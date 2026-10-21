# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""MoE FFN for GLM-5.3-Flash on one device: router + active routed experts + shared expert.

HF ``Glm5NextTextMoE``:
    logits = x_fp32 @ W_gate^T ; s = sigmoid(logits) ; choice = s + e_score_correction_bias
    idx = top8(choice) ; w = s[idx] / sum(s[idx]) * 2.5                       (n_group = 1)
    y = sum_j w_j * expert_{idx_j}(x) + shared_expert(x)
    expert(x) = (silu(clamp(gate, max=10)) * clamp(up, -10, 10)) @ W2^T

Router: the exact fp32 composite ``TtRouter.forward`` (fp32 logits as HF, sigmoid + bias, top-8 by
iterative masked argmax, renormalized and scaled). The fused ``moe_grouped_topk`` kernel is NOT used:
it ranks at bf16 resolution and swaps near-tied experts (see ``TtRouter.forward``).

Experts: ``ttnn.sparse_matmul`` with a per-expert activity mask (union over the tokens of a
32-token block) so only the experts any token in the block routes to are computed; the dense
``[tokens, E]`` routing-weight matrix (built on device from the top-k ids by one-hot compare)
weights the per-expert outputs before the reduction over experts. nnz is left to be inferred at
runtime (a static nnz that disagrees with the mask deadlocks the op). Prefill processes the
sequence in 32-token blocks with the same code.

Weight layout for sparse_matmul: W13 ``[1, E, H, 2I]`` (gate columns then up), W2 ``[1, E, I, H]``.

Expert dtype knobs (design section 9): ``expert_dtype`` is the routed W13 (gate/up) format and
``expert_down_dtype`` the routed W2 format, both ``bfloat8_b`` by default; ``bfloat4_b`` for W13 is
the candidate that halves the dominant expert-streaming term and is adopted only if the eval-set
accuracy is unchanged (``tests/test_expert_dtype.py`` carries the single-layer evidence). The shared
expert stays on ``weight_dtype`` like the dense MLP.

Expert weight cache (``weight_cache_path``, a per-layer directory): the routed W13 / W2 device
tensors are dumped as tensorbins on first construction and ``ttnn.load_tensor``-ed afterwards
(``GLMMesh.upload_cached``), which is what makes streaming all 45 layers through one chip practical
(``tt/model.py``). When the cache is complete the constructor accepts a state dict WITHOUT the
expert tensors (``TtRoutedExperts.cache_complete``), so the caller can skip dequantizing them.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.glm53_flash.custom_ops.expert_stream.op import (
    IDX_PAD,
    MAX_PICKS,
    ExpertDispatch,
    ExpertStream,
    down_stream_layout,
    gate_up_stream_layout,
    stream_memory_config,
)
from models.demos.glm53_flash.tt.config import GLM53FlashConfig
from models.demos.glm53_flash.tt.matmul_configs import DecodeLinear
from models.demos.glm53_flash.tt.mesh import GLMMesh, interleave_blocks, single_chip
from models.demos.glm53_flash.tt.mlp import TtClampedSwiGLU

TILE = 32
# prefill (T > 32 gathered tokens) routed-expert path of the hybrid layout: "blocks" = one sparse_matmul pair per 32-token
# block (the stage-1 path); "batched" = the same kernels over GLM53_PREFILL_BLOCK_GROUP blocks per call (the
# deepseek_v3 Galaxy prefill form of sparse_matmul: [1, G, 32, K] x [1, E, K, N] with a [1, G, 1, E] mask, then
# [G, E, 32, K] x [1, E, K, N] with a [1, 1, G, E] mask); "dispatch" = the deepseek_v3_d_p Blackhole prefill expert
# kernels on a one-chip dispatch group per column (masked_bincount -> region offsets -> dispatch -> ONE
# unified_routed_expert_moe program over the chip's experts (every weight read once, ClampedSiluGlu) -> combine ->
# post_combine_reduce, every chip dispatching its gathered tokens to itself along the row axis); the routed partial
# then takes the hybrid layout's reduce-scatter / all-reduce as before
# "regions" = the same d_p expert kernel (unified_routed_expert_moe + post_combine_reduce) on a region buffer built
# WITHOUT the fabric dispatch/combine (every chip of the hybrid layout already holds the gathered tokens): the router's
# dense matrix -> per-expert ranks (cumsum over the tokens) and tile-aligned region offsets (cumsum over the experts) ->
# the position of every (token, pick) -> ttnn.embedding gathers the token rows into the regions, the kernel runs, a
# second ttnn.embedding gathers every pick's output row back, post_combine_reduce applies the routing weights
PREFILL_EXPERTS = os.environ.get("GLM53_PREFILL_EXPERTS", "blocks")
# regions path: gathered tokens per kernel call (bounds the [T*k + E_local*32, H] bf16 region buffer: 1.09 GB for a 16 K
# chunk, which does not fit next to the 1 M caches; the expert weights are re-read once per sub-chunk)
REGIONS_T = int(os.environ.get("GLM53_REGIONS_T", "4096"))
# GLM53_REGIONS_KEEP_STACKED=1: keep the stacked (block / stream / dense) expert tensors next to the regions path's
# per-expert lists, so decode (T <= 32) keeps the stream custom op (75 ms vs 219 ms per step on the 8x4 through the
# regions kernel at 1 user/row) at the cost of a second copy of the expert weights (9.5 GB per chip in bfp8 on the
# 8x4: fits up to ~512 K of context, not next to the 1 M caches; 45 layers in bfp4 on a (2,4) block do not fit twice)
REGIONS_KEEP_STACKED = os.environ.get("GLM53_REGIONS_KEEP_STACKED", "0") == "1"
PREFILL_BLOCK_GROUP = int(os.environ.get("GLM53_PREFILL_BLOCK_GROUP", "32"))


def _compute(device, fidelity=ttnn.MathFidelity.HiFi2, packer_l1_acc=True):
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=packer_l1_acc,
    )


def pick_sparse_grid(device_grid, n: int, max_cores: int | None = None) -> tuple[int, int]:
    """Largest rectangular core grid inside ``device_grid`` whose core count divides the output
    tile count Nt (and is at most ``max_cores``), so every core in the mcast receiver rectangle has
    work (the sparse matmul factory requires num_cores_with_work == receiver cores)."""
    Nt = math.ceil(n / TILE)
    best = (1, 1)
    for y in range(1, device_grid[1] + 1):
        for x in range(1, device_grid[0] + 1):
            cores = x * y
            if Nt % cores or (max_cores is not None and cores > max_cores):
                continue
            # most cores; among equal counts the squarest rectangle, ties to the wider one (the measured
            # 16-core grids at N=512: (4,4) 375 us, (8,2) 377 us, (2,8) 463 us)
            if cores > best[0] * best[1] or (cores == best[0] * best[1] and abs(x - y) < abs(best[0] - best[1])):
                best = (x, y)
    return best


# in1 tiles per core per K block the sparse matmul streams between barriers (tests/test_expert_kernel_perf.py,
# doc/mesh/expert_kernel_sweep.md): the 1D mcast_in0 kernel pays one memory latency + one semaphore round trip per
# K block, so narrow outputs with few cores want big blocks (N=512 on 16 cores: 4 -> 64 tiles = 1092 -> 369 us
# per 32 active experts); with >= 32 cores the per-core streams are the limit and 16-tile blocks pipeline best.
SPARSE_BLOCK_TILES_FEW_CORES = 64
SPARSE_BLOCK_TILES = 16
SPARSE_L1_BUDGET = 1_000_000  # bytes of the kernel's dataflow buffers per core (1.46 MB L1)


def dense_matmul_config(device_grid, m: int, n: int, k: int):
    """1D mcast_in0 config for the dense expert matmuls (every core an output-width slice; the sweep of
    tests/test_expert_kernel_perf.py, stage dense2, on the 8x4 per-chip shapes): W13 [M, 4096] x
    [4096, 72 x 512] runs best on 96 cores with 4-tile K blocks (421 us = 74 % of DRAM peak at M = 32,
    631 us at M = 256), W2 [M, 72 x 256] x [72 x 256, 4096] on 32 cores x 4 tiles with 8-tile K blocks
    (240 us = 65 %) up to M = 128 and on 64 cores with 16-tile blocks at M = 256 (335 us). Output blocks are
    capped at 16 tiles per core (fp32 partials in L1); returns None (ttnn's choice) when Nt < 32."""
    Nt, Kt, Mt = math.ceil(n / TILE), math.ceil(k / TILE), math.ceil(m / TILE)
    if Nt < 32:
        return None
    if Nt >= 256:  # the gate|up width (many tiles per core)
        cores = (min(12, device_grid[0]), min(8, device_grid[1]))
        ibw = 4 if Mt <= 4 else 8
    else:  # the hidden width (Nt = 128)
        cores = (8, 4) if Mt <= 4 else (8, 8)
        ibw = 8 if Mt <= 4 else 16
    num_cores = cores[0] * cores[1]
    per_core_N = math.ceil(Nt / num_cores)
    in0_block_w = max(d for d in range(1, ibw + 1) if Kt % d == 0)
    out_block_w = max(d for d in range(1, min(per_core_N, 16) + 1) if per_core_N % d == 0)
    out_subblock_w = max(w for w in (1, 2, 4) if out_block_w % w == 0)
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=cores,
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=out_subblock_w,
        out_block_h=Mt,
        out_block_w=out_block_w,
        per_core_M=Mt,
        per_core_N=per_core_N,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )


def sparse_matmul_config(
    device_grid,
    m: int,
    n: int,
    k: int,
    in0_block_w: int | None = None,
    out_subblock_w: int | None = None,
):
    """MatmulMultiCoreReuseMultiCast1DProgramConfig for sparse_matmul (mcast_in0=True), the
    gpt_oss builder's rules plus the measured block sizes: per_core_N = ceil(Nt / cores); in0_block_w
    must divide Kt; out_subblock_w must divide out_block_w (= per_core_N); subblock area <= 4 (fp32
    accumulation, half-sync dest). Wide outputs (Nt >= 128) take 4 tiles per core (32 cores at N=4096:
    66 % of DRAM peak vs 57 % on 64 cores), narrow ones every core that divides Nt.
    ``in0_block_w=None`` picks the largest divisor of Kt whose per-core in1 block stays within
    ``SPARSE_BLOCK_TILES(_FEW_CORES)`` tiles (or the whole K when that is at most 64 tiles) and whose
    dataflow buffers (double-buffered in0 and in1, fp32 partials, bf16 output) fit the L1 budget."""
    Nt = math.ceil(n / TILE)
    Kt = math.ceil(k / TILE)
    Mt = math.ceil(m / TILE)
    if (
        os.environ.get("GLM53_SPARSE_CONFIG") == "legacy"
    ):  # the pre-sweep rule (A/B switch): most cores, 4-tile K blocks
        core_x, core_y = pick_sparse_grid(device_grid, n)
        in0_block_w, out_subblock_w = 4, 1
    else:
        core_x, core_y = pick_sparse_grid(device_grid, n, max_cores=Nt // 4 if Nt >= 128 else None)
    num_cores = core_x * core_y
    per_core_N = (Nt + num_cores - 1) // num_cores

    def l1_bytes(ibw: int) -> int:  # bf16 in0 (2048 B/tile), bfp8 in1 (1088 B), fp32 partials + bf16 out
        return 2 * Mt * ibw * 2048 + 2 * ibw * per_core_N * 1088 + Mt * per_core_N * (4096 + 2048)

    divisors = [d for d in range(1, Kt + 1) if Kt % d == 0]
    if in0_block_w is None:
        target = SPARSE_BLOCK_TILES_FEW_CORES if num_cores <= 16 else SPARSE_BLOCK_TILES
        fits = [d for d in divisors if l1_bytes(d) <= SPARSE_L1_BUDGET]
        ok = [d for d in fits if d * per_core_N <= target or (d == Kt and Kt * per_core_N <= 64)]
        in0_block_w = max(ok or fits[:1] or [1])
    elif Kt % in0_block_w != 0:
        in0_block_w = max([d for d in divisors if 2 <= d <= in0_block_w] or [Kt])
    out_block_w = per_core_N
    if out_subblock_w is None:
        out_subblock_w = 4
    out_subblock_w = max(w for w in range(1, min(out_subblock_w, 4) + 1) if out_block_w % w == 0)
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(core_x, core_y),
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=out_subblock_w,
        out_block_h=Mt,
        out_block_w=out_block_w,
        per_core_M=Mt,
        per_core_N=per_core_N,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )


class TtRouter(LightweightModule):
    def __init__(
        self,
        device,
        cfg: GLM53FlashConfig,
        gate_weight: torch.Tensor,
        correction_bias: torch.Tensor,
        *,
        mesh: GLMMesh | None = None,
    ):
        """Replicated on every chip: identical (replicated) inputs give identical routing everywhere."""
        self.device = device
        self.mesh = mesh if mesh is not None else single_chip(device)
        self.cfg = cfg
        self.E = cfg.n_routed_experts
        self.k = cfg.num_experts_per_tok
        self.w_gate = self.mesh.upload(gate_weight.float().t(), ttnn.float32)  # [H, E] fp32
        # [1,1,1,E], broadcast over the token rows
        self.bias = self.mesh.upload(correction_bias.float().reshape(1, 1, 1, self.E), ttnn.float32)
        self.iota = self.mesh.upload(torch.arange(self.E, dtype=torch.float32).reshape(1, 1, 1, self.E), ttnn.float32)
        self.iota_desc = self.mesh.upload(
            (self.E - torch.arange(self.E, dtype=torch.float32)).reshape(1, 1, 1, self.E), ttnn.float32
        )
        self.fp32 = _compute(device, ttnn.MathFidelity.HiFi4, packer_l1_acc=False)
        self.lin_gate = DecodeLinear(device, self.w_gate)
        # decode: sigmoid fused into the gate matmul's output tiles (GLM53_ROUTER_FUSED_SIGMOID=0: separate op)
        self.fused_sigmoid = os.environ.get("GLM53_ROUTER_FUSED_SIGMOID", "1") == "1"
        self.lin_gate_sig = DecodeLinear(device, self.w_gate, activation=ttnn.UnaryOpType.SIGMOID)
        self.tie_break = os.environ.get("GLM53_ROUTER_TIE_BREAK", "topk")  # topk (default) | rank | cumsum
        self._const: dict[tuple, ttnn.Tensor] = {}

    def _forward_topk(self, scores, choice, T, dense_only=False, ep_rows=False):
        """One fp32 ``ttnn.topk`` (stable bitonic network: exact fp32 ranking, lowest index on ties) over the
        biased scores, the unbiased scores gathered at the winners, renormalized (sum -> scale/sum -> multiply;
        the sum of sigmoids is > 0, no epsilon) and scattered into the dense [1,1,T,E] matrix: 8 device ops
        after the gate matmul (topk, gather, sum, rdiv, mul, typecast, scatter) instead of 8 rounds of the
        composite."""
        k = self.k
        vals, idx = ttnn.topk(
            choice, k, dim=-1, largest=True, sorted=True
        )  # [1,1,T,k] fp32 / uint32 (padded cols lose)
        ttnn.deallocate(vals)
        ttnn.deallocate(choice)
        w = ttnn.gather(scores, -1, idx)  # unbiased scores at the winners [1,1,T,k] fp32
        ttnn.deallocate(scores)
        denom = ttnn.sum(w, dim=-1, keepdim=True)  # [1,1,T,1] > 0
        norm = ttnn.rdiv(denom, self.cfg.routed_scaling_factor)  # scale / sum
        ttnn.deallocate(denom)
        w = ttnn.multiply(w, norm, output_tensor=w)
        ttnn.deallocate(norm)
        # scatter has no fp32 TILE path: the dense matrix is built in bf16, which is the dtype the expert
        # block applies anyway (it typecasts the fp32 weights to bf16 before weighting the expert outputs)
        w16 = ttnn.typecast(w, ttnn.bfloat16)
        if ep_rows:  # the EP mesh path: the bf16 weights and the uint32 ids as topk ranked them, no dense matrix
            ttnn.deallocate(w)
            return w16, idx, None
        dense = ttnn.scatter(self._dense_zeros(T), -1, idx, w16)  # [1,1,T,E] bf16: weights at their columns
        ttnn.deallocate(w16)
        if dense_only:
            ttnn.deallocate(w)
            ttnn.deallocate(idx)
            return None, None, dense
        idx32 = ttnn.typecast(idx, ttnn.float32)
        ttnn.deallocate(idx)
        return w, idx32, dense

    def _dense_zeros(self, T: int) -> ttnn.Tensor:
        key = ("zeros", T)
        if key not in self._const:
            self._const[key] = ttnn.zeros(
                (1, 1, T, self.E), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
            )
        return self._const[key]

    def forward(
        self, x: ttnn.Tensor, *, dense_only: bool = False, ep_rows: bool = False
    ) -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        """x [1,1,T,H] bf16 -> (weights [1,1,T,8] fp32, indices [1,1,T,8] fp32 (integral), dense [1,1,T,E] fp32).
        ``dense_only`` (the MoE forward): weights / indices come back as ``None`` (their host-side casts are skipped).
        ``ep_rows`` (the EP mesh path, ``topk`` tie-break only): (weights [1,1,T,8] bf16, indices [1,1,T,8] uint32,
        None): the exact ids for the all_to_all dispatch, no dense matrix.

        Exact fp32 selection (HF noaux_tc, n_group=1): ``choice = sigmoid(logits) + bias``, top-8 by
        iterative masked argmax, weights = unbiased scores at the selected experts, renormalized and
        scaled. The fused ``moe_grouped_topk`` kernel ranks with bf16 resolution (0.03 near the ~6
        biased scores), which swaps the 8th expert on ~20% of real tokens; the composite is exact.
        """
        T = x.shape[-2]
        if self.fused_sigmoid and T <= TILE and self.tie_break == "topk":
            scores = self.lin_gate_sig(x, dtype=ttnn.float32, compute_kernel_config=self.fp32)  # sigmoid fused
        else:
            logits = self.lin_gate(x, dtype=ttnn.float32, compute_kernel_config=self.fp32)
            scores = ttnn.sigmoid(logits)  # fp32 [1,1,T,E]
            ttnn.deallocate(logits)
        choice = ttnn.add(scores, self.bias)
        if self.tie_break == "topk":
            return self._forward_topk(scores, choice, T, dense_only=dense_only, ep_rows=ep_rows)
        assert not ep_rows, "ep_rows needs the topk tie-break"
        dense = None  # selected unbiased scores at their expert columns, else 0
        idx_cols, w_cols = [], []
        for _ in range(self.k):
            # first-match argmax of the biased scores (one index per row, exact fp32 ranking) -> one-hot
            # by comparing against the expert iota; replaces max + eq + single-core cumsum tie-break
            # exact fp32 first-match selection in TILE layout: max -> candidate columns; a second max over
            # the candidates weighted by a descending iota keeps only the LOWEST tied column (no cumsum,
            # no layout change; padded columns >= E weigh <= 0 and can never win)
            m = ttnn.max(choice, dim=-1, keepdim=True)  # [1,1,T,1]
            cand = ttnn.eq(choice, m)
            if self.tie_break == "cumsum":  # the stage-1 reference path (single-core cumsum), kept for A/B
                csum = ttnn.cumsum(cand, dim=-1)
                onehot = ttnn.multiply(cand, ttnn.eq(csum, 1.0))
                ttnn.deallocate(csum)
                ttnn.deallocate(m)
            else:
                ranked = ttnn.multiply(cand, self.iota_desc)  # [1,1,T,E]: E - column index on the candidates
                mx = ttnn.max(ranked, dim=-1, keepdim=True)
                onehot = ttnn.eq(ranked, mx)
                for t in (m, cand, ranked, mx):
                    ttnn.deallocate(t)
            sel = ttnn.multiply(onehot, scores)  # [1,1,T,E]
            dense = sel if dense is None else ttnn.add(dense, sel, output_tensor=dense)
            idx_cols.append(ttnn.sum(ttnn.multiply(onehot, self.iota), dim=-1, keepdim=True))
            w_cols.append(ttnn.sum(sel, dim=-1, keepdim=True))
            # remove the winner from further rounds
            choice = ttnn.subtract(choice, ttnn.multiply(onehot, 1.0e9), output_tensor=choice)
            ttnn.deallocate(onehot)
        ttnn.deallocate(choice)
        ttnn.deallocate(scores)
        weights = ttnn.concat(w_cols, dim=-1)  # [1,1,T,8] unnormalized
        indices = ttnn.concat(idx_cols, dim=-1)  # [1,1,T,8]
        denom = ttnn.add(ttnn.sum(weights, dim=-1, keepdim=True), 1e-20)
        norm = ttnn.multiply(ttnn.reciprocal(denom), self.cfg.routed_scaling_factor)  # [1,1,T,1]
        weights = ttnn.multiply(weights, norm, output_tensor=weights)
        dense = ttnn.multiply(dense, norm, output_tensor=dense)
        return weights, indices, dense


class TtRoutedExperts(LightweightModule):
    @staticmethod
    def cache_files(
        cache_dir: Path | str, mesh: GLMMesh, expert_dtype, expert_down_dtype, dense: bool = False, stream: bool = False
    ) -> dict[str, Path]:
        """Tensorbin paths of the routed-expert device tensors for this placement / format / layout."""
        tag = f"mesh{mesh.rows}x{mesh.cols}" + ("_dense" if dense else "") + ("_stream" if stream else "")
        cache_dir = Path(cache_dir)
        files = {
            "w13": cache_dir / f"experts_w13_{expert_dtype.name}_{tag}.tensorbin",
            "w2": cache_dir / f"experts_w2_{expert_down_dtype.name}_{tag}.tensorbin",
        }
        if dense:  # gate and up as two tensors (two matmuls instead of one matmul + two slices)
            files["w13_up"] = cache_dir / f"experts_w13up_{expert_dtype.name}_{tag}.tensorbin"
        return files

    @classmethod
    def cache_complete(
        cls,
        cache_dir: Path | str | None,
        mesh: GLMMesh,
        expert_dtype,
        expert_down_dtype,
        dense: bool = False,
        stream: bool = False,
    ) -> bool:
        """True when both expert tensorbins exist (the constructor then needs no expert host tensors)."""
        if cache_dir is None:
            return False
        return all(
            f.is_file()
            for f in cls.cache_files(cache_dir, mesh, expert_dtype, expert_down_dtype, dense, stream).values()
        )

    def __init__(
        self,
        device,
        cfg: GLM53FlashConfig,
        gate_up: torch.Tensor | None,  # [E, 2I, H] (HF stacked: gate rows then up rows); None = from the cache
        down: torch.Tensor | None,  # [E, H, I]
        *,
        expert_dtype=ttnn.bfloat8_b,  # W13 (gate/up) format
        expert_down_dtype=ttnn.bfloat8_b,  # W2 (down) format
        activation_dtype=ttnn.bfloat16,
        mesh: GLMMesh | None = None,
        weight_cache_path: Path | str | None = None,
        dense: bool = False,
        stream: bool = False,
    ):
        """Hybrid placement (design 5.4): the experts are split over the mesh columns (chip column
        ``c`` holds experts ``[c*E/cols, (c+1)*E/cols)``) and every expert's intermediate over the mesh
        rows (row ``r`` holds the slice ``[r*I/rows, (r+1)*I/rows)``). ``forward`` gathers the rows'
        tokens, runs the local expert slices and reduce-scatters the partials back to the token's row;
        the result is still a TP partial over the columns (the caller all-reduces once).
        ``weight_cache_path``: per-layer tensorbin directory (``cache_files``); ``gate_up`` / ``down``
        may be ``None`` when ``cache_complete`` (the host tensors are then never needed).
        ``dense``: the dense expert layout (``_block_dense``): the chip's expert slices side by side in
        ONE W13 ``[1, 1, H, 2 * E_local * I_local]`` (all gates, then all ups) and ONE W2
        ``[1, 1, E_local * I_local, H]``, run as ordinary matmuls over all gathered tokens at once; every
        local expert is read exactly once per step whatever the token count, the expert sum is the W2
        K reduction (no per-expert output slab, no zero-fill, no reduce), at the price of reading the
        inactive experts too. Right when most local experts are active (>= ~32 gathered tokens on
        the 8x4: tests/test_expert_kernel_perf.py, doc/mesh/expert_kernel_sweep.md)."""
        self.device = device
        self.mesh = mesh if mesh is not None else single_chip(device)
        self.cfg = cfg
        self.E, self.H, self.I = cfg.n_routed_experts, cfg.hidden_size, cfg.moe_intermediate_size
        self.limit = float(cfg.swiglu_limit)
        self.activation_dtype = activation_dtype
        self.expert_dtype = expert_dtype
        self.expert_down_dtype = expert_down_dtype
        self.dense = dense
        self.stream = stream
        assert not (dense and stream), "dense and stream are alternatives"
        ep, tpr = self.mesh.cols, self.mesh.rows
        assert self.E % ep == 0 and self.I % (tpr * TILE) == 0, (self.E, self.I, ep, tpr)
        self.E_local, self.I_local = self.E // ep, self.I // tpr
        files = (
            self.cache_files(weight_cache_path, self.mesh, expert_dtype, expert_down_dtype, dense, stream)
            if weight_cache_path is not None
            else {"w13": None, "w2": None, "w13_up": None}
        )
        if gate_up is None or down is None:
            if not self.cache_complete(weight_cache_path, self.mesh, expert_dtype, expert_down_dtype, dense, stream):
                raise ValueError("routed experts without host tensors need a complete weight cache")
        else:
            assert tuple(gate_up.shape) == (self.E, 2 * self.I, self.H) and tuple(down.shape) == (
                self.E,
                self.H,
                self.I,
            ), (gate_up.shape, down.shape)
        E, H, I = self.E, self.H, self.I
        E_loc, I_loc = self.E_local, self.I_local

        def make_w13() -> torch.Tensor:
            gate = gate_up[:, :I, :].transpose(1, 2)  # [E, H, I]
            up = gate_up[:, I:, :].transpose(1, 2)
            # [1,E,H,2I] in tpr blocks of [gate_r | up_r] so a row split of the last dim keeps the pairs
            return interleave_blocks([gate, up], tpr, dim=-1).reshape(1, E, H, 2 * I)

        def make_w2() -> torch.Tensor:
            return down.transpose(1, 2).contiguous().reshape(1, E, I, H)  # [1,E,I,H]

        def per_chip_dense(t):  # [E, H, I] -> [1, cols, H, rows * E_loc * I_loc]: a column split of dim 1 and a row
            # split of dim 3 hand chip (r, c) [H, (e in c, i in slice r)]
            return (
                t.reshape(ep, E_loc, H, tpr, I_loc)
                .permute(0, 2, 3, 1, 4)
                .reshape(1, ep, H, tpr * E_loc * I_loc)
                .contiguous()
            )

        def make_w13_gate_dense() -> torch.Tensor:
            return per_chip_dense(gate_up[:, :I, :].transpose(1, 2))

        def make_w13_up_dense() -> torch.Tensor:
            return per_chip_dense(gate_up[:, I:, :].transpose(1, 2))

        def make_w2_dense() -> torch.Tensor:
            # [1, cols, rows * (E_loc * I_loc), H]: chip (r, c) gets rows (e in c, i in slice r) of every W2
            w2 = down.transpose(1, 2).reshape(ep, E_loc, tpr, I_loc, H).permute(0, 2, 1, 3, 4)
            return w2.reshape(1, ep, tpr * E_loc * I_loc, H).contiguous()

        def chip_slices(r: int, c: int):
            """Chip (r, c)'s gate|up [E_loc, H, 2 I_loc] ([gate_r | up_r] columns) and down [E_loc, I_loc, H]."""
            es, isl = slice(c * E_loc, (c + 1) * E_loc), slice(r * I_loc, (r + 1) * I_loc)
            g = gate_up[es, :I, :][:, isl, :].transpose(1, 2)  # [E_loc, H, I_loc]
            u = gate_up[es, I:, :][:, isl, :].transpose(1, 2)
            w13c = torch.cat([g, u], dim=-1).float()
            w2c = down[es][:, :, isl].transpose(1, 2).float()  # [E_loc, I_loc, H]
            return w13c, w2c

        def make_w13_stream() -> torch.Tensor:  # [rows_m, cols_m, rows13, 256]: a (0, 1) mesh split
            return torch.stack(
                [torch.stack([gate_up_stream_layout(chip_slices(r, c)[0]) for c in range(ep)]) for r in range(tpr)]
            ).contiguous()

        def make_w2_stream() -> torch.Tensor:
            return torch.stack(
                [torch.stack([down_stream_layout(chip_slices(r, c)[1]) for c in range(ep)]) for r in range(tpr)]
            ).contiguous()

        if stream:
            rows13 = E_loc * 2 * (I_loc // TILE // 8) * (H // TILE) * TILE  # ppb column pairs per bank
            rows2 = E_loc * (H // TILE // 8) * (I_loc // TILE) * TILE
            self.w13 = self.mesh.upload_cached(
                make_w13_stream,
                expert_dtype,
                files["w13"],
                rows=0,
                cols=1,
                memory_config=stream_memory_config(rows13),
                expect_shape=(tpr, ep, rows13, 256),
            )
            self.w2 = self.mesh.upload_cached(
                make_w2_stream,
                expert_down_dtype,
                files["w2"],
                rows=0,
                cols=1,
                memory_config=stream_memory_config(rows2),
                expect_shape=(tpr, ep, rows2, 256),
            )
            self.stream_x_l1 = os.environ.get("GLM53_STREAM_X_L1", "1") == "1"
            self.stream_act_l1 = os.environ.get("GLM53_STREAM_ACT_L1", "1") == "1"
            # phase 2: > 32 gathered tokens go through the per-expert dispatch ops (weights read once)
            self.stream_dispatch = os.environ.get("GLM53_STREAM_DISPATCH", "1") == "1"
            self.dispatch_ops = {}
            self.dispatch_shapes = (E_loc, H, I_loc, H)
            self.stream_op = ExpertStream(
                device,
                E_loc,
                H,
                I_loc,
                H,
                groups=int(os.environ.get("GLM53_STREAM_GROUPS", "12")),
                down_readers=int(os.environ["GLM53_STREAM_DOWN_READERS"])
                if os.environ.get("GLM53_STREAM_DOWN_READERS")
                else None,  # auto: 2 at I_loc = 256, 4 at 1024, 8 at 2048
            )
        elif dense:
            shape13 = (1, ep, H, tpr * E_loc * I_loc)
            self.w13 = self.mesh.upload_cached(
                make_w13_gate_dense, expert_dtype, files["w13"], rows=3, cols=1, expect_shape=shape13
            )
            self.w13_up = self.mesh.upload_cached(
                make_w13_up_dense, expert_dtype, files["w13_up"], rows=3, cols=1, expect_shape=shape13
            )
            self.w2 = self.mesh.upload_cached(
                make_w2_dense,
                expert_down_dtype,
                files["w2"],
                rows=2,
                cols=1,
                expect_shape=(1, ep, tpr * E_loc * I_loc, H),
            )
            # routing-weight selection + expansion in ONE 0/1 constant: [E, E_loc * I_loc] per chip (column c keeps
            # its experts' rows), so w_exp = dense_w @ sel_expand repeats token t's weight for local expert e over
            # e's I_loc intermediate columns (exact in bf16; HiFi4 keeps the bf16 weights intact)
            self.sel_expand = self.mesh.upload(
                torch.eye(E, dtype=torch.bfloat16).repeat_interleave(I_loc, dim=1).reshape(1, 1, E, E * I_loc),
                ttnn.bfloat16,
                cols=-1,
            )
        else:
            self.w13 = self.mesh.upload_cached(
                make_w13, expert_dtype, files["w13"], rows=3, cols=1, expect_shape=(1, E, H, 2 * I)
            )
            self.w2 = self.mesh.upload_cached(
                make_w2, expert_down_dtype, files["w2"], rows=2, cols=1, expect_shape=(1, E, I, H)
            )
        self.prefill_experts = PREFILL_EXPERTS if not (dense or stream) else "blocks"
        self._dispatch = None
        if self.prefill_experts in ("dispatch", "regions"):
            self._build_dispatch_path(fabric=self.prefill_experts == "dispatch")
        # column c's expert selector: the [E, E_local] block of the identity for its experts
        self.select = (
            None if ep == 1 else self.mesh.upload(torch.eye(self.E, dtype=torch.float32), ttnn.float32, cols=-1)
        )
        self.select_bf16 = (
            None if ep == 1 else self.mesh.upload(torch.eye(self.E, dtype=torch.float32), ttnn.bfloat16, cols=-1)
        )
        grid = device.compute_with_storage_grid_size()
        self.grid = (grid.x, grid.y)
        self.compute = _compute(device)
        self.fp32 = _compute(device, ttnn.MathFidelity.HiFi4, packer_l1_acc=False)
        self._const_cache: dict = {}

    def _block(self, xb: ttnn.Tensor, dense_w: ttnn.Tensor) -> ttnn.Tensor:
        """One token block (<= 32 rows). xb [1,1,M,H] bf16, dense_w [1,1,M,E] bf16 (fp32 costs two
        typecasts) -> [1,1,M,H] (this chip's experts and intermediate slice only). 12 device ops plus the
        two output zero-fills inside ``sparse_matmul``: select matmul, activity mask (sum + untilize), gate|up
        sparse matmul, two slices, ``clamped_silu_glu``, the routing weights as [1,E,M,1] (one permute) and
        their multiply, down sparse matmul, sum over the experts."""
        E, H, I = self.E_local, self.H, self.I_local
        M = xb.shape[-2]
        if self.select is not None:  # keep the routing weights of this column's experts: [1,1,M,E_local]
            sel = self.select if dense_w.dtype == ttnn.float32 else self.select_bf16
            dense_w = ttnn.matmul(dense_w, sel, dtype=dense_w.dtype, compute_kernel_config=self.fp32)
        # per-expert activity mask over the block (nonzero = active); bf16 ROW_MAJOR [1,1,1,E]
        active = ttnn.sum(dense_w, dim=-2, keepdim=True)  # [1,1,1,E]
        if active.dtype != ttnn.bfloat16:
            active = ttnn.typecast(active, ttnn.bfloat16)
        sparsity = ttnn.to_layout(active, ttnn.ROW_MAJOR_LAYOUT)
        # gate|up for the active experts: [1,1,1,E,M,2I] (zeros for inactive experts)
        gu = ttnn.sparse_matmul(
            xb,
            self.w13,
            sparsity=sparsity,
            nnz=None,
            is_input_a_sparse=False,
            is_input_b_sparse=True,
            program_config=sparse_matmul_config(self.grid, M, 2 * I, H),
            dtype=self.activation_dtype,
            compute_kernel_config=self.compute,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        gu = ttnn.reshape(gu, [1, E, M, 2 * I])
        gate = ttnn.slice(gu, [0, 0, 0, 0], [1, E, M, I])
        up = ttnn.slice(gu, [0, 0, 0, I], [1, E, M, 2 * I])
        ttnn.deallocate(gu)
        h = ttnn.clamped_silu_glu(gate, up, self.limit)  # [1,E,M,I]
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        # weight each (expert, token) row by its routing weight before the down projection: [1,1,M,E] -> [1,E,M,1]
        w16 = dense_w if dense_w.dtype == self.activation_dtype else ttnn.typecast(dense_w, self.activation_dtype)
        w_cols = ttnn.permute(w16, (0, 3, 2, 1))
        h = ttnn.multiply(h, w_cols, output_tensor=h)
        ttnn.deallocate(w_cols)
        down = ttnn.sparse_matmul(
            h,
            self.w2,
            sparsity=sparsity,
            nnz=None,
            is_input_a_sparse=True,
            is_input_b_sparse=True,
            program_config=sparse_matmul_config(self.grid, M, H, I),
            dtype=self.activation_dtype,
            compute_kernel_config=self.compute,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )  # [1,E,M,H]
        ttnn.deallocate(h)
        ttnn.deallocate(sparsity)
        y = ttnn.sum(down, dim=1, keepdim=True)  # [1,1,M,H]
        ttnn.deallocate(down)
        return y

    def _block_dense(self, x: ttnn.Tensor, dense_w: ttnn.Tensor) -> ttnn.Tensor:
        """All gathered tokens at once through the dense layout, 6 device ops: x [1,1,T,H] bf16, dense_w
        [1,1,T,E] bf16 (fp32 costs one typecast) -> [1,1,T,H] (this chip's expert slices, summed over its
        experts). Routing weights: one [T,E] x [E, E_loc*I_loc] matmul; gate and up: one matmul each (no
        slices); activation: ``clamped_silu_glu``; the weight multiply; the W2 matmul (expert sum = K)."""
        E, H, I = self.E_local, self.H, self.I_local
        T = x.shape[-2]
        if dense_w.dtype != ttnn.bfloat16:
            dense_w = ttnn.typecast(dense_w, ttnn.bfloat16)
        w_exp = ttnn.matmul(
            dense_w,
            self.sel_expand,
            program_config=dense_matmul_config(self.grid, T, E * I, self.E),
            dtype=self.activation_dtype,
            compute_kernel_config=self.fp32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )  # [1,1,T,E*I]
        gate = ttnn.matmul(
            x,
            self.w13,
            program_config=dense_matmul_config(self.grid, T, E * I, H),
            dtype=self.activation_dtype,
            compute_kernel_config=self.compute,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )  # [1,1,T,E*I], (expert, i) order
        up = ttnn.matmul(
            x,
            self.w13_up,
            program_config=dense_matmul_config(self.grid, T, E * I, H),
            dtype=self.activation_dtype,
            compute_kernel_config=self.compute,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        h = ttnn.clamped_silu_glu(gate, up, self.limit)  # silu(min(gate, limit)) * clamp(up, -limit, limit)
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        h = ttnn.multiply(h, w_exp, output_tensor=h)
        ttnn.deallocate(w_exp)
        y = ttnn.matmul(
            h,
            self.w2,
            program_config=dense_matmul_config(self.grid, T, H, E * I),
            dtype=self.activation_dtype,
            compute_kernel_config=self.compute,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )  # [1,1,T,H] = sum over the chip's experts
        ttnn.deallocate(h)
        return y

    def _block_stream(self, xb: ttnn.Tensor, dense_w: ttnn.Tensor) -> ttnn.Tensor:
        """One token block (<= 32 rows) through the streaming generic ops: the column selector matmul (on a
        multi-column mesh), the gate|up stream, the down stream. xb [1,1,M,H] bf16 (M = 32: a tile row),
        dense_w [1,1,M,E] -> [1,1,32,H]."""
        if self.select is not None:
            dense_w = ttnn.matmul(dense_w, self.select_bf16, dtype=ttnn.bfloat16, compute_kernel_config=self.fp32)
        elif dense_w.dtype != ttnn.bfloat16:
            dense_w = ttnn.typecast(dense_w, ttnn.bfloat16)
        if self.stream_x_l1 and not xb.is_sharded() and xb.memory_config().buffer_type != ttnn.BufferType.L1:
            xb = ttnn.to_memory_config(xb, ttnn.L1_MEMORY_CONFIG)  # the 96 gate|up cores re-read x: L1 is cheaper
        act, out = self.stream_op.alloc(self.device, act_l1=self.stream_act_l1)
        self.stream_op.forward(xb, dense_w, self.w13, self.w2, act, out)
        ttnn.deallocate(act)
        return out

    def _dispatch_op(self, T: int):
        """The dispatch ops for T gathered tokens (one program per T, cached)."""
        if T not in self.dispatch_ops:
            E_loc, K, I_loc, H = self.dispatch_shapes
            self.dispatch_ops[T] = ExpertDispatch(
                self.device,
                E_loc,
                K,
                I_loc,
                H,
                T,
                groups=int(os.environ.get("GLM53_STREAM_GROUPS", "12")),
                down_readers=int(os.environ.get("GLM53_DISPATCH_DOWN_READERS", "8")),
            )
        return self.dispatch_ops[T]

    def _block_dispatch(self, x: ttnn.Tensor, dense_w: ttnn.Tensor) -> ttnn.Tensor:
        """All T (multiple of 32, <= 256) gathered tokens through the dispatch ops: the column selector matmul,
        the row-major copy of x and the per-expert counts, the gate|up dispatch, the down dispatch -> [1,1,T,H]."""
        if self.select is not None:
            dense_w = ttnn.matmul(dense_w, self.select_bf16, dtype=ttnn.bfloat16, compute_kernel_config=self.fp32)
        elif dense_w.dtype != ttnn.bfloat16:
            dense_w = ttnn.typecast(dense_w, ttnn.bfloat16)
        op = self._dispatch_op(x.shape[-2])
        x_rm, counts = ExpertDispatch.prepare(x, dense_w)
        act, table, out = op.alloc(self.device, act_l1=self.stream_act_l1)
        op.forward(x_rm, dense_w, counts, self.w13, self.w2, act, table, out)
        for t in (x_rm, counts, act, table):
            ttnn.deallocate(t)
        return out

    # ------------------------------------------------------------- prefill: batched sparse blocks
    def _forward_batched(self, x: ttnn.Tensor, dense_w: ttnn.Tensor) -> ttnn.Tensor:
        """All T gathered tokens (T % 32 == 0) through the block-sparse kernels in groups of
        ``PREFILL_BLOCK_GROUP`` 32-token blocks per call (the deepseek_v3 Galaxy prefill form): gate|up
        ``[1, G, 32, H] x [1, E, H, 2I]`` under the ``[1, G, 1, E]`` activity mask, the activation and routing
        weights on ``[G, E, 32, I]``, down ``[G, E, 32, I] x [1, E, I, H]`` under the ``[1, 1, G, E]`` mask, the
        expert sum. Same kernels and program configs as ``_block`` (the active experts' weights are still
        streamed once per 32-token block); G x fewer op launches and the per-block glue batched."""
        E, H, I = self.E_local, self.H, self.I_local
        T = x.shape[-2]
        if self.select is not None:
            sel = self.select if dense_w.dtype == ttnn.float32 else self.select_bf16
            dense_w = ttnn.matmul(dense_w, sel, dtype=dense_w.dtype, compute_kernel_config=self.fp32)
        if dense_w.dtype != self.activation_dtype:
            dense_w = ttnn.typecast(dense_w, self.activation_dtype)
        nb = T // TILE
        xb = ttnn.reshape(x, [1, nb, TILE, H])
        wb = ttnn.reshape(dense_w, [1, nb, TILE, E])
        cfg_gu = sparse_matmul_config(self.grid, TILE, 2 * I, H)
        cfg_down = sparse_matmul_config(self.grid, TILE, H, I)
        outs = []
        for g0 in range(0, nb, PREFILL_BLOCK_GROUP):
            g1 = min(nb, g0 + PREFILL_BLOCK_GROUP)
            G = g1 - g0
            whole = (g0, g1) == (0, nb)
            xg = xb if whole else ttnn.slice(xb, [0, g0, 0, 0], [1, g1, TILE, H])
            wg = wb if whole else ttnn.slice(wb, [0, g0, 0, 0], [1, g1, TILE, E])
            active = ttnn.sum(wg, dim=-2, keepdim=True)  # [1, G, 1, E] bf16: nonzero = active in that block
            sparsity = ttnn.to_layout(active, ttnn.ROW_MAJOR_LAYOUT)
            ttnn.deallocate(active)
            gu = ttnn.sparse_matmul(
                xg,
                self.w13,
                sparsity=sparsity,
                nnz=None,
                is_input_a_sparse=False,
                is_input_b_sparse=True,
                program_config=cfg_gu,
                dtype=self.activation_dtype,
                compute_kernel_config=self.compute,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )  # [1, G, 1, E, 32, 2I]
            gu = ttnn.reshape(gu, [G, E, TILE, 2 * I])
            gate = ttnn.slice(gu, [0, 0, 0, 0], [G, E, TILE, I])
            up = ttnn.slice(gu, [0, 0, 0, I], [G, E, TILE, 2 * I])
            ttnn.deallocate(gu)
            h = ttnn.clamped_silu_glu(gate, up, self.limit)  # [G, E, 32, I]
            ttnn.deallocate(gate)
            ttnn.deallocate(up)
            w_cols = ttnn.permute(wg, (1, 3, 2, 0))  # [G, E, 32, 1]
            h = ttnn.multiply(h, w_cols, output_tensor=h)
            ttnn.deallocate(w_cols)
            sparsity_a = ttnn.reshape(sparsity, [1, 1, G, E])  # the same mask in the [A, E] form
            down = ttnn.sparse_matmul(
                h,
                self.w2,
                sparsity=sparsity_a,
                nnz=None,
                is_input_a_sparse=True,
                is_input_b_sparse=False,
                program_config=cfg_down,
                dtype=self.activation_dtype,
                compute_kernel_config=self.compute,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )  # [G, E, 32, H]
            ttnn.deallocate(h)
            ttnn.deallocate(sparsity)
            y = ttnn.sum(down, dim=1, keepdim=True)  # [G, 1, 32, H]
            ttnn.deallocate(down)
            outs.append(ttnn.reshape(y, [1, 1, G * TILE, H]))
            if not whole:
                ttnn.deallocate(xg)
                ttnn.deallocate(wg)
        if len(outs) == 1:
            return outs[0]
        y = ttnn.concat(outs, dim=-2)
        for o in outs:
            ttnn.deallocate(o)
        return y

    # ------------------------------------------------------- prefill: one-chip dispatch + unified expert kernel
    def _build_dispatch_path(self, fabric: bool = True):
        """Per-expert weight tensors (the ``unified_routed_expert_moe`` contract: one ``[H, I_local]`` gate, one
        up and one ``[I_local, H]`` down per local expert, DRAM interleaved) sliced out of the stacked hybrid
        tensors, which are then released (this path serves prefill; the stacked form is the decode block path's,
        so a model that needs both keeps two copies: a follow-up). Plus the routing constants: the chip's dispatch
        table (``0`` on its experts, ``-1`` elsewhere: a one-chip dispatch group), the local -> global expert ids."""
        E, E_loc, H, I = self.E, self.E_local, self.H, self.I_local
        cols = self.mesh.cols
        gate, up, down = [], [], []
        for e in range(E_loc):
            g = ttnn.slice(self.w13, [0, e, 0, 0], [1, e + 1, H, I])
            u = ttnn.slice(self.w13, [0, e, 0, I], [1, e + 1, H, 2 * I])
            d = ttnn.slice(self.w2, [0, e, 0, 0], [1, e + 1, I, H])
            gate.append(ttnn.reshape(g, [H, I]))
            up.append(ttnn.reshape(u, [H, I]))
            down.append(ttnn.reshape(d, [I, H]))
        if not REGIONS_KEEP_STACKED:
            ttnn.deallocate(self.w13)
            ttnn.deallocate(self.w2)
            self.w13 = self.w2 = None
        # The dispatch / combine ops are fabric ops along one mesh axis: a device is "chip i" of its dispatch group
        # by its index along that axis, and they refuse a device without neighbours (a 1x1 mesh). The hybrid layout
        # needs every chip to keep ITS gathered tokens for ITS expert slices, so the group is the row axis (the
        # mesh rows, every chip the only source of its own tokens) and chip (r, c)'s table sends its experts
        # [c*E_loc, (c+1)*E_loc) to chip r = itself (every write a local NOC write, the fabric links idle); the
        # other columns' experts are absent (-1) and never counted / dispatched. Single-row meshes are refused.
        rows = self.mesh.rows
        if fabric and rows < 2:
            raise ValueError(
                "GLM53_PREFILL_EXPERTS=dispatch needs a multi-row mesh (the dispatch op wants fabric neighbours)"
            )
        table = torch.full((rows, cols, 1, E), -1, dtype=torch.int32)
        gidx = torch.zeros(rows, cols, 1, E_loc, dtype=torch.int32)
        for r in range(rows):
            for c in range(cols):
                table[r, c, 0, c * E_loc : (c + 1) * E_loc] = r
                gidx[r, c, 0] = torch.arange(c * E_loc, (c + 1) * E_loc)
        self._dispatch = {
            "gate": gate,
            "up": up,
            "down": down,
            "table": self.mesh.upload(
                table, ttnn.int32, rows=0, cols=1, layout=ttnn.ROW_MAJOR_LAYOUT
            ),  # [1,1,1,E] per chip
            "gidx": self.mesh.upload(gidx, ttnn.uint32, rows=0, cols=1, layout=ttnn.ROW_MAJOR_LAYOUT),  # [1,1,1,E_loc]
            "table_1d": None,
            "group": rows,
            "axis": 0,
            "fabric": fabric,
        }
        # the region path's routing constant: the chip's expert mask as an fp32 row (1 on its experts)
        self._dispatch["mask"] = self.mesh.upload((table >= 0).float(), ttnn.float32, rows=0, cols=1)  # [1,1,1,E]
        self._dispatch["table_rm"] = ttnn.reshape(self._dispatch["table"], [1, E])
        self._dispatch["table_1d"] = ttnn.reshape(self._dispatch["table"], [E])
        self._dispatch["gidx_rm"] = ttnn.reshape(self._dispatch["gidx"], [1, E_loc])
        self.dispatch_max_tokens = int(os.environ.get("GLM53_DISPATCH_MAX_TOKENS", "0"))  # 0 = T * k

    def _forward_dispatch(self, x: ttnn.Tensor, idx: ttnn.Tensor, w: ttnn.Tensor) -> ttnn.Tensor:
        """x [1,1,T,H] bf16 TILE (all gathered tokens, T % 64 == 0), idx / w [1,1,T,k] fp32 TILE (the router's
        global expert ids as exact integers and their normalized weights) -> [1,1,T,H] bf16 TILE: the chip's
        expert partial. The deepseek_v3_d_p Blackhole prefill expert pipeline on a one-chip dispatch group:
        masked_bincount (counts of the chip's experts) -> tile-aligned exclusive prefix sum (fp32 cumsum, exact
        integers) = the expert region offsets -> dispatch (NOC gather of every token's rows into its experts'
        regions, metadata for the combine) -> unified_routed_expert_moe (every local expert's gate / up /
        ClampedSiluGlu / down on its region, weights read once) -> combine (back to token order, one slot per
        top-k pick) -> post_combine_reduce (the routing weights, the sum over the picks)."""
        d = self._dispatch
        E, E_loc, H, k = self.E, self.E_local, self.H, idx.shape[-1]
        T = x.shape[-2]
        # routing tensors in the kernels' formats
        idx_u32 = ttnn.typecast(idx, ttnn.uint32)
        idx_u16 = ttnn.typecast(idx_u32, ttnn.uint16)  # [1,1,T,k] TILE
        ttnn.deallocate(idx_u32)
        counts = ttnn.experimental.deepseek_prefill.masked_bincount(
            ttnn.reshape(idx_u16, [T, k]), d["table_rm"], E, k
        )  # [E] uint32: the chip's experts' token counts, 0 elsewhere
        counts_rm = ttnn.reshape(counts, [1, E])
        # tile-aligned exclusive prefix sum over the experts (fp32 TILE: integer sums < 2^24 are exact)
        cf = ttnn.to_layout(ttnn.typecast(ttnn.reshape(counts, [1, 1, 1, E]), ttnn.float32), ttnn.TILE_LAYOUT)
        aligned = ttnn.multiply(ttnn.floor(ttnn.multiply(ttnn.add(cf, TILE - 1), 1.0 / TILE)), float(TILE))
        incl = ttnn.cumsum(aligned, dim=-1)
        excl = ttnn.subtract(incl, aligned)
        excl_u32 = ttnn.typecast(excl, ttnn.uint32)
        offsets = ttnn.reshape(ttnn.to_layout(excl_u32, ttnn.ROW_MAJOR_LAYOUT), [1, E])
        for t in (cf, aligned, incl, excl, excl_u32):
            ttnn.deallocate(t)
        # region rows = every pick + up to 31 alignment rows per local expert, rounded up to a tile multiple (the
        # kernel wants x_shape[-2] % 32 == 0; decode with 2 gathered tokens x 72 local experts gave 2320)
        max_buf = self.dispatch_max_tokens or (-(-(T * k + E_loc * TILE) // TILE) * TILE)
        x_rm = ttnn.reshape(ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT), [1, T, H])
        idx_rm = ttnn.reshape(ttnn.to_layout(idx_u16, ttnn.ROW_MAJOR_LAYOUT), [1, T, k])  # dispatch + reduce ids
        ttnn.deallocate(idx_u16)
        buf, meta = ttnn.experimental.deepseek_prefill.dispatch(
            x_rm,
            idx_rm,
            offsets,
            d["table_rm"],
            dispatch_group_size=d["group"],
            experts_per_chip=E_loc,
            num_routed_experts=E,
            num_experts_per_tok=k,
            metadata_len=3,
            max_dispatch_buffer_token_size=max_buf,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cluster_axis=d["axis"],
            num_links=1,
            topology=ttnn.Topology.Linear,
        )
        ttnn.deallocate(x_rm)
        out = ttnn.experimental.deepseek_prefill.unified_routed_expert_moe(
            ttnn.reshape(buf, [max_buf, H]),
            offsets,
            counts_rm,
            d["gidx_rm"],
            d["gate"],
            d["up"],
            d["down"],
            self.dispatch_max_tokens or T,
            compute_kernel_config=self.compute,
            activation=ttnn.RoutedExpertActivation.ClampedSiluGlu,
        )  # [max_buf, H] bf8 TILE
        ttnn.deallocate(buf)
        comb = ttnn.experimental.deepseek_prefill.combine(
            ttnn.reshape(out, [1, 1, max_buf, H]),
            meta,
            counts_rm,
            offsets,
            dispatch_group_size=d["group"],
            experts_per_chip=E_loc,
            num_experts_per_tok=k,
            seq_len_per_chip=T,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cluster_axis=d["axis"],
            num_links=1,
            topology=ttnn.Topology.Linear,
            init_zeros=True,
        )  # [1, 1, T, k, H] bf16 ROW_MAJOR, zeros at the other columns' experts
        ttnn.deallocate(out)
        ttnn.deallocate(meta)
        # the reduce's conventions (deepseek_v3_d_p tt_reduce.py): weights [1, 1, T, k, 1] bf16 ROW_MAJOR, the
        # uint16 ids of the dispatch, the dispatch table as the 1-D [E] int32 row
        w16 = ttnn.typecast(w, ttnn.bfloat16)
        w_rm = ttnn.unsqueeze(ttnn.to_layout(w16, ttnn.ROW_MAJOR_LAYOUT), -1)  # [1, 1, T, k, 1]
        ttnn.deallocate(w16)
        y = ttnn.experimental.deepseek_prefill.post_combine_reduce(
            comb, w_rm, idx_rm, d["table_1d"], expert_dim=3, output_memory_config=ttnn.DRAM_MEMORY_CONFIG
        )  # [1, 1, T, H] TILE
        for t in (comb, w_rm, idx_rm, counts, offsets):
            ttnn.deallocate(t)
        if y.dtype != self.activation_dtype:
            y = ttnn.typecast(y, self.activation_dtype)
        return y

    def _forward_regions_chunked(
        self, x: ttnn.Tensor, dense_w: ttnn.Tensor, idx: ttnn.Tensor, w: ttnn.Tensor
    ) -> ttnn.Tensor:
        """The regions path in row sub-chunks of at most REGIONS_T gathered tokens (memory bound of the region buffer
        and the per-pick gather tables at long contexts); one call when T fits."""
        T = x.shape[-2]
        if T <= REGIONS_T:
            return self._forward_regions(x, dense_w, idx, w)
        outs = []
        for t0 in range(0, T, REGIONS_T):
            t1 = min(t0 + REGIONS_T, T)
            parts = [ttnn.slice(t, [0, 0, t0, 0], [1, 1, t1, t.shape[-1]]) for t in (x, dense_w, idx, w)]
            outs.append(self._forward_regions(*parts))
            for t in parts:
                ttnn.deallocate(t)
        y = ttnn.concat(outs, dim=-2)
        for o in outs:
            ttnn.deallocate(o)
        return y

    def _forward_regions(self, x: ttnn.Tensor, dense_w: ttnn.Tensor, idx: ttnn.Tensor, w: ttnn.Tensor) -> ttnn.Tensor:
        """x [1,1,T,H] bf16 TILE (all gathered tokens), dense_w [1,1,T,E] (the router's dense routing matrix, nonzero
        at the picks), idx / w [1,1,T,k] fp32 TILE -> [1,1,T,H] bf16: the chip's expert partial through the d_p
        ``unified_routed_expert_moe`` kernel on a region buffer built with plain ttnn ops (no fabric):
        onehot[t, e] = (dense_w != 0) x the chip's expert mask; counts[e] = sum_t; rank[t, e] = cumsum over t;
        offsets[e] = tile-aligned exclusive cumsum over e; pos[t, e] = offsets[e] + rank[t, e] - 1 at the local picks;
        the picks' positions / local flags gathered at idx; the slot -> token map by one scatter, ``ttnn.embedding``
        gathers the token rows into the regions; the kernel; ``ttnn.embedding(pos, y_rows)`` = every pick's output
        row; ``post_combine_reduce`` with the routing weights zeroed at the non-local picks (its zero-weight skip)."""
        d = self._dispatch
        E, E_loc, H, k = self.E, self.E_local, self.H, idx.shape[-1]
        T = x.shape[-2]
        # the kernel wants a tile-multiple token count (decode gathers 2 tokens; a ragged last chunk any T): pad the
        # rows with zeros (zero routing = no pick, every padded row lands on the dummy slot) and slice the output back
        T_pad = -(-T // TILE) * TILE
        padded = []
        if T_pad != T:
            x, dense_w, idx, w = padded = [self._pad_rows(t, T_pad) for t in (x, dense_w, idx, w)]
        T_logical, T = T, T_pad
        onehot = ttnn.multiply(ttnn.typecast(ttnn.ne(dense_w, 0.0), ttnn.float32), d["mask"])  # [1,1,T,E] fp32 0/1
        counts_f = ttnn.sum(onehot, dim=-2, keepdim=True)  # [1,1,1,E]
        rank = ttnn.cumsum(onehot, dim=-2)  # [1,1,T,E]: inclusive rank of token t among expert e's tokens
        aligned = ttnn.multiply(ttnn.floor(ttnn.multiply(ttnn.add(counts_f, TILE - 1), 1.0 / TILE)), float(TILE))
        incl = ttnn.cumsum(aligned, dim=-1)
        offsets_f = ttnn.subtract(incl, aligned)  # [1,1,1,E] exclusive, tile-aligned
        # region rows = every pick + up to 31 alignment rows per local expert, rounded up to a tile multiple (the
        # kernel wants x_shape[-2] % 32 == 0; decode with 2 gathered tokens x 72 local experts gave 2320)
        max_buf = self.dispatch_max_tokens or (-(-(T * k + E_loc * TILE) // TILE) * TILE)
        # pos[t, e] = offsets[e] + rank[t, e] - 1 at the local picks; the non-local picks go to the dummy slot
        # max_buf - 1, which no region reaches (the tile-aligned regions total at most T*k + E_loc*31 rows), so
        # they never overwrite a region row (slot 0 is the first row of the first local expert's region)
        pos_te = ttnn.add(
            ttnn.multiply(ttnn.add(ttnn.subtract(rank, 1.0), offsets_f), onehot),
            ttnn.multiply(ttnn.rsub(onehot, 1.0), float(max_buf - 1)),
        )  # [1,1,T,E]
        for t in (rank, aligned, incl):
            ttnn.deallocate(t)
        idx_u32 = ttnn.typecast(idx, ttnn.uint32)
        pos_tk = ttnn.gather(pos_te, -1, idx_u32)  # [1,1,T,k] fp32: the region slot of every pick (dummy if not local)
        loc_tk = ttnn.gather(onehot, -1, idx_u32)  # [1,1,T,k]: 1 at the local picks
        ttnn.deallocate(pos_te)
        ttnn.deallocate(onehot)
        ttnn.deallocate(idx_u32)
        pos_rm = ttnn.reshape(ttnn.to_layout(ttnn.typecast(pos_tk, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT), [1, T * k])
        ttnn.deallocate(pos_tk)
        # slot -> token (the picks' slots are distinct; the non-local picks all land on the dummy slot)
        slot_src = ttnn.scatter(self._zero_u32_row(max_buf), -1, pos_rm, self._iota_rows(T, k))  # [1, max_buf]
        x_rm = ttnn.reshape(ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT), [T, H])
        buf = ttnn.embedding(slot_src, x_rm, layout=ttnn.ROW_MAJOR_LAYOUT)  # [1, max_buf, H] bf16: the regions
        ttnn.deallocate(slot_src)
        ttnn.deallocate(x_rm)
        counts = ttnn.reshape(ttnn.to_layout(ttnn.typecast(counts_f, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT), [1, E])
        offsets = ttnn.reshape(ttnn.to_layout(ttnn.typecast(offsets_f, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT), [1, E])
        ttnn.deallocate(counts_f)
        ttnn.deallocate(offsets_f)
        out = ttnn.experimental.deepseek_prefill.unified_routed_expert_moe(
            ttnn.reshape(buf, [max_buf, H]),
            offsets,
            counts,
            d["gidx_rm"],
            d["gate"],
            d["up"],
            d["down"],
            self.dispatch_max_tokens or T,
            compute_kernel_config=self.compute,
            activation=ttnn.RoutedExpertActivation.ClampedSiluGlu,
        )  # [max_buf, H] bf8 TILE
        ttnn.deallocate(buf)
        out16 = ttnn.typecast(out, ttnn.bfloat16)
        ttnn.deallocate(out)
        y_rm = ttnn.to_layout(out16, ttnn.ROW_MAJOR_LAYOUT)  # the gather table [max_buf, H] bf16 rows
        ttnn.deallocate(out16)
        comb = ttnn.embedding(pos_rm, y_rm, layout=ttnn.ROW_MAJOR_LAYOUT)  # [1, T*k, H]: every pick's output row
        ttnn.deallocate(y_rm)
        ttnn.deallocate(pos_rm)
        comb5 = ttnn.reshape(comb, [1, 1, T, k, H])
        w_loc = ttnn.multiply(w, loc_tk)  # the routing weights, 0 at the non-local picks (their rows are garbage)
        ttnn.deallocate(loc_tk)
        w16 = ttnn.typecast(w_loc, ttnn.bfloat16)
        w_rm = ttnn.unsqueeze(ttnn.to_layout(w16, ttnn.ROW_MAJOR_LAYOUT), -1)  # [1,1,T,k,1]
        ttnn.deallocate(w_loc)
        ttnn.deallocate(w16)
        y = ttnn.experimental.deepseek_prefill.post_combine_reduce(
            comb5, w_rm, expert_dim=3, output_memory_config=ttnn.DRAM_MEMORY_CONFIG
        )  # [1,1,T,H] TILE: sum over the picks, zero weights skipped
        for t in (comb, w_rm, counts, offsets):
            ttnn.deallocate(t)
        if y.dtype != self.activation_dtype:
            y = ttnn.typecast(y, self.activation_dtype)
        for t in padded:
            ttnn.deallocate(t)
        if T_logical != T:
            y = ttnn.slice(y, [0, 0, 0, 0], [1, 1, T_logical, H])
        return y

    def _iota_rows(self, T: int, k: int) -> ttnn.Tensor:
        """uint32 ROW_MAJOR [1, T*k] with value t at (t, j) (cached per geometry)."""
        key = ("iota_rows", T, k)
        if key not in self._const_cache:
            t = torch.arange(T, dtype=torch.int32).reshape(T, 1).expand(T, k).reshape(1, T * k).contiguous()
            self._const_cache[key] = self.mesh.upload(t, ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        return self._const_cache[key]

    @staticmethod
    def _pad_rows(t: ttnn.Tensor, T_pad: int) -> ttnn.Tensor:
        """[1,1,T,W] TILE -> [1,1,T_pad,W] TILE with zero rows appended (through ROW_MAJOR so the new rows are
        really zero, not the tile padding's leftovers)."""
        T = t.shape[-2]
        if T == T_pad:
            return t
        rm = ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT)
        rm = ttnn.pad(rm, [(0, 0), (0, 0), (0, T_pad - T), (0, 0)], 0.0)
        return ttnn.to_layout(rm, ttnn.TILE_LAYOUT)

    def _zero_u32_row(self, n: int) -> ttnn.Tensor:
        key = ("zero_u32", n)
        if key not in self._const_cache:
            self._const_cache[key] = self.mesh.upload(
                torch.zeros(1, n, dtype=torch.int32), ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
            )
        return self._const_cache[key]

    def forward(
        self, x: ttnn.Tensor, dense_w: ttnn.Tensor, idx: ttnn.Tensor | None = None, w: ttnn.Tensor | None = None
    ) -> ttnn.Tensor:
        """x [1,1,T,H] bf16, dense_w [1,1,T,E] fp32 (this row's tokens) -> [1,1,T,H] TP partial over
        the columns. ``idx`` / ``w`` [1,1,T,k] fp32 (the router's top-k ids and weights): the dispatch prefill
        path's routing (``PREFILL_EXPERTS == "dispatch"``, T > 32). Sparse layout: processed in 32-token blocks; a short last block (or T < 32, e.g.
        decode with few users) runs on its padded tile rows. Dense layout: all tokens in one pass. On a
        multi-row mesh the rows' tokens are gathered first and the expert-slice partials
        reduce-scattered back to each token's row."""
        if self.mesh.rows > 1:
            x = self.mesh.all_gather_rows(x, dim=2)
            dense_w = self.mesh.all_gather_rows(dense_w, dim=2)
            if idx is not None:
                idx = self.mesh.all_gather_rows(idx, dim=2)
                w = self.mesh.all_gather_rows(w, dim=2)
        y = self._local_experts(x, dense_w, idx, w)
        if self.mesh.rows > 1:
            ttnn.deallocate(x)
            ttnn.deallocate(dense_w)
            if idx is not None:
                ttnn.deallocate(idx)
                ttnn.deallocate(w)
            y = self.mesh.reduce_scatter_rows(y, dim=2)
        return y

    def _local_experts(
        self, x: ttnn.Tensor, dense_w: ttnn.Tensor, idx: ttnn.Tensor | None = None, w: ttnn.Tensor | None = None
    ) -> ttnn.Tensor:
        """The chip's expert slices over the (gathered) tokens x [1,1,T,H] with the routing dense_w [1,1,T,E]
        (``idx`` / ``w`` [1,1,T,k]: the router's top-k lists, needed by the dispatch / regions prefill paths):
        the layout's block path (sparse 32-token blocks / dense / stream / batched / dispatch / regions)
        -> [1,1,T,H] partial."""
        T = x.shape[-2]
        block = self._block_stream if self.stream else self._block
        if self.dense:
            y = self._block_dense(x, dense_w)
        elif self._dispatch is not None and (T > TILE or self.w13 is None):
            # the per-expert lists serve prefill (and decode too when they replaced the stacked tensors)
            assert (
                idx is not None and w is not None
            ), "the dispatch / regions prefill paths need the router's top-k lists"
            if self._dispatch["fabric"]:
                y = self._forward_dispatch(x, idx, w)
            else:
                y = self._forward_regions_chunked(x, dense_w, idx, w)
        elif self.prefill_experts == "batched" and T > TILE and not self.stream:
            y = self._forward_batched(x, dense_w)
        elif self.stream and self.stream_dispatch and TILE < T <= 256 and T % TILE == 0:
            y = self._block_dispatch(x, dense_w)
        elif T <= TILE:
            y = block(x, dense_w)
            if self.stream and T < TILE:
                y = ttnn.slice(y, [0, 0, 0, 0], [1, 1, T, self.H])
        else:
            outs = []
            for t0 in range(0, T, TILE):
                t1 = min(t0 + TILE, T)
                xb = ttnn.slice(x, [0, 0, t0, 0], [1, 1, t1, self.H])
                wb = ttnn.slice(dense_w, [0, 0, t0, 0], [1, 1, t1, self.E])
                outs.append(block(xb, wb))
            y = ttnn.concat(outs, dim=-2)
            for o in outs:
                ttnn.deallocate(o)
        return y


class TtEPExperts(LightweightModule):
    """Expert parallelism over the whole mesh (doc/mesh/orchestration_v2.md, after the DeepSeek-V3 Galaxy
    decode): chip ``d = r * cols + c`` owns the ``E_local = E / D`` FULL-width experts ``[d E_local, (d+1) E_local)``.
    The rows' tokens (replicated over the TP columns; each column holds a quarter of the experts) reach the
    experts of their column through ``all_to_all_dispatch`` along the mesh rows (axis 0), the streaming dispatch
    op in EP mode (``custom_ops/expert_stream``: the chip's active experts only, routing from the top-k index /
    weight lists, the weights applied inside) runs on the dispatched rows, ``all_to_all_combine`` (local reduce)
    returns every token's weighted sum to its row; the result is the column's partial (the caller all-reduces
    over the columns). On one chip (``D = 1``) the dispatch / combine are the identity and skipped.

    Mesh path (``forward`` with ``D > 1``, validated on a (2,4) block under FABRIC_1D: tests/test_mesh_moe.py with
    GLM53_MOE_LAYOUT=ep, tests/test_a2a_perf.py::test_dispatch_combine_golden): the routing weights ride as a K-wide
    bf16 tail of every token row (``[x | w]``, one dispatch instead of two), the ids as the dispatch's uint16
    metadata (exact: bf16 tiles cannot hold ids 257..287), both read directly by the kernels (``ep_rm``); the
    combine is ``local_reduce`` (one row per (device, token)) into a zeroed ``[K, T]`` slot tensor (it writes the
    slot of the device's first expert in the token's list and leaves the others untouched) summed over K. The
    fabric must be a 1D one (``GLM53_FABRIC=1d``, the default once GLM53_MOE_LAYOUT=ep: the dispatch writer sends 1D
    hop packets and the Blackhole CI covers these ops under FABRIC_1D only)."""

    @staticmethod
    def cache_files(cache_dir: Path | str, mesh: GLMMesh, expert_dtype, expert_down_dtype) -> dict[str, Path]:
        tag = f"mesh{mesh.rows}x{mesh.cols}_ep"
        cache_dir = Path(cache_dir)
        return {
            "w13": cache_dir / f"experts_w13_{expert_dtype.name}_{tag}.tensorbin",
            "w2": cache_dir / f"experts_w2_{expert_down_dtype.name}_{tag}.tensorbin",
        }

    @classmethod
    def cache_complete(cls, cache_dir, mesh: GLMMesh, expert_dtype, expert_down_dtype) -> bool:
        if cache_dir is None:
            return False
        return all(f.is_file() for f in cls.cache_files(cache_dir, mesh, expert_dtype, expert_down_dtype).values())

    def __init__(
        self,
        device,
        cfg: GLM53FlashConfig,
        gate_up: torch.Tensor | None,  # [E, 2I, H] (HF stacked), None = from the cache
        down: torch.Tensor | None,  # [E, H, I]
        *,
        expert_dtype=ttnn.bfloat8_b,
        expert_down_dtype=ttnn.bfloat8_b,
        mesh: GLMMesh | None = None,
        weight_cache_path: Path | str | None = None,
        picks: int = MAX_PICKS,
    ):
        self.device = device
        self.mesh = mesh if mesh is not None else single_chip(device)
        self.cfg = cfg
        self.E, self.H, self.I = cfg.n_routed_experts, cfg.hidden_size, cfg.moe_intermediate_size
        self.picks = picks
        self.expert_dtype, self.expert_down_dtype = expert_dtype, expert_down_dtype
        R, C = self.mesh.rows, self.mesh.cols
        self.D = R * C
        assert self.E % self.D == 0, (self.E, self.D)
        self.E_local = self.E // self.D
        E_local, H, I = self.E_local, self.H, self.I
        files = (
            self.cache_files(weight_cache_path, self.mesh, expert_dtype, expert_down_dtype)
            if weight_cache_path is not None
            else {"w13": None, "w2": None}
        )
        if gate_up is None or down is None:
            if not self.cache_complete(weight_cache_path, self.mesh, expert_dtype, expert_down_dtype):
                raise ValueError("EP experts without host tensors need a complete weight cache")

        def chip_experts(r: int, c: int):
            d = r * C + c
            es = slice(d * E_local, (d + 1) * E_local)
            g = gate_up[es, :I, :].transpose(1, 2)  # [E_local, H, I]
            u = gate_up[es, I:, :].transpose(1, 2)
            return torch.cat([g, u], dim=-1).float(), down[es].transpose(1, 2).float()  # [E_local, I, H]

        def make_w13() -> torch.Tensor:  # [R, C, rows13, 256]
            return torch.stack(
                [torch.stack([gate_up_stream_layout(chip_experts(r, c)[0]) for c in range(C)]) for r in range(R)]
            ).contiguous()

        def make_w2() -> torch.Tensor:
            return torch.stack(
                [torch.stack([down_stream_layout(chip_experts(r, c)[1]) for c in range(C)]) for r in range(R)]
            ).contiguous()

        rows13 = E_local * 2 * (I // TILE // 8) * (H // TILE) * TILE
        rows2 = E_local * (H // TILE // 8) * (I // TILE) * TILE
        self.w13 = self.mesh.upload_cached(
            make_w13,
            expert_dtype,
            files["w13"],
            rows=0,
            cols=1,
            memory_config=stream_memory_config(rows13),
            expect_shape=(R, C, rows13, 256),
        )
        self.w2 = self.mesh.upload_cached(
            make_w2,
            expert_down_dtype,
            files["w2"],
            rows=0,
            cols=1,
            memory_config=stream_memory_config(rows2),
            expect_shape=(R, C, rows2, 256),
        )
        # the chip's expert base id (one uint16 page per chip) and the one-hot expert -> device mapping
        base = torch.zeros(R, C, 1, 16, dtype=torch.int32)
        for r in range(R):
            for c in range(C):
                base[r, c, 0, 0] = (r * C + c) * E_local
        self.base = self.mesh.upload(base, ttnn.uint16, rows=0, cols=1, layout=ttnn.ROW_MAJOR_LAYOUT)
        self.mapping = (
            None
            if self.D == 1
            else self.mesh.upload(
                torch.eye(self.D, dtype=torch.int32).repeat_interleave(E_local, dim=0).reshape(1, 1, self.E, self.D),
                ttnn.uint16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
        )
        self.ops = {}
        self._pad = {}
        self._zero_slots = {}
        self.act_l1 = os.environ.get("GLM53_STREAM_ACT_L1", "1") == "1"
        # GLM53_EP_KSUM=fast (default): the K-slot sum as one fast_reduce_nc kernel (+ a slice below 32 rows); sum:
        # the generic ttnn.sum (permute + reduce + permute). Equal results (PCC 1.000000 / 0.999999 at 4 / 32 users)
        self.ksum_fast = os.environ.get("GLM53_EP_KSUM", "fast") == "fast"
        # the DeepSeek-V3 decode call pattern (models/demos/deepseek_v3/tt/moe.py, its decode config): dispatch and
        # combine along the mesh rows, L1 outputs, the links chosen by the op. The topology is derived from the
        # active fabric by the op (``topology=None``: FABRIC_1D -> Linear, FABRIC_2D -> Mesh, the torus -> Ring on a
        # wrapped axis): the writer kernel picks its 1D or 2D fabric send from it, and a Linear topology under a 2D
        # fabric (the first mesh run) sends 1D packets into 2D routers and hangs. GLM53_A2A_TOPOLOGY=linear|ring|mesh|torus
        # overrides for experiments.
        topo = {
            "linear": ttnn.Topology.Linear,
            "ring": ttnn.Topology.Ring,
            "mesh": ttnn.Topology.Mesh,
            "torus": ttnn.Topology.Torus,
            "": None,
        }[os.environ.get("GLM53_A2A_TOPOLOGY", "").lower()]
        # links per axis: GLM53_A2A_LINKS, default 0 = the op's own choice (the control plane's maximum: 2 on a (2,4)
        # block, where it cut the combine from 494 to 310 us at 32 users/row with identical results; 1 is the
        # Blackhole CI configuration of these ops)
        links = int(os.environ.get("GLM53_A2A_LINKS", "0")) or None
        self.a2a_kwargs = dict(cluster_axis=0, memory_config=ttnn.L1_MEMORY_CONFIG, num_links=links, topology=topo)
        # GLM53_EP_FORCE_A2A=1: run the dispatch / combine even on one chip (D = 1: the identity) to exercise the
        # mesh plumbing before a block is available
        self.force_a2a = os.environ.get("GLM53_EP_FORCE_A2A", "0") == "1"
        if self.force_a2a and self.mapping is None:
            self.mapping = self.mesh.upload(
                torch.ones(1, 1, self.E, 1, dtype=torch.int32), ttnn.uint16, layout=ttnn.ROW_MAJOR_LAYOUT
            )

    def _op(self, T: int, rm: bool = False) -> ExpertDispatch:
        if (T, rm) not in self.ops:
            self.ops[(T, rm)] = ExpertDispatch(
                self.device,
                self.E_local,
                self.H,
                self.I,
                self.H,
                T,
                groups=int(os.environ.get("GLM53_STREAM_GROUPS", "12")),
                down_readers=int(os.environ.get("GLM53_DISPATCH_DOWN_READERS", "8")),
                ep=True,
                w13_dtype=self.expert_dtype,
                units=not rm and os.environ.get("GLM53_STREAM_UNITS", "0") == "1",  # pair units (36-expert chips)
                ep_rm=rm,
            )
        return self.ops[(T, rm)]

    def _index_tiles(self, idx: ttnn.Tensor, w: ttnn.Tensor) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """idx / w [1,1,T,K] (fp32 TILE: integral ids, weights) -> the [1,1,T,32] BF16 index / weight tiles the EP
        op reads (columns >= K padded with IDX_PAD / 0)."""
        T, K = idx.shape[-2], idx.shape[-1]
        if T not in self._pad:
            self._pad[T] = (
                ttnn.full(
                    (1, 1, T, TILE - K),
                    float(IDX_PAD),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.device,
                ),
                ttnn.zeros((1, 1, T, TILE - K), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device),
            )
        pad_i, pad_w = self._pad[T]
        it = ttnn.concat([ttnn.typecast(idx, ttnn.bfloat16), pad_i], dim=-1)
        wt = ttnn.concat([ttnn.typecast(w, ttnn.bfloat16), pad_w], dim=-1)
        return it, wt

    def forward(self, x: ttnn.Tensor, w: ttnn.Tensor, idx: ttnn.Tensor) -> ttnn.Tensor:
        """x [1,1,T,H] bf16 (this row's tokens, replicated over the columns) -> [1,1,T,H]: this column's expert
        partial for the row's tokens. One chip: w / idx [1,1,T,K] fp32 (the router's top-k weights / global ids).
        Mesh: w [1,1,T,K] bf16, idx [1,1,T,K] uint32 (``TtRouter.forward(ep_rows=True)``)."""
        T = x.shape[-2]
        if self.D == 1 and not self.force_a2a:
            it, wt = self._index_tiles(idx, w)
            x_rm = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
            op = self._op(T)
            act, table, out = op.alloc(self.device, act_l1=self.act_l1)
            op.forward(x_rm, wt, it, self.w13, self.w2, act, table, out, base=self.base)
            for t in (x_rm, it, wt, act, table):
                ttnn.deallocate(t)
            return out
        # ---- mesh: three stages (timed separately by tests/test_a2a_perf.py::test_moe_decode_components)
        tok, meta = self.dispatch_rows(x, w, idx)
        out = self.run_experts(tok, meta)
        y = self.combine_rows(out, meta, T)
        for t in (tok, meta, out):
            ttnn.deallocate(t)
        return y

    def dispatch_rows(self, x: ttnn.Tensor, w: ttnn.Tensor, idx: ttnn.Tensor) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """Stage 1 (6 device ops + the dispatch): the [x | w] rows (the bf16 routing weights as a K-wide tail of
        every token row) and the uint16 ids in ROW_MAJOR L1, ``all_to_all_dispatch`` along the mesh rows ->
        (tok [1, Td, 1, H + K] = the rows of every token routed to one of this chip's experts, Td = rows * T,
        garbage rows for the others; meta [1, Td, 1, K] = every row's ids)."""
        T, K, H = x.shape[-2], idx.shape[-1], self.H
        l1 = ttnn.L1_MEMORY_CONFIG
        x_rm = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT, memory_config=l1)
        w_rm = ttnn.to_layout(w, ttnn.ROW_MAJOR_LAYOUT, memory_config=l1)  # [1,1,T,K] bf16
        xw = ttnn.concat([x_rm, w_rm], dim=-1, memory_config=l1)  # [1,1,T,H+K]
        ttnn.deallocate(x_rm)
        ttnn.deallocate(w_rm)
        xw = ttnn.reshape(xw, (T, 1, 1, H + K))
        idx16 = ttnn.typecast(idx, ttnn.uint16)
        idx_rm = ttnn.reshape(ttnn.to_layout(idx16, ttnn.ROW_MAJOR_LAYOUT, memory_config=l1), (T, 1, 1, K))
        ttnn.deallocate(idx16)
        tok, meta = ttnn.all_to_all_dispatch(xw, idx_rm, self.mapping, **self.a2a_kwargs)
        ttnn.deallocate(xw)
        ttnn.deallocate(idx_rm)
        return tok, meta

    def run_experts(self, tok: ttnn.Tensor, meta: ttnn.Tensor) -> ttnn.Tensor:
        """Stage 2 (the two expert ops): this chip's active experts over the dispatched rows, the routing read from
        the metadata rows and the row tails (``ExpertDispatch(ep=True, ep_rm=True)``), tile-padded to a multiple
        of 32 rows -> out [1, 1, Td, H] TILE (the weighted sum over the chip's experts per dispatched row)."""
        Td = tok.shape[1]
        T_pad = -(-Td // TILE) * TILE
        op = self._op(T_pad, rm=True)
        act, table, out = op.alloc(self.device, act_l1=self.act_l1, t_valid=Td)
        op.forward(tok, tok, meta, self.w13, self.w2, act, table, out, base=self.base, t_valid=Td)
        ttnn.deallocate(act)
        ttnn.deallocate(table)
        return out

    def combine_rows(self, out: ttnn.Tensor, meta: ttnn.Tensor, T: int) -> ttnn.Tensor:
        """Stage 3 (4 device ops + the combine): the contributions as rows, ``all_to_all_combine(local_reduce)``
        into a zeroed [K, T, 1, H] slot tensor (slot k of token t = the contribution of the device owning its k-th
        expert, untouched slots stay 0), tilized, summed over K -> [1, 1, T, H] (this column's partial)."""
        K, H = meta.shape[-1], self.H
        l1 = ttnn.L1_MEMORY_CONFIG
        Td = out.shape[-2]
        out_rm = ttnn.reshape(ttnn.to_layout(out, ttnn.ROW_MAJOR_LAYOUT, memory_config=l1), (1, Td, 1, H))
        # the zeroed slot tensor as a device copy of a persistent zero constant (ttnn.zeros is a host write: not
        # allowed inside a trace); the constant is made on the first (eager) call for this T
        if (K, T) not in self._zero_slots:
            self._zero_slots[(K, T)] = ttnn.zeros(
                (K, T, 1, H), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device, memory_config=l1
            )
        slots = ttnn.clone(self._zero_slots[(K, T)], memory_config=l1)
        y = ttnn.all_to_all_combine(
            out_rm, meta, self.mapping, local_reduce=True, output_tensor=slots, **self.a2a_kwargs
        )  # [K, T, 1, H]
        ttnn.deallocate(out_rm)
        y_t = ttnn.to_layout(ttnn.reshape(y, (K, 1, T, H)), ttnn.TILE_LAYOUT)
        ttnn.deallocate(y)
        if self.ksum_fast:  # one kernel over dim 0 (ttnn.sum over dim 0 is permute + reduce + permute)
            y_sum = ttnn.experimental.fast_reduce_nc(y_t, dims=[0], output=None, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            if y_sum.shape[-2] != T:  # the kernel reports the tile-padded row count (a view cannot shrink it)
                y_sum = ttnn.slice(y_sum, [0, 0, 0, 0], [1, 1, T, H])
        else:
            y_sum = ttnn.sum(y_t, dim=0, keepdim=True)  # [1,1,T,H]
        ttnn.deallocate(y_t)
        return y_sum


class TtMoE(LightweightModule):
    """Router + routed experts + shared expert (HF Glm5NextTextMoE)."""

    def __init__(
        self,
        device,
        cfg: GLM53FlashConfig,
        mlp_state_dict: dict[str, torch.Tensor],
        *,
        weight_dtype=ttnn.bfloat8_b,
        expert_dtype=ttnn.bfloat8_b,
        expert_down_dtype=ttnn.bfloat8_b,
        mesh: GLMMesh | None = None,
        weight_cache_path: Path | str | None = None,
        dense_experts: bool | None = None,
        stream_experts: bool | None = None,
        layout: str | None = None,  # "hybrid" (default) or "ep" (GLM53_MOE_LAYOUT)
    ):
        """``mlp_state_dict``: raw checkpoint tensors of ``mlp.`` (prefix stripped), experts as
        ``experts.N.{gate,up,down}_proj.weight`` or pre-stacked ``experts.gate_up_proj/down_proj``,
        or absent when the expert cache under ``weight_cache_path`` is complete (``expert_cache_complete``).
        ``weight_dtype`` is the shared expert's format; the routed experts take ``expert_dtype``
        (gate/up) and ``expert_down_dtype`` (down). On a mesh (``tt/mesh.py``) the routed experts
        use the hybrid placement, the shared expert is TP over the columns and the two partials
        share one all-reduce. ``dense_experts`` selects the dense expert layout
        (``TtRoutedExperts(dense=True)``); ``None`` reads ``GLM53_DENSE_EXPERTS`` (unset / 0 = sparse)."""
        self.cfg = cfg
        if dense_experts is None:
            dense_experts = os.environ.get("GLM53_DENSE_EXPERTS", "0") == "1"
        if stream_experts is None:
            stream_experts = os.environ.get("GLM53_STREAM_EXPERTS", "0") == "1"
        if layout is None:
            layout = os.environ.get("GLM53_MOE_LAYOUT", "hybrid")
        assert layout in ("hybrid", "ep"), layout
        self.layout = layout
        self.mesh = mesh if mesh is not None else single_chip(device)
        E, I, H = cfg.n_routed_experts, cfg.moe_intermediate_size, cfg.hidden_size
        if "experts.gate_up_proj" in mlp_state_dict:
            gate_up, down = mlp_state_dict["experts.gate_up_proj"], mlp_state_dict["experts.down_proj"]
        elif "experts.0.gate_proj.weight" not in mlp_state_dict:
            gate_up = down = None  # cached experts (TtRoutedExperts checks the cache is complete)
        else:
            gate_up = torch.stack(
                [
                    torch.cat(
                        [
                            mlp_state_dict[f"experts.{e}.gate_proj.weight"],
                            mlp_state_dict[f"experts.{e}.up_proj.weight"],
                        ],
                        dim=0,
                    )
                    for e in range(E)
                ]
            )
            down = torch.stack([mlp_state_dict[f"experts.{e}.down_proj.weight"] for e in range(E)])
        self.router = TtRouter(
            device, cfg, mlp_state_dict["gate.weight"], mlp_state_dict["gate.e_score_correction_bias"], mesh=self.mesh
        )
        if layout == "ep":
            self.experts = TtEPExperts(
                device,
                cfg,
                gate_up,
                down,
                expert_dtype=expert_dtype,
                expert_down_dtype=expert_down_dtype,
                mesh=self.mesh,
                weight_cache_path=weight_cache_path,
            )
        else:
            self.experts = TtRoutedExperts(
                device,
                cfg,
                gate_up,
                down,
                expert_dtype=expert_dtype,
                expert_down_dtype=expert_down_dtype,
                mesh=self.mesh,
                weight_cache_path=weight_cache_path,
                dense=dense_experts,
                stream=stream_experts,
            )
        self.shared = TtClampedSwiGLU(
            device,
            mlp_state_dict["shared_experts.gate_proj.weight"],
            mlp_state_dict["shared_experts.up_proj.weight"],
            mlp_state_dict["shared_experts.down_proj.weight"],
            cfg.swiglu_limit,
            weight_dtype=weight_dtype,
            mesh=self.mesh,
        )

    @staticmethod
    def expert_cache_complete(
        cache_dir,
        mesh: GLMMesh,
        expert_dtype=ttnn.bfloat8_b,
        expert_down_dtype=ttnn.bfloat8_b,
        dense_experts=None,
        stream_experts=None,
    ):
        if dense_experts is None:
            dense_experts = os.environ.get("GLM53_DENSE_EXPERTS", "0") == "1"
        if stream_experts is None:
            stream_experts = os.environ.get("GLM53_STREAM_EXPERTS", "0") == "1"
        if os.environ.get("GLM53_MOE_LAYOUT", "hybrid") == "ep":
            return TtEPExperts.cache_complete(cache_dir, mesh, expert_dtype, expert_down_dtype)
        return TtRoutedExperts.cache_complete(
            cache_dir, mesh, expert_dtype, expert_down_dtype, dense_experts, stream_experts
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """x [1,1,T,H] bf16 (T multiple of 32; replicated over the TP columns) -> [1,1,T,H] replicated."""
        if self.layout == "ep" and self.experts.D > 1:
            w16, idx, _ = self.router(x, ep_rows=True)
            routed = self.experts(x, w16, idx)
            ttnn.deallocate(w16)
            ttnn.deallocate(idx)
        elif self.layout == "ep":
            w, idx, dense = self.router(x)
            routed = self.experts(x, w, idx)
            for t in (w, idx, dense):
                ttnn.deallocate(t)
        elif getattr(self.experts, "_dispatch", None) is not None:
            w, idx, dense = self.router(x)
            routed = self.experts(x, dense, idx=idx, w=w)
            for t in (w, idx, dense):
                if t.is_allocated():
                    ttnn.deallocate(t)
        else:
            _, _, dense = self.router(x, dense_only=True)
            routed = self.experts(x, dense)
            ttnn.deallocate(dense)
        shared = self.shared(x, reduce=False)
        y = ttnn.add(routed, shared, output_tensor=routed)
        ttnn.deallocate(shared)
        return self.mesh.all_reduce_cols(y)
