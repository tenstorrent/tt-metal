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
from models.demos.glm53_flash.tt.config import GLM53FlashConfig
from models.demos.glm53_flash.tt.mesh import GLMMesh, interleave_blocks, single_chip
from models.demos.glm53_flash.tt.mlp import TtClampedSwiGLU

TILE = 32


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
    device_grid, m: int, n: int, k: int, in0_block_w: int | None = None, out_subblock_w: int | None = None
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
        self.tie_break = os.environ.get("GLM53_ROUTER_TIE_BREAK", "rank")  # rank (default) | cumsum (A/B)

    def forward(self, x: ttnn.Tensor) -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        """x [1,1,T,H] bf16 -> (weights [1,1,T,8] fp32, indices [1,1,T,8] fp32 (integral), dense [1,1,T,E] fp32).

        Exact fp32 selection (HF noaux_tc, n_group=1): ``choice = sigmoid(logits) + bias``, top-8 by
        iterative masked argmax, weights = unbiased scores at the selected experts, renormalized and
        scaled. The fused ``moe_grouped_topk`` kernel ranks with bf16 resolution (0.03 near the ~6
        biased scores), which swaps the 8th expert on ~20% of real tokens; the composite is exact.
        """
        T = x.shape[-2]
        logits = ttnn.linear(x, self.w_gate, dtype=ttnn.float32, compute_kernel_config=self.fp32)
        scores = ttnn.sigmoid(logits)  # fp32 [1,1,T,E]
        ttnn.deallocate(logits)
        choice = ttnn.add(scores, self.bias)
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
        cache_dir: Path | str, mesh: GLMMesh, expert_dtype, expert_down_dtype, dense: bool = False
    ) -> dict[str, Path]:
        """Tensorbin paths of the routed-expert device tensors for this placement / format / layout."""
        tag = f"mesh{mesh.rows}x{mesh.cols}" + ("_dense" if dense else "")
        cache_dir = Path(cache_dir)
        return {
            "w13": cache_dir / f"experts_w13_{expert_dtype.name}_{tag}.tensorbin",
            "w2": cache_dir / f"experts_w2_{expert_down_dtype.name}_{tag}.tensorbin",
        }

    @classmethod
    def cache_complete(
        cls, cache_dir: Path | str | None, mesh: GLMMesh, expert_dtype, expert_down_dtype, dense: bool = False
    ) -> bool:
        """True when both expert tensorbins exist (the constructor then needs no expert host tensors)."""
        if cache_dir is None:
            return False
        return all(
            f.is_file() for f in cls.cache_files(cache_dir, mesh, expert_dtype, expert_down_dtype, dense).values()
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
        ep, tpr = self.mesh.cols, self.mesh.rows
        assert self.E % ep == 0 and self.I % (tpr * TILE) == 0, (self.E, self.I, ep, tpr)
        self.E_local, self.I_local = self.E // ep, self.I // tpr
        files = (
            self.cache_files(weight_cache_path, self.mesh, expert_dtype, expert_down_dtype, dense)
            if weight_cache_path is not None
            else {"w13": None, "w2": None}
        )
        if gate_up is None or down is None:
            if not self.cache_complete(weight_cache_path, self.mesh, expert_dtype, expert_down_dtype, dense):
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

        def make_w13_dense() -> torch.Tensor:
            # [1, cols, H, rows * (2 * E_loc * I_loc)]: a column split of dim 1 and a row split of dim 3 hand chip
            # (r, c) [H, gates(e in c, slice r) | ups(e in c, slice r)], (expert, i) order inside each half
            def per_chip(t):  # [E, H, I] -> [cols, H, rows, E_loc * I_loc]
                return t.reshape(ep, E_loc, H, tpr, I_loc).permute(0, 2, 3, 1, 4).reshape(ep, H, tpr, E_loc * I_loc)

            gate = gate_up[:, :I, :].transpose(1, 2)
            up = gate_up[:, I:, :].transpose(1, 2)
            w = torch.cat([per_chip(gate), per_chip(up)], dim=-1)  # [cols, H, rows, 2 * E_loc * I_loc]
            return w.reshape(1, ep, H, tpr * 2 * E_loc * I_loc).contiguous()

        def make_w2_dense() -> torch.Tensor:
            # [1, cols, rows * (E_loc * I_loc), H]: chip (r, c) gets rows (e in c, i in slice r) of every W2
            w2 = down.transpose(1, 2).reshape(ep, E_loc, tpr, I_loc, H).permute(0, 2, 1, 3, 4)
            return w2.reshape(1, ep, tpr * E_loc * I_loc, H).contiguous()

        if dense:
            self.w13 = self.mesh.upload_cached(
                make_w13_dense,
                expert_dtype,
                files["w13"],
                rows=3,
                cols=1,
                expect_shape=(1, ep, H, tpr * 2 * E_loc * I_loc),
            )
            self.w2 = self.mesh.upload_cached(
                make_w2_dense,
                expert_down_dtype,
                files["w2"],
                rows=2,
                cols=1,
                expect_shape=(1, ep, tpr * E_loc * I_loc, H),
            )
            # routing weight expansion: [M, E_loc] @ [E_loc, E_loc * I_loc] repeats a token's weight for expert e
            # over e's I_loc intermediate columns (0/1 entries: exact in bf16)
            self.expand = self.mesh.upload(
                torch.eye(E_loc, dtype=torch.bfloat16)
                .repeat_interleave(I_loc, dim=1)
                .reshape(1, 1, E_loc, E_loc * I_loc),
                ttnn.bfloat16,
            )
        else:
            self.w13 = self.mesh.upload_cached(
                make_w13, expert_dtype, files["w13"], rows=3, cols=1, expect_shape=(1, E, H, 2 * I)
            )
            self.w2 = self.mesh.upload_cached(
                make_w2, expert_down_dtype, files["w2"], rows=2, cols=1, expect_shape=(1, E, I, H)
            )
        # column c's expert selector: the [E, E_local] block of the identity for its experts
        self.select = (
            None if ep == 1 else self.mesh.upload(torch.eye(self.E, dtype=torch.float32), ttnn.float32, cols=-1)
        )
        grid = device.compute_with_storage_grid_size()
        self.grid = (grid.x, grid.y)
        self.compute = _compute(device)
        self.fp32 = _compute(device, ttnn.MathFidelity.HiFi4, packer_l1_acc=False)

    def _block(self, xb: ttnn.Tensor, dense_w: ttnn.Tensor) -> ttnn.Tensor:
        """One token block (<= 32 rows). xb [1,1,M,H] bf16, dense_w [1,1,M,E] fp32 -> [1,1,M,H]
        (this chip's experts and intermediate slice only)."""
        E, H, I = self.E_local, self.H, self.I_local
        M = xb.shape[-2]
        if self.select is not None:  # keep the routing weights of this column's experts: [1,1,M,E_local]
            dense_w = ttnn.matmul(dense_w, self.select, dtype=ttnn.float32, compute_kernel_config=self.fp32)
        # per-expert activity mask over the block (nonzero = active); bf16 ROW_MAJOR [1,1,1,E]
        active = ttnn.sum(dense_w, dim=-2, keepdim=True)  # [1,1,1,E] fp32
        sparsity = ttnn.to_layout(ttnn.typecast(active, ttnn.bfloat16), ttnn.ROW_MAJOR_LAYOUT)
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
        gate = ttnn.clamp(gate, min=None, max=self.limit, output_tensor=gate)
        up = ttnn.clamp(up, min=-self.limit, max=self.limit, output_tensor=up)
        act = ttnn.silu(gate, output_tensor=gate)
        h = ttnn.multiply(act, up, output_tensor=act)  # [1,E,M,I]
        ttnn.deallocate(up)
        # weight each (expert, token) row by its routing weight before the down projection
        w_cols = ttnn.transpose(ttnn.typecast(dense_w, self.activation_dtype), -2, -1)  # [1,1,E,M]
        w_cols = ttnn.reshape(w_cols, [1, E, M, 1])
        h = ttnn.multiply(h, w_cols, output_tensor=h)
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
        """All gathered tokens at once through the dense layout. x [1,1,T,H] bf16, dense_w [1,1,T,E]
        fp32 -> [1,1,T,H] (this chip's expert slices, summed over its experts)."""
        E, H, I = self.E_local, self.H, self.I_local
        T = x.shape[-2]
        if self.select is not None:
            dense_w = ttnn.matmul(dense_w, self.select, dtype=ttnn.float32, compute_kernel_config=self.fp32)
        gu = ttnn.matmul(
            x,
            self.w13,
            program_config=dense_matmul_config(self.grid, T, 2 * E * I, H),
            dtype=self.activation_dtype,
            compute_kernel_config=self.compute,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )  # [1,1,T, 2*E*I]: all gates, then all ups, (expert, i) order
        gate = ttnn.slice(gu, [0, 0, 0, 0], [1, 1, T, E * I])
        up = ttnn.slice(gu, [0, 0, 0, E * I], [1, 1, T, 2 * E * I])
        ttnn.deallocate(gu)
        gate = ttnn.clamp(gate, min=None, max=self.limit, output_tensor=gate)
        up = ttnn.clamp(up, min=-self.limit, max=self.limit, output_tensor=up)
        act = ttnn.silu(gate, output_tensor=gate)
        h = ttnn.multiply(act, up, output_tensor=act)  # [1,1,T,E*I]
        ttnn.deallocate(up)
        w_exp = ttnn.matmul(
            ttnn.typecast(dense_w, self.activation_dtype),
            self.expand,
            dtype=self.activation_dtype,
            compute_kernel_config=self.compute,
        )  # [1,1,T,E*I]
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

    def forward(self, x: ttnn.Tensor, dense_w: ttnn.Tensor) -> ttnn.Tensor:
        """x [1,1,T,H] bf16, dense_w [1,1,T,E] fp32 (this row's tokens) -> [1,1,T,H] TP partial over
        the columns. Sparse layout: processed in 32-token blocks; a short last block (or T < 32, e.g.
        decode with few users) runs on its padded tile rows. Dense layout: all tokens in one pass. On a
        multi-row mesh the rows' tokens are gathered first and the expert-slice partials
        reduce-scattered back to each token's row."""
        if self.mesh.rows > 1:
            x = self.mesh.all_gather_rows(x, dim=2)
            dense_w = self.mesh.all_gather_rows(dense_w, dim=2)
        T = x.shape[-2]
        if self.dense:
            y = self._block_dense(x, dense_w)
        elif T <= TILE:
            y = self._block(x, dense_w)
        else:
            outs = []
            for t0 in range(0, T, TILE):
                t1 = min(t0 + TILE, T)
                xb = ttnn.slice(x, [0, 0, t0, 0], [1, 1, t1, self.H])
                wb = ttnn.slice(dense_w, [0, 0, t0, 0], [1, 1, t1, self.E])
                outs.append(self._block(xb, wb))
            y = ttnn.concat(outs, dim=-2)
            for o in outs:
                ttnn.deallocate(o)
        if self.mesh.rows > 1:
            ttnn.deallocate(x)
            ttnn.deallocate(dense_w)
            y = self.mesh.reduce_scatter_rows(y, dim=2)
        return y


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
        cache_dir, mesh: GLMMesh, expert_dtype=ttnn.bfloat8_b, expert_down_dtype=ttnn.bfloat8_b, dense_experts=None
    ):
        if dense_experts is None:
            dense_experts = os.environ.get("GLM53_DENSE_EXPERTS", "0") == "1"
        return TtRoutedExperts.cache_complete(cache_dir, mesh, expert_dtype, expert_down_dtype, dense_experts)

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """x [1,1,T,H] bf16 (T multiple of 32; replicated over the TP columns) -> [1,1,T,H] replicated."""
        _, _, dense = self.router(x)
        routed = self.experts(x, dense)
        ttnn.deallocate(dense)
        shared = self.shared(x, reduce=False)
        y = ttnn.add(routed, shared, output_tensor=routed)
        ttnn.deallocate(shared)
        return self.mesh.all_reduce_cols(y)
