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


def pick_sparse_grid(device_grid, n: int) -> tuple[int, int]:
    """Largest rectangular core grid inside ``device_grid`` whose core count divides the output
    tile count Nt, so every core in the mcast receiver rectangle has work (the sparse matmul
    factory requires num_cores_with_work == receiver cores)."""
    Nt = math.ceil(n / TILE)
    best = (1, 1)
    for x in range(1, device_grid[0] + 1):
        for y in range(1, device_grid[1] + 1):
            if Nt % (x * y) == 0 and x * y > best[0] * best[1]:
                best = (x, y)
    return best


def sparse_matmul_config(device_grid, m: int, n: int, k: int, in0_block_w: int = 4, out_subblock_w: int = 1):
    """MatmulMultiCoreReuseMultiCast1DProgramConfig for sparse_matmul (mcast_in0=True), the
    gpt_oss builder's rules: per_core_N = ceil(Nt / cores); in0_block_w must divide Kt;
    out_subblock_w must divide out_block_w (= per_core_N); subblock area <= 8."""
    core_x, core_y = pick_sparse_grid(device_grid, n)
    num_cores = core_x * core_y
    Nt = math.ceil(n / TILE)
    Kt = math.ceil(k / TILE)
    per_core_N = (Nt + num_cores - 1) // num_cores
    if Kt % in0_block_w != 0:
        divisors = [d for d in range(2, in0_block_w + 1) if Kt % d == 0]
        in0_block_w = max(divisors) if divisors else Kt
    out_block_w = per_core_N
    out_subblock_w = max(w for w in range(1, min(out_subblock_w, 8) + 1) if out_block_w % w == 0)
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(core_x, core_y),
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=out_subblock_w,
        out_block_h=math.ceil(m / TILE),
        out_block_w=out_block_w,
        per_core_M=math.ceil(m / TILE),
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
        self.fp32 = _compute(device, ttnn.MathFidelity.HiFi4, packer_l1_acc=False)

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
            m = ttnn.max(choice, dim=-1, keepdim=True)  # [1,1,T,1]
            onehot = ttnn.eq(choice, m)  # exact fp32 compare; ties -> all tied columns (see below)
            # keep only the lowest tied column: first-match via cumulative sum along E
            csum = ttnn.cumsum(onehot, dim=-1)
            onehot = ttnn.multiply(onehot, ttnn.eq(csum, 1.0))
            ttnn.deallocate(csum)
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
    def cache_files(cache_dir: Path | str, mesh: GLMMesh, expert_dtype, expert_down_dtype) -> dict[str, Path]:
        """Tensorbin paths of the routed-expert device tensors for this placement / format."""
        tag = f"mesh{mesh.rows}x{mesh.cols}"
        cache_dir = Path(cache_dir)
        return {
            "w13": cache_dir / f"experts_w13_{expert_dtype.name}_{tag}.tensorbin",
            "w2": cache_dir / f"experts_w2_{expert_down_dtype.name}_{tag}.tensorbin",
        }

    @classmethod
    def cache_complete(cls, cache_dir: Path | str | None, mesh: GLMMesh, expert_dtype, expert_down_dtype) -> bool:
        """True when both expert tensorbins exist (the constructor then needs no expert host tensors)."""
        if cache_dir is None:
            return False
        return all(f.is_file() for f in cls.cache_files(cache_dir, mesh, expert_dtype, expert_down_dtype).values())

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
    ):
        """Hybrid placement (design 5.4): the experts are split over the mesh columns (chip column
        ``c`` holds experts ``[c*E/cols, (c+1)*E/cols)``) and every expert's intermediate over the mesh
        rows (row ``r`` holds the slice ``[r*I/rows, (r+1)*I/rows)``). ``forward`` gathers the rows'
        tokens, runs the local expert slices and reduce-scatters the partials back to the token's row;
        the result is still a TP partial over the columns (the caller all-reduces once).
        ``weight_cache_path``: per-layer tensorbin directory (``cache_files``); ``gate_up`` / ``down``
        may be ``None`` when ``cache_complete`` (the host tensors are then never needed)."""
        self.device = device
        self.mesh = mesh if mesh is not None else single_chip(device)
        self.cfg = cfg
        self.E, self.H, self.I = cfg.n_routed_experts, cfg.hidden_size, cfg.moe_intermediate_size
        self.limit = float(cfg.swiglu_limit)
        self.activation_dtype = activation_dtype
        self.expert_dtype = expert_dtype
        self.expert_down_dtype = expert_down_dtype
        ep, tpr = self.mesh.cols, self.mesh.rows
        assert self.E % ep == 0 and self.I % (tpr * TILE) == 0, (self.E, self.I, ep, tpr)
        self.E_local, self.I_local = self.E // ep, self.I // tpr
        files = (
            self.cache_files(weight_cache_path, self.mesh, expert_dtype, expert_down_dtype)
            if weight_cache_path is not None
            else {"w13": None, "w2": None}
        )
        if gate_up is None or down is None:
            if not self.cache_complete(weight_cache_path, self.mesh, expert_dtype, expert_down_dtype):
                raise ValueError("routed experts without host tensors need a complete weight cache")
        else:
            assert tuple(gate_up.shape) == (self.E, 2 * self.I, self.H) and tuple(down.shape) == (
                self.E,
                self.H,
                self.I,
            ), (gate_up.shape, down.shape)
        E, H, I = self.E, self.H, self.I

        def make_w13() -> torch.Tensor:
            gate = gate_up[:, :I, :].transpose(1, 2)  # [E, H, I]
            up = gate_up[:, I:, :].transpose(1, 2)
            # [1,E,H,2I] in tpr blocks of [gate_r | up_r] so a row split of the last dim keeps the pairs
            return interleave_blocks([gate, up], tpr, dim=-1).reshape(1, E, H, 2 * I)

        def make_w2() -> torch.Tensor:
            return down.transpose(1, 2).contiguous().reshape(1, E, I, H)  # [1,E,I,H]

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

    def forward(self, x: ttnn.Tensor, dense_w: ttnn.Tensor) -> ttnn.Tensor:
        """x [1,1,T,H] bf16, dense_w [1,1,T,E] fp32 (this row's tokens) -> [1,1,T,H] TP partial over
        the columns. Processed in 32-token blocks; a short last block (or T < 32, e.g. decode with few
        users) runs on its padded tile rows. On a multi-row mesh the rows' tokens are gathered first
        and the expert-slice partials reduce-scattered back to each token's row."""
        if self.mesh.rows > 1:
            x = self.mesh.all_gather_rows(x, dim=2)
            dense_w = self.mesh.all_gather_rows(dense_w, dim=2)
        T = x.shape[-2]
        if T <= TILE:
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
    ):
        """``mlp_state_dict``: raw checkpoint tensors of ``mlp.`` (prefix stripped), experts as
        ``experts.N.{gate,up,down}_proj.weight`` or pre-stacked ``experts.gate_up_proj/down_proj``,
        or absent when the expert cache under ``weight_cache_path`` is complete (``expert_cache_complete``).
        ``weight_dtype`` is the shared expert's format; the routed experts take ``expert_dtype``
        (gate/up) and ``expert_down_dtype`` (down). On a mesh (``tt/mesh.py``) the routed experts
        use the hybrid placement, the shared expert is TP over the columns and the two partials
        share one all-reduce."""
        self.cfg = cfg
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
    def expert_cache_complete(cache_dir, mesh: GLMMesh, expert_dtype=ttnn.bfloat8_b, expert_down_dtype=ttnn.bfloat8_b):
        return TtRoutedExperts.cache_complete(cache_dir, mesh, expert_dtype, expert_down_dtype)

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """x [1,1,T,H] bf16 (T multiple of 32; replicated over the TP columns) -> [1,1,T,H] replicated."""
        _, _, dense = self.router(x)
        routed = self.experts(x, dense)
        ttnn.deallocate(dense)
        shared = self.shared(x, reduce=False)
        y = ttnn.add(routed, shared, output_tensor=routed)
        ttnn.deallocate(shared)
        return self.mesh.all_reduce_cols(y)
