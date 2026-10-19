# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""MoE FFN for GLM-5.3-Flash on one device: router + active routed experts + shared expert.

HF ``Glm5NextTextMoE``:
    logits = x_fp32 @ W_gate^T ; s = sigmoid(logits) ; choice = s + e_score_correction_bias
    idx = top8(choice) ; w = s[idx] / sum(s[idx]) * 2.5                       (n_group = 1)
    y = sum_j w_j * expert_{idx_j}(x) + shared_expert(x)
    expert(x) = (silu(clamp(gate, max=10)) * clamp(up, -10, 10)) @ W2^T

Router: ``ttnn.experimental.deepseek_prefill.moe_grouped_topk`` single-group path (n_groups=1,
sigmoid, bias, renormalize, route_scale), fp32 logits as HF.

Experts: ``ttnn.sparse_matmul`` with a per-expert activity mask (union over the tokens of a
32-token block) so only the experts any token in the block routes to are computed; the dense
``[tokens, E]`` routing-weight matrix (built on device from the top-k ids by one-hot compare)
weights the per-expert outputs before the reduction over experts. nnz is left to be inferred at
runtime (a static nnz that disagrees with the mask deadlocks the op). Prefill processes the
sequence in 32-token blocks with the same code.

Weight layout for sparse_matmul: W13 ``[1, E, H, 2I]`` (gate columns then up), W2 ``[1, E, I, H]``.
"""

from __future__ import annotations

import math

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.glm53_flash.tt.config import GLM53FlashConfig
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
    def __init__(self, device, cfg: GLM53FlashConfig, gate_weight: torch.Tensor, correction_bias: torch.Tensor):
        self.device = device
        self.cfg = cfg
        self.E = cfg.n_routed_experts
        self.k = cfg.num_experts_per_tok
        self.w_gate = ttnn.from_torch(
            gate_weight.float().t().contiguous(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device
        )  # [H, E] fp32
        self.bias_row = correction_bias.float().reshape(1, 1, 1, self.E)
        self._bias: dict[int, ttnn.Tensor] = {}
        self.iota = ttnn.from_torch(
            torch.arange(self.E, dtype=torch.float32).reshape(1, 1, 1, self.E),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        self.fp32 = _compute(device, ttnn.MathFidelity.HiFi4, packer_l1_acc=False)

    def _bias_for(self, rows: int) -> ttnn.Tensor:
        # moe_grouped_topk wants bias with the scores' shape
        if rows not in self._bias:
            self._bias[rows] = ttnn.from_torch(
                self.bias_row.expand(1, 1, rows, self.E).contiguous(),
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
            )
        return self._bias[rows]

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
        choice = ttnn.add(scores, self._bias_for(T))
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
    def __init__(
        self,
        device,
        cfg: GLM53FlashConfig,
        gate_up: torch.Tensor,  # [E, 2I, H] (HF stacked: gate rows then up rows)
        down: torch.Tensor,  # [E, H, I]
        *,
        weight_dtype=ttnn.bfloat8_b,
        activation_dtype=ttnn.bfloat16,
    ):
        self.device = device
        self.cfg = cfg
        self.E, self.H, self.I = cfg.n_routed_experts, cfg.hidden_size, cfg.moe_intermediate_size
        self.limit = float(cfg.swiglu_limit)
        self.activation_dtype = activation_dtype
        assert tuple(gate_up.shape) == (self.E, 2 * self.I, self.H) and tuple(down.shape) == (self.E, self.H, self.I)
        w13 = gate_up.transpose(1, 2).contiguous().reshape(1, self.E, self.H, 2 * self.I)  # [1,E,H,2I]
        w2 = down.transpose(1, 2).contiguous().reshape(1, self.E, self.I, self.H)  # [1,E,I,H]
        self.w13 = ttnn.from_torch(w13, dtype=weight_dtype, layout=ttnn.TILE_LAYOUT, device=device)
        self.w2 = ttnn.from_torch(w2, dtype=weight_dtype, layout=ttnn.TILE_LAYOUT, device=device)
        grid = device.compute_with_storage_grid_size()
        self.grid = (grid.x, grid.y)
        self.compute = _compute(device)

    def _block(self, xb: ttnn.Tensor, dense_w: ttnn.Tensor) -> ttnn.Tensor:
        """One token block (<= 32 rows). xb [1,1,M,H] bf16, dense_w [1,1,M,E] fp32 -> [1,1,M,H]."""
        E, H, I = self.E, self.H, self.I
        M = xb.shape[-2]
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
        """x [1,1,T,H] bf16, dense_w [1,1,T,E] fp32 -> [1,1,T,H]. Processed in 32-token blocks;
        a short last block (or T < 32, e.g. decode with few users) runs on its padded tile rows."""
        T = x.shape[-2]
        if T <= TILE:
            return self._block(x, dense_w)
        outs = []
        for t0 in range(0, T, TILE):
            t1 = min(t0 + TILE, T)
            xb = ttnn.slice(x, [0, 0, t0, 0], [1, 1, t1, self.H])
            wb = ttnn.slice(dense_w, [0, 0, t0, 0], [1, 1, t1, self.E])
            outs.append(self._block(xb, wb))
        y = ttnn.concat(outs, dim=-2)
        for o in outs:
            ttnn.deallocate(o)
        return y


class TtMoE(LightweightModule):
    """Router + routed experts + shared expert (HF Glm5NextTextMoE)."""

    def __init__(
        self, device, cfg: GLM53FlashConfig, mlp_state_dict: dict[str, torch.Tensor], *, weight_dtype=ttnn.bfloat8_b
    ):
        """``mlp_state_dict``: raw checkpoint tensors of ``mlp.`` (prefix stripped), experts as
        ``experts.N.{gate,up,down}_proj.weight`` or pre-stacked ``experts.gate_up_proj/down_proj``."""
        self.cfg = cfg
        E, I, H = cfg.n_routed_experts, cfg.moe_intermediate_size, cfg.hidden_size
        if "experts.gate_up_proj" in mlp_state_dict:
            gate_up, down = mlp_state_dict["experts.gate_up_proj"], mlp_state_dict["experts.down_proj"]
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
            device, cfg, mlp_state_dict["gate.weight"], mlp_state_dict["gate.e_score_correction_bias"]
        )
        self.experts = TtRoutedExperts(device, cfg, gate_up, down, weight_dtype=weight_dtype)
        self.shared = TtClampedSwiGLU(
            device,
            mlp_state_dict["shared_experts.gate_proj.weight"],
            mlp_state_dict["shared_experts.up_proj.weight"],
            mlp_state_dict["shared_experts.down_proj.weight"],
            cfg.swiglu_limit,
            weight_dtype=weight_dtype,
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """x [1,1,T,H] bf16 (T multiple of 32) -> [1,1,T,H]."""
        _, _, dense = self.router(x)
        routed = self.experts(x, dense)
        ttnn.deallocate(dense)
        shared = self.shared(x)
        y = ttnn.add(routed, shared, output_tensor=routed)
        ttnn.deallocate(shared)
        return y
