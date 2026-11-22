# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0

"""Inkling-2-Mini MoE block, batch-1 decode on one Blackhole card (1x1 mesh, no CCL).

HF reference: ``Inkling2SparseMoeBlock`` (transformers >= 5.24).
  router:  logits = x @ W (fp32); s = sigmoid(logits); idx = topk(s + e_score_correction_bias, 8)
           w = s[idx] / (sum + 1e-20) * route_scale(8)
  expert:  [g | u] = x @ W_gate_up[e]; h = 4 tanh(g/4) * sigmoid(g) * 25 tanh(u/25); y_e = h @ W_down[e]
  output:  sum_e w_e * y_e

Includes the layer's post_attention_layernorm (``mlp_norm``) at the input.
Device path (all on device, no host sync, trace-safe):
  rms_norm -> linear (fp32 out) -> moe_grouped_topk (single group: sigmoid, +bias, top-k, unbiased gather, norm, *scale,
  fp32 inside) -> sparse_matmul gate_up -> softcapped SwiGLU ->
  sparse_matmul down (both indexed by the top-8 ids: only those experts are read, outputs compact [8, N])
  -> routing multiply + sum over the 8 slots. Weight names/shapes follow design.md section 4.
"""

import os

import torch

import ttnn

HIDDEN = 2304
INTER = 1152
NUM_EXPERTS = 128
TOP_K = 8
ROUTE_SCALE = 8.0
GATE_SOFTCAP = 4.0
UP_SOFTCAP = 25.0

RMS_EPS = 1e-6
ROUTER_SPLIT = 9  # K chunks of the router GEMM (256 each)

WEIGHT_NAMES = ("mlp_norm", "router.w", "router.bias", "experts.w_gate_up", "experts.w_down")


def prepare_moe_weights(norm_w, router_w, router_bias, gate_up_proj, down_proj, expert_dtype=ttnn.bfloat4_b):
    """HF tensors -> host ttnn tensors in the section-4 cache form (what the converter dumps).

    norm_w [H] (post_attention_layernorm), router_w [E, H] fp32, router_bias [E] fp32, gate_up_proj [E, 2I, H] (gate rows first, already
    de-interleaved), down_proj [E, H, I]."""
    bias = torch.zeros(1, 1, 32, NUM_EXPERTS, dtype=torch.float32)
    bias[0, 0, 0] = router_bias.float()
    host = lambda t, dtype: ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT)  # noqa: E731
    return {
        "mlp_norm": ttnn.from_torch(norm_w.bfloat16().reshape(1, 1, HIDDEN // 32, 32), dtype=ttnn.bfloat16),
        "router.w": host(router_w.float().T.reshape(1, 1, HIDDEN, NUM_EXPERTS).contiguous(), ttnn.float32),
        "router.bias": host(bias, ttnn.float32),
        "experts.w_gate_up": host(gate_up_proj.transpose(1, 2).unsqueeze(0).contiguous(), expert_dtype),
        "experts.w_down": host(down_proj.transpose(1, 2).unsqueeze(0).contiguous(), expert_dtype),
    }


def load_moe_weights(layer_dir):
    """Host tensors of one layer from the v1 cache (``$INKLING_CACHE/layers/LNN``)."""
    return {name: ttnn.load_tensor(os.path.join(layer_dir, f"{name}.tensorbin")) for name in WEIGHT_NAMES}


class Inkling2MoE:
    def __init__(self, device, weights, gate_up_grid=(9, 4), down_grid=(9, 4), in0_block_w=(12, 18)):
        """``weights``: dict keyed by section-4 names (host tensors from prepare_moe_weights / load_moe_weights).

        Both projections have N = 2304 = 72 output tiles, so at most 72 of the 110 cores (11x10) get an
        output column; ``*_grid`` and ``in0_block_w`` (gate_up K-block, down K-block, in tiles) are the
        knobs tuned on p100a (see tests/test_moe.py and T4-report.md)."""
        dram = ttnn.DRAM_MEMORY_CONFIG
        self.device = device
        self.norm_w = ttnn.to_device(weights["mlp_norm"], device, dram)
        # Decode norm on one tile row: width-sharded over 18 cores (9 us) instead of one core (38 us).
        self.norm_shard = ttnn.create_sharded_memory_config(
            (32, HIDDEN // 18),
            core_grid=ttnn.CoreGrid(y=2, x=9),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        self.norm_program = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=(9, 2), subblock_w=4, block_h=1, block_w=HIDDEN // 18 // 32, inplace=False
        )
        # Router GEMM [1,H]x[H,E] split along K into ROUTER_SPLIT batched matmuls + fp32 sum: N = 4 tiles would
        # otherwise run on 4 cores (65 us -> 17 us, same fp32 weight and accumulation).
        router_w = ttnn.to_torch(weights["router.w"]).float().reshape(ROUTER_SPLIT, HIDDEN // ROUTER_SPLIT, NUM_EXPERTS)
        self.router_w = ttnn.from_torch(
            router_w.unsqueeze(0), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=dram
        )
        # The fused top-k needs bias.shape == scores.shape ([1,1,1,E]); centre it so the biased keys stay small
        # (selection is shift-invariant; see minimax_m3 centered_correction_bias).
        bias = ttnn.to_torch(weights["router.bias"])[0, 0, 0].float()
        bias = (bias - bias.mean()).reshape(1, 1, 1, NUM_EXPERTS)
        self.router_bias = ttnn.from_torch(
            bias, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=dram
        )
        self.w_gate_up = ttnn.to_device(weights["experts.w_gate_up"], device, dram)
        self.w_down = ttnn.to_device(weights["experts.w_down"], device, dram)
        # Indexed sparse_matmul still takes a sparsity operand but never reads it.
        self.sparsity = ttnn.from_torch(
            torch.ones(1, 1, 1, NUM_EXPERTS),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=dram,
        )
        self.gate_up_config = _sparse_config(2 * INTER, HIDDEN, gate_up_grid, in0_block_w[0])
        self.down_config = _sparse_config(HIDDEN, INTER, down_grid, in0_block_w[1])
        self.router_compute = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self.expert_compute = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def norm(self, h):
        """post_attention_layernorm: fp32 RMS, weight, cast to bf16 (HF: norm(h).to(bf16))."""
        hs = ttnn.to_memory_config(h, self.norm_shard)
        x = ttnn.rms_norm(
            hs,
            weight=self.norm_w,
            epsilon=RMS_EPS,
            program_config=self.norm_program,
            compute_kernel_config=self.router_compute,
            memory_config=self.norm_shard,
        )
        ttnn.deallocate(hs)
        # indexed sparse_matmul reads interleaved activation pages
        out = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16)
        ttnn.deallocate(x)
        return out

    def route(self, x):
        """x [1,1,1,H] bf16 -> (top-k weights [1,1,1,8] bf16 TILE, top-k ids [1,1,1,8] uint16 TILE)."""
        xk = ttnn.reshape(x, (1, ROUTER_SPLIT, 1, HIDDEN // ROUTER_SPLIT))
        partial = ttnn.matmul(
            xk,
            self.router_w,
            dtype=ttnn.float32,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=self.router_compute,
        )
        logits = ttnn.sum(partial, dim=1, keepdim=True)
        ttnn.deallocate(partial)
        weights, indices = ttnn.experimental.deepseek_prefill.moe_grouped_topk(
            logits,
            self.router_bias,
            n_groups=1,
            summed_experts_per_group=1,
            topk_groups=1,
            n_activated_experts=TOP_K,
            route_scale=ROUTE_SCALE,
            epsilon=1e-20,
            stable_sort=True,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.deallocate(logits)
        return weights, indices

    def _sparse(self, a, w, config, a_compact, indices):
        # Indexed/gather mode: only the 8 selected experts are read and computed; output is compact [.., 8, M, N].
        return ttnn.sparse_matmul(
            a,
            w,
            sparsity=self.sparsity,
            indices=indices,
            is_input_a_sparse=a_compact,
            is_input_b_sparse=True,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            program_config=config,
            compute_kernel_config=self.expert_compute,
            dtype=ttnn.bfloat16,
        )

    def expert_matmuls(self, x, indices):
        """The two indexed sparse_matmuls + SwiGLU: x [1,1,1,H] -> compact per-slot outputs [1,8,1,H]."""
        ids = ttnn.to_layout(indices, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
        gu = ttnn.reshape(self._sparse(x, self.w_gate_up, self.gate_up_config, False, ids), (1, TOP_K, 1, 2 * INTER))
        hidden = softcapped_swiglu(gu[..., :INTER], gu[..., INTER:])
        ttnn.deallocate(gu)
        down = ttnn.reshape(self._sparse(hidden, self.w_down, self.down_config, True, ids), (1, TOP_K, 1, HIDDEN))
        ttnn.deallocate(hidden)
        ttnn.deallocate(ids)
        return down

    def experts(self, x, weights, indices):
        """x [1,1,1,H] bf16, top-k weights / ids from route() -> [1,1,1,H] bf16 (sum of the 8 weighted experts)."""
        mem = ttnn.L1_MEMORY_CONFIG
        down = self.expert_matmuls(x, indices)
        scale = ttnn.reshape(ttnn.permute(weights, (0, 3, 2, 1)), (1, TOP_K, 1, 1))
        weighted = ttnn.mul(down, scale, memory_config=mem)
        ttnn.deallocate(down)
        out = ttnn.sum(weighted, dim=1, keepdim=True)
        ttnn.deallocate(weighted)
        return out

    def __call__(self, h):
        """h [1,1,1,H] layer state (bf16 or fp32) -> (MoE(norm(h)) [1,1,1,H] bf16, top-k ids [1,1,1,8])."""
        x = self.norm(h)
        weights, indices = self.route(x)
        out = self.experts(x, weights, indices)
        ttnn.deallocate(weights)
        ttnn.deallocate(x)
        return out, indices


def _sparse_config(n, k, grid, in0_block_w):
    """1D mcast-in0 config: N tiles spread over the grid (per_core_N = ceil), K in blocks of in0_block_w tiles."""
    n_tiles, k_tiles = n // 32, k // 32
    cores = grid[0] * grid[1]
    per_core_n = -(-n_tiles // cores)
    assert k_tiles % in0_block_w == 0, (k_tiles, in0_block_w)
    sub_w = max(d for d in range(1, min(per_core_n, 8) + 1) if per_core_n % d == 0)
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(*grid),
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=sub_w,
        out_block_h=1,
        out_block_w=per_core_n,
        per_core_M=1,
        per_core_N=per_core_n,
        fuse_batch=False,
        fused_activation=None,
        mcast_in0=True,
    )


def softcapped_swiglu(gate, up):
    """4 tanh(g/4) * sigmoid(g) * 25 tanh(u/25), as two fused binary multiplies."""
    U = ttnn.UnaryOpType
    gate_part = ttnn.mul(
        gate,
        gate,
        input_tensor_a_activations=[
            ttnn.UnaryWithParam(U.MUL_UNARY_SFPU, 1.0 / GATE_SOFTCAP),
            ttnn.UnaryWithParam(U.TANH),
        ],
        input_tensor_b_activations=[ttnn.UnaryWithParam(U.SIGMOID)],
    )
    out = ttnn.mul(
        gate_part,
        up,
        input_tensor_b_activations=[
            ttnn.UnaryWithParam(U.MUL_UNARY_SFPU, 1.0 / UP_SOFTCAP),
            ttnn.UnaryWithParam(U.TANH),
        ],
        activations=[ttnn.UnaryWithParam(U.MUL_UNARY_SFPU, GATE_SOFTCAP * UP_SOFTCAP)],
    )
    ttnn.deallocate(gate_part)
    return out
