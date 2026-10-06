# SPDX-License-Identifier: Apache-2.0
"""Kolibri optimized decoder: sandwich RMSNorm, GQA/RNoPE or sliding RoPE, sigmoid MoE.

All forward inputs, outputs and KV state are device tensors. Weight conversion
is restricted to from_state_dict. Logical prefill uses a setup-time plan and
performs chunking/padding on device. The streaming chunk entrypoint accepts
page-aligned physical chunks. Decode consumes tensor positions. See the stage
README for complete shapes, cache ownership and trace-lifetime contracts.
"""

import math
from dataclasses import dataclass

import ttnn
from models.common.lightweightmodule import LightweightModule

DRAM = ttnn.DRAM_MEMORY_CONFIG


@dataclass(frozen=True)
class DecoderPolicy:
    """Explicit per-role precision and sparse geometry; chosen with real weights."""

    expert_gate_up_dtype: str = "bfloat4_b"
    expert_down_dtype: str = "bfloat4_b"
    expert_fidelity: str = "LoFi"
    expert_fp32: bool = False
    sparse_grid: tuple = (8, 4)
    sparse_down_grid: tuple = (10, 8)
    sparse_gate_k: int = 40
    sparse_down_k: int = 16
    sparse_subblock: int = 1
    expert_l1: bool = False
    static_nnz: bool = False
    indexed_experts: bool = True
    padded_topk: bool = False
    router_grid: tuple | None = (4, 3)
    router_k: int = 80
    router_dtype: str = "bfloat4_b"
    router_fidelity: str = "LoFi"
    router_choice_bf16: bool = True
    router_score_bf16: bool = True
    decode_k_chunk: int = 128
    decode_long_k_chunk: int | None = 512
    decode_sliding_long_k_chunk: int | None = 256
    decode_long_cores_per_head: int | None = 32
    sdpa_fidelity: str = "LoFi"
    sdpa_fp32: bool = False
    sdpa_exp_approx: bool = True
    prefill_long_fidelity: str = "HiFi2"
    prefill_long_fp32: bool = True
    decode_cores_per_head: int = 16
    attention_dtype: str = "bfloat4_b"
    shared_dtype: str = "bfloat4_b"
    attention_fidelity: str = "LoFi"
    shared_fidelity: str = "LoFi"
    projection_fp32: tuple = (False, False, False, False)
    projection_cores: tuple = (8, 8, 4, 2)
    projection_k: tuple = (5, 12, 20, 8)
    projection_readers: tuple = (2, 2, 1, 2)
    residual_grid: tuple = (5, 4)
    prefill_expert_dtype: str = "bfloat4_b"
    prefill_expert_fidelity: str = "LoFi"
    prefill_expert_grid: tuple = (4, 4)
    prefill_projection_grid: tuple = (10, 8)
    prefill_projection_k: int = 16
    prefill_large_projection_k: int | None = 16
    prefill_large_output_m: int = 2
    prefill_large_kind: str = "minimal"
    prefill_large_minimal_n: int = 4
    prefill_sdpa_grid: tuple = (11, 10)
    prefill_q_chunk: int = 256
    prefill_k_chunk: int = 256
    prefill_chunk_size: int = 4096
    prefill_grouped_min: int = 32
    fused_norm_gamma: bool = True


class OptimizedDecoder(LightweightModule):
    @classmethod
    def from_state_dict(cls, state_dict, *, hf_config, layer_idx, mesh_device, **kwargs):
        import torch

        self = cls()
        self.config = hf_config
        self.layer_idx = layer_idx
        self.device = mesh_device
        self.policy = kwargs.pop("policy", DecoderPolicy())
        if kwargs:
            raise TypeError(f"Unsupported decoder options: {sorted(kwargs)}")
        if (
            hf_config.hidden_size,
            hf_config.num_attention_heads,
            hf_config.num_key_value_heads,
            hf_config.head_dim,
            hf_config.num_experts,
            hf_config.num_experts_per_tok,
        ) != (2560, 48, 4, 128, 384, 6):
            raise ValueError("This decoder requires the released Kolibri-1 dimensions")
        if hf_config.norm_topk_prob or hf_config.hidden_act != "silu":
            raise ValueError("Kolibri requires unnormalized sigmoid routing and SwiGLU")
        if hf_config.layer_types[layer_idx] not in ("sliding_attention", "full_attention"):
            raise ValueError("Unknown Kolibri layer kind")
        if hf_config.sliding_window != 513:
            raise ValueError("The released Kolibri sliding window is 513 tokens")
        self.sliding = hf_config.layer_types[layer_idx] == "sliding_attention"
        self.compute = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
        )

        self.sparse_compute = ttnn.WormholeComputeKernelConfig(
            math_fidelity=getattr(ttnn.MathFidelity, self.policy.expert_fidelity),
            math_approx_mode=False,
            fp32_dest_acc_en=self.policy.expert_fp32,
            packer_l1_acc=True,
        )
        self.sdpa_compute = ttnn.WormholeComputeKernelConfig(
            math_fidelity=getattr(ttnn.MathFidelity, self.policy.sdpa_fidelity),
            math_approx_mode=False,
            fp32_dest_acc_en=self.policy.sdpa_fp32,
            packer_l1_acc=True,
        )

        self.long_prefill_compute = ttnn.WormholeComputeKernelConfig(
            math_fidelity=getattr(ttnn.MathFidelity, self.policy.prefill_long_fidelity),
            math_approx_mode=False,
            fp32_dest_acc_en=self.policy.prefill_long_fp32,
            packer_l1_acc=True,
        )

        def convert(x, dtype=ttnn.bfloat16):
            return ttnn.from_torch(
                x.contiguous(), device=mesh_device, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=DRAM
            )

        prefix = f"model.layers.{layer_idx}."
        w = {k.removeprefix(prefix): v for k, v in state_dict.items()}
        self.norms = {
            k: convert(w[k + ".weight"].reshape(1, 1, 1, -1))
            for k in [
                "input_layernorm",
                "post_attn_norm",
                "post_attention_layernorm",
                "post_ffn_norm",
                "self_attn.q_norm",
                "self_attn.k_norm",
            ]
        }
        self.projections = {k: convert(w["self_attn." + k + ".weight"].T) for k in ["o_proj"]}
        self.projections["qkv"] = convert(
            torch.cat([w["self_attn." + k + ".weight"].T for k in ["q_proj", "k_proj", "v_proj"]], dim=-1)
        )
        self.router = convert(w["mlp.gate.weight"].T, getattr(ttnn, self.policy.router_dtype))
        self.router_compute = ttnn.WormholeComputeKernelConfig(
            math_fidelity=getattr(ttnn.MathFidelity, self.policy.router_fidelity),
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self.router_program = None
        if self.policy.router_grid:
            gx, gy = self.policy.router_grid
            pn = (12 + gx * gy - 1) // (gx * gy)
            self.router_program = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=(gx, gy),
                in0_block_w=self.policy.router_k,
                out_subblock_h=1,
                out_subblock_w=max(v for v in (1, 2, 4) if pn % v == 0),
                out_block_h=1,
                out_block_w=pn,
                per_core_M=1,
                per_core_N=pn,
                fuse_batch=False,
                mcast_in0=True,
            )
        self.expert_bias = convert(w["moe.router.expert_bias"].reshape(1, 1, 1, 384), ttnn.float32)
        self.shared = {k: convert(w["mlp.shared_experts." + k + ".weight"].T) for k in ["down_proj"]}
        self.experts = {}
        for k in ["down_proj"]:
            packed = torch.stack([w[f"mlp.experts.{e}.{k}.weight"].T for e in range(384)])[None]
            self.experts[k] = convert(packed, getattr(ttnn, self.policy.expert_down_dtype))
        packed = torch.stack(
            [
                torch.cat([w[f"mlp.experts.{e}.{k}.weight"].T for k in ["gate_proj", "up_proj"]], dim=-1)
                for e in range(384)
            ]
        )[None]
        self.experts["gate_up"] = convert(packed, getattr(ttnn, self.policy.expert_gate_up_dtype))
        self.shared["gate_up"] = convert(
            torch.cat([w["mlp.shared_experts." + k + ".weight"].T for k in ["gate_proj", "up_proj"]], dim=-1)
        )
        # Scatter produces a fresh output, so these setup constants are safe
        # across changing requests and trace replays. Keep row-major variants
        # for the common buckets to avoid converting fixed masks every call.
        self.mask_zero = convert(torch.zeros(1, 1, 128, 384))
        self.mask_one = convert(torch.ones(1, 1, 128, 6))
        self.mask_zero_rm = ttnn.to_layout(self.mask_zero, ttnn.ROW_MAJOR_LAYOUT)
        self.mask_one_rm = ttnn.to_layout(self.mask_one, ttnn.ROW_MAJOR_LAYOUT)
        self.reduce_indices = ttnn.from_torch(
            torch.arange(384, dtype=torch.int32).repeat(32, 1),
            device=self.device,
            dtype=ttnn.uint16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=DRAM,
        )
        self.reduce_mapping = ttnn.from_torch(
            torch.zeros(1, 384, dtype=torch.int32),
            device=self.device,
            dtype=ttnn.uint16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=DRAM,
        )
        # Dedicated prefill experts and packed sparse decode experts share HF values.
        self.prefill_experts = {
            role: [
                convert(w[f"mlp.experts.{e}.{role}.weight"].T, getattr(ttnn, self.policy.prefill_expert_dtype))
                for e in range(384)
            ]
            for role in ["gate_proj", "up_proj", "down_proj"]
        }
        self.expert_ids = ttnn.from_torch(
            torch.arange(384, dtype=torch.int32),
            device=self.device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=DRAM,
        )
        self.expert_sequence = convert(torch.arange(384, dtype=torch.float32).reshape(1, 1, 1, 384), ttnn.float32)
        self.expert_compute = ttnn.WormholeComputeKernelConfig(
            math_fidelity=getattr(ttnn.MathFidelity, self.policy.prefill_expert_fidelity),
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )
        # Decode weights use per-bank DRAM shards; prefill retains interleaved weights.
        values = {
            "qkv": torch.cat([w["self_attn." + role + ".weight"].T for role in ("q_proj", "k_proj", "v_proj")], -1),
            "o_proj": w["self_attn.o_proj.weight"].T,
            "gate_up": torch.cat(
                [w["mlp.shared_experts." + role + ".weight"].T for role in ("gate_proj", "up_proj")], -1
            ),
            "down_proj": w["mlp.shared_experts.down_proj.weight"].T,
        }
        self.dispatch_table = ttnn.from_torch(
            torch.zeros(1, 384, dtype=torch.int32),
            device=self.device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=DRAM,
        )
        self.projection_info = {}
        dram = self.device.dram_grid_size()
        banks = dram.x * dram.y
        dram_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram.x - 1, dram.y - 1))})
        for i, (role, value) in enumerate(values.items()):
            attention = role in ("qkv", "o_proj")
            target = self.projections if attention else self.shared
            dtype = getattr(ttnn, self.policy.attention_dtype if attention else self.policy.shared_dtype)
            target[role] = convert(value, dtype)
            k, n = value.shape
            cores = self.policy.projection_cores[i]
            readers = self.policy.projection_readers[i]
            shard_n = math.ceil(n / 32 / banks / readers) * 32 * readers
            physical_n = shard_n * banks
            weight_mem = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.DRAM,
                ttnn.ShardSpec(dram_grid, (k, shard_n), ttnn.ShardOrientation.ROW_MAJOR),
            )
            weight = ttnn.from_torch(
                torch.nn.functional.pad(value, (0, physical_n - n)).contiguous(),
                device=self.device,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=weight_mem,
            )
            grid = ttnn.num_cores_to_corerangeset(cores, self.device.compute_with_storage_grid_size(), row_wise=True)
            input_mem = ttnn.create_sharded_memory_config(
                (32, k // cores), grid, ttnn.ShardStrategy.WIDTH, use_height_and_width_as_shard_shape=True
            )
            program = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                in0_block_w=self.policy.projection_k[i],
                per_core_M=1,
                per_core_N=math.ceil(physical_n / 32 / cores),
                num_workers_per_dram_bank=readers,
            )
            compute = ttnn.WormholeComputeKernelConfig(
                math_fidelity=getattr(
                    ttnn.MathFidelity, self.policy.attention_fidelity if attention else self.policy.shared_fidelity
                ),
                math_approx_mode=False,
                fp32_dest_acc_en=self.policy.projection_fp32[i],
                packer_l1_acc=True,
            )
            self.projection_info[id(target[role])] = dict(
                weight=weight, input_mem=input_mem, program=program, compute=compute, logical_n=n, role=role
            )
        gx, gy = self.policy.residual_grid
        width = 2560 // (gx * gy)
        self.residual_mem = ttnn.create_sharded_memory_config(
            (32, width), ttnn.CoreGrid(x=gx, y=gy), ttnn.ShardStrategy.WIDTH, use_height_and_width_as_shard_shape=True
        )
        bw = width // 32
        self.norm_program = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=(gx, gy),
            subblock_w=max(v for v in (1, 2, 4) if bw % v == 0),
            block_h=1,
            block_w=bw,
            inplace=False,
        )
        # Setup constants cover the default public chunk without rebuilding or
        # untilizing masks in every large-prefill call. Keep decode allocations
        # in their original order; these buffers are never written by forward.
        mask_rows = max(128, self.policy.prefill_chunk_size)
        self.prefill_mask_zero_rm = ttnn.from_torch(
            torch.zeros(1, 1, mask_rows, 384, dtype=torch.bfloat16),
            device=self.device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=DRAM,
        )
        self.prefill_mask_one_rm = ttnn.from_torch(
            torch.ones(1, 1, mask_rows, 6, dtype=torch.bfloat16),
            device=self.device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=DRAM,
        )
        return self

    def _norm(self, x, name):
        if x.shape[-1] == 2560 and x.shape[-2] <= 32:
            x = ttnn.to_memory_config(x, self.residual_mem)
            normalized = ttnn.rms_norm(
                x,
                epsilon=self.config.rms_norm_eps,
                program_config=self.norm_program,
                weight=self.norms[name] if self.policy.fused_norm_gamma else None,
                compute_kernel_config=self.compute,
                memory_config=self.residual_mem,
            )
            if self.policy.fused_norm_gamma:
                return normalized
            return ttnn.multiply(normalized, self.norms[name], memory_config=self.residual_mem)
        if x.is_sharded():
            x = ttnn.to_memory_config(x, DRAM)
        # Fused gamma is selected by real-weight whole-decoder PCC and latency.
        normalized = ttnn.rms_norm(
            x,
            epsilon=self.config.rms_norm_eps,
            weight=self.norms[name] if self.policy.fused_norm_gamma else None,
            compute_kernel_config=self.compute,
            memory_config=DRAM,
        )
        if self.policy.fused_norm_gamma:
            return normalized
        return ttnn.multiply(normalized, self.norms[name])

    def _prefill_linear(self, x, w, dtype, compute):
        gx, gy = self.policy.prefill_projection_grid
        pm = math.ceil(x.shape[-2] / 32 / gy)
        pn = math.ceil(w.shape[-1] / 32 / gx)
        sub = max(v for v in (1, 2, 4) if pn % v == 0)
        large = x.shape[-2] > 1024 and self.policy.prefill_large_projection_k is not None
        kblock = self.policy.prefill_large_projection_k if large else self.policy.prefill_projection_k
        if large and self.policy.prefill_large_kind == "minimal":
            program = ttnn.MinimalMatmulConfig(
                M_block_size=4,
                K_block_size=kblock,
                N_block_size=self.policy.prefill_large_minimal_n,
                subblock_h=1,
                subblock_w=4,
                compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
            )
            return ttnn.experimental.minimal_matmul(
                x, w, config=program, dtype=dtype, memory_config=DRAM, compute_kernel_config=compute
            )
        bm = min(pm, self.policy.prefill_large_output_m) if large else pm
        pm = math.ceil(pm / bm) * bm
        program = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(gx, gy),
            in0_block_w=kblock,
            out_subblock_h=1,
            out_subblock_w=sub,
            out_block_h=bm,
            out_block_w=pn,
            per_core_M=pm,
            per_core_N=pn,
            transpose_mcast=False,
            fused_activation=None,
        )
        return ttnn.linear(x, w, dtype=dtype, memory_config=DRAM, program_config=program, compute_kernel_config=compute)

    def _linear(self, x, w, *, dtype=ttnn.bfloat16, activation=None):
        info = self.projection_info.get(id(w))
        if info is not None:
            if x.shape[-2] <= 32:
                a = ttnn.to_memory_config(x, info["input_mem"])
                out = ttnn.linear(
                    a,
                    info["weight"],
                    dtype=dtype,
                    memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                    program_config=info["program"],
                    compute_kernel_config=info["compute"],
                )
                if out.shape[-1] != info["logical_n"]:
                    out = ttnn.slice(out, (0,) * len(out.shape), (*tuple(out.shape)[:-1], info["logical_n"]))
                return out
            return self._prefill_linear(x, w, dtype, info["compute"])
        if w is self.router and self.router_program is not None and x.shape[-2] <= 32:
            return ttnn.linear(
                x,
                w,
                dtype=dtype,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                program_config=self.router_program,
                compute_kernel_config=self.router_compute,
            )
        if x.shape[-2] > 32:
            # Router LoFi wins at short M; large MinimalMatmul retains the
            # measured HiFi4 control because long-prefill results are mixed.
            compute = self.router_compute if w is self.router and x.shape[-2] <= 1024 else self.compute
            return self._prefill_linear(x, w, dtype, compute)
        return ttnn.linear(
            x,
            w,
            activation=activation,
            dtype=dtype,
            memory_config=DRAM,
            compute_kernel_config=self.compute,
            core_grid=ttnn.CoreGrid(y=8, x=8),
        )

    def _sparse_config(self, m, n):
        grid = self.policy.sparse_grid if n == 1024 else self.policy.sparse_down_grid
        return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=grid,
            in0_block_w=self.policy.sparse_gate_k if n == 1024 else self.policy.sparse_down_k,
            out_subblock_h=1,
            out_subblock_w=self.policy.sparse_subblock,
            out_block_h=1,
            out_block_w=self.policy.sparse_subblock,
            per_core_M=(m + 31) // 32,
            per_core_N=(n // 32 + grid[0] * grid[1] - 1) // (grid[0] * grid[1]),
            fuse_batch=False,
            mcast_in0=True,
        )

    def _moe(self, x, *, prefill=False):
        # Indexed sparse matmul consumes interleaved activation pages.
        if x.is_sharded():
            x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
        tokens = x.shape[-2]
        if prefill and tokens >= self.policy.prefill_grouped_min:
            x = ttnn.to_memory_config(x, DRAM)
        logits = self._linear(x, self.router, dtype=ttnn.float32)
        choice = ttnn.add(logits, self.expert_bias)
        if tokens == 1 and self.policy.router_choice_bf16:
            choice = ttnn.typecast(choice, ttnn.bfloat16)
        if self.policy.padded_topk:
            choice = ttnn.pad(choice, ((0, 0), (0, 0), (0, 0), (0, 128)), float("-inf"))
        _, ids = ttnn.topk(choice, k=6, dim=-1)
        if self.policy.indexed_experts and tokens == 1:
            return self._indexed_moe(x, logits, ids)
        if prefill and tokens >= self.policy.prefill_grouped_min:
            return self._grouped_prefill_moe(x, logits, ids)
        scores = logits
        if tokens <= 128:
            zero, one = self.mask_zero_rm, self.mask_one_rm
        else:
            # Larger public chunk choices remain supported entirely on device.
            repeats = (tokens + 127) // 128
            zero = ttnn.repeat(self.mask_zero, (1, 1, repeats, 1))
            one = ttnn.repeat(self.mask_one, (1, 1, repeats, 1))
        zero = ttnn.slice(zero, (0, 0, 0, 0), (1, 1, tokens, 384))
        one = ttnn.slice(one, (0, 0, 0, 0), (1, 1, tokens, 6))
        selected = ttnn.scatter(zero, dim=-1, index=ids, src=one)
        selected = ttnn.to_layout(selected, ttnn.TILE_LAYOUT)
        routing = ttnn.multiply(scores, selected, input_tensor_a_activations=[ttnn.UnaryOpType.SIGMOID])
        active = ttnn.max(selected, dim=2, keepdim=True)

        sparsity = ttnn.to_layout(active, ttnn.ROW_MAJOR_LAYOUT)

        def sparse(a, w, n, active=False):
            return ttnn.sparse_matmul(
                a,
                w,
                sparsity=sparsity,
                nnz=6 if self.policy.static_nnz and tokens == 1 else None,
                is_input_a_sparse=active,
                is_input_b_sparse=True,
                program_config=self._sparse_config(tokens, n),
                compute_kernel_config=self.sparse_compute,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.L1_MEMORY_CONFIG if self.policy.expert_l1 else DRAM,
            )

        gate_up = ttnn.reshape(sparse(x, self.experts["gate_up"], 1024), (1, 384, tokens, 1024))
        gate = ttnn.slice(gate_up, (0, 0, 0, 0), (1, 384, tokens, 512))
        up = ttnn.slice(gate_up, (0, 0, 0, 512), (1, 384, tokens, 1024))
        middle = ttnn.multiply(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
        down = ttnn.reshape(sparse(middle, self.experts["down_proj"], 2560, True), (1, 384, tokens, 2560))
        # The fused reducer allocates one score tile per expert and row tile.
        # Limit its internal row group to 32 so score tiles fit in L1. Public
        # sequence lengths stay arbitrary; prefill planning pads physical chunks.
        routed_parts = []
        for start in range(0, tokens, 32):
            count = min(32, tokens - start)
            scores_part = ttnn.slice(routing, (0, 0, start, 0), (1, 1, start + count, 384))
            scores_bf16 = ttnn.typecast(scores_part, ttnn.bfloat16)
            scores_rm = ttnn.reshape(ttnn.to_layout(scores_bf16, ttnn.ROW_MAJOR_LAYOUT), (count, 1, 1, 384))
            mask = ttnn.reshape(ttnn.permute(scores_bf16, (0, 3, 2, 1)), (1, 384, count, 1))
            piece = ttnn.slice(down, (0, 0, start, 0), (1, 384, start + count, 2560))
            clean = ttnn.where(mask, piece, 0.0)
            part = ttnn.experimental.deepseek_moe_fast_reduce_nc_fused(
                clean,
                self.reduce_indices,
                self.reduce_mapping,
                1,
                split_size=2560,
                cluster_axis=1,
                scores_tensor=scores_rm,
                compute_kernel_config=self.compute,
            )[0]
            routed_parts.append(part)
        routed = ttnn.concat(routed_parts, dim=2) if len(routed_parts) > 1 else routed_parts[0]
        both = self._linear(x, self.shared["gate_up"])
        width = both.shape[-1] // 2
        sg = ttnn.slice(both, (0, 0, 0, 0), (1, 1, tokens, width))
        su = ttnn.slice(both, (0, 0, 0, width), (1, 1, tokens, 2 * width))
        middle_shared = ttnn.multiply(sg, su, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
        shared = self._linear(middle_shared, self.shared["down_proj"])
        return ttnn.add(routed, shared, dtype=ttnn.bfloat16)

    def _grouped_prefill_moe(self, x, logits, ids):
        """Dispatch only routed rows; compute and reduce six contributions per token."""
        tokens = x.shape[-2]
        if 128 < tokens <= self.prefill_mask_zero_rm.shape[-2]:
            zero = ttnn.slice(self.prefill_mask_zero_rm, (0, 0, 0, 0), (1, 1, tokens, 384))
            one = ttnn.slice(self.prefill_mask_one_rm, (0, 0, 0, 0), (1, 1, tokens, 6))
        elif tokens > 128:
            zero = ttnn.to_layout(ttnn.repeat(self.mask_zero, (1, 1, (tokens + 127) // 128, 1)), ttnn.ROW_MAJOR_LAYOUT)
            one = ttnn.to_layout(ttnn.repeat(self.mask_one, (1, 1, (tokens + 127) // 128, 1)), ttnn.ROW_MAJOR_LAYOUT)
            zero = ttnn.slice(zero, (0, 0, 0, 0), (1, 1, tokens, 384))
            one = ttnn.slice(one, (0, 0, 0, 0), (1, 1, tokens, 6))
        else:
            zero = ttnn.slice(self.mask_zero_rm, (0, 0, 0, 0), (1, 1, tokens, 384))
            one = ttnn.slice(self.mask_one_rm, (0, 0, 0, 0), (1, 1, tokens, 6))
        selected = ttnn.to_layout(ttnn.scatter(zero, dim=-1, index=ids, src=one), ttnn.TILE_LAYOUT)
        counts = ttnn.sum(ttnn.typecast(selected, ttnn.float32), dim=2, keepdim=True)
        aligned = ttnn.multiply(ttnn.ceil(ttnn.multiply(counts, 1 / 32)), 32)
        offsets = ttnn.subtract(ttnn.cumsum(aligned, dim=3), aligned)

        def ints(t, shape):
            return ttnn.reshape(ttnn.to_layout(ttnn.typecast(t, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT), shape)

        offsets = ints(offsets, (1, 384))
        counts = ints(counts, (1, 384))
        capacity = ((6 * tokens + 31 * 384 + 31) // 32) * 32
        dispatch_ids = ttnn.to_layout(
            ids if ids.dtype == ttnn.uint16 else ttnn.typecast(ids, ttnn.uint16), ttnn.ROW_MAJOR_LAYOUT
        )
        routed, metadata = ttnn.experimental.deepseek_prefill.dispatch(
            ttnn.reshape(x, (1, tokens, 2560)),
            ttnn.reshape(dispatch_ids, (1, tokens, 6)),
            offsets,
            self.dispatch_table,
            dispatch_group_size=1,
            experts_per_chip=384,
            num_routed_experts=384,
            num_experts_per_tok=6,
            metadata_len=3,
            max_dispatch_buffer_token_size=capacity,
            cluster_axis=0,
        )
        routed = ttnn.reshape(routed, (capacity, 2560))
        output = ttnn.empty(
            (capacity, 2560), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device, memory_config=DRAM
        )
        down = ttnn.experimental.deepseek_prefill.moe_fused_swiglu(
            routed,
            self.prefill_experts["gate_proj"],
            self.prefill_experts["up_proj"],
            self.prefill_experts["down_proj"],
            ttnn.reshape(counts, (384,)),
            self.expert_ids,
            input_m_tiles=(tokens + 31) // 32,
            compute_kernel_config=self.expert_compute,
            core_grid=ttnn.CoreCoord(*self.policy.prefill_expert_grid),
            output=output,
            expert_region_offsets=ttnn.reshape(offsets, (384,)),
            read_x_at_offset=True,
        )
        combined = ttnn.experimental.deepseek_prefill.combine(
            ttnn.reshape(down, (1, 1, capacity, 2560)),
            metadata,
            ttnn.reshape(counts, (1, 1, 384)),
            ttnn.reshape(offsets, (1, 1, 384)),
            dispatch_group_size=1,
            experts_per_chip=384,
            num_experts_per_tok=6,
            seq_len_per_chip=tokens,
            cluster_axis=0,
        )
        scores = ttnn.sigmoid(ttnn.gather(logits, -1, index=ids))
        scores = ttnn.to_layout(ttnn.typecast(scores, ttnn.bfloat16), ttnn.ROW_MAJOR_LAYOUT)
        scores = ttnn.reshape(scores, (1, 1, tokens, 6, 1))
        routed = ttnn.experimental.deepseek_prefill.post_combine_reduce(
            combined, scores, dispatch_ids, self.dispatch_table, expert_dim=3, output_memory_config=DRAM
        )
        both = self._linear(x, self.shared["gate_up"])
        gate = ttnn.slice(both, (0, 0, 0, 0), (1, 1, tokens, 512))
        up = ttnn.slice(both, (0, 0, 0, 512), (1, 1, tokens, 1024))
        middle = ttnn.multiply(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
        return ttnn.add(routed, self._linear(middle, self.shared["down_proj"]), dtype=ttnn.bfloat16)

    def _indexed_moe(self, x, logits, ids):
        """Six compact slots in top-k order; expert ids stay entirely on device."""
        mem = ttnn.L1_MEMORY_CONFIG
        if self.policy.router_score_bf16:
            logits = ttnn.typecast(logits, ttnn.bfloat16, memory_config=mem)
        indices = ttnn.to_layout(
            ids if ids.dtype == ttnn.uint16 else ttnn.typecast(ids, ttnn.uint16),
            ttnn.ROW_MAJOR_LAYOUT,
            memory_config=mem,
        )
        sparsity = ttnn.slice(self.mask_zero_rm, (0, 0, 0, 0), (1, 1, 1, 384))

        def sparse(a, role, n, active=False):
            return ttnn.reshape(
                ttnn.sparse_matmul(
                    a,
                    self.experts[role],
                    sparsity=sparsity,
                    indices=indices,
                    is_input_a_sparse=active,
                    is_input_b_sparse=True,
                    program_config=self._sparse_config(1, n),
                    compute_kernel_config=self.sparse_compute,
                    dtype=ttnn.bfloat16,
                    memory_config=mem,
                ),
                (1, 6, 1, n),
            )

        both = sparse(x, "gate_up", 1024)
        gate = ttnn.slice(both, (0, 0, 0, 0), (1, 6, 1, 512), memory_config=mem)
        up = ttnn.slice(both, (0, 0, 0, 512), (1, 6, 1, 1024), memory_config=mem)
        middle = ttnn.multiply(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU], memory_config=mem)
        down = sparse(middle, "down_proj", 2560, True)
        scores = ttnn.gather(logits, -1, index=ids, memory_config=mem)
        scores = ttnn.sigmoid(scores, memory_config=mem)
        scores = ttnn.reshape(ttnn.permute(scores, (0, 3, 2, 1)), (1, 6, 1, 1))
        weighted = ttnn.multiply(down, scores, memory_config=mem)
        routed = ttnn.sum(weighted, dim=1, keepdim=True, memory_config=mem)
        shared_both = self._linear(x, self.shared["gate_up"])
        width = shared_both.shape[-1] // 2
        sg = ttnn.slice(shared_both, (0, 0, 0, 0), (1, 1, 1, width))
        su = ttnn.slice(shared_both, (0, 0, 0, width), (1, 1, 1, 2 * width))
        sm = ttnn.multiply(sg, su, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
        shared = self._linear(sm, self.shared["down_proj"])
        return ttnn.add(routed, shared, dtype=ttnn.bfloat16)

    def _finish(self, residual, attention, *, prefill=False):
        if residual.shape[-2] <= 32:
            residual = ttnn.to_memory_config(residual, self.residual_mem)
            x = ttnn.add(residual, self._norm(attention, "post_attn_norm"), memory_config=self.residual_mem)
            moe = self._moe(self._norm(x, "post_attention_layernorm"), prefill=prefill)
            return ttnn.add(x, self._norm(moe, "post_ffn_norm"), memory_config=self.residual_mem)
        x = ttnn.add(residual, self._norm(attention, "post_attn_norm"))
        return ttnn.add(
            x, self._norm(self._moe(self._norm(x, "post_attention_layernorm"), prefill=prefill), "post_ffn_norm")
        )

    def _qkv(self, x, cos, sin):
        s = x.shape[-2]
        n = self._norm(x, "input_layernorm")
        packed = self._linear(n, self.projections["qkv"])
        packed = ttnn.to_memory_config(packed, DRAM)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(packed, num_heads=48, num_kv_heads=4, transpose_k_heads=False)
        q, k, v = [ttnn.slice(t, (0, 0, 0, 0), (1, h, s, 128)) for t, h in [(q, 48), (k, 4), (v, 4)]]
        q = self._norm(q, "self_attn.q_norm")
        k = self._norm(k, "self_attn.k_norm")
        if self.sliding:
            q = self._rope(q, cos, sin)
            k = self._rope(k, cos, sin)
        return (q, k, v)

    def _rope(self, x, cos, sin):
        return ttnn.experimental.rotary_embedding_hf(x, cos, sin, compute_kernel_config=self.compute)

    def prefill_chunk_forward(
        self,
        hidden_states,
        *,
        kv_cache,
        page_table,
        chunk_page_table,
        chunk_start,
        cos=None,
        sin=None,
        chunk_start_alignment=32,
        context_length_bound=None,
    ):
        """One padded physical chunk [1,1,S,2560], S a multiple of 32.

        chunk_start: INT32 device scalar absolute offset; chunk_page_table:
        pages starting at that offset. Caller retains only valid logical rows.
        page_table maps the entire request, including its cached prefix.
        chunk_start_alignment is a setup-time guarantee about every offset used
        with these buffers. Default32 accepts any page-aligned offset. A plan
        can supply its exact offset (zero is divisible by every tile size).
        context_length_bound is an optional setup-time upper bound on the
        absolute end position of every call using these buffers. Public plans
        supply it; omitted bounds use full page-table capacity for precision.
        """
        q, k, v = self._qkv(hidden_states, cos, sin)
        ttnn.experimental.paged_fill_cache(
            kv_cache[0], ttnn.typecast(k, kv_cache[0].dtype) if k.dtype != kv_cache[0].dtype else k, chunk_page_table
        )
        ttnn.experimental.paged_fill_cache(
            kv_cache[1], ttnn.typecast(v, kv_cache[1].dtype) if v.dtype != kv_cache[1].dtype else v, chunk_page_table
        )
        # A public plan knows its maximum prefix. Stable streaming tensors may
        # omit the bound and conservatively use physical capacity instead.
        context_bound = page_table.shape[-1] * 32 if context_length_bound is None else context_length_bound
        attention_compute = (
            self.long_prefill_compute if not self.sliding and context_bound > 65536 else self.sdpa_compute
        )
        y = ttnn.transformer.chunked_scaled_dot_product_attention(
            q,
            kv_cache[0],
            kv_cache[1],
            page_table,
            chunk_start_idx_tensor=chunk_start,
            sliding_window_size=513 if self.sliding else None,
            program_config=ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=self.policy.prefill_sdpa_grid,
                q_chunk_size=max(
                    n
                    for n in (32, 64, 128, 256, 512)
                    if n <= self.policy.prefill_q_chunk
                    and hidden_states.shape[-2] % n == 0
                    and chunk_start_alignment % n == 0
                ),
                k_chunk_size=max(
                    n
                    for n in (32, 64, 128, 256, 512)
                    if n <= self.policy.prefill_k_chunk
                    and chunk_start_alignment % n == 0
                    and (page_table.shape[-1] * 32) % n == 0
                ),
                exp_approx_mode=self.policy.sdpa_exp_approx,
            ),
            compute_kernel_config=attention_compute,
        )
        y = ttnn.reshape(ttnn.transformer.concatenate_heads(y), (1, 1, hidden_states.shape[-2], 6144))
        return self._finish(hidden_states, self._linear(y, self.projections["o_proj"]), prefill=True)

    def decode_forward(self, hidden_states, *, kv_cache, page_table, current_pos, cos=None, sin=None):
        """One token per request: [1,1,B,2560], tensor positions [B], B table rows.

        Every op is device-only and can be captured in a TTNN execution trace.
        Refresh stable input/position/RoPE/page-table buffers before replay.
        """
        batch = hidden_states.shape[-2]
        if tuple(hidden_states.shape) != (1, 1, batch, 2560) or not 1 <= batch <= 32:
            raise ValueError("Decode expects [1,1,B,2560], 1 <= B <= 32")
        if tuple(current_pos.shape) != (batch,) or page_table.shape[0] != batch:
            raise ValueError("Decode positions and page-table rows must match batch")
        grid = self.device.compute_with_storage_grid_size()
        # Sharded SDPA assigns batch i to (i % grid.x, i // grid.x).
        # Q/K/V and RoPE must use that exact physical row-major prefix.
        shard_grid = ttnn.num_cores_to_corerangeset(batch, grid, row_wise=True)
        mem = ttnn.create_sharded_memory_config(
            (32, 128), shard_grid, ttnn.ShardStrategy.HEIGHT, use_height_and_width_as_shard_shape=True
        )
        n = self._norm(hidden_states, "input_layernorm")
        packed = self._linear(n, self.projections["qkv"])
        q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(packed, num_heads=48, num_kv_heads=4, memory_config=mem)
        q = self._norm(ttnn.to_memory_config(q, DRAM), "self_attn.q_norm")
        k = self._norm(ttnn.to_memory_config(k, DRAM), "self_attn.k_norm")
        if self.sliding:
            qmem = ttnn.create_sharded_memory_config(
                (64, 128), shard_grid, ttnn.ShardStrategy.HEIGHT, use_height_and_width_as_shard_shape=True
            )
            q = ttnn.to_memory_config(q, qmem)
            k = ttnn.to_memory_config(k, mem)
            c = ttnn.to_memory_config(ttnn.permute(cos, (0, 2, 1, 3)), mem)
            s = ttnn.to_memory_config(ttnn.permute(sin, (0, 2, 1, 3)), mem)
            q = ttnn.experimental.rotary_embedding_hf(q, c, s, is_decode_mode=True, compute_kernel_config=self.compute)
            k = ttnn.experimental.rotary_embedding_hf(k, c, s, is_decode_mode=True, compute_kernel_config=self.compute)
        k = ttnn.to_memory_config(k, mem)
        v = ttnn.to_memory_config(v, mem)
        ttnn.experimental.paged_update_cache(kv_cache[0], k, update_idxs_tensor=current_pos, page_table=page_table)
        ttnn.experimental.paged_update_cache(kv_cache[1], v, update_idxs_tensor=current_pos, page_table=page_table)
        k_chunk = self.policy.decode_k_chunk
        cores_per_head = self.policy.decode_cores_per_head
        capacity = page_table.shape[-1] * 32
        if self.sliding and self.policy.decode_sliding_long_k_chunk and capacity > 8192:
            k_chunk = max(
                n for n in (128, 256, 512) if n <= self.policy.decode_sliding_long_k_chunk and capacity % n == 0
            )
        if not self.sliding and self.policy.decode_long_k_chunk and capacity > 65536:
            k_chunk = max(n for n in (128, 256, 512) if n <= self.policy.decode_long_k_chunk and capacity % n == 0)
            cores_per_head = self.policy.decode_long_cores_per_head or cores_per_head
            # BF16 KV needs twice the tile payload. K512 exhausts L1 with
            # 32 requested workers/head; K256 preserves the full capacity.
            if kv_cache[0].dtype == ttnn.bfloat16:
                k_chunk = min(k_chunk, 256)
        y = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q,
            kv_cache[0],
            kv_cache[1],
            page_table,
            cur_pos_tensor=current_pos,
            sliding_window_size=513 if self.sliding else None,
            program_config=ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=(grid.x, grid.y),
                q_chunk_size=32,
                k_chunk_size=k_chunk,
                max_cores_per_head_batch=cores_per_head,
                exp_approx_mode=self.policy.sdpa_exp_approx,
            ),
            compute_kernel_config=self.sdpa_compute,
        )
        y = ttnn.reshape(y, (1, 1, batch, 6144))
        return self._finish(hidden_states, self._linear(y, self.projections["o_proj"]))

    def prepare_prefill(self, *, page_table_host, seq_len, start_pos=0, chunk_size=None):
        """Setup-only plan: bind request page mappings, absolute RoPE and offsets.

        No weights or KV are rebuilt. Prepare plans and their physical programs
        before the first trace. A generator may instead reuse stable tensors with
        prefill_chunk_forward, refreshing their contents at request boundaries.
        """
        import torch

        chunk_size = self.policy.prefill_chunk_size if chunk_size is None else chunk_size

        if not 0 < seq_len or start_pos < 0 or start_pos + seq_len > self.config.max_position_embeddings:
            raise ValueError("Prefill extent is outside the configured context")
        if chunk_size <= 0 or chunk_size % 32:
            raise ValueError("Physical chunk size must be a positive multiple of 32")

        def tt(x, dtype, layout):
            return ttnn.from_torch(x.contiguous(), device=self.device, dtype=dtype, layout=layout, memory_config=DRAM)

        if page_table_host.ndim != 2 or page_table_host.shape[0] < 1:
            raise ValueError("Page table must have at least one request row")
        if page_table_host.dtype != torch.int32 or (page_table_host < 0).any():
            raise ValueError("Page table must contain nonnegative INT32 physical page IDs")
        if page_table_host.shape[1] * 32 < start_pos + seq_len:
            raise ValueError("Page table does not cover the logical prefill extent")
        plans = []
        for slot in range(page_table_host.shape[0]):
            pages = page_table_host[slot : slot + 1]
            table = tt(pages, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
            chunks = []
            offset = 0
            while offset < seq_len:
                pos = start_pos + offset
                decode = pos % 32 != 0
                count = 1 if decode else min(chunk_size, seq_len - offset)
                physical = 1 if decode else (count + 31) // 32 * 32
                positions = torch.arange(pos, pos + physical).float()
                phase = positions[:, None] / 10000.0 ** (torch.arange(0, 128, 2).float() / 128)
                phase = torch.cat([phase, phase], -1)[None, None]
                kw = dict(
                    page_table=table,
                    cos=tt(phase.cos().bfloat16(), ttnn.bfloat16, ttnn.TILE_LAYOUT),
                    sin=tt(phase.sin().bfloat16(), ttnn.bfloat16, ttnn.TILE_LAYOUT),
                )
                position = tt(torch.tensor([pos], dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
                if decode:
                    kw["current_pos"] = position
                else:
                    kw["chunk_start"] = position
                    kw["chunk_start_alignment"] = pos
                    kw["context_length_bound"] = pos + physical
                    kw["chunk_page_table"] = tt(
                        pages[:, pos // 32 : (pos + physical) // 32], ttnn.int32, ttnn.ROW_MAJOR_LAYOUT
                    )
                chunks.append((offset, count, physical, decode, kw))
                offset += count
            plans.append(chunks)
        return dict(seq_len=seq_len, batch=len(plans), slots=plans)

    def prefill_forward(self, hidden_states, *, kv_cache, plan):
        """Logical prefill [1,B,S,2560] -> [1,B,S,2560].

        Accepts every S>=1 within context, including continuation at arbitrary
        positions. A setup-time plan supplies device page mappings and positions.
        Chunking/padding/slicing stay on device. KV entries before start_pos are
        preserved. Padding only occupies future rows in this request's own pages;
        causal attention cannot see it, and later append overwrites those rows.
        """
        seq_len = plan["seq_len"]
        batch = plan["batch"]
        if tuple(hidden_states.shape) != (1, batch, seq_len, 2560):
            raise ValueError("Prefill input shape does not match its plan")
        results = []
        for slot, chunks in enumerate(plan["slots"]):
            outputs = []
            for offset, count, physical, decode, kw in chunks:
                x = ttnn.slice(hidden_states, (0, slot, offset, 0), (1, slot + 1, offset + count, 2560))
                if physical != count:
                    x = ttnn.pad(x, ((0, 0), (0, 0), (0, physical - count), (0, 0)), value=0.0)
                y = (self.decode_forward if decode else self.prefill_chunk_forward)(x, kv_cache=kv_cache, **kw)
                y = ttnn.to_memory_config(y, DRAM)
                outputs.append(ttnn.slice(y, (0, 0, 0, 0), (1, 1, count, 2560)))
            results.append(ttnn.concat(outputs, dim=2) if len(outputs) > 1 else outputs[0])
        return ttnn.concat(results, dim=1) if batch > 1 else results[0]
