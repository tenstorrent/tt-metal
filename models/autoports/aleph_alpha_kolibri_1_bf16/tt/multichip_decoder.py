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
from models.demos.gpt_oss.tt.ccl import CCLManager

from .optimized_decoder import OptimizedDecoder

DRAM = ttnn.DRAM_MEMORY_CONFIG


class MeshCCLManager(CCLManager):
    """Initialize every core that the mesh collective planner may select."""

    def _init_subdevice(self):
        # The inherited helper assumes an 8x8 grid without restricting the
        # collective planner to it. Blackhole workers outside that rectangle
        # would read uninitialized global semaphore addresses.
        grid = self.mesh_device.compute_with_storage_grid_size()
        self.ccl_cores = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))}
        )
        self.ccl_sub_device_id = ttnn.SubDeviceId(0)


@dataclass(frozen=True)
class MeshPolicy:
    safe_compute_fidelity: str = "HiFi4"
    projection_activation_dtype: str = "bfloat16"
    expert_activation_dtype: str = "bfloat16"
    expert_gate_up_dtype: str = "bfloat4_b"
    expert_down_dtype: str = "bfloat4_b"
    expert_fidelity: str = "LoFi"
    expert_fp32: bool = False
    sparse_grid: tuple = (8, 1)
    sparse_down_grid: tuple = (10, 8)
    sparse_gate_k: int = 80
    sparse_down_k: int = 4
    sparse_subblock: int = 1
    expert_l1: bool = False
    static_nnz: bool = False
    indexed_experts: bool = True
    padded_topk: bool = False
    decode_topk_width: int = 1024
    packed_decode_router: bool = True
    router_grid: tuple = (4, 3)
    router_k: int = 80
    router_dtype: str = "bfloat4_b"
    router_fidelity: str = "LoFi"
    router_choice_bf16: bool = True
    router_score_bf16: bool = True
    decode_k_chunk: int = 128
    decode_long_k_chunk: int = 512
    decode_sliding_long_k_chunk: int = 256
    decode_long_cores_per_head: int = 64
    sdpa_fidelity: str = "LoFi"
    sdpa_fp32: bool = False
    sdpa_exp_approx: bool = True
    prefill_long_fidelity: str = "HiFi2"
    prefill_long_fp32: bool = True
    decode_cores_per_head: int = 8
    attention_dtype: str = "bfloat4_b"
    shared_dtype: str = "bfloat4_b"
    attention_fidelity: str = "LoFi"
    shared_fidelity: str = "LoFi"
    projection_fp32: tuple = (False, False, False, False)
    projection_cores: tuple = (4, 8, 4, 2)
    projection_k: tuple = (20, 6, 20, 2)
    projection_readers: tuple = (1, 2, 1, 2)
    residual_grid: tuple = (5, 4)
    prefill_projection_grid: tuple = (10, 8)
    prefill_projection_k: int = 4
    prefill_large_projection_k: int = 16
    prefill_large_output_m: int = 2
    prefill_large_kind: str = "minimal"
    prefill_large_minimal_n: int = 4
    prefill_sdpa_grid: tuple = (11, 10)
    prefill_q_chunk: int = 256
    prefill_k_chunk: int = 256
    prefill_chunk_size: int = 8192
    prefill_grouped_min: int = 32
    prefill_expert_dtype: str = "bfloat4_b"
    prefill_expert_fidelity: str = "LoFi"
    prefill_expert_grid: tuple = (4, 4)
    grouped_prefill: bool = True
    prefix_matmul: bool = True
    prefill_prefix_cores: int = 6
    fused_norm_gamma: bool = True
    residual_layout: str = "replicated"
    ccl_dtype: str = "bfloat16"
    prefill_ccl_dtype: str = "bfloat8_b"
    num_links: int = 2
    fused_qkv: bool | str = False
    fused_wo: bool = False
    ccl_mode: str = "direct"


class CollectiveWorkspace:
    """Persistent communication state shared by a serial decoder stack.

    Construct before capture. All users must execute on CQ0 without concurrent
    forwards: a later layer may overwrite scratch only after earlier consumers
    finish. Sharing avoids multiplying L1 collective buffers by50 layers.
    """

    @staticmethod
    def signature(policy):
        return (
            policy.ccl_mode,
            policy.ccl_dtype,
            policy.num_links,
            policy.fused_wo,
            policy.fused_qkv,
            tuple(policy.residual_grid),
        )

    def __init__(self, mesh_device, policy):
        self.device, self.policy = mesh_device, policy
        self.ccl = MeshCCLManager(mesh_device, policy.num_links, ttnn.Topology.Linear)
        if self.policy.ccl_mode == "direct":
            gx, gy = policy.residual_grid
            buffer_mem = ttnn.create_sharded_memory_config(
                (32, 10240 // (gx * gy)),
                ttnn.CoreGrid(x=gx, y=gy),
                ttnn.ShardStrategy.WIDTH,
                use_height_and_width_as_shard_shape=True,
            )
            self.ar_buffers = [
                ttnn.empty(
                    (1, 1, 32, 10240),
                    dtype=getattr(ttnn, self.policy.ccl_dtype),
                    layout=ttnn.TILE_LAYOUT,
                    device=self.device,
                    memory_config=buffer_mem,
                )
                for _ in range(2)
            ]
            self.ar_semaphores = [ttnn.create_global_semaphore(self.device, self.ccl.ccl_cores, 0) for _ in range(2)]
            self.ar_index = 0
        if self.policy.fused_wo:
            self.mmrs_intermediate = ttnn.empty(
                (2, 1, 1, 2560), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device, memory_config=DRAM
            )
            self.mmrs_output = ttnn.empty(
                (1, 1, 1, 640), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device, memory_config=DRAM
            )
        if policy.fused_qkv == "minimal":
            # Gather storage must match input dtype, independently of the BF16
            # matmul output. The op's automatic buffer takes output dtype.
            self.qkv_gather = ttnn.empty(
                (1, 1, 1, 2560),
                dtype=getattr(ttnn, policy.ccl_dtype),
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )


class MultichipDecoder(OptimizedDecoder):
    @classmethod
    def from_state_dict(cls, state_dict, *, hf_config, layer_idx, mesh_device, **kwargs):
        import torch

        self = cls()
        self.config = hf_config
        self.layer_idx = layer_idx
        self.device = mesh_device
        if tuple(mesh_device.shape) != (1, 4):
            raise ValueError("Kolibri multichip requires QB2 mesh1x4")
        self.policy = kwargs.pop("policy", MeshPolicy())
        workspace = kwargs.pop("collective_workspace", None)
        self.sliding_cache_tokens = kwargs.pop("sliding_cache_tokens", None)
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
        if self.sliding_cache_tokens is not None:
            if not self.sliding or self.sliding_cache_tokens % 32 or self.sliding_cache_tokens < 32 + 512:
                raise ValueError("Circular sliding cache must be page-aligned and retain at least 32+512 tokens")
        self.compute = ttnn.WormholeComputeKernelConfig(
            math_fidelity=getattr(ttnn.MathFidelity, self.policy.safe_compute_fidelity),
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
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

        self.hidden_width = 640 if self.policy.residual_layout == "sharded" else 2560
        self.collective_workspace = (
            workspace if workspace is not None else CollectiveWorkspace(mesh_device, self.policy)
        )
        if self.collective_workspace.device is not mesh_device or CollectiveWorkspace.signature(
            self.collective_workspace.policy
        ) != CollectiveWorkspace.signature(self.policy):
            raise ValueError("Collective workspace must use the same mesh and communication policy")
        self.ccl = self.collective_workspace.ccl

        def mapper(dim):
            return (
                ttnn.ReplicateTensorToMesh(mesh_device) if dim is None else ttnn.ShardTensorToMesh(mesh_device, dim=dim)
            )

        def columns(parts):
            # Every rank owns its local slice of EVERY projection, in Q/K/V
            # or gate/up order. A plain shard of globally packed weights is wrong.
            return torch.cat([torch.cat([v.chunk(4, dim=-1)[r] for v in parts], dim=-1) for r in range(4)], dim=-1)

        def convert(x, dtype=ttnn.bfloat16, dim=None):
            return ttnn.from_torch(
                x.contiguous(),
                device=mesh_device,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=DRAM,
                mesh_mapper=mapper(dim),
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
        self.shard_norms = {
            k: convert(w[k + ".weight"].reshape(1, 1, 1, -1), dim=3)
            for k in ("input_layernorm", "post_attn_norm", "post_attention_layernorm", "post_ffn_norm")
        }
        self.projections = {}
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
        if (
            self.policy.packed_decode_router
            and self.policy.indexed_experts
            and self.policy.decode_topk_width == 1024
            and not self.policy.padded_topk
        ):
            # Keep the public expert set at384. Setup-time zero weight columns
            # and -inf selection bias remove runtime top-k pad/fill operations.
            self.decode_router = convert(
                torch.nn.functional.pad(w["mlp.gate.weight"].T, (0, 640)).contiguous(),
                getattr(ttnn, self.policy.router_dtype),
            )
            self.decode_expert_bias = convert(
                torch.nn.functional.pad(
                    w["moe.router.expert_bias"].reshape(1, 1, 1, 384), (0, 640), value=float("-inf")
                ),
                ttnn.float32,
            )
            self.decode_router_program = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=(8, 4),
                in0_block_w=self.policy.router_k,
                out_subblock_h=1,
                out_subblock_w=1,
                out_block_h=1,
                out_block_w=1,
                per_core_M=1,
                per_core_N=1,
                fuse_batch=False,
                mcast_in0=True,
            )
        self.shared = {}
        self.experts = {}
        for k in ["down_proj"]:
            packed = torch.stack([w[f"mlp.experts.{e}.{k}.weight"].T for e in range(384)])[None]
            self.experts[k] = convert(packed, getattr(ttnn, self.policy.expert_down_dtype), dim=2)
        packed = torch.stack(
            [columns([w[f"mlp.experts.{e}.{k}.weight"].T for k in ["gate_proj", "up_proj"]]) for e in range(384)]
        )[None]
        self.experts["gate_up"] = convert(packed, getattr(ttnn, self.policy.expert_gate_up_dtype), dim=3)
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
        if self.policy.grouped_prefill:
            if self.policy.prefix_matmul:
                self.expert_prefix_sum = convert(torch.triu(torch.ones(384, 384, dtype=torch.bfloat16)))
            self.prefill_experts = {
                role: [
                    convert(
                        w[f"mlp.experts.{e}.{role}.weight"].T,
                        getattr(ttnn, self.policy.prefill_expert_dtype),
                        dim=0 if role == "down_proj" else 1,
                    )
                    for e in range(384)
                ]
                for role in ("gate_proj", "up_proj", "down_proj")
            }
            self.expert_ids = ttnn.from_torch(
                torch.arange(384, dtype=torch.int32),
                device=self.device,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=DRAM,
            )
            self.expert_compute = ttnn.WormholeComputeKernelConfig(
                math_fidelity=getattr(ttnn.MathFidelity, self.policy.prefill_expert_fidelity),
                math_approx_mode=False,
                fp32_dest_acc_en=False,
                packer_l1_acc=False,
            )
        # Decode weights use per-bank DRAM shards; prefill retains interleaved weights.
        values = {
            "qkv": columns([w["self_attn." + role + ".weight"].T for role in ("q_proj", "k_proj", "v_proj")]),
            "o_proj": w["self_attn.o_proj.weight"].T,
            "gate_up": columns([w["mlp.shared_experts." + role + ".weight"].T for role in ("gate_proj", "up_proj")]),
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
            axis = 1 if role in ("qkv", "gate_up") else 0
            target[role] = convert(value, dtype, dim=axis)
            local_values = value.chunk(4, dim=axis)
            k, n = local_values[0].shape
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
                torch.cat(
                    [torch.nn.functional.pad(v, (0, physical_n - n)) for v in local_values], dim=axis
                ).contiguous(),
                mesh_mapper=mapper(axis),
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
        width = self.hidden_width // (gx * gy)
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
        for name in ("ar_buffers", "ar_semaphores", "mmrs_intermediate", "mmrs_output", "qkv_gather"):
            if hasattr(self.collective_workspace, name):
                setattr(self, name, getattr(self.collective_workspace, name))
        self.ar_index = 0
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

    def _sparse_config(self, m, n):
        grid = self.policy.sparse_grid if n == 256 else self.policy.sparse_down_grid
        return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=grid,
            in0_block_w=self.policy.sparse_gate_k if n == 256 else self.policy.sparse_down_k,
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
        x = self._gather_input(x)
        # Indexed sparse matmul consumes interleaved activation pages.
        if x.is_sharded():
            x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
        tokens = x.shape[-2]
        if self.policy.grouped_prefill and prefill and tokens >= self.policy.prefill_grouped_min:
            x = ttnn.to_memory_config(x, DRAM)
        packed_router = tokens == 1 and hasattr(self, "decode_router")
        if packed_router:
            logits = ttnn.linear(
                x,
                self.decode_router,
                dtype=ttnn.float32,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                program_config=self.decode_router_program,
                compute_kernel_config=self.router_compute,
            )
        else:
            logits = self._linear(x, self.router, dtype=ttnn.float32)
        choice = ttnn.add(logits, self.decode_expert_bias if packed_router else self.expert_bias)
        if tokens == 1 and self.policy.router_choice_bf16:
            choice = ttnn.typecast(choice, ttnn.bfloat16)
        topk_width = 512 if self.policy.padded_topk else self.policy.decode_topk_width if tokens == 1 else 384
        if topk_width > choice.shape[-1]:
            # Negative-infinity columns cannot select nonexistent experts. This
            # enables the multicore top-k kernel without adding logical users.
            choice = ttnn.pad(choice, ((0, 0), (0, 0), (0, 0), (0, topk_width - choice.shape[-1])), float("-inf"))
        _, ids = ttnn.topk(choice, k=6, dim=-1)
        if self.policy.indexed_experts and tokens == 1:
            return self._indexed_moe(x, logits, ids)
        if self.policy.grouped_prefill and prefill and tokens >= self.policy.prefill_grouped_min:
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
            a = self._expert_activation(a)
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

        gate_up = ttnn.reshape(sparse(x, self.experts["gate_up"], 256), (1, 384, tokens, 256))
        gate = ttnn.slice(gate_up, (0, 0, 0, 0), (1, 384, tokens, 128))
        up = ttnn.slice(gate_up, (0, 0, 0, 128), (1, 384, tokens, 256))
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

    def _expert_activation(self, x):
        dtype = getattr(ttnn, self.policy.expert_activation_dtype)
        return ttnn.typecast(x, dtype) if x.dtype != dtype else x

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
            a = self._expert_activation(a)
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

        both = sparse(x, "gate_up", 256)
        gate = ttnn.slice(both, (0, 0, 0, 0), (1, 6, 1, 128), memory_config=mem)
        up = ttnn.slice(both, (0, 0, 0, 128), (1, 6, 1, 256), memory_config=mem)
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
            x = ttnn.add(
                residual, self._norm(self._allreduce(attention), "post_attn_norm"), memory_config=self.residual_mem
            )
            moe = self._moe(self._norm(x, "post_attention_layernorm"), prefill=prefill)
            return ttnn.add(x, self._norm(self._allreduce(moe), "post_ffn_norm"), memory_config=self.residual_mem)
        x = ttnn.add(residual, self._norm(self._allreduce(attention), "post_attn_norm"))
        return ttnn.add(
            x,
            self._norm(
                self._allreduce(self._moe(self._norm(x, "post_attention_layernorm"), prefill=prefill)), "post_ffn_norm"
            ),
        )

    def _qkv(self, x, cos, sin):
        s = x.shape[-2]
        packed = self._input_projection(x)
        packed = ttnn.to_memory_config(packed, DRAM)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(packed, num_heads=12, num_kv_heads=1, transpose_k_heads=False)
        q, k, v = [ttnn.slice(t, (0, 0, 0, 0), (1, h, s, 128)) for t, h in [(q, 12), (k, 1), (v, 1)]]
        q = self._norm(q, "self_attn.q_norm")
        k = self._norm(k, "self_attn.k_norm")
        if self.sliding:
            q = self._rope(q, cos, sin)
            k = self._rope(k, cos, sin)
        return (q, k, v)

    def prepare_prefill(self, *, page_table_host, seq_len, start_pos=0, chunk_size=None):
        """Keep logical lengths unrestricted while bounding circular-cache writes."""
        chunk_size = self.policy.prefill_chunk_size if chunk_size is None else chunk_size
        if chunk_size <= 0 or chunk_size % 32:
            raise ValueError("Physical chunk size must be a positive multiple of 32")
        if self.sliding_cache_tokens is not None:
            chunk_size = min(chunk_size, self.sliding_cache_tokens - 512)
        return super().prepare_prefill(
            page_table_host=page_table_host, seq_len=seq_len, start_pos=start_pos, chunk_size=chunk_size
        )

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
        if self.sliding_cache_tokens is not None and hidden_states.shape[-2] > self.sliding_cache_tokens - 512:
            raise ValueError("Physical prefill chunk exceeds circular-cache capacity minus the 512-token prefix")
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
        y = ttnn.reshape(ttnn.transformer.concatenate_heads(y), (1, 1, hidden_states.shape[-2], 1536))
        return self._finish(hidden_states, self._linear(y, self.projections["o_proj"]), prefill=True)

    def decode_forward(self, hidden_states, *, kv_cache, page_table, current_pos, cos=None, sin=None):
        """One token per request: [1,1,B,2560], tensor positions [B], B table rows.

        Every op is device-only and can be captured in a TTNN execution trace.
        Refresh stable input/position/RoPE/page-table buffers before replay.
        """
        batch = hidden_states.shape[-2]
        if tuple(hidden_states.shape) != (1, 1, batch, self.hidden_width) or not 1 <= batch <= 32:
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
        packed = self._input_projection(hidden_states)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(packed, num_heads=12, num_kv_heads=1, memory_config=mem)
        q = self._norm(ttnn.to_memory_config(q, DRAM), "self_attn.q_norm")
        k = self._norm(ttnn.to_memory_config(k, DRAM), "self_attn.k_norm")
        if self.sliding:
            qmem = ttnn.create_sharded_memory_config(
                (32, 128), shard_grid, ttnn.ShardStrategy.HEIGHT, use_height_and_width_as_shard_shape=True
            )
            q = ttnn.to_memory_config(q, qmem)
            k = ttnn.to_memory_config(k, mem)
            c = ttnn.to_memory_config(ttnn.permute(cos, (0, 2, 1, 3)), mem)
            s = ttnn.to_memory_config(ttnn.permute(sin, (0, 2, 1, 3)), mem)
            q = ttnn.experimental.rotary_embedding_hf(q, c, s, is_decode_mode=True, compute_kernel_config=self.compute)
            k = ttnn.experimental.rotary_embedding_hf(k, c, s, is_decode_mode=True, compute_kernel_config=self.compute)
        k = ttnn.to_memory_config(k, mem)
        v = ttnn.to_memory_config(v, mem)
        # Absolute positions still drive RoPE/attention. Only the cache write
        # lookup wraps into the configured per-request physical-page period.
        # Short ordinary page tables (e.g. batch fixtures) need no wrapping.
        modulo = self.sliding_cache_tokens
        if modulo is not None and page_table.shape[-1] * 32 <= modulo:
            modulo = None
        ttnn.experimental.paged_update_cache(
            kv_cache[0], k, update_idxs_tensor=current_pos, page_table=page_table, cache_position_modulo=modulo
        )
        ttnn.experimental.paged_update_cache(
            kv_cache[1], v, update_idxs_tensor=current_pos, page_table=page_table, cache_position_modulo=modulo
        )
        k_chunk = self.policy.decode_k_chunk
        cores_per_head = self.policy.decode_cores_per_head
        capacity = page_table.shape[-1] * 32
        if not self.sliding and capacity > 8192:
            cores_per_head = self.policy.decode_long_cores_per_head or cores_per_head
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
        y = ttnn.reshape(y, (1, 1, batch, 1536))
        if batch > 1:
            # SDPA skips inactive (-1) rows without writing its output. Select
            # zero before projection/routing so allocator history cannot enter
            # the batch's expert union. Multiplication would retain NaNs.
            positions = ttnn.to_layout(ttnn.reshape(current_pos, (1, 1, 1, batch)), ttnn.TILE_LAYOUT)
            active = ttnn.transpose(ttnn.typecast(ttnn.gez(positions), ttnn.bfloat16), -2, -1)
            y = ttnn.where(active, y, 0.0, memory_config=DRAM)
        return self._finish(hidden_states, self._linear(y, self.projections["o_proj"]))

    def _allreduce(self, x):
        if x.shape[-1] == 640:
            return x
        if self.policy.ccl_mode == "direct" and self.policy.residual_layout == "replicated" and x.shape[-2] <= 32:
            i = self.ar_index
            self.ar_index = 1 - i
            x = ttnn.to_memory_config(x, self.residual_mem)
            dtype = getattr(ttnn, self.policy.ccl_dtype)
            if x.dtype != dtype:
                x = ttnn.typecast(x, dtype)
            result = ttnn.experimental.all_reduce_async(
                x,
                self.ar_buffers[i],
                cluster_axis=1,
                mesh_device=self.device,
                multi_device_global_semaphore=self.ar_semaphores[i],
                memory_config=self.residual_mem,
                dtype=ttnn.bfloat16,
                topology=self.ccl.topology,
                num_links=self.ccl.num_links,
            )
            return result
        x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG if x.shape[-2] <= 32 else DRAM)
        mem = x.memory_config()
        dtype = getattr(ttnn, self.policy.prefill_ccl_dtype if x.shape[-2] > 32 else self.policy.ccl_dtype)
        if x.dtype != dtype:
            x = ttnn.typecast(x, dtype)
        scattered = ttnn.experimental.reduce_scatter_minimal_async(
            x,
            dim=3,
            cluster_axis=1,
            topology=self.ccl.topology,
            multi_device_global_semaphore=self.ccl.get_rs_ping_pong_semaphore(),
            barrier_semaphore=self.ccl.get_barrier_semaphore(),
            num_links=self.ccl.num_links,
            memory_config=mem,
        )
        if self.policy.residual_layout == "sharded":
            return ttnn.typecast(scattered, ttnn.bfloat16) if scattered.dtype != ttnn.bfloat16 else scattered
        result = ttnn.experimental.all_gather_async(
            scattered,
            dim=3,
            cluster_axis=1,
            mesh_device=self.device,
            topology=self.ccl.topology,
            multi_device_global_semaphore=self.ccl.get_ag_ping_pong_semaphore(),
            barrier_semaphore=self.ccl.get_barrier_semaphore(),
            num_links=self.ccl.num_links,
            memory_config=mem,
        )
        return ttnn.typecast(result, ttnn.bfloat16) if result.dtype != ttnn.bfloat16 else result

    def _gather(self, x):
        return ttnn.experimental.all_gather_async(
            x,
            dim=3,
            cluster_axis=1,
            mesh_device=self.device,
            topology=self.ccl.topology,
            multi_device_global_semaphore=self.ccl.get_ag_ping_pong_semaphore(),
            barrier_semaphore=self.ccl.get_barrier_semaphore(),
            num_links=self.ccl.num_links,
            memory_config=x.memory_config(),
        )

    def _gather_input(self, x):
        if self.policy.residual_layout != "sharded":
            return x
        x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG if x.shape[-2] <= 32 else DRAM)
        dtype = getattr(ttnn, self.policy.ccl_dtype)
        if x.dtype != dtype:
            x = ttnn.typecast(x, dtype)
        result = self._gather(x)
        return ttnn.typecast(result, ttnn.bfloat16) if result.dtype != ttnn.bfloat16 else result

    def _norm(self, x, name):
        if self.policy.residual_layout != "sharded" or x.shape[-1] != 640:
            return super()._norm(x, name)
        x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG if x.shape[-2] <= 32 else DRAM)
        stats = ttnn.rms_norm_pre_all_gather(x, compute_kernel_config=self.compute, dtype=ttnn.bfloat16)
        stats = self._gather(stats)
        return ttnn.rms_norm_post_all_gather(
            x,
            stats,
            epsilon=self.config.rms_norm_eps,
            weight=self.shard_norms[name],
            compute_kernel_config=self.compute,
            memory_config=x.memory_config(),
        )

    def prefill_forward(self, hidden_states, *, kv_cache, plan):
        """Logical prefill [1,B,S,self.hidden_width] -> [1,B,S,self.hidden_width].

        Accepts every S>=1 within context, including continuation at arbitrary
        positions. A setup-time plan supplies device page mappings and positions.
        Chunking/padding/slicing stay on device. KV entries before start_pos are
        preserved. Padding only occupies future rows in this request's own pages;
        causal attention cannot see it, and later append overwrites those rows.
        """
        seq_len = plan["seq_len"]
        batch = plan["batch"]
        if tuple(hidden_states.shape) != (1, batch, seq_len, self.hidden_width):
            raise ValueError("Prefill input shape does not match its plan")
        results = []
        for slot, chunks in enumerate(plan["slots"]):
            outputs = []
            for offset, count, physical, decode, kw in chunks:
                x = ttnn.slice(hidden_states, (0, slot, offset, 0), (1, slot + 1, offset + count, self.hidden_width))
                if physical != count:
                    x = ttnn.pad(x, ((0, 0), (0, 0), (0, physical - count), (0, 0)), value=0.0)
                y = (self.decode_forward if decode else self.prefill_chunk_forward)(x, kv_cache=kv_cache, **kw)
                y = ttnn.to_memory_config(y, DRAM)
                outputs.append(ttnn.slice(y, (0, 0, 0, 0), (1, 1, count, self.hidden_width)))
            results.append(ttnn.concat(outputs, dim=2) if len(outputs) > 1 else outputs[0])
        return ttnn.concat(results, dim=1) if batch > 1 else results[0]

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
        if self.policy.prefix_matmul and tokens <= 8192:
            # Every aligned count is a multiple of32 in[0,8192], exactly BF16.
            # FP32 accumulation/output preserves every integer prefix sum;
            # the triangular ones matrix replaces the serial cumsum kernel.
            prefix = ttnn.matmul(
                ttnn.typecast(aligned, ttnn.bfloat16),
                self.expert_prefix_sum,
                dtype=ttnn.float32,
                program_config=ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                    compute_with_storage_grid_size=(self.policy.prefill_prefix_cores, 1),
                    in0_block_w=12,
                    out_subblock_h=1,
                    out_subblock_w=12 // self.policy.prefill_prefix_cores,
                    per_core_M=1,
                    per_core_N=12 // self.policy.prefill_prefix_cores,
                    fuse_batch=False,
                    mcast_in0=True,
                ),
                compute_kernel_config=self.compute,
                memory_config=DRAM,
            )
        else:
            # Preserve exact offsets for explicitly requested larger physical
            # chunks, whose aligned counts need more than BF16 precision.
            prefix = ttnn.cumsum(aligned, dim=3)
        offsets = ttnn.subtract(prefix, aligned)

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
        gate = ttnn.slice(both, (0, 0, 0, 0), (1, 1, tokens, 128))
        up = ttnn.slice(both, (0, 0, 0, 128), (1, 1, tokens, 256))
        middle = ttnn.multiply(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
        return ttnn.add(routed, self._linear(middle, self.shared["down_proj"]), dtype=ttnn.bfloat16)

    def _input_projection(self, x):
        n = self._norm(x, "input_layernorm")
        if not self.policy.fused_qkv or x.shape[-2] > 32:
            return self._linear(self._gather_input(n), self.projections["qkv"])
        if self.policy.residual_layout != "sharded":
            raise ValueError("Fused QKV requires sharded residual")
        n = ttnn.to_memory_config(n, ttnn.L1_MEMORY_CONFIG)
        n = ttnn.typecast(n, getattr(ttnn, self.policy.ccl_dtype))
        if self.policy.fused_qkv == "minimal":
            return ttnn.experimental.all_gather_minimal_matmul_async(
                n,
                ttnn.reshape(self.projections["qkv"], (1, 1, 2560, 1792)),
                persistent_output_buffer=self.qkv_gather,
                config=ttnn.MinimalMatmulConfig(
                    M_block_size=1,
                    K_block_size=5,
                    N_block_size=2,
                    subblock_h=1,
                    subblock_w=2,
                    compute_with_storage_grid_size=(8, 4),
                ),
                multi_device_global_semaphore=self.ccl.get_ag_ping_pong_semaphore(),
                barrier_semaphore=self.ccl.get_barrier_semaphore(),
                topology=self.ccl.topology,
                cluster_axis=1,
                num_links=self.ccl.num_links,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                dtype=ttnn.bfloat16,
                compute_kernel_config=self.projection_info[id(self.projections["qkv"])]["compute"],
                force_transpose=True,
                num_workers_per_link=4,
            )[0]
        gather_mem = ttnn.create_sharded_memory_config(
            (32, 640),
            ttnn.CoreGrid(x=4, y=1),
            ttnn.ShardStrategy.WIDTH,
            use_height_and_width_as_shard_shape=True,
        )
        weight = ttnn.reshape(self.projections["qkv"], (1, 1, 2560, 1792))
        program = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(8, 4),
            in0_block_w=20,
            out_subblock_h=1,
            out_subblock_w=1,
            per_core_M=1,
            per_core_N=2,
            fuse_batch=True,
            mcast_in0=True,
        )
        _, result = ttnn.experimental.all_gather_matmul_async(
            n,
            weight,
            persistent_output_buffer=None,
            dim=3,
            multi_device_global_semaphore=self.ccl.get_ag_ping_pong_semaphore(),
            all_gather_core_grid_offset=(0, 4),
            barrier_semaphore=self.ccl.get_barrier_semaphore(),
            num_links=self.ccl.num_links,
            topology=self.ccl.topology,
            memory_config_ag=gather_mem,
            memory_config_mm=ttnn.L1_MEMORY_CONFIG,
            program_config=program,
            compute_kernel_config=self.projection_info[id(self.projections["qkv"])]["compute"],
            dtype=ttnn.bfloat16,
        )
        return result

    def _prefill_linear(self, x, w, dtype, compute):
        if x.shape[-2] <= 1024 or self.policy.prefill_large_kind != "minimal":
            return super()._prefill_linear(x, w, dtype, compute)
        kt = w.shape[-2] // 32
        requested = self.policy.prefill_large_projection_k
        if requested is None:
            return super()._prefill_linear(x, w, dtype, compute)
        kblock = max(v for v in range(1, min(kt, requested) + 1) if kt % v == 0)
        nblock = self.policy.prefill_large_minimal_n
        return ttnn.experimental.minimal_matmul(
            x,
            w,
            config=ttnn.MinimalMatmulConfig(
                M_block_size=4,
                K_block_size=kblock,
                N_block_size=nblock,
                subblock_h=1,
                subblock_w=min(nblock, 4),
                compute_with_storage_grid_size=ttnn.CoreCoord(*self.policy.prefill_projection_grid),
            ),
            dtype=dtype,
            memory_config=DRAM,
            compute_kernel_config=compute,
        )

    def _linear(self, x, w, *, dtype=ttnn.bfloat16, activation=None):
        if id(w) in self.projection_info:
            activation_dtype = getattr(ttnn, self.policy.projection_activation_dtype)
            if x.dtype != activation_dtype:
                x = ttnn.typecast(x, activation_dtype)
        if self.policy.fused_wo and w is self.projections["o_proj"] and x.shape[-2] == 1:
            if self.policy.residual_layout != "sharded":
                raise ValueError("Fused WO candidate carries sharded residual")
            x = ttnn.to_memory_config(x, DRAM)
            program = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(8, 6),
                in0_block_w=4,
                out_subblock_h=1,
                out_subblock_w=1,
                per_core_M=1,
                per_core_N=10,
                out_block_w=5,
                transpose_mcast=False,
                fuse_batch=False,
            )
            _, result = ttnn.experimental.matmul_reduce_scatter_async(
                x,
                ttnn.reshape(w, (1, 1, 1536, 2560)),
                persistent_intermediate_buffer=self.mmrs_intermediate,
                persistent_output_buffer=self.mmrs_output,
                dim=3,
                multi_device_global_semaphore=self.ccl.get_rs_ping_pong_semaphore(),
                reduce_scatter_core_grid_offset=(0, 6),
                barrier_semaphore=self.ccl.get_barrier_semaphore(),
                num_links=1,
                memory_config_rs=DRAM,
                memory_config_mm=DRAM,
                topology=self.ccl.topology,
                program_config=program,
                compute_kernel_config=self.projection_info[id(w)]["compute"],
                dtype=dtype,
            )
            return result
        return super()._linear(x, w, dtype=dtype, activation=activation)
