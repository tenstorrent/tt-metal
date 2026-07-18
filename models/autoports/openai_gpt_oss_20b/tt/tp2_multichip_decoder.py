# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Two-chip tensor-parallel GPT-OSS-20B decoder layer.

The implementation is intentionally specialized for the Blackhole P300 1x2
mesh available to the autoport run.  It keeps the optimized decoder's BF16
attention and routed BFP8 sparse-expert policy, but fractures attention heads,
expert intermediate dimensions, and the paged KV cache over TP=2.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import ttnn
from models.autoports.openai_gpt_oss_20b.tt.functional_decoder import (
    EMITTED_BATCH,
    EMITTED_CACHE_LENGTH,
    EMITTED_PREFILL_SEQUENCE,
    _as_replicated_tensor,
    _dense_expert_weight,
    _require_tensor,
)
from models.autoports.openai_gpt_oss_20b.tt.optimized_decoder import (
    OptimizationConfig,
    OptimizedDecoder,
    OptimizedGPTOSSProgramConfig,
)
from models.demos.gpt_oss.config import MeshConfig, ModeConfig
from models.demos.gpt_oss.tt.ccl import CCLManager
from models.demos.gpt_oss.tt.experts import ExpertConfig, Experts
from models.demos.gpt_oss.tt.experts.operations import apply_swiglu

TARGET_MESH_SHAPE = (1, 2)
TP_DEGREE = 2
PAGE_BLOCK_SIZE = 64
SUPPORTED_CONTEXT = 131_072


@dataclass(frozen=True)
class MultichipConfig:
    """Static P300 tensor-parallel policy."""

    num_links: int = 1
    page_block_size: int = PAGE_BLOCK_SIZE
    use_optimized_decode_layouts: bool = True
    use_sharded_decode_input_norm: bool = True
    use_sharded_decode_post_attention_norm: bool = True
    use_native_paged_sliding_attention: bool = False
    qkv_input_cores: int = 30
    qkv_in0_block_w: int = 3
    qkv_output_tiles_per_core: int = 2
    qkv_out_subblock_w: int = 2
    expert_gate_up_cores: tuple[int, int] = (5, 9)
    expert_down_cores: tuple[int, int] = (9, 10)
    expert_gate_up_in0_block_w: int = 45
    expert_down_in0_block_w: int = 45
    expert_gate_up_subblock_w: int = 1
    expert_down_subblock_w: int = 1
    prefill_expert_gate_up_cores: tuple[int, int] = (3, 5)
    prefill_expert_down_cores: tuple[int, int] = (5, 6)
    prefill_expert_gate_up_subblock_w: int = 3
    prefill_expert_down_subblock_w: int = 3
    active_prefill_chunk_size: int = 128


def _validate_qkv_geometry(config, *, k_tiles, n_tiles, grid_x, grid_y):
    """Validate and derive the P300 decode-QKV 1D matmul shard geometry."""

    named_values = {
        "qkv_input_cores": config.qkv_input_cores,
        "qkv_in0_block_w": config.qkv_in0_block_w,
        "qkv_output_tiles_per_core": config.qkv_output_tiles_per_core,
        "qkv_out_subblock_w": config.qkv_out_subblock_w,
    }
    for name, value in named_values.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")

    input_cores = config.qkv_input_cores
    in0_block_w = config.qkv_in0_block_w
    output_tiles_per_core = config.qkv_output_tiles_per_core
    out_subblock_w = config.qkv_out_subblock_w
    if k_tiles % input_cores:
        raise ValueError(f"qkv_input_cores={input_cores} must divide K tiles={k_tiles}")
    input_shard_tiles = k_tiles // input_cores
    if k_tiles % in0_block_w or input_shard_tiles % in0_block_w:
        raise ValueError(
            f"qkv_in0_block_w={in0_block_w} must divide both K tiles={k_tiles} "
            f"and per-core input shard tiles={input_shard_tiles}"
        )
    if output_tiles_per_core > n_tiles:
        raise ValueError(f"qkv_output_tiles_per_core={output_tiles_per_core} exceeds output N tiles={n_tiles}")
    if output_tiles_per_core % out_subblock_w:
        raise ValueError(
            f"qkv_out_subblock_w={out_subblock_w} must divide " f"qkv_output_tiles_per_core={output_tiles_per_core}"
        )
    # HiFi4 with FP32 destination accumulation and half-DST synchronization
    # leaves four 32x32 destination tiles for the output subblock.
    if out_subblock_w > 4:
        raise ValueError(f"qkv_out_subblock_w={out_subblock_w} exceeds the HiFi4 FP32-DST limit of 4")

    output_cores = math.ceil(n_tiles / output_tiles_per_core)
    program_rows = math.ceil(max(input_cores, output_cores) / grid_x)
    if program_rows > grid_y:
        raise ValueError(
            f"QKV geometry needs {program_rows} rows of {grid_x} cores, but the device grid is {grid_x}x{grid_y}"
        )
    return input_shard_tiles, output_cores, program_rows


def _shard_to_tp(tensor, *, mesh_device, dim, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(
        tensor.contiguous(),
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=dim),
        layout=layout,
        dtype=dtype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


class MultichipDecoder(OptimizedDecoder):
    """Real TP=2 decoder with a replicated, stack-compatible residual stream."""

    def __init__(
        self,
        *,
        multichip_config: MultichipConfig | None = None,
        optimization_config: OptimizationConfig | None = None,
        **kwargs,
    ):
        self.multichip_config = multichip_config or MultichipConfig()
        if optimization_config is None:
            optimization_config = OptimizationConfig().with_changes(
                use_shard_advisor_layouts=False,
                use_shard_advisor_attention_layouts=False,
                use_shard_advisor_moe_layouts=False,
                use_sparse_experts=True,
                use_explicit_sliding_mask=not self.multichip_config.use_native_paged_sliding_attention,
                expert_gate_up_cores=self.multichip_config.expert_gate_up_cores,
                expert_down_cores=self.multichip_config.expert_down_cores,
                expert_gate_up_in0_block_w=self.multichip_config.expert_gate_up_in0_block_w,
                expert_down_in0_block_w=self.multichip_config.expert_down_in0_block_w,
            )
        super().__init__(optimization_config=optimization_config, **kwargs)
        self._configure_multichip()

    def _configure_multichip(self):
        if tuple(self.mesh_device.shape) != TARGET_MESH_SHAPE:
            raise ValueError(
                f"MultichipDecoder is specialized for mesh {TARGET_MESH_SHAPE}, got {tuple(self.mesh_device.shape)}"
            )
        if self.num_heads % TP_DEGREE or self.num_kv_heads % TP_DEGREE:
            raise ValueError("query and KV head counts must divide evenly over TP=2")
        if self.intermediate_size % TP_DEGREE:
            raise ValueError("expert intermediate size must divide evenly over TP=2")
        if self.max_cache_len > SUPPORTED_CONTEXT:
            raise ValueError(f"max_cache_len must be in [1, {SUPPORTED_CONTEXT}], got {self.max_cache_len}")
        if self.multichip_config.page_block_size % ttnn.TILE_SIZE:
            raise ValueError("page_block_size must be tile aligned")
        if (
            self.multichip_config.active_prefill_chunk_size <= 0
            or self.multichip_config.active_prefill_chunk_size % ttnn.TILE_SIZE
        ):
            raise ValueError("active_prefill_chunk_size must be a positive multiple of 32")

        self.local_num_heads = self.num_heads // TP_DEGREE
        self.local_num_kv_heads = self.num_kv_heads // TP_DEGREE
        self.local_intermediate_size = self.intermediate_size // TP_DEGREE
        if self.sliding_window is not None and not self.multichip_config.use_native_paged_sliding_attention:
            self.sliding_sdpa_program_config = (
                ttnn.SDPAProgramConfig(
                    compute_with_storage_grid_size=self.optimization_config.sdpa_grid,
                    exp_approx_mode=False,
                    q_chunk_size=((self.local_num_heads + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE * ttnn.TILE_SIZE),
                    k_chunk_size=self.optimization_config.sdpa_k_chunk_size,
                )
                if self.optimization_config.explicit_sliding_sdpa_program_config
                else None
            )
        self.num_cache_blocks = math.ceil(self.max_cache_len / self.multichip_config.page_block_size)
        self.mesh_config = MeshConfig(
            TARGET_MESH_SHAPE,
            decode=ModeConfig(tp=TP_DEGREE, ep=1, sp=1),
            prefill=ModeConfig(tp=TP_DEGREE, ep=1, sp=1),
            tp_axis=1,
        )
        self.ccl_manager = CCLManager(
            self.mesh_device,
            num_links=self.multichip_config.num_links,
            topology=ttnn.Topology.Ring,
        )

        device_grid = self.mesh_device.compute_with_storage_grid_size()
        batch_grid = ttnn.num_cores_to_corerangeset(
            self.batch,
            ttnn.CoreCoord(device_grid.x, device_grid.y),
            row_wise=True,
        )
        self.decode_heads_memory_config = ttnn.create_sharded_memory_config(
            shape=(self.local_num_heads, self.head_dim),
            core_grid=batch_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        def width_config(cores, shard_width):
            grid = ttnn.num_cores_to_corerangeset(
                cores,
                ttnn.CoreCoord(device_grid.x, device_grid.y),
                row_wise=True,
            )
            return ttnn.create_sharded_memory_config(
                shape=(ttnn.TILE_SIZE, shard_width),
                core_grid=grid,
                strategy=ttnn.ShardStrategy.WIDTH,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )

        qkv_width = self.local_num_heads * self.head_dim + 2 * self.local_num_kv_heads * self.head_dim
        qkv_input_shard_tiles, qkv_output_cores, qkv_program_rows = _validate_qkv_geometry(
            self.multichip_config,
            k_tiles=self.hidden_size // ttnn.TILE_SIZE,
            n_tiles=qkv_width // ttnn.TILE_SIZE,
            grid_x=device_grid.x,
            grid_y=device_grid.y,
        )
        self.tp_qkv_input_config = width_config(
            self.multichip_config.qkv_input_cores,
            qkv_input_shard_tiles * ttnn.TILE_SIZE,
        )
        self.tp_qkv_output_config = width_config(
            qkv_output_cores,
            self.multichip_config.qkv_output_tiles_per_core * ttnn.TILE_SIZE,
        )
        self.tp_o_output_config = width_config(90, self.hidden_size // 90)
        self.tp_qkv_program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            # P300 is 11 cores wide.  The 1D matmul factory reconstructs the
            # input-sender and output-worker sets in physical row-major order,
            # so the extent must cover the larger set with the same width.
            compute_with_storage_grid_size=ttnn.CoreCoord(device_grid.x, qkv_program_rows),
            in0_block_w=self.multichip_config.qkv_in0_block_w,
            out_subblock_h=1,
            out_subblock_w=self.multichip_config.qkv_out_subblock_w,
            out_block_h=1,
            out_block_w=self.multichip_config.qkv_output_tiles_per_core,
            per_core_M=1,
            per_core_N=self.multichip_config.qkv_output_tiles_per_core,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        )
        self.tp_o_program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(11, 9),
            in0_block_w=8,
            out_subblock_h=1,
            out_subblock_w=1,
            out_block_h=1,
            out_block_w=1,
            per_core_M=1,
            per_core_N=1,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        )

    @classmethod
    def from_state_dict(
        cls,
        state_dict,
        *,
        hf_config,
        layer_idx,
        mesh_device,
        batch=EMITTED_BATCH,
        max_cache_len=EMITTED_CACHE_LENGTH,
        multichip_config: MultichipConfig | None = None,
        optimization_config: OptimizationConfig | None = None,
        **_kwargs,
    ):
        """Load reordered attention shards and TP sparse-expert weights."""
        import torch
        import torch.nn.functional as F
        from transformers.models.gpt_oss.modeling_gpt_oss import GptOssRotaryEmbedding

        if batch != EMITTED_BATCH:
            raise ValueError(f"The emitted workload batch is {EMITTED_BATCH}, got {batch}")
        if tuple(mesh_device.shape) != TARGET_MESH_SHAPE:
            raise ValueError(f"MultichipDecoder requires a {TARGET_MESH_SHAPE} mesh")
        if not 1 <= max_cache_len <= SUPPORTED_CONTEXT:
            raise ValueError(f"max_cache_len must be in [1, {SUPPORTED_CONTEXT}], got {max_cache_len}")
        if not 0 <= layer_idx < int(hf_config.num_hidden_layers):
            raise ValueError(f"layer_idx={layer_idx} is outside the configured layer range")

        hidden_size = int(hf_config.hidden_size)
        num_heads = int(hf_config.num_attention_heads)
        num_kv_heads = int(hf_config.num_key_value_heads)
        head_dim = int(hf_config.head_dim)
        intermediate_size = int(hf_config.intermediate_size)
        num_experts = int(hf_config.num_local_experts)
        q_dim = num_heads * head_dim
        kv_dim = num_kv_heads * head_dim
        if (hidden_size, q_dim, kv_dim, intermediate_size, num_experts) != (2880, 4096, 512, 2880, 32):
            raise ValueError("MultichipDecoder is specialized for the GPT-OSS-20B decoder geometry")

        q_weight = _require_tensor(state_dict, layer_idx, "self_attn.q_proj.weight")
        k_weight = _require_tensor(state_dict, layer_idx, "self_attn.k_proj.weight")
        v_weight = _require_tensor(state_dict, layer_idx, "self_attn.v_proj.weight")
        q_bias = _require_tensor(state_dict, layer_idx, "self_attn.q_proj.bias")
        k_bias = _require_tensor(state_dict, layer_idx, "self_attn.k_proj.bias")
        v_bias = _require_tensor(state_dict, layer_idx, "self_attn.v_proj.bias")

        qkv_rank_chunks = []
        qkv_bias_rank_chunks = []
        for rank in range(TP_DEGREE):
            qkv_rank_chunks.append(
                torch.cat(
                    [
                        torch.chunk(q_weight, TP_DEGREE, dim=0)[rank].transpose(-2, -1),
                        torch.chunk(k_weight, TP_DEGREE, dim=0)[rank].transpose(-2, -1),
                        torch.chunk(v_weight, TP_DEGREE, dim=0)[rank].transpose(-2, -1),
                    ],
                    dim=-1,
                )
            )
            qkv_bias_rank_chunks.append(
                torch.cat(
                    [
                        torch.chunk(q_bias, TP_DEGREE, dim=0)[rank],
                        torch.chunk(k_bias, TP_DEGREE, dim=0)[rank],
                        torch.chunk(v_bias, TP_DEGREE, dim=0)[rank],
                    ],
                    dim=-1,
                )
            )
        qkv_weight = torch.cat(qkv_rank_chunks, dim=-1).to(torch.bfloat16)
        qkv_bias = torch.cat(qkv_bias_rank_chunks, dim=-1).reshape(1, 1, -1).to(torch.bfloat16)

        o_weight = _require_tensor(state_dict, layer_idx, "self_attn.o_proj.weight")
        o_bias = _require_tensor(state_dict, layer_idx, "self_attn.o_proj.bias").to(torch.bfloat16)
        rank_selective_o_bias = torch.cat([o_bias, torch.zeros_like(o_bias)], dim=-1).reshape(1, 1, -1)
        router_weight = _require_tensor(state_dict, layer_idx, "mlp.router.weight")
        router_bias = _require_tensor(state_dict, layer_idx, "mlp.router.bias")
        input_norm = _require_tensor(state_dict, layer_idx, "input_layernorm.weight")
        post_attention_norm = _require_tensor(state_dict, layer_idx, "post_attention_layernorm.weight")
        sinks = _require_tensor(state_dict, layer_idx, "self_attn.sinks").to(torch.bfloat16)

        scale = head_dim**-0.5
        prefill_sinks = (sinks.reshape(1, num_heads, 1, 1) / scale).to(torch.bfloat16)
        local_decode_sinks = [
            F.pad(chunk.reshape(num_heads // TP_DEGREE, 1), (0, ttnn.TILE_SIZE - 1)) / scale
            for chunk in torch.chunk(sinks, TP_DEGREE, dim=0)
        ]
        decode_sinks = torch.cat(local_decode_sinks, dim=0).to(torch.bfloat16)

        rotary = GptOssRotaryEmbedding(hf_config)
        positions = torch.arange(max_cache_len, dtype=torch.long).unsqueeze(0)
        rotary_input = torch.empty(1, 1, max_cache_len, head_dim, dtype=torch.bfloat16)
        cos_half, sin_half = rotary(rotary_input, positions)
        cos = torch.cat([cos_half, cos_half], dim=-1).unsqueeze(1)
        sin = torch.cat([sin_half, sin_half], dim=-1).unsqueeze(1)

        norm_shape = (1, 1, hidden_size // ttnn.TILE_SIZE, ttnn.TILE_SIZE)
        weights = {
            "input_norm": _as_replicated_tensor(
                input_norm.reshape(norm_shape).to(torch.bfloat16),
                mesh_device=mesh_device,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            ),
            "post_attention_norm": _as_replicated_tensor(
                post_attention_norm.reshape(norm_shape).to(torch.bfloat16),
                mesh_device=mesh_device,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            ),
            "qkv_weight": _shard_to_tp(qkv_weight, mesh_device=mesh_device, dim=1),
            "qkv_bias": _shard_to_tp(qkv_bias, mesh_device=mesh_device, dim=2),
            "o_weight": _shard_to_tp(o_weight.transpose(-2, -1).to(torch.bfloat16), mesh_device=mesh_device, dim=0),
            "o_bias": _shard_to_tp(rank_selective_o_bias, mesh_device=mesh_device, dim=2),
            "prefill_sinks": _shard_to_tp(prefill_sinks, mesh_device=mesh_device, dim=1),
            "decode_sinks": _shard_to_tp(decode_sinks, mesh_device=mesh_device, dim=0),
            "router_weight": _as_replicated_tensor(
                router_weight.transpose(-2, -1).to(torch.bfloat16), mesh_device=mesh_device
            ),
            "router_bias": _as_replicated_tensor(
                router_bias.reshape(1, -1).float(),
                mesh_device=mesh_device,
                dtype=ttnn.float32,
            ),
        }
        decoder = cls(
            hf_config=hf_config,
            layer_idx=layer_idx,
            mesh_device=mesh_device,
            batch=batch,
            max_cache_len=max_cache_len,
            weights=weights,
            cos_cache=_as_replicated_tensor(cos, mesh_device=mesh_device),
            sin_cache=_as_replicated_tensor(sin, mesh_device=mesh_device),
            multichip_config=multichip_config,
            optimization_config=optimization_config,
        )
        tiled_input_norm = ttnn.to_layout(decoder.weights["input_norm"], ttnn.TILE_LAYOUT)
        decoder.tp_input_norm_weight = ttnn.to_memory_config(
            ttnn.reshape(tiled_input_norm, [decoder.hidden_size]),
            decoder.advisor_residual_config,
        )
        tiled_post_attention_norm = ttnn.to_layout(decoder.weights["post_attention_norm"], ttnn.TILE_LAYOUT)
        decoder.tp_post_attention_norm_weight = ttnn.to_memory_config(
            ttnn.reshape(tiled_post_attention_norm, [decoder.hidden_size]),
            decoder.advisor_residual_config,
        )

        expert_state = {
            "gate_up_proj": _dense_expert_weight(state_dict, layer_idx, "gate_up_proj"),
            "gate_up_proj_bias": _require_tensor(state_dict, layer_idx, "mlp.experts.gate_up_proj_bias"),
            "down_proj": _dense_expert_weight(state_dict, layer_idx, "down_proj"),
            "down_proj_bias": _require_tensor(state_dict, layer_idx, "mlp.experts.down_proj_bias"),
        }
        expert_config = ExpertConfig(
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            hidden_size=hidden_size,
            num_experts_per_tok=int(hf_config.num_experts_per_tok),
            swiglu_limit=float(getattr(hf_config, "swiglu_limit", 7.0)),
            alpha=1.703125,
        )
        program_config = OptimizedGPTOSSProgramConfig(
            decode_gate_up_cores=decoder.optimization_config.expert_gate_up_cores,
            decode_down_cores=decoder.optimization_config.expert_down_cores,
            prefill_gate_up_cores=decoder.multichip_config.prefill_expert_gate_up_cores,
            prefill_down_cores=decoder.multichip_config.prefill_expert_down_cores,
            decode_gate_up_in0_block_w=decoder.optimization_config.expert_gate_up_in0_block_w,
            decode_down_in0_block_w=decoder.optimization_config.expert_down_in0_block_w,
            prefill_gate_up_in0_block_w=decoder.optimization_config.expert_gate_up_in0_block_w,
            prefill_down_in0_block_w=decoder.optimization_config.expert_down_in0_block_w,
            decode_gate_up_subblock_w=decoder.multichip_config.expert_gate_up_subblock_w,
            decode_down_subblock_w=decoder.multichip_config.expert_down_subblock_w,
            prefill_gate_up_subblock_w=decoder.multichip_config.prefill_expert_gate_up_subblock_w,
            prefill_down_subblock_w=decoder.multichip_config.prefill_expert_down_subblock_w,
        )
        decoder.experts = Experts(
            mesh_device=mesh_device,
            config=expert_config,
            state_dict=expert_state,
            ccl_manager=decoder.ccl_manager,
            mesh_config=decoder.mesh_config,
            program_config=program_config,
            weight_dtype=ttnn.bfloat8_b,
        )
        return decoder

    def create_page_table(self, physical_block_ids: Sequence[int] | None = None):
        """Create a replicated logical-page to physical-block table."""
        import torch

        if physical_block_ids is None:
            physical_block_ids = range(self.num_cache_blocks)
        physical_block_ids = tuple(int(block) for block in physical_block_ids)
        if len(physical_block_ids) != self.num_cache_blocks:
            raise ValueError(f"page table requires {self.num_cache_blocks} block ids")
        if set(physical_block_ids) != set(range(self.num_cache_blocks)):
            raise ValueError("page table block ids must be a permutation of the physical cache blocks")
        table = torch.tensor(physical_block_ids, dtype=torch.int32).reshape(self.batch, self.num_cache_blocks)
        return ttnn.from_torch(
            table,
            device=self.mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def create_kv_cache(self):
        """Allocate rank-local physical pages for four KV heads per device."""
        cache_dtype = {
            "bfloat8_b": ttnn.bfloat8_b,
            "bfloat16": ttnn.bfloat16,
        }[self.optimization_config.kv_cache_dtype]
        shape = (
            self.num_cache_blocks,
            self.local_num_kv_heads,
            self.multichip_config.page_block_size,
            self.head_dim,
        )
        return (
            ttnn.zeros(
                shape,
                dtype=cache_dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            ),
            ttnn.zeros(
                shape,
                dtype=cache_dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            ),
        )

    def _validate_page_table(self, page_table):
        expected = [self.batch, self.num_cache_blocks]
        if list(page_table.shape) != expected:
            raise ValueError(f"page_table must have shape {expected}, got {list(page_table.shape)}")

    def _all_reduce(self, tensor, *, memory_config):
        return ttnn.all_reduce(
            tensor,
            num_links=self.multichip_config.num_links,
            topology=ttnn.Topology.Ring,
            cluster_axis=1,
            memory_config=memory_config,
        )

    def _project_o_and_reduce(self, attended, *, is_decode):
        """Apply the row-parallel O projection and restore the replicated stream."""

        dram_layouts = is_decode and self.optimization_config.use_dram_sharded_attention
        optimized_layouts = is_decode and self.multichip_config.use_optimized_decode_layouts and not dram_layouts
        if dram_layouts:
            attended = ttnn.to_memory_config(attended, self.tp_dram_o_input_config)
        elif optimized_layouts:
            attended = ttnn.to_memory_config(attended, ttnn.L1_MEMORY_CONFIG)
        partial = ttnn.linear(
            attended,
            self.tp_decode_o_weight if dram_layouts else self.weights["o_weight"],
            bias=None if dram_layouts else self.weights["o_bias"],
            dtype=ttnn.bfloat16,
            memory_config=(
                self.tp_dram_o_output_config
                if dram_layouts
                else self.tp_o_output_config
                if optimized_layouts
                else ttnn.DRAM_MEMORY_CONFIG
            ),
            program_config=(
                self.tp_dram_o_program_config
                if dram_layouts
                else self.tp_o_program_config
                if optimized_layouts
                else None
            ),
            compute_kernel_config=self.decode_compute_kernel_config if is_decode else self.compute_kernel_config,
        )
        if dram_layouts:
            partial = ttnn.to_memory_config(partial, ttnn.L1_MEMORY_CONFIG)
            partial = ttnn.add(partial, self.weights["o_bias"], memory_config=ttnn.L1_MEMORY_CONFIG)
        elif optimized_layouts:
            partial = ttnn.to_memory_config(partial, ttnn.L1_MEMORY_CONFIG)
        return self._all_reduce(
            partial,
            memory_config=ttnn.L1_MEMORY_CONFIG if is_decode else ttnn.DRAM_MEMORY_CONFIG,
        )

    def _prefill_attention(self, hidden_states, key_cache, value_cache, page_table, seq_len):
        normalized = ttnn.rms_norm(
            hidden_states,
            epsilon=self.eps,
            weight=self.weights["input_norm"],
            compute_kernel_config=self.compute_kernel_config,
        )
        fused = ttnn.linear(
            normalized,
            self.weights["qkv_weight"],
            bias=self.weights["qkv_bias"],
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
        fused = ttnn.reshape(fused, [self.batch, seq_len, -1])
        query, key, value = ttnn.transformer.split_query_key_value_and_split_heads(
            fused,
            None,
            num_heads=self.local_num_heads,
            num_kv_heads=self.local_num_kv_heads,
            transpose_key=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        cos = ttnn.slice(self.cos_cache, [0, 0, 0, 0], [1, 1, seq_len, self.head_dim])
        sin = ttnn.slice(self.sin_cache, [0, 0, 0, 0], [1, 1, seq_len, self.head_dim])
        query = ttnn.experimental.rotary_embedding(query, cos, sin, None, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        key = ttnn.experimental.rotary_embedding(key, cos, sin, None, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        query = ttnn.slice(query, [0, 0, 0, 0], [1, self.local_num_heads, seq_len, self.head_dim])
        key = ttnn.slice(key, [0, 0, 0, 0], [1, self.local_num_kv_heads, seq_len, self.head_dim])
        cache_key = ttnn.typecast(key, ttnn.bfloat8_b) if key_cache.dtype == ttnn.bfloat8_b else key
        cache_value = ttnn.typecast(value, ttnn.bfloat8_b) if value_cache.dtype == ttnn.bfloat8_b else value
        ttnn.experimental.paged_fill_cache(key_cache, cache_key, page_table, batch_idx=0)
        ttnn.experimental.paged_fill_cache(value_cache, cache_value, page_table, batch_idx=0)
        attended = ttnn.transformer.scaled_dot_product_attention(
            query,
            key,
            value,
            is_causal=True,
            scale=self.scale,
            sliding_window_size=self.sliding_window,
            attention_sink=self.weights["prefill_sinks"],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
        attended = ttnn.transformer.concatenate_heads(attended, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        projected = self._project_o_and_reduce(attended, is_decode=False)
        return ttnn.add(hidden_states, ttnn.reshape(projected, [1, self.batch, seq_len, self.hidden_size]))

    def _decode_attention(
        self,
        hidden_states,
        key_cache,
        value_cache,
        page_table,
        cache_position,
        cache_position_tensor,
    ):
        dram_layouts = self.optimization_config.use_dram_sharded_attention
        optimized_layouts = self.multichip_config.use_optimized_decode_layouts and not dram_layouts
        sharded_input_norm = self.multichip_config.use_sharded_decode_input_norm
        norm_input = (
            ttnn.to_memory_config(hidden_states, self.advisor_norm_memory_config)
            if sharded_input_norm
            else hidden_states
        )
        normalized = ttnn.rms_norm(
            norm_input,
            epsilon=self.eps,
            weight=self.tp_input_norm_weight if sharded_input_norm else self.weights["input_norm"],
            program_config=self.advisor_norm_program_config if sharded_input_norm else None,
            compute_kernel_config=self.decode_compute_kernel_config,
        )
        if dram_layouts:
            normalized = ttnn.to_memory_config(normalized, self.tp_dram_qkv_input_config)
        elif optimized_layouts:
            normalized = ttnn.to_memory_config(normalized, self.tp_qkv_input_config)
        fused = ttnn.linear(
            normalized,
            self.tp_decode_qkv_weight if dram_layouts else self.weights["qkv_weight"],
            bias=None if dram_layouts else self.weights["qkv_bias"],
            dtype=ttnn.bfloat16,
            memory_config=(
                self.tp_dram_qkv_output_config
                if dram_layouts
                else self.tp_qkv_output_config
                if optimized_layouts
                else ttnn.DRAM_MEMORY_CONFIG
            ),
            program_config=(
                self.tp_dram_qkv_program_config
                if dram_layouts
                else self.tp_qkv_program_config
                if optimized_layouts
                else None
            ),
            compute_kernel_config=self.decode_compute_kernel_config,
        )
        if dram_layouts:
            fused = ttnn.add(fused, self.weights["qkv_bias"], memory_config=self.tp_dram_qkv_output_config)
            fused = ttnn.to_memory_config(fused, ttnn.L1_MEMORY_CONFIG)
        elif optimized_layouts:
            fused = ttnn.to_memory_config(fused, ttnn.L1_MEMORY_CONFIG)
        fused = ttnn.reshape(fused, [1, 1, self.batch, -1])
        query, key, value = ttnn.experimental.nlp_create_qkv_heads_decode(
            fused,
            num_heads=self.local_num_heads,
            num_kv_heads=self.local_num_kv_heads,
            memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
        )
        rope_position = ttnn.typecast(cache_position_tensor, ttnn.uint32)
        rope_position = ttnn.reshape(rope_position, [1, self.batch])
        cos = ttnn.embedding(
            rope_position,
            self.decode_cos_matrix,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        sin = ttnn.embedding(
            rope_position,
            self.decode_sin_matrix,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        cos = ttnn.transpose(ttnn.unsqueeze_to_4D(cos), 1, 2)
        sin = ttnn.transpose(ttnn.unsqueeze_to_4D(sin), 1, 2)
        cos = ttnn.interleaved_to_sharded(cos, self.decode_rope_memory_config)
        sin = ttnn.interleaved_to_sharded(sin, self.decode_rope_memory_config)
        query = ttnn.experimental.rotary_embedding_hf(
            query,
            cos,
            sin,
            is_decode_mode=True,
            memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
            compute_kernel_config=self.decode_compute_kernel_config,
        )
        key = ttnn.experimental.rotary_embedding_hf(
            key,
            cos,
            sin,
            is_decode_mode=True,
            memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
            compute_kernel_config=self.decode_compute_kernel_config,
        )
        ttnn.experimental.paged_update_cache(
            key_cache,
            key,
            update_idxs_tensor=cache_position_tensor,
            page_table=page_table,
            share_cache=False,
        )
        ttnn.experimental.paged_update_cache(
            value_cache,
            value,
            update_idxs_tensor=cache_position_tensor,
            page_table=page_table,
            share_cache=False,
        )

        explicit_sliding_mask = (
            self.sliding_window is not None and not self.multichip_config.use_native_paged_sliding_attention
        )
        attention_mask = self._decode_sliding_attention_mask(cache_position_tensor) if explicit_sliding_mask else None
        attended = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            query,
            key_cache,
            value_cache,
            page_table_tensor=page_table,
            is_causal=not explicit_sliding_mask,
            attn_mask=attention_mask,
            cur_pos_tensor=None if explicit_sliding_mask else cache_position_tensor,
            attention_sink=self.weights["decode_sinks"],
            scale=self.scale,
            sliding_window_size=None if explicit_sliding_mask else self.sliding_window,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=(
                self.sliding_sdpa_program_config if explicit_sliding_mask else self.decode_sdpa_program_config
            ),
            compute_kernel_config=(
                self.sliding_compute_kernel_config if explicit_sliding_mask else self.decode_compute_kernel_config
            ),
        )
        attended = ttnn.to_memory_config(attended, self.decode_heads_memory_config)
        attended = ttnn.experimental.nlp_concat_heads_decode(attended, num_heads=self.local_num_heads)
        attended = ttnn.slice(
            attended,
            [0, 0, 0, 0],
            [1, 1, self.batch, self.local_num_heads * self.head_dim],
            [1, 1, 1, 1],
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        )
        projected = self._project_o_and_reduce(attended, is_decode=True)
        projected = ttnn.reshape(projected, [1, self.batch, 1, self.hidden_size])
        return ttnn.add(hidden_states, projected)

    def _decode_sliding_attention_mask(self, cache_position_tensor):
        position = ttnn.typecast(cache_position_tensor, ttnn.float32)
        position = ttnn.reshape(ttnn.to_layout(position, ttnn.TILE_LAYOUT), [1, 1, 1, 1])
        window_start = ttnn.subtract(position, float(self.sliding_window - 1))
        in_window = ttnn.logical_and(
            ttnn.ge(self.decode_key_positions, window_start),
            ttnn.le(self.decode_key_positions, position),
        )
        mask = ttnn.where(in_window, 0.0, -10_000.0)
        mask = ttnn.typecast(mask, ttnn.bfloat16)
        return ttnn.repeat(mask, ttnn.Shape([1, 1, self.local_num_heads, 1]))

    def _decode_post_attention_norm(self, hidden_states):
        """Run the decode post-attention norm on the advisor's 10-core shard."""
        if not self.multichip_config.use_sharded_decode_post_attention_norm:
            return ttnn.rms_norm(
                hidden_states,
                epsilon=self.eps,
                weight=self.weights["post_attention_norm"],
                compute_kernel_config=self.compute_kernel_config,
            )
        norm_input = ttnn.to_memory_config(hidden_states, self.advisor_norm_memory_config)
        normalized = ttnn.rms_norm(
            norm_input,
            epsilon=self.eps,
            weight=self.tp_post_attention_norm_weight,
            program_config=self.advisor_norm_program_config,
            compute_kernel_config=self.decode_compute_kernel_config,
        )
        # The generic sparse expert readers require interleaved operands.  Keep
        # the sharded norm local to this boundary and make the transition
        # explicit rather than relying on an implicit runtime fallback.
        return ttnn.to_memory_config(normalized, ttnn.DRAM_MEMORY_CONFIG)

    def _active_prefill_expert_chunk(self, normalized, routing_weights):
        """Run only the gate-selected experts for one tile-aligned prefill chunk.

        Sparse matmul selects batches rather than individual rows inside an M
        tile.  Giving every token its own batch therefore makes the router mask
        token-specific: gate/up consume ``[S, 1, 1, H]`` against the generic
        TP weights and down consumes ``[S, E, 1, I_local]``.  Each selected
        route owns one physical 32-row tile; padding rows are explicitly zeroed.
        """

        seq_len = normalized.shape[2]
        if seq_len % ttnn.TILE_SIZE:
            raise ValueError("active prefill expert chunks must be tile aligned")

        weights = self.experts.weights
        program_config = self.experts.program_config
        output_tile = ttnn.Tile([ttnn.TILE_SIZE, ttnn.TILE_SIZE])

        # This reshape is deliberately a device data-movement operation, not a
        # view: the explicit pad value makes the 31 unused rows in each token's
        # physical M tile deterministic zeros.
        token_major = ttnn.reshape(
            normalized,
            [seq_len, 1, 1, self.hidden_size],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        gate_up_sparsity = ttnn.to_layout(
            ttnn.reshape(routing_weights, [seq_len, 1, 1, self.num_experts]),
            ttnn.ROW_MAJOR_LAYOUT,
        )

        gate = ttnn.sparse_matmul(
            token_major,
            weights.gate_proj,
            sparsity=gate_up_sparsity,
            nnz=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_tile=output_tile,
            program_config=program_config.get_prefill_gate_up_config(1, weights.gate_proj.shape[3], k=self.hidden_size),
            dtype=ttnn.bfloat8_b,
        )
        # Move E into the tiled row dimension before elementwise work.  Keeping
        # the singleton sparse-matmul M dimension here would make every clamp,
        # sigmoid, add, and multiply traverse its 31 physical padding rows.
        gate = ttnn.reshape(gate, [seq_len, self.num_experts, weights.intermediate_size_per_device])
        gate_bias = ttnn.reshape(
            weights.gate_proj_bias,
            [1, self.num_experts, weights.intermediate_size_per_device],
        )
        gate = ttnn.add(gate, gate_bias, output_tensor=gate)

        up = ttnn.sparse_matmul(
            token_major,
            weights.up_proj,
            sparsity=gate_up_sparsity,
            nnz=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_tile=output_tile,
            program_config=program_config.get_prefill_gate_up_config(1, weights.up_proj.shape[3], k=self.hidden_size),
            dtype=ttnn.bfloat8_b,
        )
        token_major.deallocate(True)
        up = ttnn.reshape(up, [seq_len, self.num_experts, weights.intermediate_size_per_device])
        up_bias = ttnn.reshape(
            weights.up_proj_bias,
            [1, self.num_experts, weights.intermediate_size_per_device],
        )
        up = ttnn.add(up, up_bias, output_tensor=up)

        down_input = apply_swiglu(gate, up, self.experts.config)
        down_input = ttnn.reshape(
            down_input,
            [seq_len, self.num_experts, 1, weights.intermediate_size_per_device],
        )
        down_sparsity = ttnn.reshape(gate_up_sparsity, [1, 1, seq_len, self.num_experts])
        down = ttnn.sparse_matmul(
            down_input,
            weights.down_proj,
            sparsity=down_sparsity,
            nnz=None,
            is_input_a_sparse=True,
            is_input_b_sparse=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_tile=output_tile,
            program_config=program_config.get_prefill_down_config(
                1, weights.down_proj.shape[-1], k=weights.intermediate_size_per_device
            ),
            dtype=ttnn.bfloat8_b,
        )
        down_input.deallocate(True)
        down = ttnn.reshape(down, [seq_len, self.num_experts, self.hidden_size])
        down_bias = ttnn.reshape(weights.down_proj_bias, [1, self.num_experts, self.hidden_size])
        down = ttnn.add(down, down_bias, output_tensor=down)
        route_scale = ttnn.reshape(routing_weights, [seq_len, self.num_experts, 1])
        down = ttnn.mul(down, route_scale, output_tensor=down)
        down = ttnn.sum(down, dim=1)
        return ttnn.reshape(down, [1, 1, seq_len, self.hidden_size])

    def _active_prefill_sparse_moe(self, hidden_states, normalized, routing_weights, seq_len):
        """Pad/chunk active prefill experts, then perform one TP all-reduce."""

        padded_len = math.ceil(seq_len / ttnn.TILE_SIZE) * ttnn.TILE_SIZE
        if padded_len != seq_len:
            normalized = ttnn.pad(
                normalized,
                [(0, 0), (0, 0), (0, padded_len - seq_len), (0, 0)],
                value=0.0,
            )
            routing_weights = ttnn.pad(
                routing_weights,
                [(0, padded_len - seq_len), (0, 0)],
                value=0.0,
            )

        chunk_size = self.multichip_config.active_prefill_chunk_size
        if padded_len > chunk_size:
            normalized_chunks = ttnn.split(normalized, chunk_size, dim=2)
            routing_chunks = ttnn.split(routing_weights, chunk_size, dim=0)
        else:
            normalized_chunks = [normalized]
            routing_chunks = [routing_weights]

        partial_chunks = [
            self._active_prefill_expert_chunk(normalized_chunk, routing_chunk)
            for normalized_chunk, routing_chunk in zip(normalized_chunks, routing_chunks)
        ]
        partial = partial_chunks[0] if len(partial_chunks) == 1 else ttnn.concat(partial_chunks, dim=2)
        expert_output = self._all_reduce(partial, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if padded_len != seq_len:
            expert_output = ttnn.slice(
                expert_output,
                [0, 0, 0, 0],
                [1, self.batch, seq_len, self.hidden_size],
            )
        return ttnn.add(hidden_states, expert_output)

    def _moe_forward(self, hidden_states, seq_len):
        if seq_len != 1 and not self.optimization_config.use_sparse_experts:
            return super()._moe_forward(hidden_states, seq_len)
        if seq_len != 1:
            normalized = ttnn.rms_norm(
                hidden_states,
                epsilon=self.eps,
                weight=self.weights["post_attention_norm"],
                compute_kernel_config=self.compute_kernel_config,
            )
            routing_weights = self._route(normalized, self.batch * seq_len)
            return self._active_prefill_sparse_moe(
                hidden_states,
                normalized,
                routing_weights,
                seq_len,
            )
        normalized = self._decode_post_attention_norm(hidden_states)
        routing_weights = self._route(normalized, self.batch)
        if self.optimization_config.use_sparse_experts:
            return self._sparse_moe_forward(hidden_states, normalized, routing_weights, seq_len)
        return self._dense_moe_forward(hidden_states, normalized, routing_weights, seq_len)

    def prefill_forward(self, hidden_states, *, key_cache, value_cache, page_table):
        seq_len = self._validate_hidden_states(hidden_states)
        if seq_len <= 1:
            raise ValueError("prefill_forward requires seq_len > 1")
        if seq_len > self.max_cache_len:
            raise ValueError(f"seq_len={seq_len} exceeds max_cache_len={self.max_cache_len}")
        self._validate_page_table(page_table)
        hidden_states = self._prefill_attention(hidden_states, key_cache, value_cache, page_table, seq_len)
        return self._moe_forward(hidden_states, seq_len)

    def decode_forward(
        self,
        hidden_states,
        *,
        key_cache,
        value_cache,
        page_table,
        cache_position,
        cache_position_tensor,
    ):
        self._validate_hidden_states(hidden_states, expected_seq_len=1)
        if not 0 <= cache_position < self.max_cache_len:
            raise ValueError(f"cache_position must be in [0, {self.max_cache_len}), got {cache_position}")
        self._validate_page_table(page_table)
        hidden_states = self._decode_attention(
            hidden_states,
            key_cache,
            value_cache,
            page_table,
            cache_position,
            cache_position_tensor,
        )
        return self._moe_forward(hidden_states, 1)


__all__ = [
    "EMITTED_BATCH",
    "EMITTED_CACHE_LENGTH",
    "EMITTED_PREFILL_SEQUENCE",
    "MultichipConfig",
    "MultichipDecoder",
    "PAGE_BLOCK_SIZE",
    "SUPPORTED_CONTEXT",
    "TARGET_MESH_SHAPE",
    "TP_DEGREE",
]
