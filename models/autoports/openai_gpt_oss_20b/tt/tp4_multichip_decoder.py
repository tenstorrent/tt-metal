# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Full-mesh GPT-OSS-20B decoder for the four-chip P300 ring.

Attention follows the compiler-emitted 1x4 sharding prior exactly.  The
default routed-MoE policy keeps eight whole experts per rank, matching the
compiler prior and executing only gate-selected experts.  A TP4 expert mode
remains available as the measured load-balanced alternative.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import SimpleNamespace
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
from models.autoports.openai_gpt_oss_20b.tt.optimized_decoder import OptimizationConfig, OptimizedGPTOSSProgramConfig
from models.autoports.openai_gpt_oss_20b.tt.tp2_multichip_decoder import MultichipDecoder as _TP2MultichipDecoder
from models.autoports.openai_gpt_oss_20b.tt.tp2_multichip_decoder import _shard_to_tp, _validate_qkv_geometry
from models.demos.gpt_oss.config import MeshConfig, ModeConfig
from models.demos.gpt_oss.tt.ccl import CCLManager
from models.demos.gpt_oss.tt.experts import ExpertConfig, Experts
from models.demos.gpt_oss.tt.experts.operations import apply_swiglu
from models.demos.gpt_oss.tt.experts.weights import ExpertWeights

TARGET_MESH_SHAPE = (1, 4)
TP_DEGREE = 4
EP_DEGREE = 4
PAGE_BLOCK_SIZE = 64
SUPPORTED_CONTEXT = 131_072
EXPERT_STRATEGY_TP = "tp"
EXPERT_STRATEGY_EP = "ep"
DECODE_COLLECTIVE_ALL_REDUCE = "all_reduce"
DECODE_COLLECTIVE_RS_AG_PAD64 = "rs_ag_pad64"


@dataclass(frozen=True)
class MultichipConfig:
    """Static policy for the complete four-device P300 mesh."""

    num_links: int = 1
    page_block_size: int = PAGE_BLOCK_SIZE
    expert_strategy: str = EXPERT_STRATEGY_EP
    decode_collective: str = DECODE_COLLECTIVE_ALL_REDUCE
    use_optimized_decode_layouts: bool = True
    use_sharded_decode_input_norm: bool = True
    use_sharded_decode_post_attention_norm: bool = True
    use_native_paged_sliding_attention: bool = False
    qkv_input_cores: int = 10
    qkv_in0_block_w: int = 9
    qkv_output_tiles_per_core: int = 2
    qkv_out_subblock_w: int = 2
    expert_gate_up_cores: tuple[int, int] = (3, 4)
    expert_down_cores: tuple[int, int] = (9, 10)
    expert_gate_up_in0_block_w: int = 45
    expert_down_in0_block_w: int = 45
    expert_gate_up_subblock_w: int = 1
    expert_down_subblock_w: int = 1
    prefill_expert_gate_up_cores: tuple[int, int] = (3, 4)
    prefill_expert_down_cores: tuple[int, int] = (5, 6)
    prefill_expert_gate_up_subblock_w: int = 1
    prefill_expert_down_subblock_w: int = 3
    ep_decode_gate_up_cores: tuple[int, int] = (9, 10)
    ep_decode_down_cores: tuple[int, int] = (9, 10)
    ep_prefill_gate_up_cores: tuple[int, int] = (9, 10)
    ep_prefill_down_cores: tuple[int, int] = (9, 10)
    ep_prefill_gate_up_subblock_w: int = 1
    ep_prefill_down_subblock_w: int = 1
    ep_prefill_post_sparse_bf16: bool = True
    active_prefill_chunk_size: int = 128


def _validate_ep_prefill_geometry(config: MultichipConfig, *, grid_x: int = 11, grid_y: int = 10):
    """Validate EP sparse-program output blocking for the 90-tile expert width."""

    per_core_values = []
    for name, cores, subblock in (
        ("gate_up", config.ep_prefill_gate_up_cores, config.ep_prefill_gate_up_subblock_w),
        ("down", config.ep_prefill_down_cores, config.ep_prefill_down_subblock_w),
    ):
        core_x, core_y = cores
        if not 1 <= core_x <= grid_x or not 1 <= core_y <= grid_y:
            raise ValueError(f"EP prefill {name} cores={cores} exceed device grid {grid_x}x{grid_y}")
        if subblock <= 0:
            raise ValueError(f"EP prefill {name} subblock must be positive")
        per_core_n = math.ceil((2880 // ttnn.TILE_SIZE) / (core_x * core_y))
        if per_core_n % subblock:
            raise ValueError(
                f"EP prefill {name} subblock={subblock} must divide per_core_N={per_core_n} for cores={cores}"
            )
        per_core_values.append(per_core_n)
    return tuple(per_core_values)


def _shard_experts(tensor, *, mesh_device, dtype):
    """Shard the canonical expert axis while leaving each expert whole."""

    return ttnn.from_torch(
        tensor.contiguous(),
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=1),
        layout=ttnn.TILE_LAYOUT,
        dtype=dtype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


class MultichipDecoder(_TP2MultichipDecoder):
    """TP4 attention and exact EP4/TP4 active-expert MoE on a replicated stream."""

    def __init__(
        self,
        *,
        multichip_config: MultichipConfig | None = None,
        optimization_config: OptimizationConfig | None = None,
        **kwargs,
    ):
        super().__init__(
            multichip_config=multichip_config or MultichipConfig(),
            optimization_config=optimization_config,
            **kwargs,
        )

    def _configure_multichip(self):
        if tuple(self.mesh_device.shape) != TARGET_MESH_SHAPE:
            raise ValueError(
                f"MultichipDecoder is specialized for mesh {TARGET_MESH_SHAPE}, got {tuple(self.mesh_device.shape)}"
            )
        if self.multichip_config.expert_strategy not in (EXPERT_STRATEGY_TP, EXPERT_STRATEGY_EP):
            raise ValueError(
                f"expert_strategy must be {EXPERT_STRATEGY_TP!r} or {EXPERT_STRATEGY_EP!r}, "
                f"got {self.multichip_config.expert_strategy!r}"
            )
        if self.multichip_config.decode_collective not in (
            DECODE_COLLECTIVE_ALL_REDUCE,
            DECODE_COLLECTIVE_RS_AG_PAD64,
        ):
            raise ValueError(f"unsupported decode_collective={self.multichip_config.decode_collective!r}")
        if self.num_heads % TP_DEGREE or self.num_kv_heads % TP_DEGREE:
            raise ValueError("query and KV head counts must divide evenly over TP=4")
        if self.intermediate_size % TP_DEGREE:
            raise ValueError("expert intermediate size must divide logically over TP=4")
        if self.num_experts % EP_DEGREE:
            raise ValueError("expert count must divide evenly over EP=4")
        if not 1 <= self.max_cache_len <= SUPPORTED_CONTEXT:
            raise ValueError(f"max_cache_len must be in [1, {SUPPORTED_CONTEXT}], got {self.max_cache_len}")
        if self.multichip_config.page_block_size % ttnn.TILE_SIZE:
            raise ValueError("page_block_size must be tile aligned")
        if (
            self.multichip_config.active_prefill_chunk_size <= 0
            or self.multichip_config.active_prefill_chunk_size % ttnn.TILE_SIZE
        ):
            raise ValueError("active_prefill_chunk_size must be a positive multiple of 32")
        _validate_ep_prefill_geometry(
            self.multichip_config,
            grid_x=self.mesh_device.compute_with_storage_grid_size().x,
            grid_y=self.mesh_device.compute_with_storage_grid_size().y,
        )

        self.local_num_heads = self.num_heads // TP_DEGREE
        self.local_num_kv_heads = self.num_kv_heads // TP_DEGREE
        self.local_intermediate_size = self.intermediate_size // TP_DEGREE
        self.local_num_experts = self.num_experts // EP_DEGREE
        if self.sliding_window is not None and not self.multichip_config.use_native_paged_sliding_attention:
            self.sliding_sdpa_program_config = (
                ttnn.SDPAProgramConfig(
                    compute_with_storage_grid_size=self.optimization_config.sdpa_grid,
                    exp_approx_mode=False,
                    q_chunk_size=ttnn.TILE_SIZE,
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
            shape=(ttnn.TILE_SIZE, self.head_dim),
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
        """Load full-mesh attention and one of the two active-expert layouts."""

        import torch
        import torch.nn.functional as F
        from transformers.models.gpt_oss.modeling_gpt_oss import GptOssRotaryEmbedding

        multichip_config = multichip_config or MultichipConfig()
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
        q_chunks = torch.chunk(q_weight, TP_DEGREE, dim=0)
        k_chunks = torch.chunk(k_weight, TP_DEGREE, dim=0)
        v_chunks = torch.chunk(v_weight, TP_DEGREE, dim=0)
        qb_chunks = torch.chunk(q_bias, TP_DEGREE, dim=0)
        kb_chunks = torch.chunk(k_bias, TP_DEGREE, dim=0)
        vb_chunks = torch.chunk(v_bias, TP_DEGREE, dim=0)
        qkv_weight = torch.cat(
            [torch.cat([q_chunks[rank].T, k_chunks[rank].T, v_chunks[rank].T], dim=-1) for rank in range(TP_DEGREE)],
            dim=-1,
        ).to(torch.bfloat16)
        qkv_bias = (
            torch.cat(
                [torch.cat([qb_chunks[rank], kb_chunks[rank], vb_chunks[rank]], dim=-1) for rank in range(TP_DEGREE)],
                dim=-1,
            )
            .reshape(1, 1, -1)
            .to(torch.bfloat16)
        )

        o_weight = _require_tensor(state_dict, layer_idx, "self_attn.o_proj.weight")
        o_bias = _require_tensor(state_dict, layer_idx, "self_attn.o_proj.bias").to(torch.bfloat16)
        rank_selective_o_bias = torch.cat(
            [o_bias] + [torch.zeros_like(o_bias) for _ in range(TP_DEGREE - 1)], dim=-1
        ).reshape(1, 1, -1)
        router_weight = _require_tensor(state_dict, layer_idx, "mlp.router.weight")
        router_bias = _require_tensor(state_dict, layer_idx, "mlp.router.bias")
        input_norm = _require_tensor(state_dict, layer_idx, "input_layernorm.weight")
        post_attention_norm = _require_tensor(state_dict, layer_idx, "post_attention_layernorm.weight")
        sinks = _require_tensor(state_dict, layer_idx, "self_attn.sinks").to(torch.bfloat16)

        scale = head_dim**-0.5
        prefill_sinks = (sinks.reshape(1, num_heads, 1, 1) / scale).to(torch.bfloat16)
        decode_sinks = torch.cat(
            [
                F.pad(chunk.reshape(num_heads // TP_DEGREE, 1), (0, ttnn.TILE_SIZE - 1)) / scale
                for chunk in torch.chunk(sinks, TP_DEGREE, dim=0)
            ],
            dim=0,
        ).to(torch.bfloat16)

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
            "o_weight": _shard_to_tp(o_weight.T.to(torch.bfloat16), mesh_device=mesh_device, dim=0),
            "o_bias": _shard_to_tp(rank_selective_o_bias, mesh_device=mesh_device, dim=2),
            "prefill_sinks": _shard_to_tp(prefill_sinks, mesh_device=mesh_device, dim=1),
            "decode_sinks": _shard_to_tp(decode_sinks, mesh_device=mesh_device, dim=0),
            "router_weight": _as_replicated_tensor(router_weight.T.to(torch.bfloat16), mesh_device=mesh_device),
            "router_bias": _as_replicated_tensor(
                router_bias.reshape(1, -1).float(), mesh_device=mesh_device, dtype=ttnn.float32
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
        decoder.tp_input_norm_weight = ttnn.to_memory_config(
            ttnn.reshape(ttnn.to_layout(decoder.weights["input_norm"], ttnn.TILE_LAYOUT), [hidden_size]),
            decoder.advisor_residual_config,
        )
        decoder.tp_post_attention_norm_weight = ttnn.to_memory_config(
            ttnn.reshape(ttnn.to_layout(decoder.weights["post_attention_norm"], ttnn.TILE_LAYOUT), [hidden_size]),
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
        if multichip_config.expert_strategy == EXPERT_STRATEGY_TP:
            program_config = OptimizedGPTOSSProgramConfig(
                decode_gate_up_cores=decoder.optimization_config.expert_gate_up_cores,
                decode_down_cores=decoder.optimization_config.expert_down_cores,
                prefill_gate_up_cores=multichip_config.prefill_expert_gate_up_cores,
                prefill_down_cores=multichip_config.prefill_expert_down_cores,
                decode_gate_up_in0_block_w=decoder.optimization_config.expert_gate_up_in0_block_w,
                decode_down_in0_block_w=decoder.optimization_config.expert_down_in0_block_w,
                prefill_gate_up_in0_block_w=decoder.optimization_config.expert_gate_up_in0_block_w,
                prefill_down_in0_block_w=decoder.optimization_config.expert_down_in0_block_w,
                decode_gate_up_subblock_w=multichip_config.expert_gate_up_subblock_w,
                decode_down_subblock_w=multichip_config.expert_down_subblock_w,
                prefill_gate_up_subblock_w=multichip_config.prefill_expert_gate_up_subblock_w,
                prefill_down_subblock_w=multichip_config.prefill_expert_down_subblock_w,
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
        else:
            gate_proj = expert_state["gate_up_proj"][..., ::2].reshape(1, num_experts, hidden_size, intermediate_size)
            up_proj = expert_state["gate_up_proj"][..., 1::2].reshape(1, num_experts, hidden_size, intermediate_size)
            gate_bias = expert_state["gate_up_proj_bias"][..., ::2].reshape(1, num_experts, intermediate_size)
            up_bias = expert_state["gate_up_proj_bias"][..., 1::2].reshape(1, num_experts, intermediate_size)
            down_proj = expert_state["down_proj"].reshape(1, num_experts, intermediate_size, hidden_size)
            down_bias = expert_state["down_proj_bias"].reshape(1, num_experts, hidden_size)
            ep_weights = ExpertWeights(
                gate_proj=_shard_experts(gate_proj, mesh_device=mesh_device, dtype=ttnn.bfloat8_b),
                up_proj=_shard_experts(up_proj, mesh_device=mesh_device, dtype=ttnn.bfloat8_b),
                down_proj=_shard_experts(down_proj, mesh_device=mesh_device, dtype=ttnn.bfloat8_b),
                gate_proj_bias=_shard_experts(gate_bias, mesh_device=mesh_device, dtype=ttnn.bfloat16),
                up_proj_bias=_shard_experts(up_bias, mesh_device=mesh_device, dtype=ttnn.bfloat16),
                down_proj_bias=_shard_experts(down_bias, mesh_device=mesh_device, dtype=ttnn.bfloat16),
                intermediate_size_per_device=intermediate_size,
            )
            ep_program_config = OptimizedGPTOSSProgramConfig(
                decode_gate_up_cores=multichip_config.ep_decode_gate_up_cores,
                decode_down_cores=multichip_config.ep_decode_down_cores,
                prefill_gate_up_cores=multichip_config.ep_prefill_gate_up_cores,
                prefill_down_cores=multichip_config.ep_prefill_down_cores,
                decode_gate_up_in0_block_w=45,
                decode_down_in0_block_w=45,
                prefill_gate_up_in0_block_w=45,
                prefill_down_in0_block_w=45,
                decode_gate_up_subblock_w=1,
                decode_down_subblock_w=1,
                prefill_gate_up_subblock_w=multichip_config.ep_prefill_gate_up_subblock_w,
                prefill_down_subblock_w=multichip_config.ep_prefill_down_subblock_w,
            )
            decoder.experts = SimpleNamespace(
                config=expert_config,
                weights=ep_weights,
                program_config=ep_program_config,
            )
        return decoder

    def _all_reduce(self, tensor, *, memory_config):
        """Use the selected decode collective while preserving prefill behavior."""

        if self.multichip_config.decode_collective == DECODE_COLLECTIVE_ALL_REDUCE or tensor.shape[-2] != 1:
            return super()._all_reduce(tensor, memory_config=memory_config)

        # TP4 reduce-scatter requires a tile-aligned local hidden width.  Pad
        # global H=2880 by 64 so every rank receives 736=23*32 values.  The
        # shared helper's own pad guard skips M=1 decode tensors, hence this
        # localized explicit pad.
        padded = ttnn.pad(tensor, [(0, 0), (0, 0), (0, 0), (0, 64)], value=0.0)
        tensor.deallocate(True)
        gathered = self.mesh_config.allreduce(
            padded,
            self.ccl_manager,
            memory_config=memory_config,
            axis=self.mesh_config.tp_axis,
        )
        output = ttnn.slice(
            gathered,
            [0, 0, 0, 0],
            [gathered.shape[0], gathered.shape[1], gathered.shape[2], self.hidden_size],
            memory_config=memory_config,
        )
        gathered.deallocate(True)
        return output

    def create_page_table(self, physical_block_ids: Sequence[int] | None = None):
        return super().create_page_table(physical_block_ids)

    def create_kv_cache(self):
        """Allocate two local KV heads per rank for every physical page."""

        return super().create_kv_cache()

    def _ep_active_expert_chunk(self, normalized, routing_weights, *, is_decode):
        """Execute exact active routes against eight whole experts per rank."""

        seq_len = normalized.shape[2]
        if not is_decode and seq_len % ttnn.TILE_SIZE:
            raise ValueError("active prefill expert chunks must be tile aligned")
        weights = self.experts.weights
        program_config = self.experts.program_config
        local_experts = self.local_num_experts
        memory_config = ttnn.L1_MEMORY_CONFIG if is_decode else ttnn.DRAM_MEMORY_CONFIG
        output_tile = ttnn.Tile([ttnn.TILE_SIZE, ttnn.TILE_SIZE])

        token_major = ttnn.reshape(
            normalized,
            [seq_len, 1, 1, self.hidden_size],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        routing_weights_rm = ttnn.to_layout(routing_weights, ttnn.ROW_MAJOR_LAYOUT)
        local_routing_rm = ttnn.mesh_partition(
            routing_weights_rm,
            dim=1,
            cluster_axis=1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        gate_up_sparsity = ttnn.to_layout(
            ttnn.reshape(local_routing_rm, [seq_len, 1, 1, local_experts]),
            ttnn.ROW_MAJOR_LAYOUT,
        )
        local_routing = ttnn.to_layout(local_routing_rm, ttnn.TILE_LAYOUT)
        gate_up_config = (
            program_config.get_decode_gate_up_config(1, self.intermediate_size, k=self.hidden_size)
            if is_decode
            else program_config.get_prefill_gate_up_config(1, self.intermediate_size, k=self.hidden_size)
        )
        gate = ttnn.sparse_matmul(
            token_major,
            weights.gate_proj,
            sparsity=gate_up_sparsity,
            nnz=None,
            memory_config=memory_config,
            output_tile=output_tile,
            program_config=gate_up_config,
            dtype=ttnn.bfloat8_b,
        )
        if not is_decode and self.multichip_config.ep_prefill_post_sparse_bf16:
            gate = ttnn.typecast(gate, ttnn.bfloat16)
        gate = ttnn.reshape(gate, [seq_len, local_experts, self.intermediate_size])
        gate = ttnn.add(
            gate,
            ttnn.reshape(weights.gate_proj_bias, [1, local_experts, self.intermediate_size]),
            output_tensor=gate,
        )
        up = ttnn.sparse_matmul(
            token_major,
            weights.up_proj,
            sparsity=gate_up_sparsity,
            nnz=None,
            memory_config=memory_config,
            output_tile=output_tile,
            program_config=gate_up_config,
            dtype=ttnn.bfloat8_b,
        )
        token_major.deallocate(True)
        if not is_decode and self.multichip_config.ep_prefill_post_sparse_bf16:
            up = ttnn.typecast(up, ttnn.bfloat16)
        up = ttnn.reshape(up, [seq_len, local_experts, self.intermediate_size])
        up = ttnn.add(
            up,
            ttnn.reshape(weights.up_proj_bias, [1, local_experts, self.intermediate_size]),
            output_tensor=up,
        )
        down_input = apply_swiglu(gate, up, self.experts.config)
        down_input = ttnn.reshape(down_input, [seq_len, local_experts, 1, self.intermediate_size])
        down_sparsity = ttnn.reshape(gate_up_sparsity, [1, 1, seq_len, local_experts])
        down_config = (
            program_config.get_decode_down_config(1, self.hidden_size, k=self.intermediate_size)
            if is_decode
            else program_config.get_prefill_down_config(1, self.hidden_size, k=self.intermediate_size)
        )
        down = ttnn.sparse_matmul(
            down_input,
            weights.down_proj,
            sparsity=down_sparsity,
            nnz=None,
            is_input_a_sparse=True,
            is_input_b_sparse=False,
            memory_config=memory_config,
            output_tile=output_tile,
            program_config=down_config,
            dtype=ttnn.bfloat8_b,
        )
        down_input.deallocate(True)
        if not is_decode and self.multichip_config.ep_prefill_post_sparse_bf16:
            down = ttnn.typecast(down, ttnn.bfloat16)
        down = ttnn.reshape(down, [seq_len, local_experts, self.hidden_size])
        down = ttnn.add(
            down,
            ttnn.reshape(weights.down_proj_bias, [1, local_experts, self.hidden_size]),
            output_tensor=down,
        )
        down = ttnn.mul(
            down,
            ttnn.reshape(local_routing, [seq_len, local_experts, 1]),
            output_tensor=down,
        )
        down = ttnn.sum(down, dim=1)
        if not is_decode and self.multichip_config.ep_prefill_post_sparse_bf16:
            down = ttnn.typecast(down, ttnn.bfloat8_b)
        return ttnn.reshape(down, [1, 1, seq_len, self.hidden_size])

    def _active_prefill_expert_chunk(self, normalized, routing_weights):
        if self.multichip_config.expert_strategy == EXPERT_STRATEGY_TP:
            return super()._active_prefill_expert_chunk(normalized, routing_weights)
        return self._ep_active_expert_chunk(normalized, routing_weights, is_decode=False)

    def _moe_forward(self, hidden_states, seq_len):
        if self.multichip_config.expert_strategy == EXPERT_STRATEGY_TP:
            return super()._moe_forward(hidden_states, seq_len)
        if seq_len != 1:
            normalized = ttnn.rms_norm(
                hidden_states,
                epsilon=self.eps,
                weight=self.weights["post_attention_norm"],
                compute_kernel_config=self.compute_kernel_config,
            )
            routing_weights = self._route(normalized, self.batch * seq_len)
            return self._active_prefill_sparse_moe(hidden_states, normalized, routing_weights, seq_len)
        normalized = self._decode_post_attention_norm(hidden_states)
        routing_weights = self._route(normalized, self.batch)
        partial = self._ep_active_expert_chunk(normalized, routing_weights, is_decode=True)
        expert_output = self._all_reduce(partial, memory_config=ttnn.L1_MEMORY_CONFIG)
        return ttnn.add(hidden_states, expert_output)


__all__ = [
    "EMITTED_BATCH",
    "EMITTED_CACHE_LENGTH",
    "EMITTED_PREFILL_SEQUENCE",
    "DECODE_COLLECTIVE_ALL_REDUCE",
    "DECODE_COLLECTIVE_RS_AG_PAD64",
    "EP_DEGREE",
    "EXPERT_STRATEGY_EP",
    "EXPERT_STRATEGY_TP",
    "MultichipConfig",
    "MultichipDecoder",
    "PAGE_BLOCK_SIZE",
    "SUPPORTED_CONTEXT",
    "TARGET_MESH_SHAPE",
    "TP_DEGREE",
    "_validate_qkv_geometry",
    "_validate_ep_prefill_geometry",
]
