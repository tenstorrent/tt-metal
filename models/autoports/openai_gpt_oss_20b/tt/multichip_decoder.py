# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Full-mesh GPT-OSS 20B decoder for the four-chip Blackhole P300 ring.

The class deliberately subclasses the current single-chip
``OptimizedDecoder``.  It keeps that decoder's precision, bounded-prefill,
router, and active-expert policies while restoring the compiler-emitted TP4
attention and EP4 whole-expert ownership.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssRotaryEmbedding

import ttnn
from models.autoports.openai_gpt_oss_20b.tt.functional_decoder import (
    EMITTED_BATCH,
    EMITTED_CACHE_LENGTH,
    EMITTED_PREFILL_SEQUENCE,
    FunctionalDecoder,
    _config_value,
    _expert_tensor,
    _state_tensor,
)
from models.autoports.openai_gpt_oss_20b.tt.optimized_decoder import (
    OptimizationConfig,
    OptimizedDecoder,
    OptimizedGPTOSSProgramConfig,
)
from models.demos.deepseek_v3.utils.config_helpers import (
    dram_sharded_weight_config,
    get_activation_sharding_core_counts_for_dram_matmul,
    get_dram_sharded_matmul_config,
)
from models.demos.gpt_oss.config import MeshConfig, ModeConfig
from models.demos.gpt_oss.tt.ccl import CCLManager
from models.demos.gpt_oss.tt.experts import ExpertConfig
from models.demos.gpt_oss.tt.experts.operations import apply_swiglu
from models.demos.gpt_oss.tt.experts.weights import ExpertWeights

TARGET_MESH_SHAPE = (1, 4)
TP_DEGREE = 4
EP_DEGREE = 4
PAGE_BLOCK_SIZE = 64
SUPPORTED_CONTEXT = 131_072
DECODE_COLLECTIVE_ALL_REDUCE = "all_reduce"
DECODE_COLLECTIVE_MINIMAL_ALL_REDUCE = "minimal_all_reduce"
DECODE_COLLECTIVE_RS_AG_PAD64 = "rs_ag_pad64"
EXPERT_STRATEGY_EP = "ep"
EXPERT_STRATEGY_TP = "tp"


@dataclass(frozen=True)
class MultichipConfig:
    """Static policy for the complete four-card P300 mesh."""

    num_links: int = 1
    page_block_size: int = PAGE_BLOCK_SIZE
    decode_collective: str = DECODE_COLLECTIVE_MINIMAL_ALL_REDUCE
    expert_strategy: str = EXPERT_STRATEGY_EP
    use_fused_o_projection_rs: bool = False
    use_fused_o_projection_ag: bool = False
    fused_o_ag_pad_hidden: bool = False
    fused_ag_matmul_payload_dtype: str = "bfloat16"
    qkv_input_cores: int = 10
    qkv_in0_block_w: int = 9
    qkv_output_tiles_per_core: int = 2
    qkv_out_subblock_w: int = 2
    prefill_expert_cores: tuple[int, int] = (9, 10)
    decode_expert_cores: tuple[int, int] = (5, 9)
    decode_gate_up_cores: tuple[int, int] | None = (9, 10)
    decode_down_cores: tuple[int, int] | None = (9, 10)
    expert_in0_block_w: int = 45
    decode_gate_up_in0_block_w: int | None = 45
    decode_down_in0_block_w: int | None = 90
    prefill_expert_subblock_w: int = 1
    decode_expert_subblock_w: int = 2
    decode_gate_up_subblock_w: int | None = 1
    decode_down_subblock_w: int | None = 1
    prefill_expert_output_l1: bool = True
    prefill_expert_output_l1_max_seq: int = 32
    decode_expert_output_l1: bool = True
    use_packed_sparse_gate_up: bool = False
    use_dram_sharded_decode_attention: bool = False
    dram_attention_core_limit: int = 90
    active_prefill_chunk_size: int = 128
    attention_weight_dtype: str = "bfloat16"
    decode_attention_weight_dtype: str = "bfloat8_b"
    attention_math_fidelity: str = "hifi2"
    long_decode_attention_math_fidelity: str = "selected"
    expert_weight_dtype: str = "bfloat8_b"
    expert_gate_up_weight_dtype: str = "selected"
    expert_down_weight_dtype: str = "selected"
    expert_activation_dtype: str = "bfloat16"
    expert_math_fidelity: str = "lofi"
    kv_cache_dtype: str = "bfloat8_b"
    decode_ccl_dtype: str = "bfloat16"
    use_sharded_decode_norms: bool = True


def _mesh_shape(mesh_device) -> tuple[int, int]:
    return tuple(int(value) for value in mesh_device.shape)


def _replicate_tensor(
    tensor: torch.Tensor,
    *,
    mesh_device,
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
):
    return ttnn.from_torch(
        tensor.contiguous(),
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=dtype,
        layout=layout,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _shard_tensor(
    tensor: torch.Tensor,
    *,
    mesh_device,
    dim: int,
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
):
    return ttnn.from_torch(
        tensor.contiguous(),
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=dim),
        dtype=dtype,
        layout=layout,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _validate_qkv_geometry(
    config: MultichipConfig,
    *,
    k_tiles: int,
    n_tiles: int,
    grid_x: int,
    grid_y: int,
) -> tuple[int, int, int]:
    values = {
        "qkv_input_cores": config.qkv_input_cores,
        "qkv_in0_block_w": config.qkv_in0_block_w,
        "qkv_output_tiles_per_core": config.qkv_output_tiles_per_core,
        "qkv_out_subblock_w": config.qkv_out_subblock_w,
    }
    if any(value <= 0 for value in values.values()):
        raise ValueError(f"QKV geometry values must be positive, got {values}")
    # Width sharding may pad the physical K extent.  The public hidden width
    # remains logical H; TTNN owns the per-core tail padding during the
    # interleaved-to-sharded conversion.
    input_shard_tiles = math.ceil(k_tiles / config.qkv_input_cores)
    if k_tiles % config.qkv_in0_block_w or input_shard_tiles % config.qkv_in0_block_w:
        raise ValueError(
            f"qkv_in0_block_w={config.qkv_in0_block_w} must divide K tiles={k_tiles} "
            f"and the per-core shard={input_shard_tiles}"
        )
    if config.qkv_output_tiles_per_core % config.qkv_out_subblock_w:
        raise ValueError("qkv_out_subblock_w must divide qkv_output_tiles_per_core")
    output_cores = math.ceil(n_tiles / config.qkv_output_tiles_per_core)
    program_rows = math.ceil(max(config.qkv_input_cores, output_cores) / grid_x)
    if program_rows > grid_y:
        raise ValueError(f"QKV geometry needs {program_rows} rows on a {grid_x}x{grid_y} grid")
    return input_shard_tiles, output_cores, program_rows


class MultichipDecoder(OptimizedDecoder):
    """TP4 attention and EP4 active experts on a replicated BF16 stream."""

    def __init__(
        self,
        *,
        multichip_config: MultichipConfig | None = None,
        optimization_config: OptimizationConfig | None = None,
        attention_window: int | None,
        **kwargs,
    ):
        self.multichip_config = multichip_config or MultichipConfig()
        requested_optimization = optimization_config or OptimizationConfig()
        long_decode_fidelity = self.multichip_config.long_decode_attention_math_fidelity
        if long_decode_fidelity == "selected":
            long_decode_fidelity = (
                "hifi4" if attention_window is None else requested_optimization.long_decode_attention_math_fidelity
            )
        self.optimization_config = requested_optimization.with_changes(
            use_shard_advisor_attention_layouts=False,
            use_shard_advisor_router_layouts=False,
            use_shard_advisor_dense_moe_layouts=False,
            use_dram_sharded_attention=False,
            use_sparse_experts=True,
            use_dense_long_prefill=False,
            expert_input_l1=False,
            expert_weight_dtype=self.multichip_config.expert_weight_dtype,
            kv_cache_dtype=self.multichip_config.kv_cache_dtype,
            attention_math_fidelity=self.multichip_config.attention_math_fidelity,
            # TP4 QKV/O accumulation at long full-attention positions can
            # perturb a near-tied router decision even when attention PCC is
            # above 0.9997. HiFi4 restores the exact optimized top-4 route.
            long_decode_attention_math_fidelity=long_decode_fidelity,
        )

        # OptimizedDecoder's normal constructor creates a single-device dense
        # expert graph before replacing it with sparse weights.  That would
        # replicate tensors which EP4 must never allocate.  Initialize its
        # FunctionalDecoder base directly, then materialize the exact optimized
        # configs and methods used by this runtime.
        FunctionalDecoder.__init__(self, **kwargs)
        self.attention_window = attention_window
        self.experts = None
        self.prefill_rotary_views = {}
        self.decode_position_views = {}
        self.moe_policy = "sparse"

        OptimizedDecoder._configure_shard_advisor_candidate(self)
        OptimizedDecoder._configure_dram_attention_candidate(self)
        OptimizedDecoder._configure_attention_program_candidates(self)
        self._configure_multichip()

    def _configure_multichip(self) -> None:
        if _mesh_shape(self.mesh_device) != TARGET_MESH_SHAPE:
            raise ValueError(
                f"MultichipDecoder is specialized for mesh {TARGET_MESH_SHAPE}, got {_mesh_shape(self.mesh_device)}"
            )
        if self.batch != EMITTED_BATCH:
            raise ValueError(f"MultichipDecoder supports batch {EMITTED_BATCH}, got {self.batch}")
        if self.num_heads % TP_DEGREE or self.num_kv_heads % TP_DEGREE:
            raise ValueError("query and KV head counts must divide evenly over TP=4")
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
        if (
            self.multichip_config.prefill_expert_output_l1_max_seq <= 0
            or self.multichip_config.prefill_expert_output_l1_max_seq % ttnn.TILE_SIZE
        ):
            raise ValueError("prefill_expert_output_l1_max_seq must be a positive multiple of 32")
        for name in ("attention_weight_dtype", "decode_attention_weight_dtype"):
            if getattr(self.multichip_config, name) not in ("bfloat4_b", "bfloat8_b", "bfloat16"):
                raise ValueError(f"{name} must be bfloat4_b, bfloat8_b, or bfloat16")
        if self.multichip_config.attention_math_fidelity not in ("auto", "lofi", "hifi2", "hifi4"):
            raise ValueError("attention_math_fidelity must be auto, lofi, hifi2, or hifi4")
        if self.multichip_config.long_decode_attention_math_fidelity not in (
            "selected",
            "lofi",
            "hifi2",
            "hifi4",
        ):
            raise ValueError("long_decode_attention_math_fidelity must be selected, lofi, hifi2, or hifi4")
        if self.multichip_config.expert_weight_dtype not in ("bfloat4_b", "bfloat8_b", "bfloat16"):
            raise ValueError("expert_weight_dtype must be bfloat4_b, bfloat8_b, or bfloat16")
        for name in ("expert_gate_up_weight_dtype", "expert_down_weight_dtype"):
            if getattr(self.multichip_config, name) not in ("selected", "bfloat4_b", "bfloat8_b", "bfloat16"):
                raise ValueError(f"{name} must be selected, bfloat4_b, bfloat8_b, or bfloat16")
        if self.multichip_config.expert_activation_dtype not in ("bfloat8_b", "bfloat16"):
            raise ValueError("expert_activation_dtype must be bfloat8_b or bfloat16")
        if self.multichip_config.kv_cache_dtype not in ("bfloat8_b", "bfloat16"):
            raise ValueError("kv_cache_dtype must be bfloat8_b or bfloat16")
        if self.multichip_config.decode_ccl_dtype not in ("bfloat8_b", "bfloat16"):
            raise ValueError("decode_ccl_dtype must be bfloat8_b or bfloat16")
        if self.multichip_config.decode_collective not in (
            DECODE_COLLECTIVE_ALL_REDUCE,
            DECODE_COLLECTIVE_MINIMAL_ALL_REDUCE,
            DECODE_COLLECTIVE_RS_AG_PAD64,
        ):
            raise ValueError(f"unsupported decode_collective={self.multichip_config.decode_collective!r}")
        if self.multichip_config.expert_strategy not in (
            EXPERT_STRATEGY_EP,
            EXPERT_STRATEGY_TP,
        ):
            raise ValueError(f"unsupported expert_strategy={self.multichip_config.expert_strategy!r}")
        if self.multichip_config.fused_ag_matmul_payload_dtype not in (
            "bfloat8_b",
            "bfloat16",
        ):
            raise ValueError("fused_ag_matmul_payload_dtype must be bfloat8_b or bfloat16")

        self.local_num_heads = self.num_heads // TP_DEGREE
        self.local_num_kv_heads = self.num_kv_heads // TP_DEGREE
        self.local_num_experts = self.num_experts // EP_DEGREE
        self.local_intermediate_size = self.intermediate_size // TP_DEGREE
        self.num_cache_blocks = math.ceil(self.max_cache_len / self.multichip_config.page_block_size)

        self.mesh_config = MeshConfig(
            TARGET_MESH_SHAPE,
            decode=ModeConfig(tp=TP_DEGREE, ep=1, sp=1),
            prefill=ModeConfig(tp=TP_DEGREE, ep=1, sp=1),
            tp_axis=1,
        )
        self.ccl_manager = (
            CCLManager(
                self.mesh_device,
                num_links=self.multichip_config.num_links,
                topology=ttnn.Topology.Ring,
            )
            if (
                self.multichip_config.decode_collective == DECODE_COLLECTIVE_RS_AG_PAD64
                or self.multichip_config.use_fused_o_projection_rs
                or self.multichip_config.use_fused_o_projection_ag
                or self.multichip_config.expert_strategy == EXPERT_STRATEGY_TP
            )
            else None
        )
        self._fused_o_ag_persistent_buffers = {}
        device_grid = self.mesh_device.compute_with_storage_grid_size()
        one_batch_grid = ttnn.num_cores_to_corerangeset(
            self.batch,
            ttnn.CoreCoord(device_grid.x, device_grid.y),
            row_wise=True,
        )
        self.decode_heads_mem_config = ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, self.head_dim),
            core_grid=one_batch_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        self.decode_rope_memory_config = ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, self.head_dim),
            core_grid=one_batch_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        def width_config(cores: int, shard_width: int):
            core_grid = ttnn.num_cores_to_corerangeset(
                cores,
                ttnn.CoreCoord(device_grid.x, device_grid.y),
                row_wise=True,
            )
            return ttnn.create_sharded_memory_config(
                shape=(ttnn.TILE_SIZE, shard_width),
                core_grid=core_grid,
                strategy=ttnn.ShardStrategy.WIDTH,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )

        local_qkv_width = (self.local_num_heads + 2 * self.local_num_kv_heads) * self.head_dim
        qkv_input_shard_tiles, qkv_output_cores, qkv_program_rows = _validate_qkv_geometry(
            self.multichip_config,
            k_tiles=self.hidden_size // ttnn.TILE_SIZE,
            n_tiles=local_qkv_width // ttnn.TILE_SIZE,
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
        self.minimal_ar_buffer_config = width_config(
            90,
            TP_DEGREE * (self.hidden_size // 90),
        )
        self._minimal_ar_index = 0
        self._minimal_ar_buffers = []
        self._minimal_ar_semaphores = []
        if self.multichip_config.decode_collective == DECODE_COLLECTIVE_MINIMAL_ALL_REDUCE:
            ccl_dtype = {
                "bfloat8_b": ttnn.bfloat8_b,
                "bfloat16": ttnn.bfloat16,
            }[self.multichip_config.decode_ccl_dtype]
            all_worker_cores = ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(
                        ttnn.CoreCoord(0, 0),
                        ttnn.CoreCoord(device_grid.x - 1, device_grid.y - 1),
                    )
                ]
            )
            for _ in range(2):
                self._minimal_ar_semaphores.append(ttnn.create_global_semaphore(self.mesh_device, all_worker_cores, 0))
                self._minimal_ar_buffers.append(
                    ttnn.from_torch(
                        torch.zeros(
                            (
                                TARGET_MESH_SHAPE[0],
                                TARGET_MESH_SHAPE[1],
                                ttnn.TILE_SIZE,
                                TP_DEGREE * self.hidden_size,
                            )
                        ),
                        device=self.mesh_device,
                        layout=ttnn.TILE_LAYOUT,
                        dtype=ccl_dtype,
                        memory_config=self.minimal_ar_buffer_config,
                        mesh_mapper=ttnn.ShardTensor2dMesh(
                            self.mesh_device,
                            dims=(0, 1),
                            mesh_shape=TARGET_MESH_SHAPE,
                        ),
                    )
                )
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
        max_dram_cores = min(
            self.multichip_config.dram_attention_core_limit,
            device_grid.x * device_grid.y,
        )

        def dram_policy(k: int, n: int):
            input_cores = max(get_activation_sharding_core_counts_for_dram_matmul(k, max_dram_cores))
            output_cores = max(get_activation_sharding_core_counts_for_dram_matmul(n, max_dram_cores))
            return (
                get_dram_sharded_matmul_config(ttnn.TILE_SIZE, k, n, input_cores, output_cores),
                width_config(input_cores, k // input_cores),
                width_config(output_cores, n // output_cores),
            )

        self.dram_tp_qkv_program_config, self.dram_tp_qkv_input_config, self.dram_tp_qkv_output_config = dram_policy(
            self.hidden_size,
            local_qkv_width,
        )
        self.dram_tp_o_program_config, self.dram_tp_o_input_config, self.dram_tp_o_output_config = dram_policy(
            self.local_num_heads * self.head_dim,
            self.hidden_size,
        )

        self.advisor_norm_weights = {}
        for name in ("input_norm", "post_attention_norm"):
            tiled = ttnn.to_layout(getattr(self, name), ttnn.TILE_LAYOUT)
            self.advisor_norm_weights[name] = ttnn.to_memory_config(
                ttnn.reshape(tiled, [self.hidden_size]),
                self.advisor_residual_config,
            )

        self.decode_cos_matrix = ttnn.to_layout(
            ttnn.reshape(self.rotary_cos, [self.max_cache_len, self.head_dim]),
            ttnn.ROW_MAJOR_LAYOUT,
        )
        self.decode_sin_matrix = ttnn.to_layout(
            ttnn.reshape(self.rotary_sin, [self.max_cache_len, self.head_dim]),
            ttnn.ROW_MAJOR_LAYOUT,
        )

        def expert_compute_kernel(fidelity: str):
            math_fidelity = {
                "lofi": ttnn.MathFidelity.LoFi,
                "hifi2": ttnn.MathFidelity.HiFi2,
                "hifi4": ttnn.MathFidelity.HiFi4,
            }.get(fidelity)
            if math_fidelity is None:
                raise ValueError("expert math fidelity must be lofi, hifi2, or hifi4")
            return ttnn.init_device_compute_kernel_config(
                self.mesh_device.arch(),
                math_fidelity=math_fidelity,
                math_approx_mode=False,
                fp32_dest_acc_en=fidelity == "hifi4",
                packer_l1_acc=True,
            )

        self.ep_decode_compute_kernel_config = expert_compute_kernel(self.multichip_config.expert_math_fidelity)
        self.ep_sliding_prefill_compute_kernel_config = expert_compute_kernel(
            self.optimization_config.prefill_expert_math_fidelity
        )
        self.ep_full_prefill_compute_kernel_config = expert_compute_kernel(
            self.optimization_config.full_prefill_expert_math_fidelity
        )

    @classmethod
    def from_state_dict(
        cls,
        state_dict: Mapping[str, torch.Tensor],
        *,
        hf_config,
        layer_idx: int,
        mesh_device,
        batch: int = EMITTED_BATCH,
        max_cache_len: int = EMITTED_CACHE_LENGTH,
        multichip_config: MultichipConfig | None = None,
        optimization_config: OptimizationConfig | None = None,
        **_kwargs,
    ) -> "MultichipDecoder":
        multichip_config = multichip_config or MultichipConfig()
        if _mesh_shape(mesh_device) != TARGET_MESH_SHAPE:
            raise ValueError(f"MultichipDecoder requires mesh {TARGET_MESH_SHAPE}")
        if batch != EMITTED_BATCH:
            raise ValueError(f"MultichipDecoder supports batch {EMITTED_BATCH}, got {batch}")
        if not 1 <= max_cache_len <= SUPPORTED_CONTEXT:
            raise ValueError(f"max_cache_len must be in [1, {SUPPORTED_CONTEXT}], got {max_cache_len}")

        hidden_size = int(_config_value(hf_config, "hidden_size"))
        num_heads = int(_config_value(hf_config, "num_attention_heads"))
        num_kv_heads = int(_config_value(hf_config, "num_key_value_heads"))
        head_dim = int(_config_value(hf_config, "head_dim") or hidden_size // num_heads)
        intermediate_size = int(_config_value(hf_config, "intermediate_size"))
        num_experts = int(_config_value(hf_config, "num_local_experts"))
        experts_per_token = int(_config_value(hf_config, "num_experts_per_tok"))
        rms_norm_eps = float(_config_value(hf_config, "rms_norm_eps"))
        sliding_window = int(_config_value(hf_config, "sliding_window"))
        swiglu_limit = float(_config_value(hf_config, "swiglu_limit"))
        if (hidden_size, num_heads, num_kv_heads, head_dim, intermediate_size, num_experts) != (
            2880,
            64,
            8,
            64,
            2880,
            32,
        ):
            raise ValueError("MultichipDecoder is specialized for the GPT-OSS 20B decoder geometry")

        layer_types = _config_value(hf_config, "layer_types")
        if layer_types is None or not 0 <= layer_idx < len(layer_types):
            raise ValueError(f"HF config does not define layer_types[{layer_idx}]")
        layer_type = layer_types[layer_idx]
        if layer_type == "sliding_attention":
            attention_window = sliding_window
        elif layer_type == "full_attention":
            attention_window = None
        else:
            raise ValueError(f"unsupported layer_types[{layer_idx}]={layer_type!r}")

        attention_dtype = {
            "bfloat4_b": ttnn.bfloat4_b,
            "bfloat8_b": ttnn.bfloat8_b,
            "bfloat16": ttnn.bfloat16,
        }[multichip_config.attention_weight_dtype]
        q = _state_tensor(state_dict, layer_idx, "self_attn.q_proj.weight").to(torch.bfloat16)
        k = _state_tensor(state_dict, layer_idx, "self_attn.k_proj.weight").to(torch.bfloat16)
        v = _state_tensor(state_dict, layer_idx, "self_attn.v_proj.weight").to(torch.bfloat16)
        q_bias = _state_tensor(state_dict, layer_idx, "self_attn.q_proj.bias").to(torch.bfloat16)
        k_bias = _state_tensor(state_dict, layer_idx, "self_attn.k_proj.bias").to(torch.bfloat16)
        v_bias = _state_tensor(state_dict, layer_idx, "self_attn.v_proj.bias").to(torch.bfloat16)
        q_chunks = torch.chunk(q, TP_DEGREE, dim=0)
        k_chunks = torch.chunk(k, TP_DEGREE, dim=0)
        v_chunks = torch.chunk(v, TP_DEGREE, dim=0)
        qb_chunks = torch.chunk(q_bias, TP_DEGREE, dim=0)
        kb_chunks = torch.chunk(k_bias, TP_DEGREE, dim=0)
        vb_chunks = torch.chunk(v_bias, TP_DEGREE, dim=0)
        qkv = torch.cat(
            [torch.cat((q_chunks[rank].T, k_chunks[rank].T, v_chunks[rank].T), dim=-1) for rank in range(TP_DEGREE)],
            dim=-1,
        )
        packed_bias = torch.cat(
            [torch.cat((qb_chunks[rank], kb_chunks[rank], vb_chunks[rank]), dim=-1) for rank in range(TP_DEGREE)],
            dim=-1,
        ).reshape(1, 1, -1)

        output = _state_tensor(state_dict, layer_idx, "self_attn.o_proj.weight").to(torch.bfloat16)
        output_bias = _state_tensor(state_dict, layer_idx, "self_attn.o_proj.bias").to(torch.bfloat16)
        rank_selective_output_bias = torch.cat(
            [output_bias] + [torch.zeros_like(output_bias) for _ in range(TP_DEGREE - 1)],
            dim=-1,
        ).reshape(1, 1, -1)
        sinks = _state_tensor(state_dict, layer_idx, "self_attn.sinks").to(torch.bfloat16)
        scale = head_dim**-0.5
        manual_sinks = sinks.reshape(1, num_heads, 1, 1).to(torch.bfloat16)
        prefill_sinks = (manual_sinks / scale).to(torch.bfloat16)
        manual_prefill_sinks = None
        manual_prefill_mask = None
        if attention_window is None:
            manual_prefill_sinks = manual_sinks.expand(
                batch,
                num_heads,
                EMITTED_CACHE_LENGTH,
                1,
            ).contiguous()
            query_positions = torch.arange(EMITTED_CACHE_LENGTH).view(EMITTED_CACHE_LENGTH, 1)
            key_positions = torch.arange(EMITTED_CACHE_LENGTH).view(1, EMITTED_CACHE_LENGTH)
            causal_mask = torch.zeros(
                (1, 1, EMITTED_CACHE_LENGTH, EMITTED_CACHE_LENGTH),
                dtype=torch.bfloat16,
            )
            causal_mask.masked_fill_(
                (key_positions > query_positions).view(
                    1,
                    1,
                    EMITTED_CACHE_LENGTH,
                    EMITTED_CACHE_LENGTH,
                ),
                torch.finfo(torch.bfloat16).min,
            )
            manual_prefill_mask = _replicate_tensor(
                causal_mask,
                mesh_device=mesh_device,
            )
        decode_sinks = torch.cat(
            [
                F.pad(chunk.reshape(num_heads // TP_DEGREE, 1), (0, ttnn.TILE_SIZE - 1)) / scale
                for chunk in torch.chunk(sinks, TP_DEGREE, dim=0)
            ],
            dim=0,
        ).to(torch.bfloat16)

        rotary = GptOssRotaryEmbedding(hf_config)
        positions = torch.arange(max_cache_len, dtype=torch.long).unsqueeze(0)
        rope_probe = torch.zeros((1, 1, max_cache_len, head_dim), dtype=torch.bfloat16)
        cos, sin = rotary(rope_probe, positions)
        cos = torch.cat((cos, cos), dim=-1).unsqueeze(1).to(torch.bfloat16)
        sin = torch.cat((sin, sin), dim=-1).unsqueeze(1).to(torch.bfloat16)

        decoder = cls(
            mesh_device=mesh_device,
            layer_idx=layer_idx,
            batch=batch,
            max_cache_len=max_cache_len,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            experts_per_token=experts_per_token,
            rms_norm_eps=rms_norm_eps,
            sliding_window=sliding_window,
            swiglu_limit=swiglu_limit,
            input_norm=_replicate_tensor(
                _state_tensor(state_dict, layer_idx, "input_layernorm.weight").to(torch.bfloat16),
                mesh_device=mesh_device,
            ),
            post_attention_norm=_replicate_tensor(
                _state_tensor(state_dict, layer_idx, "post_attention_layernorm.weight").to(torch.bfloat16),
                mesh_device=mesh_device,
            ),
            qkv_weight=_shard_tensor(qkv, mesh_device=mesh_device, dim=1, dtype=attention_dtype),
            qkv_bias=_shard_tensor(packed_bias, mesh_device=mesh_device, dim=2),
            output_weight=_shard_tensor(output.T, mesh_device=mesh_device, dim=0, dtype=attention_dtype),
            output_bias=_shard_tensor(rank_selective_output_bias, mesh_device=mesh_device, dim=2),
            attention_sinks=_shard_tensor(prefill_sinks, mesh_device=mesh_device, dim=1),
            decode_attention_sinks=_shard_tensor(decode_sinks, mesh_device=mesh_device, dim=0),
            router_weight=_replicate_tensor(
                _state_tensor(state_dict, layer_idx, "mlp.router.weight").T.to(torch.bfloat16),
                mesh_device=mesh_device,
            ),
            router_bias=_replicate_tensor(
                _state_tensor(state_dict, layer_idx, "mlp.router.bias").to(torch.float32),
                mesh_device=mesh_device,
                dtype=ttnn.float32,
            ),
            gate_up_weight=None,
            gate_up_bias=None,
            down_weight=None,
            down_bias=None,
            rotary_cos=_replicate_tensor(cos, mesh_device=mesh_device),
            rotary_sin=_replicate_tensor(sin, mesh_device=mesh_device),
            attention_mask=manual_prefill_mask,
            position_indices=_replicate_tensor(
                torch.arange(max_cache_len, dtype=torch.int32),
                mesh_device=mesh_device,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            ),
            attention_window=attention_window,
            multichip_config=multichip_config,
            optimization_config=optimization_config,
        )
        decoder.prefill_sdpa_sink = decoder.attention_sinks
        decoder.manual_attention_sinks = _shard_tensor(manual_sinks, mesh_device=mesh_device, dim=1)
        decoder.manual_prefill_attention_sinks = (
            _shard_tensor(manual_prefill_sinks, mesh_device=mesh_device, dim=1)
            if manual_prefill_sinks is not None
            else None
        )
        decode_attention_dtype = {
            "bfloat4_b": ttnn.bfloat4_b,
            "bfloat8_b": ttnn.bfloat8_b,
            "bfloat16": ttnn.bfloat16,
        }[multichip_config.decode_attention_weight_dtype]
        if multichip_config.decode_attention_weight_dtype == multichip_config.attention_weight_dtype:
            decoder.decode_qkv_weight = decoder.qkv_weight
            decoder.decode_output_weight = decoder.output_weight
        else:
            decoder.decode_qkv_weight = _shard_tensor(
                qkv,
                mesh_device=mesh_device,
                dim=1,
                dtype=decode_attention_dtype,
            )
            decoder.decode_output_weight = _shard_tensor(
                output.T,
                mesh_device=mesh_device,
                dim=0,
                dtype=decode_attention_dtype,
            )
        decoder.fused_o_ag_weight = None
        decoder.fused_o_ag_bias = None
        decoder.fused_o_rs_weight = None
        decoder.fused_o_rs_bias = None
        decoder.dram_decode_qkv_weight = None
        decoder.dram_decode_o_weight = None
        if multichip_config.use_dram_sharded_decode_attention:
            dram_grid = mesh_device.dram_grid_size()
            decoder.dram_decode_qkv_weight = ttnn.to_memory_config(
                decoder.decode_qkv_weight,
                dram_sharded_weight_config(hidden_size, qkv.shape[-1] // TP_DEGREE, dram_grid),
            )
            decoder.dram_decode_o_weight = ttnn.to_memory_config(
                decoder.decode_output_weight,
                dram_sharded_weight_config(num_heads * head_dim // TP_DEGREE, hidden_size, dram_grid),
            )
        if multichip_config.use_fused_o_projection_rs:
            decoder.fused_o_rs_weight = _shard_tensor(
                F.pad(output.T, (0, 64)),
                mesh_device=mesh_device,
                dim=0,
                dtype=decode_attention_dtype,
            )
            padded_output_bias = F.pad(output_bias, (0, 64))
            decoder.fused_o_rs_bias = _shard_tensor(
                torch.cat(
                    [padded_output_bias] + [torch.zeros_like(padded_output_bias) for _ in range(TP_DEGREE - 1)],
                    dim=-1,
                ).reshape(1, 1, -1),
                mesh_device=mesh_device,
                dim=2,
            )
        if multichip_config.use_fused_o_projection_ag:
            o_ag_pad = 64 if multichip_config.fused_o_ag_pad_hidden else 0
            o_ag_dtype = {
                "bfloat8_b": ttnn.bfloat8_b,
                "bfloat16": ttnn.bfloat16,
            }[multichip_config.fused_ag_matmul_payload_dtype]
            decoder.fused_o_ag_weight = _shard_tensor(
                F.pad(output.T, (0, o_ag_pad)),
                mesh_device=mesh_device,
                dim=1,
                dtype=o_ag_dtype,
            )
            decoder.fused_o_ag_bias = _shard_tensor(
                F.pad(output_bias, (0, o_ag_pad)).reshape(1, 1, -1),
                mesh_device=mesh_device,
                dim=2,
            )

        gate_up = _expert_tensor(state_dict, layer_idx, "gate_up_proj")
        gate_up_bias = _state_tensor(state_dict, layer_idx, "mlp.experts.gate_up_proj_bias").to(torch.bfloat16)
        down = _expert_tensor(state_dict, layer_idx, "down_proj")
        down_bias = _state_tensor(state_dict, layer_idx, "mlp.experts.down_proj_bias").to(torch.bfloat16)
        gate = gate_up[..., ::2].reshape(1, num_experts, hidden_size, intermediate_size)
        up = gate_up[..., 1::2].reshape(1, num_experts, hidden_size, intermediate_size)
        gate_bias = gate_up_bias[..., ::2].reshape(1, num_experts, intermediate_size)
        up_bias = gate_up_bias[..., 1::2].reshape(1, num_experts, intermediate_size)
        down = down.reshape(1, num_experts, intermediate_size, hidden_size)
        down_bias = down_bias.reshape(1, num_experts, hidden_size)
        expert_dtype_map = {
            "bfloat4_b": ttnn.bfloat4_b,
            "bfloat8_b": ttnn.bfloat8_b,
            "bfloat16": ttnn.bfloat16,
        }
        gate_up_dtype_name = multichip_config.expert_gate_up_weight_dtype
        if gate_up_dtype_name == "selected":
            gate_up_dtype_name = multichip_config.expert_weight_dtype
        down_dtype_name = multichip_config.expert_down_weight_dtype
        if down_dtype_name == "selected":
            down_dtype_name = multichip_config.expert_weight_dtype
        gate_up_dtype = expert_dtype_map[gate_up_dtype_name]
        down_dtype = expert_dtype_map[down_dtype_name]
        if multichip_config.expert_strategy == EXPERT_STRATEGY_TP:
            rank_selective_down_bias = torch.cat(
                [down_bias] + [torch.zeros_like(down_bias) for _ in range(TP_DEGREE - 1)],
                dim=2,
            )
            expert_weights = ExpertWeights(
                gate_proj=_shard_tensor(gate, mesh_device=mesh_device, dim=3, dtype=gate_up_dtype),
                up_proj=_shard_tensor(up, mesh_device=mesh_device, dim=3, dtype=gate_up_dtype),
                down_proj=_shard_tensor(down, mesh_device=mesh_device, dim=2, dtype=down_dtype),
                gate_proj_bias=_shard_tensor(gate_bias, mesh_device=mesh_device, dim=2),
                up_proj_bias=_shard_tensor(up_bias, mesh_device=mesh_device, dim=2),
                down_proj_bias=_shard_tensor(rank_selective_down_bias, mesh_device=mesh_device, dim=2),
                intermediate_size_per_device=intermediate_size // TP_DEGREE,
            )
            expert_program = OptimizedGPTOSSProgramConfig(
                decode_gate_up_cores=(3, 4),
                decode_down_cores=(9, 10),
                prefill_gate_up_cores=(3, 4),
                prefill_down_cores=(5, 6),
                decode_gate_up_in0_block_w=45,
                decode_down_in0_block_w=45,
                prefill_gate_up_in0_block_w=45,
                prefill_down_in0_block_w=45,
                decode_gate_up_subblock_w=1,
                decode_down_subblock_w=1,
                prefill_gate_up_subblock_w=1,
                prefill_down_subblock_w=3,
                sequence_chunk_size=multichip_config.active_prefill_chunk_size,
            )
        else:
            decode_gate_up_cores = multichip_config.decode_gate_up_cores or multichip_config.decode_expert_cores
            decode_down_cores = multichip_config.decode_down_cores or multichip_config.decode_expert_cores
            decode_gate_up_in0_block_w = (
                multichip_config.decode_gate_up_in0_block_w or multichip_config.expert_in0_block_w
            )
            decode_down_in0_block_w = multichip_config.decode_down_in0_block_w or multichip_config.expert_in0_block_w
            decode_gate_up_subblock_w = (
                multichip_config.decode_gate_up_subblock_w or multichip_config.decode_expert_subblock_w
            )
            decode_down_subblock_w = (
                multichip_config.decode_down_subblock_w or multichip_config.decode_expert_subblock_w
            )
            expert_weights = ExpertWeights(
                gate_proj=_shard_tensor(gate, mesh_device=mesh_device, dim=1, dtype=gate_up_dtype),
                up_proj=_shard_tensor(up, mesh_device=mesh_device, dim=1, dtype=gate_up_dtype),
                down_proj=_shard_tensor(down, mesh_device=mesh_device, dim=1, dtype=down_dtype),
                gate_proj_bias=_shard_tensor(gate_bias, mesh_device=mesh_device, dim=1),
                up_proj_bias=_shard_tensor(up_bias, mesh_device=mesh_device, dim=1),
                down_proj_bias=_shard_tensor(down_bias, mesh_device=mesh_device, dim=1),
                intermediate_size_per_device=intermediate_size,
            )
            expert_program = OptimizedGPTOSSProgramConfig(
                decode_gate_up_cores=decode_gate_up_cores,
                decode_down_cores=decode_down_cores,
                prefill_gate_up_cores=multichip_config.prefill_expert_cores,
                prefill_down_cores=multichip_config.prefill_expert_cores,
                decode_gate_up_in0_block_w=decode_gate_up_in0_block_w,
                decode_down_in0_block_w=decode_down_in0_block_w,
                prefill_gate_up_in0_block_w=multichip_config.expert_in0_block_w,
                prefill_down_in0_block_w=multichip_config.expert_in0_block_w,
                decode_gate_up_subblock_w=decode_gate_up_subblock_w,
                decode_down_subblock_w=decode_down_subblock_w,
                prefill_gate_up_subblock_w=multichip_config.prefill_expert_subblock_w,
                prefill_down_subblock_w=multichip_config.prefill_expert_subblock_w,
                sequence_chunk_size=multichip_config.active_prefill_chunk_size,
            )
        decoder.experts = SimpleNamespace(
            config=ExpertConfig(
                intermediate_size=intermediate_size,
                num_experts=num_experts,
                hidden_size=hidden_size,
                num_experts_per_tok=experts_per_token,
                swiglu_limit=swiglu_limit,
                alpha=1.703125,
            ),
            weights=expert_weights,
            program_config=expert_program,
        )
        decoder.packed_gate_up_weight = None
        decoder.packed_gate_up_bias = None
        if multichip_config.use_packed_sparse_gate_up:
            decoder.packed_gate_up_weight = _shard_tensor(
                gate_up.reshape(1, num_experts, hidden_size, 2 * intermediate_size),
                mesh_device=mesh_device,
                dim=1,
                dtype=gate_up_dtype,
            )
            decoder.packed_gate_up_bias = _shard_tensor(
                gate_up_bias.reshape(1, num_experts, 2 * intermediate_size),
                mesh_device=mesh_device,
                dim=1,
            )
        return decoder

    def create_page_table(self, physical_block_ids: Sequence[int] | None = None):
        """Create a replicated logical-page to physical-page mapping."""

        if physical_block_ids is None:
            physical_block_ids = range(self.num_cache_blocks)
        physical_block_ids = tuple(int(block) for block in physical_block_ids)
        if len(physical_block_ids) != self.num_cache_blocks:
            raise ValueError(f"page table requires {self.num_cache_blocks} block ids")
        if set(physical_block_ids) != set(range(self.num_cache_blocks)):
            raise ValueError("page table block ids must be a permutation of all physical cache blocks")
        table = torch.tensor(physical_block_ids, dtype=torch.int32).reshape(self.batch, self.num_cache_blocks)
        return _replicate_tensor(
            table,
            mesh_device=self.mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )

    def create_position_tensor(self, position: int):
        if not 0 <= position < self.max_cache_len:
            raise ValueError(f"position must be in [0, {self.max_cache_len}), got {position}")
        return _replicate_tensor(
            torch.tensor([position], dtype=torch.int32),
            mesh_device=self.mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )

    def create_kv_cache(self):
        """Allocate two local BFP8 KV heads per rank in physical pages."""

        cache_dtype = {
            "bfloat8_b": ttnn.bfloat8_b,
            "bfloat16": ttnn.bfloat16,
        }[self.multichip_config.kv_cache_dtype]
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

    def _validate_page_table(self, page_table) -> None:
        expected = (self.batch, self.num_cache_blocks)
        if tuple(page_table.shape) != expected:
            raise ValueError(f"page_table must have shape {expected}, got {tuple(page_table.shape)}")

    def _validate_local_caches(self, key_cache, value_cache) -> None:
        expected = (
            self.num_cache_blocks,
            self.local_num_kv_heads,
            self.multichip_config.page_block_size,
            self.head_dim,
        )
        if tuple(key_cache.shape) != expected or tuple(value_cache.shape) != expected:
            raise ValueError(
                f"key/value caches must have local paged shape {expected}; got "
                f"{tuple(key_cache.shape)} and {tuple(value_cache.shape)}"
            )

    def _ring_all_reduce(self, tensor, *, memory_config):
        return ttnn.all_reduce(
            tensor,
            num_links=self.multichip_config.num_links,
            topology=ttnn.Topology.Ring,
            cluster_axis=1,
            memory_config=memory_config,
        )

    def _minimal_all_reduce(self, tensor, *, memory_config):
        """Run the minimal persistent-buffer AR on a width-sharded decode tensor."""

        tensor = ttnn.to_memory_config(tensor, self.tp_o_output_config)
        ccl_dtype = {
            "bfloat8_b": ttnn.bfloat8_b,
            "bfloat16": ttnn.bfloat16,
        }[self.multichip_config.decode_ccl_dtype]
        if tensor.dtype != ccl_dtype:
            tensor = ttnn.typecast(tensor, ccl_dtype, memory_config=self.tp_o_output_config)
        buffer_index = self._minimal_ar_index
        self._minimal_ar_index = (buffer_index + 1) % len(self._minimal_ar_buffers)
        output = ttnn.experimental.all_reduce_async(
            tensor,
            self._minimal_ar_buffers[buffer_index],
            cluster_axis=1,
            mesh_device=self.mesh_device,
            multi_device_global_semaphore=self._minimal_ar_semaphores[buffer_index],
            dtype=ccl_dtype,
            memory_config=self.tp_o_output_config,
            topology=ttnn.Topology.Ring,
            num_links=self.multichip_config.num_links,
        )
        if output.dtype != ttnn.bfloat16:
            output = ttnn.typecast(output, ttnn.bfloat16, memory_config=self.tp_o_output_config)
        if memory_config != self.tp_o_output_config:
            output = ttnn.to_memory_config(output, memory_config)
        return output

    def _all_reduce(self, tensor, *, memory_config):
        """Use ordinary AR or the decode-only padded persistent-semaphore RS+AG control."""

        if self.multichip_config.decode_collective == DECODE_COLLECTIVE_ALL_REDUCE or tensor.shape[-2] != 1:
            return self._ring_all_reduce(tensor, memory_config=memory_config)
        if self.multichip_config.decode_collective == DECODE_COLLECTIVE_MINIMAL_ALL_REDUCE:
            return self._minimal_all_reduce(tensor, memory_config=memory_config)

        # Four-way width reduce-scatter requires tile-aligned rank-local
        # pieces. Pad logical H=2880 to 2944 so each rank receives 736
        # values, complete the all-gather, then restore the public H=2880
        # replicated contract.
        padded = ttnn.pad(
            tensor,
            [(0, 0), (0, 0), (0, 0), (0, 64)],
            value=0.0,
        )
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

    def _fused_o_projection_reduce_scatter(self, attended, *, compute_kernel_config=None):
        """Return the padded H=2944 O projection as a rank-local H=736 shard."""

        attended = ttnn.to_memory_config(attended, ttnn.DRAM_MEMORY_CONFIG)
        if len(attended.shape) == 3:
            attended = ttnn.reshape(attended, [1, 1, attended.shape[-2], attended.shape[-1]])
        if attended.dtype != self.fused_o_rs_weight.dtype:
            attended = ttnn.typecast(attended, self.fused_o_rs_weight.dtype)
        mm_out, scattered = ttnn.experimental.minimal_matmul_strided_reduce_scatter_async(
            attended,
            self.fused_o_rs_weight,
            3,
            self.ccl_manager.get_rs_ping_pong_semaphore(),
            ttnn.CoreCoord(0, 2),
            compute_kernel_config=(
                compute_kernel_config if compute_kernel_config is not None else self.decode_compute_kernel_config
            ),
            num_links=self.ccl_manager.num_links,
            memory_config_mm=ttnn.DRAM_MEMORY_CONFIG,
            rs_output_mem_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=self.ccl_manager.topology,
            cluster_axis=self.mesh_config.tp_axis,
            bias=self.fused_o_rs_bias,
            config=ttnn.MinimalMatmulConfig(
                M_block_size=1,
                K_block_size=8,
                N_block_size=1,
                subblock_h=1,
                subblock_w=1,
                compute_with_storage_grid_size=ttnn.CoreCoord(4, 2),
            ),
            barrier_semaphore=self.ccl_manager.get_barrier_semaphore(),
            chunk_width_in_mm_blocks=1,
            num_workers_per_link=2,
        )
        attended.deallocate(True)
        mm_out.deallocate(True)
        return scattered

    def _project_o_and_reduce(
        self,
        attended,
        *,
        is_decode: bool,
        compute_kernel_config=None,
        use_long_attention_weights: bool = False,
    ):
        if is_decode and self.multichip_config.use_fused_o_projection_rs and not use_long_attention_weights:
            scattered = self._fused_o_projection_reduce_scatter(
                attended,
                compute_kernel_config=compute_kernel_config,
            )
            gathered = self.mesh_config.allgather(
                scattered,
                self.ccl_manager,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                axis=self.mesh_config.tp_axis,
            )
            scattered.deallocate(True)
            output = ttnn.slice(
                gathered,
                [0, 0, 0, 0],
                [gathered.shape[0], gathered.shape[1], gathered.shape[2], self.hidden_size],
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            gathered.deallocate(True)
            return output

        if self.multichip_config.use_fused_o_projection_ag:
            attended = ttnn.to_memory_config(attended, ttnn.DRAM_MEMORY_CONFIG)
            if len(attended.shape) == 3:
                attended = ttnn.reshape(
                    attended,
                    [1, 1, attended.shape[-2], attended.shape[-1]],
                )
            payload_dtype = {
                "bfloat8_b": ttnn.bfloat8_b,
                "bfloat16": ttnn.bfloat16,
            }[self.multichip_config.fused_ag_matmul_payload_dtype]
            if attended.dtype != payload_dtype:
                attended = ttnn.typecast(attended, payload_dtype)
            gathered_shape = list(attended.shape)
            gathered_shape[-1] *= TP_DEGREE
            persistent_key = (
                tuple(gathered_shape),
                self.multichip_config.fused_ag_matmul_payload_dtype,
            )
            persistent_gather = self._fused_o_ag_persistent_buffers.get(persistent_key)
            if persistent_gather is None:
                persistent_gather = ttnn.zeros(
                    gathered_shape,
                    dtype=payload_dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.mesh_device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                self._fused_o_ag_persistent_buffers[persistent_key] = persistent_gather
            local_projected = ttnn.experimental.all_gather_minimal_matmul_async(
                attended,
                self.fused_o_ag_weight,
                bias_tensor=self.fused_o_ag_bias,
                config=ttnn.MinimalMatmulConfig(
                    M_block_size=1,
                    K_block_size=8,
                    N_block_size=2,
                    subblock_h=1,
                    subblock_w=2,
                    compute_with_storage_grid_size=ttnn.CoreCoord(4, 4),
                ),
                multi_device_global_semaphore=self.ccl_manager.get_ag_ping_pong_semaphore(),
                topology=self.ccl_manager.topology,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                dtype=ttnn.bfloat16,
                compute_kernel_config=(
                    compute_kernel_config
                    if is_decode and compute_kernel_config is not None
                    else self.decode_compute_kernel_config
                    if is_decode
                    else self.compute_kernel_config
                ),
                persistent_output_buffer=persistent_gather,
                num_links=self.ccl_manager.num_links,
                cluster_axis=self.mesh_config.tp_axis,
                force_transpose=True,
                num_workers_per_link=4,
                num_buffers_per_channel=2,
            )[0]
            gathered = self.mesh_config.allgather(
                local_projected,
                self.ccl_manager,
                memory_config=ttnn.L1_MEMORY_CONFIG if is_decode else ttnn.DRAM_MEMORY_CONFIG,
                axis=self.mesh_config.tp_axis,
            )
            local_projected.deallocate(True)
            if not self.multichip_config.fused_o_ag_pad_hidden:
                return gathered
            output = ttnn.slice(
                gathered,
                [0, 0, 0, 0],
                [gathered.shape[0], gathered.shape[1], gathered.shape[2], self.hidden_size],
                memory_config=ttnn.L1_MEMORY_CONFIG if is_decode else ttnn.DRAM_MEMORY_CONFIG,
            )
            gathered.deallocate(True)
            return output

        dram_attention = is_decode and self.multichip_config.use_dram_sharded_decode_attention
        if dram_attention:
            attended = ttnn.to_memory_config(attended, self.dram_tp_o_input_config)
        elif is_decode:
            attended = ttnn.to_memory_config(attended, ttnn.L1_MEMORY_CONFIG)
        partial = ttnn.linear(
            attended,
            (
                self.dram_decode_o_weight
                if dram_attention
                else (
                    self.output_weight
                    if is_decode and use_long_attention_weights
                    else self.decode_output_weight
                    if is_decode
                    else self.output_weight
                )
            ),
            bias=None if dram_attention else self.output_bias,
            dtype=ttnn.bfloat16,
            memory_config=(
                self.dram_tp_o_output_config
                if dram_attention
                else self.tp_o_output_config
                if is_decode
                else ttnn.DRAM_MEMORY_CONFIG
            ),
            program_config=(
                self.dram_tp_o_program_config if dram_attention else self.tp_o_program_config if is_decode else None
            ),
            compute_kernel_config=(
                compute_kernel_config
                if is_decode and compute_kernel_config is not None
                else self.decode_compute_kernel_config
                if is_decode
                else self.compute_kernel_config
            ),
        )
        if dram_attention:
            partial = ttnn.add(partial, self.output_bias, memory_config=self.dram_tp_o_output_config)
            partial = ttnn.to_memory_config(partial, self.tp_o_output_config)
        if is_decode and self.multichip_config.decode_collective != DECODE_COLLECTIVE_MINIMAL_ALL_REDUCE:
            partial = ttnn.to_memory_config(partial, ttnn.L1_MEMORY_CONFIG)
        return self._all_reduce(
            partial,
            memory_config=ttnn.L1_MEMORY_CONFIG if is_decode else ttnn.DRAM_MEMORY_CONFIG,
        )

    def _manual_prefill_attention(self, query, key, value, seq_len: int):
        """Run exact TP-local sink-aware FP32 full attention at S=128."""

        if (
            seq_len != EMITTED_CACHE_LENGTH
            or self.attention_window is not None
            or self.attention_mask is None
            or self.manual_prefill_attention_sinks is None
        ):
            raise ValueError("manual prefill attention is specialized for full-attention S=128")
        key = ttnn.repeat_interleave(
            key,
            self.local_num_heads // self.local_num_kv_heads,
            1,
        )
        key = ttnn.permute(key, (0, 1, 3, 2), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        scores = ttnn.matmul(
            query,
            key,
            dtype=ttnn.float32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        scores = ttnn.multiply(
            scores,
            self.scale,
            dtype=ttnn.float32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        mask = ttnn.typecast(
            self.attention_mask,
            ttnn.float32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        scores = ttnn.add(
            scores,
            mask,
            dtype=ttnn.float32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        sinks = ttnn.typecast(
            self.manual_prefill_attention_sinks,
            ttnn.float32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        logits = ttnn.concat([scores, sinks], 3, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        probabilities_with_sink = ttnn.softmax(
            logits,
            3,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            numeric_stable=True,
        )
        probabilities = ttnn.slice(
            probabilities_with_sink,
            [0, 0, 0, 0],
            [self.batch, self.local_num_heads, seq_len, seq_len],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        value = ttnn.repeat_interleave(
            value,
            self.local_num_heads // self.local_num_kv_heads,
            1,
        )
        attention = ttnn.matmul(
            probabilities,
            value,
            dtype=ttnn.float32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return ttnn.typecast(
            attention,
            ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _prefill_attention(self, hidden_states, key_cache, value_cache, page_table, seq_len: int):
        residual = hidden_states
        normalized = ttnn.rms_norm(
            hidden_states,
            epsilon=self.rms_norm_eps,
            weight=self.input_norm,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
        fused = self._bounded_prefill_linear(normalized, self.qkv_weight, self.qkv_bias, seq_len)
        fused = ttnn.reshape(fused, [self.batch, seq_len, -1])
        query, key, value = ttnn.transformer.split_query_key_value_and_split_heads(
            fused,
            None,
            num_heads=self.local_num_heads,
            num_kv_heads=self.local_num_kv_heads,
            transpose_key=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        cos = ttnn.slice(self.rotary_cos, [0, 0, 0, 0], [1, 1, seq_len, self.head_dim])
        sin = ttnn.slice(self.rotary_sin, [0, 0, 0, 0], [1, 1, seq_len, self.head_dim])
        query = ttnn.experimental.rotary_embedding(query, cos, sin, None, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        key = ttnn.experimental.rotary_embedding(key, cos, sin, None, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        query = ttnn.slice(
            query,
            [0, 0, 0, 0],
            [self.batch, self.local_num_heads, seq_len, self.head_dim],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        key = ttnn.slice(
            key,
            [0, 0, 0, 0],
            [self.batch, self.local_num_kv_heads, seq_len, self.head_dim],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        cache_key = key
        cache_value = value
        if key_cache.dtype == ttnn.bfloat8_b:
            cache_key = ttnn.typecast(key, ttnn.bfloat8_b, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            cache_value = ttnn.typecast(value, ttnn.bfloat8_b, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.experimental.paged_fill_cache(key_cache, cache_key, page_table, batch_idx=0)
        ttnn.experimental.paged_fill_cache(value_cache, cache_value, page_table, batch_idx=0)

        use_manual_attention = (
            self.optimization_config.use_manual_prefill_attention
            and self.attention_window is None
            and seq_len == EMITTED_CACHE_LENGTH
        )
        if use_manual_attention:
            attention = self._manual_prefill_attention(query, key, value, seq_len)
        else:
            sdpa_chunk = self.optimization_config.prefill_sdpa_chunk_size
            if self.attention_window is None and seq_len > EMITTED_CACHE_LENGTH:
                sdpa_chunk = self.optimization_config.long_prefill_sdpa_chunk_size
            attention = ttnn.transformer.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=None,
                is_causal=True,
                scale=self.scale,
                sliding_window_size=self.attention_window,
                attention_sink=self.prefill_sdpa_sink,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=self.compute_kernel_config,
                program_config=(
                    ttnn.SDPAProgramConfig(
                        compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
                        q_chunk_size=sdpa_chunk,
                        k_chunk_size=sdpa_chunk,
                        exp_approx_mode=False,
                    )
                    if sdpa_chunk is not None
                    else None
                ),
            )
        attention = ttnn.transformer.concatenate_heads(attention, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        attention = ttnn.reshape(
            attention,
            [1, self.batch, seq_len, self.local_num_heads * self.head_dim],
        )
        partial = self._bounded_prefill_linear(attention, self.output_weight, self.output_bias, seq_len)
        attention = self._all_reduce(partial, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.add(residual, attention, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def _logical_cache_range(self, cache, page_table, start: int, end: int):
        """Gather a logical half-open range of a local paged cache on device."""

        if not 0 <= start < end <= self.max_cache_len:
            raise ValueError(f"invalid logical cache range [{start}, {end})")
        page_size = self.multichip_config.page_block_size
        first_block = start // page_size
        last_block = (end - 1) // page_size
        logical_blocks = last_block - first_block + 1
        page_ids = ttnn.slice(
            page_table,
            [0, first_block],
            [self.batch, last_block + 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        page_ids = ttnn.typecast(page_ids, ttnn.uint32)
        flat_width = self.local_num_kv_heads * page_size * self.head_dim
        flat_cache = ttnn.reshape(cache, [self.num_cache_blocks, flat_width])
        pages = ttnn.embedding(
            page_ids,
            flat_cache,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        pages = ttnn.reshape(
            pages,
            [
                self.batch,
                logical_blocks,
                self.local_num_kv_heads,
                page_size * self.head_dim,
            ],
        )
        pages = ttnn.permute(pages, (0, 2, 1, 3), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        logical = ttnn.reshape(
            pages,
            [
                self.batch,
                self.local_num_kv_heads,
                logical_blocks * page_size,
                self.head_dim,
            ],
        )
        start_in_gather = start - first_block * page_size
        length = end - start
        if start_in_gather == 0 and length == logical_blocks * page_size:
            return logical
        return ttnn.slice(
            logical,
            [0, 0, start_in_gather, 0],
            [self.batch, self.local_num_kv_heads, start_in_gather + length, self.head_dim],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _logical_cache_prefix(self, cache, page_table, valid_length: int):
        return self._logical_cache_range(cache, page_table, 0, valid_length)

    def _manual_paged_decode_attention(
        self,
        query,
        key_cache,
        value_cache,
        page_table,
        cache_position: int,
        cache_position_tensor,
    ):
        """Run trace-bank-stable sink-aware FP32 attention over logical cache data."""

        page_size = self.multichip_config.page_block_size
        bank_end = min(
            self.max_cache_len,
            math.ceil((cache_position + 1) / page_size) * page_size,
        )
        bank_start = max(0, bank_end - self.attention_window - page_size) if self.attention_window is not None else 0
        attention_length = bank_end - bank_start
        query = ttnn.to_memory_config(query, ttnn.DRAM_MEMORY_CONFIG)
        query = ttnn.slice(
            query,
            [0, 0, 0, 0],
            [1, self.batch, self.local_num_heads, self.head_dim],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        query = ttnn.permute(query, (1, 2, 0, 3), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        key = self._logical_cache_range(key_cache, page_table, bank_start, bank_end)
        value = self._logical_cache_range(value_cache, page_table, bank_start, bank_end)
        key = ttnn.repeat_interleave(key, self.local_num_heads // self.local_num_kv_heads, 1)
        value = ttnn.repeat_interleave(value, self.local_num_heads // self.local_num_kv_heads, 1)
        key = ttnn.permute(key, (0, 1, 3, 2), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        scores = ttnn.matmul(query, key, dtype=ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        scores = ttnn.multiply(scores, self.scale, dtype=ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        positions = ttnn.slice(
            self.position_indices,
            [bank_start],
            [bank_end],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        positions = ttnn.reshape(positions, [1, 1, 1, attention_length])
        current_position = ttnn.reshape(cache_position_tensor, [1, 1, 1, 1])
        valid = ttnn.le(
            positions,
            current_position,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if self.attention_window is not None:
            first_position = ttnn.subtract(
                current_position,
                self.attention_window - 1,
                dtype=ttnn.int32,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            in_window = ttnn.ge(
                positions,
                first_position,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            valid = ttnn.multiply(
                valid,
                in_window,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        attention_mask = ttnn.where(
            valid,
            0.0,
            -1.0e30,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        attention_mask = ttnn.typecast(
            attention_mask,
            ttnn.float32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        scores = ttnn.add(
            scores,
            attention_mask,
            dtype=ttnn.float32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        sinks = ttnn.typecast(self.manual_attention_sinks, ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        logits = ttnn.concat([scores, sinks], 3, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        probabilities_with_sink = ttnn.softmax(
            logits,
            3,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            numeric_stable=True,
        )
        probabilities = ttnn.slice(
            probabilities_with_sink,
            [0, 0, 0, 0],
            [self.batch, self.local_num_heads, 1, attention_length],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        attention = ttnn.matmul(
            probabilities,
            value,
            dtype=ttnn.float32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        attention = ttnn.typecast(attention, ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        attention = ttnn.permute(attention, (0, 2, 1, 3), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.reshape(
            attention,
            [1, self.batch, 1, self.local_num_heads * self.head_dim],
        )

    def _decode_norm(self, hidden_states, *, weight_name: str):
        if not self.multichip_config.use_sharded_decode_norms:
            return ttnn.rms_norm(
                hidden_states,
                epsilon=self.rms_norm_eps,
                weight=getattr(self, weight_name),
                compute_kernel_config=self.compute_kernel_config,
            )
        norm_input = ttnn.to_memory_config(hidden_states, self.advisor_norm_memory_config)
        normalized = ttnn.rms_norm(
            norm_input,
            epsilon=self.rms_norm_eps,
            weight=self.advisor_norm_weights[weight_name],
            memory_config=self.advisor_norm_memory_config,
            program_config=self.advisor_norm_program_config,
            compute_kernel_config=self.compute_kernel_config,
        )
        return normalized

    def _decode_attention(
        self,
        hidden_states,
        key_cache,
        value_cache,
        page_table,
        cache_position: int,
        cache_position_tensor,
    ):
        residual = hidden_states
        normalized = self._decode_norm(hidden_states, weight_name="input_norm")
        use_long_attention_weights = cache_position >= EMITTED_CACHE_LENGTH - 1
        # BFP8 projection copies are selected only for the short native-SDPA
        # decode path.  Long trace banks retain the BF16 projection policy:
        # a near-tied sliding-layer router decision at position 130 changes
        # with BFP8 weights even though attention PCC remains above 0.999.
        dram_attention = self.multichip_config.use_dram_sharded_decode_attention and not use_long_attention_weights
        normalized = ttnn.to_memory_config(
            normalized,
            self.dram_tp_qkv_input_config if dram_attention else self.tp_qkv_input_config,
        )
        projection_kernel = (
            self.long_decode_compute_kernel_config
            if cache_position >= EMITTED_CACHE_LENGTH - 1
            else self.decode_compute_kernel_config
        )
        fused = ttnn.linear(
            normalized,
            (
                self.dram_decode_qkv_weight
                if dram_attention
                else self.qkv_weight
                if use_long_attention_weights
                else self.decode_qkv_weight
            ),
            bias=None if dram_attention else self.qkv_bias,
            dtype=ttnn.bfloat16,
            memory_config=self.dram_tp_qkv_output_config if dram_attention else self.tp_qkv_output_config,
            program_config=self.dram_tp_qkv_program_config if dram_attention else self.tp_qkv_program_config,
            compute_kernel_config=projection_kernel,
        )
        if dram_attention:
            fused = ttnn.add(fused, self.qkv_bias, memory_config=self.dram_tp_qkv_output_config)
        fused = ttnn.to_memory_config(fused, ttnn.L1_MEMORY_CONFIG)
        fused = ttnn.reshape(fused, [1, 1, self.batch, -1])
        query, key, value = ttnn.experimental.nlp_create_qkv_heads_decode(
            fused,
            num_heads=self.local_num_heads,
            num_kv_heads=self.local_num_kv_heads,
            memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
        )
        rope_position = ttnn.reshape(ttnn.typecast(cache_position_tensor, ttnn.uint32), [1, self.batch])
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
        cos = ttnn.interleaved_to_sharded(
            ttnn.transpose(ttnn.unsqueeze_to_4D(cos), 1, 2),
            self.decode_rope_memory_config,
        )
        sin = ttnn.interleaved_to_sharded(
            ttnn.transpose(ttnn.unsqueeze_to_4D(sin), 1, 2),
            self.decode_rope_memory_config,
        )
        query = ttnn.experimental.rotary_embedding_hf(
            query,
            cos,
            sin,
            is_decode_mode=True,
            memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
            compute_kernel_config=projection_kernel,
        )
        key = ttnn.experimental.rotary_embedding_hf(
            key,
            cos,
            sin,
            is_decode_mode=True,
            memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
            compute_kernel_config=projection_kernel,
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

        use_manual_attention = cache_position >= EMITTED_CACHE_LENGTH - 1
        if use_manual_attention:
            attention = self._manual_paged_decode_attention(
                query,
                key_cache,
                value_cache,
                page_table,
                cache_position,
                cache_position_tensor,
            )
        else:
            attention = ttnn.transformer.paged_scaled_dot_product_attention_decode(
                query,
                key_cache,
                value_cache,
                page_table_tensor=page_table,
                is_causal=True,
                attn_mask=None,
                cur_pos_tensor=cache_position_tensor,
                attention_sink=self.decode_attention_sinks,
                scale=self.scale,
                sliding_window_size=None,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                program_config=self.decode_sdpa_program_config,
                compute_kernel_config=self.compute_kernel_config,
            )
            attention = ttnn.to_memory_config(attention, self.decode_heads_mem_config)
            attention = ttnn.experimental.nlp_concat_heads_decode(attention, num_heads=self.local_num_heads)
            attention = ttnn.slice(
                attention,
                [0, 0, 0, 0],
                [1, 1, self.batch, self.local_num_heads * self.head_dim],
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            )
        projected = self._project_o_and_reduce(
            attention,
            is_decode=True,
            compute_kernel_config=projection_kernel,
            use_long_attention_weights=use_long_attention_weights,
        )
        projected = ttnn.reshape(projected, [1, self.batch, 1, self.hidden_size])
        return ttnn.add(residual, projected, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def _ep_active_expert_chunk(self, normalized, routing_weights, *, is_decode: bool):
        """Execute only selected routes with EP4 whole experts or TP4 I shards."""

        seq_len = int(normalized.shape[2])
        if not is_decode and seq_len % ttnn.TILE_SIZE:
            raise ValueError("active prefill expert chunks must be tile aligned")
        weights = self.experts.weights
        program = self.experts.program_config
        use_tp_experts = self.multichip_config.expert_strategy == EXPERT_STRATEGY_TP
        local_experts = self.num_experts if use_tp_experts else self.local_num_experts
        local_intermediate = self.local_intermediate_size if use_tp_experts else self.intermediate_size
        output_tile = ttnn.Tile([ttnn.TILE_SIZE, ttnn.TILE_SIZE])
        use_l1_output = (
            self.multichip_config.decode_expert_output_l1
            if is_decode
            else (
                self.multichip_config.prefill_expert_output_l1
                and seq_len <= self.multichip_config.prefill_expert_output_l1_max_seq
            )
        )
        memory_config = ttnn.L1_MEMORY_CONFIG if use_l1_output else ttnn.DRAM_MEMORY_CONFIG
        activation_dtype = {
            "bfloat8_b": ttnn.bfloat8_b,
            "bfloat16": ttnn.bfloat16,
        }[self.multichip_config.expert_activation_dtype]
        expert_compute_kernel_config = (
            self.ep_decode_compute_kernel_config
            if is_decode
            else (
                self.ep_sliding_prefill_compute_kernel_config
                if self.attention_window is not None
                else self.ep_full_prefill_compute_kernel_config
            )
        )

        token_major = ttnn.reshape(
            normalized,
            [seq_len, 1, 1, self.hidden_size],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        routing_rm = ttnn.to_layout(routing_weights, ttnn.ROW_MAJOR_LAYOUT)
        local_routing_rm = (
            routing_rm
            if use_tp_experts
            else ttnn.mesh_partition(
                routing_rm,
                dim=1,
                cluster_axis=1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        )
        sparsity = ttnn.to_layout(
            ttnn.reshape(local_routing_rm, [seq_len, 1, 1, local_experts]),
            ttnn.ROW_MAJOR_LAYOUT,
        )
        local_routing = ttnn.to_layout(local_routing_rm, ttnn.TILE_LAYOUT)
        gate_up_width = (
            2 * local_intermediate if self.multichip_config.use_packed_sparse_gate_up else local_intermediate
        )
        gate_up_config = (
            program.get_decode_gate_up_config(1, gate_up_width, k=self.hidden_size)
            if is_decode
            else program.get_prefill_gate_up_config(1, gate_up_width, k=self.hidden_size)
        )
        if self.multichip_config.use_packed_sparse_gate_up:
            packed = ttnn.sparse_matmul(
                token_major,
                self.packed_gate_up_weight,
                sparsity=sparsity,
                nnz=None,
                memory_config=memory_config,
                output_tile=output_tile,
                program_config=gate_up_config,
                dtype=activation_dtype,
                compute_kernel_config=expert_compute_kernel_config,
            )
            token_major.deallocate(True)
            packed = ttnn.reshape(packed, [seq_len, local_experts, 2 * local_intermediate])
            packed = ttnn.add(
                packed,
                ttnn.reshape(self.packed_gate_up_bias, [1, local_experts, 2 * local_intermediate]),
                output_tensor=packed,
            )
            if packed.dtype != ttnn.bfloat16:
                packed = ttnn.typecast(packed, ttnn.bfloat16, memory_config=memory_config)
            gate = ttnn.slice(
                packed,
                [0, 0, 0],
                [seq_len, local_experts, 2 * local_intermediate],
                [1, 1, 2],
                memory_config=memory_config,
            )
            up = ttnn.slice(
                packed,
                [0, 0, 1],
                [seq_len, local_experts, 2 * local_intermediate],
                [1, 1, 2],
                memory_config=memory_config,
            )
            packed.deallocate(True)
        else:
            gate = ttnn.sparse_matmul(
                token_major,
                weights.gate_proj,
                sparsity=sparsity,
                nnz=None,
                memory_config=memory_config,
                output_tile=output_tile,
                program_config=gate_up_config,
                dtype=activation_dtype,
                compute_kernel_config=expert_compute_kernel_config,
            )
            gate = ttnn.reshape(gate, [seq_len, local_experts, local_intermediate])
            gate = ttnn.add(
                gate,
                ttnn.reshape(weights.gate_proj_bias, [1, local_experts, local_intermediate]),
                output_tensor=gate,
            )
            up = ttnn.sparse_matmul(
                token_major,
                weights.up_proj,
                sparsity=sparsity,
                nnz=None,
                memory_config=memory_config,
                output_tile=output_tile,
                program_config=gate_up_config,
                dtype=activation_dtype,
                compute_kernel_config=expert_compute_kernel_config,
            )
            token_major.deallocate(True)
            up = ttnn.reshape(up, [seq_len, local_experts, local_intermediate])
            up = ttnn.add(
                up,
                ttnn.reshape(weights.up_proj_bias, [1, local_experts, local_intermediate]),
                output_tensor=up,
            )
        down_input = apply_swiglu(gate, up, self.experts.config)
        down_input = ttnn.reshape(
            down_input,
            [seq_len, local_experts, 1, local_intermediate],
        )
        down_sparsity = ttnn.reshape(sparsity, [1, 1, seq_len, local_experts])
        down_config = (
            program.get_decode_down_config(1, self.hidden_size, k=local_intermediate)
            if is_decode
            else program.get_prefill_down_config(1, self.hidden_size, k=local_intermediate)
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
            dtype=activation_dtype,
            compute_kernel_config=expert_compute_kernel_config,
        )
        down_input.deallocate(True)
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
        return ttnn.reshape(down, [1, 1, seq_len, self.hidden_size])

    def _active_prefill_moe(self, hidden_states, normalized, routing_weights, seq_len: int):
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
        normalized_chunks = ttnn.split(normalized, chunk_size, dim=2) if padded_len > chunk_size else [normalized]
        routing_chunks = (
            ttnn.split(routing_weights, chunk_size, dim=0) if padded_len > chunk_size else [routing_weights]
        )
        partial_chunks = [
            self._ep_active_expert_chunk(normalized_chunk, routing_chunk, is_decode=False)
            for normalized_chunk, routing_chunk in zip(normalized_chunks, routing_chunks)
        ]
        partial = partial_chunks[0] if len(partial_chunks) == 1 else ttnn.concat(partial_chunks, dim=2)
        expert_output = self._all_reduce(partial, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if padded_len != seq_len:
            expert_output = ttnn.slice(
                expert_output,
                [0, 0, 0, 0],
                [1, self.batch, seq_len, self.hidden_size],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        return ttnn.add(
            hidden_states,
            expert_output,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _multichip_moe_forward(self, hidden_states, seq_len: int):
        if seq_len == 1:
            normalized = self._decode_norm(hidden_states, weight_name="post_attention_norm")
            normalized = ttnn.to_memory_config(normalized, ttnn.DRAM_MEMORY_CONFIG)
            routing_weights = self._route(normalized, seq_len)
            partial = self._ep_active_expert_chunk(normalized, routing_weights, is_decode=True)
            expert_output = self._all_reduce(partial, memory_config=ttnn.L1_MEMORY_CONFIG)
            return ttnn.add(
                hidden_states,
                expert_output,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        normalized = ttnn.rms_norm(
            hidden_states,
            epsilon=self.rms_norm_eps,
            weight=self.post_attention_norm,
            compute_kernel_config=self.compute_kernel_config,
        )
        routing_weights = self._route(normalized, seq_len)
        return self._active_prefill_moe(hidden_states, normalized, routing_weights, seq_len)

    def prefill_forward(self, hidden_states, *, key_cache, value_cache, page_table):
        seq_len = self._validate_hidden_states(hidden_states)
        if seq_len <= 1:
            raise ValueError("prefill_forward requires seq_len > 1")
        self._validate_local_caches(key_cache, value_cache)
        self._validate_page_table(page_table)
        hidden_states = self._prefill_attention(
            hidden_states,
            key_cache,
            value_cache,
            page_table,
            seq_len,
        )
        return self._multichip_moe_forward(hidden_states, seq_len)

    def decode_forward(
        self,
        hidden_states,
        *,
        key_cache,
        value_cache,
        page_table,
        cache_position: int,
        cache_position_tensor,
    ):
        self._validate_hidden_states(hidden_states, expected_seq_len=1)
        self._validate_local_caches(key_cache, value_cache)
        self._validate_page_table(page_table)
        if not 0 <= cache_position < self.max_cache_len:
            raise ValueError(f"cache_position must be in [0, {self.max_cache_len}), got {cache_position}")
        if tuple(cache_position_tensor.shape) != (self.batch,):
            raise ValueError(
                f"cache_position_tensor must have shape {(self.batch,)}, got {tuple(cache_position_tensor.shape)}"
            )
        hidden_states = self._decode_attention(
            hidden_states,
            key_cache,
            value_cache,
            page_table,
            cache_position,
            cache_position_tensor,
        )
        return self._multichip_moe_forward(hidden_states, 1)


__all__ = [
    "EMITTED_BATCH",
    "EMITTED_CACHE_LENGTH",
    "EMITTED_PREFILL_SEQUENCE",
    "DECODE_COLLECTIVE_ALL_REDUCE",
    "DECODE_COLLECTIVE_MINIMAL_ALL_REDUCE",
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
]
