# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Optimized single-device decoder for ``google/gemma-4-26B-A4B-it``.

The functional decoder owns the public tensor/cache semantics and setup-only
weight transformation.  This class deliberately overrides every material
measured projection path so optimized tests cannot silently exercise a
functional matmul fallback.  Candidate precision and sparse geometry are
constructor inputs; the defaults are the strongest cumulative policy selected
by this stage and are changed only after real-weight correctness and traced
batch-1/batch-32 measurements.
"""

from __future__ import annotations

import math
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import ttnn

from models.autoports.google_gemma_4_26b_a4b_it.tt.functional_decoder import (
    HIDDEN_SIZE,
    MLP_INTERMEDIATE_SIZE,
    MOE_INTERMEDIATE_SIZE,
    NUM_EXPERTS,
    NUM_Q_HEADS,
    PREFILL_MOE_CHUNK_SIZE,
    TILE_SIZE,
    TOP_K_EXPERTS,
    FunctionalDecoder,
    _detect_layer_prefix,
    _make_decode_height_sharded_memory_config,
    _make_decode_rope_memory_config,
    _replicate_mapper,
)
from models.demos.gemma4.tt.experts.operations import apply_geglu
from models.demos.gemma4.tt.experts.weights import ExpertWeights
from models.common.modules.mlp.mlp_1d import (
    _create_dram_sharded_mem_config,
    _dram_matmul_config,
    _dram_shard_core_grid_k_n,
)

_DTYPES = {
    "bf16": ttnn.bfloat16,
    "bfp8": ttnn.bfloat8_b,
    "bfp4": ttnn.bfloat4_b,
}
_FIDELITIES = {
    "lofi": ttnn.MathFidelity.LoFi,
    "hifi2": ttnn.MathFidelity.HiFi2,
    "hifi4": ttnn.MathFidelity.HiFi4,
}
_RESIDUAL_SHARD_GEOMETRIES = {
    0: None,
    11: (11, 1, 256, 8, 4),
    22: (11, 2, 128, 4, 4),
}
_RESIDUAL_BOUNDARY_COUNTERS = (
    "residual_entry",
    "attention_qkv_input",
    "attention_sdpa_output",
    "attention_o_input",
    "attention_output",
    "router_input",
    "expert_input",
    "expert_output",
    "residual_exit",
)
_DRAM_SHARDED_ROLES = frozenset(
    {"qkv", "o_proj", "mlp_gate", "mlp_up", "mlp_down", "packed_mlp_gate_up"}
)


def _candidate_from_env(name: str, default: Any, choices: dict[str, Any]) -> Any:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return choices[value.lower()]
    except KeyError as error:
        raise ValueError(f"{name} must be one of {sorted(choices)}, got {value!r}") from error


def _bool_from_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    if value.lower() in {"1", "true", "yes", "on"}:
        return True
    if value.lower() in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean, got {value!r}")


def _residual_shard_geometry(cores: int) -> tuple[int, int, int, int, int] | None:
    try:
        return _RESIDUAL_SHARD_GEOMETRIES[cores]
    except KeyError as error:
        raise ValueError(
            f"GEMMA4_OPT_RESIDUAL_SHARD_CORES must be one of {sorted(_RESIDUAL_SHARD_GEOMETRIES)}, got {cores}"
        ) from error


def _residual_shard_cores_from_env(default: int = 0) -> int:
    value = os.getenv("GEMMA4_OPT_RESIDUAL_SHARD_CORES", str(default))
    try:
        cores = int(value)
    except ValueError as error:
        raise ValueError(
            f"GEMMA4_OPT_RESIDUAL_SHARD_CORES must be one of {sorted(_RESIDUAL_SHARD_GEOMETRIES)}, got {value!r}"
        ) from error
    _residual_shard_geometry(cores)
    return cores


def _dram_grid(device: Any) -> ttnn.CoreRangeSet:
    size = device.dram_grid_size()
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(size.x - 1, size.y - 1))})


def _matrix_rows(tensor: ttnn.Tensor) -> int:
    rows = 1
    for index in range(len(tensor.shape) - 1):
        rows *= tensor.shape[index]
    return rows


def _load_prefill_expert_weights(
    state_dict: dict[str, Any],
    *,
    layer_idx: int,
    mesh_device: Any,
    dtype: ttnn.DataType,
    tensor_cache_path: str | Path | None,
) -> ExpertWeights:
    """Upload independent prefill experts directly from the HF source dtype."""
    prefix = _detect_layer_prefix(state_dict, layer_idx)
    gate_up = state_dict[f"{prefix}.experts.gate_up_proj"]
    expert_gate = gate_up[:, :MOE_INTERMEDIATE_SIZE, :].transpose(-2, -1).contiguous().unsqueeze(0)
    expert_up = gate_up[:, MOE_INTERMEDIATE_SIZE:, :].transpose(-2, -1).contiguous().unsqueeze(0)
    expert_down = state_dict[f"{prefix}.experts.down_proj"].transpose(-2, -1).contiguous().unsqueeze(0)
    cache_root = Path(tensor_cache_path) if tensor_cache_path is not None else None

    def upload(name: str, source: Any) -> ttnn.Tensor:
        kwargs = {
            "device": mesh_device,
            "layout": ttnn.TILE_LAYOUT,
            "dtype": dtype,
            "memory_config": ttnn.DRAM_MEMORY_CONFIG,
        }
        mapper = _replicate_mapper(mesh_device)
        if mapper is not None:
            kwargs["mesh_mapper"] = mapper
        if cache_root is not None:
            kwargs["cache_file_name"] = str(cache_root / f"layer_{layer_idx}" / f"prefill_{name}")
        return ttnn.as_tensor(source, **kwargs)

    return ExpertWeights(
        gate_proj=upload("expert_gate", expert_gate),
        up_proj=upload("expert_up", expert_up),
        down_proj=upload("expert_down", expert_down),
        intermediate_size_per_device=MOE_INTERMEDIATE_SIZE,
    )


def _dram_sharded_weight_and_config(
    weight: ttnn.Tensor,
    *,
    device: Any,
    block_w: int | None = None,
) -> tuple[
    ttnn.Tensor,
    ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig,
    ttnn.MemoryConfig,
    ttnn.MemoryConfig,
]:
    """Create a decode-only DRAM-sharded weight and its sub-tile-M config."""
    k, n = weight.shape[-2], weight.shape[-1]
    dram_cores = device.dram_grid_size().x
    memory_config = _create_dram_sharded_mem_config(
        k=k,
        n=n,
        dram_grid=_dram_grid(device),
        dram_cores=dram_cores,
    )
    grid = _dram_shard_core_grid_k_n(k, n)
    input_memory_config = ttnn.create_sharded_memory_config(
        (TILE_SIZE, k // grid.num_cores),
        grid,
        ttnn.ShardStrategy.WIDTH,
        ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    output_memory_config = ttnn.create_sharded_memory_config(
        (TILE_SIZE, math.ceil(n / (TILE_SIZE * grid.num_cores)) * TILE_SIZE),
        grid,
        ttnn.ShardStrategy.WIDTH,
        ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    config = _dram_matmul_config(m=TILE_SIZE, k=k, n=n, num_cores=grid.num_cores)
    if block_w is not None:
        config = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
            in0_block_w=block_w,
            per_core_M=config.per_core_M,
            per_core_N=config.per_core_N,
            fused_activation=config.fused_activation,
        )
    return ttnn.to_memory_config(weight, memory_config), config, input_memory_config, output_memory_config


def _compute_config(
    device: Any,
    *,
    fidelity: ttnn.MathFidelity,
    fp32_dest_acc_en: bool = False,
) -> ttnn.DeviceComputeKernelConfig:
    arch = device.arch() if hasattr(device, "arch") else ttnn.device.GetDefaultDevice().arch()
    return ttnn.init_device_compute_kernel_config(
        arch,
        math_fidelity=fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=fp32_dest_acc_en,
        packer_l1_acc=not fp32_dest_acc_en,
    )


def _rectangular_grid(device: Any, num_cores: int) -> ttnn.CoreCoord:
    grid = device.compute_with_storage_grid_size()
    for height in range(min(grid.y, num_cores), 0, -1):
        if num_cores % height == 0 and num_cores // height <= grid.x:
            return ttnn.CoreCoord(num_cores // height, height)
    raise ValueError(f"cannot place {num_cores} cores on {grid.x}x{grid.y}")


def _width_sharded_memory_config(width: int, grid: ttnn.CoreGrid) -> ttnn.MemoryConfig:
    if width % (TILE_SIZE * grid.num_cores) != 0:
        raise ValueError(f"{width=} must divide exactly over {grid.num_cores} tile-aligned width shards")
    return ttnn.create_sharded_memory_config(
        (TILE_SIZE, width // grid.num_cores),
        grid,
        ttnn.ShardStrategy.WIDTH,
        ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def _width_sharded_linear_program_config(
    *,
    k: int,
    n: int,
    grid: ttnn.CoreGrid,
    in0_block_w: int | None = None,
) -> ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig:
    if k % (TILE_SIZE * grid.num_cores) != 0 or n % (TILE_SIZE * grid.num_cores) != 0:
        raise ValueError(f"1D sharded linear requires exact tiled K/N division, got {k=} {n=} {grid.num_cores=}")
    local_k_tiles = k // TILE_SIZE // grid.num_cores
    in0_block_w = local_k_tiles if in0_block_w is None else in0_block_w
    if local_k_tiles % in0_block_w != 0:
        raise ValueError(f"{in0_block_w=} must divide the local input shard width {local_k_tiles}")
    per_core_n = n // TILE_SIZE // grid.num_cores
    out_subblock_w = next(value for value in (4, 3, 2, 1) if per_core_n % value == 0)
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(grid.x, grid.y),
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=out_subblock_w,
        per_core_M=1,
        per_core_N=per_core_n,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )


def _optimized_sparse_prefill_config(
    device: Any,
    *,
    n: int,
    groups: int,
    requested_per_core_n: int,
    in0_block_w: int,
) -> ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig:
    """Use the Blackhole grid while accounting for sparse group replication."""
    n_tiles = math.ceil(n / TILE_SIZE)
    grid = device.compute_with_storage_grid_size()
    max_cores = (grid.x * grid.y) // groups
    legal_per_core_n = [
        value
        for value in range(max(1, requested_per_core_n), n_tiles + 1)
        if n_tiles % value == 0 and n_tiles // value <= max_cores
    ]
    if not legal_per_core_n:
        raise ValueError(
            f"no sparse prefill config for n_tiles={n_tiles}, groups={groups}, "
            f"requested_per_core_n={requested_per_core_n}"
        )
    per_core_n = legal_per_core_n[0]
    projection_cores = n_tiles // per_core_n
    # sparse_matmul replicates the projection grid for each 32-token sparse
    # group. The program grid must therefore cover every replicated block.
    core_grid = _rectangular_grid(device, projection_cores * groups)
    out_subblock_w = 2 if per_core_n % 2 == 0 else 1
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=core_grid,
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=out_subblock_w,
        out_block_h=1,
        out_block_w=per_core_n,
        per_core_M=1,
        per_core_N=per_core_n,
        fuse_batch=False,
        fused_activation=None,
        mcast_in0=True,
    )


def _optimized_sparse_decode_config(
    device: Any,
    *,
    n: int,
    per_core_n: int,
    in0_block_w: int,
    out_subblock_w: int | None,
) -> ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig:
    """Place sparse decode across the full architecture grid."""
    n_tiles = math.ceil(n / TILE_SIZE)
    if n_tiles % per_core_n != 0:
        raise ValueError(f"{per_core_n=} must divide {n_tiles=} for sparse decode")
    num_cores = n_tiles // per_core_n
    core_grid = _rectangular_grid(device, num_cores)
    if out_subblock_w is None:
        out_subblock_w = next(value for value in (4, 2, 1) if per_core_n % value == 0)
    if out_subblock_w < 1 or per_core_n % out_subblock_w != 0:
        raise ValueError(f"{out_subblock_w=} must divide {per_core_n=} for sparse decode")
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=core_grid,
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=out_subblock_w,
        out_block_h=1,
        out_block_w=per_core_n,
        per_core_M=1,
        per_core_N=per_core_n,
        fuse_batch=False,
        fused_activation=None,
        mcast_in0=True,
    )


class OptimizedDecoder(FunctionalDecoder):
    """Gemma-4 decoder with explicit optimized precision/configuration paths."""

    def __init__(
        self,
        *,
        attention_math_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.HiFi4,
        full_attention_math_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.LoFi,
        mlp_math_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.LoFi,
        expert_gate_math_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.LoFi,
        expert_math_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.LoFi,
        expert_gate_in0_block_w: int = 11,
        expert_down_in0_block_w: int = 11,
        expert_gate_per_core_n: int = 2,
        expert_down_per_core_n: int = 2,
        expert_gate_out_subblock_w: int | None = None,
        expert_down_out_subblock_w: int | None = None,
        expert_decode_input_l1: bool = True,
        prefill_expert_chunk_size: int = TILE_SIZE,
        prefill_expert_per_core_n: int = 2,
        prefill_expert_gate_in0_block_w: int = 44,
        prefill_expert_down_in0_block_w: int = 11,
        prefill_expert_tail_per_core_n: int = 11,
        prefill_expert_tail_in0_block_w: int = 1,
        prefill_routed_active: bool = True,
        dense_decode_dram_sharded: bool = False,
        packed_dense_gate_up: bool = True,
        dram_in0_block_w: int | None = None,
        dram_sharded_roles: tuple[str, ...] = (),
        residual_shard_cores: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        resolved_attention_fidelity = (
            attention_math_fidelity if self.layer_kind.name == "sliding_attention" else full_attention_math_fidelity
        )
        self.attention_compute_config = _compute_config(self.mesh_device, fidelity=resolved_attention_fidelity)
        self.mlp_compute_config = _compute_config(self.mesh_device, fidelity=mlp_math_fidelity)
        self.expert_gate_compute_config = _compute_config(
            self.mesh_device,
            fidelity=expert_gate_math_fidelity,
            fp32_dest_acc_en=expert_gate_math_fidelity == ttnn.MathFidelity.HiFi4,
        )
        self.expert_compute_config = _compute_config(self.mesh_device, fidelity=expert_math_fidelity)
        self.expert_gate_in0_block_w = expert_gate_in0_block_w
        self.expert_down_in0_block_w = expert_down_in0_block_w
        self.expert_gate_per_core_n = expert_gate_per_core_n
        self.expert_down_per_core_n = expert_down_per_core_n
        self.expert_gate_out_subblock_w = expert_gate_out_subblock_w
        self.expert_down_out_subblock_w = expert_down_out_subblock_w
        self.expert_decode_input_l1 = expert_decode_input_l1
        self.prefill_expert_chunk_size = prefill_expert_chunk_size
        self.prefill_expert_per_core_n = prefill_expert_per_core_n
        self.prefill_expert_gate_in0_block_w = prefill_expert_gate_in0_block_w
        self.prefill_expert_down_in0_block_w = prefill_expert_down_in0_block_w
        self.prefill_expert_tail_per_core_n = prefill_expert_tail_per_core_n
        self.prefill_expert_tail_in0_block_w = prefill_expert_tail_in0_block_w
        self.prefill_routed_active = prefill_routed_active
        self.dense_decode_dram_sharded = dense_decode_dram_sharded
        self.packed_dense_gate_up = packed_dense_gate_up
        self.dram_in0_block_w = dram_in0_block_w
        self.dram_sharded_roles = frozenset(dram_sharded_roles)
        self.optimized_path_counters = {
            "prefill_attention": 0,
            "decode_attention": 0,
            "dense_mlp": 0,
            "expert_decode": 0,
            "expert_prefill": 0,
            "residual_chain_decode": 0,
            **{name: 0 for name in _RESIDUAL_BOUNDARY_COUNTERS},
        }
        self._configure_residual_chain(residual_shard_cores)

    def _configure_residual_chain(self, residual_shard_cores: int) -> None:
        geometry = _residual_shard_geometry(residual_shard_cores)
        self.residual_shard_cores = residual_shard_cores
        self.residual_memory_config = None
        self.residual_norm_program_config = None
        self.residual_intermediate_memory_config = None
        self.residual_dense_program_configs = {}
        self.attention_sharded_memory_configs = {}
        self.attention_sharded_program_configs = {}
        if geometry is None:
            return

        grid_x, grid_y, shard_width, norm_block_w, norm_subblock_w = geometry
        available_grid = self.mesh_device.compute_with_storage_grid_size()
        if grid_x > available_grid.x or grid_y > available_grid.y:
            raise ValueError(
                f"R{residual_shard_cores} needs a {grid_x}x{grid_y} worker grid, "
                f"but the device exposes {available_grid.x}x{available_grid.y}"
            )
        residual_grid = ttnn.CoreGrid(x=grid_x, y=grid_y)
        self.residual_memory_config = _width_sharded_memory_config(HIDDEN_SIZE, residual_grid)
        if HIDDEN_SIZE // residual_shard_cores != shard_width:
            raise AssertionError("residual shard geometry does not cover the hidden width exactly")
        self.residual_norm_program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=[grid_x, grid_y],
            subblock_w=norm_subblock_w,
            block_h=1,
            block_w=norm_block_w,
            inplace=False,
        )

        self.residual_intermediate_memory_config = _width_sharded_memory_config(
            MLP_INTERMEDIATE_SIZE,
            residual_grid,
        )
        self.residual_dense_program_configs = {
            "mlp_gate": _width_sharded_linear_program_config(
                k=HIDDEN_SIZE,
                n=MLP_INTERMEDIATE_SIZE,
                grid=residual_grid,
            ),
            "mlp_up": _width_sharded_linear_program_config(
                k=HIDDEN_SIZE,
                n=MLP_INTERMEDIATE_SIZE,
                grid=residual_grid,
            ),
            "mlp_down": _width_sharded_linear_program_config(
                k=MLP_INTERMEDIATE_SIZE,
                n=HIDDEN_SIZE,
                grid=residual_grid,
            ),
        }

        attention_grid = ttnn.CoreGrid(x=8, y=1)
        q_width = self.layer_kind.q_width
        qkv_width = self.layer_kind.qkv_width
        self.attention_sharded_memory_configs = {
            "qkv_input": _width_sharded_memory_config(HIDDEN_SIZE, attention_grid),
            "qkv_output": _width_sharded_memory_config(qkv_width, attention_grid),
            "o_input": _width_sharded_memory_config(q_width, attention_grid),
            "o_output": _width_sharded_memory_config(HIDDEN_SIZE, attention_grid),
        }
        self.attention_sharded_program_configs = {
            "qkv": _width_sharded_linear_program_config(
                k=HIDDEN_SIZE,
                n=qkv_width,
                grid=attention_grid,
                # Eleven K tiles/core with block width eleven over-allocates
                # Blackhole circular buffers by 25,408 bytes. One is the other
                # legal divisor and keeps the same coherent 8-core L1 shard.
                in0_block_w=1,
            ),
            "o_proj": _width_sharded_linear_program_config(
                k=q_width,
                n=HIDDEN_SIZE,
                grid=attention_grid,
            ),
        }

    @classmethod
    def from_state_dict(
        cls,
        state_dict: dict[str, Any],
        *,
        hf_config: Any,
        layer_idx: int,
        mesh_device: Any,
        weight_dtype: ttnn.DataType = ttnn.bfloat16,
        attention_weight_dtype: ttnn.DataType | None = None,
        mlp_weight_dtype: ttnn.DataType = ttnn.bfloat8_b,
        mlp_down_weight_dtype: ttnn.DataType | None = None,
        prefill_expert_weight_dtype: ttnn.DataType = ttnn.bfloat8_b,
        expert_weight_dtype: ttnn.DataType = ttnn.bfloat8_b,
        activation_dtype: ttnn.DataType = ttnn.bfloat16,
        tensor_cache_path: str | Path | None = None,
        attention_math_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.HiFi4,
        full_attention_math_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.LoFi,
        mlp_math_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.LoFi,
        expert_gate_math_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.LoFi,
        expert_math_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.LoFi,
        expert_gate_in0_block_w: int = 11,
        expert_down_in0_block_w: int = 11,
        expert_gate_per_core_n: int = 2,
        expert_down_per_core_n: int = 2,
        expert_gate_out_subblock_w: int | None = None,
        expert_down_out_subblock_w: int | None = None,
        expert_decode_input_l1: bool = True,
        prefill_expert_chunk_size: int = TILE_SIZE,
        prefill_expert_per_core_n: int = 2,
        prefill_expert_gate_in0_block_w: int = 44,
        prefill_expert_down_in0_block_w: int = 11,
        prefill_expert_tail_per_core_n: int = 11,
        prefill_expert_tail_in0_block_w: int = 1,
        prefill_routed_active: bool = True,
        dense_decode_dram_sharded: bool = False,
        packed_dense_gate_up: bool = True,
        dram_in0_block_w: int | None = None,
        dram_sharded_roles: tuple[str, ...] = (),
        residual_shard_cores: int = 0,
        **kwargs: Any,
    ) -> "OptimizedDecoder":
        weight_dtype = _candidate_from_env("GEMMA4_OPT_WEIGHT_DTYPE", weight_dtype, _DTYPES)
        attention_weight_dtype = _candidate_from_env(
            "GEMMA4_OPT_ATTENTION_WEIGHT_DTYPE",
            attention_weight_dtype,
            _DTYPES,
        )
        if attention_weight_dtype is None:
            layer_type = hf_config.layer_types[layer_idx]
            attention_weight_dtype = (
                ttnn.bfloat16 if layer_type == "sliding_attention" else ttnn.bfloat8_b
            )
        mlp_weight_dtype = _candidate_from_env(
            "GEMMA4_OPT_MLP_WEIGHT_DTYPE",
            mlp_weight_dtype,
            _DTYPES,
        )
        mlp_down_weight_dtype = _candidate_from_env(
            "GEMMA4_OPT_MLP_DOWN_WEIGHT_DTYPE",
            mlp_down_weight_dtype,
            _DTYPES,
        )
        if mlp_down_weight_dtype is None:
            mlp_down_weight_dtype = mlp_weight_dtype
        expert_weight_dtype = _candidate_from_env(
            "GEMMA4_OPT_EXPERT_WEIGHT_DTYPE",
            expert_weight_dtype,
            _DTYPES,
        )
        prefill_expert_weight_dtype = _candidate_from_env(
            "GEMMA4_OPT_PREFILL_EXPERT_WEIGHT_DTYPE",
            prefill_expert_weight_dtype,
            _DTYPES,
        )
        attention_math_fidelity = _candidate_from_env(
            "GEMMA4_OPT_ATTENTION_FIDELITY",
            attention_math_fidelity,
            _FIDELITIES,
        )
        full_attention_math_fidelity = _candidate_from_env(
            "GEMMA4_OPT_FULL_ATTENTION_FIDELITY",
            full_attention_math_fidelity,
            _FIDELITIES,
        )
        mlp_math_fidelity = _candidate_from_env(
            "GEMMA4_OPT_MLP_FIDELITY",
            mlp_math_fidelity,
            _FIDELITIES,
        )
        expert_math_fidelity = _candidate_from_env(
            "GEMMA4_OPT_EXPERT_FIDELITY",
            expert_math_fidelity,
            _FIDELITIES,
        )
        expert_gate_math_fidelity = _candidate_from_env(
            "GEMMA4_OPT_EXPERT_GATE_FIDELITY",
            expert_gate_math_fidelity,
            _FIDELITIES,
        )
        expert_gate_in0_block_w = int(os.getenv("GEMMA4_OPT_EXPERT_GATE_BLOCK_W", expert_gate_in0_block_w))
        expert_down_in0_block_w = int(os.getenv("GEMMA4_OPT_EXPERT_DOWN_BLOCK_W", expert_down_in0_block_w))
        expert_gate_per_core_n = int(os.getenv("GEMMA4_OPT_EXPERT_GATE_PER_CORE_N", expert_gate_per_core_n))
        expert_down_per_core_n = int(os.getenv("GEMMA4_OPT_EXPERT_DOWN_PER_CORE_N", expert_down_per_core_n))
        env_gate_subblock = os.getenv("GEMMA4_OPT_EXPERT_GATE_OUT_SUBBLOCK_W")
        if env_gate_subblock is not None:
            expert_gate_out_subblock_w = int(env_gate_subblock)
        env_down_subblock = os.getenv("GEMMA4_OPT_EXPERT_DOWN_OUT_SUBBLOCK_W")
        if env_down_subblock is not None:
            expert_down_out_subblock_w = int(env_down_subblock)
        expert_decode_input_l1 = _bool_from_env("GEMMA4_OPT_EXPERT_DECODE_INPUT_L1", expert_decode_input_l1)
        prefill_expert_chunk_size = int(os.getenv("GEMMA4_OPT_PREFILL_EXPERT_CHUNK_SIZE", prefill_expert_chunk_size))
        prefill_expert_per_core_n = int(os.getenv("GEMMA4_OPT_PREFILL_EXPERT_PER_CORE_N", prefill_expert_per_core_n))
        prefill_expert_gate_in0_block_w = int(
            os.getenv("GEMMA4_OPT_PREFILL_EXPERT_GATE_BLOCK_W", prefill_expert_gate_in0_block_w)
        )
        prefill_expert_down_in0_block_w = int(
            os.getenv("GEMMA4_OPT_PREFILL_EXPERT_DOWN_BLOCK_W", prefill_expert_down_in0_block_w)
        )
        prefill_expert_tail_per_core_n = int(
            os.getenv("GEMMA4_OPT_PREFILL_EXPERT_TAIL_PER_CORE_N", prefill_expert_tail_per_core_n)
        )
        prefill_expert_tail_in0_block_w = int(
            os.getenv("GEMMA4_OPT_PREFILL_EXPERT_TAIL_BLOCK_W", prefill_expert_tail_in0_block_w)
        )
        prefill_routed_active = _bool_from_env("GEMMA4_OPT_PREFILL_ROUTED_ACTIVE", prefill_routed_active)
        dense_decode_dram_sharded = _bool_from_env("GEMMA4_OPT_DENSE_DECODE_DRAM_SHARDED", dense_decode_dram_sharded)
        env_roles = os.getenv("GEMMA4_OPT_DRAM_SHARDED_ROLES")
        auto_selected_mlp_dram = False
        if env_roles is not None:
            dram_sharded_roles = tuple(role.strip() for role in env_roles.split(",") if role.strip())
        elif not dram_sharded_roles:
            # The advisor challenger measured O projection DRAM sharding as a
            # strict batch-1 win for sliding attention too. QKV and every
            # other losing role remain available through the env-gated knob.
            dram_sharded_roles = ("o_proj", "packed_mlp_gate_up", "mlp_down")
            auto_selected_mlp_dram = True
        invalid_roles = set(dram_sharded_roles) - _DRAM_SHARDED_ROLES
        if invalid_roles:
            raise ValueError(f"invalid DRAM-sharded roles {sorted(invalid_roles)}; choose from {sorted(_DRAM_SHARDED_ROLES)}")
        if dense_decode_dram_sharded:
            dram_sharded_roles = tuple(sorted(_DRAM_SHARDED_ROLES))
        packed_dense_gate_up = _bool_from_env("GEMMA4_OPT_PACKED_DENSE_GATE_UP", packed_dense_gate_up)
        residual_shard_cores = _residual_shard_cores_from_env(residual_shard_cores)
        env_dram_block_w = os.getenv("GEMMA4_OPT_DRAM_BLOCK_W")
        if env_dram_block_w is not None:
            dram_in0_block_w = int(env_dram_block_w)
        decoder = super().from_state_dict(
            state_dict,
            hf_config=hf_config,
            layer_idx=layer_idx,
            mesh_device=mesh_device,
            weight_dtype=weight_dtype,
            expert_weight_dtype=expert_weight_dtype,
            activation_dtype=activation_dtype,
            tensor_cache_path=tensor_cache_path,
            attention_math_fidelity=attention_math_fidelity,
            mlp_math_fidelity=mlp_math_fidelity,
            expert_gate_math_fidelity=expert_gate_math_fidelity,
            expert_math_fidelity=expert_math_fidelity,
            expert_gate_in0_block_w=expert_gate_in0_block_w,
            expert_down_in0_block_w=expert_down_in0_block_w,
            expert_gate_per_core_n=expert_gate_per_core_n,
            expert_down_per_core_n=expert_down_per_core_n,
            expert_gate_out_subblock_w=expert_gate_out_subblock_w,
            expert_down_out_subblock_w=expert_down_out_subblock_w,
            expert_decode_input_l1=expert_decode_input_l1,
            prefill_expert_chunk_size=prefill_expert_chunk_size,
            prefill_expert_per_core_n=prefill_expert_per_core_n,
            prefill_expert_gate_in0_block_w=prefill_expert_gate_in0_block_w,
            prefill_expert_down_in0_block_w=prefill_expert_down_in0_block_w,
            prefill_routed_active=prefill_routed_active,
            dense_decode_dram_sharded=dense_decode_dram_sharded,
            packed_dense_gate_up=packed_dense_gate_up,
            dram_in0_block_w=dram_in0_block_w,
            dram_sharded_roles=dram_sharded_roles,
            residual_shard_cores=residual_shard_cores,
            **kwargs,
        )
        resolved_attention_fidelity = (
            attention_math_fidelity if decoder.layer_kind.name == "sliding_attention" else full_attention_math_fidelity
        )
        decoder.attention_compute_config = _compute_config(mesh_device, fidelity=resolved_attention_fidelity)
        decoder.mlp_compute_config = _compute_config(mesh_device, fidelity=mlp_math_fidelity)
        decoder.expert_gate_compute_config = _compute_config(
            mesh_device,
            fidelity=expert_gate_math_fidelity,
            fp32_dest_acc_en=expert_gate_math_fidelity == ttnn.MathFidelity.HiFi4,
        )
        decoder.expert_compute_config = _compute_config(mesh_device, fidelity=expert_math_fidelity)
        decoder.expert_gate_in0_block_w = expert_gate_in0_block_w
        decoder.expert_down_in0_block_w = expert_down_in0_block_w
        decoder.expert_gate_per_core_n = expert_gate_per_core_n
        decoder.expert_down_per_core_n = expert_down_per_core_n
        decoder.expert_gate_out_subblock_w = expert_gate_out_subblock_w
        decoder.expert_down_out_subblock_w = expert_down_out_subblock_w
        decoder.expert_decode_input_l1 = expert_decode_input_l1
        decoder.prefill_expert_chunk_size = prefill_expert_chunk_size
        decoder.prefill_expert_per_core_n = prefill_expert_per_core_n
        decoder.prefill_expert_gate_in0_block_w = prefill_expert_gate_in0_block_w
        decoder.prefill_expert_down_in0_block_w = prefill_expert_down_in0_block_w
        decoder.prefill_expert_tail_per_core_n = prefill_expert_tail_per_core_n
        decoder.prefill_expert_tail_in0_block_w = prefill_expert_tail_in0_block_w
        decoder.prefill_routed_active = prefill_routed_active
        decoder.dense_decode_dram_sharded = dense_decode_dram_sharded
        decoder.packed_dense_gate_up = packed_dense_gate_up
        decoder.dram_in0_block_w = dram_in0_block_w
        decoder.dram_sharded_roles = frozenset(dram_sharded_roles)
        decoder._configure_residual_chain(residual_shard_cores)
        decoder.weights = replace(
            decoder.weights,
            qkv=ttnn.typecast(
                decoder.weights.qkv,
                attention_weight_dtype,
                memory_config=decoder.weights.qkv.memory_config(),
            ),
            o_proj=ttnn.typecast(
                decoder.weights.o_proj,
                attention_weight_dtype,
                memory_config=decoder.weights.o_proj.memory_config(),
            ),
            mlp_gate=ttnn.typecast(
                decoder.weights.mlp_gate,
                mlp_weight_dtype,
                memory_config=decoder.weights.mlp_gate.memory_config(),
            ),
            mlp_up=ttnn.typecast(
                decoder.weights.mlp_up,
                mlp_weight_dtype,
                memory_config=decoder.weights.mlp_up.memory_config(),
            ),
            mlp_down=ttnn.typecast(
                decoder.weights.mlp_down,
                mlp_down_weight_dtype,
                memory_config=decoder.weights.mlp_down.memory_config(),
            ),
        )
        decoder.expert_weights = _load_prefill_expert_weights(
            state_dict,
            layer_idx=layer_idx,
            mesh_device=mesh_device,
            dtype=prefill_expert_weight_dtype,
            tensor_cache_path=tensor_cache_path,
        )
        decoder.packed_mlp_gate_up = None
        if packed_dense_gate_up:
            decoder.packed_mlp_gate_up = ttnn.concat(
                [decoder.weights.mlp_gate, decoder.weights.mlp_up],
                dim=-1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        decoder.decode_dram_weights = {}
        decoder.decode_dram_configs = {}
        decoder.decode_dram_input_configs = {}
        decoder.decode_dram_output_configs = {}
        if dram_sharded_roles:
            candidates = {
                "qkv": decoder.weights.qkv,
                "o_proj": decoder.weights.o_proj,
                "mlp_gate": decoder.weights.mlp_gate,
                "mlp_up": decoder.weights.mlp_up,
                "mlp_down": decoder.weights.mlp_down,
            }
            if decoder.packed_mlp_gate_up is not None:
                candidates["packed_mlp_gate_up"] = decoder.packed_mlp_gate_up
            for name, weight in candidates.items():
                if name not in dram_sharded_roles:
                    continue
                role_block_w = os.getenv(f"GEMMA4_OPT_DRAM_BLOCK_W_{name.upper()}")
                if (
                    role_block_w is None
                    and dram_in0_block_w is None
                    and auto_selected_mlp_dram
                    and name in {"packed_mlp_gate_up", "mlp_down"}
                ):
                    role_block_w = "11"
                elif role_block_w is None and dram_in0_block_w is None and auto_selected_mlp_dram and name == "o_proj":
                    # The shipped advisor-challenger winner executed this
                    # exact role-specific geometry.
                    role_block_w = "1"
                sharded_weight, config, input_config, output_config = _dram_sharded_weight_and_config(
                    weight,
                    device=mesh_device,
                    block_w=int(role_block_w) if role_block_w is not None else dram_in0_block_w,
                )
                decoder.decode_dram_weights[name] = sharded_weight
                decoder.decode_dram_configs[name] = config
                decoder.decode_dram_input_configs[name] = input_config
                decoder.decode_dram_output_configs[name] = output_config
        return decoder

    def _linear(
        self,
        x: ttnn.Tensor,
        weight_name: str,
        *,
        compute_kernel_config: ttnn.DeviceComputeKernelConfig,
    ) -> ttnn.Tensor:
        decode_candidate = weight_name in self.decode_dram_weights and _matrix_rows(x) <= TILE_SIZE
        weight = self.decode_dram_weights[weight_name] if decode_candidate else getattr(self.weights, weight_name)
        kwargs = {}
        if decode_candidate:
            kwargs["program_config"] = self.decode_dram_configs[weight_name]
            x = ttnn.to_memory_config(x, self.decode_dram_input_configs[weight_name], dtype=x.dtype)
            kwargs["memory_config"] = self.decode_dram_output_configs[weight_name]
        result = ttnn.linear(
            x,
            weight,
            dtype=self.activation_dtype,
            memory_config=kwargs.pop("memory_config", ttnn.DRAM_MEMORY_CONFIG),
            compute_kernel_config=compute_kernel_config,
            **kwargs,
        )
        if decode_candidate:
            result = ttnn.sharded_to_interleaved(result, ttnn.DRAM_MEMORY_CONFIG)
        return result

    def _tracked_to_memory_config(
        self,
        x: ttnn.Tensor,
        memory_config: ttnn.MemoryConfig,
        counter: str,
    ) -> ttnn.Tensor:
        self.optimized_path_counters[counter] += 1
        return ttnn.to_memory_config(x, memory_config, dtype=x.dtype)

    def _tracked_sharded_to_interleaved(
        self,
        x: ttnn.Tensor,
        memory_config: ttnn.MemoryConfig,
        counter: str,
    ) -> ttnn.Tensor:
        self.optimized_path_counters[counter] += 1
        return ttnn.sharded_to_interleaved(x, memory_config)

    def _residual_rms_norm(
        self,
        x: ttnn.Tensor,
        weight: ttnn.Tensor | None,
    ) -> ttnn.Tensor:
        if self.residual_memory_config is None or self.residual_norm_program_config is None:
            raise RuntimeError("the sharded residual RMS path requires R11 or R22 configuration")
        return ttnn.rms_norm(
            x,
            epsilon=self.eps,
            weight=weight,
            program_config=self.residual_norm_program_config,
            compute_kernel_config=self.correctness_compute_config,
            memory_config=self.residual_memory_config,
        )

    def _router_weights_sharded(self, residual: ttnn.Tensor) -> ttnn.Tensor:
        """Normalize on the residual grid, then cross once into router FP32."""
        tokens = residual.shape[-2]
        router_in = self._residual_rms_norm(residual, None)
        router_in = self._tracked_sharded_to_interleaved(
            router_in,
            ttnn.DRAM_MEMORY_CONFIG,
            "router_input",
        )
        router_in = ttnn.mul(router_in, self.weights.router_scale, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        router_in = ttnn.mul(router_in, self.router_hidden_scale, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        router_in = ttnn.reshape(router_in, [tokens, HIDDEN_SIZE])
        router_in = ttnn.typecast(router_in, ttnn.float32)
        logits = ttnn.linear(
            router_in,
            self.weights.router_proj,
            dtype=ttnn.float32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        logits = ttnn.typecast(logits, ttnn.bfloat16)
        top_values, top_indices = ttnn.topk(logits, k=TOP_K_EXPERTS, dim=-1, sorted=True)
        top_values = ttnn.softmax(
            top_values,
            dim=-1,
            numeric_stable=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        routing = ttnn.scatter(ttnn.zeros_like(logits), dim=-1, index=top_indices, src=top_values)
        routing = ttnn.mul(routing, self.weights.router_per_expert_scale, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        routing = ttnn.typecast(routing, ttnn.bfloat16)
        return ttnn.reshape(routing, [1, 1, tokens, NUM_EXPERTS])

    def decode_forward(
        self,
        hidden_states: ttnn.Tensor,
        *,
        position_cos: ttnn.Tensor,
        position_sin: ttnn.Tensor,
        current_pos: ttnn.Tensor,
        page_table: ttnn.Tensor,
        kv_cache: tuple[ttnn.Tensor, ttnn.Tensor],
        cache_position_modulo: int | None = None,
    ) -> ttnn.Tensor:
        """Run the env-gated coherent R11/R22 hidden residual candidate."""
        if self.residual_shard_cores == 0:
            return super().decode_forward(
                hidden_states,
                position_cos=position_cos,
                position_sin=position_sin,
                current_pos=current_pos,
                page_table=page_table,
                kv_cache=kv_cache,
                cache_position_modulo=cache_position_modulo,
            )
        if hidden_states.shape[-2] < 1 or hidden_states.shape[-2] > TILE_SIZE:
            raise ValueError(f"sharded decode requires logical batch in [1, {TILE_SIZE}]")
        if hidden_states.shape[-1] != HIDDEN_SIZE:
            raise ValueError(f"sharded decode requires hidden width {HIDDEN_SIZE}")

        self.optimized_path_counters["residual_chain_decode"] += 1
        hidden_states = self._tracked_to_memory_config(
            hidden_states,
            self.residual_memory_config,
            "residual_entry",
        )

        residual = hidden_states
        attn_in = self._residual_rms_norm(hidden_states, self.weights.input_ln)
        attn_out = self._attention_decode(
            attn_in,
            position_cos=position_cos,
            position_sin=position_sin,
            current_pos=current_pos,
            page_table=page_table,
            kv_cache=kv_cache,
            cache_position_modulo=cache_position_modulo,
        )
        attn_out = self._residual_rms_norm(attn_out, self.weights.post_attn_ln)
        hidden_states = ttnn.add(residual, attn_out, memory_config=self.residual_memory_config)

        residual = hidden_states
        mlp_in = self._residual_rms_norm(hidden_states, self.weights.pre_ff_ln)
        mlp_out = self._dense_mlp(mlp_in)
        hidden_1 = self._residual_rms_norm(mlp_out, self.weights.post_ff_ln_1)

        router_weights = self._router_weights_sharded(residual)
        moe_in = self._residual_rms_norm(residual, self.weights.pre_ff_ln_2)
        moe_in = self._tracked_sharded_to_interleaved(
            moe_in,
            ttnn.L1_MEMORY_CONFIG,
            "expert_input",
        )
        hidden_2 = self._moe_decode(moe_in, router_weights)
        hidden_2 = self._tracked_to_memory_config(
            hidden_2,
            self.residual_memory_config,
            "expert_output",
        )
        hidden_2 = self._residual_rms_norm(hidden_2, self.weights.post_ff_ln_2)

        hidden_states = ttnn.add(hidden_1, hidden_2, memory_config=self.residual_memory_config)
        hidden_states = self._residual_rms_norm(hidden_states, self.weights.post_ff_ln)
        hidden_states = ttnn.add(residual, hidden_states, memory_config=self.residual_memory_config)
        hidden_states = ttnn.mul(
            hidden_states,
            self.weights.layer_scalar,
            memory_config=self.residual_memory_config,
        )
        return self._tracked_sharded_to_interleaved(
            hidden_states,
            ttnn.DRAM_MEMORY_CONFIG,
            "residual_exit",
        )

    def _attention_prefill(
        self,
        x: ttnn.Tensor,
        *,
        position_cos: ttnn.Tensor,
        position_sin: ttnn.Tensor,
        page_table: ttnn.Tensor,
        chunk_page_table: ttnn.Tensor | None,
        kv_cache: tuple[ttnn.Tensor, ttnn.Tensor],
        user_id: int,
        cache_position_modulo: int | None,
        logical_seq_len: int,
    ) -> ttnn.Tensor:
        self.optimized_path_counters["prefill_attention"] += 1
        kind = self.layer_kind
        seq_len = x.shape[-2]
        xqkv = self._linear(x, "qkv", compute_kernel_config=self.attention_compute_config)
        q_heads, k_heads, v_heads = ttnn.experimental.nlp_create_qkv_heads(
            xqkv,
            num_heads=NUM_Q_HEADS,
            num_kv_heads=kind.num_kv_heads,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        q_heads = self._rms_norm(q_heads, self.weights.q_norm)
        k_heads = self._rms_norm(k_heads, self.weights.k_norm)
        v_heads = self._rms_norm(v_heads, None)
        q_heads = ttnn.experimental.rotary_embedding_hf(q_heads, position_cos, position_sin, is_decode_mode=False)
        k_heads = ttnn.experimental.rotary_embedding_hf(k_heads, position_cos, position_sin, is_decode_mode=False)

        key_cache, value_cache = kv_cache
        fill_table = chunk_page_table if chunk_page_table is not None else page_table
        self._fill_prefill_cache(
            key_cache,
            value_cache,
            k_heads,
            v_heads,
            fill_table,
            user_id=user_id,
            logical_seq_len=logical_seq_len,
            cache_position_modulo=cache_position_modulo,
            fill_kwargs=self._cache_view_kwargs(prefill=True),
        )

        from models.autoports.google_gemma_4_26b_a4b_it.tt.functional_decoder import _prefill_attention_path

        attention_path = _prefill_attention_path(
            seq_len,
            is_sliding=kind.sliding_window is not None,
            has_paged_cache=fill_table is not None,
        )
        if attention_path == "sliding_chunked":
            attn_out = self._sliding_chunked_prefill_attention(q_heads, k_heads, v_heads)
        elif attention_path == "full_chunked":
            attn_out = self._full_chunked_prefill_attention(
                q_heads,
                key_cache,
                value_cache,
                fill_table,
                user_id=user_id,
            )
        else:
            attn_out = ttnn.transformer.scaled_dot_product_attention(
                q_heads,
                k_heads,
                v_heads,
                is_causal=True,
                sliding_window_size=kind.sliding_window,
                scale=1.0,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        attn_out = ttnn.reshape(attn_out, [1, NUM_Q_HEADS, seq_len, kind.head_dim])
        attn_out = ttnn.experimental.nlp_concat_heads(attn_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return self._linear(attn_out, "o_proj", compute_kernel_config=self.attention_compute_config)

    def _attention_decode(
        self,
        x: ttnn.Tensor,
        *,
        position_cos: ttnn.Tensor,
        position_sin: ttnn.Tensor,
        current_pos: ttnn.Tensor,
        page_table: ttnn.Tensor,
        kv_cache: tuple[ttnn.Tensor, ttnn.Tensor],
        cache_position_modulo: int | None,
    ) -> ttnn.Tensor:
        self.optimized_path_counters["decode_attention"] += 1
        kind = self.layer_kind
        batch = x.shape[-2]
        sharded_residual = self.residual_shard_cores != 0
        if sharded_residual:
            x = self._tracked_to_memory_config(
                x,
                self.attention_sharded_memory_configs["qkv_input"],
                "attention_qkv_input",
            )
            xqkv = ttnn.linear(
                x,
                self.weights.qkv,
                dtype=self.activation_dtype,
                program_config=self.attention_sharded_program_configs["qkv"],
                memory_config=self.attention_sharded_memory_configs["qkv_output"],
                # Sharded QKV changes the reduction partition. Preserve the
                # functional real-weight PCC bar with high accumulation while
                # the coherent layout candidate is evaluated.
                compute_kernel_config=self.correctness_compute_config,
            )
        else:
            xqkv = self._linear(x, "qkv", compute_kernel_config=self.attention_compute_config)
        head_mem_config = _make_decode_height_sharded_memory_config(self.mesh_device, batch, kind.head_dim)
        q_heads, k_heads, v_heads = ttnn.experimental.nlp_create_qkv_heads_decode(
            xqkv,
            num_heads=NUM_Q_HEADS,
            num_kv_heads=kind.num_kv_heads,
            memory_config=head_mem_config,
        )
        # The head split explicitly produces this layout. During shard
        # analysis the optimizer owns tensor layout state, so use the
        # already-declared phase config instead of querying the traced tensor.
        q_mem_config = k_mem_config = v_mem_config = head_mem_config
        q_heads = ttnn.to_memory_config(q_heads, ttnn.L1_MEMORY_CONFIG, dtype=q_heads.dtype)
        k_heads = ttnn.to_memory_config(k_heads, ttnn.L1_MEMORY_CONFIG, dtype=k_heads.dtype)
        v_heads = ttnn.to_memory_config(v_heads, ttnn.L1_MEMORY_CONFIG, dtype=v_heads.dtype)
        q_heads = self._rms_norm(q_heads, self.weights.q_norm)
        k_heads = self._rms_norm(k_heads, self.weights.k_norm)
        v_heads = self._rms_norm(v_heads, None)
        if kind.name == "full_attention":
            q_heads = ttnn.transpose(q_heads, 1, 2)
            k_heads = ttnn.transpose(k_heads, 1, 2)
            q_heads = ttnn.experimental.rotary_embedding_hf(q_heads, position_cos, position_sin, is_decode_mode=False)
            k_heads = ttnn.experimental.rotary_embedding_hf(k_heads, position_cos, position_sin, is_decode_mode=False)
            q_heads = ttnn.transpose(q_heads, 1, 2)
            k_heads = ttnn.transpose(k_heads, 1, 2)
            q_heads = ttnn.to_memory_config(q_heads, q_mem_config, dtype=q_heads.dtype)
            k_heads = ttnn.to_memory_config(k_heads, k_mem_config, dtype=k_heads.dtype)
            v_heads = ttnn.to_memory_config(v_heads, v_mem_config, dtype=v_heads.dtype)
        else:
            q_heads = ttnn.to_memory_config(q_heads, q_mem_config, dtype=q_heads.dtype)
            k_heads = ttnn.to_memory_config(k_heads, k_mem_config, dtype=k_heads.dtype)
            v_heads = ttnn.to_memory_config(v_heads, v_mem_config, dtype=v_heads.dtype)
            rope_mem_config = _make_decode_rope_memory_config(self.mesh_device, batch, kind.head_dim)
            position_cos = ttnn.interleaved_to_sharded(position_cos, rope_mem_config)
            position_sin = ttnn.interleaved_to_sharded(position_sin, rope_mem_config)
            q_heads = ttnn.experimental.rotary_embedding_hf(q_heads, position_cos, position_sin, is_decode_mode=True)
            k_heads = ttnn.experimental.rotary_embedding_hf(k_heads, position_cos, position_sin, is_decode_mode=True)

        key_cache, value_cache = kv_cache
        update_kwargs = self._cache_view_kwargs(prefill=False)
        if cache_position_modulo is not None:
            update_kwargs["cache_position_modulo"] = cache_position_modulo
        ttnn.experimental.paged_update_cache(
            key_cache,
            k_heads,
            update_idxs_tensor=current_pos,
            page_table=page_table,
            **update_kwargs,
        )
        ttnn.experimental.paged_update_cache(
            value_cache,
            v_heads,
            update_idxs_tensor=current_pos,
            page_table=page_table,
            **update_kwargs,
        )
        attn_out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q_heads,
            key_cache,
            value_cache,
            page_table_tensor=page_table,
            cur_pos_tensor=current_pos,
            scale=1.0,
            sliding_window_size=kind.sliding_window,
            program_config=self.sdpa_program_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            **self._cache_view_kwargs(prefill=False),
        )
        concat_mem_config = _make_decode_height_sharded_memory_config(self.mesh_device, batch, kind.head_dim)
        if sharded_residual:
            attn_out = self._tracked_to_memory_config(
                attn_out,
                concat_mem_config,
                "attention_sdpa_output",
            )
        else:
            attn_out = ttnn.to_memory_config(attn_out, concat_mem_config, dtype=attn_out.dtype)
        attn_out = ttnn.experimental.nlp_concat_heads_decode(attn_out, num_heads=NUM_Q_HEADS)
        if sharded_residual:
            attn_out = self._tracked_to_memory_config(
                attn_out,
                self.attention_sharded_memory_configs["o_input"],
                "attention_o_input",
            )
            attn_out = ttnn.linear(
                attn_out,
                self.weights.o_proj,
                dtype=self.activation_dtype,
                program_config=self.attention_sharded_program_configs["o_proj"],
                memory_config=self.attention_sharded_memory_configs["o_output"],
                compute_kernel_config=self.correctness_compute_config,
            )
        else:
            attn_out = ttnn.sharded_to_interleaved(attn_out, ttnn.DRAM_MEMORY_CONFIG)
            attn_out = self._linear(attn_out, "o_proj", compute_kernel_config=self.attention_compute_config)
        if attn_out.shape[-2] != batch:
            attn_out = ttnn.slice(
                attn_out,
                starts=[0, 0, 0, 0],
                ends=[1, 1, batch, HIDDEN_SIZE],
                steps=[1, 1, 1, 1],
                memory_config=(
                    self.attention_sharded_memory_configs["o_output"]
                    if sharded_residual
                    else ttnn.DRAM_MEMORY_CONFIG
                ),
            )
        if sharded_residual:
            attn_out = self._tracked_to_memory_config(
                attn_out,
                self.residual_memory_config,
                "attention_output",
            )
        return attn_out

    def _dense_mlp(self, x: ttnn.Tensor) -> ttnn.Tensor:
        self.optimized_path_counters["dense_mlp"] += 1
        if self.residual_shard_cores != 0 and x.is_sharded():
            return self._dense_mlp_residual_sharded(x)
        if self.packed_dense_gate_up:
            decode_candidate = "packed_mlp_gate_up" in self.decode_dram_weights and _matrix_rows(x) <= TILE_SIZE
            packed_weight = (
                self.decode_dram_weights["packed_mlp_gate_up"] if decode_candidate else self.packed_mlp_gate_up
            )
            kwargs = {}
            if decode_candidate:
                kwargs["program_config"] = self.decode_dram_configs["packed_mlp_gate_up"]
                x = ttnn.to_memory_config(
                    x,
                    self.decode_dram_input_configs["packed_mlp_gate_up"],
                    dtype=x.dtype,
                )
                kwargs["memory_config"] = self.decode_dram_output_configs["packed_mlp_gate_up"]
            gate_up = ttnn.linear(
                x,
                packed_weight,
                dtype=self.activation_dtype,
                memory_config=kwargs.pop("memory_config", ttnn.DRAM_MEMORY_CONFIG),
                compute_kernel_config=self.mlp_compute_config,
                **kwargs,
            )
            if decode_candidate:
                gate_up = ttnn.sharded_to_interleaved(gate_up, ttnn.DRAM_MEMORY_CONFIG)
            width = self.weights.mlp_gate.shape[-1]
            ends = [gate_up.shape[index] for index in range(len(gate_up.shape))]
            gate_ends = list(ends)
            gate_ends[-1] = width
            up_starts = [0] * len(ends)
            up_starts[-1] = width
            gate = ttnn.slice(
                gate_up,
                starts=[0] * len(ends),
                ends=gate_ends,
                steps=[1] * len(ends),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            up = ttnn.slice(
                gate_up,
                starts=up_starts,
                ends=ends,
                steps=[1] * len(ends),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            gate = self._linear(x, "mlp_gate", compute_kernel_config=self.mlp_compute_config)
            up = self._linear(x, "mlp_up", compute_kernel_config=self.mlp_compute_config)
        gate = ttnn.gelu(gate, fast_and_approximate_mode=True, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        hidden = ttnn.mul(gate, up, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return self._linear(hidden, "mlp_down", compute_kernel_config=self.mlp_compute_config)

    def _dense_mlp_residual_sharded(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """Keep the split gate/up/down family coherent with the residual grid."""
        gate = ttnn.linear(
            x,
            self.weights.mlp_gate,
            dtype=self.activation_dtype,
            program_config=self.residual_dense_program_configs["mlp_gate"],
            memory_config=self.residual_intermediate_memory_config,
            compute_kernel_config=self.correctness_compute_config,
        )
        up = ttnn.linear(
            x,
            self.weights.mlp_up,
            dtype=self.activation_dtype,
            program_config=self.residual_dense_program_configs["mlp_up"],
            memory_config=self.residual_intermediate_memory_config,
            compute_kernel_config=self.correctness_compute_config,
        )
        gate = ttnn.gelu(
            gate,
            fast_and_approximate_mode=True,
            memory_config=self.residual_intermediate_memory_config,
        )
        hidden = ttnn.mul(gate, up, memory_config=self.residual_intermediate_memory_config)
        return ttnn.linear(
            hidden,
            self.weights.mlp_down,
            dtype=self.activation_dtype,
            program_config=self.residual_dense_program_configs["mlp_down"],
            memory_config=self.residual_memory_config,
            compute_kernel_config=self.correctness_compute_config,
        )

    def _fill_prefill_cache(
        self,
        key_cache: ttnn.Tensor,
        value_cache: ttnn.Tensor,
        k_heads: ttnn.Tensor,
        v_heads: ttnn.Tensor,
        page_table: ttnn.Tensor,
        *,
        user_id: int,
        logical_seq_len: int,
        cache_position_modulo: int | None,
        fill_kwargs: dict[str, int],
    ) -> None:
        """Allow low-precision cache trials without changing update semantics.

        ``paged_fill_cache`` requires source and cache dtypes to match, whereas
        ``paged_update_cache`` intentionally consumes BF16 and repacks into a
        low-precision cache. Only the tile-aligned bulk-fill source is cast.
        Non-aligned tails retain the functional BF16 update path.
        """
        if logical_seq_len % TILE_SIZE == 0 and k_heads.dtype != key_cache.dtype:
            k_fill = ttnn.typecast(k_heads, key_cache.dtype)
            v_fill = ttnn.typecast(v_heads, value_cache.dtype)
            modulo_kwargs = (
                {"cache_position_modulo": cache_position_modulo} if cache_position_modulo is not None else {}
            )
            ttnn.experimental.paged_fill_cache(
                key_cache, k_fill, page_table, batch_idx=user_id, **fill_kwargs, **modulo_kwargs
            )
            ttnn.experimental.paged_fill_cache(
                value_cache, v_fill, page_table, batch_idx=user_id, **fill_kwargs, **modulo_kwargs
            )
            k_fill.deallocate(True)
            v_fill.deallocate(True)
            return
        super()._fill_prefill_cache(
            key_cache,
            value_cache,
            k_heads,
            v_heads,
            page_table,
            user_id=user_id,
            logical_seq_len=logical_seq_len,
            cache_position_modulo=cache_position_modulo,
            fill_kwargs=fill_kwargs,
        )

    def _moe_prefill(self, hidden_states: ttnn.Tensor, routing_weights: ttnn.Tensor) -> ttnn.Tensor:
        self.optimized_path_counters["expert_prefill"] += 1
        seq_len = hidden_states.shape[-2]
        if seq_len <= PREFILL_MOE_CHUNK_SIZE:
            return self._moe_prefill_chunk(hidden_states, routing_weights)

        chunks = []
        for start in range(0, seq_len, PREFILL_MOE_CHUNK_SIZE):
            end = min(start + PREFILL_MOE_CHUNK_SIZE, seq_len)
            hidden_chunk = ttnn.slice(
                hidden_states,
                starts=[0, 0, start, 0],
                ends=[1, 1, end, HIDDEN_SIZE],
                steps=[1, 1, 1, 1],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            routing_chunk = ttnn.slice(
                routing_weights,
                starts=[0, 0, start, 0],
                ends=[1, 1, end, NUM_EXPERTS],
                steps=[1, 1, 1, 1],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            chunks.append(self._moe_prefill_chunk(hidden_chunk, routing_chunk))
        return ttnn.concat(chunks, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def _moe_prefill_chunk(self, hidden_states: ttnn.Tensor, routing_weights: ttnn.Tensor) -> ttnn.Tensor:
        chunk_size = self.prefill_expert_chunk_size
        seq_len = hidden_states.shape[2]
        if seq_len % TILE_SIZE != 0 or chunk_size % TILE_SIZE != 0:
            raise ValueError(f"physical prefill and expert chunk must be tile aligned, got {seq_len=} {chunk_size=}")
        if seq_len > chunk_size:
            hidden_chunks = ttnn.split(hidden_states, chunk_size, dim=2)
            routing_chunks = ttnn.split(routing_weights, chunk_size, dim=2)
        else:
            hidden_chunks = [hidden_states]
            routing_chunks = [routing_weights]

        results = []
        output_tile = ttnn.Tile([TILE_SIZE, TILE_SIZE])
        for hidden_chunk, routing_chunk in zip(hidden_chunks, routing_chunks):
            physical_chunk = hidden_chunk.shape[2]
            groups = physical_chunk // TILE_SIZE
            tail_geometry = physical_chunk < self.prefill_expert_chunk_size
            per_core_n = self.prefill_expert_tail_per_core_n if tail_geometry else self.prefill_expert_per_core_n
            gate_block_w = (
                self.prefill_expert_tail_in0_block_w if tail_geometry else self.prefill_expert_gate_in0_block_w
            )
            down_block_w = (
                self.prefill_expert_tail_in0_block_w if tail_geometry else self.prefill_expert_down_in0_block_w
            )
            hidden_grouped = ttnn.reshape(hidden_chunk, (1, groups, TILE_SIZE, HIDDEN_SIZE))
            if self.prefill_routed_active:
                # sparse_matmul selects one expert set per 32-token tile group.
                # Use the on-device union of token routes, then retain per-token
                # score weighting below. This preserves exact routing semantics
                # while avoiding the all-128-expert debug topology.
                sparsity = ttnn.max(ttnn.abs(routing_chunk), dim=2, keepdim=True)
                sparsity = ttnn.to_layout(sparsity, ttnn.ROW_MAJOR_LAYOUT)
                nnz = None
            else:
                sparsity = ttnn.repeat(self.expert_prefill_sparsity, (1, 1, groups, 1))
                nnz = NUM_EXPERTS * groups
            common = {
                "sparsity": sparsity,
                "nnz": nnz,
                "memory_config": ttnn.DRAM_MEMORY_CONFIG,
                "output_tile": output_tile,
                "dtype": self.activation_dtype,
                "compute_kernel_config": self.expert_compute_config,
            }
            gate_up_config = _optimized_sparse_prefill_config(
                self.mesh_device,
                n=self.expert_weights.intermediate_size_per_device,
                groups=groups,
                requested_per_core_n=per_core_n,
                in0_block_w=gate_block_w,
            )
            down_config = _optimized_sparse_prefill_config(
                self.mesh_device,
                n=HIDDEN_SIZE,
                groups=groups,
                requested_per_core_n=per_core_n,
                in0_block_w=down_block_w,
            )
            gate = ttnn.sparse_matmul(
                hidden_grouped,
                self.expert_weights.gate_proj,
                program_config=gate_up_config,
                **common,
            )
            sparse_intermediate = gate.shape[-1]
            gate = ttnn.transpose(gate, 1, 3)
            gate = ttnn.reshape(gate, (1, NUM_EXPERTS, physical_chunk, sparse_intermediate))
            up = ttnn.sparse_matmul(
                hidden_grouped,
                self.expert_weights.up_proj,
                program_config=gate_up_config,
                **common,
            )
            up = ttnn.transpose(up, 1, 3)
            up = ttnn.reshape(up, (1, NUM_EXPERTS, physical_chunk, sparse_intermediate))
            down_input = ttnn.reshape(
                apply_geglu(gate, up),
                (1, NUM_EXPERTS, physical_chunk, sparse_intermediate),
            )
            down = ttnn.sparse_matmul(
                down_input,
                self.expert_weights.down_proj,
                sparsity=sparsity,
                nnz=nnz,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                output_tile=output_tile,
                program_config=down_config,
                is_input_a_sparse=True,
                dtype=self.activation_dtype,
                compute_kernel_config=self.expert_compute_config,
            )
            next_states = ttnn.reshape(down, (1, NUM_EXPERTS, physical_chunk, HIDDEN_SIZE))
            routing_permuted = ttnn.permute(routing_chunk, (0, 3, 2, 1))
            next_states = ttnn.mul(next_states, routing_permuted)
            next_states = ttnn.unsqueeze_to_4D(ttnn.experimental.fast_reduce_nc(next_states, dims=[1]))
            results.append(ttnn.reshape(next_states, (1, 1, physical_chunk, HIDDEN_SIZE)))
        return results[0] if len(results) == 1 else ttnn.concat(results, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def _moe_decode(self, hidden_states: ttnn.Tensor, routing_weights: ttnn.Tensor) -> ttnn.Tensor:
        if self.residual_shard_cores == 0:
            return super()._moe_decode(hidden_states, routing_weights)
        batch = hidden_states.shape[2]
        if batch == 1:
            return self._moe_decode_single_user(hidden_states, routing_weights)

        outputs = []
        for batch_index in range(batch):
            hidden_row = ttnn.slice(
                hidden_states,
                [0, 0, batch_index, 0],
                [1, 1, batch_index + 1, HIDDEN_SIZE],
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            routing_row = ttnn.slice(
                routing_weights,
                [0, 0, batch_index, 0],
                [1, 1, batch_index + 1, NUM_EXPERTS],
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            outputs.append(self._moe_decode_single_user(hidden_row, routing_row))
        return ttnn.concat(outputs, dim=2, memory_config=ttnn.L1_MEMORY_CONFIG)

    def _moe_decode_single_user(
        self,
        hidden_states: ttnn.Tensor,
        routing_weights: ttnn.Tensor,
    ) -> ttnn.Tensor:
        self.optimized_path_counters["expert_decode"] += 1
        batch = hidden_states.shape[2]
        if self.expert_decode_input_l1:
            hidden_states = ttnn.to_memory_config(hidden_states, ttnn.L1_MEMORY_CONFIG, dtype=hidden_states.dtype)
        sparsity = ttnn.to_layout(routing_weights, ttnn.ROW_MAJOR_LAYOUT)
        output_tile = ttnn.Tile([TILE_SIZE, TILE_SIZE])
        gate_up_config = _optimized_sparse_decode_config(
            self.mesh_device,
            n=self.expert_weights.intermediate_size_per_device,
            per_core_n=self.expert_gate_per_core_n,
            in0_block_w=self.expert_gate_in0_block_w,
            out_subblock_w=self.expert_gate_out_subblock_w,
        )
        down_config = _optimized_sparse_decode_config(
            self.mesh_device,
            n=HIDDEN_SIZE,
            per_core_n=self.expert_down_per_core_n,
            in0_block_w=self.expert_down_in0_block_w,
            out_subblock_w=self.expert_down_out_subblock_w,
        )
        common = {
            "sparsity": sparsity,
            "nnz": TOP_K_EXPERTS,
            "memory_config": ttnn.L1_MEMORY_CONFIG,
            "output_tile": output_tile,
            "dtype": self.activation_dtype,
        }
        gate = ttnn.sparse_matmul(
            hidden_states,
            self.weights.expert_gate,
            program_config=gate_up_config,
            compute_kernel_config=self.expert_gate_compute_config,
            **common,
        )
        sparse_intermediate = gate.shape[-1]
        gate = ttnn.reshape(gate, (batch, NUM_EXPERTS, 1, sparse_intermediate))
        gate = ttnn.transpose(gate, 1, 2)
        gate = ttnn.reshape(gate, (batch, NUM_EXPERTS, sparse_intermediate))
        up = ttnn.sparse_matmul(
            hidden_states,
            self.weights.expert_up,
            program_config=gate_up_config,
            compute_kernel_config=self.expert_compute_config,
            **common,
        )
        up = ttnn.reshape(up, (batch, NUM_EXPERTS, 1, sparse_intermediate))
        up = ttnn.transpose(up, 1, 2)
        up = ttnn.reshape(up, (batch, NUM_EXPERTS, sparse_intermediate))
        down_input = apply_geglu(gate, up)
        down_input = ttnn.transpose(down_input, 1, 0)
        down_input = ttnn.reshape(down_input, (1, NUM_EXPERTS, batch, sparse_intermediate))
        down = ttnn.sparse_matmul(
            down_input,
            self.weights.expert_down,
            program_config=down_config,
            is_input_a_sparse=True,
            compute_kernel_config=self.expert_compute_config,
            **common,
        )
        next_states = ttnn.permute(down, (0, 2, 1, 3))
        next_states = ttnn.reshape(next_states, (batch, NUM_EXPERTS, HIDDEN_SIZE))
        routing_3d = ttnn.reshape(routing_weights, (batch, NUM_EXPERTS, 1))
        next_states = ttnn.mul(next_states, routing_3d)
        next_states = ttnn.sum(next_states, dim=1)
        next_states = ttnn.unsqueeze_to_4D(next_states)
        return ttnn.reshape(
            next_states,
            (1, 1, batch, HIDDEN_SIZE),
            (1, 1, max(TILE_SIZE, batch), HIDDEN_SIZE),
        )


__all__ = ["OptimizedDecoder"]
