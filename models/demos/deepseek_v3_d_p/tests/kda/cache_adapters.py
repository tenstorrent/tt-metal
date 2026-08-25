# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Investigation-only adapters for the K3 disaggregated KDA cache contract."""

from __future__ import annotations

from dataclasses import dataclass

import ttnn
from models.demos.deepseek_v3_d_p.tt.kda.kda import KdaState

KDA_GLOBAL_HEADS = 96
KDA_HEAD_DIM = 128
KDA_CONV_HISTORY = 3
KDA_CONV_BRANCHES = 3
KDA_S_BAND_WIDTH = 32
KDA_CONV_HALF_WIDTH = 64

KDA_S_SEGMENT_BYTES = 128 * 32 * 4
KDA_CONV_SEGMENT_BYTES = 3 * 64 * 2


@dataclass(frozen=True)
class KdaCacheGeometry:
    """Exact per-topology K3 cache geometry used by the ablation."""

    sequence_parallel_size: int
    tensor_parallel_size: int

    def __post_init__(self) -> None:
        if self.sequence_parallel_size * self.tensor_parallel_size != 8:
            raise ValueError(
                "KDA cache ablation requires exactly eight devices, got "
                f"SP{self.sequence_parallel_size}xTP{self.tensor_parallel_size}"
            )
        if KDA_GLOBAL_HEADS % self.tensor_parallel_size != 0:
            raise ValueError(f"{KDA_GLOBAL_HEADS} heads cannot be divided over TP{self.tensor_parallel_size}")

    @property
    def local_heads(self) -> int:
        return KDA_GLOBAL_HEADS // self.tensor_parallel_size

    @property
    def recurrent_shape(self) -> tuple[int, int, int, int]:
        return (1, self.local_heads, KDA_HEAD_DIM, KDA_HEAD_DIM)

    @property
    def convolution_shape(self) -> tuple[int, int, int]:
        return (1, KDA_CONV_HISTORY, KDA_CONV_BRANCHES * self.local_heads * KDA_HEAD_DIM)

    @property
    def recurrent_segments_per_device(self) -> int:
        return self.local_heads * (KDA_HEAD_DIM // KDA_S_BAND_WIDTH)

    @property
    def convolution_segments_per_device(self) -> int:
        return self.local_heads * KDA_CONV_BRANCHES * (KDA_HEAD_DIM // KDA_CONV_HALF_WIDTH)

    @property
    def unique_recurrent_segments(self) -> int:
        return self.recurrent_segments_per_device * self.tensor_parallel_size

    @property
    def unique_convolution_segments(self) -> int:
        return self.convolution_segments_per_device * self.tensor_parallel_size

    @property
    def recurrent_bytes_per_device(self) -> int:
        return self.recurrent_segments_per_device * KDA_S_SEGMENT_BYTES

    @property
    def convolution_bytes_per_device(self) -> int:
        return self.convolution_segments_per_device * KDA_CONV_SEGMENT_BYTES

    @property
    def physical_recurrent_bytes(self) -> int:
        return self.recurrent_bytes_per_device * 8

    @property
    def physical_convolution_bytes(self) -> int:
        return self.convolution_bytes_per_device * 8


@dataclass(frozen=True)
class KdaContractMemoryConfigs:
    recurrent: ttnn.MemoryConfig
    convolution: ttnn.MemoryConfig


def _dram_bank_grid(device: ttnn.Device | ttnn.MeshDevice) -> ttnn.CoreRangeSet:
    num_banks = device.dram_grid_size().x
    if num_banks <= 0:
        raise ValueError("KDA cache adapters require at least one DRAM bank")
    return ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(bank, 0), ttnn.CoreCoord(bank, 0)) for bank in range(num_banks)}
    )


def _nd_dram_config(
    device: ttnn.Device | ttnn.MeshDevice,
    shard_shape: list[int],
) -> ttnn.MemoryConfig:
    return ttnn.MemoryConfig(
        buffer_type=ttnn.BufferType.DRAM,
        nd_shard_spec=ttnn.NdShardSpec(
            shard_shape=shard_shape,
            grid=_dram_bank_grid(device),
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            shard_distribution_strategy=ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
        ),
    )


def contract_memory_configs(device: ttnn.Device | ttnn.MeshDevice) -> KdaContractMemoryConfigs:
    """Return ND-DRAM layouts whose shards are exactly the migration segments."""

    return KdaContractMemoryConfigs(
        recurrent=_nd_dram_config(device, [1, 1, KDA_HEAD_DIM, KDA_S_BAND_WIDTH]),
        convolution=_nd_dram_config(device, [1, KDA_CONV_HISTORY, KDA_CONV_HALF_WIDTH]),
    )


def _validate_native_state(state: KdaState, geometry: KdaCacheGeometry) -> None:
    if tuple(state.recurrent.shape) != geometry.recurrent_shape:
        raise ValueError(f"recurrent shape {tuple(state.recurrent.shape)} != {geometry.recurrent_shape}")
    if tuple(state.convolution.shape) != geometry.convolution_shape:
        raise ValueError(f"convolution shape {tuple(state.convolution.shape)} != {geometry.convolution_shape}")
    if state.recurrent.dtype != ttnn.float32 or state.recurrent.layout != ttnn.TILE_LAYOUT:
        raise ValueError("native recurrent state must be FP32 tile layout")
    if state.convolution.dtype != ttnn.bfloat16 or state.convolution.layout != ttnn.ROW_MAJOR_LAYOUT:
        raise ValueError("native convolution state must be BF16 row-major layout")
    if state.recurrent.memory_config() != ttnn.DRAM_MEMORY_CONFIG:
        raise ValueError("native recurrent state must use interleaved DRAM")
    if state.convolution.memory_config() != ttnn.DRAM_MEMORY_CONFIG:
        raise ValueError("native convolution state must use interleaved DRAM")


def _validate_contract_state(
    state: KdaState,
    geometry: KdaCacheGeometry,
    configs: KdaContractMemoryConfigs,
) -> None:
    if tuple(state.recurrent.shape) != geometry.recurrent_shape:
        raise ValueError(f"contract recurrent shape {tuple(state.recurrent.shape)} != {geometry.recurrent_shape}")
    if tuple(state.convolution.shape) != geometry.convolution_shape:
        raise ValueError(f"contract convolution shape {tuple(state.convolution.shape)} != {geometry.convolution_shape}")
    if state.recurrent.dtype != ttnn.float32 or state.recurrent.layout != ttnn.TILE_LAYOUT:
        raise ValueError("contract recurrent state must be FP32 tile layout")
    if state.convolution.dtype != ttnn.bfloat16 or state.convolution.layout != ttnn.ROW_MAJOR_LAYOUT:
        raise ValueError("contract convolution state must be BF16 row-major layout")
    if state.recurrent.memory_config() != configs.recurrent:
        raise ValueError("contract recurrent state has the wrong ND-DRAM layout")
    if state.convolution.memory_config() != configs.convolution:
        raise ValueError("contract convolution state has the wrong ND-DRAM layout")


def allocate_contract_state(
    device: ttnn.Device | ttnn.MeshDevice,
    geometry: KdaCacheGeometry,
) -> KdaState:
    configs = contract_memory_configs(device)
    return KdaState(
        recurrent=ttnn.allocate_tensor_on_device(
            ttnn.Shape(geometry.recurrent_shape), ttnn.float32, ttnn.TILE_LAYOUT, device, configs.recurrent
        ),
        convolution=ttnn.allocate_tensor_on_device(
            ttnn.Shape(geometry.convolution_shape),
            ttnn.bfloat16,
            ttnn.ROW_MAJOR_LAYOUT,
            device,
            configs.convolution,
        ),
    )


def allocate_native_state(
    device: ttnn.Device | ttnn.MeshDevice,
    geometry: KdaCacheGeometry,
) -> KdaState:
    return KdaState(
        recurrent=ttnn.allocate_tensor_on_device(
            ttnn.Shape(geometry.recurrent_shape),
            ttnn.float32,
            ttnn.TILE_LAYOUT,
            device,
            ttnn.DRAM_MEMORY_CONFIG,
        ),
        convolution=ttnn.allocate_tensor_on_device(
            ttnn.Shape(geometry.convolution_shape),
            ttnn.bfloat16,
            ttnn.ROW_MAJOR_LAYOUT,
            device,
            ttnn.DRAM_MEMORY_CONFIG,
        ),
    )


def export_recurrent(source: ttnn.Tensor, destination: ttnn.Tensor) -> ttnn.Tensor:
    ttnn.to_memory_config(source, destination.memory_config(), output_tensor=destination)
    return destination


def export_convolution(source: ttnn.Tensor, destination: ttnn.Tensor) -> ttnn.Tensor:
    ttnn.to_memory_config(source, destination.memory_config(), output_tensor=destination)
    return destination


def import_recurrent(source: ttnn.Tensor, destination: ttnn.Tensor) -> ttnn.Tensor:
    ttnn.to_memory_config(source, destination.memory_config(), output_tensor=destination)
    return destination


def import_convolution(source: ttnn.Tensor, destination: ttnn.Tensor) -> ttnn.Tensor:
    ttnn.to_memory_config(source, destination.memory_config(), output_tensor=destination)
    return destination


def export_state(source: KdaState, destination: KdaState, geometry: KdaCacheGeometry) -> KdaState:
    _validate_native_state(source, geometry)
    configs = contract_memory_configs(source.recurrent.device())
    _validate_contract_state(destination, geometry, configs)
    return KdaState(
        recurrent=export_recurrent(source.recurrent, destination.recurrent),
        convolution=export_convolution(source.convolution, destination.convolution),
    )


def import_state(source: KdaState, destination: KdaState, geometry: KdaCacheGeometry) -> KdaState:
    configs = contract_memory_configs(source.recurrent.device())
    _validate_contract_state(source, geometry, configs)
    _validate_native_state(destination, geometry)
    return KdaState(
        recurrent=import_recurrent(source.recurrent, destination.recurrent),
        convolution=import_convolution(source.convolution, destination.convolution),
    )


def deallocate_state(state: KdaState) -> None:
    ttnn.deallocate(state.recurrent)
    ttnn.deallocate(state.convolution)
