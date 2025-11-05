# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
This module defines the MeshConfig class which manages parallelization strategies
across a mesh of devices for the GPT-OSS MoE model.
"""

from dataclasses import dataclass
from enum import Enum

import ttnn


class Mode(Enum):
    """Execution mode for model forward pass"""

    DECODE = "decode"
    PREFILL = "prefill"


@dataclass
class ModeConfig:
    """Per-mode parallelization configuration"""

    tp: int  # Tensor parallel size
    ep: int = 1  # Expert parallel size
    sp: int = 1  # Sequence parallel size

    def __post_init__(self):
        if self.tp < 1 or self.ep < 1 or self.sp < 1:
            raise ValueError(f"Parallelism values must be >= 1: tp={self.tp}, ep={self.ep}, sp={self.sp}")


class MeshConfig:
    """Mode-aware mesh parallelization with dataclass-based mode configs"""

    def __init__(
        self,
        mesh_shape,
        decode: ModeConfig,
        prefill: ModeConfig = None,
        tp_axis: int = 1,
    ):
        """
        Args:
            mesh_shape: (rows, cols) - any mesh size
            decode: ModeConfig for decode mode
            prefill: ModeConfig for prefill mode (defaults to tp=decode.tp, sp=rows, ep=1)
            tp_axis: Which mesh axis is TP (0=rows, 1=cols, default: 1)

        Default behavior:
            - Decode: Typically ep=rows (expert-parallel), sp=1
            - Prefill: Automatically sp=rows (sequence-parallel), ep=1
        """
        self.mesh_shape = tuple(mesh_shape)
        self.tp_axis = tp_axis
        self.ep_axis = 0 if tp_axis == 1 else 1
        self.sp_axis = self.ep_axis

        self.total_devices = mesh_shape[0] * mesh_shape[1]

        # Store mode configs
        self.decode = decode
        # Default prefill: Same TP, use rows for SP (sequence parallel), EP=1
        self.prefill = prefill or ModeConfig(tp=decode.tp, sp=mesh_shape[0], ep=1)

        # Validate both configs
        self._validate_config(self.decode, Mode.DECODE)
        self._validate_config(self.prefill, Mode.PREFILL)

        # Legacy attributes point to decode config
        self.tp = self.decode.tp
        self.ep = self.decode.ep
        self.sp = self.decode.sp
        self.dp = self.total_devices // (self.decode.tp * self.decode.ep)

    def _validate_config(self, config: ModeConfig, mode: Mode):
        """Validate a mode config fits the mesh"""
        dp = self.total_devices // (config.tp * config.ep)
        if config.tp * dp * config.ep != self.total_devices:
            raise ValueError(
                f"{mode.value}: TP({config.tp}) × DP({dp}) × EP({config.ep}) != total_devices({self.total_devices})"
            )

        tp_dim_size = self.mesh_shape[self.tp_axis]
        if config.tp > tp_dim_size:
            raise ValueError(f"{mode.value}: TP({config.tp}) > mesh_{self.tp_axis}_size({tp_dim_size})")

    def get_config(self, mode: Mode) -> ModeConfig:
        """Type-safe mode config access"""
        return self.decode if mode == Mode.DECODE else self.prefill

    def shard_mapper(self, mesh_device, tensor_dim=None, mesh_dims=None, mode: Mode = Mode.DECODE):
        """Unified 2D sharding - replaces all individual mappers"""
        if mesh_dims is None:
            # Default: shard along TP axis only
            mesh_dims = (None, tensor_dim) if self.tp_axis == 1 else (tensor_dim, None)

        return ttnn.ShardTensor2dMesh(mesh_device, mesh_device.shape, dims=mesh_dims)

    # Clean semantic helpers (all use unified shard_mapper)
    def column_parallel(self, mesh_device):
        """Column-parallel weights (feature dimension sharding)"""
        return self.shard_mapper(mesh_device, tensor_dim=-1)

    def row_parallel(self, mesh_device):
        """Row-parallel weights (sequence/batch dimension sharding)"""
        return self.shard_mapper(mesh_device, tensor_dim=-2)

    def sequence_parallel(self, mesh_device):
        """Sequence sharding (for KV cache)"""
        return self.shard_mapper(mesh_device, tensor_dim=-3)

    def shard_size(self, total_size, mode: Mode = Mode.DECODE):
        """Size per device for tensor parallel sharding"""
        config = self.get_config(mode)
        return total_size // config.tp

    def allreduce(self, tensor, ccl_manager, memory_config=None, pad_size=None, axis=0):
        """
        General tensor parallel allreduce (reduce-scatter + all-gather)

        Note: Caller should check if communication is needed before calling
        """
        memory_config = memory_config or ttnn.DRAM_MEMORY_CONFIG

        # Optional performance padding (caller specifies, no magic numbers)
        padded = False
        if pad_size and tensor.shape[-2] >= 32:
            tensor_padded = ttnn.pad(tensor, [(0, 0), (0, 0), (0, 0), (0, pad_size)], 0)
            tensor.deallocate(True)
            tensor = tensor_padded
            padded = True

        # Reduce-scatter along TP axis
        scattered = ttnn.experimental.reduce_scatter_minimal_async(
            tensor,
            dim=3,
            multi_device_global_semaphore=ccl_manager.get_rs_ping_pong_semaphore(),
            num_links=1,
            memory_config=memory_config,
            topology=ccl_manager.topology,
            cluster_axis=axis,
            barrier_semaphore=ccl_manager.get_barrier_semaphore(),
        )

        # All-gather back
        gathered = ttnn.experimental.all_gather_async(
            scattered,
            dim=3,
            cluster_axis=axis,
            mesh_device=ccl_manager.mesh_device,
            topology=ccl_manager.topology,
            multi_device_global_semaphore=ccl_manager.get_ag_ping_pong_semaphore(),
            num_links=1,
            memory_config=memory_config,
            barrier_semaphore=ccl_manager.get_barrier_semaphore(),
        )

        # Remove padding if applied
        if padded:
            gathered_sliced = gathered[:, :, :, :-pad_size]
            gathered.deallocate(True)
            gathered = gathered_sliced

        return gathered

    def allgather(self, tensor, ccl_manager, memory_config=None, axis=0, dim=3):
        """
        All-gather operation for tensor parallel communication

        Note: Caller should check if communication is needed before calling
        """
        memory_config = memory_config or ttnn.DRAM_MEMORY_CONFIG

        return ttnn.experimental.all_gather_async(
            tensor,
            dim=dim,
            cluster_axis=axis,
            mesh_device=ccl_manager.mesh_device,
            topology=ccl_manager.topology,
            multi_device_global_semaphore=ccl_manager.get_ag_ping_pong_semaphore(),
            num_links=1,
            memory_config=memory_config,
            barrier_semaphore=ccl_manager.get_barrier_semaphore(),
        )

    def __repr__(self):
        decode_dp = self.total_devices // (self.decode.tp * self.decode.ep)
        prefill_dp = self.total_devices // (self.prefill.tp * self.prefill.ep)
        decode_str = f"decode[TP={self.decode.tp}, EP={self.decode.ep}, SP={self.decode.sp}, DP={decode_dp}]"
        prefill_str = f"prefill[TP={self.prefill.tp}, EP={self.prefill.ep}, SP={self.prefill.sp}, DP={prefill_dp}]"
        return f"MeshConfig({self.mesh_shape}, {decode_str}, {prefill_str})"


# Convenience factory functions for common configurations
def mesh_2x4():
    # decode: TP=4, EP=2, DP=1; prefill: TP=4, SP=2, EP=1, DP=1
    return MeshConfig((2, 4), decode=ModeConfig(tp=4, ep=2))


def mesh_4x8():
    # decode: TP=8, EP=4, DP=1; prefill: TP=8, SP=4, EP=1, DP=4
    return MeshConfig((4, 8), decode=ModeConfig(tp=8, ep=4))


def mesh_4x4():
    # decode: TP=4, EP=4, DP=1; prefill: TP=4, SP=4, EP=1, DP=1
    return MeshConfig((4, 4), decode=ModeConfig(tp=4, ep=4))


def mesh_1x8():
    # decode: TP=8, EP=1, DP=1; prefill: TP=8, SP=1, EP=1, DP=1 (no SP on single row)
    return MeshConfig((1, 8), decode=ModeConfig(tp=8, ep=1))
