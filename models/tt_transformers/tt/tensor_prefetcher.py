# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Device-side Tensor Prefetcher backend for tt-transformers decode matmuls."""

import math
import os
from typing import List, Optional

from loguru import logger

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.tt_transformers.tt.common import Mode


def _model_config(model_name: str) -> Optional[dict]:
    from models.tt_transformers.tt.prefetcher import TENSOR_PREFETCHER_VERIFIED_MODEL_CONFIGS

    key = next((name for name in TENSOR_PREFETCHER_VERIFIED_MODEL_CONFIGS if name in model_name), None)
    return TENSOR_PREFETCHER_VERIFIED_MODEL_CONFIGS[key] if key is not None else None


def _rectangle_height(grid_x: int, rows_left: int, num_cores: int) -> int:
    return next(
        (
            height
            for height in range(min(rows_left, num_cores), 0, -1)
            if num_cores % height == 0 and num_cores // height <= grid_x
        ),
        0,
    )


def _candidate_supported(mesh_device: ttnn.MeshDevice, model_name: str, receivers_per_bank: int) -> bool:
    cfg = _model_config(model_name)
    if cfg is None:
        return False

    num_devices = mesh_device.get_num_devices()
    num_banks = mesh_device.dram_grid_size().x
    ring_size = num_banks * receivers_per_bank
    if cfg["n_kv_heads"] % num_devices != 0:
        return False

    head_dim = cfg["dim"] // cfg["n_heads"]
    dimensions = (
        cfg["dim"],
        cfg["hidden_dim"] // num_devices,
        (cfg["n_heads"] + 2 * cfg["n_kv_heads"]) * head_dim // num_devices,
        cfg["n_heads"] * head_dim // num_devices,
    )
    if any(dimension % (ttnn.TILE_SIZE * ring_size) != 0 for dimension in dimensions):
        return False

    grid = mesh_device.compute_with_storage_grid_size()
    if num_banks > grid.x or receivers_per_bank > grid.y:
        return False
    if _rectangle_height(grid.x, grid.y - receivers_per_bank, 32) == 0:
        return False

    # Streaming retains a partial ring in the GCB, but needs at least two
    # complete receiver blocks for producer/consumer overlap.
    streaming_gcb_budget = 788_032
    tile_bytes = 1088
    dim = cfg["dim"]
    hidden = cfg["hidden_dim"] // num_devices
    qkv = (cfg["n_heads"] + 2 * cfg["n_kv_heads"]) * head_dim // num_devices
    wo_k = cfg["n_heads"] * head_dim // num_devices
    shapes = ((dim, hidden), (hidden, dim), (dim, qkv), (wo_k, dim))
    max_block = max(
        (k // ttnn.TILE_SIZE // ring_size) * (n // ttnn.TILE_SIZE // ring_size) * tile_bytes
        for k, n in shapes
    )
    return 2 * max_block <= streaming_gcb_budget


def has_supported_receiver_configuration(mesh_device: ttnn.MeshDevice, model_name: str) -> bool:
    return any(_candidate_supported(mesh_device, model_name, receivers) for receivers in (8, 4, 2, 1))


def _ring_core(position: int, columns: int) -> ttnn.CoreCoord:
    return ttnn.CoreCoord(position % columns, position // columns)


class TensorPrefetcher(LightweightModule):
    """Tensor Prefetcher implementation of the existing model prefetcher interface."""

    def __init__(self, mesh_device: ttnn.MeshDevice, num_tensors: int, num_layers: int):
        from models.tt_transformers.tt.prefetcher import TILE_BYTES

        self.mesh_device = mesh_device
        self.num_tensors = num_tensors
        self.num_layers = num_layers
        self.num_senders = mesh_device.dram_grid_size().x
        self.model_name = os.environ.get("HF_MODEL", "")
        self.uses_tensor_prefetcher = True
        self.colocate_ops = False
        self.stream_in1 = True
        self.worker_sub_device_id = None
        self.global_cb = None
        self.mode = Mode.PREFILL
        self._started = False
        self._stopped = False
        self._tile_bytes = TILE_BYTES
        self.registered_weights: List[ttnn.Tensor] = []
        self.registered_program_configs: List = []

        self.num_receiver_cores = next(
            receivers
            for receivers in (8, 4, 2, 1)
            if _candidate_supported(mesh_device, self.model_name, receivers)
        )
        self.ring_size = self.num_senders * self.num_receiver_cores
        self._ring_cores = [_ring_core(position, self.num_senders) for position in range(self.ring_size)]
        self._bank_to_receivers = []
        for bank in range(self.num_senders):
            ranges = []
            for offset in range(self.num_receiver_cores):
                position = bank * self.num_receiver_cores + offset
                core = _ring_core(position, self.num_senders)
                ranges.append(ttnn.CoreRange(core, core))
            self._bank_to_receivers.append((bank, ttnn.CoreRangeSet(ranges)))

        grid = mesh_device.compute_with_storage_grid_size()
        self.all_worker_cores_range_set = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))]
        )

    def register_callback(self, callback) -> None:
        callback()

    def insert_tensor(self, tensor: ttnn.Tensor, program_config=None) -> None:
        assert not self._started, "Cannot register weights after the Tensor Prefetcher has started."
        assert program_config is not None, "Tensor Prefetcher weight registration requires a program config."
        if not tensor.is_sharded() or tensor.memory_config().buffer_type != ttnn.BufferType.DRAM:
            raise ValueError("Tensor Prefetcher weights must be DRAM-sharded tensors.")
        self.registered_weights.append(tensor)
        self.registered_program_configs.append(program_config)

    def init(self, mode: Mode = Mode.DECODE) -> None:
        self.mode = mode
        if mode != Mode.DECODE or self._started:
            return
        assert not self._stopped, "Tensor Prefetcher cannot restart after teardown."
        expected = self.num_tensors * self.num_layers
        assert len(self.registered_weights) == expected, (
            f"Expected {expected} registered Tensor Prefetcher weights, got {len(self.registered_weights)}."
        )
        self._build_global_cb()
        ttnn.experimental.start_tensor_prefetcher(self.mesh_device)
        self._started = True
        logger.info(f"[TensorPrefetcher] started with ring size {self.ring_size}")

    def prefetch(self) -> None:
        pass

    def run(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def teardown(self) -> None:
        if self._started and not self._stopped:
            ttnn.experimental.stop_tensor_prefetcher(self.mesh_device)
            self._stopped = True

    def __del__(self):
        try:
            self.teardown()
        except Exception:
            pass

    def to_core_range_set(self, cores: List, return_list: bool = False):
        def ranges(core):
            if isinstance(core, ttnn.CoreRangeSet):
                return core.ranges()
            if isinstance(core, ttnn.CoreRange):
                return [core]
            if isinstance(core, ttnn.CoreCoord):
                return [ttnn.CoreRange(core, core)]
            raise ValueError(f"Unsupported core type: {type(core)}")

        if return_list:
            return [ttnn.CoreRangeSet(ranges(core)) for core in cores]
        return ttnn.CoreRangeSet([core_range for core in cores for core_range in ranges(core)])

    def receiver_cores(self, sender_active=None, receiver_active=None):
        del sender_active, receiver_active
        return [ttnn.CoreRangeSet([ttnn.CoreRange(core, core)]) for core in self._ring_cores]

    def dynamic_worker_core_grid(self, num_cores: int) -> ttnn.CoreRangeSet:
        grid = self.mesh_device.compute_with_storage_grid_size()
        row = self.num_receiver_cores
        height = _rectangle_height(grid.x, grid.y - row, num_cores)
        assert height > 0, f"Cannot place {num_cores} worker cores below Tensor Prefetcher receivers."
        width = num_cores // height
        return ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(0, row), ttnn.CoreCoord(width - 1, row + height - 1))]
        )

    def weight_cache_suffix(self) -> str:
        return "_recv_contig"

    def weight_mem_config(self, k: int, n: int, default: ttnn.MemoryConfig) -> ttnn.MemoryConfig:
        del default
        dram_cores = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(self.num_senders - 1, 0))]
        )
        shard_spec = ttnn.NdShardSpec(
            ttnn.Shape([k, n // self.ring_size]),
            dram_cores,
            ttnn.ShardOrientation.ROW_MAJOR,
            ttnn.ShardDistributionStrategy.CONTIGUOUS_1D,
        )
        return ttnn.MemoryConfig(ttnn.BufferType.DRAM, shard_spec)

    def _build_global_cb(self) -> None:
        assert self.global_cb is None
        max_block_size = 0
        for tensor, program_config in zip(self.registered_weights, self.registered_program_configs):
            k_tiles = math.ceil(tensor.shape[-2] / ttnn.TILE_SIZE)
            block_size = (
                (k_tiles // self.ring_size) * program_config.per_core_N * self._tile_bytes[tensor.dtype]
            )
            max_block_size = max(max_block_size, block_size)
        window_blocks = min(self.ring_size, 788_032 // max_block_size)
        assert window_blocks >= 2, "Tensor Prefetcher streaming GCB requires at least two blocks."
        self.global_cb = ttnn.experimental.create_global_circular_buffer_for_matmul_1d(
            self.mesh_device,
            self.registered_program_configs,
            self.registered_weights,
            self._bank_to_receivers,
            window_blocks * max_block_size,
        )
