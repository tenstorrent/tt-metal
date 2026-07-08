# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Tensor-prefetcher (DRAM-core / DRISC) backend for tt-transformers decode.

This is an alternative to the worker DRAM prefetcher (``prefetcher.py`` /
``ttnn.dram_prefetcher``). Instead of filling one big global circular buffer with
every layer's weights up front, it couples a DRAM-core prefetch request to each
consuming matmul via ``ttnn.experimental.tensor_prefetcher_matmul.prefetch_and_linear``,
streaming receiver-contiguous weights through a shallow per-weight-role GCB.

It intentionally does **not** reuse the worker ``Prefetcher`` class internals; it
only matches its ring topology (``ring_size = num_dram_banks * num_receiver_cores``)
so the two backends are an apples-to-apples benchmark, and exposes the same
attribute surface (``ring_size``, ``num_receiver_cores``, ``receiver_cores``,
``to_core_range_set``, ``all_worker_cores_range_set``) that ``model_config`` reads.
"""

from __future__ import annotations

from typing import List, Optional, Union

import ttnn
from loguru import logger

from models.common.lightweightmodule import LightweightModule
from models.tt_transformers.tt.common import Mode
from models.tt_transformers.tt.prefetcher import (
    ARCH_CONFIG,
    generate_sender_receiver_mapping,
    is_prefetcher_supported,
)

# Marker read by model_config to select streaming program configs for this backend.
PREFETCHER_KIND_TENSOR = "tensor"

_BYTES_PER_TILE = {ttnn.bfloat4_b: 576, ttnn.bfloat8_b: 1088, ttnn.bfloat16: 2048}


def _bytes_per_tile(dtype) -> int:
    return _BYTES_PER_TILE[dtype]


def bank_receivers_strided(bank_idx: int, recv_per_bank: int, num_dram_banks: int, ring_cols: int) -> ttnn.CoreRangeSet:
    """Receiver arc for ROUND_ROBIN_1D weights: bank b -> ring positions
    [b, b + num_dram_banks, ...] so that shard index == ring position."""
    cores = []
    for s in range(recv_per_bank):
        ring_pos = bank_idx + s * num_dram_banks
        col = ring_pos % ring_cols
        row = ring_pos // ring_cols
        cores.append(ttnn.CoreRange(ttnn.CoreCoord(col, row), ttnn.CoreCoord(col, row)))
    return ttnn.CoreRangeSet(cores)


def bank_receivers_contiguous(bank_idx: int, recv_per_bank: int, ring_cols: int) -> ttnn.CoreRangeSet:
    """Receiver arc for CONTIGUOUS_1D weights: bank b -> ring positions
    [b*recv_per_bank .. (b+1)*recv_per_bank - 1] so that shard index == ring position."""
    cores = []
    for s in range(recv_per_bank):
        ring_pos = bank_idx * recv_per_bank + s
        col = ring_pos % ring_cols
        row = ring_pos // ring_cols
        cores.append(ttnn.CoreRange(ttnn.CoreCoord(col, row), ttnn.CoreCoord(col, row)))
    return ttnn.CoreRangeSet(cores)


class TensorPrefetcher(LightweightModule):
    """DRAM-core prefetcher backend. Drop-in alternative to ``Prefetcher`` for decode."""

    # Discriminator used by model_config to enable stream_in1 for this backend only.
    kind = PREFETCHER_KIND_TENSOR
    stream_in1 = True

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        num_tensors: int,
        num_layers: int,
        num_receiver_cores: Optional[int] = None,
        dual_senders_per_bank: bool = True,
        distribution_strategy=ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
        window_blocks: int = 2,
    ):
        import os

        self.mesh_device = mesh_device
        self.num_tensors = num_tensors
        self.num_layers = num_layers
        self.dual_senders_per_bank = dual_senders_per_bank
        self.distribution_strategy = distribution_strategy
        self.is_contiguous = distribution_strategy == ttnn.ShardDistributionStrategy.CONTIGUOUS_1D
        self.window_blocks = window_blocks

        self.model_name = os.getenv("HF_MODEL", "")
        assert self.model_name != "", "HF_MODEL is not set. Tensor prefetcher must be run with a model."
        assert ttnn.experimental.is_tensor_prefetcher_supported(
            mesh_device
        ), "Tensor prefetcher is not supported on this device (needs Blackhole + TT_METAL_ENABLE_BLACKHOLE_DRAM_PROGRAMMABLE_CORES=1)."

        # Ring topology matched to the worker prefetcher: senders == DRAM banks.
        self.num_dram_banks = mesh_device.dram_grid_size().x
        pf_config = ARCH_CONFIG["blackhole"]
        legal_receiver_cores: List[int] = pf_config["legal_receiver_cores"]
        num_devices = mesh_device.get_num_devices()

        if num_receiver_cores is not None:
            assert num_receiver_cores in legal_receiver_cores, "num_receiver_cores must be in legal_receiver_cores"
            self.num_receiver_cores = num_receiver_cores
        else:
            self.num_receiver_cores = None
            for candidate in legal_receiver_cores:
                if is_prefetcher_supported(self.model_name, num_devices, candidate * self.num_dram_banks):
                    self.num_receiver_cores = candidate
            assert self.num_receiver_cores is not None, "No legal num_receiver_cores is supported for this model/device"

        self.ring_size = self.num_dram_banks * self.num_receiver_cores
        # Receiver grid: ring_cols columns (one per bank) x ring_rows rows (recv per bank),
        # matching the recv-contig GCB arc so shard index == ring position.
        self.ring_cols = self.num_dram_banks
        self.ring_rows = self.num_receiver_cores
        self._receiver_core_range_set = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(self.ring_cols - 1, self.ring_rows - 1))}
        )
        self._dram_core_range_set = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(self.num_dram_banks - 1, 0))}
        )

        # Worker cores == everything outside the receiver ring is available; the ring itself
        # runs the matmuls. Expose the full receiver ring for sub-core-grid consumers.
        grid = mesh_device.compute_with_storage_grid_size()
        self.all_worker_cores_range_set = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))]
        )

        self.mode = Mode.PREFILL
        self._started = False
        self._gcb_cache = {}  # (K, N, dtype) -> GlobalCircularBuffer
        self.worker_sub_device_id = None  # tensor backend uses no worker sub-device

        logger.info(
            f"[TensorPrefetcher] banks={self.num_dram_banks} recv/bank={self.num_receiver_cores} "
            f"ring_size={self.ring_size} dual_senders={self.dual_senders_per_bank} "
            f"strategy={self.distribution_strategy} window_blocks={self.window_blocks}"
        )

    # --- attribute-surface parity with the worker Prefetcher (read by model_config) ---
    def to_core_range_set(self, cores, return_list: bool = False) -> Union[ttnn.CoreRangeSet, list]:
        if return_list:
            return [cores] if isinstance(cores, ttnn.CoreRangeSet) else list(cores)
        return cores if isinstance(cores, ttnn.CoreRangeSet) else ttnn.CoreRangeSet(list(cores))

    def receiver_cores(self, sender_active=None, receiver_active=None) -> ttnn.CoreRangeSet:
        """Receiver ring grid the matmuls run on (activation sharding target)."""
        return self._receiver_core_range_set

    # --- weight layout -------------------------------------------------------------
    def weight_mem_config(self, k: int, n: int) -> ttnn.MemoryConfig:
        """Per-device receiver-contiguous DRAM ND-shard mem config for a (k, n) weight.

        ``k`` / ``n`` are the per-device logical dims (same values the worker path feeds
        ``create_dram_sharded_mem_config``). ``num_shards == ring_size`` with shard
        ``(k, n // ring_size)``.
        """
        assert n % self.ring_size == 0, f"weight N={n} must divide ring_size={self.ring_size}"
        nd_shard = ttnn.NdShardSpec(
            ttnn.Shape([k, n // self.ring_size]),
            self._dram_core_range_set,
            ttnn.ShardOrientation.ROW_MAJOR,
            self.distribution_strategy,
        )
        return ttnn.MemoryConfig(ttnn.BufferType.DRAM, nd_shard)

    # --- global circular buffer (one shallow streaming GCB per weight role) --------
    def _bank_to_receivers(self):
        if self.is_contiguous:
            return [
                (b, bank_receivers_contiguous(b, self.num_receiver_cores, ring_cols=self.ring_cols))
                for b in range(self.num_dram_banks)
            ]
        return [
            (b, bank_receivers_strided(b, self.num_receiver_cores, self.num_dram_banks, ring_cols=self.ring_cols))
            for b in range(self.num_dram_banks)
        ]

    def gcb_for(self, weight: ttnn.Tensor, program_config) -> ttnn.GlobalCircularBuffer:
        """Build (and cache by weight shape/dtype) the shallow streaming GCB for a weight role."""
        key = (weight.shape[-2], weight.shape[-1], weight.dtype)
        if key in self._gcb_cache:
            return self._gcb_cache[key]

        # in1 K-block per receiver = in0_block_w tiles x per_core_N tiles (== the streaming
        # test's k_tiles_per_shard x n_tiles_per_receiver). matmul_1d_ring_config sets
        # in0_block_w = K_tiles / ring_size, i.e. the real per-shard K depth.
        in1_block_size_bytes = program_config.in0_block_w * program_config.per_core_N * _bytes_per_tile(weight.dtype)
        gcb_size = self.window_blocks * in1_block_size_bytes

        gcb = ttnn.experimental.create_global_circular_buffer_for_matmul_1d_recv_contig(
            self.mesh_device,
            [program_config],
            [weight],
            bank_to_receivers=self._bank_to_receivers(),
            size=gcb_size,
            dual_senders_per_bank=self.dual_senders_per_bank,
        )
        self._gcb_cache[key] = gcb
        logger.info(
            f"[TensorPrefetcher] built streaming GCB for weight {tuple(weight.shape)} {weight.dtype}: "
            f"{self.window_blocks} x {in1_block_size_bytes}B = {gcb_size}B"
        )
        return gcb

    # --- the consuming matmul ------------------------------------------------------
    def linear(self, input_tensor_a, weight, *, program_config, cq_id=None, **linear_kwargs):
        """Queue a DRAM-core prefetch of ``weight`` and run the consuming gather_in0 matmul,
        as a single coupled ``prefetch_and_linear`` call (never two separate calls)."""
        gcb = self.gcb_for(weight, program_config)
        return ttnn.experimental.tensor_prefetcher_matmul.prefetch_and_linear(
            input_tensor_a,
            weight,
            global_cb=gcb,
            program_config=program_config,
            cq_id=cq_id,
            **linear_kwargs,
        )

    # --- lifecycle -----------------------------------------------------------------
    # These are no-ops for the tensor backend's up-front-registration API; the tensor
    # prefetcher enqueues per matmul call instead.
    def register_callback(self, callback) -> None:
        return

    def insert_tensor(self, tensor) -> None:
        return

    def init(self, mode: Mode = Mode.DECODE) -> None:
        self.mode = mode

    def prefetch(self) -> None:
        return

    def run(self) -> None:
        """Idempotent start of the DRAM-core prefetcher (DRISC kernels + host worker).

        Started once and kept alive across trace capture *and* replay; per-forward
        stop()/run() churn would tear down the DRISC kernels between capture and replay.
        """
        if not self._started:
            ttnn.experimental.start_tensor_prefetcher(
                self.mesh_device, dual_senders_per_bank=self.dual_senders_per_bank
            )
            self._started = True

    def stop(self) -> None:
        # No-op per forward: keep the prefetcher running. Real teardown is close().
        return

    def close(self) -> None:
        if self._started:
            ttnn.experimental.stop_tensor_prefetcher(self.mesh_device)
            ttnn.synchronize_device(self.mesh_device)
            self._started = False
