# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DRAM weight prefetcher for GLM-4.7 on Galaxy Wormhole.

Adapted from Llama3 70B Galaxy prefetcher (prefetcher_common.py).
Uses the same SubDevice + GlobalCB infrastructure to overlap DRAM weight
reads with compute across layers.

Architecture:
  - Sender cores: read weights from DRAM, push to Global CB
  - Worker cores: consume weights from Global CB via global_cb= parameter
  - Prefetcher runs on dedicated SubDevice, overlapping with worker compute

Usage in model_tt.py decode:
  1. prefetcher.create_global_cb()
  2. garbage = ttnn.dram_prefetcher(tt_tensors, num_layers=N, global_cb=cb)
  3. mesh_device.set_sub_device_stall_group([worker_sub_device_id])
  4. for layer in layers: layer.forward(..., global_cb=cb, sub_device_id=worker_id)
  5. ttnn.deallocate(garbage)
"""
import os

import torch
import ttnn
from loguru import logger


def get_glm_core_ranges(num_reader_cores: int = 12, num_global_cb_receivers: int = 2):
    """Compute core ranges for GLM-4.7 on Galaxy Wormhole (8x9 grid per chip).

    Returns sender cores, receiver cores, and worker cores for the prefetcher.
    Galaxy WH has 72 cores (8 cols × 9 rows). We reserve sender + receiver cores
    for the prefetcher and give the rest to compute workers.

    Based on Llama Galaxy's get_core_ranges() but simplified for GLM.
    """
    # Galaxy WH: 8 columns (x=0..7), 9 rows (y=0..8)
    grid_x, grid_y = 8, 9

    # Sender cores: leftmost column, rows 0..num_reader_cores-1
    # These read from DRAM and write to Global CB
    sender_cores = [ttnn.CoreCoord(0, y) for y in range(num_reader_cores)]

    # Receiver cores: columns 1..num_global_cb_receivers, same rows as senders
    # These receive data from senders via Global CB
    receiver_cores = []
    for y in range(num_reader_cores):
        for x in range(1, num_global_cb_receivers + 1):
            receiver_cores.append(ttnn.CoreCoord(x, y))

    # DRAM cores for address mapping (same as sender cores)
    dram_cores = sender_cores[:num_reader_cores]

    # Worker cores: everything NOT in sender/receiver set
    # For compute (matmuls, norms, etc.)
    prefetcher_core_set = set()
    for c in sender_cores:
        prefetcher_core_set.add((c.x, c.y))
    for c in receiver_cores:
        prefetcher_core_set.add((c.x, c.y))

    worker_ranges = []
    # Use remaining columns as worker grid
    # Columns 0 is sender, 1-2 are receivers, 3-7 are workers (5 cols × 9 rows = 45 cores)
    worker_start_x = num_global_cb_receivers + 1  # e.g., 3
    if worker_start_x < grid_x:
        worker_ranges.append(
            ttnn.CoreRange(
                ttnn.CoreCoord(worker_start_x, 0),
                ttnn.CoreCoord(grid_x - 1, grid_y - 1),
            )
        )
    # Also include any free rows in sender/receiver columns (rows >= num_reader_cores)
    if num_reader_cores < grid_y:
        worker_ranges.append(
            ttnn.CoreRange(
                ttnn.CoreCoord(0, num_reader_cores),
                ttnn.CoreCoord(num_global_cb_receivers, grid_y - 1),
            )
        )

    worker_core_range_set = ttnn.CoreRangeSet(worker_ranges)

    sender_core_range_set = ttnn.CoreRangeSet(
        [ttnn.CoreRange(c, c) for c in sender_cores]
    )

    sender_receiver_mapping = list(zip(sender_cores, receiver_cores))

    logger.info(
        "GLM prefetcher core layout: {} senders, {} receivers, worker grid {}",
        len(sender_cores), len(receiver_cores), worker_core_range_set,
    )

    return (
        sender_cores,
        dram_cores,
        sender_core_range_set,
        receiver_cores,
        worker_core_range_set,
        sender_receiver_mapping,
    )


class Glm4MoePrefetcherSetup:
    """DRAM weight prefetcher for GLM-4.7 decode on Galaxy Wormhole.

    Registers attention and/or MoE weights for async prefetch during decode.
    """

    def __init__(
        self,
        mesh_device,
        n_tensors_per_layer: int,
        n_layers: int,
    ):
        self.mesh_device = mesh_device
        self.n_tensors = n_tensors_per_layer
        self.n_layers = n_layers

        (
            self.sender_cores,
            self.dram_cores,
            self.sender_core_range_set,
            self.receiver_cores,
            self.worker_core_range_set,
            self.sender_receiver_mapping,
        ) = get_glm_core_ranges()

        # Global CB — sized for double-buffering the largest weight
        # QKV weight: [5120, 1792] BF8 = 4.6 MB. At tile size 32×32×1B = 1024B,
        # that's ~4500 tiles. Double buffer = 9000 tiles × 1088 bytes = ~9.8 MB.
        # Start conservative with the Llama size (728 tiles).
        self.global_cb_size = 728 * 1088
        self.global_circular_buffer = None

        # SubDevice setup
        self.prefetcher_sub_device = ttnn.SubDevice([self.sender_core_range_set])
        self.worker_sub_device = ttnn.SubDevice([self.worker_core_range_set])
        self.prefetcher_sub_device_id = ttnn.SubDeviceId(0)
        self.worker_sub_device_id = ttnn.SubDeviceId(1)

        self.mesh_sub_device_manager_id = mesh_device.create_sub_device_manager(
            [self.prefetcher_sub_device, self.worker_sub_device], 0
        )
        mesh_device.load_sub_device_manager(self.mesh_sub_device_manager_id)
        mesh_device.set_sub_device_stall_group(
            [self.prefetcher_sub_device_id, self.worker_sub_device_id]
        )

        self.tensors = []
        self.tensor_addrs = []

        logger.info(
            "Glm4MoePrefetcherSetup: n_tensors={}, n_layers={}, global_cb_size={}",
            n_tensors_per_layer, n_layers, self.global_cb_size,
        )

    def create_global_cb(self):
        """Lazy-create the Global Circular Buffer (must be called before decode)."""
        if self.global_circular_buffer is None:
            self.global_circular_buffer = ttnn.create_global_circular_buffer(
                self.mesh_device,
                self.sender_receiver_mapping,
                self.global_cb_size,
            )
            logger.info("Global CB created, size={}", self.global_cb_size)

    def insert_tensor(self, tensor: ttnn.Tensor):
        """Register a weight tensor for prefetching."""
        self.tensors.append(tensor)
        self.tensor_addrs.append(tensor.buffer_address())

    def get_input_tensors(self):
        """Build the address tensor and return [weights..., addr_tensor]."""
        assert len(self.tensor_addrs) == self.n_tensors * self.n_layers, (
            f"Expected {self.n_tensors * self.n_layers} addresses, "
            f"got {len(self.tensor_addrs)}"
        )

        tensor_addrs = torch.tensor(self.tensor_addrs)
        tensor_addrs = tensor_addrs.repeat(len(self.dram_cores), 1)
        tensor_addrs_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                self.sender_core_range_set,
                [tensor_addrs.shape[0] // len(self.dram_cores), tensor_addrs.shape[1]],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )
        tt_tensor_addrs = ttnn.as_tensor(
            tensor_addrs,
            device=self.mesh_device,
            dtype=ttnn.uint32,
            memory_config=tensor_addrs_mem_config,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

        return self.tensors[: self.n_tensors] + [tt_tensor_addrs]

    def start_prefetch(self):
        """Start async DRAM prefetch. Call before decode loop."""
        self.create_global_cb()
        tt_tensors = self.get_input_tensors()
        garbage = ttnn.dram_prefetcher(
            tt_tensors,
            num_layers=self.n_layers,
            global_cb=self.global_circular_buffer,
        )
        # Only stall on worker cores (prefetcher runs independently)
        self.mesh_device.set_sub_device_stall_group([self.worker_sub_device_id])
        return garbage

    def stop_prefetch(self, garbage):
        """Stop prefetch and clean up."""
        ttnn.deallocate(garbage)
        self.mesh_device.set_sub_device_stall_group(
            [self.prefetcher_sub_device_id, self.worker_sub_device_id]
        )
