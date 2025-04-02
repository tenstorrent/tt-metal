# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import ttnn
import torch
from loguru import logger
from models.common.lightweightmodule import LightweightModule
from tests.ttnn.unit_tests.operations.ccl.test_ccl_common import (
    create_and_load_sub_device_manager_with_fabric_interface,
)
from tests.ttnn.unit_tests.operations.prefetcher_common import get_core_ranges


def get_buffer_address(tensor):
    addr = []
    for i, ten in enumerate(ttnn.get_device_tensors(tensor)):
        addr.append(ten.buffer_address())
        if len(addr) > 0:
            assert addr[i - 1] == addr[i], f"Expected {addr[i-1]} == {addr[i]}"
    return addr[0]


class TtLlamaPrefetcherSetup(LightweightModule):
    def __init__(self, mesh_device, n_tensors, n_layers, mode="decode"):
        """
        - sub devices
        - global cb
        - helper functions to get the weight addresses
        """
        logger.info("Running TtLlamaPrefetcherSetup")

        self.mesh_device = mesh_device
        self.n_tensors = n_tensors
        self.n_layers = n_layers

        ###### Set up GlobalCB ######
        num_reader_cores = 12
        num_global_cb_receivers = 2

        (
            self.active_sender_cores,
            self.dram_cores,
            self.all_sender_cores,
            self.active_receiver_cores_list,
            self.all_receiver_cores,
            self.worker_cores_range_set,
            self.mm_optimised_ring_cores,
            self.hop_grid,
        ) = get_core_ranges(num_reader_cores, num_global_cb_receivers, is_functional_test=False)

        ##### Set up the input tensors #####
        self.dram_core_range_set = ttnn.CoreRangeSet(
            [ttnn.CoreRange(core_coord, core_coord) for core_coord in self.dram_cores]
        )
        self.sender_core_range_set = ttnn.CoreRangeSet(
            [ttnn.CoreRange(core_coord, core_coord) for core_coord in self.active_sender_cores]
        )

        self.all_core_range_set = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(6, 9))])

        ##### Setup up sub devices #####

        if mode == "prefill":
            self.all_sub_device = ttnn.SubDevice([self.all_core_range_set])
            mesh_sub_device_manager_id = create_and_load_sub_device_manager_with_fabric_interface(
                mesh_device, [self.all_sub_device], 0, 0, True
            )
            self.mesh_sub_device_manager_id = mesh_sub_device_manager_id
            self.all_sub_device_id = ttnn.SubDeviceId(0)
            self.worker_sub_device_id = self.all_sub_device_id
        else:
            ##### Set up the global circular buffer #####
            max_tile_size = 1088
            # Global CB must be large enough to atleast double buffer weights
            # This ensures that back to back matmuls (for eg. in MLP) can run
            # without stalling on the weight prefetch
            # calculated by fitting two largest tensor with extra room, ff2 has 391680B per global CB bank, ff1 has 207360B, plus 16320B gap (one block)
            # TODO: Above calculation is not accurate, need to find a better lower bound
            self.global_cb_size = 600 * 1088
            self.sender_receiver_mapping = list(zip(self.all_sender_cores, self.all_receiver_cores))
            self.global_circular_buffer = ttnn.create_global_circular_buffer(
                self.mesh_device, self.sender_receiver_mapping, self.global_cb_size
            )
            logger.info(f"GlobalCB size {self.global_cb_size}")

            self.prefetcher_sub_device = ttnn.SubDevice([self.sender_core_range_set])
            self.worker_sub_device = ttnn.SubDevice([self.worker_cores_range_set])
            mesh_sub_device_manager_id = create_and_load_sub_device_manager_with_fabric_interface(
                mesh_device, [self.prefetcher_sub_device, self.worker_sub_device], 1, 0, True
            )
            self.mesh_sub_device_manager_id = mesh_sub_device_manager_id
            self.prefetcher_sub_device_id = ttnn.SubDeviceId(0)
            self.worker_sub_device_id = ttnn.SubDeviceId(1)

        self.tensors = []
        self.tensor_addrs = []  # List of buffer addresses

    def buffer_address(self, tensor):
        addr = []
        for i, ten in enumerate(ttnn.get_device_tensors(tensor)):
            addr.append(ten.buffer_address())
            if len(addr) > 0:
                assert addr[i - 1] == addr[i], f"Expected {addr[i-1]} == {addr[i]}"
        return addr[0]

    def insert_tensor(self, tensor: ttnn.Tensor):
        self.tensors.append(tensor)
        self.tensor_addrs.append(self.buffer_address(tensor))

    def get_tensor_addrs(self):
        assert (
            len(self.tensor_addrs) == self.n_tensors * self.n_layers
        ), f"Expected {self.n_tensors * self.n_layers} tensor addresses, got {len(self.tensor_addrs)}"

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

        return tt_tensor_addrs

    def get_input_tensors(self):
        assert (
            len(self.tensors) >= self.n_tensors
        ), f"Expected at least {self.n_tensors} tensors, got {len(self.tensors)}"

        return self.tensors[: self.n_tensors] + [self.get_tensor_addrs()]
