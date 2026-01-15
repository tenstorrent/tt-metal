# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Union

import torch
from loguru import logger

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.utility_functions import is_blackhole


@dataclass
class PrefetcherCoreConfig:
    """
    Defines the core locations of the sender and receiver cores for the prefetcher.

    Architecture layout:
    - Blackhole: Left sender column=0, Right sender column=7
                 Left receiver columns=1-6, Right receiver columns=8-10
    - Wormhole:  Left sender column=0, Right sender column=4
                 Left receiver columns=1-3, Right receiver columns=5-6
    """

    num_receiver_cores: int
    mesh_device: ttnn.MeshDevice

    # Sender core rows that are adjacent to DRAM banks (different for left and right sides)
    # Active rows are the ones adjacent to DRAM banks, inactive are the remaining rows
    SENDER_ROWS = {
        "blackhole": {
            "left": {"active": [0, 3, 7, 9], "inactive": [1, 2, 4, 5, 6, 8]},
            "right": {"active": [1, 4, 6, 9], "inactive": [0, 2, 3, 5, 7, 8]},
        },
        "wormhole": {
            "left": {"active": [0, 4, 5, 9], "inactive": [1, 2, 3, 6, 7, 8]},
            "right": {"active": [0, 1, 2, 4, 5, 6, 7, 9], "inactive": [3, 8]},
        },
    }

    # Sender core columns (left side near banks 0-3, right side near banks 4-7)
    SENDER_COLS = {
        "blackhole": {"left": 0, "right": 7},
        "wormhole": {"left": 0, "right": 4},
    }

    # Receiver core column ranges (start_col, end_col exclusive)
    RECEIVER_COLS = {
        "blackhole": {"left": (1, 7), "right": (8, 11)},
        "wormhole": {"left": (1, 4), "right": (5, 7)},
    }

    def __post_init__(self):
        arch = "blackhole" if is_blackhole() else "wormhole"
        self._sender_rows = self.SENDER_ROWS[arch]
        self._sender_cols = self.SENDER_COLS[arch]
        self._receiver_cols = self.RECEIVER_COLS[arch]

    def _get_sender_rows(self, active: Optional[bool], side: str) -> List[int]:
        """Get sender rows based on active filter for a specific side."""
        if active is True:
            return self._sender_rows[side]["active"]
        elif active is False:
            return self._sender_rows[side]["inactive"]
        else:  # None - return all rows
            return self._sender_rows[side]["active"] + self._sender_rows[side]["inactive"]

    def _get_receiver_col_range(self, active: Optional[bool], side: str) -> tuple:
        """Get receiver column range (start, end) based on active filter."""
        start, end = self._receiver_cols[side]
        if active is True:
            return (start, start + self.num_receiver_cores)
        elif active is False:
            return (start + self.num_receiver_cores, end)
        else:  # None - return all columns
            return (start, end)

    def sender_cores(self, active: Optional[bool] = None) -> List[ttnn.CoreCoord]:
        """
        Get sender cores (cores adjacent to DRAM banks).

        Args:
            active: If True, return only active sender cores (one per DRAM bank).
                   If False, return only inactive sender cores.
                   If None, return all sender cores (active first, then inactive).

        The order is: left_active, right_active, left_inactive, right_inactive.
        This ensures the first num_dram_banks cores are the active ones, which
        matches the sub-device configuration.
        """
        left_col = self._sender_cols["left"]
        right_col = self._sender_cols["right"]

        if active is True:
            left_active = self._sender_rows["left"]["active"]
            right_active = self._sender_rows["right"]["active"]
            return [ttnn.CoreCoord(left_col, r) for r in left_active] + [
                ttnn.CoreCoord(right_col, r) for r in right_active
            ]
        elif active is False:
            left_inactive = self._sender_rows["left"]["inactive"]
            right_inactive = self._sender_rows["right"]["inactive"]
            return [ttnn.CoreCoord(left_col, r) for r in left_inactive] + [
                ttnn.CoreCoord(right_col, r) for r in right_inactive
            ]
        else:  # None - return all: active first, then inactive
            left_active = self._sender_rows["left"]["active"]
            right_active = self._sender_rows["right"]["active"]
            left_inactive = self._sender_rows["left"]["inactive"]
            right_inactive = self._sender_rows["right"]["inactive"]
            return (
                [ttnn.CoreCoord(left_col, r) for r in left_active]
                + [ttnn.CoreCoord(right_col, r) for r in right_active]
                + [ttnn.CoreCoord(left_col, r) for r in left_inactive]
                + [ttnn.CoreCoord(right_col, r) for r in right_inactive]
            )

    def receiver_cores(
        self, sender_active: Optional[bool] = None, receiver_active: Optional[bool] = None
    ) -> List[ttnn.CoreRange]:
        """
        Get receiver core ranges (worker cores adjacent to sender cores).

        Each sender core has a horizontal strip of receiver cores on the same row.

        Args:
            sender_active: Filter which sender rows to create receiver ranges for.
            receiver_active: Filter which receiver columns to include.

        The order matches sender_cores: left_active, right_active, left_inactive, right_inactive.
        """
        left_recv = self._get_receiver_col_range(receiver_active, "left")
        right_recv = self._get_receiver_col_range(receiver_active, "right")

        def make_range(col_start, col_end, row):
            return ttnn.CoreRange(ttnn.CoreCoord(col_start, row), ttnn.CoreCoord(col_end - 1, row))

        if sender_active is True:
            left_active = self._sender_rows["left"]["active"]
            right_active = self._sender_rows["right"]["active"]
            return [make_range(*left_recv, r) for r in left_active] + [make_range(*right_recv, r) for r in right_active]
        elif sender_active is False:
            left_inactive = self._sender_rows["left"]["inactive"]
            right_inactive = self._sender_rows["right"]["inactive"]
            return [make_range(*left_recv, r) for r in left_inactive] + [
                make_range(*right_recv, r) for r in right_inactive
            ]
        else:  # None - return all: active first, then inactive
            left_active = self._sender_rows["left"]["active"]
            right_active = self._sender_rows["right"]["active"]
            left_inactive = self._sender_rows["left"]["inactive"]
            right_inactive = self._sender_rows["right"]["inactive"]
            return (
                [make_range(*left_recv, r) for r in left_active]
                + [make_range(*right_recv, r) for r in right_active]
                + [make_range(*left_recv, r) for r in left_inactive]
                + [make_range(*right_recv, r) for r in right_inactive]
            )


### Helper class to manage subdevices for the Prefetcher
# The class PrefetcherSubDevice provides an interface for creating subdevices is only managed by the prefetcher module
class PrefetcherSubDevice:
    def __init__(self, mesh_device):
        self.mesh_device = mesh_device
        self.num_sub_devices = 0
        self.sub_devices: List[ttnn.SubDevice] = []
        self.sub_devices_id: List[ttnn.SubDeviceId] = []

    def add_sub_device(self, core_range_set: ttnn.CoreRangeSet):
        self.sub_devices.append(ttnn.SubDevice([core_range_set]))
        self.sub_devices_id.append(ttnn.SubDeviceId(len(self.sub_devices_id)))

    def init_sub_device_manager(self):
        assert len(self.sub_devices) > 0, "No subdevices have been created. Cannot create sub device manager."
        self.manager_id = self.mesh_device.create_sub_device_manager(self.sub_devices, 0)
        self.mesh_device.load_sub_device_manager(self.manager_id)
        self.mesh_device.set_sub_device_stall_group(self.sub_devices_id)


class Prefetcher(LightweightModule):
    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        num_tensors: int,
        num_layers: int,
    ):
        """
        Prefetcher class that prefetches tensors from DRAM to
        """
        ### Device, Global CB, Parameters
        self.global_cb = None
        self.mesh_device = mesh_device
        self.num_tensors = num_tensors
        self.num_layers = num_layers
        self.enable_performance_mode = True
        self.worker_sub_device_id = None
        self.global_cb_size = 0
        self.num_receiver_cores = self.get_optimal_receiver_cores()

        # Max tensor block size is the largest block size of a tensor in bytes (1 block = tensor volume / (tile size * tile size) // (num_receiver_cores * num_reader_cores))
        self.max_tensor_block_size = 0
        self.ring_size = self.num_receiver_cores * self.mesh_device.dram_grid_size().x
        self.width_cores = self.mesh_device.compute_with_storage_grid_size().x
        self.height_cores = self.mesh_device.compute_with_storage_grid_size().y

        ### Core Config
        self.core_config = PrefetcherCoreConfig(
            num_receiver_cores=self.num_receiver_cores, mesh_device=self.mesh_device
        )

        ### Prefetcher Hardcoded Core Ranges
        self.all_core_range_set = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(self.width_cores - 1, self.height_cores - 1))]
        )

        # Remaining worker core ranges for the worker sub device
        left_range = self.core_config._receiver_cols["left"]
        right_range = self.core_config._receiver_cols["right"]
        self.all_worker_cores_range_set = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(left_range[0], 0), ttnn.CoreCoord(left_range[1] - 1, 9))]
            + [ttnn.CoreRange(ttnn.CoreCoord(right_range[0], 0), ttnn.CoreCoord(right_range[1] - 1, 9))]
        )

        ### Prefetched Tensors
        self.callbacks = []
        self.prefetched_tensors = []
        self.prefetched_tensor_addr = []
        self.prefetched_tt_addr_tensor = None

        ### Core Ranges
        self.sender_cores = None
        self.receiver_cores = None
        self.mode = "prefill"

    # NOTE: DRAM prefetched weights are prefetched in the order of the construction of the module
    def register_callback(self, callback: Callable[[], None]):
        self.callbacks.append(callback)

    # Mapping from mesh shape (as tuple) to optimal number of receiver cores
    OPTIMAL_RECEIVER_CORES = {
        (1, 1): 2,
        (1, 2): 2,
        (1, 4): 2,
        (1, 8): 1,
    }

    def get_optimal_receiver_cores(self):
        mesh_shape = tuple(self.mesh_device.shape)
        if mesh_shape not in self.OPTIMAL_RECEIVER_CORES:
            supported = list(self.OPTIMAL_RECEIVER_CORES.keys())
            raise ValueError(f"Mesh shape {mesh_shape} is not supported. Supported shapes: {supported}")
        return self.OPTIMAL_RECEIVER_CORES[mesh_shape]

    def to_core_range_set(
        self, cores: Union[List[ttnn.CoreCoord], List[ttnn.CoreRange]], return_list: bool = False
    ) -> ttnn.CoreRangeSet:
        if isinstance(cores[0], ttnn.CoreCoord):
            return ttnn.CoreRangeSet([ttnn.CoreRange(core, core) for core in cores])
        elif isinstance(cores[0], ttnn.CoreRange):
            if return_list:  # Return a list of CoreRangeSets (used for creating sender receiver mapping)
                return [ttnn.CoreRangeSet([core]) for core in cores]
            else:  # Return a single CoreRangeSet
                return ttnn.CoreRangeSet(cores)
        else:
            raise ValueError(f"Provided cores {cores} is not a list of CoreCoords or CoreRanges")

    def init(self, mode: str = "decode") -> None:
        """
        Initializes the prefetcher
        Args:
            mode: The mode to run the prefetcher in, either "decode" or "prefill"
        """
        self.mode = "decode" if mode is None else mode
        assert self.mode in [
            "decode",
            "prefill",
        ], f"Provided mode {mode} is not supported, only `decode` and `prefill` are supported"

        # Get the sender and receiver cores
        # Create a single config instance to ensure consistent state
        self.sender_cores = self.core_config.sender_cores
        self.receiver_cores = self.core_config.receiver_cores

        self.sender_receiver_mapping = list(
            zip(
                self.sender_cores(),
                self.to_core_range_set(self.receiver_cores(sender_active=None, receiver_active=True), return_list=True),
            )
        )
        match mode:
            case "decode":
                self.prefetcher_sub_device = PrefetcherSubDevice(self.mesh_device)
                self.prefetcher_sub_device.add_sub_device(self.to_core_range_set(self.sender_cores(active=True)))
                self.prefetcher_sub_device.add_sub_device(self.all_worker_cores_range_set)
                self.prefetcher_sub_device.init_sub_device_manager()
            case "prefill":
                self.prefetcher_sub_device = PrefetcherSubDevice(self.mesh_device)
                self.prefetcher_sub_device.add_sub_device(self.all_core_range_set)
                self.prefetcher_sub_device.init_sub_device_manager()

        self.worker_sub_device_id = self.prefetcher_sub_device.sub_devices_id[-1]
        logger.info("=" * 50)
        logger.info("[Prefetcher Initialization]")
        logger.info(f"  Mode: {mode}")
        logger.info(f"  Sender cores: {self.sender_cores(active=True)}")
        logger.info(f"  Receiver cores: {self.receiver_cores(sender_active=None, receiver_active=True)}")
        logger.info(f"  Number of receiver cores: {self.num_receiver_cores}")
        logger.info(f"  Number of tensors to prefetch: {self.num_tensors}")
        logger.info(f"  Number of layers: {self.num_layers}")
        logger.info("=" * 50)

    def create_address_tensor(self):
        """
        Creates a ttnn tensor which holds the addresses of the tensors to be prefetched
        The addresses are replicated on each sender core
        """
        assert (
            len(self.prefetched_tensor_addr) == self.num_tensors * self.num_layers
        ), "No tensor addresses have been inserted"

        tensor_addrs = torch.tensor(self.prefetched_tensor_addr)
        tensor_addrs = tensor_addrs.repeat(self.mesh_device.dram_grid_size().x, 1)
        tensor_addrs_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                self.to_core_range_set(self.sender_cores(active=True)),
                [tensor_addrs.shape[0] // self.mesh_device.dram_grid_size().x, tensor_addrs.shape[1]],
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

    def insert_tensor(self, tensor: ttnn.Tensor):
        """
        Populates the tensor addressess that need to be prefetched
        """
        bytes_in_tile = {ttnn.bfloat4_b: 576, ttnn.bfloat8_b: 1088, ttnn.bfloat16: 2048}
        if tensor.volume() % self.ring_size != 0:
            raise ValueError(
                f"Tensor volume ({tensor.volume()}) must be divisible by num_receiver_cores * num_reader_cores ({self.num_receiver_cores * self.width_cores}) for prefetcher."
            )
        if not tensor.is_sharded() or tensor.memory_config().buffer_type != ttnn.BufferType.DRAM:
            raise ValueError(
                f"Tensor must be DRAM sharded for prefetcher. Got sharded={tensor.is_sharded()}, "
                f"buffer_type={tensor.memory_config().buffer_type}"
            )
        self.max_tensor_block_size = max(
            (math.ceil(tensor.volume() / (ttnn.TILE_SIZE * ttnn.TILE_SIZE)) // (self.ring_size))
            * bytes_in_tile[tensor.dtype],
            self.max_tensor_block_size,
        )
        self.prefetched_tensors.append(tensor)
        self.prefetched_tensor_addr.append(tensor.buffer_address())
        logger.info(
            f"Inserted tensor of shape {tensor.shape} into prefetcher, total number of tensors in prefetcher queue: {len(self.prefetched_tensor_addr)}"
        )

    def prefetch(self):
        """
        Inserts the tensors to be prefetched in a queue
        The tensors are prefetched in the order of the registration of the callbacks
        NOTE: This only needs to be called if a callback is registered for inserting tensors
        """
        assert (
            len(self.callbacks) > 0
        ), "No tensors insertion callbacks have been inserted into the prefetcher queue. Cannot prefetch an empty queue"
        for callback in self.callbacks:
            callback()

    def run(self):
        """
        Start prefetching weights into global CB with dram_prefetcher op
        """
        # Create global cb buffer if it was not yet created.
        if self.global_cb is None:
            self.global_cb_size = self.max_tensor_block_size
            self.global_cb = ttnn.create_global_circular_buffer(
                self.mesh_device,
                self.sender_receiver_mapping,
                self.global_cb_size,
            )

        # Create address tensor if it was not created yet
        if self.prefetched_tt_addr_tensor is None:
            self.prefetched_tt_addr_tensor = self.create_address_tensor()

        # Run prefetcher op (prefetcher op will start asynchronously prefetching weights until prefetcher.stop() is called)
        self.garbage = ttnn.dram_prefetcher(
            self.prefetched_tensors[: self.num_tensors] + [self.prefetched_tt_addr_tensor],
            num_layers=self.num_layers,
            global_cb=self.global_cb,
            enable_performance_mode=self.enable_performance_mode,
        )
        # Set worker sub device stall group
        self.mesh_device.set_sub_device_stall_group([self.prefetcher_sub_device.sub_devices_id[-1]])
        return

    def stop(self):
        ttnn.deallocate(self.garbage)
        return
