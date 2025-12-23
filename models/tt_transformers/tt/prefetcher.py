import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Union

import torch
from loguru import logger

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.utility_functions import is_blackhole, is_wormhole_b0


@dataclass(frozen=True)
class PrefetcherCoreConstants:
    """
    Stores all the column number constants used in PrefetcherCoreConfig.
    """

    LEFT_START_COL: int = 1
    LEFT_END_COL_WORMHOLE_B0: int = 4
    LEFT_END_COL_BLACKHOLE: int = 6
    RIGHT_START_COL_WORMHOLE_B0: int = 5
    RIGHT_START_COL_BLACKHOLE: int = 8
    RIGHT_END_COL_WORMHOLE_B0: int = 8
    RIGHT_END_COL_BLACKHOLE: int = 12
    SENDER_COL_WORMHOLE_B0: int = 0
    SENDER_COL_BLACKHOLE: int = 7


@dataclass
class PrefetcherCoreConfig:
    """
    Defines the core locations of the sender cores and receiver cores of the prefetcher
    """

    num_receiver_cores: int
    mesh_device: ttnn.MeshDevice

    def __post_init__(self):
        constants = PrefetcherCoreConstants
        left_start_col = constants.LEFT_START_COL
        left_end_col = constants.LEFT_END_COL_WORMHOLE_B0 if is_wormhole_b0() else constants.LEFT_END_COL_BLACKHOLE
        right_start_col = (
            constants.RIGHT_START_COL_WORMHOLE_B0 if is_wormhole_b0() else constants.RIGHT_START_COL_BLACKHOLE
        )
        right_end_col = constants.RIGHT_END_COL_WORMHOLE_B0 if is_wormhole_b0() else constants.RIGHT_END_COL_BLACKHOLE

        def get_sender_range(active: Optional[bool] = None):
            left_sender_range = ([] if active == False else [0, 3, 7, 9]) + (
                [] if active == True else [1, 2, 4, 6, 5, 8]
            )
            right_sender_range = ([] if active == False else [1, 4, 6, 9] + [] if is_blackhole() else [5, 6, 7, 9]) + (
                [] if active == True else [2, 3, 5, 7, 8] if is_blackhole() else [3, 8]
            )
            return left_sender_range, right_sender_range

        def get_receiver_range(active: Optional[bool] = None):
            left_recv_range = (
                [] if active == False else list(range(left_start_col, self.num_receiver_cores + left_start_col))
            ) + ([] if active == True else list(range(self.num_receiver_cores + left_start_col, left_end_col)))
            right_recv_range = (
                [] if active == False else list(range(right_start_col, self.num_receiver_cores + right_start_col))
            ) + ([] if active == True else list(range(self.num_receiver_cores + right_start_col, right_end_col)))
            return left_recv_range, right_recv_range

        # Prefetcher sender cores (cores adjacent to dram cores/banks)
        def wh_sender_cores(active: Optional[bool] = None):
            self.left_sender_range, self.right_sender_range = [list(r) for r in get_sender_range(active)]
            return [ttnn.CoreCoord(0, i) for i in self.left_sender_range] + [
                ttnn.CoreCoord(4, i) for i in self.right_sender_range
            ]

        def bh_sender_cores(active: Optional[bool] = None):
            self.left_sender_range, self.right_sender_range = [list(r) for r in get_sender_range(active)]
            return (
                [ttnn.CoreCoord(0, i) for i in self.left_sender_range[:4]]
                + [ttnn.CoreCoord(7, i) for i in self.right_sender_range[:4]]
                + [ttnn.CoreCoord(0, i) for i in self.left_sender_range[4:]]
                + [ttnn.CoreCoord(7, i) for i in self.right_sender_range[4:]]
            )

        # Prefetcher receiver cores (num_receiver_cores worker cores adjacent to sender cores)
        def wh_receiver_cores(sender_active: Optional[bool] = None, receiver_active: Optional[bool] = None):
            self.left_sender_range, self.right_sender_range = get_sender_range(sender_active)
            self.left_recv_range, self.right_recv_range = get_receiver_range(receiver_active)
            return [
                ttnn.CoreRange(ttnn.CoreCoord(self.left_recv_range[0], i), ttnn.CoreCoord(self.left_recv_range[-1], i))
                for i in self.left_sender_range
            ] + [
                ttnn.CoreRange(
                    ttnn.CoreCoord(self.right_recv_range[0], i), ttnn.CoreCoord(self.right_recv_range[-1], i)
                )
                for i in self.right_sender_range
            ]

        def bh_receiver_cores(sender_active: Optional[bool] = None, receiver_active: Optional[bool] = None):
            self.left_sender_range, self.right_sender_range = get_sender_range(sender_active)
            self.left_recv_range, self.right_recv_range = get_receiver_range(receiver_active)
            return (
                [
                    ttnn.CoreRange(
                        ttnn.CoreCoord(self.left_recv_range[0], i), ttnn.CoreCoord(self.left_recv_range[-1], i)
                    )
                    for i in self.left_sender_range[:4]
                ]
                + [
                    ttnn.CoreRange(
                        ttnn.CoreCoord(self.right_recv_range[0], i), ttnn.CoreCoord(self.right_recv_range[-1], i)
                    )
                    for i in self.right_sender_range[:4]
                ]
                + [
                    ttnn.CoreRange(
                        ttnn.CoreCoord(self.left_recv_range[0], i), ttnn.CoreCoord(self.left_recv_range[-1], i)
                    )
                    for i in self.left_sender_range[4:]
                ]
                + [
                    ttnn.CoreRange(
                        ttnn.CoreCoord(self.right_recv_range[0], i), ttnn.CoreCoord(self.right_recv_range[-1], i)
                    )
                    for i in self.right_sender_range[4:]
                ]
            )

        self.sender_cores = wh_sender_cores if is_wormhole_b0() else bh_sender_cores
        self.receiver_cores = wh_receiver_cores if is_wormhole_b0() else bh_receiver_cores


### Helper class to manage subdevices for the Prefetcher
# The class PrefetcherSubDevice provides an interface for creating subdevices
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
        self.max_num_receiver_cores = 4 if is_blackhole() else 2

        # Max tensor block size is the largest block size of a tensor in bytes (1 block = tensor volume / (tile size * tile size) // (num_receiver_cores * num_reader_cores))
        self.max_tensor_block_size = 0
        self.ring_size = self.num_receiver_cores * self.mesh_device.dram_grid_size().x
        assert (
            self.num_receiver_cores <= self.max_num_receiver_cores
        ), f"Number of receiver cores {self.num_receiver_cores} is greater than the maximum number of receiver cores {self.max_num_receiver_cores}"
        self.width_cores = self.mesh_device.compute_with_storage_grid_size().x
        self.height_cores = self.mesh_device.compute_with_storage_grid_size().y
        # Only ring size of 24 has been tested on WH

        ### Prefetcher HardCoded Core Ranges (i dont like this)
        self.all_core_range_set = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(self.width_cores - 1, self.height_cores - 1))]
        )

        self.all_worker_cores_range_set = ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(4, 9)),
                ttnn.CoreRange(ttnn.CoreCoord(8, 0), ttnn.CoreCoord(11, 9)),
            ]
        )

        ### Prefetched Tensors
        self.callbacks = []
        self.prefetched_tensors = []
        self.prefetched_tensor_addr = []
        self.prefetched_tt_addr_tensor = None

        ### Core Ranges
        self.sender_cores = None
        self.receiver_cores = None
        self.all_cores = None
        self.mode = "prefill"

    # Todo: add a note to that weights are prefetched in the order of the construction of the module
    def register_callback(self, callback: Callable[[], None]):
        self.callbacks.append(callback)

    def get_optimal_receiver_cores(self):
        if self.mesh_device.shape == ttnn.MeshShape([1, 2]):
            return 4
        elif self.mesh_device.shape == ttnn.MeshShape([1, 8]):
            return 1
        else:
            raise ValueError(
                f"Provided mesh device shape {self.mesh_device.shape} is not supported, only [1,2] and [1,8] are supported"
            )

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
        self.sender_cores = PrefetcherCoreConfig(
            num_receiver_cores=self.num_receiver_cores, mesh_device=self.mesh_device
        ).sender_cores

        self.receiver_cores = PrefetcherCoreConfig(
            num_receiver_cores=self.num_receiver_cores, mesh_device=self.mesh_device
        ).receiver_cores
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
        logger.info(f"  All cores: {self.all_cores}")
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
            len(self.prefetched_tensor_addr) == self.num_tensors * self.num_layers,
            "No tensor addresses have been inserted",
        )
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
        self.max_tensor_block_size = max(
            (math.ceil(tensor.volume() / (ttnn.TILE_SIZE * ttnn.TILE_SIZE)) // (self.ring_size))
            * bytes_in_tile[tensor.dtype],
            self.max_tensor_block_size,
        )
        self.prefetched_tensors.append(tensor)
        self.prefetched_tensor_addr.append(tensor.buffer_address())
        logger.info(
            f"Inserted tensor {tensor.shape} into prefetcher, total number of tensors: {len(self.prefetched_tensor_addr)}"
        )

    def prefetch(self):
        """
        Inserts the tensors to be prefetched in a queue
        The tensors are prefetched in the order of the registration of the callbacks
        """
        for callback in self.callbacks:
            callback()

    def run(self):
        """
        Start prefetching weights into global CB with dram_prefetcher op
        """
        # Create global cb buffer if it was not
        if self.global_cb is None:
            self.global_cb_size = self.max_tensor_block_size  # Double buffered weights
            self.global_cb = ttnn.create_global_circular_buffer(
                self.mesh_device,
                self.sender_receiver_mapping,
                self.global_cb_size,
            )

        # Create address tensor if it was not created yet
        if self.prefetched_tt_addr_tensor is None:
            self.prefetched_tt_addr_tensor = self.create_address_tensor()

        # Run prefetcher op
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
