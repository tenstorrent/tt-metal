from dataclasses import dataclass
from typing import List

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.utility_functions import is_blackhole, is_wormhole_b0


@dataclass
class PrefetcherCoreConfig:
    """
    Defines the core locations of the sender cores and receiver cores of the prefetcher
    For wormhole,
    For blackhole,
    """

    wh_sender_cores = [ttnn.CoreCoord(0, i) for i in [0, 4, 5, 9]] + [
        ttnn.CoreCoord(4, i) for i in [0, 1, 2, 4, 5, 6, 7, 9]
    ]
    bh_sender_cores = [ttnn.CoreCoord(0, i) for i in range(10)] + [ttnn.CoreCoord(6, i) for i in range(10)]
    wh_receiver_cores = [ttnn.CoreCoord(j, i) for i in [0, 4, 5, 9] for j in [1, 2]] + [
        ttnn.CoreCoord(j, i) for i in [0, 1, 2, 4, 5, 6, 7, 9] for j in [5, 6]
    ]
    bh_receiver_cores = [ttnn.CoreCoord(j, i) for i in range(10) for j in [1, 2]] + [
        ttnn.CoreCoord(j, i) for i in range(10) for j in [7, 8]
    ]


### Helper class to manage subdevices for the Prefetcher
# The class PrefetcherSubDevice provides an interface for creating subdevices
class PrefetcherSubDevice:
    def __init__(self, mesh_device):
        self.mesh_device = mesh_device
        self.num_sub_devices = 0
        self.sub_devices: List[ttnn.SubDevice] = None
        self.sub_devices_id: List[ttnn.SubDeviceId] = None

    def add_sub_device(self, core_range_set: ttnn.CoreRangeSet):
        self.sub_devices.append(ttnn.SubDevice([core_range_set]))
        self.sub_devices_id.append(ttnn.SubDeviceId(len(self.sub_devices_id)))

    def init_sub_device_manager(self):
        assert len(self.subdevices) > 0, "No subdevices have been created. Cannot create sub device manager."
        self.manager_id = self.mesh_device.create_sub_device_manager(self.sub_devices, 0)
        self.mesh_device.load_sub_device_manager(self.manager_id)
        self.mesh_device.set_sub_device_stall_group(self.sub_devices_id)


class Prefetcher(LightweightModule):
    def __init__(self, mesh_device: ttnn.MeshDevice, num_tensors: int, num_layers: int, mode: str):
        """
        Prefetcher class that prefetches tensors from DRAM to
        """
        ### Device, Global CB, Parameters
        self.global_cb = glv
        self.mesh_device = mesh_device
        self.num_tensors = num_tensors
        self.num_layers = num_layers
        self.enable_performance_mode = False

        ### Prefetcher Subdevices
        self.prefetcher_sub_device = PrefetcherSubDevice(self.mesh_device)

        ### Prefetched Tensors
        self.prefetched_tensors = []
        self.prefetched_tensor_addr = []

        ### Core Ranges
        self.sender_cores = None
        self.receiver_cores = None
        self.all_cores = None

    def init(self, mode: str = "decode") -> None:
        """ """
        if mode == "decode":
            self.prefetcher_sub_device.add_sub_device(self.sender_cores)
            self.prefetcher_sub_device.add_sub_device(self.receiver_cores)
        if mode == "prefill":
            self.prefetcher_sub_device.add_sub_device(self.all_cores)
        else:
            raise ValueError(f"Provided mode {mode} is not supported, only `prefill` and `decode` are supported")

    def create_prefetcher_cores(self):
        if is_blackhole():
            self.sender_cores = PrefetcherCoreConfig.bh_sender_cores
            self.receiver_cores = PrefetcherCoreConfig.bh_receiver_cores
        if is_wormhole_b0():
            self.sender_cores = PrefetcherCoreConfig.wh_sender_cores
            self.receiver_cores = PrefetcherCoreConfig.wh_receiver_cores

    def create_address_tensor(self):
        """
        Creates a ttnn tensor which holds the addresses of the tensors to be prefetched
        The addresses are replicated on each sender core
        """
        tensor_addrs = torch.tensor(self.prefetched_tensor_addr)
        tensor_addrs = tensor_addrs.repeat(len(self.dram_cores), 1)
        tensor_addrs_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                self.sender_cores,
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

    def insert_tensor(self, tensor: ttnn.Tensor):
        """
        Populates the tensor addressess that need to be prefetched
        """
        self.prefetched_tensors.append(tensor)
        self.prefetched_tensor_addr.append(tensor.buffer_address())

    def run(self):
        """
        Start prefetching weights into global CB with dram_prefetcher op
        """
        # Create global cb buffer if it was not
        if self.global_cb is None:
            self.global_cb = ttnn.create_global_circular_buffer(
                self.mesh_device,
                self.sender_receiver_mapping,
                self.global_cb_size,
            )
        # Run prefetcher op
        self.garbage = ttnn.dram_prefetcher(
            self.prefetched_tensors[: self.num_tensors] + [self.prefetched_tensor_addr],
            num_layers=self.num_layers,
            global_cb=self.global_cb,
            enable_performance_mode=self.enable_performance_mode,
        )
        # Set worker sub device stall group
        self.mesh_device.set_sub_device_stall_group(self.prefetcher_sub_device.sub_devices_id[-1])
        return

    def stop(self):
        ttnn.deallocate(self.garbage)
        return
