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
from models.tt_transformers.tt.common import Mode

VERIFIED_MODEL_CONFIGS = {
    "Llama-3.2-1B": {"dim": 2048, "hidden_dim": 8192, "n_heads": 32, "n_kv_heads": 8},
    "Llama-3.2-3B": {"dim": 3072, "hidden_dim": 8192, "n_heads": 24, "n_kv_heads": 8},
    "Llama-3.1-8B": {"dim": 4096, "hidden_dim": 14336, "n_heads": 16, "n_kv_heads": 4},
    "Llama-3.3-70B": {"dim": 8192, "hidden_dim": 28672, "n_heads": 64, "n_kv_heads": 8},
    "Qwen3-32B": {"dim": 5120, "hidden_dim": 22016, "n_heads": 40, "n_kv_heads": 8},
    "Qwen3-VL-7B": {"dim": 4096, "hidden_dim": 11008, "n_heads": 32, "n_kv_heads": 8},
    "Qwen3-VL-14B": {"dim": 5120, "hidden_dim": 13824, "n_heads": 40, "n_kv_heads": 8},
    "Qwen3-VL-72B": {"dim": 8192, "hidden_dim": 28672, "n_heads": 64, "n_kv_heads": 8},
    "Gemma3-4B": {"dim": 2560, "hidden_dim": 14336, "n_heads": 20, "n_kv_heads": 20},
    "Gemma3-27B": {"dim": 4608, "hidden_dim": 24576, "n_heads": 32, "n_kv_heads": 8},
}


def is_prefetcher_supported(model_name: str, num_devices: int, ring_size: int = 80) -> bool:
    """
    Check if model weights fit in global CB constraints:
    1. Max 65535 pages (tiles) per CB
    2. CB size must fit in L1 bank with room for input shards and prefetcher CBs

    The largest weight is FF1/FF3: [dim, hidden_dim/num_devices] (N-sharded).

    Args:
        model_name: Name of the model (key in VERIFIED_MODEL_CONFIGS)
        num_devices: number of devices for tensor parallelism
        ring_size: total receiver cores (16 for default, 80 for custom mapping)

    Returns:
        True if weights fit in global CB, False otherwise
    """
    if not is_blackhole() or model_name not in VERIFIED_MODEL_CONFIGS:
        return False

    TILE_SIZE, MAX_CB_PAGES = 32, 65535
    BYTES_PER_TILE_BFP8 = 1088  # bfloat8_b tile size in bytes
    # Conservative L1 limit: account for prefetcher static CBs (~262KB) and input shards
    # Total L1 ~1.46MB, prefetcher CBs end at ~762KB, need room for input shards
    # Use ~500KB as safe limit for global CB per core
    MAX_L1_PER_BANK = 900000

    dim, hidden_dim = VERIFIED_MODEL_CONFIGS[model_name]["dim"], VERIFIED_MODEL_CONFIGS[model_name]["hidden_dim"]
    n_per_device = hidden_dim // num_devices
    n_per_core = math.ceil(n_per_device / ring_size)
    n_per_core_padded = ((n_per_core + TILE_SIZE - 1) // TILE_SIZE) * TILE_SIZE
    n_padded = n_per_core_padded * ring_size
    h_tiles = math.ceil(dim / TILE_SIZE)
    w_tiles = n_padded // TILE_SIZE
    h_tiles_padded = ((h_tiles + ring_size - 1) // ring_size) * ring_size
    tiles_per_core = (h_tiles_padded * w_tiles) // ring_size

    # Check both constraints
    pages_ok = tiles_per_core <= MAX_CB_PAGES
    bytes_per_core = tiles_per_core * BYTES_PER_TILE_BFP8
    l1_ok = bytes_per_core <= MAX_L1_PER_BANK

    return pages_ok and l1_ok


@dataclass
class PrefetcherCoreConfig:
    """
    Core locations for prefetcher sender/receiver cores.

    If receiver_mapping_override is provided, its keys become the sender cores and values
    become their receivers (all treated as "active"). This allows full custom placement.
    Otherwise, uses default architecture-specific sender/receiver layout.
    """

    ARCH_CONFIG = {
        "blackhole": {
            "dram_banks": [1, 3, 2, 0, 5, 7, 6, 4],
            "sender_cols": {"left": 0, "right": 7},
            "sender_rows": {
                "left": {"active": [0, 3, 7, 9], "inactive": [1, 2, 4, 5, 6, 8]},
                "right": {"active": [1, 4, 6, 9], "inactive": [0, 2, 3, 5, 7, 8]},
            },
            "receiver_cols": {"left": (1, 7), "right": (8, 11)},
        },
        "wormhole": {
            "dram_banks": [1, 2, 3, 0, 4, 6, 9, 10, 11, 8, 7, 5],
            "sender_cols": {"left": 0, "right": 4},
            "sender_rows": {
                "left": {"active": [0, 4, 5, 9], "inactive": [1, 2, 3, 6, 7, 8]},
                "right": {"active": [0, 1, 2, 4, 5, 6, 7, 9], "inactive": [3, 8]},
            },
            "receiver_cols": {"left": (1, 4), "right": (5, 7)},
        },
    }

    num_receiver_cores: int
    mesh_device: ttnn.MeshDevice
    receiver_mapping_override: Optional[dict] = None  # {(x,y): [(rx,ry), ...]}

    def __post_init__(self):
        cfg = self.ARCH_CONFIG["blackhole" if is_blackhole() else "wormhole"]
        self._dram_banks = [ttnn.CoreCoord(b, 0) for b in cfg["dram_banks"]]
        self._sender_cols, self._sender_rows = cfg["sender_cols"], cfg["sender_rows"]
        self._receiver_cols = cfg["receiver_cols"]
        self._use_override = self.receiver_mapping_override is not None

        # Process override: keys become senders, values become receivers
        self._override_senders = []  # List[CoreCoord] - ordered sender cores from override
        self._override_receivers = {}  # {(x,y): [CoreCoord]} - receivers per sender
        if self._use_override:
            for k, v in self.receiver_mapping_override.items():
                sender = k if isinstance(k, ttnn.CoreCoord) else ttnn.CoreCoord(k[0], k[1])
                self._override_senders.append(sender)
                key = (sender.x, sender.y)
                self._override_receivers[key] = [
                    c if isinstance(c, ttnn.CoreCoord) else ttnn.CoreCoord(c[0], c[1]) for c in v
                ]

    def _get_rows(self, active: Optional[bool], side: str) -> List[int]:
        rows = self._sender_rows[side]
        if active is True:
            return rows["active"]
        if active is False:
            return rows["inactive"]
        return rows["active"] + rows["inactive"]

    def _get_col_range(self, active: Optional[bool], side: str) -> tuple:
        start, end = self._receiver_cols[side]
        if active is True:
            return (start, start + self.num_receiver_cores)
        if active is False:
            return (start + self.num_receiver_cores, end)
        return (start, end)

    def sender_cores(self, active: Optional[bool] = None) -> List[ttnn.CoreCoord]:
        """Get sender cores. With override, all senders are 'active'. Without, uses default layout."""
        if self._use_override:
            # With override: all senders from override keys, no inactive concept
            return self._override_senders if active is None or active is True else []
        # Default behavior
        lc, rc = self._sender_cols["left"], self._sender_cols["right"]
        if active is True:
            return [ttnn.CoreCoord(lc, r) for r in self._get_rows(True, "left")] + [
                ttnn.CoreCoord(rc, r) for r in self._get_rows(True, "right")
            ]
        if active is False:
            return [ttnn.CoreCoord(lc, r) for r in self._get_rows(False, "left")] + [
                ttnn.CoreCoord(rc, r) for r in self._get_rows(False, "right")
            ]
        return (
            [ttnn.CoreCoord(lc, r) for r in self._get_rows(True, "left")]
            + [ttnn.CoreCoord(rc, r) for r in self._get_rows(True, "right")]
            + [ttnn.CoreCoord(lc, r) for r in self._get_rows(False, "left")]
            + [ttnn.CoreCoord(rc, r) for r in self._get_rows(False, "right")]
        )

    def _get_receivers(self, sender: ttnn.CoreCoord, receiver_active: Optional[bool]) -> List[ttnn.CoreCoord]:
        key = (sender.x, sender.y)
        if self._use_override:
            # With override: return all receivers for this sender (no active/inactive split)
            return self._override_receivers.get(key, [])
        # Default behavior
        side = "left" if sender.x == self._sender_cols["left"] else "right"
        col_start, col_end = self._get_col_range(receiver_active, side)
        return [ttnn.CoreCoord(c, sender.y) for c in range(col_start, col_end)]

    def receiver_cores(
        self, sender_active: Optional[bool] = None, receiver_active: Optional[bool] = None
    ) -> List[ttnn.CoreRangeSet]:
        """Get receiver ranges per sender. Returns CoreRangeSet of receiver cores for each sender.

        Always creates individual CoreRange for each receiver to ensure consistent
        CoreRangeSet size across all senders (required by global circular buffer).
        """
        result = []
        for sender in self.sender_cores(active=sender_active):
            receivers = self._get_receivers(sender, receiver_active)
            if not receivers:
                continue
            # Always create individual CoreRanges for each receiver
            # This ensures all senders have the same CoreRangeSet structure
            result.append(ttnn.CoreRangeSet([ttnn.CoreRange(r, r) for r in receivers]))
        return result

    def dram_banks(self) -> List[ttnn.CoreCoord]:
        return self._dram_banks


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
        num_receiver_cores: int = None,
        receiver_mapping_override: Optional[dict] = None,
    ):
        """
        Prefetcher class that prefetches tensors from DRAM to L1.

        Args:
            receiver_mapping_override: If provided, keys become sender cores and values become
                their receiver cores. This overrides the default column 0/7 sender placement.
        """
        ### Device, Global CB, Parameters
        assert (
            is_blackhole()
        ), "DRAM Prefetcher is currently only supported on Tenstorrent Blackhole devices on BH QB 2 (4 devices) and BH LB (8 devices). Model support is available for Llama-3.1-8B under the TT-transformers framework. Support for wormhole devices and other models is WIP."
        self.global_cb = None
        self.mesh_device = mesh_device
        self.num_tensors = num_tensors
        self.num_layers = num_layers
        self.enable_performance_mode = True
        self.worker_sub_device_id = None
        self.global_cb_size = 0
        self.receiver_mapping_override = receiver_mapping_override

        # Determine num_receiver_cores - from override or default
        if receiver_mapping_override:
            # With override: num_receiver_cores is the number of receivers per sender
            first_receivers = list(receiver_mapping_override.values())[0]
            self.num_receiver_cores = len(first_receivers)
        else:
            self.num_receiver_cores = (
                self.get_optimal_receiver_cores() if num_receiver_cores is None else num_receiver_cores
            )
            assert (
                self.num_receiver_cores > 0 and self.num_receiver_cores <= 2
            ), "Number of receiver cores must be greater than 0 and less than or equal to 2. Only a max of 2 receiver cores have been tested to be functional on BH/WH"

        # Max tensor block size is the largest block size of a tensor in bytes
        self.max_tensor_block_size = 0
        ### Core Config
        self.core_config = PrefetcherCoreConfig(
            num_receiver_cores=self.num_receiver_cores,
            mesh_device=self.mesh_device,
            receiver_mapping_override=self.receiver_mapping_override,
        )

        # ring_size = num_receivers_per_sender * num_senders (i.e., total receiver cores)
        num_senders = len(self.core_config.sender_cores(active=True))
        self.ring_size = self.num_receiver_cores * num_senders

        ### Prefetcher Hardcoded Core Ranges
        self.dram_banks = self.core_config.dram_banks

        # Dynamic worker core grid (for easily grabbing a sub core grid that is of mulitples of 8 cores)
        self.dynamic_worker_core_grid = lambda num_cores: ttnn.CoreRangeSet(
            # requested number of cores MUST be multiples of 8
            [ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(num_cores // 8, 7))]
        )

        # Worker core ranges for the worker sub device
        if receiver_mapping_override:
            # With override: collect all unique receiver cores from the mapping
            all_receiver_coords = set()
            for receivers in receiver_mapping_override.values():
                for r in receivers:
                    coord = r if isinstance(r, ttnn.CoreCoord) else ttnn.CoreCoord(r[0], r[1])
                    all_receiver_coords.add((coord.x, coord.y))
            # Create CoreRangeSet from individual cores
            worker_ranges = [ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in all_receiver_coords]
            self.all_worker_cores_range_set = ttnn.CoreRangeSet(worker_ranges)
        else:
            # Default: use receiver column ranges
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
        self.mode = Mode.PREFILL
        self.init_decode_done = False
        self.init_prefill_done = False
        self.prefetch_done = False

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
        self, cores: List, return_list: bool = False
    ) -> Union[ttnn.CoreRangeSet, List[ttnn.CoreRangeSet]]:
        """Convert cores (CoreCoord/CoreRange/CoreRangeSet) to CoreRangeSet(s)."""
        assert cores, "No cores provided"

        def to_ranges(c):
            if isinstance(c, ttnn.CoreRangeSet):
                return c.ranges()
            elif isinstance(c, ttnn.CoreRange):
                return [c]
            elif isinstance(c, ttnn.CoreCoord):
                return [ttnn.CoreRange(c, c)]
            raise ValueError(f"Unsupported core type: {type(c)}")

        if return_list:
            return [ttnn.CoreRangeSet(to_ranges(c)) for c in cores]
        return ttnn.CoreRangeSet([r for c in cores for r in to_ranges(c)])

    def init(self, mode: Mode = Mode.DECODE) -> None:
        """
        Initializes the prefetcher sub devices
        Args:
            mode: The mode to run the prefetcher in, either "decode" or "prefill"
        NOTE: All DRAM prefetcher APIs can only be called after init() is called for the given mode
        NOTE: Calling init() again for the same mode is a no-op
        """
        # If the prefetcher has already been initialized for the given mode, we do not need to initialize it again
        if mode == Mode.DECODE and self.init_decode_done or mode == Mode.PREFILL and self.init_prefill_done:
            return

        self.mode = mode
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
            case Mode.DECODE:
                self.prefetcher_sub_device = PrefetcherSubDevice(self.mesh_device)
                self.prefetcher_sub_device.add_sub_device(self.to_core_range_set(self.sender_cores(active=True)))
                self.prefetcher_sub_device.add_sub_device(self.all_worker_cores_range_set)
                self.prefetcher_sub_device.init_sub_device_manager()
            case Mode.PREFILL:
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
        logger.warning(
            "Prefetcher has only been thoroughly tested on Llama3.1-8B BH QB 2 and BH LB 1. If using for other models and other device types, expect potential errors."
        )
        logger.info("=" * 50)
        self.init_decode_done = True if mode == Mode.DECODE else False
        self.init_prefill_done = True if mode == Mode.PREFILL else False

    def create_address_tensor(self):
        """
        Creates a ttnn tensor which holds the addresses of the tensors to be prefetched
        The addresses are replicated on each sender core
        """
        assert (
            len(self.prefetched_tensor_addr) == self.num_tensors * self.num_layers
        ), f"Number of tensor addresses have been inserted does not match the number of tensors to prefetch (num_tensors * num_layers), got {len(self.prefetched_tensor_addr)} != {self.num_tensors * self.num_layers}"

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
        Populates the tensor addresses that need to be prefetched
        Args:
            tensor: The tensor to insert into the prefetcher queue
        """
        assert self.init_decode_done, "Prefetcher has not been initialized for decode mode. Cannot insert tensors"
        bytes_in_tile = {ttnn.bfloat4_b: 576, ttnn.bfloat8_b: 1088, ttnn.bfloat16: 2048}
        if tensor.volume() % self.ring_size != 0:
            raise ValueError(
                f"Tensor volume ({tensor.volume()}) must be divisible by ring_size ({self.ring_size}) for prefetcher."
            )
        if not tensor.is_sharded() or tensor.memory_config().buffer_type != ttnn.BufferType.DRAM:
            raise ValueError(
                f"Tensor must be DRAM sharded for prefetcher. Got sharded={tensor.is_sharded()}, "
                f"buffer_type={tensor.memory_config().buffer_type}"
            )
        h, w = tensor.shape[-2], tensor.shape[-1]
        h_tiles, w_tiles = math.ceil(h / ttnn.TILE_SIZE), math.ceil(w / ttnn.TILE_SIZE)
        h_tiles_padded = math.ceil(h_tiles / self.ring_size) * self.ring_size
        max_tensor_tiles = (h_tiles_padded * w_tiles) // self.ring_size
        self.max_tensor_block_size = max(max_tensor_tiles * bytes_in_tile[tensor.dtype], self.max_tensor_block_size)
        self.prefetched_tensors.append(tensor)
        self.prefetched_tensor_addr.append(tensor.buffer_address())
        logger.info(
            f"[DRAM Prefetcher] Inserted tensor of shape {tensor.shape} into prefetcher, total number of tensors in prefetcher queue: {len(self.prefetched_tensor_addr)}"
        )

    def prefetch(self):
        """
        Inserts the tensors to be prefetched in a queue
        The tensors are prefetched in the order of the registration of the callbacks
        NOTE: This only needs to be called if a callback is registered for inserting tensors
        NOTE: prefetch() only needs to be called once and in decode mode, subsequent calls are no-ops
        """
        if self.mode == Mode.DECODE:
            assert self.init_decode_done, "Prefetcher has not been initialized for decode mode. Cannot prefetch tensors"
            assert (
                len(self.callbacks) > 0
            ), "No tensors insertion callbacks have been inserted into the prefetcher queue. Cannot prefetch an empty queue"
            if not self.prefetch_done:
                for callback in self.callbacks:
                    callback()
                self.prefetch_done = True
        # NO-OP for prefill mode
        return

    def run(self):
        """
        Start prefetching weights into global CB with dram_prefetcher op
        """
        assert self.init_decode_done, "Prefetcher has not been initialized for decode mode. Cannot run prefetcher"
        # Create global cb buffer if it was not yet created.
        if self.global_cb is None:
            self.global_cb_size = self.max_tensor_block_size
            logger.info(f"[DRAM Prefetcher] Creating global CB with size: {self.global_cb_size}")
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
        assert self.init_decode_done, "Prefetcher has not been initialized for decode mode. Cannot stop prefetcher"
        assert self.garbage is not None, "Prefetcher has not been run. Cannot stop prefetcher"
        ttnn.deallocate(self.garbage)
        self.garbage = None
        return
