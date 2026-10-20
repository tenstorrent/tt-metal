# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Union

import torch
import yaml
from loguru import logger

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.utility_functions import is_blackhole
from models.tt_transformers.tt.common import Mode

# Prefetcher yaml config file describes the sender/receiver core placements
_CONFIG_PATH = Path(__file__).parent / "prefetcher/prefetcher_config.yaml"
with open(_CONFIG_PATH) as f:
    ARCH_CONFIG = yaml.safe_load(f)

# Model configurations for which DRAM prefetcher is supported
# TODO #38278: to be removed when model support matrix is unified in tt-transformers
VERIFIED_MODEL_CONFIGS = {
    "Llama-3.2-1B": {"dim": 2048, "hidden_dim": 8192, "n_heads": 32, "n_kv_heads": 8},
    "Llama-3.2-3B": {"dim": 3072, "hidden_dim": 8192, "n_heads": 24, "n_kv_heads": 8},
    "Llama-3.1-8B": {"dim": 4096, "hidden_dim": 14336, "n_heads": 32, "n_kv_heads": 8},
    "Llama-3.3-70B": {"dim": 8192, "hidden_dim": 28672, "n_heads": 64, "n_kv_heads": 8},
    "Qwen3-32B": {"dim": 5120, "hidden_dim": 22016, "n_heads": 40, "n_kv_heads": 8},
    "Qwen3-VL-7B": {"dim": 4096, "hidden_dim": 11008, "n_heads": 32, "n_kv_heads": 8},
    "Qwen3-VL-14B": {"dim": 5120, "hidden_dim": 13824, "n_heads": 40, "n_kv_heads": 8},
    "Qwen3-VL-72B": {"dim": 8192, "hidden_dim": 28672, "n_heads": 64, "n_kv_heads": 8},
    "Gemma3-4B": {"dim": 2560, "hidden_dim": 14336, "n_heads": 20, "n_kv_heads": 20},
    "Gemma3-27B": {"dim": 4608, "hidden_dim": 24576, "n_heads": 32, "n_kv_heads": 8},
}


def generate_sender_receiver_mapping(num_receivers_per_sender: int = 8) -> dict:
    """
    Generate custom sender->receiver mapping for Blackhole prefetcher.
    Args:
        num_receivers_per_sender (int): Number of receiver cores per sender (8 for 64 total, 10 for 80 total)
    Returns:
        dict: {(sender_x, sender_y): [(rx, ry), ...]} mapping
    """
    cfg = ARCH_CONFIG["blackhole"]
    left_y = cfg["bank_ordered_y_coords"]["left"]
    right_y = cfg["bank_ordered_y_coords"]["right"]
    left_sender_col = cfg["sender_cols"]["left"]
    right_sender_col = cfg["sender_cols"]["right"]
    left_senders = [(left_sender_col, r) for r in left_y]
    right_senders = [(right_sender_col, r) for r in right_y]
    mapping = {}
    for sx, sy in left_senders:
        mapping[(sx, sy)] = [(x, sy) for x in range(1, num_receivers_per_sender + 1)]
    for sx, sy in right_senders:
        # Receivers for right senders: columns 8-10, plus columns 0-6 excluding sender column
        cols = list(range(8, 11)) + [x for x in range(8) if x != right_sender_col]
        mapping[(sx, sy)] = [(x, sy) for x in cols[:num_receivers_per_sender]]
    return mapping


TILE_BYTES = {ttnn.bfloat4_b: 576, ttnn.bfloat8_b: 1088, ttnn.bfloat16: 2048}


def tiles_per_receiver(k: int, n: int, ring_size: int) -> int:
    """Tiles of a [k, n] weight that each receiver holds: dram_prefetcher pads both dims to whole tiles per ring position."""
    k_tiles = math.ceil(math.ceil(k / ttnn.TILE_SIZE) / ring_size) * ring_size
    n_tiles = math.ceil(math.ceil(n / ttnn.TILE_SIZE) / ring_size) * ring_size
    return k_tiles * n_tiles // ring_size


def max_bytes_per_receiver(model_name: str, num_devices: int, ring_size: int) -> int:
    """The largest bfloat8_b weight share any receiver holds for a verified model: an MLP weight's, FF1 or FF2."""
    dims = next(VERIFIED_MODEL_CONFIGS[m] for m in VERIFIED_MODEL_CONFIGS if m in model_name)
    hidden_per_device = dims["hidden_dim"] // num_devices
    return TILE_BYTES[ttnn.bfloat8_b] * max(
        tiles_per_receiver(dims["dim"], hidden_per_device, ring_size),
        tiles_per_receiver(hidden_per_device, dims["dim"], ring_size),
    )


def is_prefetcher_supported(model_name: str, num_devices: int, ring_size: int = 16) -> bool:
    """
    Check if model can use DRAM prefetcher: CB pages <= 65535, L1 size fits, kv_heads % num_devices == 0.
    Args:
        model_name (str): Model name (must contain a key from VERIFIED_MODEL_CONFIGS)
        num_devices (int): Number of devices for tensor parallelism
        ring_size (int): Total receiver cores (default 16, custom mapping uses 64/80)
    Returns:
        bool: True if supported on Blackhole with given config, False otherwise
    """
    verified_model_name = next((m for m in VERIFIED_MODEL_CONFIGS if m in model_name), None)
    if not is_blackhole() or verified_model_name is None:
        return False
    MAX_CB_PAGES = 65535
    MAX_L1_PER_BANK = {4: 1000000, 8: 1000000}.get(num_devices, 850000)
    kv_heads_divisible = VERIFIED_MODEL_CONFIGS[verified_model_name]["n_kv_heads"] % num_devices == 0
    dim, hidden_dim = (
        VERIFIED_MODEL_CONFIGS[verified_model_name]["dim"],
        VERIFIED_MODEL_CONFIGS[verified_model_name]["hidden_dim"],
    )
    tiles_per_core = tiles_per_receiver(dim, hidden_dim // num_devices, ring_size)
    # Check memory constraints and kv heads divisible by num_devices
    pages_ok = tiles_per_core <= MAX_CB_PAGES
    bytes_per_core = tiles_per_core * TILE_BYTES[ttnn.bfloat8_b]
    l1_ok = bytes_per_core <= MAX_L1_PER_BANK
    logger.info(
        f"DRAM Prefetcher support check: tiles_per_core: {tiles_per_core} <= {MAX_CB_PAGES} is {pages_ok}, bytes_per_core: {bytes_per_core} <= {MAX_L1_PER_BANK} is {l1_ok}, kv_heads_divisible: {kv_heads_divisible}"
    )
    return pages_ok and l1_ok and kv_heads_divisible


@dataclass
class PrefetcherCoreConfig:
    """
    Core locations for prefetcher sender/receiver cores.

    If receiver_mapping_override is provided, its keys become the sender cores and values
    become their receivers (all treated as "active"). This allows full custom placement.
    Otherwise, uses default architecture-specific sender/receiver layout.
    """

    num_receiver_cores: int
    mesh_device: ttnn.MeshDevice
    cfg: dict
    receiver_mapping_override: Optional[dict] = None  # {(x,y): [(rx,ry), ...]}

    def __post_init__(self):
        self._dram_banks = [ttnn.CoreCoord(b, 0) for b in self.cfg["dram_banks"]]
        self._sender_cols, self._sender_rows = self.cfg["sender_cols"], self.cfg["sender_rows"]
        self._receiver_cols = {k: tuple(v) for k, v in self.cfg["receiver_cols"].items()}
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

        Always creates individual CoreRange for each receiver, so a sender's receivers are listed in
        receiver order: the r-th gets column slice r of every block the sender prefetches.
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
    ):
        """
        Prefetcher class that prefetches tensors from DRAM to L1.

        Each sender core reads one DRAM bank and delivers into its own worker-sender PrefetcherPipe,
        whose receivers are the matmul ring cores that consume that bank's columns.

        Args:
            receiver_mapping_override: If provided, keys become sender cores and values become
                their receiver cores. This overrides the default column 0/7 sender placement.
        """
        ### Device, PrefetcherPipes, Parameters
        self.pf_config: dict = ARCH_CONFIG["blackhole"]
        self.legal_receiver_cores: List[int] = self.pf_config["legal_receiver_cores"]
        self.mesh_device: ttnn.MeshDevice = mesh_device
        self.enable_performance_mode: bool = True
        # The pipes' persistent L1 on the sender and receiver cores, and one pipe per sender, in DRAM bank order
        self.pipe_space: Optional[ttnn.experimental.PrefetcherPipeSpace] = None
        self.pipes: List[ttnn.experimental.PrefetcherPipe] = []
        self.worker_sub_device_id: Optional[ttnn.SubDeviceId] = None
        self.num_tensors: int = num_tensors
        self.num_layers: int = num_layers
        self.num_senders: int = len(self.pf_config["dram_banks"])
        self.receiver_mapping_override: Optional[dict] = None
        self.model_name = os.getenv("HF_MODEL", "")
        assert self.model_name != "", "HF_MODEL is not set. DRAM Prefetcher must be run with a model."
        assert (
            num_receiver_cores is None or num_receiver_cores in self.legal_receiver_cores
        ), "num_receiver_cores must be in legal_receiver_cores"
        if num_receiver_cores is not None:
            assert is_prefetcher_supported(
                self.model_name, self.mesh_device.get_num_devices(), num_receiver_cores * self.num_senders
            ), "num_receiver_cores is not supported"
            self.num_receiver_cores = num_receiver_cores
            self.receiver_mapping_override = (
                generate_sender_receiver_mapping(num_receiver_cores) if num_receiver_cores > 3 else None
            )
        else:
            for num_receivers in self.legal_receiver_cores:
                if is_prefetcher_supported(
                    self.model_name, self.mesh_device.get_num_devices(), num_receivers * self.num_senders
                ):
                    self.num_receiver_cores = num_receivers
                    self.receiver_mapping_override = (
                        generate_sender_receiver_mapping(num_receivers) if num_receivers > 3 else None
                    )
                    break

        ### Core Config
        self.core_config = PrefetcherCoreConfig(
            num_receiver_cores=self.num_receiver_cores,
            mesh_device=self.mesh_device,
            cfg=self.pf_config,
            receiver_mapping_override=self.receiver_mapping_override,
        )
        self.ring_size = self.num_receiver_cores * self.num_senders
        self.dram_banks = self.core_config.dram_banks

        ### Worker core ranges for the worker sub device
        if self.receiver_mapping_override:
            grid = self.mesh_device.compute_with_storage_grid_size()
            full_grid = ttnn.CoreRangeSet(
                [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))]
            )
            sender_cores = [
                ttnn.CoreRange(ttnn.CoreCoord(s.x, s.y), ttnn.CoreCoord(s.x, s.y))
                for s in self.core_config.sender_cores(active=True)
            ]
            sender_set = ttnn.CoreRangeSet(sender_cores)
            self.all_worker_cores_range_set = full_grid.subtract(sender_set)
        else:
            left_range = self.core_config._receiver_cols["left"]
            right_range = self.core_config._receiver_cols["right"]
            self.all_worker_cores_range_set = ttnn.CoreRangeSet(
                [ttnn.CoreRange(ttnn.CoreCoord(left_range[0], 0), ttnn.CoreCoord(left_range[1] - 1, 9))]
                + [ttnn.CoreRange(ttnn.CoreCoord(right_range[0], 0), ttnn.CoreCoord(right_range[1] - 1, 9))]
            )

        ### Dynamic worker core grid: num_cores must be multiple of 8, spans cols 1-6 rows 0-7, plus cols 8+ if needed
        def dynamic_worker_core_grid(num_cores):
            cols = num_cores // 8
            ranges = [ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(min(cols, 6), 7))]
            if cols > 6:
                ranges.append(ttnn.CoreRange(ttnn.CoreCoord(8, 0), ttnn.CoreCoord(cols + 1, 7)))
            return ttnn.CoreRangeSet(ranges)

        self.dynamic_worker_core_grid = dynamic_worker_core_grid

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

    def _reserve_pipes(self):
        """Reserve the PrefetcherPipes: one per active sender, whose receivers are that sender's ring cores.

        Each receiver keeps a whole tensor's blocks until its matmul is done with them, so a ring holds the largest
        weight share. The rings are persistent L1 at the bottom of the sender and receiver cores, which a live
        program's circular buffers there seal. Called on the first decode: clearing the program cache releases the
        programs that model construction and prefill left on those cores, and decode compiles its own above the rings.
        A prefill after that has to fit beside the rings.
        """
        ttnn.synchronize_device(self.mesh_device)
        self.mesh_device.clear_program_cache()
        senders = self.core_config.sender_cores(active=True)
        receivers = self.core_config.receiver_cores(sender_active=True, receiver_active=True)
        ring_size = max_bytes_per_receiver(self.model_name, self.mesh_device.get_num_devices(), self.ring_size)
        logger.info(f"[DRAM Prefetcher] Creating PrefetcherPipes with ring size: {ring_size}")
        self.pipe_space = ttnn.experimental.create_prefetcher_pipe_space(
            self.mesh_device,
            sender_cores=self.to_core_range_set(senders),
            receiver_domain=self.to_core_range_set(receivers),
            ring_size=ring_size,
            max_receivers_per_pipe=self.num_receiver_cores,
        )
        self.pipes = self.pipe_space.create_pipes(list(zip(senders, receivers)))

    # NOTE: DRAM prefetched weights are prefetched in the order of the construction of the module
    def register_callback(self, callback: Callable[[], None]):
        self.callbacks.append(callback)

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
        self.sender_cores = self.core_config.sender_cores
        self.receiver_cores = self.core_config.receiver_cores
        match mode:
            case Mode.DECODE:
                if self.num_tensors > 0 and not self.pipes:
                    self._reserve_pipes()
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
            f"DRAM Prefetcher has only been tested on these models: {list(VERIFIED_MODEL_CONFIGS.keys())} on BH DB, QB, LB. If using other models and other device types, expect potential errors. To check if the model is supported on the current device type, run is_prefetcher_supported(model_name, num_devices, ring_size)."
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
        assert self.pipes, "This prefetcher was created for no tensors. Cannot insert tensors"
        if tensor.volume() % self.ring_size != 0:
            raise ValueError(
                f"Tensor volume ({tensor.volume()}) must be divisible by ring_size ({self.ring_size}) for prefetcher."
            )
        if not tensor.is_sharded() or tensor.memory_config().buffer_type != ttnn.BufferType.DRAM:
            raise ValueError(
                f"Tensor must be DRAM sharded for prefetcher. Got sharded={tensor.is_sharded()}, "
                f"buffer_type={tensor.memory_config().buffer_type}"
            )
        bytes_per_receiver = (
            tiles_per_receiver(tensor.shape[-2], tensor.shape[-1], self.ring_size) * TILE_BYTES[tensor.dtype]
        )
        if bytes_per_receiver > self.pipe_space.ring_size():
            raise ValueError(
                f"Each receiver holds {bytes_per_receiver} B of the {tensor.dtype} tensor of shape {tensor.shape}, but the "
                f"prefetcher's pipe rings, sized for this model's bfloat8_b MLP weights, hold {self.pipe_space.ring_size()} B"
            )
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
        Start prefetching weights into the PrefetcherPipes with dram_prefetcher op
        """
        assert self.init_decode_done, "Prefetcher has not been initialized for decode mode. Cannot run prefetcher"

        # Create address tensor if it was not created yet
        if self.prefetched_tt_addr_tensor is None:
            self.prefetched_tt_addr_tensor = self.create_address_tensor()

        # Run prefetcher op (prefetcher op will start asynchronously prefetching weights until prefetcher.stop() is called)
        self.garbage = ttnn.dram_prefetcher(
            self.prefetched_tensors[: self.num_tensors] + [self.prefetched_tt_addr_tensor],
            num_layers=self.num_layers,
            enable_performance_mode=self.enable_performance_mode,
            prefetcher_pipes=self.pipes,
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
