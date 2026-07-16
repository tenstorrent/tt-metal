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

# Models blessed for the DRAM-core *Tensor* Prefetcher. Kept separate from
# VERIFIED_MODEL_CONFIGS because the two backends are validated independently: a model working
# on the worker-core prefetcher does not imply the recv-contig Tensor Prefetcher path has been
# exercised for it. Seed with the shared shapes; prune/extend this to only the models actually
# validated end-to-end on the Tensor Prefetcher (currently Llama-3.2-3B and Llama-3.1-8B).
# The per-weight divisibility + streaming-GCB checks in is_tensor_prefetcher_config_supported
# still gate geometry on top of this list.
TENSOR_PREFETCHER_VERIFIED_MODEL_CONFIGS = dict(VERIFIED_MODEL_CONFIGS)


def uses_tensor_prefetcher(prefetcher) -> bool:
    """Return whether a model backend queues Tensor Prefetcher requests per matmul."""
    return prefetcher is not None and getattr(prefetcher, "uses_tensor_prefetcher", False)


# Single source for ttnn weight-dtype -> tile byte size, shared by both prefetcher backends
# (worker insert_tensor block sizing, DRAM-core GCB sizing, and the support checks). Keeping
# one map avoids the two backends silently disagreeing if a dtype/tile size changes.
TILE_BYTES = {ttnn.bfloat4_b: 576, ttnn.bfloat8_b: 1088, ttnn.bfloat16: 2048}
BYTES_PER_TILE_BFP8 = TILE_BYTES[ttnn.bfloat8_b]


def full_grid_core_range_set(mesh_device: ttnn.MeshDevice) -> ttnn.CoreRangeSet:
    """The full compute-with-storage grid as a single-range CoreRangeSet.

    Shared by both prefetcher backends (worker sub-device setup, DRAM-core worker sub-device).
    """
    grid = mesh_device.compute_with_storage_grid_size()
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])


def prefetcher_linear(prefetcher, input_tensor_a, weight, *, mode, program_config, **linear_kwargs):
    """Dispatch a decode/prefill matmul through whichever prefetcher backend is active.

    Collapses the per-call-site ``prefetch_and_linear`` (DRAM-core tensor backend) vs
    ``ttnn.linear`` (worker / no-prefetcher) fork. In decode the tensor backend streams the
    weight through its GCB while the worker backend passes its GCB + sub-device; every other
    case is a plain matmul. ``linear_kwargs`` (``memory_config``, ``compute_kernel_config``,
    ``dtype``, ``core_grid``, ...) are forwarded unchanged, so each call site passes only the
    kwargs it actually uses.
    """
    if mode == Mode.DECODE and uses_tensor_prefetcher(prefetcher):
        return ttnn.experimental.tensor_prefetcher_matmul.prefetch_and_linear(
            input_tensor_a,
            weight,
            global_cb=prefetcher.global_cb,
            program_config=program_config,
            **linear_kwargs,
        )
    use_gcb = prefetcher is not None and mode == Mode.DECODE
    return ttnn.linear(
        input_tensor_a,
        weight,
        program_config=program_config,
        global_cb=prefetcher.global_cb if use_gcb else None,
        sub_device_id=prefetcher.worker_sub_device_id if use_gcb else None,
        **linear_kwargs,
    )


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


def is_tensor_prefetcher_supported(
    mesh_device: ttnn.MeshDevice,
    model_name: Optional[str] = None,
    num_receiver_cores: Optional[int] = None,
) -> bool:
    """Return whether the DRAM-core Tensor Prefetcher supports this model and mesh."""
    if not is_blackhole() or not ttnn.experimental.is_tensor_prefetcher_supported(mesh_device):
        return False

    # The Tensor Prefetcher's shape math currently assumes a single-row mesh (cluster_shape == (1,
    # num_devices)): it divides every weight's K/N by the total device count on one axis. A 2-D
    # mesh (including a Blackhole galaxy) splits K and N across different cluster axes, so the
    # recv-contig geometry would be wrong until that math is generalized. Reject it here for now so
    # make_prefetcher falls back to the worker-core Prefetcher rather than mis-sharding.
    if list(mesh_device.shape)[0] != 1:
        return False

    from models.tt_transformers.tt.tensor_prefetcher import is_tensor_prefetcher_config_supported, norm_grid_fits

    model_name = model_name or os.getenv("HF_MODEL", "")
    num_senders = mesh_device.dram_grid_size().x
    receiver_candidates = [num_receiver_cores] if num_receiver_cores is not None else [8, 4, 2, 1]
    return any(
        is_tensor_prefetcher_config_supported(
            model_name,
            mesh_device.get_num_devices(),
            num_senders,
            receivers_per_bank,
        )
        and norm_grid_fits(mesh_device, receivers_per_bank)
        for receivers_per_bank in receiver_candidates
    )


def make_prefetcher(
    mesh_device: ttnn.MeshDevice,
    num_tensors: int,
    num_layers: int,
    num_receiver_cores: Optional[int] = None,
):
    """Select the prefetcher backend.

    Prefer ``TensorPrefetcher`` whenever the model, mesh, and firmware support the
    Tensor Prefetcher. Fall back to the worker-core ``Prefetcher`` otherwise.
    """
    if is_tensor_prefetcher_supported(mesh_device, num_receiver_cores=num_receiver_cores):
        from models.tt_transformers.tt.tensor_prefetcher import TensorPrefetcher

        return TensorPrefetcher(mesh_device, num_tensors, num_layers, num_receiver_cores=num_receiver_cores)
    return Prefetcher(mesh_device, num_tensors, num_layers, num_receiver_cores=num_receiver_cores)


def colocating_prefetcher(prefetcher):
    """Return ``prefetcher`` only when it co-locates ops on the worker grid, else ``None``.

    Backends that drive weights from a separate grid (DRAM-core) report
    ``colocate_ops=False``; ops keyed off the returned value then fall back to their
    default (non-prefetcher) placement. Single source for the "does this prefetcher
    place ops on the worker grid" check that model.py / model_config.py / attention.py /
    lm_head.py would otherwise each spell out inline.
    """
    return prefetcher if (prefetcher is not None and prefetcher.colocate_ops) else None


def to_core_range_set(cores: List, return_list: bool = False) -> Union[ttnn.CoreRangeSet, List[ttnn.CoreRangeSet]]:
    """Convert cores (CoreCoord/CoreRange/CoreRangeSet, or a list thereof) to CoreRangeSet(s).

    Shared by both prefetcher backends (``Prefetcher`` and ``TensorPrefetcher``) so the core-type
    handling has a single definition.
    """
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


def resolve_verified_model_cfg(model_name: str) -> Optional[dict]:
    """Return the ``VERIFIED_MODEL_CONFIGS`` entry whose key is a substring of ``model_name``.

    Returns ``None`` off Blackhole (the only architecture the prefetcher paths are verified on) or
    when no verified model matches. Shared by ``is_prefetcher_supported`` and
    ``is_tensor_prefetcher_config_supported`` so the model lookup + Blackhole gate have one definition.
    (The per-weight shape math and L1-budget checks intentionally differ between the two backends.)
    """
    if not is_blackhole():
        return None
    name = next((m for m in VERIFIED_MODEL_CONFIGS if m in model_name), None)
    return VERIFIED_MODEL_CONFIGS[name] if name is not None else None


def resolve_tensor_prefetcher_model_cfg(model_name: str) -> Optional[dict]:
    """Return the ``TENSOR_PREFETCHER_VERIFIED_MODEL_CONFIGS`` entry matching ``model_name``.

    Tensor-Prefetcher counterpart to :func:`resolve_verified_model_cfg`. Returns ``None`` off
    Blackhole or when the model is not on the Tensor-Prefetcher-verified list (which is curated
    separately from the worker-core list — see the constant's docstring).
    """
    if not is_blackhole():
        return None
    name = next((m for m in TENSOR_PREFETCHER_VERIFIED_MODEL_CONFIGS if m in model_name), None)
    return TENSOR_PREFETCHER_VERIFIED_MODEL_CONFIGS[name] if name is not None else None


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
    cfg = resolve_verified_model_cfg(model_name)
    if cfg is None:
        return False
    TILE_SIZE, MAX_CB_PAGES = 32, 65535
    MAX_L1_PER_BANK = {4: 1000000, 8: 1000000}.get(num_devices, 850000)
    kv_heads_divisible = cfg["n_kv_heads"] % num_devices == 0
    dim, hidden_dim = cfg["dim"], cfg["hidden_dim"]
    n_per_device = hidden_dim // num_devices
    n_per_core = math.ceil(n_per_device / ring_size)
    n_per_core_padded = ((n_per_core + TILE_SIZE - 1) // TILE_SIZE) * TILE_SIZE
    n_padded = n_per_core_padded * ring_size
    h_tiles = math.ceil(dim / TILE_SIZE)
    w_tiles = n_padded // TILE_SIZE
    h_tiles_padded = ((h_tiles + ring_size - 1) // ring_size) * ring_size
    tiles_per_core = (h_tiles_padded * w_tiles) // ring_size
    # Check memory constraints and kv heads divisible by num_devices
    pages_ok = tiles_per_core <= MAX_CB_PAGES
    bytes_per_core = tiles_per_core * BYTES_PER_TILE_BFP8
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
    ):
        """
        Prefetcher class that prefetches tensors from DRAM to L1.

        Args:
            receiver_mapping_override: If provided, keys become sender cores and values become
                their receiver cores. This overrides the default column 0/7 sender placement.
        """
        ### Device, Global CB, Parameters
        self.pf_config: dict = ARCH_CONFIG["blackhole"]
        self.legal_receiver_cores: List[int] = self.pf_config["legal_receiver_cores"]
        self.mesh_device: ttnn.MeshDevice = mesh_device
        self.enable_performance_mode: bool = True
        # The worker-core backend co-locates surrounding (non-GCB-matmul) ops — SDPA, create/concat
        # heads, lm_head, rope — on its reserved worker grid so the activation stays sharded there.
        # The Tensor Prefetcher backend sets this False (those ops use the model's default placement).
        self.colocate_ops: bool = True
        # The worker backend never streams weights (batched delivery only); the Tensor Prefetcher backend
        # overrides this. Defined here so model_config can read prefetcher.stream_in1 uniformly.
        self.stream_in1: bool = False
        self.global_cb: Optional[ttnn.GlobalCircularBuffer] = None
        self.worker_sub_device_id: Optional[ttnn.SubDeviceId] = None
        self.num_tensors: int = num_tensors
        self.num_layers: int = num_layers
        self.num_senders: int = len(self.pf_config["dram_banks"])
        self.global_cb_size: int = 0  # Size of the global circular buffer in bytes storing prefetched matmul weights
        self.max_tensor_block_size: int = 0  # Max tensor block size is the largest block size of a tensor in bytes
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
            full_grid = full_grid_core_range_set(self.mesh_device)
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

    # NOTE: DRAM prefetched weights are prefetched in the order of the construction of the module
    def register_callback(self, callback: Callable[[], None]):
        self.callbacks.append(callback)

    def to_core_range_set(
        self, cores: List, return_list: bool = False
    ) -> Union[ttnn.CoreRangeSet, List[ttnn.CoreRangeSet]]:
        """Convert cores (CoreCoord/CoreRange/CoreRangeSet) to CoreRangeSet(s)."""
        return to_core_range_set(cores, return_list=return_list)

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
                # Prefill does not prefetch, so no sender columns are reserved — the sub-device is
                # the full compute grid. (Was self.all_core_range_set, an attribute that is never
                # assigned: a pre-existing typo/omission on main; all_worker_cores_range_set would
                # exclude the sender columns and trip num_intersections != num_cores.)
                full_grid = full_grid_core_range_set(self.mesh_device)
                self.prefetcher_sub_device = PrefetcherSubDevice(self.mesh_device)
                self.prefetcher_sub_device.add_sub_device(full_grid)
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

    def insert_tensor(self, tensor: ttnn.Tensor, program_config=None):
        """
        Populates the tensor addresses that need to be prefetched
        Args:
            tensor: The tensor to insert into the prefetcher queue
            program_config: The consuming matmul config. The worker backend derives block sizes
                from the tensor itself and ignores it.
        """
        del program_config
        assert self.init_decode_done, "Prefetcher has not been initialized for decode mode. Cannot insert tensors"
        bytes_in_tile = TILE_BYTES
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
        w_tiles_padded = math.ceil(w_tiles / self.ring_size) * self.ring_size
        max_tensor_tiles = (h_tiles_padded * w_tiles_padded) // self.ring_size
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

    def weight_cache_suffix(self) -> str:
        """Cache-key discriminator for the on-disk weight layout this backend produces.

        The worker-core backend keeps the caller's default (width-sharded) layout, which is the
        historical cache layout, so the suffix is empty — existing cache files stay valid. The
        Tensor Prefetcher backend overrides this to keep its receiver-contiguous weights in separate files.
        """
        return ""

    def weight_mem_config(
        self, k: int, n: int, default: ttnn.MemoryConfig, is_galaxy: bool = False
    ) -> ttnn.MemoryConfig:
        """Memory config for a prefetched (K, N) weight.

        The worker-core path keeps the caller's ``default`` (width-sharded) layout. Defined so
        model code can call ``prefetcher.weight_mem_config(...)`` uniformly; the Tensor Prefetcher backend
        overrides this to return its receiver-contiguous layout.
        """
        del k, n, is_galaxy
        return default
