# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""
Explicit Tenstorrent board / device-class specification for the standalone
SDXL and SD3.5 servers.

The taxonomy mirrors tt-inference-server/workflows/workflow_types.py so that
this module can later be unified with that one.

- DeviceClass : enum of supported board / cluster identities
- BoardSpec   : per-board static facts (descriptor file, arch, env vars,
                which models are supported)
- DeploymentSpec : per-(model, board) deployment shape (num_workers, mesh
                   shape, device ids, fabric config)

Only boards that have a concrete deployment row in DEPLOYMENTS are declared
in BOARD_SPECS. Asking for a DeviceClass without a BoardSpec entry raises
"not supported yet" — this keeps the registry honest.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Dict, FrozenSet, Optional, Tuple


# ---------------------------------------------------------------------------
# DeviceClass — full taxonomy mirrored from tt-inference-server
# ---------------------------------------------------------------------------


class DeviceClass(IntEnum):
    """Tenstorrent board / cluster identity.

    Mirrors tt-inference-server/workflows/workflow_types.py:DeviceTypes so
    `DeviceClass.from_string("p300")` resolves the same names a user would
    pass on the inference-server CLI.
    """

    # Wormhole_B0
    N150 = auto()
    N300 = auto()
    T3K = auto()
    GALAXY_T3K = auto()
    GALAXY = auto()
    DUAL_GALAXY = auto()
    QUAD_GALAXY = auto()

    # Blackhole
    P100 = auto()
    P150 = auto()
    P150X2 = auto()
    P150X4 = auto()
    P150X8 = auto()
    P300 = auto()
    P300X2 = auto()

    @classmethod
    def from_string(cls, name: str) -> "DeviceClass":
        """Parse 'p300', 'P300', 'p300x2', 'P300X2' etc."""
        try:
            return cls[name.upper()]
        except KeyError:
            valid = ", ".join(m.name.lower() for m in cls)
            raise ValueError(f"Unknown device class '{name}'. Valid: {valid}")


GALAXY_BOARDS: FrozenSet[DeviceClass] = frozenset(
    {DeviceClass.GALAXY, DeviceClass.GALAXY_T3K, DeviceClass.DUAL_GALAXY, DeviceClass.QUAD_GALAXY}
)


def is_galaxy(board: DeviceClass) -> bool:
    return board in GALAXY_BOARDS


# ---------------------------------------------------------------------------
# BoardSpec — static, per-board facts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BoardSpec:
    """Static facts about a Tenstorrent board.

    Fields are intentionally narrow: anything that varies by *model* (mesh
    shape, num_workers, fabric config) lives in DeploymentSpec instead.

    `core_grid_override` is the value for TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE;
    None means do not set the env var (use the device's native compute grid).
    Mirrors tt-inference-server/tt-media-server/utils/runner_utils.py:112-122.

    `l1_small_size` matches SDXL_L1_SMALL_SIZE / SDXL_L1_SMALL_SIZE_BH from
    models/demos/stable_diffusion_xl_base/tests/test_common.py:26-27.
    """

    descriptor_filename: str
    arch: str  # "wormhole_b0" | "blackhole"
    chip_count: int
    extra_env_vars: Dict[str, str] = field(default_factory=dict)
    supports_models: FrozenSet[str] = field(default_factory=frozenset)
    core_grid_override: Optional[str] = None
    l1_small_size: int = 32000
    trace_region_size: int = 34541598  # Wormhole-tuned default (matches tt-media-server)


BOARD_SPECS: Dict[DeviceClass, BoardSpec] = {
    DeviceClass.T3K: BoardSpec(
        descriptor_filename="t3k_mesh_graph_descriptor.textproto",
        arch="wormhole_b0",
        chip_count=8,
        supports_models=frozenset({"sdxl", "wan22"}),
        core_grid_override="7,7",  # Wormhole-tuned SDXL configs use (5,8)=40 cores; 8x8=64 fits.
        l1_small_size=32000,  # SDXL_L1_SMALL_SIZE
    ),
    DeviceClass.P150: BoardSpec(
        descriptor_filename="p150_mesh_graph_descriptor.textproto",
        arch="blackhole",
        chip_count=1,
        supports_models=frozenset({"sdxl"}),
        core_grid_override=None,  # Native BH grid; model_configs_1024x1024BH.py wants (10,8)=80 cores.
        l1_small_size=38000,  # SDXL_L1_SMALL_SIZE_BH
        trace_region_size=56_000_000,  # BH SDXL trace ~41MB measured; round up for headroom.
    ),
    DeviceClass.P150X4: BoardSpec(
        descriptor_filename="p150_x4_mesh_graph_descriptor.textproto",
        arch="blackhole",
        chip_count=4,
        supports_models=frozenset({"wan22"}),
        core_grid_override=None,
        l1_small_size=38000,
        trace_region_size=30_000_000,  # per model_spec.py wan22 override.
    ),
    DeviceClass.P150X8: BoardSpec(
        descriptor_filename="p150_x8_mesh_graph_descriptor.textproto",
        arch="blackhole",
        chip_count=8,
        supports_models=frozenset({"wan22"}),
        core_grid_override=None,
        l1_small_size=38000,
        trace_region_size=30_000_000,
    ),
    DeviceClass.P300: BoardSpec(
        descriptor_filename="p300_mesh_graph_descriptor.textproto",
        arch="blackhole",
        chip_count=2,
        supports_models=frozenset({"sdxl"}),
        core_grid_override=None,
        l1_small_size=38000,
        trace_region_size=56_000_000,  # BH SDXL trace ~41MB measured; round up for headroom.
    ),
    DeviceClass.P300X2: BoardSpec(
        descriptor_filename="p300_x2_mesh_graph_descriptor.textproto",
        arch="blackhole",
        chip_count=4,
        supports_models=frozenset({"sdxl", "wan22"}),
        core_grid_override=None,
        l1_small_size=38000,
        trace_region_size=56_000_000,  # BH SDXL trace ~41MB measured; round up for headroom.
    ),
}


# ---------------------------------------------------------------------------
# DeploymentSpec — per-(model, board) deployment shape
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DeploymentSpec:
    num_workers: int
    mesh_shape: Tuple[int, int]  # passed to ttnn.MeshShape per worker
    device_ids: Tuple[int, ...]
    fabric_config: Optional[str] = None  # e.g. "FABRIC_1D"


# Keyed by (model_name, board). Model names match server.py --model choices.
DEPLOYMENTS: Dict[Tuple[str, DeviceClass], DeploymentSpec] = {
    ("sdxl", DeviceClass.T3K): DeploymentSpec(
        num_workers=4,
        mesh_shape=(1, 1),
        device_ids=(0, 1, 2, 3),
    ),
    ("sdxl", DeviceClass.P150): DeploymentSpec(
        num_workers=1,
        mesh_shape=(1, 1),
        device_ids=(0,),
    ),
    ("sdxl", DeviceClass.P300): DeploymentSpec(
        num_workers=1,
        # (TP, DP) — multi-chip SDXL must use TP layout so cfg_parallel kicks in.
        # Mirrors tt-media-server convention (config/constants.py: N300/T3K = (2,1)).
        mesh_shape=(2, 1),
        device_ids=(0, 1),
        fabric_config="FABRIC_1D",
    ),
    ("sdxl", DeviceClass.P300X2): DeploymentSpec(
        num_workers=1,
        # 4-chip BH host: SDXL uses TP=2 only (matches tt-media-server T3K
        # convention). DP=2 (mesh (2,2)) trips a CLIP-token shard mismatch
        # in distributed_tensor.cpp:77 — TtSDXLPipeline doesn't support
        # data-parallel for the text encoder shard layout.
        mesh_shape=(2, 1),
        device_ids=(0, 1),
        fabric_config="FABRIC_1D",
    ),
    # Wan2.2 mesh shapes mirror tt-media-server/config/constants.py and are
    # validated against tt-metal's WanPipeline.create_pipeline default_config
    # (pipelines/wan/pipeline_wan.py:383-419).
    ("wan22", DeviceClass.T3K): DeploymentSpec(
        num_workers=1,
        mesh_shape=(2, 4),
        device_ids=(0, 1, 2, 3),  # PCIe L chips; R chips auto-discovered via fabric.
        fabric_config="FABRIC_1D",
    ),
    ("wan22", DeviceClass.P150X4): DeploymentSpec(
        num_workers=1,
        mesh_shape=(1, 4),
        device_ids=(0, 1, 2, 3),
        fabric_config="FABRIC_1D",
    ),
    ("wan22", DeviceClass.P150X8): DeploymentSpec(
        num_workers=1,
        mesh_shape=(2, 4),
        device_ids=(0, 1, 2, 3, 4, 5, 6, 7),
        fabric_config="FABRIC_1D",
    ),
    ("wan22", DeviceClass.P300X2): DeploymentSpec(
        num_workers=1,
        mesh_shape=(2, 2),
        device_ids=(0, 1, 2, 3),
        fabric_config="FABRIC_1D",
    ),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_board_spec(board: DeviceClass) -> BoardSpec:
    """Return the BoardSpec for `board`, or raise with a useful message."""
    spec = BOARD_SPECS.get(board)
    if spec is None:
        supported = ", ".join(b.name.lower() for b in BOARD_SPECS)
        raise ValueError(
            f"Board '{board.name.lower()}' is recognized but not supported yet. "
            f"Supported boards: {supported}"
        )
    return spec


def get_deployment(model: str, board: DeviceClass) -> DeploymentSpec:
    """Return the DeploymentSpec for (model, board), or raise."""
    dep = DEPLOYMENTS.get((model, board))
    if dep is None:
        valid_for_model = sorted(
            b.name.lower() for (m, b) in DEPLOYMENTS if m == model
        )
        raise ValueError(
            f"No deployment defined for model={model!r} board={board.name.lower()!r}. "
            f"Valid boards for {model!r}: {valid_for_model or '(none)'}"
        )
    return dep


def descriptor_path(board: DeviceClass) -> str:
    """Resolve absolute path to the mesh-graph descriptor for `board`.

    Anchored on TT_METAL_HOME (matching the existing convention in worker.py),
    falling back to the current working directory.
    """
    spec = get_board_spec(board)
    tt_metal_home = os.environ.get("TT_METAL_HOME") or os.getcwd()
    return os.path.join(
        tt_metal_home, "tt_metal", "fabric", "mesh_graph_descriptors", spec.descriptor_filename
    )


def validate_model_board(model: str, board: DeviceClass) -> None:
    """Fail fast with a clear message if (model, board) isn't supported."""
    spec = get_board_spec(board)
    if model not in spec.supports_models:
        supported = ", ".join(sorted(spec.supports_models)) or "(none)"
        raise ValueError(
            f"Model {model!r} is not supported on board {board.name.lower()!r}. "
            f"Supported models for this board: {supported}"
        )
    # Also require an explicit deployment row.
    get_deployment(model, board)
