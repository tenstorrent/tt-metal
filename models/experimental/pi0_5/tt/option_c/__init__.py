# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
pi0.5 Option C — heterogeneous 3-stage pipeline on 28 of 32 Blackhole chips.

See `docs/PI0_5_GALAXY_DEPLOYMENT_PLAN.md` §3 for the design rationale, and
`README.md` in this dir for the file map and L1-placement plan.

Key differences vs Option B:

  - No TP within stage — each chip owns whole layer(s) by itself, so there
    are no all_reduces at all. The 1.25 ms / layer all_reduce tax measured
    in test_option_b_benchmark.py disappears.
  - Heterogeneous submeshes (4-vision / 18-prefill / 6-denoise + 4 spare),
    enabled by `MeshDevice.create_submesh(shape, offset)`.
  - Every weight / bias / activation lives in L1 (memory_config=L1_MEMORY_CONFIG)
    — safe because the absence of collectives means we can use the full L1
    budget for storage without colliding with all_reduce's static CB region.
"""

from .stages import StageLayout, StageSpec, build_default_layout, build_shrunk_layout
from .mesh_setup import open_galaxy_mesh, describe_submesh
from .vision_slice import Pi0_5OptionCVisionSlice
from .vlm_slice import Pi0_5OptionCVLMSlice
from .expert_slice import Pi0_5OptionCExpertSlice
from .suffix_slice import Pi0_5OptionCSuffixSlice
from .kv_migration import KVMigration
from .transport import send_activation_via_host, send_per_chip_activation_via_host
from .stage_vision import StageVision
from .stage_prefill import StagePrefill
from .stage_denoise import StageDenoise
from .pipeline import Pi0_5PipelineC, StageTimingsC

__all__ = [
    # Layout + mesh.
    "StageLayout",
    "StageSpec",
    "build_default_layout",
    "build_shrunk_layout",
    "open_galaxy_mesh",
    "describe_submesh",
    # Per-stage slices (weights + forward).
    "Pi0_5OptionCVisionSlice",
    "Pi0_5OptionCVLMSlice",
    "Pi0_5OptionCExpertSlice",
    "Pi0_5OptionCSuffixSlice",
    # Stage orchestrators.
    "StageVision",
    "StagePrefill",
    "StageDenoise",
    # KV migration + transport + pipeline driver.
    "KVMigration",
    "send_activation_via_host",
    "send_per_chip_activation_via_host",
    "Pi0_5PipelineC",
    "StageTimingsC",
]
