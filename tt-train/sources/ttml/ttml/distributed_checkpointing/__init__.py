# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Distributed checkpointing for tt-train: every device writes its own shards, any mesh can load them.

A checkpoint is a directory of one safetensors file per device plus ``index.json``, which records where every chunk
of every tensor lives and how it maps to a layout independent of the parallelism it was saved with. This package
level holds the format and the planning, none of which touches a device: boxes and the device-to-box mapping from
topology labels (``_topology``), the canonical-layout transforms (``_layouts``), the safetensors writer and reader
(``_safetensors``), the index and header files with naming and atomic finalization (``_index``), and the save and
load planners (``_planner``). The save and load entry points that drive devices stack on top of it.

    import ttml.distributed_checkpointing as dcp
    dcp.read_header(ckpt_dir); dcp.read_index(ckpt_dir); dcp.is_complete(ckpt_dir); dcp.find_latest(root)

Topology labels are trusted. The one fact a label cannot carry, the segment order inside a fused projection, is
declared by the layer that fuses it, and a padded vocab weight declares its logical row count; see ``_layouts``.
"""

from ._errors import (
    CheckpointError,
    CoverageError,
    DtypeMismatch,
    FSDPStateError,
    IncompleteCheckpoint,
    LayoutError,
    LazyParameterError,
    MissingKeyError,
    RoleMismatch,
    ShapeMismatch,
    TopologyError,
    UnexpectedKeyError,
    VersionError,
)
from ._index import find_latest, is_complete, parse_dir_name, read_index, step_dir_name
from ._index import read_user_header as read_header


__all__ = [
    "CheckpointError",
    "CoverageError",
    "DtypeMismatch",
    "FSDPStateError",
    "IncompleteCheckpoint",
    "LayoutError",
    "LazyParameterError",
    "MissingKeyError",
    "RoleMismatch",
    "ShapeMismatch",
    "TopologyError",
    "UnexpectedKeyError",
    "VersionError",
    "find_latest",
    "is_complete",
    "parse_dir_name",
    "read_header",
    "read_index",
    "step_dir_name",
]
