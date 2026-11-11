# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Distributed checkpointing for tt-train: every device writes its own shards, any mesh can load them.

A checkpoint is a directory of one safetensors file per device plus ``index.json``, which records where every chunk
of every tensor lives and how it maps to a layout independent of the parallelism it was saved with. Saving reads
each shard off its device once and writes it from that device's own thread; there is no host-side gather. Loading
intersects each destination shard with the saved chunks and reads only the bytes it needs, so a checkpoint saved on
one mesh shape and parallelism strategy loads onto another.

    import ttml.distributed_checkpointing as dcp
    dcp.save(ckpt_dir, model=model, optimizer=optimizer, scheduler=scheduler, dataloader=loader, header={"step": step})
    result = dcp.load(ckpt_dir, model=model, optimizer=optimizer, scheduler=scheduler, dataloader=loader, rng=True)

The scheduler, the dataloader and the random generators (the ttml generator, ttml.init's, python, numpy, torch) are
saved beside the header and restored on request; see ``_train_state``.

Topology labels are trusted. The one fact a label cannot carry, the segment order inside a fused projection, is
declared by the layer that fuses it (``ColumnParallelLinear(fused_segments=...)``), and a padded vocab weight
declares its logical row count; see ``_layouts``.
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
from ._load import LoadResult, load
from ._save import SaveResult, save
from ._train_state import capture_rng, read_train_state, restore_rng


__all__ = [
    "CheckpointError",
    "CoverageError",
    "DtypeMismatch",
    "FSDPStateError",
    "IncompleteCheckpoint",
    "LayoutError",
    "LazyParameterError",
    "LoadResult",
    "MissingKeyError",
    "RoleMismatch",
    "SaveResult",
    "ShapeMismatch",
    "TopologyError",
    "UnexpectedKeyError",
    "VersionError",
    "capture_rng",
    "find_latest",
    "is_complete",
    "load",
    "parse_dir_name",
    "read_header",
    "read_index",
    "read_train_state",
    "restore_rng",
    "save",
    "step_dir_name",
]
