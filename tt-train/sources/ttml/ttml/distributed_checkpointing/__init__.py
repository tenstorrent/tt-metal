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
    result = dcp.load(ckpt_dir, model=model, optimizer=optimizer, scheduler=scheduler, dataloader=loader)

The scheduler, the dataloader and the random generators (the ttml generator, ttml.init's, python, numpy, torch) are
saved beside the header. A load restores the scheduler and dataloader passed, and with them the generators (``rng``
defaults to restoring them whenever a scheduler or dataloader is); see ``load`` and ``_train_state``.

Under ``tt-run`` with several ranks (a mesh spread over hosts, or one mesh per host) ``save`` and ``load`` are
collective: every rank of the world calls them at the same point, with the same directory and options (checked before
anything is written or read); each writes and reads only the shards of the coordinates it holds, and rank 0 merges the
ranks' parts into one ``index.json`` (``_coordinator``). A failure on any rank makes every rank raise the same error
class. While a save or load runs, no other point-to-point MPI traffic may be outstanding between the ranks, and no
other code may use the MPI tag ``DCP_TAG`` (``0x7DC0``) on the world communicator: the ranks' metadata travels under
it. A single process is the group of one and takes the same path.

Every failure of ``save`` and ``load``, on one process as on many, is a ``CheckpointError`` or one of its subclasses
(``MissingKeyError``, ``IncompleteCheckpointError``, ...). An error from outside this package (an ``OSError`` for a
full disk, a ``TypeError`` for a bad argument) is raised as a ``CheckpointError`` that names the step that failed and
the error, which is its ``__cause__`` on the rank where it was raised. A ``CheckpointFatalError`` means the ranks can no
longer agree: end the job (see its docstring).

Topology labels are trusted. The one fact a label cannot carry, the segment order inside a fused projection, is
declared by the layer that fuses it (``ColumnParallelLinear(fused_segments=...)``), and a padded vocab weight
declares its logical row count; see ``_layouts``.
"""

from ._errors import (
    CheckpointError,
    CheckpointFatalError,
    CoverageError,
    DtypeMismatchError,
    IncompleteCheckpointError,
    LayoutError,
    LazyParameterError,
    MissingKeyError,
    RoleMismatchError,
    ShapeMismatchError,
    TopologyError,
    UnexpectedKeyError,
    VersionError,
)
from ._index import find_latest, is_complete, read_index, step_dir_name
from ._index import read_user_header as read_header
from ._load import LoadResult, load
from ._save import SaveResult, save
from ._train_state import capture_rng, read_train_state, restore_rng


__all__ = [
    "CheckpointError",
    "CheckpointFatalError",
    "CoverageError",
    "DtypeMismatchError",
    "IncompleteCheckpointError",
    "LayoutError",
    "LazyParameterError",
    "LoadResult",
    "MissingKeyError",
    "RoleMismatchError",
    "SaveResult",
    "ShapeMismatchError",
    "TopologyError",
    "UnexpectedKeyError",
    "VersionError",
    "capture_rng",
    "find_latest",
    "is_complete",
    "load",
    "read_header",
    "read_index",
    "read_train_state",
    "restore_rng",
    "save",
    "step_dir_name",
]
