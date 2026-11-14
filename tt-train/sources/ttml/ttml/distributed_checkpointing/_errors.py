# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Errors raised by ``ttml.distributed_checkpointing``. Every one names the key, the layouts or the file involved."""


class CheckpointError(RuntimeError):
    """Base class: a save or load that could not be carried out as asked."""


class IncompleteCheckpoint(CheckpointError):
    """The directory is a staging directory, or has no readable ``index.json`` (missing, truncated or malformed)."""


class VersionError(CheckpointError):
    """The checkpoint was written by a newer schema version than this reader knows."""


class LayoutError(CheckpointError):
    """A layout declaration or transform does not fit the tensor it describes."""


class ShapeMismatch(CheckpointError):
    """The checkpoint's canonical shape for a key differs from the live tensor's."""


class DtypeMismatch(CheckpointError):
    """The checkpoint's dtype for a key differs from the live tensor's."""


class MissingKeyError(CheckpointError):
    """A live tensor has no entry in the checkpoint (``strict=True``)."""


class UnexpectedKeyError(CheckpointError):
    """The checkpoint has an entry no live tensor wants (``strict=True``)."""


class CoverageError(CheckpointError):
    """Boxes that must tile a shape exactly do not: a key's chunks on save, the saved chunks of a checkpoint entry,
    or the planned reads of a destination shard on load."""


class RoleMismatch(CheckpointError):
    """A live tensor resolved to a checkpoint entry of another role (a parameter to optimizer state, say)."""


class TopologyError(CheckpointError):
    """A tensor's topology label cannot be used as a layout, or changed when it must not."""


class FSDPStateError(CheckpointError):
    """An FSDP module is not in its sharded state."""


class LazyParameterError(CheckpointError):
    """A parameter has not been materialized."""
