# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""How a TTNN weight-cache miss is written while a model builds.

``ttnn.as_tensor`` writes a cache miss with ``DumpTensorMode.DISTRIBUTED_GATHER`` by default: a host all-gather
across every rank of the distributed world, a write from global rank 0 only, and a barrier. That is right when all
ranks build the same tensors over one mesh, but a pipeline-parallel prefill run builds different layers on every
rank, so its cache misses never pair up: a cold fill deadlocks, and rank 0 would write the other ranks' files.

A pipeline rank owns a whole host-local mesh, so its tensors are complete on its own host and it can write them
itself (``DumpTensorMode.LOCAL``, written to a temp file and renamed, so ranks sharing one cache directory on a
shared drive never see each other's partial files). ``TtPrefillRuntime`` builds a multi-rank pipeline inside
``per_rank_cache_dumps()``; everything else keeps the default.
"""

from __future__ import annotations

from contextlib import contextmanager

import ttnn

_mode = ttnn.DumpTensorMode.DISTRIBUTED_GATHER


def weight_cache_dump_mode() -> ttnn.DumpTensorMode:
    """The ``cache_dump_mode`` for ``ttnn.as_tensor`` calls that may write a weight-cache file."""
    return _mode


def resolve_cache_dump_mode(mode: ttnn.DumpTensorMode | None) -> ttnn.DumpTensorMode:
    """An explicit ``cache_dump_mode`` argument wins; ``None`` means the current default above."""
    return weight_cache_dump_mode() if mode is None else mode


@contextmanager
def per_rank_cache_dumps(enabled: bool = True):
    """Write weight-cache misses per rank (``LOCAL``) inside this block when ``enabled``."""
    global _mode
    previous = _mode
    if enabled:
        _mode = ttnn.DumpTensorMode.LOCAL
    try:
        yield
    finally:
        _mode = previous
