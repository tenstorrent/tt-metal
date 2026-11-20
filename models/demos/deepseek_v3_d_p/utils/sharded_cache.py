# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Weight-cache names and stamps for tensors sharded over BOTH mesh axes (2D tensor parallel).

A tensorbin's name carries only dtype and layout, and the weight-cache directory key only the mesh shape and the SP
axis (``fast_cache_checker.mesh_cache_key``). A 2D-sharded weight also depends on which mesh axis carries which
dimension, so a cache built for one split would load, silently, as another. Two guards:

* **Names from the layout** (``layout_tag``): ``{rows}x{cols}`` plus, for each role, ``{role}{size}ax{axis}``, e.g.
  ``8x4_sp8ax0_tp4ax1`` or ``8x4_h4ax1_i8ax0``. Two layouts that place any role differently never share a name.
* **A stamp next to the shards** (``LayoutCacheGuard``): ``{name}.layout.stamp`` holds the layout descriptor (module,
  format, mesh, axes, dtypes) as JSON. Loading shards with no stamp, or with a stamp that disagrees, raises. A code
  change that moves shards without moving any axis bumps the module's format, which changes the descriptor.

Used by ``ttMLA2DTP`` (``tt/mla/mla_2d.py``) and ``TtMlp2D`` (``tt/moe/tt_mlp_2d.py``) so the two cannot drift. Same
idea as the Hy4 MLA format stamp (``hy4_checkpoint.mla_cache_stamp``), which marks a whole layer rebuildable; this one
refuses instead, because a 2D shard from another split is never safe to read.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Mapping, Optional, Sequence

STAMP_SUFFIX = ".layout.stamp"
_ROLE = re.compile(r"^[a-z]+$")


def layout_tag(mesh_shape: Sequence[int], axes: Mapping[str, int]) -> str:
    """``{rows}x{cols}_{role}{size}ax{axis}...`` for ``axes`` (role -> mesh axis), in the given order.

    Every mesh axis must carry exactly one role, so the tag pins the whole split."""
    rows, cols = (int(n) for n in mesh_shape)
    shape = (rows, cols)
    if sorted(axes.values()) != [0, 1]:
        raise ValueError(f"each mesh axis must carry exactly one role, got {dict(axes)}")
    parts = [f"{rows}x{cols}"]
    for role, axis in axes.items():
        if not _ROLE.match(role):
            raise ValueError(f"role {role!r} must be lowercase letters")
        parts.append(f"{role}{shape[axis]}ax{axis}")
    return "_".join(parts)


def layout_descriptor(
    module: str,
    cache_format: int,
    mesh_shape: Sequence[int],
    axes: Mapping[str, int],
    dtypes: Mapping[str, str],
    extra: Optional[Mapping[str, str]] = None,
) -> dict:
    """The JSON-able record a stamp holds. ``extra``: other properties of the cached values that must match, e.g. the
    RoPE pairing the weights were read in (``ttMLA2DTP``)."""
    d = {
        "module": module,
        "format": int(cache_format),
        "mesh": [int(n) for n in mesh_shape],
        "axes": {role: int(axis) for role, axis in axes.items()},
        "dtypes": {name: str(dtype) for name, dtype in sorted(dtypes.items())},
    }
    if extra:
        d.update({str(k): str(v) for k, v in sorted(extra.items())})
    return d


class LayoutCacheGuard:
    """Stamp-checked access to one 2D-sharded cache entry.

    ``name``: the stamp's basename (``{name}.layout.stamp``) in ``cache_path``. ``shard_glob``: the glob matching the
    entry's tensorbins in ``cache_path``. Call ``check()`` before reading or writing the shards and ``write()`` after
    they are on disk."""

    def __init__(self, cache_path: Optional[Path], name: str, shard_glob: str, descriptor: dict):
        self.cache_path = None if cache_path is None else Path(cache_path)
        self.name = name
        self.shard_glob = shard_glob
        self.descriptor = descriptor

    @property
    def stamp(self) -> Optional[Path]:
        return None if self.cache_path is None else self.cache_path / f"{self.name}{STAMP_SUFFIX}"

    def shards(self) -> list[Path]:
        return [] if self.cache_path is None else sorted(self.cache_path.glob(self.shard_glob))

    def stamped(self) -> bool:
        """The stamp exists and matches (``check_cache_complete`` requires it on top of the shards)."""
        return self.stamp is not None and self.stamp.is_file() and self._read() == self.descriptor

    def _read(self) -> Optional[dict]:
        try:
            return json.loads(self.stamp.read_text())
        except (OSError, ValueError):
            return None

    def check(self) -> None:
        """Raise if shards exist without a stamp, or the stamp names another layout. No shards, no stamp: a cold
        cache, fine (the caller builds it and then calls ``write``)."""
        if self.stamp is None:
            return
        shards = self.shards()
        if self.stamp.is_file():
            found = self._read()
            if found != self.descriptor:
                raise RuntimeError(
                    f"2D weight cache {self.cache_path}/{self.name}: its stamp {self.stamp.name} records layout "
                    f"{found}, but this build needs {self.descriptor}. The shards were cut for another layout; delete "
                    f"{self.shard_glob} and the stamp to rebuild them."
                )
            return
        if shards:
            raise RuntimeError(
                f"2D weight cache {self.cache_path}/{self.name}: {len(shards)} shard file(s) ({self.shard_glob}) but no "
                f"layout stamp {self.stamp.name}, so their layout is unknown. Delete them to rebuild for "
                f"{self.descriptor}."
            )

    def write(self) -> None:
        if self.stamp is not None:
            self.stamp.write_text(json.dumps(self.descriptor, sort_keys=True))
