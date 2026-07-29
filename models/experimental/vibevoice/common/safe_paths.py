# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Path-containment helpers for checkpoint / asset / artifact I/O.

Every file this model reads or writes lives under a known base directory: the
VibeVoice-1.5B checkpoint dir (``VIBEVOICE_MODEL_PATH``, an HF snapshot, or the bundled
``weights/``), the auto-downloaded ``resources/`` tree, or a caller-supplied output dir.
``safe_join`` pins the join to that base, so neither a crafted env var / CLI argument nor
a shard name read out of an untrusted ``model.safetensors.index.json`` can walk outside
it.

Containment is checked on the *lexical* absolute path (``os.path.abspath``) rather than
``Path.resolve()``: HF hub snapshots populate a checkpoint directory with symlinks into a
sibling ``blobs/`` tree, and resolving them would make every legitimate shard look like an
escape.

Scope: these helpers cover path *construction* — existence checks, ``open`` targets,
mkdir targets. Use ``safe_join`` whenever a path component comes from outside the code
(env var, CLI argument, checkpoint JSON); use ``safe_output_path`` when the operator names
an artifact file outright and there is no enclosing base to pin it to.
"""

from __future__ import annotations

import os
from pathlib import Path


def safe_join(base: Path | str, *parts: str) -> Path:
    """Return ``base/*parts``, guaranteed to stay inside ``base``.

    Raises ``ValueError`` if the joined path escapes ``base`` — via ``..`` segments, or
    because a part is itself absolute (``os.path.join`` would silently discard the base).
    """
    base_dir = os.path.abspath(str(base))
    target = os.path.abspath(os.path.join(base_dir, *(str(p) for p in parts)))
    if target != base_dir and not target.startswith(base_dir + os.sep):
        raise ValueError(f"refusing path {target!r}: outside base directory {base_dir!r}")
    return Path(target)


def safe_output_path(path: Path | str, *, suffix: str | None = None) -> Path:
    """Normalize an operator-supplied output path to an absolute path.

    Used for artifact writes (wavs, CSV diagnostics, metrics JSON) where the operator
    names the file or directory outright, so there is no enclosing base to pin to.
    ``suffix`` asserts the expected file extension when the caller knows it.
    """
    target = Path(os.path.abspath(str(path)))
    if suffix is not None and target.suffix != suffix:
        raise ValueError(f"refusing output path {target!r}: expected a {suffix} file")
    return target
