# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Path-containment helper for checkpoint / asset / artifact I/O.

Every file this package reads or writes lives under a known base directory: a
checkpoint dir (``HUNYUAN_MODEL_DIR`` or an HF hub snapshot), the bundled
``ref/tokenizer/assets``, or a caller-supplied output dir. ``safe_join`` pins the
join to that base so neither a crafted env var / CLI argument nor a shard name
read out of an untrusted ``model.safetensors.index.json`` can walk outside it.

Containment is checked on the *lexical* absolute path (``os.path.abspath``), not
``Path.resolve()``: HF hub snapshots populate a directory with symlinks into a
sibling ``blobs/`` tree, and resolving them would make every legitimate shard
look like an escape.

Scope: these helpers cover path *construction* — existence checks, ``safe_open``
targets, mkdir targets. The handful of call sites that hand a checkpoint-derived
path straight to ``open`` write the same join-and-check out inline instead, so the
constraint is visible in the same function as the read (see
``weights._read_weight_index``, ``model_config.load_config``,
``tokenizer.hunyuan_tokenizer.load_config``, ``demo_i2i._checkpoint_json``). Keep the
two in step if this logic changes.
"""

from __future__ import annotations

import os
from pathlib import Path


def safe_join(base: Path | str, *parts: str) -> Path:
    """Return ``base/*parts``, guaranteed to stay inside ``base``.

    Raises ``ValueError`` if the joined path escapes ``base`` — via ``..``
    segments, or because a part is itself absolute (``os.path.join`` would
    silently discard the base).
    """
    base_dir = os.path.abspath(str(base))
    target = os.path.abspath(os.path.join(base_dir, *(str(p) for p in parts)))
    if target != base_dir and not target.startswith(base_dir + os.sep):
        raise ValueError(f"refusing path {target!r}: outside base directory {base_dir!r}")
    return Path(target)


def safe_output_path(path: Path | str, *, suffix: str | None = None) -> Path:
    """Normalize an operator-supplied output path to an absolute path.

    Used for artifact writes (CSV sweeps, PNGs) where the operator names the file
    outright, so there is no enclosing base to pin to. ``suffix`` asserts the
    expected file extension when the caller knows it.
    """
    target = Path(os.path.abspath(str(path)))
    if suffix is not None and target.suffix != suffix:
        raise ValueError(f"refusing output path {target!r}: expected a {suffix} file")
    return target
