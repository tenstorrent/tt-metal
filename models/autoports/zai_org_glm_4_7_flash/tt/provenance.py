# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Tie a generated artifact to the exact source that produced it.

Kept free of any ttnn (and torch) import so the pure-CSV tools can stamp their
output too, which is why it is its own module rather than a helper in
``model.py``.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: Hashed for every artifact: the stage implementation plus the decoder stack
#: it stands on. A script's own path is appended by the caller.
SOURCES = (
    "tt/provenance.py",
    "tt/model.py",
    "tt/generator.py",
    "tt/optimized_decoder.py",
    "tt/fused_decoder.py",
    "tt/functional_decoder.py",
)


def source_manifest(extra_paths=()):
    """sha256 prefixes of the stage-owned source, for stamping evidence files."""
    paths = [ROOT / rel for rel in SOURCES] + [Path(p) for p in extra_paths]
    manifest = {}
    for path in paths:
        if not path.is_file():
            continue
        try:
            key = str(path.resolve().relative_to(ROOT))
        except ValueError:
            key = path.name
        manifest[key] = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    return manifest
