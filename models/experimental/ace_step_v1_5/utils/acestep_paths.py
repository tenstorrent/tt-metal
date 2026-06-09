# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Paths for the demo-only ACE-Step host preprocess package."""

from __future__ import annotations

import os
from pathlib import Path

# Root added to ``sys.path`` so ``import acestep.*`` resolves for handler init and HF
# ``trust_remote_code`` modeling (``acestep.models.*``).
HOST_PREPROCESS_ROOT = Path(__file__).resolve().parent.parent / "host_preprocess"

# Backward-compatible alias for callers that still use the old vendored name.
VENDORED_ACESTEP_ROOT = HOST_PREPROCESS_ROOT

_WELL_KNOWN_REPO_ROOTS = (
    Path.home() / "proj_sdk" / "ACE-Step-1.5",
    Path.home() / "ACE-Step-1.5",
    Path("/opt") / "ACE-Step-1.5",
)


def default_acestep_repo_root() -> Path:
    """Return the bundled host-preprocess tree (contains ``acestep/``)."""
    return HOST_PREPROCESS_ROOT.resolve()


def acestep_package_dir() -> Path:
    """Directory containing the ``acestep`` Python package."""
    return (HOST_PREPROCESS_ROOT / "acestep").resolve()


def resolve_acestep_repo_root(
    *,
    ckpt_dir: str | Path | None = None,
    ace_step_repo_root: str | Path | None = None,
) -> Path | None:
    """Return a directory containing an ``acestep/`` package.

    Search order:

    1. Explicit ``ace_step_repo_root`` argument.
    2. ``ACE_STEP_REPO_ROOT`` environment variable.
    3. Bundled ``host_preprocess/`` copy (default for TTNN demos).
    4. Walk up from ``ckpt_dir`` (looks for an ``acestep/`` sibling).
    5. Well-known external ACE-Step-1.5 install paths.
    """
    candidates: list[Path] = []
    if ace_step_repo_root:
        candidates.append(Path(ace_step_repo_root).expanduser().resolve())
    env = os.environ.get("ACE_STEP_REPO_ROOT")
    if env:
        candidates.append(Path(env).expanduser().resolve())
    candidates.append(HOST_PREPROCESS_ROOT)

    if ckpt_dir:
        cur = Path(ckpt_dir).expanduser().resolve()
        for _ in range(8):
            candidates.append(cur)
            if cur.parent == cur:
                break
            cur = cur.parent
    candidates.extend(_WELL_KNOWN_REPO_ROOTS)

    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if (candidate / "acestep" / "__init__.py").is_file():
            return candidate
    return None


def ensure_host_preprocess_on_path() -> Path:
    """Add ``host_preprocess/`` to ``sys.path`` and return that root."""
    from models.experimental.ace_step_v1_5.demo.ref_decoder_compare import ensure_acestep_repo_on_path

    return ensure_acestep_repo_on_path(HOST_PREPROCESS_ROOT)


# Backward-compatible alias for tests written against the old vendored tree name.
ensure_vendored_acestep_on_path = ensure_host_preprocess_on_path
