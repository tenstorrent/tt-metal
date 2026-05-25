# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Dump STFT ``X_real`` / ``X_imag`` tensors before ``atan2`` (TT and reference paths).

Enable with ``KOKORO_DUMP_STFT_XY=1``.  Output directory: ``KOKORO_DUMP_STFT_XY_DIR``
(default: ``/tmp/pytest-of-ubuntu/kokoro_stft_xy`` — fixed path, files overwritten each run).

Each tag writes (overwrite)::

    {tag}_X_real.pt
    {tag}_X_imag.pt
    {tag}_meta.json
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

import torch

_DEFAULT_DUMP_DIR = Path("/tmp/pytest-of-ubuntu/kokoro_stft_xy")


def stft_xy_dump_enabled() -> bool:
    return os.getenv("KOKORO_DUMP_STFT_XY", "").strip().lower() in ("1", "true", "yes")


def stft_xy_dump_dir() -> Path:
    return Path(os.getenv("KOKORO_DUMP_STFT_XY_DIR", str(_DEFAULT_DUMP_DIR)))


def stft_xy_dump_paths(tag: str) -> tuple[Path, Path, Path]:
    """Return ``(X_real path, X_imag path, meta.json path)`` for a dump tag."""
    d = stft_xy_dump_dir()
    return d / f"{tag}_X_real.pt", d / f"{tag}_X_imag.pt", d / f"{tag}_meta.json"


def reset_stft_xy_dump_counter() -> None:
    """No-op (kept for test compatibility; dumps use stable names per tag)."""


def dump_stft_xy_if_enabled(
    X_real: torch.Tensor,
    X_imag: torch.Tensor,
    *,
    tag: str,
    source: str,
    extra_meta: Optional[dict[str, Any]] = None,
) -> None:
    """Save fp32 ``[B, K, F]`` (or broadcastable) real/imag STFT bins to disk (overwrite)."""
    if not stft_xy_dump_enabled():
        return

    x_cpu = X_real.detach().float().contiguous().cpu()
    y_cpu = X_imag.detach().float().contiguous().cpu()
    if x_cpu.shape != y_cpu.shape:
        raise ValueError(f"X_real shape {x_cpu.shape} != X_imag shape {y_cpu.shape}")

    dump_dir = stft_xy_dump_dir()
    dump_dir.mkdir(parents=True, exist_ok=True)
    real_path, imag_path, meta_path = stft_xy_dump_paths(tag)

    torch.save(x_cpu, real_path)
    torch.save(y_cpu, imag_path)
    meta: dict[str, Any] = {
        "tag": tag,
        "source": source,
        "shape": list(x_cpu.shape),
        "dtype": str(x_cpu.dtype),
        "dump_dir": str(dump_dir),
    }
    if extra_meta:
        meta.update(extra_meta)
    meta_path.write_text(json.dumps(meta, indent=2))
