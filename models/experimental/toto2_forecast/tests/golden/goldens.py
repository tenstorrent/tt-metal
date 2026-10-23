# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Golden files written by ``make_goldens.py`` (reference venv), read here from python_env (no reference import)."""

from __future__ import annotations

import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path

import torch

GOLDEN_DIR = Path(os.environ.get("TOTO2_GOLDEN_DIR", "/home/andy/bringup/goldens/toto2"))
MANIFEST = Path(__file__).resolve().parent / "MANIFEST.json"


def golden_path(ckpt: str, case: str) -> Path:
    return GOLDEN_DIR / ckpt / f"{case}.pt"


def have_golden(ckpt: str, case: str) -> bool:
    return golden_path(ckpt, case).is_file()


@lru_cache(maxsize=4)
def load_golden(ckpt: str, case: str, verify: bool = False) -> dict:
    """The golden dict (``inputs``, ``captures``, ``cpu_s``, ...); ``verify`` checks its sha256 against MANIFEST.json."""
    path = golden_path(ckpt, case)
    if verify:
        expected = json.loads(MANIFEST.read_text())["goldens"][f"{ckpt}/{case}.pt"]["sha256"]
        h = hashlib.sha256(path.read_bytes()).hexdigest()
        assert h == expected, f"{path}: sha256 {h} != MANIFEST {expected}"
    return torch.load(path, weights_only=False)


def rel_err(got: torch.Tensor, expected: torch.Tensor) -> float:
    """``||got - expected|| / ||expected||`` in float64."""
    got, expected = got.double(), expected.double()
    return ((got - expected).norm() / expected.norm()).item()


def pcc(got: torch.Tensor, expected: torch.Tensor) -> float:
    a, b = got.double().flatten(), expected.double().flatten()
    a, b = a - a.mean(), b - b.mean()
    return (a @ b / (a.norm() * b.norm())).item()
