# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0
"""Where the Hugging Face Inkling2 reference comes from, for the tests that compare against it.

The reference modules need transformers >= 5.24 (``transformers.models.inkling2`` / ``inkling2_loop``). If this
python has them they run in process; otherwise set ``INKLING_REF_PYTHON`` to a python that has them, and the tests
run their reference side in that python as a subprocess. With neither, the tests that need it skip.

    INKLING_REF_PYTHON  python with transformers >= 5.24 (no ttnn needed there)
    INKLING_CONFIG      the model's config.json (default: $INKLING_CKPT/config.json); needed by some references
    INKLING_REF_CACHE   directory for cached reference outputs (default: a per-user temp directory)
"""

import importlib.util
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

REF_PYTHON = os.environ.get("INKLING_REF_PYTHON")


def _has_module(name):
    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:
        return False


def in_process(module="transformers.models.inkling2_loop"):
    return _has_module(module)


def available(module="transformers.models.inkling2_loop"):
    return in_process(module) or bool(REF_PYTHON and os.path.isfile(REF_PYTHON))


def config_path():
    """The model's config.json, or None if neither INKLING_CONFIG nor $INKLING_CKPT/config.json exists."""
    p = os.environ.get("INKLING_CONFIG") or (
        os.path.join(os.environ["INKLING_CKPT"], "config.json") if os.environ.get("INKLING_CKPT") else None
    )
    return p if p and os.path.isfile(p) else None


def require(module="transformers.models.inkling2_loop", config=False):
    """Skip the calling test unless the reference (and, with ``config``, the model config) is available."""
    if not available(module):
        pytest.skip(f"HF reference unavailable: no {module} here and INKLING_REF_PYTHON unset")
    if config and config_path() is None:
        pytest.skip("model config.json not found: set INKLING_CONFIG or INKLING_CKPT")


def cache_dir(name):
    """Directory for cached reference outputs of one test file."""
    root = os.environ.get("INKLING_REF_CACHE") or os.path.join(
        tempfile.gettempdir(), f"inkling2_mini_ref-{os.getuid()}"
    )
    d = Path(root) / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def run(script, *args, isolated=False):
    """Run ``script`` (a test file's reference side) under the reference python."""
    py = sys.executable if in_process() else REF_PYTHON
    env = dict(os.environ)
    if config_path():
        env["INKLING_CONFIG"] = config_path()
    subprocess.run([py, *(["-I"] if isolated else []), os.path.abspath(script), *map(str, args)], check=True, env=env)


def finite_min(pccs):
    """min() of a list of PCCs, after asserting every one is finite (min([1.0, nan]) is 1.0)."""
    pccs = list(pccs)
    bad = [i for i, p in enumerate(pccs) if not math.isfinite(p)]
    assert not bad, f"non-finite PCC at indices {bad[:10]}: {[pccs[i] for i in bad[:10]]}"
    return min(pccs)
