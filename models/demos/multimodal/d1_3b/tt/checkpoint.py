# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The pinned d1-3B checkpoint and its remote code (`prompt.py`, `api.py`)."""

import importlib.util
import os
import sys
import types
from pathlib import Path

D1_HF_MODEL = "LiquidAI/d1-3B"
D1_REVISION = "051bcc464b01b9f92942b364d9586b0ef5912432"


def resolve_checkpoint(hf_model=None):
    """Local snapshot dir of d1-3B at the pinned revision (or HF_MODEL if it is a directory)."""
    hf_model = hf_model or os.environ.get("HF_MODEL", D1_HF_MODEL)
    if os.path.isdir(hf_model):
        return Path(hf_model)
    from huggingface_hub import snapshot_download

    try:
        return Path(snapshot_download(hf_model, revision=D1_REVISION, local_files_only=True))
    except Exception:
        return Path(snapshot_download(hf_model, revision=D1_REVISION))


def remote_modules(ckpt_dir=None, anchor=None):
    """d1's own `prompt` and `api` modules from the checkpoint, so rendering and readout are HF's code.

    `anchor` is the module name of an already loaded remote class (e.g. the HF model's runner): transformers
    can load the same remote file twice under different package hashes, and the question classes of one copy
    fail the other's isinstance checks.
    """
    if anchor is not None:
        package = anchor.rsplit(".", 1)[0]
        return sys.modules[f"{package}.prompt"], sys.modules[f"{package}.api"]
    # Loaded directly rather than through transformers' remote-code loader, which on some versions resolves
    # `api.py`'s relative import next to the HF cache blob instead of the snapshot. Both files are stdlib only.
    package = f"_d1_remote_{abs(hash(str(Path(ckpt_dir).resolve())))}"
    if package not in sys.modules:
        root = types.ModuleType(package)
        root.__path__ = [str(ckpt_dir)]
        sys.modules[package] = root
        for name in ("prompt", "api"):
            spec = importlib.util.spec_from_file_location(f"{package}.{name}", Path(ckpt_dir) / f"{name}.py")
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
    return sys.modules[f"{package}.prompt"], sys.modules[f"{package}.api"]
