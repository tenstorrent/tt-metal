# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint stamp for Hy4-Preview weight caches (same pattern as deepseek_v41_flash/tt/ckpt_stamp.py).

Decision D0-1: checkpoint id = sha256 of config.json and of model.safetensors.index.json. Unlike DSV4's stamp there is no sampling of shard bytes
(a 1.5 TB bf16 checkpoint: the index lists every tensor and shard, a different checkpoint of the same model with identical config and index is not expected;
pass ``sample=True`` to additionally hash 64 KiB at 1/3 and 2/3 of every shard, 262 small reads).
``ensure_stamp(cache_root)`` writes ``ckpt_id.json`` on first use and raises when the file names a different checkpoint.
"""

import hashlib
import json
import os
from typing import Dict, Optional

DEFAULT_CKPT = "/mnt/tt-data/Hy4-Preview"
ENV = "HY4_CKPT"
STAMP = "ckpt_id.json"
SAMPLE = 64 * 1024
_ids: Dict[tuple, dict] = {}
_checked = set()


def _sha_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def checkpoint_identity(ckpt_dir: str, sample: bool = False) -> dict:
    """{'config_sha256', 'index_sha256', 'ckpt_id'[, 'sample_sha256']}; ckpt_id = sha256(config_sha + index_sha [+ sample_sha])."""
    key = (ckpt_dir, sample)
    if key in _ids:
        return _ids[key]
    out = {
        "config_sha256": _sha_file(os.path.join(ckpt_dir, "config.json")),
        "index_sha256": _sha_file(os.path.join(ckpt_dir, "model.safetensors.index.json")),
    }
    h = hashlib.sha256((out["config_sha256"] + out["index_sha256"]).encode())
    if sample:
        s = hashlib.sha256()
        for n in sorted(f for f in os.listdir(ckpt_dir) if f.endswith(".safetensors")):
            path = os.path.join(ckpt_dir, n)
            size = os.stat(path).st_size
            s.update(f"{n}:{size}\n".encode())
            with open(path, "rb") as f:
                for frac in (1, 2):
                    f.seek(size * frac // 3)
                    s.update(f.read(SAMPLE))
        out["sample_sha256"] = s.hexdigest()
        h.update(out["sample_sha256"].encode())
    out["ckpt_id"] = h.hexdigest()
    _ids[key] = out
    return out


def checkpoint_id(ckpt_dir: str, sample: bool = False) -> str:
    return checkpoint_identity(ckpt_dir, sample)["ckpt_id"]


def ensure_stamp(cache_root: Optional[str], ckpt_dir: Optional[str] = None, sample: bool = False) -> None:
    """Write ``<cache_root>/ckpt_id.json`` if missing; raise RuntimeError if it names a different checkpoint. No-op for cache_root None / '0'."""
    if not cache_root or cache_root == "0":
        return
    ckpt_dir = ckpt_dir or os.environ.get(ENV, DEFAULT_CKPT)
    key = (str(cache_root), str(ckpt_dir), sample)
    if key in _checked:
        return
    ident = checkpoint_identity(ckpt_dir, sample)
    path = os.path.join(cache_root, STAMP)
    if os.path.exists(path):
        with open(path) as f:
            old = json.load(f).get("ckpt_id")
        if old != ident["ckpt_id"]:
            raise RuntimeError(
                f"weight cache {cache_root} was built from checkpoint id {old}, but {ENV}={ckpt_dir} has id {ident['ckpt_id']}: "
                f"use a separate cache root for this checkpoint (stamp file {path})"
            )
    else:
        os.makedirs(cache_root, exist_ok=True)
        tmp = f"{path}.{os.getpid()}"
        with open(tmp, "w") as f:
            json.dump({**ident, "ckpt_dir": str(ckpt_dir)}, f)
        os.replace(tmp, path)
    _checked.add(key)
