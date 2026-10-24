"""Checkpoint stamp for the on-disk weight caches.

The caches (routed experts, embed/head, vision, MTP, unified prefill experts) are keyed by directory / file name only, not by checkpoint.
A cache built from one checkpoint and read with another would silently give a mixed model. ``ensure_stamp(cache_root)`` writes
``ckpt_id.json`` into a cache root on first use and raises if the file exists and names a different checkpoint.

Caches that predate this (no stamp) are stamped with whatever checkpoint is configured on first use: correct only if DSV41_CKPT is the checkpoint that built them.
"""

import hashlib
import json
import os

DEFAULT_CKPT = "/mnt/tt-data/ssinghal/deepseek-v41-flash"
STAMP = "ckpt_id.json"
SAMPLE = 64 * 1024
_ids = {}  # ckpt_dir -> id
_checked = set()  # (cache_root, ckpt_dir) pairs already verified in this process


def checkpoint_id(ckpt_dir):
    """sha256 over the sorted shard names + byte sizes, the hash of the safetensors index, and two 64 KiB samples (at 1/3 and 2/3) of every shard.

    Two checkpoints of the same model have identical names, sizes and index (only the values differ), so the samples are what tells them apart; 96 small reads, no full shard read.
    """
    h = hashlib.sha256()
    for n in sorted(f for f in os.listdir(ckpt_dir) if f.endswith(".safetensors")):
        path = os.path.join(ckpt_dir, n)
        size = os.stat(path).st_size
        h.update(f"{n}:{size}\n".encode())
        with open(path, "rb") as f:
            for frac in (1, 2):
                f.seek(size * frac // 3)
                h.update(f.read(SAMPLE))
    with open(os.path.join(ckpt_dir, "model.safetensors.index.json"), "rb") as f:
        h.update(hashlib.sha256(f.read()).digest())
    return h.hexdigest()


def ensure_stamp(cache_root, ckpt_dir=None):
    """Write ``<cache_root>/ckpt_id.json`` if missing; raise RuntimeError if it names a different checkpoint. No-op for cache_root None / '0'."""
    if not cache_root or cache_root == "0":
        return
    ckpt_dir = ckpt_dir or os.environ.get("DSV41_CKPT", DEFAULT_CKPT)
    key = (str(cache_root), str(ckpt_dir))
    if key in _checked:
        return
    cid = _ids.get(ckpt_dir) or _ids.setdefault(ckpt_dir, checkpoint_id(ckpt_dir))
    path = os.path.join(cache_root, STAMP)
    if os.path.exists(path):
        with open(path) as f:
            old = json.load(f).get("ckpt_id")
        if old != cid:
            raise RuntimeError(
                f"weight cache {cache_root} was built from checkpoint id {old}, but DSV41_CKPT={ckpt_dir} has id {cid}: "
                f"use a separate cache root for this checkpoint (stamp file {path})"
            )
    else:
        os.makedirs(cache_root, exist_ok=True)
        tmp = f"{path}.{os.getpid()}"
        with open(tmp, "w") as f:
            json.dump({"ckpt_id": cid, "ckpt_dir": str(ckpt_dir)}, f)
        os.replace(tmp, path)
    _checked.add(key)
