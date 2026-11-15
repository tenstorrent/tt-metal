# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""TP32 routed-expert weight cache, layout tp32-v1 (contract: program/moe-tp32/LAYOUT.md, usage: program/moe-tp32/CACHE.md).

Per layer L: <root>/layer_<L>_tp32/{moe_w01_tp32_bfp4_r<c>.tensorbin, moe_w2_tp32_bfp4_r<c>.tensorbin (c = 0..31), moe_meta_tp32.json, moe_manifest_tp32.json}.
Each file = the host bfp4_b TILE tensor [1, 1, T*32, 256] of ONE chip (== what bank_tiles_tensor shards to that chip; bank b holds columns [32 b, 32 b + 32)).
Keys that must match on load (anything else raises, a stale cache is never read): layout tag, layer id, dtype, geometry, checkpoint id (root stamp ckpt_id.json + meta).
Env: HY4_TP32_CACHE_DIR (root; unset = feature off), HY4_TP32_CACHE_VERIFY=1 (sha256 of every file against the manifest on load), HY4_CKPT (checkpoint dir).
"""
import hashlib
import json
import os

import ttnn
from models.demos.blackhole.hy4_preview import ckpt_stamp

LAYOUT_TAG = "tp32-v1"
DTYPE_NAME = "bfloat4_b"
HIDDEN, INTER, NEXP, NCHIPS = 6144, 2048, 256, 32
T01 = 256 * 24 * 4  # tiles per bank, W01
T2 = 6 * 256 * 8  # tiles per bank, W2


def cache_root():
    r = os.environ.get("HY4_TP32_CACHE_DIR", "")
    return r if r and r != "0" else None


def layer_dir(root, layer):
    return os.path.join(root, f"layer_{layer}_tp32")


def fname(kind, c):
    return f"moe_{kind}_tp32_bfp4_r{c}.tensorbin"


def meta_for(layer, ckpt_dir=None):
    return {
        "layout": LAYOUT_TAG,
        "layer": int(layer),
        "ckpt_id": ckpt_stamp.checkpoint_id(ckpt_dir or os.environ.get(ckpt_stamp.ENV, ckpt_stamp.DEFAULT_CKPT)),
        "bfp4": True,
        "dtype": DTYPE_NAME,
        "gate_first": True,
        "hidden": HIDDEN,
        "inter": INTER,
        "experts": NEXP,
        "chips": NCHIPS,
        "tiles_per_bank": {"w01": T01, "w2": T2},
    }


def chip_host_tensor(tiles):
    """tiles [8 banks, Tn, 32, 32] (torch bf16) -> host bfp4_b TILE tensor [1, 1, Tn*32, 256] (same arrangement as bank_tiles_tensor)."""
    Tn = tiles.shape[1]
    g = tiles.permute(1, 2, 0, 3).reshape(1, 1, Tn * 32, 8 * 32).contiguous()
    return ttnn.from_torch(g, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 24), b""):
            h.update(blk)
    return h.hexdigest()


def build_layer(w0, w1, w2, layer, root, chips_per_pack=4, log=print):
    """CPU packing + dump of all 32 chips of one layer (needs a visible device only for ttnn import/cluster init; opens no mesh).
    w0, w1: [256, 6144, 2048]; w2: [256, 2048, 6144] ([in, out], bf16). Writes into a tmp dir, renames at the end (a layer dir is complete or absent).
    """
    from models.demos.blackhole.hy4_preview.tt.hy4_moe_tp32 import pack_chips

    ckpt_stamp.ensure_stamp(root)
    d = layer_dir(root, layer)
    tmp = f"{d}.tmp{os.getpid()}"
    os.makedirs(tmp, exist_ok=True)
    manifest = {}
    for c0 in range(0, NCHIPS, chips_per_pack):
        chips = list(range(c0, c0 + chips_per_pack))
        t01, t2 = pack_chips(w0, w1, w2, chips)
        for i, c in enumerate(chips):
            for kind, t in (("w01", t01[i]), ("w2", t2[i])):
                p = os.path.join(tmp, fname(kind, c))
                ttnn.dump_tensor(p, chip_host_tensor(t))
                manifest[os.path.basename(p)] = {"bytes": os.path.getsize(p), "sha256": sha256_file(p)}
        del t01, t2
    meta = meta_for(layer)
    with open(os.path.join(tmp, "moe_manifest_tp32.json"), "w") as f:
        json.dump({"layout": LAYOUT_TAG, "layer": int(layer), "files": manifest}, f, indent=0)
    with open(os.path.join(tmp, "moe_meta_tp32.json"), "w") as f:  # meta last: its presence marks a complete dir
        json.dump(meta, f)
    os.rename(tmp, d)
    return d


def check_layer(root, layer, ckpt_dir=None, verify=None):
    """-> layer dir if a valid cache exists; None if absent; raises RuntimeError on a stale/mismatching/corrupt cache."""
    d = layer_dir(root, layer)
    mp = os.path.join(d, "moe_meta_tp32.json")
    if not os.path.exists(mp):
        return None
    ckpt_stamp.ensure_stamp(root, ckpt_dir)
    with open(mp) as f:
        got = json.load(f)
    want = meta_for(layer, ckpt_dir)
    bad = {k: (got.get(k), v) for k, v in want.items() if got.get(k) != v}
    if bad:
        raise RuntimeError(f"stale TP32 cache {d}: meta mismatch (got, want) {bad}")
    if verify is None:
        verify = os.environ.get("HY4_TP32_CACHE_VERIFY", "0") == "1"
    if verify:
        with open(os.path.join(d, "moe_manifest_tp32.json")) as f:
            man = json.load(f)["files"]
        for n, e in man.items():
            p = os.path.join(d, n)
            if os.path.getsize(p) != e["bytes"] or sha256_file(p) != e["sha256"]:
                raise RuntimeError(f"corrupt TP32 cache file {p} (manifest mismatch)")
    return d


def bank_mem(md, Tn):
    dram_crs = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(md.dram_grid_size().x - 1, md.dram_grid_size().y - 1))}
    )
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(dram_crs, [Tn * 32, 32], ttnn.ShardOrientation.ROW_MAJOR),
    )


def load_host_chips(d, chips):
    """-> (list of host W01 tensors, list of host W2 tensors) for the chip ranks `chips`."""
    return (
        [ttnn.load_tensor(os.path.join(d, fname("w01", c))) for c in chips],
        [ttnn.load_tensor(os.path.join(d, fname("w2", c))) for c in chips],
    )


def load_layer(md, layer, root=None, chip0=0, ckpt_dir=None, verify=None):
    """-> (w01_t, w2_t) bank-sharded mesh tensors (same objects bank_tiles_tensor makes) loaded from the cache, or None if the layer is not cached.
    Mesh chip (r, col) gets rank chip0 + r * cols + col."""
    root = root or cache_root()
    if not root:
        return None
    d = check_layer(root, layer, ckpt_dir, verify)
    if d is None:
        return None
    rows, cols = tuple(md.shape)
    chips = [chip0 + i for i in range(rows * cols)]
    h01, h2 = load_host_chips(d, chips)
    out = []
    for hs, Tn in ((h01, T01), (h2, T2)):
        m = ttnn.from_host_shards(hs, ttnn.MeshShape(rows, cols))
        out.append(ttnn.to_device(m, md, memory_config=bank_mem(md, Tn)))
    return out[0], out[1]
