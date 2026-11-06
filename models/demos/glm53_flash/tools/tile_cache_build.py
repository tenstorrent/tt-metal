# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Build the reference-format TILE CACHE of GLM-5.3-Flash (one `.tensorbin` per tensor in its cache dtype / TILE
layout + manifest.json, the format tt/weights.TileCacheSource and the stack's byte-level expert loaders
(fused_decoder.load_fused_experts / load_xsl_experts) read directly: no host packing at load time) from the e2e test's
v2 cache (per-layer torch files of the dequantised tensors, tests/test_mk_e2e.TileCache).

    python3 tile_cache_build.py --src /data/glm53_tile_cache --dst /data/glm53_tile_cache_v3 [--workers 8]

Resumable (a tensor whose file exists with the right size is skipped). Host-only, BUT on this tree every ttnn tensor op
opens the cluster (UMD) and blocks on the chips' CHIP_IN_USE locks while a mesh job runs: run it as a SERIAL queue row
(nothing else on the chips). The manifest entry per tensor: path, shape, dtype (ttnn attribute name), layout, sha256
(of the key: a stable file tag for the permuted gate_up copies)."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import torch

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tt.weights import cache_dtype


def _name(dt) -> str:
    return str(dt).split(".")[-1].lower()  # DataType.BFLOAT4_B -> bfloat4_b (a ttnn attribute)


def _dump(path: str, t: ttnn.Tensor) -> int:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # ttnn insists on the .tensorbin extension (no temp name: a partial file fails have()'s size check); LOCAL mode: the
    # default DISTRIBUTED_GATHER initialises an MPI context (segfault from a worker thread, and no multi-host here)
    ttnn.dump_tensor(path, t, mode=ttnn.DumpTensorMode.LOCAL)
    return os.path.getsize(path)


def _entry(rel: str, shape, t: ttnn.Tensor, key: str, size: int) -> dict:
    return {"path": rel, "shape": [int(v) for v in shape], "dtype": _name(t.dtype),
            "layout": str(t.layout).split(".")[-1], "sha256": hashlib.sha256(key.encode()).hexdigest(),
            "bytes": size}  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--layers", default="")  # e.g. "0-4" for a smoke
    a = ap.parse_args()
    meta = json.load(open(os.path.join(a.src, "manifest.json")))
    os.makedirs(a.dst, exist_ok=True)
    mpath = os.path.join(a.dst, "manifest.json")
    out = json.load(open(mpath)) if os.path.isfile(mpath) else {"tensors": {}, "tt_metal_sha": "wt_mk-v3"}
    tensors = out["tensors"]

    def save() -> None:
        with open(mpath + ".tmp", "w") as f:
            json.dump(out, f)
        os.replace(mpath + ".tmp", mpath)

    def have(key: str) -> bool:
        e = tensors.get(key)
        return e is not None and os.path.isfile(os.path.join(a.dst, e["path"])) and os.path.getsize(
            os.path.join(a.dst, e["path"])) == e.get("bytes", -1)  # fmt: skip

    layers = sorted(int(k) for k in meta["layers"])
    if a.layers:
        lo, hi = (int(v) for v in a.layers.split("-"))
        layers = [L for L in layers if lo <= L <= hi]
    t_all = time.perf_counter()
    total = 0
    for L in layers:
        names = [n for n in meta["layers"][str(L)] if not have(f"layers.{L}.{n}")]
        if not names:
            continue
        t0 = time.perf_counter()
        d = torch.load(os.path.join(a.src, f"L{L:02d}.pt"), mmap=True, weights_only=True)

        def pack(n: str) -> tuple:  # a worker: the host tiling / bfp packing only
            w = d[n]
            dt = cache_dtype(n)
            h = ttnn.from_torch(w.to(torch.float32 if dt == ttnn.float32 else torch.bfloat16), dtype=dt,
                                layout=ttnn.TILE_LAYOUT)  # fmt: skip
            return n, w.shape, h

        with ThreadPoolExecutor(max_workers=a.workers) as pool:
            for n, shape, h in pool.map(pack, names):  # the main thread writes the files
                key = f"layers.{L}.{n}"
                rel = os.path.join(f"L{L:02d}", f"{n}.tensorbin")
                size = _dump(os.path.join(a.dst, rel), h)
                tensors[key] = _entry(rel, shape, h, key, size)
                total += size
        save()
        print(f"layer {L}: {len(names)} tensors in {time.perf_counter() - t0:.0f} s ({total / 2**30:.1f} GiB so far)",
              flush=True)  # fmt: skip
        del d
    # the head (mk_head reads source.tensors[name] + source._load(e)): the v2 head.pt holds the torch tensors with the
    # ttnn dtype / layout each was built with by the HF source
    hp = os.path.join(a.src, "head.pt")
    if os.path.isfile(hp) and not a.layers:
        hd = torch.load(hp, mmap=True, weights_only=True)
        for key, m in meta["head"].items():
            if have(key):
                continue
            w = hd[key]
            h = ttnn.from_torch(w, dtype=getattr(ttnn, m["dtype"]), layout=getattr(ttnn, m["layout"]))
            rel = os.path.join("head", f"{key}.tensorbin")
            size = _dump(os.path.join(a.dst, rel), h)
            tensors[key] = _entry(rel, m["shape"], h, key, size)
            print(f"head {key}: {size / 2**20:.0f} MiB", flush=True)
        save()
        if all(have(k) for k in meta["head"]) and all(
                have(f"layers.{L}.{n}") for L in meta["layers"] for n in meta["layers"][L]):  # fmt: skip
            with open(os.path.join(a.dst, "COMPLETE"), "w") as f:
                f.write(time.strftime("%Y-%m-%d %H:%M:%S\n"))
            print("COMPLETE", flush=True)
    print(
        f"done: {len(tensors)} entries, {total / 2**30:.1f} GiB written, {time.perf_counter() - t_all:.0f} s",
        flush=True,
    )


if __name__ == "__main__":
    main()
    # the cluster the host tensor ops opened is torn down at interpreter exit by MetalContext's destructor, which
    # aborted (UMD close_device) after an exception once; the files and the manifest are flushed, so leave directly
    import sys

    sys.stdout.flush()
    os._exit(0)
