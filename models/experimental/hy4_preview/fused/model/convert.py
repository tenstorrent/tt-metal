# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The decode weights of all 78 layers as the fused layers read them, on disk, with a manifest that finds a broken
layer before the board does. Host only (CPU), restartable per layer.

    TT_METAL_MOCK_CLUSTER_DESC_PATH=... nice -n 10 python -m models.experimental.hy4_preview.fused.model.convert \
        --layers 6-77 [--verify]

Per layer: fused/moe/prepare.py prepare_layer(out, layer, "tp32") writes the 32 chips' two expert streams (1/32 of
every routed expert and of the shared expert, BFP4; layer 0: the dense FFN) under <out>/layer<L>/tp32/, then this
module writes <out>/layer<L>/tp32/files.json: {"layer", "split", "complete": true, "seconds", "files": {name: [size,
sha256]}}. A layer whose files.json says complete and whose files have the recorded sizes is skipped (--verify
also re-hashes every file and reports the layers that differ). A layer that was interrupted has no files.json and
is converted again.

The attention projections are not converted here: AttnLayer.set_weights builds a layer's three streams from the
checkpoint when the layer is loaded (about 2 s a layer), in its own unit layout, which is not the layout
fused/projections/prepare.py writes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SPLIT = "tp32"
CHIPS = 32


def layer_dir(out, layer) -> Path:
    return Path(out) / f"layer{layer}" / SPLIT


def expected_files():
    return [f"chip{chip:02d}.{kind}.tensorbin" for chip in range(CHIPS) for kind in ("gate_up", "down")]


def sha256(path, chunk=1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                return h.hexdigest()
            h.update(block)


def hash_files(directory: Path, threads: int) -> dict:
    names = expected_files()
    with ThreadPoolExecutor(max_workers=threads) as pool:
        digests = list(pool.map(lambda n: sha256(directory / n), names))
    return {n: [(directory / n).stat().st_size, d] for n, d in zip(names, digests)}


def state(out, layer, rehash=False, threads=8):
    """("complete" | "missing" | "broken: ...", manifest or None) of a layer on disk."""
    d = layer_dir(out, layer)
    path = d / "files.json"
    if not path.is_file():
        return "missing", None
    manifest = json.loads(path.read_text())
    if not manifest.get("complete") or sorted(manifest["files"]) != sorted(expected_files()):
        return "broken: manifest", manifest
    for name, (size, _) in manifest["files"].items():
        p = d / name
        if not p.is_file() or p.stat().st_size != size:
            return f"broken: {name}", manifest
    if rehash:
        now = hash_files(d, threads)
        bad = [n for n in now if now[n][1] != manifest["files"][n][1]]
        if bad:
            return f"broken: checksum of {bad[:3]}", manifest
    return "complete", manifest


def parse_layers(text):
    out = []
    for part in text.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="0-77")
    ap.add_argument("--out", default="generated/hy4_preview/moe_streams")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument(
        "--verify", action="store_true", help="no conversion: check every layer's files against its manifest"
    )
    ap.add_argument(
        "--adopt", action="store_true", help="write files.json for layers that were converted before this module"
    )
    args = ap.parse_args()
    layers = parse_layers(args.layers)
    if args.verify:
        report = {layer: state(args.out, layer, rehash=True, threads=args.threads)[0] for layer in layers}
        bad = {k: v for k, v in report.items() if v != "complete"}
        total = sum(
            sum(s for s, _ in state(args.out, k)[1]["files"].values()) for k, v in report.items() if v == "complete"
        )
        print(
            f"HY4_MK11 convert verify: {len(report) - len(bad)} of {len(report)} layers complete, {total} B; not complete: {bad}",
            flush=True,
        )
        return
    assert os.environ.get("TT_METAL_MOCK_CLUSTER_DESC_PATH"), "host only: set TT_METAL_MOCK_CLUSTER_DESC_PATH"
    import torch

    torch.set_num_threads(args.threads)
    from models.experimental.hy4_preview.fused.moe import prepare

    for layer in layers:
        d = layer_dir(args.out, layer)
        found, _ = state(args.out, layer)
        if found == "complete":
            print(f"HY4_MK11 convert layer {layer}: complete, skipped", flush=True)
            continue
        t0 = time.time()
        old = d / "manifest.json"
        have_all = old.is_file() and all((d / n).is_file() for n in expected_files())
        if args.adopt and have_all:
            info = json.loads(old.read_text())
        else:
            (d / "files.json").unlink(missing_ok=True)
            info = prepare.prepare_layer(args.out, layer, SPLIT, dense=layer == 0)
        files = hash_files(d, args.threads)
        assert sum(s for s, _ in files.values()) == info["bytes"], (layer, info["bytes"])
        manifest = {
            "layer": layer,
            "split": SPLIT,
            "complete": True,
            "entries": info["entries"],
            "bytes": info["bytes"],
            "seconds": round(time.time() - t0, 1),
            "adopted": bool(args.adopt and have_all),
            "files": files,
        }
        tmp = d / "files.json.tmp"
        tmp.write_text(json.dumps(manifest, indent=1))
        tmp.replace(d / "files.json")
        print(
            f"HY4_MK11 convert layer {layer}: {info['bytes']} B in {len(files)} files, {manifest['seconds']} s{' (adopted)' if manifest['adopted'] else ''}",
            flush=True,
        )


if __name__ == "__main__":
    main()
