# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The decode weights of all 78 layers as the fused layers read them, on disk, with a manifest that finds a broken
layer before the board does. Host only (CPU), restartable per layer.

    TT_METAL_MOCK_CLUSTER_DESC_PATH=... nice -n 10 python -m models.experimental.hy4_preview.fused.model.convert \
        --layers 6-77 [--verify]

Per layer: fused/moe/prepare.py prepare_layer(out, layer, "tp32") writes the 32 chips' two expert streams (1/32 of
every routed expert and of the shared expert, BFP4; layer 0: the dense FFN) under <out>/layer<L>/tp32/, then this
module writes <out>/layer<L>/tp32/files.json: {"layer", "split", "complete": true, "seconds", "stamp", "files":
{name: [size, sha256]}}. A layer is complete when its files.json says so, its files have the recorded sizes AND its
container stamp (fused/moe/prepare.py container_stamp, in manifest.json and in files.json) is the one this tree
writes: the format number, the split and its share, the entries, the shared expert's group and terms, the source.
A complete layer is skipped (--verify also re-hashes every file and reports the layers that differ). A layer that
was interrupted has no files.json and is converted again; so is one whose stamp differs (sizes alone do not tell
a container from another: another shared group has the same sizes).
--stamp: files converted before the container had a stamp get it without a new conversion, after their recorded
manifest (split, chips, entries, bytes per chip, shared group) is checked against the stamp; without --stamp such
a layer is "unstamped" and is converted again.
--split tp8g4 writes the "8 chips per group" layout of the MoE layers (fused/moe/prepare.py, layer_mlp/PLAN.md
section 15) under <out>/layer<L>/tp8g4/ in the same way; layer 0 (the dense FFN) has no such form and is skipped.

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


def layer_dir(out, layer, split=SPLIT) -> Path:
    return Path(out) / f"layer{layer}" / split


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


def expected_stamp(layer, split=SPLIT) -> dict:
    """The container stamp this tree writes and reads for a layer (layer 0 is the dense FFN)."""
    from models.experimental.hy4_preview.fused.moe import prepare

    return prepare.container_stamp(split, dense=layer == 0)


def stamp_from_manifest(out, layer, split=SPLIT) -> dict:
    """The stamp of a layer converted before the container had one, from what its conversion recorded
    (manifest.json): the split, the chips, the entries a chip holds, the bytes a chip holds (which fix P for that
    many entries) and, for a grouped split, the shared expert's group. Raises when the record says another
    container than this tree's; what a record cannot say (the order inside a bank, the rounding) is the code's that
    wrote the files, unchanged since (re-converting a layer and comparing the checksums shows it)."""
    from models.experimental.hy4_preview.fused.moe import prepare

    want = expected_stamp(layer, split)
    info = json.loads(prepare.manifest_path(out, layer, split).read_text())
    share = prepare.split_share(split)
    payload = want["entries"] * share.expert_bytes
    recorded = {
        "split": (info["split"], want["split"]),
        "chips": (info["chips"], want["chips"]),
        "entries": (info["entries"], want["entries"]),
        "bytes_per_chip": (info["bytes_per_chip"] - info["bytes_per_chip"] % 4096, payload),  # (two file headers)
        "shared_group": (info.get("shared_group"), want["shared_group"]),
    }
    bad = {k: v for k, v in recorded.items() if v[0] != v[1]}
    assert not bad, f"layer {layer} [{split}]: its manifest records another container, (recorded, wanted): {bad}"
    return want


def state(out, layer, rehash=False, threads=8, split=SPLIT):
    """("complete" | "missing" | "unstamped" | "broken: ...", manifest or None) of a layer on disk."""
    d = layer_dir(out, layer, split)
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
    from models.experimental.hy4_preview.fused.moe import prepare

    want, on_disk = expected_stamp(layer, split), prepare.read_stamp(out, layer, split)
    if on_disk is None and "stamp" not in manifest:
        return "unstamped", manifest
    for have in (on_disk, manifest.get("stamp")):
        diff = prepare.stamp_difference(have, want)
        if diff:
            return f"broken: stamp {diff}", manifest
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
    ap.add_argument("--split", default=SPLIT, help="tp32 (default) or tp8g4 (MoE layers only)")
    ap.add_argument(
        "--verify", action="store_true", help="no conversion: check every layer's files against its manifest"
    )
    ap.add_argument(
        "--adopt", action="store_true", help="write files.json for layers that were converted before this module"
    )
    ap.add_argument(
        "--stamp", action="store_true", help="write the container stamp of complete layers converted before it existed"
    )
    args = ap.parse_args()
    layers = parse_layers(args.layers)
    split = args.split
    tag = "" if split == SPLIT else f" [{split}]"
    if split != SPLIT:  # (layer 0's dense FFN stays tp32)
        layers = [layer for layer in layers if layer != 0]
    if args.verify:
        report = {layer: state(args.out, layer, rehash=True, threads=args.threads, split=split)[0] for layer in layers}
        bad = {k: v for k, v in report.items() if v != "complete"}
        total = sum(
            sum(s for s, _ in state(args.out, k, split=split)[1]["files"].values())
            for k, v in report.items()
            if v == "complete"
        )
        print(
            f"HY4_MK11 convert verify{tag}: {len(report) - len(bad)} of {len(report)} layers complete, {total} B; not complete: {bad}",
            flush=True,
        )
        return
    assert os.environ.get("TT_METAL_MOCK_CLUSTER_DESC_PATH"), "host only: set TT_METAL_MOCK_CLUSTER_DESC_PATH"
    import torch

    torch.set_num_threads(args.threads)
    from models.experimental.hy4_preview.fused.moe import prepare

    for layer in layers:
        d = layer_dir(args.out, layer, split)
        found, known = state(args.out, layer, split=split)
        if found == "complete":
            print(f"HY4_MK11 convert{tag} layer {layer}: complete, skipped", flush=True)
            continue
        if found == "unstamped" and args.stamp:
            stamp = stamp_from_manifest(args.out, layer, split)
            for path, record in ((prepare.manifest_path(args.out, layer, split), None), (d / "files.json", known)):
                record = json.loads(path.read_text()) if record is None else record
                record["stamp"] = stamp
                tmp = path.with_suffix(".json.tmp")
                tmp.write_text(json.dumps(record, indent=1))
                tmp.replace(path)
            assert state(args.out, layer, split=split)[0] == "complete"
            print(
                f"HY4_MK11 convert{tag} layer {layer}: stamped (format {stamp['format']}, its manifest agrees), not converted",
                flush=True,
            )
            continue
        t0 = time.time()
        old = d / "manifest.json"
        have_all = old.is_file() and all((d / n).is_file() for n in expected_files())
        if args.adopt and have_all:
            info = json.loads(old.read_text())
            if "stamp" not in info:  # (converted before the container had a stamp: from what the conversion recorded)
                info["stamp"] = stamp_from_manifest(args.out, layer, split)
                old.write_text(json.dumps(info, indent=1))
        else:
            (d / "files.json").unlink(missing_ok=True)
            info = prepare.prepare_layer(args.out, layer, split, dense=layer == 0)
        files = hash_files(d, args.threads)
        assert sum(s for s, _ in files.values()) == info["bytes"], (layer, info["bytes"])
        manifest = {
            "layer": layer,
            "split": split,
            "complete": True,
            "entries": info["entries"],
            "bytes": info["bytes"],
            "seconds": round(time.time() - t0, 1),
            "adopted": bool(args.adopt and have_all),
            "files": files,
            "stamp": prepare.read_stamp(args.out, layer, split),
        }
        assert (
            manifest["stamp"] is not None
        ), f"layer {layer} [{split}]: no container stamp (files from before it: --stamp)"
        tmp = d / "files.json.tmp"
        tmp.write_text(json.dumps(manifest, indent=1))
        tmp.replace(d / "files.json")
        print(
            f"HY4_MK11 convert{tag} layer {layer}: {info['bytes']} B in {len(files)} files, {manifest['seconds']} s{' (adopted)' if manifest['adopted'] else ''}",
            flush=True,
        )


if __name__ == "__main__":
    main()
