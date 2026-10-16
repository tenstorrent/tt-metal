#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Copy a golden trace keeping the decoder-layer hidden states of only some layers.

    python trim_trace.py <src>/<hf-repo>/isl<N> <dst-root> --layers 0-5,14,15,30-32

The first ALWAYS_KEEP (5) decoder layers are kept whatever --layers says. Everything else is kept: embed, final,
every layer's cache file. Files are hard-linked when src and dst share a
filesystem (no extra space), else copied. metadata.json is rewritten: the layer-output inventory lists only the kept
layers, and "hidden_layers" records the kept set and why. The token cache next to the trace is linked too, and a
fresh <dst>/isl<N>.SHA256SUMS is written.
"""

import argparse
import hashlib
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


ALWAYS_KEEP = 5  # decoder layers 0-4: in every trace's hidden states


def parse_layers(spec: str) -> list[int]:
    out = set()
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out.update(range(int(a), int(b or a) + 1))
    return sorted(out)


def place(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(64 << 20):
            h.update(chunk)
    return h.hexdigest()


ap = argparse.ArgumentParser()
ap.add_argument("trace", type=Path, help="<root>/<hf-repo>/isl<N>")
ap.add_argument("dst_root", type=Path)
ap.add_argument("--layers", required=True, help=f"hidden-state layers to keep, e.g. 14,15 (0-{ALWAYS_KEEP - 1} always)")
ap.add_argument("--reason", default="", help="recorded in metadata.json")
args = ap.parse_args()

src = args.trace.resolve()
meta = json.loads((src / "metadata.json").read_text())
repo, n_layers = meta["hf_repo"], meta["num_layers"]
keep = sorted(set(parse_layers(args.layers)) | set(range(min(ALWAYS_KEEP, n_layers))))
assert max(keep) < n_layers, f"layers must be within 0..{n_layers - 1}"
dst = args.dst_root / repo / src.name
if (dst / "metadata.json").exists():
    raise SystemExit(f"{dst} already holds a trace")

files = [p for p in src.rglob("*") if p.is_file() and p.name not in ("metadata.json",)]
kept = [
    p
    for p in files
    if not (p.parent.name == "hidden" and p.name.startswith("layer_") and int(p.stem.split("_")[1]) not in keep)
]
for p in kept:
    place(p, dst / p.relative_to(src))
token_cache = (src / meta["token_cache"]).resolve()
place(token_cache, dst.parent / token_cache.name)

for name, entry in meta["tensors"].items():
    if entry.get("file") == "hidden/layer_{i:02d}.safetensors" and "layers" in entry:
        entry["layers"] = [i for i in entry["layers"] if i in keep]
meta["hidden_layers"] = {
    "kept": keep,
    "of": n_layers,
    "note": "decoder-layer hidden states stored only for these layers; embed, final and every layer's cache "
    "are complete. Each kept layer i's output is layer i+1's exact input.",
    "reason": args.reason,
    "trimmed_from": str(src),
}
(dst / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")

out = sorted(p for p in dst.rglob("*") if p.is_file())
with ThreadPoolExecutor(8) as ex:
    sums = list(ex.map(sha256, out))
lines = [f"{s}  {p.relative_to(dst.parent)}\n" for s, p in zip(sums, out)]
(dst.parent / f"{dst.name}.SHA256SUMS").write_text("".join(lines))
size = sum(p.stat().st_size for p in out)
print(f"{dst}: {len(out)} files, {size / 1e9:.1f} GB, hidden layers {keep}")
