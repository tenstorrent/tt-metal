#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Download CosyVoice-300M checkpoints from HuggingFace.

Stdlib only -- no huggingface_hub, no modelscope -- so it can run before (or
without) the CosyVoice venv existing. Resumable, idempotent, and it verifies
every file against the size the HF API reports rather than trusting a partial
transfer that happened to end at a plausible byte count.

Usage:
    python3 download_model.py [--dest DIR] [--repo NAME ...] [--skip-onnx-trt]

Default dest is <CosyVoice checkout>/pretrained_models, which is where
cosyvoice.cli.cosyvoice.CosyVoice(model_dir=...) expects to find them.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

HF_API = "https://huggingface.co/api/models/{repo}?blobs=true"
HF_FILE = "https://huggingface.co/{repo}/resolve/main/{path}"

DEFAULT_REPOS = [
    "FunAudioLLM/CosyVoice-300M",
    "FunAudioLLM/CosyVoice-300M-SFT",
    "FunAudioLLM/CosyVoice-300M-Instruct",
]

# Only needed for the TensorRT export path (load_trt=True). 329 MB per repo that
# a CPU or TTNN bring-up will never read.
TRT_ONLY = {"flow.decoder.estimator.fp32.onnx"}

# Never needed by any inference path.
JUNK = {".gitattributes", ".DS_Store", "README.md", "asset/dingding.png"}


def list_repo(repo: str) -> list[tuple[str, int]]:
    with urllib.request.urlopen(HF_API.format(repo=repo), timeout=60) as r:
        meta = json.load(r)
    out = []
    for sib in meta.get("siblings", []):
        name = sib["rfilename"]
        if name in JUNK:
            continue
        out.append((name, sib.get("size") or 0))
    return out


def fetch(url: str, dest: str, expect: int) -> str:
    """Download url -> dest, resuming a partial file. Returns a status word."""
    have = os.path.getsize(dest) if os.path.exists(dest) else 0
    if expect and have == expect:
        return "ok"
    if expect and have > expect:  # truncate a corrupt over-long file
        have = 0

    mode = "ab" if have else "wb"
    req = urllib.request.Request(url)
    if have:
        req.add_header("Range", f"bytes={have}-")
    try:
        resp = urllib.request.urlopen(req, timeout=120)
    except urllib.error.HTTPError as e:
        if e.code == 416:  # already complete per the server
            return "ok"
        raise
    # A server that ignores Range answers 200 with the whole body: restart.
    if have and resp.status != 206:
        have, mode = 0, "wb"

    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    done = have
    with resp, open(dest, mode) as fh:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
            done += len(chunk)
            if expect:
                pct = 100.0 * done / expect
                print(
                    f"\r    {os.path.basename(dest):<34} " f"{done/1e6:8.1f}/{expect/1e6:.1f} MB {pct:5.1f}%",
                    end="",
                    flush=True,
                )
    if expect:
        print()
    got = os.path.getsize(dest)
    if expect and got != expect:
        raise RuntimeError(f"{dest}: got {got} bytes, expected {expect}")
    return "resumed" if have else "new"


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    default_dest = os.environ.get("COSYVOICE_REPO", "/mnt/CosyVoice") + "/pretrained_models"

    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default=default_dest)
    ap.add_argument("--repo", action="append", default=None)
    ap.add_argument(
        "--skip-onnx-trt", action="store_true", help="skip flow.decoder.estimator.fp32.onnx (TRT-only, 329 MB each)"
    )
    args = ap.parse_args()
    del here

    repos = args.repo or DEFAULT_REPOS
    grand = 0
    for repo in repos:
        short = repo.split("/")[-1]
        dest_dir = os.path.join(args.dest, short)
        print(f"\n== {repo} -> {dest_dir}")
        files = list_repo(repo)
        if args.skip_onnx_trt:
            files = [(n, s) for n, s in files if n not in TRT_ONLY]
        for name, size in files:
            dest = os.path.join(dest_dir, name)
            status = fetch(HF_FILE.format(repo=repo, path=name), dest, size)
            grand += size
            if status == "ok":
                print(f"    {name:<34} {size/1e6:8.1f} MB  (already complete)")
    print(f"\ntotal {grand/1e9:.2f} GB across {len(repos)} checkpoint(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
