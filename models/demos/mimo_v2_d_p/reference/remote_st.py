# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fetch individual tensors from the remote MiMo-V2.6 safetensors shards with HTTP range reads.

The checkpoint is ~300 GB spread over EP-sharded files; a single-layer bringup only needs a few GB, so we read
each shard's header once (cached) and range-fetch only the tensors asked for.
"""

import json
import os
import struct
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

REPO = "XiaomiMiMo/MiMo-V2.6-Flash-RL"
BASE = f"https://huggingface.co/{REPO}/resolve/main/"
LOCAL = Path(os.environ.get("MIMO_V2_CKPT", "/localdev/mstaletovic/hf_models/MiMo-V2.6-Flash-RL"))

_DT = {
    "BF16": (torch.bfloat16, 2),
    "F32": (torch.float32, 4),
    "F8_E4M3": (torch.float8_e4m3fn, 1),
    "U8": (torch.uint8, 1),
    "I64": (torch.int64, 8),
}


def _get(url, start, end, retries=5):
    last = None
    for _ in range(retries):
        try:
            r = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end - 1}"})
            data = urllib.request.urlopen(r, timeout=300).read()
            assert len(data) == end - start, (len(data), end - start)
            return data
        except Exception as e:  # noqa: BLE001 - network retry
            last = e
    raise last


def weight_map():
    p = LOCAL / "model.safetensors.index.json"
    if not p.exists():
        p.write_bytes(urllib.request.urlopen(BASE + "model.safetensors.index.json").read())
    return json.loads(p.read_text())["weight_map"]


def shard_header(fn):
    cache = LOCAL / "headers" / (fn + ".json")
    if cache.exists():
        return json.loads(cache.read_text())
    n = struct.unpack("<Q", _get(BASE + fn, 0, 8))[0]
    h = json.loads(_get(BASE + fn, 8, 8 + n))
    h = {"__n__": n, **h}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(h))
    return h


def _resolve(fn):
    """Follow the HF redirect once (to the CDN URL) so range reads don't pay it per request."""
    r = urllib.request.Request(BASE + fn, headers={"Range": "bytes=0-0"})
    with urllib.request.urlopen(r, timeout=120) as resp:
        return resp.geturl()


def fetch(names, workers=16, piece=64 << 20, gap=4 << 20):
    """{name: torch tensor} for the given checkpoint tensor names (raw stored dtype).

    Tensors of one shard are coalesced into large byte ranges (gaps < ``gap`` merged), split into
    ``piece``-sized parallel reads against the resolved CDN URL.
    """
    wm = weight_map()
    files = sorted({wm[n] for n in names})
    with ThreadPoolExecutor(workers) as ex:
        headers = dict(zip(files, ex.map(shard_header, files)))
        urls = dict(zip(files, ex.map(_resolve, files)))
    jobs = []  # (file, start, end)
    spans = {}  # file -> list of (start, end, [names])
    for fn in files:
        off = 8 + headers[fn]["__n__"]
        items = sorted((off + headers[fn][n]["data_offsets"][0], off + headers[fn][n]["data_offsets"][1], n) for n in names if wm[n] == fn)
        merged = []
        for s0, e0, n in items:
            if merged and s0 - merged[-1][1] <= gap:
                merged[-1][1] = max(merged[-1][1], e0)
                merged[-1][2].append(n)
            else:
                merged.append([s0, e0, [n]])
        spans[fn] = merged
        for s0, e0, _ in merged:
            for a in range(s0, e0, piece):
                jobs.append((fn, a, min(a + piece, e0)))

    def get(job):
        fn, a, b = job
        return job, _get(urls[fn], a, b)

    with ThreadPoolExecutor(workers) as ex:
        parts = dict(ex.map(get, jobs))
    out = {}
    for fn, merged in spans.items():
        off = 8 + headers[fn]["__n__"]
        for s0, e0, ns in merged:
            buf = b"".join(parts[(fn, a, min(a + piece, e0))] for a in range(s0, e0, piece))
            for n in ns:
                meta = headers[fn][n]
                a, b = off + meta["data_offsets"][0] - s0, off + meta["data_offsets"][1] - s0
                dt, _ = _DT[meta["dtype"]]
                out[n] = torch.frombuffer(bytearray(buf[a:b]), dtype=torch.uint8).view(dt).reshape(meta["shape"]).clone()
    return out
