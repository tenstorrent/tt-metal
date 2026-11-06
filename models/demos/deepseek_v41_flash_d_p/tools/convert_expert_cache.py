#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Host only (no device): write the bank-major twin (<stem>.bm.bin, bank_major.py) of laid-out expert weight cache
files (<stem>.tensorbin, WIDTH_SHARDED). The .tensorbin files are left as they are.

    cd /scratch/dnijemcevic/tt-metal-rebase && source python_env/bin/activate
    PYTHONPATH=$PWD ionice -c3 nice -n19 python models/demos/deepseek_v41_flash_d_p/tools/convert_expert_cache.py \\
        --layers $(seq 0 39) [--impl bfp4_mse] [--verify]

--out: write the .bm.bin files there (default: beside the .tensorbin). --verify: read the written file back and
compare every shard with the source after the inverse permutation."""

from __future__ import annotations

import argparse
import glob
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from ttnn.bringup.flat_routed_expert_ttnn import bank_major as bm  # noqa: E402

CACHE = "/scratch/dnijemcevic/bringup/deepseek_v41_flash_d_p/tt_cache/8x4/v1/experts-5254448103b6/8x4"


def warm(path, threads=8):
    """Parallel preads of the whole file (page cache), so the mmap below faults warm pages."""
    size = os.path.getsize(path)
    fd = os.open(path, os.O_RDONLY)
    buf = [bytearray(bm.CHUNK) for _ in range(threads)]

    def rd(i):
        mv = memoryview(buf[i])
        for off in range(i * bm.CHUNK, size, threads * bm.CHUNK):
            os.preadv(fd, [mv], off)

    try:
        with ThreadPoolExecutor(threads) as ex:
            list(ex.map(rd, range(threads)))
    finally:
        os.close(fd)


def convert(src: str, out_dir: str | None, verify: bool) -> str:
    import ttnn

    dst = bm.bm_path(src)
    if out_dir:
        dst = str(Path(out_dir) / Path(dst).name)
    t0 = time.time()
    warm(src)
    t1 = time.time()
    t = ttnn._ttnn.tensor.load_tensor_flatbuffer(src)  # (host, mmapped: no copy)
    info = bm.width_info(t)
    mesh = tuple(int(x) for x in t.host_buffer().shape())
    srcs = bm.shard_views(t, mesh)
    n = int(srcs[0].numel())
    tiles = (info["rows"] // 32) * info["banks"]
    assert n % tiles == 0, (n, tiles)
    info["tile_bytes"] = n // tiles
    info["src"] = os.path.basename(src)
    outs = [torch.empty(n, dtype=torch.uint8) for _ in srcs]
    for s, o in zip(srcs, outs):
        bm.width_to_bank_major(s, info["banks"], info["tile_bytes"], o)
    t2 = time.time()
    bm.write_bm(dst, info, mesh, outs)
    t3 = time.time()
    gb = n * len(srcs) / 1e9
    print(
        f"{os.path.basename(dst)}: {gb:.2f} GB  read {t1 - t0:.2f} s ({gb / (t1 - t0):.2f} GB/s)  "
        f"permute {t2 - t1:.2f} s  write {t3 - t2:.2f} s ({gb / (t3 - t2):.2f} GB/s)  tile {info['tile_bytes']} B "
        f"banks {info['banks']} rows {info['rows']}"
    )
    if verify:
        hdr = bm.read_header(dst)
        back = [torch.empty(n, dtype=torch.uint8) for _ in srcs]
        with ThreadPoolExecutor(8) as pool:
            bm.read_bm_into(dst, back, pool, hdr)
        tmp = torch.empty(n, dtype=torch.uint8)
        for k, (s, b) in enumerate(zip(srcs, back)):
            bm.bank_to_width_major(b, hdr["banks"], hdr["tile_bytes"], tmp)
            assert torch.equal(tmp, s), f"{dst}: shard {k} differs from the source after the inverse permutation"
        print(f"  verified: every shard bit-identical to {os.path.basename(src)} (inverse permutation)")
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    ap.add_argument("--layers", type=int, nargs="*", default=[])
    ap.add_argument("--impl", default="bfp4_mse")
    ap.add_argument("--cache", default=CACHE)
    ap.add_argument("--out", default=None)
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(8)
    files = list(a.files)
    for L in a.layers:
        got = sorted(glob.glob(f"{a.cache}/L{L:02d}.{a.impl}.*.flat.*.tensorbin"))
        assert got, f"no cache files for layer {L} impl {a.impl} in {a.cache}"
        files += got
    if a.out:
        os.makedirs(a.out, exist_ok=True)
    for f in files:
        convert(f, a.out, a.verify)


if __name__ == "__main__":
    main()
