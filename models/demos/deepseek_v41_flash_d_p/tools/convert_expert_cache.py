#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""The migration of an old routed-expert weight cache to the only layout the loader reads (owner 2026-11-06).

Caches written before 2026-11-06 hold each laid-out weight tensor as <stem>.tensorbin (a ttnn flatbuffer of the
WIDTH_SHARDED tensor). The loader (flat_routed_expert flat_expert._load_cached_set) reads only the bank-major file
<stem>.bm.bin (bank_major.py), so on a box whose cache is .tensorbin-only every layer misses and is repacked from the
checkpoint (the host dequant + packing, ~1 h for a whole model). Converting is the fast way (host only, no device;
about 11 s a layer, 8 min for DS41's 40 layers): this writes <stem>.bm.bin beside each <stem>.tensorbin (the same
bytes per chip and DRAM bank; --verify reads them back and compares every shard after the inverse permutation). The .tensorbin files are left as they are: delete them yourself
once converted (nothing reads them; the weight cache's shared master never takes them, core/weight_cache.py).

    cd /scratch/dnijemcevic/tt-metal-rebase && source python_env/bin/activate
    PYTHONPATH=$PWD ionice -c3 nice -n19 python models/demos/deepseek_v41_flash_d_p/tools/convert_expert_cache.py \\
        --layers $(seq 0 39) [--impl bfp4_mse] [--cache <dir>] [--verify]
    ... convert_expert_cache.py <file>.tensorbin ...      # any FlatRoutedExpert cache file (e.g. another model's)

--cache: the mesh dir of the experts cache (<art>/tt_cache/<mesh>/<version>/experts-<rev>/<rows>x<cols>). --out: write
the .bm.bin files there (default: beside the .tensorbin). A complete .bm.bin already there is kept (--force rewrites).
"""

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


def convert(src: str, out_dir: str | None, verify: bool, force: bool = False) -> str:
    import ttnn

    dst = bm.bm_path(src)
    if out_dir:
        dst = str(Path(out_dir) / Path(dst).name)
    if not force and bm.is_complete(dst):
        print(f"{os.path.basename(dst)}: already converted (--force rewrites it)")
        return dst
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


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    ap.add_argument("--layers", type=int, nargs="*", default=[])
    ap.add_argument("--impl", default="bfp4_mse")
    ap.add_argument("--cache", default=CACHE)
    ap.add_argument("--out", default=None)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--force", action="store_true", help="rewrite a .bm.bin that is already complete")
    a = ap.parse_args(argv)
    torch.set_num_threads(8)
    files = list(a.files)
    for L in a.layers:
        got = sorted(glob.glob(f"{a.cache}/L{L:02d}.{a.impl}.*.flat.*.tensorbin"))
        assert got, f"no cache files for layer {L} impl {a.impl} in {a.cache}"
        files += got
    if a.out:
        os.makedirs(a.out, exist_ok=True)
    for f in files:
        convert(f, a.out, a.verify, a.force)


if __name__ == "__main__":
    main()
