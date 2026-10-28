# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Engram host-lookup benchmark (CPU + NVMe only, no device): stage 0 of the engram host-time work.

Times what tt/engram.py TtEngramRows.host_rows does per chunk and engram layer - hash the token ids, gather the fp8
rows + e8m0 scales from the memory-mapped table, dequant, assemble the bf16 rows - warm (pages in RAM) and cold (the
table's pages dropped first, so the reads hit the disk), single-threaded as today and with the gather split over N
threads (sorted ids, contiguous slices: N concurrent page faults = queue depth N). --check compares the threaded rows
with reference/weights.py EngramTable.rows byte for byte.

    python -m models.demos.deepseek_v41_flash_d_p.tools.engram_bench --ckpt <hf dir> [--layers 1,14] [--tokens 5120]
        [--threads 1,8,16,32] [--repeats 3] [--cold] [--check] [--seed 0]

Cold runs: posix_fadvise(DONTNEED) cannot evict pages a process has mapped, so the parent drops the table's byte
ranges without mapping them and every timed run is a fresh subprocess. Random token ids (uniform over the vocabulary)
stand in for a new prompt: each run uses other ids (seed + run), so a warm run does not reuse the previous run's rows
unless --same-ids.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor


def _setup(ckpt: str):
    from models.demos.deepseek_v41_flash_d_p.reference.ds41_ref import Config
    from models.demos.deepseek_v41_flash_d_p.reference.weights import Checkpoint
    from models.demos.deepseek_v41_flash_d_p.tt.engram import engram_hasher

    cfg = Config.from_dir(ckpt)
    return cfg, Checkpoint(ckpt), engram_hasher(ckpt, cfg)


def gather(table, flat, threads: int, pool=None):
    """table.rows(flat) with the gather split over ``threads`` (same bytes: sorted ids, contiguous slices)."""
    import torch

    from models.demos.deepseek_v41_flash_d_p.reference.weights import BLOCK

    order = torch.argsort(flat)
    sids = flat[order]
    n = flat.numel()
    w = torch.empty(n, table.dim, dtype=torch.uint8)
    s = torch.empty(n, table.scale.shape[1], dtype=torch.uint8)
    if os.environ.get("ENGRAM_BENCH_IO") == "pread":  # explicit reads, queue depth = threads, no page-fault path
        wfd, woff, sfd, soff = table._pread
        wd, sd = table.dim, table.scale.shape[1]
        wn, sn = w.numpy(), s.numpy()
        inv = torch.empty_like(order)
        inv[order] = torch.arange(n)
        sl = sids.tolist()
        edges = [n * k // max(threads, 1) for k in range(max(threads, 1) + 1)]

        def part(k):
            for j in range(edges[k], edges[k + 1]):
                r = sl[j]
                wn[int(order[j])] = memoryview(os.pread(wfd, wd, woff + r * wd)).cast("B")
                sn[int(order[j])] = memoryview(os.pread(sfd, sd, soff + r * sd)).cast("B")

        if threads <= 1:
            part(0)
        else:
            list(pool.map(part, range(threads)))
    elif threads <= 1:
        w[order] = table.weight[sids]
        s[order] = table.scale[sids]
    else:
        edges = [n * k // threads for k in range(threads + 1)]

        def part(k):
            a, b = edges[k], edges[k + 1]
            w[order[a:b]] = table.weight[sids[a:b]]  # (torch indexing releases the GIL: the faults run in parallel)
            s[order[a:b]] = table.scale[sids[a:b]]

        list(pool.map(part, range(threads)))
    t_gather = time.perf_counter()
    v = w.view(torch.float8_e4m3fn).float().view(n, -1, BLOCK)
    v = v * s.view(torch.float8_e8m0fnu).float().unsqueeze(-1)
    return v.view(n, table.dim), t_gather


def run_once(ckpt: str, layer: int, tokens: int, threads: int, seed: int, check: bool, cfg=None, ck=None, hasher=None):
    """One chunk of one engram layer: per-stage seconds (and the check result)."""
    import torch

    from models.demos.deepseek_v41_flash_d_p.reference.ngram import DEAD
    from models.demos.deepseek_v41_flash_d_p.reference.weights import EngramTable

    if cfg is None:
        cfg, ck, hasher = _setup(ckpt)
    table = EngramTable(ck, layer)
    if os.environ.get("ENGRAM_BENCH_IO") == "pread":
        fds = []
        for name in (f"layers.{layer}.engram.embed.weight", f"layers.{layer}.engram.embed.scale"):
            _, _, off, _ = ck.info(name)
            fd = os.open(os.path.join(ck.path, ck.weight_map[name]), os.O_RDONLY)
            if os.environ.get("ENGRAM_BENCH_MADV") == "random":
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_RANDOM)  # (pread: no file read-ahead either)
            fds += [fd, off]
        table._pread = tuple(fds)
    if os.environ.get("ENGRAM_BENCH_MADV") == "random":  # no read-around: a fault reads its page, not read_ahead_kb
        import mmap

        for name in (f"layers.{layer}.engram.embed.weight", f"layers.{layer}.engram.embed.scale"):
            ck._map(ck.weight_map[name])._mmap.madvise(mmap.MADV_RANDOM)
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, cfg.vocab_size, (tokens,), generator=g)
    tail = torch.full((hasher.max_ngram_size - 1,), DEAD, dtype=torch.int64)
    pool = ThreadPoolExecutor(threads) if threads > 1 else None
    t0 = time.perf_counter()
    h, _ = hasher.hashes(ids, layer, tail)
    t1 = time.perf_counter()
    flat = h.reshape(-1)
    io0, flt0 = _io_read_bytes(), _majflt()
    v, t2 = gather(table, flat, threads, pool)
    io1, flt1 = _io_read_bytes(), _majflt()
    t3 = time.perf_counter()
    rows = torch.zeros(tokens, h.shape[1] * table.dim, dtype=torch.bfloat16)
    rows[:] = v.view(tokens, -1).to(torch.bfloat16)
    t4 = time.perf_counter()
    out = dict(
        layer=layer,
        threads=threads,
        tokens=tokens,
        lookups=flat.numel(),
        bytes_read=flat.numel() * (table.dim + table.scale.shape[1]),
        hash_ms=1e3 * (t1 - t0),
        gather_ms=1e3 * (t2 - t1),
        dequant_ms=1e3 * (t3 - t2),
        assemble_ms=1e3 * (t4 - t3),
        total_ms=1e3 * (t4 - t0),
        gather_disk_mb=round((io1 - io0) / 2**20, 1),  # bytes the gather really read from storage (cache misses)
        gather_major_faults=flt1 - flt0,
    )
    if check:
        ref = table.rows(h).flatten(1).to(torch.bfloat16)
        out["check_equal"] = bool(torch.equal(ref.view(torch.int16), rows.view(torch.int16)))
    if pool:
        pool.shutdown()
    return out


def _io_read_bytes() -> int:
    """Bytes this process caused to be read from storage (/proc/self/io read_bytes): disk traffic, not cache hits."""
    with open("/proc/self/io") as f:
        return int(next(line for line in f if line.startswith("read_bytes:")).split()[1])


def _majflt() -> int:
    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_majflt


def load_resident(ck, layer: int, readers: int = 16):
    """The layer's engram weight + scale copied once into anonymous memory (transparent huge pages): an EngramTable-
    like object whose weight / scale are resident torch tensors. Returns (table, seconds, GiB)."""
    import mmap

    import numpy as np
    import torch

    from models.demos.deepseek_v41_flash_d_p.reference.weights import EngramTable

    table = EngramTable(ck, layer)
    t0, total = time.perf_counter(), 0
    for attr, name in (
        ("weight", f"layers.{layer}.engram.embed.weight"),
        ("scale", f"layers.{layer}.engram.embed.scale"),
    ):
        _, shape, off, nbytes = ck.info(name)
        buf = mmap.mmap(-1, nbytes, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
        try:
            buf.madvise(mmap.MADV_HUGEPAGE)
        except (AttributeError, OSError):
            pass
        fd = os.open(os.path.join(ck.path, ck.weight_map[name]), os.O_RDONLY)
        os.posix_fadvise(fd, off, nbytes, os.POSIX_FADV_SEQUENTIAL)
        step = 64 << 20
        view = memoryview(buf)

        def read(k):
            a = k * step
            n = min(step, nbytes - a)
            got = os.preadv(fd, [view[a : a + n]], off + a)
            assert got == n, (got, n)

        with ThreadPoolExecutor(readers) as ex:
            list(ex.map(read, range((nbytes + step - 1) // step)))
        os.close(fd)
        arr = np.frombuffer(buf, dtype=np.uint8).reshape(shape)
        setattr(table, attr, torch.from_numpy(arr))
        setattr(table, f"_{attr}_buf", buf)
        total += nbytes
    return table, time.perf_counter() - t0, total / 2**30


def drop_pages(ck, layer: int) -> int:
    """posix_fadvise(DONTNEED) over the byte ranges of the layer's engram weight + scale (no mapping): bytes dropped."""
    total = 0
    for name in (f"layers.{layer}.engram.embed.weight", f"layers.{layer}.engram.embed.scale"):
        _, _, off, nbytes = ck.info(name)  # (absolute offset in the file)
        fd = os.open(os.path.join(ck.path, ck.weight_map[name]), os.O_RDONLY)
        try:
            os.posix_fadvise(fd, off, nbytes, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
        total += nbytes
    return total


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--layers", default="", help="engram layers (default: every engram layer of the config)")
    ap.add_argument("--tokens", type=int, default=5120)
    ap.add_argument("--threads", default="1,8,16,32")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--cold", action="store_true", help="drop the table's pages before every run (fresh subprocess)")
    ap.add_argument("--check", action="store_true", help="threaded rows == EngramTable.rows, byte for byte")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--madv-random", action="store_true", help="madvise(MADV_RANDOM) on the table mapping (no read-around)"
    )
    ap.add_argument("--io", default="mmap", choices=("mmap", "pread"), help="gather by mmap indexing or by pread a row")
    ap.add_argument("--same-ids", action="store_true", help="the same token ids every run (the warm best case)")
    ap.add_argument("--_worker", default=None, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    os.environ["ENGRAM_BENCH_IO"] = a.io
    if a.madv_random:
        os.environ["ENGRAM_BENCH_MADV"] = "random"  # (inherited by the cold runs' subprocesses)
    if a._worker:  # one cold run in a fresh process
        layer, threads, seed = map(int, a._worker.split(","))
        print("RESULT " + json.dumps(run_once(a.ckpt, layer, a.tokens, threads, seed, a.check)), flush=True)
        return 0
    cfg, ck, hasher = _setup(a.ckpt)
    layers = [int(x) for x in a.layers.split(",") if x] or list(cfg.engram_layers)
    threads = [int(x) for x in a.threads.split(",") if x]
    print(
        f"engram layers {layers}; {a.tokens} tokens x {(hasher.max_ngram_size - 1) * hasher.n_heads} lookups a token; "
        f"{'cold' if a.cold else 'warm'}; threads {threads}; {a.repeats} runs each; {os.cpu_count()} CPUs",
        flush=True,
    )
    rows = []
    for layer in layers:
        for t in threads:
            for r in range(a.repeats):
                seed = a.seed if a.same_ids else a.seed + 1000 * r + t
                if a.cold:
                    dropped = drop_pages(ck, layer)
                    cmd = [
                        sys.executable,
                        os.path.abspath(__file__),
                        "--ckpt",
                        a.ckpt,
                        "--tokens",
                        str(a.tokens),
                        "--_worker",
                        f"{layer},{t},{seed}",
                    ] + (["--check"] if a.check else [])
                    out = subprocess.run(cmd, capture_output=True, text=True, env=os.environ)
                    line = next((l for l in out.stdout.splitlines() if l.startswith("RESULT ")), None)
                    if line is None:
                        print(out.stdout[-2000:], out.stderr[-2000:], file=sys.stderr)
                        return 1
                    res = json.loads(line[7:])
                    res["dropped_mb"] = round(dropped / 2**20)
                else:
                    if r == 0 and not a.same_ids:
                        run_once(
                            a.ckpt, layer, a.tokens, t, seed + 7, False, cfg, ck, hasher
                        )  # (page in the hasher etc.)
                    res = run_once(a.ckpt, layer, a.tokens, t, seed, a.check, cfg, ck, hasher)
                rows.append(res)
                print(json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in res.items()}), flush=True)
    print(
        f"\n{'layer':>5} {'threads':>7} {'hash':>8} {'gather':>9} {'dequant':>9} {'assemble':>9} {'total':>9}   (ms, median of {a.repeats})"
    )
    for layer in layers:
        for t in threads:
            rs = [x for x in rows if x["layer"] == layer and x["threads"] == t]
            med = lambda k: sorted(x[k] for x in rs)[len(rs) // 2]  # noqa: E731
            ok = "" if not a.check else ("  equal" if all(x.get("check_equal") for x in rs) else "  MISMATCH")
            print(
                f"{layer:>5} {t:>7} {med('hash_ms'):8.1f} {med('gather_ms'):9.1f} {med('dequant_ms'):9.1f} {med('assemble_ms'):9.1f} {med('total_ms'):9.1f}{ok}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
