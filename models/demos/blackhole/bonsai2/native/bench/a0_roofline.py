# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""A0: the BFP2 streaming roofline, and the cross-check that explains it.

Runs the existing DeepSeek ``DRAMStreamingMatmulCompressed`` with every tile
forced to BFP2 and no scales, on the projection shapes Bonsai uses, and reports
microseconds and GB/s against the 512 GB/s chip peak.  That is the fastest the
hardware can pull ternary weights with the code that already exists, so it is
the bound the new engine must reach.

Three configurations are timed per shape:

  bfp2      the ternary roofline (320 B tiles, what the engine will stream)
  bfp8      the same data at 1088 B/tile, as a bytes-vs-time control
  linear    plain ttnn.linear on bf16 weights, the path the dense Bonsai demo
            uses, as an independent calibration: whatever it reaches, this chip
            reaches, so anything far below it is a harness or kernel artefact

If bfp8 and bfp2 take the same time, the op is not bandwidth-bound and the GB/s
numbers are meaningless as a roofline -- that is a finding to report, not to
average away.

The weights are real GGUF trits, because a BFP2 tile's exponent section depends
on which 16-element face rows are all zero and ternary weights are sparse enough
for that to change the byte stream.

Usage (inside a tt-metal venv, on a chip)::

    python -m models.demos.blackhole.bonsai2.native.bench.a0_roofline \
        --gguf ~/avan/models/Ternary-Bonsai-2-27B-gguf/Ternary-Bonsai-2-27B-PQ2_0.gguf
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from .record import DRAM_PEAK_GBS, append_jsonl, new_record
from .timing import Timer

TILE = 32
TILE_BYTES = {"bfp8": 1088, "bfp4": 576, "bfp2": 320}
FMT_INDEX = {"bfp8": 0, "bfp4": 1, "bfp2": 2}

#: (name, K, N) with y = x @ W.T, so W is [N, K] exactly as the GGUF stores it.
SHAPES = [
    ("gdn_qkvz", 5120, 16384),
    ("gdn_ab", 5120, 96),
    ("gdn_out", 6144, 5120),
    ("attn_q_gate", 5120, 12288),
    ("attn_kv", 5120, 2048),
    ("attn_o", 6144, 5120),
    ("ffn_gate", 5120, 17408),
    ("ffn_up", 5120, 17408),
    ("ffn_down", 17408, 5120),
    ("lm_head", 5120, 248320),
]


def load_codes(gguf, name: str, k: int, n: int) -> np.ndarray:
    """Real trits ``[N, K]`` for a projection, matched by shape."""
    from .. import gguf_ternary as gt

    for t in gguf.tensors.values():
        if t.ggml_type in gt.TERNARY_TYPES and t.shape == (n, k):
            return gt.read_ternary(gguf, t.name).codes
    rng = np.random.default_rng(abs(hash(name)) % (2**32))
    return rng.integers(-1, 2, size=(n, k)).astype(np.int8)


def build_streaming(ttnn, device, codes: np.ndarray, cpb: int, subblock_k: int, fmt: str):
    """Set up the compressed op for one shape and format; return a callable and byte count."""
    from models.demos.deepseek_v3_b1.compressed_tensor import CompressedTensor
    from models.demos.deepseek_v3_b1.micro_ops.dram_streaming_matmul_compressed.op import (
        DRAMStreamingMatmulCompressed,
    )
    from models.demos.deepseek_v3_b1.tests.unit_tests.test_dram_streaming_matmul import shuffle_tensor_tiles

    n, k = codes.shape
    primary = device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)
    nbanks = len(primary)
    step = TILE * nbanks * cpb
    n_pad = ((n + step - 1) // step) * step
    per_core_n_tiles = n_pad // (nbanks * cpb) // TILE
    kt = k // TILE

    # The kernel's B is [K, N]; the shuffle is what puts tiles in the streaming
    # order, and it must see the whole matrix, banks included.
    w = torch.from_numpy(codes.astype(np.float32)).T.contiguous()
    if n_pad != n:
        w = torch.nn.functional.pad(w, (0, n_pad - n))
    w = shuffle_tensor_tiles(w, TILE, nbanks)

    compute = [ttnn.CoreCoord(p.x + off, p.y) for p in primary for off in range(cpb)]
    compute_set = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in compute])
    dram_grid = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0), ttnn.CoreCoord(device.dram_grid_size().x - 1, device.dram_grid_size().y - 1)
            )
        ]
    )
    b_mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(dram_grid, [k, per_core_n_tiles * TILE * cpb], ttnn.ShardOrientation.ROW_MAJOR),
    )
    assignment = np.full((kt, n_pad // TILE), FMT_INDEX[fmt], dtype=np.int8)
    ct = CompressedTensor.from_bspm(w, assignment, device=device, memory_config=b_mem)

    m = 1
    a_torch = torch.randn((m, k), dtype=torch.bfloat16).repeat(len(compute), 1)
    ttnn_a = ttnn.from_torch(
        a_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(compute_set, [m, k], ttnn.ShardOrientation.ROW_MAJOR),
        ),
        tile=ttnn.Tile([m, TILE]),
    )
    out_mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(compute_set, [m, per_core_n_tiles * TILE], ttnn.ShardOrientation.ROW_MAJOR),
    )
    out = ttnn.from_torch(
        torch.zeros((m, n_pad), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=out_mem,
        tile=ttnn.Tile([m, TILE]),
    )
    ct_ref = ct  # keep alive

    def once():
        DRAMStreamingMatmulCompressed.op(ttnn_a, ct_ref, out, subblock_k=subblock_k, cores_per_bank=cpb)
        ttnn.synchronize_device(device)

    nbytes = kt * (n_pad // TILE) * TILE_BYTES[fmt]
    return once, nbytes, n_pad, per_core_n_tiles, len(compute), nbanks


def build_linear(ttnn, device, codes: np.ndarray):
    """Plain ``ttnn.linear`` on bf16 weights: the dense path's own speed."""
    n, k = codes.shape
    wt = ttnn.from_torch(
        torch.from_numpy(codes.astype(np.float32)).bfloat16().T.contiguous(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    x = ttnn.from_torch(
        torch.randn((1, 1, 1, k), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    def once():
        y = ttnn.linear(x, wt, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.synchronize_device(device)
        ttnn.deallocate(y)

    return once, n * k * 2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--shapes", default="")
    ap.add_argument("--cores-per-bank", default="1,2,4")
    ap.add_argument("--subblock-k", default="4,16", help="4 matches the G=128 group; 16 tests pipeline depth")
    ap.add_argument("--formats", default="bfp2,bfp8", help="comma-separated; add 'linear' for the bf16 control")
    ap.add_argument("--calls", type=int, default=5)
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--out", default=os.path.expanduser("~/avan/native2bit/agentA/logs/a0_roofline.jsonl"))
    args = ap.parse_args()

    import ttnn
    from .. import gguf_ternary as gt

    gguf = gt.read_gguf(args.gguf)
    shapes = SHAPES
    if args.shapes:
        want = set(args.shapes.split(","))
        shapes = [s for s in SHAPES if s[0] in want]
    formats = [f.strip() for f in args.formats.split(",")]
    cpb_list = [int(x) for x in args.cores_per_bank.split(",")]
    sbk_list = [int(x) for x in args.subblock_k.split(",")]
    timer = Timer(warmup=2, calls=args.calls, repeat=args.repeat)

    device = ttnn.open_device(device_id=int(os.environ.get("BONSAI_DEVICE", 0)), l1_small_size=24576)
    best = {}
    try:
        for name, k, n in shapes:
            codes = load_codes(gguf, name, k, n)
            print(f"\n{name} K={k} N={n}  ({n * k / 1e6:.1f}M weights)", flush=True)

            if "linear" in formats:
                try:
                    fn, nbytes = build_linear(ttnn, device, codes)
                    med, samples = timer.best(fn)
                    gbs = nbytes / med / 1e9
                    rec = new_record(f"a0_{name}_linear", K=k, N=n, method="ttnn.linear bf16")
                    rec.samples = samples
                    print(
                        f"  linear       {med * 1e6:9.1f} us  {nbytes / 1e6:7.2f} MB  "
                        f"{gbs:7.1f} GB/s ({gbs / DRAM_PEAK_GBS * 100:5.1f}% peak)",
                        flush=True,
                    )
                    append_jsonl(args.out, rec, nbytes)
                except Exception as e:  # noqa: BLE001
                    print(f"  linear       FAILED {type(e).__name__}: {e}", flush=True)

            for fmt in [f for f in formats if f != "linear"]:
                for cpb in cpb_list:
                    for sbk in sbk_list:
                        if (k // TILE) % sbk:
                            continue
                        try:
                            fn, nbytes, n_pad, pcnt, ncores, nbanks = build_streaming(
                                ttnn, device, codes, cpb, sbk, fmt
                            )
                            med, samples = timer.best(fn)
                        except Exception as e:  # noqa: BLE001
                            print(f"  {fmt:5s} cpb={cpb} sbk={sbk:2d} FAILED {type(e).__name__}: {e}", flush=True)
                            continue
                        gbs = nbytes / med / 1e9
                        rec = new_record(
                            f"a0_{name}_{fmt}",
                            K=k,
                            N=n,
                            N_padded=n_pad,
                            per_core_n_tiles=pcnt,
                            cores=ncores,
                            banks=nbanks,
                            cores_per_bank=cpb,
                            subblock_k=sbk,
                            fmt=fmt,
                            gguf=os.path.basename(args.gguf),
                        )
                        rec.samples = samples
                        print(
                            f"  {fmt:5s} cpb={cpb} sbk={sbk:2d} {med * 1e6:9.1f} us  "
                            f"{nbytes / 1e6:7.2f} MB  {gbs:7.1f} GB/s ({gbs / DRAM_PEAK_GBS * 100:5.1f}% peak)",
                            flush=True,
                        )
                        append_jsonl(args.out, rec, nbytes)
                        key = (name, fmt)
                        if key not in best or med < best[key][0]:
                            best[key] = (med, gbs, cpb, sbk)
    finally:
        ttnn.close_device(device)

    print(f"\nbest per (shape, format); peak assumed {DRAM_PEAK_GBS} GB/s:")
    for (name, fmt), (med, gbs, cpb, sbk) in best.items():
        print(f"  {name:12s} {fmt:5s} {med * 1e6:9.1f} us  {gbs:7.1f} GB/s  (cpb={cpb} sbk={sbk})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
