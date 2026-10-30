# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""A0: the BFP2 streaming roofline, measured before writing any new kernel.

Runs the existing DeepSeek ``DRAMStreamingMatmulCompressed`` with every tile
forced to BFP2 and no scales, on the projection shapes Bonsai actually uses.
That is the fastest the hardware can pull ternary weights out of DRAM with the
code that already exists, so it is the upper bound the new engine has to reach
and the number the plan's 61 tok/s projection rests on.

The weights are the real GGUF trits, not random data, because a BFP2 tile's
exponent section depends on which 16-element face rows are all zero and ternary
weights are sparse enough for that to change the byte stream.  Every tile is
forced to format index 2 (bfp2) with no per-tile format search.

Reports, per shape: microseconds, GB/s of weight traffic, and the fraction of
the 512 GB/s chip peak, over the plan's 5-run median.

Usage (inside a tt-metal venv, on a chip)::

    python -m models.demos.blackhole.bonsai2.native.bench.a0_roofline \
        --gguf ~/avan/models/Ternary-Bonsai-2-27B-gguf/Ternary-Bonsai-2-27B-PQ2_0.gguf --runs 5
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from .record import DRAM_PEAK_GBS, append_jsonl, environment, new_record

#: The projection shapes that matter, from the plan's microbenchmark list.
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

BFP2_TILE_BYTES = 320
TILE = 32


def load_codes(gguf, name: str, k: int, n: int) -> np.ndarray:
    """Real trits ``[N, K]`` for a projection, matching it by shape.

    Shapes collide across projections of identical size (gate and up, for one),
    which is fine here: any real projection of that shape has the same sparsity
    and therefore the same BFP2 exponent distribution.
    """
    from .. import gguf_ternary as gt

    for t in gguf.tensors.values():
        if t.ggml_type in gt.TERNARY_TYPES and t.shape == (n, k):
            return gt.read_ternary(gguf, t.name).codes
    rng = np.random.default_rng(abs(hash(name)) % (2**32))
    return rng.integers(-1, 2, size=(n, k)).astype(np.int8)


def build(ttnn, device, codes: np.ndarray, cores_per_bank: int, subblock_k: int):
    """Build one reusable callable that runs the compressed matmul once.

    Weight setup happens here, once, so the measured loop contains only the op.
    """
    from models.demos.deepseek_v3_b1.compressed_tensor import CompressedTensor
    from models.demos.deepseek_v3_b1.micro_ops.dram_streaming_matmul_compressed.op import (
        DRAMStreamingMatmulCompressed,
    )
    from models.demos.deepseek_v3_b1.tests.unit_tests.test_dram_streaming_matmul import shuffle_tensor_tiles

    n, k = codes.shape
    primary = device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)
    num_banks = len(primary)
    step = TILE * num_banks * cores_per_bank
    n_padded = ((n + step - 1) // step) * step
    per_core_n = n_padded // (num_banks * cores_per_bank)
    kt = k // TILE

    w = torch.from_numpy(codes.astype(np.float32)).T.contiguous()  # [K, N]
    if n_padded != n:
        w = torch.nn.functional.pad(w, (0, n_padded - n))
    w = shuffle_tensor_tiles(w, TILE, num_banks)

    compute = []
    for p in primary:
        for off in range(cores_per_bank):
            compute.append(ttnn.CoreCoord(p.x + off, p.y))
    compute_set = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in compute])

    dram_grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(device.dram_grid_size().x - 1, device.dram_grid_size().y - 1))]
    )
    b_shard = ttnn.ShardSpec(dram_grid, [k, per_core_n * cores_per_bank], ttnn.ShardOrientation.ROW_MAJOR)
    b_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, b_shard)

    # Uniform BFP2 (format index 2), no assigner metrics.
    assignment = np.full((kt, n_padded // TILE), 2, dtype=np.int8)
    ct = CompressedTensor.from_bspm(w, assignment, device=device, memory_config=b_mem)

    m = 1
    a_torch = torch.randn((m, k), dtype=torch.bfloat16).repeat(len(compute), 1)
    a_shard = ttnn.ShardSpec(compute_set, [m, k], ttnn.ShardOrientation.ROW_MAJOR)
    a_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, a_shard)
    ttnn_a = ttnn.from_torch(
        a_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=a_mem,
        tile=ttnn.Tile([m, TILE]),
    )
    out_shard = ttnn.ShardSpec(compute_set, [m, per_core_n], ttnn.ShardOrientation.ROW_MAJOR)
    out_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, out_shard)

    def once():
        out = ttnn.from_torch(
            torch.zeros((m, n_padded), dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=out_mem,
            tile=ttnn.Tile([m, TILE]),
        )
        r = DRAMStreamingMatmulCompressed.op(ttnn_a, ct, out, subblock_k=subblock_k, cores_per_bank=cores_per_bank)
        ttnn.synchronize_device(device)
        ttnn.deallocate(r)

    # Real bytes streamed: padded tiles are read too, so they count.
    nbytes = kt * (n_padded // TILE) * BFP2_TILE_BYTES
    return once, nbytes, n_padded, per_core_n, len(compute), num_banks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--shapes", default="")
    ap.add_argument("--cores-per-bank", default="1,2,4")
    ap.add_argument("--subblock-k", default="4", help="4 matches the G=128 group")
    ap.add_argument("--out", default=os.path.expanduser("~/avan/native2bit/agentA/logs/a0_roofline.jsonl"))
    args = ap.parse_args()

    import ttnn
    from .. import gguf_ternary as gt

    gguf = gt.read_gguf(args.gguf)
    shapes = SHAPES
    if args.shapes:
        want = set(args.shapes.split(","))
        shapes = [s for s in SHAPES if s[0] in want]
    cpb_list = [int(x) for x in args.cores_per_bank.split(",")]
    sbk_list = [int(x) for x in args.subblock_k.split(",")]

    device = ttnn.open_device(device_id=0, l1_small_size=24576)
    best_per_shape = {}
    try:
        for name, k, n in shapes:
            codes = load_codes(gguf, name, k, n)
            for cpb in cpb_list:
                for sbk in sbk_list:
                    if (k // TILE) % sbk:
                        print(f"{name} cpb={cpb} sbk={sbk}: skipped, Kt={k // TILE} not divisible", flush=True)
                        continue
                    try:
                        once, nbytes, n_pad, pcn, ncores, nbanks = build(ttnn, device, codes, cpb, sbk)
                    except Exception as e:  # noqa: BLE001
                        print(f"{name} cpb={cpb} sbk={sbk}: BUILD FAILED {type(e).__name__}: {e}", flush=True)
                        continue

                    for _ in range(args.warmup):
                        once()
                    rec = new_record(
                        f"a0_{name}",
                        K=k,
                        N=n,
                        N_padded=n_pad,
                        per_core_n=pcn,
                        cores=ncores,
                        banks=nbanks,
                        cores_per_bank=cpb,
                        subblock_k=sbk,
                        gguf=os.path.basename(args.gguf),
                    )
                    for _ in range(args.runs):
                        t0 = time.perf_counter()
                        once()
                        rec.samples.append(time.perf_counter() - t0)

                    print(f"{name:12s} K={k:6d} N={n:6d} cpb={cpb} sbk={sbk:2d}  " + rec.summary(nbytes), flush=True)
                    append_jsonl(args.out, rec, nbytes)
                    if name not in best_per_shape or rec.median < best_per_shape[name][0]:
                        best_per_shape[name] = (rec.median, cpb, sbk, rec.gbs(nbytes))
    finally:
        ttnn.close_device(device)

    print()
    print("best per shape:")
    total_bytes = 0.0
    total_time = 0.0
    for name, (med, cpb, sbk, gbs) in best_per_shape.items():
        print(f"  {name:12s} {med * 1e6:9.1f} us  {gbs:7.1f} GB/s  (cpb={cpb} sbk={sbk})")
    print(f"\nDRAM peak assumed: {DRAM_PEAK_GBS} GB/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
