# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""A0-stock: the roofline that matters, measured on stock ``ttnn.linear``.

The custom streaming kernel measured 162 GB/s (31.7% of peak) in A0. Stock
``ttnn.linear`` reached 71% on the same shape in the same sweep, and the model's
own decode matmuls run at 270-400 GB/s, so the engine is now built on stock
matmuls instead (see the plan's decision). This establishes its roofline.

Three questions, each with a failure mode that would invalidate the plan:

1. **Does time scale with bytes?** bf16 -> bfp8_b is 2x fewer bytes and bfp4_b
   another 2x. If the times do not roughly halve, the matmul at M=1/32/64 is not
   bandwidth-bound and no format change will help.
2. **Which config?** The DRAM-sharded decode config and the 1D-mcast config that
   the TP GDN path uses behave differently at small M. Both are measured.
3. **Does M matter?** M=32 and M=64 are what prefill (and the block-masked scale
   trick's G rows) will use; M=1 is decode.

Every number carries the commit, chip, env and firmware through bench.record,
and the byte count used for GB/s is the real weight size in that format, so the
percentages are comparable across formats.

Usage::

    python -m models.demos.blackhole.bonsai2.native.bench.a0_stock
"""

from __future__ import annotations

import argparse
import os

import torch

from .record import DRAM_PEAK_GBS, append_jsonl, new_record
from .timing import Timer

#: (name, K, N) with y = x @ W.T.
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

#: ttnn dtypes and their bits per weight, for the byte accounting.
DTYPES = {"bf16": 16, "bfp8": 8, "bfp4": 4}


def weight_bytes(n: int, k: int, fmt: str) -> int:
    """Real on-device weight bytes for one format, including BFP block overhead."""
    if fmt == "bf16":
        return n * k * 2
    if fmt == "bfp8":
        # BFP8: 1 byte per weight + 1 exponent byte per 16-element face row.
        return n * k + (n * k // 16)
    if fmt == "bfp4":
        # BFP4: 4 bits per weight + the same exponent overhead.
        return n * k // 2 + (n * k // 16)
    raise ValueError(fmt)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default="")
    ap.add_argument("--formats", default="bf16,bfp8,bfp4")
    ap.add_argument("--m", default="1,32,64")
    ap.add_argument("--configs", default="dram,1d", help="dram = DRAM-sharded decode config, 1d = 1D mcast")
    ap.add_argument("--calls", type=int, default=10)
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--out", default=os.path.expanduser("~/avan/native2bit/agentA/logs/a0_stock.jsonl"))
    args = ap.parse_args()

    import ttnn

    device = ttnn.open_device(device_id=int(os.environ.get("BONSAI_DEVICE", 0)), l1_small_size=24576)
    timer = Timer(warmup=3, calls=args.calls, repeat=args.repeat)
    shapes = SHAPES
    if args.shapes:
        want = set(args.shapes.split(","))
        shapes = [s for s in shapes if s[0] in want]
    ms = [int(x) for x in args.m.split(",")]
    formats = args.formats.split(",")
    configs = args.configs.split(",")

    dt = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}
    results = {}
    try:
        for name, k, n in shapes:
            print(f"\n{name} K={k} N={n}", flush=True)
            # Random bf16 weights, not a ternary code tensor: the codes would have
            # to be materialized as float32 on the host (1.3 GB at N=16384), and
            # the point of A0-stock is the matmul's speed, not the values.
            w_src = torch.randn((k, n), dtype=torch.bfloat16)
            for fmt in formats:
                for m in ms:
                    # Weights [in, out] for ttnn.linear.
                    wt = ttnn.from_torch(
                        w_src,
                        dtype=dt[fmt],
                        layout=ttnn.TILE_LAYOUT,
                        device=device,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )
                    nb = weight_bytes(n, k, fmt)
                    for cfg in configs:
                        x_mem = ttnn.DRAM_MEMORY_CONFIG if cfg == "dram" else ttnn.L1_MEMORY_CONFIG
                        out_mem = ttnn.DRAM_MEMORY_CONFIG if cfg == "dram" else ttnn.L1_MEMORY_CONFIG
                        x = ttnn.from_torch(
                            torch.randn((1, 1, m, k), dtype=torch.bfloat16),
                            dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT,
                            device=device,
                            memory_config=x_mem,
                        )

                        def once():
                            y = ttnn.linear(x, wt, memory_config=out_mem)
                            ttnn.synchronize_device(device)
                            ttnn.deallocate(y)

                        try:
                            med, samples = timer.best(once)
                        except Exception as e:  # noqa: BLE001
                            print(f"  {fmt:5s} M={m:2d} {cfg:4s}: FAILED {type(e).__name__}: {e}", flush=True)
                            ttnn.deallocate(x)
                            continue

                        gbs = nb / med / 1e9
                        rec = new_record(
                            f"a0stock_{name}", K=k, N=n, M=m, fmt=fmt, config=cfg
                        )
                        rec.samples = samples
                        print(
                            f"  {fmt:5s} M={m:2d} {cfg:4s} {med * 1e6:9.1f} us  "
                            f"{nb / 1e6:7.2f} MB  {gbs:7.1f} GB/s ({gbs / DRAM_PEAK_GBS * 100:5.1f}% peak)",
                            flush=True,
                        )
                        append_jsonl(args.out, rec, nb)
                        key = (name, fmt, cfg)
                        results.setdefault(key, {})[m] = gbs
                        ttnn.deallocate(x)
                    ttnn.deallocate(wt)

        # The scaling check the plan asks for: does bfp4 beat bf16?
        print(f"\nbyte-scaling check (GB/s by format, M={ms[0]}, per config):")
        for (name, fmt, cfg), per_m in results.items():
            if fmt != "bf16":
                continue
            base = per_m.get(ms[0])
            if not base:
                continue
            parts = []
            for f in formats:
                v = results.get((name, f, cfg), {}).get(ms[0])
                if v:
                    parts.append(f"{f}={v:.1f}")
            print(f"  {name:12s} {cfg:4s} {' '.join(parts)}")
    finally:
        ttnn.close_device(device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
