# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Does the compute kernel config, not the format, set the "compute floor"?

The A0-stock and A1-bound sweeps called ``ttnn.linear`` with no
``compute_kernel_config``, so they ran at ttnn's default math fidelity. The model
itself does not: ``qwen36/tt/mlp.py`` builds

    WormholeComputeKernelConfig(math_fidelity=LoFi, fp32_dest_acc_en=True,
                                packer_l1_acc=True)

for decode and passes it on every call. LoFi is legitimate for ternary weights --
the codes are exactly representable, so the only thing LoFi can lose is
accumulation precision, not the operand values.

If the default is much slower than LoFi, then every "compute floor" and "bfp4
buys nothing on ffn_down" conclusion in the A0 tables is a property of the
benchmark, not the hardware, and the plan's byte-scaled rooflines are back in
play. That is the difference between "the format cannot help" and "the format
helps, measure it properly", so it has to be settled before anything is built on
those tables.

Reports each (shape, format) at the default config and at the model's decode
config, with the ratio.

Usage::

    python -m models.demos.blackhole.bonsai2.native.bench.a1_ckc
"""

from __future__ import annotations

import argparse
import os

import torch

from .record import DRAM_PEAK_GBS, append_jsonl, new_record
from .timing import Timer

SHAPES = [
    ("gdn_qkvz", 5120, 16384),
    ("attn_q_gate", 5120, 12288),
    ("ffn_gate", 5120, 17408),
    ("ffn_down", 17408, 5120),
    ("lm_head", 5120, 248320),
]


def weight_bytes(n: int, k: int, fmt: str) -> int:
    if fmt == "bf16":
        return n * k * 2
    if fmt == "bfp8":
        return n * k + n * k // 16
    if fmt == "bfp4":
        return n * k // 2 + n * k // 16
    raise ValueError(fmt)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default="")
    ap.add_argument("--formats", default="bfp8,bfp4")
    ap.add_argument("--m", default="1")
    ap.add_argument("--calls", type=int, default=10)
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--out", default=os.path.expanduser("~/avan/native2bit/agentA/logs/a1_ckc.jsonl"))
    args = ap.parse_args()

    import ttnn

    device = ttnn.open_device(device_id=int(os.environ.get("BONSAI_DEVICE", 0)), l1_small_size=24576)
    timer = Timer(warmup=3, calls=args.calls, repeat=args.repeat)
    dt = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}
    shapes = SHAPES
    if args.shapes:
        want = set(args.shapes.split(","))
        shapes = [s for s in shapes if s[0] in want]
    ms = [int(x) for x in args.m.split(",")]
    formats = args.formats.split(",")

    # Exactly the configs the model uses (mlp.py decode / prefill).
    ckc_decode = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, packer_l1_acc=True
    )
    ckc_prefill = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, packer_l1_acc=False
    )
    ckcs = {"default": None, "lofi-l1acc": ckc_decode, "lofi": ckc_prefill}

    print(f"Peak {DRAM_PEAK_GBS} GB/s.  Peak-relative GB/s uses the real weight bytes.")
    try:
        for name, k, n in shapes:
            w_src = torch.randn((k, n), dtype=torch.bfloat16)
            for m in ms:
                x = ttnn.from_torch(
                    torch.randn((1, 1, m, k), dtype=torch.bfloat16), dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG,
                )
                print(f"\n{name} K={k} N={n} M={m}  ({weight_bytes(n, k, 'bf16') / 1e6:.1f} MB bf16)", flush=True)
                print(f"  {'fmt':5s} " + " ".join(f"{c:>12s}" for c in ckcs))
                for fmt in formats:
                    wt = ttnn.from_torch(
                        w_src, dtype=dt[fmt], layout=ttnn.TILE_LAYOUT,
                        device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )
                    nb = weight_bytes(n, k, fmt)
                    cells = []
                    for cname, ckc in ckcs.items():
                        def once(ckc=ckc):
                            y = ttnn.linear(x, wt, compute_kernel_config=ckc,
                                            memory_config=ttnn.DRAM_MEMORY_CONFIG)
                            ttnn.synchronize_device(device)
                            ttnn.deallocate(y)

                        try:
                            med, samples = timer.best(once)
                            rec = new_record(f"a1ckc_{name}", K=k, N=n, M=m, fmt=fmt, ckc=cname)
                            rec.samples = samples
                            append_jsonl(args.out, rec, nb)
                            cells.append(med * 1e6)
                        except Exception as e:  # noqa: BLE001
                            print(f"  {fmt} {cname}: FAILED {str(e)[:80]}", flush=True)
                            cells.append(float("nan"))
                    base = cells[0]
                    line = " ".join(
                        f"{c:7.1f} us {'' if i == 0 else f'({base / c:4.2f}x)'}"
                        for i, c in enumerate(cells)
                    )
                    print(f"  {fmt:5s} {line}", flush=True)
                    best = min(c for c in cells if c == c)
                    print(f"        best {best:.1f} us = {nb / best / 1e9:.1f} GB/s "
                          f"({nb / best / 1e9 / DRAM_PEAK_GBS * 100:.1f}% peak)", flush=True)
                    ttnn.deallocate(wt)
                ttnn.deallocate(x)
    finally:
        ttnn.close_device(device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
