#!/usr/bin/env python3
"""Per-op power breakdown of a transformer block, as a tt-ember workload.

tt-ember drives any executable passed as --app-exe and parses its stdout, so this is used
unmodified: the script runs each op of a Llama-style decoder block as its own timed interval
and prints the summary rows that parser.py's RE_PROGRAM_ROW expects.

Two conventions are needed to fit that format, which only understands a compute grid:

  * the grid field carries the op index as "<idx>x1", so every op is a distinct interval;
  * the cores field also carries the op index, so tt-ember's own figures separate the points.

The op names are printed as "# OP <idx> <name> ..." lines, which RE_PROGRAM_ROW does not match
and parser.py therefore ignores. preview_op_breakdown.py reads them back out of summary.txt to
label the breakdown. Nothing in tt-ember is changed.

Batching. Telemetry samples about every 103 us, so an op call shorter than that is never
resolved and its measured power is diluted by whatever host dispatch gap follows it -- measured
at roughly 24% for a 41 us call, vanishing once a call spans ~10 samples. Each op is therefore
run on a batch large enough that one call takes --target-call-ms, which lengthens the call
without changing the per-activation work being measured. The batch is chosen per op, since call
durations across a block span more than an order of magnitude.
"""
import argparse
import os
import sys
import time
from datetime import datetime

if not os.environ.get("TT_METAL_HOME"):
    sys.stderr.write(
        "TT_METAL_HOME is not set. Export it to your tt-metal checkout and run this from a\n"
        "shell with the tt-metal python venv active (see README.md).\n")
    raise SystemExit(2)

try:
    import torch  # noqa: E402
    import ttnn  # noqa: E402
except ImportError as e:
    sys.stderr.write(
        f"Could not import ttnn ({e}).\n"
        "This must run under the tt-metal python venv -- create it with ./create_venv.sh and\n"
        "activate it before invoking tt-ember, so auto.py inherits it (see README.md).\n")
    raise SystemExit(2)


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")


# FLOPs per element for the non-matmul ops. Only the matmul counts (2*M*N*K) are exact; these
# are the usual accounting conventions, stated here so the pJ/FLOP figures can be interpreted:
#   add / multiply : 1  -- one arithmetic op per element
#   silu           : 4  -- sigmoid (exp, add, reciprocal) plus the multiply
#   rms_norm       : 4  -- square, accumulate, reciprocal-sqrt scale, multiply
#   softmax        : 5  -- max, subtract, exp, accumulate, divide
# They are conventions, not measurements. Comparing pJ/FLOP between a matmul and an
# elementwise op therefore compares an exact figure against an approximate one.
FLOPS_PER_ELEM = {"add": 1, "mul": 1, "silu": 4, "rms_norm": 4, "softmax": 5}

# Bytes of tensor data each op logically touches: inputs read plus outputs written, at 2 bytes
# per bfloat16 element. Unlike the FLOP conventions above this needs no assumption about what
# the op does internally -- it follows from the shapes alone -- so pJ/byte is a normaliser that
# does not depend on knowing the instruction mix. It is a lower bound on real traffic: it counts
# each tensor once and ignores tiling, reuse and any spill, so an op that re-reads an operand
# moves more than this.
BF16 = 2


def make_factories(device, seq, hidden, ffn, heads):
    """One Llama-style decoder block, decomposed into the ops it is made of.

    Each entry builds its tensors at a given batch and returns
    (callable, FLOPs per call, bytes of tensor data touched per call).
    """
    head_dim = hidden // heads

    def T(*shape):
        return ttnn.from_torch(
            torch.randn(*shape), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )

    def norm(b):
        x = T(b, 1, seq, hidden)
        n = b * seq * hidden
        return (lambda: ttnn.rms_norm(x)), n * FLOPS_PER_ELEM["rms_norm"], 2 * n * BF16

    def qkv(b):
        x, w = T(b, 1, seq, hidden), T(1, 1, hidden, 3 * hidden)
        by = (b * seq * hidden + hidden * 3 * hidden + b * seq * 3 * hidden) * BF16
        return (lambda: ttnn.matmul(x, w)), b * 2 * seq * hidden * 3 * hidden, by

    def qk(b):
        q, k_t = T(b, heads, seq, head_dim), T(b, heads, head_dim, seq)
        by = (2 * b * heads * seq * head_dim + b * heads * seq * seq) * BF16
        return (lambda: ttnn.matmul(q, k_t)), b * 2 * heads * seq * seq * head_dim, by

    def smax(b):
        sc = T(b, heads, seq, seq)
        n = b * heads * seq * seq
        return (lambda: ttnn.softmax(sc, dim=-1)), n * FLOPS_PER_ELEM["softmax"], 2 * n * BF16

    def av(b):
        sc, v = T(b, heads, seq, seq), T(b, heads, seq, head_dim)
        by = (b * heads * seq * seq + 2 * b * heads * seq * head_dim) * BF16
        return (lambda: ttnn.matmul(sc, v)), b * 2 * heads * seq * seq * head_dim, by

    def out_proj(b):
        x, w = T(b, 1, seq, hidden), T(1, 1, hidden, hidden)
        by = (2 * b * seq * hidden + hidden * hidden) * BF16
        return (lambda: ttnn.matmul(x, w)), b * 2 * seq * hidden * hidden, by

    def res_add(b):
        x = T(b, 1, seq, hidden)
        n = b * seq * hidden
        return (lambda: ttnn.add(x, x)), n * FLOPS_PER_ELEM["add"], 3 * n * BF16

    def ffn_up(b):
        x, w = T(b, 1, seq, hidden), T(1, 1, hidden, ffn)
        by = (b * seq * hidden + hidden * ffn + b * seq * ffn) * BF16
        return (lambda: ttnn.matmul(x, w)), b * 2 * seq * hidden * ffn, by

    def act(b):
        h = T(b, 1, seq, ffn)
        n = b * seq * ffn
        return (lambda: ttnn.silu(h)), n * FLOPS_PER_ELEM["silu"], 2 * n * BF16

    def gate(b):
        h = T(b, 1, seq, ffn)
        n = b * seq * ffn
        return (lambda: ttnn.multiply(h, h)), n * FLOPS_PER_ELEM["mul"], 3 * n * BF16

    def ffn_dn(b):
        h, w = T(b, 1, seq, ffn), T(1, 1, ffn, hidden)
        by = (b * seq * ffn + ffn * hidden + b * seq * hidden) * BF16
        return (lambda: ttnn.matmul(h, w)), b * 2 * seq * ffn * hidden, by

    return [
        ("rms_norm_in", norm), ("matmul_qkv", qkv), ("attn_qk", qk), ("softmax", smax),
        ("attn_av", av), ("matmul_out", out_proj), ("residual_add", res_add),
        ("rms_norm_post", norm), ("matmul_ffn_up", ffn_up), ("silu", act),
        ("gate_mul", gate), ("matmul_ffn_dn", ffn_dn),
    ]


def time_call(fn, device, reps=3):
    fn()
    ttnn.synchronize_device(device)
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    ttnn.synchronize_device(device)
    return (time.perf_counter() - t0) / reps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--hidden", type=int, default=2048)
    ap.add_argument("--ffn", type=int, default=8192)
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--target-call-ms", type=float, default=1.5,
                    help="Batch each op so one call takes about this long, to clear the ~103 us "
                         "telemetry sample interval by roughly 10x.")
    ap.add_argument("--max-batch", type=int, default=96)
    ap.add_argument("--target-seconds", type=float, default=4.0)
    ap.add_argument("--pause-seconds", type=float, default=5.0)
    ap.add_argument("--device-id", type=int, default=0)
    args = ap.parse_args()

    print("=== Transformer block op power breakdown (batched) ===", flush=True)
    print(f"seq={args.seq} hidden={args.hidden} ffn={args.ffn} heads={args.heads}", flush=True)
    print(f"Target call {args.target_call_ms:.2f} ms, window {args.target_seconds:.1f} s, "
          f"idle gap {args.pause_seconds:.1f} s", flush=True)

    device = ttnn.open_device(device_id=args.device_id)
    rows = []
    try:
        for idx, (name, factory) in enumerate(
                make_factories(device, args.seq, args.hidden, args.ffn, args.heads), start=1):
            if idx > 1:
                time.sleep(args.pause_seconds)

            # Size the batch from a single-activation timing, then rebuild at that batch.
            fn1, _, _ = factory(1)
            t1 = time_call(fn1, device)
            batch = max(1, min(args.max_batch, int(round(args.target_call_ms / 1000.0 / t1))))
            del fn1

            while True:
                try:
                    fn, flops, nbytes = factory(batch)
                    per_call = time_call(fn, device)
                    break
                except Exception as e:  # most likely out of device memory at this batch
                    if batch == 1:
                        raise
                    batch = max(1, batch // 2)
                    print(f"# NOTE {name}: retrying at batch={batch} ({str(e)[:60]})", flush=True)

            iters = max(1, int(args.target_seconds / per_call))
            print(f"# OP {idx} {name} iters={iters} flops_per_iter={flops} batch={batch} "
                  f"bytes_per_iter={nbytes} call_us={per_call*1e6:.1f}", flush=True)

            ttnn.synchronize_device(device)
            start = now_str()
            t0 = time.perf_counter()
            for _ in range(iters):
                fn()
            ttnn.synchronize_device(device)
            elapsed = time.perf_counter() - t0
            end = now_str()

            tflops = (flops * iters / elapsed / 1e12) if flops else 0.0
            rows.append((idx, iters, elapsed, tflops, elapsed / iters * 1000.0, start, end))
            print(f"# DONE {name}: batch={batch} {elapsed/iters*1e6:.1f} us/call "
                  f"({tflops:.1f} TFLOPS)", flush=True)
            del fn
    finally:
        ttnn.close_device(device)

    print()
    print(f"{'Grid':>8} {'Cores':>7} {'Iters':>10} {'Time [s]':>12} {'TFLOPS':>10} "
          f"{'Per iter [ms]':>16} {'Start Time':>27} {'End Time':>27}")
    for idx, iters, elapsed, tflops, per_iter_ms, start, end in rows:
        print(f"{f'{idx}x1':>8} {idx:>7} {iters:>10} {elapsed:>12.6f} {tflops:>10.2f} "
              f"{per_iter_ms:>16.6f} {start:>27} {end:>27}")
    print()
    print("Test Passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
