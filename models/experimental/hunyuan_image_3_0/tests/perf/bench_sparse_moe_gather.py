# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Decisive go/no-go micro-benchmark for capacity-gathered (sparse) MoE at the
# DENOISE shape (S~=4160, M-per-expert FLOP-bound), NOT the AR-decode shape
# (S=64, where capacity-gather was already shown to lose — see the AR verdict).
#
# It times the REAL mechanism on a single device (experts are per-device resident,
# so one device is representative of the per-device expert cost):
#
#   dense  : for 16 experts: gate_up[M=S] -> swiglu -> down[M=S]                (today)
#   sparse : for 16 experts: embedding-gather[M=C] -> gate_up -> swiglu
#                            -> down -> embedding_bw scatter-add back           (proposed)
#
# The question this answers: at S=4160 does the gather+scatter overhead amortize
# against the ~8x-smaller M=C matmul, i.e. is sparse < dense wall time per layer?
#
# MEASURED VERDICT (2x2 Blackhole, 2026-07): break-even at best.
#   dense=37.6 ms/layer ; sparse CAP=1024=57.1 (0.66x LOSS) ; CAP=512=35.4 (1.06x, drops tokens).
#   The matmul win is real (M=C matmul 0.36 ms vs dense) but the scatter-back is the wall:
#   ttnn.gather=169 ms (unusable), scatter_add=10 ms, embedding_bw=2.53 ms — all >> the 0.36 ms
#   matmul they combine. ttnn.embedding gather IS fast (0.06 ms). The fused fast-combine op
#   (ttnn.experimental.moe_compute) is Galaxy-only (16-row ring) and will not run on 2x2 BH.
#   Re-run this the day a fast Blackhole scatter-add lands: if it hits ~0.2 ms -> ~3x win.
#
# Run (needs a Blackhole device):
#   python_env/bin/python models/experimental/hunyuan_image_3_0/tests/perf/bench_sparse_moe_gather.py
# Env: HY_S (seq, default 4160), HY_CAP (capacity C, default 1024), HY_EPD (experts/device, default 16)

import os
import time

import torch
import ttnn

S = int(os.environ.get("HY_S", "4160"))
CAP = int(os.environ.get("HY_CAP", "1024"))  # tile-aligned capacity per expert
EPD = int(os.environ.get("HY_EPD", "16"))  # local experts per device
H = 4096  # hidden
I2 = 6144  # gate_and_up out (2 * intermediate)
I = 3072  # intermediate (down proj in)
TOPK = 8
E = 64
ITERS = int(os.environ.get("HY_ITERS", "5"))

assert CAP % 32 == 0, "capacity must be tile-aligned"


def _ckcfg():
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )


def _swiglu(gu):
    x1, x2 = ttnn.chunk(gu, 2, dim=-1)
    h = ttnn.multiply(x2, x1, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
    ttnn.deallocate(x1)
    ttnn.deallocate(x2)
    return h


def _time(fn, label):
    # warmup
    fn()
    ttnn.synchronize_device(dev)
    t0 = time.time()
    for _ in range(ITERS):
        fn()
    ttnn.synchronize_device(dev)
    dt = (time.time() - t0) / ITERS * 1e3
    print(f"  {label:32s} {dt:8.2f} ms / layer  ({dt / EPD * 1e3:8.1f} us / expert)")
    return dt


def main():
    global dev
    dev = ttnn.open_device(device_id=0)
    ck = _ckcfg()

    print(f"S={S} CAP={CAP} EPD={EPD} H={H} 2I={I2} I={I}  topk={TOPK}/{E}  iters={ITERS}")
    print(f"expected tokens/expert = S*topk/E = {S * TOPK / E:.0f}  (capacity C={CAP})\n")

    # weights: one representative expert's gate_up [H,2I] and down [I,H], bf8
    wgu = ttnn.from_torch(
        torch.randn(H, I2) * 0.02,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=dev,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    wdn = ttnn.from_torch(
        torch.randn(I, H) * 0.02,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=dev,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    x_full = ttnn.from_torch(
        torch.randn(1, S, H) * 0.5,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=dev,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # ---- DENSE: 16 experts at M=S ----
    def dense():
        for _ in range(EPD):
            gu = ttnn.linear(x_full, wgu, compute_kernel_config=ck, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            h = _swiglu(gu)
            ttnn.deallocate(gu)
            out = ttnn.linear(h, wdn, compute_kernel_config=ck, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(h)
            ttnn.deallocate(out)

    # ---- SPARSE: 16 experts at M=CAP + gather/scatter ----
    # Build a realistic per-expert selection: ~S*topk/E random token ids, padded to CAP.
    # Fast path: ttnn.embedding for the row-gather (0.06 ms vs ttnn.gather's 169 ms),
    # ttnn.embedding_bw for the scatter-add combine (its backward IS a scatter-add).
    tok_per_e = int(S * TOPK / E)
    ids = [
        ttnn.from_torch(
            torch.cat(
                [torch.randperm(S)[: min(tok_per_e, CAP)], torch.zeros(max(0, CAP - tok_per_e), dtype=torch.long)]
            ).reshape(1, CAP),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=dev,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        for _ in range(EPD)
    ]
    x_rm = ttnn.to_layout(x_full, ttnn.ROW_MAJOR_LAYOUT)
    wtab = ttnn.from_torch(  # embedding_bw only reads this for the vocab (=S) size
        torch.zeros(S, H),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=dev,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    def sparse():
        acc = None
        for e in range(EPD):
            g = ttnn.embedding(ids[e], x_rm)  # [1,CAP,H] ROW_MAJOR
            gt = ttnn.to_layout(g, ttnn.TILE_LAYOUT)
            ttnn.deallocate(g)
            gu = ttnn.linear(gt, wgu, compute_kernel_config=ck, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(gt)
            h = _swiglu(gu)
            ttnn.deallocate(gu)
            out = ttnn.linear(h, wdn, compute_kernel_config=ck, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(h)
            out4 = ttnn.reshape(out, (1, 1, CAP, H))  # embedding_bw grad wants [1,1,N,H]
            ttnn.deallocate(out)
            gw = ttnn.embedding_bw(ids[e], wtab, out4)  # scatter-add -> [S,H]
            ttnn.deallocate(out4)
            if acc is None:
                acc = gw
            else:
                tmp = ttnn.add(acc, gw)
                ttnn.deallocate(acc)
                ttnn.deallocate(gw)
                acc = tmp
        ttnn.deallocate(acc)

    print("timing (lower is better):")
    d = _time(dense, "dense  (16 x M=S)")
    try:
        s = _time(sparse, "sparse (16 x M=CAP + gather/scatter)")
        print(f"\n  speedup = {d / s:.2f}x   ({'WIN' if s < d else 'LOSS'})")
    except Exception as ex:
        print(f"\n  sparse path FAILED: {type(ex).__name__}: {ex}")
        print("  (gather/scatter op signature or dtype/layout mismatch — see error; mechanism needs adjustment)")

    ttnn.close_device(dev)


if __name__ == "__main__":
    main()
