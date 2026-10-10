# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Sweep: flat_routed_expert at GLM shapes on one chip with balanced routing (every one of the 36 local experts gets
the same M rows), M from 32 to 5120. The model's call: C++ op built at the model's capacity (m 8192), indexed mode
(token_index into T = 5120 gathered tokens, scattered), clamped_silu, bfp4 weights (random: perf does not depend on
the values), y bf16 row-major, fp32 down (GLM_SWEEP_YRM=0: y bfp8 tiles; GLM_SWEEP_DFP32=0: bf16 DEST down).

Per M: device time per call (program real-time profiler, median of ITERS calls; wall time when it is inactive), time
per expert, DRAM utilization (weights + x bf16 reads + y bf16 writes against 512 GB/s) and math utilization against the
LoFi peak (4096 FLOP / cycle / core at the measured clock) over the 90 compute cores (64 gate/up + 26 down) and over
the whole 110-core grid. GLM_SWEEP_M (comma list) overrides the M list; GLM_SWEEP_DUMP=path saves y at M 128."""

import os
import statistics
import time

import pytest
import torch

import ttnn

# GLM_SWEEP_E / GLM_SWEEP_CAP / GLM_SWEEP_ACT: other deployments (local experts, capacity, activation name of
# flat_expert.ACTS)
E = int(os.environ.get("GLM_SWEEP_E", "36"))
NG, T = max(288, E), 5120
CAP = int(os.environ.get("GLM_SWEEP_CAP", "8192"))
ACT = os.environ.get("GLM_SWEEP_ACT", "clamped_silu")
# GLM_SWEEP_H / GLM_SWEEP_I: other expert shapes (hidden, intermediate), e.g. 7168 / 2048, 6144 / 2048, 3584 / 3072
H = int(os.environ.get("GLM_SWEEP_H", "4096"))
I = int(os.environ.get("GLM_SWEEP_I", "2048"))
MS = (32, 64, 96, 128, 160, 192, 256, 384, 512, 768, 1024, 2048, 5120)
ITERS = int(os.environ.get("GLM_SWEEP_ITERS", "10"))
BFP4_TILE = 576  # bytes: 512 mantissa + 64 shared exponents
W_BYTES = E * 3 * (H // 32) * (I // 32) * BFP4_TILE
YRM = os.environ.get("GLM_SWEEP_YRM", "1") == "1"  # 0: y as bfp8 tiles
DFP32 = os.environ.get("GLM_SWEEP_DFP32", "1") == "1"  # 0: bf16 DEST for the down projection
# GLM_SWEEP_OP=unified: ttnn.bringup.unified_routed_expert_moe on the same bfp4 experts at LoFi (fp32 DEST, packer L1
# acc), row-major bf16 dispatched buffer in; GLM_SWEEP_UNI_HP=1 high_precision (bf16 x / h / y), 0 the default path
UNIFIED = os.environ.get("GLM_SWEEP_OP", "flat") == "unified"
XBF16 = os.environ.get("GLM_SWEEP_XBF16", "0") == "1"  # flat: x / h as bf16 tiles (else bfp8)
HBF16 = os.environ.get("GLM_SWEEP_HBF16", "0") == "1"
UNI_HP = os.environ.get("GLM_SWEEP_UNI_HP", "1") == "1"
DRAM_GBS, FLOP_CYC, N_COMPUTE, N_GRID = 512.0, 4096, 64 + 26, 110


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 0}], indirect=True)
def test_flat_balanced_sweep(device):
    from ttnn.bringup.flat_routed_expert_ttnn.flat_expert import FlatRoutedExpert

    torch.manual_seed(0)
    W = [[(torch.randn(H, I) * 0.02, torch.randn(H, I) * 0.02, torch.randn(I, H) * 0.02) for _ in range(E)]]
    rm = lambda t, d: ttnn.from_torch(  # noqa: E731
        t, dtype=d, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    if UNIFIED:  # the same bfp4 experts, the unified op's per-expert weight tensors
        to_dev = lambda w: ttnn.from_torch(  # noqa: E731
            w, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        uw = [[to_dev(W[0][e][i]) for e in range(E)] for i in range(3)]
        gidx = rm(torch.arange(E, dtype=torch.int32), ttnn.uint32)
        ucfg = ttnn.types.BlackholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
        )
    else:
        op = FlatRoutedExpert(
            device,
            W,
            m=CAP,
            H=H,
            I=I,
            gids=[list(range(E))],
            n_global=NG,
            wdtype="bf4",
            act=ACT,
            pin=1,
            x_bf16=XBF16,
            h_bf16=HBF16,
        )
    del W
    x = rm(torch.randn(T, H).to(torch.bfloat16) * 0.3, ttnn.bfloat16)

    rt = ttnn.device.IsProgramRealtimeProfilerActive()
    recs = {}

    def cb(batch):
        for r in batch.records:
            recs[r.runtime_id] = (r.end_timestamp - r.start_timestamp) / (r.frequency * 1e3), r.frequency  # µs, GHz

    hdl = ttnn.device.RegisterProgramRealtimeProfilerCallback(cb) if rt else None
    ms_list = [int(v) for v in os.environ["GLM_SWEEP_M"].split(",")] if os.environ.get("GLM_SWEEP_M") else MS
    # GLM_SWEEP_COUNTS=file, GLM_SWEEP_COUNTS_KEY=model: uneven routing ({model: {pattern: [count per expert]}});
    # M is then the pattern's mean count (the table's M_total its total)
    cases = [(M, [M] * E) for M in ms_list]
    if os.environ.get("GLM_SWEEP_COUNTS"):
        import json

        pats = json.load(open(os.environ["GLM_SWEEP_COUNTS"]))[os.environ["GLM_SWEEP_COUNTS_KEY"]]
        cases = [(name, cl) for name, cl in pats.items()]
    rows_out = []
    try:
        for M, cl in cases:
            pads = [(c + 31) // 32 * 32 for c in cl]
            counts = torch.zeros(1, NG, dtype=torch.int32)
            regions = torch.zeros(1, NG, dtype=torch.int32)
            counts[0, :E] = torch.tensor(cl, dtype=torch.int32)
            regions[0, :E] = torch.tensor([sum(pads[:e]) for e in range(E)], dtype=torch.int32)
            n_rows = max(sum(pads), 32)
            # (seeded per M: a run's inputs at M do not depend on its M list, so dumps of different runs compare)
            seed = M if isinstance(M, int) else sum(cl)
            tidx = torch.randint(0, T, (1, n_rows), dtype=torch.int32, generator=torch.Generator().manual_seed(seed))
            cd, rd, td = rm(counts, ttnn.uint32), rm(regions, ttnn.uint32), rm(tidx, ttnn.uint32)
            if UNIFIED:  # the dispatched buffer: the experts' rows back to back (>= one expert's capacity of rows)
                xb = rm(torch.randn(max(sum(cl), CAP), H).to(torch.bfloat16) * 0.3, ttnn.bfloat16)
                call = lambda: ttnn.bringup.unified_routed_expert_moe(  # noqa: E731
                    xb,
                    rd,
                    cd,
                    gidx,
                    uw[0],
                    uw[1],
                    uw[2],
                    max_dispatched_tokens_per_expert=CAP,
                    compute_kernel_config=ucfg,
                    activation=ttnn.bringup.RoutedExpertActivation.ClampedSiluGlu,
                    high_precision=UNI_HP,
                )
            else:
                call = lambda: op(x, cd, rd, token_index=td, y_row_major=YRM, down_fp32=DFP32)  # noqa: E731
            y0 = call()
            if os.environ.get("GLM_SWEEP_DUMP") and M == 128:  # y of M 128, to compare variants for equality
                torch.save(ttnn.to_torch(y0), os.environ["GLM_SWEEP_DUMP"])
            ttnn.deallocate(y0)
            ttnn.synchronize_device(device)
            ids = []
            t0 = time.time()
            for _ in range(ITERS):
                i0 = ttnn._ttnn.get_device_operation_id()
                ttnn.deallocate(call())
                ids.append((i0, ttnn._ttnn.get_device_operation_id()))
            ttnn.synchronize_device(device)
            wall = (time.time() - t0) / ITERS * 1e3
            dev, ghz = None, 1.35
            if rt:
                time.sleep(0.5)  # let the receiver thread deliver the last records
                per = [sum(recs[i][0] for i in range(a, b) if i in recs) for a, b in ids]
                hit = [i for a, b in ids for i in range(a, b) if i in recs]
                if hit:
                    dev, ghz = statistics.median(per) / 1e3, recs[hit[0]][1]
            ms = dev if dev is not None else wall
            rows = sum(cl)
            y_b = (2 if UNI_HP else 1088 / 1024) if UNIFIED else (2 if YRM else 1088 / 1024)  # y bytes per element
            dram = (W_BYTES + rows * H * (2 + y_b)) / (ms * 1e-3) / 1e9 / DRAM_GBS
            flops = rows * 6 * H * I
            peak_core = FLOP_CYC * ghz * 1e9
            mc = flops / (ms * 1e-3) / (peak_core * N_COMPUTE)
            mg = flops / (ms * 1e-3) / (peak_core * N_GRID)
            rows_out.append((M, rows, ms, wall, ms / E * 1e3, dram, mc, mg, ghz))
            print(
                f"[sweep] M {M!s:>5} rows {rows:6d}: dev {ms:8.3f} ms (wall {wall:8.3f}) {ms / E * 1e3:7.1f} us/expert "
                f"DRAM {dram:5.1%} math90 {mc:5.1%} math110 {mg:5.1%} @ {ghz:.3f} GHz",
                flush=True,
            )
            for t_ in (cd, rd, td):
                ttnn.deallocate(t_)
        if os.environ.get("TT_METAL_DEVICE_PROFILER"):
            ttnn.ReadDeviceProfiler(device)
    finally:
        if hdl is not None:
            ttnn.device.UnregisterProgramRealtimeProfilerCallback(hdl)
    print("[sweep] | M | M_total | total ms | us/expert | DRAM % | math % (90 compute) | math % (110 grid) |")
    for M, rows, ms, wall, pe, dram, mc, mg, _ in rows_out:
        print(f"[sweep] | {M} | {rows} | {ms:.3f} | {pe:.1f} | {dram:.0%} | {mc:.0%} | {mg:.0%} |")
