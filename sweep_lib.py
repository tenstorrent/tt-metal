"""
Shared conv3d sweep utilities.

bench():  runs a single config with thread-based hang timeout.
sweep():  runs a list of (T, H, W) candidates, flags hangs and slow-OOM anomalies.

Anomaly detection: if a config's per-run time exceeds ANOMALY_MULT × best_so_far,
it is flagged as "SLOW?" — this catches silent L1-OOM cases where the kernel
doesn't crash but runs ~10-100x slower (e.g. conv_out H=8 → 100ms vs H=4 → 7ms).

Hang detection: each candidate runs in a daemon thread.  If it does not finish
within TIMEOUT_S seconds the main thread continues and prints "HANG".  After a
hang os._exit(1) is called immediately to avoid blocking on device cleanup.
"""
import os
import threading
import time

import torch
import ttnn

from models.tt_dit.utils.conv3d import aligned_channels

TIMEOUT_S = 30  # seconds before declaring hang  (most configs finish <20s)
ANOMALY_MULT = 4  # flag if us > ANOMALY_MULT × best_so_far


def make_device(grid=(12, 10)):
    device = ttnn.open_device(device_id=0)
    grid_coord = ttnn.CoreCoord(*grid)
    ckc = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    return device, grid_coord, ckc


def prep_weights(device, cout, cin, kernel, cin_block):
    pc = aligned_channels(cin)
    tw = ttnn.from_torch(
        torch.randn(cout, pc, *kernel), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, pad_value=0
    )
    return ttnn.experimental.prepare_conv3d_weights(weight_tensor=tw, C_in_block=cin_block, device=device)


def prep_bias(device, cout):
    return ttnn.from_torch(
        torch.randn(1, cout), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, pad_value=0
    )


def make_input(device, B, T, H, W, C):
    return ttnn.from_torch(
        torch.randn(B, T, H, W, aligned_channels(C)), device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT
    )


def run_conv(device, grid, ckc, x, w, b, cout, kernel, pad, ci, co, t, h, ww):
    cfg = ttnn.Conv3dConfig(
        weights_dtype=ttnn.bfloat16,
        output_layout=ttnn.ROW_MAJOR_LAYOUT,
        T_out_block=t,
        W_out_block=ww,
        H_out_block=h,
        C_out_block=co,
        C_in_block=ci,
        compute_with_storage_grid_size=grid,
    )
    return ttnn.experimental.conv3d(
        input_tensor=x,
        weight_tensor=w,
        bias_tensor=b,
        config=cfg,
        output_channels=cout,
        kernel_size=kernel,
        stride=(1, 1, 1),
        padding=pad,
        padding_mode="zeros",
        dtype=ttnn.bfloat16,
        compute_kernel_config=ckc,
    )


def bench(device, grid, ckc, x, w, b, cout, kernel, pad, ci, co, t, h, ww, warmup=1, timed=2):
    """
    Returns (us_per_run, status) where status is 'ok', 'fail', or 'hang'.
    'hang': kernel did not complete within TIMEOUT_S seconds.
    'fail': exception raised (L1 OOM, invalid config, etc.).
    """
    result = [None]
    exc = [None]

    def _run():
        try:
            for _ in range(warmup):
                o = run_conv(device, grid, ckc, x, w, b, cout, kernel, pad, ci, co, t, h, ww)
                ttnn.synchronize_device(device)
                ttnn.deallocate(o)
            t0 = time.perf_counter()
            outs = [run_conv(device, grid, ckc, x, w, b, cout, kernel, pad, ci, co, t, h, ww) for _ in range(timed)]
            ttnn.synchronize_device(device)
            result[0] = (time.perf_counter() - t0) * 1e6 / timed
            for o in outs:
                ttnn.deallocate(o)
        except Exception as e:
            exc[0] = e

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    thread.join(TIMEOUT_S)

    if thread.is_alive():
        return None, "hang"
    if exc[0] is not None:
        return None, "fail"
    return result[0], "ok"


def sweep(name, device, grid, ckc, x, w, b, cout, kernel, pad, ci, co, candidates, warmup=1, timed=2):
    """
    Sweep (T, H_out_block, W_out_block) candidates.

    Prints each result immediately (flush=True).
    Flags SLOW? if us > ANOMALY_MULT × best_so_far (silent OOM indicator).
    Exits immediately via os._exit(1) on HANG to avoid blocking cleanup.

    Returns list of (T, H, W, us) for valid results.
    """
    print(f"\n=== {name} ===", flush=True)
    results = []
    best_us = float("inf")

    for t, h, ww in candidates:
        us, status = bench(device, grid, ckc, x, w, b, cout, kernel, pad, ci, co, t, h, ww, warmup=warmup, timed=timed)
        if status == "hang":
            print(f"  T={t:2d} H={h:2d} W={ww:2d}: HANG (>{TIMEOUT_S}s) — exiting", flush=True)
            print("  Run: tt-smi -r 0,1,2,3,4,5,6,7", flush=True)
            os._exit(1)
        elif status == "fail":
            print(f"  T={t:2d} H={h:2d} W={ww:2d}: FAIL", flush=True)
        else:
            anomaly = us > ANOMALY_MULT * best_us if best_us < float("inf") else False
            tag = " ← SLOW? (OOM?)" if anomaly else ""
            print(f"  T={t:2d} H={h:2d} W={ww:2d}: {us:7.0f} us{tag}", flush=True)
            if not anomaly:
                best_us = min(best_us, us)
                results.append((t, h, ww, us))

    if results:
        bt, bh, bw, bu = min(results, key=lambda x: x[3])
        print(f"  → Best: T={bt} H={bh} W={bw} ({bu:.0f} us)", flush=True)
    return results
