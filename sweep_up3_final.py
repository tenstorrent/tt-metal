"""
up3_res / up2_res / conv_out / up1_res sweep with per-config timeout.

Each candidate runs in a thread; if it doesn't finish in TIMEOUT_S seconds
we mark it HANG and move on (device is reset at end if any hang occurred).
"""
import threading
import time

import torch
import ttnn

from models.tt_dit.utils.conv3d import aligned_channels

device = ttnn.open_device(device_id=0)
GRID = ttnn.CoreCoord(12, 10)
CKC = ttnn.init_device_compute_kernel_config(
    device.arch(),
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=False,
)
WARMUP = 1
TIMED = 2
TIMEOUT_S = 45  # seconds per candidate before declaring hang


def pw(cout, cin, k, cb):
    pc = aligned_channels(cin)
    tw = ttnn.from_torch(
        torch.randn(cout, pc, *k), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, pad_value=0
    )
    return ttnn.experimental.prepare_conv3d_weights(weight_tensor=tw, C_in_block=cb, device=device)


def pb(cout):
    return ttnn.from_torch(
        torch.randn(1, cout), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, pad_value=0
    )


def mk(B, T, H, W, C):
    return ttnn.from_torch(
        torch.randn(B, T, H, W, aligned_channels(C)), device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT
    )


def run_cfg(x, w, b, cout, k, pad, ci, co, t, h, ww):
    cfg = ttnn.Conv3dConfig(
        weights_dtype=ttnn.bfloat16,
        output_layout=ttnn.ROW_MAJOR_LAYOUT,
        T_out_block=t,
        W_out_block=ww,
        H_out_block=h,
        C_out_block=co,
        C_in_block=ci,
        compute_with_storage_grid_size=GRID,
    )
    return ttnn.experimental.conv3d(
        input_tensor=x,
        weight_tensor=w,
        bias_tensor=b,
        config=cfg,
        output_channels=cout,
        kernel_size=k,
        stride=(1, 1, 1),
        padding=pad,
        padding_mode="zeros",
        dtype=ttnn.bfloat16,
        compute_kernel_config=CKC,
    )


def bench(x, wt, b, cout, k, pad, ci, co, t, h, ww):
    """Run with thread-based timeout. Returns (us, status) where status is 'ok'/'fail'/'hang'."""
    result = [None]
    exc = [None]

    def _run():
        try:
            for _ in range(WARMUP):
                o = run_cfg(x, wt, b, cout, k, pad, ci, co, t, h, ww)
                ttnn.synchronize_device(device)
                ttnn.deallocate(o)
            t0 = time.perf_counter()
            outs = [run_cfg(x, wt, b, cout, k, pad, ci, co, t, h, ww) for _ in range(TIMED)]
            ttnn.synchronize_device(device)
            us = (time.perf_counter() - t0) * 1e6 / TIMED
            for o in outs:
                ttnn.deallocate(o)
            result[0] = us
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


def sweep(name, x, wt, b, cout, k, pad, ci, co, candidates):
    print(f"\n=== {name} ===", flush=True)
    results = []
    any_hang = False
    for t, h, ww in candidates:
        us, status = bench(x, wt, b, cout, k, pad, ci, co, t, h, ww)
        if status == "ok":
            print(f"  T={t:2d} H={h:2d} W={ww:2d}: {us:7.0f} us", flush=True)
            results.append((t, h, ww, us))
        elif status == "hang":
            print(f"  T={t:2d} H={h:2d} W={ww:2d}: HANG (>{TIMEOUT_S}s) — skipping rest", flush=True)
            any_hang = True
            break  # device state may be corrupted after hang; stop this sweep
        else:
            print(f"  T={t:2d} H={h:2d} W={ww:2d}: FAIL", flush=True)

    if results:
        bt, bh, bw, bu = min(results, key=lambda x: x[3])
        print(f"  → Best: T={bt} H={bh} W={bw} ({bu:.0f} us)", flush=True)
    if any_hang:
        print("  ⚠ Device hung — exiting immediately. Run: tt-smi -r 0,1,2,3,4,5,6,7", flush=True)
        import os

        os._exit(1)  # hard exit so daemon thread can't block on cleanup
    return any_hang


hang_detected = False

# ── up3_res ───────────────────────────────────────────────────────────────────
w = pw(96, 96, (3, 3, 3), 96)
b = pb(96)

x = mk(1, 62, 186, 162, 96)
h = sweep(
    "up3_res 96→96 T=62 H=184 W=160",
    x,
    w,
    b,
    96,
    (3, 3, 3),
    (0, 1, 1),
    96,
    96,
    [(1, 8, 8), (2, 8, 8), (2, 4, 8), (2, 8, 16), (3, 8, 8), (4, 8, 8), (4, 8, 4)],
)
hang_detected |= h
ttnn.deallocate(x)

x = mk(1, 66, 186, 162, 96)
h = sweep(
    "up3_res 96→96 T=66 H=184 W=160",
    x,
    w,
    b,
    96,
    (3, 3, 3),
    (0, 1, 1),
    96,
    96,
    [(1, 8, 8), (2, 8, 8), (3, 8, 8), (3, 4, 8), (3, 8, 16), (6, 8, 8)],
)
hang_detected |= h
ttnn.deallocate(x)
ttnn.deallocate(w)
ttnn.deallocate(b)

# ── up2_res T=66 ──────────────────────────────────────────────────────────────
w = pw(192, 192, (3, 3, 3), 96)
b = pb(192)
x = mk(1, 66, 94, 82, 192)
h = sweep(
    "up2_res 192→192 T=66 H=92 W=80",
    x,
    w,
    b,
    192,
    (3, 3, 3),
    (0, 1, 1),
    96,
    96,
    [(1, 8, 4), (3, 8, 4), (6, 8, 4), (11, 8, 4), (11, 8, 8)],
)
hang_detected |= h
ttnn.deallocate(x)
ttnn.deallocate(w)
ttnn.deallocate(b)

# ── conv_out spatial ──────────────────────────────────────────────────────────
w = pw(3, 96, (3, 3, 3), 96)
b = pb(3)
x = mk(1, 62, 186, 162, 96)
h = sweep(
    "conv_out 96→3 T=31 H=184 W=160",
    x,
    w,
    b,
    3,
    (3, 3, 3),
    (0, 1, 1),
    96,
    32,
    [(31, 8, 8), (31, 8, 16), (31, 4, 8), (31, 16, 8)],
)
hang_detected |= h
ttnn.deallocate(x)
x = mk(1, 66, 186, 162, 96)
h = sweep(
    "conv_out 96→3 T=33 H=184 W=160",
    x,
    w,
    b,
    3,
    (3, 3, 3),
    (0, 1, 1),
    96,
    32,
    [(33, 8, 8), (33, 8, 16), (33, 4, 8), (33, 16, 8)],
)
hang_detected |= h
ttnn.deallocate(x)
ttnn.deallocate(w)
ttnn.deallocate(b)

# ── up1_res T=34 with Cin=96 H=8 ─────────────────────────────────────────────
w = pw(384, 384, (3, 3, 3), 96)
b = pb(384)
x = mk(1, 34, 48, 42, 384)
h = sweep(
    "up1_res 384→384 T=34 H=46 W=40",
    x,
    w,
    b,
    384,
    (3, 3, 3),
    (0, 1, 1),
    96,
    128,
    [(1, 8, 4), (2, 8, 4), (1, 4, 4), (2, 4, 4)],
)
hang_detected |= h
ttnn.deallocate(x)
ttnn.deallocate(w)
ttnn.deallocate(b)

ttnn.close_device(device)

if hang_detected:
    print("\n⚠ Hang detected — run: tt-smi -r 0,1,2,3,4,5,6,7", flush=True)
