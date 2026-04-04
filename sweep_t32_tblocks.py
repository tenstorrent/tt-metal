"""
T_out_block sweep for the actual shapes provided by the user.

Mesh: bh_4x8 (h_factor=4, w_factor=8)
Shapes include +2 causal pad on T and +1 spatial pad each side on H,W for k333/k133.

Run: python sweep_t32_tblocks.py
"""

import time

import torch
import ttnn

from models.tt_dit.utils.conv3d import aligned_channels

torch.manual_seed(42)
device = ttnn.open_device(device_id=0)
GRID = ttnn.CoreCoord(12, 10)
CKC = ttnn.init_device_compute_kernel_config(
    device.arch(),
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=False,
)
WARMUP = 3
TIMED = 4
H_FACTOR, W_FACTOR = 4, 8


def prep_w(cout, cin, k, cb):
    pc = aligned_channels(cin)
    tw = ttnn.from_torch(
        torch.randn(cout, pc, *k), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, pad_value=0
    )
    return ttnn.experimental.prepare_conv3d_weights(weight_tensor=tw, C_in_block=cb, device=device)


def prep_b(cout):
    return ttnn.from_torch(
        torch.randn(1, cout), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, pad_value=0
    )


def mk(shape):
    B, T, H, W, C = shape
    return ttnn.from_torch(
        torch.randn(B, T, H, W, aligned_channels(C)), device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT
    )


def run(x, w, b, cout, k, pad, cin_b, cout_b, t_b, h_b, w_b):
    cfg = ttnn.Conv3dConfig(
        weights_dtype=ttnn.bfloat16,
        output_layout=ttnn.ROW_MAJOR_LAYOUT,
        T_out_block=t_b,
        W_out_block=w_b,
        H_out_block=h_b,
        C_out_block=cout_b,
        C_in_block=cin_b,
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


def sweep(name, shape, cout, k, pad, cin_b, cout_b, h_b, w_b, t_candidates, current_t=None):
    x = mk(shape)
    _, T, H, W, Cin = shape
    H_out = H - (k[1] - 1)
    W_out = W - (k[2] - 1)
    w_tensor = prep_w(cout, Cin, k, cin_b)
    b_tensor = prep_b(cout)

    label = f"T={T} H={H_out} W={W_out} Cin={Cin}→{cout} {k}"
    cur_str = f" [current T_out_block={current_t}]" if current_t else ""
    print(f"\n{name}{cur_str}  {label}")

    results = []
    for t_b in t_candidates:
        try:
            for _ in range(WARMUP):
                out = run(x, w_tensor, b_tensor, cout, k, pad, cin_b, cout_b, t_b, h_b, w_b)
                ttnn.synchronize_device(device)
                ttnn.deallocate(out)
            outs = [run(x, w_tensor, b_tensor, cout, k, pad, cin_b, cout_b, t_b, h_b, w_b) for _ in range(TIMED)]
            ttnn.synchronize_device(device)
            for o in outs:
                ttnn.deallocate(o)
            t0 = time.perf_counter()
            outs2 = [run(x, w_tensor, b_tensor, cout, k, pad, cin_b, cout_b, t_b, h_b, w_b) for _ in range(TIMED)]
            ttnn.synchronize_device(device)
            us = (time.perf_counter() - t0) * 1e6 / TIMED
            for o in outs2:
                ttnn.deallocate(o)
            results.append((t_b, us))
        except Exception as e:
            results.append((t_b, float("inf")))
            print(f"  T_out_block={t_b:2d}: FAIL ({e})")
            continue

    for i, (t_b, us) in enumerate(results):
        if us == float("inf"):
            continue
        best = " ← best" if us == min(u for _, u in results if u != float("inf")) else ""
        cur = " ← current" if t_b == current_t else ""
        print(f"  T_out_block={t_b:2d}: {us:7.0f} us{best}{cur}")

    valid = [(t, u) for t, u in results if u != float("inf")]
    if valid:
        best_t, best_us = min(valid, key=lambda r: r[1])
        print(f"  → Best: T_out_block={best_t} ({best_us:.0f} us)")

    ttnn.deallocate(x)
    ttnn.deallocate(w_tensor)
    ttnn.deallocate(b_tensor)


print("=" * 70)
print(f"T_out_block sweep — actual model shapes (h={H_FACTOR}, w={W_FACTOR})")
print("=" * 70)

# ── Latent stage shapes ───────────────────────────────────────────────────────
# [1, 18, 25, 22, 32] k333 → conv_in
sweep("conv_in", (1, 18, 25, 22, 32), 384, (3, 3, 3), (0, 1, 1), 32, 128, 16, 16, [1, 2, 3, 6], current_t=1)

# [1, 17, 25, 22, 32] k333 → appears to be conv_in for different T (cached?)
sweep("conv_in_T17", (1, 17, 25, 22, 32), 384, (3, 3, 3), (0, 1, 1), 32, 128, 16, 16, [1, 2, 3], current_t=1)

# [1, 17, 25, 22, 384] k333 → lat_res (384→384 at latent)
sweep("lat_res_T17", (1, 17, 25, 22, 384), 384, (3, 3, 3), (0, 1, 1), 128, 128, 4, 4, [1, 2], current_t=1)

# [1, 18, 25, 22, 384] k333 → lat_res uncached
sweep("lat_res_T18", (1, 18, 25, 22, 384), 384, (3, 3, 3), (0, 1, 1), 128, 128, 4, 4, [1, 2, 3], current_t=1)

# [1, 17, 23, 20, 384] k311 → up0_tconv
sweep("up0_tconv_T17", (1, 17, 23, 20, 384), 768, (3, 1, 1), (0, 0, 0), 192, 256, 4, 4, [1, 2], current_t=1)

# [1, 18, 23, 20, 384] k311 → up0_tconv uncached
sweep("up0_tconv_T18", (1, 18, 23, 20, 384), 768, (3, 1, 1), (0, 0, 0), 192, 256, 4, 4, [1, 2, 3], current_t=1)

# ── Mid stage shapes ──────────────────────────────────────────────────────────
# [1, 3, 48, 42, 192] k333 → up1_res0 (192→384) cached
sweep("up1_res0_T3", (1, 3, 48, 42, 192), 384, (3, 3, 3), (0, 1, 1), 96, 128, 8, 4, [1, 3], current_t=1)

# [1, 34, 48, 42, 192] k333 → up1_res0 uncached
sweep("up1_res0_T34", (1, 34, 48, 42, 192), 384, (3, 3, 3), (0, 1, 1), 96, 128, 8, 4, [1, 2, 3, 17], current_t=1)

# [1, 3, 48, 42, 384] k333 → up1_res (384→384) cached
sweep("up1_res_T3", (1, 3, 48, 42, 384), 384, (3, 3, 3), (0, 1, 1), 128, 128, 8, 4, [1, 3], current_t=1)

# [1, 32, 48, 42, 384] k333 → up1_res uncached
sweep("up1_res_T32", (1, 32, 48, 42, 384), 384, (3, 3, 3), (0, 1, 1), 128, 128, 8, 4, [1, 2, 4, 8, 16, 32], current_t=1)

# [1, 34, 48, 42, 384] k333 → up1_res uncached variant
sweep("up1_res_T34", (1, 34, 48, 42, 384), 384, (3, 3, 3), (0, 1, 1), 128, 128, 8, 4, [1, 2, 17], current_t=1)

# [1, 32, 46, 40, 384] k311 → up1_tconv
sweep("up1_tconv_T32", (1, 32, 46, 40, 384), 768, (3, 1, 1), (0, 0, 0), 192, 384, 16, 4, [1, 2, 4, 8, 16], current_t=1)

# [1, 34, 46, 40, 384] k311 → up1_tconv uncached variant
sweep("up1_tconv_T34", (1, 34, 46, 40, 384), 768, (3, 1, 1), (0, 0, 0), 192, 384, 16, 4, [1, 2, 17], current_t=1)

# ── High-res stage shapes ─────────────────────────────────────────────────────
# [1, 3, 94, 82, 192] k333 → up2_res cached
sweep("up2_res_T3", (1, 3, 94, 82, 192), 192, (3, 3, 3), (0, 1, 1), 96, 96, 8, 4, [1, 3], current_t=1)

# [1, 62, 94, 82, 192] k333 → up2_res uncached (T=62 = 2*(31-1)+1+2−1?)
sweep("up2_res_T62", (1, 62, 94, 82, 192), 192, (3, 3, 3), (0, 1, 1), 96, 96, 8, 4, [1, 2, 31, 62], current_t=1)

# [1, 66, 94, 82, 192] k333 → up2_res uncached variant
sweep("up2_res_T66", (1, 66, 94, 82, 192), 192, (3, 3, 3), (0, 1, 1), 96, 96, 8, 4, [1, 2, 3, 6, 11, 33], current_t=1)

# ── Full-res stage shapes ─────────────────────────────────────────────────────
# [1, 3, 186, 162, 96] k333 → up3_res cached
sweep("up3_res_T3", (1, 3, 186, 162, 96), 96, (3, 3, 3), (0, 1, 1), 96, 96, 4, 8, [1, 3], current_t=1)

# [1, 62, 186, 162, 96] k333 → up3_res uncached
sweep("up3_res_T62", (1, 62, 186, 162, 96), 96, (3, 3, 3), (0, 1, 1), 96, 96, 4, 8, [1, 2, 31, 62], current_t=1)

# [1, 66, 186, 162, 96] k333 → up3_res uncached variant
sweep("up3_res_T66", (1, 66, 186, 162, 96), 96, (3, 3, 3), (0, 1, 1), 96, 96, 4, 8, [1, 2, 3, 6, 11, 33], current_t=1)

# [1, 3, 186, 162, 32] k333 → conv_out cached (note Cin=32 padded)
sweep("conv_out_T3", (1, 3, 186, 162, 32), 96, (3, 3, 3), (0, 1, 1), 96, 32, 4, 8, [1, 3], current_t=1)

# [1, 62, 186, 162, 96] k333 → conv_out uncached (same shape as up3_res but different cout)
sweep("conv_out_T62", (1, 62, 186, 162, 96), 3, (3, 3, 3), (0, 1, 1), 96, 32, 4, 8, [1, 2, 31, 62], current_t=1)

# [1, 66, 186, 162, 96] k333 → conv_out uncached variant
sweep("conv_out_T66", (1, 66, 186, 162, 96), 3, (3, 3, 3), (0, 1, 1), 96, 32, 4, 8, [1, 2, 3, 6, 11, 33], current_t=1)

ttnn.close_device(device)
