"""
T_out_block sweep for the actual shapes provided by the user.

Mesh: bh_4x8 (h_factor=4, w_factor=8)
Shapes include +2 causal pad on T and +1 spatial pad each side on H,W for k333/k133.

Uses sweep_lib for hang detection (TIMEOUT_S=30s) and anomaly detection
(flags configs >4× faster than best as likely silent OOM).

Run: python sweep_t32_tblocks.py
"""

import torch
import ttnn

from models.tt_dit.utils.conv3d import aligned_channels
from sweep_lib import ANOMALY_MULT, TIMEOUT_S, bench, make_device

torch.manual_seed(42)
device, GRID, CKC = make_device()
H_FACTOR, W_FACTOR = 4, 8


from sweep_lib import make_input, prep_bias, prep_weights


def sweep(name, shape, cout, k, pad, cin_b, cout_b, h_b, w_b, t_candidates, current_t=None):
    """Sweep T_out_block values for a given layer shape.

    Uses sweep_lib.bench for hang detection (TIMEOUT_S) and flags anomalously
    slow configs (>ANOMALY_MULT× best) as likely silent OOMs.
    """
    B, T, H, W, Cin = shape
    H_out = H - (k[1] - 1)
    W_out = W - (k[2] - 1)
    x = make_input(device, B, T, H, W, Cin)
    w_tensor = prep_weights(device, cout, Cin, k, cin_b)
    b_tensor = prep_bias(device, cout)

    cur_str = f" [current T={current_t}]" if current_t else ""
    print(f"\n{name}{cur_str}  T={T} H={H_out} W={W_out} Cin={Cin}→{cout} {k}", flush=True)

    results = []
    best_us = float("inf")
    for t_b in t_candidates:
        us, status = bench(
            device, GRID, CKC, x, w_tensor, b_tensor, cout, k, pad, cin_b, cout_b, t_b, h_b, w_b, warmup=1, timed=2
        )
        if status == "hang":
            print(f"  T={t_b:2d}: HANG (>{TIMEOUT_S}s) — exiting. Run: tt-smi -r 0,1,2,3,4,5,6,7", flush=True)
            import os

            os._exit(1)
        elif status == "fail":
            print(f"  T={t_b:2d}: FAIL", flush=True)
        else:
            anomaly = us > ANOMALY_MULT * best_us if best_us < float("inf") else False
            cur = " ← current" if t_b == current_t else ""
            tag = " SLOW?(OOM)" if anomaly else ""
            print(f"  T={t_b:2d}: {us:7.0f} us{tag}{cur}", flush=True)
            if not anomaly:
                best_us = min(best_us, us)
                results.append((t_b, us))

    if results:
        best_t, best_u = min(results, key=lambda r: r[1])
        print(f"  → Best: T={best_t} ({best_u:.0f} us)", flush=True)

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
