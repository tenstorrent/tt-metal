"""
Targeted sweep for 480p 2x4 BH LoudBox — new model T values (T_latent=16).

T=63 at full-res: 63 = 3*21 = 7*9, so T=3,7,9,21 divide cleanly.
Spatial dims per device: H=240, W=208 (up3/conv_out), H=120, W=104 (up2), H=60, W=52 (up1).
"""
from sweep_lib import make_device, make_input, prep_bias, prep_weights, sweep

device, GRID, CKC = make_device()


def s(name, x, w, b, cout, kernel, pad, ci, co, candidates):
    return sweep(name, device, GRID, CKC, x, w, b, cout, kernel, pad, ci, co, candidates)


import ttnn

# ── up3_res 96→96 T=63 H=240 W=208 ──────────────────────────────────────────
# 63 = 3×21 = 7×9. Best candidates: T=3,7,9,21 × H=4,8 × W=4,8
w = prep_weights(device, 96, 96, (3, 3, 3), 96)
b = prep_bias(device, 96)
x = make_input(device, 1, 63, 242, 210, 96)
s(
    "up3_res 96→96 T=63 H=240 W=208",
    x,
    w,
    b,
    96,
    (3, 3, 3),
    (0, 1, 1),
    96,
    96,
    [(1, 8, 8), (3, 8, 8), (3, 4, 8), (3, 8, 4), (7, 8, 8), (7, 4, 8), (9, 8, 8), (9, 4, 8), (21, 8, 8)],
)
ttnn.deallocate(x)
ttnn.deallocate(w)
ttnn.deallocate(b)

# ── conv_out 96→3 T=63 H=240 W=208 ──────────────────────────────────────────
# From 4x8 result: H=4 was critical (H=8 OOMed). Try T=3,7,9,21 × H=4 × W=4,8,16
w = prep_weights(device, 3, 96, (3, 3, 3), 96)
b = prep_bias(device, 3)
x = make_input(device, 1, 63, 242, 210, 96)
s(
    "conv_out 96→3 T=63 H=240 W=208",
    x,
    w,
    b,
    3,
    (3, 3, 3),
    (0, 1, 1),
    96,
    32,
    [(1, 4, 8), (3, 4, 8), (7, 4, 8), (9, 4, 8), (1, 8, 8), (3, 8, 8)],
)
ttnn.deallocate(x)
ttnn.deallocate(w)
ttnn.deallocate(b)

# ── up2_res 192→192 T=63 H=120 W=104 ─────────────────────────────────────────
w = prep_weights(device, 192, 192, (3, 3, 3), 96)
b = prep_bias(device, 192)
x = make_input(device, 1, 63, 122, 106, 192)
s(
    "up2_res 192→192 T=63 H=120 W=104",
    x,
    w,
    b,
    192,
    (3, 3, 3),
    (0, 1, 1),
    96,
    96,
    [(1, 8, 4), (3, 8, 4), (7, 8, 4), (9, 8, 4), (21, 8, 4), (3, 8, 8), (7, 8, 8)],
)
ttnn.deallocate(x)
ttnn.deallocate(w)
ttnn.deallocate(b)

# ── up1_res 384→384 T=33 H=60 W=52 ──────────────────────────────────────────
# T=33 = 3×11. Try T=3,11,33
w = prep_weights(device, 384, 384, (3, 3, 3), 96)
b = prep_bias(device, 384)
x = make_input(device, 1, 33, 62, 54, 384)
s(
    "up1_res 384→384 T=33 H=60 W=52",
    x,
    w,
    b,
    384,
    (3, 3, 3),
    (0, 1, 1),
    96,
    128,
    [(1, 8, 4), (3, 8, 4), (3, 4, 4), (11, 8, 4), (11, 4, 4)],
)
ttnn.deallocate(x)
ttnn.deallocate(w)
ttnn.deallocate(b)

ttnn.close_device(device)
