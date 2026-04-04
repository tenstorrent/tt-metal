"""up3_res / up2_res / conv_out / up1_res sweep using sweep_lib."""
from sweep_lib import bench, make_device, make_input, prep_bias, prep_weights, sweep

device, GRID, CKC = make_device()


def s(name, x, w, b, cout, kernel, pad, ci, co, candidates):
    return sweep(name, device, GRID, CKC, x, w, b, cout, kernel, pad, ci, co, candidates)


# ── up3_res ───────────────────────────────────────────────────────────────────
w = prep_weights(device, 96, 96, (3, 3, 3), 96)
b = prep_bias(device, 96)

x = make_input(device, 1, 62, 186, 162, 96)
s(
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

import ttnn

ttnn.deallocate(x)

x = make_input(device, 1, 66, 186, 162, 96)
s(
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
ttnn.deallocate(x)
ttnn.deallocate(w)
ttnn.deallocate(b)

# ── up2_res T=66 ──────────────────────────────────────────────────────────────
w = prep_weights(device, 192, 192, (3, 3, 3), 96)
b = prep_bias(device, 192)
x = make_input(device, 1, 66, 94, 82, 192)
s(
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
ttnn.deallocate(x)
ttnn.deallocate(w)
ttnn.deallocate(b)

# ── conv_out spatial ──────────────────────────────────────────────────────────
w = prep_weights(device, 3, 96, (3, 3, 3), 96)
b = prep_bias(device, 3)
x = make_input(device, 1, 62, 186, 162, 96)
s(
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
ttnn.deallocate(x)
x = make_input(device, 1, 66, 186, 162, 96)
s(
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
ttnn.deallocate(x)
ttnn.deallocate(w)
ttnn.deallocate(b)

# ── up1_res T=34 ──────────────────────────────────────────────────────────────
w = prep_weights(device, 384, 384, (3, 3, 3), 96)
b = prep_bias(device, 384)
x = make_input(device, 1, 34, 48, 42, 384)
s(
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
ttnn.deallocate(x)
ttnn.deallocate(w)
ttnn.deallocate(b)

ttnn.close_device(device)
