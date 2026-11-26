import torch, ttnn
from eval.golden_tests.groupnorm_sc_N_1_HW_C_b.helpers import run_groupnorm_sc_N_1_HW_C

device = ttnn.open_device(device_id=0)
B8, BF16, F32 = ttnn.bfloat8_b, ttnn.bfloat16, ttnn.float32
TILE = ttnn.TILE_LAYOUT
cases = [
    (((1, 1, 256, 256), 8), B8, "no_affine", "none", "none"),
    (((1, 1, 256, 256), 8), B8, "gamma_beta", B8, TILE),
    (((1, 1, 256, 256), 8), BF16, "gamma_beta", B8, TILE),
    (((2, 1, 64, 288), 9), B8, "gamma_beta", BF16, TILE),
    (((1, 1, 100, 64), 2), B8, "gamma_beta", B8, TILE),
    (((1, 1, 64, 100), 2), B8, "gamma_beta", B8, TILE),
]
for inputs, dt, aff, adt, al in cases:
    torch.manual_seed(0)
    try:
        run_groupnorm_sc_N_1_HW_C(
            inputs, device=device, dtype=dt, layout=TILE, affine=aff, affine_dtype=adt, affine_layout=al
        )
        print("PASS", inputs, dt, aff, adt)
    except Exception as e:
        print("FAIL", inputs, dt, aff, adt, str(e)[:400])
ttnn.close_device(device)
