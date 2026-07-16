import ttnn
from eval.golden_tests.scaled_dot_product_attention.helpers import run_scaled_dot_product_attention as run

CELLS = [
    ("fp32_True_self_none", ((1, 2, 128, 64), (1, 2, 128, 64), (1, 2, 128, 64)), ttnn.float32, True, "none", "auto"),
    (
        "fp32_True_self_custom",
        ((1, 2, 128, 64), (1, 2, 128, 64), (1, 2, 128, 64)),
        ttnn.float32,
        True,
        "custom",
        "auto",
    ),
    ("bf8b_True_self_none", ((1, 2, 128, 64), (1, 2, 128, 64), (1, 2, 128, 64)), ttnn.bfloat8_b, True, "none", "auto"),
    (
        "bf8b_True_self_custom",
        ((1, 2, 128, 64), (1, 2, 128, 64), (1, 2, 128, 64)),
        ttnn.bfloat8_b,
        True,
        "custom",
        "auto",
    ),
    (
        "bf8b_False_self_none",
        ((1, 2, 128, 64), (1, 2, 128, 64), (1, 2, 128, 64)),
        ttnn.bfloat8_b,
        False,
        "none",
        "auto",
    ),
    ("bf16_False_self_none", ((1, 2, 128, 64), (1, 2, 128, 64), (1, 2, 128, 64)), ttnn.bfloat16, False, "none", "auto"),
    ("bf8b_True_wnonalign", ((1, 1, 64, 50), (1, 1, 64, 50), (1, 1, 64, 50)), ttnn.bfloat8_b, True, "none", "auto"),
    ("bf8b_True_hnonalign", ((1, 1, 100, 64), (1, 1, 100, 64), (1, 1, 100, 64)), ttnn.bfloat8_b, True, "none", "auto"),
]

for label, inputs, dtype, acc, mask_mode, scale_mode in CELLS:
    try:
        run(
            inputs,
            device=device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            mask_mode=mask_mode,
            scale_mode=scale_mode,
            fp32_dest_acc_en=acc,
        )
        print(f"PASS  {label}")
    except Exception as e:
        msg = str(e).replace("\n", " ")[:220]
        print(f"FAIL  {label}: {type(e).__name__}: {msg}")
