import ttnn
from eval.golden_tests.scaled_dot_product_attention.helpers import run_scaled_dot_product_attention as run
from ttnn.operations.scaled_dot_product_attention import tag_alignment

# All golden non-aligned INPUTS (from feature_spec.py)
NONALIGNED = [
    ((1, 1, 32, 50), (1, 1, 32, 50), (1, 1, 32, 50)),
    ((1, 1, 47, 64), (1, 1, 47, 64), (1, 1, 47, 64)),
    ((1, 1, 50, 50), (1, 1, 50, 50), (1, 1, 50, 50)),
    ((1, 4, 47, 64), (1, 4, 47, 64), (1, 4, 47, 64)),
    ((2, 4, 100, 64), (2, 4, 100, 64), (2, 4, 100, 64)),
    ((1, 8, 64, 47), (1, 8, 64, 47), (1, 8, 64, 47)),
    ((1, 12, 33, 50), (1, 12, 33, 50), (1, 12, 33, 50)),
    ((1, 8, 47, 64), (1, 2, 47, 64), (1, 2, 47, 64)),
    ((1, 8, 47, 64), (1, 1, 47, 64), (1, 1, 47, 64)),
    ((1, 4, 100, 50), (1, 4, 47, 50), (1, 4, 47, 50)),
]

device = ttnn.open_device(device_id=0)
try:
    for acc in (True, False):
        for inp in NONALIGNED:
            q, k, _ = inp
            align = tag_alignment((list(q), list(k), list(k)), {})
            skv = k[-2]
            skv_pad = skv % 32
            for mask_mode in ("none",):
                try:
                    run(
                        inp,
                        device=device,
                        dtype=ttnn.bfloat8_b,
                        layout=ttnn.TILE_LAYOUT,
                        mask_mode=mask_mode,
                        scale_mode="auto",
                        fp32_dest_acc_en=acc,
                    )
                    st = "PASS"
                except Exception as e:
                    st = "FAIL " + str(e).replace("\n", " ")[str(e).find("pcc=") :][:60]
                print(f"acc={acc!s:5} align={align:14} Skv={skv:3}(pad{skv_pad:2}) Q={q} {st}")
finally:
    ttnn.close_device(device)
