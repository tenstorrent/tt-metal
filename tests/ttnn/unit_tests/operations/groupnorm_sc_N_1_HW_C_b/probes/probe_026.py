import torch, ttnn
from eval.golden_tests.groupnorm_sc_N_1_HW_C_b.helpers import run_groupnorm_sc_N_1_HW_C
from ttnn.operations.groupnorm_sc_N_1_HW_C_b.groupnorm_sc_N_1_HW_C_b_program_descriptor import plan_or_raise

device = ttnn.open_device(device_id=0)
try:
    for inputs in [((1, 1, 64, 17), 1), ((1, 1, 128, 48), 3), ((1, 1, 64, 200), 8)]:
        for layout in (ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT):
            torch.manual_seed(0)
            try:
                run_groupnorm_sc_N_1_HW_C(
                    inputs,
                    device=device,
                    dtype=ttnn.bfloat16,
                    layout=layout,
                    affine="gamma_beta",
                    affine_dtype=ttnn.bfloat16,
                    affine_layout=ttnn.ROW_MAJOR_LAYOUT,
                )
                print("PASS", inputs, layout)
            except Exception as e:
                print("FAIL", inputs, layout, str(e)[:400])
finally:
    ttnn.close_device(device)
