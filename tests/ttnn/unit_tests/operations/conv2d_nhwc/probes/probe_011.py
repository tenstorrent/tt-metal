import ttnn, torch
import ttnn.operations.conv2d_nhwc.conv2d_nhwc as op
from eval.golden_tests.conv2d_nhwc.helpers import run_conv2d_nhwc
from eval.golden_tests.conv2d_nhwc.feature_spec import LOOSE_CASES

orig = op.create_program_descriptor


def spy(device, i, w, b, o, plan, grid):
    print(
        "PLAN",
        plan["memory_layout"] if "memory_layout" in plan else "R0",
        "direct",
        plan.get("direct_in0"),
        "bm,bk,bn",
        plan["block_m_tiles"],
        plan["block_k_tiles"],
        plan["block_n_tiles"],
        "win_rows",
        plan.get("win_tile_rows"),
        "cm",
        plan["core_m_tiles"],
        "fp",
        plan["footprint"],
    )
    return orig(device, i, w, b, o, plan, grid)


op.create_program_descriptor = spy
device = ttnn.open_device(device_id=0, l1_small_size=16384)
TL, HS, BS, WS, IL = (
    ttnn.TILE_LAYOUT,
    ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    ttnn.TensorMemoryLayout.BLOCK_SHARDED,
    ttnn.TensorMemoryLayout.WIDTH_SHARDED,
    ttnn.TensorMemoryLayout.INTERLEAVED,
)
cases = (
    [((2, 28, 28, 64), 256, 1, 0, 1, 1, 1), HS],
    [((1, 14, 14, 1024), 256, 1, 0, 1, 1, 1), BS],
    [((2, 28, 28, 64), 64, 3, 1, 1, 1, 1), HS],
    [((1, 32, 32, 320), 320, 3, 1, 2, 1, 1), IL],
)
for inp, ml in cases:
    run_conv2d_nhwc(
        inp,
        dtype=ttnn.bfloat8_b,
        weight_dtype=ttnn.bfloat8_b,
        layout=TL,
        bias_mode="with_bias",
        memory_layout=ml,
        device=device,
    )
for c in LOOSE_CASES:
    if len(c["inputs"]) > 8 and c["inputs"][8] == TL and c["inputs"][0][3] % 32 == 0:
        run_conv2d_nhwc(
            c["inputs"],
            dtype=c["dtype"],
            weight_dtype=c["weight_dtype"],
            layout=TL,
            bias_mode="with_bias",
            memory_layout=c["inputs"][7],
            activation=c["inputs"][9] if len(c["inputs"]) > 9 else "none",
            device=device,
            extras=c["extras"],
        )
ttnn.close_device(device)
