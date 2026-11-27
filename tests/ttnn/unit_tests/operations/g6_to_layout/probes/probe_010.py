import torch, ttnn
from collections import Counter
from eval.golden_tests.g6_to_layout import helpers as H
from eval.golden_tests.g6_to_layout.feature_spec import scenario_of
from ttnn.operations.g6_to_layout import g6_to_layout

device = ttnn.open_device(device_id=0)
try:
    sc = scenario_of(29)
    x = H.make_torch_input(sc["shape"], {"input_gen": "bf16_specials", "seed": 1234})
    exp = H.host_bf8_rendering(x)
    t = ttnn.from_torch(
        x, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    y = ttnn.to_torch(g6_to_layout(t, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)).reshape(sc["shape"])
    eb, yb = exp.view(torch.int16).reshape(-1), y.view(torch.int16).reshape(-1)
    d = eb != yb
    print("mismatches", int(d.sum()))
    print(
        "pairs (expected, got)",
        Counter((hex(int(a) & 0xFFFF), hex(int(b) & 0xFFFF)) for a, b in zip(eb[d][:5000], yb[d][:5000])).most_common(
            8
        ),
    )
    print("expected NaN count", int(torch.isnan(exp).sum()))
finally:
    ttnn.close_device(device)
