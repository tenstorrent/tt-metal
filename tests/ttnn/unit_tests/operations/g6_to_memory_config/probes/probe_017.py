import torch, ttnn
from eval.golden_tests.g6_to_memory_config.feature_spec import CASES
from eval.golden_tests.g6_to_memory_config.helpers import build_memory_config
from ttnn.operations.g6_to_memory_config import plan_of

device = ttnn.open_device(device_id=0)
try:
    for n in (2178, 2179, 2182, 2183, 2184):
        sc = CASES[("Qwen", n)][0]
        t = ttnn.from_torch(
            torch.zeros(sc["shape"]),
            dtype=sc["dtype"],
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=build_memory_config(sc["in_memory"]),
        )
        p = plan_of(t, build_memory_config(sc["out_memory"]))
        print(
            f"PLAN {n} regime={p['regime']} depth={p['bounce_depth']} block={p['bounce_block_pages']} stage={p['bounce_stage_pages']} cb={p['cb_bounce_pages']} l1={p['l1_op_bytes']}"
        )
finally:
    ttnn.close_device(device)
