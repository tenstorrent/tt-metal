import torch, ttnn
from eval.golden_tests.g6_to_memory_config.feature_spec import INPUTS
from eval.golden_tests.g6_to_memory_config.helpers import build_memory_config, scenario_id
from ttnn.operations.g6_to_memory_config import plan_of
from ttnn.operations.g6_to_memory_config import g6_to_memory_config_program_descriptor as d

device = ttnn.open_device(device_id=0)
worst = 0
try:
    for inp in INPUTS:
        s = inp[0]
        om = s["out_memory"]
        if s["dtype"] == ttnn.bfloat16 and not (om["scheme"] == "width_sharded" and om["buffer"] == "dram"):
            continue
        x = ttnn.from_torch(
            torch.zeros(s["shape"]),
            dtype=s["dtype"],
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=build_memory_config(s["in_memory"]),
        )
        p = plan_of(x, build_memory_config(om))
        worst = max(worst, p["l1_op_bytes"])
        print(
            f"CELL {scenario_id((s,))} {s['dtype']} regime={p['regime']} movers={len(p['movers'])} page={p['page_size']} "
            f"block={p['bounce_block_pages']} l1_op={p['l1_op_bytes']}"
        )
        ttnn.deallocate(x)
    print(f"MAX l1_op_bytes={worst} budget={d.L1_OP_BUDGET_BYTES}")
finally:
    ttnn.close_device(device)
