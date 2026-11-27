# Verifier probe: do the block formats + DRAM width-sharded targets run bit-exact on the Phase 0
# kernels when the support gate is lifted? Scopes the refinement only; SUPPORTED is untouched on disk.
import importlib
import ttnn

opmod = importlib.import_module("ttnn.operations.g6_to_memory_config.g6_to_memory_config")
from ttnn.operations.g6_to_memory_config import plan_of
from eval.golden_tests.g6_to_memory_config.feature_spec import INPUTS
from eval.golden_tests.g6_to_memory_config.helpers import (
    run_scenario,
    scenario_id,
    build_memory_config,
    make_torch_input,
)

opmod.SUPPORTED["dtype"] = [ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat4_b]
opmod.EXCLUSIONS.clear()

device = ttnn.open_device(device_id=0)
ok = bad = 0
try:
    for entry in INPUTS:
        s = entry[0]
        if s["dtype"] == ttnn.bfloat16:
            continue
        try:
            tt_in = ttnn.from_torch(
                make_torch_input(s["shape"], s["dtype"]),
                dtype=s["dtype"],
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=build_memory_config(s["in_memory"]),
            )
            p = plan_of(tt_in, build_memory_config(s["out_memory"]))
            print(
                "PLAN",
                scenario_id(entry),
                s["dtype"],
                p["regime"],
                "pages",
                p["num_pages"],
                "page_size",
                p["page_size"],
                "bb",
                p["bounce_block_pages"],
                "l1",
                p["l1_op_bytes"],
                "movers",
                len(p["movers"]),
            )
            tt_in.deallocate()
            run_scenario(s, device, dtype=s["dtype"], layout=ttnn.TILE_LAYOUT, seed=0, op=opmod.g6_to_memory_config)
            print("CASE PASS", scenario_id(entry))
            ok += 1
        except Exception as e:
            print("CASE FAIL", scenario_id(entry), repr(e)[:300])
            bad += 1
finally:
    print("SUMMARY ok", ok, "bad", bad)
    ttnn.close_device(device)
