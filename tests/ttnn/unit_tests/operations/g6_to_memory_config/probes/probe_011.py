# Verifier probe: bounce_depth sweep on Qwen 2183/2179 (gate lifted in-process): exactness + host wall-clock
# per call (20 back-to-back calls, one sync), independent of device-profiler marker capacity.
import importlib, time, torch
import ttnn
import ttnn.operations.g6_to_memory_config.g6_to_memory_config_program_descriptor as dm

opmod = importlib.import_module("ttnn.operations.g6_to_memory_config.g6_to_memory_config")
from eval.golden_tests.g6_to_memory_config.feature_spec import CASES
from eval.golden_tests.g6_to_memory_config.helpers import build_memory_config, make_torch_input

opmod.SUPPORTED["dtype"] = [ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat4_b]
opmod.EXCLUSIONS.clear()
device = ttnn.open_device(device_id=0)
try:
    for cid in (2183, 2179):
        s = CASES[("Qwen", cid)][0]
        x = ttnn.from_torch(
            make_torch_input(s["shape"], s["dtype"]),
            dtype=s["dtype"],
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=build_memory_config(s["in_memory"]),
        )
        ref = ttnn.to_torch(x)
        cfg = build_memory_config(s["out_memory"])
        for depth in (2, 4, 8, 16, 32):
            dm.BOUNCE_DEPTH = depth
            y = opmod.g6_to_memory_config(x, cfg)
            exact = torch.equal(ttnn.to_torch(y), ref)
            y.deallocate()
            ys = []
            ttnn.synchronize_device(device)
            t0 = time.perf_counter()
            for _ in range(20):
                ys.append(opmod.g6_to_memory_config(x, cfg))
            ttnn.synchronize_device(device)
            dt = (time.perf_counter() - t0) / 20 * 1e6
            for t in ys:
                t.deallocate()
            print(f"RES Q{cid} depth={depth} exact={exact} host_us_per_call={dt:.1f}")
finally:
    ttnn.close_device(device)
