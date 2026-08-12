import sys

sys.path.insert(
    0,
    "/localdev/mstaletovic/2026_08_11/1659_mstaletovic_agent_eval/clones/rms_norm_run1/tt-metal/tests/ttnn/unit_tests/operations/rms_norm",
)
import perf_zone_harness as h
import ttnn

device = ttnn.open_device(device_id=0)
try:
    for name in ["decode7168", "prefill7168", "bshard1024", "wshard7168"]:
        ns, data = h.measure(device, name)
        if name == "decode7168":
            print("KEYS", type(data), (list(data[0].keys()) if data and isinstance(data[0], dict) else data)[:40])
        print(f"ZONE_HARNESS {name}: ns={ns}")
        import shutil, os

        src = "/localdev/mstaletovic/tt-metal/generated/profiler/.logs/profile_log_device.csv"
        if os.path.exists(src):
            shutil.copyfile(src, f"/localdev/mstaletovic/tt-metal/generated/profiler/.logs/zones_{name}.csv")
            print(f"ZONE_HARNESS {name}: zones saved {os.path.getsize(src)}")
finally:
    ttnn.close_device(device)
