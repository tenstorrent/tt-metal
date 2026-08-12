import sys, os, shutil

sys.path.insert(
    0,
    "/localdev/mstaletovic/2026_08_11/1659_mstaletovic_agent_eval/clones/rms_norm_run1/tt-metal/tests/ttnn/unit_tests/operations/rms_norm",
)
import perf_zone_harness as h
import ttnn

device = ttnn.open_device(device_id=0)
LOG = "/localdev/mstaletovic/tt-metal/generated/profiler/.logs/profile_log_device.csv"
tag = os.environ.get("ZTAG", "x")
try:
    for name in os.environ.get("ZCASES", "decode7168").split(","):
        ns, _ = h.measure(device, name)
        print(f"ZONE_HARNESS {tag} {name}: ns={ns}")
        if os.path.exists(LOG):
            shutil.copyfile(LOG, f"/localdev/mstaletovic/tt-metal/generated/profiler/.logs/zones_{tag}_{name}.csv")
finally:
    ttnn.close_device(device)
