import sys, torch, ttnn

sys.path.insert(0, "tests/ttnn/unit_tests/operations/g6_reshard")
import test_g6_reshard as T
from ttnn.operations.g6_reshard import g6_reshard, describe_plan

device = ttnn.open_device(device_id=0)
try:
    x, _, cfg = T._place_input(device, "P")
    d = describe_plan(x, cfg)
    runs = [i["runs"] for t in d["targets"].values() for i in t["issuers"]]
    print(
        "PLAN",
        d["regime"],
        d["kernel_core_ranges"],
        "targets",
        len(d["targets"]),
        "runs",
        sum(runs),
        "source_only",
        len(d["source_only"]),
    )
    for name, fn in T.BAD_MEMORY_CONFIGS.items():
        try:
            g6_reshard(x, fn(device))
            print("CASE", name, "NO ERROR")
        except Exception as e:
            print("CASE", name, type(e).__name__, str(e).splitlines()[0][:140])
    for name, fn in T.BAD_INPUTS.items():
        try:
            g6_reshard(fn(device), cfg)
            print("CASE", name, "NO ERROR")
        except Exception as e:
            print("CASE", name, type(e).__name__, str(e).splitlines()[0][:140])
finally:
    ttnn.close_device(device)
