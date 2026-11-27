# Verifier probe: widen SUPPORTED in-process to TARGET and grade every refused INPUTS / LOOSE_CASES
# cell with the golden harness, to learn whether the TARGET gap needs kernel work.
import importlib
import torch, ttnn

op = importlib.import_module("ttnn.operations.g6_clone.g6_clone")
from eval.golden_tests.g6_clone import feature_spec as fs
from eval.golden_tests.g6_clone.helpers import run_g6_clone

op.SUPPORTED["dtype"][:] = list(fs.TARGET["dtype"])
op.SUPPORTED["layout"][:] = list(fs.TARGET["layout"])

device = ttnn.open_device(device_id=0)
ok = bad = 0
try:
    cases = [(c, None) for c in fs.INPUTS] + [(lc["inputs"], lc["extras"]) for lc in fs.LOOSE_CASES]
    for inputs, extras in cases:
        sc = inputs[0]
        if sc["dtype"] == ttnn.float32:
            continue
        torch.manual_seed(0)
        try:
            run_g6_clone(inputs, dtype=sc["dtype"], layout=sc["layout"], device=device, extras=extras)
            ok += 1
            print("CELL PASS", sc["input_shape"], sc["dtype"], sc["layout"], (extras or {}).get("case", ""))
        except Exception as e:
            bad += 1
            print(
                "CELL FAIL", sc["input_shape"], sc["dtype"], sc["layout"], (extras or {}).get("case", ""), repr(e)[:300]
            )
finally:
    ttnn.close_device(device)
print("SUMMARY pass", ok, "fail", bad)
