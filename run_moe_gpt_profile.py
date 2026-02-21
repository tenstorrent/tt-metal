"""Quick profiling script for moe_gpt - run under tracy."""
import sys
import ttnn
import torch

sys.path.insert(0, "/localdev/handrews/tt-metal")
from tests.ttnn.nightly.unit_tests.operations.experimental.test_moe_gpt import run_test_moe_gpt

device = ttnn.open_device(
    device_id=0,
    dispatch_core_config=ttnn.DispatchCoreConfig(ttnn.DispatchCoreType.WORKER, ttnn.DispatchCoreAxis.ROW),
)

metrics = run_test_moe_gpt(device, M=32, K=2880, N=2880, E=4, L=1, check_accuracy=True, dump_outputs=False)

pccs = [m["pcc"] for m in metrics.values()]
status = "PASS" if all(p > 0.984 for p in pccs) else "FAIL"
print(f"Result: {status} | " + " ".join(f"E{i}={p:.4f}" for i, p in enumerate(pccs)))

ttnn.close_device(device)
