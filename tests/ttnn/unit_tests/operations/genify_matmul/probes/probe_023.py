import sys, torch, ttnn

sys.path.insert(0, ".")
import eval.golden_tests.genify_matmul.helpers as H
from eval.golden_tests.genify_matmul.feature_spec import LOOSE_CASES
import eval.metrics as M

captured = {}
orig = H.check_output


def spy(out, expected, **kw):
    m = M.compute_metrics(out, expected)
    captured["m"] = m
    return orig(out, expected, **kw)


H.check_output = spy
device = ttnn.open_device(device_id=0)
cases = [c for c in LOOSE_CASES if c["placement"] == "block_sharded"]
picks = [cases[0], cases[3], cases[9]]
for c in picks:
    H.run_genify_matmul(
        c["inputs"],
        dtype=c["dtype"],
        weight_dtype=c["weight_dtype"],
        layout=c["layout"],
        fp32_dest_acc_en=c["fp32_dest_acc_en"],
        placement=c["placement"],
        epilogue=c["epilogue"],
        device=device,
        extras=c["extras"],
    )
    print("CASE", c["inputs"], c["epilogue"], captured["m"])
H.run_genify_matmul(
    ((1, 384, 1536), (1536, 1024)),
    dtype=ttnn.bfloat16,
    weight_dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    fp32_dest_acc_en=True,
    placement="block_sharded",
    epilogue="none",
    device=device,
)
print("CASE cartesian 384x1536x1024 bf16 HiFi4", captured["m"])
ttnn.close_device(device)
