import math, torch, ttnn
import eval.golden_tests.genify_matmul.helpers as H
from eval.metrics import compute_metrics
from eval.golden_tests.genify_matmul.feature_spec import FEATURE_CASES

device = ttnn.open_device(device_id=0)
orig = H.check_output


def spy(out_t, exp, **kw):
    m = compute_metrics(out_t, exp)
    print(f"CASE {spy.name} chunk pcc={m.pcc:.6f} rms={m.rms:.5f}")
    return orig(out_t, exp, **kw)


H.check_output = spy
for c in FEATURE_CASES:
    ex = c["extras"]
    if ex.get("chunks", 1) == 1:
        continue
    spy.name = f"{c['inputs']} x{ex['chunks']}"
    H.run_genify_matmul(
        c["inputs"],
        dtype=c["dtype"],
        weight_dtype=c["weight_dtype"],
        layout=c["layout"],
        fp32_dest_acc_en=c["fp32_dest_acc_en"],
        placement=c["placement"],
        epilogue=c["epilogue"],
        device=device,
        extras=ex,
    )
ttnn.close_device(device)
