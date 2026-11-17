import torch, ttnn
import eval.golden_tests.genify_matmul.helpers as H
from eval import metrics as M

# capture metrics from check_output by wrapping compute_metrics
seen = []
_orig = H.check_output


def wrap(out, expected, **kw):
    m = M.compute_metrics(out, expected)
    seen.append((m, kw["tolerance"]))
    return _orig(out, expected, **kw)


H.check_output = wrap
from eval.golden_tests.genify_matmul.feature_spec import LOOSE_CASES

dev = ttnn.open_device(device_id=0)
try:
    cases = []
    for c in LOOSE_CASES:
        if c["placement"] != "interleaved" or c["epilogue"] == "none":
            continue
        cases.append(c)
    for c in cases[:13]:
        ex = c["extras"]
        H.run_genify_matmul(
            c["inputs"],
            dtype=c["dtype"],
            weight_dtype=c["weight_dtype"],
            layout=c["layout"],
            fp32_dest_acc_en=c["fp32_dest_acc_en"],
            placement=c["placement"],
            epilogue=c["epilogue"],
            device=dev,
            extras=ex,
        )
        m, tol = seen[-1]
        print("ACC", c["inputs"], c["epilogue"], f"pcc={m.pcc:.6f} rms={m.rms:.5f} tol={tol}")
    for ep in ["bias", "bias_relu", "swiglu", "addcmul"]:
        for dt, wdt, acc in [
            (ttnn.float32, ttnn.bfloat16, True),
            (ttnn.bfloat8_b, ttnn.bfloat8_b, False),
            (ttnn.bfloat16, ttnn.bfloat4_b, True),
        ]:
            H.run_genify_matmul(
                ((1, 384, 1536), (1536, 1024)),
                dtype=dt,
                weight_dtype=wdt,
                layout=ttnn.TILE_LAYOUT,
                fp32_dest_acc_en=acc,
                placement="interleaved",
                epilogue=ep,
                device=dev,
            )
            m, tol = seen[-1]
            print("ACC", "384x1536x1024", ep, dt, wdt, acc, f"pcc={m.pcc:.6f} rms={m.rms:.5f} tol={tol}")
finally:
    ttnn.close_device(dev)
