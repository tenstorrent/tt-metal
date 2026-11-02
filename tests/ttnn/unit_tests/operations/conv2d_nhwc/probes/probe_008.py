import torch, ttnn, sys, os

sys.path.insert(0, ".")
import eval.golden_tests.conv2d_nhwc.helpers as H
from eval.golden_tests.conv2d_nhwc.feature_spec import INPUTS
import importlib

OP = importlib.import_module("ttnn.operations.conv2d_nhwc.conv2d_nhwc")
import ttnn.operations.conv2d_nhwc.conv2d_nhwc_program_descriptor as PD

res = {}


def fake_check(out, expected, **kw):
    got = ttnn.to_torch(out).float().reshape(expected.shape)
    e = expected.float()
    pcc = torch.corrcoef(torch.stack([got.flatten(), e.flatten()]))[0, 1].item()
    rms = ((got - e).pow(2).mean().sqrt() / e.pow(2).mean().sqrt()).item()
    res["m"] = (pcc, rms)


H.check_output = fake_check
WD = [ttnn.bfloat16, ttnn.bfloat8_b]
dev = ttnn.open_device(device_id=0, l1_small_size=16384)
worst = (1, 0, None)
try:
    for inp in INPUTS:
        axes = {"dtype": ttnn.bfloat16}
        for n, t in OP.INPUT_TAGGERS.items():
            axes[n] = t(inp, axes)
        if any(
            axes[a] not in OP.SUPPORTED[a]
            for a in ("channel_alignment", "groups_mode", "dilation_mode", "memory_layout", "layout", "activation")
        ):
            continue
        for wd in WD:
            res.clear()
            try:
                H.run_conv2d_nhwc(
                    inp,
                    dtype=ttnn.bfloat16,
                    weight_dtype=wd,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    bias_mode="with_bias",
                    memory_layout=axes["memory_layout"],
                    activation=axes["activation"],
                    device=dev,
                )
                p, r = res["m"]
                flag = "BAD" if (p < 0.995 or r > 0.05) else ""
                print(f"SW {inp} {wd} pcc={p:.5f} rms={r:.4f} {flag}", flush=True)
                if r > worst[1]:
                    worst = (p, r, (inp, wd))
            except Exception as ex:
                print(f"SW {inp} {wd} ERR {type(ex).__name__}: {str(ex)[:150]}", flush=True)
finally:
    print("WORST", worst)
    ttnn.close_device(dev)
