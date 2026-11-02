import torch, ttnn, sys

sys.path.insert(0, ".")
import eval.golden_tests.conv2d_nhwc.helpers as H
from eval.golden_tests.conv2d_nhwc.feature_spec import LOOSE_CASES
import ttnn.operations.conv2d_nhwc.conv2d_nhwc_program_descriptor as PD

_IL = ttnn.TensorMemoryLayout.INTERLEAVED
res = {}


def fake_check(out, expected, **kw):
    got = ttnn.to_torch(out).float().reshape(expected.shape)
    e = expected.float()
    pcc = torch.corrcoef(torch.stack([got.flatten(), e.flatten()]))[0, 1].item()
    rms = ((got - e).pow(2).mean().sqrt() / e.pow(2).mean().sqrt()).item()
    res["m"] = (pcc, rms)


H.check_output = fake_check
cases = []
want = ["(1, 32, 32, 1920)"]
for c in LOOSE_CASES:
    if (
        str(c["inputs"][0]) in want
        and c["dtype"] == ttnn.bfloat16
        and len(c["inputs"]) > 7
        and c["inputs"][7] == ttnn.TensorMemoryLayout.BLOCK_SHARDED
    ):
        cases.append((str(c["inputs"][:2]), c["inputs"], c["weight_dtype"], c.get("extras", {}), c["inputs"][7]))
cases.append(("IL k3 (1,56,56,64)->128 bf8b", ((1, 56, 56, 64), 128, 3, 1, 1, 1, 1), ttnn.bfloat8_b, {}, _IL))
cases.append(("IL k5 (1,32,32,64)->128 bf16", ((1, 32, 32, 64), 128, 5, 0, 1, 1, 1), ttnn.bfloat16, {}, _IL))
dev = ttnn.open_device(device_id=0, l1_small_size=16384)
policies = [(ttnn.MathFidelity.HiFi2, False), (ttnn.MathFidelity.LoFi, False), (ttnn.MathFidelity.HiFi2, True)]
try:
    for name, inputs, wdt, extras, ml in cases:
        for pol in policies:
            PD.COMPUTE_POLICY[ttnn.bfloat16] = pol
            res.clear()
            try:
                H.run_conv2d_nhwc(
                    inputs,
                    dtype=ttnn.bfloat16,
                    weight_dtype=wdt,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    bias_mode="with_bias",
                    memory_layout=ml,
                    device=dev,
                    extras=extras,
                )
                print(
                    f"POL {name} w={wdt} {pol[0]} fp32acc={pol[1]} pcc={res['m'][0]:.6f} relrms={res['m'][1]:.4f}",
                    flush=True,
                )
            except Exception as ex:
                print(f"POL {name} {pol} ERR {type(ex).__name__}: {str(ex)[:200]}", flush=True)
finally:
    ttnn.close_device(dev)
