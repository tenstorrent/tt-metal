# Refinement 3: lagged norm — per-row ratio of the device y vs the reference norm on the no-norm y_b
import torch, ttnn
from eval.golden_tests.mhc_post.feature_spec import _case
from eval.golden_tests.mhc_post.helpers import *
from ttnn.operations.mhc_post import mhc_post

dev = ttnn.open_device(device_id=0)
try:
    BF16 = dict(dtype=ttnn.bfloat16, sublayer_dtype=ttnn.bfloat16)
    for shape in [(1, 1, 64, 4 * 640), (1, 1, 150, 4 * 5120)]:
        case = make_case(_case(shape), mode="collapse", coef_source="packed", seed=71, norm="on", **BF16)
        kw = dict(
            mode="collapse",
            coef_source="packed",
            device=dev,
            math_fidelity=ttnn.MathFidelity.HiFi2,
            op=mhc_post,
            **BF16,
        )
        w = case.norm_weight
        case.norm_weight = None
        plain = call_mhc_post(case, **kw)
        case.norm_weight = w
        got = call_mhc_post(case, **kw)
        yb = ttnn.to_torch(plain[3]).double().reshape(-1, shape[-1] // 4)
        y = ttnn.to_torch(got[3]).double().reshape(-1, shape[-1] // 4)
        C = yb.shape[-1]
        ref = yb * torch.rsqrt(yb.square().mean(-1, keepdim=True) + case.norm_eps) * w.double().reshape(1, -1)
        ratio = (y / ref).nanmedian(dim=-1).values
        print(shape, "rows", y.shape[0], "ratio per row (median):", [round(v, 3) for v in ratio.tolist()][:70])
        # per column tile block: which columns are off
        bad = (y - ref).abs() > 0.05 * ref.abs() + 1e-2
        print("bad frac per row", [round(v, 2) for v in bad.double().mean(-1).tolist()][:70])
        colbad = bad.double().mean(0).reshape(-1, 32).mean(-1)
        print("bad frac per col tile", [round(v, 2) for v in colbad.tolist()][:80])
finally:
    ttnn.close_device(dev)
