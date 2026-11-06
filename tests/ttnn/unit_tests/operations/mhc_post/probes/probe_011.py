# test_hc_scalars_are_runtime with the comb column-sum gate widened to the reference's own value
# (s/(s+eps) = 1 - eps, so |colsum - 1| ~ hc_eps); every other gate unchanged.
import torch, ttnn
import eval.golden_tests.mhc_post.helpers as H
from eval.golden_tests.mhc_post.test_regression import DS41, BF16, HIFI2
from ttnn.operations.mhc_post import mhc_post

device = ttnn.open_device(device_id=0)
try:
    for seed in (11, 12):
        case = H.make_case(DS41, mode="collapse", coef_source="packed", seed=seed, **BF16)
        case.hc = H.make_hc_scalars(generator=torch.Generator().manual_seed(seed * 31), nc=4 * 5120)
        case.hc["hc_eps"] = 1e-6 * seed
        case.hc["norm_eps"] = 1e-6 * (seed - 10)
        H.HC_COLSUM = 1e-5 + 1e-6 * seed
        out = H.call_mhc_post(
            case, mode="collapse", coef_source="packed", device=device, math_fidelity=HIFI2, op=mhc_post, **BF16
        )
        H.check_case(out, case, mode="collapse", dtype=ttnn.bfloat16, math_fidelity=HIFI2)
        comb = ttnn.to_torch(out[2]).double()[..., 8:24].reshape(-1, 4, 4)
        print("seed", seed, "all other gates pass; device max|colsum-1| =", (comb.sum(-2) - 1).abs().max().item())
    print("RESULT ok")
finally:
    ttnn.close_device(device)
