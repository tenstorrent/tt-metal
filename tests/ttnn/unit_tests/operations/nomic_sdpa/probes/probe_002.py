import torch, ttnn
from eval.golden_tests.nomic_sdpa.helpers import make_qkv, reference, PROFILE_TOLERANCES
import ttnn.operations.nomic_sdpa.nomic_sdpa as mod
from ttnn.operations.nomic_sdpa import nomic_sdpa

device = ttnn.open_device(device_id=0)


def pcc(a, b):
    a = a.flatten() - a.mean()
    b = b.flatten() - b.mean()
    return float(a @ b / (a.norm() * b.norm()))


try:
    for B, S in ((8, 512), (2, 512), (4, 288)):
        for prof in ("real", "flat", "peaky"):
            q, k, v = make_qkv((B, 12, S, 64), prof, seed=0)
            ref = reference(q, k, v, 1.0).reshape(B, 1, S, 768)
            tq, tk, tv = (
                ttnn.from_torch(
                    t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
                )
                for t in (q, k, v)
            )
            for form, approx in ((0, True), (1, True), (0, False)):
                mod.EXP_FORM = form
                mod._CALL_CACHE.clear()
                cfg = ttnn.ComputeConfigDescriptor(
                    math_fidelity=ttnn.MathFidelity.HiFi2,
                    fp32_dest_acc_en=False,
                    math_approx_mode=approx,
                    dst_full_sync_en=False,
                )
                out = nomic_sdpa(tq, tk, tv, compute_kernel_config=cfg)
                got = ttnn.to_torch(out).reshape(B, 1, S, 768).double()
                rel = ((got - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()
                bar = PROFILE_TOLERANCES[prof]
                ok = pcc(got, ref) >= bar[0] and rel <= bar[1]
                print(
                    f"R {B}x{S} {prof:5s} form={form} approx={approx} pcc={pcc(got,ref):.6f} rel={rel:.5f} bar={bar} {'OK' if ok else 'FAIL'}"
                )
                ttnn.deallocate(out)
            for t in (tq, tk, tv):
                ttnn.deallocate(t)
finally:
    ttnn.close_device(device)
