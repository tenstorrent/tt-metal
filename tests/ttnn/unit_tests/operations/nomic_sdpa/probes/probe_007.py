import torch, ttnn
from eval.golden_tests.nomic_sdpa.helpers import make_qkv, reference, PROFILE_TOLERANCES
from ttnn.operations.nomic_sdpa import nomic_sdpa

device = ttnn.open_device(device_id=0)
H = ttnn.MathFidelity
VARIANTS = {
    "bf16d_hifi2_approx": (H.HiFi2, False, True),
    "bf16d_hifi2_exact": (H.HiFi2, False, False),
    "bf16d_hifi3_approx": (H.HiFi3, False, True),
    "bf16d_hifi4_approx": (H.HiFi4, False, True),
}


def stats(got, ref):
    got = got.double().flatten()
    ref = ref.double().flatten()
    rms = ref.pow(2).mean().sqrt()
    rel = float((got - ref).pow(2).mean().sqrt() / rms)
    gc, ec = got - got.mean(), ref - ref.mean()
    return float(gc @ ec / (gc.norm() * ec.norm())), rel


try:
    for B, S in ((8, 512), (4, 512), (16, 256), (32, 128), (8, 384)):
        for prof in ("real", "flat", "peaky"):
            q, k, v = make_qkv((B, 12, S, 64), prof, seed=0)
            ref = reference(q, k, v, 1.0).reshape(B, 1, S, 768)
            tq, tk, tv = (
                ttnn.from_torch(
                    t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
                )
                for t in (q, k, v)
            )
            for name, (fid, fp32d, approx) in VARIANTS.items():
                cfg = ttnn.ComputeConfigDescriptor(
                    math_fidelity=fid, fp32_dest_acc_en=fp32d, math_approx_mode=approx, dst_full_sync_en=False
                )
                out = nomic_sdpa(tq, tk, tv, compute_kernel_config=cfg)
                got = ttnn.to_torch(out).reshape(B, 1, S, 768)
                pcc, rel = stats(got, ref)
                bar = PROFILE_TOLERANCES[prof]
                ok = "OK" if (pcc >= bar[0] and rel <= bar[1]) else "FAIL"
                print(f"R {B}x{S} {prof:5s} {name:22s} pcc={pcc:.6f} rel={rel:.5f} bar={bar} {ok}")
                ttnn.deallocate(out)
            for t in (tq, tk, tv):
                ttnn.deallocate(t)
finally:
    ttnn.close_device(device)
