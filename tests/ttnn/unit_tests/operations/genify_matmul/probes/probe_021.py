import math, torch, ttnn
from eval.golden_tests.genify_matmul import helpers as H
import eval.golden_tests.genify_matmul.helpers as hm

device = ttnn.open_device(device_id=0)
captured = {}
orig = hm.check_output


def spy(out, expected, **kw):
    got = ttnn.to_torch(out).float().reshape(expected.shape)
    e = expected.float()
    pcc = torch.corrcoef(torch.stack([got.flatten(), e.flatten()]))[0, 1].item()
    rms = ((got - e).pow(2).mean().sqrt() / e.pow(2).mean().sqrt()).item()
    captured["m"] = (pcc, rms, kw.get("tolerance"))
    return orig(out, expected, **kw)


hm.check_output = spy
try:
    cases = [
        (
            "perf_focus2 1x4352x5120 bf4b LoFi",
            (1, 1, 1, 4352),
            (4352, 5120),
            "none",
            dict(act_shard=([32, 4352 // 8], 8), math_fidelity=ttnn.MathFidelity.LoFi),
            ttnn.bfloat4_b,
        ),
        (
            "gate_up 32x5120x9216 swiglu",
            (1, 1, 32, 5120),
            (5120, 9216),
            "swiglu",
            dict(act_shard=([32, 512], 10), math_fidelity=ttnn.MathFidelity.LoFi),
            ttnn.bfloat4_b,
        ),
        (
            "lm_head 1x5120x16384",
            (1, 1, 1, 5120),
            (5120, 16384),
            "none",
            dict(act_shard=([32, 512], 10), math_fidelity=ttnn.MathFidelity.LoFi),
            ttnn.bfloat4_b,
        ),
        ("cart 384x1536x1024 bf16 HiFi4", (1, 384, 1536), (1536, 1024), "none", {}, ttnn.bfloat16),
    ]
    for name, a, b, ep, ex, wd in cases:
        H.run_genify_matmul(
            (a, b),
            dtype=ttnn.bfloat16,
            weight_dtype=wd,
            layout=ttnn.TILE_LAYOUT,
            fp32_dest_acc_en=True,
            placement="dram_sharded",
            epilogue=ep,
            device=device,
            extras=ex,
        )
        pcc, rms, tol = captured["m"]
        print(f"R4 {name}: PCC {pcc:.6f} relRMS {rms:.5f} gate {tol}")
finally:
    ttnn.close_device(device)
