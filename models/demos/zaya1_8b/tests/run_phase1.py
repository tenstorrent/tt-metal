"""Phase 1 PCC validation: embedding, RMSNorm, partial-RoPE cos/sin, lm_head.

Run:  TT_DEVICE=1 /home/yito/work/run_zaya.sh \
        python models/demos/zaya1_8b/tests/run_phase1.py
"""
import os
import torch
import ttnn

from models.common.utility_functions import comp_pcc
from models.demos.zaya1_8b.tt.model_args import ZayaConfig, ZayaWeights
from models.demos.zaya1_8b.tt import standard as S

GOLDEN = os.path.join(os.path.dirname(__file__), "..", "reference", "golden")
PCC_MIN = 0.99
results = []


def g(name):
    return torch.load(os.path.join(GOLDEN, f"{name}.pt"), weights_only=False)


def check(name, ttnn_out, golden_out, pcc_min=PCC_MIN):
    got = ttnn.to_torch(ttnn_out).float()
    exp = golden_out.float()
    if got.shape != exp.shape:
        got = got.reshape(exp.shape)
    ok, pcc = comp_pcc(exp, got, pcc_min)
    results.append((name, ok, pcc))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:22s} pcc={pcc}")
    return ok


def main():
    device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape((1, 1)))
    try:
        w = ZayaWeights()
        inputs = g("inputs")
        input_ids = inputs["input_ids"]  # [1, S] long
        print(f"input_ids {tuple(input_ids.shape)}")

        # 1) embedding
        emb = S.Embedding(device, w.embed())
        check("embedding", emb(input_ids), g("embed_tokens")["out"])

        # 2) RMSNorm (layer-0 input_norm) — feed golden input
        l0 = g("L0_input_norm")
        n0 = S.RMSNorm(device, w.att(0, "input_norm.weight"))
        x0 = S.to_dev(l0["in"][0], device)
        check("rmsnorm_L0", n0(x0), l0["out"])

        # 3) RMSNorm (final_norm)
        fn = g("final_norm")
        nf = S.RMSNorm(device, w.final_norm())
        xf = S.to_dev(fn["in"][0], device)
        check("rmsnorm_final", nf(xf), fn["out"])

        # 4) lm_head (tied to embed)
        lh = g("lm_head")
        head = S.LMHead(device, w.embed())
        xh = S.to_dev(lh["in"][0], device)
        check("lm_head", head(xh), lh["out"], pcc_min=0.98)

        # 5) partial-RoPE cos/sin (torch-only vs golden)
        re = g("rotary_emb")["out"]          # [cos, sin], each [1, S, rotary_dim]
        cos_g, sin_g = re[0].float(), re[1].float()
        S_len = cos_g.shape[-2]
        cos, sin = S.compute_cos_sin(S_len)
        for nm, a, b in [("rope_cos", cos, cos_g.reshape(S_len, -1)),
                         ("rope_sin", sin, sin_g.reshape(S_len, -1))]:
            ok, pcc = comp_pcc(b, a, 0.999)
            results.append((nm, ok, pcc))
            print(f"  [{'PASS' if ok else 'FAIL'}] {nm:22s} pcc={pcc}")
    finally:
        ttnn.close_mesh_device(device)

    print("\n=== Phase 1 summary ===")
    npass = sum(1 for _, ok, _ in results if ok)
    for n, ok, pcc in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {n}")
    print(f"{npass}/{len(results)} passed")
    raise SystemExit(0 if npass == len(results) else 1)


if __name__ == "__main__":
    main()
