"""Phase 3 PCC validation: CCA (compressed convolutional attention), layer 0.

(A) host/torch: conv-equivalent matrices vs reference conv golden (isolates math)
(B) device: CCAQKV (q,k,v) vs L0_cca_qkv
(C) device: full CCAAttention vs L0_self_attn

Run: TT_DEVICE=1 /home/yito/work/run_zaya.sh python models/demos/zaya1_8b/tests/run_phase3.py
"""
import os
import torch
import ttnn

from models.common.utility_functions import comp_pcc
from models.demos.zaya1_8b.tt.model_args import ZayaWeights
from models.demos.zaya1_8b.tt import standard as S
from models.demos.zaya1_8b.tt import cca as CCA

GOLDEN = os.path.join(os.path.dirname(__file__), "..", "reference", "golden")
results = []


def g(name):
    return torch.load(os.path.join(GOLDEN, f"{name}.pt"), weights_only=False)


def rec(name, ok, pcc):
    results.append((name, ok, pcc))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:24s} pcc={pcc}")


def host_conv_check(w):
    lq = g("L0_linear_q")["out"].float().reshape(-1, 1024)   # [S,1024]
    lk = g("L0_linear_k")["out"].float().reshape(-1, 256)
    qk = torch.cat([lq, lk], dim=-1)                          # [S,1280]
    Sn = qk.shape[0]
    p = "model.layers.0.self_attn.qkv"
    Am, Bm, Cm, bias = CCA.build_conv_equiv(
        w.get(f"{p}.conv_qk.0.weight"), w.get(f"{p}.conv_qk.0.bias"),
        w.get(f"{p}.conv_qk.1.weight"), w.get(f"{p}.conv_qk.1.bias"))
    sh1, sh2 = CCA.shift_matrix(Sn, 1), CCA.shift_matrix(Sn, 2)
    conv = qk @ Cm.t() + (sh1 @ qk) @ Bm.t() + (sh2 @ qk) @ Am.t() + bias
    golden = g("L0_conv_qk")["out"].float().reshape(1280, Sn).t()  # [S,1280]
    ok, pcc = comp_pcc(golden, conv, 0.9999)
    rec("conv_equiv(host)", ok, pcc)


def main():
    w = ZayaWeights()
    host_conv_check(w)

    device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape((1, 1)))
    try:
        qkv_g = g("L0_cca_qkv")
        hidden_t = qkv_g["in"][0].float()                     # already [B,S,2048] = [1,6,2048]
        S_len = hidden_t.shape[1]
        hidden = S.to_dev(hidden_t, device)

        qkv = CCA.CCAQKV(device, w, 0, S_len)
        q, k, v = qkv.forward(hidden)
        qg, kg, vg = qkv_g["out"]
        for nm, tt, go in [("cca_q", q, qg), ("cca_k", k, kg), ("cca_v", v, vg)]:
            ok, pcc = comp_pcc(go.float(), ttnn.to_torch(tt).float().reshape(go.shape), 0.98)
            rec(nm, ok, pcc)

        # full attention
        attn = CCA.CCAAttention(device, w, 0, S_len)
        out = attn.forward(hidden)
        sa = g("L0_self_attn")["out"][0]
        got_attn = ttnn.to_torch(out).float().reshape(sa.shape)
        ok, pcc = comp_pcc(sa.float(), got_attn, 0.98)
        rec("cca_attention", ok, pcc)
        # per-position breakdown
        for pp in range(sa.shape[1]):
            _, ppcc = comp_pcc(sa[0, pp].float(), got_attn[0, pp], 0.9)
            print(f"      pos{pp} cca_attn pcc={ppcc}")
    finally:
        ttnn.close_mesh_device(device)

    npass = sum(1 for _, ok, _ in results if ok)
    print(f"\n=== Phase 3 summary: {npass}/{len(results)} passed ===")
    raise SystemExit(0 if npass == len(results) else 1)


if __name__ == "__main__":
    main()
