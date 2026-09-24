import sys

import torch
import ttnn
from ttnn.operations.gated_delta_net_backward import gated_delta_net_backward

sys.path.insert(0, "/localdev/mnedeljkovic/tt-metal/tests/ttnn/unit_tests/operations/gated_delta_net_backward")
from test_gated_delta_net_backward_debug import device_algorithm, make_inputs, pcc, read_block

gdn = sys.modules["ttnn.operations.gated_delta_net_backward.gated_delta_net_backward"]


def test_bf16_scratch(device):
    shape, chunk = (1, 32, 1, 32, 32), 32
    B, T, H, K, V = shape
    ref = make_inputs(shape)

    def dev(t):
        return ttnn.from_torch(
            t.to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    gated_delta_net_backward(
        dev(ref["q"]),
        dev(ref["k"]),
        dev(ref["v"]),
        dev(ref["g"]),
        dev(ref["beta"]),
        dev(ref["do"]),
        chunk_size=chunk,
    )
    sc = gdn._LAST_SCRATCH
    flat = ttnn.to_torch(sc["sc"]).double()
    fin = ttnn.to_torch(sc["scin"]).double()
    smap, geo = sc["map"], sc["geo"]
    Ct, Kt, Vt = geo.Ct, geo.Kt, geo.Vt

    _, P_, scan = device_algorithm(ref["q"], ref["k"], ref["v"], ref["g"], ref["beta"], ref["do"], chunk_size=chunk)
    p = P_[(0, 0, 0)]

    # compact copies (in dtype) first
    qc = ref["q"][0, :, 0, :].double()
    kc = ref["k"][0, :, 0, :].double()
    vc = ref["v"][0, :, 0, :].double()
    doc = ref["do"][0, :, 0, :].double()
    for name, base, rows, cols, exp in (
        ("sc_q", smap.base_cq, Ct, Kt, qc),
        ("sc_k", smap.base_ck, Ct, Kt, kc),
        ("sc_v", smap.base_cv, Ct, Vt, vc),
        ("sc_do", smap.base_cdo, Ct, Vt, doc),
    ):
        got = read_block(fin, base, rows, cols)[: exp.shape[0], : exp.shape[1]]
        print(f"BF16 {name}: PCC {pcc(got, exp):.6f} maxdiff {float((got-exp).abs().max()):.4g}")

    vecs = read_block(flat, smap.base_vec, 3, 1)
    print(f"BF16 decay: PCC {pcc(vecs[:32*Ct, 0:1], p['decay'][:, None]):.6f}")
    print(f"BF16 beta : PCC {pcc(vecs[32*Ct:64*Ct, 0:1], ref['beta'][0,:,0].double()[:,None]):.6f}")
    for name, base, stride, rows, cols, exp in (
        ("Tinv", smap.base_attn, smap.st_attn, Ct, Ct, p["Tinv"]),
        ("kcd", smap.base_kcd, smap.st_kcd, Ct, Kt, p["kcd"]),
        ("P", smap.base_p, smap.st_p, Ct, Kt, p["Pm"]),
        ("vcorr", smap.base_vcorr, smap.st_vcorr, Ct, Vt, p["vcorr"]),
        ("u", smap.base_u, smap.st_u, Ct, Vt, p["u"]),
        ("c", smap.base_c, smap.st_c, Kt, Vt, p["c"]),
        ("vnew", smap.base_vnew, smap.st_vnew, Ct, Vt, scan["vnew"][(0, 0, 0)]),
        ("dvnew", smap.base_dvnew, smap.st_dvnew, Ct, Vt, scan["dvnew"][(0, 0, 0)]),
    ):
        got = read_block(flat, base, rows, cols)[: exp.shape[0], : exp.shape[1]]
        print(f"BF16 {name}: PCC {pcc(got, exp):.6f} maxdiff {float((got-exp).abs().max()):.4g}")
