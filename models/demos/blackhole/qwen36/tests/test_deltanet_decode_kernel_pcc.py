# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Unit PCC: fused deltanet_decode_full kernel vs main's python GDN decode math (Stage 5).

Small random single-token step (per-device GDN dims: Nk=4, Nv=12, Dk=Dv=128). Reference =
exactly what gdn/tp.py forward_decode does with the python recurrence:
  L2norm(q,k) + q*scale → recurrent_gated_delta_rule_decode_ttnn → rms_norm(Dv) → * silu(z).
Kernel = ttnn.experimental.deltanet_decode_full with the equivalent inputs packed flat
(q/k L2-normed+scaled and un-expanded; decay=exp(g); beta; z; norm_weight). Asserts the
fused output and new recurrent state match the python path (PCC), validating the kernel +
the host-side input packing before wiring it into the model (Stage 6).

Run:
    MESH_DEVICE=P150x4 pytest models/demos/blackhole/qwen36/tests/test_deltanet_decode_kernel_pcc.py -v -s
"""
import torch

import ttnn
from models.demos.blackhole.qwen36.tests.test_factory import parametrize_mesh_tp
from models.experimental.gated_attention_gated_deltanet.tt.ttnn_delta_rule_ops import (
    l2_norm_ttnn,
    recurrent_gated_delta_rule_decode_ttnn,
)


def _pcc(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


@parametrize_mesh_tp()
def test_deltanet_decode_full_pcc(mesh_device, ensure_gc):
    from loguru import logger

    import os as _os0

    Nk = int(_os0.environ.get("TEST_NK", "4"))
    Nv, Dk, Dv, K = 12, 128, 128, 4
    rf = Nv // Nk
    scale = Dk**-0.5
    key_dim, val_dim = Nk * Dk, Nv * Dv
    conv_dim = 2 * key_dim + val_dim
    torch.manual_seed(0)

    rep = ttnn.ReplicateTensorToMesh(mesh_device)

    def dev(t):
        return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=rep)

    # Raw post-conv q/k/v (un-expanded), gate z, beta (already sigmoid'd), g (log-decay), state, norm.
    q_nk = torch.randn(1, Nk, Dk) * 0.3
    k_nk = torch.randn(1, Nk, Dk) * 0.3
    if _os0.environ.get("QEQK") == "1":
        k_nk = q_nk.clone()  # probe: if read-source (q vs k) is the bug, q==k makes output match
    v_nv = torch.randn(1, Nv, Dv) * 0.3
    beta = torch.rand(1, Nv)  # in (0,1) like sigmoid(b)
    g = -torch.rand(1, Nv) * 0.1  # small negative log-decay
    rec = torch.randn(1, Nv, Dk, Dv) * 0.1

    # DIAGNOSTIC: isolate the output-path mismatch (new_state already matches). Try random vs
    # constant z (gate) and norm_w (RMSNorm weight); a jump to ~1.0 pinpoints the culprit layout.
    import os as _os

    z_rand = torch.randn(1, Nv * Dv) * 0.3
    nw_rand = torch.randn(Dv) * 0.2 + 1.0
    z_const = torch.ones(1, Nv * Dv) * 0.5
    nw_const = torch.ones(Dv)
    zc = _os.environ.get("ZCONST") == "1"
    nwc = _os.environ.get("NWCONST") == "1"
    z = z_const if zc else z_rand
    norm_w = nw_const if nwc else nw_rand
    logger.info(f"DIAG z_const={zc} nw_const={nwc}")

    # ---------- Reference: main's forward_decode math (python recurrence) ----------
    q_exp = q_nk.repeat_interleave(rf, dim=1).reshape(1, 1, Nv, Dk)
    k_exp = k_nk.repeat_interleave(rf, dim=1).reshape(1, 1, Nv, Dk)
    v_r = v_nv.reshape(1, 1, Nv, Dv)
    o, new_rec = recurrent_gated_delta_rule_decode_ttnn(
        dev(q_exp), dev(k_exp), dev(v_r),
        ttnn.reshape(dev(beta), [1, 1, Nv]), ttnn.reshape(dev(g), [1, 1, Nv]),
        scale=scale, initial_state=dev(rec), device=mesh_device, high_precision=False,
    )
    out_r = ttnn.reshape(o, (1, Nv, Dv))
    out_n = ttnn.rms_norm(out_r, weight=dev(norm_w.reshape(1, 1, Dv)), epsilon=1e-6)
    ref_out = ttnn.multiply(ttnn.reshape(out_n, (1, 1, Nv * Dv)), ttnn.silu(dev(z.reshape(1, 1, Nv * Dv))))
    ref_out_t = ttnn.to_torch(ref_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[:1]
    ref_rec_t = ttnn.to_torch(new_rec, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[:1]

    # ---------- Kernel: pack flat qkv (L2norm+scale folded into q,k; un-expanded) ----------
    qn = ttnn.multiply(l2_norm_ttnn(dev(q_nk), dim=-1), scale)  # [1,Nk,Dk]
    kn = l2_norm_ttnn(dev(k_nk), dim=-1)
    qkv_flat = ttnn.concat(
        [ttnn.reshape(qn, (1, 1, key_dim)), ttnn.reshape(kn, (1, 1, key_dim)), dev(v_nv.reshape(1, 1, val_dim))], dim=-1
    )
    qkv_proj = ttnn.reshape(qkv_flat, (1, 1, 1, conv_dim))
    z_proj = dev(z.reshape(1, 1, 1, Nv * Dv))
    b_proj = dev(beta.reshape(1, 1, 1, Nv))
    a_proj = ttnn.exp(dev(g.reshape(1, 1, 1, Nv)))  # decay
    dummy_conv = dev(torch.zeros(1, 1, conv_dim, 32))
    a_log = dev(torch.zeros(1, 1, 1, Nv))
    dt_bias = dev(torch.zeros(1, 1, 1, Nv))
    norm_weight = dev(norm_w.reshape(1, 1, 1, Dv))
    rec_state = dev(rec)  # [1,Nv,Dk,Dv]

    out_list = ttnn.experimental.deltanet_decode_full(
        qkv_proj, z_proj, b_proj, a_proj, dummy_conv, rec_state, dummy_conv, a_log, dt_bias, norm_weight,
        num_heads=Nv, num_k_heads=Nk, k_head_dim=Dk, v_head_dim=Dv,
        conv_dim=conv_dim, conv_kernel_size=K, head_expand_ratio=rf,
    )
    k_out_t = ttnn.to_torch(out_list[0], mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[:1]
    k_rec_t = ttnn.to_torch(out_list[1], mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[:1]

    # ---------- Pure-torch reference (ground truth) to arbitrate kernel vs ttnn-ref ----------
    def _l2(x):
        return x / (x.pow(2).sum(-1, keepdim=True).sqrt() + 1e-12)

    qT = _l2(q_nk.repeat_interleave(rf, dim=1)[0]) * scale  # [Nv,Dk]
    kT = _l2(k_nk.repeat_interleave(rf, dim=1)[0])  # [Nv,Dk]
    vT = v_nv[0]  # [Nv,Dv]
    hT = rec[0] * torch.exp(g[0]).reshape(Nv, 1, 1)  # decay state
    v_read = torch.einsum("hk,hkv->hv", kT, hT)
    delta = vT - v_read
    hT = hT + beta[0].reshape(Nv, 1, 1) * torch.einsum("hk,hv->hkv", kT, delta)
    oT = torch.einsum("hk,hkv->hv", qT, hT)  # [Nv,Dv]
    # Probe variant: read with k instead of q (would indicate a read-source port bug).
    o_kread = torch.einsum("hk,hkv->hv", kT, hT)
    _rk = o_kread.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt()
    torch_out_kread = (((o_kread / _rk) * norm_w.reshape(1, Dv)) * torch.nn.functional.silu(z.reshape(Nv, Dv))).reshape(-1)
    rms = oT.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt()
    torch_out = ((oT / rms) * norm_w.reshape(1, Dv)) * torch.nn.functional.silu(z.reshape(Nv, Dv))
    torch_out = torch_out.reshape(-1)
    torch_rec = hT.reshape(Nv, Dk, Dv)

    pcc_out = _pcc(ref_out_t, k_out_t)
    pcc_rec = _pcc(ref_rec_t, k_rec_t)
    logger.info(f"deltanet_decode_full PCC: output={pcc_out:.5f}  new_state={pcc_rec:.5f}")
    logger.info(
        f"vs TORCH: kernel_out={_pcc(k_out_t, torch_out):.5f}  ttnnref_out={_pcc(ref_out_t, torch_out):.5f}  "
        f"kernel_rec={_pcc(k_rec_t, torch_rec):.5f}  ttnnref_rec={_pcc(ref_rec_t, torch_rec):.5f}"
    )
    logger.info(f"PROBE kernel_out vs torch(k-read)={_pcc(k_out_t, torch_out_kread):.5f}")
    logger.info(f"PROBE-RAW kernel_out vs torch_raw(pre-norm q@S)={_pcc(k_out_t, oT.reshape(-1)):.5f}")
    assert pcc_out > 0.99, f"output PCC {pcc_out:.5f} too low (kernel packing/math mismatch)"
    assert pcc_rec > 0.99, f"new_state PCC {pcc_rec:.5f} too low"
    logger.info("PASSED: fused deltanet_decode_full matches python GDN decode math")
