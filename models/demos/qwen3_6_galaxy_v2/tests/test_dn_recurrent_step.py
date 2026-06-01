# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Contract test for ONE recurrent gated-delta-rule decode step (Fuse B).

Pins the math of ``_recurrent_delta_rule_step_fp32`` against a torch fp32
reference, for BOTH:
  - the OLD step (per-token reshape + to_layout(TILE) round-trip churn), and
  - the NEW tiled step (QWEN36_DN_RECUR_TILED=1: q/k/v stay in their
    matmul-ready [B,H,1,D] tiled layout across the whole step).

Both must match the torch reference at PCC>0.99 for BOTH the output o_t AND
the new recurrent state h.  This locks the math BEFORE the de-churn refactor.

The step takes inputs that are ALREADY l2-normed + scaled (the norm/scale
happens in the outer ``recurrent_gated_delta_rule_ttnn_fp32`` wrapper, not in
the step).  The step math, at T=1, is exactly:

    v_read = k @ h                       # [1,H,1,D] @ [1,H,D,D] -> [1,H,1,D]
    delta  = v - v_read
    h_new  = h * decay + (k_col @ delta_row) * beta
    o      = q @ h_new

where decay = exp(g), and k_col is k transposed on its last two dims.

Run (full galaxy):
  TT_VISIBLE_DEVICES unset; python -m pytest --noconftest \
    models/demos/qwen3_6_galaxy_v2/tests/test_dn_recurrent_step.py -v -s
"""
from __future__ import annotations

import pytest
import torch

import ttnn
from models.demos.qwen3_6_galaxy_v2.tt.ttnn_delta_rule_ops_fp32 import _recurrent_delta_rule_step_fp32

# --- GDN decode dims (per chip, TP=32) ---
_B, _H, _T, _D = 1, 6, 1, 128
_PCC_BAR = 0.99


def _pcc(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


@pytest.fixture(scope="module")
def mesh():
    ndev = len(ttnn.get_device_ids())
    m = ttnn.open_mesh_device(ttnn.MeshShape(8, 4) if ndev >= 32 else ttnn.MeshShape(1, min(ndev, 4) or 1))
    try:
        yield m
    finally:
        ttnn.close_mesh_device(m)


def _to_dev(t, mesh, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(
        t,
        dtype=dtype,
        layout=layout,
        device=mesh,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def _to_dev_f32(t, mesh, layout=ttnn.TILE_LAYOUT):
    return _to_dev(t, mesh, dtype=ttnn.float32, layout=layout)


def _first_replica(tt, mesh, want_shape):
    full = ttnn.to_torch(tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
    return full.reshape(-1, *want_shape[1:])[: want_shape[0]].reshape(want_shape)


def _torch_reference(q, k, v, beta, g, h):
    """fp32 torch reference for one delta-rule step.

    q,k,v: [B,H,1,D]  (already normed + scaled);  beta,g: [B,H,1];  h: [B,H,D,D]
    Returns (o, h_new) matching the ttnn step math exactly.
    """
    q = q.double()
    k = k.double()
    v = v.double()
    beta = beta.double()
    g = g.double()
    h = h.double()

    decay = torch.exp(g)  # [B,H,1]

    v_read = torch.matmul(k, h)  # [B,H,1,D]
    delta = v - v_read  # [B,H,1,D]

    k_col = k.transpose(-1, -2)  # [B,H,D,1]
    outer = torch.matmul(k_col, delta)  # [B,H,D,D]

    decay4 = decay.unsqueeze(-1)  # [B,H,1,1]
    beta4 = beta.unsqueeze(-1)  # [B,H,1,1]
    h_new = h * decay4 + outer * beta4  # [B,H,D,D]

    o = torch.matmul(q, h_new)  # [B,H,1,D]
    return o.float(), h_new.float()


def _run_device_step(mesh, q, k, v, beta, g, h, tiled: bool):
    """Run ONE _recurrent_delta_rule_step_fp32 on device.

    The step's input contract (per the T-loop slicing in
    recurrent_gated_delta_rule_ttnn_fp32):
      OLD path (tiled=False): q_t,k_t,v_t are [B,H,D] (the [:,:,0] slice),
        beta_t,decay_t are [B,H].
      NEW path (tiled=True): q_t,k_t,v_t are [B,H,1,D] tiled (no slice),
        beta_t,decay_t are [B,H,1].
    decay_t = exp(g) is precomputed by the wrapper.
    """
    import os

    decay_torch = torch.exp(g.double()).float()  # [B,H,1]

    device = mesh

    if tiled:
        os.environ["QWEN36_DN_RECUR_TILED"] = "1"
        q_t = _to_dev_f32(q, mesh)  # [B,H,1,D] tiled
        k_t = _to_dev_f32(k, mesh)
        v_t = _to_dev_f32(v, mesh)
        beta_t = _to_dev_f32(beta, mesh)  # [B,H,1]
        decay_t = _to_dev_f32(decay_torch, mesh)  # [B,H,1]
    else:
        os.environ["QWEN36_DN_RECUR_TILED"] = "0"
        # OLD path receives the [:,:,0] slice => [B,H,D] (ROW_MAJOR-equivalent
        # 3D) and [B,H] for beta/decay.  Replicate exactly.
        q_t = _to_dev_f32(q.reshape(_B, _H, _D), mesh)
        k_t = _to_dev_f32(k.reshape(_B, _H, _D), mesh)
        v_t = _to_dev_f32(v.reshape(_B, _H, _D), mesh)
        beta_t = _to_dev_f32(beta.reshape(_B, _H), mesh)
        decay_t = _to_dev_f32(decay_torch.reshape(_B, _H), mesh)

    h_t = _to_dev_f32(h, mesh)  # [B,H,D,D] tiled

    o_t, h_new = _recurrent_delta_rule_step_fp32(
        q_t,
        k_t,
        v_t,
        beta_t,
        decay_t,
        h_t,
        seq_len=_T,
        device=device,
    )

    o_torch = _first_replica(o_t, mesh, [_B, _H, _D]).reshape(_B, _H, _T, _D)
    h_torch = _first_replica(h_new, mesh, [_B, _H, _D, _D])
    return o_torch, h_torch


def _run_device_step_bf16_in(mesh, q, k, v, beta, g, h):
    """Fuse E: run ONE tiled step with q,k,v,beta as bf16 inputs (decay/g and the
    state h stay fp32 — exactly the QWEN36_DN_RECUR_BF16_IN=1 contract, which
    skips the q/k/v/beta up-casts in the wrapper).  The step itself is
    dtype-agnostic: its matmuls + subtract force fp32 output so the state chain
    stays fp32.  This pins the bf16-input PCC vs the torch fp32 reference."""
    import os

    decay_torch = torch.exp(g.double()).float()  # [B,H,1]

    os.environ["QWEN36_DN_RECUR_TILED"] = "1"
    # bf16 q/k/v (already-quantized contract); beta + decay + state stay fp32
    # (beta/decay feed the addcmul/multiply on the fp32 state — see the wrapper's
    # Fuse-E comment; addcmul with bf16 beta NaNs on this build).
    q_t = _to_dev(q, mesh, dtype=ttnn.bfloat16)  # [B,H,1,D] tiled, bf16
    k_t = _to_dev(k, mesh, dtype=ttnn.bfloat16)
    v_t = _to_dev(v, mesh, dtype=ttnn.bfloat16)
    beta_t = _to_dev_f32(beta, mesh)  # [B,H,1] fp32
    decay_t = _to_dev_f32(decay_torch, mesh)  # [B,H,1] fp32
    h_t = _to_dev_f32(h, mesh)  # [B,H,D,D] tiled fp32

    o_t, h_new = _recurrent_delta_rule_step_fp32(q_t, k_t, v_t, beta_t, decay_t, h_t, seq_len=_T, device=mesh)

    o_torch = _first_replica(o_t, mesh, [_B, _H, _D]).reshape(_B, _H, _T, _D)
    h_torch = _first_replica(h_new, mesh, [_B, _H, _D, _D])
    return o_torch, h_torch


def _make_inputs(seed):
    g = torch.Generator().manual_seed(seed)

    def r(shape, std):
        return torch.empty(*shape, dtype=torch.float32).normal_(0.0, std, generator=g)

    # already-normed q/k (unit-ish), modest v; small state for stability.
    q = torch.nn.functional.normalize(r([_B, _H, _T, _D], 1.0), dim=-1) * (_D**-0.5)
    k = torch.nn.functional.normalize(r([_B, _H, _T, _D], 1.0), dim=-1)
    v = r([_B, _H, _T, _D], 0.5)
    beta = torch.sigmoid(r([_B, _H, _T], 1.0))  # [B,H,1]
    glog = -torch.nn.functional.softplus(r([_B, _H, _T], 1.0))  # g<0 => decay in (0,1)
    h = r([_B, _H, _D, _D], 0.02)
    return q, k, v, beta, glog, h


@pytest.mark.parametrize("tiled", [False, True], ids=["old", "tiled"])
def test_recurrent_step_matches_torch(mesh, tiled):
    """Both old and new tiled step match the torch fp32 reference at PCC>0.99
    for output o_t AND new state h."""
    q, k, v, beta, glog, h = _make_inputs(seed=7)

    o_ref, h_ref = _torch_reference(q, k, v, beta, glog, h)
    o_dev, h_dev = _run_device_step(mesh, q, k, v, beta, glog, h, tiled=tiled)

    pcc_o = _pcc(o_dev, o_ref)
    pcc_h = _pcc(h_dev, h_ref)
    label = "tiled" if tiled else "old"
    print(f"[dn-step:{label}] o_t PCC={pcc_o:.6f}  h_new PCC={pcc_h:.6f}")
    assert pcc_o > _PCC_BAR, f"[{label}] o_t PCC {pcc_o:.5f} < {_PCC_BAR}"
    assert pcc_h > _PCC_BAR, f"[{label}] h_new PCC {pcc_h:.5f} < {_PCC_BAR}"


def test_recurrent_step_bf16_inputs(mesh):
    """Fuse E: tiled step with bf16 q,k,v,beta inputs (state h + decay fp32) vs
    the torch fp32 reference.  bf16 inputs are already-quantized, so PCC may be
    slightly < 1.0 but must clear 0.99.  Also report the fp32-input PCC delta so
    the precision cost of dropping the up-casts is quantified."""
    q, k, v, beta, glog, h = _make_inputs(seed=7)

    o_ref, h_ref = _torch_reference(q, k, v, beta, glog, h)

    o_f32, h_f32 = _run_device_step(mesh, q, k, v, beta, glog, h, tiled=True)
    pcc_o_f32 = _pcc(o_f32, o_ref)
    pcc_h_f32 = _pcc(h_f32, h_ref)

    o_bf, h_bf = _run_device_step_bf16_in(mesh, q, k, v, beta, glog, h)
    pcc_o_bf = _pcc(o_bf, o_ref)
    pcc_h_bf = _pcc(h_bf, h_ref)

    print(
        f"[dn-step:bf16-in] o_t PCC={pcc_o_bf:.6f} (fp32-in {pcc_o_f32:.6f}, "
        f"delta {pcc_o_bf - pcc_o_f32:+.6f})  "
        f"h_new PCC={pcc_h_bf:.6f} (fp32-in {pcc_h_f32:.6f}, "
        f"delta {pcc_h_bf - pcc_h_f32:+.6f})"
    )
    assert pcc_o_bf > _PCC_BAR, f"[bf16-in] o_t PCC {pcc_o_bf:.5f} < {_PCC_BAR}"
    assert pcc_h_bf > _PCC_BAR, f"[bf16-in] h_new PCC {pcc_h_bf:.5f} < {_PCC_BAR}"


if __name__ == "__main__":
    pytest.main([__file__, "-s", "-v"])
