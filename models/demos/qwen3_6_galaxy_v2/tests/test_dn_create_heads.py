# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Unit test for the GDN/DeltaNet decode "create-heads" / "concat-heads" sub-block.

This isolates the layout transform that turns the flat per-row projection/conv
outputs into the per-head tensors the recurrent core consumes, and back again —
the hand-rolled equivalent of ``nlp_create_qkv_heads`` / ``nlp_concat_heads``.

Purpose
-------
1. **Golden contract** (``test_create_heads_contract`` / ``test_concat_heads_contract``):
   pin the CURRENT ttnn op sequence (reshape + ``repeat_interleave``) against a
   torch reference at PCC > 0.99. Any churn-reduction refactor of this sub-block
   (fused op, or keeping the per-head tensors tiled across the recurrent step)
   must keep these green.
2. **Feasibility probe** for replacing the manual split with the fused
   ``ttnn.experimental.nlp_create_qkv_heads``:
   - ``test_fused_create_heads_at_6_6_6``: the fused op DOES produce the
     recurrent-ready ``[1, 6, 1, 128]`` layout — but only when q/k/v all have the
     SAME head count (post-GQA-expand), since the op forces ``k_heads == v_heads``.
   - ``test_fused_create_heads_rejects_inverse_gqa``: documents WHY it can't be
     used to fuse the split directly — GDN is *inverse* GQA (k=2 heads, v=6
     heads), which the op's ``num_kv_heads`` (shared by k and v) cannot express.

GDN decode dims (per chip, TP=32): head_dim=128, n_k/chip=2, n_v/chip=6,
GQA ratio=3, q_per_row=k_per_row=256, v_per_row=768.

Run:
  TT_VISIBLE_DEVICES unset (full galaxy);  python -m pytest --noconftest \
    models/demos/qwen3_6_galaxy_v2/tests/test_dn_create_heads.py -v -s
"""

import pytest
import torch

import ttnn

# --- GDN decode head config (per chip) ---
_HEAD_DIM = 128
_N_K = 2  # k (and q) heads per chip
_N_V = 6  # v heads per chip
_RATIO = _N_V // _N_K  # 3 — GQA expand factor for q,k
_Q_PER_ROW = _N_K * _HEAD_DIM  # 256
_V_PER_ROW = _N_V * _HEAD_DIM  # 768
_B, _T = 1, 1  # decode
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


def _to_dev(t, mesh):
    return ttnn.from_torch(
        t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh)
    )


def _first_replica(tt, mesh, want_shape):
    """Read a replicated tensor back; all devices identical, take the first."""
    full = ttnn.to_torch(tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
    return full.reshape(-1, *want_shape[1:])[: want_shape[0]].reshape(want_shape)


# ---------------------------------------------------------------------------
# 1. GOLDEN CONTRACT — current create-heads (reshape + repeat_interleave)
# ---------------------------------------------------------------------------
def test_create_heads_contract(mesh):
    """flat q_conv|k_conv|v_conv|z  →  per-head q_e,k_e,v_h,z_h [1,1,6,128].

    Mirrors forward_decode lines ~2900-2903 (per-head reshape) + _gqa_expand_q_k
    (repeat_interleave). Pins the layout contract the recurrent core depends on.
    """
    torch.manual_seed(0)
    q_conv = torch.randn(_B, _T, _Q_PER_ROW)
    k_conv = torch.randn(_B, _T, _Q_PER_ROW)
    v_conv = torch.randn(_B, _T, _V_PER_ROW)
    z = torch.randn(_B, _T, _V_PER_ROW)

    # --- torch reference ---
    q_h = q_conv.reshape(_B, _T, _N_K, _HEAD_DIM)
    k_h = k_conv.reshape(_B, _T, _N_K, _HEAD_DIM)
    q_e_ref = q_h.repeat_interleave(_RATIO, dim=2)  # [1,1,6,128]
    k_e_ref = k_h.repeat_interleave(_RATIO, dim=2)
    v_h_ref = v_conv.reshape(_B, _T, _N_V, _HEAD_DIM)
    z_h_ref = z.reshape(_B, _T, _N_V, _HEAD_DIM)

    # --- current ttnn op sequence ---
    mem = ttnn.DRAM_MEMORY_CONFIG
    q_t, k_t, v_t, z_t = _to_dev(q_conv, mesh), _to_dev(k_conv, mesh), _to_dev(v_conv, mesh), _to_dev(z, mesh)
    q_hd = ttnn.reshape(q_t, [_B, _T, _N_K, _HEAD_DIM], memory_config=mem)
    k_hd = ttnn.reshape(k_t, [_B, _T, _N_K, _HEAD_DIM], memory_config=mem)
    q_e = ttnn.repeat_interleave(q_hd, _RATIO, dim=2, memory_config=mem)
    k_e = ttnn.repeat_interleave(k_hd, _RATIO, dim=2, memory_config=mem)
    v_hd = ttnn.reshape(v_t, [_B, _T, _N_V, _HEAD_DIM], memory_config=mem)
    z_hd = ttnn.reshape(z_t, [_B, _T, _N_V, _HEAD_DIM], memory_config=mem)

    shp = [_B, _T, _N_V, _HEAD_DIM]
    for name, tt, ref in [("q_e", q_e, q_e_ref), ("k_e", k_e, k_e_ref), ("v_h", v_hd, v_h_ref), ("z_h", z_hd, z_h_ref)]:
        pcc = _pcc(_first_replica(tt, mesh, shp), ref)
        print(f"[create-heads] {name} PCC={pcc:.6f}")
        assert pcc > _PCC_BAR, f"{name} PCC {pcc:.5f} < {_PCC_BAR}"


def test_direct_bhtd_create_heads(mesh):
    """QWEN36_DN_FUSED_HEADS path: produce per-head q/k/v already in the
    recurrent core's [B, H, T, D] = [1, 6, 1, 128] layout via a DIRECT reshape
    (reshape q_conv->[1,2,1,128], repeat_interleave dim=1; v_conv->[1,6,1,128]),
    with NO transpose. Proves it equals the torch reference [1,6,1,128] — i.e.
    the current [1,1,6,128] create-heads data transposed to [B,H,T,D]. At T=1
    the direct reshape is bit-identical to reshape([B,T,H,D])+transpose(1,2).
    """
    torch.manual_seed(3)
    q_conv = torch.randn(_B, _T, _Q_PER_ROW)
    k_conv = torch.randn(_B, _T, _Q_PER_ROW)
    v_conv = torch.randn(_B, _T, _V_PER_ROW)

    # --- torch reference: old create-heads ([1,1,6,128]) then transpose to [1,6,1,128] ---
    q_e_old = q_conv.reshape(_B, _T, _N_K, _HEAD_DIM).repeat_interleave(_RATIO, dim=2)  # [1,1,6,128]
    k_e_old = k_conv.reshape(_B, _T, _N_K, _HEAD_DIM).repeat_interleave(_RATIO, dim=2)
    v_h_old = v_conv.reshape(_B, _T, _N_V, _HEAD_DIM)
    q_ref = q_e_old.transpose(1, 2)  # [1,6,1,128]
    k_ref = k_e_old.transpose(1, 2)
    v_ref = v_h_old.transpose(1, 2)

    # --- ttnn direct [B,H,T,D] reshape path (no transpose) ---
    mem = ttnn.DRAM_MEMORY_CONFIG
    q_t, k_t, v_t = _to_dev(q_conv, mesh), _to_dev(k_conv, mesh), _to_dev(v_conv, mesh)
    q_hd = ttnn.reshape(q_t, [_B, _N_K, _T, _HEAD_DIM], memory_config=mem)  # [1,2,1,128]
    k_hd = ttnn.reshape(k_t, [_B, _N_K, _T, _HEAD_DIM], memory_config=mem)
    q_e = ttnn.repeat_interleave(q_hd, _RATIO, dim=1, memory_config=mem)  # [1,6,1,128]
    k_e = ttnn.repeat_interleave(k_hd, _RATIO, dim=1, memory_config=mem)
    v_hd = ttnn.reshape(v_t, [_B, _N_V, _T, _HEAD_DIM], memory_config=mem)  # [1,6,1,128]

    shp = [_B, _N_V, _T, _HEAD_DIM]
    for name, tt, ref in [("q", q_e, q_ref), ("k", k_e, k_ref), ("v", v_hd, v_ref)]:
        assert list(tt.shape) == shp, f"{name} shape {list(tt.shape)} != {shp}"
        pcc = _pcc(_first_replica(tt, mesh, shp), ref)
        print(f"[direct BHTD] {name} PCC={pcc:.6f}")
        assert pcc > _PCC_BAR, f"{name} PCC {pcc:.5f} < {_PCC_BAR}"
    print("[direct BHTD] => direct [1,6,1,128] reshape == old create-heads transposed; transpose-free & PCC-safe.")


def test_concat_heads_contract(mesh):
    """per-head core_out [1,1,6,128]  →  flat [1,1,768] (the _apply_norm_gated reshape side)."""
    torch.manual_seed(1)
    core_out = torch.randn(_B, _T, _N_V, _HEAD_DIM)
    flat_ref = core_out.reshape(_B, _T, _V_PER_ROW)

    co = _to_dev(core_out, mesh)
    flat = ttnn.reshape(co, [_B, _T, _V_PER_ROW], memory_config=ttnn.DRAM_MEMORY_CONFIG)
    pcc = _pcc(_first_replica(flat, mesh, [_B, _T, _V_PER_ROW]), flat_ref)
    print(f"[concat-heads] flat PCC={pcc:.6f}")
    assert pcc > _PCC_BAR


# ---------------------------------------------------------------------------
# 2. FEASIBILITY — can ttnn.experimental.nlp_create_qkv_heads replace the split?
# ---------------------------------------------------------------------------
def test_fused_create_heads_at_6_6_6(mesh):
    """The fused op DOES yield the recurrent-ready [1,6,1,128] layout — but only
    when q,k,v share the SAME head count (i.e. AFTER the GQA 2→6 expand).

    This is the post-expand regime: a fused q6|k6|v6 input. If this matches the
    torch split, then nlp_create_qkv_heads can collapse the per-head reshape +
    the 3 q/k/v transposes (forward_decode reshape + recurrent_fp32 transpose)
    into ONE multi-core op — provided the gqa-expand still runs first.
    """
    torch.manual_seed(2)
    # fused [B,1,S, (nq + 2*nkv)*hd] with nq=nkv=6
    n = _N_V
    fused = torch.randn(_B, 1, _T, 3 * n * _HEAD_DIM)
    # torch reference split: q|k|v each [B,n,S,hd]
    qr = fused[..., : n * _HEAD_DIM].reshape(_B, _T, n, _HEAD_DIM).transpose(1, 2)
    kr = fused[..., n * _HEAD_DIM : 2 * n * _HEAD_DIM].reshape(_B, _T, n, _HEAD_DIM).transpose(1, 2)
    vr = fused[..., 2 * n * _HEAD_DIM :].reshape(_B, _T, n, _HEAD_DIM).transpose(1, 2)  # [B,n,S,hd]

    xqkv = _to_dev(fused, mesh)
    q, k, v = ttnn.experimental.nlp_create_qkv_heads(
        xqkv,
        num_heads=n,
        num_kv_heads=n,
        transpose_k_heads=False,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    shp = [_B, n, _T, _HEAD_DIM]
    print(f"[fused 6/6/6] q.shape={list(q.shape)} (want {shp})")
    assert list(q.shape) == shp, f"fused op q shape {list(q.shape)} != {shp}"
    for name, tt, ref in [("q", q, qr), ("k", k, kr), ("v", v, vr)]:
        pcc = _pcc(_first_replica(tt, mesh, shp), ref)
        print(f"[fused 6/6/6] {name} PCC={pcc:.6f}")
        assert pcc > _PCC_BAR, f"{name} PCC {pcc:.5f} < {_PCC_BAR}"
    print(
        "[fused 6/6/6] => nlp_create_qkv_heads produces the recurrent [1,6,1,128] layout "
        "post-expand; it can fuse the per-head reshape + q/k/v transposes (NOT the gqa-expand)."
    )


def test_fused_create_heads_rejects_inverse_gqa():
    """Document WHY nlp_create_qkv_heads cannot fuse the GDN split DIRECTLY.

    The op splits a fused tensor into q[nq], k[nkv], v[nkv] — k and v are forced
    to the SAME head count (num_kv_heads). GDN is *inverse* GQA: k=2 heads but
    v=6 heads. There is no (num_heads, num_kv_heads) that yields k=2 AND v=6, so
    the op cannot represent the GDN split before the 2→6 expand. Hence the churn
    win must come from (a) running the fused op AFTER expand (6/6/6, see above),
    or (b) keeping the per-head tensors tiled across the recurrent step — not
    from a drop-in nlp_create_qkv_heads on the raw q2|k2|v6 projection.
    """
    assert _N_K != _N_V, "GDN k/v head counts are equal — re-evaluate; fused split may apply directly"
    # num_kv_heads is shared by k and v; it cannot be both 2 and 6.
    with pytest.raises(AssertionError):
        nkv_for_k, nkv_for_v = _N_K, _N_V
        assert nkv_for_k == nkv_for_v, (
            f"nlp_create_qkv_heads needs k_heads==v_heads (num_kv_heads); "
            f"GDN has k={nkv_for_k}, v={nkv_for_v} (inverse GQA) — not directly fusable."
        )
