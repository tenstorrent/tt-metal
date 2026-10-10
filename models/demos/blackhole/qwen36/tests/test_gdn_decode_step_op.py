# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Layer tests of the gdn_decode_step decode path of TPGatedDeltaNet (QWEN36_GDN_DECODE_STEP_OP) on a TP mesh.

  QWEN36_FABRIC_RING=1 MESH_DEVICE=P150x8 HF_MODEL=... pytest models/demos/blackhole/qwen36/tests/test_gdn_decode_step_op.py -sv

Random weights, no checkpoint (needs the model's config.json via HF_MODEL).
"""
import os

import pytest
import torch

import ttnn
from models.demos.blackhole.qwen36.tests.test_factory import (
    compute_pcc,
    model_path,
    parametrize_mesh_tp,
    random_gdn_state_dict,
    replicate_to_device,
    tp_composer,
)
from models.demos.blackhole.qwen36.tt.gdn.tp import TPGatedDeltaNet, load_gdn_weights_tp, pack_head_tiles
from models.demos.blackhole.qwen36.tt.model_config import Qwen36ModelArgs

PCC = 0.999


def _cat0(mesh, t):
    return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)).float()


def _build(mesh, Bmax, monkeypatch):
    os.environ.setdefault("HF_MODEL", model_path())
    args = Qwen36ModelArgs(mesh, max_batch_size=Bmax, max_seq_len=256)
    li = next(i for i, t in enumerate(args.attention_type_list) if t == "linear_attention")
    sd = random_gdn_state_dict(args, seed=li)
    from models.tt_transformers.tt.ccl import TT_CCL

    tt_ccl = TT_CCL(mesh) if mesh.get_num_devices() > 1 else None
    monkeypatch.setenv("QWEN36_GDN_FUSED_DECODE", "1")
    monkeypatch.setenv("QWEN36_GDN_DECODE_STEP_OP", "1")
    tw = load_gdn_weights_tp(mesh, sd, args)
    go = TPGatedDeltaNet(mesh, args, tw, tt_ccl)
    monkeypatch.setenv("QWEN36_GDN_DECODE_STEP_OP", "0")
    gr = TPGatedDeltaNet(mesh, args, tw, tt_ccl)
    assert go._decode_op and not gr._decode_op and gr._fused_batched
    return args, go, gr


def _seed(mesh, args, g, rec0, hist0):
    Bmax = len(rec0)
    g._stable_state = True
    g.reset_state()
    rec = [
        ttnn.from_torch(
            r, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh)
        )
        for r in rec0
    ]
    g.assemble_batched_state(rec, [replicate_to_device(mesh, h) for h in hist0])
    assert len(rec) == Bmax


def _hist_rm(mesh, g, nd, Bmax, C):
    """The conv history [nd, Bmax, 3, C] in the canonical RM format, whatever the current format is."""
    if g._conv_fmt == "packed":
        g._unpack_conv_hist()
        return _cat0(mesh, g._conv_hist_rm).reshape(nd, Bmax, 3, C)
    if g._conv_fmt == "states":
        return torch.stack([_cat0(mesh, g.conv_states[m]).reshape(nd, Bmax, C) for m in (1, 2, 3)], dim=2)
    return _cat0(mesh, g._conv_hist_rm).reshape(nd, Bmax, 3, C)


@torch.no_grad()
@parametrize_mesh_tp()
def test_gdn_pack_roundtrip(mesh_device, reset_seeds, ensure_gc, monkeypatch):
    """RM history -> packed (embedding gather) matches the host packer the kernel expects (slot-parity aware), and
    packed -> RM restores the history exactly."""
    mesh, Bmax = mesh_device, 32
    args, go, _ = _build(mesh, Bmax, monkeypatch)
    nd = mesh.get_num_devices()
    Nv, Nk, Dk, Dv, C = go.Nv, go.Nk, go.Dk, go.Dv, go.qkv_dim_tp
    go.reset_state()
    hist = torch.randn(Bmax, 3, C, dtype=torch.bfloat16)
    ttnn.copy(replicate_to_device(mesh, hist, layout=ttnn.ROW_MAJOR_LAYOUT), go._ensure_conv_hist())
    go._pack_conv_hist()
    packed = ttnn.to_torch(ttnn.get_device_tensors(go._conv_hist_packed)[0])  # [Bmax, Nv, 4, 32, 32]
    exp = torch.stack(
        [
            pack_head_tiles([torch.zeros(C)] + [hist[b, j] for j in range(3)], Nv, Nk, Dk, Dv, parity=b & 1)
            for b in range(Bmax)
        ]
    )
    exp[:, :, 0] = 0
    # the op never reads slot 0 / the other parity; the gather writes zeros there (== the host packer with a zero row)
    assert torch.equal(packed.bfloat16(), exp), "packed history differs from the host packer"
    ttnn.copy(replicate_to_device(mesh, torch.zeros_like(hist), layout=ttnn.ROW_MAJOR_LAYOUT), go._conv_hist_rm)
    go._unpack_conv_hist()
    back = _cat0(mesh, go._conv_hist_rm).reshape(nd, Bmax, 3, C)
    assert torch.equal(back[0], hist.float()), "packed -> RM round trip is not exact"


@torch.no_grad()
@parametrize_mesh_tp()
@pytest.mark.parametrize("B", [1, 8, 32])
def test_gdn_decode_step_op_vs_existing(mesh_device, B, reset_seeds, ensure_gc, monkeypatch):
    """B decode steps through the op path vs the existing path (gate off: fused-batched for B <= SCAN_MAX, the
    original decode above), from identical random state (Bmax = 32). Between the steps: write_slot of slot 1 and a
    remap_slots that swaps slots (0,1) and (2,3) (odd <-> even: the packed-row parity flips)."""
    mesh, Bmax = mesh_device, 32
    args, go, gr = _build(mesh, Bmax, monkeypatch)
    nd = mesh.get_num_devices()
    Nv, Dk, Dv, C = go.Nv, go.Dk, go.Dv, go.qkv_dim_tp
    rec0 = [0.1 * torch.randn(1, Nv, Dk, Dv) for _ in range(Bmax)]
    hist0 = [torch.randn(1, 3, C, dtype=torch.bfloat16) for _ in range(Bmax)]
    _seed(mesh, args, go, rec0, hist0)
    _seed(mesh, args, gr, rec0, hist0)

    slot_rec = 0.1 * torch.randn(1, Nv, Dk, Dv)
    slot_convs = [torch.randn(1, 1, C, dtype=torch.bfloat16) for _ in range(4)]
    remap = list(range(Bmax))
    remap[0], remap[1], remap[2], remap[3] = 1, 0, 3, 2

    def do_write_slot(g):
        rec = ttnn.from_torch(
            slot_rec,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        g.write_slot(1, rec, [replicate_to_device(mesh, c) for c in slot_convs])

    def compare(tag, out_o=None, out_r=None):
        rec_o = _cat0(mesh, go.rec_state).reshape(nd, Bmax, Nv, Dk, Dv)
        rec_r = _cat0(mesh, gr.rec_state).reshape(nd, Bmax, Nv, Dk, Dv)
        h_o = _hist_rm(mesh, go, nd, Bmax, C)
        h_r = _hist_rm(mesh, gr, nd, Bmax, C)
        for u in range(B):
            p_rec, p_hist = compute_pcc(rec_o[:, u], rec_r[:, u]), compute_pcc(h_o[:, u], h_r[:, u])
            assert p_rec >= PCC, f"{tag} user {u}: rec_state PCC {p_rec:.5f}"
            assert p_hist >= PCC, f"{tag} user {u}: conv history PCC {p_hist:.5f}"
        # idle users must not move in the op path
        assert torch.isfinite(rec_o).all() and torch.isfinite(h_o).all()
        if out_o is not None:
            worst = min(compute_pcc(out_o[0, 0, u], out_r[0, 0, u]) for u in range(B))
            assert worst >= PCC, f"{tag}: output PCC {worst:.5f}"
            print(f"{tag}: worst out PCC {worst:.5f}")

    comp = tp_composer(mesh)
    for step in range(5):
        if step == 2:
            do_write_slot(go)
            do_write_slot(gr)
            compare("after write_slot")
        if step == 3:
            go.prepare_decode_width(B)  # vLLM order: prepare, then the deferred remap, then the decode
            gr.prepare_decode_width(B)
            go.remap_slots(remap)
            gr.remap_slots(remap)
            compare("after remap")
        go.prepare_decode_width(B)
        gr.prepare_decode_width(B)
        assert go._conv_fmt == "packed"
        rec_idle = _cat0(mesh, go.rec_state).reshape(nd, Bmax, Nv, Dk, Dv)[:, B:].clone()
        x = torch.randn(1, 1, B, args.dim, dtype=torch.bfloat16)
        out_o = ttnn.to_torch(go.forward_decode(replicate_to_device(mesh, x)), mesh_composer=comp).float()
        out_r = ttnn.to_torch(gr.forward_decode(replicate_to_device(mesh, x)), mesh_composer=comp).float()
        assert torch.isfinite(out_o).all()
        assert torch.equal(_cat0(mesh, go.rec_state).reshape(nd, Bmax, Nv, Dk, Dv)[:, B:], rec_idle), "idle rows moved"
        compare(f"step {step} (B={B})", out_o, out_r)
