# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Layer tests of the gdn_decode_step decode path of TPGatedDeltaNet (QWEN36_GDN_DECODE_STEP_OP) on a TP mesh.

  QWEN36_FABRIC_RING=1 MESH_DEVICE=P150x8 HF_MODEL=... pytest models/demos/blackhole/qwen36/tests/test_gdn_decode_step_op.py -sv

Random weights, no checkpoint (needs the model's config.json via HF_MODEL), except test_gdn_b1_decode_step_vs_fused,
which loads one real GDN layer from the checkpoint.
"""
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.tests.test_factory import (
    compute_pcc,
    load_gdn_layer,
    model_path,
    parametrize_mesh_tp,
    random_gdn_state_dict,
    replicate_to_device,
    shard_to_device,
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


def _packed_host(g):
    return ttnn.to_torch(ttnn.get_device_tensors(g._conv_hist_packed)[0]).bfloat16().clone()


def _set_packed(mesh, g, host):
    ttnn.copy(replicate_to_device(mesh, host), g._conv_hist_packed)


def _random_packed_state(mesh, g, Bmax, C):
    """Random RM history -> packed (so only valid rows are non-zero, like the op leaves it), format 'packed'."""
    g.reset_state()
    hist = torch.randn(Bmax, 3, C, dtype=torch.bfloat16)
    ttnn.copy(replicate_to_device(mesh, hist, layout=ttnn.ROW_MAJOR_LAYOUT), g._ensure_conv_hist())
    g._pack_conv_hist()
    g._conv_fmt = "packed"


@torch.no_grad()
@parametrize_mesh_tp()
def test_gdn_packed_write_slot_exact(mesh_device, reset_seeds, ensure_gc, monkeypatch):
    """write_slot under the 'packed' format writes the slot's tiles directly and stays 'packed'; the whole packed
    tensor is bit-identical to the old unpack -> RM slot write -> re-pack path (even and odd slots)."""
    mesh, Bmax = mesh_device, 32
    args, go, _ = _build(mesh, Bmax, monkeypatch)
    Nv, Dk, Dv, C = go.Nv, go.Dk, go.Dv, go.qkv_dim_tp
    _random_packed_state(mesh, go, Bmax, C)
    base = _packed_host(go)
    for slot in (0, 1, 6, 13, 30, 31):
        convs = [replicate_to_device(mesh, torch.randn(1, 1, C, dtype=torch.bfloat16)) for _ in range(4)]
        rec = ttnn.from_torch(
            0.1 * torch.randn(1, Nv, Dk, Dv),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        # new path
        _set_packed(mesh, go, base)
        go._conv_fmt = "packed"
        go.write_slot(slot, rec, convs, consume=False)
        assert go._conv_fmt == "packed"
        new = _packed_host(go)
        # old path: unpack -> RM slot write -> re-pack
        _set_packed(mesh, go, base)
        go._unpack_conv_hist()
        go._conv_fmt = "hist"
        go._write_conv_hist_slot(slot, convs)
        go._pack_conv_hist()
        old = _packed_host(go)
        assert torch.equal(new, old), f"slot {slot}: packed write differs from unpack/write/repack"
        assert not torch.equal(new, base)


@torch.no_grad()
@parametrize_mesh_tp()
def test_gdn_write_slot_inplace_keeps_addresses(mesh_device, reset_seeds, ensure_gc, monkeypatch):
    """The in-place write_slot (L1-sharded sources) never re-seats rec_state / the packed history (decode traces bake
    their addresses) and is bit-exact vs a host reference of the slot write, for every slot, written twice."""
    mesh, Bmax = mesh_device, 32
    args, go, _ = _build(mesh, Bmax, monkeypatch)
    monkeypatch.setenv("QWEN36_GDN_INPLACE_SLOT_WRITE", "1")
    Nv, Dk, Dv, C = go.Nv, go.Dk, go.Dv, go.qkv_dim_tp
    go.reset_state()
    _random_packed_state(mesh, go, Bmax, C)
    go._conv_fmt = "packed"
    rec_addr, pk_addr = go.rec_state.buffer_address(), go._conv_hist_packed.buffer_address()
    ref_rec = ttnn.to_torch(ttnn.get_device_tensors(go.rec_state)[0]).clone()
    for rep in range(2):
        for slot in range(Bmax):
            convs = [replicate_to_device(mesh, torch.randn(1, 1, C, dtype=torch.bfloat16)) for _ in range(4)]
            rec_h = 0.1 * torch.randn(1, Nv, Dk, Dv)
            rec = ttnn.from_torch(
                rec_h,
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )
            pre = _packed_host(go)
            go.write_slot(slot, rec, convs, consume=False)
            ref_rec[slot] = rec_h[0].float()
            assert go.rec_state.buffer_address() == rec_addr, f"rep {rep} slot {slot}: rec_state moved"
            assert go._conv_hist_packed.buffer_address() == pk_addr, f"rep {rep} slot {slot}: packed history moved"
            new = _packed_host(go)
            # reference: the same slot through the old unpack -> RM slot write -> re-pack path, from the pre-write state
            _set_packed(mesh, go, pre)
            go._unpack_conv_hist()
            go._conv_fmt = "hist"
            go._write_conv_hist_slot(slot, convs)
            go._pack_conv_hist()
            go._conv_fmt = "packed"
            assert torch.equal(new, _packed_host(go)), f"rep {rep} slot {slot}: packed write differs from reference"
    got_rec = ttnn.to_torch(ttnn.get_device_tensors(go.rec_state)[0])
    assert torch.equal(got_rec, ref_rec), "rec_state differs from the host reference"
    for t in ttnn.get_device_tensors(go.rec_state)[1:]:
        assert torch.equal(ttnn.to_torch(t), got_rec)


@torch.no_grad()
@parametrize_mesh_tp()
def test_gdn_packed_remap_exact(mesh_device, reset_seeds, ensure_gc, monkeypatch):
    """remap_slots under 'packed' (one parity-aware gather) is bit-identical to unpack -> RM gather -> re-pack:
    odd<->even swaps, identity entries, shifts (condense) and duplicated sources."""
    mesh, Bmax = mesh_device, 32
    args, go, _ = _build(mesh, Bmax, monkeypatch)
    C = go.qkv_dim_tp
    _random_packed_state(mesh, go, Bmax, C)
    base = _packed_host(go)
    swap = list(range(Bmax))
    swap[0], swap[1], swap[2], swap[3] = 1, 0, 3, 2
    cond = [i + 1 if i < 31 else 31 for i in range(Bmax)]  # shift down by one (odd->even, even->odd)
    cond2 = [(i + 2) % Bmax if i < 20 else i for i in range(Bmax)]  # even->even with identity tail
    rev = list(reversed(range(Bmax)))
    perm = torch.randperm(Bmax).tolist()
    for name, remap in (("swap", swap), ("condense", cond), ("shift2", cond2), ("reverse", rev), ("random", perm)):
        _set_packed(mesh, go, base)
        go._conv_fmt = "packed"
        go.remap_slots(remap)
        assert go._conv_fmt == "packed"
        new = _packed_host(go)
        _set_packed(mesh, go, base)
        go._unpack_conv_hist()
        go._gather_indices(go._conv_hist_rm, remap, dim=0)
        go._pack_conv_hist()
        old = _packed_host(go)
        assert torch.equal(new, old), f"remap {name}: packed gather differs from unpack/gather/repack"


@torch.no_grad()
@parametrize_mesh_tp()
def test_gdn_rec_row_remap_exact(mesh_device, reset_seeds, ensure_gc, monkeypatch):
    """In-place rec_state row remap is bit-exact on every row (an idle row may hold an unscheduled request), keeps the
    buffer address, and compiles nothing after the warm reversal."""
    import random

    mesh, Bmax = mesh_device, 32
    args, go, _ = _build(mesh, Bmax, monkeypatch)
    nd = mesh.get_num_devices()
    Nv, Dk, Dv, C = go.Nv, go.Dk, go.Dv, go.qkv_dim_tp
    rec0 = [torch.randn(1, Nv, Dk, Dv) for _ in range(Bmax)]
    hist0 = [torch.randn(1, 3, C, dtype=torch.bfloat16) for _ in range(Bmax)]
    _seed(mesh, args, go, rec0, hist0)
    assert go._rec_row_remap_ok()
    addr = go.rec_state.buffer_address()

    def host():
        return _cat0(mesh, go.rec_state).reshape(nd, Bmax, Nv, Dk, Dv)

    ref = host().clone()
    # warm: full reversal compiles every 1-row read / write (all 2-cycles)
    rev = list(range(Bmax - 1, -1, -1))
    go.remap_slots(rev)
    ref = ref[:, rev]
    assert torch.equal(host(), ref), "reversal"
    entries = mesh.num_program_cache_entries()

    rng = random.Random(1234)
    for it in range(20):
        k = rng.randint(1, 8)
        finished = set(rng.sample(range(Bmax), k))
        n_live = Bmax - k
        survivors = [r for r in range(Bmax) if r not in finished]
        # vLLM condense: move the highest live rows into the lowest holes
        row_slots = list(range(Bmax))
        holes = sorted(finished)
        live_rows = set(survivors)
        high = sorted((r for r in live_rows if r >= n_live), reverse=True)
        low_holes = [h for h in holes if h < n_live]
        for h, r in zip(low_holes, high):
            row_slots[h] = r
        free = sorted(set(range(Bmax)) - set(row_slots[:n_live]))
        remap = row_slots[:n_live] + free
        assert sorted(remap) == list(range(Bmax))
        go.remap_slots(remap)
        ref = ref[:, remap]
        assert torch.equal(host(), ref), f"condense {it} (k={k}): rows differ"
        assert go.rec_state.buffer_address() == addr

    for it in range(5):
        perm = torch.randperm(Bmax).tolist()
        go.remap_slots(perm)
        ref = ref[:, perm]
        assert torch.equal(host(), ref), f"random perm {it}: rows differ"
        assert go.rec_state.buffer_address() == addr

    assert mesh.num_program_cache_entries() == entries, "remap compiled new programs after the warm reversal"


def _hist_b1(mesh, g):
    if g._conv_fmt == "packed":
        g._unpack_conv_hist()  # RM becomes current too; the tag stays "packed"
    return _cat0(mesh, g._conv_hist_rm)


@torch.no_grad()
@parametrize_mesh_tp()
def test_gdn_b1_decode_step_vs_fused(mesh_device, reset_seeds, ensure_gc, monkeypatch):
    """B=1 decode, gdn_decode_step vs the single-user fused kernels on one real GDN layer: prefill handoff,
    snapshot / restore, reset + re-prefill, prepare path."""
    mesh = mesh_device
    os.environ.setdefault("HF_MODEL", model_path())
    args = Qwen36ModelArgs(mesh, max_batch_size=1, max_seq_len=512)
    li = next(i for i, t in enumerate(args.attention_type_list) if t == "linear_attention")
    sd = load_gdn_layer(args.CKPT_DIR, li)
    from models.tt_transformers.tt.ccl import TT_CCL

    tt_ccl = TT_CCL(mesh) if mesh.get_num_devices() > 1 else None
    monkeypatch.setenv("QWEN36_GDN_FUSED_DECODE", "1")
    monkeypatch.setenv("QWEN36_GDN_DECODE_STEP_OP", "1")
    tw = load_gdn_weights_tp(mesh, sd, args)
    go = TPGatedDeltaNet(mesh, args, tw, tt_ccl)
    monkeypatch.setenv("QWEN36_GDN_DECODE_STEP_OP", "0")
    gr = TPGatedDeltaNet(mesh, args, tw, tt_ccl)
    assert go._decode_op and not go._fused_batched, "gdn_decode_step not enabled at max_batch_size == 1"
    assert not gr._decode_op and gr._fused_decode, "reference is not the single-user fused path"
    comp = tp_composer(mesh)
    T = 128

    def prefill(x):
        for g in (go, gr):
            g.forward_prefill(shard_to_device(mesh, x, dim=-1), chunk_size=T, capture_state=True)
        assert go._conv_fmt == "packed", f"prefill left the op layer in format {go._conv_fmt!r}"

    def decode(tag, x):
        oo = ttnn.to_torch(go.forward_decode(replicate_to_device(mesh, x)), mesh_composer=comp).float()
        orr = ttnn.to_torch(gr.forward_decode(replicate_to_device(mesh, x)), mesh_composer=comp).float()
        assert torch.isfinite(oo).all(), f"{tag}: non-finite op output"
        p_out = compute_pcc(oo, orr)
        p_rec = compute_pcc(_cat0(mesh, go.rec_state), _cat0(mesh, gr.rec_state))
        p_h = compute_pcc(_hist_b1(mesh, go), _hist_b1(mesh, gr))
        logger.info(f"B1CHECK {tag}: out PCC {p_out:.6f}  rec_state PCC {p_rec:.6f}  conv hist PCC {p_h:.6f}")
        assert p_out >= PCC and p_rec >= PCC and p_h >= PCC, f"{tag}: below PCC {PCC}"
        return oo

    def snapshot(g):
        return _cat0(mesh, g.rec_state), g.snapshot_fused_decode_state()

    def restore(g, snap):
        rec, hist = snap
        r = ttnn.from_torch(
            rec,
            dtype=g.rec_state.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
        )
        ttnn.copy(r, g.rec_state)
        ttnn.deallocate(r)
        g.restore_fused_decode_state(hist)

    xs = lambda: torch.randn(1, 1, 1, args.dim, dtype=torch.bfloat16)  # noqa: E731

    # 1) prefill -> decode handoff
    for g in (go, gr):
        g.reset_state()
    prefill(torch.randn(1, 1, T, args.dim, dtype=torch.bfloat16))
    for s in range(4):
        decode(f"seq1 step{s}", xs())

    # 2) snapshot -> 2 steps -> restore -> the same 2 steps (op path must reproduce itself exactly)
    snaps = (snapshot(go), snapshot(gr))
    xa = [xs(), xs()]
    first = [decode(f"pre-restore step{i}", x) for i, x in enumerate(xa)]
    restore(go, snaps[0])
    restore(gr, snaps[1])
    assert go._conv_fmt == "packed", f"restore left format {go._conv_fmt!r}"
    assert go.prepare_decode_width(1) is False, "prepare re-packed an already packed history"
    again = [decode(f"post-restore step{i}", x) for i, x in enumerate(xa)]
    for i, (a, b) in enumerate(zip(first, again)):
        assert torch.equal(a, b), f"op path not reproducible after restore (step {i})"

    # 3) new sequence after the op ran: reset in place + re-prefill must refresh the packed history
    for g in (go, gr):
        g.reset_state_inplace()
    assert go._conv_fmt == "both"
    prefill(torch.randn(1, 1, T, args.dim, dtype=torch.bfloat16))
    for s in range(3):
        decode(f"seq2 step{s}", xs())

    # 4) prepare path: a reset leaves "both"; prepare_decode_width(1) packs eagerly
    for g in (go, gr):
        g.reset_state_inplace()
    assert go.prepare_decode_width(1) is True and go._conv_fmt == "packed"
    decode("from-zero step0", xs())
    logger.info("B1CHECK PASSED")
