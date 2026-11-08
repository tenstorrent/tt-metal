# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""max_batch_size == 1 GDN decode through gdn_decode_step (op on) vs the single-user fused kernels (op off), one real
Qwen3.8 GDN layer: prefill -> decode handoff, snapshot / restore, reset + re-prefill after the op ran, prepare path.

  QWEN36_FABRIC_RING=1 MESH_DEVICE=P150x8 HF_MODEL=... pytest models/demos/blackhole/qwen36/tests/test_gdn_decode_step_b1.py -sv
"""
import os

import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.tests.test_factory import (
    compute_pcc,
    load_gdn_layer,
    model_path,
    parametrize_mesh_tp,
    replicate_to_device,
    shard_to_device,
    tp_composer,
)
from models.demos.blackhole.qwen36.tt.gdn.tp import TPGatedDeltaNet, load_gdn_weights_tp
from models.demos.blackhole.qwen36.tt.model_config import Qwen36ModelArgs

PCC = 0.999


def _cat0(mesh, t):
    return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)).float()


def _hist(mesh, g):
    if g._conv_fmt == "packed":
        g._unpack_conv_hist()  # RM becomes current too; the tag stays "packed"
    return _cat0(mesh, g._conv_hist_rm)


@torch.no_grad()
@parametrize_mesh_tp()
def test_gdn_b1_decode_step_vs_fused(mesh_device, reset_seeds, ensure_gc, monkeypatch):
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
        p_h = compute_pcc(_hist(mesh, go), _hist(mesh, gr))
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
