# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""TP validation for Qwen3.5/3.6 Gated DeltaNet on a Blackhole mesh.

One file per component (decode / chunk-prefill), sharing the loaders and mesh
parametrization from ``test_factory``:

* ``test_gdn_tp``         — decode PCC @ pos0 (recurrent state starts at zero, so
  o = beta*(q̂·k̂)*v); the torch reference covers the sharded QKV/Z/AB reorder,
  per-channel conv, GQA head expansion, L2 norm, gated RMSNorm, Z-gate, output
  projection, and reduce-scatter. Plus a second decode step for shape/NaN.
* ``test_gdn_tp_prefill`` — chunk-prefill (FIR conv + shared chunk kernel) must
  agree with step-by-step decode over the same tokens (zero init state). An
  internal-consistency check across two code paths; no hand-written reference.

Run:
    MESH_DEVICE=P150x4 HF_MODEL=Qwen/Qwen3.6-27B \
      pytest models/demos/blackhole/qwen36/tests/test_gdn_tp.py -v -s
"""
import os

import torch
import torch.nn.functional as F
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.blackhole.qwen36.tests.test_factory import (
    compute_pcc,
    get_pcc_threshold,
    load_gdn_layer,
    model_path,
    parametrize_batch,
    parametrize_mesh_tp,
    replicate_to_device,
    shard_to_device,
    tp_composer,
)
from models.demos.blackhole.qwen36.tt.gdn.tp import TPGatedDeltaNet, load_gdn_weights_tp
from models.demos.blackhole.qwen36.tt.model_config import Qwen36ModelArgs


@torch.no_grad()
@parametrize_mesh_tp()
@parametrize_batch()
def test_gdn_tp(mesh_device, B, reset_seeds, ensure_gc, request):
    """Validate TP decode output against a hand-written PyTorch reference at pos0 (batch sweep).

    Checks PCC for the full GDN forward pass (QKV proj, conv tap, L2 norm, beta gating,
    gated RMSNorm, output proj) and runs a second decode step to catch shape/NaN regressions.
    """
    os.environ.setdefault("HF_MODEL", model_path())
    args = Qwen36ModelArgs(mesh_device, max_batch_size=B, max_seq_len=256)
    nd = mesh_device.get_num_devices()
    li = next(i for i, t in enumerate(args.attention_type_list) if t == "linear_attention")
    logger.info(f"devices={nd} gdn layer={li} Nk_tp={args.gdn_nk_tp} Nv_tp={args.gdn_nv_tp}")

    # args.CKPT_DIR is the resolved local snapshot dir (Qwen36ModelArgs downloads the hub id).
    sd = load_gdn_layer(args.CKPT_DIR, li)
    from models.tt_transformers.tt.ccl import TT_CCL

    tt_ccl = TT_CCL(mesh_device) if nd > 1 else None
    tw = load_gdn_weights_tp(mesh_device, sd, args)
    gdn = TPGatedDeltaNet(mesh_device, args, tw, tt_ccl)

    x = torch.randn(1, 1, B, args.dim, dtype=torch.bfloat16)
    x_tt = replicate_to_device(mesh_device, x)
    out = gdn.forward_decode(x_tt)
    out_t = ttnn.to_torch(out, mesh_composer=tp_composer(mesh_device))[0, 0].float()
    assert out_t.shape[-1] == args.dim and not torch.isnan(out_t).any() and out_t.abs().max() > 0

    # ---- torch reference @ pos0 (full, unsharded) ----
    Nk, Nv, Dk, Dv = args.gdn_nk, args.gdn_nv, args.gdn_dk, args.gdn_dv
    key_dim, value_dim = args.gdn_key_dim, args.gdn_value_dim
    xf = x[0, 0].float()
    qkv = xf @ sd["linear_attn.in_proj_qkv.weight"].float().T  # [B, 2*key_dim+value_dim]
    z = xf @ sd["linear_attn.in_proj_z.weight"].float().T
    b = xf @ sd["linear_attn.in_proj_b.weight"].float().T  # [B, Nv]
    tap3 = sd["linear_attn.conv1d.weight"].float()[:, 0, 3]  # [qkv_dim], newest-token tap
    conv = F.silu(qkv * tap3)
    q = conv[:, :key_dim].reshape(B, Nk, Dk)
    k = conv[:, key_dim : 2 * key_dim].reshape(B, Nk, Dk)
    v = conv[:, 2 * key_dim :].reshape(B, Nv, Dv)
    rf = Nv // Nk
    q = q.repeat_interleave(rf, dim=1)
    k = k.repeat_interleave(rf, dim=1)
    q = F.normalize(q, dim=-1) * (Dk**-0.5)
    k = F.normalize(k, dim=-1)
    beta = torch.sigmoid(b)  # [B, Nv]
    qk = (q * k).sum(-1)  # [B, Nv]
    o = beta[..., None] * qk[..., None] * v  # [B, Nv, Dv]
    # gated RMSNorm over Dv (weight only, NO +1)
    o_n = o / torch.sqrt(o.pow(2).mean(-1, keepdim=True) + 1e-6) * sd["linear_attn.norm.weight"].float()
    gated = (o_n * F.silu(z.reshape(B, Nv, Dv))).reshape(B, value_dim)
    ref = gated @ sd["linear_attn.out_proj.weight"].float().T  # [B, dim]

    # Per-row PCC: x has distinct random content per user, so a flattened/aggregate PCC over the
    # whole [B, dim] tensor could mask a single contaminated row.
    thr = get_pcc_threshold(request)
    pccs = [compute_pcc(ref[u], out_t[u]) for u in range(B)]
    worst = min(pccs)
    logger.info(f"GDN TP PCC (pos0) min={worst:.5f} max={max(pccs):.5f}")
    bad = [(u, p) for u, p in enumerate(pccs) if p < thr]
    assert not bad, f"users below PCC {thr}: {bad}"

    x2 = replicate_to_device(mesh_device, torch.randn(1, 1, B, args.dim, dtype=torch.bfloat16))
    out2 = gdn.forward_decode(x2)
    out2_t = ttnn.to_torch(out2, mesh_composer=tp_composer(mesh_device))
    assert not torch.isnan(out2_t).any() and out2_t.abs().max() > 0
    logger.info("PASSED: GDN TP decode (pos0 PCC + pos1 shape/NaN)")


@torch.no_grad()
@parametrize_mesh_tp()
@parametrize_batch(batches=(8, 32))
def test_gdn_tp_peruser_state(mesh_device, B, reset_seeds, ensure_gc, request):
    """Per-user GDN prefill stitched into the batched decode state.

    B users are prefilled independently via forward_prefill(return_state=True);
    assemble_batched_state stitches each user's recurrent + conv state into row u of the
    batched buffers. A single batched decode must then match, row-by-row, B independent B=1
    prefill+decode runs, proving correct row assembly with no cross-user contamination.
    """
    os.environ.setdefault("HF_MODEL", model_path())
    args = Qwen36ModelArgs(mesh_device, max_batch_size=B, max_seq_len=256)
    # forward_decode keys all shapes off self.B, so the B=1 reference needs its own
    # max_batch_size=1 args (weights tw are batch-independent and shared).
    args1 = Qwen36ModelArgs(mesh_device, max_batch_size=1, max_seq_len=256)
    nd = mesh_device.get_num_devices()
    li = next(i for i, t in enumerate(args.attention_type_list) if t == "linear_attention")
    logger.info(f"devices={nd} gdn layer={li} B={B}")

    sd = load_gdn_layer(args.CKPT_DIR, li)
    from models.tt_transformers.tt.ccl import TT_CCL

    tt_ccl = TT_CCL(mesh_device) if nd > 1 else None
    tw = load_gdn_weights_tp(mesh_device, sd, args)
    comp = tp_composer(mesh_device)
    T = 128  # one prefill chunk (gated_delta_attn_seq kernel chunk_size)

    xp = [torch.randn(1, 1, T, args.dim, dtype=torch.bfloat16) for _ in range(B)]
    xd = [torch.randn(1, 1, 1, args.dim, dtype=torch.bfloat16) for _ in range(B)]

    # ---- reference: B independent B=1 prefill (capture_state) + decode ----
    ref_rows = []
    for u in range(B):
        g = TPGatedDeltaNet(mesh_device, args1, tw, tt_ccl)
        g.reset_state()
        g.forward_prefill(shard_to_device(mesh_device, xp[u], dim=-1), chunk_size=T, capture_state=True)
        out_u = g.forward_decode(replicate_to_device(mesh_device, xd[u]))
        ref_rows.append(ttnn.to_torch(out_u, mesh_composer=comp)[0, 0, 0].float())

    # ---- batched: per-user prefill(return_state) -> assemble -> single batched decode ----
    gb = TPGatedDeltaNet(mesh_device, args, tw, tt_ccl)
    rec_list, conv_list = [], []
    for u in range(B):
        _, rec_u, conv_u = gb.forward_prefill(
            shard_to_device(mesh_device, xp[u], dim=-1), chunk_size=T, return_state=True
        )
        rec_list.append(rec_u)
        conv_list.append(conv_u)
    gb.assemble_batched_state(rec_list, conv_list)
    x_dec = torch.cat(xd, dim=2)  # [1, 1, B, dim], row u = user u's decode token
    out_b = gb.forward_decode(replicate_to_device(mesh_device, x_dec))
    out_t = ttnn.to_torch(out_b, mesh_composer=comp)  # [1, 1, B, dim]

    # ---- per-row comparison (flattened PCC would mask a single contaminated user) ----
    thr = get_pcc_threshold(request)
    pccs = [compute_pcc(ref_rows[u], out_t[0, 0, u].float()) for u in range(B)]
    worst = min(pccs)
    logger.info(f"per-user GDN state (B={B}) PCC min={worst:.5f} max={max(pccs):.5f}")
    bad = [(u, p) for u, p in enumerate(pccs) if p < thr]
    assert not bad, f"users below PCC {thr}: {bad}"
    logger.info(f"PASSED: per-user GDN state (B={B}) worst PCC = {worst:.5f}")


@torch.no_grad()
@parametrize_mesh_tp()
# Batches capped at (2, 4): the gated_delta_attn_seq kernel maps one BH = B*Nv_tp row per
# core and is L1-bound, so BH must stay <= ~32 (at TP=4/Nv_tp=8, B=4 -> BH=32 fits; B>=8
# clashes/trips the BH <= compute_grid assert). Batched prefill itself is bit-exact (PCC 1.0);
# serving B=32 would need grouped launches, so the model still prefills per-user.
@parametrize_batch(batches=(2, 4))
def test_gdn_tp_batched_prefill(mesh_device, B, reset_seeds, ensure_gc, request):
    """True batched GDN prefill (one pass over all B users) vs B independent B=1 prefills.

    Each user has a distinct length (padded to a common bucket + per-row valid_len) and distinct
    content. forward_prefill_batched runs the projection / conv-FIR / chunk-parallel recurrence over
    the whole [B,T] batch in one shot and writes the batched decode state directly. A batched decode
    must then match, row-by-row, B independent B=1 prefill+decode runs, proving the chunk-seq kernel
    batches correctly with per-row masking. B capped at <=4 (see kernel BH limit note above).
    """
    os.environ.setdefault("HF_MODEL", model_path())
    args = Qwen36ModelArgs(mesh_device, max_batch_size=B, max_seq_len=256)
    args1 = Qwen36ModelArgs(mesh_device, max_batch_size=1, max_seq_len=256)
    nd = mesh_device.get_num_devices()
    li = next(i for i, t in enumerate(args.attention_type_list) if t == "linear_attention")
    logger.info(f"devices={nd} gdn layer={li} B={B}")

    sd = load_gdn_layer(args.CKPT_DIR, li)
    from models.tt_transformers.tt.ccl import TT_CCL

    tt_ccl = TT_CCL(mesh_device) if nd > 1 else None
    tw = load_gdn_weights_tp(mesh_device, sd, args)
    comp = tp_composer(mesh_device)

    Tmax = 128  # common bucket (one fused-chunk kernel bucket; must be a 32-multiple)
    # Distinct real lengths, each a 32-multiple > TILE_SIZE: the fused chunk op requires the per-call
    # bucket T to be a multiple of the fused chunk size (32), and the B=1 reference routes S<=32 to the
    # decode matmul (replicated input) — so keep lens in {64, 96, 128}. The batched path pads to Tmax
    # and masks via valid_lens; the reference runs each user at its own length.
    lens = [Tmax - 32 * (u % 3) for u in range(B)]  # {128, 96, 64}
    xp = [torch.randn(1, 1, lens[u], args.dim, dtype=torch.bfloat16) for u in range(B)]
    xd = [torch.randn(1, 1, 1, args.dim, dtype=torch.bfloat16) for u in range(B)]

    # ---- reference: B independent B=1 prefill(capture_state) + decode ----
    ref_rows = []
    for u in range(B):
        g = TPGatedDeltaNet(mesh_device, args1, tw, tt_ccl)
        g.reset_state()
        g.forward_prefill(shard_to_device(mesh_device, xp[u], dim=-1), chunk_size=Tmax, capture_state=True)
        out_u = g.forward_decode(replicate_to_device(mesh_device, xd[u]))
        ref_rows.append(ttnn.to_torch(out_u, mesh_composer=comp)[0, 0, 0].float())

    # ---- batched: pad each user to Tmax, ONE batched prefill, then a batched decode step ----
    gb = TPGatedDeltaNet(mesh_device, args, tw, tt_ccl)
    gb.reset_state()
    x_pad = torch.zeros(B, Tmax, args.dim, dtype=torch.bfloat16)
    for u in range(B):
        x_pad[u, : lens[u], :] = xp[u][0, 0]
    gb.forward_prefill_batched(shard_to_device(mesh_device, x_pad, dim=-1), chunk_size=Tmax, valid_lens=lens)
    x_dec = torch.cat(xd, dim=2)  # [1, 1, B, dim]
    out_b = gb.forward_decode(replicate_to_device(mesh_device, x_dec))
    out_t = ttnn.to_torch(out_b, mesh_composer=comp)  # [1, 1, B, dim]

    thr = get_pcc_threshold(request)
    pccs = [compute_pcc(ref_rows[u], out_t[0, 0, u].float()) for u in range(B)]
    worst = min(pccs)
    logger.info(f"batched GDN prefill (B={B}) PCC min={worst:.5f} max={max(pccs):.5f} lens={lens}")
    bad = [(u, lens[u], p) for u, p in enumerate(pccs) if p < thr]
    assert not bad, f"users below PCC {thr}: {bad}"
    logger.info(f"PASSED: batched GDN prefill (B={B}) worst PCC = {worst:.5f}")


@torch.no_grad()
@parametrize_mesh_tp()
@parametrize_batch(batches=(2,))
def test_gdn_tp_batched_prefill_chunked(mesh_device, B, reset_seeds, ensure_gc, request):
    """Chunk-outer BATCHED GDN prefill (forward_prefill_batched carry=True) == single-shot.

    Prefilling a 2-chunk sequence as TWO carried chunks must match prefilling it in ONE call
    (the kernel runs <=16 sub-chunks per call, so the single-shot is the ground truth). Validates
    the batched cross-chunk recurrent + conv-state carry in isolation — the foundation for grouped
    long-context batched prefill.
    """
    os.environ.setdefault("HF_MODEL", model_path())
    args = Qwen36ModelArgs(mesh_device, max_batch_size=B, max_seq_len=256)
    nd = mesh_device.get_num_devices()
    li = next(i for i, t in enumerate(args.attention_type_list) if t == "linear_attention")
    sd = load_gdn_layer(args.CKPT_DIR, li)
    from models.tt_transformers.tt.ccl import TT_CCL

    tt_ccl = TT_CCL(mesh_device) if nd > 1 else None
    tw = load_gdn_weights_tp(mesh_device, sd, args)
    comp = tp_composer(mesh_device)

    C = 128  # GDN kernel chunk size
    T = 2 * C  # two full chunks
    x = torch.randn(B, T, args.dim, dtype=torch.bfloat16)
    xd = torch.randn(1, 1, B, args.dim, dtype=torch.bfloat16)  # one decode token per user

    # ---- reference: single-shot batched prefill over the full T (ground truth) ----
    gref = TPGatedDeltaNet(mesh_device, args, tw, tt_ccl)
    gref.reset_state()
    gref._stable_state = True
    gref.forward_prefill_batched(shard_to_device(mesh_device, x.unsqueeze(0), dim=-1), chunk_size=C)
    out_ref = ttnn.to_torch(gref.forward_decode(replicate_to_device(mesh_device, xd)), mesh_composer=comp)

    # ---- test: two CARRIED chunks ----
    g = TPGatedDeltaNet(mesh_device, args, tw, tt_ccl)
    g.reset_state()
    g._stable_state = True
    g.reset_state_inplace()  # zero state + clear the batched conv carry at sequence start
    g.forward_prefill_batched(shard_to_device(mesh_device, x[:, :C].unsqueeze(0), dim=-1), chunk_size=C, carry=True)
    g.forward_prefill_batched(shard_to_device(mesh_device, x[:, C:].unsqueeze(0), dim=-1), chunk_size=C, carry=True)
    out_t = ttnn.to_torch(g.forward_decode(replicate_to_device(mesh_device, xd)), mesh_composer=comp)

    thr = get_pcc_threshold(request, default=0.99)
    pccs = [compute_pcc(out_ref[0, 0, u].float(), out_t[0, 0, u].float()) for u in range(B)]
    worst = min(pccs)
    logger.info(f"batched chunk-outer carry (B={B}) PCC min={worst:.5f} max={max(pccs):.5f}")
    assert worst >= thr, f"carry vs single-shot PCC {worst:.5f} < {thr}: {pccs}"
    logger.info(f"PASSED: batched chunk-outer GDN prefill carry (B={B}) worst PCC = {worst:.5f}")


@torch.no_grad()
@parametrize_mesh_tp()
def test_gdn_tp_prefill(mesh_device, reset_seeds, ensure_gc, request):
    """Check that chunk-prefill and step-by-step decode agree on the same T=128 tokens.

    Both paths start from zero state. No hand-written reference — this is a
    self-consistency check between forward_prefill and forward_decode.
    """
    os.environ.setdefault("HF_MODEL", model_path())
    T = 128
    args = Qwen36ModelArgs(mesh_device, max_batch_size=1, max_seq_len=256)
    nd = mesh_device.get_num_devices()
    li = next(i for i, t in enumerate(args.attention_type_list) if t == "linear_attention")
    logger.info(f"devices={nd} gdn layer={li} T={T}")

    sd = load_gdn_layer(args.CKPT_DIR, li)
    from models.tt_transformers.tt.ccl import TT_CCL

    tt_ccl = TT_CCL(mesh_device) if nd > 1 else None
    tw = load_gdn_weights_tp(mesh_device, sd, args)
    gdn = TPGatedDeltaNet(mesh_device, args, tw, tt_ccl)

    x = torch.randn(1, 1, T, args.dim, dtype=torch.bfloat16)
    # Prefill input is K-sharded (the model's prefill norm skips its AG; the fused in-proj gathers).
    x_tt = shard_to_device(mesh_device, x, dim=-1)
    composer = tp_composer(mesh_device)

    # ---- Prefill ----
    gdn.reset_state()
    out_pf = gdn.forward_prefill(x_tt, chunk_size=128)
    pf = ttnn.to_torch(out_pf, mesh_composer=composer)[0, 0].float()  # [T, dim]

    # ---- Decode the same tokens one at a time ----
    gdn.reset_state()
    dec_rows = []
    for t in range(T):
        xt = replicate_to_device(mesh_device, x[:, :, t : t + 1, :])
        ot = gdn.forward_decode(xt)
        dec_rows.append(ttnn.to_torch(ot, mesh_composer=composer)[0, 0, 0].float())  # [dim]
    dec = torch.stack(dec_rows, dim=0)  # [T, dim]

    passing, pcc = comp_pcc(dec, pf, get_pcc_threshold(request))
    logger.info(f"GDN TP PREFILL vs DECODE PCC (T={T}) = {pcc}")
    assert passing, f"GDN prefill/decode mismatch PCC: {pcc}"


@torch.no_grad()
@parametrize_mesh_tp()
def test_gdn_tp_fused_chunk_prefill(mesh_device, monkeypatch, reset_seeds, ensure_gc, request):
    """Isolate main's fused chunk_gated_delta_rule kernel (the DEFAULT prefill path).

    forward_prefill routes single-user prefill through ttnn.transformer.chunk_gated_delta_rule
    (fused_chunk_enabled() is on by default) — this is the per-user prefill path the model
    actually runs (prefill_chunked_peruser -> forward_prefill_collect -> forward_prefill). Here we
    run the SAME tokens twice — once fused (default), once with fused_chunk_enabled forced off so
    forward_prefill falls back to the trusted chunk_gated_delta_rule_seq_adapter — and require the
    fused output to match the seq path. Cross-checked against step-by-step decode for absolute
    grounding (both agree AND are correct). Multi-chunk (T > chunk_size) exercises the recurrence.
    """
    os.environ.setdefault("HF_MODEL", model_path())
    T, chunk = 256, 128  # T > chunk => multiple internal chunks (cross-chunk recurrence exercised)
    args = Qwen36ModelArgs(mesh_device, max_batch_size=1, max_seq_len=512)
    nd = mesh_device.get_num_devices()
    li = next(i for i, t in enumerate(args.attention_type_list) if t == "linear_attention")
    logger.info(f"devices={nd} gdn layer={li} T={T} chunk={chunk}")

    sd = load_gdn_layer(args.CKPT_DIR, li)
    from models.tt_transformers.tt.ccl import TT_CCL

    tt_ccl = TT_CCL(mesh_device) if nd > 1 else None
    tw = load_gdn_weights_tp(mesh_device, sd, args)
    gdn = TPGatedDeltaNet(mesh_device, args, tw, tt_ccl)

    x = torch.randn(1, 1, T, args.dim, dtype=torch.bfloat16)
    # Prefill input is K-sharded (the model's prefill norm skips its AG; the fused in-proj gathers).
    x_tt = shard_to_device(mesh_device, x, dim=-1)
    composer = tp_composer(mesh_device)

    import models.demos.blackhole.qwen36.tt.gdn.fused_chunk as fc

    assert fc.fused_chunk_enabled(), "fused chunk must be ON by default (production prefill path)"

    # ---- Fused chunk kernel (default) ----
    gdn.reset_state()
    out_fused = gdn.forward_prefill(x_tt, chunk_size=chunk)
    fused = ttnn.to_torch(out_fused, mesh_composer=composer)[0, 0].float()  # [T, dim]
    assert not torch.isnan(fused).any() and fused.abs().max() > 0

    # ---- Seq adapter (fused forced off) on the SAME tokens ----
    monkeypatch.setattr(fc, "fused_chunk_enabled", lambda: False)
    gdn.reset_state()
    out_seq = gdn.forward_prefill(x_tt, chunk_size=chunk)
    seq = ttnn.to_torch(out_seq, mesh_composer=composer)[0, 0].float()  # [T, dim]

    thr = get_pcc_threshold(request)
    passing_fs, pcc_fs = comp_pcc(seq, fused, thr)
    logger.info(f"GDN fused-chunk vs seq-adapter prefill PCC (T={T}) = {pcc_fs}")
    assert passing_fs, f"fused chunk kernel disagrees with seq adapter: PCC {pcc_fs} < {thr}"

    # ---- Absolute grounding: fused prefill must also match step-by-step decode ----
    monkeypatch.undo()  # restore fused-on (decode path is unaffected, but keep state clean)
    gdn.reset_state()
    dec_rows = []
    for t in range(T):
        xt = replicate_to_device(mesh_device, x[:, :, t : t + 1, :])
        ot = gdn.forward_decode(xt)
        dec_rows.append(ttnn.to_torch(ot, mesh_composer=composer)[0, 0, 0].float())
    dec = torch.stack(dec_rows, dim=0)  # [T, dim]
    passing_fd, pcc_fd = comp_pcc(dec, fused, thr)
    logger.info(f"GDN fused-chunk prefill vs step-decode PCC (T={T}) = {pcc_fd}")
    assert passing_fd, f"fused chunk prefill disagrees with step-by-step decode: PCC {pcc_fd} < {thr}"
