# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""TP validation for Qwen3.5/3.6 Gated DeltaNet on a Blackhole mesh.
Run:
    MESH_DEVICE=P150x4 HF_MODEL=Qwen/Qwen3.6-27B \
      pytest models/demos/blackhole/qwen36/tests/test_gdn_tp.py -v -s
The prefill conv-path tests at the end (test_conv_paths_*, test_kda_*) use random data and need no checkpoint.
"""
import os

import pytest
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
    random_gdn_state_dict,
    replicate_to_device,
    shard_to_device,
    tp_composer,
)
from models.demos.blackhole.qwen36.tt.gdn.tp import (
    TPGatedDeltaNet,
    kda_channel_chunk_size,
    kda_conv_prefill,
    load_gdn_weights_tp,
)
from models.demos.blackhole.qwen36.tt.model_config import GDN_CONV1D_L1_SMALL_SIZE, Qwen36ModelArgs
from models.experimental.gated_attention_gated_deltanet.tt.ttnn_delta_rule_ops import (
    recurrent_gated_delta_rule_decode_ttnn,
)
from models.experimental.gated_attention_gated_deltanet.tt.ttnn_gated_deltanet import _causal_conv1d_fir


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
@pytest.mark.parametrize("high_precision", [pytest.param(True, id="fp32"), pytest.param(False, id="bf16")])
def test_gdn_tp_decode_recurrence_state(mesh_device, high_precision, reset_seeds, ensure_gc, request):
    """Multi-step T=1 decode: output AND recurrent state vs a torch reference each step.

    ``test_gdn_tp`` validates pos0 only, where the recurrent state is zero and the
    state-decay multiply (h * exp(g), fused as an EXP pre-activation on the multiply)
    is invisible. This test drives ``recurrent_gated_delta_rule_decode_ttnn`` — the
    exact T=1 function ``forward_decode`` dispatches to — for several steps from a
    NONZERO initial state, checking both the step output and the carried state, so a
    broken decay (or an ignored fused activation) collapses the PCC immediately.
    fp32 mirrors the TP default (``high_precision=True``); bf16 covers the
    ``QWEN35_GDN_DECODE_BF16=1`` fallback.
    """
    B, H, K, V = 2, 8, 128, 128
    steps = 4
    # One threshold per node via pcc_thresholds.json; fp32/bf16 defaults differ (bf16
    # accumulates state quantization error over the 4 steps).
    thr = get_pcc_threshold(request, default=0.9999 if high_precision else 0.99)

    # Pre-quantize inputs to bf16 so device and reference consume identical values —
    # the remaining error is device math, not input rounding.
    def _bf16(t):
        return t.to(torch.bfloat16).float()

    q = _bf16(torch.randn(steps, B, H, K))
    k = _bf16(torch.randn(steps, B, H, K))
    v = _bf16(torch.randn(steps, B, H, V))
    beta = _bf16(torch.sigmoid(torch.randn(steps, B, H)))
    g = _bf16(-F.softplus(torch.randn(steps, B, H)))  # log-decay <= 0, exp(g) in (0,1]
    h0 = _bf16(torch.randn(B, H, K, V))  # nonzero: decay must actually act on it

    state_dtype = ttnn.float32 if high_precision else ttnn.bfloat16
    h_tt = replicate_to_device(mesh_device, h0, dtype=state_dtype)
    h_ref = h0.clone()

    def _first_shard(t):
        return ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float()

    for s in range(steps):
        o_tt, h_tt = recurrent_gated_delta_rule_decode_ttnn(
            replicate_to_device(mesh_device, q[s].reshape(B, 1, H, K)),
            replicate_to_device(mesh_device, k[s].reshape(B, 1, H, K)),
            replicate_to_device(mesh_device, v[s].reshape(B, 1, H, V)),
            replicate_to_device(mesh_device, beta[s].reshape(B, 1, H)),
            replicate_to_device(mesh_device, g[s].reshape(B, 1, H)),
            initial_state=h_tt,
            device=mesh_device,
            high_precision=high_precision,
        )

        # ---- torch reference: mirrors the ttnn function step-for-step ----
        qh = F.normalize(q[s], dim=-1) * (K**-0.5)  # [B,H,K]
        kh = F.normalize(k[s], dim=-1)
        h_ref = h_ref * torch.exp(g[s])[..., None, None]  # decay BEFORE read
        v_read = kh.unsqueeze(-2) @ h_ref  # [B,H,1,V]
        delta = v[s].unsqueeze(-2) - v_read
        h_ref = h_ref + beta[s][..., None, None] * (kh.unsqueeze(-1) @ delta)
        o_ref = (qh.unsqueeze(-2) @ h_ref).reshape(B, H, V)

        o_t = _first_shard(o_tt).reshape(B, H, V)
        pcc_o = compute_pcc(o_ref, o_t)
        pcc_h = compute_pcc(h_ref, _first_shard(h_tt))
        logger.info(f"step {s}: out PCC={pcc_o:.6f} state PCC={pcc_h:.6f}")
        assert pcc_o >= thr, f"step {s} output PCC {pcc_o:.6f} < {thr}"
        assert pcc_h >= thr, f"step {s} state PCC {pcc_h:.6f} < {thr}"

    logger.info(f"PASSED: GDN TP decode recurrence ({steps} steps, {'fp32' if high_precision else 'bf16'})")


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
@parametrize_batch(batches=(8,))
def test_gdn_tp_write_slot_and_remap(mesh_device, B, reset_seeds, ensure_gc, request):
    """Per-slot GDN state edits for vLLM continuous batching: write_slot + remap_slots.

    write_slot writes ONE user's B=1 prefill state into a single decode row without disturbing the
    others — the incremental analogue of assemble_batched_state (which builds the whole batch at
    once). remap_slots reindexes the rows on a vLLM batch condense. Validates:
      (a) writing B users one slot at a time (in reverse order, so each write must preserve the
          rows written before it) then ONE batched decode matches B independent B=1 runs, row by row;
      (b) remap_slots(reverse) makes decode row i carry user (B-1-i)'s state, and the permuted state
          is exactly the pre-remap rows reindexed (no cross-row contamination).
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
    T = 128

    xp = [torch.randn(1, 1, T, args.dim, dtype=torch.bfloat16) for _ in range(B)]
    xd = [torch.randn(1, 1, 1, args.dim, dtype=torch.bfloat16) for _ in range(B)]

    # ---- reference: B independent B=1 prefill(capture_state) + decode ----
    ref_rows = []
    for u in range(B):
        g = TPGatedDeltaNet(mesh_device, args1, tw, tt_ccl)
        g.reset_state()
        g.forward_prefill(shard_to_device(mesh_device, xp[u], dim=-1), chunk_size=T, capture_state=True)
        out_u = g.forward_decode(replicate_to_device(mesh_device, xd[u]))
        ref_rows.append(ttnn.to_torch(out_u, mesh_composer=comp)[0, 0, 0].float())

    # ---- batched via write_slot: each user prefilled B=1, its state written into ITS slot ----
    gb = TPGatedDeltaNet(mesh_device, args, tw, tt_ccl)
    gb.reset_state()
    for u in reversed(range(B)):  # reverse order: every write must preserve the already-written rows
        gu = TPGatedDeltaNet(mesh_device, args1, tw, tt_ccl)
        gu.reset_state()
        gu.forward_prefill(shard_to_device(mesh_device, xp[u], dim=-1), chunk_size=T, capture_state=True)
        gb.write_slot(u, gu.rec_state, list(gu.conv_states))  # consumes gu's rec/conv buffers
        gu.rec_state, gu.conv_states = None, None

    x_dec = torch.cat(xd, dim=2)  # [1, 1, B, dim], row u = user u's decode token
    out_b = gb.forward_decode(replicate_to_device(mesh_device, x_dec))
    out_t = ttnn.to_torch(out_b, mesh_composer=comp)  # [1, 1, B, dim]
    thr = get_pcc_threshold(request)
    pccs = [compute_pcc(ref_rows[u], out_t[0, 0, u].float()) for u in range(B)]
    bad = [(u, p) for u, p in enumerate(pccs) if p < thr]
    assert not bad, f"write_slot users below PCC {thr}: {bad} (min={min(pccs):.5f})"
    logger.info(f"write_slot (B={B}) worst PCC = {min(pccs):.5f}")

    # ---- remap_slots(reverse): row i must become the exact pre-remap row (B-1-i) ----
    remap = [B - 1 - i for i in range(B)]
    pre = ttnn.to_torch(gb.rec_state, mesh_composer=comp).float()  # [nd*B?, ...] mesh dim 0 = devices
    gb.remap_slots(remap)
    post = ttnn.to_torch(gb.rec_state, mesh_composer=comp).float()
    # rec_state per device is [B, Nv, Dk, Dv]; mesh-concat stacks devices on dim 0 -> [nd*B, ...].
    # Compare row i to pre row remap[i] within each device block.
    ndev = pre.shape[0] // B
    max_diff = 0.0
    for d in range(ndev):
        for i in range(B):
            max_diff = max(max_diff, (post[d * B + i] - pre[d * B + remap[i]]).abs().max().item())
    assert max_diff < 1e-3, f"remap_slots rec mismatch: max_diff={max_diff}"
    logger.info(f"remap_slots (B={B}) exact-permutation max_diff = {max_diff:.2e}")
    logger.info(f"PASSED: write_slot + remap_slots (B={B})")


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
    gref.forward_prefill_batched(shard_to_device(mesh_device, x.unsqueeze(0), dim=-1), chunk_size=C)
    out_ref = ttnn.to_torch(gref.forward_decode(replicate_to_device(mesh_device, xd)), mesh_composer=comp)

    # ---- test: two CARRIED chunks ----
    g = TPGatedDeltaNet(mesh_device, args, tw, tt_ccl)
    g.reset_state()
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


def _snapshot_layer_state(gdn, mesh_device):
    """Host copy of one GDN layer's (rec_state, conv_carry, conv_states)."""
    comp = ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    return (
        ttnn.to_torch(gdn.rec_state, mesh_composer=comp),
        ttnn.to_torch(gdn.conv_carry, mesh_composer=comp) if gdn.conv_carry is not None else None,
        [ttnn.to_torch(c, mesh_composer=comp) for c in gdn.conv_states] if gdn.conv_states is not None else None,
    )


def _restore_layer_state(gdn, mesh_device, snap):
    rec, carry, convs = snap
    mapper = ttnn.ShardTensorToMesh(mesh_device, dim=0)

    def _back(t, dtype):
        return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=mapper)

    r = _back(rec, gdn.rec_state.dtype)
    ttnn.copy(r, gdn.rec_state)
    ttnn.deallocate(r)
    if carry is not None and gdn.conv_carry is not None:
        c = _back(carry, gdn.conv_carry.dtype)
        ttnn.copy(c, gdn.conv_carry)
        ttnn.deallocate(c)
    if convs is not None and gdn.conv_states is not None:
        for j, cs in enumerate(convs):
            cc = _back(cs, gdn.conv_states[j].dtype)
            ttnn.copy(cc, gdn.conv_states[j])
            ttnn.deallocate(cc)


@torch.no_grad()
@parametrize_mesh_tp()
def test_gdn_chunk_vs_recurrent_attribution(mesh_device, reset_seeds, ensure_gc, request):
    """Localize the chunk-vs-recurrent divergence that speculative decoding runs into.

    Two facts already established elsewhere:
      * torch chunk == torch recurrent to >= 0.9999 PCC in exactly the verify regime, including a
        nonzero carried state (test_gdn_chunk_recurrent_parity, CPU) — so the ALGORITHM is exact;
      * on device, chunk prefill == step decode over T=128 tokens FROM ZERO STATE
        (test_gdn_tp_prefill / test_gdn_tp_fused_chunk_prefill).

    Neither covers what verify actually does: a SHORT chunk continuing from a warmed state. A
    whole-model measurement (tests/test_spec_decode_features.py) shows the two paths' hidden states
    at cosine 0.93 after a single step from a bit-identical state, which is far too large to be
    rounding — so the natural suspicion is the carried recurrent/conv state handoff. This narrows it
    to one GDN layer and three regimes:

        cold  : both paths from zero state over T tokens      (the already-covered case)
        warm  : both continue from the SAME warmed state      (the verify case)
        warm1 : ONE real token in a masked bucket             (what verify_forward runs per step)

    MEASURED: all three agree at PCC ~0.99999 (cold 0.999955, warm 0.999991, warm1 0.999965), so
    the suspicion is wrong — the carried-state handoff is sound and one GDN layer is faithful in
    exactly the regime verify uses. The model-level gap is instead the COMPOUNDING of per-layer
    differences of this size: test_spec_decode_tp.py::test_verify_layer_localization measures the
    hidden PCC decaying 0.999999 (layer 0) -> 0.99990 (layer 31) -> 0.9932 (layer 63) with no jump
    at any single layer, GDN or attention. Two kernel pairs contribute along the way (the delta-rule
    scan vs its recurrence, and prefill vs decode SDPA), and 64 residual+RMSNorm stages amplify.

    Consequences, both measured elsewhere rather than assumed:
      * acceptance barely cares — injecting 30% relative noise into the drafter's hidden costs only
        ~0.19 committed tokens/iter (tests/mtp_cpu_check.py), and the chunk and recurrent feature
        sets give the same ceiling;
      * output text does care — the two paths' greedy trajectories fork within a couple of tokens
        and the chunk path degenerates into repetition on some prompts, which is why
        test_spec_decode_align.py pins the base at the recurrent kernel and guards degeneracy.

    This test therefore stands as a regression guard on the per-layer kernels, not as a reproduction
    of the model-level gap.
    """
    os.environ.setdefault("HF_MODEL", model_path())
    # T must exceed TILE_SIZE (32): at exactly 32 the in-projection takes the decode-sized branch,
    # which expects a replicated rather than K-sharded activation. verify_forward's real bucket is
    # 128, so anything above the boundary reproduces the production path.
    W, T = 128, 64  # warmup tokens, then a short chunk (both 32-multiples for the fused kernel)
    args = Qwen36ModelArgs(mesh_device, max_batch_size=1, max_seq_len=256)
    nd = mesh_device.get_num_devices()
    li = next(i for i, t in enumerate(args.attention_type_list) if t == "linear_attention")
    logger.info(f"devices={nd} gdn layer={li} warmup={W} chunk={T}")

    sd = load_gdn_layer(args.CKPT_DIR, li)
    from models.tt_transformers.tt.ccl import TT_CCL

    tt_ccl = TT_CCL(mesh_device) if nd > 1 else None
    gdn = TPGatedDeltaNet(mesh_device, args, load_gdn_weights_tp(mesh_device, sd, args), tt_ccl)
    composer = tp_composer(mesh_device)

    warm_x = torch.randn(1, 1, W, args.dim, dtype=torch.bfloat16)
    x = torch.randn(1, 1, T, args.dim, dtype=torch.bfloat16)

    def _chunk_rows(valid_len=None, n=None):
        out = gdn.forward_prefill(
            shard_to_device(mesh_device, x, dim=-1), chunk_size=T, valid_len=valid_len, capture_state=True
        )
        rows = ttnn.to_torch(out, mesh_composer=composer)[0, 0].float()
        return rows[: n if n is not None else T]

    def _decode_rows(n):
        rows = []
        for t in range(n):
            ot = gdn.forward_decode(replicate_to_device(mesh_device, x[:, :, t : t + 1, :]))
            rows.append(ttnn.to_torch(ot, mesh_composer=composer)[0, 0, 0].float())
        return torch.stack(rows, dim=0)

    results = {}

    # ---- cold: both paths from zero state ----
    gdn.reset_state()
    cold_chunk = _chunk_rows()
    gdn.reset_state()
    cold_dec = _decode_rows(T)
    results["cold"] = (cold_chunk, cold_dec)

    # ---- warm: one prefill establishes the state (populating BOTH conv_carry for the chunk path
    # and conv_states for the decode path), then each path continues from that exact snapshot ----
    gdn.reset_state()
    ttnn.deallocate(gdn.forward_prefill(shard_to_device(mesh_device, warm_x, dim=-1), chunk_size=W, capture_state=True))
    snap = _snapshot_layer_state(gdn, mesh_device)

    _restore_layer_state(gdn, mesh_device, snap)
    warm_chunk = _chunk_rows()
    _restore_layer_state(gdn, mesh_device, snap)
    warm_dec = _decode_rows(T)
    results["warm"] = (warm_chunk, warm_dec)

    # ---- warm1: a single real token in a masked bucket — verify_forward's per-step shape ----
    _restore_layer_state(gdn, mesh_device, snap)
    warm1_chunk = _chunk_rows(valid_len=1, n=1)
    _restore_layer_state(gdn, mesh_device, snap)
    warm1_dec = _decode_rows(1)
    results["warm1"] = (warm1_chunk, warm1_dec)

    for label, (chunk_rows, dec_rows) in results.items():
        n = chunk_rows.shape[0]
        pcc = compute_pcc(dec_rows, chunk_rows)
        cos = torch.nn.functional.cosine_similarity(dec_rows, chunk_rows, dim=-1)
        per_pos = " ".join(f"t{t}={float(cos[t]):.5f}" for t in range(min(n, 4)))
        logger.info(f"[{label}] chunk-vs-decode PCC={pcc:.6f} cos(min)={float(cos.min()):.6f} [{per_pos}]")

    cold_pcc = compute_pcc(results["cold"][1], results["cold"][0])
    warm_pcc = compute_pcc(results["warm"][1], results["warm"][0])
    warm1_pcc = compute_pcc(results["warm1"][1], results["warm1"][0])
    logger.info(f"ATTRIBUTION: cold={cold_pcc:.6f} warm={warm_pcc:.6f} warm1={warm1_pcc:.6f}")
    if cold_pcc > 0.99 and warm_pcc < 0.99:
        logger.error(
            "chunk and decode agree from ZERO state but not from a CARRIED state -> the defect is "
            "in the carried recurrent/conv state handoff, not kernel precision"
        )

    # Measured at ~0.99999 for all three; hold them near that rather than at a loose 0.99, since the
    # whole point is that a single layer is far more faithful than the 64-layer stack.
    thr = get_pcc_threshold(request, default=0.9995)
    assert cold_pcc > thr, f"cold-start chunk vs decode regressed (PCC={cold_pcc:.6f})"
    # The warm regimes are what speculative decoding depends on, and torch parity says they are
    # exact; a drop here would be a real carried recurrent/conv state defect.
    assert warm_pcc > thr, (
        f"chunk vs decode from a CARRIED state is only PCC={warm_pcc:.6f} (cold-start is "
        f"{cold_pcc:.6f}, and torch parity for this regime is >= 0.9999) — the carried "
        "recurrent/conv state handoff regressed"
    )
    assert warm1_pcc > thr, (
        f"a single-token masked chunk from a carried state is only PCC={warm1_pcc:.6f} — this is "
        "exactly what verify_forward runs per step"
    )


@torch.no_grad()
@parametrize_mesh_tp()
@pytest.mark.parametrize(
    "OUTER_CHUNK_SIZE",
    [64, 512, 2048],
    ids=["OUTER_CHUNK_SIZE64", "OUTER_CHUNK_SIZE512", "OUTER_CHUNK_SIZE2048"],
)
def test_gdn_out_agmm_vs_mmrs(mesh_device, OUTER_CHUNK_SIZE, reset_seeds, ensure_gc):
    """GDN prefill out-projection: column-parallel AG+matmul vs the row-parallel matmul+reduce-scatter.
    Runs one forward_prefill with out-AGMM prefill enabled and disabled, and PCCs the two outputs against each other.

    OUTER_CHUNK_SIZE: how the prompt is split; one forward_prefill call receives one outer chunk as input;
    64 covers a short prefill that is not a multiple of 128 (reachable: the TP paged prefill passes the
    raw prompt length). T <= TILE_SIZE is not tested: on TP such a prefill already fails in the QKV
    in-proj, whose S <= TILE_SIZE branch needs a full-width x while prefill hands GDN a K-sharded one.
    """
    os.environ.setdefault("HF_MODEL", model_path())
    args = Qwen36ModelArgs(mesh_device, max_batch_size=1, max_seq_len=4096)
    nd = mesh_device.get_num_devices()
    if nd == 1:
        pytest.skip("TP-only")
    li = next(i for i, t in enumerate(args.attention_type_list) if t == "linear_attention")
    sd = load_gdn_layer(args.CKPT_DIR, li)
    from models.tt_transformers.tt.ccl import TT_CCL

    tt_ccl = TT_CCL(mesh_device)
    tw = load_gdn_weights_tp(mesh_device, sd, args)
    gdn = TPGatedDeltaNet(mesh_device, args, tw, tt_ccl)
    assert gdn._out_colpar_prefill, "column-parallel prefill out-proj not active"
    composer = tp_composer(mesh_device)

    x = torch.randn(1, 1, OUTER_CHUNK_SIZE, args.dim, dtype=torch.bfloat16)
    x_tt = shard_to_device(mesh_device, x, dim=-1)

    # CHUNK_SIZE: how the input to forward_prefill is split and processed by the GDN kernel;
    # The parameter is used by the sequential GDN kernel only. The (default and used here)
    # fused/phased kernel hardcodes 32.
    CHUNK_SIZE = 128
    logger.info(f"[AGMM] OUTER_CHUNK_SIZE={OUTER_CHUNK_SIZE} starting AGMM out-proj arm")
    gdn.reset_state()
    o = gdn.forward_prefill(x_tt, chunk_size=CHUNK_SIZE)
    got = ttnn.to_torch(o, mesh_composer=composer).reshape(OUTER_CHUNK_SIZE, -1).float()
    ttnn.deallocate(o)
    logger.info(f"[AGMM] OUTER_CHUNK_SIZE={OUTER_CHUNK_SIZE} AGMM arm OK, out shape {tuple(got.shape)}")

    logger.info(f"[AGMM] OUTER_CHUNK_SIZE={OUTER_CHUNK_SIZE} starting MMRS reference arm")
    gdn._out_colpar_prefill = False
    gdn.reset_state()
    o2 = gdn.forward_prefill(x_tt, chunk_size=CHUNK_SIZE)
    ref = ttnn.to_torch(o2, mesh_composer=composer).reshape(OUTER_CHUNK_SIZE, -1).float()
    ttnn.deallocate(o2)
    gdn._out_colpar_prefill = True
    logger.info(f"[AGMM] OUTER_CHUNK_SIZE={OUTER_CHUNK_SIZE} MMRS arm OK")

    passing, pcc = comp_pcc(ref, got, 0.99)
    logger.info(f"GDN out-proj AGMM vs MMRS PCC (OUTER_CHUNK_SIZE={OUTER_CHUNK_SIZE}) = {pcc}")
    assert passing, f"AGMM/MMRS mismatch at OUTER_CHUNK_SIZE={OUTER_CHUNK_SIZE}: {pcc}"


@torch.no_grad()
@parametrize_mesh_tp()
def test_gdn_out_agmm_deterministic_under_device_skew(mesh_device, monkeypatch, reset_seeds, ensure_gc):
    """The column-parallel out-projection must not depend on device timing.

    all_gather_minimal_matmul_async writes each device's K-slice straight into its peers' gather buffer, with
    no receiver-ready handshake. A per-call gather buffer is allocated on the host from L1 the previous ops
    just freed (here: the gate multiply's fp32 input), so a device that reaches the op early overwrites data a
    lagging peer's gate multiply is still reading. Delay each device in turn right before the gate
    (ttnn.apply_device_delay) and require every run to be bit-identical to a reference computed with the
    devices synchronized before the out-projection."""
    import models.demos.blackhole.qwen36.tt.gdn.tp as gdn_tp
    from models.demos.blackhole.qwen36.tt import tp_common as tpc

    os.environ.setdefault("HF_MODEL", model_path())
    nd = mesh_device.get_num_devices()
    if nd == 1:
        pytest.skip("TP-only")
    T = 2048
    args = Qwen36ModelArgs(mesh_device, max_batch_size=1, max_seq_len=4096)
    li = next(i for i, t in enumerate(args.attention_type_list) if t == "linear_attention")
    sd = load_gdn_layer(args.CKPT_DIR, li)
    from models.tt_transformers.tt.ccl import TT_CCL

    gdn = TPGatedDeltaNet(mesh_device, args, load_gdn_weights_tp(mesh_device, sd, args), TT_CCL(mesh_device))
    assert gdn._out_colpar_prefill, "column-parallel prefill out-proj not active"
    composer = tp_composer(mesh_device)
    x_tt = shard_to_device(mesh_device, torch.randn(1, 1, T, args.dim, dtype=torch.bfloat16), dim=-1)

    def run():
        gdn.reset_state()
        o = gdn.forward_prefill(x_tt)
        out = ttnn.to_torch(o, mesh_composer=composer)[0, 0].float().clone()
        ttnn.deallocate(o)
        return out

    agmm, silu_mul = tpc.all_gather_matmul_prefill, gdn_tp._silu_mul

    def synced_agmm(*a, **kw):
        ttnn.synchronize_device(mesh_device)
        return agmm(*a, **kw)

    with monkeypatch.context() as m:
        m.setattr(tpc, "all_gather_matmul_prefill", synced_agmm)
        ref = run()

    for late in range(nd):
        delays = [[600_000 if d == late else 0 for d in range(nd)]]

        def delayed_silu_mul(x, z, memory_config, dtype=None):
            if dtype is not None:  # the column-parallel arm only
                ttnn.apply_device_delay(mesh_device, delays)
            return silu_mul(x, z, memory_config, dtype)

        with monkeypatch.context() as m:
            m.setattr(gdn_tp, "_silu_mul", delayed_silu_mul)
            got = run()
        bad = torch.nonzero((got - ref).abs().sum(-1)).flatten()
        assert bad.numel() == 0, (
            f"device {late} late: {bad.numel()} output rows differ from the synchronized reference "
            f"(first {bad[:8].tolist()}): the out-projection gather overwrote data the late device still used"
        )


@torch.no_grad()
@parametrize_mesh_tp()
@pytest.mark.parametrize("T", [256, 2048], ids=lambda t: f"T{t}")
def test_gdn_tp_prefill_fused_vs_phased_bit_exact(mesh_device, T, reset_seeds, ensure_gc):
    """Layer-level test for the fused prep->scan op: TPGatedDeltaNet.forward_prefill with real weights,
    run with a phased and a fused program config on the same tokens.
    """
    os.environ.setdefault("HF_MODEL", model_path())
    if mesh_device.get_num_devices() == 1:
        pytest.skip("TP-only")
    args = Qwen36ModelArgs(mesh_device, max_batch_size=1, max_seq_len=4096)
    li = next(i for i, t in enumerate(args.attention_type_list) if t == "linear_attention")
    sd = load_gdn_layer(args.CKPT_DIR, li)
    from models.tt_transformers.tt.ccl import TT_CCL

    gdn = TPGatedDeltaNet(mesh_device, args, load_gdn_weights_tp(mesh_device, sd, args), TT_CCL(mesh_device))
    composer = tp_composer(mesh_device)
    x_tt = shard_to_device(mesh_device, torch.randn(1, 1, T, args.dim, dtype=torch.bfloat16), dim=-1)

    def run(program_config):
        gdn.gdn_program_config = program_config
        gdn.reset_state()
        o = gdn.forward_prefill(x_tt)
        out = ttnn.to_torch(o, mesh_composer=composer)[0, 0].float().clone()
        ttnn.deallocate(o)
        return out

    phased = run(ttnn.ChunkGdnPhasedProgramConfig())
    phased_again = run(ttnn.ChunkGdnPhasedProgramConfig())
    n_phased = mesh_device.num_program_cache_entries()
    fused = run(ttnn.ChunkGdnFusedProgramConfig())
    n_fused = mesh_device.num_program_cache_entries()
    assert torch.equal(phased, phased_again), "phased layer output is not deterministic"
    assert n_fused > n_phased, "the fused program config compiled no new program: the fused prim did not run"
    d = (phased - fused).abs()
    assert torch.equal(phased, fused), (
        f"T={T}: fused layer output differs from phased (max|d|={d.max().item():.3e}, "
        f"first differing row {int(torch.nonzero(d.sum(-1))[0])})"
    )


@torch.no_grad()
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
            "l1_small_size": GDN_CONV1D_L1_SMALL_SIZE,
            "trace_region_size": 268435456,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [pytest.param((1, 4), id="1x4")], indirect=True)
@pytest.mark.parametrize("weights", ["checkpoint", "random"])
def test_gdn_tp_prefill_trace_replay(mesh_device, weights, reset_seeds, ensure_gc, request):
    """Chunk-outer prefill through ONE captured trace equals the eager chunks, bit for bit.

    Three 2048-token chunks with the persistent carry, first eagerly, then as the
    model's chunked prefill runs them: state zeroed in place, one forward_prefill captured, the trace
    replayed three times with each chunk ttnn.copy'd into the persistent input buffer and the baked
    output read back. Catches anything on the prefill path that allocates or writes from the host
    inside the trace (constants must be built by reset_state) or whose programs differ per chunk.
    The property needs no trained weights: the `random` variant runs wherever HF_MODEL holds the
    model's config.json, the `checkpoint` variant where the layer's safetensors are too.
    """
    os.environ.setdefault("HF_MODEL", model_path())
    T, n_chunks = 2048, 3
    mesh = mesh_device
    args = Qwen36ModelArgs(mesh, max_batch_size=1, max_seq_len=T * n_chunks)
    li = next(i for i, t in enumerate(args.attention_type_list) if t == "linear_attention")
    sd = load_gdn_layer(args.CKPT_DIR, li) if weights == "checkpoint" else random_gdn_state_dict(args, seed=li)
    from models.tt_transformers.tt.ccl import TT_CCL

    tt_ccl = TT_CCL(mesh)
    tw = load_gdn_weights_tp(mesh, sd, args)
    gdn = TPGatedDeltaNet(mesh, args, tw, tt_ccl)
    logger.info(f"conv impl={gdn._conv_impl} kda={gdn._gdn_kda_conv} layer={li}")
    gdn.reset_state()  # persistent state and the prefill constants, before any capture
    x = torch.randn(1, 1, T * n_chunks, args.dim, dtype=torch.bfloat16)
    chunks = [x[:, :, c * T : (c + 1) * T, :] for c in range(n_chunks)]
    comp = tp_composer(mesh)

    # Persistent K-sharded input buffer (its address is baked into the trace); chunks are copied in.
    x_buf = shard_to_device(mesh, chunks[0], dim=-1)

    def load_chunk(c):
        src = shard_to_device(mesh, chunks[c], dim=-1)
        ttnn.copy(src, x_buf)
        ttnn.deallocate(src)

    eager = []
    for c in range(n_chunks):
        load_chunk(c)
        out = gdn.forward_prefill(x_buf, chunk_size=args.gdn_chunk_size)
        eager.append(ttnn.to_torch(out, mesh_composer=comp).float())
        ttnn.deallocate(out)
    ttnn.synchronize_device(mesh)

    gdn.reset_state_inplace()
    ttnn.synchronize_device(mesh)
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    out_t = gdn.forward_prefill(x_buf, chunk_size=args.gdn_chunk_size)
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    replay = []
    for c in range(n_chunks):
        load_chunk(c)
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
        replay.append(ttnn.to_torch(out_t, mesh_composer=comp).float())
    ttnn.release_trace(mesh, tid)

    for c in range(n_chunks):
        eq = torch.equal(eager[c], replay[c])
        _, pcc = comp_pcc(eager[c], replay[c])
        logger.info(f"chunk {c}: trace replay == eager: {eq}; pcc {pcc}")
        assert eq, f"chunk {c}: trace replay differs from eager (pcc {pcc})"
    logger.info("PASSED: traced chunk-outer GDN prefill matches eager on every chunk")


# ---------------------------------------------------------------------------------------------------------------
# Prefill conv paths (the KDA fused op and the FIR) against one torch reference. Random data, no checkpoint;
# data is replicated to every device of the mesh and asserted on device 0.
# ---------------------------------------------------------------------------------------------------------------

CONV_K = 4  # conv kernel width; the carry holds K-1 = 3 rows
CONV_PCC_VS_REF = 0.999
# The first K-1 rows are the only ones that read the carry, and 3 rows in 2048 do not move a whole-chunk PCC (a
# dropped carry still scores 0.9996), so their max-abs error is held to a multiple of the other rows' (measured:
# 0.016 vs 0.045 with the carry, 2.45 without).
CARRY_ROWS_HEADROOM = 4

# (T, kd, vd, fir) per device: the 27B TP-4 production shape (C = 2560), the 35B-A3B TP-4 shape (C = 2048), the 9B
# single-device shape (C = 8192; the KDA op only: the FIR's shifted copies do not fit L1 next to the L1 qkv) and a
# small one (C = 192).
CONV_SHAPES = [
    pytest.param(2048, 512, 1536, True, id="T2048-kd512-vd1536"),
    pytest.param(2048, 512, 1024, True, id="T2048-kd512-vd1024"),
    pytest.param(2048, 2048, 4096, False, id="T2048-kd2048-vd4096"),
    pytest.param(64, 64, 64, True, id="T64-kd64-vd64"),
]


def _ref_conv(x, hist, w):
    """Depthwise causal conv + SiLU in fp32: silu(sum_j w[:, j] * xpad[t + j]) with the K-1 history rows
    prepended, so tap j multiplies input row t - 3 + j. x [1,T,C], hist [1,3,C] bf16; w [C,K] bf16."""
    T = x.shape[1]
    xp = torch.cat([hist.float(), x.float()], dim=1)  # [1, K-1+T, C]
    return F.silu(sum(w[:, j].float() * xp[:, j : j + T, :] for j in range(CONV_K)))


def _split_qkv(t, kd, vd):
    return t[..., :kd], t[..., kd : 2 * kd], t[..., 2 * kd :]


def _dev0(t):
    """Device 0's copy of a replicated mesh tensor, as a torch tensor."""
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0])


def _pcc(golden, calculated):
    return comp_pcc(golden, calculated)[1]


def _random_inputs(T, C, seed):
    torch.manual_seed(seed)
    x = torch.randn(1, T, C).to(torch.bfloat16)
    hist = torch.randn(1, CONV_K - 1, C).to(torch.bfloat16)  # nonzero carry: exercises the history rows
    w = (torch.randn(C, CONV_K) * 0.3).to(torch.bfloat16)  # taps [C, CONV_K]; tap j multiplies row t-3+j
    return x, hist, w


def _to_l1(mesh, x):
    """qkv as the layer hands it to the conv: bf16 TILE in L1, replicated."""
    return ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )


def _taps(mesh, w):
    """The KDA / FIR tap contract: four [1, 1, C] bf16 TILE tensors in kernel-position order."""
    C = w.shape[0]
    return [replicate_to_device(mesh, w[:, j].reshape(1, 1, C).contiguous()) for j in range(CONV_K)]


def _actual_start(mesh):
    return ttnn.from_torch(
        torch.tensor([0], dtype=torch.int64),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def _slice3(conv, T, kd, vd):
    """q | k | v column split of a [1, T, C] conv output (the non-fused paths' epilogue)."""
    C = 2 * kd + vd
    q = ttnn.slice(conv, (0, 0, 0), (1, T, kd))
    k = ttnn.slice(conv, (0, 0, kd), (1, T, 2 * kd))
    v = ttnn.slice(conv, (0, 0, 2 * kd), (1, T, C))
    ttnn.deallocate(conv)
    return q, k, v


def _check_contract(name, q, k, v, new_state, T, kd, vd, x):
    """Output contract shared by every path: q/k/v [1,T,kd|kd|vd] and new_state [1,3,C], all TILE in DRAM;
    new_state is bit-exactly the last K-1 rows of the bf16 input."""
    C = 2 * kd + vd
    for label, t, shape in (
        ("q", q, (1, T, kd)),
        ("k", k, (1, T, kd)),
        ("v", v, (1, T, vd)),
        ("new_state", new_state, (1, 3, C)),
    ):
        assert tuple(t.shape) == shape, f"{name} {label}: shape {tuple(t.shape)} != {shape}"
        assert t.layout == ttnn.TILE_LAYOUT, f"{name} {label}: layout {t.layout}"
        assert t.memory_config().buffer_type == ttnn.BufferType.DRAM, f"{name} {label}: {t.memory_config()}"
    assert torch.equal(
        _dev0(new_state), x[:, T - (CONV_K - 1) :, :]
    ), f"{name}: new_state != last {CONV_K - 1} input rows"


def _compare(name, got, ref):
    """PCC (asserted) and max-abs (logged) of q/k/v against the reference triple, and the carry-dependent rows
    (the first K-1) held to CARRY_ROWS_HEADROOM x the max-abs of the other rows. got/ref: fp32 torch."""
    pccs = [_pcc(r, g) for r, g in zip(ref, got)]
    mads = [(g - r).abs().max().item() for r, g in zip(ref, got)]
    head = max((g[:, : CONV_K - 1] - r[:, : CONV_K - 1]).abs().max().item() for r, g in zip(ref, got))
    tail = max((g[:, CONV_K - 1 :] - r[:, CONV_K - 1 :]).abs().max().item() for r, g in zip(ref, got))
    logger.info(
        f"{name:7s}: pcc q/k/v {pccs[0]:.6f} {pccs[1]:.6f} {pccs[2]:.6f} | max-abs q/k/v "
        f"{mads[0]:.3e} {mads[1]:.3e} {mads[2]:.3e} | carry rows {head:.3e} vs rest {tail:.3e}"
    )
    for label, p in zip("qkv", pccs):
        assert p >= CONV_PCC_VS_REF, f"{name} {label}: pcc {p:.6f} < {CONV_PCC_VS_REF}"
    assert (
        head <= CARRY_ROWS_HEADROOM * tail
    ), f"{name}: carry rows max-abs {head:.3e} > {CARRY_ROWS_HEADROOM} x the other rows' {tail:.3e}"
    return pccs, mads


@torch.no_grad()
@parametrize_mesh_tp()
@pytest.mark.parametrize("T, kd, vd, fir", CONV_SHAPES)
def test_conv_paths_match_reference(mesh_device, T, kd, vd, fir, reset_seeds, ensure_gc, request):
    """The KDA fused op and the FIR (where it fits) on the same L1 qkv and the same nonzero TILE carry: each
    within PCC of the fp32 reference, the carry rows included, and both honour the same output contract."""
    mesh = mesh_device
    C = 2 * kd + vd
    x, hist, w = _random_inputs(T, C, seed=223)
    ref = [r.contiguous() for r in _split_qkv(_ref_conv(x, hist, w), kd, vd)]

    qkv = _to_l1(mesh, x)
    carry = replicate_to_device(mesh, hist)  # TILE DRAM, as the layer's conv_carry
    taps = _taps(mesh, w)
    start = _actual_start(mesh)

    def run_kda():
        return kda_conv_prefill(qkv, T, carry, taps, (kd, kd, vd), start)

    def run_fir():
        conv, ns = _causal_conv1d_fir(
            qkv,
            None,
            None,
            CONV_K,
            mesh,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            conv_state=carry,
            weight_taps=taps,
            bias_dev=None,
            valid_len=None,
        )
        return (*_slice3(conv, T, kd, vd), ns)

    arms = [("kda", run_kda)] + ([("fir", run_fir)] if fir else [])
    outs = {}
    for name, fn in arms:
        q, k, v, ns = fn()
        _check_contract(name, q, k, v, ns, T, kd, vd, x)
        outs[name] = [_dev0(t).float() for t in (q, k, v)]
        for t in (q, k, v, ns):
            ttnn.deallocate(t)
        _compare(name, outs[name], ref)


@torch.no_grad()
@parametrize_mesh_tp()
def test_kda_carry_across_chunks(mesh_device, reset_seeds, ensure_gc, request):
    """Two consecutive T-row chunks through kda_conv_prefill: chunk 1 from an all-zero ROW_MAJOR history (the
    layer's _kda_zero_history), chunk 2 from chunk 1's TILE new_state. Both must match the reference computed
    over the concatenated 2T rows."""
    mesh = mesh_device
    T, kd, vd = 2048, 512, 1536
    C = 2 * kd + vd
    x, _, w = _random_inputs(2 * T, C, seed=224)
    zeros = torch.zeros(1, CONV_K - 1, C, dtype=torch.bfloat16)
    ref_full = _ref_conv(x, zeros, w)  # [1, 2T, C]

    history = ttnn.from_torch(
        zeros,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    taps = _taps(mesh, w)
    start = _actual_start(mesh)

    for ci in range(2):
        xc = x[:, ci * T : (ci + 1) * T, :]
        qkv = _to_l1(mesh, xc)
        q, k, v, ns = kda_conv_prefill(qkv, T, history, taps, (kd, kd, vd), start)
        ttnn.deallocate(qkv)
        _check_contract(f"kda chunk{ci}", q, k, v, ns, T, kd, vd, xc)
        got = [_dev0(t).float() for t in (q, k, v)]
        for t in (q, k, v):
            ttnn.deallocate(t)
        ref = [r.contiguous() for r in _split_qkv(ref_full[:, ci * T : (ci + 1) * T, :], kd, vd)]
        _compare(f"chunk{ci}", got, ref)
        history = ns  # TILE: chunk 2 takes the TILE -> ROW_MAJOR branch of kda_conv_prefill


@torch.no_grad()
@parametrize_mesh_tp()
@pytest.mark.parametrize("T", [1, 4, 8, 12])
def test_kda_conv_padded_rows_match_reference(mesh_device, T, reset_seeds, ensure_gc, request):
    """The spec verify's conv (tp._verify_fullbatch): T non-tile-aligned tokens (seed T=1, verify T=K+1),
    right-padded to 32 rows for the KDA op. The conv is causal, so rows [0, T) must match the reference
    built from the T real rows alone; the padded tail is never read."""
    mesh = mesh_device
    kd, vd = 512, 1536  # 27B at TP4
    C = 2 * kd + vd
    x, hist, w = _random_inputs(T, C, seed=225)
    ref = [r.contiguous() for r in _split_qkv(_ref_conv(x, hist, w), kd, vd)]

    x_l1 = _to_l1(mesh, x)
    qkv = ttnn.pad(x_l1, [(0, 0), (0, 32 - T), (0, 0)], 0.0)  # same pad call as the verify
    ttnn.deallocate(x_l1)
    carry = replicate_to_device(mesh, hist)  # TILE DRAM, like _conv_win_buf's slice
    q, k, v, ns = kda_conv_prefill(qkv, 32, carry, _taps(mesh, w), (kd, kd, vd), _actual_start(mesh), emit_state=False)
    assert ns is None, "emit_state=False must not return new_state"
    for label, t, r in zip("qkv", (q, k, v), ref):
        p = _pcc(r, _dev0(t).float()[:, :T])
        logger.info(f"T={T} {label}: pcc {p:.6f}")
        assert p >= CONV_PCC_VS_REF, f"T={T} {label}: pcc {p:.6f} < {CONV_PCC_VS_REF}"
    for t in (qkv, q, k, v):
        ttnn.deallocate(t)


@pytest.mark.parametrize(
    "channels, expected",
    [(2560, 512), (1280, 320), (5120, 512), (96, 96), (64, 64)],
)
def test_kda_channel_chunk_size(channels, expected):
    """Largest tile-aligned divisor of the channel count not above the cap (pure python)."""
    assert kda_channel_chunk_size(channels) == expected


def test_kda_channel_chunk_size_rejects_unaligned(expect_error):
    with expect_error(ValueError, "no tile-aligned channel chunk divides 100"):
        kda_channel_chunk_size(100)
