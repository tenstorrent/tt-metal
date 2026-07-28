# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""TP validation for the Qwen3.6-27B MTP (multi-token prediction) drafter head.

The MTP head is the speculative-decode drafter (the ``mtp.*`` checkpoint weights):

    h'  = fc( concat[ enorm(embed(token)), hnorm(hidden) ] )
    h'' = DecoderLayer(h')                         # reuses the full-attention decoder layer
    logits = LMHead( mtp.norm(h'') )

This composes the same per-component torch references the other TP tests validate
(gated causal attention, RMSNorm, SwiGLU MLP, LM head) end-to-end and PCC-checks the full
head. It specifically pins the NEW wiring: the concat order ([embedding, hidden]) and which
pre-fc norm applies to which input, plus the fc + norm sharding on the mesh.

The torch reference lives in mtp_torch_ref.py, shared with the off-device acceptance oracle
(mtp_cpu_check.py) so the device fidelity check and the acceptance ceiling cannot drift apart.

A tiny ``n_layers=0`` model is built (embedding + LM head + final norm + MTP head only) so the
64 base layers are never loaded — the head consumes a RANDOM hidden, so base layers are unused.

Run:
    MESH_DEVICE=P150x4 HF_MODEL=Qwen/Qwen3.6-27B \
      pytest models/demos/blackhole/qwen36/tests/test_mtp_tp.py -v -s
"""
import os

import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.blackhole.qwen36.tests.mtp_torch_ref import load_head_sd, mtp_reference
from models.demos.blackhole.qwen36.tests.test_factory import (
    assert_argmaxes_match_except_near_ties,
    get_pcc_threshold,
    model_path,
    shard_to_device,
)
from models.demos.blackhole.qwen36.tt.attention.rope_tp import rot_mats_prefill
from models.demos.blackhole.qwen36.tt.model_config import Qwen36ModelArgs
from models.tt_transformers.tt.common import Mode

from .test_factory import parametrize_mesh_tp


@torch.no_grad()
@parametrize_mesh_tp()
def test_mtp_head_tp_pcc(mesh_device, reset_seeds, request):
    """Full MTP head (prefill path, internal attention cache) vs the composed torch reference."""
    os.environ.setdefault("HF_MODEL", model_path())
    args = Qwen36ModelArgs(mesh_device, max_batch_size=1, max_seq_len=256)
    nd = mesh_device.get_num_devices()
    logger.info(f"devices={nd} dim={args.dim} NH={args.n_heads} NKV={args.n_kv_heads} rope_hd={args.rope_head_dim}")

    sd = load_head_sd(args.CKPT_DIR)
    assert "mtp.fc.weight" in sd, "mtp.* missing (weight_mapping regression)"

    # n_layers=0: build ONLY embedding + LM head + final norm + MTP head (no base layers loaded).
    args.n_layers = 0
    from models.demos.blackhole.qwen36.tt.model import Qwen36Model

    # The framework Embedding/LM-head require a (writable) cache path; reuse the model's standard one.
    model = Qwen36Model(mesh_device, args, sd, tensor_cache_path=args.weight_cache_path())
    assert model.mtp is not None, "MTP head was not constructed"

    # S must exceed TILE_SIZE (32) so the attention prefill takes the all-gather-matmul path
    # (the fused in-proj gathers the K-sharded activation); a smaller S hits the decode-sized branch.
    S = 64
    hidden = torch.randn(1, S, args.dim, dtype=torch.bfloat16)
    tokens = torch.randint(0, args.vocab_size, (1, S), dtype=torch.long)
    ref_logits = mtp_reference(hidden, tokens, sd, args)  # [S, vocab]

    cos, sin = rot_mats_prefill(mesh_device, args.rope_head_dim, S, args.rope_theta)
    # Prefill activation is K-sharded (the norm skips its AG; the fused in-proj gathers).
    hidden_tt = shard_to_device(mesh_device, hidden.reshape(1, 1, S, args.dim), dim=-1)
    tok_tt = ttnn.from_torch(
        tokens.to(torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    out = model.mtp.forward_prefill(hidden_tt, tok_tt, cos, sin, page_table=None)
    normed = model.mtp.head_norm(out, mode=Mode.PREFILL)
    logits_tt = model.mtp._lm_head(normed)
    # LM head all-gathers -> full logits replicated per device; read replica 0.
    out_torch = ttnn.to_torch(logits_tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    out_torch = out_torch.reshape(-1, out_torch.shape[-1])[:S, : args.vocab_size].float()

    assert not torch.isnan(out_torch).any(), "MTP logits contain NaN"
    passing, pcc = comp_pcc(ref_logits, out_torch, get_pcc_threshold(request, default=0.97))
    logger.info(f"MTP HEAD TP PCC (S={S}) = {pcc}")

    # Argmax agreement is the property speculative decode actually relies on: the drafter's output
    # IS an argmax, so a logit PCC that passes while argmaxes flip would cap acceptance invisibly.
    #
    # The raw agreement rate is LOW here (~0.83) and that is expected: the hidden is random, which
    # is far off the head's training distribution and yields a nearly flat output — the reference's
    # median top-2 gap is only ~0.53. Measured on this input, every flip sits at a reference gap
    # <= 0.51 and none of the ~19 rows with gap >= 1.0 flip, which is the signature of faithful
    # arithmetic resolving near-ties differently, not of a broken head. Hence the 1.0 threshold
    # below. test_mtp_head_on_real_features carries the strict check on in-distribution hiddens.
    ref_ids = ref_logits.argmax(-1).tolist()
    tt_ids = out_torch.argmax(-1).tolist()
    agree = sum(int(a == b) for a, b in zip(ref_ids, tt_ids))
    logger.info(f"MTP argmax agreement {agree}/{S} (random hidden -> near-flat logits)")
    assert_argmaxes_match_except_near_ties(
        [ref_logits[i] for i in range(S)],
        [out_torch[i] for i in range(S)],
        "MTP head vs torch reference (random hidden)",
        near_tie_gap=1.0,
    )
    assert passing, f"MTP head TP PCC too low: {pcc}"


@torch.no_grad()
@parametrize_mesh_tp()
def test_mtp_head_on_real_features(mesh_device, reset_seeds, request):
    """Device MTP head vs the torch reference on REAL base hiddens (Gemma4's
    test_assistant_first_step_vs_hf_realistic rung).

    test_mtp_head_tp_pcc feeds a random hidden, which is off-distribution and leaves the output
    nearly flat — so it cannot tell whether the head's argmax (the thing a draft IS) is faithful.
    This replays the exported base features instead, and additionally reports the drafter's top-1
    agreement with the base's own next token, which is the acceptance rate the device can reach.

    Needs features from test_spec_decode_features.py; skipped when absent.
    """
    import pytest

    from models.demos.blackhole.qwen36.tests.mtp_torch_ref import MTPTorchHead

    feat_path = os.environ.get("QWEN36_MTP_FEATURES", "/tmp/qwen36_spec_features_recurrent.pt").split(",")[0]
    if not os.path.isfile(feat_path):
        pytest.skip(f"no exported features at {feat_path}; run test_spec_decode_features.py first")
    feats = torch.load(feat_path, weights_only=False)

    os.environ.setdefault("HF_MODEL", model_path())
    args = Qwen36ModelArgs(mesh_device, max_batch_size=1, max_seq_len=1024)
    sd = load_head_sd(args.CKPT_DIR)
    args.n_layers = 0  # embedding + LM head + final norm + MTP head only
    from models.demos.blackhole.qwen36.tt.model import Qwen36Model

    model = Qwen36Model(mesh_device, args, sd, tensor_cache_path=args.weight_cache_path())

    # 'shift' alignment: slot i is fused from (base hidden_i, token_{i+1}) and predicts token_{i+2}
    # — the convention mtp_cpu_check.py measured as this checkpoint's, so it is what the device
    # head should be scored under.
    S = 128
    assert feats["hidden"].shape[0] >= S + 2, "need more exported steps"
    hidden = feats["hidden"][:S]
    tokens = feats["tokens"][1 : S + 1]
    target = feats["greedy"][1 : S + 1]  # base's prediction at slot i+1 == token at i+2

    head = MTPTorchHead(sd, rope_dim=args.rope_head_dim, rope_theta=args.rope_theta)
    ref_logits, _, _, _ = head.forward_sequence(hidden, tokens, positions=torch.arange(S))

    cos, sin = rot_mats_prefill(mesh_device, args.rope_head_dim, S, args.rope_theta)
    hidden_tt = shard_to_device(mesh_device, hidden.reshape(1, 1, S, args.dim).to(torch.bfloat16), dim=-1)
    tok_tt = ttnn.from_torch(
        tokens.reshape(1, S).to(torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    out = model.mtp.forward_prefill(hidden_tt, tok_tt, cos, sin, page_table=None)
    normed = model.mtp.head_norm(out, mode=Mode.PREFILL)
    logits_tt = model.mtp._lm_head(normed)
    got = ttnn.to_torch(logits_tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    got = got.reshape(-1, got.shape[-1])[:S, : args.vocab_size].float()

    _, pcc = comp_pcc(ref_logits, got, get_pcc_threshold(request, default=0.97))
    ref_ids, tt_ids = ref_logits.argmax(-1), got.argmax(-1)
    agree = int((ref_ids == tt_ids).sum())
    ref_hit = int((ref_ids == target).sum())
    tt_hit = int((tt_ids == target).sum())
    gaps = torch.topk(ref_logits, 2, dim=-1).values
    gaps = (gaps[:, 0] - gaps[:, 1]).float()
    flip_gaps = gaps[ref_ids != tt_ids]
    logger.info(f"real-feature MTP head PCC={pcc:.6f}  device-vs-reference argmax {agree}/{S}")
    logger.info(f"reference median top-2 gap={gaps.median():.3f} (vs random-hidden ~0.53)")
    logger.info(f"max top-2 gap among flips={float(flip_gaps.max()) if len(flip_gaps) else 0.0:.3f}")
    logger.info(f"draft top-1 vs base next token: reference {ref_hit}/{S}, device {tt_hit}/{S}")

    assert_argmaxes_match_except_near_ties(
        [ref_logits[i] for i in range(S)], [got[i] for i in range(S)], "MTP head on real features"
    )
    # The device must not lose meaningful drafting accuracy relative to its own fp32 reference.
    assert tt_hit >= ref_hit - max(2, int(0.03 * S)), (
        f"device drafter accuracy {tt_hit}/{S} is materially below the fp32 reference {ref_hit}/{S} "
        "— device MTP fidelity is capping acceptance"
    )
