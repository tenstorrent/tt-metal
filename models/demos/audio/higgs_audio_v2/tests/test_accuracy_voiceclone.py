# SPDX-FileCopyrightText: (c) 2026 Tenstorrent
# SPDX-License-Identifier: Apache-2.0
"""Teacher-forced token-accuracy check for the VOICE-CLONE conditioning path.

Companion to ``test_accuracy_native.py`` (which scores the short text-only TTS
path at 0.967). That test never exercises the reference-audio conditioning — the
prefill mask-blend that splices reference-audio embeddings into the
``<|AUDIO_OUT|>`` placeholders and routes them through the audio DualFFN branch.
This test exercises it, and guards against a conditioning regression.

Method (identical scoring to the TTS gate, conditioning added):
  1. Build a real voice-clone prompt with the HF processor (system + a reference
     ``assistant`` audio turn + target text) -> input_ids (with audio
     placeholders) + audio_input_ids/_mask (the encoded reference codes).
     (VC_MODE=tts builds a LONG text-only prompt of comparable length with NO
     conditioning — the control used to establish the finding below.)
  2. HF (native-B) ``generate()`` a conditioned trajectory. NOTE: native-B's
     ``audio_sequences`` PREPENDS the reference frames (verified seq[:n_valid]==
     reference); we skip those and teacher-force only the *generated* tail.
  3. Teacher-force BOTH native-B and TTNN through the same incremental path,
     prefilling WITH the reference-audio merge on both sides, and compare
     per-step argmax over the delay-pattern active mask.

FINDING (measured): the conditioning is FAITHFUL — it is NOT a source of
degradation. Teacher-forced accuracy tracks PREFILL LENGTH, not conditioning:
short TTS (S=36) = 0.967, long text-only (S=71) = 0.86, voice-clone (S=124) =
0.86. Voice-clone matches a length-matched text-only prompt, so the audio
conditioning adds no error beyond what prompt length alone costs. (Conditioned
modes still gibber when free-running because they start from this lower ~0.86
per-step fidelity and drift compounds — addressed at generation time with
best-of-N selection, not here.) The gate below is a regression floor: a genuinely
broken conditioning path (wrong DualFFN branch / bad splice) collapses well below
0.7, like the bf4-accuracy failures; faithful conditioning sits at ~0.86.
"""
import gc
import os
import pathlib

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.audio.higgs_audio_v2.tt.audio_decode import (
    apply_delay_pattern_to_greedy_audio_tokens,
    initialize_delay_pattern_state,
)
from models.demos.audio.higgs_audio_v2.tt.model import HiggsAudioTTModel
from models.demos.audio.higgs_audio_v2.tt.model_args import HiggsModelArgs
from models.demos.audio.higgs_audio_v2.tt.reference import HiggsAudioV2Config, load_higgs_v2_state_dict
from models.tt_transformers.tt.ccl import TT_CCL
from models.tt_transformers.tt.rope import HfRotarySetup, RotarySetup

HIGGS_MODEL_DIR = "/data/hf_cache/higgs"
REF_WAV = os.environ.get("VC_REF_WAV", "/tmp/higgs_pick/tts_male_voice.wav")
TARGET_TEXT = "The quick brown fox jumps over the lazy dog."
T_FRAMES = int(os.environ.get("VC_FRAMES", "48"))
# Target: the conditioning path should match the short-TTS gate (>=0.95). The
# baseline long-prefill measurement was ~0.86 (prefill-length accumulation error);
# the goal is to recover that to >=0.95, not to relax the gate.
ACCURACY_MIN_TOKEN_ACCURACY = 0.95


class _DelayCfg:
    def __init__(self, args):
        self.audio_num_codebooks = args.audio_num_codebooks
        self.audio_stream_bos_id = args.audio_stream_bos_id
        self.audio_stream_eos_id = args.audio_stream_eos_id
        self.use_delay_pattern = True


@pytest.fixture(scope="module")
def mesh_device():
    dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    yield dev
    ttnn.close_mesh_device(dev)


def _rot_inputs(rope_setup, mesh_device, args, pos: int):
    cp = torch.tensor([pos], dtype=torch.int32)
    rope_idxs = rope_setup.get_rot_idxs(cp, on_host=True)
    rope_idxs = ttnn.to_device(rope_idxs, mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    cp_tt = ttnn.from_torch(
        cp,
        device=mesh_device,
        dtype=ttnn.int32,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, None), mesh_shape=args.cluster_shape),
    )
    return cp_tt, rope_setup.get_rot_mats(rope_idxs)


_LONG_TEXT = (
    "The quick brown fox jumps over the lazy dog while the morning sun rises slowly "
    "over the quiet valley, and a gentle breeze carries the distant sound of church "
    "bells across the rolling green hills toward the sleepy little town below."
)


def _build_voiceclone_inputs():
    """Build the prompt. VC_MODE=voiceclone (default) splices a reference audio
    turn (the conditioning under test); VC_MODE=tts builds a LONG text-only prompt
    of comparable length (~120 tokens) with NO conditioning — the control that
    isolates 'long prefill' from 'audio conditioning'."""
    from transformers import AutoProcessor

    mode = os.environ.get("VC_MODE", "voiceclone")
    proc = AutoProcessor.from_pretrained(HIGGS_MODEL_DIR)
    if mode == "tts":
        conv = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "Generate speech in the style of a calm neutral male voice."}],
            },
            {"role": "user", "content": [{"type": "text", "text": _LONG_TEXT}]},
        ]
    else:
        assert pathlib.Path(REF_WAV).exists(), f"reference wav missing: {REF_WAV}"
        conv = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "Generate speech in the style of a calm neutral male voice."}],
            },
            {"role": "user", "content": [{"type": "text", "text": "Please speak in this voice."}]},
            {"role": "assistant", "content": [{"type": "audio", "url": REF_WAV}]},
            {"role": "user", "content": [{"type": "text", "text": TARGET_TEXT}]},
        ]
    enc = proc.apply_chat_template(
        conv, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
    )
    return proc, enc


def test_accuracy_voiceclone(mesh_device):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    proc, enc = _build_voiceclone_inputs()
    input_ids = enc["input_ids"][0].to(torch.int64)
    audio_in = enc.get("audio_input_ids", None)
    audio_mask = enc.get("audio_input_ids_mask", None)
    audio_token_id = proc.audio_token_id
    S = input_ids.shape[0]
    n_ph = int((input_ids == audio_token_id).sum())
    logger.info(
        f"VC prompt: mode={os.environ.get('VC_MODE','voiceclone')} S={S} placeholders={n_ph} "
        f"audio_in={None if audio_in is None else tuple(audio_in.shape)} "
        f"valid={None if audio_mask is None else int(audio_mask.sum())}"
    )

    # ---- native-B: generate a conditioned trajectory, then teacher-force it ----
    from transformers import HiggsAudioV2ForConditionalGeneration
    from transformers.cache_utils import DynamicCache

    logger.info("[HF] loading native-B (CPU bf16)...")
    hf_model = HiggsAudioV2ForConditionalGeneration.from_pretrained(HIGGS_MODEL_DIR, dtype=torch.bfloat16).to("cpu")
    hf_model.eval()

    with torch.no_grad():
        out = hf_model.generate(
            **enc, max_new_tokens=T_FRAMES, do_sample=False, return_dict_in_generate=True, output_scores=False
        )
    # HF's audio_sequences PREPENDS the reference frames (verified: seq[:n_valid]==ref).
    # The genuinely *generated* continuation — the only thing decoded at positions >= S —
    # is the tail after those reference frames. Teacher-force only that.
    n_skip = 0 if audio_in is None else audio_in.shape[1]
    ref = torch.as_tensor(out.audio_sequences[0], dtype=torch.long)[n_skip:]  # [T, K] generated only
    T, K = ref.shape
    logger.info(
        f"[HF] trajectory {tuple(out.audio_sequences[0].shape)} -> skipped {n_skip} reference frames, "
        f"scoring {T} generated frames"
    )
    logger.info(f"[HF] conditioned target trajectory: {tuple(ref.shape)}")

    backbone = hf_model.model if hasattr(hf_model, "model") else hf_model
    captured = []
    h = backbone.norm.register_forward_hook(lambda _m, _i, o: captured.append(o.detach().cpu().float()))
    with torch.no_grad():
        hf_cache = DynamicCache(config=hf_model.config)
        # PREFILL WITH the reference-audio merge (the conditioning under test)
        _ = backbone(
            input_ids=input_ids.unsqueeze(0),
            audio_input_ids=audio_in,
            audio_input_ids_mask=audio_mask,
            past_key_values=hf_cache,
            use_cache=True,
        )
        for k in range(1, T):
            _ = backbone(audio_input_ids=ref[k - 1].view(1, 1, K), past_key_values=hf_cache, use_cache=True)
    h.remove()
    audio_head_w = hf_model.audio_lm_head.weight.detach().float()
    ref_logits = {k: (captured[k][0, -1, :].float() @ audio_head_w.t()).reshape(K, -1) for k in range(1, T)}
    del hf_model, backbone, hf_cache, captured
    gc.collect()

    # ---- TTNN model (accuracy preset; same conditioning prefill) --------------
    higgs_cfg = HiggsAudioV2Config.from_json(pathlib.Path(HIGGS_MODEL_DIR) / "config.json")
    precision = os.environ.get("HIGGS_PRECISION", "accuracy")
    max_seq = int(os.environ.get("HIGGS_MAXSEQ", "1024"))
    if precision == "accuracy":
        args = HiggsModelArgs(mesh_device=mesh_device, higgs_config=higgs_cfg, max_batch_size=1, max_seq_len=max_seq)
    else:
        from models.demos.audio.higgs_audio_v2.tt.model_args import BASE_TEXT_MODEL
        from models.demos.audio.higgs_audio_v2.tt.precision_presets import build_precision

        opt = build_precision(precision, higgs_cfg.num_hidden_layers, BASE_TEXT_MODEL)
        args = HiggsModelArgs(
            mesh_device=mesh_device, higgs_config=higgs_cfg, max_batch_size=1, max_seq_len=max_seq, optimizations=opt
        )
    logger.info(f"HIGGS_PRECISION={precision} max_seq_len={max_seq}")
    _, state_dict = load_higgs_v2_state_dict(HIGGS_MODEL_DIR)
    tt_ccl = TT_CCL(mesh_device)
    RopeCls = HfRotarySetup if args.use_hf_rope else RotarySetup
    rope_setup = RopeCls(
        device=mesh_device,
        batch_size=args.max_batch_size,
        head_dim=args.head_dim,
        max_seq_len=args.max_seq_len,
        rope_theta=args.rope_theta,
        rope_scaling=args.rope_scaling,
        use_qk_fused=args.use_qk_fused,
        prefetcher=None,
    )
    act_dtype = {"bf8": ttnn.bfloat8_b, "bf16": ttnn.bfloat16}[os.environ.get("HIGGS_ACT_DTYPE", "bf8")]
    logger.info(f"activation/block dtype = {act_dtype}")
    model = HiggsAudioTTModel(
        args=args,
        mesh_device=mesh_device,
        tt_ccl=tt_ccl,
        state_dict=state_dict,
        transformation_mats=rope_setup.get_both_trans_mats(),
        dtype=act_dtype,
    )
    cfg = _DelayCfg(args)
    _ = model.prefill_text(
        input_ids, rope_setup, audio_input_ids=audio_in, audio_input_ids_mask=audio_mask, audio_token_id=audio_token_id
    )

    # ---- score both sides through the same delay state machine ----------------
    num_delay_r, num_rem_r = initialize_delay_pattern_state(ref[:1].transpose(0, 1), cfg)
    num_delay_t, num_rem_t = initialize_delay_pattern_state(ref[:1].transpose(0, 1), cfg)
    token_matches, token_total, raw_matches, raw_total = 0, 0, 0, 0
    pccs = []  # per-step,per-codebook logit PCC (TTNN vs HF)
    miss_margins = []  # for active mismatches: HF's logit margin between its pick and TTNN's pick
    per_step = []  # (k, n_active, n_match, mean_pcc) to localize prefill-context vs decode-position
    for k in range(1, T):
        pos = S + k - 1
        cp_tt, rot_mats = _rot_inputs(rope_setup, mesh_device, args, pos)
        audio_in_k = ref[k - 1].long().clamp(min=0, max=args.audio_codebook_size - 1)
        tt_logits = model.decode_step_audio(audio_in_k, cp_tt, rot_mats)  # [K, cb]

        ref_next, active_mask, num_delay_r, num_rem_r, fin_r = apply_delay_pattern_to_greedy_audio_tokens(
            ref_logits[k], cfg, num_delay_r, num_rem_r
        )
        tt_next, _tt_active, num_delay_t, num_rem_t, _fin_t = apply_delay_pattern_to_greedy_audio_tokens(
            tt_logits, cfg, num_delay_t, num_rem_t
        )
        # diagnostics: is the divergence near-tie noise or gross? per-codebook logit PCC,
        # and for each ACTIVE mismatch the margin HF assigns between its argmax and TTNN's.
        rl, tl = ref_logits[k].float(), tt_logits.float()
        step_pccs, step_active, step_match = [], 0, 0
        for cb in range(rl.shape[0]):
            if not bool(active_mask[cb]):
                continue
            step_active += 1
            p = float(torch.corrcoef(torch.stack([rl[cb], tl[cb]]))[0, 1])
            pccs.append(p)
            step_pccs.append(p)
            r_arg, t_arg = int(rl[cb].argmax()), int(tl[cb].argmax())
            if r_arg == t_arg:
                step_match += 1
            else:
                miss_margins.append(float(rl[cb][r_arg] - rl[cb][t_arg]))
        per_step.append((k, step_active, step_match, sum(step_pccs) / max(1, len(step_pccs))))
        match = tt_next == ref_next
        token_matches += int((match & active_mask).sum().item())
        token_total += int(active_mask.sum().item())
        raw_matches += int(match.sum().item())
        raw_total += int(match.numel())
        if k in (1, 5, 9, 15, 23):
            logger.info(
                f"  step {k:>2d}: active={active_mask.int().tolist()} tt={tt_next.tolist()} ref={ref_next.tolist()}"
            )
        if fin_r:
            break

    active_acc = token_matches / max(1, token_total)
    raw_acc = raw_matches / max(1, raw_total)
    import numpy as _np

    mean_pcc = float(_np.mean(pccs)) if pccs else float("nan")
    mm = _np.array(miss_margins) if miss_margins else _np.array([0.0])
    near = int((mm < 0.05).sum())
    small = int((mm < 0.10).sum())
    logger.info(f"[VC teacher-forced] active-mask token accuracy: {active_acc:.4f} ({token_matches}/{token_total})")
    logger.info(f"[VC teacher-forced] raw token accuracy:         {raw_acc:.4f} ({raw_matches}/{raw_total})")
    logger.info(f"[diag] mean active-codebook logit PCC = {mean_pcc:.5f}")
    logger.info(
        f"[diag] active mismatches: {len(miss_margins)}; HF margin over TTNN's pick: "
        f"min={mm.min():.4f} median={_np.median(mm):.4f} max={mm.max():.4f}; "
        f"<0.05: {near}/{len(miss_margins)}  <0.10: {small}/{len(miss_margins)}"
    )
    # localize: accuracy + PCC across thirds of the trajectory (decode position increases with k)
    nps = len(per_step)
    for lbl, lo, hi in [
        ("first-third", 0, nps // 3),
        ("mid-third", nps // 3, 2 * nps // 3),
        ("last-third", 2 * nps // 3, nps),
    ]:
        chunk = per_step[lo:hi]
        a = sum(m for _, _, m, _ in chunk)
        n = sum(act for _, act, _, _ in chunk)
        pc = _np.mean([p for _, _, _, p in chunk]) if chunk else float("nan")
        ks = f"k={chunk[0][0]}..{chunk[-1][0]}" if chunk else "-"
        logger.info(
            f"[diag] {lbl:11s} ({ks}, pos {S + (chunk[0][0] if chunk else 0)}..): "
            f"acc={a}/{n}={a / max(1, n):.3f}  logitPCC={pc:.5f}"
        )
    print(
        f"VC_TF_ACTIVE_ACC={active_acc:.4f} ACTIVE={token_matches}/{token_total} RAW={raw_acc:.4f} "
        f"LOGIT_PCC={mean_pcc:.5f} NEARTIE_MISS={near}/{len(miss_margins)} (TTS baseline 0.967)"
    )

    assert active_acc >= ACCURACY_MIN_TOKEN_ACCURACY, (
        f"voice-clone active-mask token accuracy {active_acc:.4f} < regression floor "
        f"{ACCURACY_MIN_TOKEN_ACCURACY} -> the conditioning path regressed (faithful conditioning is "
        f"~0.86, == a length-matched text-only prompt; this low suggests a broken DualFFN branch or splice)"
    )
