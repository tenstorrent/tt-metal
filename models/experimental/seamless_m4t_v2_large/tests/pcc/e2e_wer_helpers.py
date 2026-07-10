# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""E2E WER helpers for speech-output tasks (T2ST, S2ST).

Compares TT ``generate(generate_speech=True)`` intermediate translation text to offline HF
reference text (same inputs and generation kwargs). Uses ``jiwer.wer`` like the Whisper demo.

Includes sweep utilities (ISL ladder, thresholds, reference ensure), in-process result storage
for pytest summaries, and device run helpers.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jiwer
import numpy as np
import pytest
import torch
import ttnn
from loguru import logger

from models.experimental.seamless_m4t_v2_large.tests.pcc.decoder_pcc_fixtures import (
    TextDecoderPccInputs,
    _hf_speech_encoder_hidden_and_mask,
    _hf_text_encoder_hidden,
    align_case_for_tt_prefill,
    decoder_seed_ids,
)
from models.experimental.seamless_m4t_v2_large.tests.pcc.e2e_logit_pcc_helpers import (
    _hf_teacher_forced_decode_reference,
)
from models.experimental.seamless_m4t_v2_large.tests.pcc.e2e_task_config import (
    maybe_skip_short_speech_input,
)
from models.experimental.seamless_m4t_v2_large.tests.pcc.e2e_token_matching_helpers import (
    TEXT_INPUT_TASKS,
)
from models.experimental.seamless_m4t_v2_large.tests.pcc.e2e_tt_model_helpers import (
    make_tt_model,
    torch_feats_to_ttnn,
    torch_ids_to_ttnn,
)
from models.experimental.seamless_m4t_v2_large.tt.common import (
    build_causal_with_padding_4d,
    build_cross_attn_mask_4d,
    to_torch_replicated_first_shard,
    tt_position_ids,
    tt_position_ids_decode_step,
)
from models.experimental.seamless_m4t_v2_large.tt.mesh_helpers import (
    DEVICE_PARAMS_E2E_2CQ_GENERATE,
    DEVICE_PARAMS_E2E_2CQ_GENERATE_SINGLE,
    MESH_DEVICE_PARAMETRIZE_E2E_2CQ_GENERATE_SINGLE_AND_1X4,
    MESH_SHAPE_SINGLE,
    _mesh_device_param,
    _mesh_device_param_for_shape,
    from_torch_uint32_rm,
    get_tp,
    mesh_default_device,
)
from models.experimental.seamless_m4t_v2_large.tt.tt_text_decoder import (
    init_text_decoder_kv_cache,
    warm_text_decoder_kv_cache_prefill,
)

# ---------------------------------------------------------------------------
# WER thresholds and sweep lengths
# ---------------------------------------------------------------------------

# WER-sweep tasks: T2ST (text in), S2ST (speech in → speech out), ASR (speech in → text out).
WER_SWEEP_TASKS = ("t2st", "s2st", "asr")

# Target language per WER-sweep task. T2ST here targets SPANISH (not the token-matching suite's Hindi)
# so the Whisper round-trip is a FAITHFUL metric across both speech-output tasks: whisper-large-v3
# transcribes Spanish reliably, but amplifies perceptually-inaudible acoustic deltas on Hindi into huge
# word divergence (proven: TT-vs-HF vocoder mel-PCC 0.994 on Hindi, yet whisper WER 1.77 vs Spanish 0.03).
# S2ST (spa) and ASR (eng) are unchanged. This mapping is WER-sweep-local; the token-matching / PCC
# suites keep using e2e_token_matching_helpers.TASK_TGT_LANG (T2ST=hin) via their own references.
WER_TASK_TGT_LANG = {"t2st": "spa", "s2st": "spa", "asr": "eng"}

T2ST_TF_WER_THRESHOLD = 0.20
S2ST_TF_WER_THRESHOLD = 0.30
ASR_TF_WER_THRESHOLD = 0.30

# Whisper round-trip WER (voxtral quality-metrics style): transcribe the GENERATED speech with
# Whisper for HF and TT, then WER between the two transcriptions. Unlike the token-level WER (which
# stops at the intermediate translation), this exercises the full audio path (T2U + vocoder). Only
# for speech-output tasks; ASR already emits text. HF's ASR errors cancel (both use the same Whisper).
WHISPER_ASR_MODEL = "openai/whisper-large-v3"  # large-v3: far better Hindi ASR + less long-audio hallucination
WHISPER_SPEECH_OUTPUT_TASKS = ("t2st", "s2st")
_WHISPER_LANG_BY_TGT = {"hin": "hi", "spa": "es", "eng": "en"}
_WHISPER_ASR_SAMPLE_RATE = 16000
T2ST_WHISPER_WER_THRESHOLD = 0.35
S2ST_WHISPER_WER_THRESHOLD = 0.45
# HF greedy decode horizon for teacher forcing (matches generate max_new_tokens=256).
TEACHER_FORCED_DECODE_STEPS = 256
_TF_EOS_TOKEN_ID = 3

_REF_DIR = Path(__file__).resolve().parent.parent / "teacher_forced_sweep_outputs" / "wer_references"

_P150_WER_SCALE = 1.25

# ---------------------------------------------------------------------------
# In-process result store (pytest summary)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WerResult:
    label: str
    seq_len: int
    wer: float
    wer_threshold: float
    reference_words: int
    tt_words: int
    passed: bool


_WER_RESULTS: list[WerResult] = []


def clear_wer_results() -> None:
    _WER_RESULTS.clear()


def get_wer_results() -> list[WerResult]:
    return list(_WER_RESULTS)


def record_wer_result(
    *,
    label: str,
    seq_len: int,
    wer: float,
    wer_threshold: float,
    reference_words: int,
    tt_words: int,
) -> None:
    _WER_RESULTS.append(
        WerResult(
            label=label,
            seq_len=seq_len,
            wer=wer,
            wer_threshold=wer_threshold,
            reference_words=reference_words,
            tt_words=tt_words,
            passed=wer <= wer_threshold,
        )
    )


def print_wer_summary() -> None:
    results = get_wer_results()
    if not results:
        return
    print("\n" + "=" * 72)
    print("Seamless M4T v2 E2E WER sweep summary")
    print("=" * 72)
    header = f"{'Label':<18} {'Len':>5}  {'WER':>7}  {'Thr':>7}  {'RefW':>5}  {'TTW':>5}  {'Status':>6}"
    print(header)
    print("-" * len(header))
    for row in results:
        status = "PASS" if row.passed else "FAIL"
        print(
            f"{row.label:<18} {row.seq_len:>5}  {row.wer:>6.3f}  {row.wer_threshold:>6.3f}  "
            f"{row.reference_words:>5}  {row.tt_words:>5}  {status:>6}"
        )
    passed = sum(1 for r in results if r.passed)
    print("-" * len(header))
    print(f"Total: {passed}/{len(results)} passed\n")


# ---------------------------------------------------------------------------
# Reference paths and text metrics
# ---------------------------------------------------------------------------


def wer_refpt_path(task: str, seq_len: int) -> Path:
    return _REF_DIR / f"seamless_m4t_v2_{task}_len{seq_len}.refpt"


def normalize_text_for_wer(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def word_count(text: str) -> int:
    norm = normalize_text_for_wer(text)
    return len(norm.split()) if norm else 0


def compute_wer(reference_text: str, hypothesis_text: str) -> float:
    ref = normalize_text_for_wer(reference_text)
    hyp = normalize_text_for_wer(hypothesis_text)
    if not ref and not hyp:
        return 0.0
    if not ref or not hyp:
        return 1.0
    return float(jiwer.wer(ref, hyp))


# ---------------------------------------------------------------------------
# Whisper transcription (round-trip WER on generated speech)
# ---------------------------------------------------------------------------

_WHISPER_CACHE: dict[str, Any] = {}


def _load_whisper():
    """Load and cache Whisper (module-level; CPU) so it isn't reloaded per sweep point."""
    if "model" not in _WHISPER_CACHE:
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        _WHISPER_CACHE["processor"] = WhisperProcessor.from_pretrained(WHISPER_ASR_MODEL)
        # Force fp32: large-v3 ships fp16 weights, but the fp32 input features (and CPU inference)
        # need fp32 or generate() raises "Input type (float) vs bias type (Half)".
        _WHISPER_CACHE["model"] = WhisperForConditionalGeneration.from_pretrained(
            WHISPER_ASR_MODEL, torch_dtype=torch.float32
        ).eval()
    return _WHISPER_CACHE["processor"], _WHISPER_CACHE["model"]


def transcribe_waveform_whisper(waveform_np, src_sr: int, language: str) -> str:
    """Transcribe a 1-D fp32 waveform with Whisper (large-v3), suppressing hallucination/looping.

    Whisper repeats/hallucinates on long or degenerate audio (inflating transcriptions — e.g. a
    150-word translation transcribed to 447 words). We decode independent 30 s chunks — which gives
    ``condition_on_prev_tokens=False`` behavior for free (no cross-chunk context to compound) — and
    pass ``no_repeat_ngram_size=3`` to hard-stop in-chunk phrase loops. (The transformers long-form
    ``compression_ratio_threshold`` path needs a temperature-fallback schedule to populate the
    logprobs it compares, and misfires with a ``None`` comparison otherwise, so we avoid it.)
    """
    import librosa

    processor, whisper = _load_whisper()
    audio = np.asarray(waveform_np, dtype=np.float32).reshape(-1)
    if src_sr != _WHISPER_ASR_SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=src_sr, target_sr=_WHISPER_ASR_SAMPLE_RATE)
    chunk = 30 * _WHISPER_ASR_SAMPLE_RATE
    segments: list[str] = []
    with torch.no_grad():
        for start in range(0, max(len(audio), 1), chunk):
            seg = audio[start : start + chunk]
            if seg.size == 0:
                continue
            inputs = processor(
                seg, sampling_rate=_WHISPER_ASR_SAMPLE_RATE, return_tensors="pt", return_attention_mask=True
            )
            gen_kwargs = {
                "input_features": inputs.input_features,
                "language": language,
                "task": "transcribe",
                "no_repeat_ngram_size": 3,  # hard-stop repeated 3-grams (the hallucination loops)
            }
            if getattr(inputs, "attention_mask", None) is not None:
                gen_kwargs["attention_mask"] = inputs.attention_mask
            ids = whisper.generate(**gen_kwargs)
            segments.append(processor.batch_decode(ids, skip_special_tokens=True)[0].strip())
    return " ".join(s for s in segments if s).strip()


def whisper_language_for_tgt(tgt_lang: str) -> str:
    return _WHISPER_LANG_BY_TGT.get(tgt_lang, "en")


@dataclass(frozen=True)
class WerSweepReference:
    task: str
    seq_len: int
    tgt_lang: str
    reference_text: str
    src_ids: torch.Tensor | None = None
    src_mask: torch.Tensor | None = None
    input_features: torch.Tensor | None = None
    mel_attention_mask: torch.Tensor | None = None
    # Whisper transcription of the HF-generated waveform (speech-output tasks only) — the reference
    # for the whisper round-trip WER. Empty for ASR / older refpts (test skips gracefully then).
    hf_whisper_text: str = ""
    # HF intermediate translation token ids — fed to the TT speech pipeline (T2U+vocoder) for the
    # TEACHER-FORCED whisper round-trip, so the text-decode cascade is excluded. None for ASR / old refpts.
    hf_intermediate_ids: torch.Tensor | None = None
    # The exact (offset-applied) unit ids HF feeds to its vocoder — eos/pad mapped to pad_id and real
    # units shifted by ``-vocoder_offset``. Fed straight to the TT vocoder to ALSO teacher-force the T2U
    # (text→units), leaving a pure vocoder-fidelity whisper metric. None for ASR / old refpts.
    hf_vocoder_input_ids: torch.Tensor | None = None


def load_wer_sweep_reference(path: Path) -> WerSweepReference:
    data = torch.load(path, map_location="cpu", weights_only=False)
    task = str(data["task"])
    return WerSweepReference(
        task=task,
        seq_len=int(data["seq_len"]),
        tgt_lang=str(data["tgt_lang"]),
        reference_text=str(data["reference_text"]),
        src_ids=data.get("src_ids"),
        src_mask=data.get("src_mask"),
        input_features=data.get("input_features"),
        mel_attention_mask=data.get("mel_attention_mask"),
        hf_whisper_text=str(data.get("hf_whisper_text", "")),
        hf_intermediate_ids=data.get("hf_intermediate_ids"),
        hf_vocoder_input_ids=data.get("hf_vocoder_input_ids"),
    )


# ---------------------------------------------------------------------------
# Sweep helpers
# ---------------------------------------------------------------------------


def sweep_auto_ref_enabled() -> bool:
    return os.environ.get("SEAMLESS_SWEEP_AUTO_REF", "1") != "0"


def wer_sweep_mesh_parametrize():
    """Pytest mesh params for WER sweep.

    * ``MESH_DEVICE=P150`` → 1×1 only
    * ``MESH_DEVICE=BH-QB`` → 1×4 only
    * unset → both when hardware allows
    """
    mesh_env = os.environ.get("MESH_DEVICE", "").strip()
    if mesh_env == "P150":
        return (
            "mesh_device,device_params",
            [_mesh_device_param_for_shape(MESH_SHAPE_SINGLE, DEVICE_PARAMS_E2E_2CQ_GENERATE_SINGLE, id="1x1")],
        )
    if mesh_env == "BH-QB":
        return ("mesh_device,device_params", [_mesh_device_param(DEVICE_PARAMS_E2E_2CQ_GENERATE)])
    return MESH_DEVICE_PARAMETRIZE_E2E_2CQ_GENERATE_SINGLE_AND_1X4


def sweep_teacher_forced_wer_threshold_for_task(task: str, seq_len: int, *, mesh_id: str = "") -> float:
    """Teacher-forced WER threshold (length-independent; scaled up on 1×1 like free-running)."""
    _ = seq_len
    if task not in WER_SWEEP_TASKS:
        raise ValueError(f"teacher-forced WER expects one of {WER_SWEEP_TASKS}, got {task!r}")
    base = {"t2st": T2ST_TF_WER_THRESHOLD, "s2st": S2ST_TF_WER_THRESHOLD, "asr": ASR_TF_WER_THRESHOLD}[task]
    if mesh_id == "1x1":
        return min(1.0, base * _P150_WER_SCALE)
    return base


def sweep_whisper_wer_threshold_for_task(task: str, seq_len: int, *, mesh_id: str = "") -> float:
    """Whisper round-trip WER threshold (speech-output tasks only; scaled up on 1×1)."""
    _ = seq_len
    if task not in WHISPER_SPEECH_OUTPUT_TASKS:
        raise ValueError(f"whisper WER expects one of {WHISPER_SPEECH_OUTPUT_TASKS}, got {task!r}")
    base = {"t2st": T2ST_WHISPER_WER_THRESHOLD, "s2st": S2ST_WHISPER_WER_THRESHOLD}[task]
    if mesh_id == "1x1":
        return min(1.0, base * _P150_WER_SCALE)
    return base


def ensure_wer_sweep_reference(task: str, seq_len: int, weights_dir: str) -> Path:
    if task not in WER_SWEEP_TASKS:
        raise ValueError(f"WER sweep expects one of {WER_SWEEP_TASKS}, got {task!r}")
    ref_path = wer_refpt_path(task, seq_len)
    if ref_path.is_file():
        return ref_path
    if not sweep_auto_ref_enabled():
        gen = "models/experimental/seamless_m4t_v2_large/scripts/generate_wer_sweep_reference.py"
        pytest.skip(
            f"Missing WER sweep reference {ref_path}. Run: "
            f"python {gen} --task {task} --seq_len {seq_len} "
            f"(or set SEAMLESS_SWEEP_AUTO_REF=1 to generate on first run)"
        )
    from models.experimental.seamless_m4t_v2_large.scripts.generate_wer_sweep_reference import (
        generate_speech_wer_sweep_reference,
        generate_text_wer_sweep_reference,
    )

    if task in TEXT_INPUT_TASKS:
        generate_text_wer_sweep_reference(
            weights_dir=weights_dir,
            output_file=ref_path,
            task=task,
            seq_len=seq_len,
        )
    else:
        generate_speech_wer_sweep_reference(
            weights_dir=weights_dir,
            output_file=ref_path,
            task=task,
            mel_frames=seq_len,
        )
    return ref_path


def maybe_skip_empty_wer_reference(ref_text: str, *, task: str, seq_len: int) -> None:
    if ref_text.strip():
        return
    pytest.skip(f"{task.upper()} WER sweep len={seq_len}: HF reference text is empty " f"(likely EOS on short input)")


def maybe_skip_short_speech_wer(task: str, ref_text: str, seq_len: int) -> None:
    """Skip speech-input WER points with too few mel frames (see ``maybe_skip_short_speech_input``).

    ``ref_text`` is unused; kept for call-site compatibility.
    """
    _ = ref_text
    maybe_skip_short_speech_input(task, seq_len)


# ---------------------------------------------------------------------------
# TT device run
# ---------------------------------------------------------------------------


def _tt_waveform_to_mono_np(waveform_tt: ttnn.Tensor, lengths_tt: ttnn.Tensor) -> "np.ndarray":
    """TT vocoder output ``[B, T, 1]`` → 1-D fp32 numpy, trimmed to the valid sample count."""
    arr = to_torch_replicated_first_shard(waveform_tt).float().reshape(-1).cpu().numpy()
    valid_len = int(to_torch_replicated_first_shard(lengths_tt).long().reshape(-1)[0].item())
    if 0 < valid_len <= arr.size:
        arr = arr[:valid_len]
    return arr


def run_speech_output_whisper_wer(
    mesh_device: ttnn.Device,
    hf_model: Any,
    tokenizer: Any,
    ref: WerSweepReference,
    *,
    wer_threshold: float,
    log_label: str,
) -> None:
    """Teacher-forced whisper round-trip WER — pure vocoder fidelity (T2U teacher-forced too).

    Feeds HF's *exact* vocoder input (``ref.hf_vocoder_input_ids`` — the offset-applied unit ids HF
    fed to its own vocoder) straight into the TT vocoder via ``tt_model.vocode_units(...)``, then
    transcribes the TT waveform with Whisper and WERs it against the HF-generated waveform's Whisper
    transcription (``ref.hf_whisper_text``). Because BOTH the text decode and the T2U (text→units) are
    teacher-forced from HF, the only variables are the TT vocoder (+ Whisper ASR) — so this gate is
    unaffected by the free-running T2U divergence. Speech-output tasks only.
    """
    if ref.task not in WHISPER_SPEECH_OUTPUT_TASKS:
        raise ValueError(f"whisper WER expects one of {WHISPER_SPEECH_OUTPUT_TASKS}, got {ref.task!r}")
    if not ref.hf_whisper_text.strip() or ref.hf_vocoder_input_ids is None:
        pytest.skip(
            f"{log_label}: refpt lacks hf_whisper_text/hf_vocoder_input_ids — regenerate references "
            f"(delete the .refpt or run generate_wer_sweep_reference.py) for the teacher-forced whisper round-trip"
        )
    cfg = hf_model.config
    t2u_cfg = hf_model.t2u_model.config
    sample_rate = int(getattr(cfg, "sampling_rate", 16000))
    language = whisper_language_for_tgt(ref.tgt_lang)

    with mesh_default_device(mesh_device):
        tt_model = make_tt_model(mesh_device, hf_model, cfg, t2u_cfg)
        try:
            # Feed HF's exact (offset-applied) unit ids to ONLY the TT vocoder — teacher-forces the
            # T2U so the metric measures vocoder fidelity alone (no encoder / text-decode / T2U run).
            wav_tt, lengths_tt = tt_model.vocode_units(ref.hf_vocoder_input_ids, ref.tgt_lang, speaker_id=0)
            tt_wav = _tt_waveform_to_mono_np(wav_tt, lengths_tt)
            ttnn.deallocate(wav_tt)
            ttnn.deallocate(lengths_tt)
        finally:
            tt_model.release_generation_runtime()

    tt_text = transcribe_waveform_whisper(tt_wav, sample_rate, language)
    wer = compute_wer(ref.hf_whisper_text, tt_text)
    ref_words = word_count(ref.hf_whisper_text)
    tt_words = word_count(tt_text)
    logger.info(
        f"SeamlessM4Tv2 E2E teacher-forced whisper-WER ({log_label}) seq={ref.seq_len} lang={language} "
        f"WER={wer:.4f} (threshold<={wer_threshold:.3f}, hf_words={ref_words}, tt_words={tt_words})"
    )
    logger.info(f"  HF whisper: {ref.hf_whisper_text[:240]}{'…' if len(ref.hf_whisper_text) > 240 else ''}")
    logger.info(f"  TT whisper: {tt_text[:240]}{'…' if len(tt_text) > 240 else ''}")

    record_wer_result(
        label=log_label,
        seq_len=ref.seq_len,
        wer=wer,
        wer_threshold=wer_threshold,
        reference_words=ref_words,
        tt_words=tt_words,
    )
    assert wer <= wer_threshold, (
        f"{log_label}: whisper WER {wer:.4f} > threshold {wer_threshold:.3f} "
        f"(hf_words={ref_words}, tt_words={tt_words}). TT whisper: {tt_text!r}"
    )


# ---------------------------------------------------------------------------
# Teacher-forced WER (fidelity gate; no free-running cascade)
# ---------------------------------------------------------------------------


def _tt_encoder_for_task(mesh_device: ttnn.Device, tt_model: Any, ref: WerSweepReference):
    """Run the PRODUCTION TT encoder (same path as ``generate``) on the real sweep inputs.

    Returns ``(enc_tt, enc_mask_tt)``. Uses ``tt_model._encode_text`` / ``_encode_speech`` so the
    speech encoder is built exactly as in production (``matmul_token_rows=64``) — the logit-PCC
    helper's ``tt_encode_speech`` rebuilds it per-call with a different config and corrupts the
    cross-attention context at mel 1024/2048 (enc_seq 129/257).
    """
    if ref.task == "t2st":
        assert ref.src_ids is not None and ref.src_mask is not None
        ids_tt = torch_ids_to_ttnn(mesh_device, ref.src_ids)
        mask_tt = torch_ids_to_ttnn(mesh_device, ref.src_mask)
        enc_tt, enc_mask_tt, _ = tt_model._encode_text(ids_tt, mask_tt)
        ttnn.deallocate(ids_tt)
        if enc_mask_tt is not mask_tt:
            ttnn.deallocate(mask_tt)
        return enc_tt, enc_mask_tt

    assert ref.task in ("s2st", "asr")  # both are speech-input; only the decoder target lang differs
    assert ref.input_features is not None and ref.mel_attention_mask is not None
    tt_model.prewarm_speech_encoder([int(ref.mel_attention_mask.sum().item())])
    feats_tt = torch_feats_to_ttnn(mesh_device, ref.input_features)
    mask_tt = torch_ids_to_ttnn(mesh_device, ref.mel_attention_mask)
    enc_tt, enc_mask_tt, _ = tt_model._encode_speech(feats_tt, mask_tt)
    ttnn.deallocate(feats_tt)
    if enc_mask_tt is not mask_tt:
        ttnn.deallocate(mask_tt)
    return enc_tt, enc_mask_tt


def _teacher_forced_tt_predictions(
    mesh_device: ttnn.Device,
    hf_model: Any,
    tt_dec: Any,
    enc_tt: ttnn.Tensor,
    enc_mask_tt: ttnn.Tensor,
    dec_seed: torch.Tensor,
    decode_tokens: list[int],
    horizon: int,
) -> list[int]:
    """Feed HF teacher tokens to the production TT decoder step-by-step; return per-step argmax tokens.

    ``tt_dec`` is the full model's ``text_decoder``. ``tt_pred[0]`` is TT's prediction after prefill
    (aligns to ``decode_tokens[0]``); ``tt_pred[i]`` after consuming ``decode_tokens[i-1]``.
    Deallocates ``enc_tt`` / ``enc_mask_tt`` and the fresh KV caches it builds.
    """
    cfg = hf_model.config
    lm_head = hf_model.lm_head
    pad_id = int(cfg.pad_token_id)
    hidden_size = int(cfg.hidden_size)
    n_heads = int(cfg.decoder_attention_heads)
    p_dtype = next(hf_model.text_decoder.parameters()).dtype

    logical_dec = int(dec_seed.shape[1])
    padded_dec = ((logical_dec + 31) // 32) * 32
    # Encoder length comes from the TT encoder tensor itself (native TT timeline), matching enc_mask_tt.
    padded_enc = int(enc_tt.shape[1])
    max_seq_len = max(64, logical_dec + horizon + 8)

    def _greedy(row: torch.Tensor) -> int:
        with torch.no_grad():
            return int(lm_head(row.to(p_dtype))[0].argmax().item())

    kv_cache, cross_attn_cache = init_text_decoder_kv_cache(
        mesh_device,
        num_hidden_layers=cfg.decoder_layers,
        num_attention_heads=n_heads,
        hidden_size=hidden_size,
        max_batch_size=1,
        max_seq_len=max_seq_len,
        encoder_seq_len=padded_enc,
        tp=get_tp(mesh_device),
    )

    if padded_dec > logical_dec:
        tail = torch.full((1, padded_dec - logical_dec), pad_id, dtype=dec_seed.dtype)
        ids_padded = torch.cat([dec_seed, tail], dim=1)
    else:
        ids_padded = dec_seed
    ids_tt = from_torch_uint32_rm(mesh_device, ids_padded)
    pos_tt = tt_position_ids(ids_tt, pad_id)
    causal_tt = build_causal_with_padding_4d(None, 1, padded_dec, mesh_device)
    cross_prefill_tt = build_cross_attn_mask_4d(enc_mask_tt, tgt_seq=padded_dec, device=mesh_device)
    prefill_dev = warm_text_decoder_kv_cache_prefill(
        tt_dec,
        ids_tt,
        pos_tt,
        enc_tt,
        causal_tt,
        cross_prefill_tt,
        kv_cache,
        cross_attn_cache,
        kv_cache_fill_len=logical_dec,
    )
    tt_prefill = (to_torch_replicated_first_shard(prefill_dev).to(torch.bfloat16).reshape(1, padded_dec, hidden_size))[
        :, :logical_dec, :
    ].contiguous()
    ttnn.deallocate(prefill_dev)
    ttnn.deallocate(ids_tt)
    ttnn.deallocate(pos_tt)
    ttnn.deallocate(causal_tt)
    ttnn.deallocate(cross_prefill_tt)

    tt_pred = [_greedy(tt_prefill[:, -1, :])]
    cross_decode_tt = build_cross_attn_mask_4d(enc_mask_tt, tgt_seq=1, device=mesh_device)
    for step in range(horizon):
        position = logical_dec + step
        token_ids = from_torch_uint32_rm(mesh_device, torch.full((1, 1), decode_tokens[step], dtype=torch.int32))
        step_pos = tt_position_ids_decode_step(token_ids, pad_id, position)
        cur_pos = tt_dec.borrow_current_decode_pos_tensor(position, batch_size=1)
        dec_dev = tt_dec.forward(
            token_ids,
            step_pos,
            enc_tt,
            None,
            cross_decode_tt,
            kv_cache=kv_cache,
            cross_attn_cache=cross_attn_cache,
            cross_attn_cache_valid=True,
            current_decode_pos=cur_pos,
            cache_seq_len=position + 1,
        )
        tt_step = to_torch_replicated_first_shard(dec_dev).to(torch.bfloat16).reshape(1, 1, hidden_size).contiguous()
        ttnn.deallocate(dec_dev)
        ttnn.deallocate(token_ids)
        ttnn.deallocate(step_pos)
        tt_pred.append(_greedy(tt_step[:, -1, :]))

    ttnn.deallocate(enc_tt)
    ttnn.deallocate(enc_mask_tt)
    ttnn.deallocate(cross_decode_tt)
    for layer in kv_cache:
        ttnn.deallocate(layer[0])
        ttnn.deallocate(layer[1])
    for layer in cross_attn_cache:
        ttnn.deallocate(layer[0])
        ttnn.deallocate(layer[1])
    return tt_pred


def run_speech_output_teacher_forced_wer(
    mesh_device: ttnn.Device,
    hf_model: Any,
    tokenizer: Any,
    ref: WerSweepReference,
    *,
    wer_threshold: float,
    log_label: str,
    decode_steps: int = TEACHER_FORCED_DECODE_STEPS,
) -> None:
    """Teacher-forced WER: feed HF's reference tokens to the TT decoder (no free-running cascade).

    Measures per-step decoder fidelity to HF — a stable gate, unlike free-running WER which
    amplifies a single low-margin token flip on degenerate input into a divergent generation.
    """
    cfg = hf_model.config
    t2u_cfg = hf_model.t2u_model.config
    pad_id = int(cfg.pad_token_id)
    # Use the refpt's tgt_lang (WER-sweep-local, e.g. T2ST=spa) — the source of truth for this ref.
    dec_seed = decoder_seed_ids(hf_model, ref.tgt_lang)

    # HF greedy teacher tokens from the HF encoder hidden (the reference the TT decoder must match).
    if ref.task == "t2st":
        assert ref.src_ids is not None and ref.src_mask is not None
        enc_hidden = _hf_text_encoder_hidden(hf_model, ref.src_ids, ref.src_mask)
        enc_attn_mask = ref.src_mask.to(enc_hidden.device)
    else:
        assert ref.task in ("s2st", "asr")  # speech-input; decoder target lang differs (spa vs eng)
        assert ref.input_features is not None and ref.mel_attention_mask is not None
        enc_hidden, enc_attn_mask = _hf_speech_encoder_hidden_and_mask(
            hf_model, ref.input_features, ref.mel_attention_mask
        )
    case = TextDecoderPccInputs(
        input_ids=dec_seed,
        attention_mask=torch.ones_like(dec_seed),
        encoder_hidden_states=enc_hidden,
        encoder_attention_mask=enc_attn_mask,
    )
    aligned = align_case_for_tt_prefill(case, pad_id)
    hf_logical_enc = aligned.logical_enc_seq
    _, _, decode_tokens = _hf_teacher_forced_decode_reference(
        hf_model, hf_model.text_decoder, case, aligned, decode_steps
    )
    horizon = decode_steps - 1
    for s, tok in enumerate(decode_tokens):
        if tok == _TF_EOS_TOKEN_ID:
            horizon = min(horizon, s)
            break

    # TT side: production encoder + production decoder (same as generate), teacher-forced.
    with mesh_default_device(mesh_device):
        tt_model = make_tt_model(mesh_device, hf_model, cfg, t2u_cfg)
        try:
            enc_tt, enc_mask_tt = _tt_encoder_for_task(mesh_device, tt_model, ref)
            tt_pred = _teacher_forced_tt_predictions(
                mesh_device, hf_model, tt_model.text_decoder, enc_tt, enc_mask_tt, dec_seed, decode_tokens, horizon
            )
        finally:
            tt_model.release_generation_runtime()

    hf_seq = decode_tokens[: horizon + 1]
    hf_text = tokenizer.batch_decode([hf_seq], skip_special_tokens=True)[0]
    tt_text = tokenizer.batch_decode([tt_pred], skip_special_tokens=True)[0]
    wer = compute_wer(hf_text, tt_text)
    ref_words = word_count(hf_text)
    tt_words = word_count(tt_text)
    logger.info(
        f"SeamlessM4Tv2 E2E teacher-forced WER ({log_label}) seq={ref.seq_len} enc_seq={hf_logical_enc} "
        f"WER={wer:.4f} (threshold<={wer_threshold:.3f}, hf_words={ref_words}, tt_words={tt_words}, "
        f"horizon={horizon + 1})"
    )
    logger.info(f"  HF teacher: {hf_text[:240]}{'…' if len(hf_text) > 240 else ''}")
    logger.info(f"  TT forced:  {tt_text[:240]}{'…' if len(tt_text) > 240 else ''}")

    record_wer_result(
        label=log_label,
        seq_len=ref.seq_len,
        wer=wer,
        wer_threshold=wer_threshold,
        reference_words=ref_words,
        tt_words=tt_words,
    )
    assert wer <= wer_threshold, (
        f"{log_label}: teacher-forced WER {wer:.4f} > threshold {wer_threshold:.3f} "
        f"(hf_words={ref_words}, tt_words={tt_words}). TT text: {tt_text!r}"
    )
