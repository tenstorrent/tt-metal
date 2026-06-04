# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end PCC test for SeamlessM4Tv2 — ``TT generate()`` vs ``HF generate()`` across all 5 tasks.

The TT model exposes only ``generate()`` (no ``forward()``) — this matches HF's public API and the
demo's runtime path. A single pytest test exercises all five canonical inference tasks of
SeamlessM4T v2 on the production device config (1×N mesh — TP — + 2CQ + decode-Trace), with the
same default greedy decoding HF uses (``do_sample=False, num_beams=1``):

  1. **T2TT** (text → text)    — long-greedy common-prefix match with HF.
  2. **T2ST** (text → speech)  — voiced-audio plausibility (RMS band + voicing fraction).
  3. **S2TT** (speech → text)  — soft: BOS+lang-code prefix match, non-empty content.
  4. **S2ST** (speech → speech)— soft: both produce voiced audio with RMS in HF's plausible band.
  5. **ASR**  (speech → same-lang text) — soft (same as S2TT; ``repetition_penalty=1.0`` like HF).

This test runs at **demo length**: a long real prompt (~165-token translation) and, for the
speech-input tasks, the chained ~33 s Hindi T2ST waveform → mel seq ≈ 1792 (the encoder's
long-audio regime — chunked matmul, DRAM-resident conformer residual, uncached rel-pos tables).
An earlier short-prompt version (10 tokens, ~1.3 s / mel seq ≈ 256) exercised none of those paths,
which let a speech-encoder rewrite pass while garbaging long audio — hence the long inputs here.

Because everything decodes a long greedy sequence, TT (device-bf16) and HF (CPU-bf16) accumulate
different per-op rounding and **desync after some step** (greedy can't re-converge once one token
differs). So full-sequence / exact-length parity is infeasible: T2TT checks a substantial matching
prefix, the speech-output tasks check valid voiced audio, and the speech→text tasks check the
seed + non-empty content. Tight numerical bounds live in the per-block PCC suite
(``test_speech_encoder.py`` runs the conformer at mel seq = 3000).

For the speech-input tasks (S2TT/S2ST/ASR) the Hindi audio is the HF T2ST output of the same
prompt — same chaining strategy as the demo — so the audio is realistic, reproducible, and
generated once per test invocation.

Real weights only: ``ensure_seamless_m4t_v2_large_weights()`` auto-downloads if missing and
``pytest.skip`` is raised on download failure. No synthetic-weight fallback.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import pytest
import torch
import ttnn
from loguru import logger
from transformers import AutoProcessor, AutoTokenizer

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from models.experimental.seamless_m4t_v2_large.reference.torch_seamless_m4t_v2_model import (
    load_pretrained_seamless_m4t_v2_model,
)
from models.experimental.seamless_m4t_v2_large.scripts.download_weights import ensure_seamless_m4t_v2_large_weights
from models.experimental.seamless_m4t_v2_large.tt.common import (
    hf_aligned_generation_kwargs,
    to_torch_replicated_first_shard,
)
from models.experimental.seamless_m4t_v2_large.tt.mesh_helpers import (
    MESH_DEVICE_PARAMETRIZE_E2E_2CQ_GENERATE,
    mesh_default_device,
)
from models.experimental.seamless_m4t_v2_large.tt.model_preprocessing import create_seamless_m4t_v2_model_parameters
from models.experimental.seamless_m4t_v2_large.tt.tt_seamless_m4t_v2_model import (
    TTSeamlessM4Tv2GenerationOutput,
    TTSeamlessM4Tv2GreedySearchOutput,
    TTSeamlessM4Tv2Model,
)

# --- Test hyperparameters --------------------------------------------------------------------
# LONG real prompt (the demo's story) — drives the long-sequence code paths the old short
# "Hello, my name is SeamlessM4T." / ``max_new_tokens=10`` prompt never reached:
#   * text encoder long prefill + ~165-token greedy decode (T2TT / T2ST), and
#   * via the chained ~33 s Hindi T2ST waveform, the speech encoder's long-audio regime
#     (mel seq ≈ 1792 → chunked 1-D matmul, DRAM-resident conformer residual at seq > 1024,
#     uncached relative-position tables) — the exact path the demo runs.
# The short prompt produced ~1.3 s / mel seq ≈ 256 audio, which exercised none of this. That
# blind spot let a speech-encoder rewrite (efficient relative attention) pass this very test
# while garbaging long audio — so the speech-input tasks below MUST run at demo length.
_PROMPT = """Maya lived in a small coastal town where every morning began with the sound of fishing boats leaving the harbor. She worked at her grandfather's old bookstore, a narrow shop filled with dusty shelves, handwritten notes, and the smell of paper that had aged for decades. Most customers came looking for schoolbooks or travel guides, but Maya loved recommending forgotten stories hidden in the back corners of the store.

One rainy evening, while organizing a stack of returned books, she discovered a small blue journal tucked between two novels. The cover had no title, only a silver compass symbol that shimmered faintly under the light. Curious, she opened it and found detailed sketches of places around the town along with cryptic messages about a hidden lighthouse path that only appeared during storms.

At first, Maya thought someone was playing a prank. But the next night, as heavy clouds gathered over the sea, she noticed something unusual from the bookstore window. A narrow trail of lantern lights stretched along the cliffs where no road existed before. Holding the journal tightly, she followed the glowing path through the rain until she reached an abandoned lighthouse overlooking the crashing waves."""
# T2TT common-prefix floor (TT-device-bf16 vs HF-cpu-bf16 desync after some greedy step; see
# ``_assert_text_prefix_matches``). Measured first divergence at token 32 (a single near-tie flip,
# after which the two sequences re-converge and the ~200-token tails match); 16 leaves a 2× margin
# so a near-tie shifting earlier on a different build doesn't make this flaky.
_MIN_TEXT_PREFIX = 16

# Language pairs (same as the demo's chain):
#   T2TT / T2ST: eng → hin     (Hindi text / Hindi speech)
#   S2TT       : hin speech → eng text
#   S2ST       : hin speech → spa speech
#   ASR        : hin speech → hin text
_TGT_HIN = "hin"
_TGT_ENG = "eng"
_TGT_SPA = "spa"

# Speech-output tolerances (vs HF on the same inputs). Audio length parity is not asserted: every
# speech-output task here decodes a long greedy text/unit sequence that desyncs from HF (bf16), so
# sample counts legitimately differ — we check valid voiced output (RMS band + voicing) instead.
_RMS_RATIO_LO = 0.70  # symmetric in log space with _RMS_RATIO_HI
_RMS_RATIO_HI = 1.43
_VOICING_FRAC_TOL = 0.15


# --- Helpers ---------------------------------------------------------------------------------


def _weights_dir_or_skip() -> str:
    try:
        return ensure_seamless_m4t_v2_large_weights()
    except ImportError as e:
        pytest.skip(str(e))
        raise
    except Exception as e:
        pytest.skip(f"Could not prepare seamless-m4t-v2-large weights: {e}")
        raise


def _torch_ids_to_ttnn(device: ttnn.Device, t: torch.Tensor) -> ttnn.Tensor:
    return ttnn.from_torch(
        t.to(torch.int32).cpu(),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _torch_feats_to_ttnn(device: ttnn.Device, t: torch.Tensor) -> ttnn.Tensor:
    return ttnn.from_torch(
        t.to(torch.bfloat16).cpu().contiguous(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )


def _make_tt_model(device: ttnn.Device, model: Any, cfg: Any, t2u_cfg: Any) -> TTSeamlessM4Tv2Model:
    params = create_seamless_m4t_v2_model_parameters(model, device=device)
    return TTSeamlessM4Tv2Model(
        device,
        params,
        layer_norm_eps=cfg.layer_norm_eps,
        encoder_layers=cfg.encoder_layers,
        encoder_attention_heads=cfg.encoder_attention_heads,
        decoder_layers=cfg.decoder_layers,
        decoder_attention_heads=cfg.decoder_attention_heads,
        hidden_size=cfg.hidden_size,
        feature_projection_input_dim=cfg.feature_projection_input_dim,
        speech_encoder_attention_heads=cfg.speech_encoder_attention_heads,
        speech_encoder_intermediate_size=cfg.speech_encoder_intermediate_size,
        speech_encoder_layers=cfg.speech_encoder_layers,
        speech_encoder_chunk_size=cfg.speech_encoder_chunk_size,
        speech_encoder_left_chunk_num=cfg.speech_encoder_left_chunk_num,
        pad_token_id=cfg.pad_token_id,
        decoder_start_token_id=cfg.decoder_start_token_id,
        vocab_size=cfg.vocab_size,
        adaptor_kernel_size=cfg.adaptor_kernel_size,
        adaptor_stride=cfg.adaptor_stride,
        t2u_eos_token_id=cfg.t2u_eos_token_id,
        t2u_pad_token_id=t2u_cfg.pad_token_id,
        vocoder_offset=cfg.vocoder_offset,
        t2u_layer_norm_eps=t2u_cfg.layer_norm_eps,
        t2u_encoder_layers=t2u_cfg.encoder_layers,
        t2u_encoder_attention_heads=t2u_cfg.encoder_attention_heads,
        t2u_decoder_layers=t2u_cfg.decoder_layers,
        t2u_decoder_attention_heads=t2u_cfg.decoder_attention_heads,
        variance_predictor_embed_dim=t2u_cfg.variance_predictor_embed_dim,
        variance_predictor_hidden_dim=t2u_cfg.variance_predictor_hidden_dim,
        variance_predictor_kernel_size=t2u_cfg.variance_predictor_kernel_size,
        vocoder_config=cfg,
        generation_config=model.generation_config,
        hf_config=cfg,
    )


def _tt_waveform_to_np(wav_tt: ttnn.Tensor, lengths_tt: ttnn.Tensor) -> np.ndarray:
    arr = to_torch_replicated_first_shard(wav_tt).float().squeeze().cpu().numpy()
    if arr.ndim > 1:
        arr = arr.reshape(-1)
    valid_len = int(to_torch_replicated_first_shard(lengths_tt).long().reshape(-1)[0].item())
    if 0 < valid_len <= arr.size:
        arr = arr[:valid_len]
    return arr


def _audio_stats(wav: np.ndarray) -> Tuple[int, float, float]:
    n = wav.size
    rms = float(np.sqrt((wav.astype(np.float64) ** 2).mean())) if n else 0.0
    voicing = float((np.abs(wav) > 0.01).mean()) if n else 0.0
    return n, rms, voicing


def _unpack_hf_speech(out: Any) -> Tuple[np.ndarray, int]:
    """HF ``generate(generate_speech=True)`` returns either a ``(waveform, lengths)`` tuple or a
    ``SeamlessM4Tv2GenerationOutput`` dataclass (when ``return_intermediate_token_ids=True``).
    Accept both; trim the waveform to the valid sample count."""
    if isinstance(out, (tuple, list)) and len(out) == 2:
        wav_t, lens_t = out
    elif hasattr(out, "waveform"):
        wav_t, lens_t = out.waveform, getattr(out, "waveform_lengths", None)
    else:
        raise RuntimeError(f"Unexpected HF speech-generate return type: {type(out)}")
    wav = wav_t.detach().cpu().float().squeeze().numpy()
    if wav.ndim > 1:
        wav = wav.reshape(-1)
    if lens_t is None:
        return wav, wav.size
    valid_len = int(lens_t.detach().cpu().reshape(-1)[0].item())
    return wav[:valid_len], valid_len


def _unpack_tt_text(out: Any) -> list:
    """TT text-only ``generate()`` returns ``TTSeamlessM4Tv2GreedySearchOutput(sequences=...)``."""
    assert isinstance(out, TTSeamlessM4Tv2GreedySearchOutput), type(out)
    ids = to_torch_replicated_first_shard(out.sequences).long().cpu().reshape(-1).tolist()
    ttnn.deallocate(out.sequences)
    return ids


def _unpack_tt_speech(out: Any) -> np.ndarray:
    """TT ``generate(generate_speech=True)`` returns a tuple ``(wav, lengths)`` by default and the
    dataclass when ``return_intermediate_token_ids=True``. Accept either."""
    if isinstance(out, TTSeamlessM4Tv2GenerationOutput):
        wav_tt, lens_tt = out.waveform, out.waveform_lengths
    else:
        assert isinstance(out, (tuple, list)) and len(out) == 2, type(out)
        wav_tt, lens_tt = out
    wav = _tt_waveform_to_np(wav_tt, lens_tt)
    ttnn.deallocate(wav_tt)
    ttnn.deallocate(lens_tt)
    return wav


def _hf_text_ids(out: Any) -> list:
    """HF text-only ``generate`` returns ``GreedySearchEncoderDecoderOutput`` (or similar) with
    ``.sequences``, or a plain ``Tensor``. Accept either."""
    if hasattr(out, "sequences"):
        return out.sequences[0].cpu().tolist()
    return out[0].cpu().tolist()


def _longest_common_prefix(a: list, b: list) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def _assert_text_prefix_matches(hf_ids: list, tt_ids: list, *, task: str, min_prefix: int) -> None:
    """Long-greedy text-output check (T2TT).

    TT decodes on-device (bf16) while HF decodes on CPU (bf16-emulated); over a ~165-token greedy
    decode the two accumulate different per-op rounding and **desync after some step** — greedy
    decoding has no mechanism to re-converge once one token differs, so full-sequence equality is
    infeasible at long length (the short-prompt version of this test could assert strict equality
    only because 10 tokens stayed in lockstep). We instead require a substantial matching prefix
    (the text encoder + early decode must be correct) and that both sides produce a long output.
    """
    lcp = _longest_common_prefix(hf_ids, tt_ids)
    logger.info(f"[{task}] HF tokens ({len(hf_ids)}): {hf_ids}")
    logger.info(f"[{task}] TT tokens ({len(tt_ids)}): {tt_ids}")
    logger.info(f"[{task}] longest common prefix = {lcp} (min required {min_prefix})")
    assert len(hf_ids) >= min_prefix and len(tt_ids) >= min_prefix, (
        f"[{task}] output too short for a long prompt (HF={len(hf_ids)} TT={len(tt_ids)}, "
        f"min {min_prefix}) — encoder/decode likely collapsed"
    )
    assert lcp >= min_prefix, (
        f"[{task}] common prefix {lcp} < {min_prefix} — early decode diverged from HF. "
        f"HF[:24]={hf_ids[:24]} TT[:24]={tt_ids[:24]}"
    )


def _assert_tokens_close_after_speech(hf_ids: list, tt_ids: list, *, task: str, min_prefix: int = 2) -> None:
    """Softer token check for speech-input → text-output tasks (S2TT, ASR).

    The HF speech encoder runs at fp32 while the TT one is bf16; accumulated rounding through
    24 conformer layers + the long attention timeline pushes the first divergent token within a
    handful of greedy steps. The demo already shows this as light paraphrasing in S2TT/ASR
    outputs. Instead of token-for-token equality we require:

    * The seed (BOS + lang-code) must match — this verifies ``tgt_lang`` plumbing and the
      decoder-input-ids construction.
    * TT produced *some* tokens after the seed (not collapsed to a bare seed).
    * TT didn't terminate immediately after the seed (>= 2 content tokens produced).

    Token *count* may legitimately differ — bf16 drift can cause one side to emit EOS one or two
    steps earlier than the other, which is a valid generation outcome.
    """
    logger.info(f"[{task}] HF tokens ({len(hf_ids)}): {hf_ids}")
    logger.info(f"[{task}] TT tokens ({len(tt_ids)}): {tt_ids}")
    assert (
        hf_ids[:min_prefix] == tt_ids[:min_prefix]
    ), f"[{task}] Seed token mismatch (first {min_prefix}): HF {hf_ids[:min_prefix]} vs TT {tt_ids[:min_prefix]}"
    assert (
        len(tt_ids) >= min_prefix + 2 and len(hf_ids) >= min_prefix + 2
    ), f"[{task}] Too few content tokens (HF={len(hf_ids)} TT={len(tt_ids)} min_prefix={min_prefix})"


def _assert_audio_plausible_voiced(hf_wav: np.ndarray, tt_wav: np.ndarray, *, task: str) -> None:
    """Soft audio check (S2ST): both back-ends produce voiced speech of comparable energy/voicing.

    Speech-input speech-output compounds bf16 drift through the speech encoder *and* the
    intermediate-text decode + T2U, so the unit count (and audio length) can diverge substantially
    at small ``max_new_tokens`` budgets — the demo shows ~10 % length drift at realistic prompts.
    The strict ±2 % length bound is meaningless here; instead we require valid voiced output
    (RMS in HF's plausible band, voicing fraction close).
    """
    hf_n, hf_rms, hf_voice = _audio_stats(hf_wav)
    tt_n, tt_rms, tt_voice = _audio_stats(tt_wav)
    logger.info(f"[{task}] HF audio: samples={hf_n} rms={hf_rms:.4f} voicing={hf_voice:.3f}")
    logger.info(f"[{task}] TT audio: samples={tt_n} rms={tt_rms:.4f} voicing={tt_voice:.3f}")

    assert hf_n > 0 and tt_n > 0, f"[{task}] empty audio (HF={hf_n}, TT={tt_n})"
    assert hf_rms > 0.0 and tt_rms > 0.0, f"[{task}] zero-energy audio (HF={hf_rms}, TT={tt_rms})"
    ratio = tt_rms / hf_rms
    assert (
        _RMS_RATIO_LO <= ratio <= _RMS_RATIO_HI
    ), f"[{task}] RMS ratio TT/HF={ratio:.3f} outside [{_RMS_RATIO_LO}, {_RMS_RATIO_HI}]"
    # Slightly wider voicing tolerance than T2ST — different content can shift the voiced fraction.
    assert (
        abs(tt_voice - hf_voice) <= 2 * _VOICING_FRAC_TOL
    ), f"[{task}] voicing frac diff > {2 * _VOICING_FRAC_TOL} (TT={tt_voice:.3f} HF={hf_voice:.3f})"


# --- The single all-5-tasks test -------------------------------------------------------------


@pytest.mark.timeout(5400)
@pytest.mark.parametrize(*MESH_DEVICE_PARAMETRIZE_E2E_2CQ_GENERATE, indirect=["mesh_device", "device_params"])
def test_seamless_m4t_v2_generate_matches_hf_all_tasks(mesh_device, device_params, reset_seeds):
    """``TT generate()`` matches ``HF generate()`` across all 5 tasks on TP + 2CQ + decode-Trace.

    Speech-input tasks chain on the HF T2ST waveform (Hindi) generated earlier in the same test,
    mirroring the demo's flow exactly.
    """
    _ = reset_seeds
    _ = device_params

    weights_dir = _weights_dir_or_skip()
    path = os.fspath(weights_dir)

    torch.manual_seed(0)
    model, cfg = load_pretrained_seamless_m4t_v2_model(weights_dir, dtype=torch.bfloat16)
    t2u_cfg = model.t2u_model.config
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    processor = AutoProcessor.from_pretrained(path, local_files_only=True)
    sr = int(getattr(cfg, "sampling_rate", 16000))

    text_enc = tokenizer([_PROMPT], return_tensors="pt", padding=True)
    text_input_ids = text_enc["input_ids"]
    text_attn = text_enc["attention_mask"]

    common_kwargs = hf_aligned_generation_kwargs(model.generation_config)
    asr_kwargs = common_kwargs
    tt_extra = dict(use_kv_cache=True, use_decode_trace=True, use_2cq=True)

    with mesh_default_device(mesh_device):
        tt_model = _make_tt_model(mesh_device, model, cfg, t2u_cfg)

        # =============================================================================
        # 1. T2TT (eng text → hin text) — long-greedy common-prefix match
        # =============================================================================
        with torch.no_grad():
            hf_out = model.generate(
                input_ids=text_input_ids,
                attention_mask=text_attn,
                generate_speech=False,
                tgt_lang=_TGT_HIN,
                **common_kwargs,
            )
        hf_t2tt_ids = _hf_text_ids(hf_out)
        tt_out = tt_model.generate(
            input_ids=_torch_ids_to_ttnn(mesh_device, text_input_ids),
            attention_mask=_torch_ids_to_ttnn(mesh_device, text_attn),
            generate_speech=False,
            tgt_lang=_TGT_HIN,
            **common_kwargs,
            **tt_extra,
        )
        _assert_text_prefix_matches(hf_t2tt_ids, _unpack_tt_text(tt_out), task="T2TT", min_prefix=_MIN_TEXT_PREFIX)

        # =============================================================================
        # 2. T2ST (eng text → hin speech) — plausible voiced audio (length parity dropped)
        # =============================================================================
        # T2ST shares T2TT's greedy text decode, so its intermediate Hindi text desyncs from HF at
        # the same step → the unit sequence and audio length diverge. The old ±2 % length bound held
        # only at the 10-token prompt; at ~165 tokens it cannot. Check valid voiced output instead.
        with torch.no_grad():
            hf_out = model.generate(
                input_ids=text_input_ids,
                attention_mask=text_attn,
                generate_speech=True,
                tgt_lang=_TGT_HIN,
                **common_kwargs,
            )
        hf_hin_wav, _hf_hin_len = _unpack_hf_speech(hf_out)
        tt_out = tt_model.generate(
            input_ids=_torch_ids_to_ttnn(mesh_device, text_input_ids),
            attention_mask=_torch_ids_to_ttnn(mesh_device, text_attn),
            generate_speech=True,
            tgt_lang=_TGT_HIN,
            speaker_id=0,
            **common_kwargs,
            **tt_extra,
        )
        _assert_audio_plausible_voiced(hf_hin_wav, _unpack_tt_speech(tt_out), task="T2ST")

        # ---- Build speech inputs for tasks 3-5 from the HF Hindi waveform ----------
        # Same chaining pattern as the demo (S2TT/S2ST/ASR consume T2ST's audio).
        audio_inputs = processor(audios=hf_hin_wav, sampling_rate=sr, return_tensors="pt")
        sp_input_features = audio_inputs["input_features"]
        sp_attn = audio_inputs["attention_mask"]

        # =============================================================================
        # 3. S2TT (hin speech → eng text) — soft seed + non-empty content (long audio, mel≈1792)
        # =============================================================================
        with torch.no_grad():
            hf_out = model.generate(
                input_features=sp_input_features.float(),
                attention_mask=sp_attn,
                generate_speech=False,
                tgt_lang=_TGT_ENG,
                **common_kwargs,
            )
        hf_s2tt_ids = _hf_text_ids(hf_out)
        tt_out = tt_model.generate(
            input_features=_torch_feats_to_ttnn(mesh_device, sp_input_features),
            attention_mask=_torch_ids_to_ttnn(mesh_device, sp_attn),
            generate_speech=False,
            tgt_lang=_TGT_ENG,
            **common_kwargs,
            **tt_extra,
        )
        _assert_tokens_close_after_speech(hf_s2tt_ids, _unpack_tt_text(tt_out), task="S2TT")

        # =============================================================================
        # 4. S2ST (hin speech → spa speech) — audio match
        # =============================================================================
        with torch.no_grad():
            hf_out = model.generate(
                input_features=sp_input_features.float(),
                attention_mask=sp_attn,
                generate_speech=True,
                tgt_lang=_TGT_SPA,
                **common_kwargs,
            )
        hf_spa_wav, _ = _unpack_hf_speech(hf_out)
        tt_out = tt_model.generate(
            input_features=_torch_feats_to_ttnn(mesh_device, sp_input_features),
            attention_mask=_torch_ids_to_ttnn(mesh_device, sp_attn),
            generate_speech=True,
            tgt_lang=_TGT_SPA,
            speaker_id=0,
            **common_kwargs,
            **tt_extra,
        )
        _assert_audio_plausible_voiced(hf_spa_wav, _unpack_tt_speech(tt_out), task="S2ST")

        # =============================================================================
        # 5. ASR (hin speech → hin text) — soft seed + non-empty content, rep-penalty=1.0 (long audio)
        # =============================================================================
        with torch.no_grad():
            hf_out = model.generate(
                input_features=sp_input_features.float(),
                attention_mask=sp_attn,
                generate_speech=False,
                tgt_lang=_TGT_HIN,
                **asr_kwargs,
            )
        hf_asr_ids = _hf_text_ids(hf_out)
        tt_out = tt_model.generate(
            input_features=_torch_feats_to_ttnn(mesh_device, sp_input_features),
            attention_mask=_torch_ids_to_ttnn(mesh_device, sp_attn),
            generate_speech=False,
            tgt_lang=_TGT_HIN,
            **asr_kwargs,
            **tt_extra,
        )
        _assert_tokens_close_after_speech(hf_asr_ids, _unpack_tt_text(tt_out), task="ASR")
        tt_model.release_generation_runtime()
