# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""E2E logit PCC helpers for sweep tests and reference generation.

Full-vocabulary logits PCC (``run_e2e_logits_pcc_loop``): Devstral-style ISL sweep; HF-greedy
decode feeds both HF and TT after each logits comparison.

Also provides speech E2E input builders (reference scripts) and ``_hf_teacher_forced_decode_reference``
(teacher-forced WER gate).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, List, Tuple

import pytest
import torch
import ttnn
from loguru import logger
from transformers import AutoProcessor, SeamlessM4Tv2Model
from transformers.cache_utils import DynamicCache

from models.common.utility_functions import comp_allclose, comp_pcc
from models.experimental.seamless_m4t_v2_large.tests.pcc.decoder_pcc_fixtures import (
    TextDecoderPccInputs,
    _hf_speech_encoder_hidden_and_mask,
    _truncate_encoder_timeline,
    decoder_seed_ids,
)
from models.experimental.seamless_m4t_v2_large.tests.pcc.e2e_tt_model_helpers import (
    make_tt_model,
    torch_feats_to_ttnn,
    torch_ids_to_ttnn,
)
from models.experimental.seamless_m4t_v2_large.tt.mesh_helpers import mesh_default_device, mesh_num_devices
from models.experimental.seamless_m4t_v2_large.tt.tt_seamless_m4t_v2_model import (
    _ttnn_ids_from_list,
)
from models.experimental.seamless_m4t_v2_large.tt.tt_text_decoder import init_text_decoder_kv_cache

if TYPE_CHECKING:
    from models.experimental.seamless_m4t_v2_large.tests.pcc.e2e_token_matching_helpers import (
        SpeechTokenAccuracyReference,
        T2ttTokenAccuracyReference,
    )

MAX_ENC_SEQ = 256
LOGIT_PCC_DECODE_STEPS = 10
LOGIT_PCC_REQUIRED = 0.90


def resolve_preamble_wav_for_tests() -> Path:
    """Demo preamble WAV for S2ST token-matching reference (download + validate if missing)."""
    from models.experimental.seamless_m4t_v2_large.demo.demo import PREAMBLE_WAV, ensure_demo_audio

    return ensure_demo_audio(dest=PREAMBLE_WAV.expanduser().resolve())


@dataclass(frozen=True)
class SpeechE2eInputs:
    """Speech-path E2E inputs: mel features plus decoder PCC case."""

    case: TextDecoderPccInputs
    input_features: torch.Tensor
    mel_attention_mask: torch.Tensor


def make_speech_e2e_inputs(
    model: SeamlessM4Tv2Model,
    processor: AutoProcessor,
    *,
    tgt_lang: str,
    enc_seq_len: int = MAX_ENC_SEQ,
    wav_seconds: float = 1.0,
    seed: int = 42,
    wav_path: Path | None = None,
) -> SpeechE2eInputs:
    if wav_path is not None:
        from models.experimental.seamless_m4t_v2_large.demo.demo import _load_mono_wav_resampled

        wav, _ = _load_mono_wav_resampled(wav_path.expanduser().resolve(), 16_000)
    else:
        seconds = max(wav_seconds, 1.0)
        torch.manual_seed(seed)
        n_samples = max(1, int(16_000 * seconds))
        wav = (torch.randn(n_samples, dtype=torch.float32) * 0.01).numpy().reshape(-1)

    audio = processor(audio=wav, sampling_rate=16_000, return_tensors="pt")
    input_features = audio["input_features"].to(dtype=next(model.parameters()).dtype)
    mel_mask = audio["attention_mask"]
    encoder_hidden, enc_attn = _hf_speech_encoder_hidden_and_mask(model, input_features, mel_mask)

    if wav_path is None:
        got = int(encoder_hidden.shape[1])
        seconds = max(wav_seconds, 1.0)
        while got < enc_seq_len:
            seconds *= 1.5
            torch.manual_seed(seed)
            n_samples = max(1, int(16_000 * seconds))
            wav = (torch.randn(n_samples, dtype=torch.float32) * 0.01).numpy().reshape(-1)
            audio = processor(audio=wav, sampling_rate=16_000, return_tensors="pt")
            input_features = audio["input_features"].to(dtype=next(model.parameters()).dtype)
            mel_mask = audio["attention_mask"]
            encoder_hidden, enc_attn = _hf_speech_encoder_hidden_and_mask(model, input_features, mel_mask)
            got = int(encoder_hidden.shape[1])
        enc_target = enc_seq_len
    else:
        enc_target = min(enc_seq_len, int(encoder_hidden.shape[1]))

    encoder_hidden, enc_attn = _truncate_encoder_timeline(encoder_hidden, enc_attn, enc_target)

    dec_ids = decoder_seed_ids(model, tgt_lang)
    dec_mask = torch.ones_like(dec_ids)
    case = TextDecoderPccInputs(
        input_ids=dec_ids,
        attention_mask=dec_mask,
        encoder_hidden_states=encoder_hidden,
        encoder_attention_mask=enc_attn,
    )
    return SpeechE2eInputs(case=case, input_features=input_features, mel_attention_mask=mel_mask)


def tt_encode_speech_via_model(
    mesh_device: ttnn.Device,
    tt_model,
    input_features: torch.Tensor,
    mel_attention_mask: torch.Tensor,
) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
    """Production speech encoder path: ``prewarm_speech_encoder`` + ``_encode_speech``."""
    mel_frames = int(mel_attention_mask.sum().item())
    tt_model.prewarm_speech_encoder([mel_frames])
    feats_tt = torch_feats_to_ttnn(mesh_device, input_features)
    mask_tt = torch_ids_to_ttnn(mesh_device, mel_attention_mask)
    enc_tt, enc_mask_tt, attn_owned = tt_model._encode_speech(feats_tt, mask_tt)
    ttnn.deallocate(feats_tt)
    if attn_owned and enc_mask_tt is not mask_tt:
        ttnn.deallocate(mask_tt)
    if mesh_num_devices(mesh_device) == 1:
        tt_model._clear_decode_and_t2u_programs(preserve_vocoder=True)
    return enc_tt, enc_mask_tt


def _hf_teacher_forced_decode_reference(
    hf_model: SeamlessM4Tv2Model,
    decoder,
    case: TextDecoderPccInputs,
    aligned,
    decode_steps: int,
) -> Tuple[torch.Tensor, List[torch.Tensor], List[int]]:
    """HF incremental decode; returns prefill hidden, per-step decode hiddens, teacher tokens."""
    batch = int(aligned.input_ids.shape[0])
    logical_dec = aligned.logical_dec_seq
    p0 = next(decoder.parameters())
    enc_hidden = aligned.encoder_hidden_states.to(device=p0.device, dtype=p0.dtype)
    enc_mask = aligned.encoder_attention_mask.to(device=p0.device)
    seed_ids = case.input_ids.to(device=p0.device)
    seed_mask = case.attention_mask.to(device=p0.device)
    lm_head = hf_model.lm_head

    def _greedy_last(hidden: torch.Tensor) -> int:
        with torch.no_grad():
            logits = lm_head(hidden[:, -1, :].to(p0.dtype))
        return int(logits[0].argmax().item())

    with torch.no_grad():
        prefill_out = decoder(
            input_ids=seed_ids,
            attention_mask=seed_mask,
            encoder_hidden_states=enc_hidden,
            encoder_attention_mask=enc_mask,
            use_cache=True,
            return_dict=True,
        )
    ref_prefill = prefill_out.last_hidden_state[:, :logical_dec, :].to(torch.bfloat16).contiguous()
    past = prefill_out.past_key_values

    decode_tokens: list[int] = []
    ref_decode: list[torch.Tensor] = []
    tok = _greedy_last(prefill_out.last_hidden_state)
    for _ in range(decode_steps):
        decode_tokens.append(tok)
        step_ids = torch.full((batch, 1), tok, dtype=torch.long, device=p0.device)
        with torch.no_grad():
            step_out = decoder(
                input_ids=step_ids,
                encoder_hidden_states=enc_hidden,
                encoder_attention_mask=enc_mask,
                past_key_values=past,
                use_cache=True,
                return_dict=True,
            )
        ref_decode.append(step_out.last_hidden_state.to(torch.bfloat16).contiguous())
        past = step_out.past_key_values
        tok = _greedy_last(step_out.last_hidden_state)

    return ref_prefill, ref_decode, decode_tokens


def _logits_row_for_pcc(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim == 1:
        return logits.unsqueeze(0)
    if logits.ndim == 2:
        return logits[:1]
    if logits.ndim == 3:
        return logits[:, -1, :]
    raise ValueError(f"Expected 1D–3D logits, got shape {logits.shape}")


def assert_logits_pcc(
    ref_logits: torch.Tensor,
    tt_logits: torch.Tensor,
    *,
    label: str,
    pcc_required: float = LOGIT_PCC_REQUIRED,
) -> None:
    ref_row = _logits_row_for_pcc(ref_logits.float().cpu())
    tt_row = _logits_row_for_pcc(tt_logits.float().cpu())
    passing, msg = comp_pcc(ref_row, tt_row, pcc_required)
    logger.info(comp_allclose(ref_row, tt_row))
    if passing:
        logger.info(f"{label}: PASS — {msg}")
    else:
        logger.warning(f"{label}: FAIL — {msg}")
    assert passing, f"{label}: PCC below {pcc_required}: {msg}"


def _hf_greedy_token_id(logits: torch.Tensor) -> int:
    row = _logits_row_for_pcc(logits.float().cpu())
    return int(row.argmax(dim=-1).item())


@torch.no_grad()
def _hf_prefill_last_logits(
    hf_model,
    *,
    encoder_hidden: torch.Tensor,
    enc_mask: torch.Tensor,
    seed_ids: torch.Tensor,
) -> Tuple[torch.Tensor, DynamicCache]:
    decoder = hf_model.text_decoder
    lm_head = hf_model.lm_head
    p0 = next(decoder.parameters())
    enc_hidden = encoder_hidden.to(device=p0.device, dtype=p0.dtype)
    enc_mask_dev = enc_mask.to(device=p0.device, dtype=torch.long)
    seed_ids_dev = seed_ids.to(device=p0.device)
    seed_mask = torch.ones_like(seed_ids_dev)

    prefill = decoder(
        input_ids=seed_ids_dev,
        attention_mask=seed_mask,
        encoder_hidden_states=enc_hidden,
        encoder_attention_mask=enc_mask_dev,
        use_cache=True,
        return_dict=True,
    )
    prefill_logits = lm_head(prefill.last_hidden_state[:, -1:, :]).float().cpu()
    return prefill_logits, prefill.past_key_values


@torch.no_grad()
def _hf_decode_step_logits(
    hf_model,
    token_id: int,
    *,
    encoder_hidden: torch.Tensor,
    enc_mask: torch.Tensor,
    cache: DynamicCache,
) -> Tuple[torch.Tensor, DynamicCache]:
    decoder = hf_model.text_decoder
    lm_head = hf_model.lm_head
    p0 = next(decoder.parameters())
    enc_hidden = encoder_hidden.to(device=p0.device, dtype=p0.dtype)
    enc_mask_dev = enc_mask.to(device=p0.device, dtype=torch.long)
    step_ids = torch.full((1, 1), int(token_id), dtype=torch.long, device=p0.device)
    step_out = decoder(
        input_ids=step_ids,
        encoder_hidden_states=enc_hidden,
        encoder_attention_mask=enc_mask_dev,
        past_key_values=cache,
        use_cache=True,
        return_dict=True,
    )
    return lm_head(step_out.last_hidden_state).float().cpu(), step_out.past_key_values


@torch.no_grad()
def _tt_prefill_last_logits(
    tt_model,
    mesh_device: ttnn.Device,
    hf_model,
    *,
    enc_tt: ttnn.Tensor,
    enc_mask_tt: ttnn.Tensor,
    seed_ids: torch.Tensor,
    kv_cache,
    cross_attn_cache,
) -> torch.Tensor:
    seed_len = int(seed_ids.shape[1])
    batch = 1
    hidden_size = int(hf_model.config.hidden_size)

    seed_tt = _ttnn_ids_from_list([seed_ids[0].tolist()], mesh_device)
    warm_out = tt_model._prefill_text_decoder_kv_cache(
        seed_tt,
        enc_tt,
        enc_mask_tt,
        kv_cache,
        cross_attn_cache,
    )
    ttnn.deallocate(seed_tt)

    local_pos = seed_len - 1
    last_h = ttnn.slice(warm_out, [0, local_pos, 0], [batch, local_pos + 1, hidden_size], (1, 1, 1))
    prefill_logits_tt = tt_model._lm_head(last_h)
    ttnn.deallocate(last_h)
    tt_prefill = tt_model._logits_row_to_host(prefill_logits_tt, dec_len=1, sharded=False)
    ttnn.deallocate(prefill_logits_tt)
    if warm_out is not None:
        ttnn.deallocate(warm_out)
    return tt_prefill


@torch.no_grad()
def _tt_decode_step_logits(
    tt_model,
    *,
    token_id: int,
    position: int,
    enc_tt: ttnn.Tensor,
    enc_mask_tt: ttnn.Tensor,
    kv_cache,
    cross_attn_cache,
    cross_attn_cache_valid: bool,
) -> torch.Tensor:
    logits_tt = tt_model._decode_token_with_kv_cache(
        int(token_id),
        position,
        enc_tt,
        enc_mask_tt,
        kv_cache,
        cross_attn_cache,
        cross_attn_cache_valid=cross_attn_cache_valid,
        batch_size=1,
    )
    tt_logits = tt_model._logits_row_to_host(logits_tt, dec_len=1, sharded=False)
    ttnn.deallocate(logits_tt)
    return tt_logits


@torch.no_grad()
def run_e2e_logits_pcc_loop(
    tt_model,
    mesh_device: ttnn.Device,
    hf_model,
    *,
    encoder_hidden: torch.Tensor,
    enc_mask: torch.Tensor,
    seed_ids: torch.Tensor,
    enc_tt: ttnn.Tensor,
    enc_mask_tt: ttnn.Tensor,
    decode_steps: int,
    log_label: str,
    pcc_required: float = LOGIT_PCC_REQUIRED,
) -> None:
    """Compare HF vs TT logits: seed prefill last position + ``decode_steps`` HF-greedy decode steps."""
    cfg = hf_model.config
    seed_len = int(seed_ids.shape[1])
    max_seq_len = max(64, seed_len + decode_steps + 8)
    padded_enc = int(enc_tt.shape[1])

    kv_cache, cross_attn_cache = init_text_decoder_kv_cache(
        mesh_device,
        num_hidden_layers=cfg.decoder_layers,
        num_attention_heads=cfg.decoder_attention_heads,
        hidden_size=cfg.hidden_size,
        max_batch_size=1,
        max_seq_len=max_seq_len,
        encoder_seq_len=padded_enc,
        tp=tt_model._tp,
    )

    ref_prefill, hf_cache = _hf_prefill_last_logits(
        hf_model,
        encoder_hidden=encoder_hidden,
        enc_mask=enc_mask,
        seed_ids=seed_ids,
    )
    tt_prefill = _tt_prefill_last_logits(
        tt_model,
        mesh_device,
        hf_model,
        enc_tt=enc_tt,
        enc_mask_tt=enc_mask_tt,
        seed_ids=seed_ids,
        kv_cache=kv_cache,
        cross_attn_cache=cross_attn_cache,
    )
    assert_logits_pcc(
        ref_prefill,
        tt_prefill,
        label=f"{log_label} prefill last logits",
        pcc_required=pcc_required,
    )

    next_tok = _hf_greedy_token_id(ref_prefill)
    current_pos = seed_len
    cross_valid = True

    for step in range(decode_steps):
        ref_logits, hf_cache = _hf_decode_step_logits(
            hf_model,
            next_tok,
            encoder_hidden=encoder_hidden,
            enc_mask=enc_mask,
            cache=hf_cache,
        )
        tt_logits = _tt_decode_step_logits(
            tt_model,
            token_id=next_tok,
            position=current_pos,
            enc_tt=enc_tt,
            enc_mask_tt=enc_mask_tt,
            kv_cache=kv_cache,
            cross_attn_cache=cross_attn_cache,
            cross_attn_cache_valid=cross_valid,
        )
        assert_logits_pcc(
            ref_logits,
            tt_logits,
            label=f"{log_label} decode step={step} pos={current_pos} tok={next_tok}",
            pcc_required=pcc_required,
        )
        next_tok = _hf_greedy_token_id(ref_logits)
        current_pos += 1

    ttnn.deallocate(enc_tt)
    ttnn.deallocate(enc_mask_tt)
    for layer in kv_cache:
        ttnn.deallocate(layer[0])
        ttnn.deallocate(layer[1])
    for layer in cross_attn_cache:
        ttnn.deallocate(layer[0])
        ttnn.deallocate(layer[1])
    ttnn.synchronize_device(mesh_device)


def effective_logit_pcc_decode_steps(ref_teacher_steps: int, *, task: str, seq_len: int) -> int:
    if ref_teacher_steps <= 0:
        pytest.skip(
            f"{task.upper()} logit-PCC sweep len={seq_len}: HF reference has no decode steps "
            f"(EOS on prefill or empty mel input)"
        )
    effective = min(LOGIT_PCC_DECODE_STEPS, ref_teacher_steps)
    if effective < LOGIT_PCC_DECODE_STEPS:
        logger.info(
            f"{task.upper()} logit-PCC sweep len={seq_len}: capping decode steps "
            f"{effective}/{LOGIT_PCC_DECODE_STEPS} (HF greedy EOS after {ref_teacher_steps} steps)"
        )
    return effective


def run_t2tt_e2e_logits_pcc_from_ref(
    mesh_device: ttnn.Device,
    hf_model,
    ref: T2ttTokenAccuracyReference,
    *,
    seq_len: int,
    decode_steps: int = LOGIT_PCC_DECODE_STEPS,
    pcc_required: float = LOGIT_PCC_REQUIRED,
) -> None:
    from models.experimental.seamless_m4t_v2_large.tests.pcc.decoder_pcc_fixtures import _hf_text_encoder_hidden

    cfg = hf_model.config
    t2u_cfg = hf_model.t2u_model.config
    log_label = f"T2TT-len{seq_len}"

    encoder_hidden = _hf_text_encoder_hidden(hf_model, ref.src_ids, ref.src_mask)
    enc_mask = ref.src_mask.to(device=encoder_hidden.device, dtype=torch.long)

    with mesh_default_device(mesh_device):
        tt_model = make_tt_model(mesh_device, hf_model, cfg, t2u_cfg)
        src_ids_tt = torch_ids_to_ttnn(mesh_device, ref.src_ids.to(torch.int32))
        src_mask_tt = torch_ids_to_ttnn(mesh_device, ref.src_mask.to(torch.int32))
        enc_tt, enc_mask_tt, attn_owned = tt_model._encode_text(src_ids_tt, src_mask_tt)
        ttnn.deallocate(src_ids_tt)
        if attn_owned:
            ttnn.deallocate(src_mask_tt)

        logger.info(
            f"SeamlessM4Tv2 E2E logits PCC ({log_label}): decode_steps={decode_steps}, "
            f"pcc>={pcc_required}, teacher=HF_greedy"
        )
        run_e2e_logits_pcc_loop(
            tt_model,
            mesh_device,
            hf_model,
            encoder_hidden=encoder_hidden,
            enc_mask=enc_mask,
            seed_ids=ref.seed_ids,
            enc_tt=enc_tt,
            enc_mask_tt=enc_mask_tt,
            decode_steps=decode_steps,
            log_label=log_label,
            pcc_required=pcc_required,
        )


def run_speech_e2e_logits_pcc_from_ref(
    mesh_device: ttnn.Device,
    hf_model,
    ref: SpeechTokenAccuracyReference,
    *,
    task: str,
    seq_len: int,
    decode_steps: int = LOGIT_PCC_DECODE_STEPS,
    pcc_required: float = LOGIT_PCC_REQUIRED,
) -> None:
    cfg = hf_model.config
    t2u_cfg = hf_model.t2u_model.config
    log_label = f"{task.upper()}-len{seq_len}"

    encoder_hidden, enc_mask = _hf_speech_encoder_hidden_and_mask(
        hf_model,
        ref.input_features,
        ref.mel_attention_mask,
    )

    with mesh_default_device(mesh_device):
        tt_model = make_tt_model(mesh_device, hf_model, cfg, t2u_cfg)
        try:
            enc_tt, enc_mask_tt = tt_encode_speech_via_model(
                mesh_device,
                tt_model,
                ref.input_features,
                ref.mel_attention_mask,
            )

            logger.info(
                f"SeamlessM4Tv2 E2E logits PCC ({log_label}): decode_steps={decode_steps}, "
                f"pcc>={pcc_required}, teacher=HF_greedy"
            )
            run_e2e_logits_pcc_loop(
                tt_model,
                mesh_device,
                hf_model,
                encoder_hidden=encoder_hidden,
                enc_mask=enc_mask,
                seed_ids=ref.seed_ids,
                enc_tt=enc_tt,
                enc_mask_tt=enc_mask_tt,
                decode_steps=decode_steps,
                log_label=log_label,
                pcc_required=pcc_required,
            )
        finally:
            tt_model.release_generation_runtime()
