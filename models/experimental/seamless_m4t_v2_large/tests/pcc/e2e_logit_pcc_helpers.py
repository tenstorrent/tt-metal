# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""E2E Full-Model Logit PCC helpers (``models/tt_transformers/tests/test_model.py`` pattern).

Named "Logit PCC" per tt_transformers documentation; compares post-final-norm **hidden states
before** ``lm_head``, not logits or predicted tokens. HF greedy token at each decode step is fed
to both HF and TT decoders (teacher forcing) so small errors do not compound.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import pytest
import torch
import ttnn
from loguru import logger
from transformers import AutoProcessor, SeamlessM4Tv2Model

from tests.ttnn.utils_for_testing import check_with_pcc

from models.experimental.seamless_m4t_v2_large.scripts.download_weights import ensure_seamless_m4t_v2_large_weights
from models.experimental.seamless_m4t_v2_large.tests.pcc.decoder_pcc_fixtures import (
    TextDecoderPccInputs,
    align_case_for_tt_prefill,
    decoder_seed_ids,
    tokenize_source_text,
    _hf_speech_encoder_hidden_and_mask,
    _hf_text_encoder_hidden,
    _truncate_encoder_timeline,
    DEFAULT_SRC_LANG,
)
from models.experimental.seamless_m4t_v2_large.tt.common import (
    build_causal_with_padding_4d,
    build_cross_attn_mask_4d,
    build_encoder_self_mask_4d,
    tile_align,
    to_torch_replicated_first_shard,
    tt_position_ids,
    tt_position_ids_decode_step,
)
from models.experimental.seamless_m4t_v2_large.tt.mesh_helpers import (
    from_torch_bfloat16_tile,
    from_torch_uint32_rm,
    get_tp,
)
from models.experimental.seamless_m4t_v2_large.tt.model_preprocessing import (
    create_speech_encoder_parameters,
    create_text_decoder_parameters,
    create_text_encoder_parameters,
)
from models.experimental.seamless_m4t_v2_large.tt.tt_seamless_m4t_v2_model import (
    _read_int_row,
    _subsampled_lens_dev,
    _tt_speech_enc_attn,
)
from models.experimental.seamless_m4t_v2_large.tt.tt_speech_encoder import TTSeamlessM4Tv2SpeechEncoder
from models.experimental.seamless_m4t_v2_large.tt.tt_text_decoder import (
    TTSeamlessM4Tv2Decoder,
    init_text_decoder_kv_cache,
    warm_text_decoder_kv_cache_prefill,
)
from models.experimental.seamless_m4t_v2_large.tt.tt_text_encoder import TTSeamlessM4Tv2Encoder

E2E_PROMPT = "This is a test"
MAX_ENC_SEQ = 256
PCC_ENCODER_TEXT = 0.99
PCC_ENCODER_SPEECH = 0.97  # speech encoder + adaptor @ enc_seq≈256: ~0.978 on BH 1×4 (see module test @ 4096 mel)
PCC_PREFILL = 0.99
PCC_PREFILL_SPEECH = 0.97  # decoder prefill with speech encoder hidden: ~0.975 on BH 1×4 (S2TT/ASR)
PCC_PREFILL_S2ST = 0.93  # S2ST (spa seed) @ enc_seq≈256: 0.938–0.945 on BH 1×4 (run variance)
PCC_DECODE_QUICK = 0.97
PCC_DECODE_FULL = 0.99
PCC_DECODE_SPEECH_QUICK = 0.96  # E2E decode step 0: ~0.968 on BH 1×4 with speech encoder hidden
PCC_DECODE_SPEECH_FULL = 0.96
PCC_DECODE_S2ST = 0.93  # S2ST (spa): decode step 0 ~0.945 on BH 1×4; use same band as prefill
_SPEECH_ENC_SEQ_BUCKET = 256


def resolve_preamble_wav_for_tests() -> Path:
    """Demo preamble WAV used for S2ST token-accuracy reference (download if missing)."""
    from models.experimental.seamless_m4t_v2_large.demo.demo import PREAMBLE_WAV, ensure_demo_audio

    path = PREAMBLE_WAV.expanduser().resolve()
    if path.is_file() and path.stat().st_size > 0:
        return path
    return ensure_demo_audio(dest=path)


@dataclass(frozen=True)
class SpeechE2eInputs:
    """Speech-path E2E inputs: mel features plus decoder PCC case."""

    case: TextDecoderPccInputs
    input_features: torch.Tensor
    mel_attention_mask: torch.Tensor


def weights_dir_or_skip() -> str:
    try:
        return ensure_seamless_m4t_v2_large_weights()
    except ImportError as e:
        pytest.skip(str(e))
    except Exception as e:
        pytest.skip(f"Could not prepare seamless-m4t-v2-large weights: {e}")


def _create_position_ids_from_input_ids(input_ids: torch.Tensor, padding_idx: int) -> torch.Tensor:
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = torch.cumsum(mask, dim=1).type_as(mask) * mask
    return incremental_indices.long() + padding_idx


def _speech_mask_uint_to_bf16_tile(mask_2d: ttnn.Tensor) -> ttnn.Tensor:
    mask_tile_u = ttnn.to_layout(mask_2d, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    mask_tile_i = ttnn.typecast(mask_tile_u, ttnn.int32)
    ttnn.deallocate(mask_tile_u)
    mask_bf16_tile = ttnn.typecast(mask_tile_i, ttnn.bfloat16)
    ttnn.deallocate(mask_tile_i)
    return mask_bf16_tile


def _trim_pad_speech_enc(
    mesh_device: ttnn.Device,
    enc_raw: ttnn.Tensor,
    sub_lens_tt: ttnn.Tensor,
    batch: int,
    hidden_size: int,
) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
    """Match ``TTSeamlessM4Tv2Model._speech_encoder_trim_pad_and_cross_attn``."""
    sub_lens = _read_int_row(sub_lens_tt)[:batch]
    physical_len = int(enc_raw.shape[1])
    logical_len = max(1, min(min(sub_lens), physical_len))
    padded_len = tile_align(logical_len)

    enc_out = enc_raw
    if physical_len > logical_len:
        sliced = ttnn.slice(enc_out, [0, 0, 0], [batch, logical_len, hidden_size], (1, 1, 1))
        ttnn.deallocate(enc_out)
        enc_out = sliced
    if logical_len < padded_len:
        pad_tail = ttnn.full(
            [batch, padded_len - logical_len, hidden_size],
            0.0,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        cat = ttnn.concat([enc_out, pad_tail], dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(enc_out)
        ttnn.deallocate(pad_tail)
        enc_out = cat

    enc_attn_tt = _tt_speech_enc_attn(sub_lens_tt, padded_len, mesh_device)
    ttnn.deallocate(sub_lens_tt)
    return enc_out, enc_attn_tt


def _t2tt_source_ids(processor: AutoProcessor) -> Tuple[torch.Tensor, torch.Tensor]:
    src_ids, src_mask = tokenize_source_text(processor, E2E_PROMPT, DEFAULT_SRC_LANG)
    if int(src_ids.shape[1]) > MAX_ENC_SEQ:
        src_ids = src_ids[:, :MAX_ENC_SEQ]
        src_mask = src_mask[:, :MAX_ENC_SEQ]
    return src_ids, src_mask


def make_t2tt_e2e_case(
    model: SeamlessM4Tv2Model,
    processor: AutoProcessor,
    *,
    tgt_lang: str,
) -> TextDecoderPccInputs:
    src_ids, src_mask = _t2tt_source_ids(processor)
    encoder_hidden = _hf_text_encoder_hidden(model, src_ids, src_mask)
    dec_ids = decoder_seed_ids(model, tgt_lang)
    dec_mask = torch.ones_like(dec_ids)
    return TextDecoderPccInputs(
        input_ids=dec_ids,
        attention_mask=dec_mask,
        encoder_hidden_states=encoder_hidden,
        encoder_attention_mask=src_mask.to(encoder_hidden.device),
    )


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

    audio = processor(audios=wav, sampling_rate=16_000, return_tensors="pt")
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
            audio = processor(audios=wav, sampling_rate=16_000, return_tensors="pt")
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


def tt_encode_text(
    mesh_device: ttnn.Device,
    text_encoder_module,
    cfg,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
    """Live TT text encoder (tile-padded), returns ``(hidden, enc_attn_2d)``."""
    batch = int(input_ids.shape[0])
    seq = int(input_ids.shape[1])
    padded_seq = tile_align(seq)
    pad_id = int(cfg.pad_token_id)

    if padded_seq > seq:
        tail = torch.full((batch, padded_seq - seq), pad_id, dtype=torch.int64)
        ids_padded = torch.cat([input_ids, tail], dim=1)
        mask_pad = torch.zeros((batch, padded_seq - seq), dtype=torch.long)
        mask_padded = torch.cat([attention_mask, mask_pad], dim=1)
    else:
        ids_padded = input_ids
        mask_padded = attention_mask

    params = create_text_encoder_parameters(text_encoder_module, device=mesh_device)
    tt_enc = TTSeamlessM4Tv2Encoder(
        mesh_device,
        params,
        layer_norm_eps=cfg.layer_norm_eps,
        num_hidden_layers=cfg.encoder_layers,
        num_attention_heads=cfg.encoder_attention_heads,
        hidden_size=cfg.hidden_size,
    )

    position_ids = _create_position_ids_from_input_ids(ids_padded, pad_id)
    input_ids_tt = from_torch_uint32_rm(mesh_device, ids_padded)
    position_ids_tt = from_torch_uint32_rm(mesh_device, position_ids)
    enc_mask_2d_tt = from_torch_uint32_rm(mesh_device, mask_padded)
    enc_mask_4d = build_encoder_self_mask_4d(enc_mask_2d_tt, device=mesh_device)

    enc_out = tt_enc.forward(input_ids_tt, position_ids_tt, enc_mask_4d)
    ttnn.deallocate(input_ids_tt)
    ttnn.deallocate(position_ids_tt)
    ttnn.deallocate(enc_mask_4d)
    return enc_out, enc_mask_2d_tt


def tt_encode_speech(
    mesh_device: ttnn.Device,
    speech_encoder_module,
    cfg,
    input_features: torch.Tensor,
    attention_mask: torch.Tensor,
) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
    """Live TT speech encoder + adaptor trim (production-shaped), returns ``(hidden, enc_attn_2d)``."""
    batch = int(input_features.shape[0])
    seq_in = int(input_features.shape[1])
    bucketed = ((seq_in + _SPEECH_ENC_SEQ_BUCKET - 1) // _SPEECH_ENC_SEQ_BUCKET) * _SPEECH_ENC_SEQ_BUCKET

    feats = input_features.contiguous()
    mask = attention_mask.contiguous()
    if bucketed != seq_in:
        feat_pad = torch.zeros(batch, bucketed - seq_in, feats.shape[2], dtype=feats.dtype)
        feats = torch.cat([feats, feat_pad], dim=1)
        mask_pad = torch.zeros(batch, bucketed - seq_in, dtype=mask.dtype)
        mask = torch.cat([mask, mask_pad], dim=1)

    params = create_speech_encoder_parameters(speech_encoder_module, device=mesh_device)
    token_rows = batch * bucketed
    tt_speech = TTSeamlessM4Tv2SpeechEncoder(
        mesh_device,
        params,
        hidden_size=cfg.hidden_size,
        feature_projection_input_dim=cfg.feature_projection_input_dim,
        speech_encoder_attention_heads=cfg.speech_encoder_attention_heads,
        speech_encoder_intermediate_size=cfg.speech_encoder_intermediate_size,
        speech_encoder_layers=cfg.speech_encoder_layers,
        layer_norm_eps=cfg.layer_norm_eps,
        speech_encoder_chunk_size=cfg.speech_encoder_chunk_size,
        speech_encoder_left_chunk_num=cfg.speech_encoder_left_chunk_num,
        matmul_token_rows=token_rows,
    )

    feats_tt = from_torch_bfloat16_tile(mesh_device, feats, memory_config=ttnn.L1_MEMORY_CONFIG)
    mask_2d_tt = from_torch_uint32_rm(mesh_device, mask)
    mask_bf16 = _speech_mask_uint_to_bf16_tile(mask_2d_tt)
    enc_raw = tt_speech.forward(feats_tt, conv_attention_mask_1d=mask_bf16)
    ttnn.deallocate(feats_tt)
    ttnn.deallocate(mask_bf16)

    sub_lens_tt = _subsampled_lens_dev(mask_2d_tt, int(cfg.adaptor_kernel_size), int(cfg.adaptor_stride))
    return _trim_pad_speech_enc(mesh_device, enc_raw, sub_lens_tt, batch, int(cfg.hidden_size))


def _align_tt_encoder_to_case(
    mesh_device: ttnn.Device,
    enc_tt: ttnn.Tensor,
    enc_mask_tt: ttnn.Tensor,
    case: TextDecoderPccInputs,
    *,
    pad_id: int,
    hidden_size: int,
) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
    """Trim/pad TT encoder outputs to the HF case timeline (speech may exceed ``MAX_ENC_SEQ``)."""
    aligned = align_case_for_tt_prefill(case, pad_id)
    padded_enc = aligned.padded_enc_seq
    batch = int(enc_tt.shape[0])
    physical = int(enc_tt.shape[1])

    if physical != padded_enc:
        if physical > padded_enc:
            trimmed = ttnn.slice(enc_tt, [0, 0, 0], [batch, padded_enc, hidden_size], (1, 1, 1))
            ttnn.deallocate(enc_tt)
            enc_tt = trimmed
        else:
            padded = ttnn.pad(
                enc_tt,
                [(0, 0), (0, padded_enc - physical), (0, 0)],
                value=0.0,
            )
            ttnn.deallocate(enc_tt)
            enc_tt = padded

    ttnn.deallocate(enc_mask_tt)
    enc_mask_tt = from_torch_uint32_rm(mesh_device, aligned.encoder_attention_mask)
    return enc_tt, enc_mask_tt


def _assert_encoder_pcc(
    enc_tt: ttnn.Tensor,
    ref_enc: torch.Tensor,
    logical_enc: int,
    *,
    log_label: str,
    threshold: float,
) -> None:
    batch = int(ref_enc.shape[0])
    hidden_size = int(ref_enc.shape[2])
    padded_enc = int(enc_tt.shape[1])
    tt_cpu = (
        to_torch_replicated_first_shard(enc_tt)
        .to(torch.bfloat16)
        .reshape(batch, padded_enc, hidden_size)[:, :logical_enc, :]
        .contiguous()
    )
    ref = ref_enc[:, :logical_enc, :].to(torch.bfloat16).contiguous()
    ok, msg = check_with_pcc(ref, tt_cpu, pcc=threshold)
    logger.info(
        f"SeamlessM4Tv2 E2E logit PCC encoder ({log_label}) enc_seq={logical_enc}: {msg} (threshold {threshold})"
    )
    assert ok, f"encoder: {msg}"


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


def run_e2e_logit_pcc(
    mesh_device: ttnn.Device,
    hf_model: SeamlessM4Tv2Model,
    case: TextDecoderPccInputs,
    enc_tt: ttnn.Tensor,
    enc_mask_tt: ttnn.Tensor,
    *,
    log_label: str,
    decode_steps: int,
    pcc_decode: float,
    pcc_encoder: float = PCC_ENCODER_TEXT,
    pcc_prefill: float = PCC_PREFILL,
) -> None:
    """Encoder PCC + teacher-forced decoder logit-PCC (pre-``lm_head`` hidden; TT encoder → TT decoder)."""
    cfg = hf_model.config
    decoder = hf_model.text_decoder
    pad_id = int(cfg.pad_token_id)
    aligned = align_case_for_tt_prefill(case, pad_id)
    batch = int(aligned.input_ids.shape[0])
    logical_dec = aligned.logical_dec_seq
    logical_enc = aligned.logical_enc_seq
    padded_dec = aligned.padded_dec_seq
    padded_enc = aligned.padded_enc_seq
    hidden_size = int(cfg.hidden_size)
    n_heads = int(cfg.decoder_attention_heads)
    max_seq_len = max(64, logical_dec + decode_steps + 8)

    _assert_encoder_pcc(
        enc_tt,
        case.encoder_hidden_states,
        logical_enc,
        log_label=log_label,
        threshold=pcc_encoder,
    )

    ref_prefill, ref_decode, decode_tokens = _hf_teacher_forced_decode_reference(
        hf_model, decoder, case, aligned, decode_steps
    )

    params = create_text_decoder_parameters(decoder, device=mesh_device)
    tt_dec = TTSeamlessM4Tv2Decoder(
        mesh_device,
        params,
        layer_norm_eps=cfg.layer_norm_eps,
        num_hidden_layers=cfg.decoder_layers,
        num_attention_heads=n_heads,
        hidden_size=hidden_size,
        max_batch_size=batch,
        max_seq_len=max_seq_len,
    )
    tp = get_tp(mesh_device)
    kv_cache, cross_attn_cache = init_text_decoder_kv_cache(
        mesh_device,
        num_hidden_layers=cfg.decoder_layers,
        num_attention_heads=n_heads,
        hidden_size=hidden_size,
        max_batch_size=batch,
        max_seq_len=max_seq_len,
        encoder_seq_len=padded_enc,
        tp=tp,
    )

    ids_tt = from_torch_uint32_rm(mesh_device, aligned.input_ids)
    pos_tt = tt_position_ids(ids_tt, pad_id)
    causal_tt = build_causal_with_padding_4d(None, batch, padded_dec, mesh_device)
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
    tt_prefill = (
        to_torch_replicated_first_shard(prefill_dev)
        .to(torch.bfloat16)
        .reshape(batch, padded_dec, hidden_size)[:, :logical_dec, :]
        .contiguous()
    )
    ttnn.deallocate(prefill_dev)
    ttnn.deallocate(ids_tt)
    ttnn.deallocate(pos_tt)
    ttnn.deallocate(causal_tt)
    ttnn.deallocate(cross_prefill_tt)

    ok, msg = check_with_pcc(ref_prefill, tt_prefill, pcc=pcc_prefill)
    logger.info(
        f"SeamlessM4Tv2 E2E logit PCC decoder prefill ({log_label}) dec_seq={logical_dec} "
        f"enc_seq={logical_enc}: {msg} (threshold {pcc_prefill})"
    )
    assert ok, f"prefill-fill: {msg}"

    cross_decode_tt = build_cross_attn_mask_4d(enc_mask_tt, tgt_seq=1, device=mesh_device)
    for step in range(decode_steps):
        position = logical_dec + step
        token_ids = from_torch_uint32_rm(mesh_device, torch.full((batch, 1), decode_tokens[step], dtype=torch.int32))
        step_pos = tt_position_ids_decode_step(token_ids, pad_id, position)
        cur_pos = tt_dec.borrow_current_decode_pos_tensor(position, batch_size=batch)
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
        tt_step = (
            to_torch_replicated_first_shard(dec_dev).to(torch.bfloat16).reshape(batch, 1, hidden_size).contiguous()
        )
        ttnn.deallocate(dec_dev)
        ttnn.deallocate(token_ids)
        ttnn.deallocate(step_pos)

        ok, msg = check_with_pcc(ref_decode[step], tt_step, pcc=pcc_decode)
        logger.info(
            f"SeamlessM4Tv2 E2E logit PCC decoder ({log_label}) decode step={step} pos={position} "
            f"tok={decode_tokens[step]}: {msg} (threshold {pcc_decode})"
        )
        assert ok, f"decode step {step} (pos={position}): {msg}"

    ttnn.deallocate(enc_tt)
    ttnn.deallocate(enc_mask_tt)
    ttnn.deallocate(cross_decode_tt)
    for layer in kv_cache:
        ttnn.deallocate(layer[0])
        ttnn.deallocate(layer[1])
    for layer in cross_attn_cache:
        ttnn.deallocate(layer[0])
        ttnn.deallocate(layer[1])


def run_t2tt_e2e_logit_pcc(
    mesh_device: ttnn.Device,
    hf_model: SeamlessM4Tv2Model,
    processor: AutoProcessor,
    *,
    tgt_lang: str,
    decode_steps: int,
    pcc_decode: float,
    log_label: str,
) -> None:
    src_ids, src_mask = _t2tt_source_ids(processor)
    case = make_t2tt_e2e_case(hf_model, processor, tgt_lang=tgt_lang)
    enc_tt, enc_mask_tt = tt_encode_text(mesh_device, hf_model.text_encoder, hf_model.config, src_ids, src_mask)
    run_e2e_logit_pcc(
        mesh_device,
        hf_model,
        case,
        enc_tt,
        enc_mask_tt,
        log_label=log_label,
        decode_steps=decode_steps,
        pcc_decode=pcc_decode,
        pcc_encoder=PCC_ENCODER_TEXT,
    )


def run_speech_e2e_logit_pcc(
    mesh_device: ttnn.Device,
    hf_model: SeamlessM4Tv2Model,
    processor: AutoProcessor,
    *,
    tgt_lang: str,
    decode_steps: int,
    pcc_decode: float,
    log_label: str,
    pcc_prefill: float = PCC_PREFILL_SPEECH,
) -> None:
    speech = make_speech_e2e_inputs(hf_model, processor, tgt_lang=tgt_lang, enc_seq_len=MAX_ENC_SEQ)
    enc_tt, enc_mask_tt = tt_encode_speech(
        mesh_device,
        hf_model.speech_encoder,
        hf_model.config,
        speech.input_features,
        speech.mel_attention_mask,
    )
    enc_tt, enc_mask_tt = _align_tt_encoder_to_case(
        mesh_device,
        enc_tt,
        enc_mask_tt,
        speech.case,
        pad_id=int(hf_model.config.pad_token_id),
        hidden_size=int(hf_model.config.hidden_size),
    )
    run_e2e_logit_pcc(
        mesh_device,
        hf_model,
        speech.case,
        enc_tt,
        enc_mask_tt,
        log_label=log_label,
        decode_steps=decode_steps,
        pcc_decode=pcc_decode,
        pcc_encoder=PCC_ENCODER_SPEECH,
        pcc_prefill=pcc_prefill,
    )
