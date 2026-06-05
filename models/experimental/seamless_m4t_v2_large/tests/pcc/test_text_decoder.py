# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""PCC tests for the SeamlessM4Tv2 text decoder using production-shaped inputs.

The decoder is fed the same tensors ``generate()`` uses:

  * **T2TT** — ``text_encoder`` hidden states on a tokenized source prompt + a two-token seed
    ``[decoder_start_token_id, tgt_lang_code]``.
  * **S2TT** — ``speech_encoder`` hidden states on processor ``input_features`` from audio +
    the same decoder seed, with subsampled encoder attention masks.

Random ``input_ids`` / ``randn(encoder_hidden)`` are intentionally avoided: they do not match
the activation distribution of the HF or TT stack and falsely cap PCC at short seq.

Both HF reference and TT paths tile-pad encoder/decoder timelines and build masks the same way
``TTSeamlessM4Tv2Model._prefill_text_decoder_kv_cache`` does before ``text_decoder.forward``.
PCC is checked on the logical (unpadded) decoder prefix only.

Hardware prefill L1 for very long encoder timelines is a separate ceiling from decoder seed
length; production only prefills the two-token seed regardless of source length.
"""

from __future__ import annotations

import pytest
import torch
import ttnn
from loguru import logger

from tests.ttnn.utils_for_testing import check_with_pcc

from models.experimental.seamless_m4t_v2_large.reference.torch_text_decoder import forward_torch_reference
from models.experimental.seamless_m4t_v2_large.scripts.download_weights import ensure_seamless_m4t_v2_large_weights
from models.experimental.seamless_m4t_v2_large.tests.pcc.decoder_pcc_fixtures import (
    TextDecoderPccInputs,
    align_case_for_tt_prefill,
    load_hf_model_and_processor,
    make_s2tt_decoder_pcc_inputs,
    make_t2tt_decoder_pcc_inputs,
)
from models.experimental.seamless_m4t_v2_large.tt.common import (
    build_causal_with_padding_4d,
    build_cross_attn_mask_4d,
    to_torch_replicated_first_shard,
    tt_position_ids,
    tt_position_ids_decode_step,
)
from models.experimental.seamless_m4t_v2_large.tt.mesh_helpers import (
    MESH_DEVICE_PARAMETRIZE_TEXT,
    from_torch_bfloat16_tile,
    from_torch_uint32_rm,
    get_tp,
    mesh_default_device,
)
from models.experimental.seamless_m4t_v2_large.tt.model_preprocessing import create_text_decoder_parameters
from models.experimental.seamless_m4t_v2_large.tt.tt_text_decoder import (
    TTSeamlessM4Tv2Decoder,
    init_text_decoder_kv_cache,
    warm_text_decoder_kv_cache_prefill,
)

PCC_THRESHOLD = 0.99
# Longest tokenized source length (text-encoder output) validated with production seed on BH 1×4.
# Next power-of-two (1024) overflows L1 on cross-attention prefill at tile-padded encoder length.
MAX_ENC_SEQ = 512
# Number of greedy decode steps to validate after the prefill cache-fill. Each step is one KV-cache
# decoder forward (``seq_len=1``); a single step already exercises the decode self/cross-attn,
# paged-cache-write, and position-stepping paths. Bump for deeper cache-drift / decode profiling.
DECODE_STEPS = 1


def _weights_dir_or_skip() -> str:
    try:
        return ensure_seamless_m4t_v2_large_weights()
    except ImportError as e:
        pytest.skip(str(e))
    except Exception as e:
        pytest.skip(f"Could not prepare seamless-m4t-v2-large weights: {e}")


def _run_decoder_pcc(
    mesh_device,
    decoder,
    cfg,
    case: TextDecoderPccInputs,
    *,
    log_label: str,
) -> None:
    aligned = align_case_for_tt_prefill(case, int(cfg.pad_token_id))
    batch = int(aligned.input_ids.shape[0])
    logical_dec = aligned.logical_dec_seq
    padded_dec = aligned.padded_dec_seq

    ref = forward_torch_reference(
        decoder,
        aligned.input_ids,
        aligned.encoder_hidden_states,
        aligned.attention_mask,
        aligned.encoder_attention_mask,
    ).to(torch.bfloat16)
    ref = ref[:, :logical_dec, :].contiguous()

    params = create_text_decoder_parameters(decoder, device=mesh_device)
    tt_dec = TTSeamlessM4Tv2Decoder(
        mesh_device,
        params,
        layer_norm_eps=cfg.layer_norm_eps,
        num_hidden_layers=cfg.decoder_layers,
        num_attention_heads=cfg.decoder_attention_heads,
        hidden_size=cfg.hidden_size,
    )

    input_ids_tt = from_torch_uint32_rm(mesh_device, aligned.input_ids)
    encoder_tt = from_torch_bfloat16_tile(mesh_device, aligned.encoder_hidden_states)
    enc_mask_tt = from_torch_uint32_rm(mesh_device, aligned.encoder_attention_mask)
    pos_tt = tt_position_ids(input_ids_tt, int(cfg.pad_token_id))
    causal_tt = build_causal_with_padding_4d(None, batch, padded_dec, mesh_device)
    cross_tt = build_cross_attn_mask_4d(enc_mask_tt, tgt_seq=padded_dec, device=mesh_device)

    out_tt = tt_dec.forward(input_ids_tt, pos_tt, encoder_tt, causal_tt, cross_tt)
    tt_cpu = (
        to_torch_replicated_first_shard(out_tt)
        .to(torch.bfloat16)
        .reshape(batch, padded_dec, cfg.hidden_size)[:, :logical_dec, :]
        .contiguous()
    )

    ttnn.deallocate(out_tt)
    ttnn.deallocate(input_ids_tt)
    ttnn.deallocate(encoder_tt)
    ttnn.deallocate(enc_mask_tt)
    ttnn.deallocate(pos_tt)
    ttnn.deallocate(causal_tt)
    ttnn.deallocate(cross_tt)

    ok, msg = check_with_pcc(ref, tt_cpu, pcc=PCC_THRESHOLD)
    logger.info(
        f"SeamlessM4Tv2 text decoder PCC ({log_label}) dec_seq={logical_dec} enc_seq={aligned.logical_enc_seq}: "
        f"{msg} (threshold {PCC_THRESHOLD})"
    )
    assert ok, msg


def _run_decoder_prefill_decode_pcc(
    mesh_device,
    hf_model,
    decoder,
    cfg,
    case: TextDecoderPccInputs,
    *,
    log_label: str,
    decode_steps: int = DECODE_STEPS,
) -> None:
    """Prefill (cache-fill) PCC + ``decode_steps`` KV-cache decode-step PCCs against an incremental HF run.

    Mirrors production ``generate()``: one batched prefill forward fills the self/cross KV caches
    (``prefill_kv_cache_fill=True``), then greedy single-token decode steps read the caches. The HF
    reference runs the same path with ``use_cache=True`` and teacher-forces the TT decode loop with
    the HF-greedy token at each step so both stacks step identical inputs. PCC is checked on the
    decoder hidden states (post final layer-norm) at the prefill seed and at every decode step.
    """
    aligned = align_case_for_tt_prefill(case, int(cfg.pad_token_id))
    batch = int(aligned.input_ids.shape[0])
    logical_dec = aligned.logical_dec_seq
    padded_dec = aligned.padded_dec_seq
    padded_enc = aligned.padded_enc_seq
    pad_id = int(cfg.pad_token_id)
    hidden_size = int(cfg.hidden_size)
    n_heads = int(cfg.decoder_attention_heads)
    # Cache horizon: seed + decode steps, with headroom. init pads this to a 256 chunk internally.
    max_seq_len = max(64, logical_dec + decode_steps + 8)

    # --- HF reference: incremental decode with use_cache=True ---
    # Prefill the LOGICAL seed against the tile-padded encoder (matches TT cross-attn cache K).
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

    # --- TT path: prefill cache-fill, then teacher-forced decode loop ---
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
    enc_tt = from_torch_bfloat16_tile(mesh_device, aligned.encoder_hidden_states)
    enc_mask_tt = from_torch_uint32_rm(mesh_device, aligned.encoder_attention_mask)
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

    ok, msg = check_with_pcc(ref_prefill, tt_prefill, pcc=PCC_THRESHOLD)
    logger.info(
        f"SeamlessM4Tv2 text decoder PCC ({log_label}) prefill-fill dec_seq={logical_dec} "
        f"enc_seq={aligned.logical_enc_seq}: {msg} (threshold {PCC_THRESHOLD})"
    )
    assert ok, f"prefill-fill: {msg}"

    cross_decode_tt = build_cross_attn_mask_4d(enc_mask_tt, tgt_seq=1, device=mesh_device)
    for step in range(decode_steps):
        position = logical_dec + step  # cache slot the generated token is written to
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

        ok, msg = check_with_pcc(ref_decode[step], tt_step, pcc=PCC_THRESHOLD)
        logger.info(
            f"SeamlessM4Tv2 text decoder PCC ({log_label}) decode step={step} pos={position} "
            f"tok={decode_tokens[step]}: {msg} (threshold {PCC_THRESHOLD})"
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


@pytest.mark.timeout(1800)
@pytest.mark.parametrize(*MESH_DEVICE_PARAMETRIZE_TEXT, indirect=["mesh_device", "device_params"])
def test_seamless_m4t_v2_text_decoder_t2tt_prefill_pcc(mesh_device, device_params, reset_seeds):
    """Decoder prefill PCC ≥ 0.99 on T2TT-shaped inputs (text-encoder hidden + lang seed)."""
    _ = reset_seeds
    _ = device_params
    weights_dir = _weights_dir_or_skip()

    with mesh_default_device(mesh_device):
        hf_model, processor, _ = load_hf_model_and_processor(weights_dir, dtype=torch.bfloat16)
        case = make_t2tt_decoder_pcc_inputs(hf_model, processor)
        _run_decoder_pcc(mesh_device, hf_model.text_decoder, hf_model.config, case, log_label="T2TT")


@pytest.mark.timeout(1800)
@pytest.mark.parametrize(*MESH_DEVICE_PARAMETRIZE_TEXT, indirect=["mesh_device", "device_params"])
def test_seamless_m4t_v2_text_decoder_s2tt_prefill_pcc(mesh_device, device_params, reset_seeds):
    """Decoder prefill PCC ≥ 0.99 on S2TT-shaped inputs (speech-encoder hidden + lang seed)."""
    _ = reset_seeds
    _ = device_params
    weights_dir = _weights_dir_or_skip()

    with mesh_default_device(mesh_device):
        hf_model, processor, _ = load_hf_model_and_processor(weights_dir, dtype=torch.bfloat16)
        case = make_s2tt_decoder_pcc_inputs(hf_model, processor)
        _run_decoder_pcc(mesh_device, hf_model.text_decoder, hf_model.config, case, log_label="S2TT")


@pytest.mark.timeout(3600)
@pytest.mark.parametrize(*MESH_DEVICE_PARAMETRIZE_TEXT, indirect=["mesh_device", "device_params"])
def test_seamless_m4t_v2_text_decoder_t2tt_max_enc_seq_pcc(mesh_device, device_params, reset_seeds):
    """T2TT decoder prefill PCC ≥ 0.99 at ``MAX_ENC_SEQ`` text-encoder timeline (decoder seed = 2)."""
    _ = reset_seeds
    _ = device_params
    weights_dir = _weights_dir_or_skip()

    with mesh_default_device(mesh_device):
        hf_model, processor, _ = load_hf_model_and_processor(weights_dir, dtype=torch.bfloat16)
        case = make_t2tt_decoder_pcc_inputs(hf_model, processor, enc_seq_len=MAX_ENC_SEQ)
        assert int(case.encoder_hidden_states.shape[1]) == MAX_ENC_SEQ
        _run_decoder_pcc(mesh_device, hf_model.text_decoder, hf_model.config, case, log_label="T2TT-max-enc")


@pytest.mark.timeout(3600)
@pytest.mark.parametrize(*MESH_DEVICE_PARAMETRIZE_TEXT, indirect=["mesh_device", "device_params"])
def test_seamless_m4t_v2_text_decoder_s2tt_max_enc_seq_pcc(mesh_device, device_params, reset_seeds):
    """S2TT decoder PCC ≥ 0.99 at ``MAX_ENC_SEQ`` speech-encoder timeline: prefill cache-fill (seed = 2)
    plus ``DECODE_STEPS`` KV-cache decode steps, validating both the prefill and decode paths."""
    _ = reset_seeds
    _ = device_params
    weights_dir = _weights_dir_or_skip()

    with mesh_default_device(mesh_device):
        hf_model, processor, _ = load_hf_model_and_processor(weights_dir, dtype=torch.bfloat16)
        case = make_s2tt_decoder_pcc_inputs(hf_model, processor, enc_seq_len=MAX_ENC_SEQ)
        assert int(case.encoder_hidden_states.shape[1]) == MAX_ENC_SEQ
        _run_decoder_prefill_decode_pcc(
            mesh_device, hf_model, hf_model.text_decoder, hf_model.config, case, log_label="S2TT-max-enc"
        )
