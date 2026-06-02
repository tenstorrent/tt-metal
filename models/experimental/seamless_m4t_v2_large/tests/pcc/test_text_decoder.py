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
)
from models.experimental.seamless_m4t_v2_large.tt.mesh_helpers import (
    MESH_DEVICE_PARAMETRIZE_TEXT,
    from_torch_bfloat16_tile,
    from_torch_uint32_rm,
    mesh_default_device,
)
from models.experimental.seamless_m4t_v2_large.tt.model_preprocessing import create_text_decoder_parameters
from models.experimental.seamless_m4t_v2_large.tt.tt_text_decoder import TTSeamlessM4Tv2Decoder

PCC_THRESHOLD = 0.99
# Longest tokenized source length (text-encoder output) validated with production seed on BH 1×4.
# Next power-of-two (1024) overflows L1 on cross-attention prefill at tile-padded encoder length.
MAX_ENC_SEQ = 512


def _weights_dir_or_skip() -> str:
    try:
        return ensure_seamless_m4t_v2_large_weights()
    except ImportError as e:
        pytest.skip(str(e))
        raise
    except Exception as e:
        pytest.skip(f"Could not prepare seamless-m4t-v2-large weights: {e}")
        raise


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
    """S2TT decoder prefill PCC ≥ 0.99 at ``MAX_ENC_SEQ`` speech-encoder timeline (decoder seed = 2)."""
    _ = reset_seeds
    _ = device_params
    weights_dir = _weights_dir_or_skip()

    with mesh_default_device(mesh_device):
        hf_model, processor, _ = load_hf_model_and_processor(weights_dir, dtype=torch.bfloat16)
        case = make_s2tt_decoder_pcc_inputs(hf_model, processor, enc_seq_len=MAX_ENC_SEQ)
        assert int(case.encoder_hidden_states.shape[1]) == MAX_ENC_SEQ
        _run_decoder_pcc(mesh_device, hf_model.text_decoder, hf_model.config, case, log_label="S2TT-max-enc")
