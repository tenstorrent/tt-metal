# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Single-layer text-decoder prefill+decode workload for Tracy / device-perf profiling.

Runs **one** ``TTSeamlessM4Tv2Decoder`` layer (layer 0 weights only) with prefill
``PREFILL_SEQ_LEN`` decoder tokens and one KV-cache decode step — same pattern as
``devstral2_123B_instruct/tests/perf/test_profile_single_layer_prefill_decode.py``.

Each measured iteration runs prefill then decode inside the ``start``/``stop`` signpost
window. Warmup iteration runs first without signposts.

Standalone Tracy capture::

    python -m tracy -p -v -r --dump-device-data-mid-run \\
        pytest models/experimental/seamless_m4t_v2_large/tests/perf/test_profile_single_layer_prefill_decode.py \\
        ::test_profile_single_layer_prefill_decode -v

Device perf CSV/JSON dump (outer driver wraps the command via ``run_device_perf``)::

    pytest models/experimental/seamless_m4t_v2_large/tests/perf/test_device_perf_single_layer_prefill_decode.py \\
        -v -m models_device_performance_bare_metal
"""

from __future__ import annotations

from typing import NamedTuple

import pytest
import torch
import ttnn
from loguru import logger

from models.experimental.seamless_m4t_v2_large.scripts.download_weights import ensure_seamless_m4t_v2_large_weights
from models.experimental.seamless_m4t_v2_large.tests.pcc.decoder_pcc_fixtures import (
    TextDecoderPccInputs,
    align_case_for_tt_prefill,
    load_hf_model_and_processor,
    make_t2tt_decoder_pcc_inputs,
)
from models.experimental.seamless_m4t_v2_large.tt.common import (
    build_causal_with_padding_4d,
    build_cross_attn_mask_4d,
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

NUM_LAYERS = 1
PREFILL_SEQ_LEN = 128
DECODE_POS = PREFILL_SEQ_LEN
ENC_SEQ_LEN = 128
NUM_WARMUP_ITERS = 1


class _LayerPerfFixtures(NamedTuple):
    cfg: object
    tt_dec: TTSeamlessM4Tv2Decoder
    aligned: object
    pad_id: int
    batch: int
    logical_dec: int
    padded_dec: int
    padded_enc: int
    decode_token: int
    enc_tt: ttnn.Tensor
    enc_mask_tt: ttnn.Tensor


def _weights_dir_or_skip() -> str:
    try:
        return ensure_seamless_m4t_v2_large_weights()
    except ImportError as e:
        pytest.skip(str(e))
    except Exception as e:
        pytest.skip(f"Could not prepare seamless-m4t-v2-large weights: {e}")


def _tracy_signpost_available() -> bool:
    try:
        from tracy import signpost  # noqa: F401

        return True
    except ImportError:
        return False


def _dealloc_kv_caches(
    kv_cache: list[list[ttnn.Tensor]],
    cross_attn_cache: list[list[ttnn.Tensor]],
) -> None:
    for layer in kv_cache:
        ttnn.deallocate(layer[0])
        ttnn.deallocate(layer[1])
    for layer in cross_attn_cache:
        ttnn.deallocate(layer[0])
        ttnn.deallocate(layer[1])


def _make_layer_perf_case(hf_model, processor) -> TextDecoderPccInputs:
    """T2TT encoder timeline @ ``ENC_SEQ_LEN`` with a ``PREFILL_SEQ_LEN`` decoder prefill."""
    base = make_t2tt_decoder_pcc_inputs(hf_model, processor, enc_seq_len=ENC_SEQ_LEN)
    torch.manual_seed(42)
    dec_ids = torch.randint(0, int(hf_model.config.vocab_size), (1, PREFILL_SEQ_LEN), dtype=torch.long)
    dec_mask = torch.ones_like(dec_ids)
    return TextDecoderPccInputs(
        input_ids=dec_ids,
        attention_mask=dec_mask,
        encoder_hidden_states=base.encoder_hidden_states,
        encoder_attention_mask=base.encoder_attention_mask,
    )


def _setup_seamless_decoder_one_layer_from_case(
    mesh_device,
    hf_model,
    case: TextDecoderPccInputs,
) -> _LayerPerfFixtures:
    cfg = hf_model.config
    pad_id = int(cfg.pad_token_id)
    aligned = align_case_for_tt_prefill(case, pad_id)
    batch = int(aligned.input_ids.shape[0])
    logical_dec = aligned.logical_dec_seq
    padded_dec = aligned.padded_dec_seq
    padded_enc = aligned.padded_enc_seq
    max_seq_len = max(64, logical_dec + 8)

    params = create_text_decoder_parameters(hf_model.text_decoder, device=mesh_device)
    params.layers = params.layers[:NUM_LAYERS]
    tt_dec = TTSeamlessM4Tv2Decoder(
        mesh_device,
        params,
        layer_norm_eps=cfg.layer_norm_eps,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=cfg.decoder_attention_heads,
        hidden_size=cfg.hidden_size,
        max_batch_size=batch,
        max_seq_len=max_seq_len,
    )

    enc_tt = from_torch_bfloat16_tile(mesh_device, aligned.encoder_hidden_states)
    enc_mask_tt = from_torch_uint32_rm(mesh_device, aligned.encoder_attention_mask)
    decode_token = int(cfg.decoder_start_token_id)

    return _LayerPerfFixtures(
        cfg=cfg,
        tt_dec=tt_dec,
        aligned=aligned,
        pad_id=pad_id,
        batch=batch,
        logical_dec=logical_dec,
        padded_dec=padded_dec,
        padded_enc=padded_enc,
        decode_token=decode_token,
        enc_tt=enc_tt,
        enc_mask_tt=enc_mask_tt,
    )


def _run_prefill_decode_step(mesh_device, fixtures: _LayerPerfFixtures) -> None:
    """One KV prefill + one decode forward on the single decoder layer."""
    tt_dec = fixtures.tt_dec
    aligned = fixtures.aligned
    tp = get_tp(mesh_device)
    kv_cache, cross_attn_cache = init_text_decoder_kv_cache(
        mesh_device,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=tt_dec.num_attention_heads,
        hidden_size=tt_dec.hidden_size,
        max_batch_size=fixtures.batch,
        max_seq_len=tt_dec.max_seq_len,
        encoder_seq_len=fixtures.padded_enc,
        tp=tp,
    )

    ids_tt = from_torch_uint32_rm(mesh_device, aligned.input_ids)
    pos_tt = tt_position_ids(ids_tt, fixtures.pad_id)
    causal_tt = build_causal_with_padding_4d(None, fixtures.batch, fixtures.padded_dec, mesh_device)
    cross_prefill_tt = build_cross_attn_mask_4d(fixtures.enc_mask_tt, tgt_seq=fixtures.padded_dec, device=mesh_device)

    prefill_out = warm_text_decoder_kv_cache_prefill(
        tt_dec,
        ids_tt,
        pos_tt,
        fixtures.enc_tt,
        causal_tt,
        cross_prefill_tt,
        kv_cache,
        cross_attn_cache,
        kv_cache_fill_len=fixtures.logical_dec,
    )
    ttnn.deallocate(prefill_out)
    ttnn.deallocate(ids_tt)
    ttnn.deallocate(pos_tt)
    ttnn.deallocate(causal_tt)
    ttnn.deallocate(cross_prefill_tt)

    cross_decode_tt = build_cross_attn_mask_4d(fixtures.enc_mask_tt, tgt_seq=1, device=mesh_device)
    token_ids = from_torch_uint32_rm(
        mesh_device,
        torch.full((fixtures.batch, 1), fixtures.decode_token, dtype=torch.int32),
    )
    step_pos = tt_position_ids_decode_step(token_ids, fixtures.pad_id, DECODE_POS)
    cur_pos = tt_dec.borrow_current_decode_pos_tensor(DECODE_POS, batch_size=fixtures.batch)
    dec_out = tt_dec.forward(
        token_ids,
        step_pos,
        fixtures.enc_tt,
        None,
        cross_decode_tt,
        kv_cache=kv_cache,
        cross_attn_cache=cross_attn_cache,
        cross_attn_cache_valid=True,
        current_decode_pos=cur_pos,
        cache_seq_len=DECODE_POS + 1,
    )
    ttnn.deallocate(dec_out)
    ttnn.deallocate(token_ids)
    ttnn.deallocate(step_pos)
    ttnn.deallocate(cross_decode_tt)
    _dealloc_kv_caches(kv_cache, cross_attn_cache)


@pytest.mark.timeout(3600)
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(*MESH_DEVICE_PARAMETRIZE_TEXT, indirect=["mesh_device", "device_params"])
def test_profile_single_layer_prefill_decode(mesh_device, device_params):
    """Prefill 128 + decode 1 on a 1-layer ``TTSeamlessM4Tv2Decoder`` (Tracy profile target)."""
    _ = device_params
    weights_dir = _weights_dir_or_skip()

    use_signpost = _tracy_signpost_available()
    if use_signpost:
        from tracy import signpost
    else:
        logger.info("tracy.signpost unavailable; running profile workload without signpost markers.")

    with mesh_default_device(mesh_device):
        hf_model, processor, _ = load_hf_model_and_processor(weights_dir, dtype=torch.bfloat16)
        case = _make_layer_perf_case(hf_model, processor)
        assert int(case.encoder_hidden_states.shape[1]) == ENC_SEQ_LEN
        assert int(case.input_ids.shape[1]) == PREFILL_SEQ_LEN

        fixtures = _setup_seamless_decoder_one_layer_from_case(mesh_device, hf_model, case)

        for _ in range(NUM_WARMUP_ITERS):
            _run_prefill_decode_step(mesh_device, fixtures)
            ttnn.synchronize_device(mesh_device)

        if use_signpost:
            signpost("start")

        _run_prefill_decode_step(mesh_device, fixtures)
        ttnn.synchronize_device(mesh_device)

        if use_signpost:
            signpost("stop")

        ttnn.deallocate(fixtures.enc_tt)
        ttnn.deallocate(fixtures.enc_mask_tt)

    logger.info(
        f"Profile workload complete: layers={NUM_LAYERS}, prefill_seq_len={PREFILL_SEQ_LEN}, "
        f"decode_pos={DECODE_POS}, enc_seq={ENC_SEQ_LEN}, signposts={'on' if use_signpost else 'off'}"
    )
