# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Conformer sub-modules vs HF (teacher forced: each gets the HF input of that module, real audio).

test_feed_forward   FFN1 and FFN2 (Macaron half steps) of layers 0, 5, 11
test_attention      chunked causal-local attention of layers 0, 5, 11 on the layer cases, and of
                    layer 0 on crops of 1, 11, 12, 13, 24, 25, 37, 100 tokens (first / last block,
                    lengths not divisible by 12); PCC over valid tokens plus the first and the last
                    12-token chunk on their own
test_light_conv     LightConv1d of layers 0, 5, 11
test_host_mask_matches_hf
                    the host-built band mask keeps exactly the keys the HF 5D blocked mask keeps
"""

import pytest
import torch

import ttnn
from models.demos.blackhole.embedding_gemma2.tests.common import PCC_LAYER, log_result, pcc
from models.demos.blackhole.embedding_gemma2.tests.pcc.audio.audio_common import (
    CROP_TOKENS,
    LAYER_CASES,
    SUBLAYERS,
    download_rows,
    upload_rows,
    valid,
)
from models.demos.blackhole.embedding_gemma2.tt.audio.attention import attention_mask

LAYER_PARAMS = [(c, i) for c in LAYER_CASES for i in SUBLAYERS]
LAYER_IDS = [f"{c}-L{i}" for c, i in LAYER_PARAMS]
CROP_CASES = tuple(f"crop{n}" for n in CROP_TOKENS)


def _upload_mask(device, ref, S, cfg):
    amask = attention_mask(ref.token_mask.sum(-1).tolist(), S, cfg.attention)
    return ttnn.from_torch(amask, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)


@pytest.mark.parametrize("case,layer", LAYER_PARAMS, ids=LAYER_IDS)
@pytest.mark.parametrize("which", ["ffn1", "ffn2"])
def test_feed_forward(case, layer, which, eg2_device, tt_audio_encoder, audio_reference_for):
    feats, mask, ref = audio_reference_for(case)
    S = tt_audio_encoder.prepare_inputs(feats, mask).seq_len
    N = ref.token_mask.shape[1]
    L = tt_audio_encoder.model.layers[layer]
    x_in = ref.ffn1_in[layer] if which == "ffn1" else ref.ffn2_in[layer]
    y_ref = ref.ffn1_out[layer] if which == "ffn1" else ref.ffn2_out[layer]
    y = download_rows(getattr(L, which)(upload_rows(eg2_device, x_in, S)), N)
    value = pcc(*valid(y_ref, y, ref.token_mask))
    assert log_result(f"audio_{which}_pcc", value, PCC_LAYER, case=case, layer=layer)


def _check_attention(case, layer, eg2_device, encoder, ref, S):
    N = ref.token_mask.shape[1]
    L = encoder.model.layers[layer]
    m = _upload_mask(eg2_device, ref, S, encoder.config)
    y = download_rows(L.attn(upload_rows(eg2_device, ref.attn_in[layer], S), m), N)
    tm = ref.token_mask
    ok = log_result("audio_attention_pcc", pcc(*valid(ref.attn_out[layer], y, tm)), PCC_LAYER, case=case, layer=layer)
    n_min = int(tm.sum(-1).min())
    chunk = encoder.config.chunk_size
    first = slice(0, min(chunk, n_min))
    last_start = ((n_min - 1) // chunk) * chunk
    for name, sl in (("first", first), ("last", slice(last_start, n_min))):
        a, b = ref.attn_out[layer][:, sl], y[:, sl]
        if a.numel() > 1:
            ok &= log_result(f"audio_attention_{name}_chunk_pcc", pcc(a, b), PCC_LAYER, case=case, layer=layer)
    return ok


@pytest.mark.parametrize("case,layer", LAYER_PARAMS, ids=LAYER_IDS)
def test_attention(case, layer, eg2_device, tt_audio_encoder, audio_reference_for):
    feats, mask, ref = audio_reference_for(case)
    S = tt_audio_encoder.prepare_inputs(feats, mask).seq_len
    assert _check_attention(case, layer, eg2_device, tt_audio_encoder, ref, S)


@pytest.mark.parametrize("case", CROP_CASES)
def test_attention_crop(case, eg2_device, tt_audio_encoder, audio_reference_for):
    feats, mask, ref = audio_reference_for(case)
    host = tt_audio_encoder.prepare_inputs(feats, mask)
    assert host.num_tokens == [int(case[4:])]
    assert _check_attention(case, 0, eg2_device, tt_audio_encoder, ref, host.seq_len)


@pytest.mark.parametrize("case,layer", LAYER_PARAMS, ids=LAYER_IDS)
def test_light_conv(case, layer, eg2_device, tt_audio_encoder, audio_reference_for):
    feats, mask, ref = audio_reference_for(case)
    S = tt_audio_encoder.prepare_inputs(feats, mask).seq_len
    N = ref.token_mask.shape[1]
    L = tt_audio_encoder.model.layers[layer]
    y = download_rows(L.lconv(upload_rows(eg2_device, ref.lconv_in[layer], S)), N)
    value = pcc(*valid(ref.lconv_out[layer], y, ref.token_mask))
    assert log_result("audio_lconv_pcc", value, PCC_LAYER, case=case, layer=layer)


@pytest.mark.parametrize("case", ("short", "b2_s10_s2", "crop13", "crop25"))
def test_host_mask_matches_hf(case, audio_config, tt_audio_encoder, audio_reference_for):
    """Kept (query, key) pairs of the host band mask == those of HF's blocked mask, on valid queries."""
    feats, mask, ref = audio_reference_for(case)
    m5 = ref.attn_mask_5d  # [B, 1, nb, 12, 24], 0 = keep
    B, N = ref.token_mask.shape
    chunk, ctx = m5.shape[3], m5.shape[4]
    past = ctx - chunk
    keep_hf = torch.zeros(B, N, N, dtype=torch.bool)
    for blk in range(m5.shape[2]):
        for i in range(chunk):
            q = blk * chunk + i
            if q >= N:
                continue
            for j in range(ctx):
                kv = blk * chunk - past + j
                if 0 <= kv < N:
                    keep_hf[:, q, kv] = m5[:, 0, blk, i, j] == 0
    S = tt_audio_encoder.prepare_inputs(feats, mask).seq_len
    mine = attention_mask(ref.token_mask.sum(-1).tolist(), S, audio_config.attention)[:, 0]
    keep_mine = torch.zeros(B, N, N, dtype=torch.bool)
    for q in range(N):
        for j in range(mine.shape[-1]):
            kv = (q // 32) * 32 - 32 + j
            if 0 <= kv < N:
                keep_mine[:, q, kv] = mine[:, q, j] == 0
    vq = ref.token_mask
    assert torch.equal(keep_hf[vq], keep_mine[vq])
    # every valid query keeps itself and at most 12 keys
    assert int(keep_mine[vq].sum(-1).max()) <= audio_config.context_left - 1
    assert bool(keep_mine[vq].sum(-1).min() >= 1)
