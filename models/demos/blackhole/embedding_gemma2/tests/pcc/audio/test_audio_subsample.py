# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Subsampling (two on-device conv2d + LayerNorm + ReLU with input masking, flatten) and input_proj vs HF.

test_conv_stack     log-mel -> flattened conv output (HF input_proj_linear input), valid tokens
test_input_proj     input_proj_linear on the HF flattened conv output (teacher forced)
test_subsample      log-mel -> layer-0 input (whole HF Gemma4AudioSubSampleConvProjection)
Also checks the token count against the HF output mask.
"""

import pytest

from models.demos.blackhole.embedding_gemma2.tests.common import PCC_LAYER, log_result, pcc
from models.demos.blackhole.embedding_gemma2.tests.pcc.audio.audio_common import (
    ALL_CASES,
    device_inputs,
    download_rows,
    upload_rows,
    valid,
)


@pytest.mark.parametrize("case", ALL_CASES + ("crop1", "crop13"))
def test_conv_stack(case, tt_audio_encoder, audio_reference_for):
    feats, mask, ref = audio_reference_for(case)
    host, x = device_inputs(tt_audio_encoder, feats, mask)
    assert host.num_tokens == ref.token_mask.sum(-1).tolist()
    N = ref.token_mask.shape[1]
    flat = download_rows(tt_audio_encoder.model.subsample.conv_stack(x), N)
    value = pcc(*valid(ref.proj_in, flat, ref.token_mask))
    assert log_result("audio_conv_stack_pcc", value, PCC_LAYER, case=case, tokens=host.num_tokens)


@pytest.mark.parametrize("case", ("short", "s30", "b2_s10_s2"))
def test_input_proj(case, eg2_device, tt_audio_encoder, audio_reference_for):
    feats, mask, ref = audio_reference_for(case)
    host = tt_audio_encoder.prepare_inputs(feats, mask)
    N = ref.token_mask.shape[1]
    out = tt_audio_encoder.model.subsample.input_proj(upload_rows(eg2_device, ref.proj_in, host.seq_len))
    value = pcc(*valid(ref.subsample_out, download_rows(out, N), ref.token_mask))
    assert log_result("audio_input_proj_pcc", value, PCC_LAYER, case=case)


@pytest.mark.parametrize("case", ALL_CASES + ("crop1", "crop13"))
def test_subsample(case, tt_audio_encoder, audio_reference_for):
    feats, mask, ref = audio_reference_for(case)
    host, x = device_inputs(tt_audio_encoder, feats, mask)
    N = ref.token_mask.shape[1]
    out = download_rows(tt_audio_encoder.model.subsample(x), N)
    value = pcc(*valid(ref.subsample_out, out, ref.token_mask))
    assert log_result("audio_subsample_pcc", value, PCC_LAYER, case=case, tokens=host.num_tokens)
