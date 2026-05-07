# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from loguru import logger

from tests.ttnn.utils_for_testing import check_with_pcc

from models.experimental.seamless_m4t_v2_large.reference.torch_code_hifigan import (
    forward_torch_code_hifigan_reference,
    load_pretrained_code_hifigan,
)
from models.experimental.seamless_m4t_v2_large.scripts.download_weights import ensure_seamless_m4t_v2_large_weights
from models.experimental.seamless_m4t_v2_large.tt.model_preprocessing import create_code_hifigan_parameters
from models.experimental.seamless_m4t_v2_large.tt.tt_code_hifigan import TTSeamlessM4Tv2CodeHifiGan

PCC_THRESHOLD = 0.95


@pytest.mark.parametrize("device_params", [{"l1_small_size": 65536}], indirect=True)
def test_seamless_m4t_v2_code_hifigan_pcc(device, reset_seeds):
    try:
        weights_dir = ensure_seamless_m4t_v2_large_weights()
    except ImportError as e:
        pytest.skip(str(e))
    except Exception as e:
        pytest.skip(f"Could not prepare seamless-m4t-v2-large weights: {e}")

    torch.manual_seed(0)
    vocoder, cfg = load_pretrained_code_hifigan(weights_dir, dtype=torch.bfloat16)

    batch, seq = 1, 16
    pad_id = int(cfg.t2u_pad_token_id)
    vocab = int(cfg.unit_hifi_gan_vocab_size)
    low = max(pad_id + 1, 2)
    high = max(low + 1, min(vocab - 1, low + 9999))
    input_ids = torch.randint(low, high, (batch, seq), dtype=torch.int64)
    speaker_id = torch.zeros(batch, 1, dtype=torch.int64)
    lang_id = torch.zeros(batch, 1, dtype=torch.int64)

    ref_wav, ref_lengths = forward_torch_code_hifigan_reference(vocoder, input_ids, speaker_id, lang_id)
    ref_wav = ref_wav.to(torch.bfloat16)

    params = create_code_hifigan_parameters(vocoder, device=device)
    tt_v = TTSeamlessM4Tv2CodeHifiGan(device, params, cfg)

    input_ids_tt = ttnn.from_torch(
        input_ids.to(torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    sp_tt = ttnn.from_torch(
        speaker_id.to(torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    lang_tt = ttnn.from_torch(
        lang_id.to(torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    out_tt, lengths_tt = tt_v.forward(input_ids_tt, sp_tt, lang_tt, input_ids_torch=input_ids)

    tt_wav = ttnn.to_torch(ttnn.from_device(out_tt)).to(torch.bfloat16).squeeze(-1).contiguous()
    ref_1d = ref_wav.squeeze(1).contiguous()
    min_len = min(tt_wav.shape[-1], ref_1d.shape[-1])
    tt_wav = tt_wav[..., :min_len]
    ref_1d = ref_1d[..., :min_len]

    ok, msg = check_with_pcc(ref_1d, tt_wav, pcc=PCC_THRESHOLD)
    logger.info(f"SeamlessM4Tv2 CodeHiFi-GAN PCC: {msg} (threshold {PCC_THRESHOLD})")

    assert torch.allclose(lengths_tt.cpu(), ref_lengths.cpu()), "lengths mismatch vs HF"

    assert ok, msg
