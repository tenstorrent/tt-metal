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

PCC_THRESHOLD = 0.99


@pytest.mark.parametrize("device_params", [{"l1_small_size": 65536}], indirect=True)
@pytest.mark.parametrize("batch", [1, 2])
def test_seamless_m4t_v2_code_hifigan_pcc(device, batch, reset_seeds):
    try:
        weights_dir = ensure_seamless_m4t_v2_large_weights()
    except ImportError as e:
        pytest.skip(str(e))
    except Exception as e:
        pytest.skip(f"Could not prepare seamless-m4t-v2-large weights: {e}")

    torch.manual_seed(0)
    vocoder, cfg = load_pretrained_code_hifigan(weights_dir, dtype=torch.bfloat16)

    seq = 16
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

    out_tt, lengths_tt = tt_v.forward(input_ids_tt, sp_tt, lang_tt)

    tt_wav = ttnn.to_torch(ttnn.from_device(out_tt)).to(torch.bfloat16).squeeze(-1).contiguous()
    lengths_torch = ttnn.to_torch(ttnn.from_device(lengths_tt)).to(torch.long).reshape(-1)

    # ``ref_wav`` is ``[B, T_wav_ref]`` after ``squeeze(1)``; ``ref_lengths`` is ``[B]``.
    if ref_wav.dim() == 3:
        ref_2d = ref_wav.squeeze(1)
    else:
        ref_2d = ref_wav  # already 2D for some HF versions
    ref_lengths_t = ref_lengths.reshape(-1).to(torch.long)

    assert torch.allclose(
        lengths_torch, ref_lengths_t
    ), f"lengths mismatch vs HF: tt={lengths_torch.tolist()} ref={ref_lengths_t.tolist()}"

    pcc_min = 1.0
    for b in range(batch):
        valid_len = int(lengths_torch[b].item())
        min_len = min(valid_len, ref_2d.shape[-1], tt_wav.shape[-1])
        tt_b = tt_wav[b, :min_len].contiguous()
        ref_b = ref_2d[b, :min_len].contiguous()
        ok, msg = check_with_pcc(ref_b, tt_b, pcc=PCC_THRESHOLD)
        # ``msg`` is the float PCC value when passing, or a string when failing.
        try:
            pcc_val = float(msg)
        except (TypeError, ValueError):
            pcc_val = -1.0
        pcc_min = min(pcc_min, pcc_val) if pcc_val >= 0 else pcc_min
        logger.info(
            f"SeamlessM4Tv2 CodeHiFi-GAN PCC [B={batch} row={b} valid_len={valid_len}]: "
            f"{msg} (threshold {PCC_THRESHOLD})"
        )
        assert ok, f"row {b}: {msg}"

    logger.info(f"SeamlessM4Tv2 CodeHiFi-GAN PCC [B={batch}] min over rows: {pcc_min}")
