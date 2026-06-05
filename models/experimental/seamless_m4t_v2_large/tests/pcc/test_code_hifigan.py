# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""CodeHiFi-GAN vocoder PCC at maximum unit sequence length (1024 tokens).

Real weights only — skipped when download fails.
"""

import pytest
import torch
from loguru import logger

from tests.ttnn.utils_for_testing import check_with_pcc

from models.experimental.seamless_m4t_v2_large.reference.torch_code_hifigan import (
    forward_torch_code_hifigan_reference,
    load_pretrained_code_hifigan,
)
from models.experimental.seamless_m4t_v2_large.scripts.download_weights import ensure_seamless_m4t_v2_large_weights
from models.experimental.seamless_m4t_v2_large.tt.common import to_torch_replicated_first_shard
from models.experimental.seamless_m4t_v2_large.tt.mesh_helpers import (
    MESH_DEVICE_PARAMETRIZE_TEXT,
    from_torch_uint32_rm,
    mesh_default_device,
)
from models.experimental.seamless_m4t_v2_large.tt.model_preprocessing import create_code_hifigan_parameters
from models.experimental.seamless_m4t_v2_large.tt.tt_code_hifigan import TTSeamlessM4Tv2CodeHifiGan

PCC_THRESHOLD = 0.99
MAX_UNIT_SEQ = 1024


@pytest.mark.timeout(3600)
@pytest.mark.parametrize(*MESH_DEVICE_PARAMETRIZE_TEXT, indirect=["mesh_device", "device_params"])
def test_seamless_m4t_v2_code_hifigan_max_unit_seq_pcc(mesh_device, device_params, reset_seeds):
    """CodeHiFi-GAN PCC ≥ 0.99 at unit_seq=1024 (chunked conv1d on every upsample stage)."""
    _ = reset_seeds
    _ = device_params

    try:
        weights_dir = ensure_seamless_m4t_v2_large_weights()
    except ImportError as e:
        pytest.skip(str(e))
    except Exception as e:
        pytest.skip(f"Could not prepare seamless-m4t-v2-large weights: {e}")

    with mesh_default_device(mesh_device):
        vocoder, cfg = load_pretrained_code_hifigan(weights_dir, dtype=torch.bfloat16)

        torch.manual_seed(0)
        batch = 1
        unit_seq = MAX_UNIT_SEQ
        pad_id = int(cfg.t2u_pad_token_id)
        vocab = int(cfg.unit_hifi_gan_vocab_size)
        low = max(pad_id + 1, 2)
        high = max(low + 1, min(vocab - 1, low + 9999))
        input_ids = torch.randint(low, high, (batch, unit_seq), dtype=torch.int64)
        speaker_id = torch.zeros(batch, 1, dtype=torch.int64)
        lang_id = torch.zeros(batch, 1, dtype=torch.int64)

        ref_wav, ref_lengths = forward_torch_code_hifigan_reference(vocoder, input_ids, speaker_id, lang_id)
        ref_wav = ref_wav.to(torch.bfloat16)
        ref_2d = ref_wav.squeeze(1) if ref_wav.dim() == 3 else ref_wav

        params = create_code_hifigan_parameters(vocoder, device=mesh_device)
        tt_v = TTSeamlessM4Tv2CodeHifiGan(mesh_device, params, cfg)

        input_ids_tt = from_torch_uint32_rm(mesh_device, input_ids)
        sp_tt = from_torch_uint32_rm(mesh_device, speaker_id)
        lang_tt = from_torch_uint32_rm(mesh_device, lang_id)

        out_tt, lengths_tt = tt_v.forward(input_ids_tt, sp_tt, lang_tt)

        tt_wav = to_torch_replicated_first_shard(out_tt).to(torch.bfloat16).squeeze(-1).contiguous()
        lengths_torch = to_torch_replicated_first_shard(lengths_tt).to(torch.long).reshape(-1)
        ref_lengths_t = ref_lengths.reshape(-1).to(torch.long)
        assert torch.allclose(
            lengths_torch, ref_lengths_t
        ), f"lengths mismatch vs HF: tt={lengths_torch.tolist()} ref={ref_lengths_t.tolist()}"

        valid_len = int(lengths_torch[0].item())
        min_len = min(valid_len, ref_2d.shape[-1], tt_wav.shape[-1])
        ok, msg = check_with_pcc(ref_2d[0, :min_len].contiguous(), tt_wav[0, :min_len].contiguous(), pcc=PCC_THRESHOLD)
        logger.info(
            f"SeamlessM4Tv2 CodeHiFi-GAN PCC @ unit_seq={unit_seq} valid_len={valid_len}: "
            f"{msg} (threshold {PCC_THRESHOLD})"
        )
        assert ok, msg
        try:
            mesh_device.clear_program_cache()
        except Exception:
            pass
