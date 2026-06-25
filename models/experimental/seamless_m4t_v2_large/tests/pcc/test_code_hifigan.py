# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""CodeHiFi-GAN vocoder PCC at maximum unit sequence length (1024 tokens).

Real weights only — skipped when download fails.
"""

import os

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
from models.experimental.seamless_m4t_v2_large.tt.common import to_torch_replicated_first_shard
from models.experimental.seamless_m4t_v2_large.tt.mesh_helpers import (
    from_torch_uint32_rm,
    mesh_default_device,
)
from models.experimental.seamless_m4t_v2_large.tt.model_preprocessing import create_code_hifigan_parameters
from models.experimental.seamless_m4t_v2_large.tt.tt_code_hifigan import TTSeamlessM4Tv2CodeHifiGan

PCC_THRESHOLD = 0.99
MAX_UNIT_SEQ = 1024


def _mesh_device_param():
    mesh_env = os.environ.get("MESH_DEVICE")
    if mesh_env in {"P150": (1, 1), "BH-QB": (1, 4)}:
        return {"P150": (1, 1), "BH-QB": (1, 4)}[mesh_env]
    if "TT_MESH_WIDTH" in os.environ:
        return int(os.environ["TT_MESH_WIDTH"])
    try:
        return (1, 4) if ttnn.get_num_devices() >= 4 else (1, 1)
    except Exception:
        return (1, 1)


def _device_params():
    mesh_param = _mesh_device_param()
    params = {"l1_small_size": 32768, "num_command_queues": 2}
    if mesh_param != (1, 1) and mesh_param != 1:
        params["fabric_config"] = ttnn.FabricConfig.FABRIC_1D
    return params


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("mesh_device", [_mesh_device_param()], indirect=True)
@pytest.mark.parametrize("device_params", [_device_params()], indirect=True)
def test_seamless_m4t_v2_code_hifigan_max_unit_seq_pcc(mesh_device, device_params, reset_seeds):
    """CodeHiFi-GAN PCC ≥ 0.99 at unit_seq=1024 (chunked conv1d on every upsample stage)."""
    _ = reset_seeds
    _ = device_params

    try:
        weights_dir = ensure_seamless_m4t_v2_large_weights()
    except ImportError as e:
        raise pytest.skip.Exception(str(e))
    except Exception as e:
        raise pytest.skip.Exception(f"Could not prepare seamless-m4t-v2-large weights: {e}")

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
        except Exception as clear_err:
            # Program-cache clear is optional post-test cleanup.
            logger.warning("clear_program_cache failed (ignored): %s", clear_err)
