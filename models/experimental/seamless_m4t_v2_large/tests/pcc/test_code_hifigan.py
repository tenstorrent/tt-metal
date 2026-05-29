# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Single-shot PCC test for the SeamlessM4Tv2 CodeHiFi-GAN vocoder at its design maximum unit seq.

Vocoder design: the HiFi-GAN itself has no position embeddings — input length is bounded only by
the per-conv1d L1 budget (``_HIFIGAN_MAX_CONV1D_TLEN`` = 4096) above which ``_conv1d`` chunks the
timeline using a wide ``_VOCODER_CONV1D_INTERIOR`` window (3968 — sized just under the L1 ceiling).
Inputs *larger* than ``unit_seq=1024`` keep working via the same chunked conv1d path; this test is
the longest unit-token PCC we run in CI because a 4096-unit input would drive multi-hour HiFi-GAN
runs.

Real weights only — if ``huggingface_hub`` is missing or the snapshot download fails the test is
skipped.
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
from models.experimental.seamless_m4t_v2_large.tests.pcc.prof_capture_limits import CODE_HIFIGAN_UNIT_SEQ
from models.experimental.seamless_m4t_v2_large.tt.tt_code_hifigan import TTSeamlessM4Tv2CodeHifiGan

PCC_THRESHOLD = 0.99
MAX_UNIT_SEQ = 1024
PROF_CAPTURE_UNIT_SEQ = CODE_HIFIGAN_UNIT_SEQ


@pytest.mark.timeout(1800)
@pytest.mark.parametrize(*MESH_DEVICE_PARAMETRIZE_TEXT, indirect=["mesh_device", "device_params"])
def test_seamless_m4t_v2_code_hifigan_prof_capture_unit_seq_pcc(mesh_device, device_params, reset_seeds):
    """CodeHiFi-GAN PCC ≥ 0.99 at the tracy-safe unit length (``PROF_CAPTURE_UNIT_SEQ`` = 128).

    Highest power-of-two ``unit_seq`` where ``python3 -m tracy -r -v`` and PCC both pass on BH 1×4.
    ``unit_seq=256`` overflows Tracy's 32K source locations; ``test_*_max_unit_seq_pcc`` at 1024
    also breaks capture even though the forward completes.
    """
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
        unit_seq = PROF_CAPTURE_UNIT_SEQ
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
            f"SeamlessM4Tv2 CodeHiFi-GAN prof-capture PCC @ unit_seq={unit_seq} valid_len={valid_len}: "
            f"{msg} (threshold {PCC_THRESHOLD})"
        )
        assert ok, msg


@pytest.mark.timeout(3600)
@pytest.mark.parametrize(*MESH_DEVICE_PARAMETRIZE_TEXT, indirect=["mesh_device", "device_params"])
def test_seamless_m4t_v2_code_hifigan_max_unit_seq_pcc(mesh_device, device_params, reset_seeds):
    """CodeHiFi-GAN PCC ≥ 0.99 at ``unit_seq=1024`` — exercises chunked conv1d on every upsample stage.

    Longer inputs are designed to work via the same chunked path; this PCC bar covers the conv1d
    chunking + DRAM-sliced ConvTranspose + chunked expand matmul end-to-end against HF.
    """
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
