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
from models.experimental.seamless_m4t_v2_large.tt.common import to_torch_replicated_first_shard
from models.experimental.seamless_m4t_v2_large.tt.mesh_helpers import (
    MESH_DEVICE_PARAMETRIZE_TEXT,
    MESH_DEVICE_PARAMETRIZE_VOCODER_TRACE,
    from_torch_uint32_rm,
    mesh_default_device,
)
from models.experimental.seamless_m4t_v2_large.tt.model_preprocessing import create_code_hifigan_parameters
from models.experimental.seamless_m4t_v2_large.tt.tt_code_hifigan import TTSeamlessM4Tv2CodeHifiGan

PCC_THRESHOLD = 0.99
PCC_THRESHOLD_LONG_UNIT = 0.97
SHORT_UNIT_SEQ = 16
# Chunked expand matmul, DRAM-sliced upsample, HiFi-GAN mel chunks (~minutes on 1×4).
LONG_UNIT_SEQ = 512
# Max **unit-token** PCC length. Implementation chunks ``t_audio`` up to ``_HIFIGAN_MAX_CONV1D_TLEN``
# (4096); a 4096-token PCC run sums thousands of mel frames and is not practical in CI.
MAX_UNIT_SEQ = 1024


def _synthetic_unit_inputs(
    cfg, *, batch: int, unit_seq: int, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    pad_id = int(cfg.t2u_pad_token_id)
    vocab = int(cfg.unit_hifi_gan_vocab_size)
    low = max(pad_id + 1, 2)
    high = max(low + 1, min(vocab - 1, low + 9999))
    input_ids = torch.randint(low, high, (batch, unit_seq), dtype=torch.int64)
    speaker_id = torch.zeros(batch, 1, dtype=torch.int64)
    lang_id = torch.zeros(batch, 1, dtype=torch.int64)
    return input_ids, speaker_id, lang_id


def _run_code_hifigan_pcc(
    device,
    batch: int,
    *,
    unit_seq: int,
    pcc_threshold: float = PCC_THRESHOLD,
) -> None:
    """Shared PCC body. Works on either a single-device 1×1 mesh or a multi-device 1×N mesh."""
    try:
        weights_dir = ensure_seamless_m4t_v2_large_weights()
    except ImportError as e:
        pytest.skip(str(e))
    except Exception as e:
        pytest.skip(f"Could not prepare seamless-m4t-v2-large weights: {e}")

    vocoder, cfg = load_pretrained_code_hifigan(weights_dir, dtype=torch.bfloat16)
    input_ids, speaker_id, lang_id = _synthetic_unit_inputs(cfg, batch=batch, unit_seq=unit_seq)

    ref_wav, ref_lengths = forward_torch_code_hifigan_reference(vocoder, input_ids, speaker_id, lang_id)
    ref_wav = ref_wav.to(torch.bfloat16)

    params = create_code_hifigan_parameters(vocoder, device=device)
    tt_v = TTSeamlessM4Tv2CodeHifiGan(device, params, cfg)

    input_ids_tt = from_torch_uint32_rm(device, input_ids)
    sp_tt = from_torch_uint32_rm(device, speaker_id)
    lang_tt = from_torch_uint32_rm(device, lang_id)

    out_tt, lengths_tt = tt_v.forward(input_ids_tt, sp_tt, lang_tt)

    tt_wav = to_torch_replicated_first_shard(out_tt).to(torch.bfloat16).squeeze(-1).contiguous()
    lengths_torch = to_torch_replicated_first_shard(lengths_tt).to(torch.long).reshape(-1)

    if ref_wav.dim() == 3:
        ref_2d = ref_wav.squeeze(1)
    else:
        ref_2d = ref_wav
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
        ok, msg = check_with_pcc(ref_b, tt_b, pcc=pcc_threshold)
        try:
            pcc_val = float(msg)
        except (TypeError, ValueError):
            pcc_val = -1.0
        pcc_min = min(pcc_min, pcc_val) if pcc_val >= 0 else pcc_min
        logger.info(
            f"SeamlessM4Tv2 CodeHiFi-GAN PCC (unit_seq={unit_seq}) [B={batch} row={b} valid_len={valid_len}]: "
            f"{msg} (threshold {pcc_threshold})"
        )
        assert ok, f"row {b}: {msg}"

    logger.info(f"SeamlessM4Tv2 CodeHiFi-GAN PCC (unit_seq={unit_seq}) [B={batch}] min over rows: {pcc_min}")


def _run_code_hifigan_trace_pcc(
    device,
    *,
    unit_seq: int,
    pcc_threshold: float = PCC_THRESHOLD,
) -> None:
    """PCC for vocoder forward via Metal trace replay (2CQ device params, execute on CQ0).

    One compile ``forward`` for PCC + ``_last_t_audio``, then trace capture/replay (no extra compile forward).
    """
    try:
        weights_dir = ensure_seamless_m4t_v2_large_weights()
    except ImportError as e:
        pytest.skip(str(e))
    except Exception as e:
        pytest.skip(f"Could not prepare seamless-m4t-v2-large weights: {e}")

    vocoder, cfg = load_pretrained_code_hifigan(weights_dir, dtype=torch.bfloat16)
    input_ids, speaker_id, lang_id = _synthetic_unit_inputs(cfg, batch=1, unit_seq=unit_seq)

    ref_wav, ref_lengths = forward_torch_code_hifigan_reference(vocoder, input_ids, speaker_id, lang_id)
    ref_wav = ref_wav.to(torch.bfloat16)

    params = create_code_hifigan_parameters(vocoder, device=device)
    tt_v = TTSeamlessM4Tv2CodeHifiGan(device, params, cfg)

    input_ids_tt = from_torch_uint32_rm(device, input_ids)
    sp_tt = from_torch_uint32_rm(device, speaker_id)
    lang_tt = from_torch_uint32_rm(device, lang_id)

    # Compile path + PCC reference for trace inputs (sets ``_last_t_audio``).
    compile_wav, compile_len = tt_v.forward(input_ids_tt, sp_tt, lang_tt)
    ttnn.deallocate(compile_wav)
    ttnn.deallocate(compile_len)

    try:
        tt_v.capture_forward_trace(input_ids_tt, sp_tt, lang_tt, after_compile=True)
        out_tt, lengths_tt = tt_v.execute_forward_trace()
    finally:
        tt_v.release_forward_trace()

    tt_wav = to_torch_replicated_first_shard(out_tt).to(torch.bfloat16).squeeze(-1).contiguous()
    lengths_torch = to_torch_replicated_first_shard(lengths_tt).to(torch.long).reshape(-1)

    if ref_wav.dim() == 3:
        ref_2d = ref_wav.squeeze(1)
    else:
        ref_2d = ref_wav
    ref_lengths_t = ref_lengths.reshape(-1).to(torch.long)

    assert torch.allclose(lengths_torch, ref_lengths_t)

    valid_len = int(lengths_torch[0].item())
    min_len = min(valid_len, ref_2d.shape[-1], tt_wav.shape[-1])
    ok, msg = check_with_pcc(ref_2d[0, :min_len].contiguous(), tt_wav[0, :min_len].contiguous(), pcc=pcc_threshold)
    logger.info(f"SeamlessM4Tv2 CodeHiFi-GAN trace PCC (unit_seq={unit_seq}): {msg} (threshold {pcc_threshold})")
    assert ok, msg


@pytest.mark.parametrize(*MESH_DEVICE_PARAMETRIZE_TEXT, indirect=["mesh_device", "device_params"])
def test_seamless_m4t_v2_code_hifigan_short_seq_pcc(mesh_device, device_params, reset_seeds):
    """PCC at ``unit_seq=16`` — short 1D matmul / single-shot HiFi-GAN path."""
    _ = reset_seeds
    _ = device_params
    with mesh_default_device(mesh_device):
        _run_code_hifigan_pcc(mesh_device, batch=1, unit_seq=SHORT_UNIT_SEQ)


@pytest.mark.parametrize(*MESH_DEVICE_PARAMETRIZE_TEXT, indirect=["mesh_device", "device_params"])
def test_seamless_m4t_v2_code_hifigan_short_seq_b2_pcc(mesh_device, device_params, reset_seeds):
    """PCC at ``unit_seq=16``, ``batch=2`` (P150 only when ``1x1`` is the active mesh)."""
    _ = reset_seeds
    _ = device_params
    if int(mesh_device.shape[0]) * int(mesh_device.shape[1]) > 1:
        pytest.skip("batch=2 PCC is validated on 1x1 only.")
    with mesh_default_device(mesh_device):
        _run_code_hifigan_pcc(mesh_device, batch=2, unit_seq=SHORT_UNIT_SEQ)


@pytest.mark.timeout(1800)
@pytest.mark.parametrize(*MESH_DEVICE_PARAMETRIZE_TEXT, indirect=["mesh_device", "device_params"])
def test_seamless_m4t_v2_code_hifigan_long_seq_pcc(mesh_device, device_params, reset_seeds):
    """PCC at ``unit_seq=512`` — chunked expand matmul, DRAM-sliced upsample, HiFi-GAN mel chunks."""
    _ = reset_seeds
    _ = device_params
    with mesh_default_device(mesh_device):
        _run_code_hifigan_pcc(
            mesh_device,
            batch=1,
            unit_seq=LONG_UNIT_SEQ,
            pcc_threshold=PCC_THRESHOLD_LONG_UNIT,
        )


@pytest.mark.timeout(3600)
@pytest.mark.parametrize(*MESH_DEVICE_PARAMETRIZE_TEXT, indirect=["mesh_device", "device_params"])
def test_seamless_m4t_v2_code_hifigan_max_unit_seq_pcc(mesh_device, device_params, reset_seeds):
    """PCC at ``unit_seq=1024`` — longest practical unit-token PCC (chunked vocoder paths).

    Code supports ``t_audio`` timelines up to ``_HIFIGAN_MAX_CONV1D_TLEN`` (4096) via chunking; a
    4096-token PCC would drive multi-hour HiFi-GAN runs and is not run in CI.
    """
    _ = reset_seeds
    _ = device_params
    with mesh_default_device(mesh_device):
        _run_code_hifigan_pcc(
            mesh_device,
            batch=1,
            unit_seq=MAX_UNIT_SEQ,
            pcc_threshold=PCC_THRESHOLD_LONG_UNIT,
        )


@pytest.mark.parametrize(*MESH_DEVICE_PARAMETRIZE_VOCODER_TRACE, indirect=["mesh_device", "device_params"])
def test_seamless_m4t_v2_code_hifigan_short_seq_trace_pcc(mesh_device, device_params, reset_seeds):
    """Vocoder Metal trace replay PCC at ``unit_seq=16`` (2CQ + trace region device params)."""
    _ = reset_seeds
    _ = device_params
    with mesh_default_device(mesh_device):
        _run_code_hifigan_trace_pcc(mesh_device, unit_seq=SHORT_UNIT_SEQ)
