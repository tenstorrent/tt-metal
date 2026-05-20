# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

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
    DEVICE_PARAMS_BH_QB_FULL,
    DEVICE_PARAMS_P150_FULL,
    MESH_SHAPE_BH_QB,
    MESH_SHAPE_P150,
    _requires_num_devices,
    from_torch_uint32_rm,
    mesh_default_device,
)
from models.experimental.seamless_m4t_v2_large.tt.model_preprocessing import create_code_hifigan_parameters
from models.experimental.seamless_m4t_v2_large.tt.tt_code_hifigan import TTSeamlessM4Tv2CodeHifiGan

PCC_THRESHOLD = 0.99


def _run_code_hifigan_pcc(device, batch: int) -> None:
    """Shared PCC body. Works on either a single-device 1×1 mesh or a multi-device 1×N mesh.

    Multi-device readbacks of replicated tensors go through ``to_torch_replicated_first_shard``,
    which attaches a ``ConcatMeshToTensor(dim=0)`` composer and slices the first device's copy
    out — required because ``ttnn.to_torch`` errors on a >1-shard tensor without a composer.
    On a 1×1 mesh that helper is a bit-identical pass-through.
    """
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

    input_ids_tt = from_torch_uint32_rm(device, input_ids)
    sp_tt = from_torch_uint32_rm(device, speaker_id)
    lang_tt = from_torch_uint32_rm(device, lang_id)

    out_tt, lengths_tt = tt_v.forward(input_ids_tt, sp_tt, lang_tt)

    tt_wav = to_torch_replicated_first_shard(out_tt).to(torch.bfloat16).squeeze(-1).contiguous()
    lengths_torch = to_torch_replicated_first_shard(lengths_tt).to(torch.long).reshape(-1)

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


@pytest.mark.parametrize(
    "mesh_device,device_params,batch",
    [
        pytest.param(
            MESH_SHAPE_P150,
            DEVICE_PARAMS_P150_FULL,
            1,
            id="1x1-b1",
            marks=pytest.mark.skipif(_requires_num_devices(1), reason="P150 (1 device)"),
        ),
        pytest.param(
            MESH_SHAPE_P150,
            DEVICE_PARAMS_P150_FULL,
            2,
            id="1x1-b2",
            marks=pytest.mark.skipif(_requires_num_devices(1), reason="P150 (1 device)"),
        ),
        pytest.param(
            MESH_SHAPE_BH_QB,
            DEVICE_PARAMS_BH_QB_FULL,
            1,
            id="1x4-b1",
            marks=pytest.mark.skipif(_requires_num_devices(4), reason="BH QB (4 devices)"),
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_seamless_m4t_v2_code_hifigan_pcc(mesh_device, device_params, batch, reset_seeds):
    _ = reset_seeds
    _ = device_params
    with mesh_default_device(mesh_device):
        _run_code_hifigan_pcc(mesh_device, batch)
