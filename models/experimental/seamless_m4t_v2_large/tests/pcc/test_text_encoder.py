# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger

from tests.ttnn.utils_for_testing import check_with_pcc

from models.experimental.seamless_m4t_v2_large.reference.torch_text_encoder import (
    forward_torch_reference,
    load_pretrained_text_encoder,
)
from models.experimental.seamless_m4t_v2_large.scripts.download_weights import ensure_seamless_m4t_v2_large_weights
from models.experimental.seamless_m4t_v2_large.tt.common import to_torch_replicated_first_shard
from models.experimental.seamless_m4t_v2_large.tt.mesh_helpers import (
    MESH_DEVICE_PARAMETRIZE_TEXT,
    from_torch_uint32_rm,
    mesh_default_device,
)
from models.experimental.seamless_m4t_v2_large.tt.model_preprocessing import create_text_encoder_parameters
from models.experimental.seamless_m4t_v2_large.tt.tt_text_encoder import TTSeamlessM4Tv2Encoder

PCC_THRESHOLD = 0.99


def _create_position_ids_from_input_ids(
    input_ids: torch.Tensor, padding_idx: int, past_key_values_length: int = 0
) -> torch.Tensor:
    """
    Copy of HF helper used by SeamlessM4Tv2 positional embedding.
    """
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
    return incremental_indices.long() + padding_idx


def _run_text_encoder_pcc(device, *, seq: int = 32) -> None:
    """Shared PCC body; mesh-safe readback via ``to_torch_replicated_first_shard``.

    ``seq`` parametrises the input length so long-sequence regressions can be checked against the
    HF ``max_position_embeddings = 4096`` upper bound (the sinusoidal table is sized for that).
    """
    try:
        weights_dir = ensure_seamless_m4t_v2_large_weights()
    except ImportError as e:
        pytest.skip(str(e))
    except Exception as e:
        pytest.skip(f"Could not prepare seamless-m4t-v2-large weights: {e}")

    torch.manual_seed(0)
    encoder, cfg = load_pretrained_text_encoder(weights_dir, dtype=torch.bfloat16)

    batch = 1
    input_ids = torch.randint(1, min(cfg.vocab_size - 1, 2**31 - 1), (batch, seq), dtype=torch.int64)
    attn_mask = torch.ones(batch, seq, dtype=torch.long)

    with torch.no_grad():
        position_ids = _create_position_ids_from_input_ids(input_ids, cfg.pad_token_id, past_key_values_length=0)

    ref = forward_torch_reference(encoder, input_ids, attn_mask).to(torch.bfloat16)

    params = create_text_encoder_parameters(encoder, device=device)
    tt_enc = TTSeamlessM4Tv2Encoder(
        device,
        params,
        layer_norm_eps=cfg.layer_norm_eps,
        num_hidden_layers=cfg.encoder_layers,
        num_attention_heads=cfg.encoder_attention_heads,
        hidden_size=cfg.hidden_size,
    )

    input_ids_tt = from_torch_uint32_rm(device, input_ids)
    position_ids_tt = from_torch_uint32_rm(device, position_ids)
    # All-ones ``attn_mask`` → no additive SDPA mask needed (same as E2E when ``attention_mask`` is omitted).
    assert attn_mask.min() == 1, "PCC test uses a fully valid attention mask"

    out_tt = tt_enc.forward(input_ids_tt, position_ids_tt, attention_mask=None)
    tt_cpu = (
        to_torch_replicated_first_shard(out_tt).to(torch.bfloat16).reshape(batch, seq, cfg.hidden_size).contiguous()
    )

    ok, msg = check_with_pcc(ref, tt_cpu, pcc=PCC_THRESHOLD)
    logger.info(f"SeamlessM4Tv2 text encoder PCC: {msg} (threshold {PCC_THRESHOLD})")
    if ok:
        logger.info("SeamlessM4Tv2 text encoder PCC check passed.")
    else:
        logger.warning("SeamlessM4Tv2 text encoder PCC check failed.")

    assert ok, msg


@pytest.mark.parametrize(*MESH_DEVICE_PARAMETRIZE_TEXT, indirect=["mesh_device", "device_params"])
def test_seamless_m4t_v2_text_encoder_pcc(mesh_device, device_params, reset_seeds):
    _ = reset_seeds
    _ = device_params
    with mesh_default_device(mesh_device):
        _run_text_encoder_pcc(mesh_device)


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("seq", [128, 4096], ids=["seq128", "seq4096"])
@pytest.mark.parametrize(*MESH_DEVICE_PARAMETRIZE_TEXT, indirect=["mesh_device", "device_params"])
def test_seamless_m4t_v2_text_encoder_max_seq_len_pcc(mesh_device, device_params, reset_seeds, seq):
    """PCC at long sequences, capped at HF ``max_position_embeddings = 4096``.

    Exercises the chunked DRAM-sharded matmul path in ``TTSeamlessM4Tv2Encoder._linear`` —
    the kernel is hard-coded to ``M == TILE`` (per_core_M=1), so long-seq prefill runs the
    matmul ``ceil(m_actual / TILE)`` times per call and concatenates the results. PCC is
    preserved because each chunk uses the same kernel as the short-seq fast path.

    ``seq=128`` is a small-multi-chunk smoke case; ``seq=4096`` covers HF's upper bound and
    exercises the long-seq DRAM activation path (FC1 intermediate would be 64 MB in L1).
    """
    _ = reset_seeds
    _ = device_params
    with mesh_default_device(mesh_device):
        _run_text_encoder_pcc(mesh_device, seq=seq)
