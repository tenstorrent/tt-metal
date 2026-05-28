# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Single-shot PCC test for the SeamlessM4Tv2 text encoder at its maximum designed sequence length.

Encoder design max = ``max_position_embeddings = 4096`` (HF NLLB-style sinusoidal positions).
A single test at ``seq=4096`` exercises the chunked DRAM-sharded matmul path and the long-seq
activation-in-DRAM policy that kick in above ``MATMUL_1D_SEQ_THRESHOLD`` (128).

Inputs are derived from the real downloaded weights (no synthetic weights). If
``huggingface_hub`` is missing or the snapshot download fails the test is skipped — no fallback,
per the project policy "real weights or skip".
"""

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
MAX_SEQ = 4096  # HF ``max_position_embeddings``


def _create_position_ids_from_input_ids(input_ids: torch.Tensor, padding_idx: int) -> torch.Tensor:
    """HF helper used by SeamlessM4Tv2 positional embedding."""
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = torch.cumsum(mask, dim=1).type_as(mask) * mask
    return incremental_indices.long() + padding_idx


@pytest.mark.timeout(1800)
@pytest.mark.parametrize(*MESH_DEVICE_PARAMETRIZE_TEXT, indirect=["mesh_device", "device_params"])
def test_seamless_m4t_v2_text_encoder_max_seq_pcc(mesh_device, device_params, reset_seeds):
    """Text encoder PCC ≥ 0.99 at the HF maximum sequence length (``max_position_embeddings`` = 4096).

    Exercises every long-sequence code path: chunked DRAM-sharded matmul, DRAM activations, and the
    sinusoidal position embedding at the full position table extent.
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
        torch.manual_seed(0)
        encoder, cfg = load_pretrained_text_encoder(weights_dir, dtype=torch.bfloat16)

        batch = 1
        seq = MAX_SEQ
        input_ids = torch.randint(1, min(cfg.vocab_size - 1, 2**31 - 1), (batch, seq), dtype=torch.int64)
        attn_mask = torch.ones(batch, seq, dtype=torch.long)

        with torch.no_grad():
            position_ids = _create_position_ids_from_input_ids(input_ids, cfg.pad_token_id)
            ref = forward_torch_reference(encoder, input_ids, attn_mask).to(torch.bfloat16)

        params = create_text_encoder_parameters(encoder, device=mesh_device)
        tt_enc = TTSeamlessM4Tv2Encoder(
            mesh_device,
            params,
            layer_norm_eps=cfg.layer_norm_eps,
            num_hidden_layers=cfg.encoder_layers,
            num_attention_heads=cfg.encoder_attention_heads,
            hidden_size=cfg.hidden_size,
        )

        input_ids_tt = from_torch_uint32_rm(mesh_device, input_ids)
        position_ids_tt = from_torch_uint32_rm(mesh_device, position_ids)

        out_tt = tt_enc.forward(input_ids_tt, position_ids_tt, attention_mask=None)
        tt_cpu = (
            to_torch_replicated_first_shard(out_tt).to(torch.bfloat16).reshape(batch, seq, cfg.hidden_size).contiguous()
        )

        ok, msg = check_with_pcc(ref, tt_cpu, pcc=PCC_THRESHOLD)
        logger.info(f"SeamlessM4Tv2 text encoder PCC @ seq={seq}: {msg} (threshold {PCC_THRESHOLD})")
        assert ok, msg
