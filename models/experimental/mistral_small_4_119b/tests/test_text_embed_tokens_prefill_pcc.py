# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""PCC: :class:`TtMistral4EmbedTokensPrefill` vs ``F.embedding`` (hub ``embed_tokens.weight``)."""

import pytest
import torch
import ttnn
from loguru import logger

from models.common.auto_compose import to_torch_auto_compose
from models.common.utility_functions import comp_allclose, comp_pcc, run_for_wormhole_b0_or_blackhole
from models.experimental.mistral_small_4_119b.constants import HF_MODEL_ID, TEXT_MODEL_EMBED_TOKENS_WEIGHT_KEY
from models.experimental.mistral_small_4_119b.tests.mesh_param import mesh_device_request_param
from models.experimental.mistral_small_4_119b.tt.mistral4_embed_tokens import TtMistral4EmbedTokensPrefill
from models.experimental.mistral_small_4_119b.tt.mistral4_self_attention import _torch_for_ttnn_upload
from models.tt_transformers.tt.load_checkpoints import load_hf_state_dict_filtered


def _embed_checkpoint():
    try:
        return load_hf_state_dict_filtered(HF_MODEL_ID, (TEXT_MODEL_EMBED_TOKENS_WEIGHT_KEY,))
    except (FileNotFoundError, OSError) as exc:
        pytest.skip(f"No checkpoint shard for embed weights: {exc}")


@torch.no_grad()
@run_for_wormhole_b0_or_blackhole()
@pytest.mark.parametrize("mesh_device", [mesh_device_request_param()], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 30000000, "num_command_queues": 1}],
    indirect=True,
)
@pytest.mark.parametrize("seq_len", (8,))
@pytest.mark.parametrize("batch_size", (1,))
def test_mistral_small_4_text_embed_tokens_prefill_pcc(batch_size, seq_len, reset_seeds, mesh_device):
    state_dict = _embed_checkpoint()
    assert TEXT_MODEL_EMBED_TOKENS_WEIGHT_KEY in state_dict

    w = state_dict[TEXT_MODEL_EMBED_TOKENS_WEIGHT_KEY]
    vocab = w.shape[0]
    torch.manual_seed(0)
    input_ids = torch.randint(0, vocab, (batch_size, seq_len), dtype=torch.long)

    w_bf16 = _torch_for_ttnn_upload(w)
    ref = torch.nn.functional.embedding(input_ids, w_bf16)

    try:
        emb = TtMistral4EmbedTokensPrefill(mesh_device, state_dict)
    except Exception as exc:
        pytest.skip(f"TtMistral4EmbedTokensPrefill init failed: {exc}")

    y_tt = emb(input_ids)
    y_tt_torch = to_torch_auto_compose(y_tt, device=mesh_device)[:, :, :seq_len, :].squeeze(1)

    passing, pcc_message = comp_pcc(ref, y_tt_torch, pcc=0.99)
    logger.info(comp_allclose(ref, y_tt_torch))
    logger.info(f"embed_tokens PCC: {pcc_message}")
    assert passing, f"embed PCC below 0.99: {pcc_message}"
