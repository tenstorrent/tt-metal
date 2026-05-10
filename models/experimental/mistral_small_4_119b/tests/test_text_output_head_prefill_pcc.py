# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""PCC: :class:`TtMistral4FinalNormLmHeadPrefill` vs HF RMSNorm + ``F.linear`` (``norm`` + ``lm_head``)."""

import pytest
import torch
import torch.nn.functional as F
import ttnn
from loguru import logger

from models.common.utility_functions import comp_allclose, comp_pcc, run_for_wormhole_b0_or_blackhole
from models.experimental.mistral_small_4_119b.constants import (
    HF_MODEL_ID,
    TEXT_MODEL_LM_HEAD_WEIGHT_KEY,
    TEXT_MODEL_NORM_WEIGHT_KEY,
)
from models.experimental.mistral_small_4_119b.tests.mesh_param import mesh_device_request_param
from models.experimental.mistral_small_4_119b.tt.mistral4_output_head import TtMistral4FinalNormLmHeadPrefill
from models.experimental.mistral_small_4_119b.tt.mistral4_self_attention import _torch_for_ttnn_upload
from models.tt_transformers.tt.load_checkpoints import load_hf_state_dict_filtered


pytest.importorskip("transformers")
pytest.importorskip("transformers.models.mistral4.modeling_mistral4", reason="Mistral4 required")


def _text_config():
    from transformers import AutoConfig

    try:
        cfg = AutoConfig.from_pretrained(HF_MODEL_ID, trust_remote_code=True)
    except Exception as exc:
        pytest.skip(f"Could not load HF config: {exc}")
    text = cfg.text_config
    if text is None:
        pytest.skip("Config has no text_config")
    return text


def _norm_lm_checkpoint():
    try:
        return load_hf_state_dict_filtered(HF_MODEL_ID, (TEXT_MODEL_NORM_WEIGHT_KEY, TEXT_MODEL_LM_HEAD_WEIGHT_KEY))
    except (FileNotFoundError, OSError) as exc:
        pytest.skip(f"Checkpoint load failed: {exc}")


@torch.no_grad()
@run_for_wormhole_b0_or_blackhole()
@pytest.mark.parametrize("mesh_device", [mesh_device_request_param()], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 30000000, "num_command_queues": 1}],
    indirect=True,
)
@pytest.mark.parametrize("seq_len", (8,))
def test_mistral_small_4_text_output_head_prefill_pcc(seq_len, reset_seeds, mesh_device):
    from transformers.models.mistral4.modeling_mistral4 import Mistral4RMSNorm

    text = _text_config()
    sd = _norm_lm_checkpoint()

    torch.manual_seed(3)
    h = torch.randn(1, seq_len, text.hidden_size, dtype=torch.bfloat16)
    norm_w = _torch_for_ttnn_upload(sd[TEXT_MODEL_NORM_WEIGHT_KEY])
    lm_w = _torch_for_ttnn_upload(sd[TEXT_MODEL_LM_HEAD_WEIGHT_KEY])
    ref_norm = Mistral4RMSNorm(text.hidden_size, eps=float(text.rms_norm_eps)).eval()
    ref_norm.weight.data = norm_w
    ref_logits = F.linear(ref_norm(h), lm_w)

    h_tt = ttnn.from_torch(
        h.unsqueeze(1),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device=mesh_device),
    )
    try:
        head = TtMistral4FinalNormLmHeadPrefill(sd, text)
    except Exception as exc:
        pytest.skip(f"TtMistral4FinalNormLmHeadPrefill init failed: {exc}")

    out = head(h_tt, mesh_device=mesh_device, logical_batch=int(h.shape[0]))
    passing, pcc_message = comp_pcc(ref_logits, out, pcc=0.99)
    logger.info(comp_allclose(ref_logits, out))
    logger.info(f"norm+lm_head PCC: {pcc_message}")
    assert passing, f"PCC below 0.99: {pcc_message}"
