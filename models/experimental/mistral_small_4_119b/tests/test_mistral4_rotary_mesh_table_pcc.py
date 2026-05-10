# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""PCC: device RoPE table gather vs one-shot HF cos/sin upload (short ``seq``)."""

import pytest
import torch
import ttnn
from loguru import logger

from models.common.auto_compose import to_torch_auto_compose
from models.common.utility_functions import comp_allclose, comp_pcc, run_for_wormhole_b0_or_blackhole
from models.experimental.mistral_small_4_119b.constants import HF_MODEL_ID
from models.experimental.mistral_small_4_119b.tests.mesh_param import mesh_device_request_param
from models.experimental.mistral_small_4_119b.tt.mistral4_rotary_mesh_table import TtMistral4RotaryEmbeddingMeshTable
from models.experimental.mistral_small_4_119b.tt.mistral4_self_attention import upload_mistral4_rotary_cos_sin_to_mesh

pytest.importorskip("transformers")
pytest.importorskip("transformers.models.mistral4.modeling_mistral4", reason="Mistral4 required")


def _text_config_eager_attn():
    from transformers import AutoConfig

    try:
        cfg = AutoConfig.from_pretrained(HF_MODEL_ID, trust_remote_code=True)
    except Exception as exc:
        pytest.skip(f"Could not load HF config: {exc}")
    text = cfg.text_config
    if text is None:
        pytest.skip("Config has no text_config")
    if hasattr(text, "attn_implementation"):
        text.attn_implementation = "eager"
    if hasattr(text, "_attn_implementation"):
        text._attn_implementation = "eager"
    return text


@torch.no_grad()
@run_for_wormhole_b0_or_blackhole()
@pytest.mark.parametrize("mesh_device", [mesh_device_request_param()], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 30000000, "num_command_queues": 1}],
    indirect=True,
)
@pytest.mark.parametrize("seq_len", (8,))
def test_mistral4_rotary_mesh_table_gather_matches_hf_upload(mesh_device, seq_len):
    from transformers.models.mistral4.modeling_mistral4 import Mistral4RotaryEmbedding

    text = _text_config_eager_attn()
    # Small table keeps CPU HF rotary init fast; positions 0..seq_len-1 stay in-range.
    num_positions = max(512, seq_len + 16)
    torch.manual_seed(1)
    hidden0 = torch.randn(1, seq_len, text.hidden_size, dtype=torch.bfloat16)
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    rotary = Mistral4RotaryEmbedding(text).eval().to(torch.bfloat16)
    position_embeddings = rotary(hidden0, position_ids)

    cos_u, sin_u = upload_mistral4_rotary_cos_sin_to_mesh(mesh_device, position_embeddings)
    try:
        table = TtMistral4RotaryEmbeddingMeshTable(mesh_device, text, num_positions=num_positions)
        cos_g, sin_g = table.gather(position_ids)
        try:
            ref_c = to_torch_auto_compose(cos_u, device=mesh_device).squeeze(1)
            got_c = to_torch_auto_compose(cos_g, device=mesh_device).squeeze(1)
            ref_s = to_torch_auto_compose(sin_u, device=mesh_device).squeeze(1)
            got_s = to_torch_auto_compose(sin_g, device=mesh_device).squeeze(1)
        finally:
            ttnn.deallocate(cos_g)
            ttnn.deallocate(sin_g)
    finally:
        ttnn.deallocate(cos_u)
        ttnn.deallocate(sin_u)

    pc, msg_c = comp_pcc(ref_c[:, :seq_len, :], got_c[:, :seq_len, :], pcc=0.99)
    ps, msg_s = comp_pcc(ref_s[:, :seq_len, :], got_s[:, :seq_len, :], pcc=0.99)
    logger.info(comp_allclose(ref_c[:, :seq_len, :], got_c[:, :seq_len, :]))
    logger.info(f"rotary cos PCC: {msg_c}; sin PCC: {msg_s}")
    assert pc, f"cos PCC below 0.99: {msg_c}"
    assert ps, f"sin PCC below 0.99: {msg_s}"
