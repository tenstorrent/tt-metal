# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""PCC: TTNN ``TtMistral4SelfAttentionPrefill`` vs HF ``Mistral4Attention`` (random init, short S).

Covers the full prefill path on device (including RoPE, Llama-4 Q scaling, and SDPA);
only rotary cos/sin and ``position_ids`` for scaling are supplied from the host.
"""

import os

import pytest
import torch
import torch.nn as nn
import ttnn
from loguru import logger

from models.common.auto_compose import to_torch_auto_compose
from models.common.utility_functions import comp_allclose, comp_pcc, run_for_wormhole_b0_or_blackhole
from models.experimental.mistral_small_4_119b.constants import HF_MODEL_ID
from models.experimental.mistral_small_4_119b.tt.mistral4_self_attention import (
    TtMistral4SelfAttentionPrefill,
    upload_mistral4_rotary_cos_sin_to_mesh,
)


pytest.importorskip("transformers")
pytest.importorskip(
    "transformers.models.mistral4.modeling_mistral4", reason="Mistral4 attention requires recent transformers"
)


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


def _init_mistral4_attention_weights(module: nn.Module) -> None:
    from transformers.models.mistral4.modeling_mistral4 import Mistral4RMSNorm

    with torch.no_grad():
        for m in module.modules():
            if isinstance(m, Mistral4RMSNorm):
                nn.init.ones_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


@torch.no_grad()
@run_for_wormhole_b0_or_blackhole()
@pytest.mark.parametrize(
    "device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize("seq_len", (8,))
def test_mistral_small_4_text_layer0_self_attn_prefill_pcc(seq_len, reset_seeds, device):
    from transformers.models.mistral4.modeling_mistral4 import Mistral4Attention, Mistral4RotaryEmbedding

    text = _text_config_eager_attn()
    hf_attn = Mistral4Attention(text, layer_idx=0).eval()
    _init_mistral4_attention_weights(hf_attn)
    rotary = Mistral4RotaryEmbedding(text).eval()
    # Default HF modules use float32 parameters; activations are bf16 — align dtypes for F.linear.
    hf_attn = hf_attn.to(torch.bfloat16)
    rotary = rotary.to(torch.bfloat16)

    torch.manual_seed(0)
    x = torch.randn(1, seq_len, text.hidden_size, dtype=torch.bfloat16)
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    position_embeddings = rotary(x, position_ids)

    y_hf, _ = hf_attn(
        x,
        attention_mask=None,
        position_ids=position_ids,
        past_key_values=None,
        use_cache=False,
        position_embeddings=position_embeddings,
    )

    tt_attn = TtMistral4SelfAttentionPrefill(device, text, hf_attn.state_dict())
    x_tt = ttnn.from_torch(
        x.unsqueeze(1),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device=device),
    )
    y_tt = tt_attn(x_tt, position_ids=position_ids, position_embeddings=position_embeddings)
    y_tt_torch = to_torch_auto_compose(y_tt, device=device)[:, :, :seq_len, :]
    y_tt_torch = y_tt_torch.squeeze(1)

    passing, pcc_message = comp_pcc(y_hf, y_tt_torch, pcc=0.92)
    logger.info(comp_allclose(y_hf, y_tt_torch))
    logger.info(f"self_attn prefill PCC: {pcc_message}")

    assert passing, f"self_attn PCC below 0.92: {pcc_message}"

    x_tt2 = ttnn.from_torch(
        x.unsqueeze(1),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device=device),
    )
    cos_tt, sin_tt = upload_mistral4_rotary_cos_sin_to_mesh(device, position_embeddings)
    try:
        y_tt_shared = tt_attn(
            x_tt2,
            position_ids=position_ids,
            position_embeddings_tt=(cos_tt, sin_tt),
        )
    finally:
        ttnn.deallocate(cos_tt)
        ttnn.deallocate(sin_tt)
    y_shared = to_torch_auto_compose(y_tt_shared, device=device)[:, :, :seq_len, :].squeeze(1)
    passing_shared, msg_shared = comp_pcc(y_hf, y_shared, pcc=0.92)
    logger.info(f"self_attn prefill PCC (device cos/sin): {msg_shared}")
    assert passing_shared, f"PCC below 0.92 with position_embeddings_tt: {msg_shared}"
