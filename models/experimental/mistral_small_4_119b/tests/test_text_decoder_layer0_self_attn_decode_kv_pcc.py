# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""PCC: prefill + one decode step with KV vs HF ``Mistral4Attention`` + ``DynamicCache``."""

import os

import pytest
import torch
import torch.nn as nn
import ttnn
from loguru import logger

from models.common.auto_compose import to_torch_auto_compose
from models.common.utility_functions import comp_allclose, comp_pcc, run_for_wormhole_b0_or_blackhole
from models.experimental.mistral_small_4_119b.constants import HF_MODEL_ID
from models.experimental.mistral_small_4_119b.tt.mistral4_self_attention import TtMistral4SelfAttentionPrefill

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
@pytest.mark.parametrize("prefill_len", (4,))
def test_mistral_small_4_text_layer0_self_attn_prefill_then_decode_kv_pcc(prefill_len, reset_seeds, device):
    from transformers.cache_utils import DynamicCache
    from transformers.models.mistral4.modeling_mistral4 import Mistral4Attention, Mistral4RotaryEmbedding

    text = _text_config_eager_attn()
    hf_attn = Mistral4Attention(text, layer_idx=0).eval()
    _init_mistral4_attention_weights(hf_attn)
    rotary = Mistral4RotaryEmbedding(text).eval()
    hf_attn = hf_attn.to(torch.bfloat16)
    rotary = rotary.to(torch.bfloat16)

    torch.manual_seed(0)
    h = text.hidden_size
    x_pref = torch.randn(1, prefill_len, h, dtype=torch.bfloat16)
    pos_pref = torch.arange(prefill_len, dtype=torch.long).unsqueeze(0)
    pe_pref = rotary(x_pref, pos_pref)

    cache = DynamicCache()
    y_pref_hf, _ = hf_attn(
        x_pref,
        pe_pref,
        None,
        pos_pref,
        past_key_values=cache,
    )

    x_dec = torch.randn(1, 1, h, dtype=torch.bfloat16)
    pos_dec = torch.tensor([[prefill_len]], dtype=torch.long)
    pe_dec = rotary(x_dec, pos_dec)
    y_dec_hf, _ = hf_attn(
        x_dec,
        pe_dec,
        None,
        pos_dec,
        past_key_values=cache,
    )

    tt_attn = TtMistral4SelfAttentionPrefill(device, text, hf_attn.state_dict())
    x_pref_tt = ttnn.from_torch(
        x_pref.unsqueeze(1),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device=device),
    )
    out_pref_tt, kv_k, kv_v = tt_attn.forward_prefill_with_kv(
        x_pref_tt,
        position_ids=pos_pref,
        position_embeddings=pe_pref,
    )
    y_pref_tt = to_torch_auto_compose(out_pref_tt, device=device)[:, :, :prefill_len, :].squeeze(1)

    passing_pref, msg_pref = comp_pcc(y_pref_hf, y_pref_tt, pcc=0.92)
    logger.info(comp_allclose(y_pref_hf, y_pref_tt))
    logger.info(f"prefill PCC: {msg_pref}")
    assert passing_pref, f"prefill PCC below 0.92: {msg_pref}"
    ttnn.deallocate(out_pref_tt)

    x_dec_tt = ttnn.from_torch(
        x_dec.unsqueeze(1),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device=device),
    )
    out_dec_tt, kv_k2, kv_v2 = tt_attn.forward_decode_extend_kv(
        x_dec_tt,
        position_ids=pos_dec,
        past_key_states=kv_k,
        past_value_states=kv_v,
        position_embeddings=pe_dec,
    )
    ttnn.deallocate(kv_k)
    ttnn.deallocate(kv_v)

    y_dec_tt = to_torch_auto_compose(out_dec_tt, device=device)[:, :, :1, :].squeeze(1)

    passing_dec, msg_dec = comp_pcc(y_dec_hf, y_dec_tt, pcc=0.92)
    logger.info(comp_allclose(y_dec_hf, y_dec_tt))
    logger.info(f"decode PCC: {msg_dec}")
    ttnn.deallocate(out_dec_tt)
    ttnn.deallocate(kv_k2)
    ttnn.deallocate(kv_v2)
    assert passing_dec, f"decode PCC below 0.92: {msg_dec}"
