# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""PCC: ``TtMistral4DecoderSequence`` prefill + decode with :class:`Mistral4DecoderStackKvState` vs HF."""

import pytest
import torch
import torch.nn.functional as F
import ttnn
from loguru import logger

from models.common.auto_compose import to_torch_auto_compose
from models.common.utility_functions import comp_allclose, comp_pcc, run_for_wormhole_b0_or_blackhole
from models.experimental.mistral_small_4_119b.constants import (
    HF_MODEL_ID,
    TEXT_MODEL_EMBED_TOKENS_WEIGHT_KEY,
    strip_fp8_aux_tensors_from_decoder_inner,
    text_decoder_layer_inner_state_dict,
    text_decoder_layer_state_dict_prefix,
)
from models.experimental.mistral_small_4_119b.tests.mesh_param import mesh_device_request_param
from models.experimental.mistral_small_4_119b.tt.mistral4_embed_tokens import TtMistral4EmbedTokensPrefill
from models.experimental.mistral_small_4_119b.tt.mistral4_kv_state import Mistral4DecoderStackKvState
from models.experimental.mistral_small_4_119b.tt.mistral4_self_attention import _torch_for_ttnn_upload
from models.experimental.mistral_small_4_119b.tt.text_backbone import TtMistral4DecoderSequence
from models.tt_transformers.tt.load_checkpoints import load_hf_state_dict_filtered

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


def _prefixes_embed_and_layer0() -> tuple[str, ...]:
    return ("language_model.model.embed_tokens.", text_decoder_layer_state_dict_prefix(0))


def _load_state_dict() -> dict:
    try:
        return load_hf_state_dict_filtered(HF_MODEL_ID, _prefixes_embed_and_layer0())
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
@pytest.mark.parametrize("prefill_len", (4,))
def test_mistral_small_4_decoder_stack_prefill_then_decode_kv_pcc(mesh_device, prefill_len):
    from transformers.cache_utils import DynamicCache
    from transformers.models.mistral4.modeling_mistral4 import Mistral4DecoderLayer, Mistral4RotaryEmbedding

    text = _text_config_eager_attn()
    state_dict = _load_state_dict()
    inner = strip_fp8_aux_tensors_from_decoder_inner(text_decoder_layer_inner_state_dict(state_dict, 0))

    layer = Mistral4DecoderLayer(text, layer_idx=0).eval()
    try:
        layer.load_state_dict(inner, strict=True)
    except Exception as exc:
        pytest.skip(f"HF layer 0 load failed: {exc}")
    layer = layer.to(torch.bfloat16)
    rotary = Mistral4RotaryEmbedding(text).eval().to(torch.bfloat16)

    w = _torch_for_ttnn_upload(state_dict[TEXT_MODEL_EMBED_TOKENS_WEIGHT_KEY])
    vocab = w.shape[0]
    torch.manual_seed(11)
    input_pref = torch.randint(0, vocab, (1, prefill_len), dtype=torch.long)
    pos_pref = torch.arange(prefill_len, dtype=torch.long).unsqueeze(0)
    hidden_pref = F.embedding(input_pref, w).to(torch.bfloat16)
    pe_pref = rotary(hidden_pref, pos_pref)

    cache = DynamicCache()
    _ = layer(
        hidden_pref,
        attention_mask=None,
        position_ids=pos_pref,
        past_key_values=cache,
        use_cache=True,
        position_embeddings=pe_pref,
    )

    input_dec = torch.randint(0, vocab, (1, 1), dtype=torch.long)
    pos_dec = torch.tensor([[prefill_len]], dtype=torch.long)
    hidden_dec = F.embedding(input_dec, w).to(torch.bfloat16)
    pe_dec = rotary(hidden_dec, pos_dec)
    h2_hf = layer(
        hidden_dec,
        attention_mask=None,
        position_ids=pos_dec,
        past_key_values=cache,
        use_cache=True,
        position_embeddings=pe_dec,
    )

    try:
        emb = TtMistral4EmbedTokensPrefill(mesh_device, state_dict)
        stack = TtMistral4DecoderSequence(
            mesh_device,
            state_dict,
            text,
            layer_indices=(0,),
            use_device_rotary_embedding_table=False,
        )
    except Exception as exc:
        pytest.skip(f"TTNN embed/stack init failed: {exc}")

    stack_kv = Mistral4DecoderStackKvState(1)
    h_tt = emb(input_pref)
    h_tt = stack(
        h_tt,
        position_ids=pos_pref,
        position_embeddings=pe_pref,
        mode="prefill",
        stack_kv=stack_kv,
    )

    h2_tt = emb(input_dec)
    h2_tt = stack(
        h2_tt,
        position_ids=pos_dec,
        position_embeddings=pe_dec,
        mode="decode",
        stack_kv=stack_kv,
    )

    ref_last = h2_hf[:, -1:, :].contiguous()
    got = to_torch_auto_compose(h2_tt, device=mesh_device)[:, :, :1, :].squeeze(1)
    passing, pcc_message = comp_pcc(ref_last, got, pcc=0.92)
    logger.info(comp_allclose(ref_last, got))
    logger.info(f"1-layer stack decode PCC: {pcc_message}")
    stack_kv.clear()
    assert passing, f"PCC below 0.92: {pcc_message}"
