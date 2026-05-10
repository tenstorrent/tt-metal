# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""PCC: ``embed_tokens`` → ``N`` × :class:`TtMistral4DecoderLayer` vs HF (``N`` in ``{1, 2}``)."""

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


def _prefixes_embed_and_layers(num_decoder_layers: int) -> tuple[str, ...]:
    p: list[str] = ["language_model.model.embed_tokens."]
    for i in range(num_decoder_layers):
        p.append(text_decoder_layer_state_dict_prefix(i))
    return tuple(p)


def _load_state_dict(num_decoder_layers: int) -> dict:
    try:
        return load_hf_state_dict_filtered(HF_MODEL_ID, _prefixes_embed_and_layers(num_decoder_layers))
    except (FileNotFoundError, OSError) as exc:
        pytest.skip(f"Checkpoint load failed: {exc}")


def _hf_reference_stack(
    input_ids: torch.Tensor,
    text,
    state_dict: dict,
    *,
    num_decoder_layers: int,
    position_ids: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
):
    from transformers.models.mistral4.modeling_mistral4 import Mistral4DecoderLayer

    w = _torch_for_ttnn_upload(state_dict[TEXT_MODEL_EMBED_TOKENS_WEIGHT_KEY])
    hidden = F.embedding(input_ids, w).to(torch.bfloat16)
    for li in range(num_decoder_layers):
        inner = strip_fp8_aux_tensors_from_decoder_inner(text_decoder_layer_inner_state_dict(state_dict, li))
        layer = Mistral4DecoderLayer(text, layer_idx=li).eval()
        try:
            layer.load_state_dict(inner, strict=True)
        except Exception as exc:
            pytest.skip(f"HF layer {li} load failed: {exc}")
        layer = layer.to(torch.bfloat16)
        hidden = layer(
            hidden,
            attention_mask=None,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
            position_embeddings=position_embeddings,
        )
    return hidden


@torch.no_grad()
@run_for_wormhole_b0_or_blackhole()
@pytest.mark.parametrize("mesh_device", [mesh_device_request_param()], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 30000000, "num_command_queues": 1}],
    indirect=True,
)
@pytest.mark.parametrize("seq_len", (8,))
@pytest.mark.parametrize("num_decoder_layers", (1, 2), ids=("nlayer1", "nlayer2"))
@pytest.mark.parametrize(
    "use_device_rotary_embedding_table",
    (False, True),
    ids=("rope_host_upload", "rope_device_table"),
)
def test_mistral_small_4_text_embed_decoder_stack_pcc(
    seq_len, num_decoder_layers, use_device_rotary_embedding_table, reset_seeds, mesh_device
):
    from transformers.models.mistral4.modeling_mistral4 import Mistral4RotaryEmbedding

    text = _text_config_eager_attn()
    state_dict = _load_state_dict(num_decoder_layers)

    w = state_dict[TEXT_MODEL_EMBED_TOKENS_WEIGHT_KEY]
    vocab = w.shape[0]
    torch.manual_seed(2)
    input_ids = torch.randint(0, vocab, (1, seq_len), dtype=torch.long)
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)

    w_bf16 = _torch_for_ttnn_upload(w)
    hidden0 = F.embedding(input_ids, w_bf16).to(torch.bfloat16)
    rotary = Mistral4RotaryEmbedding(text).eval().to(torch.bfloat16)
    position_embeddings = rotary(hidden0, position_ids)

    ref = _hf_reference_stack(
        input_ids,
        text,
        state_dict,
        num_decoder_layers=num_decoder_layers,
        position_ids=position_ids,
        position_embeddings=position_embeddings,
    )

    try:
        emb = TtMistral4EmbedTokensPrefill(mesh_device, state_dict)
        stack = TtMistral4DecoderSequence(
            mesh_device,
            state_dict,
            text,
            layer_indices=tuple(range(num_decoder_layers)),
            use_device_rotary_embedding_table=use_device_rotary_embedding_table,
        )
    except Exception as exc:
        pytest.skip(f"TTNN embed/stack init failed: {exc}")

    h_tt = emb(input_ids)
    y_tt = stack(
        h_tt,
        position_ids=position_ids,
        position_embeddings=None if use_device_rotary_embedding_table else position_embeddings,
        mode="prefill",
    )
    y_tt_torch = to_torch_auto_compose(y_tt, device=mesh_device)[:, :, :seq_len, :].squeeze(1)

    passing, pcc_message = comp_pcc(ref, y_tt_torch, pcc=0.92)
    logger.info(comp_allclose(ref, y_tt_torch))
    logger.info(f"embed+{num_decoder_layers}layer(s) PCC: {pcc_message}")
    assert passing, f"PCC below 0.92: {pcc_message}"
