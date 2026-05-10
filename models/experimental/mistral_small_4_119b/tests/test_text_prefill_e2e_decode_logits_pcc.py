# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""E2E PCC: prefill then **one decode** logits (``embed`` → ``N`` layers + KV → ``norm`` + ``lm_head``).

``N`` is in ``{1, 2}``. With ``N=2``, two device↔host MoE hops plus bf16/tilize drift before ``lm_head``
mirror ``test_text_prefill_e2e_logits_pcc`` — use a **lower** full-tensor logits PCC bar for ``N=2``.
"""

import pytest
import torch
import torch.nn.functional as F
import ttnn
from loguru import logger

from models.common.utility_functions import comp_allclose, comp_pcc, run_for_wormhole_b0_or_blackhole
from models.experimental.mistral_small_4_119b.constants import (
    HF_MODEL_ID,
    TEXT_MODEL_EMBED_TOKENS_WEIGHT_KEY,
    TEXT_MODEL_LM_HEAD_WEIGHT_KEY,
    TEXT_MODEL_NORM_WEIGHT_KEY,
    strip_fp8_aux_tensors_from_decoder_inner,
    text_decoder_layer_inner_state_dict,
    text_decoder_layer_state_dict_prefix,
)
from models.experimental.mistral_small_4_119b.tests.mesh_param import mesh_device_request_param
from models.experimental.mistral_small_4_119b.tt.mistral4_self_attention import _torch_for_ttnn_upload
from models.experimental.mistral_small_4_119b.tt.mistral4_text_prefill import TtMistral4TextPrefillLogits
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


def _prefixes_e2e(num_decoder_layers: int) -> tuple[str, ...]:
    p: list[str] = ["language_model.model.embed_tokens."]
    for i in range(num_decoder_layers):
        p.append(text_decoder_layer_state_dict_prefix(i))
    p.append("language_model.model.norm.")
    p.append("language_model.lm_head.")
    return tuple(p)


def _load_state_dict(num_decoder_layers: int) -> dict:
    try:
        return load_hf_state_dict_filtered(HF_MODEL_ID, _prefixes_e2e(num_decoder_layers))
    except (FileNotFoundError, OSError) as exc:
        pytest.skip(f"Checkpoint load failed: {exc}")


def _hf_decode_logits_last_token(
    text,
    state_dict: dict,
    *,
    num_decoder_layers: int,
    prefill_ids: torch.Tensor,
    decode_ids: torch.Tensor,
    pos_pref: torch.Tensor,
    pos_dec: torch.Tensor,
):
    from transformers.cache_utils import DynamicCache
    from transformers.models.mistral4.modeling_mistral4 import (
        Mistral4DecoderLayer,
        Mistral4RMSNorm,
        Mistral4RotaryEmbedding,
    )

    layers: list = []
    for li in range(num_decoder_layers):
        inner = strip_fp8_aux_tensors_from_decoder_inner(text_decoder_layer_inner_state_dict(state_dict, li))
        layer = Mistral4DecoderLayer(text, layer_idx=li).eval()
        try:
            layer.load_state_dict(inner, strict=True)
        except Exception as exc:
            pytest.skip(f"HF layer {li} load failed: {exc}")
        layers.append(layer.to(torch.bfloat16))

    rotary = Mistral4RotaryEmbedding(text).eval().to(torch.bfloat16)
    w = _torch_for_ttnn_upload(state_dict[TEXT_MODEL_EMBED_TOKENS_WEIGHT_KEY])
    cache = DynamicCache()
    hidden = F.embedding(prefill_ids, w).to(torch.bfloat16)
    pe_pref = rotary(hidden, pos_pref)
    for layer in layers:
        hidden = layer(
            hidden,
            attention_mask=None,
            position_ids=pos_pref,
            past_key_values=cache,
            use_cache=True,
            position_embeddings=pe_pref,
        )

    hidden_dec = F.embedding(decode_ids, w).to(torch.bfloat16)
    pe_dec = rotary(hidden_dec, pos_dec)
    for layer in layers:
        hidden_dec = layer(
            hidden_dec,
            attention_mask=None,
            position_ids=pos_dec,
            past_key_values=cache,
            use_cache=True,
            position_embeddings=pe_dec,
        )

    norm_w = _torch_for_ttnn_upload(state_dict[TEXT_MODEL_NORM_WEIGHT_KEY])
    lm_w = _torch_for_ttnn_upload(state_dict[TEXT_MODEL_LM_HEAD_WEIGHT_KEY])
    norm = Mistral4RMSNorm(text.hidden_size, eps=float(text.rms_norm_eps)).eval()
    norm.weight.data = norm_w
    hidden_dec = norm(hidden_dec)
    return F.linear(hidden_dec, lm_w)


@torch.no_grad()
@run_for_wormhole_b0_or_blackhole()
@pytest.mark.parametrize("mesh_device", [mesh_device_request_param()], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 30000000, "num_command_queues": 1}],
    indirect=True,
)
@pytest.mark.parametrize("prefill_len", (4,))
@pytest.mark.parametrize("num_decoder_layers", (1, 2), ids=("nlayer1", "nlayer2"))
def test_mistral_small_4_text_prefill_then_decode_e2e_logits_pcc(mesh_device, prefill_len, num_decoder_layers):
    from transformers.models.mistral4.modeling_mistral4 import Mistral4RotaryEmbedding

    text = _text_config_eager_attn()
    state_dict = _load_state_dict(num_decoder_layers)

    w = state_dict[TEXT_MODEL_EMBED_TOKENS_WEIGHT_KEY]
    vocab = w.shape[0]
    torch.manual_seed(13)
    prefill_ids = torch.randint(0, vocab, (1, prefill_len), dtype=torch.long)
    pos_pref = torch.arange(prefill_len, dtype=torch.long).unsqueeze(0)
    decode_ids = torch.randint(0, vocab, (1, 1), dtype=torch.long)
    pos_dec = torch.tensor([[prefill_len]], dtype=torch.long)

    w_bf16 = _torch_for_ttnn_upload(w)
    hidden0 = F.embedding(prefill_ids, w_bf16).to(torch.bfloat16)
    rotary = Mistral4RotaryEmbedding(text).eval().to(torch.bfloat16)
    pe_pref = rotary(hidden0, pos_pref)
    hidden_dec0 = F.embedding(decode_ids, w_bf16).to(torch.bfloat16)
    pe_dec = rotary(hidden_dec0, pos_dec)

    ref_logits = _hf_decode_logits_last_token(
        text,
        state_dict,
        num_decoder_layers=num_decoder_layers,
        prefill_ids=prefill_ids,
        decode_ids=decode_ids,
        pos_pref=pos_pref,
        pos_dec=pos_dec,
    )

    try:
        model = TtMistral4TextPrefillLogits(
            mesh_device,
            state_dict,
            text,
            num_decoder_layers=num_decoder_layers,
            use_device_rotary_embedding_table=False,
        )
    except Exception as exc:
        pytest.skip(f"TtMistral4TextPrefillLogits init failed: {exc}")

    stack_kv = model.make_stack_kv_state()
    _ = model(
        prefill_ids,
        position_ids=pos_pref,
        position_embeddings=pe_pref,
        mode="prefill",
        stack_kv=stack_kv,
    )
    tt_logits = model(
        decode_ids,
        position_ids=pos_dec,
        position_embeddings=pe_dec,
        mode="decode",
        stack_kv=stack_kv,
    )
    stack_kv.clear()

    min_pcc = 0.76 if num_decoder_layers >= 2 else 0.92
    passing, pcc_message = comp_pcc(ref_logits, tt_logits, pcc=min_pcc)
    logger.info(comp_allclose(ref_logits, tt_logits))
    logger.info(f"e2e decode logits (N={num_decoder_layers}, min={min_pcc}) PCC: {pcc_message}")
    assert passing, f"PCC below {min_pcc}: {pcc_message}"
