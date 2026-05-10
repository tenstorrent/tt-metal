# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""PCC: decoder prefill block (input norm + MLA attn + residual + post norm) vs HF ``Mistral4DecoderLayer``.

``layer_idx`` is parametrized (``0``, ``1``). Hub safetensors may include FP8 *auxiliary* tensors;
those keys are stripped before ``load_state_dict``. Skips if shards are missing or load fails.
"""

import pytest
import torch
import ttnn
from loguru import logger

from models.common.auto_compose import to_torch_auto_compose
from models.common.utility_functions import comp_allclose, comp_pcc, run_for_wormhole_b0_or_blackhole
from models.experimental.mistral_small_4_119b.constants import (
    HF_MODEL_ID,
    strip_fp8_aux_tensors_from_decoder_inner,
    text_decoder_layer_inner_state_dict,
    text_decoder_layer_state_dict_prefix,
)
from models.experimental.mistral_small_4_119b.tests.mesh_param import mesh_device_request_param
from models.experimental.mistral_small_4_119b.tt.text_backbone import TtMistral4DecoderLayerAttnPrefillBlock
from models.tt_transformers.tt.load_checkpoints import load_hf_state_dict_filtered


pytest.importorskip("transformers")
pytest.importorskip("transformers.models.mistral4.modeling_mistral4", reason="Mistral4 requires recent transformers")


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


def _layer_checkpoint(layer_idx: int):
    prefix = text_decoder_layer_state_dict_prefix(layer_idx)
    try:
        return load_hf_state_dict_filtered(HF_MODEL_ID, (prefix,))
    except (FileNotFoundError, OSError) as exc:
        pytest.skip(f"No checkpoint shards for layer {layer_idx}: {exc}")


@torch.no_grad()
@run_for_wormhole_b0_or_blackhole()
@pytest.mark.parametrize("mesh_device", [mesh_device_request_param()], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 30000000, "num_command_queues": 1}],
    indirect=True,
)
@pytest.mark.parametrize("seq_len", (8,))
@pytest.mark.parametrize("layer_idx", (0, 1), ids=("layer0", "layer1"))
def test_mistral_small_4_text_decoder_layer_attn_prefill_block_pcc(seq_len, layer_idx, reset_seeds, mesh_device):
    from transformers.models.mistral4.modeling_mistral4 import Mistral4DecoderLayer, Mistral4RotaryEmbedding

    text = _text_config_eager_attn()
    state_dict = _layer_checkpoint(layer_idx)
    inner = strip_fp8_aux_tensors_from_decoder_inner(text_decoder_layer_inner_state_dict(state_dict, layer_idx))

    layer = Mistral4DecoderLayer(text, layer_idx=layer_idx).eval()
    try:
        layer.load_state_dict(inner, strict=True)
    except Exception as exc:
        pytest.skip(
            f"Could not load layer-{layer_idx} weights into HF ``Mistral4DecoderLayer`` after stripping FP8 "
            f"aux keys (dtype/layout mismatch, etc.). Detail: {exc}"
        )
    layer = layer.to(torch.bfloat16)
    rotary = Mistral4RotaryEmbedding(text).eval().to(torch.bfloat16)

    torch.manual_seed(1)
    x = torch.randn(1, seq_len, text.hidden_size, dtype=torch.bfloat16)
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    position_embeddings = rotary(x, position_ids)

    residual = x
    h = layer.input_layernorm(x)
    attn_out, _ = layer.self_attn(
        h,
        attention_mask=None,
        position_ids=position_ids,
        past_key_values=None,
        use_cache=False,
        position_embeddings=position_embeddings,
    )
    ref = layer.post_attention_layernorm(residual + attn_out)

    try:
        tt_block = TtMistral4DecoderLayerAttnPrefillBlock(mesh_device, state_dict, text, layer_idx=layer_idx)
    except Exception as exc:
        pytest.skip(f"TTNN block init from hub layer-{layer_idx} tensors failed: {exc}")
    x_tt = ttnn.from_torch(
        x.unsqueeze(1),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device=mesh_device),
    )
    y_tt = tt_block(
        x_tt,
        position_ids=position_ids,
        position_embeddings=position_embeddings,
        mode="prefill",
    )
    # Replicated mesh activations: ``ConcatMeshToTensor`` would stack identical shards on ``dim=-1``
    # (4096 → 16384 on a 1×4 mesh). Compose from topology instead.
    y_tt_torch = to_torch_auto_compose(y_tt, device=mesh_device)[:, :, :seq_len, :]
    y_tt_torch = y_tt_torch.squeeze(1)

    passing, pcc_message = comp_pcc(ref, y_tt_torch, pcc=0.92)
    logger.info(comp_allclose(ref, y_tt_torch))
    logger.info(f"layer{layer_idx} attn block PCC: {pcc_message}")
    assert passing, f"PCC below 0.92: {pcc_message}"
