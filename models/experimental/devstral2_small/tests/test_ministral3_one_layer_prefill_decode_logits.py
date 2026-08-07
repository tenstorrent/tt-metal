# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import types

import pytest
import torch
from loguru import logger
from transformers.cache_utils import DynamicCache
from transformers.masking_utils import create_causal_mask
from transformers.models.ministral3.modeling_ministral3 import Ministral3DecoderLayer

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.experimental.devstral2_small.devstral_utils import apply_fp8_dequantize_compat, resolve_rope_parameters
from models.experimental.devstral2_small.tt.tt_ministral3_decoder_layer import TtMinistral3DecoderLayer
from models.experimental.devstral2_small.tt.tt_ministral_rotary_emb import TtMinistral3RotaryEmbedding
from models.tt_transformers.tt.ccl import TT_CCL
from models.tt_transformers.tt.common import Mode
from models.tt_transformers.tt.model_config import ModelArgs

apply_fp8_dequantize_compat()

DEVSTRAL_REPO_ID = "mistralai/Devstral-Small-2-24B-Instruct-2512"
PCC_TARGET = float(os.environ.get("MINISTRAL3_DECODER_LOGITS_PCC", "0.99"))
_MESH_DEVICE_PRESETS = {"P150": (1, 1), "BH-QB": (1, 4)}
LAYER_NUM = 0


def _mesh_device_param():
    mesh_env = os.environ.get("MESH_DEVICE")
    if mesh_env in _MESH_DEVICE_PRESETS:
        return _MESH_DEVICE_PRESETS[mesh_env]
    try:
        return ttnn._ttnn.multi_device.SystemMeshDescriptor().shape().mesh_size()
    except Exception:
        return int(os.environ.get("TT_MESH_WIDTH", "4"))


def _text_model_root(multimodal_inner):
    lm = multimodal_inner.language_model
    return lm.model if hasattr(lm, "model") else lm


def _hf_lm_head(hf_full, text_root):
    candidates = (
        hf_full,
        getattr(hf_full, "model", None),
        getattr(hf_full, "language_model", None),
        getattr(getattr(hf_full, "model", None), "language_model", None),
        text_root,
        getattr(text_root, "language_model", None),
    )
    for candidate in candidates:
        if candidate is not None and hasattr(candidate, "lm_head"):
            return candidate.lm_head
    raise AttributeError("Could not find HF lm_head on Devstral model.")


def _layer_hidden(layer_out):
    return layer_out[0] if isinstance(layer_out, tuple) else layer_out


def _logits_from_hidden(hf_full, text_root, hidden):
    normed = text_root.norm(hidden)
    return _hf_lm_head(hf_full, text_root)(normed).float()


def _as_tile_tensor(torch_tensor, mesh_device):
    host_tensor = ttnn.from_torch(
        torch_tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    return ttnn.to_device(host_tensor, mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def _as_position_tensor(position_ids, mesh_device):
    return ttnn.from_torch(
        position_ids.to(torch.int32),
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _tt_prefill_hidden_to_bsh(tt_out, mesh_device, ref_shape):
    tt_torch = ttnn.to_torch(tt_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[0]
    while tt_torch.dim() > 3:
        tt_torch = tt_torch.squeeze(0)
    if tt_torch.dim() == 2:
        tt_torch = tt_torch.unsqueeze(0)
    return tt_torch.reshape(ref_shape)


def _tt_decode_hidden_to_bsh(tt_out, mesh_device, batch_size, hidden_dim):
    tt_torch = ttnn.to_torch(tt_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1))
    if tt_torch.dim() == 4:
        return tt_torch[:, :1, :batch_size, :hidden_dim].reshape(batch_size, 1, hidden_dim)
    while tt_torch.dim() > 3:
        tt_torch = tt_torch.squeeze(0)
    return tt_torch.reshape(batch_size, 1, hidden_dim)


def _build_model_args(mesh_device, batch_size, max_seq_len):
    model_args = ModelArgs(
        mesh_device,
        max_batch_size=batch_size,
        max_seq_len=max_seq_len,
        dummy_weights=False,
        use_hf_rope=True,
        cache_hf=True,
    )
    model_args.n_layers = 1
    model_args.is_distributed_norm = types.MethodType(
        lambda self, mode: model_args.is_multichip and mode == Mode.DECODE,
        model_args,
    )
    return model_args


def _load_devstral_context(mesh_device, batch_size, max_seq_len):
    model_args = _build_model_args(mesh_device, batch_size, max_seq_len)
    try:
        state_dict = model_args.load_state_dict()
    except Exception as exc:
        pytest.skip(f"Full checkpoint load failed (memory / hub / env): {exc}")

    hf_full = model_args.cached_hf_model
    assert hf_full is not None

    text_root = _text_model_root(hf_full.model)
    hf_layer = text_root.layers[LAYER_NUM]
    assert isinstance(hf_layer, Ministral3DecoderLayer), type(hf_layer)
    hf_layer.eval()
    text_root.norm.eval()
    _hf_lm_head(hf_full, text_root).eval()

    rope_params = resolve_rope_parameters(model_args.hf_config.text_config)
    tt_layer = TtMinistral3DecoderLayer(
        mesh_device=mesh_device,
        tt_ccl=TT_CCL(mesh_device),
        model_args=model_args,
        meta_state_dict=state_dict,
        weight_cache_path=model_args.weight_cache_path(ttnn.bfloat16),
        layer_num=LAYER_NUM,
        dtype=ttnn.bfloat16,
        transformation_mats={"decode": None, "prefill": None},
        configuration=model_args,
        llama_4_scaling_beta=rope_params.get("llama_4_scaling_beta"),
        original_max_position_embeddings=rope_params.get("original_max_position_embeddings"),
    )
    tt_rope = TtMinistral3RotaryEmbedding(
        device=mesh_device,
        batch_size=batch_size,
        head_dim=model_args.head_dim,
        max_seq_len=model_args.max_seq_len,
        config=model_args.hf_config.text_config,
    )
    return model_args, hf_full, text_root, hf_layer, tt_layer, tt_rope


@pytest.fixture
def trust_remote_ministral(monkeypatch):
    from models.tt_transformers.tt import model_config as mc

    orig_set = mc.ModelArgs._set_hf_params

    def _set_hf_params_trust(self, checkpoint_dir: str):
        self.trust_remote_code_hf = True
        return orig_set(self, checkpoint_dir)

    monkeypatch.setattr(mc.ModelArgs, "_set_hf_params", _set_hf_params_trust)

    def _get_hf_model_cls_devstral_safe(self):
        from transformers import AutoModelForCausalLM
        from transformers.models.auto.modeling_auto import AutoModelForImageTextToText

        if not self.is_multimodal:
            return AutoModelForCausalLM
        if type(self.hf_config) in AutoModelForImageTextToText._model_mapping:
            return AutoModelForImageTextToText
        raise ValueError(
            f"Test supports multimodal configs in AutoModelForImageTextToText only; got {type(self.hf_config)}"
        )

    monkeypatch.setattr(mc.ModelArgs, "get_hf_model_cls", _get_hf_model_cls_devstral_safe)


@torch.no_grad()
@pytest.mark.timeout(900)
@pytest.mark.models_performance_bare_metal
@pytest.mark.usefixtures("trust_remote_ministral", "ensure_gc")
@pytest.mark.parametrize("mesh_device", [_mesh_device_param()], indirect=True)
@pytest.mark.parametrize("seq_len", (128,))
@pytest.mark.parametrize("batch_size", (1,))
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 60000000, "num_command_queues": 1}],
    indirect=True,
)
def test_ministral3_one_layer_prefill_then_decode_logits(
    mesh_device,
    seq_len,
    batch_size,
    monkeypatch,
):
    monkeypatch.setenv("HF_MODEL", DEVSTRAL_REPO_ID)
    torch.manual_seed(0)

    model_args, hf_full, text_root, hf_layer, tt_layer, tt_rope = _load_devstral_context(
        mesh_device, batch_size, max_seq_len=max(512, seq_len + 1)
    )

    prompt = torch.randn(batch_size, seq_len, model_args.dim, dtype=torch.bfloat16)
    decode_x = torch.randn(batch_size, 1, model_args.dim, dtype=torch.bfloat16)

    prompt_position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
    prompt_position_embeddings = text_root.rotary_emb(prompt, position_ids=prompt_position_ids)
    prompt_mask = create_causal_mask(
        config=model_args.hf_config.text_config,
        inputs_embeds=prompt,
        attention_mask=None,
        past_key_values=None,
        position_ids=prompt_position_ids,
    )

    hf_cache = DynamicCache()
    ref_prefill_hidden = _layer_hidden(
        hf_layer(
            hidden_states=prompt,
            attention_mask=prompt_mask,
            position_ids=prompt_position_ids,
            past_key_values=hf_cache,
            use_cache=True,
            cache_position=prompt_position_ids[0],
            position_embeddings=prompt_position_embeddings,
        )
    )
    ref_prefill_logits = _logits_from_hidden(hf_full, text_root, ref_prefill_hidden)

    decode_position_ids = torch.full((batch_size, 1), seq_len, dtype=torch.long)
    decode_position_embeddings = text_root.rotary_emb(decode_x, position_ids=decode_position_ids)
    ref_decode_hidden = _layer_hidden(
        hf_layer(
            hidden_states=decode_x,
            attention_mask=None,
            position_ids=decode_position_ids,
            past_key_values=hf_cache,
            use_cache=True,
            cache_position=decode_position_ids[0],
            position_embeddings=decode_position_embeddings,
        )
    )
    ref_decode_logits = _logits_from_hidden(hf_full, text_root, ref_decode_hidden)

    prompt_tt = _as_tile_tensor(prompt.unsqueeze(1), mesh_device)
    prompt_pos_tt = _as_position_tensor(prompt_position_ids, mesh_device)
    tt_prefill_hidden = tt_layer.forward_prefill(
        prompt_tt,
        tt_rope.slice_rot_mats_prefill(0, seq_len),
        position_ids=prompt_pos_tt,
    )
    tt_prefill_hidden_torch = _tt_prefill_hidden_to_bsh(tt_prefill_hidden, mesh_device, ref_prefill_hidden.shape)
    tt_prefill_logits = _logits_from_hidden(hf_full, text_root, tt_prefill_hidden_torch)
    ttnn.deallocate(tt_prefill_hidden)

    current_pos = torch.full((batch_size,), seq_len, dtype=torch.int32)
    current_pos_tt = ttnn.from_torch(
        current_pos.reshape(1, batch_size),
        device=mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    current_pos_rope_tt = ttnn.from_torch(
        current_pos.reshape(1, batch_size),
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    residual_mem_cfg = model_args.get_residual_mem_config(Mode.DECODE, None)
    decode_host = model_args.prepare_residual_tensor_decode(
        decode_x,
        residual_mem_cfg,
        on_host=True,
    )
    decode_tt = ttnn.to_device(decode_host, mesh_device, memory_config=residual_mem_cfg)
    ttnn.deallocate(decode_host)
    tt_decode_hidden = tt_layer.forward_decode(
        decode_tt,
        current_pos_tt,
        tt_rope.get_rot_mats(current_pos_rope_tt),
    )
    tt_decode_hidden_torch = _tt_decode_hidden_to_bsh(tt_decode_hidden, mesh_device, batch_size, model_args.dim)
    tt_decode_logits = _logits_from_hidden(hf_full, text_root, tt_decode_hidden_torch)

    prefill_passing, prefill_pcc_message = comp_pcc(ref_prefill_logits, tt_prefill_logits, PCC_TARGET)
    logger.info(comp_allclose(ref_prefill_logits, tt_prefill_logits))
    logger.info(f"One-layer prefill logits PCC: {prefill_pcc_message}")

    decode_passing, decode_pcc_message = comp_pcc(ref_decode_logits, tt_decode_logits, PCC_TARGET)
    logger.info(comp_allclose(ref_decode_logits, tt_decode_logits))
    logger.info(f"One-layer decode logits after {seq_len}-token prefill PCC: {decode_pcc_message}")

    assert prefill_passing, f"Prefill logits PCC below {PCC_TARGET}: {prefill_pcc_message}"
    assert decode_passing, f"Decode logits PCC below {PCC_TARGET}: {decode_pcc_message}"
