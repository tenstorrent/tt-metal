# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
PCC test: Hugging Face ``Ministral3Attention`` vs ``TtMinistralAttention`` on Devstral weights.

Loads the **full** multimodal checkpoint via ``ModelArgs.load_state_dict()`` (same conversion path as
production: ``standardize_hf_keys_multimodal`` + ``convert_vision_hf_to_meta_no_qkv_permute``), so
TT attention sees keys such as ``layers.0.attention.wq.weight``.

The reference submodule is taken from ``cached_hf_model.model.language_model`` (layer 0 ``self_attn``
and ``rotary_emb``). Rotary cos/sin are uploaded from HF so TT matches reference rope tables.

Requirements:
- Enough host RAM / VRAM for ``transformers`` to load Devstral (~24B parameters / FP8→BF16).
- Sequence length multiple of 128 (TT attention prefill path).

The ``trust_remote_ministral`` fixture monkeypatches ``ModelArgs.get_hf_model_cls`` to use
``AutoModelForImageTextToText`` from ``modeling_auto`` so this test does not import
``AutoModelForVision2Seq`` (absent in some ``transformers`` versions). No changes to ``model_config.py``.

At import time this module patches ``Fp8Dequantize._dequantize_one`` so scalar FP8 scales from
Devstral checkpoints do not trip Hugging Face's fine-grained FP8 dequantizer (same workaround as
``demo_devstral2_tt_multimodal.py`` / ``reference/inference.py``).

Reference attention is asserted to be Hugging Face ``Ministral3Attention`` from
``transformers.models.ministral3.modeling_ministral3`` before PCC vs TT.
"""

from __future__ import annotations


import os

import pytest
import torch
from loguru import logger
from transformers.integrations.finegrained_fp8 import Fp8Dequantize
from transformers.masking_utils import create_causal_mask
from transformers.models.ministral3.modeling_ministral3 import Ministral3Attention

# Devstral HF checkpoints can ship scalar FP8 scales; ``Fp8Dequantize._dequantize_one`` expects a 2D
# scale grid and raises unless we patch before ``ModelArgs.load_state_dict()`` / ``from_pretrained``.
_ORIGINAL_FP8_DEQUANTIZE_ONE = Fp8Dequantize._dequantize_one


def _dequantize_one_compat(self, quantized: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    if scales.ndim == 0:
        fp4_dtype = getattr(torch, "float4_e2m1fn_x2", None)
        if quantized.dtype == torch.int8 or (fp4_dtype is not None and quantized.dtype == fp4_dtype):
            quantized_fp32 = self._unpack_fp4(quantized)
        else:
            quantized_fp32 = quantized.to(torch.float32)
        out_dtype = scales.dtype if scales.dtype.is_floating_point and scales.element_size() >= 2 else torch.bfloat16
        scale = scales.to(torch.float32)
        return (quantized_fp32 * scale).to(out_dtype)
    return _ORIGINAL_FP8_DEQUANTIZE_ONE(self, quantized, scales)


Fp8Dequantize._dequantize_one = _dequantize_one_compat

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.experimental.devstarl2_small.tt.tt_ministralattn import TtMinistralAttention
from models.tt_transformers.tt.ccl import TT_CCL
from models.tt_transformers.tt.model_config import ModelArgs

DEVSTRAL_REPO_ID = "mistralai/Devstral-Small-2-24B-Instruct-2512"


def _text_model_root(multimodal_inner):
    """``Mistral3Model.language_model`` → inner causal stack with ``layers`` and ``rotary_emb``."""
    lm = multimodal_inner.language_model
    return lm.model if hasattr(lm, "model") else lm


@pytest.fixture
def trust_remote_ministral(monkeypatch):
    """Trust remote code for config/load, and avoid ``AutoModelForVision2Seq`` where needed."""

    from models.tt_transformers.tt import model_config as mc

    orig_set = mc.ModelArgs._set_hf_params

    def _set_hf_params_trust(self, checkpoint_dir: str):
        self.trust_remote_code_hf = True
        return orig_set(self, checkpoint_dir)

    monkeypatch.setattr(mc.ModelArgs, "_set_hf_params", _set_hf_params_trust)

    def _get_hf_model_cls_devstral_safe(self):
        """Mistral3 / Devstral map to ``AutoModelForImageTextToText``; skip broken top-level Vision2Seq import."""
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
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize("seq_len", (128,))
@pytest.mark.parametrize("batch_size", (1,))
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 30000000, "num_command_queues": 1}],
    indirect=True,
)
def test_ministral3_attention_pcc_devstral_weights(
    mesh_device,
    seq_len,
    batch_size,
    monkeypatch,
    trust_remote_ministral,
):
    monkeypatch.setenv("HF_MODEL", DEVSTRAL_REPO_ID)

    dtype = ttnn.bfloat16
    model_args = ModelArgs(
        mesh_device,
        max_batch_size=batch_size,
        max_seq_len=max(512, seq_len),
        dummy_weights=False,
        use_hf_rope=True,
        cache_hf=True,
    )

    text_cfg = model_args.hf_config.text_config

    try:
        meta_state_dict = model_args.load_state_dict()
    except Exception as exc:
        pytest.skip(f"Full checkpoint load failed (memory / hub / env): {exc}")

    hf_full = model_args.cached_hf_model
    assert hf_full is not None, "Expected cached HF model after load_state_dict with cache_hf=True."

    text_root = _text_model_root(hf_full.model)
    hf_attn = text_root.layers[0].self_attn
    assert isinstance(
        hf_attn, Ministral3Attention
    ), f"PCC reference must be HF Ministral3Attention; got {type(hf_attn).__module__}.{type(hf_attn).__name__}"
    rotary = text_root.rotary_emb

    hf_attn.eval()
    rotary.eval()

    hidden_size = model_args.dim
    x = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.bfloat16)
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)

    position_embeddings = rotary(x, position_ids=position_ids)
    causal_mask = create_causal_mask(
        config=text_cfg,
        inputs_embeds=x,
        attention_mask=None,
        past_key_values=None,
        position_ids=position_ids,
    )

    ref_out, _ = hf_attn(
        x,
        position_embeddings=position_embeddings,
        attention_mask=causal_mask,
        position_ids=position_ids,
        past_key_values=None,
    )

    rope_params = text_cfg.rope_parameters or {}
    if not isinstance(rope_params, dict):
        rope_params = dict(rope_params)

    tt_ccl = TT_CCL(mesh_device)
    transformation_mats = {"decode": None, "prefill": None}

    tt_attn = TtMinistralAttention(
        mesh_device,
        tt_ccl,
        model_args,
        meta_state_dict,
        model_args.weight_cache_path(dtype),
        layer_num=0,
        dtype=dtype,
        transformation_mats=transformation_mats,
        configuration=model_args,
        llama_4_scaling_beta=rope_params.get("llama_4_scaling_beta"),
        original_max_position_embeddings=rope_params.get("original_max_position_embeddings"),
    )

    cos, sin = position_embeddings
    cos_tt = ttnn.from_torch(
        cos.unsqueeze(0),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    sin_tt = ttnn.from_torch(
        sin.unsqueeze(0),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    rot_mats = [cos_tt, sin_tt]

    x_tt = ttnn.from_torch(
        x.unsqueeze(1),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    pos_tt = ttnn.from_torch(
        position_ids.to(torch.int32),
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    tt_out = tt_attn.forward_prefill(x_tt, rot_mats, position_ids=pos_tt)

    tt_torch = ttnn.to_torch(tt_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1))
    tt_torch = tt_torch.reshape(ref_out.shape)

    pcc_required = 0.94
    passing, pcc_message = comp_pcc(ref_out, tt_torch, pcc_required)
    logger.info(comp_allclose(ref_out, tt_torch))
    logger.info(f"PCC: {pcc_message}")
    assert passing, f"PCC below {pcc_required}: {pcc_message}"
