# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared Hugging Face load path for Devstral-2-123B (matches ``devstral2_123b_inference.py``)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from transformers import AutoConfig, AutoModelForCausalLM, FineGrainedFP8Config
from transformers.integrations.finegrained_fp8 import FP8Linear, Fp8Dequantize
from transformers.models.ministral3.configuration_ministral3 import Ministral3Config

if TYPE_CHECKING:
    from transformers import PreTrainedModel

DEVSTRAL2_MODEL_ID = "mistralai/Devstral-2-123B-Instruct-2512"
DEFAULT_OFFLOAD_FOLDER = "./hf_offload_devstral2_123b"

_ORIGINAL_DEQUANTIZE_ONE = Fp8Dequantize._dequantize_one


def hf_local_files_only() -> bool:
    """When true, only use files already in the HF cache (no Hub download)."""
    return os.getenv("DEVSTRAL2_HF_LOCAL_ONLY", "").lower() in ("1", "true", "yes")


def hf_reference_torch_device() -> torch.device:
    """Torch device for HF reference *inputs* (CPU on machines without a CUDA GPU)."""
    override = os.getenv("DEVSTRAL2_HF_DEVICE", "").strip()
    if override:
        return torch.device(override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def resolve_hf_input_device(causal_lm: PreTrainedModel) -> torch.device:
    """Device for token/logit tensors fed into a loaded HF model."""
    override = os.getenv("DEVSTRAL2_HF_DEVICE", "").strip()
    if override:
        return torch.device(override)
    model_device = getattr(causal_lm, "device", None)
    if model_device is not None:
        return torch.device(model_device)
    if not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        return next(causal_lm.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def hf_reference_device_map() -> str | dict[str, str]:
    """``device_map`` for ``from_pretrained`` — CPU + disk offload when no CUDA GPU."""
    override = os.getenv("DEVSTRAL2_HF_DEVICE_MAP", "").strip()
    if override:
        return override
    if torch.cuda.is_available():
        return "auto"
    return "cpu"


def apply_fp8_dequantize_compat_patch() -> None:
    """Patch HF FP8 dequant for scalar ``weight_scale_inv`` (same as inference script)."""
    if Fp8Dequantize._dequantize_one is not _ORIGINAL_DEQUANTIZE_ONE:
        return

    def _dequantize_one_compat(
        self,
        quantized: torch.Tensor,
        scales: torch.Tensor,
        output_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        if scales.ndim == 0:
            fp4_dtype = getattr(torch, "float4_e2m1fn_x2", None)
            if quantized.dtype == torch.int8 or (fp4_dtype is not None and quantized.dtype == fp4_dtype):
                quantized_fp32 = self._unpack_fp4(quantized)
            else:
                quantized_fp32 = quantized.to(torch.float32)
            if output_dtype is None:
                output_dtype = (
                    scales.dtype if scales.dtype.is_floating_point and scales.element_size() >= 2 else torch.bfloat16
                )
            scale = scales.to(torch.float32)
            return (quantized_fp32 * scale).to(output_dtype)
        return _ORIGINAL_DEQUANTIZE_ONE(self, quantized, scales, output_dtype=output_dtype)

    Fp8Dequantize._dequantize_one = _dequantize_one_compat


def load_devstral2_text_config() -> Ministral3Config:
    """Load the text ``Ministral3Config`` from the Devstral-2 Hub repo."""
    hf_cfg = AutoConfig.from_pretrained(
        DEVSTRAL2_MODEL_ID,
        trust_remote_code=True,
        local_files_only=hf_local_files_only(),
    )
    text = getattr(hf_cfg, "text_config", None) or hf_cfg
    if not isinstance(text, Ministral3Config):
        raise TypeError(f"Expected Ministral3Config, got {type(text)!r}")
    return text


def load_devstral2_causal_lm(
    *,
    offload_folder: str | Path | None = None,
    local_files_only: bool | None = None,
) -> PreTrainedModel:
    """Load ``AutoModelForCausalLM`` with FineGrained FP8 (inference script parity)."""
    apply_fp8_dequantize_compat_patch()
    local_only = hf_local_files_only() if local_files_only is None else local_files_only
    folder = Path(offload_folder or os.getenv("DEVSTRAL2_HF_OFFLOAD_FOLDER", DEFAULT_OFFLOAD_FOLDER))
    folder.mkdir(parents=True, exist_ok=True)

    config = AutoConfig.from_pretrained(
        DEVSTRAL2_MODEL_ID,
        trust_remote_code=True,
        local_files_only=local_only,
    )
    model_quant_cfg = getattr(config, "quantization_config", {}) or {}
    quantization_config = FineGrainedFP8Config(
        activation_scheme=model_quant_cfg.get("activation_scheme", "static"),
        weight_block_size=model_quant_cfg.get("weight_block_size", None),
        dequantize=False,
        modules_to_not_convert=model_quant_cfg.get("modules_to_not_convert", None),
    )

    model = AutoModelForCausalLM.from_pretrained(
        DEVSTRAL2_MODEL_ID,
        dtype=torch.bfloat16,
        device_map=hf_reference_device_map(),
        offload_folder=str(folder),
        offload_state_dict=True,
        low_cpu_mem_usage=True,
        quantization_config=quantization_config,
        trust_remote_code=True,
        local_files_only=local_only,
    )
    model.eval()
    return model


def prepare_ministral3_backbone_for_pcc(causal_lm: PreTrainedModel):
    """Return ``causal_lm.model`` with eager attention for deterministic PCC."""
    ref = causal_lm.model
    ref.config._attn_implementation = "eager"
    return ref


def prepare_causal_lm_for_pcc(causal_lm: PreTrainedModel) -> PreTrainedModel:
    """Eager attention on the full causal LM for deterministic logit PCC."""
    causal_lm.config._attn_implementation = "eager"
    if hasattr(causal_lm, "model"):
        causal_lm.model.config._attn_implementation = "eager"
    causal_lm.eval()
    return causal_lm


def ministral3_backbone_weight_keys(num_layers: int) -> list[str]:
    """State-dict keys for ``TtMinistral3Model`` (``model.`` prefix)."""
    keys = ["model.embed_tokens.weight", "model.norm.weight"]
    for layer_idx in range(num_layers):
        p = f"model.layers.{layer_idx}"
        keys.extend(
            [
                f"{p}.input_layernorm.weight",
                f"{p}.post_attention_layernorm.weight",
                f"{p}.self_attn.q_proj.weight",
                f"{p}.self_attn.k_proj.weight",
                f"{p}.self_attn.v_proj.weight",
                f"{p}.self_attn.o_proj.weight",
                f"{p}.mlp.gate_proj.weight",
                f"{p}.mlp.up_proj.weight",
                f"{p}.mlp.down_proj.weight",
            ]
        )
    return keys


def _tensor_to_cpu_bf16(t: torch.Tensor) -> torch.Tensor:
    return t.detach().cpu().to(torch.bfloat16).clone()


def _linear_weight_to_bf16(linear: torch.nn.Module, dequant: Fp8Dequantize) -> torch.Tensor:
    weight = linear.weight
    if isinstance(linear, FP8Linear) and weight.element_size() == 1:
        weight = dequant._dequantize_one(weight, linear.weight_scale_inv)
    return _tensor_to_cpu_bf16(weight)


def extract_backbone_bf16_state_dict(causal_lm: PreTrainedModel, num_layers: int) -> dict[str, torch.Tensor]:
    """Dequantize FP8 linears and gather bf16 CPU tensors for the TT upload path."""
    apply_fp8_dequantize_compat_patch()
    dequant = Fp8Dequantize(hf_quantizer=None)
    backbone = causal_lm.model
    out: dict[str, torch.Tensor] = {}

    out["model.embed_tokens.weight"] = _tensor_to_cpu_bf16(backbone.embed_tokens.weight)
    out["model.norm.weight"] = _tensor_to_cpu_bf16(backbone.norm.weight)

    for layer_idx in range(num_layers):
        layer = backbone.layers[layer_idx]
        p = f"model.layers.{layer_idx}"
        out[f"{p}.input_layernorm.weight"] = _tensor_to_cpu_bf16(layer.input_layernorm.weight)
        out[f"{p}.post_attention_layernorm.weight"] = _tensor_to_cpu_bf16(layer.post_attention_layernorm.weight)
        attn = layer.self_attn
        out[f"{p}.self_attn.q_proj.weight"] = _linear_weight_to_bf16(attn.q_proj, dequant)
        out[f"{p}.self_attn.k_proj.weight"] = _linear_weight_to_bf16(attn.k_proj, dequant)
        out[f"{p}.self_attn.v_proj.weight"] = _linear_weight_to_bf16(attn.v_proj, dequant)
        out[f"{p}.self_attn.o_proj.weight"] = _linear_weight_to_bf16(attn.o_proj, dequant)
        mlp = layer.mlp
        out[f"{p}.mlp.gate_proj.weight"] = _linear_weight_to_bf16(mlp.gate_proj, dequant)
        out[f"{p}.mlp.up_proj.weight"] = _linear_weight_to_bf16(mlp.up_proj, dequant)
        out[f"{p}.mlp.down_proj.weight"] = _linear_weight_to_bf16(mlp.down_proj, dequant)

    expected = ministral3_backbone_weight_keys(num_layers)
    missing = [k for k in expected if k not in out]
    if missing:
        raise KeyError(f"Missing backbone weights after extraction: {missing[:5]}")
    return out


def extract_causal_lm_bf16_state_dict(causal_lm: PreTrainedModel, num_layers: int) -> dict[str, torch.Tensor]:
    """Backbone bf16 weights plus ``lm_head`` when embeddings are untied."""
    out = extract_backbone_bf16_state_dict(causal_lm, num_layers)
    tie = bool(getattr(causal_lm.config, "tie_word_embeddings", False))
    if not tie and hasattr(causal_lm, "lm_head"):
        dequant = Fp8Dequantize(hf_quantizer=None)
        out["lm_head.weight"] = _linear_weight_to_bf16(causal_lm.lm_head, dequant)
    return out
