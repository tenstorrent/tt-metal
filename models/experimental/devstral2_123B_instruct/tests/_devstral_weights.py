# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared Hugging Face download / dequantization helpers for the Devstral-2-123B tests.

The Devstral-2-123B checkpoint on the Hub is stored in FineGrained FP8 with per-block weight
scales. The TT path expects ``bf16`` host tensors. Loading the full checkpoint is out of scope
for unit tests, so each test downloads only the safetensor shards it needs via the index file
and dequantizes FP8 weights with their ``weight_scale_inv`` tensors when present.

By default helpers **download from the Hub** when tensors are not already cached. Tests should
call :func:`require_text_config` and :func:`require_hf_weights` (or the layer/model wrappers),
which invoke ``pytest.skip`` if config or weight download fails.

Set ``DEVSTRAL2_HF_LOCAL_ONLY=1`` to forbid network access (CI with a pre-populated HF cache).
"""

from __future__ import annotations

import json
import os
from typing import NoReturn

import pytest
import torch
from transformers import AutoConfig
from transformers.models.ministral3.configuration_ministral3 import Ministral3Config

from models.experimental.devstral2_123B_instruct.tt.weight_loading import DEVSTRAL2_LARGE_REPO_ID

# Fixed KV/RoPE budget for device tests so weight cache paths align (``seq_{max_seq_len}``).
DEVSTRAL2_TEST_MAX_SEQ_LEN = 98304


def devstral2_weight_cache_seq_len() -> int:
    """On-disk tiled-weight cache key (``seq_{N}``); default 256K from demo/agent runs."""
    return int(os.getenv("DEVSTRAL2_WEIGHT_CACHE_SEQ_LEN", "262144"))


def resolve_devstral2_weight_cache_path(mesh_device, text_cfg: Ministral3Config, num_layers: int) -> str:
    """Path to ``…/layers_{N}/seq_{devstral2_weight_cache_seq_len()}/`` for reusing tiled weights."""
    from models.experimental.devstral2_123B_instruct.tt.model_args import Devstral2Args
    from models.experimental.devstral2_123B_instruct.tt.weight_loading import resolve_weight_cache_path

    cache_args = Devstral2Args.from_hf_config(
        text_cfg,
        mesh_shape=tuple(mesh_device.shape),
        max_seq_len=devstral2_weight_cache_seq_len(),
        max_batch_size=1,
    )
    path = resolve_weight_cache_path(None, cache_args, num_layers=num_layers)
    assert path is not None
    return path


_FP8_DTYPES = tuple(
    dt for name in ("float8_e4m3fn", "float8_e5m2", "float8_e4m3fnuz") if (dt := getattr(torch, name, None)) is not None
)


def _hf_local_files_only() -> bool:
    """When true, only use files already in the HF cache (no Hub download)."""
    return os.getenv("DEVSTRAL2_HF_LOCAL_ONLY", "").lower() in ("1", "true", "yes")


def _skip_download_failure(what: str, exc: BaseException) -> NoReturn:
    pytest.skip(
        f"Could not download {what} from {DEVSTRAL2_LARGE_REPO_ID} "
        f"(set HF_TOKEN if gated, or pre-cache weights). Error: {exc}"
    )
    raise AssertionError("unreachable")  # pytest.skip always raises; satisfies static analysis


def require_text_config() -> Ministral3Config:
    """Load HF config, downloading if needed; ``pytest.skip`` on failure."""
    try:
        return load_text_config()
    except Exception as exc:
        _skip_download_failure("Ministral3Config", exc)


def require_hf_weights(keys: list[str]) -> dict[str, torch.Tensor]:
    """Download/dequantize the given state-dict keys; ``pytest.skip`` on failure."""
    try:
        return load_hf_tensors_for_keys(keys)
    except Exception as exc:
        _skip_download_failure(f"weights {keys!r}", exc)


def require_layer_weights(layer_idx: int) -> dict[str, torch.Tensor]:
    """Download all tensors for one decoder layer."""
    return require_hf_weights(layer_decoder_weight_keys(layer_idx))


def require_attention_weights(layer_idx: int = 0) -> dict[str, torch.Tensor]:
    p = f"model.layers.{layer_idx}.self_attn"
    return require_hf_weights(
        [
            f"{p}.q_proj.weight",
            f"{p}.k_proj.weight",
            f"{p}.v_proj.weight",
            f"{p}.o_proj.weight",
        ]
    )


def require_mlp_weights(layer_idx: int = 0) -> dict[str, torch.Tensor]:
    p = f"model.layers.{layer_idx}.mlp"
    return require_hf_weights(
        [
            f"{p}.gate_proj.weight",
            f"{p}.up_proj.weight",
            f"{p}.down_proj.weight",
        ]
    )


def require_model_weights(num_layers: int) -> dict[str, torch.Tensor]:
    """Download embed + ``num_layers`` decoder blocks + final norm."""
    return require_hf_weights(model_prefill_weight_keys(num_layers))


def load_text_config() -> Ministral3Config:
    """Download (cached) the HF config and return the inner ``Ministral3Config``.

    The Devstral-2 repo may publish a multimodal wrapper config; the underlying text stack is
    ``Ministral3Config`` either way. We always return the text config so the rest of the test
    can treat it uniformly.
    """
    hf_cfg = AutoConfig.from_pretrained(
        DEVSTRAL2_LARGE_REPO_ID,
        trust_remote_code=True,
        local_files_only=_hf_local_files_only(),
    )
    text = getattr(hf_cfg, "text_config", None) or hf_cfg
    if not isinstance(text, Ministral3Config):
        raise TypeError(f"Expected Ministral3Config, got {type(text)!r}")
    return text


def dequantize_fp8_weight(weight: torch.Tensor, scale_inv: torch.Tensor) -> torch.Tensor:
    """Dequant matching HF ``Fp8Dequantize._dequantize_one`` (scalar or per-block ``weight_scale_inv``)."""
    if scale_inv.numel() == 1:
        out_dtype = (
            scale_inv.dtype if scale_inv.dtype.is_floating_point and scale_inv.element_size() >= 2 else torch.bfloat16
        )
        return (weight.to(torch.float32) * scale_inv.to(torch.float32)).to(out_dtype)

    quantized_fp32 = weight.to(torch.float32)
    rows, cols = quantized_fp32.shape[-2:]
    scale_rows, scale_cols = scale_inv.shape[-2:]
    if rows % scale_rows or cols % scale_cols:
        raise ValueError(f"Weight shape ({rows}, {cols}) not divisible by scale grid ({scale_rows}, {scale_cols}).")
    block_m = rows // scale_rows
    block_n = cols // scale_cols
    out_dtype = (
        scale_inv.dtype if scale_inv.dtype.is_floating_point and scale_inv.element_size() >= 2 else torch.bfloat16
    )
    original_shape = quantized_fp32.shape
    q = quantized_fp32.reshape(-1, scale_rows, block_m, scale_cols, block_n)
    s = scale_inv.to(torch.float32).reshape(-1, scale_rows, scale_cols).unsqueeze(-1).unsqueeze(2)
    return (q * s).to(out_dtype).reshape(original_shape)


def to_bf16_host_if_fp8(t: torch.Tensor, scale_inv: torch.Tensor | None = None) -> torch.Tensor:
    """FP8 → bf16 on host. Uses block scales when ``scale_inv`` is provided."""
    if scale_inv is not None and _FP8_DTYPES and t.dtype in _FP8_DTYPES:
        return dequantize_fp8_weight(t, scale_inv).to(torch.bfloat16)
    if _FP8_DTYPES and t.dtype in _FP8_DTYPES:
        return t.to(torch.bfloat16)
    return t


def layer_decoder_weight_keys(layer_idx: int) -> list[str]:
    """HF state-dict keys for one ``Ministral3DecoderLayer`` (with ``model.`` prefix)."""
    p = f"model.layers.{layer_idx}"
    return [
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


def model_prefill_weight_keys(num_layers: int) -> list[str]:
    """Weights for ``TtMinistral3Model`` prefill PCC (embed + ``num_layers`` decoder blocks + final norm)."""
    keys = ["model.embed_tokens.weight", "model.norm.weight"]
    for layer_idx in range(num_layers):
        keys.extend(layer_decoder_weight_keys(layer_idx))
    return keys


def load_hf_tensors_for_keys(keys: list[str]) -> dict[str, torch.Tensor]:
    """Download shards for ``keys`` and return host tensors (weights dequantized to bf16 when FP8)."""
    from huggingface_hub import hf_hub_download
    from safetensors.torch import safe_open as safetensors_safe_open

    index_path = hf_hub_download(
        repo_id=DEVSTRAL2_LARGE_REPO_ID,
        filename="model.safetensors.index.json",
        local_files_only=_hf_local_files_only(),
    )
    with open(index_path, encoding="utf-8") as f:
        weight_map = json.load(f)["weight_map"]

    fetch_keys: set[str] = set(keys)
    for key in keys:
        if key.endswith(".weight"):
            scale_key = key[: -len(".weight")] + ".weight_scale_inv"
            if scale_key in weight_map:
                fetch_keys.add(scale_key)

    raw: dict[str, torch.Tensor] = {}
    for key in fetch_keys:
        if key not in weight_map:
            if key in keys:
                raise KeyError(f"Key {key!r} not in weight_map for {DEVSTRAL2_LARGE_REPO_ID}")
            continue
        shard_path = hf_hub_download(
            repo_id=DEVSTRAL2_LARGE_REPO_ID,
            filename=weight_map[key],
            local_files_only=_hf_local_files_only(),
        )
        with safetensors_safe_open(shard_path, framework="pt", device="cpu") as sf:
            if key not in sf.keys():
                raise KeyError(f"Key {key!r} missing from shard {weight_map[key]}")
            raw[key] = sf.get_tensor(key).clone()

    out: dict[str, torch.Tensor] = {}
    for key in keys:
        t = raw[key]
        if key.endswith(".weight"):
            scale_key = key[: -len(".weight")] + ".weight_scale_inv"
            scale = raw.get(scale_key)
            out[key] = to_bf16_host_if_fp8(t, scale).to(torch.bfloat16)
        else:
            out[key] = t
    return out


def load_ministral3_decoder_layer_weights(
    layer: "Ministral3DecoderLayer",
    state_dict: dict[str, torch.Tensor],
    layer_idx: int,
) -> None:
    """Copy ``model.layers.<i>.*`` tensors from ``state_dict`` into a HF decoder layer."""
    from transformers.models.ministral3.modeling_ministral3 import Ministral3DecoderLayer

    if not isinstance(layer, Ministral3DecoderLayer):
        raise TypeError(f"Expected Ministral3DecoderLayer, got {type(layer)!r}")
    p = f"model.layers.{layer_idx}"
    layer.input_layernorm.weight.data.copy_(state_dict[f"{p}.input_layernorm.weight"])
    layer.post_attention_layernorm.weight.data.copy_(state_dict[f"{p}.post_attention_layernorm.weight"])
    attn = layer.self_attn
    attn.q_proj.weight.data.copy_(state_dict[f"{p}.self_attn.q_proj.weight"])
    attn.k_proj.weight.data.copy_(state_dict[f"{p}.self_attn.k_proj.weight"])
    attn.v_proj.weight.data.copy_(state_dict[f"{p}.self_attn.v_proj.weight"])
    attn.o_proj.weight.data.copy_(state_dict[f"{p}.self_attn.o_proj.weight"])
    mlp = layer.mlp
    mlp.gate_proj.weight.data.copy_(state_dict[f"{p}.mlp.gate_proj.weight"])
    mlp.up_proj.weight.data.copy_(state_dict[f"{p}.mlp.up_proj.weight"])
    mlp.down_proj.weight.data.copy_(state_dict[f"{p}.mlp.down_proj.weight"])


def load_ministral3_model_weights(
    model: "Ministral3Model",
    state_dict: dict[str, torch.Tensor],
) -> None:
    """Copy downloaded weights into a HF ``Ministral3Model`` (layer count must match)."""
    from transformers.models.ministral3.modeling_ministral3 import Ministral3Model

    if not isinstance(model, Ministral3Model):
        raise TypeError(f"Expected Ministral3Model, got {type(model)!r}")
    model.embed_tokens.weight.data.copy_(state_dict["model.embed_tokens.weight"])
    model.norm.weight.data.copy_(state_dict["model.norm.weight"])
    for layer_idx, layer in enumerate(model.layers):
        load_ministral3_decoder_layer_weights(layer, state_dict, layer_idx)


def replicated_tt_to_torch(tensor: "ttnn.Tensor", *, reshape: tuple[int, ...] | None = None) -> torch.Tensor:
    """Read a replicated mesh tensor from one device (post all-reduce activations)."""
    import ttnn

    tt = ttnn.to_torch(ttnn.get_device_tensors(tensor)[0])
    if tt.ndim == 4:
        tt = tt[0:1]
    if reshape is not None:
        tt = tt.reshape(*reshape)
    return tt
