# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Hugging Face adaptor for the TTTv2 Llama-3.1-8B path."""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from pathlib import Path

import torch
from loguru import logger
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import ttnn


def nearest_multiple(value: int, multiple: int) -> int:
    return math.ceil(value / multiple) * multiple


@dataclass(frozen=True)
class RopeScaling:
    rope_type: str
    factor: float | None = None
    original_max_position_embeddings: int | None = None
    low_freq_factor: float = 1.0
    high_freq_factor: float = 4.0


def rope_scaling_model_factory(
    rope_scaling_params: dict | None, original_max_context_len: int | None = None
) -> RopeScaling | None:
    if rope_scaling_params is None:
        return None

    rope_type = rope_scaling_params.get("rope_type") or rope_scaling_params.get("type")
    if rope_type in ("default", "mrope"):
        logger.warning(
            f"Rope scaling type was set to {rope_type}, defaulting to no rope scaling as this rope type is not supported yet by TTTv2"
        )
        return None
    if rope_type not in ("linear", "llama3"):
        raise ValueError(f"Unsupported RoPE scaling type for Llama-3.1-8B TTTv2 path: {rope_type}")

    return RopeScaling(
        rope_type=rope_type,
        factor=rope_scaling_params.get("factor"),
        original_max_position_embeddings=rope_scaling_params.get(
            "original_max_position_embeddings", original_max_context_len
        ),
        low_freq_factor=rope_scaling_params.get("low_freq_factor", 1.0),
        high_freq_factor=rope_scaling_params.get("high_freq_factor", 4.0),
    )


def _permute_to_meta_format(cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos[:, : cos.shape[1] // 2]
    cos = torch.stack((cos, cos), dim=-1).flatten(-2)

    sin = sin[:, : sin.shape[1] // 2]
    sin = torch.stack((sin, sin), dim=-1).flatten(-2)

    return cos.unsqueeze(0).unsqueeze(0), sin.unsqueeze(0).unsqueeze(0)


def _gather_cos_sin(position_ids: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    position_id_expanded = position_ids.unsqueeze(1).expand(-1, cos.shape[-1])
    cos = cos.gather(0, position_id_expanded)
    sin = sin.gather(0, position_id_expanded)
    cos = torch.stack([cos, cos], dim=-1).flatten(-2).unsqueeze(0).unsqueeze(0)
    sin = torch.stack([sin, sin], dim=-1).flatten(-2).unsqueeze(0).unsqueeze(0)
    return cos, sin


def _llama3_scaled_inv_freq(freqs: torch.Tensor, scaling: RopeScaling) -> torch.Tensor:
    assert scaling.factor is not None
    assert scaling.original_max_position_embeddings is not None

    low_freq_wavelen = scaling.original_max_position_embeddings / scaling.low_freq_factor
    high_freq_wavelen = scaling.original_max_position_embeddings / scaling.high_freq_factor
    new_freqs = []
    for freq in freqs:
        wavelen = 2 * math.pi / freq
        if wavelen < high_freq_wavelen:
            new_freqs.append(freq)
        elif wavelen > low_freq_wavelen:
            new_freqs.append(freq / scaling.factor)
        else:
            smooth = (scaling.original_max_position_embeddings / wavelen - scaling.low_freq_factor) / (
                scaling.high_freq_factor - scaling.low_freq_factor
            )
            new_freqs.append((1 - smooth) * freq / scaling.factor + smooth * freq)
    return torch.tensor(new_freqs, dtype=freqs.dtype, device=freqs.device)


def compute_gather_cos_sin(
    dhead: int, end: int, theta: float, rope_scaling: RopeScaling | None
) -> tuple[torch.Tensor, torch.Tensor]:
    seq_len = end // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, dhead, 2).float() / dhead))

    if rope_scaling is None:
        t = torch.arange(seq_len, dtype=inv_freq.dtype)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return _permute_to_meta_format(emb.cos(), emb.sin())

    if rope_scaling.rope_type == "linear":
        assert rope_scaling.factor is not None
        inv_freq = inv_freq / rope_scaling.factor
    elif rope_scaling.rope_type == "llama3":
        inv_freq = _llama3_scaled_inv_freq(inv_freq, rope_scaling)
    else:
        raise ValueError(f"Unsupported RoPE scaling type for Llama-3.1-8B TTTv2 path: {rope_scaling.rope_type}")

    t = torch.arange(seq_len * 2.0)
    freqs = torch.outer(t, inv_freq).float()
    cos, sin = torch.cos(freqs), torch.sin(freqs)
    return _gather_cos_sin(torch.arange(seq_len), cos, sin)


def should_pad_sampling_logits_to_power_of_2(padded_vocab_size: int, sampling_splits: int) -> bool:
    if sampling_splits < 1:
        return False
    per_device_vocab = padded_vocab_size // sampling_splits
    return per_device_vocab > 0 and (per_device_vocab & (per_device_vocab - 1)) != 0


def resolve_hf_model_id(hf_model: str | None = None) -> str:
    hf_model = hf_model or os.getenv("HF_MODEL")
    if not hf_model:
        raise ValueError("Please set HF_MODEL to a HuggingFace name e.g. meta-llama/Llama-3.1-8B-Instruct")
    return hf_model


def _replace_keys(state_dict, replacements):
    output = {}
    for key, value in state_dict.items():
        new_key = key
        for pattern, repl in replacements:
            new_key = re.sub(pattern, repl, new_key)
        output[new_key] = value
    return output


def _standardize_hf_keys(state_dict):
    key_meta = "lm_head.weight"
    key_hf = "model.embed_tokens.weight"
    if key_meta not in state_dict and key_hf in state_dict:
        state_dict[key_meta] = state_dict[key_hf]
        del state_dict[key_hf]
    return state_dict


def _split_hf_keys(loaded_weights, n_heads=None, n_kv_heads=None):
    converted_weights = {}
    for key, tensor in loaded_weights.items():
        if "qkv_proj" in key:
            q_key = key.replace("qkv_proj", "q_proj")
            k_key = key.replace("qkv_proj", "k_proj")
            v_key = key.replace("qkv_proj", "v_proj")
            if n_heads is not None and n_kv_heads is not None and n_heads != n_kv_heads:
                head_dim = tensor.shape[0] // (n_heads + 2 * n_kv_heads)
                q_size = n_heads * head_dim
                kv_size = n_kv_heads * head_dim
                q_tensor = tensor[:q_size]
                k_tensor = tensor[q_size : q_size + kv_size]
                v_tensor = tensor[q_size + kv_size : q_size + 2 * kv_size]
            else:
                q_tensor, k_tensor, v_tensor = torch.split(tensor, tensor.shape[0] // 3, dim=0)
            converted_weights[q_key] = q_tensor
            converted_weights[k_key] = k_tensor
            converted_weights[v_key] = v_tensor
        elif "gate_up_proj" in key:
            gate_key = key.replace("gate_up_proj", "gate_proj")
            up_key = key.replace("gate_up_proj", "up_proj")
            gate_tensor, up_tensor = torch.split(tensor, tensor.shape[0] // 2, dim=0)
            converted_weights[gate_key] = gate_tensor
            converted_weights[up_key] = up_tensor
        else:
            converted_weights[key] = tensor
    return converted_weights


def _reverse_permute(tensor, n_heads, dim1, dim2):
    return tensor.view(n_heads, 2, dim1 // n_heads // 2, dim2).transpose(1, 2).reshape(dim1, dim2)


def _reverse_permute_1d(tensor):
    dim = tensor.shape[-1]
    assert dim % 2 == 0, "Last dimension must be even"
    reals = tensor[..., : dim // 2]
    imags = tensor[..., dim // 2 :]
    return torch.stack((reals, imags), dim=-1).flatten(start_dim=len(tensor.shape) - 1)


def _convert_hf_qkv_to_meta_format(loaded_weights, head_dim):
    converted_weights = {}
    for key, tensor in loaded_weights.items():
        if "q_proj.weight" in key or "k_proj.weight" in key:
            n_heads = tensor.shape[0] // head_dim
            converted_weights[key] = _reverse_permute(tensor, n_heads, tensor.shape[0], tensor.shape[1])
        elif "q_proj.bias" in key or "k_proj.bias" in key:
            n_heads = tensor.shape[0] // head_dim
            converted_weights[key] = _reverse_permute(tensor, n_heads, tensor.shape[0], 1).squeeze(-1)
        elif "q_norm.weight" in key or "k_norm.weight" in key:
            converted_weights[key] = _reverse_permute_1d(tensor)
        else:
            converted_weights[key] = tensor
    return converted_weights


def _map_hf_to_meta_keys(loaded_weights):
    replacements = [
        ("^emb.weight", "weight"),
        ("model.", ""),
        ("embed_tokens", "tok_embeddings"),
        ("lm_head", "output"),
        ("input_layernorm", "attention_norm"),
        ("post_attention_layernorm", "ffn_norm"),
        ("self_attn", "attention"),
        ("mlp", "feed_forward"),
        ("gate_proj", "w1"),
        ("down_proj", "w2"),
        ("up_proj", "w3"),
        ("q_proj", "wq"),
        ("k_proj", "wk"),
        ("v_proj", "wv"),
        ("o_proj", "wo"),
        ("q_norm", "q_norm"),
        ("k_norm", "k_norm"),
    ]
    return _replace_keys(loaded_weights, replacements)


def convert_hf_state_dict_to_meta(state_dict, *, head_dim: int, n_heads: int, n_kv_heads: int):
    state_dict = _split_hf_keys(state_dict, n_heads, n_kv_heads)
    state_dict = _convert_hf_qkv_to_meta_format(state_dict, head_dim)
    return _map_hf_to_meta_keys(state_dict)


def _chat_template_ids(encoded):
    if hasattr(encoded, "keys") and "input_ids" in encoded:
        encoded = encoded["input_ids"]
    if hasattr(encoded, "ids"):
        return list(encoded.ids)
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if isinstance(encoded, (list, tuple)) and len(encoded) == 1 and isinstance(encoded[0], (list, tuple)):
        encoded = encoded[0]
    return list(encoded)


def _encode_prompt_with_chat_template(tokenizer, prompt_text, system_prompt_text=None):
    chat = []
    if isinstance(prompt_text, str):
        if system_prompt_text:
            chat.append({"role": "system", "content": system_prompt_text})
        if prompt_text:
            chat.append({"role": "user", "content": prompt_text})
        encoded = tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=True)
    else:
        encoded = tokenizer.apply_chat_template(prompt_text, add_generation_prompt=True, tokenize=True)
    return _chat_template_ids(encoded)


def encode_prompt(tokenizer, prompt_text, system_prompt_text=None, *, instruct=True):
    if instruct:
        try:
            return _encode_prompt_with_chat_template(tokenizer, prompt_text, system_prompt_text)
        except ValueError as exc:
            logger.warning(f"Failed to encode chat prompt, falling back to base encoding: {exc}")
    return tokenizer.encode(prompt_text, add_special_tokens=False)


def load_hf_model_info(
    hf_model: str,
    *,
    num_devices: int,
    cluster_shape: list[int],
    trust_remote_code: bool = False,
):
    hf_config = AutoConfig.from_pretrained(
        hf_model,
        trust_remote_code=trust_remote_code,
        local_files_only=os.getenv("CI") == "true",
    )
    config = hf_config.to_dict()
    text_config = config.get("text_config", config)
    dim = text_config.get("dim", text_config.get("hidden_size"))
    n_heads = text_config.get("n_heads", text_config.get("num_attention_heads"))
    n_kv_heads = text_config.get("n_kv_heads", text_config.get("num_key_value_heads"))
    n_layers = text_config.get("n_layers", text_config.get("num_hidden_layers"))
    vocab_size = text_config["vocab_size"]
    padded_vocab_size = nearest_multiple(vocab_size, ttnn.TILE_SIZE * num_devices)
    hidden_dim = text_config["intermediate_size"]
    model_name = os.path.basename(os.path.normpath(config["_name_or_path"])) if config.get("_name_or_path") else None
    sampling_splits = num_devices if cluster_shape != [1, 1] else 2
    rope_parameters = text_config.get("rope_parameters") or {}
    rope_scaling_params = text_config.get("rope_scaling")
    if not rope_scaling_params and rope_parameters.get("rope_type") not in (None, "default"):
        rope_scaling_params = rope_parameters
    original_max_context_len = text_config.get("original_max_position_embeddings", None)

    return {
        "hf_config": hf_config,
        "dim": dim,
        "n_heads": n_heads,
        "n_kv_heads": n_kv_heads,
        "n_layers": n_layers,
        "full_model_n_layers": n_layers,
        "norm_eps": text_config.get("norm_eps", text_config.get("rms_norm_eps")),
        "vocab_size": vocab_size,
        "padded_vocab_size": padded_vocab_size,
        "head_dim": text_config.get("head_dim", dim // n_heads) or dim // n_heads,
        "max_context_len": text_config.get("max_position_embeddings"),
        "hidden_dim": hidden_dim,
        "model_name": model_name,
        "pad_logits_to_power_of_2": cluster_shape != [1, 1]
        and should_pad_sampling_logits_to_power_of_2(padded_vocab_size, sampling_splits),
        "unpadded_hidden_dim": hidden_dim,
        "layer_types": text_config.get("layer_types", None),
        "sliding_window": text_config.get("sliding_window", None),
        "rope_theta": text_config.get("rope_theta") or rope_parameters.get("rope_theta"),
        "rope_theta_local": text_config.get("rope_local_base_freq"),
        "use_sliding_window": text_config.get("use_sliding_window", None),
        "rope_scaling_params": rope_scaling_params,
        "original_max_context_len": original_max_context_len,
        "rope_scaling": (
            rope_scaling_model_factory(rope_scaling_params, original_max_context_len) if rope_scaling_params else None
        ),
        "query_pre_attn_scalar": text_config.get("query_pre_attn_scalar", None),
        "mlp_activation_type": ttnn.UnaryOpType.SILU,
        "is_multimodal": False,
        "state_dict_text_prefix": "",
        "state_dict_vision_prefix": "visual.",
    }


def load_tokenizer(hf_model: str, *, trust_remote_code: bool = False):
    tokenizer = AutoTokenizer.from_pretrained(
        hf_model,
        local_files_only=os.getenv("CI") == "true",
        trust_remote_code=trust_remote_code,
    )
    if not hasattr(tokenizer, "stop_tokens") or tokenizer.stop_tokens is None:
        tokenizer.stop_tokens = [tokenizer.eos_token_id]
    return tokenizer


def load_converted_state_dict(
    hf_model: str,
    *,
    head_dim: int,
    n_heads: int,
    n_kv_heads: int,
    n_layers: int,
    trust_remote_code: bool = False,
):
    model = AutoModelForCausalLM.from_pretrained(
        hf_model,
        torch_dtype="auto",
        trust_remote_code=trust_remote_code,
        local_files_only=os.getenv("CI") == "true",
    )
    state_dict = model.state_dict()
    fuse_qkv = any("qkv" in layer_name for layer_name in state_dict)
    fuse_mlp = any("gate_up" in layer_name for layer_name in state_dict)
    state_dict = _standardize_hf_keys(state_dict)
    state_dict = convert_hf_state_dict_to_meta(
        state_dict,
        head_dim=head_dim,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
    )
    for key in list(state_dict.keys()):
        if "layers." in key:
            layer_num = int(key.split("layers.")[1].split(".")[0])
            if layer_num >= n_layers:
                state_dict.pop(key)
    return state_dict, fuse_qkv, fuse_mlp


def _device_name(mesh_device) -> str:
    num_devices = mesh_device.get_num_devices()
    dram_grid_size = mesh_device.dram_grid_size()
    if ttnn.device.is_blackhole(mesh_device):
        return {
            1: "P100" if dram_grid_size and dram_grid_size.x == 7 else "P150",
            2: "P300",
            4: "P150x4",
            8: "P150x8",
            32: "BHGLX",
        }[num_devices]
    if ttnn.device.is_wormhole_b0(mesh_device):
        return {1: "N150", 2: "N300", 4: "N150x4", 8: "T3K", 32: "TG"}[num_devices]
    raise ValueError(f"Unsupported architecture: {ttnn.get_arch_name()}")


def _model_cache_path(hf_model: str, mesh_device) -> Path:
    cache_path = os.getenv("TT_CACHE_PATH")
    if cache_path:
        return Path(cache_path) / _device_name(mesh_device)
    return Path("model_cache") / hf_model / _device_name(mesh_device)


def from_pretrained(
    mesh_device,
    *,
    hf_model: str | None = None,
    instruct: bool | None = None,
    max_batch_size: int,
    max_seq_len: int,
    optimizations="performance",
    n_layers: int | None = None,
    dtype=ttnn.bfloat8_b,
    paged_attention_config=None,
):
    """Build a TTTv2 Llama-3.1-8B model from an HF checkpoint."""
    from models.common.models.llama3_8b.model import (
        Llama3Transformer1D,
        build_llama3_transformer_1d_config,
        create_llama31_runtime_args,
    )

    hf_model = resolve_hf_model_id(hf_model)
    if instruct is None:
        instruct = "Instruct" in Path(hf_model).name

    hf_model_info = load_hf_model_info(
        hf_model,
        num_devices=mesh_device.get_num_devices(),
        cluster_shape=list(mesh_device.shape),
    )
    model_name = hf_model_info.get("model_name") or hf_model.strip("/").split("/")[-1]
    tokenizer = load_tokenizer(hf_model)

    rope_scaling = hf_model_info.get("rope_scaling")
    rope_cos, rope_sin = compute_gather_cos_sin(
        dhead=hf_model_info["head_dim"],
        end=2 * max_seq_len,
        theta=hf_model_info["rope_theta"],
        rope_scaling=rope_scaling,
    )
    model_info = {
        key: value
        for key, value in hf_model_info.items()
        if key
        not in {
            "hf_config",
            "rope_parameters",
            "rope_scaling",
            "rope_scaling_params",
            "original_max_context_len",
            "rope_theta",
            "rope_theta_local",
        }
    }
    model_info["rope_cos"] = rope_cos
    model_info["rope_sin"] = rope_sin

    def state_dict_loader(args):
        return load_converted_state_dict(
            hf_model,
            head_dim=args.head_dim,
            n_heads=args.n_heads,
            n_kv_heads=args.n_kv_heads,
            n_layers=args.n_layers,
        )

    def prompt_encoder(prompt_text, system_prompt_text=None, *, instruct=True):
        return encode_prompt(tokenizer, prompt_text, system_prompt_text, instruct=instruct)

    model_args = create_llama31_runtime_args(
        mesh_device=mesh_device,
        instruct=instruct,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        model_name=model_name,
        model_info=model_info,
        model_cache_path=_model_cache_path(hf_model, mesh_device),
        optimizations=optimizations,
        n_layers=n_layers,
        tokenizer=tokenizer,
        prompt_encoder=prompt_encoder,
        state_dict_loader=state_dict_loader,
    )
    if paged_attention_config is not None:
        model_args.paged_attention_config = paged_attention_config

    state_dict = model_args.load_state_dict()
    model_config = build_llama3_transformer_1d_config(
        mesh_device=mesh_device,
        args=model_args,
        state_dict=state_dict,
        weight_cache_path=model_args.weight_cache_path(dtype),
        dtype=dtype,
        paged_attention_config=paged_attention_config,
    )
    return Llama3Transformer1D(model_config), model_args
