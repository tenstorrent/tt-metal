# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Hugging Face adaptor for the TTTv2 Llama-3.1-8B path."""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import torch
from loguru import logger
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import ttnn
from models.common.device_utils import get_device_name
from models.common.tensor_utils import nearest_multiple


@dataclass(frozen=True)
class RopeScaling:
    rope_type: str
    factor: float
    original_max_position_embeddings: int
    low_freq_factor: float
    high_freq_factor: float


def llama3_rope_scaling(rope_parameters: dict) -> RopeScaling:
    rope_type = rope_parameters["rope_type"]
    if rope_type != "llama3":
        raise ValueError(f"Unsupported RoPE scaling type for Llama-3.1-8B TTTv2 path: {rope_type}")

    return RopeScaling(
        rope_type=rope_type,
        factor=rope_parameters["factor"],
        original_max_position_embeddings=rope_parameters["original_max_position_embeddings"],
        low_freq_factor=rope_parameters["low_freq_factor"],
        high_freq_factor=rope_parameters["high_freq_factor"],
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
    dhead: int, end: int, theta: float, rope_scaling: RopeScaling
) -> tuple[torch.Tensor, torch.Tensor]:
    seq_len = end // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, dhead, 2).float() / dhead))

    if rope_scaling.rope_type != "llama3":
        raise ValueError(f"Unsupported RoPE scaling type for Llama-3.1-8B TTTv2 path: {rope_scaling.rope_type}")
    inv_freq = _llama3_scaled_inv_freq(inv_freq, rope_scaling)

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


def load_tokenizer(hf_model: str, *, trust_remote_code: bool = False):
    tokenizer = AutoTokenizer.from_pretrained(
        hf_model,
        local_files_only=os.getenv("CI") == "true",
        trust_remote_code=trust_remote_code,
    )
    if not hasattr(tokenizer, "stop_tokens") or tokenizer.stop_tokens is None:
        tokenizer.stop_tokens = [tokenizer.eos_token_id]
    return tokenizer


@dataclass(frozen=True)
class Llama3GenerationConfig:
    """Text-generation defaults for the Llama 3.1-8B product model."""

    max_decode_tokens: int = 128
    temperature: float = 0.0
    top_k: int = 32
    top_p: float = 0.08
    stop_token_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class Llama3RuntimeConfig:
    """Executor/runtime metadata kept outside the tensor graph config."""

    model_name: str
    model_cache_path: Path
    max_prefill_chunk_size: int
    max_context_len: int
    trace_prefill_supported_seq_lens: tuple[int, ...] = (128, 1024)

    def can_enable_trace(self, prefill_seq_len, num_cached_tokens=0):
        return (
            num_cached_tokens == 0
            and prefill_seq_len in self.trace_prefill_supported_seq_lens
            and prefill_seq_len <= self.max_prefill_chunk_size
        )


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


@dataclass
class Llama3ForCausalLM:
    """Usable Llama 3.1-8B model product: tokenizer plus TT tensor model.

    The tokenizer interface is intentionally documented rather than enforced
    through a Protocol for now. The object must provide encode/decode behavior,
    EOS/stop token IDs, and chat-template application for instruct models.
    """

    model: object
    tokenizer: object
    runtime_config: Llama3RuntimeConfig
    instruct: bool
    generation_config: Llama3GenerationConfig = field(default_factory=Llama3GenerationConfig)

    def __post_init__(self):
        self.model.model_args = self.runtime_config
        if not self.generation_config.stop_token_ids:
            stop_tokens = tuple(getattr(self.tokenizer, "stop_tokens", []) or [])
            self.generation_config = Llama3GenerationConfig(
                max_decode_tokens=self.generation_config.max_decode_tokens,
                temperature=self.generation_config.temperature,
                top_k=self.generation_config.top_k,
                top_p=self.generation_config.top_p,
                stop_token_ids=stop_tokens,
            )

    @property
    def model_name(self):
        return self.runtime_config.model_name

    @property
    def model_cache_path(self):
        return self.runtime_config.model_cache_path

    @property
    def max_seq_len(self):
        return self.model.config.max_seq_len

    @property
    def max_context_len(self):
        return self.runtime_config.max_context_len

    def encode_prompt(self, prompt_text, system_prompt_text=None, instruct=None):
        use_instruct = self.instruct if instruct is None else instruct
        return encode_prompt(self.tokenizer, prompt_text, system_prompt_text, instruct=use_instruct)

    def encode_chat(self, messages):
        return self.encode_prompt(messages, instruct=True)


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
    return state_dict


def _model_cache_path(hf_model: str, mesh_device) -> Path:
    cache_path = os.getenv("TT_CACHE_PATH")
    if cache_path:
        return Path(cache_path) / get_device_name(mesh_device)
    return Path("model_cache") / hf_model / get_device_name(mesh_device)


def _max_prefill_chunk_size(mesh_device) -> int:
    override = os.getenv("MAX_PREFILL_CHUNK_SIZE")
    if override is not None:
        return int(override) * 1024
    return {"N150": 4, "N300": 64, "T3K": 128}[get_device_name(mesh_device)] * 1024


def _trace_prefill_supported_seq_lens(
    device_name: str, max_prefill_chunk_size: int, max_seq_len: int
) -> tuple[int, ...]:
    supported_seq_lens_by_device = {
        "N150": (128, 1024),
        "N300": (128, 1024, 2048, 4096, 8192),
        "T3K": (128, 1024, 2048, 4096, 8192),
    }
    supported_seq_lens = supported_seq_lens_by_device[device_name]
    return tuple(seq_len for seq_len in supported_seq_lens if seq_len <= min(max_prefill_chunk_size, max_seq_len))


def _weight_cache_path(model_cache_path: Path, *, instruct: bool, dtype):
    if instruct:
        return (
            model_cache_path
            / {
                ttnn.bfloat16: "tensor_cache_instruct_bf16",
                ttnn.bfloat8_b: "tensor_cache_instruct_bfp8",
            }[dtype]
        )
    return model_cache_path / {ttnn.bfloat16: "tensor_cache_bf16", ttnn.bfloat8_b: "tensor_cache_bfp8"}[dtype]


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
    """Build a product-level TTTv2 Llama-3.1-8B model from an HF checkpoint."""
    from models.common.models.llama3_8b.model import Llama3Transformer1D, build_llama3_transformer_1d_config

    hf_model = resolve_hf_model_id(hf_model)
    if instruct is None:
        instruct = "Instruct" in Path(hf_model).name

    hf_config = AutoConfig.from_pretrained(
        hf_model,
        local_files_only=os.getenv("CI") == "true",
    )
    text_config = hf_config.to_dict()
    model_name = Path(hf_model).name
    tokenizer = load_tokenizer(hf_model)
    num_hidden_layers = n_layers if n_layers is not None else text_config["num_hidden_layers"]

    rope_cos, rope_sin = compute_gather_cos_sin(
        dhead=text_config["hidden_size"] // text_config["num_attention_heads"],
        end=2 * max_seq_len,
        theta=text_config["rope_parameters"]["rope_theta"],
        rope_scaling=llama3_rope_scaling(text_config["rope_parameters"]),
    )
    model_cache_path = _model_cache_path(hf_model, mesh_device)

    model_config = build_llama3_transformer_1d_config(
        mesh_device=mesh_device,
        instruct=instruct,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        model_name=model_name,
        dim=text_config["hidden_size"],
        n_heads=text_config["num_attention_heads"],
        n_kv_heads=text_config["num_key_value_heads"],
        n_layers=num_hidden_layers,
        head_dim=text_config["hidden_size"] // text_config["num_attention_heads"],
        hidden_dim=text_config["intermediate_size"],
        vocab_size=text_config["vocab_size"],
        norm_eps=text_config["rms_norm_eps"],
        padded_vocab_size=nearest_multiple(text_config["vocab_size"], ttnn.TILE_SIZE * mesh_device.get_num_devices()),
        rope_cos=rope_cos,
        rope_sin=rope_sin,
        model_cache_path=model_cache_path,
        state_dict=load_converted_state_dict(
            hf_model,
            head_dim=text_config["hidden_size"] // text_config["num_attention_heads"],
            n_heads=text_config["num_attention_heads"],
            n_kv_heads=text_config["num_key_value_heads"],
            n_layers=num_hidden_layers,
        ),
        optimizations=optimizations,
        weight_cache_path=_weight_cache_path(model_cache_path, instruct=instruct, dtype=dtype),
        dtype=dtype,
        paged_attention_config=paged_attention_config,
        pad_logits_to_power_of_2=list(mesh_device.shape) != [1, 1]
        and should_pad_sampling_logits_to_power_of_2(
            nearest_multiple(text_config["vocab_size"], ttnn.TILE_SIZE * mesh_device.get_num_devices()),
            mesh_device.get_num_devices() if list(mesh_device.shape) != [1, 1] else 2,
        ),
    )
    max_prefill_chunk_size = _max_prefill_chunk_size(mesh_device)
    trace_prefill_supported_seq_lens = _trace_prefill_supported_seq_lens(
        get_device_name(mesh_device),
        max_prefill_chunk_size,
        max_seq_len,
    )
    runtime_config = Llama3RuntimeConfig(
        model_name=model_name,
        model_cache_path=model_cache_path,
        max_prefill_chunk_size=max_prefill_chunk_size,
        max_context_len=text_config["max_position_embeddings"],
        trace_prefill_supported_seq_lens=trace_prefill_supported_seq_lens,
    )
    return Llama3ForCausalLM(
        model=Llama3Transformer1D(model_config),
        tokenizer=tokenizer,
        runtime_config=runtime_config,
        instruct=instruct,
    )
