# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Layer-only HuggingFace reference for ``meta-models/Muse-Glimmer-30B``.

Everything here is test-side PyTorch: the real ``transformers`` decoder-layer
class is instantiated directly (no full ``ForConditionalGeneration`` load) so
the TTNN implementation is compared against the actual HF math rather than a
re-implementation.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import torch
from transformers import AutoConfig
from transformers.cache_utils import DynamicCache
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.models.muse_glimmer.modeling_muse_glimmer import (
    MuseGlimmerTextDecoderLayer,
    MuseGlimmerTextRotaryEmbedding,
)

MODEL_ID = "meta-models/Muse-Glimmer-30B"

#: Weight suffixes the TTNN layer consumes, i.e. the canonical key contract.
LAYER_WEIGHT_SUFFIXES = (
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "pre_feedforward_layernorm.weight",
    "post_feedforward_layernorm.weight",
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "self_attn.gate_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
)

_STATS_PATH = Path(__file__).with_name("layer_weight_stats.json")


def layer_prefix(layer_idx: int) -> str:
    return f"model.language_model.layers.{layer_idx}"


@lru_cache(maxsize=1)
def hf_config():
    config = AutoConfig.from_pretrained(MODEL_ID, local_files_only=True)
    config.text_config._attn_implementation = "sdpa"
    return config


def text_config():
    return hf_config().text_config


@lru_cache(maxsize=1)
def weight_stats() -> dict:
    with open(_STATS_PATH) as handle:
        return json.load(handle)


def synthetic_state_dict(layer_idx: int, *, seed: int = 20260811) -> dict[str, torch.Tensor]:
    """Deterministic synthetic weights drawn from the real checkpoint's stats."""
    stats = weight_stats()["layers"][str(layer_idx)]
    generator = torch.Generator(device="cpu").manual_seed(seed + layer_idx)
    state_dict: dict[str, torch.Tensor] = {}
    for suffix in LAYER_WEIGHT_SUFFIXES:
        entry = stats[suffix]
        tensor = torch.normal(
            mean=entry["mean"],
            std=max(entry["std"], 1e-8),
            size=tuple(entry["shape"]),
            generator=generator,
            dtype=torch.float32,
        )
        state_dict[f"{layer_prefix(layer_idx)}.{suffix}"] = tensor.to(torch.bfloat16)
    return state_dict


def weights_snapshot_dir() -> Path:
    """Cache snapshot that actually holds the safetensors shards.

    ``refs/main`` can point at a metadata-only revision (e.g. a later
    config-only fetch), so resolve by looking for the weight index rather than
    trusting the default revision.
    """
    from huggingface_hub.constants import HF_HUB_CACHE

    repo = Path(HF_HUB_CACHE) / f"models--{MODEL_ID.replace('/', '--')}"
    candidates = sorted(repo.glob("snapshots/*/model.safetensors.index.json"))
    if not candidates:
        raise FileNotFoundError(
            f"no cached safetensors index for {MODEL_ID} under {repo}; real-weight tests need the checkpoint"
        )
    return candidates[0].parent


def real_state_dict(layer_idx: int) -> dict[str, torch.Tensor]:
    """Load exactly this layer's tensors out of the real safetensors shards."""
    from safetensors import safe_open

    snapshot = weights_snapshot_dir()
    index = json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"]
    state_dict: dict[str, torch.Tensor] = {}
    wanted: dict[str, list[str]] = {}
    for suffix in LAYER_WEIGHT_SUFFIXES:
        key = f"{layer_prefix(layer_idx)}.{suffix}"
        wanted.setdefault(index[key], []).append(key)
    for shard, keys in wanted.items():
        shard_path = snapshot / shard
        if not shard_path.exists():  # pragma: no cover - safetensors not cached
            raise FileNotFoundError(f"missing weight shard {shard_path}")
        with safe_open(str(shard_path), framework="pt") as handle:
            for key in keys:
                state_dict[key] = handle.get_tensor(key)
    return state_dict


def reference_layer(
    layer_idx: int, state_dict: dict[str, torch.Tensor], dtype: torch.dtype = torch.bfloat16
) -> MuseGlimmerTextDecoderLayer:
    """Instantiate the real HF decoder layer.

    ``dtype`` defaults to the checkpoint's bfloat16.  Passing ``torch.float32``
    gives a higher-precision control: comparing the BF16 TTNN layer against it
    rules out errors that are common-mode between two bfloat16 implementations.
    """
    config = text_config()
    layer = MuseGlimmerTextDecoderLayer(config, layer_idx=layer_idx).to(dtype).eval()
    prefix = f"{layer_prefix(layer_idx)}."
    local = {name.removeprefix(prefix): tensor.to(dtype) for name, tensor in state_dict.items()}
    layer.load_state_dict(local, strict=True)
    return layer


@lru_cache(maxsize=1)
def _rotary() -> MuseGlimmerTextRotaryEmbedding:
    return MuseGlimmerTextRotaryEmbedding(text_config()).eval()


def rope_embeddings(position_ids: torch.Tensor, dtype: torch.dtype = torch.bfloat16):
    dummy = torch.zeros(position_ids.shape[0], 1, 1, dtype=dtype)
    with torch.no_grad():
        return _rotary()(dummy, position_ids)


def uses_rope(layer_idx: int) -> bool:
    return bool(text_config().layer_rope_theta[layer_idx])


def is_sliding(layer_idx: int) -> bool:
    return text_config().layer_types[layer_idx] == "sliding_attention"


def _mask(layer_idx: int, *, inputs_embeds, attention_mask, past_key_values, position_ids):
    config = text_config()
    kwargs = {
        "config": config,
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
        "past_key_values": past_key_values,
        "position_ids": position_ids,
    }
    if is_sliding(layer_idx):
        return create_sliding_window_causal_mask(**kwargs)
    return create_causal_mask(**kwargs)


@torch.no_grad()
def reference_prefill(
    layer: MuseGlimmerTextDecoderLayer,
    layer_idx: int,
    hidden_states: torch.Tensor,
    *,
    past_key_values: DynamicCache | None = None,
    start_pos: int = 0,
) -> tuple[torch.Tensor, DynamicCache]:
    """Run the HF layer over a prompt, returning ``(output, cache)``."""
    if past_key_values is None:
        past_key_values = DynamicCache(config=text_config())
    batch, seq_len, _ = hidden_states.shape
    position_ids = (torch.arange(seq_len) + start_pos).unsqueeze(0).expand(batch, -1)
    mask = _mask(
        layer_idx,
        inputs_embeds=hidden_states,
        attention_mask=None,
        past_key_values=past_key_values,
        position_ids=position_ids,
    )
    position_embeddings = rope_embeddings(position_ids, hidden_states.dtype) if uses_rope(layer_idx) else None
    out = layer(
        hidden_states,
        position_embeddings=position_embeddings,
        attention_mask=mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
    )
    return out, past_key_values


@torch.no_grad()
def reference_decode(
    layer: MuseGlimmerTextDecoderLayer,
    layer_idx: int,
    hidden_states: torch.Tensor,
    *,
    past_key_values: DynamicCache,
    positions: torch.Tensor,
) -> torch.Tensor:
    """Decode one token per batch row at absolute ``positions`` ``[batch]``."""
    position_ids = positions.reshape(-1, 1)
    mask = _mask(
        layer_idx,
        inputs_embeds=hidden_states,
        attention_mask=None,
        past_key_values=past_key_values,
        position_ids=position_ids,
    )
    position_embeddings = rope_embeddings(position_ids, hidden_states.dtype) if uses_rope(layer_idx) else None
    return layer(
        hidden_states,
        position_embeddings=position_embeddings,
        attention_mask=mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
    )


def hidden_state_scale() -> float:
    """RMS of the activations entering a decoder layer.

    ``MuseGlimmerTextNormedEmbedding`` applies a scale-less RMSNorm to the token
    embeddings, so the tensor arriving at layer 0 has unit RMS per position.
    Deeper layers grow, but unit RMS is the right synthetic-input distribution
    for a layer-level harness (and is what the recorded stats assume).
    """
    return 1.0


def synthetic_hidden_states(batch: int, seq_len: int, *, seed: int = 1234) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    hidden = torch.randn(batch, seq_len, text_config().hidden_size, generator=generator, dtype=torch.float32)
    hidden = hidden * torch.rsqrt(hidden.pow(2).mean(-1, keepdim=True))
    return (hidden * hidden_state_scale()).to(torch.bfloat16)


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value is None else int(value)
