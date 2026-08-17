# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
HuggingFace reference / weight-loading boundary for the Qwen3.6-35B-A3B functional decoder.

Everything in this module is *test and setup* scope: it imports torch, reads the HF
checkpoint and runs the HF decoder layer to produce golden tensors. The TTNN runtime path
(``functional_decoder.py`` prefill/decode) never imports it.

Responsibilities
----------------
* resolve the local HF snapshot and text config,
* extract a single decoder layer's real state dict out of the sharded safetensors without
  materialising the whole 67 GiB checkpoint,
* record per-tensor stats and regenerate deterministic *synthetic* weights with the real
  shapes from those stats (so CI tests need no checkpoint),
* build an HF ``Qwen3_5MoeDecoderLayer`` and drive it for prefill / decode goldens,
  including a cheap "tail chunk against a pre-filled KV cache" mode that makes a
  262144-token full-attention reference tractable on CPU.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoConfig
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeDecoderLayer,
    Qwen3_5MoeTextRotaryEmbedding,
    apply_rotary_pos_emb,
)

HF_MODEL_ID = "Qwen/Qwen3.6-35B-A3B"

#: layer index used to exercise each decoder layer kind. Layer types repeat with period 4
#: ("linear, linear, linear, full"), so these two indices cover every distinct kind.
LINEAR_ATTENTION_LAYER_IDX = 0
FULL_ATTENTION_LAYER_IDX = 3

AUTOPORT_DIR = Path(__file__).resolve().parents[1]
WEIGHT_STATS_DIR = AUTOPORT_DIR / "doc" / "functional_decoder" / "weight_stats"

_LAYER_KEY_PREFIX = "model.language_model.layers.{idx}."


# ---------------------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------------------
def hf_snapshot_dir() -> Path:
    """Local snapshot dir for HF_MODEL_ID, or raise with an actionable message."""
    override = os.environ.get("QWEN36_35B_A3B_SNAPSHOT")
    if override:
        return Path(override)
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(HF_MODEL_ID, local_files_only=True))


def load_hf_text_config() -> Qwen3_5MoeTextConfig:
    """The *text* decoder config (``config.json -> text_config``)."""
    cfg = AutoConfig.from_pretrained(str(hf_snapshot_dir()))
    text_config = cfg.text_config
    # The decoder layer builds its attention with the config's attn implementation.
    text_config._attn_implementation = "sdpa"
    return text_config


def layer_kind(hf_config: Qwen3_5MoeTextConfig, layer_idx: int) -> str:
    return hf_config.layer_types[layer_idx]


# ---------------------------------------------------------------------------------------
# real weights
# ---------------------------------------------------------------------------------------
def real_layer_state_dict(layer_idx: int, dtype: torch.dtype = torch.float32) -> dict[str, torch.Tensor]:
    """Load one decoder layer's real weights, keyed relative to the layer.

    Only the shards holding this layer are opened, and only this layer's tensors are read,
    so peak host memory is ~one layer (~1.7 GiB in bf16, ~3.4 GiB upcast to fp32) rather
    than the full 67 GiB checkpoint. Reading per-key from the sharded index is what makes
    a full-model download unnecessary for stats collection.
    """
    from safetensors import safe_open

    snapshot = hf_snapshot_dir()
    index = json.loads((snapshot / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    prefix = _LAYER_KEY_PREFIX.format(idx=layer_idx)
    keys = sorted(k for k in weight_map if k.startswith(prefix))
    if not keys:
        raise KeyError(f"no checkpoint tensors found for layer {layer_idx} (prefix {prefix!r})")

    by_shard: dict[str, list[str]] = {}
    for key in keys:
        by_shard.setdefault(weight_map[key], []).append(key)

    state_dict: dict[str, torch.Tensor] = {}
    for shard, shard_keys in by_shard.items():
        with safe_open(str(snapshot / shard), framework="pt") as handle:
            for key in shard_keys:
                state_dict[key[len(prefix) :]] = handle.get_tensor(key).to(dtype)
    return state_dict


# ---------------------------------------------------------------------------------------
# weight stats <-> synthetic weights
# ---------------------------------------------------------------------------------------
def weight_stats(state_dict: dict[str, torch.Tensor]) -> dict[str, dict]:
    """name -> {shape, dtype, mean, std, min, max} for every tensor in the layer."""
    stats = {}
    for name, tensor in sorted(state_dict.items()):
        flat = tensor.detach().float().reshape(-1)
        stats[name] = {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype).replace("torch.", ""),
            "mean": float(flat.mean()),
            "std": float(flat.std()),
            "min": float(flat.min()),
            "max": float(flat.max()),
            "numel": int(flat.numel()),
        }
    return stats


def weight_stats_path(layer_idx: int) -> Path:
    return WEIGHT_STATS_DIR / f"layer_{layer_idx:02d}.json"


def save_weight_stats(layer_idx: int, stats: dict[str, dict]) -> Path:
    WEIGHT_STATS_DIR.mkdir(parents=True, exist_ok=True)
    path = weight_stats_path(layer_idx)
    path.write_text(json.dumps({"hf_model_id": HF_MODEL_ID, "layer_idx": layer_idx, "tensors": stats}, indent=2))
    return path


def load_weight_stats(layer_idx: int) -> dict[str, dict]:
    path = weight_stats_path(layer_idx)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing. Regenerate with "
            f"pytest models/autoports/qwen_qwen3_6_35b_a3b/tests/test_weight_stats.py"
        )
    return json.loads(path.read_text())["tensors"]


def synthetic_layer_state_dict(
    layer_idx: int, stats: dict[str, dict] | None = None, seed: int = 0, dtype: torch.dtype = torch.float32
) -> dict[str, torch.Tensor]:
    """Deterministic synthetic weights with the *real* shapes and per-tensor statistics.

    Each tensor gets its own generator seeded from ``seed`` and the tensor name, so adding
    or reordering tensors never perturbs the others. Values are drawn N(mean, std) and
    clamped into the recorded [min, max] range, which matters for ``A_log`` (log-uniform)
    and ``dt_bias`` (constant) where an unclamped normal would produce out-of-range gates.
    """
    stats = stats if stats is not None else load_weight_stats(layer_idx)
    out: dict[str, torch.Tensor] = {}
    for name, meta in sorted(stats.items()):
        gen = torch.Generator().manual_seed((seed * 1_000_003 + _stable_hash(name)) % (2**31))
        shape = tuple(meta["shape"])
        if meta["std"] == 0.0:
            tensor = torch.full(shape, meta["mean"], dtype=torch.float32)
        else:
            tensor = torch.normal(meta["mean"], meta["std"], size=shape, generator=gen)
            tensor = tensor.clamp(meta["min"], meta["max"])
        out[name] = tensor.to(dtype)
    return out


def _stable_hash(text: str) -> int:
    h = 1469598103934665603
    for byte in text.encode():
        h = ((h ^ byte) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


def synthetic_hidden_states(
    hf_config: Qwen3_5MoeTextConfig, batch: int, seq_len: int, seed: int = 0, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Activations shaped like what actually enters a decoder layer.

    Qwen3.6 embeddings feed straight into an RMSNorm, so the interesting property is the
    per-token RMS rather than the raw scale. Real layer-0 input RMS is ~1 per channel; we
    draw unit-normal and leave it there instead of using an arbitrary large scale.
    """
    gen = torch.Generator().manual_seed(1234 + seed)
    return torch.randn(batch, seq_len, hf_config.hidden_size, generator=gen, dtype=torch.float32).to(dtype)


# ---------------------------------------------------------------------------------------
# HF layer construction / driving
# ---------------------------------------------------------------------------------------
def build_hf_layer(
    hf_config: Qwen3_5MoeTextConfig,
    layer_idx: int,
    state_dict: dict[str, torch.Tensor],
    dtype: torch.dtype = torch.float32,
) -> Qwen3_5MoeDecoderLayer:
    """Instantiate the real HF decoder layer and load ``state_dict`` strictly."""
    with torch.device("meta"):
        layer = Qwen3_5MoeDecoderLayer(hf_config, layer_idx)
    layer.to_empty(device="cpu")
    missing, unexpected = layer.load_state_dict({k: v.to(dtype) for k, v in state_dict.items()}, strict=False)
    if unexpected:
        raise KeyError(f"unexpected keys for layer {layer_idx}: {unexpected}")
    if missing:
        raise KeyError(f"missing keys for layer {layer_idx}: {missing}")
    layer.eval()
    return layer.to(dtype)


def rope_cos_sin(
    hf_config: Qwen3_5MoeTextConfig, positions: torch.Tensor, dtype: torch.dtype = torch.float32
) -> tuple[torch.Tensor, torch.Tensor]:
    """HF text RoPE cos/sin for ``positions`` (``[batch, seq]`` int tensor).

    Uses HF's own ``Qwen3_5MoeTextRotaryEmbedding`` with all three mRoPE rows set to the
    text positions, which is exactly what ``Qwen3_5MoeTextModel.forward`` does for
    text-only input. Returns ``[batch, seq, rotary_dim]`` where ``rotary_dim`` is
    ``head_dim * partial_rotary_factor`` (64 here).
    """
    rope = Qwen3_5MoeTextRotaryEmbedding(hf_config).to(dtype)
    dummy = torch.zeros(positions.shape[0], positions.shape[1], hf_config.hidden_size, dtype=dtype)
    pos3 = positions[None, ...].expand(3, positions.shape[0], positions.shape[1]).contiguous()
    return rope(dummy, pos3)


def rotary_dim(hf_config: Qwen3_5MoeTextConfig) -> int:
    return int(hf_config.head_dim * hf_config.rope_parameters.get("partial_rotary_factor", 1.0))


@dataclass
class HfPrefillGolden:
    """HF golden for one prefill call."""

    output: torch.Tensor  # [batch, seq, hidden]
    cache: DynamicCache


def make_cache(hf_config: Qwen3_5MoeTextConfig) -> DynamicCache:
    return DynamicCache(config=hf_config)


def hf_prefill(
    layer: Qwen3_5MoeDecoderLayer,
    hf_config: Qwen3_5MoeTextConfig,
    hidden_states: torch.Tensor,
    *,
    start_pos: int = 0,
    cache: DynamicCache | None = None,
) -> HfPrefillGolden:
    """Run the HF decoder layer over ``hidden_states`` starting at absolute ``start_pos``.

    The cache is threaded through so a long prefill can be issued in several calls, which
    is also how the equivalent TTNN chunk-continuation path is validated.
    """
    batch, seq_len, _ = hidden_states.shape
    cache = cache if cache is not None else make_cache(hf_config)
    positions = torch.arange(start_pos, start_pos + seq_len).view(1, -1).expand(batch, -1)
    cos, sin = rope_cos_sin(hf_config, positions, dtype=hidden_states.dtype)
    mask = _additive_causal_mask(seq_len, start_pos + seq_len, start_pos, hidden_states.dtype)
    with torch.no_grad():
        out = layer(
            hidden_states,
            position_embeddings=(cos, sin),
            attention_mask=mask if layer.layer_type == "full_attention" else None,
            position_ids=positions,
            past_key_values=cache,
        )
    return HfPrefillGolden(output=out, cache=cache)


def hf_decode(
    layer: Qwen3_5MoeDecoderLayer,
    hf_config: Qwen3_5MoeTextConfig,
    hidden_states: torch.Tensor,
    *,
    positions: torch.Tensor,
    cache: DynamicCache,
) -> torch.Tensor:
    """One decode step. ``hidden_states`` is ``[batch, 1, hidden]``, ``positions`` ``[batch]``.

    HF's ``DynamicCache`` appends one token per batch row at a single shared length, so
    per-row positions are only honoured through RoPE; tests that use ragged positions keep
    the cache lengths equal (the TTNN side is what carries genuinely ragged ``cur_pos``).
    """
    batch = hidden_states.shape[0]
    pos2d = positions.view(batch, 1)
    cos, sin = rope_cos_sin(hf_config, pos2d, dtype=hidden_states.dtype)
    with torch.no_grad():
        return layer(
            hidden_states,
            position_embeddings=(cos, sin),
            attention_mask=None,  # single query token attends to everything cached
            position_ids=pos2d,
            past_key_values=cache,
        )


def _additive_causal_mask(q_len: int, kv_len: int, offset: int, dtype: torch.dtype) -> torch.Tensor:
    """[1, 1, q_len, kv_len] additive mask; query i (absolute offset+i) sees keys <= offset+i."""
    q_idx = torch.arange(q_len).view(-1, 1) + offset
    k_idx = torch.arange(kv_len).view(1, -1)
    allowed = k_idx <= q_idx
    mask = torch.zeros(q_len, kv_len, dtype=dtype)
    mask.masked_fill_(~allowed, torch.finfo(dtype).min)
    return mask.view(1, 1, q_len, kv_len)


# ---------------------------------------------------------------------------------------
# cheap long-context reference for full_attention layers
# ---------------------------------------------------------------------------------------
def hf_fill_full_attention_cache(
    layer: Qwen3_5MoeDecoderLayer,
    hf_config: Qwen3_5MoeTextConfig,
    hidden_states: torch.Tensor,
    *,
    start_pos: int = 0,
    cache: DynamicCache | None = None,
    chunk: int = 8192,
) -> DynamicCache:
    """Populate a ``DynamicCache`` with the K/V the HF layer *would* produce, in O(seq).

    Calls the layer's own ``k_proj``/``k_norm``/``v_proj`` modules and HF's own
    ``apply_rotary_pos_emb``, i.e. exactly the lines of ``Qwen3_5MoeAttention.forward``
    that produce cache entries — it skips only the O(seq^2) attention that a full HF
    prefill would spend on query positions the test does not compare. Used to make a
    262144-token reference tractable on CPU.
    """
    attn = layer.self_attn
    batch, seq_len, _ = hidden_states.shape
    cache = cache if cache is not None else make_cache(hf_config)
    head_dim = hf_config.head_dim
    with torch.no_grad():
        for lo in range(0, seq_len, chunk):
            hi = min(lo + chunk, seq_len)
            h = layer.input_layernorm(hidden_states[:, lo:hi])
            shape = (batch, hi - lo, -1, head_dim)
            k = attn.k_norm(attn.k_proj(h).view(shape)).transpose(1, 2)
            v = attn.v_proj(h).view(shape).transpose(1, 2)
            positions = torch.arange(start_pos + lo, start_pos + hi).view(1, -1).expand(batch, -1)
            cos, sin = rope_cos_sin(hf_config, positions, dtype=hidden_states.dtype)
            _, k = apply_rotary_pos_emb(k, k, cos, sin)
            cache.update(k, v, attn.layer_idx)
    return cache


def hf_prefill_tail(
    layer: Qwen3_5MoeDecoderLayer,
    hf_config: Qwen3_5MoeTextConfig,
    hidden_states: torch.Tensor,
    *,
    tail: int,
) -> torch.Tensor:
    """HF golden for only the last ``tail`` query positions of a full-attention prefill.

    Exact for those positions: the cache is filled with every earlier K/V (see
    ``hf_fill_full_attention_cache``) and the layer then runs normally over the tail, so
    the tail queries attend to the complete context. Returns ``[batch, tail, hidden]``.
    """
    if layer.layer_type != "full_attention":
        raise ValueError("hf_prefill_tail only applies to full_attention layers")
    batch, seq_len, _ = hidden_states.shape
    head = seq_len - tail
    cache = hf_fill_full_attention_cache(layer, hf_config, hidden_states[:, :head])
    golden = hf_prefill(layer, hf_config, hidden_states[:, head:], start_pos=head, cache=cache)
    return golden.output


def hf_linear_attention_chunked(
    layer: Qwen3_5MoeDecoderLayer,
    hf_config: Qwen3_5MoeTextConfig,
    hidden_states: torch.Tensor,
    *,
    cache: DynamicCache | None = None,
    chunk: int = 2048,
) -> DynamicCache:
    """Advance the HF conv + recurrent state over ``hidden_states``, mixer only, in O(seq).

    Calls the layer's own ``input_layernorm`` and ``linear_attn`` (i.e. the exact lines of
    ``Qwen3_5MoeDecoderLayer.forward`` that touch the state) and drops the MoE, which cannot
    affect the state. Makes a 262144-token state reference tractable: the MoE is ~90% of a
    full HF prefill's CPU cost and contributes nothing here.
    """
    if layer.layer_type != "linear_attention":
        raise ValueError("hf_linear_attention_chunked only applies to linear_attention layers")
    cache = cache if cache is not None else make_cache(hf_config)
    with torch.no_grad():
        for lo in range(0, hidden_states.shape[1], chunk):
            piece = layer.input_layernorm(hidden_states[:, lo : lo + chunk])
            layer.linear_attn(hidden_states=piece, cache_params=cache, attention_mask=None)
    return cache


def hf_linear_prefill_tail(
    layer: Qwen3_5MoeDecoderLayer,
    hf_config: Qwen3_5MoeTextConfig,
    hidden_states: torch.Tensor,
    *,
    tail: int,
    chunk: int = 2048,
) -> tuple[torch.Tensor, DynamicCache]:
    """HF golden for the last ``tail`` positions of a long linear-attention prefill.

    Exact for those positions: the conv + recurrent state is advanced over every earlier
    token (see ``hf_linear_attention_chunked``) and the *full* layer then runs over the tail.
    Returns ``(output [batch, tail, hidden], cache after the tail)``.
    """
    head = hidden_states.shape[1] - tail
    cache = hf_linear_attention_chunked(layer, hf_config, hidden_states[:, :head], chunk=chunk)
    golden = hf_prefill(layer, hf_config, hidden_states[:, head:], start_pos=head, cache=cache)
    return golden.output, golden.cache


def hf_linear_attention_state(cache: DynamicCache, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(conv_states [b, conv_dim, 4], recurrent_states [b, 32, 128, 128]) from an HF cache."""
    entry = cache.layers[layer_idx]
    return entry.conv_states, entry.recurrent_states


def seed_hf_linear_attention_state(
    cache: DynamicCache,
    layer_idx: int,
    conv_state: torch.Tensor,
    recurrent_state: torch.Tensor,
) -> None:
    """Force an HF cache into a given linear-attention state (for random-state decode tests).

    ``conv_state`` is the 3-column TTNN state; HF keeps 4 columns whose oldest entry is
    provably dead (see ``tests/test_reference_math.py::test_hf_conv_state_oldest_column_is_dead``),
    so it is left-padded with zeros here.
    """
    entry = cache.layers[layer_idx]
    padded = torch.cat([torch.zeros_like(conv_state[..., :1]), conv_state], dim=-1)
    entry.update_conv_state(padded)
    entry.update_recurrent_state(recurrent_state)
    entry.has_previous_state = True


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    """Pearson correlation between two tensors, computed in float64."""
    x = a.detach().reshape(-1).double()
    y = b.detach().reshape(-1).double()
    x = x - x.mean()
    y = y - y.mean()
    denom = x.norm() * y.norm()
    if denom == 0:
        return 1.0 if torch.allclose(a.double(), b.double()) else 0.0
    value = float((x @ y) / denom)
    return value if not math.isnan(value) else 0.0
