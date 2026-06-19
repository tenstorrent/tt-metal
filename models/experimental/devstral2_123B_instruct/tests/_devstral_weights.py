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
from pathlib import Path
from typing import NoReturn

import pytest
import torch
from loguru import logger
from transformers import AutoConfig
from transformers.models.ministral3.configuration_ministral3 import Ministral3Config

from models.experimental.devstral2_123B_instruct.tt.weight_loading import DEVSTRAL2_LARGE_REPO_ID

# Default on-disk cache key; prefer :func:`devstral2_test_max_seq_len` (respects env at runtime).
DEVSTRAL2_TEST_MAX_SEQ_LEN = 262144

# Longest teacher-forced eval window in E2E sweeps (logit PCC uses 10 decode steps).
DEVSTRAL2_E2E_MAX_POST_PREFILL_TOKENS = 500


def devstral2_weight_cache_seq_len() -> int:
    """On-disk tiled-weight cache key (``seq_{N}``); default 256K from demo/agent runs."""
    return int(os.getenv("DEVSTRAL2_WEIGHT_CACHE_SEQ_LEN", "262144"))


def devstral2_test_max_seq_len() -> int:
    """Runtime KV/RoPE budget for unit/PCC tests; matches the shared ``seq_{N}`` cache key."""
    return devstral2_weight_cache_seq_len()


def devstral2_tt_weight_cache_layer_count(text_cfg: Ministral3Config) -> int:
    """Full-model layer count for the shared cache directory (``layers_{N}``)."""
    return int(text_cfg.num_hidden_layers)


def devstral2_test_model_args(
    text_cfg: Ministral3Config,
    mesh_device=None,
    *,
    max_seq_len: int | None = None,
    max_batch_size: int = 1,
) -> "Devstral2Args":
    """``Devstral2Args`` with default ``max_seq_len`` aligned to the shared cache key."""
    from models.experimental.devstral2_123B_instruct.tt.model_args import Devstral2Args

    seq = max_seq_len if max_seq_len is not None else devstral2_test_max_seq_len()
    mesh_shape = tuple(mesh_device.shape) if mesh_device is not None else (1, 8)
    return Devstral2Args.from_hf_config(
        text_cfg,
        mesh_shape=mesh_shape,
        max_seq_len=seq,
        max_batch_size=max_batch_size,
    )


def devstral2_e2e_sweep_model_max_seq_len(*, max_post_prefill_tokens: int | None = None) -> int:
    """Single KV/RoPE budget and ``seq_{N}`` weight-cache key for all E2E sweep points.

    Sized for the worst prefill in the sweep (``devstral2_weight_cache_seq_len()``) plus
    post-prefill generation (500 teacher-forced eval tokens by default). Every sweep
    point reuses this budget so RoPE tables and tiled weights are not recompiled per
    input length.
    """
    from models.experimental.devstral2_123B_instruct.demo.text_demo import _round_up
    from models.experimental.devstral2_123B_instruct.tt.model_args import Devstral2Args

    post = DEVSTRAL2_E2E_MAX_POST_PREFILL_TOKENS if max_post_prefill_tokens is None else max_post_prefill_tokens
    kv_block = Devstral2Args.kv_block_size
    max_prefill = devstral2_weight_cache_seq_len()
    need = max_prefill + post + 1
    return _round_up(need, kv_block)


def devstral2_max_new_tokens() -> int:
    """Decode budget for tests — same as ``text_demo`` (``DEVSTRAL2_MAX_NEW_TOKENS``, default 100)."""
    return int(os.environ.get("DEVSTRAL2_MAX_NEW_TOKENS") or "100")


def devstral2_isl_perf_decode_replay_iters() -> int:
    """Decode trace replays to time (``max_new_tokens``, wall-clock capped)."""
    budget = devstral2_max_new_tokens()
    if budget <= 0:
        return 0
    cap = int(os.getenv("DEVSTRAL2_ISL_PERF_DECODE_REPLAY_CAP", "32"))
    if cap <= 0:
        return budget
    return min(budget, cap)


def devstral2_isl_perf_kv_max_seq_len(isl_lengths: list[int]) -> int:
    """Single KV/RoPE budget for an ISL sweep (``text_demo``-style, trace-safe alignment).

    For each sweep point: ``padded_ISL + max_new_tokens``, take the maximum, apply
    ``DEVSTRAL2_MIN_MAX_SEQ_LEN`` floor (default 262144), then round with
    ``_round_up_max_seq_len`` so logical KV blocks are a multiple of 8 (prefill trace safe).
    """
    from models.experimental.devstral2_123B_instruct.demo.text_demo import (
        _min_max_seq_len,
        _round_up,
        _round_up_max_seq_len,
    )
    from models.experimental.devstral2_123B_instruct.tt.model_args import Devstral2Args

    max_new = devstral2_max_new_tokens()
    kv_block = Devstral2Args.kv_block_size
    need = 0
    for isl in isl_lengths:
        num_chunks = max(1, (isl + kv_block - 1) // kv_block)
        padded_len = num_chunks * kv_block
        need = max(need, _round_up(padded_len + max_new, kv_block))
    # ``DEVSTRAL2_MIN_MAX_SEQ_LEN`` (default 262144) is a prefill budget floor, not a decode-safe
    # cap: a prompt at the floor still needs ``max_new_tokens`` slots after prefill (e.g. ISL=262144
    # decode starts at pos 262144 = logical KV block 2048, which does not exist when max_seq_len is
    # exactly 262144).
    floor = _round_up(_min_max_seq_len() + max_new, kv_block)
    return _round_up_max_seq_len(max(need, floor), kv_block)


def devstral2_tt_weight_cache_dir(mesh_device, text_cfg: Ministral3Config) -> str:
    """Shared on-disk TT weight cache for all Devstral-2 large tests.

    Always ``…/layers_{num_hidden_layers}/seq_{devstral2_weight_cache_seq_len()}/``.
    Matmul / norm / embed tiled weights are keyed only by mesh and layer count, not
    sweep input length. Partial-layer models still read/write layer-0 weights here.
    """
    from models.experimental.devstral2_123B_instruct.tt.model_args import Devstral2Args
    from models.experimental.devstral2_123B_instruct.tt.weight_loading import resolve_weight_cache_path

    n_layers = devstral2_tt_weight_cache_layer_count(text_cfg)
    cache_args = Devstral2Args.from_hf_config(
        text_cfg,
        mesh_shape=tuple(mesh_device.shape),
        max_seq_len=devstral2_weight_cache_seq_len(),
        max_batch_size=1,
    )
    path = resolve_weight_cache_path(None, cache_args, num_layers=n_layers)
    assert path is not None
    return path


def resolve_devstral2_weight_cache_path(
    mesh_device,
    text_cfg: Ministral3Config,
    num_layers: int | None = None,
    *,
    max_seq_len: int | None = None,
) -> str:
    """Deprecated alias for :func:`devstral2_tt_weight_cache_dir` (``num_layers`` ignored)."""
    if num_layers is not None and num_layers != devstral2_tt_weight_cache_layer_count(text_cfg):
        logger.warning(
            f"resolve_devstral2_weight_cache_path(num_layers={num_layers}) ignored; "
            f"using full model layer count {devstral2_tt_weight_cache_layer_count(text_cfg)}"
        )
    if max_seq_len is not None and max_seq_len != devstral2_weight_cache_seq_len():
        logger.warning(
            f"resolve_devstral2_weight_cache_path(max_seq_len={max_seq_len}) ignored; "
            f"using devstral2_weight_cache_seq_len()={devstral2_weight_cache_seq_len()}"
        )
    return devstral2_tt_weight_cache_dir(mesh_device, text_cfg)


def tt_weight_cache_marker_files(num_layers: int) -> list[str]:
    """Flatbuffer filenames that indicate a populated TT weight cache."""
    markers = ["model_embed_tokens_weight", "model_norm_weight"]
    if num_layers > 0:
        markers.append("model_layers_0_self_attn_q_proj_weight")
    return markers


def is_tt_weight_cache_populated(cache_path: str, num_layers: int) -> bool:
    """True when representative tiled weights already exist under ``cache_path``."""
    cache_dir = Path(cache_path)
    if not cache_dir.is_dir():
        return False
    return all((cache_dir / name).is_file() for name in tt_weight_cache_marker_files(num_layers))


def log_tt_weight_cache_status(cache_path: str, num_layers: int) -> bool:
    """Log cache hit/miss; return whether on-disk TT weight cache is ready to reuse."""
    if is_tt_weight_cache_populated(cache_path, num_layers):
        logger.info(f"Reusing existing TT weight cache at {cache_path}")
        return True
    logger.info(
        f"TT weight cache missing or incomplete at {cache_path} — "
        "will upload from HF host weights and write tiled cache files"
    )
    return False


def _hf_weight_shard_names(keys: list[str]) -> set[str]:
    """Map state-dict keys to safetensor shard filenames via the Hub index."""
    from huggingface_hub import hf_hub_download

    index_path = hf_hub_download(
        repo_id=DEVSTRAL2_LARGE_REPO_ID,
        filename="model.safetensors.index.json",
        local_files_only=True,
    )
    with open(index_path, encoding="utf-8") as f:
        weight_map = json.load(f)["weight_map"]

    fetch_keys: set[str] = set(keys)
    for key in keys:
        if key.endswith(".weight"):
            scale_key = key[: -len(".weight")] + ".weight_scale_inv"
            if scale_key in weight_map:
                fetch_keys.add(scale_key)

    shard_names: set[str] = set()
    for key in fetch_keys:
        if key in weight_map:
            shard_names.add(weight_map[key])
    return shard_names


def hf_hub_weights_cached(keys: list[str]) -> bool:
    """True when every safetensor shard for ``keys`` is already in the HF Hub cache."""
    from huggingface_hub import hf_hub_download

    try:
        shard_names = _hf_weight_shard_names(keys)
    except Exception:
        return False
    if not shard_names:
        return False
    try:
        for shard in shard_names:
            hf_hub_download(
                repo_id=DEVSTRAL2_LARGE_REPO_ID,
                filename=shard,
                local_files_only=True,
            )
    except Exception:
        return False
    return True


def log_hf_hub_weights_status(keys: list[str]) -> bool:
    """Log whether HF shards are cached; return True if local Hub cache is complete."""
    if _hf_local_files_only():
        logger.info("DEVSTRAL2_HF_LOCAL_ONLY=1 — HF Hub download disabled")
        return hf_hub_weights_cached(keys)
    if hf_hub_weights_cached(keys):
        logger.info("HF Hub weight shards already cached — loading locally")
        return True
    logger.info("HF Hub weight shards not cached — downloading from Hub")
    return False


def prepare_e2e_tt_model_budget(
    mesh_device,
    text_cfg: Ministral3Config,
) -> tuple[int, str]:
    """Return ``(model_max_seq_len, weight_cache_path)`` for E2E sweeps.

    - ``weight_cache_path`` uses ``devstral2_weight_cache_seq_len()`` (default
      ``seq_262144``) so an existing demo/CI cache is reused via ``ttnn.as_tensor``.
    - ``model_max_seq_len`` covers worst-case prefill + post-prefill tokens for KV
      and RoPE; only RoPE tables may be extended on first run if this exceeds the
      cached RoPE size — matmul weights still load from the existing cache.
    """
    model_max_seq_len = devstral2_e2e_sweep_model_max_seq_len()
    cache_layers = devstral2_tt_weight_cache_layer_count(text_cfg)
    weight_cache_path = devstral2_tt_weight_cache_dir(mesh_device, text_cfg)
    log_tt_weight_cache_status(weight_cache_path, cache_layers)
    logger.info(f"E2E TT budget: model_max_seq_len={model_max_seq_len}, " f"weight_cache_dir={weight_cache_path}")
    return model_max_seq_len, weight_cache_path


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

    log_hf_hub_weights_status(keys)

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


def decode_tt_to_torch(tt_out: "ttnn.Tensor", *, hidden_size: int, batch_size: int = 1) -> torch.Tensor:
    """Convert decode output to ``[batch, 1, hidden]``, trimming TILE height padding."""
    import ttnn

    tt = ttnn.to_torch(ttnn.get_device_tensors(tt_out)[0])
    if tt.ndim == 4:
        tt = tt[:, :, :batch_size, :hidden_size]
    return tt.reshape(batch_size, 1, hidden_size)


def replicated_tt_to_torch(tensor: "ttnn.Tensor", *, reshape: tuple[int, ...] | None = None) -> torch.Tensor:
    """Read a replicated mesh tensor from one device (post all-reduce activations)."""
    import ttnn

    tt = ttnn.to_torch(ttnn.get_device_tensors(tensor)[0])
    if tt.ndim == 4:
        tt = tt[0:1]
    if reshape is not None:
        tt = tt.reshape(*reshape)
    return tt
