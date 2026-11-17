# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint reading helpers shared by the TT loader (``tt/model_config.py``) and the torch reference.

Quantized checkpoints are dequantized to bf16 at load time, so everything downstream sees the same
plain ``<name>.weight`` keys and bf16 tensors as the bf16 checkpoint (the MoE router tensors stay fp32,
see ``ROUTER_FP32_SUFFIXES``):

* MXFP8 linears: ``<name>.weight`` (float8_e4m3fn ``[N, K]``) + ``<name>.weight_scale_inv``
  (uint8 UE8M0 exponents ``[N, K/32]``, one per 32 consecutive K elements):
  ``w = fp8 * 2^(e - 127)``.
* NVFP4 routed experts (compressed-tensors ``nvfp4-pack-quantized``): ``<name>.weight_packed``
  (uint8 ``[N, K/2]``, two E2M1 codes per byte, element ``2j`` in the low nibble and ``2j+1`` in the
  high nibble) + ``<name>.weight_scale`` (float8_e4m3fn ``[N, K/16]``) + ``<name>.weight_global_scale``
  (fp32 scalar): ``w = e2m1 * scale / global_scale``. ``input_global_scale`` is the activation scale
  of the W4A4 kernels and is not needed for the weights.

Pure torch (no ttnn) so host tests can import it.
"""

from __future__ import annotations

from typing import Callable, Iterable, Iterator

import torch

MXFP8_BLOCK_SIZE = 32
NVFP4_GROUP_SIZE = 16

# E2M1 code -> value; bit 3 is the sign.
E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)

MXFP8_SCALE_SUFFIX = "weight_scale_inv"
NVFP4_PACKED_SUFFIX = "weight_packed"
NVFP4_SCALE_SUFFIX = "weight_scale"
NVFP4_GLOBAL_SCALE_SUFFIX = "weight_global_scale"
# Scale / auxiliary tensors that are consumed by dequantization (or unused) and never reach the model.
AUX_SUFFIXES = (
    MXFP8_SCALE_SUFFIX,
    NVFP4_SCALE_SUFFIX,
    NVFP4_GLOBAL_SCALE_SUFFIX,
    "input_global_scale",
)


def dequant_mxfp8(weight: torch.Tensor, scale_inv: torch.Tensor, block_size: int = MXFP8_BLOCK_SIZE) -> torch.Tensor:
    """float8_e4m3fn ``[N, K]`` + UE8M0 uint8 ``[N, K/block]`` -> bf16 ``[N, K]``."""
    if weight.dtype != torch.float8_e4m3fn:
        raise ValueError(f"MXFP8 weight must be float8_e4m3fn, got {weight.dtype}")
    if scale_inv.dtype != torch.uint8:
        raise ValueError(f"MXFP8 scale must be uint8 (UE8M0), got {scale_inv.dtype}")
    n, k = weight.shape
    if k % block_size or tuple(scale_inv.shape) != (n, k // block_size):
        raise ValueError(f"MXFP8 scale shape {tuple(scale_inv.shape)} does not match weight {(n, k)} / {block_size}")
    scale = torch.exp2(scale_inv.to(torch.float32) - 127.0).unsqueeze(-1)
    out = weight.to(torch.float32).view(n, k // block_size, block_size) * scale
    return out.view(n, k).to(torch.bfloat16)


def unpack_e2m1(packed: torch.Tensor) -> torch.Tensor:
    """uint8 ``[..., K/2]`` -> fp32 ``[..., K]`` (low nibble first)."""
    if packed.dtype != torch.uint8:
        raise ValueError(f"packed E2M1 must be uint8, got {packed.dtype}")
    table = torch.tensor(E2M1_VALUES, dtype=torch.float32)
    codes = torch.stack((packed & 0x0F, packed >> 4), dim=-1).flatten(-2)
    return table[codes.long()]


def dequant_nvfp4(
    packed: torch.Tensor,
    scale: torch.Tensor,
    global_scale: torch.Tensor,
    group_size: int = NVFP4_GROUP_SIZE,
) -> torch.Tensor:
    """uint8 ``[N, K/2]`` + float8_e4m3fn ``[N, K/group]`` + fp32 scalar -> bf16 ``[N, K]``."""
    if scale.dtype != torch.float8_e4m3fn:
        raise ValueError(f"NVFP4 block scale must be float8_e4m3fn, got {scale.dtype}")
    if global_scale.numel() != 1:
        raise ValueError(f"NVFP4 global scale must be a scalar, got shape {tuple(global_scale.shape)}")
    n, k = packed.shape[0], packed.shape[1] * 2
    if k % group_size or tuple(scale.shape) != (n, k // group_size):
        raise ValueError(f"NVFP4 scale shape {tuple(scale.shape)} does not match weight {(n, k)} / {group_size}")
    values = unpack_e2m1(packed).view(n, k // group_size, group_size)
    block_scale = scale.to(torch.float32).unsqueeze(-1) / global_scale.to(torch.float32).reshape(())
    return (values * block_scale).view(n, k).to(torch.bfloat16)


# MoE router tensors, kept fp32 on the host. The checkpoint stores both fp32 and the vendor routes in fp32:
# the correction bias sits at ~10-37 with a spread of ~0.05, so bf16 (step 0.06-0.25 there) collapses it
# to a handful of distinct values and changes the top-4 set on most tokens.
ROUTER_FP32_SUFFIXES = ("block_sparse_moe.gate.weight", "block_sparse_moe.e_score_correction_bias")


def is_router_fp32_key(key: str) -> bool:
    """True for the MoE gate weight and correction bias (kept fp32 by ``read_text_backbone``)."""
    return key.endswith(ROUTER_FP32_SUFFIXES)


def centered_correction_bias(bias: torch.Tensor) -> torch.Tensor:
    """``e_score_correction_bias`` in fp32 minus its mean.

    Top-k selection on ``sigmoid + bias`` is invariant to a constant shift, and the combine weights use the
    unbiased sigmoid, so routing is unchanged. Centering keeps the biased keys near the sigmoid's [0, 1]
    instead of ~10-37, where ``moe_grouped_topk``'s sort keys (fp32 truncated to 10 mantissa bits) resolve
    only ~0.008-0.03, more than the typical 4th-vs-5th expert gap."""
    bias = bias.to(torch.float32)
    return bias - bias.mean()


def is_aux_key(key: str) -> bool:
    """True for scale / activation-scale tensors that dequantization consumes."""
    return key.rpartition(".")[2] in AUX_SUFFIXES


def plain_key(key: str) -> str:
    """The model-facing name of a checkpoint key (``x.weight_packed`` -> ``x.weight``)."""
    base, _, suffix = key.rpartition(".")
    return f"{base}.weight" if suffix == NVFP4_PACKED_SUFFIX else key


def checkpoint_key_for(name: str, has: Callable[[str], bool]) -> str:
    """The checkpoint key that holds model weight ``name`` (inverse of ``plain_key``)."""
    if not has(name) and name.endswith(".weight"):
        packed = name[: -len("weight")] + NVFP4_PACKED_SUFFIX
        if has(packed):
            return packed
    return name


def load_plain(key: str, get: Callable[[str], torch.Tensor]) -> torch.Tensor:
    """Read checkpoint ``key`` (never an aux key) and return the model-facing tensor, dequantized to
    bf16 when it is quantized. ``get`` reads any checkpoint key (its scales may live in another shard)."""
    base = key.rpartition(".")[0] + "."
    if key.endswith("." + NVFP4_PACKED_SUFFIX):
        return dequant_nvfp4(get(key), get(base + NVFP4_SCALE_SUFFIX), get(base + NVFP4_GLOBAL_SCALE_SUFFIX))
    tensor = get(key)
    if tensor.dtype == torch.float8_e4m3fn:
        return dequant_mxfp8(tensor, get(base + MXFP8_SCALE_SUFFIX))
    return tensor


def iter_dequantized(keys: Iterable[str], get: Callable[[str], torch.Tensor]) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield ``(plain_key, tensor)`` for every non-aux key in ``keys``, one tensor at a time."""
    for key in keys:
        if not is_aux_key(key):
            yield plain_key(key), load_plain(key, get)


def read_text_backbone(weights_path, weight_map: dict, shards: Iterable[str], prefix: str = "language_model.") -> dict:
    """The text-backbone state dict from ``shards`` of a sharded safetensors checkpoint: keys under ``prefix``
    with the prefix stripped, quantized weights dequantized to bf16 under their plain ``.weight`` key (their
    scales may live in other shards), fp32 tensors cast to bf16 except the MoE router's (``is_router_fp32_key``),
    everything else (multimodal towers) dropped."""
    import os

    from safetensors import safe_open

    shard_keys = {}
    for key, shard in weight_map.items():
        if key.startswith(prefix):
            shard_keys.setdefault(shard, []).append(key)
    handles = {}

    def get(key):
        shard = weight_map[key]
        if shard not in handles:
            handles[shard] = safe_open(os.path.join(str(weights_path), shard), framework="pt")
        return handles[shard].get_tensor(key)

    state_dict = {}
    for shard in shards:
        for key, tensor in iter_dequantized(shard_keys.get(shard, []), get):
            # language_model.model.X -> model.X ; language_model.lm_head -> lm_head
            if tensor.dtype == torch.float32 and not is_router_fp32_key(key):
                tensor = tensor.to(torch.bfloat16)
            state_dict[key[len(prefix) :]] = tensor
    return state_dict


def text_config_of(cfg):
    """The text-backbone config of a (VL-wrapped) checkpoint config, as a ``PretrainedConfig``. Some
    checkpoints' remote config code leaves ``text_config`` a plain dict."""
    text = getattr(cfg, "text_config", cfg)
    if isinstance(text, dict):
        from transformers import PretrainedConfig

        text = PretrainedConfig(**text)
    check_supported_config(text)
    check_supported_quantization(getattr(cfg, "quantization_config", None))
    return text


def _getter(cfg):
    """``get(key, default)`` over a config that is a dict or an attribute object."""
    if isinstance(cfg, dict):
        return cfg.get
    return lambda key, default=None: getattr(cfg, key, default)


# Text-config fields the model reads directly (no defaults: a missing one is an error, not an M3 fallback).
REQUIRED_CONFIG_KEYS = (
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "rotary_dim",
    "rope_theta",
    "rms_norm_eps",
    "vocab_size",
    "use_qk_norm",
    "use_gemma_norm",
    "hidden_act",
    "swiglu_alpha",
    "swiglu_limit",
    "intermediate_size",
    "dense_intermediate_size",
    "shared_intermediate_size",
    "num_local_experts",
    "num_experts_per_tok",
    "n_shared_experts",
    "scoring_func",
    "use_routing_bias",
    "routed_scaling_factor",
    "moe_layer_freq",
    "sparse_attention_config",
    "tie_word_embeddings",
)

# (alpha, limit) baked into the routed-expert kernels' clamped swigluoai (ttnn.RoutedExpertActivation.SwiGluOai).
# The dense MLP and the shared expert take both from the config.
ROUTED_EXPERT_SWIGLU = (1.702, 7.0)


def check_supported_config(cfg) -> None:
    """Reject text configs this implementation cannot run as written: missing fields, and values that are
    baked into kernels or have a single code path (listed with the reason). ``cfg`` is a dict or an object."""
    get = _getter(cfg)
    problems = [f"{key} missing" for key in REQUIRED_CONFIG_KEYS if get(key) is None]
    if problems:
        raise ValueError(f"MiniMax-M3 text config is missing fields: {problems}")

    def expect(ok, what):
        if not ok:
            problems.append(what)

    n_layers = get("num_hidden_layers")
    head_dim, rotary_dim = get("head_dim"), get("rotary_dim")
    expect(get("hidden_act") == "swigluoai", f"hidden_act={get('hidden_act')!r}: only swigluoai is implemented")
    # Per-layer lists may be longer than num_hidden_layers: a partial-model run lowers the layer count only.
    expect(
        len(get("moe_layer_freq")) >= n_layers,
        f"moe_layer_freq has {len(get('moe_layer_freq'))} entries for {n_layers} layers",
    )
    if any(get("moe_layer_freq")):
        expect(
            (get("swiglu_alpha"), get("swiglu_limit")) == ROUTED_EXPERT_SWIGLU,
            f"swiglu alpha/limit {(get('swiglu_alpha'), get('swiglu_limit'))}: the routed-expert kernels bake "
            f"{ROUTED_EXPERT_SWIGLU}",
        )
        expect(get("scoring_func") == "sigmoid", f"scoring_func={get('scoring_func')!r}: the MoE gate is sigmoid-only")
        expect(get("n_shared_experts") == 1, f"n_shared_experts={get('n_shared_experts')}: the MoE builds exactly one")
    expect(
        0 < rotary_dim <= head_dim and rotary_dim % 2 == 0,
        f"rotary_dim={rotary_dim} must be even and in (0, head_dim={head_dim}]",
    )
    factor = get("partial_rotary_factor")
    expect(
        factor is None or factor * head_dim == rotary_dim,
        f"partial_rotary_factor={factor} disagrees with rotary_dim={rotary_dim} / head_dim={head_dim}",
    )
    expect(not get("rope_scaling"), f"rope_scaling={get('rope_scaling')!r}: only plain RoPE is implemented")
    # Flags with a single code path: QK-norm on, gemma (1 + w) norms, the routing correction bias.
    for flag in ("use_qk_norm", "use_gemma_norm", "use_routing_bias"):
        expect(get(flag) is True, f"{flag}={get(flag)!r}: only {flag}=true is implemented")
    expect(
        get("qk_norm_type", "per_head") == "per_head",
        f"qk_norm_type={get('qk_norm_type')!r}: only per_head QK-norm is implemented",
    )
    expect(not get("attention_output_gate"), "attention_output_gate is not implemented")
    expect(not get("sliding_window"), "sliding_window attention is not implemented")
    expect(not get("tie_word_embeddings"), "tie_word_embeddings: the LM head loads its own lm_head.weight")
    if problems:
        raise NotImplementedError(f"unsupported MiniMax-M3 text config: {problems}")
    check_supported_sparse_attention(get("sparse_attention_config"), cfg)


# sparse_attention_config switches for MSA variants this implementation does not have.
_UNSUPPORTED_SPARSE_FLAGS = (
    "sparse_enable_kv_share",
    "sparse_enable_vo_rope",
    "sparse_enable_estimate_lse",
    "sparse_kvshare_value_drop_rope_dim",
)


def check_supported_sparse_attention(sparse_cfg, text_cfg=None) -> None:
    """Reject sparse-attention variants the model does not implement (index value branch, KV share, ...).
    With ``text_cfg`` also check the MSA parameters the kernels fix: ``topk_large_indices`` force-selects
    exactly the current block (local block 1, no init / sink block), the indexer block-max-pools
    (score type "max") and scores one index head per GQA group, and the index_k cache shares the K head width."""
    if sparse_cfg is None:
        return
    get = _getter(sparse_cfg)
    enabled = [flag for flag in _UNSUPPORTED_SPARSE_FLAGS if get(flag, False)]
    # The index value branch only exists on MSA layers (dense layers ignore the flag).
    freq = get("sparse_attention_freq", None) or []
    disable_index_value = get("sparse_disable_index_value", None)
    if disable_index_value is not None:
        with_value = [i for i, (f, d) in enumerate(zip(freq, disable_index_value)) if f and not d]
        if with_value:
            enabled.append(f"index value branch on MSA layers {with_value}")
    if text_cfg is not None:
        text = _getter(text_cfg)
        if freq and len(freq) < text("num_hidden_layers"):
            enabled.append(f"sparse_attention_freq has {len(freq)} entries for {text('num_hidden_layers')} layers")
        if get("use_sparse_attention") and any(freq):
            fixed = {
                "sparse_score_type": "max",
                "sparse_init_block": 0,
                "sparse_local_block": 1,
                "sparse_num_index_heads": text("num_key_value_heads"),
                "sparse_index_dim": text("head_dim"),
            }
            enabled += [f"{k}={get(k)!r} (supported: {v!r})" for k, v in fixed.items() if get(k) != v]
            enabled += [f"{k} missing" for k in ("sparse_block_size", "sparse_topk_blocks") if not get(k)]
    if enabled:
        raise NotImplementedError(f"unsupported sparse attention options in the checkpoint config: {enabled}")


def check_supported_quantization(quant_cfg) -> None:
    """Reject checkpoint quantization formats ``load_plain`` does not decode. None = a bf16 checkpoint."""
    if quant_cfg is None:
        return
    get = _getter(quant_cfg)
    expected = {
        "quant_method": "mxfp8",
        "weight_block_size": [1, MXFP8_BLOCK_SIZE],
        "group_size": NVFP4_GROUP_SIZE,
        "moe_quant_format": "nvfp4-pack-quantized",
    }
    bad = {k: get(k) for k, v in expected.items() if get(k) is not None and _as_list(get(k)) != v}
    if get("quant_method") != "mxfp8" or bad:
        raise NotImplementedError(
            f"unsupported checkpoint quantization {bad or get('quant_method')!r}; expected {expected}"
        )


def _as_list(v):
    return list(v) if isinstance(v, (list, tuple)) else v
