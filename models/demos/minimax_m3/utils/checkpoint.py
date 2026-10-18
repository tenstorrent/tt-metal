# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint reading helpers shared by the TT loader (``tt/model_config.py``) and the torch reference.

Quantized checkpoints are dequantized to bf16 at load time, so everything downstream sees the same
plain ``<name>.weight`` keys and bf16 tensors as the bf16 checkpoint:

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
    scales may live in other shards), fp32 tensors cast to bf16, everything else (multimodal towers) dropped."""
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
            state_dict[key[len(prefix) :]] = tensor.to(torch.bfloat16) if tensor.dtype == torch.float32 else tensor
    return state_dict


def text_config_of(cfg):
    """The text-backbone config of a (VL-wrapped) checkpoint config, as a ``PretrainedConfig``. Some
    checkpoints' remote config code leaves ``text_config`` a plain dict."""
    text = getattr(cfg, "text_config", cfg)
    if isinstance(text, dict):
        from transformers import PretrainedConfig

        text = PretrainedConfig(**text)
    check_supported_sparse_attention(getattr(text, "sparse_attention_config", None))
    return text


# sparse_attention_config switches for MSA variants this implementation does not have.
_UNSUPPORTED_SPARSE_FLAGS = (
    "sparse_enable_kv_share",
    "sparse_enable_vo_rope",
    "sparse_enable_estimate_lse",
    "sparse_kvshare_value_drop_rope_dim",
)


def check_supported_sparse_attention(sparse_cfg) -> None:
    """Reject sparse-attention variants the model does not implement (index value branch, KV share, ...)."""
    if sparse_cfg is None:
        return
    get = sparse_cfg.get if isinstance(sparse_cfg, dict) else lambda k, d=None: getattr(sparse_cfg, k, d)
    enabled = [flag for flag in _UNSUPPORTED_SPARSE_FLAGS if get(flag, False)]
    # The index value branch only exists on MSA layers (dense layers ignore the flag).
    freq = get("sparse_attention_freq", None) or []
    disable_index_value = get("sparse_disable_index_value", None)
    if disable_index_value is not None:
        with_value = [i for i, (f, d) in enumerate(zip(freq, disable_index_value)) if f and not d]
        if with_value:
            enabled.append(f"index value branch on MSA layers {with_value}")
    if enabled:
        raise NotImplementedError(f"unsupported sparse attention options in the checkpoint config: {enabled}")
