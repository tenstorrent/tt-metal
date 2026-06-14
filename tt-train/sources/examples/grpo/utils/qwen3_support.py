# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Self-contained helpers for the Qwen3 GRPO completer.

Ports the pieces the standalone ``examples/qwen3`` scripts keep in their own
``utils`` package (weight name mapping, HF -> ttml weight conversion, the
QK / RoPE row permutations) plus a small named-mesh builder so the GRPO
example can construct and FSDP-shard a ttml Qwen3 model without depending on
that example's import layout.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import List

import torch
import ttnn

import ttml
from ttml.common.config import DeviceConfig


# ---------------------------------------------------------------------------
# Named mesh
# ---------------------------------------------------------------------------


def build_mesh(device_config: DeviceConfig) -> "ttml.Mesh":
    """Build a named device mesh from a :class:`DeviceConfig`.

    Axis assignment order is ``(dp -> fsdp -> tp)``; the first enabled name
    claims axis 0, the next claims axis 1, etc. Mirrors ``build_mesh`` in the
    nano_gpt example so ``ttml.fsdp.fully_shard`` and ``ttml.sync_gradients``
    bind to the same ``"fsdp"`` / ``"dp"`` axes the rest of ttml expects.

    Line meshes (at most one axis > 1) require exactly one parallelism enabled;
    2D meshes require one parallelism per axis.
    """
    shape = tuple(int(s) for s in device_config.mesh_shape)
    n = len(shape)
    nontrivial = [i for i, s in enumerate(shape) if s > 1]
    is_line = len(nontrivial) <= 1

    enabled_names: List[str] = []
    if device_config.enable_ddp:
        enabled_names.append("dp")
    if device_config.enable_fsdp:
        enabled_names.append("fsdp")
    if device_config.enable_tp:
        enabled_names.append("tp")

    axis_names = [f"_{i}" for i in range(n)]
    if not enabled_names:
        return ttml.Mesh(shape, tuple(axis_names))

    if is_line:
        if len(enabled_names) != 1:
            raise ValueError(
                f"Line mesh {shape} requires exactly one of enable_ddp / enable_fsdp / enable_tp; "
                f"got enabled={enabled_names}"
            )
        active = nontrivial[0] if nontrivial else 0
        axis_names[active] = enabled_names[0]
    else:
        if len(enabled_names) != n:
            raise ValueError(
                f"Mesh {shape} requires {n} parallelisms enabled "
                f"(any subset of enable_ddp / enable_fsdp / enable_tp); got enabled={enabled_names}"
            )
        for i, name in enumerate(enabled_names):
            axis_names[i] = name

    return ttml.Mesh(shape, tuple(axis_names))


# ---------------------------------------------------------------------------
# HF -> ttml weight conversion
# ---------------------------------------------------------------------------


def torch_to_ttml(t: torch.Tensor) -> "ttml.autograd.Tensor":
    """Convert a torch tensor to a bfloat16 TILE tensor on the active device."""
    device = ttml.autograd.AutoContext.get_instance().get_device()
    ttnn_host = ttnn.from_torch(t, dtype=ttnn.bfloat16)
    ttnn_dev = ttnn.to_device(ttnn_host, device)
    ttnn_tiled = ttnn.tilize_with_zero_padding(ttnn_dev)
    return ttml.autograd.create_tensor(ttnn_tiled)


def unpermute_proj_rows(weight: torch.Tensor, num_heads: int) -> torch.Tensor:
    """Reorder rows: HF ``[real_half, imag_half]`` -> ttml interleaved ``[r0,i0,...]``."""
    if weight.dim() == 1:
        total = weight.shape[0]
        D = total // num_heads
        half = D // 2
        w = weight.view(num_heads, D)
        first_half = w[:, :half]
        second_half = w[:, half:]
        interleaved = torch.stack([first_half, second_half], dim=2)
        return interleaved.reshape(total)
    elif weight.dim() == 2:
        rows, cols = weight.shape
        D = rows // num_heads
        half = D // 2
        w = weight.view(num_heads, D, cols)
        first_half = w[:, :half, :]
        second_half = w[:, half:, :]
        interleaved = torch.stack([first_half, second_half], dim=2)
        return interleaved.reshape(rows, cols)
    raise ValueError(f"Expected 1D or 2D tensor, got {weight.dim()}D")


def unpermute_norm_weights(weight: torch.Tensor) -> torch.Tensor:
    """Reorder QK-Norm: HF ``[x1,x2,...,y1,y2,...]`` -> ttml ``[x1,y1,x2,y2,...]``."""
    head_dim = weight.shape[0]
    assert head_dim % 2 == 0
    half = head_dim // 2
    w = weight.view(2, half)
    return w.t().contiguous().flatten()


def build_weight_mapping_single(config, root_prefix: str, tie_word_embeddings: bool):
    """Build HF->ttml parameter name mapping + transform specs (single-device).

    Returns ``(mapping, transforms)`` where transform values are
    ``("unpermute_proj", num_heads)`` or ``("unpermute_norm",)``.
    """
    mapping = {}
    transforms = {}

    if tie_word_embeddings:
        mapping["model.embed_tokens.weight"] = f"{root_prefix}/fc/weight"
    else:
        mapping["model.embed_tokens.weight"] = f"{root_prefix}/tok_emb/weight"
        mapping["lm_head.weight"] = f"{root_prefix}/fc/weight"

    for i in range(config.num_hidden_layers):
        hp = f"model.layers.{i}"
        tp = f"{root_prefix}/blocks/{i}"

        mapping[f"{hp}.self_attn.q_proj.weight"] = f"{tp}/self_attn/q_proj/weight"
        transforms[f"{hp}.self_attn.q_proj.weight"] = ("unpermute_proj", config.num_attention_heads)
        mapping[f"{hp}.self_attn.k_proj.weight"] = f"{tp}/self_attn/k_proj/weight"
        transforms[f"{hp}.self_attn.k_proj.weight"] = ("unpermute_proj", config.num_key_value_heads)
        mapping[f"{hp}.self_attn.v_proj.weight"] = f"{tp}/self_attn/v_proj/weight"
        mapping[f"{hp}.self_attn.o_proj.weight"] = f"{tp}/self_attn/o_proj/weight"

        if config.attention_bias:
            mapping[f"{hp}.self_attn.q_proj.bias"] = f"{tp}/self_attn/q_proj/bias"
            transforms[f"{hp}.self_attn.q_proj.bias"] = ("unpermute_proj", config.num_attention_heads)
            mapping[f"{hp}.self_attn.k_proj.bias"] = f"{tp}/self_attn/k_proj/bias"
            transforms[f"{hp}.self_attn.k_proj.bias"] = ("unpermute_proj", config.num_key_value_heads)
            mapping[f"{hp}.self_attn.v_proj.bias"] = f"{tp}/self_attn/v_proj/bias"
            mapping[f"{hp}.self_attn.o_proj.bias"] = f"{tp}/self_attn/o_proj/bias"

        mapping[f"{hp}.self_attn.q_norm.weight"] = f"{tp}/self_attn/q_norm/weight"
        transforms[f"{hp}.self_attn.q_norm.weight"] = ("unpermute_norm",)
        mapping[f"{hp}.self_attn.k_norm.weight"] = f"{tp}/self_attn/k_norm/weight"
        transforms[f"{hp}.self_attn.k_norm.weight"] = ("unpermute_norm",)

        mapping[f"{hp}.input_layernorm.weight"] = f"{tp}/input_layernorm/weight"
        mapping[f"{hp}.post_attention_layernorm.weight"] = f"{tp}/post_attention_layernorm/weight"
        mapping[f"{hp}.mlp.gate_proj.weight"] = f"{tp}/mlp/gate_proj/weight"
        mapping[f"{hp}.mlp.up_proj.weight"] = f"{tp}/mlp/up_proj/weight"
        mapping[f"{hp}.mlp.down_proj.weight"] = f"{tp}/mlp/down_proj/weight"

    mapping["model.norm.weight"] = f"{root_prefix}/ln_fc/weight"
    return mapping, transforms


def load_weights_from_hf(ttml_model, hf_state_dict: dict, config, tie_word_embeddings: bool = False) -> None:
    """Load a HuggingFace Qwen3 state-dict into a (replicated) ttml Qwen3 model.

    Must be called BEFORE ``ttml.fsdp.fully_shard`` so the parameters are still
    replicated on the FSDP axis; ``fully_shard`` then reshards them in place.
    """
    ttml_params = ttml_model.parameters()
    any_key = next(iter(ttml_params))
    root_prefix = any_key.split("/")[0]

    mapping, transforms = build_weight_mapping_single(config, root_prefix, tie_word_embeddings)
    ttml_shapes = {name: list(ttml_params[name].shape()) for name in ttml_params}

    def _prepare(hf_name, ttml_name):
        # CPU-only prep (float cast, permutation, padding). The device transfer
        # (torch_to_ttml -> ttnn.to_device/tilize) is done serially in the main
        # loop below: running those device ops concurrently across worker threads
        # races on the program-cache binary commit (TT_FATAL "Expected Program
        # Binaries to be committed to DRAM").
        if hf_name not in hf_state_dict or ttml_name not in ttml_shapes:
            return None

        weight = hf_state_dict[hf_name].float()

        if hf_name in transforms:
            tr = transforms[hf_name]
            if tr[0] == "unpermute_proj":
                weight = unpermute_proj_rows(weight, num_heads=tr[1])
            elif tr[0] == "unpermute_norm":
                weight = unpermute_norm_weights(weight)

        ttml_shape = ttml_shapes[ttml_name]
        if weight.dim() == 2:
            rows, cols = weight.shape
            tgt_rows, tgt_cols = ttml_shape[2], ttml_shape[3]
            if rows != tgt_rows or cols != tgt_cols:
                padded = torch.zeros(tgt_rows, tgt_cols, dtype=weight.dtype)
                padded[: min(rows, tgt_rows), : min(cols, tgt_cols)] = weight[
                    : min(rows, tgt_rows), : min(cols, tgt_cols)
                ]
                weight = padded
            weight = weight.unsqueeze(0).unsqueeze(0)
        elif weight.dim() == 1:
            dim = weight.shape[0]
            tgt_dim = ttml_shape[-1]
            if dim != tgt_dim:
                padded = torch.zeros(tgt_dim, dtype=weight.dtype)
                padded[: min(dim, tgt_dim)] = weight[: min(dim, tgt_dim)]
                weight = padded
            weight = weight.unsqueeze(0).unsqueeze(0).unsqueeze(0)

        return weight

    items = list(mapping.items())
    loaded = 0
    skipped: List[str] = []

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [(hf_name, ttml_name, pool.submit(_prepare, hf_name, ttml_name)) for hf_name, ttml_name in items]
        for hf_name, ttml_name, future in futures:
            weight = future.result()
            if weight is None:
                skipped.append(hf_name)
                continue
            # Device transfer runs serially in the main thread (see _prepare).
            ttml_params[ttml_name].assign(torch_to_ttml(weight))
            loaded += 1

    print(f"  Qwen3 weight loading: {loaded} loaded, {len(skipped)} skipped")
    if skipped:
        print(f"  Skipped: {skipped}")
