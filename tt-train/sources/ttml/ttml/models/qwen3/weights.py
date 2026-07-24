# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Qwen3 HF <-> ttml weight conversion and loading.

This module is the single source of truth for the Qwen3 weight helpers that the
``examples/qwen3`` and ``examples/grpo`` examples both need; it lives beside the
Qwen3 model implementation (``ttml.models.qwen3``) so neither example has to
vendor a private copy or depend on the other's import layout.

Contents:

  - Weight permutation / inverse-permutation helpers (HF <-> ttml layout)
  - ``build_weight_mapping_single``: HF -> ttml parameter name mapping
  - ``torch_to_ttml``: torch tensor -> bfloat16 TILE ttml tensor, with an
    optional ``mapper`` for FSDP-distributed upload
  - ``load_weights_from_hf``: HF state-dict -> ttml Qwen3 model, supporting both
    the single-device / replicated path (``sharded=False``) and the
    FSDP-sharded, already-materialized path (``sharded=True``)

Imported lazily (not from ``ttml.models.qwen3.__init__``) so the heavy
``torch`` / ``ttnn`` module-level imports only happen when weight loading is
actually used.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, List

import torch
import ttnn

import ttml
from ttml.sharding import Sharding


# =====================================================================
# Weight permutation utilities (HF -> ttml)
# =====================================================================


def unpermute_proj_rows(weight: torch.Tensor, num_heads: int) -> torch.Tensor:
    """Reorder rows: HF [real_half, imag_half] -> ttml interleaved [r0,i0,r1,i1,...]."""
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
    else:
        raise ValueError(f"Expected 1D or 2D tensor, got {weight.dim()}D")


def unpermute_norm_weights(weight: torch.Tensor) -> torch.Tensor:
    """Reorder QK-Norm: HF [x1,x2,...,y1,y2,...] -> ttml [x1,y1,x2,y2,...]."""
    head_dim = weight.shape[0]
    assert head_dim % 2 == 0
    half = head_dim // 2
    w = weight.view(2, half)
    return w.t().contiguous().flatten()


# =====================================================================
# Inverse permutation utilities (ttml -> HF format, for gradient comparison)
# =====================================================================


def repermute_proj_rows(weight: torch.Tensor, num_heads: int) -> torch.Tensor:
    """Inverse of unpermute_proj_rows: ttml interleaved [r0,i0,r1,i1,...] -> HF [real_half, imag_half]."""
    if weight.dim() == 1:
        total = weight.shape[0]
        D = total // num_heads
        w = weight.view(num_heads, D)
        reals = w[:, 0::2]  # even positions -> reals
        imags = w[:, 1::2]  # odd positions -> imags
        return torch.cat([reals, imags], dim=1).reshape(total)
    elif weight.dim() == 2:
        rows, cols = weight.shape
        D = rows // num_heads
        w = weight.view(num_heads, D, cols)
        reals = w[:, 0::2, :]
        imags = w[:, 1::2, :]
        return torch.cat([reals, imags], dim=1).reshape(rows, cols)
    else:
        raise ValueError(f"Expected 1D or 2D tensor, got {weight.dim()}D")


def repermute_norm_weights(weight: torch.Tensor) -> torch.Tensor:
    """Inverse of unpermute_norm_weights: ttml [x1,y1,x2,y2,...] -> HF [x1,x2,...,y1,y2,...]."""
    head_dim = weight.shape[0]
    assert head_dim % 2 == 0
    half = head_dim // 2
    w = weight.view(half, 2)
    return w.t().contiguous().flatten()


# =====================================================================
# Parameter name mapping builder
# =====================================================================


def build_weight_mapping_single(config, root_prefix, tie_word_embeddings):
    """Build HF->ttml name mapping and transform specs for single-device weight loading.

    Returns ``(mapping, transforms)``.
    ``transforms`` values: ``("unpermute_proj", num_heads)`` | ``("unpermute_norm",)``
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
        transforms[f"{hp}.self_attn.q_proj.weight"] = (
            "unpermute_proj",
            config.num_attention_heads,
        )
        # Fused KV: the single ttml kv_proj weight is built from the two HF weights
        # k_proj and v_proj. The mapping key is the K weight; the "combine_kv"
        # transform pulls v_proj from the state-dict and concatenates
        # [unpermute_proj(k) ; v] on the row axis (K rows first, then V rows) to
        # match the [K | V] output layout grouped_heads_creation expects. v_proj is
        # NOT permuted (only K carries the RoPE-permute, mirroring q_proj/k_proj).
        mapping[f"{hp}.self_attn.k_proj.weight"] = f"{tp}/self_attn/kv_proj/weight"
        transforms[f"{hp}.self_attn.k_proj.weight"] = (
            "combine_kv",
            config.num_key_value_heads,
            f"{hp}.self_attn.v_proj.weight",
        )
        mapping[f"{hp}.self_attn.o_proj.weight"] = f"{tp}/self_attn/o_proj/weight"

        if config.attention_bias:
            mapping[f"{hp}.self_attn.q_proj.bias"] = f"{tp}/self_attn/q_proj/bias"
            transforms[f"{hp}.self_attn.q_proj.bias"] = (
                "unpermute_proj",
                config.num_attention_heads,
            )
            mapping[f"{hp}.self_attn.k_proj.bias"] = f"{tp}/self_attn/kv_proj/bias"
            transforms[f"{hp}.self_attn.k_proj.bias"] = (
                "combine_kv",
                config.num_key_value_heads,
                f"{hp}.self_attn.v_proj.bias",
            )
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


# =====================================================================
# torch -> ttml tensor upload
# =====================================================================


def torch_to_ttml(t: torch.Tensor, mapper: Any = None) -> "ttml.autograd.Tensor":
    """Convert a torch tensor to a bfloat16 TILE tensor on the active device.

    When ``mapper`` is ``None`` the tensor is uploaded to a single device (or
    replicated by the default mesh behaviour) -- the original eager path. When a
    ``mapper`` (a ``ttnn`` mesh mapper) is provided the full host tensor is
    distributed across the mesh according to the mapper's placements, so an
    FSDP-sharded parameter only ever lands on device as its per-device slice
    (the full weight stays in host RAM). This mirrors how
    ``ttml.fsdp._shard_replicated_param`` redistributes via
    ``Tensor.from_numpy(full_np, TILE, dtype, mapper)``.
    """
    if mapper is not None:
        full_np = t.float().numpy()
        return ttml.autograd.Tensor.from_numpy(full_np, ttnn.Layout.TILE, ttnn.DataType.BFLOAT16, mapper)
    device = ttml.autograd.AutoContext.get_instance().get_device()
    ttnn_host = ttnn.from_torch(t, dtype=ttnn.bfloat16)
    ttnn_dev = ttnn.to_device(ttnn_host, device)
    ttnn_tiled = ttnn.tilize_with_zero_padding(ttnn_dev)
    return ttml.autograd.create_tensor(ttnn_tiled)


# =====================================================================
# Weight loading from HuggingFace
# =====================================================================


def load_weights_from_hf(
    ttml_model,
    hf_state_dict: dict,
    config,
    tie_word_embeddings: bool = False,
    sharded: bool = False,
    verbose: bool = False,
) -> None:
    """Load a HuggingFace Qwen3 state-dict into a ttml Qwen3 model.

    Two modes:

    - ``sharded=False`` (default, eager path): the model is replicated on the
      FSDP axis (or single-device). When used with FSDP this must be called
      BEFORE ``ttml.fsdp.fully_shard`` so ``fully_shard`` can reshard the (full,
      replicated) parameters in place. CPU prep is threaded; the device transfer
      runs serially in the main thread (concurrent device ops race on the
      program-cache binary commit -- same as the sharded path below).

    - ``sharded=True`` (lazy + FSDP path): the model has already been
      ``fully_shard``-ed and materialized, so each parameter is allocated
      already-sharded. Each HF weight is uploaded distributed to match the
      destination parameter's existing placements, so the full unsharded weight
      never lands on a single device (only its per-device slice does; the full
      tensor stays in host RAM). The materialized parameter's value is swapped
      in place via ``set_value`` so any FSDP markers attached to the autograd
      tensor at materialize time are preserved. The device transfer runs
      serially in the main thread (concurrent device ops race on the
      program-cache binary commit), so only the CPU prep is threaded.
    """
    ttml_params = ttml_model.parameters()

    if verbose:
        print("\n  TTML parameter names:")
        for name in sorted(ttml_params.keys()):
            shape = ttml_params[name].shape()
            print(f"    {name}: {list(shape)}")

    any_key = next(iter(ttml_params))
    root_prefix = any_key.split("/")[0]

    mapping, transforms = build_weight_mapping_single(config, root_prefix, tie_word_embeddings)

    # For a materialized FSDP-sharded parameter, ``.shape()`` returns the LOCAL
    # per-device shape, not the global tensor shape. ``_prepare`` pads each HF
    # weight to ``ttml_shapes[name]`` and (in the sharded path) hands the result
    # to ``Tensor.from_numpy(full_np, ..., mapper)``, where the mapper expects
    # the GLOBAL tensor and slices the per-device shards itself. So in the
    # sharded path we must record the global shape (local shape scaled back up
    # by the distribution-shape size on every sharded dim); otherwise the weight
    # is truncated to one shard and then sharded a second time (uneven shards ->
    # "Shard shape mismatch"). The matching per-param mapper is cached for reuse
    # in the upload loop below.
    #
    # The per-param mesh layout (placements + distribution shape) and its mapper
    # are read via the parallelism-agnostic ``ttml.sharding.Sharding`` helper --
    # the public owner of "read a tensor's live topology, rebuild the mapper that
    # redistributes a host array onto the mesh exactly as the tensor was distributed".
    ttml_shapes = {name: list(ttml_params[name].shape()) for name in ttml_params}
    sharded_mappers: dict = {}
    if sharded:
        device = ttml.autograd.AutoContext.get_instance().get_device()
        replicate_mapper = ttml.core.distributed.replicate_tensor_to_mesh_mapper(device)
        for name in ttml_params:
            sharding = Sharding.from_tensor(ttml_params[name])
            mapper = sharding.derive_mapper()
            if mapper is not None and not sharding.is_fully_replicated:
                # The param is genuinely sharded on some mesh axis. Scale the
                # local per-device shape back up to the global shape on each
                # sharded dim using the live distribution shape, writing it back
                # into ``ttml_shapes`` so ``_prepare`` pads to the global shape
                # (the mapper re-slices the per-device shards). Then reuse the
                # mapper Sharding built from that same topology.
                for axis_idx, placement in enumerate(sharding.placements):
                    if isinstance(placement, ttnn.PlacementShard):
                        ttml_shapes[name][placement.dim] *= sharding.dist_shape[axis_idx]
                sharded_mappers[name] = mapper
            else:
                # Replicated (any param fully_shard leaves un-sharded) or
                # single-device: global == local shape, fan the full host copy out.
                sharded_mappers[name] = replicate_mapper

    def _prepare(hf_name, ttml_name):
        # CPU-only prep (float cast, permutation, padding). The device transfer
        # (torch_to_ttml) is done serially in the main loop below for BOTH
        # paths: running those device ops concurrently across worker threads
        # races on the program-cache binary commit (TT_FATAL "Expected Program
        # Binaries to be committed to DRAM"). TTNN device ops on a single mesh
        # device are not thread-safe, so only the CPU prep is threaded here.
        if hf_name not in hf_state_dict or ttml_name not in ttml_shapes:
            return None

        weight = hf_state_dict[hf_name].float()

        if hf_name in transforms:
            tr = transforms[hf_name]
            if tr[0] == "unpermute_proj":
                weight = unpermute_proj_rows(weight, num_heads=tr[1])
            elif tr[0] == "unpermute_norm":
                weight = unpermute_norm_weights(weight)
            elif tr[0] == "combine_kv":
                # tr = ("combine_kv", num_kv_heads, v_hf_name). Build the fused
                # kv_proj weight: unpermute K (RoPE row-permute), leave V as-is,
                # then concatenate [K ; V] on the row axis (dim 0) so the fused
                # projection outputs K features first, then V.
                num_kv_heads, v_hf_name = tr[1], tr[2]
                if v_hf_name not in hf_state_dict:
                    return None
                k_w = unpermute_proj_rows(weight, num_heads=num_kv_heads)
                v_w = hf_state_dict[v_hf_name].float()
                weight = torch.cat([k_w, v_w], dim=0)

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

        # Return the host weight; the device transfer is serialized in the main
        # loop below (both paths), see the thread-safety note above.
        return weight

    items = list(mapping.items())
    loaded = 0
    skipped: List[str] = []

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [(hf_name, ttml_name, pool.submit(_prepare, hf_name, ttml_name)) for hf_name, ttml_name in items]
        for hf_name, ttml_name, future in futures:
            prepared = future.result()
            if prepared is None:
                if ttml_name not in ttml_shapes:
                    print(f"  WARNING: ttml param '{ttml_name}' not found for HF '{hf_name}'")
                skipped.append(hf_name)
                continue
            # ``prepared`` is a padded host torch weight from the worker thread;
            # the device transfer below runs serially in the main thread (TTNN
            # device ops are not thread-safe -- see the note in ``_prepare``).
            param = ttml_params[ttml_name]
            if sharded:
                # ``prepared`` is the padded GLOBAL-shape host weight (see above).
                # Distribute it with the param's precomputed mapper (FSDP-shard or
                # replicate), then swap the value in place to preserve FSDP markers.
                param.set_value(torch_to_ttml(prepared, mapper=sharded_mappers[ttml_name]).get_value())
            else:
                param.assign(torch_to_ttml(prepared))
            loaded += 1

    print(f"  Qwen3 weight loading: {loaded} loaded, {len(skipped)} skipped")
    if skipped:
        print(f"  Skipped: {skipped}")
