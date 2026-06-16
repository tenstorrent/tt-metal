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
from typing import Any, List

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


def torch_to_ttml(t: torch.Tensor, mapper: Any = None) -> "ttml.autograd.Tensor":
    """Convert a torch tensor to a bfloat16 TILE tensor on the active device.

    When ``mapper`` is ``None`` the tensor is uploaded to a single device (or
    replicated by the default mesh behaviour) — the original eager path. When a
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


def _placements_of(param_tensor: "ttml.autograd.Tensor") -> Any:
    """Best-effort read of a materialized parameter's mesh placements.

    Returns the list of ``ttnn`` placements (``PlacementShard`` /
    ``PlacementReplicate``) backing the autograd tensor's value, or ``None`` if
    the topology is unavailable (e.g. unit mesh). Mirrors
    ``ttml.fsdp._get_placements``; kept local so this module doesn't depend on
    an underscore-private symbol from ``ttml.fsdp``.
    """
    try:
        value = param_tensor.get_value()
        topology = value.tensor_topology()
        return list(topology.placements())
    except Exception:
        return None


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


def load_weights_from_hf(
    ttml_model, hf_state_dict: dict, config, tie_word_embeddings: bool = False, sharded: bool = False
) -> None:
    """Load a HuggingFace Qwen3 state-dict into a ttml Qwen3 model.

    Two modes:

    - ``sharded=False`` (default, eager path): the model is still replicated on
      the FSDP axis. Must be called BEFORE ``ttml.fsdp.fully_shard`` so
      ``fully_shard`` can reshard the (full, replicated) parameters in place.

    - ``sharded=True`` (lazy + FSDP path): the model has already been
      ``fully_shard``-ed and materialized, so each parameter is allocated
      already-sharded. Each HF weight is uploaded distributed to match the
      destination parameter's existing placements, so the full unsharded weight
      never lands on a single device (only its per-device slice does; the full
      tensor stays in host RAM). The materialized parameter's value is swapped
      in place via ``set_value`` so any FSDP markers attached to the autograd
      tensor at materialize time are preserved.
    """
    ttml_params = ttml_model.parameters()
    any_key = next(iter(ttml_params))
    root_prefix = any_key.split("/")[0]

    mapping, transforms = build_weight_mapping_single(config, root_prefix, tie_word_embeddings)

    # For a materialized FSDP-sharded parameter, ``.shape()`` returns the LOCAL
    # per-device shape, not the global tensor shape. ``_prepare`` pads each HF
    # weight to ``ttml_shapes[name]`` and (in the sharded path) hands the result
    # to ``Tensor.from_numpy(full_np, ..., mapper)``, where the mapper expects
    # the GLOBAL tensor and slices the per-device shards itself. So in the
    # sharded path we must record the global shape (local shape scaled back up
    # by the mesh axis size on every sharded dim); otherwise the weight is
    # truncated to one shard and then sharded a second time (uneven shards ->
    # "Shard shape mismatch"). The matching per-param mapper is cached for reuse
    # in the upload loop below.
    ttml_shapes = {name: list(ttml_params[name].shape()) for name in ttml_params}
    sharded_placements: dict = {}
    if sharded:
        mesh_shape = list(ttml.mesh().shape)
        for name in ttml_params:
            placements = _placements_of(ttml_params[name])
            sharded_placements[name] = placements
            if not placements:
                continue
            global_shape = ttml_shapes[name]
            for axis_idx, placement in enumerate(placements):
                if isinstance(placement, ttnn.PlacementShard) and axis_idx < len(mesh_shape):
                    global_shape[placement.dim] *= mesh_shape[axis_idx]

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

    device = ttml.autograd.AutoContext.get_instance().get_device() if sharded else None

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [(hf_name, ttml_name, pool.submit(_prepare, hf_name, ttml_name)) for hf_name, ttml_name in items]
        for hf_name, ttml_name, future in futures:
            weight = future.result()
            if weight is None:
                skipped.append(hf_name)
                continue
            # Device transfer runs serially in the main thread (see _prepare).
            param = ttml_params[ttml_name]
            if sharded:
                # ``weight`` is already padded to the GLOBAL shape (see above).
                # Distribute it to match the destination parameter's existing
                # (FSDP-sharded or replicated) placements, then swap the value in
                # place to preserve FSDP markers.
                placements = sharded_placements.get(ttml_name)
                if placements:
                    mapper = ttnn.create_mesh_mapper(device, ttnn.MeshMapperConfig(placements))
                else:
                    mapper = ttml.core.distributed.replicate_tensor_to_mesh_mapper(device)
                param.set_value(torch_to_ttml(weight, mapper=mapper).get_value())
            else:
                param.assign(torch_to_ttml(weight))
            loaded += 1

    print(f"  Qwen3 weight loading: {loaded} loaded, {len(skipped)} skipped")
    if skipped:
        print(f"  Skipped: {skipped}")
