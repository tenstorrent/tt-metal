# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""DiffusionGemma-local causal-prefill MoE optimizations.

The dense fallback tunes sparse-matmul geometry. The default ragged path packs
only routed token/expert pairs into compact, zero-drop batches, preserves the
shared BF16 operation/reduction order, and is bit-identical to the shared
128-expert path while computing only the routed experts.

Prompts longer than ``RAGGED_PREFILL_CHUNK`` are processed in ``RAGGED_PREFILL_CHUNK``-token slices
through the same ragged path (``chunked_ragged_sparse_prefill_forward``, default on via
``DG_PREFILL_RAGGED_LONG``) instead of the shared dense fallback that recomputes all 128 experts.
Chunking also keeps the ``top_k*S*H`` index volumes under the int32 limit that a single full-S
call would hit past ~128K context.

The shared Gemma4 source remains untouched. A context-local selector activates
these paths only for the current DiffusionGemma prefill.
"""

from __future__ import annotations

from contextvars import ContextVar
import os
from contextlib import contextmanager
from threading import Lock

import torch
import ttnn
import models.demos.gemma4.tt.experts as gemma4_experts
import models.demos.gemma4.tt.attention.operations as gemma4_attention_operations
import models.demos.gemma4.tt.experts.prefill as gemma4_prefill
import models.demos.gemma4.tt.shared_mlp as gemma4_shared_mlp
from models.demos.gemma4.tt.router import Gemma4Router
from models.demos.gemma4.tt.shared_mlp import SharedMLP
from models.experimental.diffusion_gemma.tt.ccl import ccl_allreduce as dg_ccl_allreduce
from models.experimental.diffusion_gemma.tt.expert_operations import (
    shared_mlp_forward,
    use_tanh_expert_activations,
)
from models.experimental.diffusion_gemma.tt.sparse_moe import (
    RAGGED_PREFILL_CHUNK,
    chunked_ragged_sparse_prefill_forward,
    ragged_router_forward,
    ragged_sparse_prefill_forward,
)

FLAG = "DG_PREFILL_MOE_TUNED"
RAGGED_FLAG = "DG_PREFILL_MOE_RAGGED"
RAGGED_LONG_FLAG = "DG_PREFILL_RAGGED_LONG"

_HIDDEN_SIZE = 2816
_INTERMEDIATE_PER_DEVICE = 192
_MOE_INTERMEDIATE_SIZE = 704
_NUM_EXPERTS = 128
_TOP_K = 8
_MESH_SHAPE = (1, 4)
_COMPUTE_GRID = (11, 10)

_tuned_geometry_active: ContextVar[bool] = ContextVar("diffusion_gemma_tuned_prefill_moe", default=False)
_ragged_prefill_active: ContextVar[bool] = ContextVar("diffusion_gemma_ragged_prefill_moe", default=False)
_tanh_gelu_active: ContextVar[bool] = ContextVar("diffusion_gemma_tanh_gelu", default=False)
_dg_ccl_active: ContextVar[bool] = ContextVar("diffusion_gemma_prefill_ccl", default=False)
_builder_install_lock = Lock()
_original_builder = gemma4_prefill._build_sparse_matmul_config
_original_prefill_forward = gemma4_experts.prefill_forward
_original_router_forward = Gemma4Router.__call__
_original_shared_mlp_forward = SharedMLP.__call__
_original_ccl_allreduce = gemma4_attention_operations.ccl_allreduce
_original_sdpa = ttnn.transformer.scaled_dot_product_attention
_causal_mask_cache = {}


def tuned_prefill_moe_enabled() -> bool:
    """Whether the exact dense prefill-MoE geometry is enabled (default on)."""

    return os.environ.get(FLAG, "1").strip().lower() not in ("0", "false", "no", "off")


def ragged_prefill_moe_enabled() -> bool:
    """Whether zero-drop compact sparse prefill is enabled (default ON; bit-identical to the shared path)."""

    return os.environ.get(RAGGED_FLAG, "1").strip().lower() not in ("0", "false", "no", "off")


def ragged_long_prefill_enabled() -> bool:
    """Whether the ragged path is extended past ``RAGGED_PREFILL_CHUNK`` via token-dim chunking.

    Default ON: long prefill is processed in ``RAGGED_PREFILL_CHUNK``-token slices through the
    ragged path (see ``chunked_ragged_sparse_prefill_forward``), bit-identical to the shared dense
    prefill. Set ``DG_PREFILL_RAGGED_LONG=0`` to force the shared 128-expert dense fallback for
    prompts beyond one chunk."""

    return os.environ.get(RAGGED_LONG_FLAG, "1").strip().lower() not in ("0", "false", "no", "off")


def _use_ragged_for(seq_len: int) -> bool:
    """Whether this prefill sequence length should take the ragged path (vs shared dense).

    Both the router and prefill hooks MUST agree: the ragged router emits a ``RaggedRouting`` object
    that only the ragged prefill can consume, so the two gates move together. With long-prefill
    chunking off, this is the original ``1 < S <= RAGGED_PREFILL_CHUNK`` window; with it on, any
    multi-token prefill is ragged (the chunked wrapper handles S beyond one chunk)."""

    if seq_len <= 1:
        return False
    if ragged_long_prefill_enabled():
        return True
    return seq_len <= RAGGED_PREFILL_CHUNK


def _find_supported_experts(model):
    """Return all experts only when every measured QB2 invariant matches."""

    mesh_device = getattr(model, "mesh_device", None)
    mesh_config = getattr(model, "mesh_config", None)
    if mesh_device is None or mesh_config is None:
        return None

    try:
        grid = mesh_device.compute_with_storage_grid_size()
        prefill_config = mesh_config.prefill
        supported_mesh = (
            mesh_device.arch() == ttnn.device.Arch.BLACKHOLE
            and tuple(mesh_device.shape) == _MESH_SHAPE
            and mesh_device.get_num_devices() == 4
            and tuple(mesh_config.mesh_shape) == _MESH_SHAPE
            and mesh_config.tp_axis == 1
            and (prefill_config.tp, prefill_config.ep, prefill_config.sp) == (4, 1, 1)
            and (int(grid.x), int(grid.y)) == _COMPUTE_GRID
        )
    except (AttributeError, TypeError):
        return None
    if not supported_mesh or gemma4_prefill.PREFILL_CHUNK_SIZE != ttnn.TILE_SIZE:
        return None

    layers = tuple(getattr(model, "layers", ()))
    experts_per_layer = tuple(getattr(getattr(layer, "moe", None), "experts", None) for layer in layers)
    if not experts_per_layer or any(experts is None for experts in experts_per_layer):
        return None

    for experts in experts_per_layer:
        weights = experts.weights
        config = experts.config
        expert_weights = (weights.gate_proj, weights.up_proj, weights.down_proj)
        if (
            config.hidden_size != _HIDDEN_SIZE
            or config.moe_intermediate_size != _MOE_INTERMEDIATE_SIZE
            or config.num_experts != _NUM_EXPERTS
            or config.top_k != _TOP_K
            or weights.intermediate_size_per_device != _INTERMEDIATE_PER_DEVICE
            or any(weight.get_dtype() != ttnn.bfloat16 for weight in expert_weights)
        ):
            return None
    return experts_per_layer


def _contextual_config_builder(m, n, in0_block_w=1):
    """Select the measured QB2 geometry only in the active call context."""

    if not _tuned_geometry_active.get() or m != ttnn.TILE_SIZE:
        return _original_builder(m, n, in0_block_w)

    if n == _INTERMEDIATE_PER_DEVICE:
        # gate/up: M=32, K=2816 (88 tiles), N=192 (6 tiles).
        grid_x, grid_y = 6, 1
        block_w = 44
        per_core_n = 1
    elif n == _HIDDEN_SIZE:
        # down: M=32, K=192 (6 tiles), N=2816 (88 tiles).
        grid_x, grid_y = 11, 4
        block_w = 3
        per_core_n = 2
    else:
        return _original_builder(m, n, in0_block_w)

    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(grid_x, grid_y),
        in0_block_w=block_w,
        out_subblock_h=1,
        out_subblock_w=1,
        out_block_h=1,
        out_block_w=per_core_n,
        per_core_M=1,
        per_core_N=per_core_n,
        fuse_batch=False,
        fused_activation=None,
        mcast_in0=True,
    )


def _install_contextual_builder() -> None:
    """Install one stable dispatcher; call-local state controls its behavior."""

    if gemma4_prefill._build_sparse_matmul_config is _contextual_config_builder:
        return
    with _builder_install_lock:
        if gemma4_prefill._build_sparse_matmul_config is _contextual_config_builder:
            return
        gemma4_prefill._build_sparse_matmul_config = _contextual_config_builder


def _contextual_prefill_forward(*args, **kwargs):
    """Dispatch only the active DiffusionGemma call to the ragged path."""

    hidden_states = kwargs.get("hidden_states", args[0] if args else None)
    if _ragged_prefill_active.get() and hidden_states is not None and _use_ragged_for(hidden_states.shape[2]):
        if ragged_long_prefill_enabled():
            return chunked_ragged_sparse_prefill_forward(*args, **kwargs)
        return ragged_sparse_prefill_forward(*args, **kwargs)
    return _original_prefill_forward(*args, **kwargs)


def _install_contextual_prefill_forward() -> None:
    if gemma4_experts.prefill_forward is _contextual_prefill_forward:
        return
    with _builder_install_lock:
        if gemma4_experts.prefill_forward is _contextual_prefill_forward:
            return
        gemma4_experts.prefill_forward = _contextual_prefill_forward


def _contextual_router_forward(router, hidden_states):
    if _ragged_prefill_active.get() and _use_ragged_for(hidden_states.shape[2]):
        return ragged_router_forward(router, hidden_states)
    return _original_router_forward(router, hidden_states)


def _install_contextual_router_forward() -> None:
    if Gemma4Router.__call__ is _contextual_router_forward:
        return
    with _builder_install_lock:
        if Gemma4Router.__call__ is _contextual_router_forward:
            return
        Gemma4Router.__call__ = _contextual_router_forward


def _contextual_shared_mlp_forward(mlp, hidden_states):
    if _tanh_gelu_active.get():
        return shared_mlp_forward(mlp, hidden_states)
    return _original_shared_mlp_forward(mlp, hidden_states)


def _install_contextual_shared_mlp_forward() -> None:
    if SharedMLP.__call__ is _contextual_shared_mlp_forward:
        return
    with _builder_install_lock:
        if SharedMLP.__call__ is _contextual_shared_mlp_forward:
            return
        SharedMLP.__call__ = _contextual_shared_mlp_forward


def _contextual_ccl_allreduce(tensor, mesh_config, ccl_manager, memory_config=None):
    if _dg_ccl_active.get():
        return dg_ccl_allreduce(tensor, mesh_config, ccl_manager, memory_config)
    return _original_ccl_allreduce(tensor, mesh_config, ccl_manager, memory_config)


def _install_contextual_ccl_allreduce() -> None:
    with _builder_install_lock:
        gemma4_attention_operations.ccl_allreduce = _contextual_ccl_allreduce
        gemma4_prefill.ccl_allreduce = _contextual_ccl_allreduce
        gemma4_shared_mlp.ccl_allreduce = _contextual_ccl_allreduce


def _manual_one_tile_causal_attention(query, key, value):
    from models.experimental.diffusion_gemma.tt.commit_batched import _manual_gqa_attention_masked

    sequence_length = int(query.shape[-2])
    key_length = int(key.shape[-2])
    host_mask = torch.triu(
        torch.full(
            (1, 1, sequence_length, key_length),
            torch.finfo(torch.bfloat16).min,
            dtype=torch.bfloat16,
        ),
        diagonal=1,
    )
    mesh_device = query.device()
    mask = ttnn.from_torch(
        host_mask,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    output = _manual_gqa_attention_masked(query, key, value, mask)
    mask.deallocate(True)
    return output


def _cached_causal_mask_slice(query, query_start: int, query_end: int, key_start: int, key_end: int, sliding_window):
    mesh_device = query.device()
    window = 0 if sliding_window is None else int(sliding_window)
    cache_key = (id(mesh_device), query_start, query_end, key_start, key_end, window)
    mask = _causal_mask_cache.get(cache_key)
    if mask is not None and mask.is_allocated():
        return mask

    query_positions = torch.arange(query_start, query_end).unsqueeze(1)
    key_positions = torch.arange(key_start, key_end).unsqueeze(0)
    visible = key_positions <= query_positions
    if window > 0:
        visible &= key_positions >= query_positions - window + 1
    host_mask = torch.where(
        visible,
        torch.tensor(0.0, dtype=torch.bfloat16),
        torch.tensor(torch.finfo(torch.bfloat16).min, dtype=torch.bfloat16),
    ).reshape(1, 1, query_end - query_start, key_end - key_start)
    mask = ttnn.from_torch(
        host_mask,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    _causal_mask_cache[cache_key] = mask
    return mask


def _manual_chunked_causal_attention(query, key, value, sliding_window, chunk_size=512):
    """Staged causal GQA with bounded score tensors; no SDPA kernel."""
    from models.experimental.diffusion_gemma.tt.commit_batched import _manual_gqa_attention_masked

    sequence_length = int(query.shape[-2])
    key_length = int(key.shape[-2])
    outputs = []
    for query_start in range(0, sequence_length, chunk_size):
        query_end = min(sequence_length, query_start + chunk_size)
        if sliding_window is None:
            key_start = 0
        else:
            first_visible = max(0, query_start - int(sliding_window) + 1)
            key_start = first_visible // ttnn.TILE_SIZE * ttnn.TILE_SIZE
        key_end = min(key_length, query_end)
        query_chunk = ttnn.slice(
            query,
            [0, 0, query_start, 0],
            [query.shape[0], query.shape[1], query_end, query.shape[3]],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        key_chunk = ttnn.slice(
            key,
            [0, 0, key_start, 0],
            [key.shape[0], key.shape[1], key_end, key.shape[3]],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        value_chunk = ttnn.slice(
            value,
            [0, 0, key_start, 0],
            [value.shape[0], value.shape[1], key_end, value.shape[3]],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        mask = _cached_causal_mask_slice(query, query_start, query_end, key_start, key_end, sliding_window)
        outputs.append(_manual_gqa_attention_masked(query_chunk, key_chunk, value_chunk, mask))
        query_chunk.deallocate(True)
        key_chunk.deallocate(True)
        value_chunk.deallocate(True)

    if len(outputs) == 1:
        return outputs[0]
    output = ttnn.concat(outputs, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    for chunk in outputs:
        chunk.deallocate(True)
    return output


def _contextual_sdpa(query, key, value, *args, **kwargs):
    """Avoid the Blackhole causal SDPA deadlock during DG prefill.

    The stock kernel hangs on QB2 with both implicit-causal and explicit-mask
    modes, independent of grid size, CCL, and KV layout. Route DG causal prefill
    through staged GQA matmuls with bounded query chunks instead.
    """
    sequence_length = int(query.shape[-2])
    key_length = int(key.shape[-2])
    sliding_window = kwargs.get("sliding_window_size")
    use_manual = (
        _dg_ccl_active.get()
        and kwargs.get("is_causal", False)
        and kwargs.get("attn_mask") is None
        and float(kwargs.get("scale", 1.0)) == 1.0
    )
    if not use_manual:
        return _original_sdpa(query, key, value, *args, **kwargs)
    if sequence_length == ttnn.TILE_SIZE and key_length == ttnn.TILE_SIZE:
        return _manual_one_tile_causal_attention(query, key, value)
    return _manual_chunked_causal_attention(query, key, value, sliding_window)


def _install_contextual_sdpa() -> None:
    transformer = getattr(ttnn, "transformer", None)
    if transformer is None or transformer.scaled_dot_product_attention is _contextual_sdpa:
        return
    with _builder_install_lock:
        if transformer.scaled_dot_product_attention is not _contextual_sdpa:
            transformer.scaled_dot_product_attention = _contextual_sdpa


@contextmanager
def use_tuned_prefill_moe(model):
    """Apply supported DiffusionGemma prefill optimizations in this call context."""

    tuned = tuned_prefill_moe_enabled()
    ragged = ragged_prefill_moe_enabled()
    tanh_gelu = True
    supported_experts = _find_supported_experts(model)
    if not tanh_gelu and ((not tuned and not ragged) or supported_experts is None):
        yield
        return

    if tuned and supported_experts is not None:
        _install_contextual_builder()
    if ragged and supported_experts is not None:
        _install_contextual_prefill_forward()
        _install_contextual_router_forward()
    if tanh_gelu:
        _install_contextual_shared_mlp_forward()
    _install_contextual_ccl_allreduce()
    _install_contextual_sdpa()
    geometry_token = _tuned_geometry_active.set(tuned and supported_experts is not None)
    ragged_token = _ragged_prefill_active.set(ragged and supported_experts is not None)
    tanh_token = _tanh_gelu_active.set(tanh_gelu)
    ccl_token = _dg_ccl_active.set(True)
    try:
        with use_tanh_expert_activations(tanh_gelu):
            yield
    finally:
        _dg_ccl_active.reset(ccl_token)
        _tanh_gelu_active.reset(tanh_token)
        _ragged_prefill_active.reset(ragged_token)
        _tuned_geometry_active.reset(geometry_token)
