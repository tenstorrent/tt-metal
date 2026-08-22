# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for the Gemma4 decoder-layer PCC tests (multi-step decode + prefill ISL sweep).

Both tests drive **one** decoder layer against the HuggingFace
``Gemma4TextDecoderLayer`` of the same index, loaded with the checkpoint's real
weights, through the *same* KV cache the demo uses (paged, ``block_size=64``):

* ``test_decode.py`` — N decode steps that each append to the KV cache, so step
  ``k`` attends to what steps ``0…k-1`` wrote. The existing
  ``test_layer_forward_decode`` pre-fills the cache with ``torch.randn`` and takes
  a single step; nothing there gates the *write*.
* ``test_prefill.py`` (lands separately) — a prefill input-sequence-length sweep,
  single-call below the demo's chunk size and generator-style multi-chunk above it.
  ``tt_prefill_chunk`` / ``hf_forward_span`` are shared with it; ``test_decode.py``
  already uses both to seed its KV cache before decoding.

Design notes
------------
* **Real weights, not the constructor's init.** Every scale here is the
  checkpoint's (12B layer 0's input LayerNorm reaches ~190), and the shipped
  ``precision_overrides.json`` dtypes (bfp8 attention + shared MLP on 31B/12B)
  are applied, so the test gates the arithmetic the demo actually runs.
* **HF runs in fp32** (the module is built at the default dtype and the real bf16
  tensors are upcast on load) — same convention as ``test_layer.py``, i.e. the
  reference is more precise than TT rather than equally lossy.
* **HF can be chunked too.** ``hf_forward_span`` takes any span at any start
  position and accumulates into a caller-owned ``DynamicCache``, so a long
  reference forward can be split into bounded pieces — eager attention
  materializes ``[heads, span, keys]`` scores, which a full-sequence forward
  cannot afford at 32K+ keys. Splitting is mathematically the same forward
  (softmax rows cover the same keys), and the span size is independent of TT's
  chunk size: comparison happens on the concatenated per-token outputs.
* **eager ignores ``sliding_window``** (it only adds ``attention_mask``), so the
  window has to be in the mask or a sliding layer diverges from TT's SDPA past
  position 1024.
"""

from __future__ import annotations

import os
from typing import NamedTuple

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.gemma4.config import MeshConfig, ModeConfig
from models.demos.gemma4.tt.attention import Gemma4AttentionConfig
from models.demos.gemma4.tt.attention.kv_cache import init_kv_cache
from models.demos.gemma4.tt.attention.kv_cache_hybrid import build_hybrid_page_tables
from models.demos.gemma4.tt.ccl import CCLManager
from models.demos.gemma4.tt.layer import Gemma4DecoderLayer
from models.demos.gemma4.tt.model import create_rope_caches
from models.demos.gemma4.tt.model_config import Gemma4ModelArgs
from models.demos.gemma4.tt.precision import Gemma4Precision
from models.tt_transformers.tt.common import PagedAttentionConfig, num_blocks_in_seq

from ...tests.test_factory import _get_model_path, find_layer_idx, load_real_weights_into, skip_unless_real_weights

PCC_BATCH_SIZE = 1

# Decode steps per run. 10 is enough for the KV write to matter (step 9 attends to
# nine cache entries the device wrote) while keeping the test a few seconds.
DECODE_STEPS = int(os.environ.get("GEMMA4_DECODER_PCC_DECODE_STEPS", "10"))

# Paged KV cache geometry — the demo's block size, so the page-table arithmetic
# under test is the shipped one.
KV_BLOCK_SIZE = 64

# KV-cache readback threshold. The cache stores bf16 while HF keeps fp32, so an
# exact match is not on the table; anything below this is a wrong row, not
# rounding.
KV_PCC_REQUIRED = 0.99


def hf_text_config():
    """HF text config for the active ``HF_MODEL``, with per-layer input disabled.

    ``hidden_size_per_layer_input`` is not ported to TT (it is 0 on 31B/12B
    anyway); zeroing it keeps the reference on the same graph as the TT layer.
    """
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(_get_model_path(), trust_remote_code=True)
    tc = getattr(config, "text_config", config)
    tc.hidden_size_per_layer_input = 0
    tc._attn_implementation = "eager"
    return tc


def _skip_if_l1_overflow(hf_text_cfg, layer_type: str, mesh_device) -> None:
    """Skip the single-card case a wide global head cannot fit (same rule as test_attention).

    A ``head_dim >= 512`` full-attention layer of a ``hidden_size > 4096`` model
    overflows L1 at TP=1. Gemma4's 31B/12B run ``head_dim=256``, so this only fires
    on variants that would fail in the kernel rather than in the comparison.
    """
    tp = mesh_device.shape[1] if hasattr(mesh_device, "shape") else 1
    if tp > 1 or layer_type != "full_attention":
        return
    if getattr(hf_text_cfg, "head_dim", 0) >= 512 and hf_text_cfg.hidden_size > 4096:
        pytest.skip("Global attention head_dim>=512 overflows L1 on a single device for large models")


class DecoderPccContext(NamedTuple):
    hf_text_cfg: object
    model_args: Gemma4ModelArgs
    attn_cfg: Gemma4AttentionConfig
    hf_layer: object
    tt_layer: Gemma4DecoderLayer
    kv_cache: list
    page_table: torch.Tensor  # [1, blocks] int32, host copy
    page_table_tt: ttnn.Tensor
    rope_4d: tuple  # (cos, sin) [1, 1, max_seq_len, head_dim] — prefill
    rope_2d: tuple  # (cos, sin) [max_seq_len, head_dim]       — decode gather
    layer_idx: int
    layer_type: str
    sliding_window: int | None
    hidden_size: int
    max_seq_len: int
    block_size: int
    tp: int
    is_mesh: bool
    # sliding_window when the bounded ring cache is on, else None. Mirrors
    # ``config.cache_position_modulo``: the value the paged ops wrap positions by.
    cache_position_modulo: int | None


def build_decoder_pcc_context(
    mesh_device, layer_type: str, *, max_seq_len: int, bounded: bool = False
) -> DecoderPccContext:
    """One decoder layer, HF + TT, real weights, shipped dtypes, paged KV cache.

    ``max_seq_len`` sizes both the RoPE caches and the page pool, so it must cover
    the longest position the caller will touch.

    ``bounded=True`` builds the **bounded sliding KV cache** instead of the
    unbounded one — the layout the demo auto-enables above its per-(model, system)
    ISL cutover (``GEMMA4_LONG_CONTEXT_POLICY``). A sliding layer can only ever
    attend to the last ``sliding_window`` tokens, so rather than one cache slot per
    position it gets a ring of ``sliding_window / block_size`` blocks, and the three
    paged ops wrap the absolute position into that ring before the page-table
    lookup (``config.cache_position_modulo``). The page table stays full width with
    a zero-padded tail past the valid prefix, exactly as vLLM emits for a
    SlidingWindowSpec layer — so this also checks the wrap never walks onto the
    padding and clobbers block 0.

    Only meaningful on a sliding layer: ``Gemma4Attention`` leaves the modulo unset
    on full-attention layers, so ``bounded=True`` there would silently test the
    unbounded path. Skipped rather than silently downgraded.
    """
    skip_unless_real_weights()

    hf_text_cfg = hf_text_config()
    layer_idx = find_layer_idx(hf_text_cfg, layer_type)
    _skip_if_l1_overflow(hf_text_cfg, layer_type, mesh_device)

    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextDecoderLayer

    hf_layer = Gemma4TextDecoderLayer(hf_text_cfg, layer_idx=layer_idx)
    load_real_weights_into(hf_layer, f"layers.{layer_idx}")
    hf_layer.eval()

    # ``Gemma4DecoderLayer`` looks for "model.language_model.layers.{idx}" then
    # "model.layers.{idx}"; the HF module's own key names give the latter. Handing
    # over the whole layer state dict guarantees TT and HF read the same tensors.
    tt_state = {f"model.layers.{layer_idx}.{k}": v for k, v in hf_layer.state_dict().items()}

    model_args = Gemma4ModelArgs.from_hf_config(hf_text_cfg)
    attn_cfg = Gemma4AttentionConfig(model_args, layer_idx)

    is_mesh = hasattr(mesh_device, "shape")
    num_devices = mesh_device.get_num_devices() if is_mesh else 1
    tp = mesh_device.shape[1] if is_mesh else 1
    mesh_shape = tuple(mesh_device.shape) if is_mesh else (1, 1)
    mesh_config = MeshConfig(mesh_shape, decode=ModeConfig(tp=tp))
    ccl_manager = CCLManager(mesh_device, is_moe=bool(model_args.enable_moe_block)) if num_devices > 1 else None

    # Shipped per-module dtypes (31B/12B run attention + shared MLP at bfp8).
    # Same resolution create_tt_model does, so the layer under test is not
    # quietly more precise than the demo's.
    precision = Gemma4Precision.load(_get_model_path(), mesh_shape, hf_config=model_args)

    sliding_window = attn_cfg.sliding_window if attn_cfg.is_sliding else None
    if bounded:
        if sliding_window is None:
            pytest.skip(f"bounded sliding KV is only defined for a sliding layer, not {layer_type}")
        if sliding_window % KV_BLOCK_SIZE != 0:
            pytest.skip(f"sliding_window ({sliding_window}) is not a multiple of block_size ({KV_BLOCK_SIZE})")

    # Bounded: a ring of sliding_window/block_size blocks per user, and a full-width
    # page table whose tail past that prefix is zero-padded (what vLLM emits, and
    # what the wrap has to stay clear of). Unbounded: one block per block_size
    # positions and a plain ascending table.
    full_width_blocks = num_blocks_in_seq(max_seq_len, KV_BLOCK_SIZE)
    if bounded:
        pool_blocks = (sliding_window // KV_BLOCK_SIZE) * PCC_BATCH_SIZE
    else:
        pool_blocks = full_width_blocks
    paged_config = PagedAttentionConfig(block_size=KV_BLOCK_SIZE, max_num_blocks=pool_blocks)
    kv_cache = init_kv_cache(
        mesh_device=mesh_device,
        config=attn_cfg,
        max_batch_size=PCC_BATCH_SIZE,
        max_seq_len=max_seq_len,
        paged_attention_config=paged_config,
        cache_dtype=ttnn.bfloat16,
    )

    tt_layer = Gemma4DecoderLayer(
        mesh_device=mesh_device,
        hf_config=model_args,
        state_dict=tt_state,
        layer_idx=layer_idx,
        ccl_manager=ccl_manager,
        dtype=ttnn.bfloat16,
        tensor_cache_path=None,
        mesh_config=mesh_config,
        max_seq_len=max_seq_len,
        max_local_batch_size=PCC_BATCH_SIZE,
        shared_mlp_dtype=precision.get("shared_mlp", ttnn.bfloat16),
        attention_dtype=precision.get("attention", ttnn.bfloat16),
        experts_dtype=precision.get("experts", ttnn.bfloat16),
        router_dtype=precision.get("router", ttnn.bfloat16),
        bounded_sliding_kv_cache=bounded,
    )
    tt_layer.self_attn.kv_cache = kv_cache

    # Wiring check, not a formality: if the flag failed to reach the config the
    # ops would run unbounded and every assertion below would still pass — the
    # test would be green for the wrong layout.
    modulo = tt_layer.self_attn.config.cache_position_modulo
    expected_modulo = sliding_window if bounded else None
    assert modulo == expected_modulo, (
        f"bounded={bounded} on a {layer_type} layer should give "
        f"cache_position_modulo={expected_modulo}, got {modulo}"
    )

    if bounded:
        page_table = build_hybrid_page_tables(
            num_layers=1,
            sliding_layers_mask=[True],
            num_users=PCC_BATCH_SIZE,
            block_size=KV_BLOCK_SIZE,
            max_seq_len=max_seq_len,
            sliding_window=sliding_window,
        )[0]
    else:
        page_table = torch.arange(full_width_blocks, dtype=torch.int32).reshape(1, full_width_blocks)
    page_table_tt = ttnn.from_torch(
        page_table,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.int32,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None,
    )

    rope_4d, rope_2d = create_rope_caches(mesh_device, hf_text_cfg, max_seq_len)

    logger.info(
        "Decoder PCC context: layer_idx={} ({}), sliding_window={}, kv={}, hidden={}, "
        "max_seq_len={}, blocks={}x{}, tp={}, attn_dtype={}, mlp_dtype={}",
        layer_idx,
        layer_type,
        sliding_window,
        f"bounded(ring={sliding_window})" if bounded else "unbounded",
        model_args.hidden_size,
        max_seq_len,
        pool_blocks,
        KV_BLOCK_SIZE,
        tp,
        precision.get("attention", ttnn.bfloat16),
        precision.get("shared_mlp", ttnn.bfloat16),
    )

    return DecoderPccContext(
        hf_text_cfg=hf_text_cfg,
        model_args=model_args,
        attn_cfg=attn_cfg,
        hf_layer=hf_layer,
        tt_layer=tt_layer,
        kv_cache=kv_cache,
        page_table=page_table,
        page_table_tt=page_table_tt,
        rope_4d=rope_4d[layer_type],
        rope_2d=rope_2d[layer_type],
        layer_idx=layer_idx,
        layer_type=layer_type,
        sliding_window=sliding_window,
        hidden_size=model_args.hidden_size,
        max_seq_len=max_seq_len,
        block_size=KV_BLOCK_SIZE,
        tp=tp,
        is_mesh=is_mesh,
        cache_position_modulo=sliding_window if bounded else None,
    )


# ── HF reference forwards ─────────────────────────────────────────────────


def build_attention_mask(
    *,
    query_start: int,
    query_len: int,
    kv_len: int,
    sliding_window: int | None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """``[1, 1, query_len, kv_len]`` additive mask: causal, plus the sliding window.

    Query ``i`` sits at absolute position ``query_start + i`` and may see key ``j``
    when ``j <= query_start + i`` and, for a sliding layer,
    ``j > query_start + i - sliding_window`` — the same visibility TT's SDPA gets
    from ``sliding_window_size``. HF's eager path reads *only* this mask (it
    ignores the ``sliding_window`` it is handed), so the window has to be here.
    """
    q_pos = torch.arange(query_start, query_start + query_len).unsqueeze(1)
    k_pos = torch.arange(kv_len).unsqueeze(0)
    visible = k_pos <= q_pos
    if sliding_window:
        visible &= k_pos > q_pos - sliding_window
    mask = torch.zeros(query_len, kv_len, dtype=dtype)
    mask.masked_fill_(~visible, float("-inf"))
    return mask.reshape(1, 1, query_len, kv_len)


def hf_rope(ctx: DecoderPccContext, positions: torch.Tensor) -> tuple:
    """(cos, sin) for absolute ``positions`` ([1, n] long) on this layer's rope base."""
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding

    rope = Gemma4TextRotaryEmbedding(ctx.hf_text_cfg)
    return rope(torch.zeros(1, dtype=torch.float32), positions, layer_type=ctx.layer_type)


def hf_forward_span(ctx: DecoderPccContext, hidden: torch.Tensor, *, start_pos: int, cache) -> torch.Tensor:
    """One HF forward over ``hidden`` ``[1, n, hidden]`` at positions ``start_pos…``.

    ``cache`` (a ``DynamicCache``) accumulates K/V across calls, so consecutive
    spans reconstruct the full-sequence forward one bounded piece at a time.
    """
    span = hidden.shape[1]
    positions = torch.arange(start_pos, start_pos + span, dtype=torch.long).unsqueeze(0)
    cos, sin = hf_rope(ctx, positions)
    mask = build_attention_mask(
        query_start=start_pos,
        query_len=span,
        kv_len=start_pos + span,
        sliding_window=ctx.sliding_window,
    )
    with torch.no_grad():
        out = ctx.hf_layer(
            hidden,
            position_embeddings=(cos, sin),
            attention_mask=mask,
            position_ids=positions,
            past_key_values=cache,
        )
    return out


# ── TT forwards ───────────────────────────────────────────────────────────


def _replicate(mesh_device, is_mesh):
    return ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None


def tt_from_torch_hidden(ctx: DecoderPccContext, mesh_device, hidden: torch.Tensor) -> ttnn.Tensor:
    """``[1, n, hidden]`` host → ``[1, 1, n, hidden]`` bf16 TILE on device (replicated)."""
    return ttnn.from_torch(
        hidden.reshape(1, 1, hidden.shape[-2], hidden.shape[-1]).to(torch.bfloat16),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=_replicate(mesh_device, ctx.is_mesh),
    )


def tt_to_torch(ctx: DecoderPccContext, tensor: ttnn.Tensor) -> torch.Tensor:
    """Layer output → ``[1, n, hidden]`` fp32 on host (device 0's copy on a mesh).

    The per-layer all-reduce leaves the hidden state replicated, so device 0 holds
    the whole row.
    """
    local = ttnn.get_device_tensors(tensor)[0] if ctx.is_mesh else tensor
    return ttnn.to_torch(local).float().reshape(1, -1, ctx.hidden_size)


def tt_prefill_chunk(
    ctx: DecoderPccContext,
    mesh_device,
    hidden: torch.Tensor,
    *,
    chunk_start: int,
) -> torch.Tensor:
    """One TT prefill call for ``hidden`` ``[1, chunk_len, hidden]`` at ``chunk_start``.

    ``chunk_start > 0`` takes the generator's continuation path: the full page
    table for the SDPA read plus a ``chunk_page_table`` naming only this chunk's
    blocks for the ``paged_fill_cache`` write.
    """
    chunk_len = hidden.shape[1]
    rope_cos, rope_sin = ctx.rope_4d
    rope_mats = (
        rope_cos[:, :, chunk_start : chunk_start + chunk_len, :],
        rope_sin[:, :, chunk_start : chunk_start + chunk_len, :],
    )

    chunk_page_table_tt = None
    chunk_start_idx = None
    if chunk_start > 0:
        first_block = chunk_start // ctx.block_size
        last_block = num_blocks_in_seq(chunk_start + chunk_len, ctx.block_size)
        chunk_page_table_tt = ttnn.from_torch(
            ctx.page_table[:, first_block:last_block],
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.int32,
            mesh_mapper=_replicate(mesh_device, ctx.is_mesh),
        )
        chunk_start_idx = chunk_start

    tt_in = tt_from_torch_hidden(ctx, mesh_device, hidden)
    tt_out = ctx.tt_layer(
        tt_in,
        rope_mats=rope_mats,
        position_idx=None,
        page_table=ctx.page_table_tt,
        kv_cache=ctx.kv_cache,
        is_decode=False,
        user_id=0,
        chunk_start_idx=chunk_start_idx,
        chunk_page_table=chunk_page_table_tt,
    )
    out = tt_to_torch(ctx, tt_out)
    tt_out.deallocate(True)
    return out


def tt_decode_step(
    ctx: DecoderPccContext,
    mesh_device,
    hidden: torch.Tensor,
    *,
    position: int,
) -> torch.Tensor:
    """One TT decode step at ``position``; appends this token's K/V to the cache.

    Mirrors ``prepare_decode_inputs_host``: a ``[1, 32]`` uint32 position tensor
    for the RoPE row gather and a ``[batch]`` int32 one for the cache write /
    SDPA bound.
    """
    replicate = _replicate(mesh_device, ctx.is_mesh)
    cos_2d, sin_2d = ctx.rope_2d

    pos_rope = ttnn.from_torch(
        torch.nn.functional.pad(torch.tensor([[position]], dtype=torch.int64), (0, 32 - PCC_BATCH_SIZE), "constant", 0),
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint32,
        mesh_mapper=replicate,
    )
    pos_cache = ttnn.from_torch(
        torch.tensor([position] * PCC_BATCH_SIZE, dtype=torch.int32),
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.int32,
        mesh_mapper=replicate,
    )

    # Gather this position's cos/sin row once, exactly as Gemma4Model does for the
    # whole stack, and hand it down pre-sliced.
    cos_pos = ttnn.unsqueeze_to_4D(ttnn.embedding(pos_rope, cos_2d, layout=ttnn.TILE_LAYOUT))
    sin_pos = ttnn.unsqueeze_to_4D(ttnn.embedding(pos_rope, sin_2d, layout=ttnn.TILE_LAYOUT))

    tt_in = tt_from_torch_hidden(ctx, mesh_device, hidden)
    tt_out = ctx.tt_layer(
        tt_in,
        rope_mats=(cos_pos, sin_pos),
        position_idx=pos_rope,
        page_table=ctx.page_table_tt,
        kv_cache=ctx.kv_cache,
        is_decode=True,
        token_index=None,
        position_idx_cache=pos_cache,
        rope_presliced=True,
        batch_size=PCC_BATCH_SIZE,
    )
    out = tt_to_torch(ctx, tt_out)
    tt_out.deallocate(True)
    cos_pos.deallocate(True)
    sin_pos.deallocate(True)
    return out


# ── KV cache readback ─────────────────────────────────────────────────────


def kv_head_assignment(ctx: DecoderPccContext) -> list[list[int]]:
    """HF KV-head indices each TP column holds, in column order.

    Mirrors ``load_attention_weights`` exactly: with ``num_kv_heads >= tp`` the K/V
    projection is chunked evenly over dim 0, so column ``c`` owns heads
    ``[c*local, (c+1)*local)``; with ``num_kv_heads < tp`` each column instead gets
    the single head its Q heads map to under GQA
    (``(c * q_per_col) * num_kv // num_q``). Getting this wrong is exactly the bug
    the readback is meant to catch, so it is derived from the same formula rather
    than assumed.
    """
    num_kv = ctx.attn_cfg.num_key_value_heads
    num_q = ctx.attn_cfg.num_attention_heads
    if num_kv < ctx.tp:
        q_per_col = num_q // ctx.tp
        return [[(c * q_per_col) * num_kv // num_q] for c in range(ctx.tp)]
    local = num_kv // ctx.tp
    return [list(range(c * local, (c + 1) * local)) for c in range(ctx.tp)]


def _hf_cache_kv(cache, layer_idx: int):
    """HF cache K/V for ``layer_idx`` as ``[1, num_kv_heads, seq, head_dim]``."""
    if hasattr(cache, "layers"):
        return cache.layers[layer_idx].keys, cache.layers[layer_idx].values
    return cache.key_cache[layer_idx], cache.value_cache[layer_idx]


def cache_slot(ctx: DecoderPccContext, position: int) -> tuple[int, int]:
    """``(block, row)`` the device op writes ``position`` to.

    Resolved by hand rather than by reusing the production indexing, so the readback
    is an independent check. On the bounded ring the absolute position is first
    wrapped into ``[0, cache_position_modulo)`` — the same wrap
    ``paged_update_cache`` / ``paged_fill_cache`` apply — which is what keeps the
    lookup inside the valid page-table prefix instead of on the zero-padded tail.
    """
    slot = position % ctx.cache_position_modulo if ctx.cache_position_modulo else position
    return int(ctx.page_table[0, slot // ctx.block_size]), slot % ctx.block_size


def assert_positions_resident(ctx: DecoderPccContext, positions: list[int], written_through: int) -> None:
    """Fail loudly if a requested position has already been overwritten by the ring.

    On the bounded cache slot ``p % window`` holds the *newest* position congruent to
    ``p``, so once ``written_through`` tokens exist only the last ``window`` of them
    are still readable. Comparing an evicted position against HF (which keeps
    everything) would report a phantom KV mismatch, so ask for the impossible and
    get an error naming the cause instead.
    """
    window = ctx.cache_position_modulo
    if window is None:
        return
    evicted = [p for p in positions if p + window < written_through]
    assert not evicted, (
        f"positions {evicted} were evicted from the bounded ring (window={window}, "
        f"{written_through} positions written — only [{max(0, written_through - window)}, "
        f"{written_through}) are still resident); ask for a resident position instead"
    )


def compare_kv_cache(ctx: DecoderPccContext, cache, positions: list[int], *, written_through: int | None = None):
    """TT-vs-HF K/V at ``positions``, over *every* device, as flat aligned tensors.

    Returns ``((tt_k, hf_k), (tt_v, hf_v))``, each pair the same shape, built by
    walking every device tensor and comparing its resident KV heads against the HF
    heads that device is supposed to hold. Checking the whole mesh rather than
    device 0 alone is what catches a head landing on the wrong column.

    ``written_through`` (total positions written so far) enables the bounded-ring
    residency check; pass it whenever the context may be bounded.
    """
    if written_through is not None:
        assert_positions_resident(ctx, positions, written_through)
    k_cache, v_cache = ctx.kv_cache
    tt_k_dev = ttnn.get_device_tensors(k_cache) if ctx.is_mesh else [k_cache]
    tt_v_dev = ttnn.get_device_tensors(v_cache) if ctx.is_mesh else [v_cache]
    hf_k, hf_v = _hf_cache_kv(cache, ctx.layer_idx)
    assignment = kv_head_assignment(ctx)

    rows = [(pos, *cache_slot(ctx, pos)) for pos in positions]
    tt_k_out, tt_v_out, hf_k_out, hf_v_out = [], [], [], []

    for dev_idx, (dev_k, dev_v) in enumerate(zip(tt_k_dev, tt_v_dev)):
        # Device order is row-major over the mesh and TP runs along the columns,
        # so a data-parallel row replicates the same column assignment.
        heads = assignment[dev_idx % ctx.tp]
        k_local = ttnn.to_torch(dev_k).float()  # [blocks, local_kv, block, head_dim]
        v_local = ttnn.to_torch(dev_v).float()
        assert k_local.shape[1] == len(heads), (
            f"device {dev_idx} holds {k_local.shape[1]} KV heads but the assignment "
            f"expects {len(heads)} ({heads}) — kv_head_assignment is out of sync with weights.py"
        )
        for pos, block, row in rows:
            tt_k_out.append(k_local[block, :, row, :])
            tt_v_out.append(v_local[block, :, row, :])
            hf_k_out.append(hf_k[0, heads, pos, :].float())
            hf_v_out.append(hf_v[0, heads, pos, :].float())

    return (
        (torch.stack(tt_k_out), torch.stack(hf_k_out)),
        (torch.stack(tt_v_out), torch.stack(hf_v_out)),
    )


# ── Reporting ─────────────────────────────────────────────────────────────


def check_pcc(label: str, reference: torch.Tensor, actual: torch.Tensor, threshold: float) -> tuple[bool, float]:
    """``comp_pcc`` plus a one-line PASS/FAIL log; returns ``(passing, pcc)``."""
    passing, pcc = comp_pcc(reference, actual, threshold)
    logger.log(
        "INFO" if passing else "WARNING",
        "{}: PCC {:.6f} (threshold {}) [{}]",
        label,
        float(pcc),
        threshold,
        "PASS" if passing else "FAIL",
    )
    return passing, float(pcc)
