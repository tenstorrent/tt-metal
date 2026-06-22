# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Gemma4 vLLM-style hybrid kv_cache_groups wiring.

This file does NOT stand up the full Gemma4 Model — that's covered by the
existing model accuracy / vLLM-parity tests. Instead it exercises the
bounded-cache plumbing in isolation:

  - ``build_hybrid_page_tables`` returns the SlidingWindowSpec-shaped
    page_table (valid prefix sized to sliding_window/block_size, zero-padded
    tail to max_seq_len/block_size) for sliding layers, and the full
    FullAttentionSpec page_table for non-sliding layers.
  - The page_table that the helper produces, fed to the three paged ops with
    ``cache_position_modulo=sliding_window``, correctly addresses a decode
    walk that crosses the sliding-window boundary multiple times.

The kernel-level "without the kwarg, positions past the boundary clobber block
0" bug is already locked in by
``tests/ttnn/unit_tests/operations/sdpa/test_bounded_sliding_kv_cache.py``;
this file focuses on the Gemma4-specific page_table builder and end-to-end
correctness with the bounded physical pool.

A larger end-to-end check (real weights, full model forward) is intentionally
out of scope; the Gemma4 vLLM parity suite already pins the bridge wiring,
and the bounded mode is opt-in via the ``bounded_sliding_kv_cache`` flag.
"""

from __future__ import annotations

import pytest
import torch

import ttnn
from models.demos.gemma4.tt.attention.kv_cache_hybrid import build_hybrid_page_tables
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc


def _sharded_kv_input(device, x_padded):
    """Mimic the decode-time height-sharded KV tensor that paged_update_cache wants."""
    num_users = x_padded.shape[1]
    xt = ttnn.Tensor(x_padded, ttnn.bfloat16).to(ttnn.TILE_LAYOUT)
    shard_grid = ttnn.num_cores_to_corerangeset(num_users, device.compute_with_storage_grid_size(), True)
    shard_spec = ttnn.ShardSpec(
        shard_grid,
        [xt.volume() // xt.padded_shape[-1] // num_users, xt.padded_shape[-1]],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    mem_cfg = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)
    return xt.to(device, mem_cfg)


def _torch_sliding_ref(k_hist, v_hist, q, cur_pos, sliding_window, scale):
    """Single-user sliding-window attention reference."""
    lo = max(0, cur_pos - sliding_window + 1)
    k_win = k_hist[lo : cur_pos + 1].float()
    v_win = v_hist[lo : cur_pos + 1].float()
    q_b = q[0, 0, :, :].float()
    repeat = q_b.shape[0] // k_win.shape[1]
    k_rep = k_win.repeat_interleave(repeat, dim=1)
    v_rep = v_win.repeat_interleave(repeat, dim=1)
    scores = torch.einsum("hd,shd->hs", q_b, k_rep) * scale
    weights = torch.softmax(scores, dim=-1)
    out = torch.einsum("hs,shd->hd", weights, v_rep)
    return out.unsqueeze(0).unsqueeze(0)


# ── helper-shape tests (pure CPU) ──────────────────────────────────────────


def test_build_hybrid_page_tables_shapes_and_padding():
    """Sliding layers: first sliding_blocks entries valid (per-user contiguous),
    rest zero-padded. Full layers: all max_blocks entries valid."""
    num_layers = 5  # Gemma4-style 4-sliding-1-full pattern
    sliding_layers_mask = [True, True, True, True, False]
    num_users = 3
    block_size = 32
    sliding_window = 128
    max_seq_len = 1024
    sliding_blocks = sliding_window // block_size
    max_blocks = max_seq_len // block_size

    page_tables = build_hybrid_page_tables(
        num_layers,
        sliding_layers_mask,
        num_users=num_users,
        block_size=block_size,
        max_seq_len=max_seq_len,
        sliding_window=sliding_window,
    )

    assert len(page_tables) == num_layers
    for layer_idx, pt in enumerate(page_tables):
        assert pt.shape == (num_users, max_blocks), f"layer {layer_idx} wrong shape"
        assert pt.dtype == torch.int32

        if sliding_layers_mask[layer_idx]:
            for u in range(num_users):
                expected_valid = torch.arange(u * sliding_blocks, (u + 1) * sliding_blocks, dtype=torch.int32)
                assert torch.equal(pt[u, :sliding_blocks], expected_valid)
                assert torch.all(pt[u, sliding_blocks:] == 0)
        else:
            for u in range(num_users):
                expected = torch.arange(u * max_blocks, (u + 1) * max_blocks, dtype=torch.int32)
                assert torch.equal(pt[u], expected)


def test_build_hybrid_page_tables_rejects_non_multiple_sliding_window():
    with pytest.raises(ValueError, match="must be a multiple of block_size"):
        build_hybrid_page_tables(
            num_layers=1,
            sliding_layers_mask=[True],
            num_users=1,
            block_size=32,
            max_seq_len=256,
            sliding_window=100,  # not a multiple of 32
        )


def test_build_hybrid_page_tables_no_sliding_window_falls_back_to_full():
    """sliding_window=None makes every layer use the full max_blocks allocation
    regardless of the mask — there's no SlidingWindowSpec to apply."""
    pts = build_hybrid_page_tables(
        num_layers=2,
        sliding_layers_mask=[True, False],
        num_users=1,
        block_size=32,
        max_seq_len=128,
        sliding_window=None,
    )
    for pt in pts:
        assert torch.equal(pt[0], torch.arange(4, dtype=torch.int32))


# ── on-device end-to-end: bounded sliding decode round-trip ─────────────────


def _run_bounded_decode_walk(device, sliding_window, use_modulo, decode_steps):
    """Drive paged_update_cache + paged_sdpa_decode through a bounded sliding-window
    page_table built by build_hybrid_page_tables (sliding spec for layer 0)."""
    torch.manual_seed(0)

    block_size = 32
    sliding_blocks = sliding_window // block_size
    max_seq_len = 4 * sliding_window  # zero-pad ratio 4× to match vLLM behaviour
    max_blocks = max_seq_len // block_size

    num_kv_heads = 1
    num_q_heads = 1
    head_dim = 128
    num_users = 1
    PADDED_HEADS = 32
    scale = 1.0 / (head_dim**0.5)

    # Bounded physical cache: sliding_blocks * num_users blocks total.
    cache_blocks = sliding_blocks * num_users
    k_init = torch.zeros(cache_blocks, num_kv_heads, block_size, head_dim).bfloat16().float()
    v_init = torch.zeros(cache_blocks, num_kv_heads, block_size, head_dim).bfloat16().float()
    k_cache_tt = ttnn.Tensor(k_init, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device)
    v_cache_tt = ttnn.Tensor(v_init, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device)

    pts = build_hybrid_page_tables(
        num_layers=1,
        sliding_layers_mask=[True],
        num_users=num_users,
        block_size=block_size,
        max_seq_len=max_seq_len,
        sliding_window=sliding_window,
    )
    page_table = pts[0]
    assert page_table.shape == (num_users, max_blocks)
    page_table_tt = ttnn.Tensor(page_table, ttnn.int32).to(device)

    k_hist = torch.zeros(decode_steps, num_kv_heads, head_dim).bfloat16().float()
    v_hist = torch.zeros(decode_steps, num_kv_heads, head_dim).bfloat16().float()

    modulo_kwargs = {"cache_position_modulo": sliding_window} if use_modulo else {}
    results = []
    for pos in range(decode_steps):
        K_new = torch.randn(1, 1, num_kv_heads, head_dim).bfloat16().float()
        V_new = torch.randn(1, 1, num_kv_heads, head_dim).bfloat16().float()
        Q = torch.randn(1, 1, num_q_heads, head_dim).bfloat16().float()
        k_hist[pos, :, :] = K_new[0, 0]
        v_hist[pos, :, :] = V_new[0, 0]

        K_padded = torch.nn.functional.pad(K_new, (0, 0, 0, PADDED_HEADS - num_kv_heads))
        V_padded = torch.nn.functional.pad(V_new, (0, 0, 0, PADDED_HEADS - num_kv_heads))
        Kt = _sharded_kv_input(device, K_padded)
        Vt = _sharded_kv_input(device, V_padded)
        pos_tt = ttnn.Tensor(torch.tensor([pos], dtype=torch.int32), ttnn.int32).to(device)

        ttnn.experimental.paged_update_cache(
            k_cache_tt, Kt, update_idxs_tensor=pos_tt, page_table=page_table_tt, **modulo_kwargs
        )
        ttnn.experimental.paged_update_cache(
            v_cache_tt, Vt, update_idxs_tensor=pos_tt, page_table=page_table_tt, **modulo_kwargs
        )

        Q_padded = torch.nn.functional.pad(Q, (0, 0, 0, PADDED_HEADS - num_q_heads))
        Qt = ttnn.Tensor(Q_padded, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device)

        out_tt = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            Qt,
            k_cache_tt,
            v_cache_tt,
            cur_pos_tensor=pos_tt,
            page_table_tensor=page_table_tt,
            scale=scale,
            sliding_window_size=sliding_window,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            **modulo_kwargs,
        )
        out_torch_padded = ttnn.to_torch(out_tt)
        out_tt_first = out_torch_padded[..., :num_q_heads, :]

        ref = _torch_sliding_ref(k_hist, v_hist, Q, pos, sliding_window, scale)
        passing, msg = comp_pcc(ref, out_tt_first, pcc=0.99)
        results.append((pos, passing, msg))
    return results


@pytest.mark.timeout(180)
def test_bounded_sliding_decode_with_hybrid_page_table(device):
    """End-to-end: hybrid page_table built by build_hybrid_page_tables, fed to the
    paged ops with cache_position_modulo set, correctly tracks the sliding window
    across multiple wrap cycles in a bounded physical cache."""
    sliding_window = 128
    decode_steps = sliding_window * 2 + 16  # well past the boundary
    results = _run_bounded_decode_walk(device, sliding_window, use_modulo=True, decode_steps=decode_steps)

    failing = [(p, m) for (p, ok, m) in results if not ok]
    if failing:
        msg = "\n".join(f"  pos={p}: {m}" for p, m in failing[:10])
        pytest.fail(f"{len(failing)}/{decode_steps} steps failed PCC>=0.99 with hybrid page_table:\n{msg}")


# ── end-to-end flag wiring: bounded vs unbounded decode parity ──────────────


def _build_sliding_window_mask(cache_len: int, sliding_window: int | None) -> torch.Tensor:
    """HF-compatible decode attention mask with sliding window applied. Mirrors
    the helper in ``test_attention.py``; duplicated to keep this file
    self-contained."""
    total_len = cache_len + 1  # cache entries + current query token
    mask = torch.zeros(1, 1, 1, total_len)
    if sliding_window is not None:
        current_pos = cache_len
        for j in range(total_len):
            if j < current_pos - sliding_window + 1:
                mask[0, 0, 0, j] = float("-inf")
    return mask


@pytest.mark.parametrize("mesh_device", [(1, 1)], ids=["1x1"], indirect=True)
@pytest.mark.parametrize("cache_len", [32, 512, 1500], ids=lambda c: f"cache{c}")
def test_attention_decode_bounded_vs_unbounded_parity(cache_len, mesh_device, reset_seeds, request):
    """Drive a sliding-attention decode through both the unbounded path and the
    bounded path; assert both agree with the HF reference and pairwise with
    each other. This is the only test that exercises the full flag wiring
    end-to-end:

        bounded_sliding_kv_cache=True
        -> Gemma4Attention.__init__ sets config.cache_position_modulo
        -> decode.py paged_modulo_kwargs gets the value
        -> paged_update_cache + paged_scaled_dot_product_attention_decode
           interpret positions correctly against the smaller bounded buffer.

    Three cache lengths exercise both regimes:
      - cache_len=32:    within the sliding window; bounded wrap is a no-op.
      - cache_len=512:   still within the window (1024), large enough to span
                         multiple blocks per user.
      - cache_len=1500:  past the window; the bounded buffer wraps while SDPA
                         (with sliding_window_size=1024) still attends only
                         to the last 1024 entries -- which both layouts
                         expose, so outputs must match.
    """
    from transformers.cache_utils import DynamicCache
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding

    from models.demos.gemma4.config import MeshConfig, ModeConfig
    from models.demos.gemma4.tt.attention import Gemma4Attention, Gemma4AttentionConfig
    from models.demos.gemma4.tt.attention.kv_cache import init_kv_cache
    from models.tt_transformers.tt.common import PagedAttentionConfig

    from ..test_factory import TestFactory, compare_tensors, get_pcc_threshold

    layer_idx = 0
    hf_text_config = TestFactory.create_hf_text_config()
    probe_config = Gemma4AttentionConfig(TestFactory.create_hf_config(), layer_idx)
    if not probe_config.is_sliding:
        pytest.skip("layer 0 is not a sliding layer for this model variant")
    sliding_window = probe_config.sliding_window

    block_size = 64
    if sliding_window % block_size != 0:
        pytest.skip(f"sliding_window ({sliding_window}) is not a multiple of block_size ({block_size})")

    hf_layer = TestFactory.create_hf_reference_layer(hf_text_config, layer_idx)
    hf_attn = hf_layer.self_attn
    state_dict = {k: v.clone() for k, v in hf_attn.state_dict().items() if not k.startswith("v_norm")}

    mesh_config = MeshConfig(mesh_device.shape, decode=ModeConfig(tp=1))

    sliding_blocks = sliding_window // block_size

    def _build_variant(bounded):
        config = Gemma4AttentionConfig(TestFactory.create_hf_config(), layer_idx)
        if bounded:
            max_num_blocks = sliding_blocks
            # Zero-pad the page_table out to a few sliding windows -- matches
            # what build_hybrid_page_tables produces and what vLLM emits for
            # SlidingWindowSpec layers.
            max_seq_len = sliding_blocks * 4 * block_size
        else:
            max_num_blocks = (cache_len + block_size) // block_size + 1
            max_seq_len = max_num_blocks * block_size

        paged_attention_config = PagedAttentionConfig(block_size=block_size, max_num_blocks=max_num_blocks)
        kv_cache = init_kv_cache(
            mesh_device=mesh_device,
            config=config,
            paged_attention_config=paged_attention_config,
            cache_dtype=ttnn.bfloat16,
        )

        tt_attn = Gemma4Attention(
            mesh_device=mesh_device,
            config=config,
            state_dict=state_dict,
            ccl_manager=None,
            mesh_config=mesh_config,
            program_config=None,
            layer_idx=layer_idx,
            bounded_sliding_kv_cache=bounded,
        )
        tt_attn.kv_cache = kv_cache

        # Wiring check: the flag should propagate to the config field that
        # decode.py reads.
        if bounded:
            assert (
                tt_attn.config.cache_position_modulo == sliding_window
            ), f"flag set but config.cache_position_modulo = {tt_attn.config.cache_position_modulo}"
        else:
            assert (
                tt_attn.config.cache_position_modulo is None
            ), f"flag off but config.cache_position_modulo = {tt_attn.config.cache_position_modulo}"

        if bounded:
            page_tables = build_hybrid_page_tables(
                num_layers=1,
                sliding_layers_mask=[True],
                num_users=1,
                block_size=block_size,
                max_seq_len=max_seq_len,
                sliding_window=sliding_window,
            )
            page_table = page_tables[0]
        else:
            page_table = torch.arange(max_num_blocks, dtype=torch.int32).reshape(1, max_num_blocks)
        page_table_tt = ttnn.from_torch(page_table, device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.int32)
        return tt_attn, kv_cache, page_table_tt, max_seq_len

    # Shared inputs across both variants.
    k_data = torch.randn(1, probe_config.num_key_value_heads, cache_len, probe_config.head_dim)
    v_data = torch.randn(1, probe_config.num_key_value_heads, cache_len, probe_config.head_dim)
    x_torch = torch.randn(1, 1, probe_config.hidden_size, dtype=torch.float32)

    # HF reference.
    hf_cache = DynamicCache()
    hf_cache.update(k_data.clone(), v_data.clone(), layer_idx=layer_idx)
    rope = Gemma4TextRotaryEmbedding(hf_text_config)
    layer_type = hf_text_config.layer_types[layer_idx]
    cos, sin = rope(x_torch, torch.tensor([[cache_len]]), layer_type=layer_type)
    mask = _build_sliding_window_mask(cache_len, sliding_window)
    with torch.no_grad():
        ref_output, _ = hf_attn(
            x_torch,
            position_embeddings=(cos, sin),
            past_key_values=hf_cache,
            attention_mask=mask,
            shared_kv_states=None,
        )

    outputs = {}
    for bounded in (False, True):
        tt_attn, kv_cache, page_table_tt, max_seq_len = _build_variant(bounded)
        k_cache_tt, v_cache_tt = kv_cache

        # paged_fill_cache: on the bounded path the kernel needs cache_position_modulo
        # so wrap-skips happen on-device. On the unbounded path we omit it -- same
        # legacy kwarg list as the existing test_attention_decode_paged.
        modulo_kwargs = {"cache_position_modulo": sliding_window} if bounded else {}
        k_fill = ttnn.from_torch(
            k_data.to(torch.bfloat16), device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
        )
        v_fill = ttnn.from_torch(
            v_data.to(torch.bfloat16), device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
        )
        ttnn.experimental.paged_fill_cache(k_cache_tt, k_fill, page_table_tt, batch_idx=0, **modulo_kwargs)
        ttnn.experimental.paged_fill_cache(v_cache_tt, v_fill, page_table_tt, batch_idx=0, **modulo_kwargs)

        cos_tt, sin_tt = TestFactory.create_tt_rope_cache(mesh_device, hf_text_config, max_seq_len, layer_idx)
        x_tt = ttnn.from_torch(
            x_torch.unsqueeze(0).to(torch.bfloat16),
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
        )
        position_idx_tt = ttnn.from_torch(
            torch.tensor([[cache_len]], dtype=torch.int32),
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.int32,
        )
        tt_out = tt_attn(
            x_tt,
            rope_mats=(cos_tt, sin_tt),
            position_idx=position_idx_tt,
            is_decode=True,
            token_index=cache_len,
            page_table=page_table_tt,
        )
        outputs[bounded] = ttnn.to_torch(tt_out).squeeze(0).float()

    pcc_threshold = get_pcc_threshold(request)

    passing_unbounded, msg_unbounded = compare_tensors(outputs[False], ref_output, pcc_threshold=pcc_threshold)
    passing_bounded, msg_bounded = compare_tensors(outputs[True], ref_output, pcc_threshold=pcc_threshold)
    passing_pair, msg_pair = compare_tensors(outputs[True], outputs[False], pcc_threshold=pcc_threshold)

    assert passing_unbounded, f"unbounded vs HF (cache_len={cache_len}): {msg_unbounded}"
    assert passing_bounded, f"bounded vs HF (cache_len={cache_len}): {msg_bounded}"
    assert passing_pair, f"bounded vs unbounded (cache_len={cache_len}): {msg_pair}"


def test_bounded_pool_rejects_without_modulo(device):
    """The relaxed paged_update_cache validation only allows page_table.shape[1] >
    cache.shape[0] when cache_position_modulo is set. Confirm the strict legacy
    check still fires when a caller forgets the kwarg — keeps the bounded layout
    from silently clobbering block 0."""
    torch.manual_seed(7)
    block_size = 32
    sliding_window = 128
    sliding_blocks = sliding_window // block_size
    max_seq_len = 4 * sliding_window
    max_blocks = max_seq_len // block_size
    num_users = 1
    num_kv_heads = 1
    head_dim = 128
    PADDED_HEADS = 32

    cache_blocks = sliding_blocks * num_users  # bounded
    cache_torch = torch.zeros(cache_blocks, num_kv_heads, block_size, head_dim).bfloat16().float()
    cache_tt = ttnn.Tensor(cache_torch, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device)

    pts = build_hybrid_page_tables(
        num_layers=1,
        sliding_layers_mask=[True],
        num_users=num_users,
        block_size=block_size,
        max_seq_len=max_seq_len,
        sliding_window=sliding_window,
    )
    page_table_tt = ttnn.Tensor(pts[0], ttnn.int32).to(device)
    cache_idxs_tt = ttnn.Tensor(torch.zeros(num_users, dtype=torch.int32), ttnn.int32).to(device)

    x = torch.randn([1, num_users, num_kv_heads, head_dim]).bfloat16().float()
    x_padded = torch.nn.functional.pad(x, (0, 0, 0, PADDED_HEADS - num_kv_heads))
    xt = _sharded_kv_input(device, x_padded)

    # page_table.shape[1] (max_blocks=16) > cache.shape[0] (sliding_blocks=4) → strict check fires
    assert max_blocks > cache_blocks
    with pytest.raises(RuntimeError, match="max_num_blocks_per_seq must be less than max_num_blocks"):
        ttnn.experimental.paged_update_cache(
            cache_tt,
            xt,
            update_idxs_tensor=cache_idxs_tt,
            page_table=page_table_tt,
        )
