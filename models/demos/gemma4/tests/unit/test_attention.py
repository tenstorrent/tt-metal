# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Gemma4 Attention — uses HF Gemma4TextAttention as reference.

Tests prefill and decode for both sliding and global layers, across all TP factors.

    pytest -k "1x1"              # single card
    pytest -k "1x8"              # T3K
    pytest -k "sliding"          # sliding attention only
    pytest -k "global"           # global attention only
    pytest -k "prefill"          # prefill only
    pytest -k "decode"           # decode only
"""

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.gemma4.config import MeshConfig, ModeConfig
from models.demos.gemma4.tt.attention import Gemma4Attention, Gemma4AttentionConfig
from models.demos.gemma4.tt.ccl import CCLManager

from ...tests.test_factory import (
    PREFILL_BUCKETS,
    TestFactory,
    build_hf_prefill_mask,
    compare_tensors,
    find_layer_idx,
    get_pcc_threshold,
    mesh_tp_and_axis,
    parametrize_mesh_with_fabric,
)


def _skip_if_l1_overflow(config, mesh_device):
    """Skip if global attention head_dim overflows L1 on this mesh config."""
    tp, _ = mesh_tp_and_axis(mesh_device)
    # Global layers with large head_dim (512) overflow L1 when hidden_size > 4096 on single device
    if not config.is_sliding and config.head_dim >= 512 and tp == 1:
        hf_config = TestFactory.create_hf_config()
        if hf_config.hidden_size > 4096:
            pytest.skip("Global attention head_dim=512 overflows L1 on single device for large models")


def _setup_attention(mesh_device, layer_idx, create_kv_cache=False, max_seq_len=128):
    """Create HF reference and TT attention module for a given mesh."""
    hf_text_config = TestFactory.create_hf_text_config()
    hf_layer = TestFactory.create_hf_reference_layer(hf_text_config, layer_idx)
    hf_attn = hf_layer.self_attn
    config = Gemma4AttentionConfig(TestFactory.create_hf_config(), layer_idx)

    state_dict = {k: v.clone() for k, v in hf_attn.state_dict().items() if not k.startswith("v_norm")}

    tp, tp_axis = mesh_tp_and_axis(mesh_device)
    mesh_config = MeshConfig(mesh_device.shape, decode=ModeConfig(tp=tp), tp_axis=tp_axis)
    ccl_manager = CCLManager(mesh_device, num_links=1) if tp > 1 else None

    tt_attn = Gemma4Attention(
        mesh_device=mesh_device,
        config=config,
        state_dict=state_dict,
        ccl_manager=ccl_manager,
        mesh_config=mesh_config,
        program_config=None,
        layer_idx=layer_idx,
        create_kv_cache=create_kv_cache,
        max_batch_size=1,
        max_seq_len=max_seq_len,
    )

    return hf_text_config, hf_attn, config, tt_attn, mesh_config


def _to_device(tensor, mesh_device):
    """Send tensor to mesh device with appropriate mapper."""
    is_mesh = hasattr(mesh_device, "shape") and mesh_device.get_num_devices() > 1
    return ttnn.from_torch(
        tensor,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None,
    )


def _from_device(tensor, mesh_device):
    """Read tensor back from device 0."""
    is_mesh = hasattr(mesh_device, "shape") and mesh_device.get_num_devices() > 1
    if is_mesh:
        return ttnn.to_torch(ttnn.get_device_tensors(tensor)[0])
    return ttnn.to_torch(tensor)


# ── Prefill PCC Test ──────────────────────────────────────────────────────


@parametrize_mesh_with_fabric()
@pytest.mark.parametrize("layer_type", ["sliding_attention", "full_attention"], ids=["sliding", "global"])
@pytest.mark.parametrize("seq_len", PREFILL_BUCKETS, ids=[f"seq{b}" for b in PREFILL_BUCKETS])
def test_attention_prefill(layer_type, seq_len, mesh_device, reset_seeds, request):
    """Test prefill attention against HF reference; threshold per
    pcc_thresholds.json (defaults to 0.99 when unmeasured).

    Resolves the layer index by layer_type so the test runs against an actual
    sliding/global layer regardless of model variant (E2B has full layers at
    indices 4/9/14/...; 26B-A4B places them differently).
    """
    hf_text_config = TestFactory.create_hf_text_config()
    try:
        layer_idx = find_layer_idx(hf_text_config, layer_type)
    except ValueError:
        pytest.skip(f"No {layer_type} layer in this model")

    hf_text_config, hf_attn, config, tt_attn, mesh_config = _setup_attention(mesh_device, layer_idx)
    _skip_if_l1_overflow(config, mesh_device)

    x_torch = torch.randn(1, seq_len, config.hidden_size, dtype=torch.float32)

    # HF reference — sliding layers need a sliding+causal mask, otherwise the
    # reference attends to far-history tokens that TT's SDPA correctly masks
    # out via sliding_window_size, and PCC drops once seq_len > sliding_window.
    hf_rope = TestFactory.create_hf_rope(hf_text_config, seq_len, layer_idx)
    sliding = config.sliding_window if config.is_sliding else None
    attn_mask = build_hf_prefill_mask(seq_len, sliding_window=sliding)
    with torch.no_grad():
        ref_output, _ = hf_attn(x_torch, position_embeddings=hf_rope, attention_mask=attn_mask, shared_kv_states=None)

    # TT forward
    cos_tt, sin_tt = TestFactory.create_tt_rope_cache(mesh_device, hf_text_config, max(seq_len, 128), layer_idx)
    x_tt = _to_device(x_torch.unsqueeze(0).to(torch.bfloat16), mesh_device)
    tt_output = tt_attn(x_tt, rope_mats=(cos_tt, sin_tt), is_decode=False)
    tt_output_torch = _from_device(tt_output, mesh_device).squeeze(0).float()

    passing, pcc_msg = compare_tensors(tt_output_torch, ref_output, pcc_threshold=get_pcc_threshold(request))
    assert passing, (
        f"Attention prefill (layer_type={layer_type}, layer_idx={layer_idx}, "
        f"seq={seq_len}, tp={mesh_config.tp}) PCC too low: {pcc_msg}"
    )


# ── RoPE PCC Test at high positions ──────────────────────────────────────


@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1)])
@pytest.mark.parametrize("layer_idx", [0, 5], ids=["sliding", "global"])
@pytest.mark.parametrize("position", [0, 32, 512, 1023, 1024, 1500, 2047], ids=lambda p: f"pos{p}")
def test_rope_pcc(layer_idx, position, mesh_device, reset_seeds, request):
    """Test RoPE cos/sin values and apply_rope PCC at various decode positions up to 2k.

    Compares TT rotary embedding output against HF reference at each position.
    This catches bfloat16 precision loss or indexing bugs at high positions.
    """
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding

    from models.demos.gemma4.tt.attention.operations import apply_rope

    hf_text_config = TestFactory.create_hf_text_config()
    config = Gemma4AttentionConfig(TestFactory.create_hf_config(), layer_idx)
    head_dim = config.head_dim
    max_seq_len = 2048

    # HF reference rope at the target position
    rope = Gemma4TextRotaryEmbedding(hf_text_config)
    layer_type = hf_text_config.layer_types[layer_idx]
    x_dummy = torch.randn(1, 1, hf_text_config.hidden_size)
    cos_ref, sin_ref = rope(x_dummy, torch.tensor([[position]]), layer_type=layer_type)
    # cos_ref, sin_ref: [1, 1, head_dim]

    # Random Q-like tensor [1, num_heads, 1, head_dim]
    num_heads = config.num_attention_heads
    q_torch = torch.randn(1, num_heads, 1, head_dim, dtype=torch.float32)

    # HF-style manual RoPE application (reference)
    cos_expanded = cos_ref.unsqueeze(0).expand(1, num_heads, 1, head_dim)  # [1, heads, 1, head_dim]
    sin_expanded = sin_ref.unsqueeze(0).expand(1, num_heads, 1, head_dim)

    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    ref_rotated = (q_torch * cos_expanded) + (rotate_half(q_torch) * sin_expanded)

    # TT RoPE — build 4D cache [1, 1, max_seq_len, head_dim] and index at position
    cos_tt, sin_tt = TestFactory.create_tt_rope_cache(mesh_device, hf_text_config, max_seq_len, layer_idx)
    q_tt = ttnn.from_torch(q_torch.to(torch.bfloat16), device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    tt_rotated = apply_rope(q_tt, cos_tt, sin_tt, token_index=position)
    tt_rotated_torch = ttnn.to_torch(tt_rotated).float()

    # Also verify the cos/sin cache values themselves at this position
    cos_cache_torch = ttnn.to_torch(cos_tt).float()  # [1, 1, max_seq_len, head_dim]
    sin_cache_torch = ttnn.to_torch(sin_tt).float()
    cos_at_pos = cos_cache_torch[0, 0, position, :]
    sin_at_pos = sin_cache_torch[0, 0, position, :]
    cos_ref_flat = cos_ref[0, 0, :]
    sin_ref_flat = sin_ref[0, 0, :]

    # Check cos/sin cache PCC at this position
    cos_pcc = torch.corrcoef(torch.stack([cos_at_pos, cos_ref_flat]))[0, 1].item()
    sin_pcc = torch.corrcoef(torch.stack([sin_at_pos, sin_ref_flat]))[0, 1].item()

    # Check rotated output PCC
    passing, pcc_msg = compare_tensors(tt_rotated_torch, ref_rotated, pcc_threshold=get_pcc_threshold(request))
    assert passing, (
        f"RoPE output PCC too low at position {position} (layer={layer_idx}): {pcc_msg}\n"
        f"  cos PCC: {cos_pcc:.6f}, sin PCC: {sin_pcc:.6f}"
    )


# ── Decode PCC Test (paged attention, high positions) ────────────────────


def _build_sliding_window_mask(cache_len, sliding_window):
    """Build HF-compatible decode attention mask with sliding window.

    Returns [1, 1, 1, cache_len+1] mask where positions outside the window are -inf.
    For global layers (sliding_window=None), all positions are visible (mask=0).
    """
    total_len = cache_len + 1  # cache entries + current query token
    mask = torch.zeros(1, 1, 1, total_len)
    if sliding_window is not None:
        current_pos = cache_len
        for j in range(total_len):
            if j < current_pos - sliding_window + 1:
                mask[0, 0, 0, j] = float("-inf")
    return mask


@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1)])
@pytest.mark.parametrize("layer_idx", [0, 5], ids=["sliding", "global"])
@pytest.mark.parametrize("cache_len", [32, 512, 1023, 1500], ids=lambda c: f"cache{c}")
def test_attention_decode_paged(layer_idx, cache_len, mesh_device, reset_seeds, request):
    """Test decode attention with paged KV cache against HF reference.

    Tests at various cache lengths including positions beyond the sliding window (1024).
    At cache_len=1500 with sliding_window=1024, SDPA must correctly mask out old entries.
    Global layers (no sliding window) should attend to all cache positions.
    """
    from transformers.cache_utils import DynamicCache
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding

    from models.demos.gemma4.tt.attention.kv_cache import init_kv_cache
    from models.tt_transformers.tt.common import PagedAttentionConfig

    hf_text_config = TestFactory.create_hf_text_config()
    hf_layer = TestFactory.create_hf_reference_layer(hf_text_config, layer_idx)
    hf_attn = hf_layer.self_attn
    config = Gemma4AttentionConfig(TestFactory.create_hf_config(), layer_idx)
    _skip_if_l1_overflow(config, mesh_device)

    state_dict = {k: v.clone() for k, v in hf_attn.state_dict().items() if not k.startswith("v_norm")}

    # Paged attention: enough blocks to hold cache_len + some headroom
    block_size = 64
    max_num_blocks = (cache_len + block_size) // block_size + 1
    max_seq_len = max_num_blocks * block_size
    paged_attention_config = PagedAttentionConfig(block_size=block_size, max_num_blocks=max_num_blocks)

    mesh_config = MeshConfig(mesh_device.shape, decode=ModeConfig(tp=1))
    kv_cache = init_kv_cache(
        mesh_device=mesh_device, config=config, paged_attention_config=paged_attention_config, cache_dtype=ttnn.bfloat16
    )

    tt_attn = Gemma4Attention(
        mesh_device=mesh_device,
        config=config,
        state_dict=state_dict,
        ccl_manager=None,
        mesh_config=mesh_config,
        program_config=None,
        layer_idx=layer_idx,
    )
    tt_attn.kv_cache = kv_cache

    # Random KV cache data [1, num_kv_heads, cache_len, head_dim]
    k_data = torch.randn(1, config.num_key_value_heads, cache_len, config.head_dim)
    v_data = torch.randn(1, config.num_key_value_heads, cache_len, config.head_dim)

    # HF cache
    hf_cache = DynamicCache()
    hf_cache.update(k_data.clone(), v_data.clone(), layer_idx=layer_idx)

    # TT paged cache fill
    page_table = torch.arange(max_num_blocks, dtype=torch.int32).reshape(1, max_num_blocks)
    page_table_tt = ttnn.from_torch(page_table, device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.int32)
    k_cache_tt, v_cache_tt = kv_cache
    k_fill = ttnn.from_torch(
        k_data.to(torch.bfloat16), device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
    )
    v_fill = ttnn.from_torch(
        v_data.to(torch.bfloat16), device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
    )
    ttnn.experimental.paged_fill_cache(k_cache_tt, k_fill, page_table_tt, batch_idx=0)
    ttnn.experimental.paged_fill_cache(v_cache_tt, v_fill, page_table_tt, batch_idx=0)

    # Decode input
    x_torch = torch.randn(1, 1, config.hidden_size, dtype=torch.float32)

    # HF reference with proper sliding-window mask
    rope = Gemma4TextRotaryEmbedding(hf_text_config)
    layer_type = hf_text_config.layer_types[layer_idx]
    cos, sin = rope(x_torch, torch.tensor([[cache_len]]), layer_type=layer_type)
    sliding_window = config.sliding_window if config.is_sliding else None
    mask = _build_sliding_window_mask(cache_len, sliding_window)
    with torch.no_grad():
        ref_output, _ = hf_attn(
            x_torch,
            position_embeddings=(cos, sin),
            past_key_values=hf_cache,
            attention_mask=mask,
            shared_kv_states=None,
        )

    # TT decode with paged attention
    cos_tt, sin_tt = TestFactory.create_tt_rope_cache(mesh_device, hf_text_config, max_seq_len, layer_idx)
    x_tt = ttnn.from_torch(
        x_torch.unsqueeze(0).to(torch.bfloat16), device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
    )
    position_idx_tt = ttnn.from_torch(
        torch.tensor([[cache_len]], dtype=torch.int32),
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.int32,
    )
    tt_output = tt_attn(
        x_tt,
        rope_mats=(cos_tt, sin_tt),
        position_idx=position_idx_tt,
        is_decode=True,
        token_index=cache_len,
        page_table=page_table_tt,
    )
    tt_output_torch = ttnn.to_torch(tt_output).squeeze(0).float()

    passing, pcc_msg = compare_tensors(tt_output_torch, ref_output, pcc_threshold=get_pcc_threshold(request))
    assert passing, (
        f"Attention paged decode (layer={layer_idx}, cache_len={cache_len}, "
        f"sliding_window={sliding_window}) PCC too low: {pcc_msg}"
    )


# ── Batched Decode PCC Test (paged attention, batch > 1) ─────────────────


def _kv_fill_to_tt(k_torch, mesh_device, *, num_kv_heads, num_attention_heads, tp, num_devices):
    """Send a [1, num_kv_heads, seq, head_dim] KV tensor to the mesh in the same
    layout production's K_proj weight sharding produces (mirrors the helper in
    test_vllm_parity._kv_torch_to_tt).

    * Single device: bare from_torch.
    * Sharded (num_kv_heads >= tp): ShardTensorToMesh(dim=1).
    * GQA-replicated (num_kv_heads < tp): one GQA-assigned KV head per device.
    """
    if num_devices <= 1:
        return ttnn.from_torch(
            k_torch.to(torch.bfloat16), device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
        )
    if num_kv_heads >= tp:
        return ttnn.from_torch(
            k_torch.to(torch.bfloat16),
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=1),
        )
    q_per_device = num_attention_heads // tp
    per_device_slices = []
    for dev_i in range(num_devices):
        kv_idx = (dev_i * q_per_device) * num_kv_heads // num_attention_heads
        per_device_slices.append(k_torch[:, kv_idx : kv_idx + 1, :, :])
    stacked = torch.cat(per_device_slices, dim=0)
    return ttnn.from_torch(
        stacked.to(torch.bfloat16),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )


@pytest.mark.skipif(
    os.environ.get("CI") == "true",
    reason="Batched decode PCC test is slow; excluded from CI (run locally for coverage).",
)
@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1), (1, 4)])
@pytest.mark.parametrize("layer_idx", [0, 5], ids=["sliding", "global"])
@pytest.mark.parametrize("batch", [1, 16, 32], ids=lambda b: f"batch{b}")
@pytest.mark.parametrize("cache_len", [512, 1500], ids=lambda c: f"cache{c}")
@pytest.mark.parametrize("qkv_path", ["composite", "fused_presliced"])
def test_attention_decode_paged_batched(layer_idx, batch, cache_len, qkv_path, mesh_device, reset_seeds, request):
    """True batched decode (batch > 1 single forward) with paged KV cache.

    ``qkv_path``: ``composite`` = the per-head norms + RoPE composite with the 2D cache gathered inside attention
    (the historical path); ``fused_presliced`` = the model's decode form (cos / sin gathered once per step into
    ``[1, 1, 32, head_dim]`` rows, ``rope_presliced=True``) with the fused ``custom_ops/qkv_norm_rope`` program
    (GEMMA4_QKV_NORM_ROPE_FUSED=1, GEMMA4_QKV_L1_OUT=1, GEMMA4_HEAD_NORM_4D=1): the end-to-end attention check the
    op's own unit test cannot give (sharded q into SDPA, k / v into the paged cache, per-layer-type tables).

    Each user gets independent random K/V, an independent query, and a distinct
    position ``cache_len - b`` so the per-user RoPE path (apply_rope_decode_peruser
    on [1, batch, 1, head_dim]) is exercised — a single broadcast position would
    not catch a per-user-position bug. Compares each user's TT output against an HF
    reference computed per user. Uses the 2D embedding-lookup RoPE cache (the path
    real decode takes); the 4D legacy cache cannot represent per-user positions.

    Skipped in CI (slow — runs across every gemma4 variant's unit job); run
    locally for batched-decode coverage.
    """
    from transformers.cache_utils import DynamicCache
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding

    from models.demos.gemma4.tt.attention.kv_cache import init_kv_cache
    from models.tt_transformers.tt.common import PagedAttentionConfig

    hf_text_config = TestFactory.create_hf_text_config()
    hf_layer = TestFactory.create_hf_reference_layer(hf_text_config, layer_idx)
    hf_attn = hf_layer.self_attn
    config = Gemma4AttentionConfig(TestFactory.create_hf_config(), layer_idx)
    _skip_if_l1_overflow(config, mesh_device)

    tp, tp_axis = mesh_tp_and_axis(mesh_device)
    num_devices = mesh_device.get_num_devices() if hasattr(mesh_device, "get_num_devices") else 1
    is_mesh = num_devices > 1

    state_dict = {k: v.clone() for k, v in hf_attn.state_dict().items() if not k.startswith("v_norm")}

    # Per-user positions (distinct → distinct RoPE). cache_len is the fill length
    # for every user; each user attends to its first L_b = cache_len - b entries.
    positions = [cache_len - b for b in range(batch)]

    # Paged cache: each user gets its own contiguous block range.
    block_size = 64
    blocks_per_user = (cache_len + block_size) // block_size + 1
    max_num_blocks = blocks_per_user * batch
    max_seq_len = blocks_per_user * block_size
    paged_attention_config = PagedAttentionConfig(block_size=block_size, max_num_blocks=max_num_blocks)

    mesh_config = MeshConfig(mesh_device.shape, decode=ModeConfig(tp=tp), tp_axis=tp_axis)
    ccl_manager = CCLManager(mesh_device, num_links=1) if tp > 1 else None
    kv_cache = init_kv_cache(
        mesh_device=mesh_device, config=config, paged_attention_config=paged_attention_config, cache_dtype=ttnn.bfloat16
    )

    tt_attn = Gemma4Attention(
        mesh_device=mesh_device,
        config=config,
        state_dict=state_dict,
        ccl_manager=ccl_manager,
        mesh_config=mesh_config,
        program_config=None,
        layer_idx=layer_idx,
    )
    tt_attn.kv_cache = kv_cache

    # Page table [batch, max_num_blocks]: user b owns blocks [b*bpu, (b+1)*bpu).
    page_table = torch.zeros(batch, max_num_blocks, dtype=torch.int32)
    for b in range(batch):
        page_table[b, :blocks_per_user] = torch.arange(
            b * blocks_per_user, (b + 1) * blocks_per_user, dtype=torch.int32
        )
    page_table_tt = ttnn.from_torch(
        page_table,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.int32,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None,
    )

    # Per-user random K/V + query; fill each user's cache and run HF reference.
    k_cache_tt, v_cache_tt = kv_cache
    sliding_window = config.sliding_window if config.is_sliding else None
    x_users = []
    ref_outputs = []
    for b in range(batch):
        k_data = torch.randn(1, config.num_key_value_heads, cache_len, config.head_dim)
        v_data = torch.randn(1, config.num_key_value_heads, cache_len, config.head_dim)
        k_fill = _kv_fill_to_tt(
            k_data,
            mesh_device,
            num_kv_heads=config.num_key_value_heads,
            num_attention_heads=config.num_attention_heads,
            tp=tp,
            num_devices=num_devices,
        )
        v_fill = _kv_fill_to_tt(
            v_data,
            mesh_device,
            num_kv_heads=config.num_key_value_heads,
            num_attention_heads=config.num_attention_heads,
            tp=tp,
            num_devices=num_devices,
        )
        ttnn.experimental.paged_fill_cache(k_cache_tt, k_fill, page_table_tt, batch_idx=b)
        ttnn.experimental.paged_fill_cache(v_cache_tt, v_fill, page_table_tt, batch_idx=b)

        # HF reference: this user attends to its first L_b cache entries.
        pos = positions[b]
        x_b = torch.randn(1, 1, config.hidden_size, dtype=torch.float32)
        x_users.append(x_b)
        hf_cache = DynamicCache()
        hf_cache.update(k_data[:, :, :pos, :].clone(), v_data[:, :, :pos, :].clone(), layer_idx=layer_idx)
        rope = Gemma4TextRotaryEmbedding(hf_text_config)
        layer_type = hf_text_config.layer_types[layer_idx]
        cos, sin = rope(x_b, torch.tensor([[pos]]), layer_type=layer_type)
        mask = _build_sliding_window_mask(pos, sliding_window)
        with torch.no_grad():
            ref_b, _ = hf_attn(
                x_b,
                position_embeddings=(cos, sin),
                past_key_values=hf_cache,
                attention_mask=mask,
                shared_kv_states=None,
            )
        ref_outputs.append(ref_b.reshape(config.hidden_size))

    # Batched TT decode input: [1, 1, batch, hidden_size], users on dim 2.
    x_torch = torch.cat([x.reshape(1, 1, 1, config.hidden_size) for x in x_users], dim=2)
    x_tt = ttnn.from_torch(
        x_torch.to(torch.bfloat16),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None,
    )

    # 2D embedding-lookup RoPE cache (real decode path). position_idx packs the
    # B per-user positions into the first B slots of a [1, 32] uint32 tensor;
    # position_idx_cache carries the [B] int32 positions for cache update + SDPA.
    cos_tt, sin_tt = TestFactory.create_tt_rope_cache_2d(
        mesh_device, hf_text_config, max(cache_len + 32, 128), layer_idx
    )
    pos_padded = torch.zeros(1, 32, dtype=torch.int32)
    pos_padded[0, :batch] = torch.tensor(positions, dtype=torch.int32)
    position_idx_tt = ttnn.from_torch(
        pos_padded,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint32,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None,
    )
    position_idx_cache_tt = ttnn.from_torch(
        torch.tensor(positions, dtype=torch.int32),
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.int32,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None,
    )

    fused_env = {"GEMMA4_QKV_NORM_ROPE_FUSED": "1", "GEMMA4_QKV_L1_OUT": "1", "GEMMA4_HEAD_NORM_4D": "1"}
    saved_env = {k: os.environ.get(k) for k in fused_env}
    try:
        if qkv_path == "fused_presliced":
            os.environ.update(fused_env)
            # the model's per-step gather (tt/model.py): row u of the [1, 1, 32, head_dim] table = user u's position
            cos_pos = ttnn.unsqueeze_to_4D(ttnn.embedding(position_idx_tt, cos_tt, layout=ttnn.TILE_LAYOUT))
            sin_pos = ttnn.unsqueeze_to_4D(ttnn.embedding(position_idx_tt, sin_tt, layout=ttnn.TILE_LAYOUT))
            tt_output = tt_attn(
                x_tt,
                rope_mats=(cos_pos, sin_pos),
                position_idx=position_idx_tt,
                is_decode=True,
                page_table=page_table_tt,
                position_idx_cache=position_idx_cache_tt,
                rope_presliced=True,
            )
        else:
            tt_output = tt_attn(
                x_tt,
                rope_mats=(cos_tt, sin_tt),
                position_idx=position_idx_tt,
                is_decode=True,
                page_table=page_table_tt,
                position_idx_cache=position_idx_cache_tt,
            )
    finally:
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    # tt_output: [1, 1, batch_padded(32), hidden_size] — the decode batch dim is
    # tile-padded; production slices [:, :, :B] in process_output_decode.
    tt_output_torch = _from_device(tt_output, mesh_device)[:, :, :batch, :].reshape(batch, config.hidden_size).float()

    ref_stacked = torch.stack(ref_outputs, dim=0)  # [batch, hidden_size]
    passing, pcc_msg = compare_tensors(tt_output_torch, ref_stacked, pcc_threshold=get_pcc_threshold(request))
    assert passing, (
        f"Batched decode (layer={layer_idx}, batch={batch}, cache_len={cache_len}, "
        f"tp={tp}, sliding_window={sliding_window}) PCC too low: {pcc_msg}"
    )


# ── Cross-call chunked prefill tail lifetime regression ──────────────────


@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1)])
def test_sliding_tail_survives_cross_call_chunking(mesh_device, reset_seeds, request):
    """Regression test for the sliding-window prefill-tail-lifetime bug.

    vLLM-driven token-chunked prefill makes separate ``__call__`` invocations
    for each chunk. The first call slices the sliding tail from tt_k/tt_v and
    stores it in ``self._sliding_tails_by_key`` (demo/non-vLLM path keys the single legacy ``None`` slot). After the call returns, tt_k
    and tt_v are deallocated. If the tail is a view (not an independent copy)
    of tt_k, the second call finds an unallocated tensor and crashes with
    ``TT_FATAL: Input Tensor is not allocated``.

    This test simulates two separate prefill calls with ``chunk_start_idx``
    and verifies the second call consumes the persisted tail without error.
    """
    from models.demos.gemma4.tt.attention.kv_cache import init_kv_cache
    from models.tt_transformers.tt.common import PagedAttentionConfig

    hf_text_config = TestFactory.create_hf_text_config()
    layer_idx = find_layer_idx(hf_text_config, "sliding_attention")
    config = Gemma4AttentionConfig(TestFactory.create_hf_config(), layer_idx)

    hf_layer = TestFactory.create_hf_reference_layer(hf_text_config, layer_idx)
    state_dict = {k: v.clone() for k, v in hf_layer.self_attn.state_dict().items() if not k.startswith("v_norm")}

    chunk_size = 1024
    block_size = 64
    total_seq = chunk_size * 2  # two chunks
    max_num_blocks = total_seq // block_size + 2
    max_seq_len = max_num_blocks * block_size
    paged_attention_config = PagedAttentionConfig(block_size=block_size, max_num_blocks=max_num_blocks)

    mesh_config = MeshConfig(mesh_device.shape, decode=ModeConfig(tp=1))
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
    )
    tt_attn.kv_cache = kv_cache

    page_table = torch.arange(max_num_blocks, dtype=torch.int32).reshape(1, -1)
    page_table_tt = ttnn.from_torch(page_table, device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.int32)

    cos_tt, sin_tt = TestFactory.create_tt_rope_cache(mesh_device, hf_text_config, max_seq_len, layer_idx)

    # region Chunk 1 (chunk_start_idx=0)
    x1 = torch.randn(1, 1, chunk_size, config.hidden_size, dtype=torch.bfloat16)
    x1_tt = _to_device(x1, mesh_device)
    chunk1_pt = page_table[:, : chunk_size // block_size]
    chunk1_pt_tt = ttnn.from_torch(chunk1_pt, device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.int32)
    out1 = tt_attn(
        x1_tt,
        rope_mats=(cos_tt, sin_tt),
        is_decode=False,
        page_table=page_table_tt,
        kv_cache=kv_cache,
        chunk_start_idx=0,
        chunk_page_table=chunk1_pt_tt,
    )
    # Force deallocation of the output to simulate the model runner dropping it
    # between scheduling steps (mimics real vLLM inter-step behavior).
    out1.deallocate(True)

    # The tail must be alive after the first call.
    tail = (getattr(tt_attn, "_sliding_tails_by_key", None) or {}).get(None)
    assert tail is not None, "Sliding tail was not persisted after chunk 1"
    assert all(t.is_allocated() for t in tail), "Sliding tail tensor(s) are not allocated — the clone fix is missing"
    # endregion

    # region Chunk 2 (chunk_start_idx=chunk_size, consumes persisted tail)
    x2 = torch.randn(1, 1, chunk_size, config.hidden_size, dtype=torch.bfloat16)
    x2_tt = _to_device(x2, mesh_device)
    chunk2_pt = page_table[:, chunk_size // block_size : total_seq // block_size]
    chunk2_pt_tt = ttnn.from_torch(chunk2_pt, device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.int32)
    # This call used to crash with "Input Tensor is not allocated" before the
    # `ttnn.clone` fix in `_prefill_forward_single.
    out2 = tt_attn(
        x2_tt,
        rope_mats=(cos_tt, sin_tt),
        is_decode=False,
        page_table=page_table_tt,
        kv_cache=kv_cache,
        chunk_start_idx=chunk_size,
        chunk_page_table=chunk2_pt_tt,
    )
    # If we get here, the tail survived the cross-call deallocation.
    out2_torch = _from_device(out2, mesh_device)
    assert out2_torch.shape[-2] == chunk_size, f"Expected seq_len={chunk_size} in output, got {out2_torch.shape[-2]}"
    # endregion

    # region Async-safe tail lifetime across decode
    # Under vLLM async_scheduling, another request's decode can interleave
    # between APC continuation prefills. Decode must NOT wipe the sliding
    # tail; a new prefill at chunk_start_idx==0 releases it instead.
    tail_before_decode = (getattr(tt_attn, "_sliding_tails_by_key", None) or {}).get(None)
    assert tail_before_decode is not None, "Tail should still be present after chunk 2 (not yet in decode)"

    x_dec = torch.randn(1, 1, 1, config.hidden_size, dtype=torch.bfloat16)
    x_dec_tt = _to_device(x_dec, mesh_device)
    position_idx_tt = ttnn.from_torch(
        torch.tensor([[total_seq]], dtype=torch.int32),
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.int32,
    )
    tt_attn(
        x_dec_tt,
        rope_mats=(cos_tt, sin_tt),
        position_idx=position_idx_tt,
        is_decode=True,
        token_index=total_seq,
        page_table=page_table_tt,
        kv_cache=kv_cache,
    )

    tail_after_decode = (getattr(tt_attn, "_sliding_tails_by_key", None) or {}).get(None)
    assert tail_after_decode is not None, "Tail must survive decode so async APC continuations keep sliding_tail_in"
    assert all(t.is_allocated() for t in tail_after_decode), "Tail deallocated during decode"

    # A fresh prefill at chunk_start==0 must release the prior request's tail.
    x_new = torch.randn(1, 1, chunk_size, config.hidden_size, dtype=torch.bfloat16)
    x_new_tt = _to_device(x_new, mesh_device)
    chunk_new_pt = page_table[:, : chunk_size // block_size]
    chunk_new_pt_tt = ttnn.from_torch(chunk_new_pt, device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.int32)
    out_new = tt_attn(
        x_new_tt,
        rope_mats=(cos_tt, sin_tt),
        is_decode=False,
        page_table=page_table_tt,
        kv_cache=kv_cache,
        chunk_start_idx=0,
        chunk_page_table=chunk_new_pt_tt,
    )
    out_new.deallocate(True)
    tail_after_reset = (getattr(tt_attn, "_sliding_tails_by_key", None) or {}).get(None)
    assert tail_after_reset is not None, "New prefill at start=0 should stash a fresh tail"
    assert all(t.is_allocated() for t in tail_after_reset)
    # endregion


@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1)])
def test_short_first_chunk_stashes_padded_sliding_tail(mesh_device, reset_seeds, request):
    """APC remnant: first grant < sliding_window must still stash a hist-wide tail.

    Reproduces the shield failure mode where chunk_start=384 arrives on the
    continuation without sliding_tail_in because the prior short chunk skipped
    the post-SDPA stash (kseq < hist).
    """
    from models.demos.gemma4.tt.attention.kv_cache import init_kv_cache
    from models.tt_transformers.tt.common import PagedAttentionConfig

    hf_text_config = TestFactory.create_hf_text_config()
    layer_idx = find_layer_idx(hf_text_config, "sliding_attention")
    config = Gemma4AttentionConfig(TestFactory.create_hf_config(), layer_idx)
    sliding_window = int(config.sliding_window)
    hist = ((sliding_window + 31) // 32) * 32
    assert hist > 384, f"test needs hist>384, got hist={hist}"

    hf_layer = TestFactory.create_hf_reference_layer(hf_text_config, layer_idx)
    state_dict = {k: v.clone() for k, v in hf_layer.self_attn.state_dict().items() if not k.startswith("v_norm")}

    short_len = 384
    cont_len = 1024
    short_cont = 128
    block_size = 64
    total_seq = short_len + cont_len
    final_seq = total_seq + short_cont
    max_num_blocks = (final_seq + block_size - 1) // block_size + 2
    max_seq_len = max_num_blocks * block_size
    paged_attention_config = PagedAttentionConfig(block_size=block_size, max_num_blocks=max_num_blocks)

    mesh_config = MeshConfig(mesh_device.shape, decode=ModeConfig(tp=1))
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
    )
    tt_attn.kv_cache = kv_cache
    page_table = torch.arange(max_num_blocks, dtype=torch.int32).reshape(1, -1)
    page_table_tt = ttnn.from_torch(page_table, device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.int32)
    cos_tt, sin_tt = TestFactory.create_tt_rope_cache(mesh_device, hf_text_config, max_seq_len, layer_idx)

    # Short first grant — single-chunk path (no chunk_page_table), like vLLM APC.
    x1 = torch.randn(1, 1, short_len, config.hidden_size, dtype=torch.bfloat16)
    out1 = tt_attn(
        _to_device(x1, mesh_device),
        rope_mats=(cos_tt, sin_tt),
        is_decode=False,
        page_table=page_table_tt,
        kv_cache=kv_cache,
        chunk_start_idx=0,
        valid_seq_len=short_len,
    )
    out1.deallocate(True)
    tail = (getattr(tt_attn, "_sliding_tails_by_key", None) or {}).get(None)
    assert tail is not None, "Short first chunk must stash a padded sliding tail"
    assert all(t.is_allocated() for t in tail)
    assert int(tail[0].shape[-2]) == hist, f"Expected padded hist={hist}, got {tail[0].shape[-2]}"

    # Continuation at chunk_start=384 with chunk_page_table — needs the tail.
    x2 = torch.randn(1, 1, cont_len, config.hidden_size, dtype=torch.bfloat16)
    chunk2_pt = page_table[:, short_len // block_size : total_seq // block_size]
    chunk2_pt_tt = ttnn.from_torch(chunk2_pt, device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.int32)
    out2 = tt_attn(
        _to_device(x2, mesh_device),
        rope_mats=(cos_tt, sin_tt),
        is_decode=False,
        page_table=page_table_tt,
        kv_cache=kv_cache,
        chunk_start_idx=short_len,
        chunk_page_table=chunk2_pt_tt,
    )
    out2_torch = _from_device(out2, mesh_device)
    assert out2_torch.shape[-2] == cont_len

    # Short continuation with seq < hist must still consume the padded tail
    # (Q filler previously sliced hist rows from a short tt_q → TT_FATAL).
    x3 = torch.randn(1, 1, short_cont, config.hidden_size, dtype=torch.bfloat16)
    chunk3_pt = page_table[:, total_seq // block_size : final_seq // block_size]
    chunk3_pt_tt = ttnn.from_torch(chunk3_pt, device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.int32)
    out3 = tt_attn(
        _to_device(x3, mesh_device),
        rope_mats=(cos_tt, sin_tt),
        is_decode=False,
        page_table=page_table_tt,
        kv_cache=kv_cache,
        chunk_start_idx=total_seq,
        chunk_page_table=chunk3_pt_tt,
    )
    out3_torch = _from_device(out3, mesh_device)
    assert out3_torch.shape[-2] == short_cont, "short continuation with seq < hist failed"


# ── Chunked prefill numerics (served 8K / 32K path) ──────────────────────


@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1), (1, 4)])
@pytest.mark.parametrize(
    "layer_type",
    [
        "full_attention",
        pytest.param(
            "sliding_attention",
            marks=pytest.mark.skipif(os.environ.get("CI") == "true", reason="CI subset: the global layer only"),
        ),
    ],
)
def test_chunked_prefill_matches_one_shot(layer_type, mesh_device, reset_seeds, request):
    """Two 1024-token chunks through the vLLM-style chunked prefill (``chunk_start_idx`` + ``chunk_page_table``,
    the served 8K / 32K path) match the one-shot 2048-token prefill of the same layer: the global layer's second
    chunk attends the paged prefix through ``chunked_prefill_sdpa`` and the sliding layer's through its stashed
    tail, both on the prefill SDPA program config (step 3b units-fill rule) that the one-shot path uses."""
    from models.demos.gemma4.tt.attention.kv_cache import init_kv_cache
    from models.tt_transformers.tt.common import PagedAttentionConfig

    hf_text_config = TestFactory.create_hf_text_config()
    try:
        layer_idx = find_layer_idx(hf_text_config, layer_type)
    except ValueError:
        pytest.skip(f"no {layer_type} layer in this checkpoint")

    chunk_size = 1024
    block_size = 64
    total_seq = chunk_size * 2
    max_num_blocks = total_seq // block_size + 2
    max_seq_len = max_num_blocks * block_size
    paged_attention_config = PagedAttentionConfig(block_size=block_size, max_num_blocks=max_num_blocks)
    _, _, config, tt_attn, mesh_config = _setup_attention(mesh_device, layer_idx, max_seq_len=max_seq_len)
    _skip_if_l1_overflow(config, mesh_device)
    cos_tt, sin_tt = TestFactory.create_tt_rope_cache(mesh_device, hf_text_config, max_seq_len, layer_idx)

    hd = int(config.head_dim)

    def rope_rows(lo, hi):
        # the prefill RoPE consumes cos / sin rows [0, rows) of what it is given: a chunk needs its own positions
        # (the generator builds them from chunk_start_idx), the one-shot call the first total_seq rows.
        return tuple(ttnn.slice(t, [0, 0, lo, 0], [1, 1, hi, hd]) for t in (cos_tt, sin_tt))

    x = torch.randn(1, 1, total_seq, config.hidden_size, dtype=torch.bfloat16)
    # one-shot reference: the whole prompt in one prefill call
    one_shot = tt_attn(_to_device(x, mesh_device), rope_mats=rope_rows(0, total_seq), is_decode=False)
    one_shot_torch = _from_device(one_shot, mesh_device)
    one_shot.deallocate(True)

    kv_cache = init_kv_cache(
        mesh_device=mesh_device,
        config=config,
        paged_attention_config=paged_attention_config,
        cache_dtype=ttnn.bfloat16,
        mesh_config=mesh_config,
    )
    tt_attn.kv_cache = kv_cache
    page_table = torch.arange(max_num_blocks, dtype=torch.int32).reshape(1, -1)
    page_table_tt = ttnn.from_torch(page_table, device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.int32)
    outs = []
    for c in range(2):
        lo, hi = c * chunk_size, (c + 1) * chunk_size
        chunk_pt = page_table[:, lo // block_size : hi // block_size]
        chunk_pt_tt = ttnn.from_torch(chunk_pt, device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.int32)
        out = tt_attn(
            _to_device(x[:, :, lo:hi].contiguous(), mesh_device),
            rope_mats=rope_rows(lo, hi),
            is_decode=False,
            page_table=page_table_tt,
            kv_cache=kv_cache,
            chunk_start_idx=lo,
            chunk_page_table=chunk_pt_tt,
        )
        outs.append(_from_device(out, mesh_device))
        out.deallocate(True)
    chunked_torch = torch.cat(outs, dim=-2)
    assert chunked_torch.shape == one_shot_torch.shape, (chunked_torch.shape, one_shot_torch.shape)
    for name, lo, hi in (("chunk 1", 0, chunk_size), ("chunk 2", chunk_size, total_seq)):
        passing, pcc_msg = compare_tensors(
            chunked_torch[:, :, lo:hi], one_shot_torch[:, :, lo:hi], pcc_threshold=get_pcc_threshold(request)
        )
        logger.info(f"chunked prefill vs one-shot, {layer_type} layer {layer_idx}, {name}: {pcc_msg}")
        assert passing, f"{layer_type} layer {layer_idx} {name} vs one-shot prefill: {pcc_msg}"


# ===== merged from tests/unit/test_attention_fracture.py: Attention under the galaxy one-instance fracture gate vs HF: heads shard over the tp axis and are
def _setup_fractured_attention(mesh_device, layer_idx, max_seq_len=128):
    hf_text_config = TestFactory.create_hf_text_config()
    hf_layer = TestFactory.create_hf_reference_layer(hf_text_config, layer_idx)
    hf_attn = hf_layer.self_attn
    config = Gemma4AttentionConfig(TestFactory.create_hf_config(), layer_idx)

    state_dict = {k: v.clone() for k, v in hf_attn.state_dict().items() if not k.startswith("v_norm")}

    rows, _cols = tuple(mesh_device.shape)
    mesh_config = MeshConfig(
        mesh_device.shape,
        decode=ModeConfig(tp=rows),
        tp_axis=0,
        weight_fracture=True,
    )
    ccl_manager = CCLManager(mesh_device, num_links=1)

    tt_attn = Gemma4Attention(
        mesh_device=mesh_device,
        config=config,
        state_dict=state_dict,
        ccl_manager=ccl_manager,
        mesh_config=mesh_config,
        program_config=None,
        layer_idx=layer_idx,
        create_kv_cache=False,
        max_batch_size=1,
        max_seq_len=max_seq_len,
    )
    return hf_text_config, hf_attn, config, tt_attn, mesh_config


@parametrize_mesh_with_fabric(mesh_shapes=[(8, 4)])
@pytest.mark.parametrize("layer_type", ["sliding_attention", "full_attention"], ids=["sliding", "global"])
@pytest.mark.parametrize("seq_len", [128, 1024], ids=["seq128", "seq1024"])
def test_attention_fracture_prefill(layer_type, seq_len, mesh_device, reset_seeds, request):
    """Prefill PCC vs HF: heads over the tp axis, replicated across the lane axis."""
    hf_text_config = TestFactory.create_hf_text_config()
    try:
        layer_idx = find_layer_idx(hf_text_config, layer_type)
    except ValueError:
        pytest.skip(f"No {layer_type} layer in this model")

    hf_text_config, hf_attn, config, tt_attn, _ = _setup_fractured_attention(
        mesh_device, layer_idx, max_seq_len=max(seq_len, 128)
    )

    x_torch = torch.randn(1, seq_len, config.hidden_size, dtype=torch.float32)

    hf_rope = TestFactory.create_hf_rope(hf_text_config, seq_len, layer_idx)
    sliding = config.sliding_window if config.is_sliding else None
    attn_mask = build_hf_prefill_mask(seq_len, sliding_window=sliding)
    with torch.no_grad():
        ref_output, _ = hf_attn(x_torch, position_embeddings=hf_rope, attention_mask=attn_mask, shared_kv_states=None)

    cos_tt, sin_tt = TestFactory.create_tt_rope_cache(mesh_device, hf_text_config, max(seq_len, 128), layer_idx)
    x_tt = ttnn.from_torch(
        x_torch.unsqueeze(0).to(torch.bfloat16),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    tt_output = tt_attn(x_tt, rope_mats=(cos_tt, sin_tt), is_decode=False)

    device_tensors = ttnn.get_device_tensors(tt_output)
    out0 = ttnn.to_torch(device_tensors[0])
    out_last = ttnn.to_torch(device_tensors[-1])
    assert torch.equal(out0, out_last), "fractured attention output not replicated across the mesh"

    tt_output_torch = out0.squeeze(0).float()
    passing, pcc_msg = compare_tensors(tt_output_torch, ref_output, pcc_threshold=get_pcc_threshold(request))
    assert passing, (
        f"Fractured attention prefill (layer_type={layer_type}, layer_idx={layer_idx}, "
        f"seq={seq_len}) PCC too low: {pcc_msg}"
    )


# ===== merged from tests/unit/test_cp_prefill.py: CP prefill (galaxy one-instance): engagement, needle retrieval through multi-chunk prefill, and the
GEN_TOKENS = 14


FILLER = "The sky is blue and the grass is green. "


@parametrize_mesh_with_fabric(mesh_shapes=[(8, 4)])
def test_cp_prefill(mesh_device, reset_seeds, request):
    os.environ["GEMMA4_GALAXY_FRACTURE"] = "1"
    os.environ["GEMMA4_CP_PREFILL"] = "1"
    os.environ.pop("GEMMA4_GALAXY_LANES", None)  # CP and lanes are exclusive
    # No chunk override: the CP-paired default (24576, via
    # gemma4_cp_prefill_engaged) must kick in — the production pairing.

    from models.demos.gemma4.tt.attention import operations as attn_ops
    from models.demos.gemma4.tt.generator import Gemma4Generator
    from models.tt_transformers.tt.common import PagedAttentionConfig

    model_path = os.environ.get("HF_MODEL", "google/gemma-4-31B-it")

    blocks_per_user = 514  # 32896 tokens: ~26K prompt + generation
    paged_cfg = PagedAttentionConfig(block_size=64, max_num_blocks=1 + blocks_per_user)
    generator, tt_kv_cache, tokenizer = Gemma4Generator.from_pretrained(
        mesh_device,
        model_path,
        max_batch_size=32,
        max_seq_len=65536,
        paged_attention_config=paged_cfg,
    )
    assert getattr(generator.model[0].mesh_config, "cp_prefill", False), "CP gate did not engage"
    block_ids = torch.arange(1, 1 + blocks_per_user, dtype=torch.int32)

    def ask(content, expect, min_len=0):
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}], add_generation_prompt=True, tokenize=False
        )
        ids = tokenizer.encode(text, add_special_tokens=False)
        if ids and isinstance(ids[0], str):
            ids = tokenizer.convert_tokens_to_ids(ids)
        assert len(ids) >= min_len, f"prompt too short to exercise chunking: {len(ids)} < {min_len}"
        t = torch.tensor(ids, dtype=torch.long)
        plen = int(t.shape[-1])
        padded = 1 << max(int(plen - 1).bit_length(), 7)
        tok1 = torch.zeros(1, padded, dtype=torch.long)
        tok1[0, :plen] = t
        out = generator.prefill_forward_text(
            tok1,
            page_table=block_ids.unsqueeze(0),
            kv_cache=tt_kv_cache,
            prompt_lens=torch.tensor([plen]),
            empty_slots=[0],
            enable_trace=False,
            sampling_params=None,
            warmup_prefill=False,
        )
        lg = out[0] if isinstance(out, (list, tuple)) else out
        cur = int(torch.argmax(lg.reshape(-1)).item())
        toks, pos = [], plen
        for _ in range(GEN_TOKENS):
            logits = generator.decode_forward(
                torch.tensor([[cur]], dtype=torch.long),
                torch.tensor([pos], dtype=torch.long),
                page_table=block_ids.unsqueeze(0),
                kv_cache=tt_kv_cache,
                enable_trace=False,
                read_from_device=True,
                sampling_params=None,
            )
            l0 = logits[0] if isinstance(logits, (list, tuple)) else logits
            cur = int(torch.argmax(l0.reshape(-1)).item())
            toks.append(cur)
            pos += 1
        answer = tokenizer.decode(toks, skip_special_tokens=True)
        print(f"[cp] plen={plen} expect={expect!r} -> {answer!r}")
        assert expect in answer.lower(), f"expected {expect!r} in {answer!r}"
        return plen

    assert int(generator.model_args[0].max_prefill_chunk_size) == 24576, (
        "CP-paired chunk default did not resolve: " f"{generator.model_args[0].max_prefill_chunk_size}"
    )

    # ~26K tokens -> two CP chunks; the needle sits in the first and the question in the second,
    # so a per-column offset or gather bug (wrong prefix visibility) loses the needle.
    filler = " ".join(f"Entry {i}: shipment {i} arrived at dock {i % 40} on day {i % 28}." for i in range(1150))
    needle_pos = len(filler) // 4
    long_content = (
        filler[:needle_pos]
        + " The secret access code is 7291. "
        + filler[needle_pos:]
        + " What is the secret access code? Answer with just the number."
    )
    ask(long_content, "7291", min_len=24700)
    assert getattr(
        attn_ops.chunked_prefill_sdpa, "_cp_logged", False
    ), "CP branch never engaged on the aligned long prompt"

    # A short prompt stays single-chunk and must still answer correctly.
    ask("What is the opposite of hot? Answer in one word.", "cold")
