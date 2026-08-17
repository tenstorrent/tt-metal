# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests that reproduce vLLM's hybrid kv-cache layout offline.

These tests are differential: they take the same Gemma4 model and run it
twice — once with the demo's uniform paged config, once with the vLLM
harness (HMA tensor sharing + per-group block sizes + kv-shared
aliasing). Any divergence between the two flags a bug in the paged
attention machinery that's specific to the vLLM-shape path. The
existing ``test_attention.py`` / ``test_layer.py`` / ``test_model.py``
suite covers the uniform-shape side of that comparison.

Each test runs at the system's largest available mesh, matching the
e2e tests' parametrization — so on an 8-device runner 31B is exercised
at TP=8 as it would be in vLLM serving, not at an unrealistic 1x1
where the full-attention concat-heads CB overflows L1.
"""

import pytest
import torch

import ttnn
from models.demos.gemma4.config import MeshConfig, ModeConfig
from models.demos.gemma4.tt.attention import Gemma4Attention, Gemma4AttentionConfig
from models.demos.gemma4.tt.ccl import CCLManager
from models.demos.gemma4.tt.model import Gemma4Model
from models.demos.gemma4.tt.model_config import Gemma4ModelArgs

from ..test_factory import TestFactory, compare_tensors, find_layer_idx, get_pcc_threshold, parametrize_mesh_with_fabric
from ..vllm_harness import Gemma4VllmLayout, Gemma4VllmRequestPool, allocate_vllm_kv_cache


def _skip_full_model_parity_if_mesh_too_small(mesh_device):
    """Skip multi-layer parity shapes that overflow L1 on single-device 31B."""
    tp = mesh_device.shape[1] if hasattr(mesh_device, "shape") else 1
    hf_config = TestFactory.create_hf_config()
    if getattr(hf_config, "hidden_size", 0) > 4096 and tp < 2:
        pytest.skip(f"Full-model vLLM parity overflows L1 on single device (hidden={hf_config.hidden_size})")


# ── Pure-Python layout tests (no device) ─────────────────────────────────


def test_layout_matches_unifier_for_gemma4_e2b():
    """The page-size unifier doubles the smaller-byte spec's block_size.

    For Gemma4-E2B (num_kv_heads=1 on both layer types, sliding
    head_dim=256, full head_dim=512), starting from a requested
    ``block_size=64``: sliding's page_size is half of full's, so the
    unifier doubles sliding's block_size to 128 while full stays at 64.

    Other variants (E4B, 26B-A4B, 31B) have sliding's per-block bytes
    ≥ full's (more sliding kv-heads), so the unifier either no-ops or
    doubles *full* instead — opposite of E2B. The assertions below are
    E2B-specific; skip on those variants.
    """
    hf_config = TestFactory.create_hf_config()
    sliding_units = hf_config.num_key_value_heads * hf_config.head_dim
    full_units = hf_config.num_global_key_value_heads * hf_config.global_head_dim
    if sliding_units >= full_units:
        pytest.skip(
            f"Sliding per-block bytes ({sliding_units}) ≥ full ({full_units}) — "
            "unifier doesn't double sliding for this variant"
        )

    layout = Gemma4VllmLayout.from_hf_config(
        hf_config,
        num_blocks=64,
        max_model_len=4096,
        requested_block_size=64,
    )
    sliding_bs = {li.block_size for li in layout.per_layer if li.layer_type == "sliding_attention"}
    full_bs = {li.block_size for li in layout.per_layer if li.layer_type == "full_attention"}
    assert sliding_bs == {128}, f"expected sliding block_size=128 after unifier, got {sliding_bs}"
    assert full_bs == {64}, f"expected full block_size=64 (unchanged), got {full_bs}"

    # Per-block byte counts should match — the unifier's invariant.
    units_per_layer = {
        (li.layer_type, li.block_size * li.num_kv_heads_per_dev * li.head_dim) for li in layout.per_layer
    }
    units = {u for _, u in units_per_layer}
    assert len(units) == 1, f"per-block units differ across layer types: {units_per_layer}"


def test_layout_hma_tensor_sharing_alternates_sliding_full():
    """Two layers of different types at group positions 0..min(|S|, |F|)-1
    share one buffer (HMA); past that, only the larger group's layers
    occupy further tensor_idx slots.

    For Gemma4's 4-sliding-then-1-full pattern (E2B truncated to a few
    layers), the first sliding and first full share ``tensor_idx=0``.
    """
    hf_config = TestFactory.create_hf_config()
    # Find the smallest prefix that includes one full-attention layer
    # (= 5 for E2B, 6 for the larger variants).
    n = find_layer_idx(hf_config, "full_attention") + 1
    layout = Gemma4VllmLayout.from_hf_config(
        hf_config,
        num_blocks=8,
        max_model_len=256,
        requested_block_size=64,
        num_layers=n,
    )

    # tensor_idx per layer = position within its group's member list.
    # For Gemma4-E2B with layer_types=[s,s,s,s,f,...] truncated to 5:
    # sliding group = [0,1,2,3] → tensor_idx [0,1,2,3]
    # full group   = [4]       → tensor_idx [0]
    # so layers 0 and 4 share tensor_idx=0.
    first_full = find_layer_idx(hf_config, "full_attention")
    assert layout.per_layer[0].tensor_idx == 0
    assert layout.per_layer[first_full].tensor_idx == 0
    # And the allocator's recorded shape for tensor 0 is sliding's
    # (first encountered in layer-index order).
    sliding_li = layout.per_layer[0]
    full_li = layout.per_layer[first_full]
    expected_shape = (
        layout.num_blocks,
        sliding_li.num_kv_heads_per_dev,
        sliding_li.block_size,
        sliding_li.head_dim,
    )
    assert layout.tensor_alloc_shape[0] == expected_shape
    # Sanity: full layer's "native" view ≠ alloc shape — it'll use
    # block_size_override in attention/decode.py.
    full_native = (
        layout.num_blocks,
        full_li.num_kv_heads_per_dev,
        full_li.block_size,
        full_li.head_dim,
    )
    assert full_native != expected_shape


def test_layout_carries_model_kv_shared_map():
    """``kv_shared_layer_map`` derivation mirrors
    :class:`Gemma4Model`'s logic — if upstream changes either side, the
    parity tests should fail loudly here.
    """
    hf_config = TestFactory.create_hf_config()
    num_kv_shared = int(getattr(hf_config, "num_kv_shared_layers", 0) or 0)
    if num_kv_shared == 0:
        pytest.skip("Model has no kv-shared layers")

    # Build at the full layer count so the shared range falls inside.
    layout = Gemma4VllmLayout.from_hf_config(
        hf_config,
        num_blocks=4,
        max_model_len=256,
        requested_block_size=64,
        num_layers=hf_config.num_hidden_layers,
    )
    assert len(layout.kv_shared_map) == num_kv_shared, (
        f"expected {num_kv_shared} kv-shared layers, got {len(layout.kv_shared_map)}: " f"{layout.kv_shared_map}"
    )
    # Source indices must be strictly less than shared indices and
    # have the same layer_type.
    for shared_idx, source_idx in layout.kv_shared_map.items():
        assert source_idx < shared_idx, f"source {source_idx} >= shared {shared_idx}"
        assert layout.per_layer[shared_idx].layer_type == layout.per_layer[source_idx].layer_type


def test_request_pool_block_ids_disjoint_across_groups():
    """vLLM's ``BlockPool`` guarantees no cross-group block-ID collision
    per request. Our harness mimics this with a single bump cursor; the
    invariant must hold so HMA-shared layers don't address the same
    physical slot through different views.
    """
    hf_config = TestFactory.create_hf_config()
    n = find_layer_idx(hf_config, "full_attention") + 1
    layout = Gemma4VllmLayout.from_hf_config(
        hf_config,
        num_blocks=64,
        max_model_len=2048,
        requested_block_size=64,
        num_layers=n,
    )
    pool = Gemma4VllmRequestPool(layout)
    req = pool.allocate_request(num_prefill_tokens=256)

    page_tables = pool.per_layer_page_tables(req)
    # Group all non-zero (= real) block IDs by layer_type and assert
    # no duplicate across types.
    ids_by_type: dict[str, set[int]] = {}
    for li, pt in zip(layout.per_layer, page_tables):
        used_width = max(1, (256 + li.block_size - 1) // li.block_size)  # actual content extent
        row = pt[0, :used_width].tolist()
        ids_by_type.setdefault(li.layer_type, set()).update(row)
    if len(ids_by_type) > 1:
        types = list(ids_by_type)
        a, b = ids_by_type[types[0]], ids_by_type[types[1]]
        assert a.isdisjoint(b), f"block IDs collide across groups: {a & b}"


# ── Device tests: paged attention round-trip via harness ─────────────


def _replicate_mapper(mesh_device):
    is_mesh = hasattr(mesh_device, "shape") and mesh_device.get_num_devices() > 1
    return ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None


def _from_device(tensor, mesh_device):
    is_mesh = hasattr(mesh_device, "shape") and mesh_device.get_num_devices() > 1
    if is_mesh:
        return ttnn.to_torch(ttnn.get_device_tensors(tensor)[0])
    return ttnn.to_torch(tensor)


def _kv_torch_to_tt(k_torch, mesh_device, *, num_kv_heads, num_attention_heads, tp, num_devices):
    """Send a [1, num_kv_heads, seq, head_dim] KV tensor to the mesh in
    the same layout production's K_proj weight sharding produces.

    * Single device: bare ``from_torch``.
    * Sharded (``num_kv_heads >= tp``): ``ShardTensorToMesh(dim=1)`` —
      contiguous chunk of ``num_kv_heads/tp`` heads per device.
    * GQA-replicated (``num_kv_heads < tp``): build a per-device stack
      on dim 0 with each device i's KV head chosen via the production
      mapping ``kv_idx = (i * q_per_device) * num_kv_heads //
      num_attention_heads`` (see ``attention/weights.py``).
    """
    is_mesh = num_devices > 1
    if not is_mesh:
        return ttnn.from_torch(
            k_torch.to(torch.bfloat16), device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
        )
    kv_replicated = num_kv_heads < tp
    if not kv_replicated:
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


@parametrize_mesh_with_fabric()
@pytest.mark.parametrize("layer_type", ["sliding_attention", "full_attention"], ids=["sliding", "full"])
def test_attention_paged_decode_via_harness(layer_type, mesh_device, reset_seeds, request):
    """Single-layer paged decode using the vllm-harness-allocated cache.

    The classic ``test_attention_decode_paged`` uses a uniform
    ``PagedAttentionConfig(block_size=64)`` and a sequential
    ``arange(num_blocks)`` page table. This test substitutes the
    vllm-harness allocation: HMA-shape buffer (post-unifier block
    sizes), disjoint block IDs via the request pool, padded page table.
    Same PCC bar — any divergence isolates a vllm-shape bug to a
    single layer + a single forward, far easier to triage than a
    multi-token decode that's already diverged into garbage by the
    time you see logits.
    """
    hf_text_config = TestFactory.create_hf_text_config()
    try:
        layer_idx = find_layer_idx(hf_text_config, layer_type)
    except ValueError:
        pytest.skip(f"No {layer_type} layer in this model")

    hf_config = TestFactory.create_hf_config()
    tp = mesh_device.shape[1] if hasattr(mesh_device, "shape") else 1
    num_devices = mesh_device.get_num_devices() if hasattr(mesh_device, "get_num_devices") else 1
    is_mesh = num_devices > 1

    # Build a tiny vllm-style layout — just enough layers to land a
    # ``layer_idx`` of the requested type and (for ``full_attention``)
    # also include the first sliding so HMA sharing is exercised.
    n_layers = layer_idx + 1
    cache_len = 64  # small enough to fit on single device for any variant
    requested_block_size = 64
    num_blocks_per_tensor = max(
        4,
        (cache_len + requested_block_size - 1) // requested_block_size * 2,
    )
    layout = Gemma4VllmLayout.from_hf_config(
        hf_config,
        num_blocks=num_blocks_per_tensor,
        max_model_len=cache_len + 64,
        requested_block_size=requested_block_size,
        num_layers=n_layers,
        tp=tp,
        num_devices=num_devices,
    )

    hf_layer = TestFactory.create_hf_reference_layer(hf_text_config, layer_idx)
    hf_attn = hf_layer.self_attn
    config = Gemma4AttentionConfig(hf_config, layer_idx)

    state_dict = {k: v.clone() for k, v in hf_attn.state_dict().items() if not k.startswith("v_norm")}
    mesh_config = MeshConfig(mesh_device.shape, decode=ModeConfig(tp=tp))
    ccl_manager = CCLManager(mesh_device, num_links=1) if tp > 1 else None
    tt_attn = Gemma4Attention(
        mesh_device=mesh_device,
        config=config,
        state_dict=state_dict,
        ccl_manager=ccl_manager,
        mesh_config=mesh_config,
        program_config=None,
        layer_idx=layer_idx,
    )
    # Substitute the harness-allocated cache for layer ``layer_idx``.
    kv_per_layer = allocate_vllm_kv_cache(mesh_device, layout, dtype=ttnn.bfloat16)
    tt_attn.kv_cache = kv_per_layer[layer_idx]

    # Allocate a request worth of blocks, then fill the cache up to
    # ``cache_len`` so we can decode at position ``cache_len``.
    pool = Gemma4VllmRequestPool(layout)
    req = pool.allocate_request(num_prefill_tokens=cache_len)
    per_layer_pts = pool.per_layer_page_tables(req)
    pt_torch = per_layer_pts[layer_idx]
    pt_tt = ttnn.from_torch(
        pt_torch,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.int32,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None,
    )

    # Fill K/V with random reference content at the request's blocks.
    # On multi-device, _kv_torch_to_tt mirrors production K_proj sharding
    # so each device receives the slice its Q heads will attend against.
    k_ref = torch.randn(1, config.num_key_value_heads, cache_len, config.head_dim)
    v_ref = torch.randn(1, config.num_key_value_heads, cache_len, config.head_dim)
    k_fill = _kv_torch_to_tt(
        k_ref,
        mesh_device,
        num_kv_heads=config.num_key_value_heads,
        num_attention_heads=config.num_attention_heads,
        tp=tp,
        num_devices=num_devices,
    )
    v_fill = _kv_torch_to_tt(
        v_ref,
        mesh_device,
        num_kv_heads=config.num_key_value_heads,
        num_attention_heads=config.num_attention_heads,
        tp=tp,
        num_devices=num_devices,
    )
    k_cache_tt, v_cache_tt = kv_per_layer[layer_idx]
    # Per-block element-count invariant under HMA cross-group sharing:
    # input_kv * eff_bs * input_hd == cache_kv * cache_bs * cache_hd
    # → eff_bs = cache_kv * cache_bs * cache_hd // (input_kv * input_hd)
    # Required when sliding (kv=8/16) and full (kv=2/4) layers share one buffer
    # on 26B-A4B / 31B at small TP. Mirrors the production helper at
    # models/demos/gemma4/tt/attention/operations.py:effective_block_size.
    # input_kv is the per-device kv-heads of the *fill* tensor (what
    # _kv_torch_to_tt produces above), which matches production's
    # split_qkv_heads_decode: 1 under GQA replication, else config.kv // tp.
    # This may differ from cache.padded_shape[1] when the buffer was
    # allocated for a different layer type under HMA (e.g. full layer
    # reading a sliding-allocated cache on 26B-A4B at TP=4).
    input_kv_per_dev = 1 if config.num_key_value_heads < tp else config.num_key_value_heads // tp
    effective_block_size = (k_cache_tt.padded_shape[1] * k_cache_tt.padded_shape[2] * k_cache_tt.padded_shape[-1]) // (
        input_kv_per_dev * config.head_dim
    )
    # Same call shape ``attention/prefill.py`` uses on the vllm path.
    ttnn.experimental.paged_fill_cache(k_cache_tt, k_fill, pt_tt, batch_idx=0, block_size=effective_block_size)
    ttnn.experimental.paged_fill_cache(v_cache_tt, v_fill, pt_tt, batch_idx=0, block_size=effective_block_size)

    # ── HF reference (DynamicCache) ──────────────────────────────────
    from transformers.cache_utils import DynamicCache
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding

    hf_cache = DynamicCache()
    hf_cache.update(k_ref.clone(), v_ref.clone(), layer_idx=layer_idx)

    x_torch = torch.randn(1, 1, config.hidden_size, dtype=torch.float32)
    rope = Gemma4TextRotaryEmbedding(hf_text_config)
    cos, sin = rope(x_torch, torch.tensor([[cache_len]]), layer_type=layer_type)
    sliding_window = config.sliding_window if config.is_sliding else None

    # Mask: causal + sliding-window if applicable.
    total_len = cache_len + 1
    mask = torch.zeros(1, 1, 1, total_len)
    if sliding_window is not None:
        for j in range(total_len):
            if j < cache_len - sliding_window + 1:
                mask[0, 0, 0, j] = float("-inf")
    with torch.no_grad():
        ref_output, _ = hf_attn(
            x_torch,
            position_embeddings=(cos, sin),
            past_key_values=hf_cache,
            attention_mask=mask,
            shared_kv_states=None,
        )

    # ── TT decode ───────────────────────────────────────────────────
    cos_tt, sin_tt = TestFactory.create_tt_rope_cache(mesh_device, hf_text_config, max(cache_len + 64, 128), layer_idx)
    x_tt = ttnn.from_torch(
        x_torch.unsqueeze(0).to(torch.bfloat16),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None,
    )
    position_idx_tt = ttnn.from_torch(
        torch.tensor([[cache_len]], dtype=torch.int32),
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.int32,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None,
    )
    tt_output = tt_attn(
        x_tt,
        rope_mats=(cos_tt, sin_tt),
        position_idx=position_idx_tt,
        is_decode=True,
        token_index=cache_len,
        page_table=pt_tt,
    )
    tt_output_torch = _from_device(tt_output, mesh_device).squeeze(0).float()
    passing, pcc_msg = compare_tensors(tt_output_torch, ref_output, pcc_threshold=get_pcc_threshold(request))
    assert passing, (
        f"vllm-harness paged decode (layer_type={layer_type}, "
        f"layer_idx={layer_idx}, cache_len={cache_len}, "
        f"effective_block_size={effective_block_size}) PCC too low: {pcc_msg}"
    )


# ── KV-sharing alias round trip ──────────────────────────────────────


@parametrize_mesh_with_fabric()
def test_kv_shared_alias_round_trip(mesh_device, reset_seeds):
    """A shared layer reads what its source wrote, via the alias.

    Constructs a 2-sliding-layer layout where layer 1 shares from
    layer 0 (forced kv_shared_map), allocates the cache, writes to
    layer 0's cache, and asserts that reading layer 1's cache returns
    the same data — which is the invariant
    :class:`Gemma4ForCausalLM.allocate_kv_cache_per_layer` exists to
    enforce for the real shared-layer indices.
    """
    hf_config = TestFactory.create_hf_config()
    try:
        find_layer_idx(hf_config, "sliding_attention")
    except ValueError:
        pytest.skip("No sliding layer in this model")

    # Force a tiny "fake" shared map for the test: layer 1 shares from layer 0.
    # We still need both layers to be sliding so they have matching shape,
    # which holds for any prefix of Gemma4 starting with sliding.
    if hf_config.layer_types[1] != "sliding_attention":
        pytest.skip("Test assumes layer 1 is also sliding (Gemma4 layer-type pattern)")

    tp = mesh_device.shape[1] if hasattr(mesh_device, "shape") else 1
    num_devices = mesh_device.get_num_devices() if hasattr(mesh_device, "get_num_devices") else 1
    is_mesh = num_devices > 1

    layout = Gemma4VllmLayout.from_hf_config(
        hf_config,
        num_blocks=8,
        max_model_len=512,
        requested_block_size=64,
        num_layers=2,
        kv_shared_map={1: 0},
        tp=tp,
        num_devices=num_devices,
    )
    assert layout.kv_shared_map == {1: 0}

    kv_per_layer = allocate_vllm_kv_cache(mesh_device, layout, dtype=ttnn.bfloat16)
    # Aliasing is identity at the Python-object level — the model side
    # relies on this so ``caches[shared] is caches[source]``.
    assert kv_per_layer[1][0] is kv_per_layer[0][0], "K aliasing missing"
    assert kv_per_layer[1][1] is kv_per_layer[0][1], "V aliasing missing"

    # Write known data through layer 0's view, read back via layer 1's
    # view — same buffer, so the read should match.
    config0 = Gemma4AttentionConfig(hf_config, 0)
    pool = Gemma4VllmRequestPool(layout)
    req = pool.allocate_request(num_prefill_tokens=64)
    per_layer_pts = pool.per_layer_page_tables(req)
    pt_layer_0 = ttnn.from_torch(
        per_layer_pts[0],
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.int32,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None,
    )

    # Per-device K shape matches the harness-allocated cache shape; data
    # content doesn't matter (the test only checks Python identity of
    # the alias, not data correctness — that's covered by the
    # full-model parity tests).
    k_cache_tt, _ = kv_per_layer[0]
    num_local_kv_heads = k_cache_tt.padded_shape[1]
    k_ref = torch.randn(1, num_local_kv_heads, 64, config0.head_dim)
    k_fill = ttnn.from_torch(
        k_ref.to(torch.bfloat16),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None,
    )
    # See test_attention_paged_decode_via_harness for the asymmetric-kv formula.
    eff_bs = (k_cache_tt.padded_shape[1] * k_cache_tt.padded_shape[2] * k_cache_tt.padded_shape[-1]) // (
        num_local_kv_heads * config0.head_dim
    )
    ttnn.experimental.paged_fill_cache(k_cache_tt, k_fill, pt_layer_0, batch_idx=0, block_size=eff_bs)

    # Read back: tensor identity guarantees this is the same buffer.
    k_layer_1, _ = kv_per_layer[1]
    assert k_layer_1 is k_cache_tt, "post-alloc identity broken — alias diverged after fill"


# ── Decoder layer via harness ────────────────────────────────────────


@parametrize_mesh_with_fabric()
def test_layer_forward_decode_via_harness(mesh_device, reset_seeds, request):
    """Decoder-layer decode with the vLLM harness allocator.

    Mirrors ``test_layer_forward_decode`` in ``test_layer.py`` but
    swaps its contiguous ``init_kv_cache`` + ``fill_cache`` setup for
    the harness's paged layout (post-unifier block sizes, HMA-shared
    buffer, per-layer page tables). Same PCC bar against the same HF
    reference. Catches regressions where a decoder layer works under
    the contiguous path but breaks under the paged + vLLM-shape path
    — which is the only path exercised in production.

    Single layer at index 0 (sliding); the attention sub-test in this
    file already exercises layer 4 (full) on a HMA-shared buffer.
    """
    from transformers.cache_utils import DynamicCache
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding

    from models.demos.gemma4.tt.layer import Gemma4DecoderLayer

    from ..unit.test_layer import (
        _create_gemma4_model_args,
        _create_hf_reference_layer,
        _create_hf_text_config,
        _hf_state_to_tt_state,
    )

    layer_idx = 0
    cache_len = 32

    hf_text_config = _create_hf_text_config(num_experts=4, top_k=2)
    hf_layer = _create_hf_reference_layer(hf_text_config, layer_idx)
    tt_state = _hf_state_to_tt_state(hf_layer.state_dict(), layer_idx)
    model_args = _create_gemma4_model_args(hf_text_config)
    attn_cfg = Gemma4AttentionConfig(model_args, layer_idx)

    tp = mesh_device.shape[1] if hasattr(mesh_device, "shape") else 1
    num_devices = mesh_device.get_num_devices() if hasattr(mesh_device, "get_num_devices") else 1
    is_mesh = num_devices > 1
    mesh_config = MeshConfig(mesh_device.shape, decode=ModeConfig(tp=tp))
    ccl_manager = CCLManager(mesh_device, num_links=1) if tp > 1 else None
    tt_layer = Gemma4DecoderLayer(
        mesh_device=mesh_device,
        hf_config=model_args,
        state_dict=tt_state,
        layer_idx=layer_idx,
        ccl_manager=ccl_manager,
        dtype=ttnn.bfloat16,
        tensor_cache_path=None,
        mesh_config=mesh_config,
        max_seq_len=cache_len + 32,
        max_local_batch_size=1,
    )

    # Harness allocation — covers just this layer (one sliding layer is
    # enough to land tensor_idx=0 at the layer's own native shape; HMA
    # cross-view is exercised by the attention sub-test).
    layout = Gemma4VllmLayout.from_hf_config(
        hf_text_config,
        num_blocks=max(4, (cache_len + 64 + 63) // 64 * 2),
        max_model_len=cache_len + 64,
        requested_block_size=64,
        num_layers=1,
        tp=tp,
        num_devices=num_devices,
    )
    kv_per_layer = allocate_vllm_kv_cache(mesh_device, layout, dtype=ttnn.bfloat16)
    tt_layer.self_attn.kv_cache = kv_per_layer[layer_idx]

    # Fill cache via the same paged_fill_cache call the model uses.
    pool = Gemma4VllmRequestPool(layout)
    req = pool.allocate_request(num_prefill_tokens=cache_len)
    per_layer_pts = pool.per_layer_page_tables(req)
    pt_tt = ttnn.from_torch(
        per_layer_pts[layer_idx],
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.int32,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None,
    )
    k_data = torch.randn(1, attn_cfg.num_key_value_heads, cache_len, attn_cfg.head_dim)
    v_data = torch.randn(1, attn_cfg.num_key_value_heads, cache_len, attn_cfg.head_dim)
    k_cache_tt, v_cache_tt = kv_per_layer[layer_idx]
    # See test_attention_paged_decode_via_harness for the asymmetric-kv formula.
    # input_kv is the per-device kv-heads count of the *fill* tensor
    # (what _kv_torch_to_tt produces below), which mirrors production's
    # split_qkv_heads_decode: 1 under GQA replication, else attn_cfg.kv // tp.
    # Differs from cache.padded_shape[1] under HMA cross-group sharing.
    input_kv_per_dev = 1 if attn_cfg.num_key_value_heads < tp else attn_cfg.num_key_value_heads // tp
    eff_bs = (k_cache_tt.padded_shape[1] * k_cache_tt.padded_shape[2] * k_cache_tt.padded_shape[-1]) // (
        input_kv_per_dev * attn_cfg.head_dim
    )
    k_fill = _kv_torch_to_tt(
        k_data,
        mesh_device,
        num_kv_heads=attn_cfg.num_key_value_heads,
        num_attention_heads=attn_cfg.num_attention_heads,
        tp=tp,
        num_devices=num_devices,
    )
    v_fill = _kv_torch_to_tt(
        v_data,
        mesh_device,
        num_kv_heads=attn_cfg.num_key_value_heads,
        num_attention_heads=attn_cfg.num_attention_heads,
        tp=tp,
        num_devices=num_devices,
    )
    ttnn.experimental.paged_fill_cache(k_cache_tt, k_fill, pt_tt, batch_idx=0, block_size=eff_bs)
    ttnn.experimental.paged_fill_cache(v_cache_tt, v_fill, pt_tt, batch_idx=0, block_size=eff_bs)

    # HF reference (DynamicCache holds the same K/V at the same positions).
    hf_cache = DynamicCache()
    hf_cache.update(k_data.clone(), v_data.clone(), layer_idx=layer_idx)
    x_torch = torch.randn(1, 1, model_args.hidden_size, dtype=torch.float32)
    rope = Gemma4TextRotaryEmbedding(hf_text_config)
    cos, sin = rope(x_torch, torch.tensor([[cache_len]]), layer_type=hf_text_config.layer_types[layer_idx])
    mask = torch.zeros(1, 1, 1, cache_len + 1)
    sliding_window = attn_cfg.sliding_window if attn_cfg.is_sliding else None
    if sliding_window is not None and cache_len + 1 > sliding_window:
        for j in range(cache_len + 1):
            if j < cache_len - sliding_window + 1:
                mask[0, 0, 0, j] = float("-inf")
    with torch.no_grad():
        hf_output = hf_layer(
            x_torch, per_layer_input=None, position_embeddings=(cos, sin), past_key_values=hf_cache, attention_mask=mask
        )

    # TT forward on the harness-shaped cache + page table.
    cos_tt, sin_tt = TestFactory.create_tt_rope_cache(mesh_device, hf_text_config, cache_len + 64, layer_idx)
    x_tt = ttnn.from_torch(
        x_torch.unsqueeze(0).to(torch.bfloat16),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None,
    )
    position_idx_tt = ttnn.from_torch(
        torch.tensor([[cache_len]], dtype=torch.int32),
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.int32,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None,
    )
    tt_output = tt_layer(
        x_tt,
        rope_mats=(cos_tt, sin_tt),
        position_idx=position_idx_tt,
        page_table=pt_tt,
        kv_cache=kv_per_layer[layer_idx],
        is_decode=True,
        token_index=cache_len,
    )
    tt_output_torch = _from_device(tt_output, mesh_device).squeeze(0).float()
    passing, pcc_msg = compare_tensors(tt_output_torch, hf_output, pcc_threshold=get_pcc_threshold(request))
    assert passing, f"layer-decode-via-harness PCC too low: {pcc_msg}"


# ── Multi-layer parity: uniform vs vLLM-shape ────────────────────────


@parametrize_mesh_with_fabric()
def test_full_model_parity_uniform_vs_vllm(mesh_device, reset_seeds, request):
    """Run a multi-layer Gemma4 model twice and compare logits.

    First pass: demo-style — uniform ``PagedAttentionConfig(block_size=64)``,
    sequential ``arange`` page table, model-side ``self.tt_kv_cache``
    (with kv-share aliasing already applied in ``Gemma4Model.__init__``).

    Second pass: vllm-style — :func:`allocate_vllm_kv_cache` for the
    cache, :class:`Gemma4VllmRequestPool` for the page_tables (per
    layer, padded, disjoint block IDs across groups), and the model's
    ``page_tables_per_layer=`` argument to route them through to
    attention.

    Both passes feed the *same* prompt through the *same* weights, then
    run ``N`` decode steps starting from the same prefill output.
    Logit PCC at every step should stay above the configured threshold;
    any drop flags a bug that's specific to the vLLM-shape paged
    attention codepath (HMA tensor sharing, block_size_override,
    per-layer routing) — exactly the class of bug uniform-shape tests
    miss.
    """
    _skip_full_model_parity_if_mesh_too_small(mesh_device)

    from ...tests.test_factory import num_layers_for_full_attention_group
    from ..unit.test_model import _create_hf_model, _create_hf_text_config, _hf_model_state_to_tt_state

    # Smallest variant of the layer pattern that includes both sliding
    # and full layers (4 sliding + 1 full for Gemma4-E2B). That covers
    # the cross-group HMA sharing path.
    base_for_count = _create_hf_text_config(vocab_size=256, num_layers=1)
    num_layers = num_layers_for_full_attention_group(base_for_count)

    hf_text_config = _create_hf_text_config(vocab_size=256, num_layers=num_layers)
    hf_model = _create_hf_model(hf_text_config)
    model_args = Gemma4ModelArgs.from_hf_config(hf_text_config)
    model_args._hf_text_config = hf_text_config
    tt_state = _hf_model_state_to_tt_state(hf_model)

    seq_len = 32
    decode_steps = 4
    tp = mesh_device.shape[1] if hasattr(mesh_device, "shape") else 1
    num_devices = mesh_device.get_num_devices() if hasattr(mesh_device, "get_num_devices") else 1
    mesh_config = MeshConfig(mesh_device.shape, decode=ModeConfig(tp=tp))
    ccl_manager = CCLManager(mesh_device, num_links=1) if tp > 1 else None

    # Single shared model across both passes — weights are heavy enough on
    # 26B-A4B (128 experts × 6 layers) that two simultaneous instances OOM
    # the per-bank DRAM budget on wh_llmbox. The model's own kv_cache feeds
    # the uniform pass via the kv_cache=None fall-through, and the vLLM pass
    # supplies its harness-allocated cache + per-layer page tables as
    # explicit kwargs. Both passes touch disjoint cache buffers so neither
    # pollutes the other's state.
    tt_model = Gemma4Model(
        mesh_device=mesh_device,
        hf_config=model_args,
        state_dict=tt_state,
        ccl_manager=ccl_manager,
        dtype=ttnn.bfloat16,
        tensor_cache_path=None,
        mesh_config=mesh_config,
        max_seq_len=seq_len + decode_steps + 32,
        max_local_batch_size=1,
        num_layers=num_layers,
    )

    tokens = torch.randint(0, model_args.vocab_size, (1, seq_len), dtype=torch.long)

    # ── Uniform path: prefill ───────────────────────────────────────
    replicate = _replicate_mapper(mesh_device)
    tt_tokens = ttnn.from_torch(
        tokens.to(torch.int32),
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint32,
        mesh_mapper=replicate,
    )
    tt_embeds = tt_model.embed_tokens(tt_tokens)
    tt_embeds = ttnn.reshape(tt_embeds, (1, 1, seq_len, model_args.hidden_size))
    tt_embeds = ttnn.to_layout(tt_embeds, ttnn.TILE_LAYOUT)
    uniform_prefill_logits = tt_model(
        tt_embeds, rope_mats=None, position_idx=None, page_table=None, kv_caches=None, is_decode=False
    )
    uniform_prefill_torch = _from_device(uniform_prefill_logits, mesh_device).float()
    uniform_prefill_logits.deallocate(True)

    # ── vLLM path: same model, harness-allocated cache + page tables ─
    requested_block_size = 64
    max_model_len = seq_len + decode_steps + 32
    layout = Gemma4VllmLayout.from_hf_config(
        hf_text_config,
        num_blocks=max(
            16,
            (max_model_len + requested_block_size - 1) // requested_block_size * 2,
        ),
        max_model_len=max_model_len,
        requested_block_size=requested_block_size,
        num_layers=num_layers,
        tp=tp,
        num_devices=num_devices,
    )
    kv_per_layer = allocate_vllm_kv_cache(mesh_device, layout, dtype=ttnn.bfloat16)
    pool = Gemma4VllmRequestPool(layout)
    req = pool.allocate_request(num_prefill_tokens=seq_len)
    per_layer_pts = pool.per_layer_page_tables(req)

    tt_tokens_v = ttnn.from_torch(
        tokens.to(torch.int32),
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint32,
        mesh_mapper=replicate,
    )
    tt_embeds_v = tt_model.embed_tokens(tt_tokens_v)
    tt_embeds_v = ttnn.reshape(tt_embeds_v, (1, 1, seq_len, model_args.hidden_size))
    tt_embeds_v = ttnn.to_layout(tt_embeds_v, ttnn.TILE_LAYOUT)
    vllm_prefill_logits = tt_model.ttnn_prefill_forward(
        tt_embeds_v,
        page_table=None,
        kv_cache=kv_per_layer,
        input_ids_torch=tokens,
        embeds_torch=None,
        page_tables_per_layer=per_layer_pts,
    )
    vllm_prefill_torch = _from_device(vllm_prefill_logits, mesh_device).float()
    vllm_prefill_logits.deallocate(True)

    # Prefill parity — same prompt + same weights + same RoPE, the
    # only difference is paged cache layout. Mismatch here is a bug in
    # ``paged_fill_cache`` / ``Gemma4Model.__call__`` routing.
    passing, pcc_msg = compare_tensors(
        vllm_prefill_torch, uniform_prefill_torch, pcc_threshold=get_pcc_threshold(request)
    )
    assert passing, f"prefill parity uniform vs vllm-shape: {pcc_msg}"

    # ── Decode N steps in lockstep ──────────────────────────────────
    # Sample greedy from prefill's last position on both paths, then
    # feed the same token into both decoders for ``decode_steps`` and
    # PCC the logits at each step.
    last_pos = seq_len - 1
    uniform_token = uniform_prefill_torch[0, 0, last_pos, : model_args.vocab_size].argmax().item()
    vllm_token = vllm_prefill_torch[0, 0, last_pos, : model_args.vocab_size].argmax().item()
    assert (
        uniform_token == vllm_token
    ), f"first-decoded token diverges already: uniform={uniform_token}, vllm={vllm_token}"

    del pool, req  # prefill-only parity here; decode covered by the dedicated test below


# ── Decode parity: where the chat-completion garbage lives ───────────


def _build_parity_models(
    mesh_device,
    hf_text_config,
    model_args,
    tt_state,
    mesh_config,
    num_layers,
    max_total_len,
    uniform_paged_cfg,
    ccl_manager=None,
):
    """Build the Gemma4Model(s) used by the parity tests.

    Originally returned two distinct instances (one owning the uniform
    kv-cache, the other with ``create_kv_cache=False`` so its forward
    paths consumed the harness-allocated cache via ``kv_cache=``). For
    26B-A4B (128 experts × 6 layers) two weight copies plus the
    harness cache overflow the per-bank DRAM budget on wh_llmbox.

    Now returns the *same* instance twice: the model's internal kv-cache
    feeds the uniform pass (``kv_caches=None`` fall-through), and the
    vLLM pass supplies its harness-allocated cache + per-layer page
    tables as explicit ``kv_cache=`` / ``page_tables_per_layer=`` kwargs.
    Both passes touch disjoint cache buffers, so weights stay shared
    without state cross-pollution. The two-variable signature is kept
    so existing call sites (uniform / vllm method calls on separately
    named locals) need no churn.

    The ``_active_page_tables_per_layer`` stash on the vllm side is
    fine under this single-model arrangement because every vllm caller
    in this file either passes ``page_tables_per_layer=`` explicitly
    (overriding the stash) or ``del``-s it before the next uniform
    call — none rely on the model carrying it between passes.
    """
    _inject_missing_kv_shared_attention_weights(tt_state, hf_text_config, num_layers)

    tt_model = Gemma4Model(
        mesh_device=mesh_device,
        hf_config=model_args,
        state_dict=tt_state,
        ccl_manager=ccl_manager,
        dtype=ttnn.bfloat16,
        tensor_cache_path=None,
        mesh_config=mesh_config,
        max_seq_len=max_total_len,
        max_local_batch_size=1,
        num_layers=num_layers,
        paged_attention_config=uniform_paged_cfg,
    )
    return tt_model, tt_model


def _decode_step_inputs(model, mesh_device, token_id, position):
    """Build the host-side decode inputs both paths consume.

    Mirrors what ``Gemma4Model.prepare_decode_inputs_host`` packs into
    its return tuple. Returns ``(tokens, pos_uint32, pos_int32, pli)``;
    ``pli`` is ``None`` when the model has no PLI configured. Caller is
    responsible for deallocating the returned device tensors after
    each step.
    """
    import torch.nn.functional as F

    pli = model.compute_host_pli(token_id)
    is_mesh = hasattr(mesh_device, "shape") and mesh_device.get_num_devices() > 1
    mapper = ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None

    tokens_tt = ttnn.from_torch(
        torch.tensor([[token_id]], dtype=torch.int32),
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint32,
        mesh_mapper=mapper,
    )
    pos_padded = F.pad(torch.tensor([position], dtype=torch.int32).reshape(1, 1), (0, 31), "constant", 0)
    pos_tt = ttnn.from_torch(
        pos_padded, device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32, mesh_mapper=mapper
    )
    pos_int32_tt = ttnn.from_torch(
        torch.tensor([position], dtype=torch.int32),
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.int32,
        mesh_mapper=mapper,
    )
    pli_tt = None
    if pli is not None:
        pli_tt = ttnn.from_torch(
            pli.to(torch.bfloat16),
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=mapper,
        )
    return tokens_tt, pos_tt, pos_int32_tt, pli_tt


def _page_table_to_tt(page_table_torch, mesh_device):
    is_mesh = hasattr(mesh_device, "shape") and mesh_device.get_num_devices() > 1
    return ttnn.from_torch(
        page_table_torch,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.int32,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None,
    )


def _compute_num_layers_for_kv_shared(hf_text_config) -> int | None:
    """Smallest layer count that engages at least one kv-shared layer.

    Reads the *full* model's layer count + kv-shared count from the
    real HF config rather than from any truncated test config — those
    truncate ``num_hidden_layers`` but leave ``num_kv_shared_layers``
    at the original value, which makes
    ``num_hidden_layers - num_kv_shared_layers`` go negative.

    Returns ``None`` when the model has no kv-shared layers.
    """
    real = TestFactory.create_hf_config()
    full_n = int(getattr(real, "num_hidden_layers", 0) or 0)
    kv_shared = int(getattr(real, "num_kv_shared_layers", 0) or 0)
    if kv_shared <= 0 or full_n <= 0:
        return None
    # First shared layer index = full_n - kv_shared. Truncating to that
    # +1 keeps exactly one shared layer in the test config.
    return max(1, full_n - kv_shared + 1)


def _inject_missing_kv_shared_attention_weights(state_dict, hf_text_config, num_layers, prefix="model."):
    """Pad synthetic HF states whose kv-shared layers omit K/V weights.

    HF checkpoints for Gemma4 variants with ``num_kv_shared_layers`` can omit
    K/V projection tensors on shared layers because those layers reuse a source
    layer's cache at runtime. The TT attention constructor still builds a fused
    QKV tensor for every layer before runtime can discard K/V under
    ``is_kv_shared=True``, so test-generated full-layer states need zero K/V
    placeholders for any omitted entries.
    """
    hidden = int(hf_text_config.hidden_size)
    for layer_idx in range(num_layers):
        attn_prefix = f"{prefix}layers.{layer_idx}.self_attn"
        layer_type = hf_text_config.layer_types[layer_idx]
        is_sliding = layer_type == "sliding_attention"
        if is_sliding:
            num_kv_heads = int(hf_text_config.num_key_value_heads)
            head_dim = int(hf_text_config.head_dim)
        else:
            num_kv_heads = int(
                getattr(hf_text_config, "num_global_key_value_heads", None) or hf_text_config.num_key_value_heads
            )
            head_dim = int(getattr(hf_text_config, "global_head_dim", None) or hf_text_config.head_dim)
        kv_size = num_kv_heads * head_dim

        state_dict.setdefault(
            f"{attn_prefix}.k_proj.weight",
            torch.zeros((kv_size, hidden), dtype=torch.bfloat16),
        )
        use_kv_tying = bool(getattr(hf_text_config, "attention_k_eq_v", False)) and not is_sliding
        if not use_kv_tying:
            state_dict.setdefault(
                f"{attn_prefix}.v_proj.weight",
                torch.zeros((kv_size, hidden), dtype=torch.bfloat16),
            )
        state_dict.setdefault(
            f"{attn_prefix}.k_norm.weight",
            torch.ones((head_dim,), dtype=torch.bfloat16),
        )
    return state_dict


@parametrize_mesh_with_fabric()
@pytest.mark.parametrize("decode_steps", [4, 8], ids=lambda n: f"steps{n}")
@pytest.mark.parametrize(
    "layer_set", ["small", "with_kv_shared", "all_kv_shared"], ids=["small", "kv-shared", "all-kv-shared"]
)
def test_full_model_parity_decode_uniform_vs_vllm(layer_set, decode_steps, mesh_device, reset_seeds, request):
    """Multi-token decode parity between uniform-paged and vLLM-shape paths.

    This is the test that turns "garbage logits at vllm chat completion
    step 2+" into a single PCC failure on a 5-layer model in seconds.
    Both paths share weights, prompt, RoPE, sampling. The only
    differences are:

    * uniform path runs against a uniform-block-size paged cache
      allocated by ``Gemma4Model.__init__`` with the demo's
      ``PagedAttentionConfig`` and a single ``arange`` page table fed
      into every layer;
    * vLLM path runs against :func:`allocate_vllm_kv_cache` (HMA tensor
      sharing + kv-share aliasing) with per-layer page tables from
      :class:`Gemma4VllmRequestPool` and the persistent-device-buffer
      refresh dance that :class:`Gemma4ForCausalLM.decode_forward`
      normally does via the bridge.

    Each step does (a) ``update_persistent_per_layer_page_tables`` on
    the vllm side, (b) ``ttnn_decode_forward`` on both sides, (c) PCC
    on the produced logits, (d) greedy sample on the *uniform* side
    and feed that token into both for the next step (so the test
    measures parity, not "do they pick the same EOS").
    """
    _skip_full_model_parity_if_mesh_too_small(mesh_device)

    from models.tt_transformers.tt.common import PagedAttentionConfig

    from ...tests.test_factory import num_layers_for_full_attention_group
    from ..unit.test_model import _create_hf_model, _create_hf_text_config, _hf_model_state_to_tt_state

    base_for_count = _create_hf_text_config(vocab_size=256, num_layers=1)

    if layer_set == "small":
        num_layers = num_layers_for_full_attention_group(base_for_count)
        kv_shared_override = None
    elif layer_set == "with_kv_shared":
        kv_layers = _compute_num_layers_for_kv_shared(base_for_count)
        if kv_layers is None:
            pytest.skip("Model has no num_kv_shared_layers — kv-shared parity not applicable")
        num_layers = kv_layers
        # When the test truncates ``num_hidden_layers`` to ``num_layers``,
        # the model's ``first_shared_idx = num_hidden_layers -
        # num_kv_shared_layers`` formula needs ``num_kv_shared_layers``
        # to be rescaled too — otherwise it goes negative and the
        # kv-shared loop indexes ``layer_types`` past the truncated end.
        # Engage exactly one shared layer at the tail of the test
        # config: ``num_kv_shared_layers = 1`` ⇒ ``first_shared_idx =
        # num_layers - 1``.
        kv_shared_override = 1
    else:
        # "all_kv_shared": run the full layer count from the real
        # model so every kv-shared layer in
        # ``Gemma4Model.kv_shared_layer_map`` is exercised
        # simultaneously. This matches the production shape and is the
        # closest pure-model test to the failing vLLM chat scenario.
        real = TestFactory.create_hf_config()
        num_layers = int(real.num_hidden_layers)
        kv_shared_override = int(getattr(real, "num_kv_shared_layers", 0) or 0)
        if kv_shared_override <= 0:
            pytest.skip("Model has no num_kv_shared_layers — all-kv-shared case not applicable")

    hf_text_config = _create_hf_text_config(vocab_size=256, num_layers=num_layers)
    if kv_shared_override is not None:
        hf_text_config.num_kv_shared_layers = kv_shared_override
    hf_model = _create_hf_model(hf_text_config)
    model_args = Gemma4ModelArgs.from_hf_config(hf_text_config)
    model_args._hf_text_config = hf_text_config
    tt_state = _hf_model_state_to_tt_state(hf_model)

    seq_len = 32
    max_total_len = seq_len + decode_steps + 32
    tp = mesh_device.shape[1] if hasattr(mesh_device, "shape") else 1
    num_devices = mesh_device.get_num_devices() if hasattr(mesh_device, "get_num_devices") else 1
    mesh_config = MeshConfig(mesh_device.shape, decode=ModeConfig(tp=tp))
    ccl_manager = CCLManager(mesh_device, num_links=1) if tp > 1 else None

    # ── Uniform paged config ────────────────────────────────────────
    uniform_block_size = 64
    uniform_num_blocks = max(
        16,
        (max_total_len + uniform_block_size - 1) // uniform_block_size * 2,
    )
    uniform_paged_cfg = PagedAttentionConfig(block_size=uniform_block_size, max_num_blocks=uniform_num_blocks)

    tt_model_uniform, tt_model_vllm = _build_parity_models(
        mesh_device,
        hf_text_config,
        model_args,
        tt_state,
        mesh_config,
        num_layers,
        max_total_len,
        uniform_paged_cfg,
        ccl_manager=ccl_manager,
    )

    # ── Harness allocation for the vllm path ────────────────────────
    requested_block_size = 64
    layout = Gemma4VllmLayout.from_hf_config(
        hf_text_config,
        num_blocks=max(
            16,
            (max_total_len + requested_block_size - 1) // requested_block_size * 2,
        ),
        max_model_len=max_total_len,
        requested_block_size=requested_block_size,
        num_layers=num_layers,
        tp=tp,
        num_devices=num_devices,
    )
    kv_vllm = allocate_vllm_kv_cache(mesh_device, layout, dtype=ttnn.bfloat16)
    pool = Gemma4VllmRequestPool(layout)
    req = pool.allocate_request(num_prefill_tokens=seq_len)
    pts_prefill_torch = pool.per_layer_page_tables(req)

    # ── Prefill on both ─────────────────────────────────────────────
    tokens = torch.randint(0, model_args.vocab_size, (1, seq_len), dtype=torch.long)
    replicate = _replicate_mapper(mesh_device)

    def _embed_tokens(tt_model, toks):
        tt_tokens = ttnn.from_torch(
            toks.to(torch.int32),
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.uint32,
            mesh_mapper=replicate,
        )
        e = tt_model.embed_tokens(tt_tokens)
        e = ttnn.reshape(e, (1, 1, seq_len, model_args.hidden_size))
        return ttnn.to_layout(e, ttnn.TILE_LAYOUT)

    embeds_u = _embed_tokens(tt_model_uniform, tokens)
    embeds_v = _embed_tokens(tt_model_vllm, tokens)
    uniform_pt_prefill = _page_table_to_tt(
        torch.arange(uniform_num_blocks, dtype=torch.int32).reshape(1, uniform_num_blocks), mesh_device
    )

    uniform_prefill_logits = tt_model_uniform.ttnn_prefill_forward(
        embeds_u,
        page_table=uniform_pt_prefill,
        kv_cache=None,  # uses tt_model_uniform.tt_kv_cache (paged uniform)
        input_ids_torch=tokens,
        embeds_torch=None,
    )
    vllm_prefill_logits = tt_model_vllm.ttnn_prefill_forward(
        embeds_v,
        page_table=None,
        kv_cache=kv_vllm,
        input_ids_torch=tokens,
        embeds_torch=None,
        page_tables_per_layer=pts_prefill_torch,
    )

    uniform_prefill_torch = _from_device(uniform_prefill_logits, mesh_device).float()
    vllm_prefill_torch = _from_device(vllm_prefill_logits, mesh_device).float()

    last_pos = seq_len - 1
    passing, pcc_msg = compare_tensors(
        vllm_prefill_torch[..., last_pos, : model_args.vocab_size],
        uniform_prefill_torch[..., last_pos, : model_args.vocab_size],
        pcc_threshold=get_pcc_threshold(request),
    )
    assert passing, f"prefill last-token parity: {pcc_msg}"

    # Greedy from the uniform side; this is the canonical "next token"
    # we feed into both decoders so any divergence is on logits, not
    # on token selection.
    cur_tok = uniform_prefill_torch[0, 0, last_pos, : model_args.vocab_size].argmax().item()

    # Free prefill output tensors before stepping into decode — these
    # are big enough that holding them across decode allocations risks
    # DRAM pressure on single-device runs.
    uniform_prefill_logits.deallocate(True)
    vllm_prefill_logits.deallocate(True)

    # ── Decode steps ────────────────────────────────────────────────
    failures: list[str] = []
    for step in range(decode_steps):
        position = seq_len + step

        # Advance the harness pool to ensure block IDs exist for
        # this position before the decode reads/writes them.
        pool.reserve_decode_token(req)
        pts_step_torch = pool.per_layer_page_tables(req)

        # Uniform decode inputs.
        u_embeds, u_pos, u_pos_int32, u_pli = _decode_step_inputs(tt_model_uniform, mesh_device, cur_tok, position)
        u_pt = _page_table_to_tt(
            torch.arange(uniform_num_blocks, dtype=torch.int32).reshape(1, uniform_num_blocks)[0:1],
            mesh_device,
        )

        u_logits, _ = tt_model_uniform.ttnn_decode_forward(
            u_embeds, u_pos, u_pos_int32, u_pt, kv_cache=None, pli_combined=u_pli
        )

        # vLLM decode: refresh persistent buffers, build the legacy
        # page_table view from the same request (= group 0's
        # post-padding block IDs), then call the same entrypoint.
        tt_model_vllm.update_persistent_per_layer_page_tables(pts_step_torch)
        v_embeds, v_pos, v_pos_int32, v_pli = _decode_step_inputs(tt_model_vllm, mesh_device, cur_tok, position)
        v_pt = _page_table_to_tt(pool.legacy_page_table(req), mesh_device)
        v_logits, _ = tt_model_vllm.ttnn_decode_forward(
            v_embeds,
            v_pos,
            v_pos_int32,
            v_pt,
            kv_cache=kv_vllm,
            page_tables_per_layer=pts_step_torch,
            pli_combined=v_pli,
        )

        u_logits_torch = _from_device(u_logits, mesh_device).float()
        v_logits_torch = _from_device(v_logits, mesh_device).float()

        # Logits shapes are ``[1, 1, 1, vocab_size]``; PCC over the
        # vocab dimension at the only generated position.
        u_slice = u_logits_torch[..., : model_args.vocab_size]
        v_slice = v_logits_torch[..., : model_args.vocab_size]
        passing, pcc_msg = compare_tensors(v_slice, u_slice, pcc_threshold=get_pcc_threshold(request))
        if not passing:
            u_top = u_slice.flatten().argmax().item()
            v_top = v_slice.flatten().argmax().item()
            failures.append(f"step {step} (pos {position}): {pcc_msg} — uniform_top={u_top} vllm_top={v_top}")

        # Next-token sample from uniform side so both decoders see
        # identical inputs every step.
        cur_tok = u_slice.flatten().argmax().item()

        u_logits.deallocate(True)
        v_logits.deallocate(True)

    assert not failures, "decode parity diverged:\n" + "\n".join(failures)


# ── PLI parity: enables hidden_size_per_layer_input ──────────────────


def _create_hf_text_config_with_pli(vocab_size, num_layers, pli_size):
    """Like ``test_model._create_hf_text_config`` but keeps PLI on.

    Per-layer-input (``hidden_size_per_layer_input``) is the Gemma3n-style
    per-layer signal that ``Gemma4Model.compute_host_embeddings`` builds
    on host every decode step from token IDs + main-embedding state. The
    simplified test config in ``test_model.py`` zeroes it out for speed;
    when chasing the chat-completion drift we want it engaged so the
    decode path computes a real PLI tensor every step.

    ``pli_size`` should match the real model's value (E2B = 256). Smaller
    values keep the random-weight reference cheaper without changing
    which code paths run.
    """
    from transformers import AutoConfig

    from ..test_factory import _get_model_path

    config = AutoConfig.from_pretrained(_get_model_path(), trust_remote_code=True)
    tc = config.text_config
    tc.vocab_size = vocab_size
    tc.num_hidden_layers = num_layers
    tc.hidden_size_per_layer_input = pli_size
    tc.vocab_size_per_layer_input = vocab_size  # match main vocab to keep PLI embed table small
    tc._attn_implementation = "eager"
    return tc


def _augment_state_with_pli_weights(state_dict, hf_text_config, num_layers, prefix="model."):
    """Inject random PLI weights into a state_dict.

    ``Gemma4Model.__init__`` looks for three keyed tensors when
    ``hidden_size_per_layer_input > 0``:

    * ``{prefix}embed_tokens_per_layer.weight`` — shape
      ``[vocab_size_per_layer_input, full_n_layers * pli_size]``.
      ``_compute_per_layer_inputs`` infers ``full_n_layers`` from
      ``embed_w.shape[-1] // pli_size``, so the second dim must match
      the full layer count of the test config (= ``num_layers`` here).
    * ``{prefix}per_layer_model_projection.weight`` — shape
      ``[full_n_layers * pli_size, hidden_size]``.
    * ``{prefix}per_layer_projection_norm.weight`` — shape ``[pli_size]``.

    Both TT models in the parity test load the *same* augmented state
    so PLI is computed deterministically and identically on both sides.
    """
    pli_size = int(hf_text_config.hidden_size_per_layer_input)
    vocab_pli = int(hf_text_config.vocab_size_per_layer_input)
    hidden = int(hf_text_config.hidden_size)
    state_dict[f"{prefix}embed_tokens_per_layer.weight"] = (
        torch.randn(vocab_pli, num_layers * pli_size, dtype=torch.bfloat16).float() * 0.02
    )
    state_dict[f"{prefix}per_layer_model_projection.weight"] = (
        torch.randn(num_layers * pli_size, hidden, dtype=torch.bfloat16).float() * 0.02
    )
    state_dict[f"{prefix}per_layer_projection_norm.weight"] = torch.ones(pli_size)
    return state_dict


@parametrize_mesh_with_fabric()
@pytest.mark.parametrize("decode_steps", [4, 16], ids=lambda n: f"steps{n}")
@pytest.mark.parametrize("layer_set", ["small", "all_kv_shared"], ids=["small", "all-kv-shared"])
def test_full_model_parity_decode_with_pli(layer_set, decode_steps, mesh_device, reset_seeds, request):
    """Decode parity with per-layer-input (PLI) engaged.

    Same shape as ``test_full_model_parity_decode_uniform_vs_vllm`` but
    with ``hidden_size_per_layer_input`` set to ``256`` (E2B's value)
    and the three PLI weight tensors injected into the state_dict so
    ``compute_host_embeddings`` produces a real per-token PLI tensor on
    both paths. If PLI is the source of the chat-completion drift, the
    bug should surface here (likely on ``all-kv-shared`` where every
    layer in the kv-share map sees PLI).
    """
    _skip_full_model_parity_if_mesh_too_small(mesh_device)

    from models.tt_transformers.tt.common import PagedAttentionConfig

    from ...tests.test_factory import num_layers_for_full_attention_group
    from ..unit.test_model import _create_hf_model, _hf_model_state_to_tt_state

    if layer_set == "small":
        base = _create_hf_text_config_with_pli(vocab_size=256, num_layers=1, pli_size=64)
        num_layers = num_layers_for_full_attention_group(base)
        kv_shared_override = None
    else:
        real = TestFactory.create_hf_config()
        num_layers = int(real.num_hidden_layers)
        kv_shared_override = int(getattr(real, "num_kv_shared_layers", 0) or 0)
        if kv_shared_override <= 0:
            pytest.skip("Model has no kv-shared layers — all-kv-shared case not applicable")

    pli_size = 64  # small enough to keep the random PLI table cheap
    hf_text_config = _create_hf_text_config_with_pli(vocab_size=256, num_layers=num_layers, pli_size=pli_size)
    if kv_shared_override is not None:
        hf_text_config.num_kv_shared_layers = kv_shared_override
    assert hf_text_config.hidden_size_per_layer_input == pli_size, "PLI must be enabled for this test"

    # Build HF ref → state_dict, then augment with PLI weights at the
    # full layer count so ``Gemma4Model._compute_per_layer_inputs``'s
    # ``full_n_layers = embed_w.shape[-1] // pli_size`` matches.
    hf_model = _create_hf_model(hf_text_config)
    tt_state = _hf_model_state_to_tt_state(hf_model)
    _augment_state_with_pli_weights(tt_state, hf_text_config, num_layers, prefix="model.")

    model_args = Gemma4ModelArgs.from_hf_config(hf_text_config)
    model_args._hf_text_config = hf_text_config

    seq_len = 32
    max_total_len = seq_len + decode_steps + 32
    tp = mesh_device.shape[1] if hasattr(mesh_device, "shape") else 1
    num_devices = mesh_device.get_num_devices() if hasattr(mesh_device, "get_num_devices") else 1
    mesh_config = MeshConfig(mesh_device.shape, decode=ModeConfig(tp=tp))
    ccl_manager = CCLManager(mesh_device, num_links=1) if tp > 1 else None

    uniform_block_size = 64
    uniform_num_blocks = max(16, (max_total_len + uniform_block_size - 1) // uniform_block_size * 2)
    uniform_paged_cfg = PagedAttentionConfig(block_size=uniform_block_size, max_num_blocks=uniform_num_blocks)
    tt_model_uniform, tt_model_vllm = _build_parity_models(
        mesh_device,
        hf_text_config,
        model_args,
        tt_state,
        mesh_config,
        num_layers,
        max_total_len,
        uniform_paged_cfg,
        ccl_manager=ccl_manager,
    )
    # Sanity: PLI weights actually loaded on both sides.
    assert tt_model_uniform.per_layer_input_weights, "uniform model missing PLI weights"
    assert tt_model_vllm.per_layer_input_weights, "vllm model missing PLI weights"

    requested_block_size = 64
    layout = Gemma4VllmLayout.from_hf_config(
        hf_text_config,
        num_blocks=max(16, (max_total_len + requested_block_size - 1) // requested_block_size * 2),
        max_model_len=max_total_len,
        requested_block_size=requested_block_size,
        num_layers=num_layers,
        tp=tp,
        num_devices=num_devices,
    )
    kv_vllm = allocate_vllm_kv_cache(mesh_device, layout, dtype=ttnn.bfloat16)
    pool = Gemma4VllmRequestPool(layout)
    req = pool.allocate_request(num_prefill_tokens=seq_len)
    pts_prefill_torch = pool.per_layer_page_tables(req)

    tokens = torch.randint(0, model_args.vocab_size, (1, seq_len), dtype=torch.long)
    replicate = _replicate_mapper(mesh_device)

    def _embed_tokens(tt_model, toks):
        tt_tokens = ttnn.from_torch(
            toks.to(torch.int32),
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.uint32,
            mesh_mapper=replicate,
        )
        e = tt_model.embed_tokens(tt_tokens)
        e = ttnn.reshape(e, (1, 1, seq_len, model_args.hidden_size))
        return ttnn.to_layout(e, ttnn.TILE_LAYOUT)

    embeds_u = _embed_tokens(tt_model_uniform, tokens)
    embeds_v = _embed_tokens(tt_model_vllm, tokens)
    uniform_pt_prefill = _page_table_to_tt(
        torch.arange(uniform_num_blocks, dtype=torch.int32).reshape(1, uniform_num_blocks), mesh_device
    )

    # PLI prefill needs the host-side embedding to project against —
    # ``_compute_per_layer_inputs`` raises if it's missing.
    import torch.nn.functional as F

    embed_w = tt_model_uniform._embed_weight_cpu
    embeds_torch_cpu = (F.embedding(tokens.long(), embed_w) * tt_model_uniform.embed_scale).float()

    uniform_prefill_logits = tt_model_uniform.ttnn_prefill_forward(
        embeds_u,
        page_table=uniform_pt_prefill,
        kv_cache=None,
        input_ids_torch=tokens,
        embeds_torch=embeds_torch_cpu,
    )
    vllm_prefill_logits = tt_model_vllm.ttnn_prefill_forward(
        embeds_v,
        page_table=None,
        kv_cache=kv_vllm,
        input_ids_torch=tokens,
        embeds_torch=embeds_torch_cpu,
        page_tables_per_layer=pts_prefill_torch,
    )

    uniform_prefill_torch = _from_device(uniform_prefill_logits, mesh_device).float()
    vllm_prefill_torch = _from_device(vllm_prefill_logits, mesh_device).float()

    last_pos = seq_len - 1
    passing, pcc_msg = compare_tensors(
        vllm_prefill_torch[..., last_pos, : model_args.vocab_size],
        uniform_prefill_torch[..., last_pos, : model_args.vocab_size],
        pcc_threshold=get_pcc_threshold(request),
    )
    assert passing, f"PLI prefill parity (layer_set={layer_set}): {pcc_msg}"

    cur_tok = uniform_prefill_torch[0, 0, last_pos, : model_args.vocab_size].argmax().item()
    uniform_prefill_logits.deallocate(True)
    vllm_prefill_logits.deallocate(True)

    failures: list[str] = []
    for step in range(decode_steps):
        position = seq_len + step
        pool.reserve_decode_token(req)
        pts_step_torch = pool.per_layer_page_tables(req)

        u_embeds, u_pos, u_pos_int32, u_pli = _decode_step_inputs(tt_model_uniform, mesh_device, cur_tok, position)
        u_pt = _page_table_to_tt(
            torch.arange(uniform_num_blocks, dtype=torch.int32).reshape(1, uniform_num_blocks)[0:1],
            mesh_device,
        )
        u_logits, _ = tt_model_uniform.ttnn_decode_forward(
            u_embeds, u_pos, u_pos_int32, u_pt, kv_cache=None, pli_combined=u_pli
        )

        tt_model_vllm.update_persistent_per_layer_page_tables(pts_step_torch)
        v_embeds, v_pos, v_pos_int32, v_pli = _decode_step_inputs(tt_model_vllm, mesh_device, cur_tok, position)
        v_pt = _page_table_to_tt(pool.legacy_page_table(req), mesh_device)
        v_logits, _ = tt_model_vllm.ttnn_decode_forward(
            v_embeds,
            v_pos,
            v_pos_int32,
            v_pt,
            kv_cache=kv_vllm,
            page_tables_per_layer=pts_step_torch,
            pli_combined=v_pli,
        )

        u_slice = _from_device(u_logits, mesh_device).float()[..., : model_args.vocab_size]
        v_slice = _from_device(v_logits, mesh_device).float()[..., : model_args.vocab_size]
        passing, pcc_msg = compare_tensors(v_slice, u_slice, pcc_threshold=get_pcc_threshold(request))
        if not passing:
            failures.append(
                f"step {step} (pos {position}): {pcc_msg} — "
                f"uniform_top={u_slice.flatten().argmax().item()} "
                f"vllm_top={v_slice.flatten().argmax().item()}"
            )
        cur_tok = u_slice.flatten().argmax().item()
        u_logits.deallocate(True)
        v_logits.deallocate(True)

    assert not failures, "PLI decode parity diverged:\n" + "\n".join(failures)


# ── Trace-replay parity: the path the chat-completion server actually uses ──


@parametrize_mesh_with_fabric()
@pytest.mark.parametrize("decode_steps", [4], ids=lambda n: f"steps{n}")
@pytest.mark.parametrize("layer_set", ["small", "all_kv_shared"], ids=["small", "all-kv-shared"])
@pytest.mark.parametrize("pli", [False, True], ids=["no-pli", "pli"])
def test_full_model_parity_decode_trace(layer_set, decode_steps, pli, mesh_device, reset_seeds, request):
    """Decode parity with the *traced* decode path on both sides.

    This is the path the vLLM server actually runs at inference: a
    decode trace is captured once (with warmup-style inputs), then
    every subsequent decode step is a trace-replay refreshed by
    ``copy_host_to_device`` against ``Generator.trace_inputs_decode``.
    The non-trace parity tests above bypass that machinery entirely;
    this one wraps both Gemma4Model instances in
    :class:`Gemma4Generator` and calls
    ``Generator.decode_forward(enable_trace=True)`` so we go through
    ``_capture_decode_trace_text`` + ``_decode_forward_trace_text``
    end-to-end.

    The vLLM-side prelude (``update_persistent_per_layer_page_tables``
    + ``_active_page_tables_per_layer`` stash) is inlined here to
    mirror what :class:`Gemma4ForCausalLM.decode_forward` does in the
    bridge; we don't need the bridge itself, just its setup steps.

    If the chat-completion drift is a trace-binding bug analogous to
    the :class:`Gemma4Model._decode_pli_combined` issue I fixed
    earlier, it should surface here as a PCC drop at decode step 2+.
    """
    _skip_full_model_parity_if_mesh_too_small(mesh_device)

    import torch.nn.functional as F

    from models.demos.gemma4.tt.generator import Gemma4Generator
    from models.tt_transformers.tt.common import PagedAttentionConfig

    from ...tests.test_factory import num_layers_for_full_attention_group
    from ..unit.test_model import _create_hf_model, _hf_model_state_to_tt_state

    # ── Build config ─────────────────────────────────────────────────
    if pli:
        if layer_set == "small":
            base = _create_hf_text_config_with_pli(vocab_size=256, num_layers=1, pli_size=64)
            num_layers = num_layers_for_full_attention_group(base)
            kv_shared_override = None
        else:
            real = TestFactory.create_hf_config()
            num_layers = int(real.num_hidden_layers)
            kv_shared_override = int(getattr(real, "num_kv_shared_layers", 0) or 0)
            if kv_shared_override <= 0:
                pytest.skip("Model has no kv-shared layers — all-kv-shared case not applicable")
        hf_text_config = _create_hf_text_config_with_pli(vocab_size=256, num_layers=num_layers, pli_size=64)
        if kv_shared_override is not None:
            hf_text_config.num_kv_shared_layers = kv_shared_override
    else:
        from ..unit.test_model import _create_hf_text_config

        if layer_set == "small":
            base = _create_hf_text_config(vocab_size=256, num_layers=1)
            num_layers = num_layers_for_full_attention_group(base)
            kv_shared_override = None
        else:
            real = TestFactory.create_hf_config()
            num_layers = int(real.num_hidden_layers)
            kv_shared_override = int(getattr(real, "num_kv_shared_layers", 0) or 0)
            if kv_shared_override <= 0:
                pytest.skip("Model has no kv-shared layers — all-kv-shared case not applicable")
        hf_text_config = _create_hf_text_config(vocab_size=256, num_layers=num_layers)
        if kv_shared_override is not None:
            hf_text_config.num_kv_shared_layers = kv_shared_override

    hf_model = _create_hf_model(hf_text_config)
    tt_state = _hf_model_state_to_tt_state(hf_model)
    if pli:
        _augment_state_with_pli_weights(tt_state, hf_text_config, num_layers, prefix="model.")

    model_args = Gemma4ModelArgs.from_hf_config(hf_text_config)
    model_args._hf_text_config = hf_text_config
    # ``Generator`` indexes ``self.model_args[i].mesh_device`` at trace
    # capture; ``Gemma4ModelArgs.from_hf_config`` doesn't populate it
    # (that's normally done by the ``_patch_model_args`` helper in
    # ``Gemma4Generator.from_pretrained``).
    model_args.mesh_device = mesh_device

    seq_len = 32
    max_total_len = seq_len + decode_steps + 32
    tp = mesh_device.shape[1] if hasattr(mesh_device, "shape") else 1
    num_devices = mesh_device.get_num_devices() if hasattr(mesh_device, "get_num_devices") else 1
    mesh_config = MeshConfig(mesh_device.shape, decode=ModeConfig(tp=tp))
    ccl_manager = CCLManager(mesh_device, num_links=1) if tp > 1 else None
    uniform_block_size = 64
    uniform_num_blocks = max(16, (max_total_len + uniform_block_size - 1) // uniform_block_size * 2)
    uniform_paged_cfg = PagedAttentionConfig(block_size=uniform_block_size, max_num_blocks=uniform_num_blocks)
    tt_model_uniform, tt_model_vllm = _build_parity_models(
        mesh_device,
        hf_text_config,
        model_args,
        tt_state,
        mesh_config,
        num_layers,
        max_total_len,
        uniform_paged_cfg,
        ccl_manager=ccl_manager,
    )

    # ── Wrap each model in a minimal Generator ──────────────────────
    # ``Generator.decode_forward`` is what eventually routes to
    # ``_decode_forward_trace_text``; one Generator instance per
    # model so each gets its own ``trace_inputs_decode`` cache.
    gen_uniform = Gemma4Generator(model=[tt_model_uniform], model_args=[model_args], mesh_device=mesh_device)
    gen_vllm = Gemma4Generator(model=[tt_model_vllm], model_args=[model_args], mesh_device=mesh_device)

    requested_block_size = 64
    layout = Gemma4VllmLayout.from_hf_config(
        hf_text_config,
        num_blocks=max(16, (max_total_len + requested_block_size - 1) // requested_block_size * 2),
        max_model_len=max_total_len,
        requested_block_size=requested_block_size,
        num_layers=num_layers,
        tp=tp,
        num_devices=num_devices,
    )
    kv_vllm = allocate_vllm_kv_cache(mesh_device, layout, dtype=ttnn.bfloat16)
    pool = Gemma4VllmRequestPool(layout)
    req = pool.allocate_request(num_prefill_tokens=seq_len)
    pts_prefill_torch = pool.per_layer_page_tables(req)

    tokens = torch.randint(0, model_args.vocab_size, (1, seq_len), dtype=torch.long)
    replicate = _replicate_mapper(mesh_device)

    def _embed_tokens(tt_model, toks):
        tt_tokens = ttnn.from_torch(
            toks.to(torch.int32),
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.uint32,
            mesh_mapper=replicate,
        )
        e = tt_model.embed_tokens(tt_tokens)
        e = ttnn.reshape(e, (1, 1, seq_len, model_args.hidden_size))
        return ttnn.to_layout(e, ttnn.TILE_LAYOUT)

    # PLI requires host embeds; harmless to compute either way.
    embed_w = tt_model_uniform._embed_weight_cpu
    embeds_torch_cpu = (F.embedding(tokens.long(), embed_w) * tt_model_uniform.embed_scale).float()

    embeds_u = _embed_tokens(tt_model_uniform, tokens)
    embeds_v = _embed_tokens(tt_model_vllm, tokens)
    uniform_pt_prefill = _page_table_to_tt(
        torch.arange(uniform_num_blocks, dtype=torch.int32).reshape(1, uniform_num_blocks), mesh_device
    )

    # Prefill once on both sides (same path as the existing parity
    # tests — trace coverage is exclusively for decode).
    uniform_prefill_logits = tt_model_uniform.ttnn_prefill_forward(
        embeds_u,
        page_table=uniform_pt_prefill,
        kv_cache=None,
        input_ids_torch=tokens,
        embeds_torch=embeds_torch_cpu,
    )
    vllm_prefill_logits = tt_model_vllm.ttnn_prefill_forward(
        embeds_v,
        page_table=None,
        kv_cache=kv_vllm,
        input_ids_torch=tokens,
        embeds_torch=embeds_torch_cpu,
        page_tables_per_layer=pts_prefill_torch,
    )

    uniform_prefill_torch = _from_device(uniform_prefill_logits, mesh_device).float()
    vllm_prefill_torch = _from_device(vllm_prefill_logits, mesh_device).float()

    last_pos = seq_len - 1
    cur_tok_u = uniform_prefill_torch[0, 0, last_pos, : model_args.vocab_size].argmax().item()
    cur_tok_v = vllm_prefill_torch[0, 0, last_pos, : model_args.vocab_size].argmax().item()
    assert cur_tok_u == cur_tok_v, (
        f"prefill first-decoded-token diverged (no trace involved yet): " f"uniform={cur_tok_u} vllm={cur_tok_v}"
    )
    cur_tok = cur_tok_u
    uniform_prefill_logits.deallocate(True)
    vllm_prefill_logits.deallocate(True)

    # ── Decode through trace path ───────────────────────────────────
    failures: list[str] = []
    for step in range(decode_steps):
        position = seq_len + step
        tokens_step = torch.tensor([cur_tok], dtype=torch.int32).reshape(1)
        pos_step = torch.tensor([position], dtype=torch.int32).reshape(1)

        # Uniform path: straight Generator.decode_forward, no per-layer routing.
        u_out = gen_uniform.decode_forward(
            tokens=tokens_step,
            start_pos=pos_step,
            page_table=torch.arange(uniform_num_blocks, dtype=torch.int32).reshape(1, uniform_num_blocks),
            kv_cache=None,  # uses tt_model_uniform.tt_kv_cache
            enable_trace=True,
            read_from_device=False,
            sampling_params=None,
            reset_batch=(step == 0),
        )

        # vLLM path: refresh persistent per-layer page tables + stash
        # the per-submesh routing list, then go through the same
        # Generator.decode_forward. Exactly mirrors what
        # ``Gemma4ForCausalLM.decode_forward`` does in the bridge.
        pool.reserve_decode_token(req)
        pts_step_torch = pool.per_layer_page_tables(req)
        tt_model_vllm.update_persistent_per_layer_page_tables(pts_step_torch)
        tt_model_vllm._active_page_tables_per_layer = pts_step_torch
        try:
            v_out = gen_vllm.decode_forward(
                tokens=tokens_step,
                start_pos=pos_step,
                page_table=pool.legacy_page_table(req),
                kv_cache=[kv_vllm],
                enable_trace=True,
                read_from_device=False,
                sampling_params=None,
                reset_batch=(step == 0),
            )
        finally:
            # Bridge would do this via context manager — mirror that.
            if hasattr(tt_model_vllm, "_active_page_tables_per_layer"):
                del tt_model_vllm._active_page_tables_per_layer

        # ``Gemma4Model.ttnn_decode_forward`` returns ``(logits, None)``;
        # ``_decode_forward_trace_text`` propagates that, so each
        # DP entry is a 2-tuple. DP=1 → ``[0][0]`` is the logits.
        u_entry = u_out[0]
        v_entry = v_out[0]
        u_logits = u_entry[0] if isinstance(u_entry, tuple) else u_entry
        v_logits = v_entry[0] if isinstance(v_entry, tuple) else v_entry
        u_logits_torch = _from_device(u_logits, mesh_device).float()
        v_logits_torch = _from_device(v_logits, mesh_device).float()
        u_slice = u_logits_torch[..., : model_args.vocab_size]
        v_slice = v_logits_torch[..., : model_args.vocab_size]

        passing, pcc_msg = compare_tensors(v_slice, u_slice, pcc_threshold=get_pcc_threshold(request))
        if not passing:
            failures.append(
                f"step {step} (pos {position}): {pcc_msg} — "
                f"uniform_top={u_slice.flatten().argmax().item()} "
                f"vllm_top={v_slice.flatten().argmax().item()}"
            )
        cur_tok = u_slice.flatten().argmax().item()

    assert not failures, "trace-replay decode parity diverged:\n" + "\n".join(failures)


# ── Warmup-then-inference trace parity: the exact vLLM production flow ──


@parametrize_mesh_with_fabric()
@pytest.mark.parametrize("decode_steps", [4], ids=lambda n: f"steps{n}")
@pytest.mark.parametrize("layer_set", ["small", "all_kv_shared"], ids=["small", "all-kv-shared"])
@pytest.mark.parametrize("pli", [False, True], ids=["no-pli", "pli"])
def test_full_model_parity_warmup_then_inference(layer_set, decode_steps, pli, mesh_device, reset_seeds, request):
    """End-to-end mirror of vLLM's warmup→inference flow.

    The previous parity tests captured the decode trace at the *first*
    decode step of the actual inference inputs — real tokens, real
    positions, real block IDs. The vLLM plugin captures the decode
    trace during ``warmup_model_decode`` with **all-zero** inputs
    (``tokens=zeros``, ``start_pos=zeros``, ``page_table=zeros``),
    and only then runs real inference against that warmup-bound trace.

    If anything inside ``Gemma4Model.ttnn_decode_forward`` reads a value
    *eagerly* at trace-capture time instead of reading it from a device
    tensor that gets refreshed at replay, that value gets *frozen* at
    its warmup-zero state. The drift symptom in chat completions would
    be exactly what you'd expect from one frozen sub-graph (PLI, RoPE
    lookup, page_table interpretation, sampling — any of them).

    Run order matches the plugin's ``TTModelRunner.warmup``:

        warmup_model_decode(enable_trace=False)   # compile, no trace
        warmup_model_decode(enable_trace=True)    # capture trace
        prefill(real_prompt)                      # writes K/V at real
                                                  # block IDs
        decode_step × N                           # trace replays

    PCC against the uniform baseline at every replay step. If we are
    going to reproduce the chat garble in a unit test, this is where.
    """
    _skip_full_model_parity_if_mesh_too_small(mesh_device)

    import torch.nn.functional as F

    from models.demos.gemma4.tt.generator import Gemma4Generator
    from models.tt_transformers.tt.common import PagedAttentionConfig

    from ...tests.test_factory import num_layers_for_full_attention_group
    from ..unit.test_model import _create_hf_model, _hf_model_state_to_tt_state

    if pli:
        if layer_set == "small":
            base = _create_hf_text_config_with_pli(vocab_size=256, num_layers=1, pli_size=64)
            num_layers = num_layers_for_full_attention_group(base)
            kv_shared_override = None
        else:
            real = TestFactory.create_hf_config()
            num_layers = int(real.num_hidden_layers)
            kv_shared_override = int(getattr(real, "num_kv_shared_layers", 0) or 0)
        hf_text_config = _create_hf_text_config_with_pli(vocab_size=256, num_layers=num_layers, pli_size=64)
        if kv_shared_override is not None:
            hf_text_config.num_kv_shared_layers = kv_shared_override
    else:
        from ..unit.test_model import _create_hf_text_config

        if layer_set == "small":
            base = _create_hf_text_config(vocab_size=256, num_layers=1)
            num_layers = num_layers_for_full_attention_group(base)
            kv_shared_override = None
        else:
            real = TestFactory.create_hf_config()
            num_layers = int(real.num_hidden_layers)
            kv_shared_override = int(getattr(real, "num_kv_shared_layers", 0) or 0)
        hf_text_config = _create_hf_text_config(vocab_size=256, num_layers=num_layers)
        if kv_shared_override is not None:
            hf_text_config.num_kv_shared_layers = kv_shared_override

    hf_model = _create_hf_model(hf_text_config)
    tt_state = _hf_model_state_to_tt_state(hf_model)
    if pli:
        _augment_state_with_pli_weights(tt_state, hf_text_config, num_layers, prefix="model.")

    model_args = Gemma4ModelArgs.from_hf_config(hf_text_config)
    model_args._hf_text_config = hf_text_config
    model_args.mesh_device = mesh_device

    seq_len = 32
    max_total_len = seq_len + decode_steps + 32
    tp = mesh_device.shape[1] if hasattr(mesh_device, "shape") else 1
    num_devices = mesh_device.get_num_devices() if hasattr(mesh_device, "get_num_devices") else 1
    mesh_config = MeshConfig(mesh_device.shape, decode=ModeConfig(tp=tp))
    ccl_manager = CCLManager(mesh_device, num_links=1) if tp > 1 else None
    uniform_block_size = 64
    uniform_num_blocks = max(16, (max_total_len + uniform_block_size - 1) // uniform_block_size * 2)
    uniform_paged_cfg = PagedAttentionConfig(block_size=uniform_block_size, max_num_blocks=uniform_num_blocks)
    tt_model_uniform, tt_model_vllm = _build_parity_models(
        mesh_device,
        hf_text_config,
        model_args,
        tt_state,
        mesh_config,
        num_layers,
        max_total_len,
        uniform_paged_cfg,
        ccl_manager=ccl_manager,
    )

    gen_uniform = Gemma4Generator(model=[tt_model_uniform], model_args=[model_args], mesh_device=mesh_device)
    gen_vllm = Gemma4Generator(model=[tt_model_vllm], model_args=[model_args], mesh_device=mesh_device)

    requested_block_size = 64
    layout = Gemma4VllmLayout.from_hf_config(
        hf_text_config,
        num_blocks=max(16, (max_total_len + requested_block_size - 1) // requested_block_size * 2),
        max_model_len=max_total_len,
        requested_block_size=requested_block_size,
        num_layers=num_layers,
        tp=tp,
        num_devices=num_devices,
    )
    kv_vllm = allocate_vllm_kv_cache(mesh_device, layout, dtype=ttnn.bfloat16)

    # ── Phase 1: warmup decode trace capture (zero inputs) ───────────
    # This is the exact thing the plugin does in ``TTModelRunner.warmup``:
    # warm up at all-zero tokens/positions/page_tables, capture the
    # decode trace, then stash for reuse at inference.
    #
    # Uniform path: just call warmup_model_decode; the model's own
    # tt_kv_cache (paged uniform) is used.
    #
    # vLLM path: same call but we have to (a) pass the harness-allocated
    # kv_cache (the bridge's ``submit_decode`` does this, the
    # ``warmup_model_decode`` mixin doesn't know about it), and (b)
    # pre-populate the model's ``_active_page_tables_per_layer`` /
    # ``_persistent_per_layer_page_tables`` with warmup-zero per-layer
    # tables so the trace binds against the per-layer routing path
    # rather than the legacy single page_table fallback.
    warmup_num_blocks = layout.max_num_blocks_per_req
    warmup_page_table_per_layer = [torch.zeros(1, warmup_num_blocks, dtype=torch.int32) for _ in range(num_layers)]

    # Uniform warmup — passes ``page_table`` as a single tensor,
    # broadcast trivially by the model.
    gen_uniform.warmup_model_decode(
        kv_cache=None,  # model uses self.tt_kv_cache (paged uniform)
        enable_trace=True,
        max_batch_size=1,
        num_blocks=uniform_num_blocks,
        can_sample_on_device=False,
        greedy_only=True,
    )

    # vLLM warmup — replicate the bridge's pre-decode setup so the
    # captured trace lands on the per-layer routing path. Without this
    # the warmup trace would bind against the legacy page_table only,
    # and at inference our update_persistent_per_layer_page_tables
    # writes wouldn't reach anything the trace actually reads.
    tt_model_vllm.update_persistent_per_layer_page_tables(warmup_page_table_per_layer)
    tt_model_vllm._active_page_tables_per_layer = warmup_page_table_per_layer
    try:
        gen_vllm.warmup_model_decode(
            kv_cache=[kv_vllm],  # list-of-DP: harness kv-cache
            enable_trace=True,
            max_batch_size=1,
            num_blocks=warmup_num_blocks,
            can_sample_on_device=False,
            greedy_only=True,
        )
    finally:
        if hasattr(tt_model_vllm, "_active_page_tables_per_layer"):
            del tt_model_vllm._active_page_tables_per_layer

    # ── Phase 2: real prefill (writes K/V at real block IDs) ─────────
    pool = Gemma4VllmRequestPool(layout)
    req = pool.allocate_request(num_prefill_tokens=seq_len)
    pts_prefill_torch = pool.per_layer_page_tables(req)

    tokens = torch.randint(0, model_args.vocab_size, (1, seq_len), dtype=torch.long)
    replicate = _replicate_mapper(mesh_device)

    def _embed_tokens(tt_model, toks):
        tt_tokens = ttnn.from_torch(
            toks.to(torch.int32),
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.uint32,
            mesh_mapper=replicate,
        )
        e = tt_model.embed_tokens(tt_tokens)
        e = ttnn.reshape(e, (1, 1, seq_len, model_args.hidden_size))
        return ttnn.to_layout(e, ttnn.TILE_LAYOUT)

    embed_w = tt_model_uniform._embed_weight_cpu
    embeds_torch_cpu = (F.embedding(tokens.long(), embed_w) * tt_model_uniform.embed_scale).float()
    embeds_u = _embed_tokens(tt_model_uniform, tokens)
    embeds_v = _embed_tokens(tt_model_vllm, tokens)
    uniform_pt_prefill = _page_table_to_tt(
        torch.arange(uniform_num_blocks, dtype=torch.int32).reshape(1, uniform_num_blocks), mesh_device
    )
    uniform_prefill_logits = tt_model_uniform.ttnn_prefill_forward(
        embeds_u,
        page_table=uniform_pt_prefill,
        kv_cache=None,
        input_ids_torch=tokens,
        embeds_torch=embeds_torch_cpu,
    )
    vllm_prefill_logits = tt_model_vllm.ttnn_prefill_forward(
        embeds_v,
        page_table=None,
        kv_cache=kv_vllm,
        input_ids_torch=tokens,
        embeds_torch=embeds_torch_cpu,
        page_tables_per_layer=pts_prefill_torch,
    )
    uniform_prefill_torch = _from_device(uniform_prefill_logits, mesh_device).float()
    vllm_prefill_torch = _from_device(vllm_prefill_logits, mesh_device).float()
    last_pos = seq_len - 1
    cur_tok_u = uniform_prefill_torch[0, 0, last_pos, : model_args.vocab_size].argmax().item()
    cur_tok_v = vllm_prefill_torch[0, 0, last_pos, : model_args.vocab_size].argmax().item()
    assert cur_tok_u == cur_tok_v, f"prefill diverges: u={cur_tok_u} v={cur_tok_v}"
    cur_tok = cur_tok_u
    uniform_prefill_logits.deallocate(True)
    vllm_prefill_logits.deallocate(True)

    # ── Phase 3: decode steps via WARMUP-CAPTURED trace ──────────────
    failures: list[str] = []
    for step in range(decode_steps):
        position = seq_len + step
        tokens_step = torch.tensor([cur_tok], dtype=torch.int32).reshape(1)
        pos_step = torch.tensor([position], dtype=torch.int32).reshape(1)

        u_out = gen_uniform.decode_forward(
            tokens=tokens_step,
            start_pos=pos_step,
            page_table=torch.arange(uniform_num_blocks, dtype=torch.int32).reshape(1, uniform_num_blocks),
            kv_cache=None,
            enable_trace=True,
            read_from_device=False,
            sampling_params=None,
            reset_batch=(step == 0),
        )

        pool.reserve_decode_token(req)
        pts_step_torch = pool.per_layer_page_tables(req)
        tt_model_vllm.update_persistent_per_layer_page_tables(pts_step_torch)
        tt_model_vllm._active_page_tables_per_layer = pts_step_torch
        try:
            v_out = gen_vllm.decode_forward(
                tokens=tokens_step,
                start_pos=pos_step,
                page_table=pool.legacy_page_table(req),
                kv_cache=[kv_vllm],
                enable_trace=True,
                read_from_device=False,
                sampling_params=None,
                reset_batch=(step == 0),
            )
        finally:
            if hasattr(tt_model_vllm, "_active_page_tables_per_layer"):
                del tt_model_vllm._active_page_tables_per_layer

        u_entry, v_entry = u_out[0], v_out[0]
        u_logits = u_entry[0] if isinstance(u_entry, tuple) else u_entry
        v_logits = v_entry[0] if isinstance(v_entry, tuple) else v_entry
        u_slice = _from_device(u_logits, mesh_device).float()[..., : model_args.vocab_size]
        v_slice = _from_device(v_logits, mesh_device).float()[..., : model_args.vocab_size]
        passing, pcc_msg = compare_tensors(v_slice, u_slice, pcc_threshold=get_pcc_threshold(request))
        if not passing:
            failures.append(
                f"step {step} (pos {position}): {pcc_msg} — "
                f"u_top={u_slice.flatten().argmax().item()} "
                f"v_top={v_slice.flatten().argmax().item()}"
            )
        cur_tok = u_slice.flatten().argmax().item()

    assert not failures, "warmup-then-inference parity diverged:\n" + "\n".join(failures)


# ── KV-share page-table alias: regression test for chat-completion garble ──


def test_bridge_aliases_kv_share_page_tables_to_source():
    """The bridge must alias kv-shared layers' *page_tables* to source.

    Regression test for the chat-completion-garbage bug fixed by
    :meth:`Gemma4ForCausalLM._apply_kv_share_to_per_layer_page_tables`.
    Background: vLLM's hybrid kv-cache manager does *not* put all
    sliding layers in one group — for Gemma4-E2B's
    4-sliding-then-1-full pattern it produces 5 sub-groups of 7
    layers each, and each physical KVCacheTensor is shared by one
    layer from each sub-group. A kv-shared layer (sink) and its
    source therefore have the same attention type but live in
    *different* sub-groups, so the plugin's
    ``_block_tables_per_layer`` hands them different per-layer
    page_tables. Aliasing only the kv-cache *buffer* (which we do in
    :meth:`Gemma4ForCausalLM.allocate_kv_cache_per_layer`) is
    necessary but not sufficient — the sink layer would index that
    shared buffer with its own sub-group's block IDs and read a
    completely different layer's K/V. The fix replaces the sink's
    per-layer page_table with the source's.

    This test pins the function's behaviour with a small mock so
    a future refactor that forgets to call the helper, or breaks
    its semantics, fails here loud and fast — without needing to
    spin up vLLM and inspect text outputs.

    Pure-Python (no device): the routing logic is just a list
    rewrite keyed off ``kv_shared_layer_map``.
    """
    from types import SimpleNamespace

    import torch

    from models.demos.gemma4.tt.generator_vllm import Gemma4ForCausalLM

    # Mock model holds the kv_shared_layer_map the way the real
    # Gemma4Model does — last 2 layers (3, 4) share KV from layer 0
    # and 1 respectively. The bridge looks up the map via
    # ``self.model[0].kv_shared_layer_map``.
    kv_shared_map = {3: 0, 4: 1}
    mock_model = SimpleNamespace(kv_shared_layer_map=kv_shared_map)

    # Build a bridge instance without going through the full
    # initialize_vllm_model machinery (it requires a real model
    # path). Direct attribute assignment is enough since the helper
    # only reads ``self.model[0].kv_shared_layer_map``.
    bridge = Gemma4ForCausalLM.__new__(Gemma4ForCausalLM)
    bridge.model = [mock_model]

    # Per-layer page tables: each layer gets distinct block IDs so
    # an unaliased call leaves them distinct, and an aliased call
    # makes shared layers' tables identical to their source's.
    page_tables = [
        torch.tensor([[10, 11, 0]], dtype=torch.int32),  # layer 0
        torch.tensor([[20, 21, 0]], dtype=torch.int32),  # layer 1
        torch.tensor([[30, 31, 0]], dtype=torch.int32),  # layer 2
        torch.tensor([[40, 41, 0]], dtype=torch.int32),  # layer 3 (sink → 0)
        torch.tensor([[50, 51, 0]], dtype=torch.int32),  # layer 4 (sink → 1)
    ]

    out = bridge._apply_kv_share_to_per_layer_page_tables(page_tables)

    # Non-shared layers untouched (Python identity, not equality —
    # the routing must be a list rewrite, not a tensor copy).
    assert out[0] is page_tables[0], "layer 0 (source) page_table should pass through"
    assert out[1] is page_tables[1], "layer 1 (source) page_table should pass through"
    assert out[2] is page_tables[2], "layer 2 (non-shared) page_table should pass through"
    # Shared layers re-pointed to their source layer's page_table.
    # Same Python object — that's the whole point: the sink and source
    # must dereference the same block IDs in the shared buffer.
    assert out[3] is page_tables[0], (
        "layer 3 (kv-shared from 0) page_table must alias to layer 0's, "
        "not be a copy — content match alone wouldn't catch a list-rewrite bug "
        "that produces equal but distinct objects"
    )
    assert out[4] is page_tables[1], "layer 4 (kv-shared from 1) page_table must alias to layer 1's"


def test_bridge_no_alias_when_kv_share_map_empty():
    """When the model has no kv-shared layers (e.g. truncated test
    config), the routing helper must be a pass-through.

    Guards against an accidental rewrite that breaks every other
    Gemma4 variant or a future model that doesn't use Gemma3n-style
    kv-share but still goes through this bridge.
    """
    from types import SimpleNamespace

    import torch

    from models.demos.gemma4.tt.generator_vllm import Gemma4ForCausalLM

    mock_model = SimpleNamespace(kv_shared_layer_map={})
    bridge = Gemma4ForCausalLM.__new__(Gemma4ForCausalLM)
    bridge.model = [mock_model]

    page_tables = [
        torch.tensor([[10, 11, 0]], dtype=torch.int32),
        torch.tensor([[20, 21, 0]], dtype=torch.int32),
    ]
    out = bridge._apply_kv_share_to_per_layer_page_tables(page_tables)
    assert out[0] is page_tables[0]
    assert out[1] is page_tables[1]


def test_bridge_alias_handles_distinct_source_page_tables():
    """Sink layers can map to *different* source layers (e.g. one
    sliding source, one full source). The helper must handle each
    independently — a sink-to-wrong-source swap would mean the shared
    layer reads a different attention type's KV cache layout, which
    is exactly the cross-talk class of bug the fix exists to prevent.
    """
    from types import SimpleNamespace

    import torch

    from models.demos.gemma4.tt.generator_vllm import Gemma4ForCausalLM

    # Two distinct sources: layer 13 (sliding source) and layer 14
    # (full source). Sinks split between them as the real Gemma4-E2B
    # map does — sliding sinks → 13, full sinks → 14.
    kv_shared_map = {15: 13, 16: 13, 19: 14, 24: 14}
    mock_model = SimpleNamespace(kv_shared_layer_map=kv_shared_map)
    bridge = Gemma4ForCausalLM.__new__(Gemma4ForCausalLM)
    bridge.model = [mock_model]

    # Distinct page_tables so an alias-to-wrong-source bug would
    # show up as a mismatched Python identity in the output.
    page_tables = [torch.tensor([[100 + i, 0, 0]], dtype=torch.int32) for i in range(25)]

    out = bridge._apply_kv_share_to_per_layer_page_tables(page_tables)
    assert out[15] is page_tables[13], "sliding sink 15 must point at sliding source 13"
    assert out[16] is page_tables[13], "sliding sink 16 must point at sliding source 13"
    assert out[19] is page_tables[14], "full sink 19 must point at full source 14"
    assert out[24] is page_tables[14], "full sink 24 must point at full source 14"
    # Source layers themselves stay self-pointing.
    assert out[13] is page_tables[13]
    assert out[14] is page_tables[14]
