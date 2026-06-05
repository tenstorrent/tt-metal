# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Paged KV-cache primitive round-trip for Gemma4 (issue #44946).

Exercises the three paged-cache ops the decode/prefill paths rely on —
``paged_fill_cache`` (prefill), ``paged_update_cache`` (decode token append) and
``paged_scaled_dot_product_attention_decode`` (read) — *directly* (not through
the ``Gemma4Attention`` module), so a block-size-dependent indexing bug surfaces
at the op boundary rather than being masked by the rest of attention.

The headline test ``test_paged_kv_cache`` is a fill -> update -> SDPA-read
round-trip parametrized over ``batch in {1, 16, 32}`` and
``block_size in {32, 64, 128, 256}`` on the supported TP grids. It is also the
isolated profiling vehicle for the page_block_size audit: filtering to a single
``paged_update_cache`` shape (e.g. ``-k "1x4 and update and batch1"``) gives Tracy
a clean, low-op-count capture where ``PagedUpdateCacheDeviceOperation`` is
reliably attributed a device kernel duration + DRAM BW (it is not, inside the
full multi-device decode loop).

Cache geometry mirrors one device's production view: the cache is replicated
across the mesh with the *local* KV-head count (``num_key_value_heads // tp``),
which is exactly the ``[max_num_blocks, num_kv_local, block_size, head_dim]``
buffer ``init_kv_cache`` allocates and the ``[1, B, num_kv_local, head_dim]``
per-token write target of ``paged_update_cache``. Cross-device KV-head sharding
and the full SDPA-vs-HF numerics across block sizes are covered by
``test_attention.py::test_attention_decode_paged_batched``.

    pytest models/demos/gemma4/tests/unit/test_paged_kv_cache.py -k "1x4"
    pytest models/demos/gemma4/tests/unit/test_paged_kv_cache.py -k "block64 and batch1"
"""

import pytest
import torch

import ttnn

# Tracy signpost: brackets the measured iterations of the microbench so a
# profiling capture isolates the steady-state paged_update_cache region from the
# first-iteration op compile/program-cache warmup. No-op outside a tracy run.
try:
    from tracy import signpost as _tracy_signpost
except ModuleNotFoundError:

    def _tracy_signpost(*_args, **_kwargs):
        pass


from models.demos.gemma4.tt.attention import Gemma4AttentionConfig

from ...tests.test_factory import TestFactory, compare_tensors, get_pcc_threshold, parametrize_mesh_with_fabric

# Supported decode block sizes for the audit sweep.
BLOCK_SIZES = [32, 64, 128, 256]
# Meshes matching the issue focus (31B 1x4 quietbox) plus single-card portability.
_MESH_SHAPES = [(1, 1), (1, 4)]


def _tp_of(mesh_device):
    return mesh_device.shape[1] if hasattr(mesh_device, "shape") else 1


def _is_mesh(mesh_device):
    return hasattr(mesh_device, "shape") and mesh_device.get_num_devices() > 1


def _local_kv_heads(config, tp):
    """Per-device KV-head count, mirroring ``init_kv_cache`` / ``split_qkv_heads_decode``."""
    return 1 if config.num_key_value_heads < tp else config.num_key_value_heads // tp


def _ceil_to_tile(n, tile=32):
    return ((n + tile - 1) // tile) * tile


def _replicate(t, mesh_device, layout, dtype):
    """Send a torch tensor to the mesh, replicated to every device."""
    return ttnn.from_torch(
        t,
        device=mesh_device,
        layout=layout,
        dtype=dtype,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device) if _is_mesh(mesh_device) else None,
    )


def _from_device(tensor, mesh_device):
    """Read a (replicated) tensor back from device 0."""
    if _is_mesh(mesh_device):
        return ttnn.to_torch(ttnn.get_device_tensors(tensor)[0])
    return ttnn.to_torch(tensor)


def _height_sharded_decode_input(x_padded, mesh_device):
    """Convert ``(1, B, padded_heads, head_dim)`` to the height-sharded L1 layout
    ``paged_update_cache`` expects (one user per core), matching what
    ``decode.py`` feeds the op after ``split_qkv_heads_decode``.
    """
    num_users = x_padded.shape[1]
    xt = _replicate(x_padded, mesh_device, ttnn.TILE_LAYOUT, ttnn.bfloat16)
    grid = mesh_device.compute_with_storage_grid_size()
    shard_grid = ttnn.num_cores_to_corerangeset(num_users, grid, True)
    shard_spec = ttnn.ShardSpec(
        shard_grid,
        [xt.volume() // xt.padded_shape[-1] // num_users, xt.padded_shape[-1]],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)
    return ttnn.to_memory_config(xt, mem)


def _alloc_paged_cache(max_num_blocks, num_kv_heads, block_size, head_dim, mesh_device):
    """Zero-initialised paged cache ``[max_num_blocks, num_kv_heads, block_size, head_dim]``."""
    cache_torch = torch.zeros(max_num_blocks, num_kv_heads, block_size, head_dim)
    return _replicate(cache_torch, mesh_device, ttnn.TILE_LAYOUT, ttnn.bfloat16)


def _contiguous_page_table(batch, blocks_per_user):
    """Page table where user ``b`` owns the contiguous physical block range
    ``[b * blocks_per_user, (b + 1) * blocks_per_user)``.
    """
    page_table = torch.zeros(batch, blocks_per_user, dtype=torch.int32)
    for b in range(batch):
        page_table[b] = torch.arange(b * blocks_per_user, (b + 1) * blocks_per_user, dtype=torch.int32)
    return page_table


def _torch_sdpa_reference(q, k, v, cur_pos, scale):
    """Causal SDPA reference in fp32. ``q`` is ``(1, B, num_q_heads, head_dim)``,
    ``k``/``v`` are ``(B, num_kv_heads, seq, head_dim)``, ``cur_pos[b]`` is the
    inclusive most-recent token index for user ``b``. Mirrors the reference in
    ``tests/ttnn/.../test_paged_sdpa_decode_flexible_geometry.py``.
    """
    B = q.shape[1]
    num_q_heads = q.shape[2]
    num_kv_heads = k.shape[1]
    head_dim_v = v.shape[3]
    repeat = num_q_heads // num_kv_heads
    out = torch.zeros(1, B, num_q_heads, head_dim_v, dtype=torch.float32)
    for b in range(B):
        pos = int(cur_pos[b])
        k_b = k[b, :, : pos + 1, :].float()
        v_b = v[b, :, : pos + 1, :].float()
        q_b = q[0, b, :, :].float()
        k_b_rep = k_b.repeat_interleave(repeat, dim=0)
        v_b_rep = v_b.repeat_interleave(repeat, dim=0)
        scores = torch.einsum("hd,hsd->hs", q_b, k_b_rep) * scale
        weights = torch.softmax(scores, dim=-1)
        out[0, b, :, :] = torch.einsum("hs,hsd->hd", weights, v_b_rep)
    return out


# ── fill → update → SDPA-read round-trip ────────────────────────────────────


@parametrize_mesh_with_fabric(mesh_shapes=_MESH_SHAPES)
@pytest.mark.parametrize("batch", [1, 16, 32], ids=lambda b: f"batch{b}")
@pytest.mark.parametrize("block_size", BLOCK_SIZES, ids=lambda s: f"block{s}")
def test_paged_kv_cache(batch, block_size, mesh_device, reset_seeds, request):
    """fill (prefill chunk) -> update (1 decode token) -> paged SDPA-read, vs a
    torch SDPA reference, swept over ``block_size`` and ``batch``.

    Uses the sliding-layer geometry (``head_dim=256``); the SDPA op runs scale=1.0
    (Gemma4 has no 1/sqrt(d) scaling). Each user gets ``fill_len = 2 * block_size``
    prefill tokens (two full physical blocks) plus one appended decode token, so the
    write straddles a block boundary at every block size.
    """
    layer_idx = 0  # sliding attention layer
    config = Gemma4AttentionConfig(TestFactory.create_hf_config(), layer_idx)
    tp = _tp_of(mesh_device)

    num_kv_local = _local_kv_heads(config, tp)
    num_q_local = config.num_attention_heads // tp
    head_dim = config.head_dim

    fill_len = 2 * block_size
    cur_pos = fill_len  # the appended decode token's index (same for every user)
    blocks_per_user = fill_len // block_size + 1  # +1 for the decode token's block
    max_num_blocks = blocks_per_user * batch

    page_table = _contiguous_page_table(batch, blocks_per_user)
    page_table_tt = _replicate(page_table, mesh_device, ttnn.ROW_MAJOR_LAYOUT, ttnn.int32)

    k_cache_tt = _alloc_paged_cache(max_num_blocks, num_kv_local, block_size, head_dim, mesh_device)
    v_cache_tt = _alloc_paged_cache(max_num_blocks, num_kv_local, block_size, head_dim, mesh_device)
    assert k_cache_tt.padded_shape[2] == block_size
    assert k_cache_tt.padded_shape[1] == num_kv_local

    # Per-user K/V: fill_len prefill tokens + 1 decode token = fill_len + 1.
    k_full = torch.randn(batch, num_kv_local, fill_len + 1, head_dim)
    v_full = torch.randn(batch, num_kv_local, fill_len + 1, head_dim)

    # 1. Prefill fill: write the first fill_len tokens per user.
    for b in range(batch):
        k_fill = _replicate(k_full[b : b + 1, :, :fill_len, :], mesh_device, ttnn.TILE_LAYOUT, ttnn.bfloat16)
        v_fill = _replicate(v_full[b : b + 1, :, :fill_len, :], mesh_device, ttnn.TILE_LAYOUT, ttnn.bfloat16)
        ttnn.experimental.paged_fill_cache(k_cache_tt, k_fill, page_table_tt, batch_idx=b)
        ttnn.experimental.paged_fill_cache(v_cache_tt, v_fill, page_table_tt, batch_idx=b)

    # 2. Decode update: append token at cur_pos for every user. Decode layout is
    # [1, B, num_kv_local, head_dim] with the head axis tile-padded to 32.
    padded_kv = _ceil_to_tile(num_kv_local)
    # Decode token [1, B, num_kv_local, head_dim]: element [0, b, h, :] is user b's
    # head-h K/V at position fill_len. k_full[:, :, fill_len, :] is already
    # [B, num_kv_local, head_dim], so a plain reshape preserves (b, h) order.
    k_tok = k_full[:, :, fill_len, :].reshape(1, batch, num_kv_local, head_dim)
    v_tok = v_full[:, :, fill_len, :].reshape(1, batch, num_kv_local, head_dim)
    k_tok_p = torch.nn.functional.pad(k_tok, (0, 0, 0, padded_kv - num_kv_local))
    v_tok_p = torch.nn.functional.pad(v_tok, (0, 0, 0, padded_kv - num_kv_local))
    cur_pos_tt = _replicate(
        torch.full((batch,), cur_pos, dtype=torch.int32), mesh_device, ttnn.ROW_MAJOR_LAYOUT, ttnn.int32
    )
    ttnn.experimental.paged_update_cache(
        k_cache_tt,
        _height_sharded_decode_input(k_tok_p, mesh_device),
        update_idxs_tensor=cur_pos_tt,
        page_table=page_table_tt,
    )
    ttnn.experimental.paged_update_cache(
        v_cache_tt,
        _height_sharded_decode_input(v_tok_p, mesh_device),
        update_idxs_tensor=cur_pos_tt,
        page_table=page_table_tt,
    )

    # 3. SDPA-read the freshly-updated cache (attends to fill_len + 1 tokens).
    # Keep Q's logical head count = num_q_local (TILE layout pads internally). The
    # decode SDPA op derives the GQA group size from Q's logical head dim, so a
    # manual pad to 32 here would inflate num_q_heads and corrupt the grouping at
    # TP>1 (num_q_local < 32).
    q = torch.randn(1, batch, num_q_local, head_dim)
    q_tt = _replicate(q, mesh_device, ttnn.TILE_LAYOUT, ttnn.bfloat16)
    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(8, 4),
        q_chunk_size=32,
        k_chunk_size=32,
        exp_approx_mode=False,
    )
    # Use 1/sqrt(d) scaling (not Gemma's production scale=1.0) purely for numerical
    # conditioning: this test validates the paged fill/update/read *indexing*, not
    # the model's scaling, and large unscaled logits amplify bf16 softmax error at
    # the smallest GQA shapes. The scale is applied identically to op and reference.
    scale = 1.0 / (head_dim**0.5)
    out_tt = ttnn.transformer.paged_scaled_dot_product_attention_decode(
        q_tt,
        k_cache_tt,
        v_cache_tt,
        cur_pos_tensor=cur_pos_tt,
        page_table_tensor=page_table_tt,
        scale=scale,
        program_config=program_config,
    )
    out = _from_device(out_tt, mesh_device)[:, :, :num_q_local, :].float()

    cur_pos_vec = [cur_pos] * batch
    ref = _torch_sdpa_reference(q, k_full, v_full, cur_pos_vec, scale=scale)
    passing, pcc_msg = compare_tensors(out, ref, pcc_threshold=get_pcc_threshold(request))
    assert passing, (
        f"paged fill->update->SDPA round-trip (batch={batch}, block_size={block_size}, "
        f"tp={tp}) PCC too low: {pcc_msg}"
    )


# ── isolated paged_update_cache microbenchmark (Tracy profiling vehicle) ─────


@parametrize_mesh_with_fabric(mesh_shapes=_MESH_SHAPES)
@pytest.mark.parametrize("batch", [1], ids=lambda b: f"batch{b}")
@pytest.mark.parametrize("block_size", BLOCK_SIZES, ids=lambda s: f"block{s}")
def test_paged_update_cache_microbench(batch, block_size, mesh_device, reset_seeds, request):
    """Repeated ``paged_update_cache`` on the decode ``[1, B, num_kv_local, head_dim]``
    layout — the isolated profiling target for the dispatch/DRAM measurement.

    Profile with (device profiler on, program-support count raised so the
    repeated dispatch does not exhaust the profiler program buffer)::

        TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=20000 \
          python -m tracy -p -r -v -m pytest \
          models/demos/gemma4/tests/unit/test_paged_kv_cache.py \
          -k "1x4 and update and batch1 and block64" -sv

    The low op count (vs the full multi-device decode loop) lets Tracy
    post-processing complete and attribute ``PagedUpdateCacheDeviceOperation`` a
    ``DEVICE KERNEL DURATION [ns]`` + ``DRAM BW UTIL (%)``.

    The measured iterations are bracketed by ``tracy.signpost("start"/"stop")``
    so the capture isolates the steady-state region from the first-iteration op
    compile / program-cache warmup — filter to the signpost zone in the Tracy
    timeline to read pure dispatch + device duration.
    """
    layer_idx = 0
    config = Gemma4AttentionConfig(TestFactory.create_hf_config(), layer_idx)
    tp = _tp_of(mesh_device)
    num_kv_local = _local_kv_heads(config, tp)
    head_dim = config.head_dim

    blocks_per_user = 8
    max_num_blocks = blocks_per_user * batch
    max_seq_len = blocks_per_user * block_size

    page_table = _contiguous_page_table(batch, blocks_per_user)
    page_table_tt = _replicate(page_table, mesh_device, ttnn.ROW_MAJOR_LAYOUT, ttnn.int32)
    k_cache_tt = _alloc_paged_cache(max_num_blocks, num_kv_local, block_size, head_dim, mesh_device)

    padded_kv = _ceil_to_tile(num_kv_local)
    tok = torch.randn(1, batch, num_kv_local, head_dim)
    tok_p = torch.nn.functional.pad(tok, (0, 0, 0, padded_kv - num_kv_local))
    tok_tt = _height_sharded_decode_input(tok_p, mesh_device)

    def _update(pos):
        cur_pos_tt = _replicate(
            torch.full((batch,), pos, dtype=torch.int32), mesh_device, ttnn.ROW_MAJOR_LAYOUT, ttnn.int32
        )
        ttnn.experimental.paged_update_cache(
            k_cache_tt,
            tok_tt,
            update_idxs_tensor=cur_pos_tt,
            page_table=page_table_tt,
        )

    # Warmup: first call compiles the op + populates the program cache; keep it
    # outside the signpost zone so the measured region is steady-state only.
    num_warmup = 2
    for i in range(num_warmup):
        _update((i * block_size) % (max_seq_len - 1))
    ttnn.synchronize_device(mesh_device)

    # Sweep the write position across the sequence so the op touches every block.
    num_iters = 32
    _tracy_signpost("start")
    for i in range(num_iters):
        pos = ((num_warmup + i) * block_size) % (max_seq_len - 1)
        _update(pos)
    ttnn.synchronize_device(mesh_device)
    _tracy_signpost("stop")

    # Light correctness anchor: the last write must land in the expected block.
    last_pos = ((num_warmup + num_iters - 1) * block_size) % (max_seq_len - 1)
    got = _from_device(k_cache_tt, mesh_device).float()
    phys_block = int(page_table[0, last_pos // block_size].item())
    row = last_pos % block_size
    written = got[phys_block, :num_kv_local, row, :]
    expected = tok[0, 0, :, :]
    passing, pcc_msg = compare_tensors(written, expected, pcc_threshold=get_pcc_threshold(request))
    assert passing, f"microbench last write mismatch (block_size={block_size}, tp={tp}): {pcc_msg}"


# ── batched prefill fill + sliced page table ────────────────────────────────


@parametrize_mesh_with_fabric(mesh_shapes=_MESH_SHAPES)
@pytest.mark.parametrize("padded_batch", [8, 16, 32], ids=lambda b: f"batch{b}")
@pytest.mark.parametrize("block_size", [64, 128], ids=lambda s: f"block{s}")
def test_paged_fill_cache_batched(padded_batch, block_size, mesh_device, reset_seeds, request):
    """Batched-prefill ``paged_fill_cache``: each user's chunk must land in exactly
    the physical blocks named by ``page_table[user, :]`` and nowhere else.

    Mirrors the ``batch_size > 1`` branch in ``prefill.py`` (fill user-by-user via
    ``batch_idx``) and the sliced-``page_table`` indexing — a per-user fill that
    wrote into a neighbour's blocks would be caught by comparing every user's
    gathered blocks against its own input.
    """
    layer_idx = 0
    config = Gemma4AttentionConfig(TestFactory.create_hf_config(), layer_idx)
    tp = _tp_of(mesh_device)
    num_kv_local = _local_kv_heads(config, tp)
    head_dim = config.head_dim

    seq_len = 2 * block_size  # two full blocks per user
    blocks_per_user = seq_len // block_size
    max_num_blocks = blocks_per_user * padded_batch

    # Interleave block ownership so a user's blocks are NOT a trivial 0..N prefix —
    # forces the fill to honour the page_table indirection rather than write linearly.
    perm = torch.randperm(max_num_blocks)
    page_table = perm.reshape(padded_batch, blocks_per_user).to(torch.int32)
    page_table_tt = _replicate(page_table, mesh_device, ttnn.ROW_MAJOR_LAYOUT, ttnn.int32)

    k_cache_tt = _alloc_paged_cache(max_num_blocks, num_kv_local, block_size, head_dim, mesh_device)

    k_users = torch.randn(padded_batch, num_kv_local, seq_len, head_dim)
    for b in range(padded_batch):
        k_fill = _replicate(k_users[b : b + 1], mesh_device, ttnn.TILE_LAYOUT, ttnn.bfloat16)
        ttnn.experimental.paged_fill_cache(k_cache_tt, k_fill, page_table_tt, batch_idx=b)

    got = _from_device(k_cache_tt, mesh_device).float()
    # Reassemble each user's sequence from its (non-contiguous) physical blocks.
    assembled = torch.zeros(padded_batch, num_kv_local, seq_len, head_dim)
    for b in range(padded_batch):
        for vb in range(blocks_per_user):
            pb = int(page_table[b, vb].item())
            assembled[b, :, vb * block_size : (vb + 1) * block_size, :] = got[pb, :num_kv_local, :, :]

    passing, pcc_msg = compare_tensors(assembled, k_users, pcc_threshold=get_pcc_threshold(request))
    assert passing, (
        f"batched paged_fill_cache (padded_batch={padded_batch}, block_size={block_size}, "
        f"tp={tp}) mismatch — a user's chunk did not land in its page_table blocks: {pcc_msg}"
    )
