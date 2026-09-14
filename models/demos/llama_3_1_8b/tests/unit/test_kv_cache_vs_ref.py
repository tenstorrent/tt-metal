# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""KV cache write and read-back at the spec's shape (sp=8 x tp=4, head_dim 128, 2 KV heads/chip).

What this has to catch is the class of bug that does not raise: the block-cyclic write lands at the
wrong offset, or in the wrong layer's slot, or the head sharding is transposed. All of those produce
a populated cache full of plausible numbers.

So: write, read the whole slot back, un-rotate, and compare against the host tensor that was written
— plus the three controls that make the comparison mean something (a second chunk must not disturb
the first, a layer must not read another layer's slot, and a shifted un-rotation must NOT match).
"""

from __future__ import annotations

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.llama_3_1_8b.tests.common import assert_pcc, cfg_full, galaxy_mesh, pcc, spec_mesh_config
from models.demos.llama_3_1_8b.tt.attention.kv_cache import allocate_kv_cache, read_slot_kv, write_kv_chunk
from models.demos.llama_3_1_8b.tt.runners.prefill_kv_validation import naturalize

CHUNK = 5120
CACHE = 10240
NUM_LAYERS = 4  # a cache-shape test; the layer count is about slot arithmetic, not model depth


def _shard_kv(host, mesh_device, mc, dtype=ttnn.bfloat16):
    """``[1, num_kv_heads, chunk, head_dim]`` -> heads on TP, sequence on SP."""
    dims = [None, None]
    dims[mc.sp_axis] = 2
    dims[mc.tp_axis] = 1
    return ttnn.from_torch(
        host,
        device=mesh_device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_device.shape, dims=tuple(dims)),
    )


def _read_layer(mesh_device, cache, layer, n_tokens, mc):
    k_blk, v_blk = read_slot_kv(mesh_device, cache, 0, NUM_LAYERS)
    return (
        naturalize(k_blk[layer], n_tokens, mc.sp, CHUNK, CACHE),
        naturalize(v_blk[layer], n_tokens, mc.sp, CHUNK, CACHE),
    )


@galaxy_mesh()
def test_kv_cache_shape_matches_the_spec(mesh_device, device_params):
    """The per-chip cache must carry ``num_kv_heads / tp`` heads — 2 here, not the sources' 1.

    ``update_padded_kv_cache`` enforces ``cache_shape[1] == input_shape[1]``, so a cache allocated
    with the borrowed literal 1 fails loudly on the first write rather than corrupting anything.
    That is worth an explicit test because it is the single place this model's shape departs from
    the canonical layout.
    """
    cfg = cfg_full()
    mc = spec_mesh_config(mesh_device)
    cache = allocate_kv_cache(
        mesh_device,
        num_layers=NUM_LAYERS,
        max_seq_len=CACHE,
        num_kv_heads=cfg.num_key_value_heads,
        tp=mc.tp,
        sp_axis=mc.sp_axis,
        head_dim=cfg.head_dim,
    )
    assert cache.n_kv_local == cfg.num_key_value_heads // mc.tp == 2
    assert tuple(cache.k.shape) == (NUM_LAYERS, 2, CACHE // mc.sp, cfg.head_dim)
    assert cache.k.dtype == ttnn.bfloat8_b, "the spec binds kv_cache dataformat bfloat8_b"


@galaxy_mesh()
@pytest.mark.parametrize("layer", [0, 3], ids=["L0", "L3"])
def test_kv_cache_write_vs_ref(mesh_device, device_params, layer, topology_name):
    """One chunk in, the same chunk back out of the right layer's slot."""
    cfg = cfg_full()
    mc = spec_mesh_config(mesh_device)
    cache = allocate_kv_cache(
        mesh_device,
        num_layers=NUM_LAYERS,
        max_seq_len=CACHE,
        num_kv_heads=cfg.num_key_value_heads,
        tp=mc.tp,
        sp_axis=mc.sp_axis,
        head_dim=cfg.head_dim,
    )
    torch.manual_seed(layer)
    k = torch.randn(1, cfg.num_key_value_heads, CHUNK, cfg.head_dim)
    v = torch.randn(1, cfg.num_key_value_heads, CHUNK, cfg.head_dim)

    write_kv_chunk(
        cache, _shard_kv(k, mesh_device, mc), _shard_kv(v, mesh_device, mc),
        slot_idx=0, layer_idx=layer, kv_actual=0, sp_axis=mc.sp_axis,
    )
    got_k, got_v = _read_layer(mesh_device, cache, layer, CHUNK, mc)
    assert_pcc(f"kv_write K[{topology_name}] L{layer}", pcc(k[0], got_k))
    assert_pcc(f"kv_write V[{topology_name}] L{layer}", pcc(v[0], got_v))

    # Control: an un-rotation that is off by one chunk-local block must NOT match, otherwise the
    # block-cyclic inverse is not actually doing anything.
    shifted = torch.roll(got_k, shifts=CHUNK // mc.sp, dims=-2)
    assert pcc(k[0], shifted) < 0.5, "the block-cyclic un-rotation is not position-sensitive"


@galaxy_mesh()
def test_kv_cache_two_chunk_append(mesh_device, device_params, topology_name):
    """Chunk 1 appends at ``kv_actual=5120`` without disturbing chunk 0 — the mechanism P2 rests on."""
    cfg = cfg_full()
    mc = spec_mesh_config(mesh_device)
    cache = allocate_kv_cache(
        mesh_device,
        num_layers=NUM_LAYERS,
        max_seq_len=CACHE,
        num_kv_heads=cfg.num_key_value_heads,
        tp=mc.tp,
        sp_axis=mc.sp_axis,
        head_dim=cfg.head_dim,
    )
    torch.manual_seed(11)
    k = torch.randn(1, cfg.num_key_value_heads, CACHE, cfg.head_dim)
    v = torch.randn(1, cfg.num_key_value_heads, CACHE, cfg.head_dim)

    for i, start in enumerate((0, CHUNK)):
        write_kv_chunk(
            cache,
            _shard_kv(k[:, :, start : start + CHUNK], mesh_device, mc),
            _shard_kv(v[:, :, start : start + CHUNK], mesh_device, mc),
            slot_idx=0,
            layer_idx=1,
            kv_actual=start,
            sp_axis=mc.sp_axis,
        )
        got_k, _ = _read_layer(mesh_device, cache, 1, start + CHUNK, mc)
        p = pcc(k[0, :, : start + CHUNK], got_k)
        logger.info(f"after chunk {i}: cumulative K PCC over {start + CHUNK} tokens = {p:.6f}")
        assert_pcc(f"kv_two_chunk[{topology_name}] after chunk {i}", p)

    got_k, got_v = _read_layer(mesh_device, cache, 1, CACHE, mc)
    assert_pcc(f"kv_two_chunk K[{topology_name}]", pcc(k[0], got_k))
    assert_pcc(f"kv_two_chunk V[{topology_name}]", pcc(v[0], got_v))


@galaxy_mesh()
def test_kv_cache_layer_slots_are_isolated(mesh_device, device_params):
    """Each layer must land in its own slot.

    ``slot = user*num_layers + layer`` is the arithmetic both the writer and the ring SDPA's
    ``kv_cache_batch_idx`` depend on; if they ever disagree, every layer reads layer 0's K/V and the
    model is wrong in a way that still runs.
    """
    cfg = cfg_full()
    mc = spec_mesh_config(mesh_device)
    cache = allocate_kv_cache(
        mesh_device,
        num_layers=NUM_LAYERS,
        max_seq_len=CACHE,
        num_kv_heads=cfg.num_key_value_heads,
        tp=mc.tp,
        sp_axis=mc.sp_axis,
        head_dim=cfg.head_dim,
    )
    per_layer = {}
    for layer in range(NUM_LAYERS):
        torch.manual_seed(100 + layer)
        k = torch.randn(1, cfg.num_key_value_heads, CHUNK, cfg.head_dim)
        per_layer[layer] = k
        write_kv_chunk(
            cache, _shard_kv(k, mesh_device, mc), _shard_kv(k, mesh_device, mc),
            slot_idx=0, layer_idx=layer, kv_actual=0, sp_axis=mc.sp_axis,
        )

    k_blk, _ = read_slot_kv(mesh_device, cache, 0, NUM_LAYERS)
    for layer in range(NUM_LAYERS):
        got = naturalize(k_blk[layer], CHUNK, mc.sp, CHUNK, CACHE)
        assert_pcc(f"kv_slot L{layer}", pcc(per_layer[layer][0], got))
        for other in range(NUM_LAYERS):
            if other != layer:
                assert pcc(per_layer[other][0], got) < 0.5, f"layer {layer}'s slot reads like layer {other}'s"
