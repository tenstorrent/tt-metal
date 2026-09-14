# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The SP-sharded ring SDPA, both ways it is used — live K/V and reading the block-cyclic KV cache.

These two tests are the mechanism chunked prefill is built on, so they are op-level and deliberately
independent of the attention block: Q/K/V go in as plain random tensors, sharded exactly the way the
model shards them, and the answer is compared against unsharded causal GQA on the host.

* **live** — each device's query shard attends the full sequence reconstructed across the SP ring by
  online softmax. This is chunk 0.
* **cache-read** — a short Q attends a longer accumulated prefix that is already in the cache. This
  is chunk N > 0, and it is where the KV layout, the ``kv_cache_batch_idx`` slot arithmetic and the
  ``kv_actual_isl`` rotation all have to agree at once.

GQA is grouped natively: 8 query heads over 2 KV heads per chip, with the cache never inflated to 8.
"""

from __future__ import annotations

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.llama_3_1_8b.reference.model import causal_sdpa
from models.demos.llama_3_1_8b.tests.common import (
    assert_pcc,
    cfg_full,
    galaxy_mesh,
    make_ccl,
    pcc,
    spec_mesh_config,
)
from models.demos.llama_3_1_8b.tt.attention.dense_sp import ring_sdpa_cache_read, ring_sdpa_live
from models.demos.llama_3_1_8b.tt.attention.kv_cache import allocate_kv_cache, write_kv_chunk
from models.demos.llama_3_1_8b.tt.compute import ring_sdpa_compute_config, ring_sdpa_program_config

NUM_LAYERS = 4


def _shard_heads_seq(host, mesh_device, mc, dtype=ttnn.bfloat16):
    """``[1, heads, seq, d]`` -> heads across TP, seq across SP."""
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


def _gather_heads_seq(tt, mesh_device, mc):
    dims = [None, None]
    dims[mc.sp_axis] = 2
    dims[mc.tp_axis] = 1
    composer = ttnn.ConcatMesh2dToTensor(mesh_device, dims=tuple(dims), mesh_shape=mesh_device.shape)
    return ttnn.to_torch(tt, mesh_composer=composer).float()


@galaxy_mesh()
@pytest.mark.parametrize("seq", [1024, 5120], ids=["s1024", "chunk5120"])
def test_ring_sdpa_live_vs_ref(mesh_device, device_params, seq, topology_name):
    """Ring SDPA over the chunk's own SP-sharded K/V vs unsharded causal GQA."""
    cfg = cfg_full()
    mc = spec_mesh_config(mesh_device)
    ccl = make_ccl(mesh_device)
    torch.manual_seed(0)
    q = torch.randn(1, cfg.num_attention_heads, seq, cfg.head_dim) * 0.5
    k = torch.randn(1, cfg.num_key_value_heads, seq, cfg.head_dim) * 0.5
    v = torch.randn(1, cfg.num_key_value_heads, seq, cfg.head_dim) * 0.5

    out = ring_sdpa_live(
        _shard_heads_seq(q, mesh_device, mc),
        _shard_heads_seq(k, mesh_device, mc),
        _shard_heads_seq(v, mesh_device, mc),
        mesh_config=mc,
        ccl_manager=ccl,
        logical_n=seq,
        n_kv_global=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
        scale=cfg.scale if hasattr(cfg, "scale") else cfg.head_dim**-0.5,
        program_config=ring_sdpa_program_config(mesh_device),
        compute_kernel_config=ring_sdpa_compute_config(),
    )
    got = _gather_heads_seq(out, mesh_device, mc)
    ref = causal_sdpa(q.to(torch.float16), k.to(torch.float16), v.to(torch.float16), cfg.head_dim**-0.5).float()
    assert_pcc(f"ring_sdpa_live[{topology_name}] s={seq}", pcc(ref, got))


@galaxy_mesh()
@pytest.mark.parametrize("chunk", [1024, 5120], ids=["chunk1024", "chunk5120"])
def test_ring_sdpa_cache_read_vs_ref(mesh_device, device_params, chunk, topology_name):
    """Chunk 1's Q against the accumulated prefix in the block-cyclic cache.

    Both chunks' K/V are written to the cache first (which is what the model does), then the second
    chunk's Q reads back over the whole ``2*chunk`` prefix. The reference is plain causal GQA over
    the full sequence, sliced to the second chunk's query rows.
    """
    cfg = cfg_full()
    mc = spec_mesh_config(mesh_device)
    ccl = make_ccl(mesh_device)
    cache_len = 2 * chunk
    layer = 2  # not 0 — a wrong kv_cache_batch_idx is invisible at layer 0

    cache = allocate_kv_cache(
        mesh_device,
        num_layers=NUM_LAYERS,
        max_seq_len=cache_len,
        num_kv_heads=cfg.num_key_value_heads,
        tp=mc.tp,
        sp_axis=mc.sp_axis,
        head_dim=cfg.head_dim,
    )

    torch.manual_seed(5)
    q = torch.randn(1, cfg.num_attention_heads, cache_len, cfg.head_dim) * 0.5
    k = torch.randn(1, cfg.num_key_value_heads, cache_len, cfg.head_dim) * 0.5
    v = torch.randn(1, cfg.num_key_value_heads, cache_len, cfg.head_dim) * 0.5

    for start in (0, chunk):
        write_kv_chunk(
            cache,
            _shard_heads_seq(k[:, :, start : start + chunk], mesh_device, mc),
            _shard_heads_seq(v[:, :, start : start + chunk], mesh_device, mc),
            slot_idx=0,
            layer_idx=layer,
            kv_actual=start,
            sp_axis=mc.sp_axis,
        )

    out = ring_sdpa_cache_read(
        _shard_heads_seq(q[:, :, chunk:], mesh_device, mc),
        cache.k,
        cache.v,
        mesh_config=mc,
        ccl_manager=ccl,
        kv_actual=chunk,
        logical_n=cache_len,
        cache_global=cache_len,
        n_kv_global=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
        scale=cfg.head_dim**-0.5,
        program_config=ring_sdpa_program_config(mesh_device),
        compute_kernel_config=ring_sdpa_compute_config(),
        slot_idx=0,
        layer_idx=layer,
        num_layers=NUM_LAYERS,
    )
    got = _gather_heads_seq(out, mesh_device, mc)

    full = causal_sdpa(q.to(torch.float16), k.to(torch.float16), v.to(torch.float16), cfg.head_dim**-0.5).float()
    ref = full[:, :, chunk:]
    assert_pcc(f"ring_sdpa_cache_read[{topology_name}] chunk={chunk}", pcc(ref, got))

    # Control: the second chunk must genuinely be attending the prefix, not just itself.
    self_only = causal_sdpa(
        q[:, :, chunk:].to(torch.float16), k[:, :, chunk:].to(torch.float16), v[:, :, chunk:].to(torch.float16),
        cfg.head_dim**-0.5,
    ).float()
    p_self = pcc(self_only, got)
    logger.info(f"chunk-1 output vs prefix-blind attention: PCC {p_self:.6f}")
    assert p_self < 0.99, "cache-read output looks like the chunk attending only itself"


@galaxy_mesh()
def test_ring_sdpa_reads_the_right_layer_slot(mesh_device, device_params):
    """Reading layer L must return layer L's K/V.

    Two layers get deliberately different K/V; attention over layer 2's slot must match layer 2's
    reference and NOT layer 0's. This is the direct test for the ``kv_cache_batch_idx = slot *
    num_layers + layer`` folding, whose absence is silent on layer 0 and wrong everywhere else.
    """
    cfg = cfg_full()
    mc = spec_mesh_config(mesh_device)
    ccl = make_ccl(mesh_device)
    chunk, cache_len = 1024, 2048

    cache = allocate_kv_cache(
        mesh_device,
        num_layers=NUM_LAYERS,
        max_seq_len=cache_len,
        num_kv_heads=cfg.num_key_value_heads,
        tp=mc.tp,
        sp_axis=mc.sp_axis,
        head_dim=cfg.head_dim,
    )
    torch.manual_seed(9)
    q = torch.randn(1, cfg.num_attention_heads, cache_len, cfg.head_dim) * 0.5
    kv = {}
    for layer in (0, 2):
        torch.manual_seed(50 + layer)
        k = torch.randn(1, cfg.num_key_value_heads, cache_len, cfg.head_dim) * 0.5
        v = torch.randn(1, cfg.num_key_value_heads, cache_len, cfg.head_dim) * 0.5
        kv[layer] = (k, v)
        for start in (0, chunk):
            write_kv_chunk(
                cache,
                _shard_heads_seq(k[:, :, start : start + chunk], mesh_device, mc),
                _shard_heads_seq(v[:, :, start : start + chunk], mesh_device, mc),
                slot_idx=0,
                layer_idx=layer,
                kv_actual=start,
                sp_axis=mc.sp_axis,
            )

    out = ring_sdpa_cache_read(
        _shard_heads_seq(q[:, :, chunk:], mesh_device, mc),
        cache.k,
        cache.v,
        mesh_config=mc,
        ccl_manager=ccl,
        kv_actual=chunk,
        logical_n=cache_len,
        cache_global=cache_len,
        n_kv_global=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
        scale=cfg.head_dim**-0.5,
        program_config=ring_sdpa_program_config(mesh_device),
        compute_kernel_config=ring_sdpa_compute_config(),
        slot_idx=0,
        layer_idx=2,
        num_layers=NUM_LAYERS,
    )
    got = _gather_heads_seq(out, mesh_device, mc)

    def ref_for(layer):
        k, v = kv[layer]
        return causal_sdpa(
            q.to(torch.float16), k.to(torch.float16), v.to(torch.float16), cfg.head_dim**-0.5
        ).float()[:, :, chunk:]

    p2, p0 = pcc(ref_for(2), got), pcc(ref_for(0), got)
    logger.info(f"read layer 2: PCC vs L2 ={p2:.6f}, vs L0 ={p0:.6f}")
    assert_pcc("ring_sdpa_layer_slot", p2)
    assert p0 < 0.9, "reading layer 2 returned something that looks like layer 0's cache"
