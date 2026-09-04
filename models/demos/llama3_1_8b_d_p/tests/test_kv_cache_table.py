# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""P4: the KV chunk address table, checked BIT-EXACTLY against device DRAM.

Ported from ``deepseek_v3_d_p/tests/test_kv_cache_table.py`` rather than the gpt-oss one, per the
bring-up recipe: the gpt-oss variant is parametrized only for a ``(2,4)`` submesh, which cannot bring
fabric up on a Galaxy.

Allocates a real cache on the target mesh, fills it with a known pattern, then reads raw bytes back
**from device DRAM at the addresses the table computed** and compares them to what should be there.
That proves four things at once, none of which any PCC test can:

  * the address arithmetic (bank id + per-bank offset from slot/layer/head/position);
  * the DRAM ND-shard ROUND_ROBIN_1D bank walk;
  * the packed bf8 byte decode;
  * that a protobuf round-trip preserves config ids and lookups.

It is the highest-value check on the layout for this model, because Llama is the first one here with
**2 KV heads per chip** — the table has to step over a head dim the donors' tables never had (see
``tt/runners/kv_chunk_table.py``). A mistake produces valid-looking addresses pointing at the wrong
tokens: migration would move real bytes to the wrong place, silently.

Runs no model and moves nothing over fabric.
"""

import os
import tempfile

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.tt.mla.utils import block_cyclic_reorder
from models.demos.llama3_1_8b_d_p.tt.attention.kv_cache import NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK, allocate_kv_cache
from models.demos.llama3_1_8b_d_p.tt.runners.kv_chunk_table import (
    build_kv_chunk_address_table,
    chunk_size_bytes,
    config_specs,
    stable_config_name,
)
from tests.ttnn.utils_for_testing import assert_equal

from .test_factory import llama_config, make_mesh_config, parametrize_mesh_with_fabric


@parametrize_mesh_with_fabric(mesh_shapes=[(8, 4)])
@pytest.mark.parametrize("num_layers", [1, 3], ids=["L1", "L3"])
@pytest.mark.parametrize("num_users", [1, 2], ids=["u1", "u2"])
@pytest.mark.timeout(0)
def test_kv_cache_table(mesh_device, device_params, num_layers, num_users, reset_seeds):
    """Fill the cache with a known pattern, then verify every table address reads it back exactly.

    ``num_users=2`` and ``num_layers=3`` are what pin the user-major slot packing
    ``slot = user*num_layers + layer``: with one user and one layer every packing formula agrees.
    """
    cfg = llama_config()
    hd = cfg.head_dim
    n_kv = cfg.num_key_value_heads
    mesh_config = make_mesh_config(mesh_device)
    rows, cols = tuple(mesh_device.shape)
    sp, sp_axis, tp_axis = mesh_config.sp, mesh_config.sp_axis, mesh_config.tp_axis
    n_kv_local = n_kv // mesh_config.tp

    # Small but multi-chunk: the block-cyclic mapping is only exercised with >1 chunk.
    chunk_size = 512
    n_chunks = 4
    seq_len = n_chunks * chunk_size
    chunk_local = chunk_size // sp
    assert chunk_local % NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK == 0

    kv_cache = allocate_kv_cache(
        mesh_device,
        num_layers=num_layers,
        max_seq_len=seq_len,
        sp_axis=sp_axis,
        num_users=num_users,
        head_dim=hd,
        num_kv_heads_local=n_kv_local,
    )

    # A pattern that is distinct per (slot, layer, head, position, column) so any mis-addressing
    # lands on visibly wrong data rather than a plausible neighbour.
    def pattern(cache_name):
        base = 0.0 if cache_name == "k" else 0.5
        t = torch.zeros(num_users * num_layers, n_kv, seq_len, hd)
        for b in range(num_users * num_layers):
            for h in range(n_kv):
                pos = torch.arange(seq_len, dtype=torch.float32)[:, None]
                col = torch.arange(hd, dtype=torch.float32)[None, :]
                t[b, h] = base + b * 1000.0 + h * 100.0 + pos * 0.01 + col * 0.0001
        return t

    written = {}
    for cache_name, cache_tensor in (("k", kv_cache.k), ("v", kv_cache.v)):
        host = pattern(cache_name)
        written[cache_name] = host
        # Lay the host tensor out exactly as the device holds it: sequence block-cyclic on the SP
        # rows, KV heads sharded on the TP cols.
        host_bc = block_cyclic_reorder(host, chunk_local, sp, seq_dim=2)
        dims = [None, None]
        dims[sp_axis], dims[tp_axis] = 2, 1
        staged = ttnn.from_torch(
            host_bc,
            dtype=cache_tensor.dtype,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=cache_tensor.memory_config(),
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(rows, cols), dims=tuple(dims)),
        )
        ttnn.copy(staged, cache_tensor)
        staged.deallocate(True)
    ttnn.synchronize_device(mesh_device)

    table = build_kv_chunk_address_table(
        mesh_device=mesh_device,
        kv_cache=kv_cache,
        seq_len=seq_len,
        num_layers=num_layers,
        mesh_shape=(rows, cols),
        sp_axis=sp_axis,
        num_users=num_users,
        chunk_size=chunk_size,
        num_kv_heads=n_kv,
        head_dim=hd,
    )

    specs = config_specs(n_kv)
    assert table.num_configs() == len(specs) == 2 * n_kv

    # Reference in the cache's own dtype, so the comparison is bit-exact rather than approximate.
    def to_cache_dtype(t):
        return ttnn.to_torch(ttnn.from_torch(t, dtype=kv_cache.k.dtype, layout=ttnn.TILE_LAYOUT)).to(torch.bfloat16)

    ref_bf8 = {name: to_cache_dtype(t) for name, t in written.items()}
    chunk_shape = [1, 1, NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK, hd]

    checked = 0
    for config_id, (label, cache_name, head) in enumerate(specs):
        for slot in range(num_users):
            for layer in range(num_layers):
                batch = slot * num_layers + layer
                for position in range(0, seq_len, NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK):
                    raw = table.read_device_chunk(layer=layer, position=position, slot=slot, config_id=config_id)
                    chunk_tt = ttnn.experimental.disaggregation.tensor_from_bfp8_bytes(raw, chunk_shape)
                    got = ttnn.to_torch(chunk_tt).to(torch.bfloat16)
                    want = ref_bf8[cache_name][
                        batch : batch + 1,
                        head : head + 1,
                        position : position + NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK,
                        :,
                    ]
                    assert_equal(got, want)
                    checked += 1

    logger.info(
        f"KV chunk table: {checked} chunks verified bit-exactly (configs={len(specs)}, users={num_users}, layers={num_layers})"
    )


@parametrize_mesh_with_fabric(mesh_shapes=[(8, 4)])
def test_kv_cache_table_protobuf_round_trip(mesh_device, device_params, reset_seeds):
    """Export the table, re-import it, and check every lookup survives.

    Protobuf rebuilds configs through a ``std::map``, so config NAMES decide the order on import. With
    16 configs, unpadded names would put ``"10"`` before ``"2"`` and silently renumber the ids — this
    asserts the zero-padded naming actually prevents that end-to-end.
    """
    cfg = llama_config()
    hd, n_kv = cfg.head_dim, cfg.num_key_value_heads
    mesh_config = make_mesh_config(mesh_device)
    rows, cols = tuple(mesh_device.shape)
    n_kv_local = n_kv // mesh_config.tp
    chunk_size, n_chunks, num_layers, num_users = 512, 2, 2, 1
    seq_len = n_chunks * chunk_size

    kv_cache = allocate_kv_cache(
        mesh_device,
        num_layers=num_layers,
        max_seq_len=seq_len,
        sp_axis=mesh_config.sp_axis,
        num_users=num_users,
        head_dim=hd,
        num_kv_heads_local=n_kv_local,
    )
    table = build_kv_chunk_address_table(
        mesh_device=mesh_device,
        kv_cache=kv_cache,
        seq_len=seq_len,
        num_layers=num_layers,
        mesh_shape=(rows, cols),
        sp_axis=mesh_config.sp_axis,
        num_users=num_users,
        chunk_size=chunk_size,
        num_kv_heads=n_kv,
        head_dim=hd,
    )

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "kv_table.pb")
        ttnn.experimental.disaggregation.export_to_protobuf_file(table, path)
        reloaded = ttnn.experimental.disaggregation.import_from_protobuf_file(path)

    assert reloaded.num_configs() == table.num_configs()
    n_cfg = table.num_configs()
    for i in range(n_cfg):
        assert reloaded.config_name(i) == stable_config_name(i, n_cfg) == table.config_name(i), (
            f"config {i} name changed across the protobuf round trip: "
            f"{table.config_name(i)!r} -> {reloaded.config_name(i)!r}"
        )
    assert reloaded.total_entries() == table.total_entries()
    logger.info(f"protobuf round trip preserved {n_cfg} configs / {reloaded.total_entries()} entries")


def test_chunk_size_bytes_matches_tile_geometry():
    """Host-only: the per-chunk byte size must match the bf8 tile geometry the writer assumes.

    ``[1, 1, 32, 128]`` bf8_b = 4 tiles x 1088 B = 4352 B — the spec's ``kv_cache`` note. If this
    drifts from the allocator's shard shape, every address after the first is off by a multiple of
    the error.
    """
    cfg = llama_config()
    assert chunk_size_bytes(ttnn.bfloat8_b, cfg.head_dim) == 4352
    assert chunk_size_bytes(ttnn.bfloat16, cfg.head_dim) == 4 * 2048
    with pytest.raises(AssertionError):
        chunk_size_bytes(ttnn.float32, cfg.head_dim)
