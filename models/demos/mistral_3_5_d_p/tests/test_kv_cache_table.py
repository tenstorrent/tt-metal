# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""P4: the KV chunk address table, verified by reading raw bytes back FROM device DRAM at the
addresses the table computed and comparing BIT-EXACTLY.

Ported from ``deepseek_v3_d_p/tests/test_kv_cache_table.py``, and deliberately NOT from
``gpt_oss_d_p/tests/test_kv_cache_table.py``: the gpt-oss variant is parametrized only for a ``(2,4)``
submesh, and no proper submesh can bring up the fabric on a Galaxy (carving chips out leaves their
ethernet partners outside the submesh with no router kernel running). The DeepSeek variant uses the
FULL mesh and has a no-weights ``random`` case, which is what runs here.

What this proves, and what nothing else does:
  * the address ARITHMETIC — every ``(layer, position, slot, config)`` maps to the DRAM address that
    actually holds that chunk;
  * the DRAM bank WALK — 32-token blocks round-robin across the bank grid, the per-bank offset
    advancing after each full sweep;
  * the packed-byte DECODE — bfp8 chunk bytes reconstruct the tile they were written from;
  * that a protobuf ROUND TRIP preserves lookups, which is where the zero-padded config names earn
    their keep (a ``std::map`` rebuild reorders unpadded ``"0".."15"``, and this table has 16
    configs).

It runs no model and moves nothing over fabric: allocate, write known K/V, read back through the
table. Comparison is bit-exact (``assert_equal``), not PCC — an address is either right or wrong.
"""

from __future__ import annotations

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.tt.mla.utils import block_cyclic_reorder, blockcyclic_positions
from models.demos.mistral_3_5_d_p.reference.mistral_config import MistralMedium35Config as C
from models.demos.mistral_3_5_d_p.spec import SPEC
from models.demos.mistral_3_5_d_p.tt.attention import allocate_kv_cache, write_kv_chunk
from models.demos.mistral_3_5_d_p.tt.attention.kv_cache import NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK
from models.demos.mistral_3_5_d_p.tt.runners.kv_chunk_table import (
    build_and_serialize_kv_chunk_table,
    build_kv_chunk_address_table,
)
from tests.ttnn.utils_for_testing import assert_equal

from .test_factory import parametrize_mesh, sp_tp_shard_mapper

HEAD_DIM, NKV = C.HEAD_DIM, C.NUM_KEY_VALUE_HEADS


def _fill_cache(mesh_device, kv_cache, *, chunk, capacity, n_kv, n_chunks):
    """Write known per-head K/V over the whole cache through the production write seam.

    Returns the natural-order ``(k, v)`` that was written, each ``[n_kv, capacity, head_dim]``.
    """
    sp = mesh_device.shape[SPEC.sp_axis]
    sent_k = torch.randn(n_kv, capacity, HEAD_DIM)
    sent_v = torch.randn(n_kv, capacity, HEAD_DIM)
    mapper = sp_tp_shard_mapper(mesh_device, seq_dim=2, head_dim=1)
    for i in range(n_chunks):
        lo = i * chunk

        # Reorder this chunk into the per-chip block-cyclic row order the writer expects.
        def to_device(src):
            bc = block_cyclic_reorder(
                src[:, lo : lo + chunk, :].reshape(1, n_kv, chunk, HEAD_DIM), chunk // sp, sp, seq_dim=2
            )
            return ttnn.from_torch(
                bc,
                device=mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mapper,
            )

        tt_k, tt_v = to_device(sent_k), to_device(sent_v)
        write_kv_chunk(kv_cache, tt_k, tt_v, slot_idx=0, layer_idx=0, kv_actual=lo, sp_axis=SPEC.sp_axis)
        tt_k.deallocate(True)
        tt_v.deallocate(True)
    ttnn.synchronize_device(mesh_device)
    return sent_k, sent_v


def _device_cache_rows(cache_tensor, mesh_device, *, slot, head):
    """The RAW on-device rows for one (slot, head): concat the SP rows' shards, no un-rotation.

    The table addresses raw cache rows, so the expected bytes are the raw layout — un-rotating here
    would compare the table against a tensor the table does not describe.
    """
    rows, cols = tuple(mesh_device.shape)
    shards = ttnn.get_device_tensors(cache_tensor)
    return torch.cat([ttnn.to_torch(shards[r * cols + head])[slot, 0] for r in range(rows)], dim=0)


@parametrize_mesh()
@pytest.mark.parametrize("n_chunks", [2], ids=["c2"])
@pytest.mark.parametrize("num_layers", [1], ids=["l1"])
@pytest.mark.timeout(0)
def test_kv_chunk_table_addresses_are_bit_exact(mesh_device, device_params, n_chunks, num_layers, reset_seeds):
    """Every chunk the table addresses must hold exactly the bytes the cache holds there.

    Walks all 16 configs (k head 0..7, then v head 0..7) and every 32-token position, reads the
    chunk back through the table, and compares bit-exactly against the raw device rows.
    """
    rows, cols = tuple(mesh_device.shape)
    sp = rows
    assert cols == NKV, "this table maps KV head c -> TP column c"
    chunk = ttnn.TILE_SIZE * sp * 4  # 512 at sp=4: tile-aligned, several bank sweeps per slot
    capacity = chunk * n_chunks

    kv_cache = allocate_kv_cache(
        mesh_device,
        num_layers=num_layers,
        max_seq_len=capacity,
        sp_axis=SPEC.sp_axis,
        num_users=1,
        head_dim=HEAD_DIM,
    )
    _fill_cache(mesh_device, kv_cache, chunk=chunk, capacity=capacity, n_kv=NKV, n_chunks=n_chunks)

    table = build_kv_chunk_address_table(
        mesh_device=mesh_device,
        kv_cache=kv_cache,
        seq_len=capacity,
        num_layers=num_layers,
        mesh_shape=(rows, cols),
        sp_axis=SPEC.sp_axis,
        num_users=1,
        chunk_size=chunk,
        num_kv_heads=NKV,
        head_dim=HEAD_DIM,
    )
    assert table.num_configs() == 2 * NKV, f"expected {2 * NKV} configs (k+v per head), got {table.num_configs()}"

    chunk_shape = [1, 1, NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK, HEAD_DIM]
    seq_local = capacity // sp
    # shard row -> global position, and its inverse.
    positions = blockcyclic_positions(sp, chunk, capacity)
    shard_row_of = torch.empty(capacity, dtype=torch.long)
    shard_row_of[positions] = torch.arange(capacity)
    compared = 0
    for config_id in range(table.num_configs()):
        head = config_id % NKV
        cache_tensor = kv_cache.k if config_id < NKV else kv_cache.v
        device_rows = _device_cache_rows(cache_tensor, mesh_device, slot=0, head=head)
        for position in range(0, capacity, NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK):
            raw = table.read_device_chunk(layer=0, position=position, slot=0, config_id=config_id)
            chunk_tt = ttnn.experimental.disaggregation.tensor_from_bfp8_bytes(raw, chunk_shape)
            got = ttnn.to_torch(chunk_tt).reshape(NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK, HEAD_DIM)
            # The table keys on the GLOBAL token position, but the bytes at that address are a RAW
            # cache row, and the cache is block-cyclic: global position p lives at shard row
            # shard_row_of[p] (the inverse of blockcyclic_positions). Comparing at `position`
            # directly would only work while the layout is the identity — which it is for a
            # single-chunk cache and is NOT here (capacity is 2 chunks), so this indirection is the
            # part of the test that actually exercises the layout.
            #
            # A 32-token bank block always lies inside one (slab, SP row), so those 32 global
            # positions are 32 CONSECUTIVE shard rows and one slice suffices.
            first = int(shard_row_of[position])
            expected = device_rows[first : first + NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK]
            assert_equal(got.to(expected.dtype), expected)
            compared += 1
    logger.info(
        f"KV chunk table: {compared} chunks read back bit-exactly across {table.num_configs()} configs "
        f"(capacity={capacity}, seq_local={seq_local}, chunk={chunk})"
    )
    assert compared == table.num_configs() * (capacity // NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK)


@parametrize_mesh()
@pytest.mark.timeout(0)
def test_kv_chunk_table_survives_a_protobuf_round_trip(mesh_device, device_params, tmp_path, reset_seeds):
    """Serialize, re-import, and confirm every lookup is unchanged.

    This is the zero-padded-config-name guard. With 16 configs, unpadded names (``"0".."15"``) come
    back from a ``std::map`` in lexicographic order — ``"10"`` before ``"2"`` — so config_id 2 would
    silently resolve to a different head's addresses. Producer read-backs are by integer config_id,
    so the failure would show up as a KV PCC mismatch a long way from here.
    """
    rows, cols = tuple(mesh_device.shape)
    sp = rows
    chunk = ttnn.TILE_SIZE * sp * 2
    capacity = chunk * 2
    kv_cache = allocate_kv_cache(
        mesh_device, num_layers=1, max_seq_len=capacity, sp_axis=SPEC.sp_axis, num_users=1, head_dim=HEAD_DIM
    )
    path = str(tmp_path / "kv_table.pb")
    build_and_serialize_kv_chunk_table(
        mesh_device=mesh_device,
        kv_cache=kv_cache,
        seq_len=capacity,
        num_layers=1,
        mesh_shape=(rows, cols),
        sp_axis=SPEC.sp_axis,
        num_users=1,
        chunk_size=chunk,
        num_kv_heads=NKV,
        head_dim=HEAD_DIM,
        path=path,
    )
    original = build_kv_chunk_address_table(
        mesh_device=mesh_device,
        kv_cache=kv_cache,
        seq_len=capacity,
        num_layers=1,
        mesh_shape=(rows, cols),
        sp_axis=SPEC.sp_axis,
        num_users=1,
        chunk_size=chunk,
        num_kv_heads=NKV,
        head_dim=HEAD_DIM,
    )
    imported = ttnn.experimental.disaggregation.import_from_protobuf_file(path)

    assert imported.num_configs() == original.num_configs() == 2 * NKV
    for config_id in range(original.num_configs()):
        assert imported.config_name(config_id) == original.config_name(config_id), (
            f"config_id {config_id} name changed across the round trip: "
            f"{original.config_name(config_id)!r} -> {imported.config_name(config_id)!r} — the "
            f"zero-padded naming is what prevents this"
        )
        for position in range(0, capacity, NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK):
            want = original.lookup(0, position, 0, config_id)
            got = imported.lookup(0, position, 0, config_id)
            assert (
                got.noc_addr == want.noc_addr
            ), f"config {config_id} position {position}: address {got.noc_addr:#x} != {want.noc_addr:#x}"
            assert got.size_bytes == want.size_bytes
    logger.info(f"protobuf round trip preserved every lookup across {original.num_configs()} configs")


@parametrize_mesh()
def test_table_rejects_a_head_count_that_does_not_match_tp(mesh_device, device_params):
    """The table maps head ``h`` to TP column ``h``, so a mismatch must fail LOUDLY.

    Without this the builder would happily emit addresses for a column that holds a different head,
    and the only symptom would be a wrong-looking KV read on the destination.
    """
    rows, cols = tuple(mesh_device.shape)
    sp = rows
    chunk = ttnn.TILE_SIZE * sp
    kv_cache = allocate_kv_cache(
        mesh_device, num_layers=1, max_seq_len=chunk, sp_axis=SPEC.sp_axis, num_users=1, head_dim=HEAD_DIM
    )
    with pytest.raises(AssertionError, match="TP column"):
        build_kv_chunk_address_table(
            mesh_device=mesh_device,
            kv_cache=kv_cache,
            seq_len=chunk,
            num_layers=1,
            mesh_shape=(rows, cols),
            sp_axis=SPEC.sp_axis,
            num_users=1,
            chunk_size=chunk,
            num_kv_heads=NKV - 1,  # deliberately wrong
            head_dim=HEAD_DIM,
        )
