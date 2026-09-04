# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""`G-KV-TABLE` — the KV chunk address table, proved by reading DRAM back through it (P10.4, `R-030`).

The address table is what a **device-less** reader — the migration worker, and the producer's
`PREFILL_PRODUCER_CHECK_PCC` gate — uses to find the bytes prefill wrote. Nothing downstream can
detect a wrong one: a structurally valid table with the wrong addresses migrates the wrong DRAM
ranges and surfaces as a corrupted decode long after prefill. So this file does not check that the
table *exists* or that it *parses*; it writes a known pattern into a real cache on the real `(4,8)`
galaxy and then reads every chunk back **through the table over UMD**, exactly as the producer does,
and requires bit-exact agreement with the runtime's own `gather_layer`.

Four properties, each of which a plausible table bug breaks and nothing else here would:

1. **Round trip.** The table is serialized to protobuf and re-imported before use, so the
   zero-padded config naming (`std::map` lexicographic order == numeric `config_id`) is exercised
   rather than asserted in-process. Unpadded names put head 10 where head 2 should be, and *only*
   for `N > 10` — which this model, at 16 configs, is.
2. **Head -> config -> chip.** Config `h` must resolve to K head `h` and config `N+h` to V head `h`,
   on TP column `h`. Cross-wiring heads is invisible to any single-head check.
3. **Position -> address.** The block-cyclic sequence layout means global position `p` lives on SP
   row `(p // chunk_local) % ...`; the table has to invert `update_padded_kv_cache`'s writer exactly.
   The pattern written below is unique per (slot, layer, head, position), so any mix-up fails.
4. **K and V are distinct buffers.** They have different base addresses, and a table that pointed
   both halves at one of them would still PCC well against a golden whose K and V are similar.

Negative control: reading the same slot with the config ids rotated by one must **fail** the
comparison, so "the table maps head h to config h" is a claim this test can refute.

**Mesh.** The full `(4,8)` galaxy (`DEC-080`: a top-level partial mesh cannot bring the fabric up
here), `Topology.Ring` (`DEC-081`). `sp = 4`, so the block-cyclic layout is non-trivial — the point.
Only ONE mesh is used, so `R-032`'s overlapping-submesh hang cannot arise.

**Input distribution:** a deterministic integer-valued pattern, chosen so bf8_b stores it exactly and
the comparison can be `rtol = atol = 0` rather than a PCC. Nothing here is a numerical claim; it is
an addressing claim, and an addressing claim deserves an exact test (Appendix E.1).
**Reference dtype policy:** not applicable — both sides of every comparison are the same device bytes,
read two different ways.

Run::

    export TT_MESH_GRAPH_DESC_PATH=$TT_METAL_HOME/tt_metal/fabric/mesh_graph_descriptors/\
single_bh_galaxy_torus_xy_graph_descriptor.textproto
    pytest models/demos/llama31_8b_d_p/tests/unit/test_kv_chunk_table.py -x -q
"""

from __future__ import annotations

import json

import torch

import ttnn
from models.demos.common.prefill.runners.migration import serialize_device_map
from models.demos.llama31_8b_d_p.tests.test_factory import TestFactory, parametrize_galaxy_submeshes
from models.demos.llama31_8b_d_p.tt.attention.kv_cache import (
    NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK,
    allocate_kv_cache,
    write_kv_chunk,
)
from models.demos.llama31_8b_d_p.tt.runners.kv_chunk_table import (
    build_and_serialize_kv_chunk_table,
    chunk_size_bytes,
    stable_config_name,
)

# Small on purpose: this is an addressing test, and every extra layer/user multiplies the UMD read
# count. 2 users x 2 layers x 4 chunks x 8 heads x 2 (K,V) = 512 blocks of 32 tokens.
NUM_LAYERS = 2
NUM_USERS = 2
CHUNK = 128  # must satisfy CHUNK % (TILE_SIZE * sp) == 0 -> 128 % 128 == 0 at sp=4
MAX_SEQ_LEN = 4 * CHUNK  # 512; a multiple of CHUNK so the block-cyclic period tiles the cache
HEAD_DIM = 128
N_KV = 8


def _pattern(slot, layer, head, positions):
    """`[len(positions), HEAD_DIM]` labelling every axis a wrong table entry could confuse.

    Lane blocks, 32 wide each: ``[0:32]`` = ``pos % 128``, ``[32:64]`` = ``pos // 128``,
    ``[64:96]`` = head, ``[96:128]`` = ``slot * NUM_LAYERS + layer``. So a read that lands on the
    wrong head, the wrong position, the wrong slot **or** the wrong layer differs in a block that
    names which one — the failure identifies the bug instead of merely reporting inequality.

    Every value is an integer in ``[0, 127]``, which ``bfloat8_b`` stores **exactly**: its decoder is
    ``magnitude(7 bits) * 2 ** (exponent - 133)``, and the exponent is shared across 16 consecutive
    *lanes* (`prefill_producer._decode_bfp8_chunk`), which these 32-lane constant blocks never cross.
    That is what lets the comparison be `torch.equal` rather than a PCC: this is an addressing claim,
    not a numerical one.
    """
    n = len(positions)
    out = torch.empty(n, HEAD_DIM, dtype=torch.float32)
    out[:, 0:32] = (positions % 128).float().unsqueeze(-1)
    out[:, 32:64] = (positions // 128).float().unsqueeze(-1)
    out[:, 64:96] = float(head)
    out[:, 96:128] = float(slot * NUM_LAYERS + layer)
    assert out.max() <= 127, "bfloat8_b is exact only for integer magnitudes <= 127"
    return out


def _write_pattern(dev, kv, *, sp, cols):
    """Fill every (slot, layer) with the pattern, chunk by chunk, through the real writer.

    Uses the **model's own** mesh mapper — sequence sharded on the SP rows (dim 2), heads on the TP
    columns (dim 1) — so the on-device arrangement is the one `Model.prefill_forward` produces, not
    one this test invented.
    """
    chunk_local = CHUNK // sp
    for slot in range(NUM_USERS):
        for layer in range(NUM_LAYERS):
            for chunk_idx in range(MAX_SEQ_LEN // CHUNK):
                start = chunk_idx * CHUNK
                positions = torch.arange(start, start + CHUNK)
                sent = torch.stack([_pattern(slot, layer, h, positions) for h in range(cols)], dim=0)
                tt_chunk = ttnn.from_torch(
                    sent.reshape(1, cols, CHUNK, HEAD_DIM),
                    device=dev,
                    dtype=ttnn.bfloat8_b,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ShardTensor2dMesh(dev, mesh_shape=(sp, cols), dims=(2, 1)),
                )
                per_chip = tuple(ttnn.get_device_tensors(tt_chunk)[0].shape)
                assert per_chip == (1, 1, chunk_local, HEAD_DIM), (
                    f"the mesh mapper gave each chip {per_chip}, not one KV head's SP shard; "
                    f"write_kv_chunk would then write the wrong rows"
                )
                write_kv_chunk(kv, tt_chunk, tt_chunk, slot_idx=slot, layer_idx=layer, kv_actual=start, sp_axis=0)
                tt_chunk.deallocate(True)


def _device_map(mesh_device, tmp_path) -> dict:
    """The same JSON sidecar the runner publishes, parsed the way the producer parses it."""
    path = str(tmp_path / "device_map.json")
    serialize_device_map(mesh_device, path)
    with open(path) as fh:
        return {tuple(int(x) for x in key.split(":")): int(uid) for key, uid in json.load(fh).items()}


def _read_through_table(table, device_map, *, config_id, layer, slot, n_tokens, head_dim):
    """Read `[0, n_tokens)` of one (config, layer, slot) back over UMD, in natural token order.

    Deliberately the producer's own path — `table.lookup` -> `_resolve_unique_id` -> `read_dram_umd`
    -> `_decode_bfp8_chunk` — rather than a local re-implementation, so a decode or lookup convention
    that the producer gets wrong cannot pass here.
    """
    from models.demos.common.prefill.runners.prefill_producer import _decode_bfp8_chunk, _resolve_unique_id

    rows = []
    for pos in range(0, n_tokens, NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK):
        loc = table.lookup(layer, pos, slot, config_id)
        unique_id = _resolve_unique_id(table.get_device_group(loc.device_group_index).fabric_node_ids, device_map)
        raw = ttnn.experimental.disaggregation.read_dram_umd(unique_id, loc.noc_addr, loc.size_bytes)
        rows.append(_decode_bfp8_chunk(raw, head_dim))
    return torch.cat(rows, dim=0)[:n_tokens]


@parametrize_galaxy_submeshes([(4, 8)])
def test_kv_chunk_table_addresses_the_bytes_prefill_wrote(mesh_device, submesh_shape, tmp_path):
    objs = TestFactory.setup_submesh(mesh_device, submesh_shape)
    dev = objs["mesh_device"]
    sp, cols = submesh_shape
    assert cols == N_KV, "R-027: one KV head per chip requires TP == num_key_value_heads == 8"

    kv = allocate_kv_cache(
        dev,
        num_layers=NUM_LAYERS,
        max_seq_len=MAX_SEQ_LEN,
        sp_axis=0,
        num_users=NUM_USERS,
        head_dim=HEAD_DIM,
        cache_dtype=ttnn.bfloat8_b,
    )
    _write_pattern(dev, kv, sp=sp, cols=cols)
    ttnn.synchronize_device(dev)

    table_path = str(tmp_path / "kv_chunk_table.pb")
    build_and_serialize_kv_chunk_table(
        mesh_device=dev,
        kv_cache=kv,
        seq_len=MAX_SEQ_LEN,
        num_layers=NUM_LAYERS,
        mesh_shape=submesh_shape,
        sp_axis=0,
        num_users=NUM_USERS,
        chunk_size=CHUNK,
        num_kv_heads=N_KV,
        head_dim=HEAD_DIM,
        path=table_path,
    )

    # Property 1: survive the protobuf round trip, config ids intact.
    table = ttnn.experimental.disaggregation.import_from_protobuf_file(table_path)
    assert table.num_configs() == 2 * N_KV, table.num_configs()
    for i in range(2 * N_KV):
        assert table.config_name(i) == stable_config_name(i, 2 * N_KV), (
            f"config {i} re-imported as {table.config_name(i)!r}: the protobuf std::map reordered "
            f"the configs, so every lookup above id 9 reads another head's bytes"
        )
    cfg0 = table.config(0)
    assert cfg0.num_layers == NUM_LAYERS
    assert cfg0.num_slots == NUM_USERS
    assert cfg0.max_sequence_length == MAX_SEQ_LEN
    assert cfg0.chunk_n_tokens == NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK
    assert cfg0.chunk_size_bytes == chunk_size_bytes(ttnn.bfloat8_b, HEAD_DIM) == 4 * 1088

    device_map = _device_map(dev, tmp_path)
    assert len(device_map) == sp * cols

    # Properties 2 + 3: every (slot, layer, head) reads back exactly what was written there.
    mismatches = []
    for slot in range(NUM_USERS):
        for layer in range(NUM_LAYERS):
            for head in range(N_KV):
                expected = _pattern(slot, layer, head, torch.arange(MAX_SEQ_LEN))
                for label, config_id in (("k", head), ("v", N_KV + head)):
                    got = _read_through_table(
                        table,
                        device_map,
                        config_id=config_id,
                        layer=layer,
                        slot=slot,
                        n_tokens=MAX_SEQ_LEN,
                        head_dim=HEAD_DIM,
                    )
                    if not torch.equal(got, expected):
                        bad = int((got[:, 0] != expected[:, 0]).sum())
                        mismatches.append(f"{label} slot={slot} layer={layer} head={head}: {bad}/{MAX_SEQ_LEN} rows")
    assert not mismatches, "table read-back != what was written:\n  " + "\n  ".join(mismatches[:20])

    # Property 4: K and V really are two buffers.
    assert kv.k.buffer_address() != kv.v.buffer_address()
    k_addr = table.lookup(0, 0, 0, 0).noc_addr & 0xFFFFFFFF
    v_addr = table.lookup(0, 0, 0, N_KV).noc_addr & 0xFFFFFFFF
    assert k_addr != v_addr, "config 0 (k head 0) and config N (v head 0) resolve to the same address"

    # Negative control: rotate the config ids by one. Head h's golden read through head h+1's config
    # must NOT match, or this test is not measuring the head->config map at all.
    control = _read_through_table(
        table, device_map, config_id=1, layer=0, slot=0, n_tokens=MAX_SEQ_LEN, head_dim=HEAD_DIM
    )
    expected_head0 = _pattern(0, 0, 0, torch.arange(MAX_SEQ_LEN))
    assert not torch.equal(control, expected_head0), (
        "reading head 0's data through config 1 SUCCEEDED — the configs are not head-specific, so "
        "the positive assertions above prove nothing about the head mapping"
    )


@parametrize_galaxy_submeshes([(4, 8)])
def test_runtime_hook_refuses_a_multi_rank_merge(mesh_device, submesh_shape, tmp_path, expect_error):
    """`R-040`: the merged multi-rank table is not implemented, so it must raise, not publish.

    Built with `num_layers=1` and no weights: this exercises the hook's argument checking, which runs
    before anything touches the cache.
    """
    from models.demos.llama31_8b_d_p.tt.tt_prefill_runtime import TtPrefillRuntime

    objs = TestFactory.setup_submesh(mesh_device, submesh_shape)
    dev = objs["mesh_device"]
    runtime = TtPrefillRuntime.__new__(TtPrefillRuntime)  # no model build: only the hook is under test
    runtime.mesh_device = dev
    runtime.hf_config = objs["hf_config"]
    from models.demos.llama31_8b_d_p.tt.tt_prefill_runtime import TtPrefillRuntimeConfig

    runtime.config = TtPrefillRuntimeConfig(
        num_layers=NUM_LAYERS, max_seq_len=MAX_SEQ_LEN, default_chunk_size=CHUNK, num_users=NUM_USERS
    )
    runtime.chunk_sizes = (CHUNK,)
    runtime.kv_cache = None

    with expect_error(NotImplementedError, "R-040"):
        runtime.build_kv_chunk_table(None, str(tmp_path / "t.pb"), first_layer_idx=16)
    with expect_error(NotImplementedError, "R-040"):
        runtime.build_kv_chunk_table(None, str(tmp_path / "t.pb"), num_my_layers=NUM_LAYERS + 1)
    with expect_error(NotImplementedError, "R-040"):
        runtime.build_kv_chunk_table(None, str(tmp_path / "t.pb"), stage_layout=[{"rank": 0}, {"rank": 1}])
