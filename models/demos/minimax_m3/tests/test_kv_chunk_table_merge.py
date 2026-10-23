# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Device-free checks of the M3 multi-stage (pipeline-parallel) KV chunk table merge.

Drives ``build_and_serialize_kv_chunk_table`` with synthetic 2-stage ``stage_layouts`` (one gathered
layout per cache — k, v, index_k) and a stub kv_cache (the merged path reads only dtypes/shapes from it —
addresses, fabric nodes, hosts and bank counts all come from the gathered layouts), then asserts the
table's addressing against the layouts: per-(stage, cache) base addresses at global layer indices,
single-member per-head device groups vs the full-row index_k replica group, and the per-(config, stage,
row) bank-walk restart, and the index_k rows limited to the MSA layers. The ``dedup_table`` cases rebuild
it with index_k TP-deduped (``index_k_tp_axis``): index_k stays one config but every chip owns a distinct
stripe, addressed by a single-chip group and its own bank walk, while K / V are unchanged.

The strided-row cases compare every chunk of the STRIDED_ROWS table with a replay of each chip's page walk
across geometries (bank counts, period lengths), then check the 1M-token single-slot table the 16-chip
prefill partition publishes, and that it pairs with a two-slot decode-side table of a different depth.
"""

import os
import zlib
from types import SimpleNamespace

import pytest

import ttnn
from models.demos.common.prefill.runners.migration import validate_stage_layout_contiguous
from models.demos.minimax_m3.tt.attention.kv_cache import BH_NUM_DRAM_BANKS, NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK
from models.demos.minimax_m3.tt.runners.kv_chunk_table import _chunk_size_bytes, build_and_serialize_kv_chunk_table

SP = 2
COLS = 4  # == num_kv_heads (the builder asserts the 1:1 head->column map)
CHUNK_SIZE = 64  # tokens_per_chunk_local = 32 == NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK
SEQ_LEN = 128
NUM_USERS = 2
HEAD_DIM = 128
STAGE_COUNTS = (2, 3)  # layers per stage; global total 5
NUM_BANKS = 8


@pytest.mark.parametrize("alias", [None, "", "   ", "galaxy-prefill", " galaxy-decode "])
def test_colocated_table_host_identity(monkeypatch, alias):
    import zlib

    from models.demos.common.prefill.runners import migration

    monkeypatch.setattr(migration.socket, "gethostname", lambda: "physical-galaxy")
    if alias is None:
        monkeypatch.delenv("TT_MIGRATION_TABLE_HOST", raising=False)
    else:
        monkeypatch.setenv("TT_MIGRATION_TABLE_HOST", alias)
    name = (alias or "").strip() or "physical-galaxy"
    assert migration._host_tag_int() == zlib.crc32(name.encode()) & 0x7FFFFFFF


def _fnids(mesh_id):
    return [[ttnn.FabricNodeId(ttnn.MeshId(mesh_id), r * COLS + c) for c in range(COLS)] for r in range(SP)]


def _base(stage_idx: int, cache_idx: int) -> int:
    """Distinct per-(stage, cache) base so a wrong (stage, cache) pick is visible in the noc_addr."""
    return 0x10000 * (stage_idx + 1) + cache_idx * 0x4000


def _stage_layouts():
    """One gathered layout per cache (k, v, index_k — the kv_migration_stages order), each spanning the
    two pipeline stages. All three share fnids/banks/ranges; only the base address differs per cache."""
    layouts = []
    for cache_idx in range(3):
        stages = []
        first = 0
        for i, count in enumerate(STAGE_COUNTS):
            stages.append(
                {
                    "rank": i,
                    "first_layer": first,
                    "count": count,
                    "base_addr": _base(i, cache_idx),
                    "num_banks": NUM_BANKS,
                    "host_tag": 0x1000 + i,
                    "fnids": _fnids(i),
                }
            )
            first += count
        layouts.append(stages)
    return layouts


def _stub_cache(num_layers, seq_len=SEQ_LEN, index_k_tp_axis=None):
    def t(dtype, rows):
        return SimpleNamespace(shape=(NUM_USERS * num_layers, 1, rows, HEAD_DIM), dtype=dtype)

    ik_rows = seq_len if index_k_tp_axis is None else seq_len // COLS
    # Migration requires a bf8 index_k (the decode peer's dtype); the builder rejects bf16.
    return SimpleNamespace(
        k=t(ttnn.bfloat8_b, seq_len),
        v=t(ttnn.bfloat8_b, seq_len),
        index_k=t(ttnn.bfloat8_b, ik_rows),
        index_k_tp_axis=index_k_tp_axis,
    )


def _build(
    tmp_path, stage_layouts, *, index_k_layers=None, seq_len=SEQ_LEN, chunk_size=CHUNK_SIZE, index_k_tp_axis=None
):
    path = os.path.join(str(tmp_path), "m3_merge_table.pb")
    # num_layers is THIS rank's stage count (rank 0 builds), used only for the local shape assert.
    return build_and_serialize_kv_chunk_table(
        mesh_device=None,
        kv_cache=_stub_cache(STAGE_COUNTS[0], seq_len=seq_len, index_k_tp_axis=index_k_tp_axis),
        seq_len=seq_len,
        num_layers=STAGE_COUNTS[0],
        mesh_shape=(SP, COLS),
        sp_axis=0,
        num_users=NUM_USERS,
        chunk_size=chunk_size,
        num_kv_heads=COLS,
        head_dim=HEAD_DIM,
        path=path,
        stage_layouts=stage_layouts,
        index_k_layers=index_k_layers,
    )


def test_single_rank_uses_kvm_hashed_host_identity(tmp_path, monkeypatch):
    import zlib

    name = "galaxy-prefill"
    monkeypatch.setenv("TT_MIGRATION_TABLE_HOST", name)
    cache = _stub_cache(STAGE_COUNTS[0])
    for i, tensor in enumerate((cache.k, cache.v, cache.index_k)):
        tensor.buffer_address = lambda i=i: _base(0, i)
    nodes = _fnids(0)
    mesh = SimpleNamespace(get_fabric_node_id=lambda coord: nodes[coord[0]][coord[1]])
    path = tmp_path / "single.pb"
    build_and_serialize_kv_chunk_table(
        mesh_device=mesh,
        kv_cache=cache,
        seq_len=SEQ_LEN,
        num_layers=STAGE_COUNTS[0],
        mesh_shape=(SP, COLS),
        sp_axis=0,
        num_users=NUM_USERS,
        chunk_size=CHUNK_SIZE,
        num_kv_heads=COLS,
        head_dim=HEAD_DIM,
        path=str(path),
    )
    # KVM hashes KV_MANAGER_TABLE_HOST into this string before selecting its local table.
    expected = f"host-{zlib.crc32(name.encode()) & 0x7FFFFFFF:08x}"
    assert expected.encode() in path.read_bytes()
    assert name.encode() not in path.read_bytes()


@pytest.fixture(scope="module")
def merged_table(tmp_path_factory):
    path = _build(tmp_path_factory.mktemp("m3_merge"), _stage_layouts())
    return ttnn.experimental.disaggregation.import_from_protobuf_file(path)


def test_configs_and_global_layer_extent(merged_table):
    assert merged_table.num_configs() == 2 * COLS + 1
    total = sum(STAGE_COUNTS)
    for cfg_id in range(merged_table.num_configs()):
        assert merged_table.config(cfg_id).num_layers == total
    assert merged_table.config(2 * COLS).chunk_size_bytes == _chunk_size_bytes(ttnn.bfloat8_b, HEAD_DIM)
    # The KV manager pairs prefill and decode configs by name: blaze names them by zero-padded position.
    assert [merged_table.config_name(i) for i in range(merged_table.num_configs())] == [
        f"{i:02d}" for i in range(2 * COLS + 1)
    ]


def test_stage_addressing_and_bank_walk(merged_table):
    stages = _stage_layouts()[0]  # k's layout; fnids/ranges identical across the three caches
    k_bytes = _chunk_size_bytes(ttnn.bfloat8_b, HEAD_DIM)
    for stage in stages:
        for local_layer in (0, stage["count"] - 1):
            global_layer = stage["first_layer"] + local_layer
            # Row 0's first chunk of (slot 0, local layer): the walk runs slot -> local layer -> chunk,
            # 32 tokens per bank step, restarting per (config, stage, row) — so this chunk's global
            # index within the walk is local_layer * (SEQ_LEN // CHUNK_SIZE) * (CHUNK_SIZE // SP / 32).
            chunks_before = (
                local_layer * (SEQ_LEN // CHUNK_SIZE) * (CHUNK_SIZE // SP // NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK)
            )
            loc = merged_table.lookup(global_layer, 0, 0, 0)  # config 0 = k_h0, position 0 lives in row 0
            bank = chunks_before % NUM_BANKS
            offset = (chunks_before // NUM_BANKS) * k_bytes
            assert loc.noc_addr == (bank << 32) | (stage["base_addr"] + offset), (
                f"stage {stage['rank']} global layer {global_layer}: k_h0 chunk 0 landed at "
                f"{loc.noc_addr:#x}, expected bank {bank} offset {offset:#x} off base "
                f"{stage['base_addr']:#x}"
            )
            assert loc.size_bytes == k_bytes


def test_v_and_index_k_use_their_own_base(merged_table):
    stage_idx = 1
    gl = _stage_layouts()[0][stage_idx]["first_layer"]
    v_loc = merged_table.lookup(gl, 0, 0, COLS)  # config COLS = v_h0
    assert (v_loc.noc_addr & 0xFFFFFFFF) == _base(stage_idx, 1)
    ik_loc = merged_table.lookup(gl, 0, 0, 2 * COLS)
    assert (ik_loc.noc_addr & 0xFFFFFFFF) == _base(stage_idx, 2)


def test_device_groups_per_head_and_replica(merged_table):
    stages = _stage_layouts()[0]
    for stage in stages:
        gl = stage["first_layer"]
        for h in range(COLS):
            loc = merged_table.lookup(gl, 0, 0, h)  # row 0 owns position 0
            group = merged_table.get_device_group(loc.device_group_index).fabric_node_ids
            assert [(int(f.mesh_id), int(f.chip_id)) for f in group] == [
                (int(stage["fnids"][0][h].mesh_id), int(stage["fnids"][0][h].chip_id))
            ], f"k_h{h} of stage {stage['rank']} must be a single-member group on column {h}"
        ik = merged_table.lookup(gl, 0, 0, 2 * COLS)
        group = merged_table.get_device_group(ik.device_group_index).fabric_node_ids
        assert sorted((int(f.mesh_id), int(f.chip_id)) for f in group) == sorted(
            (int(f.mesh_id), int(f.chip_id)) for f in stage["fnids"][0]
        ), f"index_k of stage {stage['rank']} must replicate across the full row"


def test_row_sharding_positions(merged_table):
    # Position CHUNK_SIZE//SP (= row 1's first token of chunk 0) must resolve to row 1's chips.
    stage = _stage_layouts()[0][0]
    loc = merged_table.lookup(0, CHUNK_SIZE // SP, 0, 0)
    group = merged_table.get_device_group(loc.device_group_index).fabric_node_ids
    assert (int(group[0].mesh_id), int(group[0].chip_id)) == (
        int(stage["fnids"][1][0].mesh_id),
        int(stage["fnids"][1][0].chip_id),
    )


def test_non_contiguous_stage_layout_rejected(tmp_path, expect_error):
    layouts = _stage_layouts()
    for layout in layouts:
        layout[1]["first_layer"] += 1  # gap after stage 0, in every cache's layout
    with expect_error(RuntimeError, "not contiguous"):
        _build(tmp_path, layouts)
    with expect_error(RuntimeError, "not contiguous"):
        validate_stage_layout_contiguous(layouts[0])


def test_mismatched_cache_ranges_rejected(tmp_path, expect_error):
    # M3's caches share one layer-index space; a v layout whose ranges differ from k's must be refused.
    layouts = _stage_layouts()
    layouts[1][0]["count"] += 1
    layouts[1][1]["first_layer"] += 1
    with expect_error(RuntimeError, "share one layer-index space"):
        _build(tmp_path, layouts)


def test_index_k_rows_limited_to_msa_layers(tmp_path):
    # Layers 0 and 2 (one per stage, including each stage's first) carry no index_k. Their rows must be
    # empty, while every other row keeps the address of the unfiltered table: the dense layers' regions are
    # still walked, so the bank walk of later layers and slots doesn't shift.
    total = sum(STAGE_COUNTS)
    msa = {1, 3, 4}
    imp = ttnn.experimental.disaggregation.import_from_protobuf_file
    (tmp_path / "full").mkdir()
    (tmp_path / "msa").mkdir()
    full = imp(_build(tmp_path / "full", _stage_layouts()))
    filtered = imp(_build(tmp_path / "msa", _stage_layouts(), index_k_layers=msa))
    for cfg_id in range(2 * COLS + 1):
        for slot in range(NUM_USERS):
            for layer in range(total):
                for pos in range(0, SEQ_LEN, NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK):
                    want = full.lookup(layer, pos, slot, cfg_id)
                    got = filtered.lookup(layer, pos, slot, cfg_id)
                    if cfg_id == 2 * COLS and layer not in msa:
                        assert got.size_bytes == 0, f"index_k row published on dense layer {layer}"
                        continue
                    assert (got.noc_addr, got.size_bytes, got.device_group_index) == (
                        want.noc_addr,
                        want.size_bytes,
                        want.device_group_index,
                    ), f"config {cfg_id} slot {slot} layer {layer} pos {pos} moved"


# --- index_k TP dedup -----------------------------------------------------------------------------------------
# Each SP row's chunk_local rows split into COLS stripes of whole 32-token chunks: chunk_local = 128, stripe = 32.
DEDUP_CHUNK_SIZE = SP * COLS * NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK  # 256
DEDUP_SEQ_LEN = 2 * DEDUP_CHUNK_SIZE  # 512: two slabs, so the walk crosses a slab boundary
DEDUP_CHUNK_LOCAL = DEDUP_CHUNK_SIZE // SP
DEDUP_STRIPE = DEDUP_CHUNK_LOCAL // COLS


@pytest.fixture(scope="module")
def dedup_table(tmp_path_factory):
    path = _build(
        tmp_path_factory.mktemp("m3_merge_dedup"),
        _stage_layouts(),
        seq_len=DEDUP_SEQ_LEN,
        chunk_size=DEDUP_CHUNK_SIZE,
        index_k_tp_axis=1,
    )
    return ttnn.experimental.disaggregation.import_from_protobuf_file(path)


def _ids(group):
    return [(int(f.mesh_id), int(f.chip_id)) for f in group]


def _dedup_owner(position):
    """(row, col, local row) of the chip holding ``position`` in a TP-deduped cache."""
    n, off = divmod(position, DEDUP_CHUNK_SIZE)
    row, in_row = divmod(off, DEDUP_CHUNK_LOCAL)
    col, i = divmod(in_row, DEDUP_STRIPE)
    return row, col, n * DEDUP_STRIPE + i


def test_dedup_keeps_config_list(dedup_table):
    # Same 2N+1 configs in the same order: the src<->dst contract does not change.
    assert dedup_table.num_configs() == 2 * COLS + 1
    assert dedup_table.config(2 * COLS).chunk_size_bytes == _chunk_size_bytes(ttnn.bfloat8_b, HEAD_DIM)


def test_dedup_index_k_single_chip_groups_and_addresses(dedup_table):
    """Every 32-token index_k chunk resolves to the ONE chip owning it, at that chip's own ND-shard page
    (ROUND_ROBIN_1D over its [B, 1, rows, D] pages, B = slot * stage_count + local_layer)."""
    ik_bytes = _chunk_size_bytes(ttnn.bfloat8_b, HEAD_DIM)
    rows_per_chip = DEDUP_SEQ_LEN // (SP * COLS)
    for stage_idx, stage in enumerate(_stage_layouts()[2]):
        for slot in range(NUM_USERS):
            for local_layer in range(stage["count"]):
                batch = slot * stage["count"] + local_layer
                for position in range(0, DEDUP_SEQ_LEN, NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK):
                    row, col, local_row = _dedup_owner(position)
                    loc = dedup_table.lookup(stage["first_layer"] + local_layer, position, slot, 2 * COLS)
                    assert _ids(dedup_table.get_device_group(loc.device_group_index).fabric_node_ids) == _ids(
                        [stage["fnids"][row][col]]
                    ), f"stage {stage_idx} pos {position}: index_k must live on chip ({row}, {col}) alone"
                    page = (batch * rows_per_chip + local_row) // NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK
                    bank, offset = page % NUM_BANKS, (page // NUM_BANKS) * ik_bytes
                    assert loc.noc_addr == (bank << 32) | (_base(stage_idx, 2) + offset), (
                        f"stage {stage_idx} slot {slot} layer {local_layer} pos {position}: {loc.noc_addr:#x}, "
                        f"expected bank {bank} offset {offset:#x}"
                    )


def test_dedup_leaves_k_v_layout(dedup_table):
    # K / V stay TP-head-sharded: whole chunk_local rows per SP row, a single-member group on the head's column.
    stage = _stage_layouts()[0][0]
    for position in range(0, DEDUP_CHUNK_SIZE, NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK):
        row = position // DEDUP_CHUNK_LOCAL
        for h in range(COLS):
            for cfg in (h, COLS + h):
                loc = dedup_table.lookup(0, position, 0, cfg)
                assert _ids(dedup_table.get_device_group(loc.device_group_index).fabric_node_ids) == _ids(
                    [stage["fnids"][row][h]]
                )


def test_dedup_index_k_rows_limited_to_msa_layers(dedup_table, tmp_path):
    # The MSA-layer filter composes with the TP dedup: dense layers 0 and 2 get no index_k rows on any
    # stripe, and every other row (K / V everywhere, index_k on MSA layers) keeps its unfiltered address
    # and single-chip group, since each chip's own bank walk still steps over the dense layers' regions.
    total = sum(STAGE_COUNTS)
    msa = {1, 3, 4}
    filtered = ttnn.experimental.disaggregation.import_from_protobuf_file(
        _build(
            tmp_path,
            _stage_layouts(),
            index_k_layers=msa,
            seq_len=DEDUP_SEQ_LEN,
            chunk_size=DEDUP_CHUNK_SIZE,
            index_k_tp_axis=1,
        )
    )
    for cfg_id in range(2 * COLS + 1):
        for slot in range(NUM_USERS):
            for layer in range(total):
                for pos in range(0, DEDUP_SEQ_LEN, NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK):
                    want = dedup_table.lookup(layer, pos, slot, cfg_id)
                    got = filtered.lookup(layer, pos, slot, cfg_id)
                    if cfg_id == 2 * COLS and layer not in msa:
                        assert got.size_bytes == 0, f"index_k row published on dense layer {layer} pos {pos}"
                        continue
                    assert (got.noc_addr, got.size_bytes) == (
                        want.noc_addr,
                        want.size_bytes,
                    ), f"config {cfg_id} slot {slot} layer {layer} pos {pos} moved"
                    assert _ids(filtered.get_device_group(got.device_group_index).fabric_node_ids) == _ids(
                        dedup_table.get_device_group(want.device_group_index).fabric_node_ids
                    ), f"config {cfg_id} slot {slot} layer {layer} pos {pos} changed device group"


# --- strided rows ---------------------------------------------------------------------------------------------


def _layouts(sp, cols, counts, num_banks, *, mesh_id=0, base=0x100000, host="prefill"):
    """Gathered k / v / index_k layouts over ``len(counts)`` stages; each (stage, cache) has its own base."""
    layouts = []
    for cache_idx in range(3):
        stages, first = [], 0
        for i, count in enumerate(counts):
            stages.append(
                {
                    "rank": i,
                    "first_layer": first,
                    "count": count,
                    "base_addr": base + 0x10000000 * i + 0x4000000 * cache_idx,
                    "num_banks": num_banks,
                    "host_tag": zlib.crc32(host.encode()) & 0x7FFFFFFF,
                    "fnids": [
                        [ttnn.FabricNodeId(ttnn.MeshId(mesh_id + i), r * cols + c) for c in range(cols)]
                        for r in range(sp)
                    ],
                }
            )
            first += count
        layouts.append(stages)
    return layouts


def _build_table(path, layouts, *, sp, cols, chunk_size, seq_len, num_users, dedup, index_k_layers=None):
    local_layers = layouts[0][0]["count"]
    rows = seq_len // sp // (cols if dedup else 1)

    def t(n_rows):
        return SimpleNamespace(shape=(num_users * local_layers, 1, n_rows, HEAD_DIM), dtype=ttnn.bfloat8_b)

    cache = SimpleNamespace(
        k=t(seq_len // sp), v=t(seq_len // sp), index_k=t(rows), index_k_tp_axis=1 if dedup else None
    )
    build_and_serialize_kv_chunk_table(
        mesh_device=None,
        kv_cache=cache,
        seq_len=seq_len,
        num_layers=local_layers,
        mesh_shape=(sp, cols),
        sp_axis=0,
        num_users=num_users,
        chunk_size=chunk_size,
        num_kv_heads=cols,
        head_dim=HEAD_DIM,
        path=str(path),
        stage_layouts=layouts,
        index_k_layers=index_k_layers,
    )
    return ttnn.experimental.disaggregation.import_from_protobuf_file(str(path))


def _walk_reference(layouts, *, sp, cols, chunk_size, seq_len, num_users, dedup):
    """(config, slot, layer, position) -> (noc_addr, chips) by replaying each chip's ROUND_ROBIN_1D page walk."""
    chunk_local = chunk_size // sp
    specs = [(0, [[h]]) for h in range(cols)] + [(1, [[h]]) for h in range(cols)]
    specs.append((2, [[c] for c in range(cols)] if dedup else [list(range(cols))]))
    chunk_bytes = _chunk_size_bytes(ttnn.bfloat8_b, HEAD_DIM)
    out = {}
    for config_id, (cache_idx, row_groups) in enumerate(specs):
        group_tokens = chunk_local // len(row_groups)
        for stage in layouts[cache_idx]:
            for row in range(sp):
                for stripe, group_cols in enumerate(row_groups):
                    chips = sorted(_ids([stage["fnids"][row][c] for c in group_cols]))
                    page = 0
                    for slot in range(num_users):
                        for local_layer in range(stage["count"]):
                            for slab in range(seq_len // chunk_size):
                                start = slab * chunk_size + row * chunk_local + stripe * group_tokens
                                for pos in range(start, start + group_tokens, NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK):
                                    bank, sweep = divmod(page, stage["num_banks"])[::-1]
                                    noc = (bank << 32) | (stage["base_addr"] + sweep * chunk_bytes)
                                    out[config_id, slot, stage["first_layer"] + local_layer, pos] = (noc, chips)
                                    page += 1
    return out


@pytest.mark.parametrize(
    "sp, cols, chunk_size, seq_len, num_banks, dedup",
    [
        (2, 4, 64, 128, 8, False),  # one page per slab: the period is the whole row
        (2, 4, 128, 1024, 8, False),  # two pages per slab: four-slab period, two chunks per residue
        (2, 4, 256, 1024, 8, True),  # TP-deduped index_k, one page per stripe
        (4, 4, 4096, 16384, 8, False),  # 16-chip prefill partition geometry
        (4, 4, 4096, 16384, 8, True),
        (2, 4, 2048, 16384, 7, False),  # bank count coprime to the pages per slab
    ],
)
def test_strided_rows_match_page_walk(tmp_path, sp, cols, chunk_size, seq_len, num_banks, dedup):
    layouts = _layouts(sp, cols, STAGE_COUNTS, num_banks)
    geometry = dict(sp=sp, cols=cols, chunk_size=chunk_size, seq_len=seq_len, num_users=2, dedup=dedup)
    table = _build_table(tmp_path / "t.pb", layouts, **geometry)
    disagg = ttnn.experimental.disaggregation
    assert all(table.compression(c) == disagg.ChunkCompression.STRIDED_ROWS for c in range(table.num_configs()))
    chunk_bytes = _chunk_size_bytes(ttnn.bfloat8_b, HEAD_DIM)
    for (config_id, slot, layer, pos), (noc, chips) in _walk_reference(layouts, **geometry).items():
        loc = table.lookup(layer, pos, slot, config_id)
        assert (loc.noc_addr, loc.size_bytes) == (
            noc,
            chunk_bytes,
        ), f"config {config_id} slot {slot} layer {layer} pos {pos}: {loc.noc_addr:#x}, expected {noc:#x}"
        assert sorted(_ids(table.get_device_group(loc.device_group_index).fabric_node_ids)) == chips


PREFILL_1M = dict(sp=4, cols=4, chunk_size=4096, seq_len=1 << 20, layers=60)


def test_prefill_1m_single_slot_table(tmp_path, monkeypatch, expect_error):
    # The 16-chip prefill partition at 1M positions, one slot, with runner_16chip_kvm_1m.yaml's dual-write budget.
    monkeypatch.setenv("KV_CHUNK_TABLE_DUAL_WRITE_MAX_BYTES", "2080374784")
    sp, cols, chunk, seq, layers = (PREFILL_1M[k] for k in ("sp", "cols", "chunk_size", "seq_len", "layers"))
    msa = set(range(3, layers))
    layouts = _layouts(sp, cols, (layers,), BH_NUM_DRAM_BANKS)
    path = tmp_path / "prefill_1m.pb"
    table = _build_table(
        path, layouts, sp=sp, cols=cols, chunk_size=chunk, seq_len=seq, num_users=1, dedup=False, index_k_layers=msa
    )
    # Runs only: the per-chunk entries mirror alone would be ~0.85 GB.
    assert os.path.getsize(path) < 64 << 20
    disagg = ttnn.experimental.disaggregation
    for config_id in range(2 * cols + 1):
        cfg = table.config(config_id)
        assert (cfg.num_slots, cfg.max_sequence_length, cfg.num_layers) == (1, seq, layers)
        assert table.compression(config_id) == disagg.ChunkCompression.STRIDED_ROWS

    chunk_bytes = _chunk_size_bytes(ttnn.bfloat8_b, HEAD_DIM)
    chunk_local = chunk // sp
    pages_per_slab = chunk_local // NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK
    stage = [layout[0] for layout in layouts]
    for layer in (0, 2, 3, layers - 1):
        for pos in (0, chunk_local - 32, chunk_local, chunk + 3 * chunk_local, seq - chunk_local, seq - 32):
            slab, in_slab = divmod(pos, chunk)
            row, in_row = divmod(in_slab, chunk_local)
            page = (layer * (seq // chunk) + slab) * pages_per_slab + in_row // NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK
            bank, sweep = page % BH_NUM_DRAM_BANKS, page // BH_NUM_DRAM_BANKS
            for config_id in range(2 * cols + 1):
                cache_idx = min(config_id // cols, 2)
                loc = table.lookup(layer, pos, 0, config_id)
                if cache_idx == 2 and layer not in msa:
                    assert loc.size_bytes == 0
                    continue
                assert loc.noc_addr == (bank << 32) | (stage[cache_idx]["base_addr"] + sweep * chunk_bytes)
                head = config_id % cols
                expected = [stage[cache_idx]["fnids"][row][c] for c in (range(cols) if cache_idx == 2 else [head])]
                assert sorted(_ids(table.get_device_group(loc.device_group_index).fabric_node_ids)) == sorted(
                    _ids(expected)
                )
    with expect_error(RuntimeError, "num_slots"):
        table.lookup(0, 0, 1, 0)


def _kv_manager_plan_errors(src, dst, *, src_slot, dst_slot, layers, pos_end):
    """The checks the KV manager's migration planner runs per (layer, config) before booking a copy: both sides
    cover the row or neither, chunk geometry matches, and every slot / position is inside its OWN table."""
    errors = []
    for layer in layers:
        for config_id in range(src.num_configs()):
            s, d = src.config(config_id), dst.config(config_id)
            if src_slot >= s.num_slots or dst_slot >= d.num_slots:
                errors.append(f"layer {layer} config {config_id}: slot {src_slot}->{dst_slot} out of range")
                continue
            if pos_end > s.max_sequence_length or pos_end > d.max_sequence_length:
                errors.append(f"layer {layer} config {config_id}: pos_end {pos_end} past a table's depth")
                continue
            if s.chunk_n_tokens != d.chunk_n_tokens:
                errors.append(f"config {config_id}: chunk_n_tokens {s.chunk_n_tokens} vs {d.chunk_n_tokens}")
                continue
            for pos in range(0, pos_end, s.chunk_n_tokens):
                a, b = src.lookup(layer, pos, src_slot, config_id), dst.lookup(layer, pos, dst_slot, config_id)
                if a.size_bytes != b.size_bytes:
                    errors.append(f"layer {layer} config {config_id} pos {pos}: {a.size_bytes} vs {b.size_bytes} B")
                    break
    return errors


def test_single_slot_prefill_pairs_with_two_slot_decode(tmp_path):
    # Prefill: one slot at its own depth. Decode: two slots, twice the depth, a different slab and its own chips,
    # so the per-slot strides differ. Only names, coverage and chunk geometry have to agree.
    layers, msa = 4, {1, 2, 3}
    prefill = _build_table(
        tmp_path / "prefill.pb",
        _layouts(4, 4, (layers,), 8),
        sp=4,
        cols=4,
        chunk_size=4096,
        seq_len=8192,
        num_users=1,
        dedup=False,
        index_k_layers=msa,
    )
    decode = _build_table(
        tmp_path / "decode.pb",
        _layouts(4, 4, (layers,), 8, mesh_id=7, base=0x2000000, host="decode"),
        sp=4,
        cols=4,
        chunk_size=8192,
        seq_len=16384,
        num_users=2,
        dedup=False,
        index_k_layers=msa,
    )
    names = [prefill.config_name(i) for i in range(prefill.num_configs())]
    assert names == [decode.config_name(i) for i in range(decode.num_configs())]
    assert (prefill.config(0).num_slots, decode.config(0).num_slots) == (1, 2)
    assert decode.lookup(0, 0, 1, 0).noc_addr != decode.lookup(0, 0, 0, 0).noc_addr

    plan = dict(layers=range(layers), pos_end=8192)
    for dst_slot in (0, 1):
        assert _kv_manager_plan_errors(prefill, decode, src_slot=0, dst_slot=dst_slot, **plan) == []
    assert _kv_manager_plan_errors(prefill, decode, src_slot=0, dst_slot=2, **plan)
    assert _kv_manager_plan_errors(prefill, decode, src_slot=1, dst_slot=0, **plan)
    assert _kv_manager_plan_errors(prefill, decode, src_slot=0, dst_slot=1, layers=range(layers), pos_end=16384)
