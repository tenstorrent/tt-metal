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
stripe, addressed by a single-chip group and its own bank walk, while K / V are unchanged. The ``paged`` cases
give the second stage a paged pool: each chunk of it must sit where the slot walk puts the same page of a pool
built as a slot cache (one slot per page), and the runner's fixed page table must map slots one to one onto pages.
"""

import os
from types import SimpleNamespace

import pytest

import ttnn
from models.demos.common.prefill.runners.migration import validate_stage_layout_contiguous
from models.demos.minimax_m3.tt import tt_prefill_runtime
from models.demos.minimax_m3.tt.attention.kv_cache import (
    NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK,
    PAGE_SIZE,
    fixed_page_table,
    fixed_pool_pages,
    page_owner,
)
from models.demos.minimax_m3.tt.runners.kv_chunk_table import _chunk_size_bytes, build_and_serialize_kv_chunk_table

SP = 2
COLS = 4  # == num_kv_heads (the builder asserts the 1:1 head->column map)
CHUNK_SIZE = 64  # tokens_per_chunk_local = 32 == NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK
SEQ_LEN = 128
NUM_USERS = 2
HEAD_DIM = 128
STAGE_COUNTS = (2, 3)  # layers per stage; global total 5
NUM_BANKS = 8


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


def _stub_cache(num_layers, seq_len=SEQ_LEN, index_k_tp_axis=None, num_users=NUM_USERS):
    def t(dtype, rows):
        return SimpleNamespace(shape=(num_users * num_layers, 1, rows, HEAD_DIM), dtype=dtype)

    ik_rows = seq_len if index_k_tp_axis is None else seq_len // COLS
    # Migration requires a bf8 index_k (the decode peer's dtype); the builder rejects bf16.
    return SimpleNamespace(
        k=t(ttnn.bfloat8_b, seq_len),
        v=t(ttnn.bfloat8_b, seq_len),
        index_k=t(ttnn.bfloat8_b, ik_rows),
        index_k_tp_axis=index_k_tp_axis,
    )


def _build(
    tmp_path,
    stage_layouts,
    *,
    index_k_layers=None,
    seq_len=SEQ_LEN,
    chunk_size=CHUNK_SIZE,
    index_k_tp_axis=None,
    num_users=NUM_USERS,
    **paging,
):
    path = os.path.join(str(tmp_path), "m3_merge_table.pb")
    # num_layers is THIS rank's stage count (rank 0 builds), used only for the local shape assert.
    return build_and_serialize_kv_chunk_table(
        mesh_device=None,
        kv_cache=_stub_cache(STAGE_COUNTS[0], seq_len=seq_len, index_k_tp_axis=index_k_tp_axis, num_users=num_users),
        seq_len=seq_len,
        num_layers=STAGE_COUNTS[0],
        mesh_shape=(SP, COLS),
        sp_axis=0,
        num_users=num_users,
        chunk_size=chunk_size,
        num_kv_heads=COLS,
        head_dim=HEAD_DIM,
        path=path,
        stage_layouts=stage_layouts,
        index_k_layers=index_k_layers,
        **paging,
    )


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


# Two pages per rank per chunk, two chunks per slot.
PAGED_CHUNK_SIZE = 2 * PAGE_SIZE * SP
PAGED_SEQ_LEN = 2 * PAGED_CHUNK_SIZE


def test_paged_stage_addressing(tmp_path):
    """Stage 1 holds a paged pool (stage 0 keeps slots). Oracle: the pool is a slot cache of one slot per page and
    PAGE_SIZE rows per chip, so the slot walk over it places page p of rank r at slot p, positions [r * PAGE_SIZE,
    (r + 1) * PAGE_SIZE). Every chunk of the paged stage, on every config, must sit at its page's oracle entry, and
    stage 0's entries must equal a slots-only table's."""
    imp = ttnn.experimental.disaggregation.import_from_protobuf_file
    layouts = _stage_layouts()
    for name in ("paged", "slots", "pool"):
        (tmp_path / name).mkdir()
    sizes = dict(seq_len=PAGED_SEQ_LEN, chunk_size=PAGED_CHUNK_SIZE)
    paged = imp(_build(tmp_path / "paged", layouts, paged_stage=lambda first, count: first > 0, **sizes))
    slots = imp(_build(tmp_path / "slots", layouts, **sizes))
    num_pages = fixed_pool_pages(num_users=NUM_USERS, max_seq_len=PAGED_SEQ_LEN, sp=SP)
    pool = imp(
        _build(
            tmp_path / "pool",
            layouts,
            seq_len=PAGE_SIZE * SP,
            chunk_size=PAGE_SIZE * SP,
            num_users=num_pages,
        )
    )
    table = fixed_page_table(num_users=NUM_USERS, max_seq_len=PAGED_SEQ_LEN, sp=SP, chunk_size=PAGED_CHUNK_SIZE)
    chunk_local = PAGED_CHUNK_SIZE // SP
    total = sum(STAGE_COUNTS)
    checked = 0
    for cfg in range(2 * COLS + 1):
        for slot in range(NUM_USERS):
            for position in range(0, PAGED_SEQ_LEN, NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK):
                rank = page_owner(position // PAGE_SIZE, chunk_local=chunk_local, sp=SP)
                page = int(table[slot, position // PAGE_SIZE])
                for layer in range(total):
                    got = paged.lookup(layer, position, slot, cfg)
                    if layer < STAGE_COUNTS[0]:
                        want = slots.lookup(layer, position, slot, cfg)
                    else:
                        want = pool.lookup(layer, rank * PAGE_SIZE + position % PAGE_SIZE, page, cfg)
                    assert (got.noc_addr, got.size_bytes) == (want.noc_addr, want.size_bytes), (
                        f"config {cfg} slot {slot} layer {layer} position {position}: "
                        f"{got.noc_addr:#x} != {want.noc_addr:#x}"
                    )
                    assert _ids(paged.get_device_group(got.device_group_index).fabric_node_ids) == _ids(
                        pool.get_device_group(want.device_group_index).fabric_node_ids
                        if layer >= STAGE_COUNTS[0]
                        else slots.get_device_group(want.device_group_index).fabric_node_ids
                    )
                    checked += 1
    assert checked == (2 * COLS + 1) * NUM_USERS * total * PAGED_SEQ_LEN // NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK


def test_fixed_page_table_is_one_to_one():
    """Every rank's pool pages are owned by exactly one (slot, page) of the fixed table."""
    table = fixed_page_table(num_users=NUM_USERS, max_seq_len=PAGED_SEQ_LEN, sp=SP, chunk_size=PAGED_CHUNK_SIZE)
    num_pages = fixed_pool_pages(num_users=NUM_USERS, max_seq_len=PAGED_SEQ_LEN, sp=SP)
    for rank in range(SP):
        owned = [q for q in range(table.shape[1]) if page_owner(q, chunk_local=PAGED_CHUNK_SIZE // SP, sp=SP) == rank]
        assert sorted(table[:, owned].flatten().tolist()) == list(range(num_pages)), f"rank {rank}"


def _hf_config(msa_layers, num_layers=6):
    return SimpleNamespace(
        sparse_attention_config={
            "use_sparse_attention": True,
            "sparse_attention_freq": [int(i in msa_layers) for i in range(num_layers)],
        }
    )


@pytest.mark.parametrize(
    "enabled, first, count, paged",
    [(False, 3, 2, False), (True, 3, 2, True), (True, 2, 2, False), (True, 0, 6, False)],
    ids=["off", "msa_stage", "dense_layer", "whole_model"],
)
def test_stage_is_paged(monkeypatch, enabled, first, count, paged):
    """Only a stage of MSA layers is paged, and only with PAGED_KV_CACHE on."""
    monkeypatch.setattr(tt_prefill_runtime, "PAGED_KV_CACHE", enabled)
    assert tt_prefill_runtime.stage_is_paged(_hf_config({3, 4, 5}), first, count) == paged


def test_paging_mismatch_raises(monkeypatch, expect_error):
    """A stage whose cache disagrees with stage_is_paged would publish a wrong migration table: it raises."""
    monkeypatch.setattr(tt_prefill_runtime, "PAGED_KV_CACHE", True)
    runtime = SimpleNamespace(hf_config=_hf_config({3, 4, 5}))
    check = tt_prefill_runtime.TtPrefillRuntime._check_paging
    with expect_error(ValueError, "slot KV cache"):
        check(runtime, SimpleNamespace(page_table=None), 3, 2)
    with expect_error(ValueError, "paged KV cache"):
        check(runtime, SimpleNamespace(page_table=object()), 0, 3)
    check(runtime, SimpleNamespace(page_table=object()), 3, 2)
