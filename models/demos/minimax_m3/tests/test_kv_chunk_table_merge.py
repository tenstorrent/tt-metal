# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Device-free checks of the M3 multi-stage (pipeline-parallel) KV chunk table merge.

Drives ``build_and_serialize_kv_chunk_table`` with synthetic 2-stage ``stage_layouts`` (one gathered
layout per cache — k, v, index_k) and a stub kv_cache (the merged path reads only dtypes/shapes from it —
addresses, fabric nodes, hosts and bank counts all come from the gathered layouts), then asserts the
table's addressing against the layouts: per-(stage, cache) base addresses at global layer indices,
single-member per-head device groups vs the full-row index_k replica group, and the per-(config, stage,
row) bank-walk restart.
"""

import os
from types import SimpleNamespace

import pytest

import ttnn
from models.demos.common.prefill.runners.migration import validate_stage_layout_contiguous
from models.demos.minimax_m3.tt.attention.kv_cache import NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK
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


def _stub_cache(num_layers):
    def t(dtype):
        return SimpleNamespace(shape=(NUM_USERS * num_layers, 1, SEQ_LEN, HEAD_DIM), dtype=dtype)

    return SimpleNamespace(k=t(ttnn.bfloat8_b), v=t(ttnn.bfloat8_b), index_k=t(ttnn.bfloat16))


def _build(tmp_path, stage_layouts):
    path = os.path.join(str(tmp_path), "m3_merge_table.pb")
    # num_layers is THIS rank's stage count (rank 0 builds), used only for the local shape assert.
    return build_and_serialize_kv_chunk_table(
        mesh_device=None,
        kv_cache=_stub_cache(STAGE_COUNTS[0]),
        seq_len=SEQ_LEN,
        num_layers=STAGE_COUNTS[0],
        mesh_shape=(SP, COLS),
        sp_axis=0,
        num_users=NUM_USERS,
        chunk_size=CHUNK_SIZE,
        num_kv_heads=COLS,
        head_dim=HEAD_DIM,
        path=path,
        stage_layouts=stage_layouts,
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
    # index_k carries the bf16 chunk size, K/V the bfp8 one.
    assert merged_table.config(0).chunk_size_bytes == _chunk_size_bytes(ttnn.bfloat8_b, HEAD_DIM)
    assert merged_table.config(2 * COLS).chunk_size_bytes == _chunk_size_bytes(ttnn.bfloat16, HEAD_DIM)


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
