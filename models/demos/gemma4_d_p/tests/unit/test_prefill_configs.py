# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host checks for the prefill SDPA, matmul and RMSNorm config choices."""

import json
from types import SimpleNamespace

import pytest

import ttnn
from models.demos.gemma4_d_p.tt.attention import ring_prefill
from models.demos.gemma4_d_p.tt.attention.operations import projection_math_fidelity
from models.demos.gemma4_d_p.tt.attention.ring_prefill import ring_sdpa_chunk_sizes
from models.demos.gemma4_d_p.tt.matmul_config import prefill_1d_matmul_program_config, short_m_sharded_memcfg
from models.demos.gemma4_d_p.tt.model import prefill_chunk_geometry_error
from models.demos.gemma4_d_p.tt.rms_norm import _block_shard_geometry

GRID = SimpleNamespace(x=11, y=10)


@pytest.mark.parametrize(
    "slab, sliding, expected",
    [
        (32, False, (32, 224, 4, False)),
        (256, False, (64, 288, 3, False)),
        (448, False, (128, 288, 3, False)),
        (512, False, (128, 256, 3, False)),
        (768, False, (128, 256, 2, False)),
        (832, False, (64, 256, 1, True)),
        (1024, False, (96, 256, 1, True)),
        (256, True, (128, 128, 3, False)),
        (512, True, (128, 128, 3, False)),
        (1024, True, (128, 128, 1, False)),
        (64, False, (64, 256, 4, False)),
        (160, False, (96, 224, 4, False)),
        (1536, False, (128, 256, 1, True)),
        (1664, False, (128, 256, 1, True)),
        (2048, False, (96, 256, 1, False)),
        (4096, False, (96, 256, 1, False)),
    ],
)
def test_ring_sdpa_chunk_sizes(slab, sliding, expected):
    assert ring_sdpa_chunk_sizes(slab, sliding, num_heads=8, grid=(11, 10), use_tuned_configs=True) == expected


@pytest.fixture
def ring_sdpa_table_file(tmp_path, monkeypatch):
    monkeypatch.setattr(ring_prefill, "__file__", str(tmp_path / "ring_prefill.py"))
    ring_prefill._load_global_ring_sdpa_configs.cache_clear()
    yield tmp_path / "ring_sdpa_configs.json"
    ring_prefill._load_global_ring_sdpa_configs.cache_clear()


@pytest.mark.parametrize("contents", [None, "{", "null", "[]", "{}", '{"configs": []}'])
def test_ring_sdpa_table_unavailable_uses_heuristic(ring_sdpa_table_file, contents):
    if contents is not None:
        ring_sdpa_table_file.write_text(contents)
    assert ring_sdpa_chunk_sizes(768, False, use_tuned_configs=True) == (96, 256, 1, True)
    # 112 Q units on 110 cores: the op refuses both a K split and segments, so neither is requested.
    assert ring_sdpa_chunk_sizes(448, False, use_tuned_configs=True) == (32, 256, 1, False)
    assert ring_sdpa_chunk_sizes(448, True, use_tuned_configs=True) == (128, 128, 3, False)


def test_ring_sdpa_table_is_relative_to_module_and_cached(ring_sdpa_table_file, tmp_path, monkeypatch):
    other = tmp_path / "different_working_directory"
    other.mkdir()
    monkeypatch.chdir(other)
    ring_sdpa_table_file.write_text(json.dumps({"configs": {"6144": [128, 256, 3, True]}}))
    assert ring_sdpa_chunk_sizes(768, False, use_tuned_configs=True) == (128, 256, 3, True)
    ring_sdpa_table_file.unlink()
    assert ring_sdpa_chunk_sizes(768, False, use_tuned_configs=True) == (128, 256, 3, True)


def test_ring_sdpa_tuning_is_optional_and_geometry_specific():
    assert ring_sdpa_chunk_sizes(768, False) == (96, 256, 1, True)
    assert ring_sdpa_chunk_sizes(768, False, use_tuned_configs=False) == (96, 256, 1, True)
    assert ring_sdpa_chunk_sizes(768, False, num_heads=4, use_tuned_configs=True) == (96, 256, 1, True)
    assert ring_sdpa_chunk_sizes(768, False, grid=(12, 10), use_tuned_configs=True) == (96, 256, 1, True)


@pytest.mark.parametrize(
    "cp,tp,cols,rows,kv_dtype,expected",
    [
        (8, 4, 12, 10, ttnn.bfloat8_b, True),
        (4, 8, 12, 10, ttnn.bfloat8_b, False),
        (8, 4, 11, 11, ttnn.bfloat8_b, False),
        (8, 4, 12, 10, ttnn.bfloat16, False),
    ],
)
def test_ring_sdpa_global_caller_guards_tuned_workload(
    monkeypatch, expect_error, cp, tp, cols, rows, kv_dtype, expected
):
    class SelectionReached(Exception):
        pass

    def select(*args, **kwargs):
        assert kwargs["use_tuned_configs"] is expected
        raise SelectionReached("SDPA configuration selected")

    monkeypatch.setattr(ring_prefill, "ring_sdpa_chunk_sizes", select)
    with expect_error(SelectionReached, "SDPA configuration selected"):
        ring_prefill.global_ring_prefill_attention(
            SimpleNamespace(shape=(1, 8, 768, 512), dtype=ttnn.bfloat16),
            SimpleNamespace(dtype=kv_dtype),
            SimpleNamespace(device=SimpleNamespace(arch=lambda: ttnn.Arch.BLACKHOLE), cp_degree=cp, tp_degree=tp),
            SimpleNamespace(compute_grid_size=SimpleNamespace(x=cols, y=rows)),
            prefill_metadata=None,
            num_local_kv_heads=1,
            max_seq_len=270336,
            logical_n=262144,
            kv_actual_global=256000,
        )


@pytest.mark.parametrize(
    "slab, expected",
    [
        (256, (64, 256, 3, False)),
        (512, (128, 256, 3, False)),
        # A quarter slab that is not whole tiles falls back to one tile.
        (64, (32, 256, 3, False)),
        # 40 units take 4 rows of 11, so three bands need 12 > 10 rows: unsplit, segmented.
        (160, (32, 256, 1, True)),
    ],
)
def test_ring_sdpa_heuristic_never_pairs_segments_with_k_split(slab, expected):
    assert ring_sdpa_chunk_sizes(slab, False, num_heads=8, grid=(11, 10)) == expected


@pytest.mark.parametrize(
    "slab, num_heads, grid, k_splits",
    [
        # Three bands of every (head, q 128 chunk) unit must fit in whole rows of the grid.
        (512, 8, (11, 10), 3),
        (512, 8, (11, 8), 1),
        (256, 16, (11, 10), 3),
        (512, 16, (11, 10), 1),
        # 3 x 32 units fit 104 cores, but each band needs 3 of 8 rows.
        (512, 8, (13, 8), 1),
    ],
)
def test_ring_sdpa_sliding_k_splits(slab, num_heads, grid, k_splits):
    assert ring_sdpa_chunk_sizes(slab, True, num_heads=num_heads, grid=grid)[2] == k_splits


@pytest.mark.parametrize(
    "rows, expected",
    [
        (256, ttnn.MathFidelity.HiFi2),
        (511, ttnn.MathFidelity.HiFi2),
        (512, ttnn.MathFidelity.LoFi),
        (1024, ttnn.MathFidelity.LoFi),
    ],
)
def test_projection_math_fidelity(rows, expected):
    assert projection_math_fidelity(rows) == expected


def _tensor(rows, cols, shard_cols=None):
    """A host stand-in for a tensor; shard_cols makes it width-sharded with that shard width."""
    shard_spec = SimpleNamespace(shape=(rows, shard_cols))
    return SimpleNamespace(
        padded_shape=(1, 1, rows, cols),
        is_sharded=lambda: shard_cols is not None,
        memory_config=lambda: SimpleNamespace(shard_spec=shard_spec),
    )


@pytest.mark.parametrize("n", [2048, 5376])
def test_1d_matmul_config_at_short_m(n):
    config = prefill_1d_matmul_program_config(_tensor(256, 5376), _tensor(5376, n), GRID)
    assert config is not None
    assert (config.per_core_M, config.per_core_N) == (8, 2)
    assert 5376 // 32 % config.in0_block_w == 0 and config.in0_block_w <= 16


def test_1d_matmul_config_reads_sharded_activation_one_shard_per_k_block():
    # Chunk 2048's MLP down: 8 tile rows, K 5376 in 8-tile shards, 4 weight columns per core.
    config = prefill_1d_matmul_program_config(
        _tensor(256, 5376, shard_cols=256), _tensor(5376, 5376), GRID, per_core_n=4
    )
    assert (config.in0_block_w, config.per_core_N) == (8, 4)
    # fp32 dest: at most 4 tiles per output subblock, shaped so the output can be width-sharded.
    assert config.out_subblock_h * config.out_subblock_w <= 4
    assert config.out_subblock_w == config.per_core_N or config.out_subblock_h == 1


@pytest.mark.parametrize(
    "rows, n",
    [(512, 2048), (256, 2 * 32 * 111), (256, 3 * 32)],  # M above 8 tiles, more columns than cores, odd width
)
def test_1d_matmul_config_falls_back(rows, n):
    assert prefill_1d_matmul_program_config(_tensor(rows, 5376), _tensor(5376, n), GRID) is None


@pytest.mark.parametrize("k_tiles", [13, 8 * 111])
def test_short_m_sharded_layout_rejects_unshardable_k(k_tiles, expect_error):
    device = SimpleNamespace(compute_with_storage_grid_size=lambda: GRID)
    with expect_error(ValueError, f"{k_tiles} K tiles"):
        short_m_sharded_memcfg(device, 256, k_tiles)


# Rows per device after the TP row split: 64 / 128 / 256 at chunk 2048 / 4096 / 8192. Counts the minimum block
# height doesn't divide (32 / 96 / 160 / 288 / 576) take the smallest divisor that fits.
@pytest.mark.parametrize(
    "rows, expected",
    [
        (64, (2, 14, 1, 12)),
        (128, (2, 14, 2, 12)),
        (256, (2, 14, 4, 12)),
        (1024, (4, 21, 8, 8)),
        (32, (1, 14, 1, 12)),
        (96, (1, 14, 3, 12)),
        (160, (1, 14, 5, 12)),
        (288, (3, 14, 3, 12)),
        (576, (3, 21, 6, 8)),
    ],
)
def test_rms_norm_block_shard_geometry(rows, expected):
    assert _block_shard_geometry(rows, 5376) == expected


def test_rms_norm_block_shard_falls_back_when_blocks_overflow_l1():
    assert _block_shard_geometry(2048, 5376) is None


@pytest.mark.parametrize(
    "chunk, cp, tp, max_seq_len, error",
    [
        (2048, 8, 4, 262144, None),
        (8192, 8, 4, 262144, None),
        (4096, 4, 8, 262144, None),
        # Issue #57836's sizes: 416 rows per CP rank split four ways is not whole tiles.
        (3328, 8, 4, 19968, "sequence-parallel residual"),
        # 4x8: a 384-token slab split eight ways is 48 rows, not whole tiles.
        (1536, 4, 8, 6144, "sequence-parallel residual"),
        # Without TP the residual split allows a 160-token slab, which is not whole 128-token sliding K chunks.
        (1280, 8, 1, 2560, "sliding attention"),
        (12288, 8, 4, 262144, "must divide max_seq_len"),
    ],
)
def test_prefill_chunk_geometry_error(chunk, cp, tp, max_seq_len, error):
    message = prefill_chunk_geometry_error(chunk, cp, max_seq_len, tp_degree=tp)
    if error is None:
        assert message is None
    else:
        assert error in message
