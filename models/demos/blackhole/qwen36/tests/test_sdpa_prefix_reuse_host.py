# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Host-only tests (no device; ttnn importable) of the chunked prefill SDPA prefix-reuse flags:

* QWEN36_SDPA_PREFIX_REUSE (tp_common.sdpa_prefix_reuse_bits): default 0 (today's program), parsing, read once;
* QWEN36_SDPA_PREFILL_QSPLIT (tp_common.sdpa_prefill_q_chunk): the Q chunk per call size, K chunk untouched;
* QWEN36_SDPA_PREFIX_REUSE=auto per call (tp_common.sdpa_prefill_reuse_bits), QWEN36_SDPA_PREFILL_READ_BATCH_UNITS;
* SDPAProgramConfig.prefix_reuse: default 0, settable, in the repr.

  PYTHONPATH=<worktree>:$TT_METAL_HOME/ttnn:$TT_METAL_HOME/tools $TT_METAL_HOME/python_env/bin/python -m pytest -q \\
      models/demos/blackhole/qwen36/tests/test_sdpa_prefix_reuse_host.py
"""

import pytest

import ttnn
from models.demos.blackhole.qwen36.tt import tp_common as tc


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    for name in (
        "_SDPA_PREFIX_REUSE",
        "_SDPA_PREFILL_QSPLIT",
        "_SDPA_PREFILL_QSPLIT_UNITS",
        "_SDPA_PREFILL_READ_BATCH_UNITS",
    ):
        monkeypatch.setattr(tc, name, None)
    for env in (
        "QWEN36_SDPA_PREFIX_REUSE",
        "QWEN36_SDPA_PREFILL_QSPLIT",
        "QWEN36_SDPA_PREFILL_QSPLIT_UNITS",
        "QWEN36_SDPA_PREFILL_READ_BATCH_UNITS",
    ):
        monkeypatch.delenv(env, raising=False)
    yield


def test_prefix_reuse_default_off():
    assert tc.sdpa_prefix_reuse_bits() == 0


@pytest.mark.parametrize("raw,bits", [("0", 0), ("", 0), ("7", 7), ("0x407", 0x407), ("2", 2), (" 4 ", 4)])
def test_prefix_reuse_parse(monkeypatch, raw, bits):
    monkeypatch.setattr(tc, "SDPA_PREFIX_REUSE_SUPPORTED", True)
    monkeypatch.setenv("QWEN36_SDPA_PREFIX_REUSE", raw)
    assert tc.sdpa_prefix_reuse_bits() == bits


def test_prefix_reuse_auto_per_call(monkeypatch):
    monkeypatch.setattr(tc, "SDPA_PREFIX_REUSE_SUPPORTED", True)
    monkeypatch.setenv("QWEN36_SDPA_PREFIX_REUSE", "auto")
    # 8192 rows: 192 Q chunks of 128 > 120 cores (light/heavy pairs) -> prefix chains only
    assert tc.sdpa_prefill_reuse_bits(8192, 128, 3, 120) == 2
    # Q chunks that fit the grid -> one per core
    assert tc.sdpa_prefill_reuse_bits(2048, 128, 3, 120) == 4
    assert tc.sdpa_prefill_reuse_bits(1024, 64, 3, 120) == 4  # split, 48 Q chunks > 24
    assert tc.sdpa_prefill_reuse_bits(512, 32, 3, 120) == 4  # split, 48 Q chunks > 24
    # split calls on at most 24 cores -> one per core + the read batches of the cores that read DRAM
    assert tc.sdpa_prefill_reuse_bits(256, 32, 3, 120) == 5  # 24 Q chunks
    assert tc.sdpa_prefill_reuse_bits(128, 32, 3, 120) == 5  # 12 Q chunks
    assert tc.sdpa_prefill_reuse_bits(128, 64, 3, 120) == 5
    # unsplit calls never take the read batches, however few their Q chunks
    assert tc.sdpa_prefill_reuse_bits(1024, 128, 3, 120) == 4  # 24 Q chunks of 128
    assert tc.sdpa_prefill_reuse_bits(128, 128, 3, 120) == 4
    assert tc.sdpa_prefill_reuse_bits(32, 32, 3, 120, k_chunk=32) == 4  # the 32-row verify bucket
    assert tc.sdpa_prefill_reuse_bits(1024, 64, 3, 120, k_chunk=64) == 4  # the S < 2048 scalar path's K chunk 64
    assert tc.sdpa_prefill_reuse_bits(8192, 128, 3, 64) == 2


def test_prefix_reuse_auto_read_batch_units(monkeypatch):
    monkeypatch.setattr(tc, "SDPA_PREFIX_REUSE_SUPPORTED", True)
    monkeypatch.setenv("QWEN36_SDPA_PREFIX_REUSE", "auto")
    assert tc.sdpa_prefill_read_batch_units(120) == 24
    assert tc.sdpa_prefill_read_batch_units(64) == 12
    monkeypatch.setattr(tc, "_SDPA_PREFILL_READ_BATCH_UNITS", None)
    monkeypatch.setenv("QWEN36_SDPA_PREFILL_READ_BATCH_UNITS", "0")  # never: one Q chunk per core only
    assert tc.sdpa_prefill_reuse_bits(128, 32, 3, 120) == 4
    monkeypatch.setattr(tc, "_SDPA_PREFILL_READ_BATCH_UNITS", None)
    monkeypatch.setenv("QWEN36_SDPA_PREFILL_READ_BATCH_UNITS", "48")
    assert tc.sdpa_prefill_reuse_bits(512, 32, 3, 120) == 5
    assert tc.sdpa_prefill_reuse_bits(2048, 128, 3, 120) == 4  # unsplit
    assert tc.sdpa_prefill_reuse_bits(8192, 128, 3, 120) == 2


@pytest.mark.parametrize("raw", ["-1", "x"])
def test_prefix_reuse_read_batch_units_bad_value(monkeypatch, raw):
    monkeypatch.setenv("QWEN36_SDPA_PREFILL_READ_BATCH_UNITS", raw)
    with pytest.raises(ValueError):
        tc.sdpa_prefill_read_batch_units(120)


@pytest.mark.parametrize("raw,bits", [("0", 0), ("7", 7), ("2", 2)])
def test_prefix_reuse_explicit_value_for_every_call(monkeypatch, raw, bits):
    monkeypatch.setattr(tc, "SDPA_PREFIX_REUSE_SUPPORTED", True)
    monkeypatch.setenv("QWEN36_SDPA_PREFIX_REUSE", raw)
    for rows, q in ((8192, 128), (2048, 128), (128, 32)):
        assert tc.sdpa_prefill_reuse_bits(rows, q, 3, 120) == bits


def test_prefix_reuse_auto_needs_the_ttnn_field(monkeypatch):
    monkeypatch.setattr(tc, "SDPA_PREFIX_REUSE_SUPPORTED", False)
    monkeypatch.setenv("QWEN36_SDPA_PREFIX_REUSE", "auto")
    assert tc.sdpa_prefill_reuse_bits(8192, 128, 3, 120) == 0


@pytest.mark.parametrize("raw", ["-1", "0x10000", "on"])
def test_prefix_reuse_bad_value(monkeypatch, raw):
    monkeypatch.setattr(tc, "SDPA_PREFIX_REUSE_SUPPORTED", True)
    monkeypatch.setenv("QWEN36_SDPA_PREFIX_REUSE", raw)
    with pytest.raises(ValueError):
        tc.sdpa_prefix_reuse_bits()


def test_prefix_reuse_read_once(monkeypatch):
    monkeypatch.setattr(tc, "SDPA_PREFIX_REUSE_SUPPORTED", True)
    monkeypatch.setenv("QWEN36_SDPA_PREFIX_REUSE", "7")
    assert tc.sdpa_prefix_reuse_bits() == 7
    monkeypatch.setenv("QWEN36_SDPA_PREFIX_REUSE", "0")
    assert tc.sdpa_prefix_reuse_bits() == 7


def test_prefix_reuse_needs_the_ttnn_field(monkeypatch):
    monkeypatch.setattr(tc, "SDPA_PREFIX_REUSE_SUPPORTED", False)
    monkeypatch.setenv("QWEN36_SDPA_PREFIX_REUSE", "7")
    assert tc.sdpa_prefix_reuse_bits() == 0


def test_qsplit_default_off_is_today():
    # flag off: Q chunk = K chunk for every call the model makes
    for k, rows in ((128, 8192), (128, 2048), (128, 128), (64, 1024), (32, 32)):
        assert tc.sdpa_prefill_q_chunk(k, rows, 3, 120) == k


@pytest.mark.parametrize(
    "q_min,rows,expect",
    [
        (64, 8192, 128),  # 192 Q chunks of 128 already exceed the 48-chunk target
        (64, 4096, 128),
        (64, 2048, 128),  # 3 * 32 = 96 > 48: the 48 chunks of 128 rows stay
        (32, 2048, 128),
        (64, 1024, 64),  # 3 * 16 = 48
        (32, 1024, 64),  # 32 rows would make 96 > 48
        (32, 512, 32),  # 48
        (32, 256, 32),  # 24
        (32, 128, 32),  # 12
        (64, 128, 64),
    ],
)
def test_qsplit_table(monkeypatch, q_min, rows, expect):
    monkeypatch.setenv("QWEN36_SDPA_PREFILL_QSPLIT", str(q_min))
    assert tc.sdpa_prefill_q_chunk(128, rows, 3, 120) == expect


def test_qsplit_units_default_and_override(monkeypatch):
    assert tc.sdpa_prefill_qsplit_units(120) == 48
    assert tc.sdpa_prefill_qsplit_units(64) == 25
    monkeypatch.setattr(tc, "_SDPA_PREFILL_QSPLIT_UNITS", None)
    monkeypatch.setenv("QWEN36_SDPA_PREFILL_QSPLIT_UNITS", "120")
    monkeypatch.setenv("QWEN36_SDPA_PREFILL_QSPLIT", "32")
    assert tc.sdpa_prefill_qsplit_units(120) == 120
    assert tc.sdpa_prefill_q_chunk(128, 2048, 3, 120) == 64  # 96 <= 120
    assert tc.sdpa_prefill_q_chunk(128, 1024, 3, 120) == 32


def test_qsplit_keeps_other_k_chunks(monkeypatch):
    monkeypatch.setenv("QWEN36_SDPA_PREFILL_QSPLIT", "32")
    assert tc.sdpa_prefill_q_chunk(64, 1024, 3, 120) == 64  # the S < 2048 scalar path's 64-row K chunk
    assert tc.sdpa_prefill_q_chunk(128, 64, 3, 120) == 128  # fewer rows than the K chunk: untouched


def test_qsplit_respects_the_chunk_start(monkeypatch):
    monkeypatch.setenv("QWEN36_SDPA_PREFILL_QSPLIT", "32")
    # the model's chunk starts are multiples of 128: same choice as at 0
    assert tc.sdpa_prefill_q_chunk(128, 512, 3, 120, chunk_start=128 * 7) == 32
    # a start the smaller Q chunks would not divide keeps today's chunk
    assert tc.sdpa_prefill_q_chunk(128, 512, 3, 120, chunk_start=32 * 7) == 128


def test_qsplit_small_grid(monkeypatch):
    monkeypatch.setenv("QWEN36_SDPA_PREFILL_QSPLIT", "32")
    assert tc.sdpa_prefill_q_chunk(128, 1024, 3, 64) == 128  # target 25 chunks: 3 * 16 = 48 > 25
    assert tc.sdpa_prefill_q_chunk(128, 512, 3, 64) == 64  # 24 <= 25; 48 > 25


@pytest.mark.parametrize("raw", ["16", "128", "x"])
def test_qsplit_bad_value(monkeypatch, raw):
    monkeypatch.setenv("QWEN36_SDPA_PREFILL_QSPLIT", raw)
    with pytest.raises(ValueError):
        tc.sdpa_prefill_qsplit_min()


@pytest.mark.skipif(not tc.SDPA_PREFIX_REUSE_SUPPORTED, reason="ttnn build without SDPAProgramConfig.prefix_reuse")
def test_program_config_field():
    grid = ttnn.CoreCoord(12, 10)
    cfg = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=grid, q_chunk_size=128, k_chunk_size=128)
    assert cfg.prefix_reuse == 0
    cfg7 = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=grid, q_chunk_size=64, k_chunk_size=128, prefix_reuse=0x407
    )
    assert cfg7.prefix_reuse == 0x407 and cfg7.q_chunk_size == 64 and cfg7.k_chunk_size == 128
    assert "prefix_reuse=1031" in repr(cfg7)
