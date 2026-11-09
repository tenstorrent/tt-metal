# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests for QWEN36_DECODE_AR_ONESHOT_LIVE_ROWS (one-shot all_reduce_async + live_rows_packing at <= 16 rows).

Host-checkable parts only, over the recording fake of test_decode_ar_oneshot_host.py:
* flag off (and instances built without __init__) leave the one-shot call exactly as today (no live_rows_packing);
* flag on adds live_rows_packing=True to bf16 one-shot calls of <= 16 logical rows, with or without a fused residual,
  and never at 17..32 rows;
* the flag needs QWEN36_DECODE_AR_ONESHOT=1 (no effect on the default path) and an fp32 GDN partial keeps the default op;
* the env helper parses only "1" as on.
Device numerics / latency: tests/ttnn/unit_tests/operations/ccl/test_all_reduce_async_one_shot.py (live arms). Run:
pytest models/demos/blackhole/qwen36/tests/test_decode_ar_oneshot_live_rows_host.py -q
"""

import pytest

import ttnn
from models.demos.blackhole.qwen36.tests.test_decode_ar_oneshot_host import DEFAULT_KW, FT, _ar_kwargs, _dar, rec  # noqa: F401
from models.demos.blackhole.qwen36.tt import tp_common as tpc


def _dar_live(one_shot=True, oneshot_live_rows=True, **kw):
    d = _dar(one_shot, **kw)
    if oneshot_live_rows is not None:
        d.oneshot_live_rows = oneshot_live_rows
    return d


@pytest.mark.parametrize("oneshot_live_rows", [None, False])
@pytest.mark.parametrize("rows", [4, 16, 32])
def test_flag_off_keeps_today(rec, oneshot_live_rows, rows):
    _dar_live(True, oneshot_live_rows).all_reduce(FT((1, 1, rows, 5120), ttnn.bfloat16, mem="DRAM"))
    k = _ar_kwargs(rec)
    assert set(k) == DEFAULT_KW | {"one_shot"} and "live_rows_packing" not in k


@pytest.mark.parametrize("rows", [1, 4, 8, 12, 16])
def test_flag_on_small(rec, rows):
    _dar_live().all_reduce(FT((1, 1, rows, 5120), ttnn.bfloat16, mem="DRAM"))
    k = _ar_kwargs(rec)
    assert k.get("one_shot") is True and k.get("live_rows_packing") is True
    assert set(k) == DEFAULT_KW | {"one_shot", "live_rows_packing"}


@pytest.mark.parametrize("rows", [17, 24, 32])
def test_flag_on_large_rows_full_tiles(rec, rows):
    _dar_live().all_reduce(FT((1, 1, rows, 5120), ttnn.bfloat16, mem="DRAM"))
    k = _ar_kwargs(rec)
    assert k.get("one_shot") is True and "live_rows_packing" not in k


def test_flag_on_fused_residual(rec):
    d = _dar_live(fuse=True)
    res = FT((1, 1, 8, 5120), ttnn.bfloat16)
    d.arm_residual(res)
    out = d.all_reduce(FT((1, 1, 8, 5120), ttnn.bfloat16, mem="DRAM"))
    k = _ar_kwargs(rec)
    assert k.get("one_shot") is True and k.get("live_rows_packing") is True and k.get("residual_tensor") is res
    assert d.add_residual(res, out) is out


def test_needs_one_shot(rec):
    # oneshot_live_rows without one_shot: the default path, untouched (no live rows unless QWEN36_DECODE_AR_LIVE_ROWS).
    _dar_live(one_shot=False).all_reduce(FT((1, 1, 8, 5120), ttnn.bfloat16, mem="DRAM"))
    assert set(_ar_kwargs(rec)) == DEFAULT_KW


def test_fp32_gdn_partial_keeps_default(rec):
    d = _dar_live(gdn_fp32=True)
    d.all_reduce(FT((1, 1, 8, 5120), ttnn.float32, mem="DRAM"), keep_fp32=True)
    k = _ar_kwargs(rec)
    assert "one_shot" not in k and "live_rows_packing" not in k


@pytest.mark.parametrize("val,on", [(None, False), ("0", False), ("1", True), ("true", False)])
def test_env_helper(monkeypatch, val, on):
    if val is None:
        monkeypatch.delenv(tpc.DECODE_AR_ONESHOT_LIVE_ROWS_ENV, raising=False)
    else:
        monkeypatch.setenv(tpc.DECODE_AR_ONESHOT_LIVE_ROWS_ENV, val)
    assert tpc.decode_ar_oneshot_live_rows_enabled() is on
