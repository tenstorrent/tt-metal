# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host checks (no device) for QWEN36_SPEC_SDPA_KV_CLAMP / the paged SDPA decode's clamp_kv_tail.

  * the flag: unset follows QWEN36_SPEC_ADAPTIVE_K, "0" / "1" force it; the call kwargs are {} when off (today's call).
  * every paged SDPA-decode call in tt/attention/tp.py passes **sdpa_kv_clamp_kwargs() (none can be missed).
  * a host model of the reader's K/V range (rt_args_common.hpp: get_dynamic_Sk_chunk_t, get_workload_for_core) gives the
    over-read rule the device tests rely on: a group with last position p_max resolves page-table entry
    last_block(p_max) = ceil((p_max + 1) / chunk) * chunk / 64 - 1; a row is over-read iff that is >= its width. With the
    spec cap of 4 tiles that is p_max >= 128 and p_max % 128 < 64 for rows ceil((p_max + 1) / 64) wide (the old adaptive-k
    test tables), never inside the serving rows (4097 entries) below p_max 262144, and always from p_max 262144 on (a
    verify window that ends within K of max_model_len 262144, or a drafter step there).
"""
import math
import os
import re

import pytest

from models.demos.blackhole.qwen36.tt import tp_common as tpc
from models.demos.blackhole.qwen36.tt.attention import tp as atp
from models.demos.blackhole.qwen36.tt.spec_engine import spec_table_width

BLOCK = 64
TILE = 32
SPEC_CAP_TILES = 4  # kSpecMaxDynamicChunkTiles


def _pow2_up_to(x, cap):
    if x == 0:
        return 1
    r = 1
    while r < x:
        r *= 2
    return min(r, cap)


def _last_block(p_max, cap_tiles=SPEC_CAP_TILES):
    """Index of the last page-table entry the legacy reader resolves for a group whose last position is p_max."""
    chunk = _pow2_up_to(p_max // TILE + 1, cap_tiles) * TILE  # get_dynamic_Sk_chunk_t
    valid = math.ceil((p_max + 1) / chunk) * chunk  # get_workload_for_core: nearest_n(cur_pos + 1, chunk)
    return (valid - 1) // BLOCK


def _clamped_last_block(p_max):
    """With clamp_kv_tail: rows past the tile holding p_max read that tile."""
    return min(_last_block(p_max), (p_max // TILE) * TILE // BLOCK)


@pytest.mark.parametrize(
    "clamp, ak, want",
    [(None, None, False), (None, "1", True), ("0", "1", False), ("1", None, True), ("", "1", True), ("1", "1", True)],
)
def test_kv_clamp_flag(monkeypatch, clamp, ak, want):
    for k, v in (("QWEN36_SPEC_SDPA_KV_CLAMP", clamp), ("QWEN36_SPEC_ADAPTIVE_K", ak)):
        if v is None:
            monkeypatch.delenv(k, raising=False)
        else:
            monkeypatch.setenv(k, v)
    assert tpc.spec_sdpa_kv_clamp_enabled() is want


def test_kv_clamp_kwargs(monkeypatch):
    monkeypatch.delenv("QWEN36_SPEC_ADAPTIVE_K", raising=False)
    monkeypatch.delenv("QWEN36_SPEC_SDPA_KV_CLAMP", raising=False)
    monkeypatch.setattr(atp, "SDPA_KV_CLAMP_SUPPORTED", False)
    assert atp.sdpa_kv_clamp_kwargs() == {}  # flag off: today's call, whatever the build
    monkeypatch.setenv("QWEN36_SPEC_SDPA_KV_CLAMP", "1")
    with pytest.raises(AssertionError):
        atp.sdpa_kv_clamp_kwargs()  # requested on a build without the op argument: refuse to run unguarded
    monkeypatch.setattr(atp, "SDPA_KV_CLAMP_SUPPORTED", True)
    assert atp.sdpa_kv_clamp_kwargs() == {"clamp_kv_tail": True}
    monkeypatch.setenv("QWEN36_SPEC_SDPA_KV_CLAMP", "0")
    monkeypatch.setenv("QWEN36_SPEC_ADAPTIVE_K", "1")
    assert atp.sdpa_kv_clamp_kwargs() == {}


def test_every_paged_decode_call_passes_the_clamp_kwargs():
    src = open(os.path.join(os.path.dirname(atp.__file__), "tp.py")).read()
    calls = [m.start() for m in re.finditer(r"ttnn\.transformer\.paged_scaled_dot_product_attention_decode\(", src)]
    assert len(calls) == 4, f"expected the 4 known call sites, found {len(calls)}"
    for start in calls:
        depth, i = 0, src.index("(", start)
        for j in range(i, len(src)):
            depth += {"(": 1, ")": -1}.get(src[j], 0)
            if depth == 0:
                break
        call = src[i : j + 1]
        line = src.count("\n", 0, start) + 1
        assert "**sdpa_kv_clamp_kwargs()" in call, f"tt/attention/tp.py:{line}: paged SDPA decode without the clamp kwargs"


def test_overread_rule_tight_rows():
    """Rows ceil((p_max + 1) / 64) wide (the old test_spec_adaptive_k_sdpa.py tables): over-read exactly when p_max >= 128
    and p_max % 128 < 64; never with the clamp."""
    for p_max in range(0, 70000):
        width = math.ceil((p_max + 1) / BLOCK)
        over = _last_block(p_max) >= width
        assert over == (p_max >= 128 and p_max % 128 < 64), p_max
        assert _clamped_last_block(p_max) < width
        assert _last_block(p_max) <= width  # at most ONE entry past the row (the page's padding when width % 16 != 0)


def test_overread_rule_serving_rows():
    """Serving rows: spec_table_width(262144, K) = 4097 for K <= 63. The legacy reader stays inside them for every
    p_max < 262144 and leaves them from 262144 on (r5: verify windows starting at 262141..262143, K = 3; adaptive-K wide
    family K = 5: 262139..262143); the clamp keeps every lookup inside up to the last position the row can hold."""
    for K in (3, 5, 7):
        w = spec_table_width(262144, K, BLOCK)
        assert w == 4097
        for p_max in range(260000, 262144 + 64):
            assert (_last_block(p_max) >= w) == (p_max >= 262144), (K, p_max)
            if p_max < w * BLOCK:
                assert _clamped_last_block(p_max) < w
        first_base = 262144 - K  # window [base, base + K] reaches p_max 262144
        assert _last_block(first_base + K) >= w and _last_block(first_base + K - 1) < w


def test_tight_plus_one_rows_never_overread():
    """The safety stage's tight + 1 layout (one spare entry per row) is in bounds for every p_max: the legacy reader
    resolves at most one entry past the row."""
    for p_max in range(0, 70000):
        assert _last_block(p_max) < math.ceil((p_max + 1) / BLOCK) + 1
