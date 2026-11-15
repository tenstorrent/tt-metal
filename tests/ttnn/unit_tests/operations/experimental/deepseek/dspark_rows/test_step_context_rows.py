# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""``decode_step_context`` for a DSpark verify block of s = 4 rows: one call builds every per-row integer ([4] INT32,
plus the 16 causal ring rows) from the verify packet and every RoPE family's 4 rows as ``(4, 32)``-tiled
``[1, 1, 4, Rd]`` cos / sin / -sin, row j at position p + j.

* ``test_step_context_rows``: the integers exactly (host ``eval_term``), the RoPE rows per row within a bf16 ulp of
  float64 cos / sin, -sin == -sin, padding rows zero, the 1-row outputs next to them unchanged;
* ``test_step_context_rows_perf``: the production-shaped s = 1 step (``_step_context_fused``) against the s = 4
  verify step. Profile::

    python -m tracy -r -p -v --no-web-server -t 8123 -o <dir> -m pytest <this file>::test_step_context_rows_perf -s
    python3 dv_prof.py <dir>/reports/<ts>/ops_perf_results_*.csv StepContext
"""

import pytest
import torch

import ttnn
from models.experimental.deepseek_v41_flash.tt.decode.step_context import (
    StepContextBuilder,
    eval_term,
    rope_table,
    term,
)

RD, MAX_POS, WINDOW, TOPK, RATE, WIN_ROWS, STASH = 64, 1 << 20, 128, 512, 4, 4, 32
FAMILIES = {"main": (10000.0, 1.0), "compress": (160000.0, 1.25)}
T = 4
# Verify packet fields (as _verify_packet lays them out, abridged): tokens, positions, the closing groups' starts.
F_TOKEN, F_POS, F_START2, F_SLOT, F_RING_OFF, F_WIN_OFF = 0, 4, 8, 10, 11, 12
PKT_W = 16


def _inv_freq(rd: int, base: float) -> torch.Tensor:
    half = 1.0 / (base ** (torch.arange(0, rd, 2, dtype=torch.float64) / rd))
    return half.repeat_interleave(2).reshape(1, 1, 1, rd).to(torch.float32)


def _tables(device) -> dict:
    return {
        name: ttnn.from_torch(
            rope_table(_inv_freq(RD, base), scale, MAX_POS),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        for name, (base, scale) in FAMILIES.items()
    }


def _packet(device, host: list[int]) -> ttnn.Tensor:
    return ttnn.from_torch(
        torch.tensor(host, dtype=torch.int32).reshape(1, 1, 1, -1),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _verify_host_packet(pos: int) -> list[int]:
    pkt = [0] * PKT_W
    for j in range(T):
        pkt[F_TOKEN + j] = 1000 + 7 * j
        pkt[F_POS + j] = pos + j
    # Ratio 2 closes the groups ending at the odd positions of the block (2 of them).
    starts = [p + 1 - 2 for p in range(pos, pos + T) if (p + 1) % 2 == 0]
    for i, s in enumerate(starts):
        pkt[F_START2 + i] = s
    pkt[F_SLOT], pkt[F_RING_OFF], pkt[F_WIN_OFF] = 3, 640, 8
    return pkt


def _verify_ints() -> dict:
    """The verify block's per-row integers (what ``_verify_packet`` holds precomputed), as terms of the positions."""
    rows = range(T)
    pos = [F_POS + j for j in rows]
    ints = {
        "token": ([1, T], [term(F_TOKEN + j) for j in rows], ttnn.uint32),
        "pos": ([T], [term(p) for p in pos], ttnn.int32),
        "sliding_pos": ([T], [term(p, mode="mod", div=WINDOW) for p in pos], ttnn.int32),
        # Row j reads ring rows (pos + i) % W for i <= j, -1 past it.
        "ring_rows": (
            [T * T],
            [
                term(F_POS, pre_add=i, mode="mod", div=WINDOW) if i <= j else term(-1, post_add=-1)
                for j in rows
                for i in rows
            ],
            ttnn.int32,
        ),
        "sliding_curpos": ([T], [term(p, clamp_max=WINDOW - 1) for p in pos], ttnn.int32),
        "stash_slot": ([T], [term(p, mode="mod", div=STASH) for p in pos], ttnn.int32),
        "page_row": ([1], [term(F_SLOT)], ttnn.int32),
    }
    for r in (1, 2):
        ints[f"curpos{r}"] = (
            [T],
            [term(p, pre_add=1, mode="div", div=r, post_add=WINDOW - 1, clamp_max=WINDOW + TOPK - 1) for p in pos],
            ttnn.int32,
        )
        ints[f"index_pos{r}"] = (
            [T],
            [term(p, pre_add=1, mode="div", div=r, mul=RATE, post_add=-1) for p in pos],
            ttnn.int32,
        )
        ints[f"win_slot{r}"] = (
            [T],
            [term(p, mode="mod", div=WIN_ROWS if r > 1 else 1, add_field=F_WIN_OFF) for p in pos],
            ttnn.int32,
        )
    ints["entry1"] = ([T], [term(p) for p in pos], ttnn.int32)
    ints["entry2"] = ([2], [term(F_START2 + i, mode="div", div=2) for i in range(2)], ttnn.int32)
    return ints


def _verify_ropes() -> list:
    """(key, family, field, offset, rows, tile height): every family's 4 rows in (4, 32) tiles, ratio 2's closing
    groups' window rows (2, at their starts) in 32 x 32 tiles, and a legacy 1-row triple."""
    return [
        ("main", "main", F_POS, 0, T, T),
        ("compress", "compress", F_POS, 0, T, T),
        ("win2", "compress", F_START2, 0, 2, 32),
        ("prev", "compress", F_POS, -1, 1, 32),
    ]


def _build(pkt, tables, ints: dict, ropes: list) -> dict:
    b = StepContextBuilder(pkt, tables)
    for key, (shape, terms, dtype) in ints.items():
        b.add_int(key, shape, terms, dtype)
    for key, family, field, offset, rows, height in ropes:
        b.add_rope(key, family, field, offset, rows=rows, tile_height=height)
    return b.run()


@pytest.mark.parametrize("pos", (0, 1, 126, 1022, 4095, 65537, MAX_POS - 4))
def test_step_context_rows(device, pos):
    host = _verify_host_packet(pos)
    ints, ropes = _verify_ints(), _verify_ropes()
    pkt, tables = _packet(device, host), _tables(device)
    for _ in range(2):  # the second call runs from the program cache
        out = _build(pkt, tables, ints, ropes)
        for key, (shape, terms, dtype) in ints.items():
            got = ttnn.to_torch(out[key]).to(torch.int64).reshape(-1).tolist()
            want = [eval_term(t, host) for t in terms]
            assert list(out[key].shape) == shape and out[key].dtype == dtype, key
            assert got == want, (key, got, want)
        for key, family, field, offset, rows, height in ropes:
            base, scale = FAMILIES[family]
            inv = _inv_freq(RD, base).to(torch.float64).reshape(-1)
            ulp = 2.0**-7 * scale
            cos, sin, neg_sin = out[key]
            for t in (cos, sin, neg_sin):
                assert t.layout == ttnn.TILE_LAYOUT and t.dtype == ttnn.bfloat16, key
                assert list(t.shape) == [1, 1, rows, RD], (key, t.shape)
                assert t.tile.tile_shape == [height, 32], (key, t.tile.tile_shape)
            cos, sin, neg_sin = (ttnn.to_torch(t).to(torch.float64).reshape(rows, RD) for t in (cos, sin, neg_sin))
            assert torch.equal(neg_sin, -sin), key
            for j in range(rows):
                p = host[field + j] + offset
                cos_want = (torch.cos(p * inv) * scale).to(torch.bfloat16).to(torch.float64)
                sin_want = (torch.sin(p * inv) * scale).to(torch.bfloat16).to(torch.float64)
                for got, want, name in ((cos[j], cos_want, "cos"), (sin[j], sin_want, "sin")):
                    err = (got - want).abs().max().item()
                    assert err <= ulp, (key, name, j, p, err)


def _step_ints_s1() -> dict:
    """``_step_context_fused``'s integers for a sliding + ratio 1/2 stage that runs the r=2 pool and select."""
    lengths = lambda r: [  # noqa: E731
        term(2, pre_add=1, clamp_max=WINDOW),
        term(2, pre_add=1, mode="div", div=r, clamp_max=TOPK) if r else term(-1, mul=0),
        term(4),
    ]
    ints = {
        "token": ([1, 1], [term(0)], ttnn.uint32),
        "sliding_pos": ([1], [term(1)], ttnn.int32),
        "page_row": ([1], [term(3)], ttnn.int32),
        "ring_offset": ([1], [term(4)], ttnn.int32),
        "win_offset": ([1], [term(5)], ttnn.int32),
    }
    for r in (0, 1, 2):
        ints[f"lengths{r}"] = ([1, 1, 1, 3], lengths(r), ttnn.int32)
    ints["win_slot2"] = ([1], [term(2, mode="mod", div=WIN_ROWS, add_field=5)], ttnn.int32)
    for r in (1, 2):
        ints[f"entry{r}"] = ([1], [term(2, mode="div", div=r)], ttnn.int32)
        ints[f"index_pos{r}"] = ([1], [term(2, pre_add=1, mode="div", div=r, mul=RATE, post_add=-1)], ttnn.int32)
    ints["stash_slot"] = ([1], [term(2, mode="mod", div=STASH)], ttnn.int32)
    return ints


@pytest.mark.parametrize("s", (1, 4))
def test_step_context_rows_perf(device, s):
    from tracy import signpost

    tables = _tables(device)
    if s == 1:
        pkt = _packet(device, [77, 5, 1000, 3, 640, 9] + [0] * 10)
        ints = _step_ints_s1()
        ropes = [
            ("main", "main", 2, 0, 1, 32),
            ("compress", "compress", 2, 0, 1, 32),
            ("win2", "compress", 2, -1, 1, 32),
        ]
    else:
        pkt = _packet(device, _verify_host_packet(1000))
        ints = _verify_ints()
        ropes = [("main", "main", F_POS, 0, T, T), ("compress", "compress", F_POS, 0, T, T)]
        ropes.append(("win2", "compress", F_START2, 0, 2, 32))
    _build(pkt, tables, ints, ropes)
    ttnn.synchronize_device(device)
    signpost(f"step_context-s{s}")
    for _ in range(20):
        _build(pkt, tables, ints, ropes)
    ttnn.synchronize_device(device)
