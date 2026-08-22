# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Sort-network library SIM acceptance (lane FG, X5).

Runs sources/sfpu_sortnet_probe.cpp -- the GENERATED sfpi_sortnet.h
bitonic networks, truncated per stage -- on the pinned simulator and
compares EVERY independent machine at EVERY stage against lane FB's
oracle (helpers/crosslane_oracle.py bitonic_sort_trace /
bitonic_sort_kv_trace, the source of crosslane_fixtures/
bitonic_stages.json).  The recorded fixture cases ride machines 0 and 1
verbatim, so the fixture traces are checked byte-for-byte; the remaining
machines carry varied splitmix stimuli (genericity: nothing depends on
the recorded values).

Element placement (matches sfpi_sortnet.h's machine geometry):
  bitonic_sort8    machine = lane l;   element e at (row e, lane l)
  bitonic_sort32   machine = column c; element e at (row e&7,
                                       lane 8*(e>>3) + c)
  bitonic_sort16_kv machine = column c; key e at (row e&3,
                                       lane 8*(e>>2) + c), payload at
                                       (row 4 + (e&3), same lane)

KV-32 is deliberately ABSENT: the library refuses it by name
(crosslane-kv32-register-file -- 32 keys + 32 companions = twice the
LReg file); the recorded n=32 KV fixture documents the spec that
refusal points away from.  The KV-16 network is validated against the
same oracle generator at n=16.  Tie caveat: stimuli are tie-free
(SFPSWAP tie behavior is the unadjudicated doc-vs-sim divergence).

Run: pytest -q --run-simulator test_crosslane_sortnet.py
"""

import json
import os
from dataclasses import dataclass

import pytest
import torch
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import DestAccumulation
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import TemplateParameter

from helpers import crosslane_oracle as co

M32 = 0xFFFFFFFF
ELEMS = 1024
ROWS = 16
LANES = 32

FIXDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "crosslane_fixtures")


@dataclass
class UIntTemplate(TemplateParameter):
    name: str
    value: int

    def convert_to_cpp(self) -> str:
        return f"constexpr std::uint32_t {self.name} = {self.value}u;"


def run_probe(net, order, stages, input_vec):
    formats = InputOutputFormat(DataFormat.UInt32, DataFormat.UInt32)
    src = torch.tensor(input_vec, dtype=torch.int64)
    config = TestConfig(
        "sources/sfpu_sortnet_probe.cpp",
        formats,
        templates=[
            UIntTemplate("SORT_NET", net),
            UIntTemplate("SORT_ORDER", order),
            UIntTemplate("SORT_STAGES", stages),
        ],
        runtimes=[],
        variant_stimuli=StimuliConfig(
            src,
            formats.input_format,
            torch.zeros_like(src),
            formats.input_format,
            formats.output_format,
            tile_count_A=1,
            tile_count_B=1,
            tile_count_res=4,
        ),
        dest_acc=DestAccumulation.Yes,
        unpack_to_dest=True,
        disable_format_inference=True,
        compile_time_formats=True,
    )
    res = config.run().result
    assert len(res) == 4 * ELEMS, f"expected 4 tiles back, got {len(res)}"
    return [int(v) & M32 for v in res]


# ---------------------------------------------------------------------------
# calibration (module-scoped; lane DS/FB's empirical method)
# ---------------------------------------------------------------------------


class Cal:
    T = None
    m1inv = None


@pytest.fixture(scope="module")
def cal():
    if Cal.T is not None:
        return Cal
    ramp = list(range(ELEMS))
    rowtag = run_probe(101, 0, 0, ramp)
    lanetag = run_probe(102, 0, 0, ramp)
    ident = run_probe(100, 0, 0, ramp)

    row_positions = {}
    for pos, v in enumerate(rowtag):
        if 0x00A00000 <= v < 0x00A00000 + ROWS:
            row_positions.setdefault(v - 0x00A00000, []).append(pos)
    assert sorted(row_positions.keys()) == list(range(ROWS))
    T = {}
    for i, positions in row_positions.items():
        assert len(positions) == LANES
        tags = [lanetag[p] for p in positions]
        assert sorted(tags) == [2 * l for l in range(LANES)]
        for k, p in enumerate(positions):
            T[(i, tags[k] // 2)] = p
    m1inv = {}
    for i in range(ROWS):
        for l in range(LANES):
            v = ident[T[(i, l)]]
            assert 0 <= v < ELEMS
            m1inv[(i, l)] = v
    assert len(set(m1inv.values())) == ROWS * LANES
    Cal.T, Cal.m1inv = T, m1inv
    return Cal


def build_input(cal, rows):
    vec = [0] * ELEMS
    for i, lane_vals in rows.items():
        for l in range(LANES):
            vec[cal.m1inv[(i, l)]] = lane_vals[l] & M32
    return vec


def read_rows(cal, out, indices):
    return {i: [out[cal.T[(i, l)]] for l in range(LANES)] for i in indices}


# ---------------------------------------------------------------------------
# stimuli: fixture cases on machines 0/1, tie-free varied elsewhere
# ---------------------------------------------------------------------------


def fixture_cases():
    with open(os.path.join(FIXDIR, "bitonic_stages.json")) as f:
        return json.load(f)["cases"]


def tie_free_keys(seed, n):
    out, seen, i = [], set(), 0
    while len(out) < n:
        r = co.splitmix32(seed * 7919 + i)
        i += 1
        x = ((r % 2000) - 1000) + (r >> 20) / 4096.0
        b = co.f32_to_bits(x if r & 1 else -x)
        if b in seen:
            continue
        seen.add(b)
        out.append(b)
    return out


def machines_sort8(order):
    """32 machines of 8; fixture seeds 51/52 on machines 0/1."""
    fixt = [c for c in fixture_cases()
            if not c["kv"] and c["n"] == 8 and c["order"] == order]
    ms = []
    for m in range(LANES):
        if m < len(fixt):
            ms.append(([int(x, 16) for x in fixt[m]["input"]], fixt[m]))
        else:
            ms.append((tie_free_keys(9000 + m, 8), None))
    return ms


def machines_sort32(order):
    fixt = [c for c in fixture_cases()
            if not c["kv"] and c["n"] == 32 and c["order"] == order]
    ms = []
    for m in range(8):
        if m < len(fixt):
            ms.append(([int(x, 16) for x in fixt[m]["input"]], fixt[m]))
        else:
            ms.append((tie_free_keys(9100 + m, 32), None))
    return ms


def machines_kv16():
    ms = []
    for m in range(8):
        keys = tie_free_keys(9200 + m, 16)
        pays = [(0xC0DE0000 | (m << 8) | e) for e in range(16)]
        ms.append((keys, pays))
    return ms


# ---------------------------------------------------------------------------
# the per-stage acceptance sweeps
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("order", ["asc", "desc"])
@pytest.mark.parametrize("stages", list(range(1, 7)))
def test_sortnet8_stage(cal, order, stages):
    ms = machines_sort8(order)
    rows = {e: [ms[m][0][e] for m in range(LANES)] for e in range(8)}
    out = run_probe(0, 0 if order == "asc" else 1, stages,
                    build_input(cal, rows))
    got = read_rows(cal, out, range(8))
    bad = []
    for m in range(LANES):
        _, trace = co.bitonic_sort_trace(ms[m][0], order)
        want = trace[stages - 1]
        if ms[m][1] is not None:
            fx = [int(x, 16) for x in ms[m][1]["stages"][stages - 1]]
            assert fx == [w & M32 for w in want], "fixture/oracle drift"
        for e in range(8):
            if got[e][m] != want[e] & M32:
                bad.append((m, e, got[e][m], want[e] & M32))
    if bad:
        print(f"SORTNET8 MISMATCH order={order} stages={stages}: "
              f"{len(bad)} elements")
        for m, e, g, w in bad[:16]:
            print(f"  machine={m} elem={e} got={g:08x} want={w:08x}")
    assert not bad


@pytest.mark.parametrize("order", ["asc", "desc"])
@pytest.mark.parametrize("stages", list(range(1, 16)))
def test_sortnet32_stage(cal, order, stages):
    ms = machines_sort32(order)
    rows = {}
    for g in range(8):
        rows[g] = [0] * LANES
        for l in range(LANES):
            r, c = l // 8, l % 8
            rows[g][l] = ms[c][0][r * 8 + g]
    out = run_probe(1, 0 if order == "asc" else 1, stages,
                    build_input(cal, rows))
    got = read_rows(cal, out, range(8))
    # mid-sandwich truncations (7 and 11..12 boundaries) already close
    # the sandwich in the header, so the readout is always in element
    # layout.
    bad = []
    for c in range(8):
        _, trace = co.bitonic_sort_trace(ms[c][0], order)
        want = trace[stages - 1]
        if ms[c][1] is not None:
            fx = [int(x, 16) for x in ms[c][1]["stages"][stages - 1]]
            assert fx == [w & M32 for w in want], "fixture/oracle drift"
        for e in range(32):
            g, r = e & 7, e >> 3
            v = got[g][r * 8 + c]
            if v != want[e] & M32:
                bad.append((c, e, v, want[e] & M32))
    if bad:
        print(f"SORTNET32 MISMATCH order={order} stages={stages}: "
              f"{len(bad)} elements")
        for c, e, g, w in bad[:16]:
            print(f"  machine={c} elem={e} got={g:08x} want={w:08x}")
    assert not bad


@pytest.mark.parametrize("order", ["asc", "desc"])
@pytest.mark.parametrize("stages", list(range(1, 11)))
def test_sortnet16_kv_stage(cal, order, stages):
    ms = machines_kv16()
    rows = {}
    for g in range(4):
        rows[g] = [0] * LANES
        rows[4 + g] = [0] * LANES
        for l in range(LANES):
            r, c = l // 8, l % 8
            rows[g][l] = ms[c][0][r * 4 + g]
            rows[4 + g][l] = ms[c][1][r * 4 + g]
    out = run_probe(2, 0 if order == "asc" else 1, stages,
                    build_input(cal, rows))
    got = read_rows(cal, out, range(8))
    bad = []
    for c in range(8):
        _, _, trace = co.bitonic_sort_kv_trace(ms[c][0], ms[c][1], order)
        wk, wp = trace[stages - 1]
        for e in range(16):
            g, r = e & 3, e >> 2
            vk = got[g][r * 8 + c]
            vp = got[4 + g][r * 8 + c]
            if vk != wk[e] & M32:
                bad.append(("k", c, e, vk, wk[e] & M32))
            if vp != wp[e] & M32:
                bad.append(("p", c, e, vp, wp[e] & M32))
    if bad:
        print(f"SORTNET16KV MISMATCH order={order} stages={stages}: "
              f"{len(bad)} elements")
        for kind, c, e, g, w in bad[:16]:
            print(f"  {kind} machine={c} elem={e} got={g:08x} want={w:08x}")
    assert not bad


@pytest.mark.parametrize("order", ["asc", "desc"])
def test_sortnet_fixture_sorted(cal, order):
    """Full networks reproduce the recorded fixtures' sorted outputs."""
    for n, net in ((8, 0), (32, 1)):
        ms = machines_sort8(order) if n == 8 else machines_sort32(order)
        nm = LANES if n == 8 else 8
        rows = {}
        nrows = 8
        for g in range(nrows):
            rows[g] = [0] * LANES
            for l in range(LANES):
                if n == 8:
                    rows[g][l] = ms[l][0][g]
                else:
                    r, c = l // 8, l % 8
                    rows[g][l] = ms[c][0][r * 8 + g]
        stages = 6 if n == 8 else 15
        out = run_probe(net, 0 if order == "asc" else 1, stages,
                        build_input(cal, rows))
        got = read_rows(cal, out, range(8))
        for m in range(nm):
            srt, _ = co.bitonic_sort_trace(ms[m][0], order)
            if ms[m][1] is not None:
                fx = [int(x, 16) for x in ms[m][1]["sorted"]]
                assert fx == [x & M32 for x in srt]
            for e in range(n):
                if n == 8:
                    v = got[e][m]
                else:
                    v = got[e & 7][(e >> 3) * 8 + m]
                assert v == srt[e] & M32, (
                    f"n={n} {order} machine {m} elem {e}: "
                    f"{v:08x} != {srt[e] & M32:08x}")
