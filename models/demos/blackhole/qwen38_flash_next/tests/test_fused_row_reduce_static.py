# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The fused row-packed output reduce (``ttnn/fused/row_reduce``) without a device: its registry entry and switch, its
admission contract, the argument layout between the Python builder and the transport kernel, the zone markers, and the
summation tree of the compute kernel against the stock line reduce-scatter's tree (a model of the kernels' feed and
add order; the bits are pinned by tests/test_fused_row_reduce_device.py)."""

from __future__ import annotations

import inspect
import re
from types import SimpleNamespace

import pytest

import ttnn
from models.demos.blackhole.qwen38_flash_next.ttnn import fused
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import program as fp
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import row_reduce as rr

TRANSPORT = (fp.REPO_ROOT / rr.TRANSPORT_KERNEL).read_text()
COMPUTE = (fp.REPO_ROOT / rr.COMPUTE_KERNEL).read_text()
MODULE = inspect.getsource(rr)


def _partial(shape=(1, 1, 1, 2560), dtype=ttnn.bfloat16, ring=8, layout=ttnn.TILE_LAYOUT):
    padded = (*shape[:2], 32, shape[3])
    mesh = SimpleNamespace(shape=(1, ring))
    return SimpleNamespace(shape=shape, padded_shape=padded, dtype=dtype, layout=layout, device=lambda: mesh)


@pytest.fixture
def on(monkeypatch):
    monkeypatch.setenv(rr.ENV, "1")
    monkeypatch.setattr(rr.gr_fold, "line", lambda mesh: None)
    monkeypatch.setattr(rr.ttnn, "get_tt_fabric_max_payload_size_bytes", lambda: 4352, raising=False)
    monkeypatch.setattr(rr, "TP_SIZE", 8)


def test_registered_bitwise_opt_in_with_an_admission():
    entry = fused.kernel("row_reduce")
    assert entry.tolerance == fused.BITWISE
    assert entry.fused is rr.row_reduce and entry.composed is rr.composed and entry.admits is rr.admits
    assert not entry.default_on and "row_reduce" not in fused.DEFAULT_ON
    assert inspect.signature(rr.row_reduce).parameters.keys() == inspect.signature(rr.composed).parameters.keys()
    # on by default (x33 A/B); QWEN38_ROW_REDUCE=0 keeps the stock reduce-scatter
    assert rr.enabled_by_env({}) and rr.enabled_by_env({rr.ENV: "1"}) and not rr.enabled_by_env({rr.ENV: "0"})
    assert fused.resolve_admitted("row_reduce", {}) is entry.composed


def test_admission_contract(on):
    bf16, fp32 = ttnn.bfloat16, ttnn.float32
    assert rr.admits(_partial()) and rr.admits(_partial((1, 1, 2, 2560)))
    assert not rr.admits(_partial((1, 1, 3, 2560)))  # bf16 takes rows <= 2
    assert rr.admits(_partial(dtype=fp32)) and not rr.admits(_partial((1, 1, 2, 2560), dtype=fp32))  # fp32 rows <= 1
    assert not rr.admits(_partial((1, 1, 1, 2560 + 32)))  # tiles not a multiple of the ring
    assert not rr.admits(_partial(layout=ttnn.ROW_MAJOR_LAYOUT))
    assert not rr.admits(_partial(dtype=ttnn.bfloat8_b))
    assert not rr.admits(_partial(), cluster_axis=0)
    assert not rr.admits(_partial(ring=4)) and not rr.admits(_partial((2, 1, 1, 2560)))
    assert not rr.admits(_partial(), memory_config=ttnn.L1_MEMORY_CONFIG)


def test_off_admits_nothing(monkeypatch):
    monkeypatch.setenv(rr.ENV, "0")
    assert not rr.admits(_partial())


def test_geometry_and_packets():
    # the valid rows of faces 0 and 1 of every tile: bf16 row 32 B, fp32 row 64 B; 80 tiles of the 2560-wide partial
    for dtype, rows, packed_tile, total in (
        (ttnn.bfloat16, 1, 64, 5120),
        (ttnn.bfloat16, 2, 128, 10240),
        (ttnn.float32, 1, 128, 10240),
    ):
        p = _partial((1, 1, rows, 2560), dtype)
        r, tiles, out_tiles, chunk, packed, slot, scratch = rr._geometry(p)
        assert (r, tiles, packed, tiles * packed) == (rows, 80, packed_tile, total)
        assert out_tiles == 80 // rr.TP_SIZE and slot % 16 == 0 and slot >= total
        assert (scratch > 0) == (chunk % rr.READ_ALIGN != 0)  # only a 32 B chunk bounces
        # one packet per direction with a payload that holds the rows; 4352 B (the default) takes ceil(total / 4352)
        assert -(-total // 4352) == (2 if total == 5120 else 3) and -(-total // 15232) == 1


@pytest.mark.parametrize(
    "dtype, rows",
    [
        (ttnn.bfloat16, 1),
        pytest.param(
            ttnn.bfloat16,
            2,
            marks=pytest.mark.xfail(
                rr.TP_SIZE == 4,
                strict=True,
                reason="SUSPECTED REAL BUG (x65 report): at TP=4 a 2-row bf16 partial has out_tiles=20, 20 x 128 B > 2048 B; "
                "admits() accepts it and _row_reduce_v2 raises 'the dense tile of a slice does not fit a tile'",
            ),
        ),
        (ttnn.float32, 1),
    ],
)
def test_the_dense_tile_of_a_slice_fits_the_tile_page(dtype, rows):
    # (was an inline assert of test_geometry_and_packets; split so the rest of that test still runs at TP=4)
    _r, _tiles, out_tiles, _chunk, packed, _slot, _scratch = rr._geometry(_partial((1, 1, rows, 2560), dtype))
    assert out_tiles * packed <= fp.TILE_BYTES[dtype]


def test_transport_compile_args_match_the_builder():
    ids = [int(i) for i in re.findall(r"get_compile_time_arg_val\((\d+)\)", TRANSPORT)]
    base = int(re.search(r"ACCESSOR_BASE = (\d+);", TRANSPORT).group(1))
    extra = (
        re.search(r"def row_reduce\(partial.*?extra_args=\(([^)]*)\)", MODULE, re.S).group(1).split(",")
    )  # v1's builder
    extra = [e for e in extra if e.strip()]
    assert base == 11 + len(extra) == max(ids) + 1  # arg 23 is the last scalar; the accessors follow
    # the transport spec's own words: 0 CB_IN (scratch cb), 1 tiles, 2 go, 3 ring, then the extras from 11
    assert [
        get for get in re.findall(r"constexpr uint32_t (\w+) = get_compile_time_arg_val\((?:0|1|2|3|11)\);", TRANSPORT)
    ] == [
        "CB_IN",
        "TILES",
        "SEM_GO",
        "RING",
        "ROWS",
    ]
    compute_ids = sorted({int(i) for i in re.findall(r"get_compile_time_arg_val\((\d+)\)", COMPUTE)})
    assert compute_ids == list(range(9))
    assert "[CB_IN, CB_FS, CB_BS, CB_F, CB_B, CB_G, CB_OUT, 1, TP_SIZE]" in MODULE


def test_cbs_are_distinct():
    assert sorted((rr.CB_IN, rr.PACK_CB, rr.CB_FS, rr.CB_BS, rr.CB_F, rr.CB_B, rr.CB_G, rr.CB_OUT)) == list(range(8))


def test_zones_follow_the_fused_conventions():
    for text in (TRANSPORT, COMPUTE):
        assert '#include "../../kernels/zones.h"' in text and "DeviceZoneScopedN" not in text
        lines = text.split("\n")
        for i, line in enumerate(lines):
            if "FUSED_ZONE(" in line:
                assert re.fullmatch(r'FUSED_ZONE\("fz_rr_[a-z0-9_]+"\);', line.strip()), line
                previous = next(l for l in reversed(lines[:i]) if l.strip())
                assert re.sub(r"//.*", "", previous).rstrip().endswith("{"), (i + 1, previous)
    names = re.findall(r'FUSED_ZONE\("([^"]+)"\)', TRANSPORT + COMPUTE)
    assert len(names) == len(set(names)) == 7  # v1: transport (6) + compute (1); v2 is checked below


# ------------------------------------------------------------------------------ the tree: a model of the two kernels


def _kernels_tree(rank: int, ring: int, mirror: bool):
    """The sum of device ``rank``'s slice as the kernels build it: the feed's push order (transport.cpp) consumed by the
    compute kernel's add order (compute.cpp), with strings for the partials."""

    L = ring - 1 - rank if mirror else rank
    p = lambda s: (
        f"p{ring - 1 - s}" if mirror else f"p{s}"
    )  # logical source s is physical device ring-1-s when mirrored
    fs = [p(0)]  # CB_FS: p_0 (the own term for L == 0)
    bs = [p(ring - 1)] if L < ring - 1 else []
    cin = [p(s) for s in range(1, L + 1)] + ([p(s) for s in range(ring - 2, L, -1)] if L < ring - 1 else [])
    add = lambda a, b: f"({a}+{b})"
    own = fs[0]
    if L >= 1:
        acc = fs[0]
        for _ in range(1, L):
            acc = add(cin.pop(0), acc)
        own = add(cin.pop(0), acc)
    if L < ring - 1:
        acc = bs[0]
        for _ in range(ring - 2, L, -1):
            acc = add(cin.pop(0), acc)
        own = add(own, acc)
    assert not cin
    return own


def _stock_tree(d: int, ring: int):
    """out_d = (p_d + F_d) + B_d, F_d = p_{d-1} + (p_{d-2} + ... + (p_1 + p_0)), B_d = p_{d+1} + (... + (p_{R-2} + p_{R-1}))."""

    add = lambda a, b: f"({a}+{b})"
    out = f"p{d}"
    if d >= 1:
        f = "p0"
        for k in range(1, d):
            f = add(f"p{k}", f)
        out = add(out, f)
    if d < ring - 1:
        b = f"p{ring - 1}"
        for k in range(ring - 2, d, -1):
            b = add(f"p{k}", b)
        out = add(out, b)
    return out


@pytest.mark.parametrize("ring", [4, 8])
def test_kernel_order_is_the_stock_tree(ring):
    for d in range(ring):
        assert _kernels_tree(d, ring, mirror=False) == _stock_tree(d, ring), d
    # the mirror flips the line: device d takes the tree of device ring-1-d over the reversed partials
    for d in range(ring):
        flipped = _stock_tree(ring - 1 - d, ring)
        flipped = re.sub(r"p(\d+)", lambda m: f"p{ring - 1 - int(m.group(1))}", flipped)
        assert _kernels_tree(d, ring, mirror=True) == flipped, d


def test_the_kernels_implement_the_modelled_order():
    # the pins of the model: the feed's order and the compute's per-add operand pairs
    flat = re.sub(r"\s+", "", TRANSPORT)  # whitespace-agnostic: clang-format wraps these loop headers
    assert "for(uint32_ts=1;s<=L;++s,at+=tile_bytes)" in flat  # p_1 .. p_{L-1}, then the own term p_L
    assert "for(uint32_ts=RING-2;s>L&&L<RING-1;--s,at+=tile_bytes)" in flat  # p_{R-2} .. p_{L+1}
    assert "fill(cfs.get_write_ptr(),0);" in flat and "fill(cbs.get_write_ptr(),RING-1);" in flat
    assert "add_pop(CB_IN, acc, CB_F);" in COMPUTE and "add_pop(CB_IN, acc, CB_B);" in COMPUTE
    assert (
        "add_pop(CB_IN, acc, L == RING - 1 ? CB_OUT : CB_G);" in COMPUTE and "add_pop(own_sum, acc, CB_OUT);" in COMPUTE
    )
    # one add_tiles and one pack per binary add: the rounding after every add is the stock reduction's
    assert COMPUTE.count("add_tiles(") == 1 and COMPUTE.count("pack_tile(") == 1
    # fp32 accumulates in the fp32 dest (as the stock op's resolved HiFi4 + fp32_dest_acc config), bf16 in the 16-bit dest
    assert (
        "fp32_dest=dtype == ttnn.float32" in MODULE
        and "fidelity" not in MODULE.split("def compute")[1].split("def ")[0]
    )


def test_v2_transport_layout_and_parity_pins():
    text = (fp.REPO_ROOT / rr.TRANSPORT2_KERNEL).read_text()
    ids = sorted({int(i) for i in re.findall(r"get_compile_time_arg_val\((\d+)\)", text)})
    base = int(re.search(r"ACCESSOR_BASE = (\d+);", text).group(1))
    extra = re.search(r"def _row_reduce_v2.*?extra_args=\(([^)]*)\)", MODULE, re.S).group(1)
    assert base == 11 + len([e for e in re.sub(r"\s+", " ", extra).split(",") if e.strip()]) == max(ids) + 1
    assert rr.version({}) == 2 and rr.version({rr.ENV: "1"}) == 1 and rr.version({rr.ENV: "2"}) == 2  # v2 default
    assert rr.enabled_by_env({rr.ENV: "2"}) and not rr.enabled_by_env({rr.ENV: "0"})
    assert text.index("const uint32_t parity = *counter & 1;") < text.index(
        "noc_semaphore_wait_min(data, (RING - 1) * PACKETS);"
    )
    assert text.index("done.wait(1);") < text.index("*counter = *counter + 1;") and "arrive(" not in text
    assert text.count("fabric_multicast_noc_fused_unicast_with_atomic_inc(") == 1
    names = re.findall(r'FUSED_ZONE\("([^"]+)"\)', text)
    assert len(names) == len(set(names)) == 5 and all(n.startswith("fz_rr_v2_") for n in names)


def test_rr_write_kernels_layout_and_hooks():
    t3 = (fp.REPO_ROOT / rr.TRANSPORT3_KERNEL).read_text()
    c3 = (fp.REPO_ROOT / rr.COMPUTE3_KERNEL).read_text()
    hp = (fp.REPO_ROOT / rr.HELPER_KERNEL).read_text()
    ids = sorted({int(i) for i in re.findall(r"get_compile_time_arg_val\((\d+)\)", t3)})
    base = int(re.search(r"ACCESSOR_BASE = (\d+);", t3).group(1))
    extra = (
        MODULE.split("def _row_reduce_write", 1)[1]
        .split("extra_args=(", 1)[1]
        .split("*fp.accessor_args(residual)", 1)[0]
    )
    scalars = [e for e in re.sub(r"\s+", " ", extra).split(",") if e.strip()]
    assert base == 11 + len(scalars) == max(ids) + 1  # the three accessor lists follow the scalars
    assert sorted({int(i) for i in re.findall(r"get_compile_time_arg_val\((\d+)\)", c3)}) == list(range(7)) + list(
        range(8, 17)
    )
    assert "constexpr uint32_t HELPERS = 4;" in t3 and rr.BRANCHES == 4 and len(rr.HELPER_CORES) == 4
    # gr_write's op order in the helper's compute: copy, COL broadcast, SFPU mul, pack, FPU add, pack
    order = [
        "copy_tile(CB_A",
        "unary_bcast<BroadcastType::COL>",
        "mul_binary_tile(",
        "pack_tile(dst_upd",
        "add_tiles(CB_RES",
        "pack_tile(0, CB_NEW",
    ]
    positions = [c3.index(o) for o in order]
    assert positions == sorted(positions)
    assert (
        "unpack_to_dest_fp32=(CB_OUT,)" in MODULE
    )  # the narrowing reads the fp32 sum exactly (ttnn.typecast's unpack mode)
    for text in (t3, c3, hp):
        assert '#include "../../kernels/zones.h"' in text
    names = re.findall(r'FUSED_ZONE\("([^"]+)"\)', t3 + c3 + hp)
    assert len(names) == len(set(names)) == 10 and all(n.startswith("fz_rr_") for n in names)


def test_rr_write_switch_and_handoff():
    assert rr.write_enabled({}) and not rr.write_enabled({rr.RR_WRITE_ENV: "1", rr.ENV: "1"})  # default on; needs v2
    assert rr.write_enabled({rr.RR_WRITE_ENV: "1", rr.ENV: "2"}) and not rr.write_enabled(
        {rr.RR_WRITE_ENV: "0", rr.ENV: "2"}
    )
    state, block = object(), object()
    rr.WRITTEN[id(block)] = (block, state, "new")
    assert rr.take_written(block, object()) is None  # another state: not handed over
    rr.WRITTEN[id(block)] = (block, state, "new")
    assert rr.take_written(block, state) == "new" and rr.take_written(block, state) is None
