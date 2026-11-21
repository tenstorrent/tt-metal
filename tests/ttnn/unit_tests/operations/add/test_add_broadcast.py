# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 1: broadcasting (over_w, over_h, over_hw, over_leading) and a Python-number operand.

1. Regime pinning: each broadcast value selects its regime (Col / Row / Scalar resident-b walk, leading page
   remap, number), and the result is within 1 bf16 ulp of the exact sum, either operand order (R4).
2. Structural count (the Done-when): b's pages read per cluster equal the distinct b pages its out range needs
   (never more than one b slice), for every broadcast entry of the specification's INPUTS; b's ring holds
   RING_DEPTH_BLOCKS of the largest window.
3. The walk under stress: cluster ranges that start mid-row and cross rows and images (MIN_TILES_PER_CORE turned
   up), and resident windows chunked (BCAST_WINDOW_TILES turned down), all into a NaN-prefilled output.
4. Number (R7): taken as its float32 value, exact on elements that nearly cancel it.
5. Thread plans: leading / number at (4, 4, 2) into a NaN-prefilled output; resident regimes refuse C > 1.
6. R9 placement of a broadcast result; two operands that both broadcast are refused.
7. One device session: a tt-metal program before, every regime, and tt-metal's to_layout after.
"""

import pytest
import torch

import ttnn
from ttnn.operations.add import add, default_compute_kernel_config
from ttnn.operations.add import add_program_artifacts as P
from ttnn.operations.add.add_program_artifacts import create_program_artifacts, io_tensors

from eval.memory import build, place
from eval.specs.add.spec import INPUTS, tag_broadcast


def _put(t, device, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config)


def _host(shape, seed):
    torch.manual_seed(seed)
    if shape == ():
        return torch.tensor(float(torch.randn(1).item()), dtype=torch.float32)
    return torch.randn(*shape).to(torch.bfloat16)


def _ordered(bits16):
    b = bits16.to(torch.int32) & 0xFFFF
    return torch.where(b >= 0x8000, 0x8000 - b, b)


def _check_ulp(got, ta, tb, max_ulp=1):
    """At most `max_ulp` bf16 steps from the exact sum rounded once to bf16 (the specification's grading)."""
    assert not torch.isnan(got).any(), "NaN pages left: a group or a thread did not write its pages"
    ref = (ta.double() + tb.double()).to(torch.bfloat16)
    g = got.to(torch.bfloat16)
    dist = (_ordered(g.view(torch.int16)) - _ordered(ref.view(torch.int16))).abs()
    assert int(dist.max()) <= max_ulp, f"max {int(dist.max())} ulp, at {int((dist > max_ulp).sum())} elements"


def _operand(t, device, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    return float(t) if t.dim() == 0 else _put(t, device, memory_config)


class _Shape:
    def __init__(self, shape):
        self.shape = shape


def _mode(sa, sb):
    return P.plan_operands(_Shape(sa), float(1.0) if sb == () else _Shape(sb))


REGIMES = [
    ((1, 1, 32, 64), (1, 1, 32, 1), P.MODE_COL, False),
    ((4, 8, 32, 256), (4, 8, 32, 1), P.MODE_COL, False),
    ((1, 1, 17, 50), (1, 1, 17, 1), P.MODE_COL, False),
    ((1, 1, 32, 1), (1, 1, 32, 64), P.MODE_COL, True),
    ((2, 4, 128, 512), (2, 4, 1, 512), P.MODE_ROW, False),
    ((1, 1, 50, 128), (1, 1, 1, 128), P.MODE_ROW, False),
    ((1, 1, 128, 512), (512,), P.MODE_ROW, False),
    ((1, 512), (128, 512), P.MODE_ROW, True),
    ((1, 1, 64, 128), (1, 1, 1, 1), P.MODE_SCALAR, False),
    ((4, 128, 512), (4, 1, 1), P.MODE_SCALAR, False),
    ((4, 8, 32, 256), (1, 8, 32, 256), P.MODE_LEADING, False),
    ((4, 8, 32, 64), (32, 64), P.MODE_LEADING, False),
    ((1, 1, 17, 50), (), P.MODE_NUMBER, False),
    ((4, 8, 32, 256), (), P.MODE_NUMBER, False),
]


@pytest.mark.parametrize("sa,sb,mode,swap", REGIMES, ids=lambda v: str(v))
def test_regime(device, sa, sb, mode, swap):
    got_swap, got_mode, _ = _mode(sa, sb)
    assert (got_mode, got_swap) == (mode, swap)
    ta, tb = _host(sa, 1), _host(sb, 2)
    out = add(_put(ta, device), _operand(tb, device))
    assert list(out.shape) == list(torch.broadcast_shapes(tuple(sa), tuple(sb)))
    _check_ulp(ttnn.to_torch(out).float(), ta, tb)


def _pages(shape):
    s = (1,) * (4 - len(shape)) + tuple(shape)
    return s[0] * s[1] * -(-s[2] // 32) * -(-s[3] // 32)


def _b_page(mode, g, p):
    """b's page of out page p, straight from the broadcast rule (independent of the walk)."""
    hbt = 1 if mode in (P.MODE_ROW, P.MODE_SCALAR) else g["Ht"]
    wbt = 1 if mode in (P.MODE_COL, P.MODE_SCALAR) else g["Wt"]
    img = g["Ht"] * g["Wt"]
    n, r, c = p // img, (p // g["Wt"]) % g["Ht"], p % g["Wt"]
    return P._nb(g, n) * hbt * wbt + (r if hbt > 1 else 0) * wbt + (c if wbt > 1 else 0)


BCAST_ENTRIES = [(sa, sb) for sa, sb in INPUTS if tag_broadcast((sa, sb), {}) not in ("none", "number")]


@pytest.mark.parametrize("sa,sb", BCAST_ENTRIES, ids=lambda v: "x".join(map(str, v)))
def test_b_read_once_per_cluster(device, sa, sb):
    """The Done-when count: per cluster, b pages read == distinct b pages its out range needs (<= one b slice);
    the runs cover the range exactly, in order, and index b inside their windows; b's ring is sized so that no
    window straddles its end (Row: it holds every window of the cluster; Col / Scalar: one-tile windows)."""
    _, mode, g = _mode(sa, sb)
    out_shape = tuple(torch.broadcast_shapes(tuple(sa), tuple(sb)))
    _, counts, starts = P.split_pages(_pages(out_shape), device.compute_with_storage_grid_size(), 1)
    for core in counts:
        st, ct = starts[core], counts[core]
        needed = {_b_page(mode, g, p) for p in range(st, st + ct)}
        assert P.b_pages_read(mode, g, st, ct) == len(needed), (core, st, ct)
        if mode in P.RESIDENT_B_MODES:
            runs = list(P.bcast_runs(mode, g, st, ct))
            assert [q for r in runs for q in range(r[0], r[0] + r[1])] == list(range(st, st + ct))
            for first, ln, _, _, win_first, win_tiles, lo in runs:
                for p in range(first, first + ln):
                    idx = (p % g["Wt"] - lo) if mode == P.MODE_ROW else 0
                    assert 0 <= idx < win_tiles and win_first + idx == _b_page(mode, g, p)
    ta, tb = _host(sa, 3), _host(sb, 4)
    a, b = _put(ta, device), _put(tb, device)
    out = ttnn.allocate_tensor_on_device(
        ttnn.TensorSpec(ttnn.Shape(out_shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.DRAM_MEMORY_CONFIG), device
    )
    spec, _, _ = create_program_artifacts(a, b, out, default_compute_kernel_config())
    if mode in P.RESIDENT_B_MODES:
        dfb_b = next(d for d in spec.dataflow_buffers if d.unique_id == P.CB_INPUT_B)
        dfb_a = next(d for d in spec.dataflow_buffers if d.unique_id == P.CB_INPUT_A)
        block = dfb_a.num_entries // P.RING_DEPTH_BLOCKS
        if mode == P.MODE_ROW:  # never wraps: every window of the busiest cluster
            assert dfb_b.num_entries == max(P.b_pages_read(mode, g, starts[c], counts[c]) for c in counts)
        else:
            assert dfb_b.num_entries == P.RING_DEPTH_BLOCKS * block + 1


def _nan_out(device, shape):
    return _put(torch.full(shape, float("nan"), dtype=torch.bfloat16), device)


def _run_knobs(device, ta, tb, **knobs):
    a, b = _put(ta, device), _operand(tb, device)
    out_shape = tuple(torch.broadcast_shapes(tuple(ta.shape), tuple(tb.shape)))
    out = _nan_out(device, out_shape)
    spec, run_args, idx = create_program_artifacts(a, b, out, default_compute_kernel_config(), **knobs)
    return ttnn.to_torch(ttnn.generic_op(io_tensors(a, b, out), spec, run_args, idx)).float()


# Ranges that start mid-row and cross rows and images (min tiles per core); a small window cap forces Row's
# one-tile-window fallback.
STRESS = [
    ((3, 1, 64, 224), (1, 1, 1, 224)),  # Row, b's row shared across images (nb constant)
    ((3, 1, 64, 224), (3, 1, 1, 224)),  # Row, a new window per image
    ((2, 3, 96, 160), (1, 3, 1, 160)),  # Row, nb cycling over the inner leading dim
    ((2, 1, 160, 96), (2, 1, 160, 1)),  # Col, rows across images
    ((1, 1, 320, 64), (1, 1, 320, 1)),  # Col, many full rows in one image
    ((4, 3, 64, 64), (4, 1, 1, 1)),  # Scalar, nb runs of 3 images
    ((2, 3, 64, 96), (1, 3, 64, 96)),  # leading remap
]


@pytest.mark.parametrize("min_tiles,window_cap", [(1, 128), (7, 128), (13, 2), (40, 3)], ids=lambda v: str(v))
@pytest.mark.parametrize("sa,sb", STRESS, ids=lambda v: "x".join(map(str, v)))
def test_walk_stress(device, sa, sb, min_tiles, window_cap):
    ta, tb = _host(sa, 5), _host(sb, 6)
    got = _run_knobs(device, ta, tb, min_tiles_per_core=min_tiles, window_cap=window_cap)
    _check_ulp(got, ta, tb)


@pytest.mark.parametrize("block_tiles,ring_depth", [(1, 2), (2, 3), (4, 2)], ids=str)
@pytest.mark.parametrize("sa,sb", [((2, 1, 160, 96), (2, 1, 160, 1)), ((3, 1, 64, 224), (1, 1, 1, 224)), ((1, 1, 17, 50), ())], ids=str)
def test_block_and_depth_knobs(device, sa, sb, block_tiles, ring_depth):
    ta, tb = _host(sa, 7), _host(sb, 8)
    got = _run_knobs(device, ta, tb, block_tiles=block_tiles, ring_depth=ring_depth, min_tiles_per_core=9)
    _check_ulp(got, ta, tb)


@pytest.mark.parametrize("number", [0.1, -1.7, 3.0e-3, 1234.567, 1.0 / 3.0], ids=str)
def test_number_is_float32_exact_on_cancellation(device, number):
    """R7: the number as a float32 value. Elements of a close to -number make the sum small, where a number
    rounded to bf16 (or tf32) first would be many ulp off; the exact fp32 SFPU add stays within 1 ulp."""
    n32 = torch.tensor(number, dtype=torch.float32)
    torch.manual_seed(9)
    ta = (-n32 + torch.randn(1, 1, 64, 128) * abs(number) * 0.01).to(torch.bfloat16)
    out = add(_put(ta, device), number)
    _check_ulp(ttnn.to_torch(out).float(), ta, n32)


@pytest.mark.parametrize("sa,sb", [((1, 3, 320, 320), (1, 1, 320, 320)), ((1, 3, 320, 320), ())], ids=str)
def test_thread_plan_streamed_regimes(device, sa, sb):
    ta, tb = _host(sa, 10), _host(sb, 11)
    _check_ulp(_run_knobs(device, ta, tb, threads=(4, 4, 2)), ta, tb)


def test_resident_regimes_need_plan_one(device):
    a, b = _put(_host((1, 1, 64, 64), 0), device), _put(_host((1, 1, 1, 64), 0), device)
    out = _nan_out(device, (1, 1, 64, 64))
    with pytest.raises(ValueError):
        create_program_artifacts(a, b, out, default_compute_kernel_config(), threads=(4, 4, 2))


def test_both_operands_broadcast_refused(device):
    a, b = _put(_host((1, 1, 32, 1), 0), device), _put(_host((1, 1, 1, 64), 0), device)
    with pytest.raises(NotImplementedError):
        add(a, b)


@pytest.mark.parametrize("sharding", ["height", "width"])
def test_broadcast_result_placement(device, sharding):
    """R9: a's sharded config when a has the output's shape; b's (the output-shaped operand) when a broadcasts."""
    sa, sb = (1, 1, 64, 128), (1, 1, 1, 128)
    ta, tb = _host(sa, 12), _host(sb, 13)
    a = _put(ta, device, place(build("L1", sharding), sa, ttnn.TILE_LAYOUT, device))
    b = _put(tb, device, place(build("L1", sharding), sb, ttnn.TILE_LAYOUT, device))
    out = add(a, b)
    assert out.memory_config() == a.memory_config()
    _check_ulp(ttnn.to_torch(out).float(), ta, tb)
    out_swapped = add(b, a)
    assert out_swapped.memory_config() == a.memory_config()
    _check_ulp(ttnn.to_torch(out_swapped).float(), ta, tb)
    n = add(a, 0.25)
    assert n.memory_config() == a.memory_config()
    _check_ulp(ttnn.to_torch(n).float(), ta, torch.tensor(0.25))
    for t, buf in ((add(_put(ta, device, ttnn.L1_MEMORY_CONFIG), _put(tb, device, ttnn.L1_MEMORY_CONFIG)), ttnn.BufferType.L1),):
        assert t.memory_config().buffer_type == buf


def test_one_device_session_every_regime(device):
    """tt-metal's untilize before, then every regime in one session (each leaves its tag handed back), then
    tt-metal's to_layout on each result, issuing untagged NoC transfers."""
    sa = (1, 1, 64, 128)
    ta = _host(sa, 14)
    a = _put(ta, device, ttnn.L1_MEMORY_CONFIG)
    assert torch.equal(ttnn.to_torch(ttnn.to_layout(a, ttnn.ROW_MAJOR_LAYOUT)), ta)
    results = []
    for sb in [(1, 1, 64, 1), (1, 1, 1, 128), (1, 1, 1, 1), (2, 1, 64, 128), ()]:
        tb = _host(sb, 15)
        if sb == (2, 1, 64, 128):  # a broadcasts over the leading dim: operands swapped
            out = add(a, _put(tb, device, ttnn.L1_MEMORY_CONFIG))
        else:
            out = add(a, _operand(tb, device, ttnn.L1_MEMORY_CONFIG))
        results.append((out, tb))
    for out, tb in results:
        rm = ttnn.to_layout(out, ttnn.ROW_MAJOR_LAYOUT)
        _check_ulp(ttnn.to_torch(ttnn.from_device(rm)).float(), ta, tb)
    # and an add of an add's result, then tt-metal again
    tb = _host((1, 1, 1, 128), 16)
    b = _put(tb, device, ttnn.L1_MEMORY_CONFIG)
    twice = ttnn.to_layout(add(add(a, b), b), ttnn.ROW_MAJOR_LAYOUT)
    exp = ((ta.float() + tb.float()).to(torch.bfloat16).float() + tb.float()).to(torch.bfloat16).float()
    assert torch.allclose(ttnn.to_torch(ttnn.from_device(twice)).float(), exp, rtol=1e-2, atol=1e-2)
