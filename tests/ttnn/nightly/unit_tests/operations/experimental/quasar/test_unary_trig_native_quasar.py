# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""ttnn.sin / cos / tan on Quasar: from a tile count per core up, a lone op on a DRAM-interleaved TILE tensor runs
through the native binary program as add(x, -0.0) with the op as a post activation (four Neos instead of one).

These tests check which device operation runs on each side of the cutoffs and for each configuration the routing
leaves alone, and that a routed result is bit-identical to the unary program's on the same data.

Run on the Quasar simulator:
    TT_METAL_SIMULATOR=<path>/libttsim.so TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_DISABLE_SFPLOADMACRO=1 \
        pytest tests/ttnn/nightly/unit_tests/operations/experimental/quasar/test_unary_trig_native_quasar.py
"""

import math
import os
import subprocess
import sys
import textwrap

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc
from tt_py_test_utils_common.utility_functions import is_quasar

# The routing runs only at the native binary program's default configuration.
_KNOBS_SET = os.environ.get("TTNN_QSR_NATIVE", "") not in ("", "1") or any(
    k.startswith("TTNN_QSR_") and k != "TTNN_QSR_NATIVE" and os.environ[k] for k in os.environ
)
pytestmark = pytest.mark.skipif(
    not is_quasar() or _KNOBS_SET, reason="Quasar, with the native binary program at its default configuration"
)

TORCH = {"sin": torch.sin, "cos": torch.cos, "tan": torch.tan}
TTNN_DTYPE = {torch.bfloat16: ttnn.bfloat16, torch.float32: ttnn.float32}
# Tiles per core from which the op is routed (unary.cpp); fp32 sin / cos never are.
CUTOFF = {
    ("sin", torch.bfloat16): 96,
    ("cos", torch.bfloat16): 64,
    ("tan", torch.bfloat16): 64,
    ("tan", torch.float32): 64,
}


def _cores(device):
    grid = device.compute_with_storage_grid_size()
    return grid.x * grid.y


def _shape(tiles):
    # One tile row, so the tile count is exact.
    return (1, 1, 32, 32 * tiles)


def _input(shape, dtype):
    x = torch.rand(shape) * 20 - 10
    x.view(-1)[:6] = torch.tensor([0.0, -0.0, math.inf, -math.inf, math.nan, 1e-40])
    return x.to(dtype)


def _to_device(device, x, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    return ttnn.from_torch(x, dtype=TTNN_DTYPE[x.dtype], layout=layout, device=device, memory_config=memory_config)


def _device_ops(call):
    ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
    out = call()
    graph = ttnn.graph.end_graph_capture()
    ops = {
        node["params"]["name"]
        for node in graph
        if node.get("node_type") == "function_start"
        and node.get("params", {}).get("name", "").endswith("DeviceOperation")
    }
    return out, ops


def _run(device, op, x):
    out, ops = _device_ops(lambda: getattr(ttnn, op)(_to_device(device, x)))
    return ttnn.to_torch(out).to(x.dtype), ops


def _same_bits(a, b):
    view = torch.int16 if a.dtype == torch.bfloat16 else torch.int32
    return bool(((a.view(view) == b.view(view)) | (torch.isnan(a) & torch.isnan(b))).all())


def _check_routed(device, op, x):
    routed, ops = _run(device, op, x)
    assert ops == {"BinaryNgDeviceOperation"}, f"{op} {x.dtype} with {x.numel() // 1024} tiles ran {ops}"

    # The same data in two halves, each below the cutoff, so each stays on the unary program.
    width = x.shape[-1] // 2
    sliced = []
    for i in range(2):
        part, part_ops = _run(device, op, x[..., i * width : (i + 1) * width].contiguous())
        assert part_ops == {"UnaryDeviceOperation"}, f"{op} {x.dtype} half ran {part_ops}"
        sliced.append(part)
    assert _same_bits(
        routed, torch.cat(sliced, dim=-1)
    ), f"{op} {x.dtype}: routed result differs from the unary program"

    finite = torch.isfinite(x) & (x.abs() < 1e30)
    assert_with_pcc(TORCH[op](x.float())[finite], routed.float()[finite], 0.999)


@pytest.mark.parametrize("op, dtype", list(CUTOFF))
def test_routed_matches_unary(device, op, dtype):
    torch.manual_seed(0)
    _check_routed(device, op, _input(_shape(CUTOFF[(op, dtype)] * _cores(device)), dtype))


def test_routed_padded_shape_matches_unary(device):
    # Batch dims and a padded height and width; the two batches below run on the unary program.
    torch.manual_seed(0)
    k = math.ceil(CUTOFF[("cos", torch.bfloat16)] * _cores(device) / 24)
    x = _input((2, 3, 100, 32 * k + 7), torch.bfloat16)
    routed, ops = _run(device, "cos", x)
    assert ops == {"BinaryNgDeviceOperation"}, f"cos on {tuple(x.shape)} ran {ops}"
    halves = []
    for i in range(2):
        part, part_ops = _run(device, "cos", x[i : i + 1].contiguous())
        assert part_ops == {"UnaryDeviceOperation"}, f"cos on {tuple(x[i : i + 1].shape)} ran {part_ops}"
        halves.append(part)
    assert _same_bits(routed, torch.cat(halves, dim=0)), "cos on a padded shape: routed result differs from unary"


def test_cutoff_rounds_tiles_per_core_up(device):
    # One tile past (cutoff - 1) tiles a core: the busiest core holds the cutoff, so the op routes.
    torch.manual_seed(0)
    _check_routed(
        device, "cos", _input(_shape((CUTOFF[("cos", torch.bfloat16)] - 1) * _cores(device) + 1), torch.bfloat16)
    )


@pytest.mark.parametrize("op, dtype", list(CUTOFF))
def test_below_cutoff_stays_unary(device, op, dtype):
    torch.manual_seed(0)
    _, ops = _run(device, op, _input(_shape((CUTOFF[(op, dtype)] - 1) * _cores(device)), dtype))
    assert ops == {"UnaryDeviceOperation"}, f"{op} {dtype} below its cutoff ran {ops}"


@pytest.mark.parametrize("op", ["sin", "cos"])
def test_fp32_sin_cos_stay_unary(device, op):
    torch.manual_seed(0)
    _, ops = _run(device, op, _input(_shape(128 * _cores(device)), torch.float32))
    assert ops == {"UnaryDeviceOperation"}, f"{op} float32 ran {ops}"


@pytest.mark.parametrize(
    "config",
    [
        "l1_input",
        "l1_input_dram_output",
        "l1_output",
        "width_sharded",
        "width_sharded_dram_output",
        "row_major",
        "output_tensor",
        "sub_core_grids",
        "op_chain",
    ],
)
def test_unrouted_configs_stay_unary(device, config):
    torch.manual_seed(0)
    grid = device.compute_with_storage_grid_size()
    tiles_per_core = CUTOFF[("tan", torch.bfloat16)]
    shape = _shape(tiles_per_core * _cores(device))
    x = _input(shape, torch.bfloat16)
    tan = ttnn.UnaryWithParam(ttnn.UnaryOpType.TAN)
    width_sharded = ttnn.create_sharded_memory_config(
        (32, 32 * tiles_per_core),
        core_grid=ttnn.CoreGrid(y=grid.y, x=grid.x),
        strategy=ttnn.ShardStrategy.WIDTH,
        use_height_and_width_as_shard_shape=True,
    )
    calls = {
        "l1_input": lambda: ttnn.tan(_to_device(device, x, memory_config=ttnn.L1_MEMORY_CONFIG)),
        "l1_input_dram_output": lambda: ttnn.tan(
            _to_device(device, x, memory_config=ttnn.L1_MEMORY_CONFIG), memory_config=ttnn.DRAM_MEMORY_CONFIG
        ),
        "l1_output": lambda: ttnn.tan(_to_device(device, x), memory_config=ttnn.L1_MEMORY_CONFIG),
        "width_sharded": lambda: ttnn.tan(_to_device(device, x, memory_config=width_sharded)),
        "width_sharded_dram_output": lambda: ttnn.tan(
            _to_device(device, x, memory_config=width_sharded), memory_config=ttnn.DRAM_MEMORY_CONFIG
        ),
        "row_major": lambda: ttnn.tan(_to_device(device, x, layout=ttnn.ROW_MAJOR_LAYOUT)),
        "output_tensor": lambda: ttnn.tan(_to_device(device, x), output_tensor=_to_device(device, torch.zeros_like(x))),
        "sub_core_grids": lambda: ttnn.tan(
            _to_device(device, x),
            sub_core_grids=ttnn.CoreRangeSet(
                {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))}
            ),
        ),
        "op_chain": lambda: ttnn.unary_chain(_to_device(device, x), [tan, ttnn.UnaryWithParam(ttnn.UnaryOpType.NEG)]),
    }
    out, ops = _device_ops(calls[config])
    assert ops == {"UnaryDeviceOperation"}, f"tan with {config} ran {ops}"
    want = -torch.tan(x.float()) if config == "op_chain" else torch.tan(x.float())
    got = ttnn.to_torch(out).float()
    finite = torch.isfinite(want) & torch.isfinite(got) & (x.float().abs() < 1e30)
    assert_with_pcc(want[finite], got[finite], 0.999)


def test_non_32x32_tile_stays_unary(device, expect_error):
    # The unary program, which rejects tiles other than 32x32, still gets the call.
    torch.manual_seed(0)
    tiles = CUTOFF[("tan", torch.bfloat16)] * _cores(device)
    x = torch.rand(1, 1, 16, 32 * 2 * tiles).bfloat16()
    tx = ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        tile=ttnn.Tile((16, 32)),
    )
    with expect_error(RuntimeError, "Unary supports only 32x32 tiles"):
        ttnn.tan(tx)


@pytest.mark.parametrize("op, dtype", [("cos", torch.bfloat16), ("tan", torch.float32)])
def test_trig_succeeds_under_l1_pressure(device, op, dtype):
    # With L1 buffers filling the space the native program's rings would take, the unary program still fits: the call
    # must not route there. Fill L1 from the top in steps and check every level the unary program fits in.
    torch.manual_seed(0)
    cores = _cores(device)
    x = _input(_shape(CUTOFF[(op, dtype)] * cores), dtype)
    want = TORCH[op](x.float())
    seen = set()
    for per_core in range(1500, 2200, 10):
        filler = ttnn.empty(
            (1, 1, 32, 32 * per_core * cores),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        lowest = device.lowest_occupied_compute_l1_address()
        if lowest is not None and lowest < 450 * 1024:
            ttnn.deallocate(filler)
            break
        got, ops = _run(device, op, x)
        got = got.float()
        finite = torch.isfinite(want) & torch.isfinite(got) & (x.float().abs() < 1e30)
        assert_with_pcc(want[finite], got[finite], 0.999)
        seen |= ops
        ttnn.deallocate(filler)
    assert seen == {
        "BinaryNgDeviceOperation",
        "UnaryDeviceOperation",
    }, f"{op} {dtype} over the L1 levels ran {seen}; expected the routed and the unary program"


# Runs tan bf16 at its cutoff and prints the device operations and the PCC against torch. With TRIG_CACHED_KNOB set to
# a variable's name, a native add reads the settings first and that variable is then removed.
_SUBPROCESS_SCRIPT = textwrap.dedent(
    """
    import os, torch, ttnn
    device = ttnn.open_device(device_id=0)
    try:
        cached_knob = os.environ.pop("TRIG_CACHED_KNOB", "")
        if cached_knob:
            small = ttnn.from_torch(torch.rand(1, 1, 32, 32).bfloat16(), dtype=ttnn.bfloat16,
                                    layout=ttnn.TILE_LAYOUT, device=device)
            ttnn.experimental.quasar.add(small, 1.0)
            os.environ.pop(cached_knob)
        grid = device.compute_with_storage_grid_size()
        torch.manual_seed(0)
        x = (torch.rand(1, 1, 32, 32 * 64 * grid.x * grid.y) * 2 - 1).bfloat16()
        tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
                             memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
        out = ttnn.tan(tx)
        graph = ttnn.graph.end_graph_capture()
        names = sorted({n["params"]["name"] for n in graph if n.get("node_type") == "function_start"
                        and n.get("params", {}).get("name", "").endswith("DeviceOperation")})
        got = ttnn.to_torch(out).float().flatten()
        want = torch.tan(x.float()).flatten()
        pcc = torch.corrcoef(torch.stack([got, want]))[0, 1].item()
        print("DEVICE_OPS", ",".join(names), "PCC", pcc)
    finally:
        ttnn.close_device(device)
    """
)


def _device_ops_in_subprocess(knob, cached=False):
    # A fresh process with one TTNN_QSR_* variable set, since the native factory reads its settings once per process.
    name, value = knob.split("=")
    env = dict(os.environ, **{name: value})
    if cached:
        env["TRIG_CACHED_KNOB"] = name
    result = subprocess.run(
        [sys.executable, "-c", _SUBPROCESS_SCRIPT], env=env, capture_output=True, text=True, timeout=1200
    )
    lines = [line for line in result.stdout.splitlines() if line.startswith("DEVICE_OPS")]
    assert result.returncode == 0 and lines, f"{knob}: subprocess failed: {result.returncode}\n{result.stderr[-2000:]}"
    _, ops, _, pcc = lines[-1].split()
    return ops, float(pcc)


@pytest.mark.parametrize(
    "knob",
    [
        "TTNN_QSR_NATIVE=0",
        "TTNN_QSR_READER_THREADS=1",
        "TTNN_QSR_COMPUTE_THREADS=2",
        "TTNN_QSR_WRITER_THREADS=1",
        "TTNN_QSR_ENTRIES_PER_THREAD=2",
        "TTNN_QSR_DM_BATCH=2",
        "TTNN_QSR_TILES_PER_CYCLE=2",
        # Values the native factory rejects keep the call on the unary program too, as on main.
        "TTNN_QSR_COMPUTE_THREADS=3",
        "TTNN_QSR_IMPLICIT_SYNC=1",
        "TTNN_QSR_NATIVE=false",
    ],
)
def test_knob_set_stays_unary(knob):
    ops, pcc = _device_ops_in_subprocess(knob)
    assert ops == "UnaryDeviceOperation", f"tan with {knob} ran {ops}"
    assert pcc >= 0.999, f"tan with {knob}: PCC {pcc}"


@pytest.mark.parametrize("knob", ["TTNN_QSR_NATIVE=1", "TTNN_QSR_NATIVE="])
def test_default_native_switch_routes(knob):
    ops, pcc = _device_ops_in_subprocess(knob)
    assert ops == "BinaryNgDeviceOperation", f"tan with {knob} ran {ops}"
    assert pcc >= 0.999, f"tan with {knob}: PCC {pcc}"


@pytest.mark.parametrize("knob", ["TTNN_QSR_COMPUTE_THREADS=2", "TTNN_QSR_NATIVE=0"])
def test_cached_knob_stays_unary(knob):
    # The native factory keeps the settings it read while the variable was set.
    ops, pcc = _device_ops_in_subprocess(knob, cached=True)
    assert ops == "UnaryDeviceOperation", f"tan after a native add with {knob} ran {ops}"
    assert pcc >= 0.999, f"tan after a native add with {knob}: PCC {pcc}"
