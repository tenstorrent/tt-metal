# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Device kernel time of the packer's ReLU stage against the TT-NN path it replaces.

The reference runs that path in this build: a chain of the op and IDENTITY runs the unary program
with TT-NN's own kernel, since the generated kernel serves a lone op only, and a composite runs its
own TT-NN ops. Before anything is timed the reference must be TT-NN's own path: on FP32 input, which
the generated route refuses, it matches the public op bit for bit and launches the same device
operations, and on a board the route does not install on it matches the public op's BF16 output.
Both paths then run in one profiled process on the same BF16 operands; a call's time is the device
kernel duration summed over the programs it launches, the median of REPEATS calls, and the
generated program must be faster by more than MARGIN.
"""

import pandas as pd
import pytest
import torch
import ttnn
from loguru import logger

SHAPES = [(512, 512), (1024, 1024)]
REPEATS = 5
MARGIN = 0.05
# Inputs inside the range the generated kernel is fitted on.
LOW, HIGH = -10.0, 10.0
PATH = "tests/ttnn/perf_tests/operations/eltwise/test_pack_relu_device_perf.py"

# op: (this change's call, the call of the path it replaces, the boards the change installs on)
CASES = {
    "relu": (
        lambda x: ttnn.relu(x),
        lambda x: ttnn.unary_chain(
            x, [ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU), ttnn.UnaryWithParam(ttnn.UnaryOpType.IDENTITY)]
        ),
        ("blackhole", "wormhole_b0"),
    ),
    "relu_min": (
        lambda x: ttnn.relu_min(x, 0.0),
        lambda x: ttnn.unary_chain(
            x, [ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU_MIN, 0.0), ttnn.UnaryWithParam(ttnn.UnaryOpType.IDENTITY)]
        ),
        ("blackhole", "wormhole_b0"),
    ),
    "threshold": (
        lambda x: ttnn.threshold(x, 0.0, 0.0),
        lambda x: ttnn.unary_chain(
            x,
            [ttnn.UnaryWithParam(ttnn.UnaryOpType.THRESHOLD, 0.0, 0.0), ttnn.UnaryWithParam(ttnn.UnaryOpType.IDENTITY)],
        ),
        ("blackhole", "wormhole_b0"),
    ),
    "relu6": (
        lambda x: ttnn.relu6(x),
        lambda x: ttnn.unary_chain(
            x, [ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU6), ttnn.UnaryWithParam(ttnn.UnaryOpType.IDENTITY)]
        ),
        ("blackhole", "wormhole_b0"),
    ),
    "relu_max": (
        lambda x: ttnn.relu_max(x, 6.0),
        lambda x: ttnn.unary_chain(
            x, [ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU_MAX, 6.0), ttnn.UnaryWithParam(ttnn.UnaryOpType.IDENTITY)]
        ),
        ("blackhole", "wormhole_b0"),
    ),
}


def _board(device=None):
    return "blackhole" if ttnn.device.is_blackhole(device) else "wormhole_b0"


def _inputs(shape, dtype, device):
    generator = torch.Generator().manual_seed(0)
    x = LOW + (HIGH - LOW) * (0.05 + 0.9 * torch.rand(shape, generator=generator))
    return ttnn.from_torch(
        x.to(torch.bfloat16).to(dtype),
        dtype=ttnn.bfloat16 if dtype == torch.bfloat16 else ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )


def _device_operations(call):
    """The result of ``call`` and the device operations it launches, as graph capture names them."""
    ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
    try:
        result = call()
    finally:
        graph = ttnn.graph.end_graph_capture()
    return result, [name for name in ttnn.graph.extract_calltrace(graph) if "DeviceOperation" in name]


def _same_bits(a, b):
    a, b = ttnn.to_torch(a), ttnn.to_torch(b)
    width = torch.int32 if a.dtype == torch.float32 else torch.int16
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(a.view(width), b.view(width))


def _prove_reference(name, fused, reference, boards, device):
    """Fail, and time nothing, unless ``reference`` is TT-NN's own path for ``name``."""
    height, width = SHAPES[0]
    x32 = _inputs((1, 1, height, width), torch.float32, device)
    x16 = _inputs((1, 1, height, width), torch.bfloat16, device)
    own, own_operations = _device_operations(lambda: fused(x32))
    mine, my_operations = _device_operations(lambda: reference(x32))
    assert _same_bits(mine, own), f"{name}: on FP32 input the reference differs from TT-NN's own path; not timing"
    _, my_bf16_operations = _device_operations(lambda: reference(x16))
    assert (
        my_bf16_operations == own_operations
    ), f"{name}: the reference launches {my_bf16_operations}, TT-NN's own path {own_operations}; not timing"
    if _board(device) not in boards:
        assert _same_bits(
            reference(x16), fused(x16)
        ), f"{name}: on a board that keeps TT-NN's path the reference differs from it"
    logger.info(f"REFERENCE {name} {_board(device)} is TT-NN's own path: {len(own_operations)} device operations")


def test_pack_relu_profiled_calls(device):
    """The proofs, then the calls the device-perf test profiles, between signposts that name each one."""
    from tracy import signpost

    if not (ttnn.device.is_blackhole(device) or ttnn.device.is_wormhole_b0(device)):
        pytest.skip("the generated kernels exist for Blackhole and Wormhole only")
    for name, (fused, reference, boards) in CASES.items():
        _prove_reference(name, fused, reference, boards, device)
        if _board(device) not in boards:
            continue
        for height, width in SHAPES:
            x = _inputs((1, 1, height, width), torch.bfloat16, device)
            for path, call in (("old", reference), ("fused", fused)):
                ttnn.deallocate(call(x))
                for repeat in range(REPEATS):
                    signpost(f"{name}-{path}-{height}x{width}-{repeat}-start")
                    ttnn.deallocate(call(x))
                    signpost(f"{name}-{path}-{height}x{width}-{repeat}-end")
            ttnn.synchronize_device(device)


def _kernel_ns_per_region(csv_path):
    """Per pair of signposts, the device kernel duration summed over the programs between them, and their count."""
    regions, current = {}, None
    for _, row in pd.read_csv(csv_path).iterrows():
        if row["OP TYPE"] == "signpost":
            code = row["OP CODE"]
            current = code.removesuffix("-start") if code.endswith("-start") else None
            if current is not None:
                regions[current] = [0.0, 0]
        elif current is not None and str(row["DEVICE KERNEL DURATION [ns]"]) not in ("-", "nan"):
            regions[current][0] += float(row["DEVICE KERNEL DURATION [ns]"])
            regions[current][1] += 1
    return regions


@pytest.mark.models_device_performance_bare_metal
def test_pack_relu_device_perf():
    from tracy.common import clear_profiler_runtime_artifacts
    from tracy.process_model_log import get_latest_ops_log_filename, run_device_profiler

    board = _board()
    subdir = "pack_relu_device_perf"
    clear_profiler_runtime_artifacts()
    run_device_profiler(
        f"pytest {PATH}::test_pack_relu_profiled_calls", subdir, device_analysis_types=["device_kernel_duration"]
    )
    timed = [name for name, (_, _, boards) in CASES.items() if board in boards]
    for name in CASES:
        if name not in timed:
            logger.info(f"DEVICE_PERF {name} {board} keeps TT-NN's path; the reference equals it")
    if not timed:
        return
    regions = _kernel_ns_per_region(get_latest_ops_log_filename(subdir))
    slow = []
    for name in timed:
        for height, width in SHAPES:
            per_call = {}
            for path in ("old", "fused"):
                measured = [regions[f"{name}-{path}-{height}x{width}-{repeat}"] for repeat in range(REPEATS)]
                assert all(programs for _, programs in measured), f"a {path} call of {name} launched no program"
                per_call[path] = (sorted(ns for ns, _ in measured)[REPEATS // 2], measured[0][1])
            (old, old_programs), (fused, fused_programs) = per_call["old"], per_call["fused"]
            logger.info(
                f"DEVICE_PERF {name} {board} {height}x{width} old={old:.0f}ns ({old_programs} programs) "
                f"fused={fused:.0f}ns ({fused_programs} programs) speedup={old / fused:.2f}x"
            )
            if fused > old * (1 - MARGIN):
                slow.append(f"{name} {height}x{width}: fused {fused:.0f} ns, old {old:.0f} ns")
    assert not slow, f"the generated program is not {MARGIN:.0%} faster than the path it replaces: {slow}"
