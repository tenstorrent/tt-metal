# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Synthetic-weight performance acceptance for Kimi-K3 KDA."""

from __future__ import annotations

import json
import os
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass

import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import (
    fabric_1d_device_params,
    torus_xy_device_params,
    tp_axis_is_wrapped,
)
from models.demos.deepseek_v3_d_p.tests.kda.utils import (
    KIMI_K3_CONVOLUTION_NRMSE,
    LayerNrmseThresholds,
    deallocate_state,
    make_kimi_k3_device_case,
    make_synthetic_kimi_k3_test_case,
)
from models.demos.deepseek_v3_d_p.tt.kda.config import KDA_PARTIAL_OUTPUT_L1_BYTES_PER_CORE, l1_when_it_fits
from models.demos.deepseek_v3_d_p.tt.kda.kda import KdaState, reduce_scatter_tp_partials, ttKDA
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology
from models.demos.deepseek_v3_d_p.utils.chunk_config import PREFILL_CHUNK_TOKENS, PREFILL_CHUNK_TOKENS_PER_CHIP
from models.tt_transformers.tt.ccl import TT_CCL
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc
from tests.ttnn.profiling.realtime_profiler_utils import profile_realtime_program_merged, require_realtime_profiler
from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import make_actual_start

pytestmark = [
    run_for_blackhole(),
    pytest.mark.perf,
    pytest.mark.timeout(900),
]

_REPETITIONS = 10
_TIMING_SAMPLES = 5
_LOUDBOX_PERF_SKU = "bh_loudbox"
_QUIETBOX_PERF_SKU = "bh_quietbox"
_PERF_MARGIN = 0.03
# Host wall time of one traced call: the median over _TIMING_SAMPLES samples, each of back-to-back trace replays
# through device synchronization.
_PERF_REFERENCE_MS = {
    _LOUDBOX_PERF_SKU: {
        "SP8xTP1": 1.432,
        "TP4-linear": 0.188,
    },
    _QUIETBOX_PERF_SKU: {
        "TP4-ring": 0.086,
    },
}
# Device time of one traced call: the sum over its programs of the slowest chip's realtime-profiler duration, the
# median over _TIMING_SAMPLES single replays. As in the K3 MLA perf test, each replay is profiled on its own after a
# discarded first one, never back to back: sustained replays push QuietBox to its power limit, and the firmware then
# lowers the clock by an amount that varies by runner. The cross-chip programs' wait for their neighbours still moves
# a single replay by a few percent, hence the wider margin.
_DEVICE_PERF_MARGIN = 0.05
_DEVICE_PERF_REFERENCE_MS = {
    _QUIETBOX_PERF_SKU: {
        "SP4xTP1": 1.437,
    },
}
# Galaxy-equivalent layouts outside CI: they report wall time but are not gated.
_REPORT_ONLY_LAYOUTS = ("SP4xTP2", "SP2xTP2")
# The reduce-scatter takes a fraction of a layer, so its trace holds several calls to amortize the replay.
_REDUCE_SCATTER_CALLS_PER_TRACE = 20
# Overrides the chunk's start position for investigation; the performance gates are skipped when it is set.
_ACTUAL_START_OVERRIDE = os.environ.get("KDA_PERF_ACTUAL_START")
# Galaxy target: a full chunk whose start falls mid-device. A resumed request starts at an arbitrary aligned
# prefix, and every full chunk of it then wraps onto its first rank, which adds the final-carry barrier and
# broadcast that a rank-aligned start skips. Production always passes actual_end, so the gate does too.
_GALAXY_PERF_REFERENCE_MS = 1.210
_GALAXY_TP = 4
_GALAXY_HEADS_PER_CHIP = KimiK3Config.KDA_NUM_HEADS // _GALAXY_TP
_GALAXY_ACTUAL_START = 4 * PREFILL_CHUNK_TOKENS_PER_CHIP + 320
# QuietBox SP2xTP2 runs Galaxy's per-chip workload: 640 rows and 24 heads per chip, so the chunk shrinks to 1280
# tokens and the synthetic config halves the global head count. Every per-chip shape, L1 placement and program
# config then matches Galaxy, and the chunk starts 320 rows into a device as on Galaxy. The SP chain is 2 ranks
# instead of 8, and the 11x10 grid splits each head's values over 2 cores instead of 4.
_QUIETBOX_SEQUENCE = 2 * PREFILL_CHUNK_TOKENS_PER_CHIP
_QUIETBOX_NUM_HEADS = 2 * _GALAXY_HEADS_PER_CHIP
_QUIETBOX_ACTUAL_START = PREFILL_CHUNK_TOKENS_PER_CHIP + 320
# QuietBox SP4xTP1 runs the same per-chip workload over a 4-rank SP ring: 2560 tokens of Galaxy's 24 heads per chip.
# With one TP rank the output projection writes its result to DRAM and skips the TP reduce-scatter;
# test_kimi_k3_tp_reduce_scatter_perf times that collective on its own.
_QUIETBOX_SP4_SEQUENCE = 4 * PREFILL_CHUNK_TOKENS_PER_CHIP
_QUIETBOX_SP4_NUM_HEADS = _GALAXY_HEADS_PER_CHIP
_QUIETBOX_SP4_ACTUAL_START = PREFILL_CHUNK_TOKENS_PER_CHIP + 320
# LoudBox SP4xTP2 runs the same per-chip workload on a 2560-token chunk. The SP chain is 4 ranks and the TP
# reduce-scatter 2; the 11x10 grid splits each head's values over 2 cores as on QuietBox.
_LOUDBOX_SEQUENCE = 4 * PREFILL_CHUNK_TOKENS_PER_CHIP
_LOUDBOX_NUM_HEADS = 2 * _GALAXY_HEADS_PER_CHIP
_LOUDBOX_ACTUAL_START = 2 * PREFILL_CHUNK_TOKENS_PER_CHIP + 320
# LoudBox SP8xTP1 runs Galaxy's SP chain and chunk with Galaxy's 24 heads per chip. Its 8 chips close into a ring, as
# Galaxy's SP axis does, and so do QuietBox SP4xTP1's 4; FABRIC_1D_RING rings only mesh axis 0, so SP takes that axis.
# As on QuietBox SP4xTP1, it skips the TP reduce-scatter, which test_kimi_k3_tp_reduce_scatter_perf times on its own.
# Like the model's prefill tests, give CCL semaphores their own L1_SMALL region: allocated in L1 on the first call,
# they would land between the chunk terms and fragment L1, so the traced call would plan ops on less free L1 than
# the warm-up compiled for.
_L1_SMALL_SIZE = 1152


def _perf_reference_ms(layout: str) -> float:
    """Return the wall-time reference for a gated layout on the KDA_PERF_SKU it opts in to."""
    if layout == "SP8xTP4":
        return _GALAXY_PERF_REFERENCE_MS
    skus = sorted(sku for sku, references in _PERF_REFERENCE_MS.items() if layout in references)
    if not skus:
        raise ValueError(f"{layout} has no host wall-time reference")
    sku = os.environ.get("KDA_PERF_SKU")
    if sku not in skus:
        raise ValueError(f"set KDA_PERF_SKU to one of {skus} to opt in to this hardware-specific performance gate")
    return _PERF_REFERENCE_MS[sku][layout]


def _device_perf_reference_ms(layout: str) -> float | None:
    """Return the device-time reference for a device-gated layout on its KDA_PERF_SKU, or None if it has none."""
    skus = sorted(sku for sku, references in _DEVICE_PERF_REFERENCE_MS.items() if layout in references)
    if not skus:
        return None
    sku = os.environ.get("KDA_PERF_SKU")
    if sku not in skus:
        raise ValueError(f"set KDA_PERF_SKU to one of {skus} to opt in to this hardware-specific performance gate")
    return _DEVICE_PERF_REFERENCE_MS[sku][layout]


def _allocate_state(layer: ttKDA) -> KdaState:
    return layer.allocate_state(batch_size=1)


def _assert_within_margin(what: str, measured_ms: float, reference_ms: float, margin: float) -> None:
    lower = reference_ms * (1.0 - margin)
    upper = reference_ms * (1.0 + margin)
    assert lower <= measured_ms <= upper, (
        f"{what} {measured_ms:.3f} ms is outside performance range "
        f"[{lower:.3f}, {upper:.3f}] ms "
        f"(reference {reference_ms:.3f} ms ± {margin:.0%})"
    )


@pytest.mark.parametrize(
    "layout,sku",
    [
        ("SP8xTP1", _LOUDBOX_PERF_SKU),
        ("TP4-linear", _LOUDBOX_PERF_SKU),
        ("TP4-ring", _QUIETBOX_PERF_SKU),
        # The Galaxy gate needs no KDA_PERF_SKU opt-in.
        ("SP8xTP4", None),
    ],
)
def test_synthetic_performance_uses_two_sided_margin(layout, sku, monkeypatch, expect_error) -> None:
    if sku is None:
        monkeypatch.delenv("KDA_PERF_SKU", raising=False)
    else:
        monkeypatch.setenv("KDA_PERF_SKU", sku)
    reference_ms = _perf_reference_ms(layout)
    margin = _PERF_MARGIN
    _assert_within_margin(layout, reference_ms * (1 - 0.99 * margin), reference_ms, margin)
    _assert_within_margin(layout, reference_ms * (1 + 0.99 * margin), reference_ms, margin)
    with expect_error(AssertionError, "outside performance range"):
        _assert_within_margin(layout, reference_ms * (0.99 - margin), reference_ms, margin)
    with expect_error(AssertionError, "outside performance range"):
        _assert_within_margin(layout, reference_ms * (1.01 + margin), reference_ms, margin)


def test_synthetic_device_performance_uses_two_sided_margin(monkeypatch, expect_error) -> None:
    monkeypatch.setenv("KDA_PERF_SKU", _QUIETBOX_PERF_SKU)
    reference_ms = _device_perf_reference_ms("SP4xTP1")
    _assert_within_margin("SP4xTP1", reference_ms * (1 + 0.99 * _DEVICE_PERF_MARGIN), reference_ms, _DEVICE_PERF_MARGIN)
    with expect_error(AssertionError, "outside performance range"):
        _assert_within_margin("SP4xTP1", reference_ms * (1.01 + _DEVICE_PERF_MARGIN), reference_ms, _DEVICE_PERF_MARGIN)


def _profile_replay(mesh_device: ttnn.MeshDevice, trace_id: int) -> tuple[float, float]:
    """Return the host wall and device time (ms) of one profiled trace replay."""

    def replay() -> float:
        start = time.perf_counter()
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        return (time.perf_counter() - start) * 1e3

    wall_ms, programs = profile_realtime_program_merged(mesh_device, replay)
    return wall_ms, sum(program["duration_ns"] for program in programs.values()) / 1e6


@dataclass
class _LayerTrace:
    """A traced layer call and the device buffers it holds until released."""

    state: KdaState
    trace_id: int | None = None
    output: ttnn.Tensor | None = None
    next_state: KdaState | None = None


def _capture_layer_trace(
    mesh_device: ttnn.MeshDevice,
    layer: ttKDA,
    hidden: ttnn.Tensor,
    actual_start: int = 0,
    actual_end: int | None = None,
) -> _LayerTrace:
    """Capture one layer call in a trace after a warm-up call, and replay it once."""
    actual_start_tt = make_actual_start(mesh_device, actual_start)
    actual_end_tt = None if actual_end is None else make_actual_start(mesh_device, actual_end)
    trace = _LayerTrace(state=_allocate_state(layer))
    try:
        warm_output, warm_state = layer.forward(hidden, trace.state, actual_start_tt, actual_end_tt)
        ttnn.synchronize_device(mesh_device)
        ttnn.deallocate(warm_output)
        deallocate_state(warm_state)

        trace.trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        trace.output, trace.next_state = layer.forward(hidden, trace.state, actual_start_tt, actual_end_tt)
        ttnn.end_trace_capture(mesh_device, trace.trace_id, cq_id=0)
        ttnn.execute_trace(mesh_device, trace.trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
    except BaseException:
        _release_layer_trace(mesh_device, trace)
        raise
    return trace


def _release_layer_trace(mesh_device: ttnn.MeshDevice, trace: _LayerTrace) -> None:
    try:
        if trace.trace_id is not None:
            ttnn.release_trace(mesh_device, trace.trace_id)
    finally:
        if trace.output is not None:
            ttnn.deallocate(trace.output)
        if trace.next_state is not None:
            deallocate_state(trace.next_state)
        deallocate_state(trace.state)


def _trace_wall_samples_ms(
    mesh_device: ttnn.MeshDevice,
    layer: ttKDA,
    hidden: ttnn.Tensor,
    repetitions: int,
    validate_first_replay: Callable[[KdaState, ttnn.Tensor], dict[str, float]] | None = None,
    actual_start: int = 0,
    actual_end: int | None = None,
) -> tuple[list[float], dict[str, float] | None]:
    trace = _capture_layer_trace(mesh_device, layer, hidden, actual_start, actual_end)
    try:
        validation = (
            validate_first_replay(
                trace.next_state,
                trace.output,
                nrmse=LayerNrmseThresholds(output=0.03, recurrent=0.04, convolution=KIMI_K3_CONVOLUTION_NRMSE),
            )
            if validate_first_replay is not None
            else None
        )

        samples_ms = []
        for _ in range(_TIMING_SAMPLES):
            start = time.perf_counter()
            for _ in range(repetitions):
                ttnn.execute_trace(mesh_device, trace.trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)
            samples_ms.append((time.perf_counter() - start) * 1e3 / repetitions)
        return samples_ms, validation
    finally:
        _release_layer_trace(mesh_device, trace)


def _trace_device_samples_ms(
    mesh_device: ttnn.MeshDevice,
    layer: ttKDA,
    hidden: ttnn.Tensor,
    actual_start: int = 0,
    actual_end: int | None = None,
) -> tuple[list[float], list[float]]:
    """Return the host wall and device time (ms) of single profiled trace replays, never back to back."""
    trace = _capture_layer_trace(mesh_device, layer, hidden, actual_start, actual_end)
    try:
        _profile_replay(mesh_device, trace.trace_id)  # exclude first-replay jitter
        samples = [_profile_replay(mesh_device, trace.trace_id) for _ in range(_TIMING_SAMPLES)]
    finally:
        _release_layer_trace(mesh_device, trace)
    return [wall for wall, _ in samples], [device for _, device in samples]


@pytest.mark.parametrize(
    "mesh_device,tensor_parallel_axis,device_params",
    [
        pytest.param(
            (2, 4),
            0,
            fabric_1d_device_params(l1_small_size=_L1_SMALL_SIZE),
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(2, 4), topology="mesh-2x4"),
            id="SP4xTP2-fabric-1d",
        ),
        pytest.param(
            (8, 1),
            1,
            fabric_1d_device_params(fabric_config=ttnn.FabricConfig.FABRIC_1D_RING, l1_small_size=_L1_SMALL_SIZE),
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 1), topology="ring"),
            id="SP8xTP1-ring",
        ),
        pytest.param(
            (2, 2),
            1,
            fabric_1d_device_params(l1_small_size=_L1_SMALL_SIZE),
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(2, 2), topology="mesh-2x2"),
            id="SP2xTP2-fabric-1d",
        ),
        pytest.param(
            (4, 1),
            1,
            fabric_1d_device_params(fabric_config=ttnn.FabricConfig.FABRIC_1D_RING, l1_small_size=_L1_SMALL_SIZE),
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(4, 1), topology="ring"),
            id="SP4xTP1-ring",
        ),
        pytest.param(
            (8, 4),
            1,
            torus_xy_device_params(l1_small_size=_L1_SMALL_SIZE),
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="SP8xTP4-torus-xy",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_synthetic_kimi_k3_perf(
    mesh_device: ttnn.MeshDevice,
    tensor_parallel_axis: int,
    device_params: dict,
) -> None:
    """Gate checkpoint-free production K3 latency on LoudBox, QuietBox and Galaxy."""
    mesh_shape = tuple(mesh_device.shape)
    sequence_parallel_axis = 1 - tensor_parallel_axis
    layout = f"SP{mesh_shape[sequence_parallel_axis]}xTP{mesh_shape[tensor_parallel_axis]}"
    report_only = layout in _REPORT_ONLY_LAYOUTS
    device_reference_ms = None if report_only else _device_perf_reference_ms(layout)
    reference_ms = None if report_only or device_reference_ms is not None else _perf_reference_ms(layout)
    sequence, num_heads, actual_start = {
        "SP8xTP4": (PREFILL_CHUNK_TOKENS, None, _GALAXY_ACTUAL_START),
        "SP4xTP2": (_LOUDBOX_SEQUENCE, _LOUDBOX_NUM_HEADS, _LOUDBOX_ACTUAL_START),
        "SP8xTP1": (PREFILL_CHUNK_TOKENS, _GALAXY_HEADS_PER_CHIP, _GALAXY_ACTUAL_START),
        "SP2xTP2": (_QUIETBOX_SEQUENCE, _QUIETBOX_NUM_HEADS, _QUIETBOX_ACTUAL_START),
        "SP4xTP1": (_QUIETBOX_SP4_SEQUENCE, _QUIETBOX_SP4_NUM_HEADS, _QUIETBOX_SP4_ACTUAL_START),
    }[layout]
    if _ACTUAL_START_OVERRIDE is not None:
        actual_start = int(_ACTUAL_START_OVERRIDE)
    actual_end = None if actual_start == 0 else actual_start + sequence
    case = make_synthetic_kimi_k3_test_case(sequence=sequence, num_heads=num_heads)
    layer, hidden_tt = make_kimi_k3_device_case(
        mesh_device,
        case,
        tensor_parallel_axis=tensor_parallel_axis,
        cache_weights=False,
    )
    assert (layer.active_seq_len_local, layer.config.num_heads) == (
        PREFILL_CHUNK_TOKENS_PER_CHIP,
        _GALAXY_HEADS_PER_CHIP,
    ), f"{layout} must run Galaxy's per-chip workload"
    if device_reference_ms is not None:
        require_realtime_profiler(f"the {layout} device-time gate")
    if device_reference_ms is not None:
        samples_ms, device_samples_ms = _trace_device_samples_ms(
            mesh_device, layer, hidden_tt, actual_start=actual_start, actual_end=actual_end
        )
    else:
        samples_ms, _ = _trace_wall_samples_ms(
            mesh_device, layer, hidden_tt, _REPETITIONS, actual_start=actual_start, actual_end=actual_end
        )
        device_samples_ms = None
    median_wall_ms = statistics.median(samples_ms)
    median_device_ms = None if device_samples_ms is None else statistics.median(device_samples_ms)
    result = {
        "fabric_config": ttnn.get_fabric_config().name,
        "layout": layout,
        "sequence": sequence,
        "num_heads": layer.config.num_heads * layer.tensor_parallel_size,
        "actual_start": actual_start,
        "actual_end": actual_end,
        "weights": "deterministic synthetic",
        "repetitions": 1 if device_reference_ms is not None else _REPETITIONS,
        "trace_wall_samples_ms": samples_ms,
        "median_trace_wall_ms": median_wall_ms,
        "timing_sample_count": _TIMING_SAMPLES,
        "reference_trace_wall_ms": reference_ms,
        "perf_margin_pct": _PERF_MARGIN * 100.0,
        "device_samples_ms": device_samples_ms,
        "median_device_ms": median_device_ms,
        "reference_device_ms": device_reference_ms,
        "device_perf_margin_pct": _DEVICE_PERF_MARGIN * 100.0,
    }
    print("KDA_SYNTHETIC_PERF=" + json.dumps(result, sort_keys=True))
    if report_only:
        return
    if _ACTUAL_START_OVERRIDE is not None:
        pytest.skip(f"KDA_PERF_ACTUAL_START={_ACTUAL_START_OVERRIDE} overrides the gated start; not gating")
    if layout == "SP8xTP4" and not tp_axis_is_wrapped(mesh_device):
        pytest.skip(
            f"TP axis not wrapped: measured {median_wall_ms:.3f} ms on Linear TP; "
            "the Galaxy reference assumes the TP ring"
        )
    if device_reference_ms is not None:
        _assert_within_margin(f"{layout} median device", median_device_ms, device_reference_ms, _DEVICE_PERF_MARGIN)
    else:
        _assert_within_margin(f"{layout} median trace wall", median_wall_ms, reference_ms, _PERF_MARGIN)


@pytest.mark.parametrize(
    "layout,mesh_device,cluster_axis,expected_topology,device_params",
    [
        pytest.param(
            "TP4-ring",
            (4, 1),
            0,
            ttnn.Topology.Ring,
            fabric_1d_device_params(fabric_config=ttnn.FabricConfig.FABRIC_1D_RING, l1_small_size=_L1_SMALL_SIZE),
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(4, 1), topology="ring"),
            id="TP4-ring",
        ),
        # LoudBox has no 4-chip ring, so its TP4 rows reduce-scatter over a line.
        pytest.param(
            "TP4-linear",
            (2, 4),
            1,
            ttnn.Topology.Linear,
            fabric_1d_device_params(l1_small_size=_L1_SMALL_SIZE),
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(2, 4), topology="mesh-2x4"),
            id="TP4-linear",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_kimi_k3_tp_reduce_scatter_perf(
    layout: str, mesh_device: ttnn.MeshDevice, cluster_axis: int, expected_topology: ttnn.Topology, device_params: dict
) -> None:
    """Gate Galaxy's TP reduce-scatter on 4 chips, with the layer's per-chip shapes and arguments.

    The output projection's partial sums, [640, hidden] per chip in L1 as on Galaxy, are reduce-scattered over
    4 TP ranks into DRAM. SP4xTP1 (QuietBox) and SP8xTP1 (LoudBox) skip this collective, so together with it they
    cover Galaxy's per-chip layer.
    """
    reference_ms = _perf_reference_ms(layout)
    tp_size = tuple(mesh_device.shape)[cluster_axis]
    topology = per_axis_topology()[cluster_axis]
    assert (tp_size, topology) == (
        4,
        expected_topology,
    ), f"must reduce-scatter over 4 TP ranks on a {expected_topology}"
    rows, hidden_size = PREFILL_CHUNK_TOKENS_PER_CHIP, KimiK3Config.EMB_SIZE
    input_memory_config = l1_when_it_fits(mesh_device, rows * hidden_size * 2, KDA_PARTIAL_OUTPUT_L1_BYTES_PER_CORE)
    assert input_memory_config == ttnn.L1_MEMORY_CONFIG, "Galaxy keeps the partial sums in L1"

    torch.manual_seed(0)
    partials = torch.randn(tp_size, 1, rows, hidden_size, dtype=torch.bfloat16)
    partials_tt = ttnn.from_torch(
        partials,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=input_memory_config,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device, dims=(0, None) if cluster_axis == 0 else (None, 0), mesh_shape=tuple(mesh_device.shape)
        ),
    )
    tt_ccl = TT_CCL(mesh_device)
    num_links = tt_ccl.get_num_links(cluster_axis)

    def reduce_scatter() -> ttnn.Tensor:
        return reduce_scatter_tp_partials(partials_tt, tt_ccl, cluster_axis, topology)

    warm_output = reduce_scatter()
    ttnn.synchronize_device(mesh_device)
    output = ttnn.to_torch(warm_output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1))
    ttnn.deallocate(warm_output)
    # Each TP group holds the full reduced output; replicas along the other axis follow one another.
    expected = partials.float().sum(dim=0, keepdim=True).repeat(1, 1, 1, output.shape[-1] // hidden_size)
    passed, pcc = comp_pcc(expected, output.float(), 0.9999)
    assert passed, f"{layout} PCC {pcc}"

    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    for _ in range(_REDUCE_SCATTER_CALLS_PER_TRACE):
        # Released inside the trace, as the layer releases its reduced output after use.
        ttnn.deallocate(reduce_scatter())
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    try:
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
        samples_ms = []
        for _ in range(_TIMING_SAMPLES):
            start = time.perf_counter()
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)
            samples_ms.append((time.perf_counter() - start) * 1e3 / _REDUCE_SCATTER_CALLS_PER_TRACE)
    finally:
        ttnn.release_trace(mesh_device, trace_id)
        ttnn.deallocate(partials_tt)
    median_wall_ms = statistics.median(samples_ms)
    result = {
        "fabric_config": ttnn.get_fabric_config().name,
        "layout": layout,
        "input_shape": [1, 1, rows, hidden_size],
        "num_links": num_links,
        "topology": topology.name,
        "pcc": pcc,
        "calls_per_trace": _REDUCE_SCATTER_CALLS_PER_TRACE,
        "trace_wall_samples_ms": samples_ms,
        "median_trace_wall_ms": median_wall_ms,
        "reference_trace_wall_ms": reference_ms,
        "perf_margin_pct": _PERF_MARGIN * 100.0,
    }
    print("KDA_TP_REDUCE_SCATTER_PERF=" + json.dumps(result, sort_keys=True))
    _assert_within_margin(f"{layout} median trace wall", median_wall_ms, reference_ms, _PERF_MARGIN)
