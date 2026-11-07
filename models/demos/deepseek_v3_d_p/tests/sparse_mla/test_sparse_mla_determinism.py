# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Sparse MLA determinism at the perf gate's production shapes: one 5k chunk over a 50k or 500k cache (Galaxy sizes,
scaled by SP on smaller boxes), the scaled-FP8 KVPE cache, each model's fabric packet size, for K3.1 (last DSA layer's
slot of 24), GLM-5.3 (last full indexer layer) and a GLM-5.3 shared layer.

Like the ring joint SDPA determinism tests, each case enqueues RUNS forwards back to back with every output kept on
device (no host sync or readback in between), then syncs once and compares: every eager output must equal the first
bit for bit. The same for RUNS replays of one captured trace, each copied on device into a buffer allocated before the
capture (every replay writes the same output buffer), and the traced output must equal the eager one.

Prefixes:
  * zero: the perf gate's caches, zero-initialised. The indexer then scores the whole prefix equally, so top-k breaks
    many ties.
  * filled: the prefix written by real chunk forwards of distinct inputs, as in production (warm only: the long prefix
    would take 100 forwards). Not for the GLM-5.3 shared layer: its top-k comes from a full layer's forward at the
    final start, so it cannot run the earlier chunks.
"""

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_d_p.tests.sparse_mla.sparse_mla_reference import make_hidden
from models.demos.deepseek_v3_d_p.tests.sparse_mla.test_sparse_mla_perf import (
    _BOX,
    CHUNK_TOKENS,
    GALAXY_SP,
    GATE_CASES,
    SCENARIO_CACHE_TOKENS,
    _device_tensors,
    _gate_params,
    build_sparse_perf_case,
    local_cache_tokens,
)
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology
from models.demos.deepseek_v3_d_p.utils.sub_device_trace import SubDeviceTraceController

RUNS = 10
# (scenario, prefix)
PREFIXES = [("warm", "zero"), ("warm", "filled"), ("long", "zero")]


def _params():
    """The perf gate's cases x PREFIXES, without a filled prefix for a shared layer."""
    params = []
    for gate in _gate_params():
        mesh, dp, variant, case_name = gate.values
        for scenario, prefix in PREFIXES:
            if prefix == "filled" and GATE_CASES[case_name][1]:
                continue
            params.append(
                pytest.param(
                    mesh,
                    dp,
                    variant,
                    case_name,
                    scenario,
                    prefix,
                    marks=gate.marks,
                    id=f"{gate.id or 'unsupported'}-{scenario}-{prefix}",
                )
            )
    return params


def _fill_prefix(mesh_device, case, cache: int, chunk: int, hidden_size: int):
    """Write the prefix [0, cache) with chunk forwards of distinct inputs, like a real prefill before `cache`."""
    x = case.x
    try:
        for i, start in enumerate(range(0, cache, chunk)):
            case.x = ttnn.from_torch(
                make_hidden(chunk, hidden_size, seed=1000 + i)[:, :chunk].unsqueeze(0),
                device=mesh_device,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=[-2, -1]),
            )
            try:
                ttnn.deallocate(case.forward(start))
            finally:
                ttnn.deallocate(case.x)
    finally:
        case.x = x
    ttnn.synchronize_device(mesh_device)


def _drain(mesh_device):
    """Best-effort sync before freeing: on an error path, earlier enqueued work may still use the buffers."""
    try:
        ttnn.synchronize_device(mesh_device)
    except Exception:
        pass


def _eager_outputs(mesh_device, case, start):
    """RUNS forwards back to back, every output kept on device; then one sync and the per-device outputs."""
    outputs = []
    try:
        for _ in range(RUNS):
            outputs.append(case.forward(start))
        ttnn.synchronize_device(mesh_device)
        return [_device_tensors(out) for out in outputs]
    finally:
        _drain(mesh_device)
        for out in outputs:
            ttnn.deallocate(out)


def _replay_outputs(mesh_device, case, start):
    """RUNS replays of one captured forward back to back, each copied on device; then one sync and the outputs."""
    controller = SubDeviceTraceController(mesh_device)
    case.mla.set_trace_controller(controller)
    capture_out, capturing, outputs = None, False, []
    try:
        # Compile under the controller, and allocate each replay's copy now: a buffer allocated while the trace is
        # live can land on its scratch and be overwritten by the next replay.
        out = case.forward(start)
        try:
            for _ in range(RUNS):
                outputs.append(ttnn.clone(out))
        finally:
            ttnn.deallocate(out)
        ttnn.synchronize_device(mesh_device)
        controller.begin_capture()
        capturing = True
        capture_out = case.forward(start)
        controller.end_capture()
        capturing = False
        for copy in outputs:  # same CQ: each copy runs after its replay, before the next replay
            controller.replay(sync=False)
            ttnn.copy(capture_out, copy)
        ttnn.synchronize_device(mesh_device)
        return [_device_tensors(copy) for copy in outputs]
    finally:
        _drain(mesh_device)
        for copy in outputs:
            ttnn.deallocate(copy)
        if capturing:
            try:
                controller.end_capture()
            except Exception:
                pass
        try:
            controller.release()
        finally:
            case.mla.set_trace_controller(None)
            if capture_out is not None:
                ttnn.deallocate(capture_out)


def _bit_equal(a, b):
    """Bitwise equality (a NaN equals the same NaN), not numeric: compare the raw bits of each element."""
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    if a.is_floating_point():
        bits = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[a.element_size()]
        return torch.equal(a.contiguous().view(bits), b.contiguous().view(bits))
    return torch.equal(a, b)


def _differences(what, reference, outputs, first=1):
    """One line per run (numbered from `first`; run 0 is the reference) that is not bit-exact with `reference`: the
    devices that differ and their PCC."""
    lines = []
    for run, devices in enumerate(outputs, start=first):
        bad = [d for d, (a, b) in enumerate(zip(reference, devices)) if not _bit_equal(a, b)]
        if bad:
            shapes = [d for d in bad if reference[d].shape != devices[d].shape]
            if shapes:
                d = shapes[0]
                detail = f"shape {tuple(devices[d].shape)} vs {tuple(reference[d].shape)} on device {d}"
            else:
                detail = f"min PCC {min(comp_pcc(reference[d], devices[d], 0.0)[1] for d in bad):.6f}"
            lines.append(f"{what} {run}: {len(bad)}/{len(devices)} devices differ (first {bad[:4]}), {detail}")
    return lines


@pytest.mark.timeout(900)
@pytest.mark.parametrize(
    "mesh_device, device_params, variant, case_name, scenario, prefix",
    _params(),
    indirect=["mesh_device", "device_params", "variant"],
)
def test_sparse_mla_determinism(mesh_device, device_params, variant, case_name, config_only, scenario, prefix):
    label = f"{case_name} sparse MLA {_BOX[0]} {scenario} {prefix} prefix"
    sp, _ = mesh_device.shape
    chunk = CHUNK_TOKENS * sp // GALAXY_SP
    cache = local_cache_tokens(SCENARIO_CACHE_TOKENS[scenario], sp)
    case = build_sparse_perf_case(
        mesh_device,
        variant,
        config_only,
        cache=cache,
        chunk=chunk,
        topology=per_axis_topology(device_params["fabric_config"]),
        shared_layer=GATE_CASES[case_name][1],
    )
    if prefix == "filled":
        _fill_prefix(mesh_device, case, cache, chunk, config_only.hidden_size)
    ttnn.deallocate(case.forward(cache))  # compile
    ttnn.synchronize_device(mesh_device)

    eager = _eager_outputs(mesh_device, case, cache)
    replays = _replay_outputs(mesh_device, case, cache)
    failures = _differences("eager run", eager[0], eager[1:]) + _differences("replay", replays[0], replays[1:])
    failures += _differences("traced vs eager, replay", eager[0], replays[:1], first=0)
    assert not failures, f"{label} is not deterministic:\n  " + "\n  ".join(failures)
