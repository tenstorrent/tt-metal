# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Sparse MLA determinism at the perf gate's production shapes: one 5k chunk over a 50k or 500k cache (Galaxy sizes,
scaled by SP on smaller boxes), the scaled-FP8 KVPE cache, each model's fabric packet size, for K3.1 (last DSA layer's
slot of 24), GLM-5.3 (last full indexer layer) and a GLM-5.3 shared layer.

The check stays on device, as in tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py (helpers in
attn_res/device_determinism.py): run 0's output is the reference, every later output folds max(ne(reference, output))
into one marker, and the markers are read back once at the end. Nothing syncs inside the loops, so a fast chip can get
ahead of a slow one, which is what exposes a race. Only a failure reads outputs back, to say which chips differ.
ttnn.ne compares values, not bits: a NaN never equals itself (so a NaN fails the check) and -0 equals +0.

  * eager: RUNS forwards back to back; run 0's and the latest output are kept, each one in between is freed once the
    next run's compare is enqueued.
  * traced: RUNS replays of one captured trace, each copied on device into a buffer allocated before the capture (a
    buffer allocated while the trace is live can be overwritten by the next replay), folded after the last replay.
  * traced vs eager: the first replay must equal the eager reference.

Prefixes, checked in this order on one built case per scenario:
  * zero: the perf gate's caches, zero-initialised. The indexer then scores the whole prefix equally, so top-k breaks
    many ties.
  * filled: then the prefix written by real chunk forwards of distinct inputs, as in production (warm only: the long
    prefix would take 100 forwards). Not for the GLM-5.3 shared layer: its top-k comes from a full layer's forward at
    the final start, so it cannot run the earlier chunks.

Everything a run needs is built once and reused: the layer, its caches, the RoPE tables and the input chunk. CI runs
this with DS_PREFILL_REUSE_MODEL=1, so the mesh stays open across cases.
"""

import gc

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.tests.attn_res.device_determinism import (
    device_mismatch_marker,
    device_mismatch_marker_is_set,
    fold_mismatches,
    merge_device_mismatch_markers,
    mismatch_report,
)
from models.demos.deepseek_v3_d_p.tests.sparse_mla.test_sparse_mla_perf import (
    _BOX,
    CHUNK_TOKENS,
    GALAXY_SP,
    GATE_CASES,
    SCENARIO_CACHE_TOKENS,
    _gate_params,
    build_sparse_perf_case,
    local_cache_tokens,
)
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology
from models.demos.deepseek_v3_d_p.utils.sub_device_trace import SubDeviceTraceController

RUNS = 10
PREFIXES = {"warm": ("zero", "filled"), "long": ("zero",)}


def _params():
    """The perf gate's cases x scenarios; each case checks its scenario's PREFIXES on one build."""
    params = []
    for gate in _gate_params():
        mesh, dp, variant, case_name = gate.values
        for scenario in PREFIXES:
            params.append(
                pytest.param(
                    mesh,
                    dp,
                    variant,
                    case_name,
                    scenario,
                    marks=gate.marks,
                    id=f"{gate.id or 'unsupported'}-{scenario}",
                )
            )
    return params


def _fill_prefix(mesh_device, case, cache: int, chunk: int, hidden_size: int):
    """Write the prefix [0, cache) with chunk forwards of distinct inputs, like a real prefill before `cache`.

    Each chunk is its own seeded draw (fp32, then bf16: 4x faster than drawing bf16), so no two prefix positions see
    the same hidden row. The forwards run back to back; each input is freed once its forward is enqueued."""
    x, inputs = case.x, []
    try:
        for i, start in enumerate(range(0, cache, chunk)):
            g = torch.Generator().manual_seed(1000 + i)
            inputs.append(
                ttnn.from_torch(
                    torch.randn(1, 1, chunk, hidden_size, generator=g).to(torch.bfloat16),
                    device=mesh_device,
                    dtype=ttnn.bfloat16,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    layout=ttnn.TILE_LAYOUT,
                    mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=[-2, -1]),
                )
            )
            case.x = inputs[-1]
            ttnn.deallocate(case.forward(start))
            # Freeing returns the address on the host now; anything that reuses it is enqueued later on the same CQ,
            # so it runs after this forward has read the input.
            ttnn.deallocate(inputs.pop())
        ttnn.synchronize_device(mesh_device)
    finally:
        case.x = x
        if inputs:
            _drain(mesh_device)
            for t in inputs:
                ttnn.deallocate(t)


def _drain(mesh_device):
    """Best-effort sync before freeing (on success it is a cheap second sync). Frees on one command queue are ordered
    after earlier work anyway; this guards anything that touches the buffers outside the queue, and leaves the device
    idle."""
    try:
        ttnn.synchronize_device(mesh_device)
    except Exception as e:
        logger.debug(f"drain before cleanup failed: {e}")


def _eager_runs(mesh_device, case, start):
    """RUNS forwards back to back with no sync. Returns (reference, last output, marker): run 0's output, the last
    run's, and max(ne(reference, output)) over runs 1.. folded on device. Outputs in between are freed once folded."""
    reference, last, marker = case.forward(start), None, None
    try:
        for _ in range(1, RUNS):
            out = case.forward(start)
            marker = merge_device_mismatch_markers(marker, device_mismatch_marker(reference, out))
            if last is not None:
                ttnn.deallocate(last)  # safe: later work reusing it runs after the queued compare (one CQ)
            last = out
        return reference, last, marker
    except BaseException:
        _drain(mesh_device)
        for t in (reference, last, marker):
            if t is not None:
                ttnn.deallocate(t)
        raise


def _replay_runs(mesh_device, case, start):
    """RUNS replays of one captured forward back to back, each copied on device into a buffer allocated before the
    capture. Returns (copies, marker): the copies, and max(ne(copies[0], copy)) over the rest, folded after the last
    replay. The caller frees the copies."""
    controller = SubDeviceTraceController(mesh_device)
    case.mla.set_trace_controller(controller)
    capture_out, capturing, capture_stuck, copies = None, False, False, []
    try:
        # Compile under the controller, and allocate each replay's copy now: a buffer allocated while the trace is
        # live can land on its scratch and be overwritten by the next replay.
        out = case.forward(start)
        try:
            for _ in range(RUNS):
                copies.append(ttnn.clone(out))
        finally:
            ttnn.deallocate(out)
        ttnn.synchronize_device(mesh_device)
        controller.begin_capture()
        capturing = True
        capture_out = case.forward(start)
        controller.end_capture()
        capturing = False
        for copy in copies:  # same CQ: each copy runs after its replay, before the next replay
            controller.replay(sync=False)
            ttnn.copy(capture_out, copy)
        marker = fold_mismatches(None, [copies[0]] * (RUNS - 1), copies[1:])  # after the last replay
        return copies, marker
    except BaseException:
        if capturing:  # end a failed capture first: the device cannot sync while capturing
            try:
                controller.end_capture()
            except Exception as e:
                capture_stuck = True  # still capturing: release() would raise and replace the first error
                logger.warning(f"ending the failed capture failed: {e}")
        if not capture_stuck:
            _drain(mesh_device)
        for copy in copies:
            ttnn.deallocate(copy)
        raise
    finally:
        try:
            if not capture_stuck:
                _drain(mesh_device)
                controller.release()
        finally:
            case.mla.set_trace_controller(None)
            if capture_out is not None:
                ttnn.deallocate(capture_out)


def _check(mesh_device, case, start):
    """RUNS eager runs, RUNS replays and traced vs eager, compared on device. Returns failure text, or None."""
    reference, last, eager_marker = _eager_runs(mesh_device, case, start)
    copies, replay_marker, traced_marker = [], None, None
    try:
        copies, replay_marker = _replay_runs(mesh_device, case, start)
        traced_marker = device_mismatch_marker(reference, copies[0])
        failed = [
            name
            for name, marker in (
                ("eager runs", eager_marker),
                ("replays", replay_marker),
                ("traced vs eager", traced_marker),
            )
            if device_mismatch_marker_is_set(marker)  # the only readbacks on a pass
        ]
        if not failed:
            return None
        # Host diffs only on failure; a diagnostics error must not replace the failure itself.
        try:
            detail = ""
            if "eager runs" in failed:  # eager keeps only its last run: name every chip that differed in any run
                chips = [
                    chip
                    for chip, shard in enumerate(ttnn.get_device_tensors(ttnn.from_device(eager_marker)))
                    if float(ttnn.to_torch(shard).item()) != 0.0
                ]
                detail = f"eager runs differ on chips {chips}\n"
            detail += mismatch_report(
                ["eager last run"] + [f"replay {i}" for i in range(1, RUNS)] + ["replay 0 vs eager"],
                [reference] + [copies[0]] * (RUNS - 1) + [reference],
                [last] + copies[1:] + [copies[0]],
            )
        except Exception as e:
            detail = f"(diagnostics failed: {e})"
        return f"{', '.join(failed)}:\n{detail}"
    finally:
        _drain(mesh_device)
        for t in [reference, last, eager_marker, replay_marker, traced_marker, *copies]:
            if t is not None:
                ttnn.deallocate(t)


@pytest.mark.timeout(900)
@pytest.mark.parametrize(
    "mesh_device, device_params, variant, case_name, scenario",
    _params(),
    indirect=["mesh_device", "device_params", "variant"],
)
def test_sparse_mla_determinism(mesh_device, device_params, variant, case_name, config_only, scenario):
    shared_layer = GATE_CASES[case_name][1]
    prefixes = [p for p in PREFIXES[scenario] if not (p == "filled" and shared_layer)]
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
        shared_layer=shared_layer,
    )
    try:
        failures = _run_prefixes(
            mesh_device, case, case_name, scenario, prefixes, cache, chunk, config_only.hidden_size
        )
    finally:
        # ttMLA holds reference cycles: collect now so its buffers are gone before the next case on a reused mesh.
        # (If a check raised, its traceback still holds the case; a failed test closes the reused mesh anyway.)
        _drain(mesh_device)
        del case
        gc.collect()
    # Assert after the cleanup, so a determinism failure does not keep the case alive through its traceback.
    assert not failures, "sparse MLA is not deterministic:\n" + "\n".join(failures)


def _run_prefixes(mesh_device, case, case_name, scenario, prefixes, cache, chunk, hidden_size):
    ttnn.deallocate(case.forward(cache))  # compile
    ttnn.synchronize_device(mesh_device)

    failures = []
    for prefix in prefixes:
        label = f"{case_name} sparse MLA {_BOX[0]} {scenario} {prefix} prefix"
        if prefix == "filled":
            _fill_prefix(mesh_device, case, cache, chunk, hidden_size)
        failure = _check(mesh_device, case, cache)
        if failure:
            failures.append(f"{label}: {failure}")
        else:
            logger.info(f"{label}: {RUNS} eager runs, {RUNS} replays and traced vs eager are identical")
    return failures
