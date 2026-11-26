# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Sparse MLA determinism at the perf gate's production shapes and boxes (test_sparse_mla_perf.py): one chunk over a
50k or 500k cache, the scaled-FP8 KVPE cache, each model's fabric packet size, for K3.1 (last DSA layer's
slot of 24), GLM-5.3 (last full indexer layer) and a GLM-5.3 shared layer.

The check stays on device, as in tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py (tests/measure/determinism.py):
run 0's output is the reference, every later output folds max(ne(reference, output)) into one marker, and the markers
are read back once at the end. Nothing syncs inside the loops, so a fast chip can get ahead of a slow one, which is
what exposes a race. Only a failure reads outputs back, to say which chips differ.
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
from models.demos.deepseek_v3_d_p.tests.measure.determinism import RUNS, check_determinism
from models.demos.deepseek_v3_d_p.tests.measure.trace import TracedStep, drain
from models.demos.deepseek_v3_d_p.tests.sparse_mla.perf_cases import (
    BOX,
    GATE_CASES,
    SCENARIO_CACHE_TOKENS,
    build_sparse_perf_case,
    gate_params,
)
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology

PREFIXES = {"warm": ("zero", "filled"), "long": ("zero",)}


def _params():
    """The perf gate's cases x scenarios; each case checks its scenario's PREFIXES on one build."""
    params = []
    for gate in gate_params():
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
                    mesh_mapper=ttnn.ShardTensor2dMesh(
                        mesh_device, mesh_shape=tuple(mesh_device.shape), dims=case.input_shard_dims
                    ),
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
            drain(mesh_device)
            for t in inputs:
                ttnn.deallocate(t)


def _check(mesh_device, case, start):
    """RUNS eager runs, RUNS replays and traced vs eager, compared on device. Returns failure text, or None."""
    return check_determinism(
        mesh_device,
        lambda: case.forward(start),
        lambda: TracedStep(
            mesh_device,
            lambda: case.forward(start),
            attach=case.mla.set_trace_controller,
            detach=lambda: case.mla.set_trace_controller(None),
        ),
    )


@pytest.mark.timeout(900)
@pytest.mark.parametrize(
    "mesh_device, device_params, variant, case_name, scenario",
    _params(),
    indirect=["mesh_device", "device_params", "variant"],
)
def test_sparse_mla_determinism(mesh_device, device_params, variant, case_name, config_only, scenario):
    shared_layer = GATE_CASES[case_name][1]
    prefixes = [p for p in PREFIXES[scenario] if not (p == "filled" and shared_layer)]
    chunk = BOX.chunk
    cache = BOX.cache_tokens(SCENARIO_CACHE_TOKENS[scenario])
    case = build_sparse_perf_case(
        mesh_device,
        variant,
        config_only,
        cache=cache,
        chunk=chunk,
        topology=per_axis_topology(device_params["fabric_config"]),
        shared_layer=shared_layer,
        sp_axis=BOX.sp_axis,
    )
    try:
        failures = _run_prefixes(
            mesh_device, case, case_name, scenario, prefixes, cache, chunk, config_only.hidden_size
        )
    finally:
        # ttMLA holds reference cycles: collect now so its buffers are gone before the next case on a reused mesh.
        # (If a check raised, its traceback still holds the case; a failed test closes the reused mesh anyway.)
        drain(mesh_device)
        del case
        gc.collect()
    # Assert after the cleanup, so a determinism failure does not keep the case alive through its traceback.
    assert not failures, "sparse MLA is not deterministic:\n" + "\n".join(failures)


def _run_prefixes(mesh_device, case, case_name, scenario, prefixes, cache, chunk, hidden_size):
    ttnn.deallocate(case.forward(cache))  # compile
    ttnn.synchronize_device(mesh_device)

    failures = []
    for prefix in prefixes:
        label = f"{case_name} sparse MLA {BOX.name} {scenario} {prefix} prefix"
        if prefix == "filled":
            _fill_prefix(mesh_device, case, cache, chunk, hidden_size)
        failure = _check(mesh_device, case, cache)
        if failure:
            failures.append(f"{label}: {failure}")
        else:
            logger.info(f"{label}: {RUNS} eager runs, {RUNS} replays and traced vs eager are identical")
    return failures
