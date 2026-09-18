# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Existing collective alternatives for replicated KDA final states."""

import json
import os
import time

import pytest
import torch

import ttnn
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric_1d_device_params


def publish(state: ttnn.Tensor, *, axis: int, source: int, variant: str) -> ttnn.Tensor:
    if tuple(state.device().shape)[axis] == 1:
        return state
    if variant == "all_broadcast":
        results = ttnn.all_broadcast(state, cluster_axis=axis)
        result = results[source]
        for i, tensor in enumerate(results):
            if i != source:
                ttnn.deallocate(tensor)
        return result
    gathered = ttnn.all_gather(state, dim=0, cluster_axis=axis)
    shape = tuple(state.shape)
    lo = [0] * len(shape)
    hi = list(shape)
    lo[0], hi[0] = source * shape[0], (source + 1) * shape[0]
    return ttnn.slice(gathered, lo, hi)


@pytest.mark.parametrize(
    "mesh_device,tp_axis,device_params",
    [
        pytest.param((1, 8), 1, fabric_1d_device_params(trace_region_size=8 * 1024 * 1024), id="SP1xTP8"),
        pytest.param((2, 4), 1, fabric_1d_device_params(trace_region_size=8 * 1024 * 1024), id="SP2xTP4"),
        pytest.param((4, 2), 1, fabric_1d_device_params(trace_region_size=8 * 1024 * 1024), id="SP4xTP2"),
        pytest.param((4, 2), 0, fabric_1d_device_params(trace_region_size=8 * 1024 * 1024), id="SP2xTP4-axis1"),
        pytest.param((2, 4), 0, fabric_1d_device_params(trace_region_size=8 * 1024 * 1024), id="SP4xTP2-axis1"),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("kind", ["recurrent", "convolution"])
def test_publication_candidates(mesh_device, tp_axis, device_params, kind):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        _run_publication(mesh_device, tp_axis, kind)
    finally:
        torch.set_num_threads(previous_threads)


def _run_publication(mesh_device, tp_axis, kind):
    axis = 1 - tp_axis
    sp = tuple(mesh_device.shape)[axis]
    tp = tuple(mesh_device.shape)[tp_axis]
    shape = (sp, 96, 128, 128) if kind == "recurrent" else (sp, 3, 96 * 128 * 3)
    dims = [None, None]
    dims[axis], dims[tp_axis] = 0, 1 if kind == "recurrent" else 2
    # Each SP partition contains all heads, then TP shards its head dimension.
    host = torch.randn(shape, generator=torch.Generator().manual_seed(41))
    if kind == "convolution":
        host = host.bfloat16()
    state = ttnn.from_torch(
        host,
        dtype=ttnn.float32 if kind == "recurrent" else ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT if kind == "recurrent" else ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=tuple(dims), mesh_shape=tuple(mesh_device.shape)),
    )
    if kind == "recurrent":
        state = ttnn.reshape(state, (96 // tp, 128, 128))
    originals = [ttnn.to_torch(t).clone() for t in ttnn.get_device_tensors(state)]
    variants = ["all_gather", "all_broadcast"] if sp > 1 else ["all_gather"]

    def check(output, source):
        actual = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(output)]
        columns = tuple(mesh_device.shape)[1]
        for rank in range(sp):
            for shard in range(tp):
                dest = (rank, shard) if axis == 0 else (shard, rank)
                src = (source, shard) if axis == 0 else (shard, source)
                assert torch.equal(actual[dest[0] * columns + dest[1]], originals[src[0] * columns + src[1]])

    def release(output):
        if output is not state:
            ttnn.deallocate(output)

    for source in range(sp):
        for variant in variants:
            for _ in range(2):
                output = publish(state, axis=axis, source=source, variant=variant)
                check(output, source)
                release(output)
        if source == 0:
            from models.demos.deepseek_v3_d_p.tests.kda.perf.carry_experiment_resources import (
                capture_resources,
                profile_resources,
            )

            for variant in variants:
                resource_output = capture_resources(
                    lambda: publish(state, axis=axis, source=source, variant=variant),
                    mesh_device,
                    f"publication-{kind}-SP{sp}TP{tp}-axis{axis}-{variant}",
                )
                if resource_output is not None:
                    check(resource_output, source)
                    release(resource_output)
                    if os.environ.get("KDA_PROFILE_CARRY") == "1":
                        label = f"publication-{kind}-SP{sp}TP{tp}-axis{axis}-{variant}"
                        if sp == 1:
                            print("KDA_PROGRAM_EXPERIMENT=" + json.dumps(dict(label=label, programs=[])))
                        else:
                            profiled_output = profile_resources(
                                lambda: publish(state, axis=axis, source=source, variant=variant), mesh_device, label
                            )
                            check(profiled_output, source)
                            release(profiled_output)
        samples = {v: [] for v in variants}
        for round_index in range(10):
            order = variants if round_index % 2 == 0 else list(reversed(variants))
            for variant in order:
                trace = ttnn.begin_trace_capture(mesh_device, cq_id=0)
                output = publish(state, axis=axis, source=source, variant=variant)
                ttnn.end_trace_capture(mesh_device, trace, cq_id=0)
                try:
                    ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=True)
                    check(output, source)
                    start = time.perf_counter()
                    for _ in range(100):
                        ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=False)
                    ttnn.synchronize_device(mesh_device)
                    samples[variant].append((time.perf_counter() - start) * 1e4)
                    check(output, source)
                finally:
                    ttnn.release_trace(mesh_device, trace)
                    release(output)
        print(
            "KDA_PUBLICATION_EXPERIMENT="
            + json.dumps(
                dict(kind=kind, sp=sp, tp=tp, axis=axis, source=source, repetitions=100, trace_us=samples),
                sort_keys=True,
            )
        )
    assert all(
        torch.equal(ttnn.to_torch(t), original) for t, original in zip(ttnn.get_device_tensors(state), originals)
    )
    old_state = state
    cache_entries = mesh_device.num_program_cache_entries()
    state = ttnn.from_torch(
        -host,
        dtype=ttnn.float32 if kind == "recurrent" else ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT if kind == "recurrent" else ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=tuple(dims), mesh_shape=tuple(mesh_device.shape)),
    )
    if kind == "recurrent":
        state = ttnn.reshape(state, (96 // tp, 128, 128))
    assert state.buffer_address() != old_state.buffer_address()
    originals = [-tensor for tensor in originals]
    for source in range(sp):
        for variant in variants:
            output = publish(state, axis=axis, source=source, variant=variant)
            check(output, source)
            release(output)
    assert mesh_device.num_program_cache_entries() == cache_entries
    print(f"KDA_PUBLICATION_CACHE_REBIND {kind} SP={sp} TP={tp} axis={axis} entries={cache_entries} PASS")
