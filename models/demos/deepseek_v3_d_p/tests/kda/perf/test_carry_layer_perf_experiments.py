# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Paired production-dimension layer costs of KDA carry candidates."""
import json
import time
from contextlib import contextmanager
from functools import partial

import pytest
import torch

import ttnn
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric_1d_device_params
from models.demos.deepseek_v3_d_p.tests.kda.perf.test_carry_experiments import composed_exchange
from models.demos.deepseek_v3_d_p.tests.kda.perf.test_layer_perf import _deallocate_state
from models.demos.deepseek_v3_d_p.tests.kda.perf.test_state_publication_experiments import publish
from models.demos.deepseek_v3_d_p.tests.kda.utils import make_kimi_k3_device_case, make_synthetic_kimi_k3_test_case
from models.demos.deepseek_v3_d_p.tt.kda import kda as kda_module
from models.demos.deepseek_v3_d_p.tt.kda import recurrence


def snapshot(output, state):
    return [
        ttnn.to_torch(shard).clone()
        for tensor in (output, state.recurrent, state.convolution)
        for shard in ttnn.get_device_tensors(tensor)
    ]


@contextmanager
def candidate_patch(monkeypatch, variant, axis, source):
    with monkeypatch.context() as patch:
        if variant in ("shared", "separate_padded", "separate_compact", "separate_broadcast"):
            patch.setattr(kda_module, "exchange_split_convolution_carry", partial(composed_exchange, variant=variant))
        elif variant == "recurrent_broadcast":
            # Mark the exact final-state tensor; intercept only its gather and following slice.
            original_last = recurrence._last_group_state
            original_gather, original_slice = ttnn.all_gather, ttnn.slice
            marked = {}

            def last(*args, **kwargs):
                result = original_last(*args, **kwargs)
                marked["input"] = result
                return result

            def gather(tensor, *args, **kwargs):
                if tensor is marked.get("input") and kwargs.get("cluster_axis") == axis:
                    result = publish(tensor, axis=axis, source=source, variant="all_broadcast")
                    marked["published"] = result
                    return result
                return original_gather(tensor, *args, **kwargs)

            def sliced(tensor, *args, **kwargs):
                if tensor is marked.get("published"):
                    marked.pop("published")
                    return tensor
                return original_slice(tensor, *args, **kwargs)

            patch.setattr(recurrence, "_last_group_state", last)
            patch.setattr(ttnn, "all_gather", gather)
            patch.setattr(ttnn, "slice", sliced)
        yield


@pytest.mark.parametrize(
    "mesh_device,tp_axis,device_params",
    [
        pytest.param((1, 8), 1, fabric_1d_device_params(trace_region_size=32 * 1024 * 1024), id="SP1xTP8"),
        pytest.param((2, 4), 1, fabric_1d_device_params(trace_region_size=32 * 1024 * 1024), id="SP2xTP4"),
        pytest.param((4, 2), 1, fabric_1d_device_params(trace_region_size=32 * 1024 * 1024), id="SP4xTP2"),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("sequence", [1280, 5120])
def test_layer_candidate_costs(mesh_device, tp_axis, device_params, sequence, monkeypatch):
    torch.set_num_threads(1)
    axis = 1 - tp_axis
    sp = tuple(mesh_device.shape)[axis]
    local_rows = sequence // sp
    start_position = (sp - 1) * local_rows + 32
    case = make_synthetic_kimi_k3_test_case(sequence=sequence)
    layer, hidden = make_kimi_k3_device_case(mesh_device, case, tensor_parallel_axis=tp_axis, cache_weights=False)
    state = layer.allocate_state(batch_size=1)
    # Create a nonzero continuation state through the actual baseline layer.
    output, nonzero_state = layer.forward(hidden, state, 0)
    ttnn.synchronize_device(mesh_device)
    ttnn.deallocate(output)
    _deallocate_state(state)
    state = nonzero_state
    before = [
        ttnn.to_torch(shard).clone()
        for tensor in (state.recurrent, state.convolution)
        for shard in ttnn.get_device_tensors(tensor)
    ]
    expected = None
    variants = (
        ["baseline", "shared", "separate_padded", "separate_compact", "separate_broadcast", "recurrent_broadcast"]
        if sp > 1
        else ["baseline"]
    )
    samples = {v: [] for v in variants}
    eager = {v: [] for v in variants}
    for round_index in range(10):
        ordered = variants[round_index % len(variants) :] + variants[: round_index % len(variants)]
        for variant in ordered:
            with candidate_patch(monkeypatch, variant, axis, sp - 1):
                for _ in range(2):
                    begun = time.perf_counter()
                    output, next_state = layer.forward(hidden, state, start_position)
                    ttnn.synchronize_device(mesh_device)
                    elapsed = (time.perf_counter() - begun) * 1e3
                    if expected is None:
                        expected = snapshot(output, next_state)
                    ttnn.deallocate(output)
                    _deallocate_state(next_state)
                eager[variant].append(elapsed)
                trace = ttnn.begin_trace_capture(mesh_device, cq_id=0)
                output, next_state = layer.forward(hidden, state, start_position)
                ttnn.end_trace_capture(mesh_device, trace, cq_id=0)
                try:
                    ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=True)
                    assert all(torch.equal(a, b) for a, b in zip(snapshot(output, next_state), expected))
                    begun = time.perf_counter()
                    for _ in range(10):
                        ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=False)
                    ttnn.synchronize_device(mesh_device)
                    samples[variant].append((time.perf_counter() - begun) * 100)
                    assert all(torch.equal(a, b) for a, b in zip(snapshot(output, next_state), expected))
                finally:
                    ttnn.release_trace(mesh_device, trace)
                    ttnn.deallocate(output)
                    _deallocate_state(next_state)
    after = [
        ttnn.to_torch(shard)
        for tensor in (state.recurrent, state.convolution)
        for shard in ttnn.get_device_tensors(tensor)
    ]
    assert all(torch.equal(a, b) for a, b in zip(before, after))
    print(
        "KDA_LAYER_CANDIDATES="
        + json.dumps(
            dict(
                sp=sp,
                tp=tuple(mesh_device.shape)[tp_axis],
                sequence=sequence,
                actual_start=start_position,
                trace_ms=samples,
                eager_ms=eager,
                nonzero_state=True,
                bit_identical=True,
            ),
            sort_keys=True,
        )
    )
    _deallocate_state(state)


@pytest.mark.parametrize(
    "mesh_device,device_params",
    [pytest.param((2, 4), fabric_1d_device_params(), id="SP2xTP4-C2560")],
    indirect=["mesh_device", "device_params"],
)
def test_production_group_and_program_counts(mesh_device, device_params, monkeypatch):
    """Profile the combined composition separately from trace timing samples."""
    from models.demos.deepseek_v3_d_p.tests.kda.perf.test_layer_perf import _log_device_program_times

    torch.set_num_threads(1)
    case = make_synthetic_kimi_k3_test_case(sequence=5120)
    layer, hidden = make_kimi_k3_device_case(mesh_device, case, tensor_parallel_axis=1, cache_weights=False)
    for variant in ("baseline", "separate_broadcast"):
        with candidate_patch(monkeypatch, variant, 0, 1):
            state = layer.allocate_state(batch_size=1)
            for _ in range(2):
                output, next_state = layer.forward(hidden, state, 2592)
                ttnn.synchronize_device(mesh_device)
                ttnn.deallocate(output)
                _deallocate_state(next_state)
            _deallocate_state(state)
            groups = []
            original = ttnn.experimental.kda.recurrent_chunk_scan
            with monkeypatch.context() as patch:

                def scan(*args, **kwargs):
                    groups.append(kwargs["groups_per_head"])
                    return original(*args, **kwargs)

                patch.setattr(ttnn.experimental.kda, "recurrent_chunk_scan", scan)
                programs = _log_device_program_times(
                    mesh_device, layer, hidden, f"{variant}-SP2xTP4-C2560", actual_start=2592
                )
            assert groups == [4], groups
            assert sum(p["name"] == "experimental.kda.recurrent_chunk_scan" for p in programs) == 1
            assert all(p["complete"] for p in programs)
            print(f"KDA_COMBINED_GROUP_GATE variant={variant} groups={groups} recurrent_scan_count=1 PASS")
