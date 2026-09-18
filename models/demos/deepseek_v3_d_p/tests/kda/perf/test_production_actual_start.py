# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Matched None/zero timings on calls intercepted from production K3 execution."""

import json
import os
import time
from dataclasses import replace
from pathlib import Path

import pytest
import torch

import ttnn
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric_1d_device_params
from models.demos.deepseek_v3_d_p.tests.kda.checkpoint_utils import kda_state_dict_sha256
from models.demos.deepseek_v3_d_p.tests.kda.utils import (
    KimiK3TestCase,
    _deallocate_state,
    _kda_config_from_kimi_k3_constants,
    make_kimi_k3_device_case,
    random_weights,
)
from tests.ttnn.profiling.realtime_profiler_utils import profile_realtime_program
from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import make_actual_start


@pytest.mark.parametrize(
    "mesh_device,local_rows",
    [((2, 4), 2560), ((2, 4), 640), ((4, 2), 640)],
    indirect=["mesh_device"],
    ids=["production", "proxy-sp2tp4", "proxy-sp4tp2"],
)
@pytest.mark.parametrize("device_params", [fabric_1d_device_params(trace_region_size=8000000)], indirect=True)
@pytest.mark.parametrize("capture_order", [("none", "zero"), ("zero", "none")], ids=["none-first", "zero-first"])
def test_production_actual_start(
    mesh_device: ttnn.MeshDevice,
    device_params: dict,
    capture_order: tuple[str, str],
    monkeypatch: pytest.MonkeyPatch,
    local_rows: int,
) -> None:
    mesh_shape = tuple(mesh_device.shape)
    sequence = local_rows * mesh_shape[0]
    config = replace(_kda_config_from_kimi_k3_constants(), num_heads=24 * mesh_shape[1])
    weights = random_weights(config)
    case = KimiK3TestCase(
        config=config,
        state_dict=weights,
        hidden=torch.randn(
            1, sequence, config.hidden_size, generator=torch.Generator().manual_seed(1607), dtype=torch.bfloat16
        ),
        weights_identity=kda_state_dict_sha256(weights),
    )
    layer, hidden = make_kimi_k3_device_case(mesh_device, case, tensor_parallel_axis=1, cache_weights=False)
    assert layer.config.num_heads == 24
    assert layer.config.head_k_dim == layer.config.head_v_dim == 128
    actual_start = make_actual_start(mesh_device, 0)
    state = layer.allocate_state(batch_size=1)
    operations = (
        "qkv_causal_conv1d_silu",
        "summarize_chunk_recurrence",
        "reduce_affine_transforms",
        "affine_exclusive_scan",
        "recurrent_chunk_scan",
    )
    evidence = dict(
        capture_order=capture_order,
        mesh=mesh_shape,
        sequence=sequence,
        local_rows=local_rows,
        global_heads=config.num_heads,
        local_heads=24,
        operations=[],
    )
    destination = (
        Path(os.environ["KDA_EVIDENCE_DIR"])
        / f"sp{mesh_shape[0]}tp{mesh_shape[1]}-rows{local_rows}-{capture_order[0]}-first.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)

    def metadata(value):
        if isinstance(value, ttnn.Tensor):
            return dict(
                shape=list(value.shape),
                dtype=str(value.dtype),
                layout=str(value.layout),
                memory=str(value.memory_config()),
            )
        return str(value)

    def measure(name, operation, args, kwargs):
        reference = operation(*args, **kwargs)
        ordinary_args = args
        ordinary_kwargs = dict(kwargs, actual_start=None)
        for key in ("tail_a", "tail_b", "tail_state", "predecessor_carry"):
            ordinary_kwargs.pop(key, None)
        effective_history = None
        if name == "qkv_causal_conv1d_silu":
            assert tuple(args[0].shape) == (1, local_rows, 9216)
            assert args[6:9] == (3072, 3072, 3072)
            composer = ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=mesh_shape, dims=(1, 2))
            history = ttnn.to_torch(args[1], mesh_composer=composer)
            predecessor = ttnn.to_torch(kwargs["predecessor_carry"], mesh_composer=composer)
            # At start zero, SP rank 0 reads layer history; all other ranks read their predecessor.
            host_history = torch.cat((history[:, :3], predecessor[:, 3:]), dim=1)
            effective_history = ttnn.from_torch(
                host_history,
                device=mesh_device,
                dtype=args[1].dtype,
                layout=args[1].layout,
                memory_config=args[1].memory_config(),
                mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(1, 2), mesh_shape=mesh_shape),
            )
            ordinary_args = (args[0], effective_history, *args[2:])
        elif name in ("summarize_chunk_recurrence", "recurrent_chunk_scan"):
            groups = kwargs["groups_per_head"]
            assert tuple(args[0].shape) == (24 * groups, local_rows // (32 * groups), 32, 128)

        def run(variant):
            result = operation(
                *(ordinary_args if variant == "none" else args), **(ordinary_kwargs if variant == "none" else kwargs)
            )
            return list(result) if isinstance(result, (tuple, list)) else [result]

        reference = list(reference) if isinstance(reference, (tuple, list)) else [reference]
        reference_metadata = [metadata(t) for t in reference]
        live_count = 2 if name == "summarize_chunk_recurrence" else len(reference)
        reference_host = [
            [ttnn.to_torch(shard) for shard in ttnn.get_device_tensors(t)] for t in reference[:live_count]
        ]
        for tensor in reference:
            ttnn.deallocate(tensor)
        traces, outputs = {}, {}
        try:
            for variant in capture_order:
                for _ in range(2):
                    for tensor in run(variant):
                        ttnn.deallocate(tensor)
                trace = ttnn.begin_trace_capture(mesh_device, cq_id=0)
                outputs[variant] = run(variant)
                ttnn.end_trace_capture(mesh_device, trace, cq_id=0)
                traces[variant] = trace
                ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=True)
            assert [metadata(t) for t in outputs["none"]] == [metadata(t) for t in outputs["zero"]]
            max_abs = []
            for i in range(live_count):
                difference = 0.0
                for expected, none, zero in zip(
                    reference_host[i],
                    ttnn.get_device_tensors(outputs["none"][i]),
                    ttnn.get_device_tensors(outputs["zero"][i]),
                    strict=True,
                ):
                    none, zero = (ttnn.to_torch(t) for t in (none, zero))
                    assert torch.isfinite(expected).all()
                    torch.testing.assert_close(none, expected, rtol=0, atol=0)
                    torch.testing.assert_close(zero, expected, rtol=0, atol=0)
                    difference = max(difference, float((none.float() - zero.float()).abs().max()))
                max_abs.append(difference)
            for _ in range(5):
                for variant in capture_order:
                    for _ in range(100):
                        ttnn.execute_trace(mesh_device, traces[variant], cq_id=0, blocking=False)
                    ttnn.synchronize_device(mesh_device)
            samples = dict(none=[], zero=[])
            for sample in range(20):
                for variant in capture_order if sample % 2 == 0 else capture_order[::-1]:
                    begin = time.perf_counter()
                    for _ in range(100):
                        ttnn.execute_trace(mesh_device, traces[variant], cq_id=0, blocking=False)
                    ttnn.synchronize_device(mesh_device)
                    samples[variant].append((time.perf_counter() - begin) * 1e6 / 100)
            for trace in traces.values():
                ttnn.release_trace(mesh_device, trace)
            traces.clear()
            for tensors in outputs.values():
                for tensor in tensors:
                    ttnn.deallocate(tensor)
            outputs.clear()
            profiles = dict(none=[], zero=[])
            for sample in range(3):
                for variant in capture_order if sample % 2 == 0 else capture_order[::-1]:
                    profiled, records = profile_realtime_program(
                        mesh_device, lambda: run(variant), collect_all=True, record_timeout_seconds=30
                    )
                    assert len({record["chip_id"] for record in records}) == 8
                    assert len({record["runtime_id"] for record in records}) == 1
                    profiles[variant].append(records)
                    for tensor in profiled:
                        ttnn.deallocate(tensor)
            evidence["operations"].append(
                dict(
                    name=name,
                    args=[metadata(a) for a in args],
                    kwargs={k: metadata(v) for k, v in kwargs.items()},
                    outputs=reference_metadata,
                    live_max_abs_difference=max_abs,
                    trace_samples_us=samples,
                    device_profiles=profiles,
                )
            )
            destination.write_text(json.dumps(evidence, indent=2) + "\n")
        finally:
            for trace in traces.values():
                ttnn.release_trace(mesh_device, trace)
            for tensors in outputs.values():
                for tensor in tensors:
                    ttnn.deallocate(tensor)
            if effective_history is not None:
                ttnn.deallocate(effective_history)
        return operation(*args, **kwargs)

    for name in operations:
        operation = getattr(ttnn.experimental.kda, name)
        monkeypatch.setattr(
            ttnn.experimental.kda,
            name,
            lambda *args, _name=name, _operation=operation, **kwargs: measure(_name, _operation, args, kwargs),
        )
    output, next_state = layer.forward(hidden, state, actual_start)
    assert {r["name"] for r in evidence["operations"]} == set(operations)
    ttnn.deallocate(output)
    _deallocate_state(next_state)
    _deallocate_state(state)
    ttnn.deallocate(actual_start)
