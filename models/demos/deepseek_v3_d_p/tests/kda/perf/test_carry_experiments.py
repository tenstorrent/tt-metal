# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Local KDA offset experiments; candidates are deliberately outside production dispatch."""

from __future__ import annotations

import json
import time

import pytest
import torch

import ttnn
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric_1d_device_params
from models.demos.deepseek_v3_d_p.tests.kda.components.test_convolution import _sp_carries, _to_device
from models.demos.deepseek_v3_d_p.tt.kda.chronological_topology import ChronologicalTopology, _chronological_topology
from models.demos.deepseek_v3_d_p.tt.kda.convolution import exchange_split_convolution_carry


@pytest.fixture(autouse=True)
def one_host_thread():
    """Avoid a large idle torch thread pool competing with trace submission."""
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _rows(tensor: ttnn.Tensor, start: int, count: int) -> ttnn.Tensor:
    return ttnn.slice(tensor, (0, start, 0), (tensor.shape[0], start + count, tensor.shape[2]))


def _transport(tensor: ttnn.Tensor, padded: bool) -> ttnn.Tensor:
    if not padded:
        return tensor
    tensor = ttnn.pad(tensor, ((0, 0), (0, 32 - tensor.shape[1]), (0, 0)), value=0.0)
    return ttnn.to_layout(tensor, ttnn.TILE_LAYOUT)


def composed_exchange(
    qkv: ttnn.Tensor,
    initial: ttnn.Tensor,
    *,
    sequence_parallel_axis: int,
    topology: ChronologicalTopology,
    wrap_indicator: ttnn.Tensor | None = None,
    variant: str = "shared",
) -> tuple[ttnn.Tensor, ttnn.Tensor]:
    """Compare shared six-row control with separated three-row halo/state exchange."""
    axis = sequence_parallel_axis
    sp = topology.sp_size
    end = _rows(qkv, topology.local_rows - 3, 3)
    if sp == 1:
        return initial, end
    head = _rows(qkv, topology.head_rows - 3, 3)
    if variant == "shared":
        published = _transport(ttnn.concat([head, end], dim=1), True)
        gathered = ttnn.all_gather(published, dim=1, cluster_axis=axis)

        def selected(rank, final=False):
            tile = _rows(gathered, rank * 32, 32)
            rm = ttnn.to_layout(tile, ttnn.ROW_MAJOR_LAYOUT)
            return _rows(rm, 3 if final or rank != topology.boundary_chip else 0, 3)

        final = selected(topology.boundary_chip, True)
    else:
        # mesh_partition chooses this rank's contribution without a new selector op.
        outgoing = ttnn.mesh_partition(
            ttnn.concat([head if rank == topology.boundary_chip else end for rank in range(sp)], dim=1),
            dim=1,
            cluster_axis=axis,
        )
        padded = variant == "separate_padded"
        published = _transport(outgoing, padded)
        gathered = ttnn.all_gather(published, dim=1, cluster_axis=axis)
        stride = 32 if padded else 3

        def selected(rank, final=False):
            item = _rows(gathered, rank * stride, stride)
            return _rows(ttnn.to_layout(item, ttnn.ROW_MAJOR_LAYOUT), 0, 3)

        # Hold final-state publication fixed between padded/compact halo variants.
        if variant == "separate_broadcast":
            finals = ttnn.all_broadcast(end, cluster_axis=axis)
            final = finals[topology.boundary_chip]
            for rank, tensor in enumerate(finals):
                if rank != topology.boundary_chip:
                    ttnn.deallocate(tensor)
        else:
            final_gather = ttnn.all_gather(_transport(end, True), dim=1, cluster_axis=axis)
            final = _rows(
                ttnn.to_layout(_rows(final_gather, topology.boundary_chip * 32, 32), ttnn.ROW_MAJOR_LAYOUT), 0, 3
            )
    entries = []
    for rank in range(sp):
        entry = initial if rank == topology.boundary_chip else selected(topology.predecessor_chip(rank))
        entries.append(entry)
        entries.append(selected(topology.predecessor_chip(rank)) if rank == topology.boundary_chip else entry)
    return ttnn.mesh_partition(ttnn.concat(entries, dim=1), dim=1, cluster_axis=axis), final


def _release(outputs: tuple[ttnn.Tensor, ttnn.Tensor], initial: ttnn.Tensor) -> None:
    for tensor in outputs:
        if tensor is not initial:
            ttnn.deallocate(tensor)


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
@pytest.mark.parametrize("local_rows", [640, 2560])
def test_carry_candidates(mesh_device, tp_axis, device_params, local_rows):
    """Sentinel routing for every boundary, trace replay, input preservation and paired timing."""
    axis = 1 - tp_axis
    sp = tuple(mesh_device.shape)[axis]
    tp = tuple(mesh_device.shape)[tp_axis]
    width = 96 * 128 * 3
    generator = torch.Generator().manual_seed(128)
    qkv = torch.randn(1, sp * local_rows, width, generator=generator).bfloat16()
    initial = torch.randn(1, 3, width, generator=generator).bfloat16()
    dims = [None, None]
    dims[axis], dims[tp_axis] = 1, 2
    state_dims = [None, None]
    state_dims[tp_axis] = 2
    qkv_tt = _to_device(qkv, mesh_device, tuple(dims))
    initial_tt = _to_device(initial, mesh_device, tuple(state_dims))
    variants = (
        ["baseline", "shared", "separate_padded", "separate_compact", "separate_broadcast"] if sp > 1 else ["shared"]
    )
    for boundary in range(sp):
        for tail_rows in (32, local_rows // 2, local_rows - 32):
            topology = _chronological_topology(boundary * local_rows + tail_rows, sp, local_rows)
            indicator_dims = [None, None]
            indicator_dims[axis] = 0
            indicator = torch.zeros(sp, 1, 1)
            indicator[boundary] = 1
            indicator_tt = ttnn.from_torch(
                indicator,
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensor2dMesh(
                    mesh_device, dims=tuple(indicator_dims), mesh_shape=tuple(mesh_device.shape)
                ),
            )

            def run(variant):
                fn = exchange_split_convolution_carry if variant == "baseline" else composed_exchange
                kwargs = {} if variant == "baseline" else {"variant": variant}
                return fn(
                    qkv_tt,
                    initial_tt,
                    sequence_parallel_axis=axis,
                    topology=topology,
                    wrap_indicator=indicator_tt,
                    **kwargs,
                )

            expected_entries = []
            for rank in range(sp):
                previous = (rank - 1) % sp
                end_row = previous * local_rows + (topology.head_rows if previous == boundary else local_rows)
                outgoing = qkv[:, end_row - 3 : end_row]
                entry = initial if rank == boundary else outgoing
                expected_entries.append(
                    torch.cat([entry, outgoing if rank == boundary else entry], dim=1) if sp > 1 else initial
                )
            expected_entries = torch.stack(expected_entries)
            end_row = (boundary + 1) * local_rows
            expected_final = qkv[:, end_row - 3 : end_row]

            def check(outputs):
                actual_entries, actual_final = [_sp_carries(t, mesh_device, axis, tp_axis) for t in outputs]
                assert torch.equal(actual_entries, expected_entries)
                assert all(torch.equal(item, expected_final) for item in actual_final)

            for variant in variants:
                for _ in range(2):
                    outputs = run(variant)
                    check(outputs)
                    _release(outputs, initial_tt)
            # Timing subset; exhaustive owners/wraps above remain correctness checks.
            if boundary == 0 and tail_rows == local_rows // 2:
                from models.demos.deepseek_v3_d_p.tests.kda.perf.carry_experiment_resources import capture_resources

                for variant in variants:
                    resource_outputs = capture_resources(
                        lambda: run(variant), mesh_device, f"halo-SP{sp}TP{tp}-axis{axis}-C{local_rows}-{variant}"
                    )
                    if resource_outputs is not None:
                        check(resource_outputs)
                        _release(resource_outputs, initial_tt)
                samples = {variant: [] for variant in variants}
                eager = {variant: [] for variant in variants}
                for round_index in range(10):
                    order = variants[round_index % len(variants) :] + variants[: round_index % len(variants)]
                    for variant in order:
                        ttnn.synchronize_device(mesh_device)
                        start = time.perf_counter()
                        outputs = run(variant)
                        ttnn.synchronize_device(mesh_device)
                        eager[variant].append((time.perf_counter() - start) * 1e6)
                        _release(outputs, initial_tt)
                        trace = ttnn.begin_trace_capture(mesh_device, cq_id=0)
                        outputs = run(variant)
                        ttnn.end_trace_capture(mesh_device, trace, cq_id=0)
                        try:
                            for _ in range(2):
                                ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=True)
                                check(outputs)
                            start = time.perf_counter()
                            for _ in range(100):
                                ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=False)
                            ttnn.synchronize_device(mesh_device)
                            samples[variant].append((time.perf_counter() - start) * 1e4)
                            check(outputs)
                        finally:
                            ttnn.release_trace(mesh_device, trace)
                            _release(outputs, initial_tt)
                print(
                    "KDA_CARRY_EXPERIMENT="
                    + json.dumps(
                        dict(
                            sp=sp,
                            tp=tp,
                            axis=axis,
                            local_rows=local_rows,
                            trace_us=samples,
                            eager_us=eager,
                            repetitions=100,
                            logical_halo_bytes=3 * width // tp * 2,
                            padded_halo_bytes=32 * width // tp * 2,
                        ),
                        sort_keys=True,
                    )
                )
            if boundary == 0 and tail_rows == local_rows // 2:
                # Fresh addresses AND different values exercise cached runtime rebinding.
                old_qkv, old_initial = qkv_tt, initial_tt
                cache_entries = mesh_device.num_program_cache_entries()
                qkv_tt = _to_device(-qkv, mesh_device, tuple(dims))
                initial_tt = _to_device(-initial, mesh_device, tuple(state_dims))
                assert qkv_tt.buffer_address() != old_qkv.buffer_address()
                expected_entries, expected_final = -expected_entries, -expected_final
                for variant in variants:
                    outputs = run(variant)
                    check(outputs)
                    _release(outputs, initial_tt)
                assert mesh_device.num_program_cache_entries() == cache_entries
                ttnn.deallocate(qkv_tt)
                ttnn.deallocate(initial_tt)
                qkv_tt, initial_tt = old_qkv, old_initial
                print(f"KDA_CARRY_CACHE_REBIND SP={sp} TP={tp} axis={axis} C={local_rows} entries={cache_entries} PASS")
            ttnn.deallocate(indicator_tt)
    assert torch.equal(_sp_carries(qkv_tt, mesh_device, axis, tp_axis), qkv.reshape(sp, 1, local_rows, width))
    assert all(torch.equal(item, initial) for item in _sp_carries(initial_tt, mesh_device, axis, tp_axis))
