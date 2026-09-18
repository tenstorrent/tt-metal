# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Matched ordinary/zero-offset accuracy and interleaved warm trace timings."""

import json
import time

import pytest
import torch
import ttnn

from models.common.utility_functions import run_for_blackhole
from tests.ttnn.nightly.unit_tests.operations.experimental.kda.recurrent_chunk_scan_test_utils import (
    device_protocol,
    host_protocol,
    initial_state,
    recurrent_oracle,
    summary_oracle,
    to_device,
)
from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import (
    assert_accurate,
    qkv_device_inputs,
    qkv_reference,
)

pytestmark = run_for_blackhole()


@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 8000000}], indirect=True)
@pytest.mark.parametrize("groups,dim,chunks", [(1, 32, 4), (4, 128, 4), (4, 128, 20)])
def test_actual_start_equivalence(mesh_device, device_params, groups, dim, chunks):
    host = host_protocol(2 * groups, chunks, dim, dim, seed=991)
    inputs = device_protocol(host, mesh_device)
    seed = initial_state(2 * groups, dim, dim)
    state = to_device(seed, mesh_device)
    tail_state = to_device(seed[::groups].contiguous(), mesh_device)
    zero = ttnn.from_torch(
        torch.tensor([0], dtype=torch.int32), device=mesh_device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
    )
    affine_host = tuple(t.bfloat16().float() for t in summary_oracle(host))
    affine_a, affine_b = (to_device(t.bfloat16(), mesh_device) for t in affine_host)
    entries, reduced_a, reduced_b = [], [], []
    for head in range(2):
        carry = seed[head * groups].clone()
        total_a, total_b = torch.eye(dim), torch.zeros(dim, dim)
        for group in range(groups):
            index = head * groups + group
            entries.append(carry.clone())
            carry = affine_host[0][index] @ carry + affine_host[1][index]
            total_a = affine_host[0][index] @ total_a
            total_b = affine_host[0][index] @ total_b + affine_host[1][index]
        reduced_a.append(total_a)
        reduced_b.append(total_b)
    conv_host, (conv_input, history, taps) = qkv_device_inputs(
        mesh_device, sequence=groups * chunks * 32, widths=(dim, dim, dim)
    )
    cases = (
        ("summary", summary_oracle(host)),
        ("recurrent", recurrent_oracle(host, seed)),
        ("affine", (torch.stack(entries),)),
        ("reduce", (torch.stack(reduced_a), torch.stack(reduced_b))),
        ("convolution", qkv_reference(*conv_host, (dim, dim, dim))),
    )
    for operation, expected in cases:

        def run(actual_start):
            kwargs = dict(groups_per_head=groups, actual_start=actual_start, sequence_parallel_axis=0)
            if operation == "summary":
                return ttnn.experimental.kda.summarize_chunk_recurrence(*inputs, **kwargs)
            if operation == "recurrent":
                return ttnn.experimental.kda.recurrent_chunk_scan(
                    *inputs, state, tail_state=tail_state if actual_start is not None else None, **kwargs
                )
            if operation == "affine":
                extra = (
                    dict(tail_a=affine_a, tail_b=affine_b, tail_state=tail_state) if actual_start is not None else {}
                )
                return [
                    ttnn.experimental.kda.affine_exclusive_scan(
                        affine_a, affine_b, tail_state, local_rows=groups * chunks * 32, **kwargs, **extra
                    )
                ]
            if operation == "reduce":
                return ttnn.experimental.kda.reduce_affine_transforms(
                    affine_a, affine_b, local_rows=groups * chunks * 32, **kwargs
                )
            return ttnn.experimental.kda.qkv_causal_conv1d_silu(
                conv_input,
                history,
                *taps,
                q_width=dim,
                k_width=dim,
                v_width=dim,
                program_config=ttnn.QkvCausalConv1dSiluProgramConfig(channel_chunk_size=3 * dim),
                actual_start=actual_start,
                sequence_parallel_axis=0,
                predecessor_carry=history if actual_start is not None else None,
            )

        traces, outputs = [], []
        for actual_start in (None, zero):
            for _ in range(2):
                for tensor in run(actual_start):
                    ttnn.deallocate(tensor)
            trace = ttnn.begin_trace_capture(mesh_device, cq_id=0)
            outputs.append(run(actual_start))
            ttnn.end_trace_capture(mesh_device, trace, cq_id=0)
            traces.append(trace)
            ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=True)
        try:
            assert [(t.dtype, tuple(t.shape)) for t in outputs[0]] == [(t.dtype, tuple(t.shape)) for t in outputs[1]]
            for variant, tensors in enumerate(outputs):
                for index, golden in enumerate(expected):
                    for shard in ttnn.get_device_tensors(tensors[index]):
                        assert_accurate(
                            golden.float(),
                            ttnn.to_torch(shard).float(),
                            name=f"{operation} variant={variant} output={index}",
                            pcc_threshold=0.999,
                        )
            # Prime asynchronous replay submission as well as the captured programs.
            for trace in traces:
                for _ in range(20):
                    ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh_device)
            samples = [[], []]
            for repetition in range(10):
                for variant in (0, 1) if repetition % 2 == 0 else (1, 0):
                    begin = time.perf_counter()
                    for _ in range(20):
                        ttnn.execute_trace(mesh_device, traces[variant], cq_id=0, blocking=False)
                    ttnn.synchronize_device(mesh_device)
                    samples[variant].append((time.perf_counter() - begin) * 1000 / 20)
            differences = []
            for index in range(len(expected)):
                differences.append(
                    max(
                        float((ttnn.to_torch(a).float() - ttnn.to_torch(b).float()).abs().max())
                        for a, b in zip(
                            ttnn.get_device_tensors(outputs[0][index]),
                            ttnn.get_device_tensors(outputs[1][index]),
                            strict=True,
                        )
                    )
                )
            print(
                "KDA_OFFSET_EQUIVALENCE="
                + json.dumps(
                    dict(
                        operation=operation,
                        groups=groups,
                        dim=dim,
                        chunks=chunks,
                        none_ms=samples[0],
                        zero_ms=samples[1],
                        max_abs_difference=differences,
                        dtypes=[[str(t.dtype) for t in tensors] for tensors in outputs],
                    )
                )
            )
        finally:
            for trace in traces:
                ttnn.release_trace(mesh_device, trace)
            for tensors in outputs:
                for tensor in tensors:
                    ttnn.deallocate(tensor)
