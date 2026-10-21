# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Slab attention normalization versus independent FP32 attention and a constant-V invariant."""

import hashlib
import os

import pytest
import torch

import ttnn
from models.demos.blackhole.qwen38_flash_next.tests.tp_harness import DEVICE_PARAMS, pcc
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import sparse_sdpa_tiled as sst
from models.demos.blackhole.qwen38_flash_next.ttnn.fused.sparse_sdpa_tiled import synthetic

pytestmark = pytest.mark.skipif(os.environ.get("QWEN38_FUSED_DEVICE_TEST") != "1", reason="requires held four-die mesh")


def host(tensor):
    copies = [ttnn.to_torch(local).float() for local in ttnn.get_device_tensors(tensor)]
    assert all(torch.equal(copies[0], other) for other in copies[1:])
    return copies[0]


@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [{**DEVICE_PARAMS, "trace_region_size": 2_000_000}], indirect=True)
@pytest.mark.parametrize("rows,position,ids,capacity", [(32, 0, 64, 256), (128, 2048, 512, 2304)])
@pytest.mark.parametrize("config", [sst.HIFI2, sst.DEFAULT], ids=["served-hifi2", "hifi4-control"])
def test_slab_attention_normalization_and_reuse(mesh_device, rows, position, ids, capacity, config, record_property):
    cached = None
    observed_ones = []
    for seed in (712, 713):
        inputs = synthetic.make_inputs(S=rows, T=capacity, IDS=ids, H=6, P=position, pattern="random", seed=seed)
        inputs.kv[..., 0] = 1
        reference = sst.reference_fp32(inputs.q, inputs.kv, inputs.block_ids, inputs.positions, config=config)
        tensors = synthetic.upload(inputs, mesh_device)
        output = control = None
        try:
            output = sst.sparse_sdpa_tiled(*tensors, config=config)
            result = host(output)
            control = sst.sparse_sdpa_tiled_composed(*tensors, config=config)
            control_host = host(control)
            control_pcc = pcc(control_host, reference)
            control_relative = float(
                torch.linalg.vector_norm(control_host - reference) / torch.linalg.vector_norm(reference)
            )
            control_ones = float((control_host[..., 0] - 1).abs().max())
            record_property(
                f"control_seed{seed}",
                f"PCC={control_pcc:.9f}; relative_RMS={control_relative:.9f}; ones_error={control_ones:.9f}",
            )
            assert control_pcc >= 0.999
            assert control_relative <= 0.05
            record_property(f"tokens_seed{seed}", hashlib.sha256(result.contiguous().numpy().tobytes()).hexdigest())
            score = pcc(result, reference)
            relative = float(torch.linalg.vector_norm(result - reference) / torch.linalg.vector_norm(reference))
            ones_error = float((result[..., 0] - 1).abs().max())
            record_property(f"seed{seed}", f"PCC={score:.9f}; relative_RMS={relative:.9f}; ones_error={ones_error:.9f}")
            assert score >= 0.999
            assert relative <= 0.05
            observed_ones.append(ones_error)
            count = mesh_device.num_program_cache_entries()
            if cached is not None:
                assert count == cached
            cached = count
            trace = ttnn.begin_trace_capture(mesh_device, cq_id=0)
            try:
                sst.sparse_sdpa_tiled(*tensors, config=config, out=output)
            finally:
                ttnn.end_trace_capture(mesh_device, trace, cq_id=0)
            try:
                for _ in range(3):
                    ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=True)
                    assert torch.equal(host(output), result)
                assert mesh_device.num_program_cache_entries() == cached
            finally:
                ttnn.release_trace(mesh_device, trace)
        finally:
            for tensor in (*tensors, output, control):
                if tensor is not None:
                    ttnn.deallocate(tensor)
    # The pinned slab explicitly selects nonlegacy reciprocal. Constant V must
    # remain one to within the frozen BF16 normalization tolerance.
    assert max(observed_ones) <= 0.03
