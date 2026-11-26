# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host cold/hot timing of all_to_all_matmul: native C++ op versus the frozen generic_op route."""
import json
import os

import all_to_all_matmul_generic  # frozen generic route, provided by the driver
import intree_host_comparison as hc
import pytest
import torch
import ttnn

from eval.golden_tests.all_to_all_matmul.helpers import (
    _tolerance_for,
    check_output,
    create_ttnn_input_tensor,
    output_shape,
    pytorch_all_to_all_matmul,
    read_back,
    replicate,
)
from eval.metrics import LOWER_FIDELITY_PCC
from ttnn.operations import all_to_all_matmul as op_pkg

BLOCKS = int(os.environ.get("BENCH_BLOCKS", "20"))
G = 4


def _cfg(fp32_dest_acc_en, fidelity):
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = fidelity
    cfg.fp32_dest_acc_en = fp32_dest_acc_en
    cfg.math_approx_mode = False
    return cfg


# (x_shape, w_shape, weight dtype, output buffer, fp32 DEST, fidelity)
CASES = [
    pytest.param(
        (1, 4, 128, 32),
        (1, 16, 32, 32),
        ttnn.bfloat16,
        ttnn.BufferType.DRAM,
        True,
        ttnn.MathFidelity.HiFi4,
        id="smallest",
    ),
    pytest.param(
        (1, 24, 640, 128),
        (1, 96, 128, 512),
        ttnn.bfloat16,
        ttnn.BufferType.L1,
        False,
        ttnn.MathFidelity.HiFi2,
        id="kimi_model",
    ),
    pytest.param(
        (1, 32, 640, 128),
        (1, 128, 128, 512),
        ttnn.bfloat8_b,
        ttnn.BufferType.DRAM,
        True,
        ttnn.MathFidelity.HiFi4,
        id="deepseek_bf8b",
    ),
    pytest.param(
        (1, 24, 160, 128),
        (1, 96, 128, 512),
        ttnn.bfloat16,
        ttnn.BufferType.DRAM,
        True,
        ttnn.MathFidelity.HiFi4,
        id="non_aligned_40_rows",
    ),
]


@pytest.mark.parametrize(("x_shape", "w_shape", "w_dtype", "out_buffer", "fp32", "fidelity"), CASES)
def test_host_perf(mesh_device, x_shape, w_shape, w_dtype, out_buffer, fp32, fidelity, request):
    rows, cols = tuple(mesh_device.shape)
    torch.manual_seed(0)
    x = torch.randn((rows, cols, *x_shape), dtype=torch.float32).to(torch.bfloat16)
    w = (torch.randn(w_shape, dtype=torch.float32) / w_shape[-2] ** 0.5).to(torch.bfloat16)
    if w_dtype == ttnn.bfloat8_b:  # the reference multiplies by the weight as the devices hold it
        held = create_ttnn_input_tensor(replicate(w, mesh_device), mesh_device, dtype=w_dtype, layout=ttnn.TILE_LAYOUT)
        w = read_back(held, (rows, cols))[0, 0]
        ttnn.deallocate(held)
    expected = pytorch_all_to_all_matmul(x, w, 1)
    pcc, rms = _tolerance_for(ttnn.bfloat16, fp32)
    if fidelity != ttnn.MathFidelity.HiFi4:
        pcc = LOWER_FIDELITY_PCC
    kwargs = dict(
        cluster_axis=1,
        topology=ttnn.Topology.Ring,
        num_links=2,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, out_buffer),
        compute_kernel_config=_cfg(fp32, fidelity),
    )

    def make_inputs():
        return (
            create_ttnn_input_tensor(x, mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT),
            create_ttnn_input_tensor(replicate(w, mesh_device), mesh_device, dtype=w_dtype, layout=ttnn.TILE_LAYOUT),
        )

    def validate(out):
        assert out.memory_config().buffer_type == out_buffer
        check_output(
            out,
            expected,
            shape=output_shape(x_shape, w_shape, G),
            dtype=ttnn.bfloat16,
            expected_layout=ttnn.TILE_LAYOUT,
            group=G,
            tolerance=(pcc, rms),
            fp32_dest_acc_en=fp32,
        )

    def route(name, op):
        return hc.RouteAdapter(name, lambda case, phase, block: hc.BoundCall(op, validate, make_inputs(), dict(kwargs)))

    native = lambda *a, **k: op_pkg.all_to_all_matmul(*a, **k)  # noqa: E731
    case_id = request.node.callspec.id
    result = hc.HostComparison(
        route("generic", all_to_all_matmul_generic.all_to_all_matmul),
        route("native", native),
        hc.BenchmarkRuntime.from_ttnn(mesh_device),
    ).run(case_id, None, blocks=BLOCKS)
    out = hc.evidence_path(case_id)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result.as_dict(), indent=1, sort_keys=True))
