# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host cold/hot timing of all_gather_rms_norm: native C++ op versus the frozen generic_op route.

Runs on the 1x4 group (conftest ``mesh_device``: FABRIC_2D_TORUS_XY, production router payload). Cases come from the
acceptance suite: the smallest (one tile-row, no gamma, Linear, 1 link), the q_a PERF FOCUS (640 x 384, ROW_MAJOR
gamma, Ring, 2 links), a HEIGHT_SHARDED input with its L1 shard output, and the indexer LayerNorm with every optional
argument set (gamma + beta, L1 interleaved output, explicit compute config). Each output is checked with the golden
suite's check_output against its pure-torch reference.
"""

import json
import os

import intree_host_comparison as hc
import all_gather_rms_norm_generic  # frozen generic route, provided by the driver
import pytest
import torch
import ttnn
from ttnn.operations import all_gather_rms_norm as op_pkg

from eval.golden_tests.all_gather_rms_norm.helpers import (
    build_memory_config,
    check_output,
    create_ttnn_gamma,
    create_ttnn_input_tensor,
    gathered_shape,
    pytorch_all_gather_rms_norm,
    random_gamma,
    random_stacked,
)

BLOCKS = int(os.environ.get("BENCH_BLOCKS", "20"))

HS = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
RM, TILE = ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT

CASES = [
    # (per-device shape, case spec)
    pytest.param(
        (1, 1, 32, 64),
        dict(
            gamma_layout=None,
            norm_type="rms_norm",
            beta=False,
            memory="dram",
            topology=ttnn.Topology.Linear,
            num_links=1,
        ),
        id="small_nogamma_linear",
    ),
    pytest.param(
        (1, 1, 640, 384),
        dict(
            gamma_layout=RM, norm_type="rms_norm", beta=False, memory="dram", topology=ttnn.Topology.Ring, num_links=2
        ),
        id="qa_focus_ring_L2",
    ),
    pytest.param(
        (1, 1, 640, 384),
        dict(
            gamma_layout=RM,
            norm_type="rms_norm",
            beta=False,
            memory="height_sharded",
            topology=ttnn.Topology.Ring,
            num_links=2,
        ),
        id="height_sharded",
    ),
    pytest.param(
        (1, 1, 640, 32),
        dict(gamma_layout=RM, norm_type="layer_norm", beta=True, memory="l1", topology=ttnn.Topology.Ring, num_links=2),
        id="ln_beta_l1_out",
    ),
]


@pytest.mark.parametrize(("shape", "spec"), CASES)
def test_host_perf(mesh_device, shape, spec, request):
    mesh = mesh_device
    cluster_axis = 1
    G = tuple(mesh.shape)[cluster_axis]
    out_shape = gathered_shape(shape, G)
    ln = spec["norm_type"] == "layer_norm"
    x = random_stacked(mesh, shape, torch.bfloat16, seed=0, row_offset_scale=2.0 if ln else 0.0)
    gamma = random_gamma(out_shape[-1], torch.bfloat16, seed=1) if spec["gamma_layout"] is not None else None
    beta = random_gamma(out_shape[-1], torch.bfloat16, seed=2) if spec["beta"] else None
    expected = pytorch_all_gather_rms_norm(x, gamma, cluster_axis, 1e-6, norm_type=spec["norm_type"], beta=beta)

    in_layout = HS if spec["memory"] == "height_sharded" else ttnn.TensorMemoryLayout.INTERLEAVED
    kwargs = {
        "cluster_axis": cluster_axis,
        "epsilon": 1e-6,
        "topology": spec["topology"],
        "num_links": spec["num_links"],
    }
    if ln:
        kwargs["norm_type"] = "layer_norm"
        config = ttnn.ComputeConfigDescriptor()
        config.math_fidelity = ttnn.MathFidelity.HiFi4
        config.fp32_dest_acc_en = True
        config.math_approx_mode = False
        kwargs["compute_kernel_config"] = config
    if spec["memory"] == "l1":
        kwargs["memory_config"] = ttnn.L1_MEMORY_CONFIG
    elif spec["memory"] == "height_sharded":
        kwargs["memory_config"] = build_memory_config(HS, TILE, out_shape, mesh, dtype=ttnn.bfloat16)

    def make_input():
        return create_ttnn_input_tensor(x, mesh, dtype=ttnn.bfloat16, layout=TILE, memory_layout=in_layout)

    def make_kwargs():
        k = dict(kwargs)
        if gamma is not None:
            k["gamma"] = create_ttnn_gamma(gamma, mesh, dtype=ttnn.bfloat16, layout=spec["gamma_layout"])
        if beta is not None:
            k["beta"] = create_ttnn_gamma(beta, mesh, dtype=ttnn.bfloat16, layout=spec["gamma_layout"])
        return k

    def validate(out):
        check_output(
            out, expected, shape=out_shape, dtype=ttnn.bfloat16, fp32_dest_acc_en=True, expected_layout=TILE, group=G
        )

    def route(name, op):
        return hc.RouteAdapter(
            name, lambda case, phase, block: hc.BoundCall(op, validate, (make_input(),), make_kwargs())
        )

    native = lambda *a, **k: op_pkg.all_gather_rms_norm(*a, **k)  # noqa: E731
    case_id = request.node.callspec.id
    result = hc.HostComparison(
        route("generic", all_gather_rms_norm_generic.all_gather_rms_norm),
        route("native", native),
        hc.BenchmarkRuntime.from_ttnn(mesh),
    ).run(case_id, None, blocks=BLOCKS)
    out = hc.evidence_path(case_id)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result.as_dict(), indent=1, sort_keys=True))
