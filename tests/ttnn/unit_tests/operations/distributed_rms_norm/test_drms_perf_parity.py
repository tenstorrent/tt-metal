# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""distributed_rms_norm device time on the golden perf cases, by the drms_ccl_perf protocol (spin + 5 calls, traced).

The cases and config are the eval's LOOSE_CASES: bf16 TILE interleaved x, bf16 TILE gamma, HiFi4 + fp32 dest,
cluster_axis 1, Ring, 2 links, on the 1x4 ring stand-in. Every result lands in DRMS_PERF_OUT/<case>.json.

    DRMS_PERF_EXPERIMENTS=20 scripts/run_safe_pytest.sh --run-all \\
        tests/ttnn/unit_tests/operations/distributed_rms_norm/test_drms_perf_parity.py -s
"""

import os
from pathlib import Path

from tests.ttnn.unit_tests.operations.distributed_rms_norm import drms_ccl_perf as ccl_perf

for _k, _v in ccl_perf.PROFILER_ENV.items():  # before the mesh opens
    os.environ.setdefault(_k, _v)

import pytest  # noqa: E402
import torch  # noqa: E402
import ttnn  # noqa: E402

from tests.ttnn.unit_tests.operations.distributed_rms_norm.drms_mesh import (  # noqa: E402
    TP,
    close_ring_mesh,
    open_ring_mesh,
    pcc,
    rms_norm_ref,
    width_sharded,
)

EXPERIMENTS = int(os.environ.get("DRMS_PERF_EXPERIMENTS", "20"))
OUT = Path(os.environ.get("DRMS_PERF_OUT", "generated/drms_perf"))
EPS = 1e-6


@pytest.fixture(scope="module")
def ring_mesh():
    mesh = open_ring_mesh(ccl_perf.TRACE_REGION_SIZE)
    yield mesh
    close_ring_mesh(mesh)


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("M,W_local", [(640, 1792), (32, 1792), (2048, 1280)], ids=["640x1792", "32x1792", "2048x1280"])
def test_perf_parity(ring_mesh, M, W_local):
    mesh = ring_mesh
    torch.manual_seed(0)
    W = W_local * TP
    x = torch.randn(1, 1, M, W).to(torch.bfloat16)
    gamma = torch.randn(1, 1, 1, W).to(torch.bfloat16)
    tx = width_sharded(x, mesh, ttnn.bfloat16, ttnn.TILE_LAYOUT)
    tg = width_sharded(gamma, mesh, ttnn.bfloat16, ttnn.TILE_LAYOUT)
    ckc = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False
    )

    def call():
        return ttnn.distributed_rms_norm(
            tx,
            cluster_axis=1,
            gamma=tg,
            epsilon=EPS,
            topology=ttnn.Topology.Ring,
            num_links=2,
            compute_kernel_config=ckc,
        )

    ref = rms_norm_ref(x, gamma, EPS)
    for i, t in enumerate(ttnn.get_device_tensors(call())):
        got = ttnn.to_torch(t)
        want = ref[..., i * W_local : (i + 1) * W_local]
        assert pcc(got, want) > 0.9999, f"chip {i}: pcc {pcc(got, want)}"
    case = f"{M}x{W_local}"
    ccl_perf.run(mesh, call, experiments=EXPERIMENTS, out_path=OUT / f"{case}.json", label=case)
