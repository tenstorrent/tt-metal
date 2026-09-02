# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""vsa_sdpa re-runnability: run-to-run determinism and trace replay (the model runs traced).
Regression for a hang seen only on the traced transformer block."""

import pytest
import torch
import ttnn

from models.common.utility_functions import skip_for_wormhole_b0
from tests.ttnn.unit_tests.operations.sdpa.test_vsa_sdpa_perf import make_inputs
from tests.ttnn.utils_for_testing import comp_pcc


def _run_checks(dev, order):
    q, k, v, idx, counts, _ = make_inputs(
        dev, s_local=14464, n_blocks=1808, row_blocks=197, dense_rows=0, order=order
    )
    multi = isinstance(dev, ttnn.MeshDevice) and dev.get_num_devices() > 1
    composer = ttnn.ConcatMeshToTensor(dev, dim=0) if multi else None
    host = lambda t: ttnn.to_torch(t, mesh_composer=composer).float()
    ref = ttnn.transformer.vsa_sdpa(q, k, v, idx, counts)  # compile + untraced reference
    rep = ttnn.transformer.vsa_sdpa(q, k, v, idx, counts)  # untraced repeat
    ttnn.synchronize_device(dev)
    ref_t = host(ref)
    rep_t = host(rep)
    _, pcc_rep = comp_pcc(ref_t, rep_t, 0.9999)
    exact_rep = torch.equal(ref_t, rep_t)

    tid = ttnn.begin_trace_capture(dev, cq_id=0)
    out = ttnn.transformer.vsa_sdpa(q, k, v, idx, counts)
    ttnn.end_trace_capture(dev, tid, cq_id=0)
    for _ in range(3):
        ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(dev)
    out_t = host(out)
    ttnn.release_trace(dev, tid)
    _, pcc_tr = comp_pcc(ref_t, out_t, 0.9999)
    print(f"VSA_DETERMINISM order={order} untraced-repeat exact={exact_rep} pcc={pcc_rep:.6f} traced pcc={pcc_tr:.6f}")
    # Trace must not add error beyond the op's own run-to-run variance (window partitioning is
    # timing-dependent, so untraced repeats already differ at bf16 rounding-order level).
    assert pcc_tr >= pcc_rep - 5e-4, f"traced PCC {pcc_tr} vs untraced-repeat PCC {pcc_rep}"


@skip_for_wormhole_b0("vsa_sdpa is Blackhole-only")
@pytest.mark.parametrize(
    "device_params", [{"trace_region_size": 20_000_000, "l1_small_size": 65536}], indirect=True
)
@pytest.mark.parametrize("order", ["topk", "model"])
def test_vsa_sdpa_trace_replay(device, order):
    _run_checks(device, order)


@skip_for_wormhole_b0("vsa_sdpa is Blackhole-only")
@pytest.mark.parametrize(
    "mesh_device, device_params",
    [((4, 8), {"trace_region_size": 20_000_000, "l1_small_size": 65536})],
    indirect=["mesh_device", "device_params"],
)
def test_vsa_sdpa_trace_replay_mesh(mesh_device):
    _run_checks(mesh_device, "model")
