# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Golden coverage cut of all_gather_rms_norm, run through the acceptance suite's ``all_gather_rms_norm_op`` fixture.

Every value of every axis the op's SUPPORTED lists appears in at least one case (combined freely, never their
product), plus the smallest ((1, 1, 1, 384)) and the largest ((1, 1, 2048, 1280), on each topology) golden INPUTS.
Each case is built, referenced and gated by the golden suite's own ``prepare_all_gather_rms_norm`` (pure-torch
reference, helpers.TOLERANCES on every device and every group-position slice), with the op swapped for the fixture.
No case is an INVALID cell (gamma "none" coupling, beta only with gamma + layer_norm, output placement following the
input placement) nor an EXCLUSIONS cell. Each case runs on the mesh ``topology.registry_mesh`` gives its
(topology, cluster_axis): on a 4-chip box without wrap links, its snake ring opened as 1x4 / 4x1.

Coverage (case ids below):
  dtype            float32: fp32_ln_beta_gTILE_f32 | bfloat16: most | bfloat8_b: bf8b_dest16_nogamma
  fp32_dest_acc_en True: most | False: bf8b_dest16_nogamma, w_tail_dest16_ln
  alignment        tile_aligned: largest_* | w_non_aligned: w_tail_dest16_ln | h_non_aligned: smallest, axis0_linear
  rank             4: most | 3: rank3_height_sharded | 2: rank2_width_sharded_ln
  gamma            gamma / no_gamma; dtype float32 / bfloat16 / none; layout TILE / ROW_MAJOR / none
  norm_type        rms_norm / layer_norm; beta_mode beta / no_beta
  memory_layout    INTERLEAVED / HEIGHT_SHARDED / WIDTH_SHARDED
  output_placement dram_interleaved / l1_interleaved (l1_out_real_call) / l1_sharded (the sharded cases)
  cluster_axis     1 / 0 (axis0_*); topology Ring / Linear; num_links 2 / 1
"""

from __future__ import annotations

import pytest
import ttnn

from eval.golden_tests.all_gather_rms_norm.helpers import prepare_all_gather_rms_norm
from tests.ttnn.unit_tests.operations.all_gather_rms_norm.pre_migration.agrn_cases import RegistryGroup
from tests.ttnn.unit_tests.operations.all_gather_rms_norm.pre_migration.test_acceptance import (  # noqa: F401
    all_gather_rms_norm_op,
)

BF16, F32, BF8 = ttnn.bfloat16, ttnn.float32, ttnn.bfloat8_b
TILE, RM = ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT
IL = ttnn.TensorMemoryLayout.INTERLEAVED
HS = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
WS = ttnn.TensorMemoryLayout.WIDTH_SHARDED

# The q_a RMSNorm production cell (feature_spec._PHASE0) every case varies around.
BASE = dict(
    dtype=BF16,
    fp32_dest_acc_en=True,
    layout=TILE,
    gamma_mode="gamma",
    gamma_dtype=BF16,
    gamma_layout=RM,
    norm_type="rms_norm",
    beta_mode="no_beta",
    memory_layout=IL,
    output_placement="dram_interleaved",
    cluster_axis=1,
    topology=ttnn.Topology.Ring,
    num_links=2,
)
NO_GAMMA = dict(gamma_mode="no_gamma", gamma_dtype="none", gamma_layout="none")
LN_BETA = dict(norm_type="layer_norm", beta_mode="beta")

CUT = [
    ("smallest", (1, 1, 1, 384), {}),
    ("largest_ring_L2", (1, 1, 2048, 1280), {}),
    (
        "largest_linear_L1_gTILE_f32",
        (1, 1, 2048, 1280),
        dict(topology=ttnn.Topology.Linear, num_links=1, gamma_layout=TILE, gamma_dtype=F32),
    ),
    ("ln_perf_focus_one_tile", (1, 1, 640, 32), LN_BETA),
    ("l1_out_real_call", (1, 1, 640, 384), dict(output_placement="l1_interleaved")),
    ("fp32_ln_beta_gTILE_f32", (1, 1, 640, 384), dict(dtype=F32, gamma_dtype=F32, gamma_layout=TILE, **LN_BETA)),
    ("bf8b_dest16_nogamma", (1, 1, 640, 384), dict(dtype=BF8, fp32_dest_acc_en=False, **NO_GAMMA)),
    (
        "w_tail_dest16_ln",
        (1, 1, 32, 50),
        dict(
            fp32_dest_acc_en=False,
            gamma_layout=TILE,
            norm_type="layer_norm",
            topology=ttnn.Topology.Linear,
            num_links=1,
        ),
    ),
    ("rank3_height_sharded", (1, 640, 1792), dict(memory_layout=HS, output_placement="l1_sharded")),
    ("rank2_width_sharded_ln", (640, 384), dict(memory_layout=WS, output_placement="l1_sharded", **LN_BETA)),
    ("axis0_ring_L2", (1, 1, 640, 384), dict(cluster_axis=0)),
    (
        "axis0_linear_L1_nogamma",
        (1, 1, 17, 1280),
        dict(cluster_axis=0, topology=ttnn.Topology.Linear, num_links=1, **NO_GAMMA),
    ),
]


def _params():
    out = []
    for cid, shape, over in CUT:
        axes = {**BASE, **over}
        group = RegistryGroup(axes["topology"], axes["cluster_axis"])
        out.append(pytest.param(group, (shape,), axes, id=cid))
    return out


@pytest.mark.parametrize("mesh_device,inputs,axes", _params(), indirect=["mesh_device"])
def test_golden_cut(mesh_device, inputs, axes, all_gather_rms_norm_op):
    call, check = prepare_all_gather_rms_norm(inputs, mesh_device=mesh_device, op=all_gather_rms_norm_op, **axes)
    out = call()
    if all_gather_rms_norm_op.mode == "source":
        assert all_gather_rms_norm_op.last_dispatches == 1
    check(out)
    mc = out.memory_config()
    want = {
        "dram_interleaved": (IL, ttnn.BufferType.DRAM),
        "l1_interleaved": (IL, ttnn.BufferType.L1),
        "l1_sharded": (axes["memory_layout"], ttnn.BufferType.L1),
    }[axes["output_placement"]]
    assert (mc.memory_layout, mc.buffer_type) == want, f"output placed at {mc}"
