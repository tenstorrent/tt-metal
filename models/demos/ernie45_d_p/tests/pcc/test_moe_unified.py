# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Gate P3.1: MoE via the fused unified_routed_expert_moe EP pipeline on the 1x4 mesh (dispatch group of 1 chip).

Checks the routed part alone (vs golden routed_out) and the full MoE (vs golden mlp_out), both chunks of
2k->2k, and times it against the dense-EP TtMoE on the same inputs (warm, second run).
"""

import time

import pytest

import ttnn
from models.demos.ernie45_d_p.bringup import metrics
from models.demos.ernie45_d_p.tests.conftest import mesh_1x4
from models.demos.ernie45_d_p.tt.common import replicated_to_torch, to_mesh_activation
from models.demos.ernie45_d_p.tt.moe import TtMoE
from models.demos.ernie45_d_p.tt.moe_unified import TtMoEUnified

TASK = "P3.2"


def _timed(fn, mesh, reps=2):
    out = None
    for _ in range(reps):  # first rep compiles; report the last
        t0 = time.time()
        out = fn()
        ttnn.synchronize_device(mesh)
        dt = time.time() - t0
    return out, dt


@mesh_1x4
@pytest.mark.parametrize("layer", [1, 14, 27])
def test_moe_unified(mesh_device, cfg, layer_weights, golden_2k, record, layer):
    w = layer_weights(layer)
    moe = TtMoEUnified(mesh_device, cfg, layer, w)
    for c in range(golden_2k.seq // golden_2k.chunk):
        g = golden_2k.layer(c, layer)
        x = to_mesh_activation(mesh_device, g["ffn_norm"].float())
        dbg = {}
        y, dt = _timed(lambda: moe(x, debug=dbg), mesh_device)
        routed = replicated_to_torch(dbg["routed"])[0, 0]
        record(f"pcc_routed_L{layer:02d}_c{c}", routed, g["routed_out"], 0.99)
        record(f"pcc_moe_L{layer:02d}_c{c}", replicated_to_torch(y)[0, 0], g["mlp_out"], 0.99)
        metrics.record(record.task, f"unified_moe_ms_L{layer:02d}_c{c}", round(dt * 1e3, 2))
        if c == 0:
            dense = TtMoE(mesh_device, cfg, layer, w)
            _, dt_dense = _timed(lambda: dense(x), mesh_device)
            metrics.record(record.task, f"dense_ep_moe_ms_L{layer:02d}", round(dt_dense * 1e3, 2))
            metrics.record(record.task, f"speedup_L{layer:02d}", round(dt_dense / dt, 2))
            print(f"L{layer} S={golden_2k.chunk}: unified {dt * 1e3:.1f} ms vs dense-EP {dt_dense * 1e3:.1f} ms")
            del dense
    record.check()
