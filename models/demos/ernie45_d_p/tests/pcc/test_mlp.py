# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Gate P2.7: dense SwiGLU (layer 0, FFN 12288) and shared experts (FFN 3072), TP4 + all_reduce."""

import pytest

from models.demos.ernie45_d_p.tests.conftest import mesh_1x4
from models.demos.ernie45_d_p.tt.common import replicated_to_torch, to_mesh_activation
from models.demos.ernie45_d_p.tt.ops import TtSwiGLU, all_reduce

TASK = "P2.7"


@mesh_1x4
@pytest.mark.parametrize("layer,which", [(0, "dense"), (1, "shared"), (27, "shared")])
def test_swiglu(mesh_device, cfg, layer_weights, golden_2k, record, layer, which):
    w = layer_weights(layer)
    mlp = TtSwiGLU(mesh_device, w.w_gate, w.w_up, w.w_down, name=f"L{layer}/{'mlp' if which == 'dense' else 'shared'}")
    g = golden_2k.layer(0, layer)
    y = replicated_to_torch(all_reduce(mlp(to_mesh_activation(mesh_device, g["ffn_norm"].float()))))[0, 0]
    want = g["mlp_out"] if which == "dense" else g["shared_out"]
    record(f"pcc_{which}_L{layer:02d}", y, want, 0.99)
    record.check()
