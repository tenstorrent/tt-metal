# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Gate P2.9: full MoE block (router + EP4 routed experts + TP4 shared experts + all_reduce) vs golden mlp_out."""

import pytest

from models.demos.ernie45_d_p.tests.conftest import mesh_1x4
from models.demos.ernie45_d_p.tt.common import replicated_to_torch, to_mesh_activation
from models.demos.ernie45_d_p.tt.moe import TtMoE

TASK = "P2.9"


@mesh_1x4
@pytest.mark.parametrize("layer", [1, 14, 27])
def test_moe(mesh_device, cfg, layer_weights, golden_2k, record, layer):
    moe = TtMoE(mesh_device, cfg, layer, layer_weights(layer))
    for c in range(golden_2k.seq // golden_2k.chunk):
        g = golden_2k.layer(c, layer)
        y = replicated_to_torch(moe(to_mesh_activation(mesh_device, g["ffn_norm"].float())))[0, 0]
        record(f"pcc_moe_L{layer:02d}_c{c}", y, g["mlp_out"], 0.99)
    record.check()
