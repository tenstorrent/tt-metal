# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Distributed KDA PCC: mesh (SP,TP) vs torch reference. Phase 8 Option B.

import pytest
import torch
import ttnn
from loguru import logger

from models.experimental.kimi_delta_attention.torch_functional import KimiDeltaAttentionRef
from models.experimental.kimi_delta_attention.tt.ttnn_kda_dist import TtKimiDeltaAttentionMesh
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import create_fabric_router_config, get_max_payload_size
from models.common.utility_functions import comp_pcc

torch.manual_seed(2)

_FABRIC_2D = {
    "fabric_config": ttnn.FabricConfig.FABRIC_2D,
    "fabric_router_config": create_fabric_router_config(max_payload_size=get_max_payload_size()),
    "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
}


@pytest.mark.parametrize("device_params", [_FABRIC_2D], indirect=True)
@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
@pytest.mark.parametrize("T,use_conv", [(16, True), (16, False)])
def test_kda_dist_tp(mesh_device, T, use_conv):
    """TP=4 head-sharded KDA layer vs torch reference on LoudBox (2,4); sequence replicated over SP."""
    hidden, head_dim, nh = 256, 64, 4  # 4 heads / TP4 = 1 head/chip
    m = KimiDeltaAttentionRef(
        hidden_size=hidden, head_dim=head_dim, num_heads=nh, num_v_heads=nh,
        conv_size=4, use_short_conv=use_conv, mode="recurrent",
    ).eval()
    x = torch.randn(1, T, hidden)
    with torch.no_grad():
        y_ref = m(x)

    tt = TtKimiDeltaAttentionMesh(m, mesh_device)
    y = tt.forward(x)

    ok, pcc = comp_pcc(y_ref, y, pcc=0.98)
    logger.info(f"[kda_dist_tp] T={T} use_conv={use_conv} mesh=(2,4) SP-replicated PCC={pcc}")
    assert ok, f"PCC too low: {pcc}"
