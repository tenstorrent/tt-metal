# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Gate P2.8: MoE router. Order-insensitive top-6 expert-set agreement per token, and PCC of the
renormalized weights on the dense [S, 64] routing matrix."""

import pytest
import torch

from models.demos.ernie45_d_p.bringup import metrics
from models.demos.ernie45_d_p.tests.conftest import mesh_1x4
from models.demos.ernie45_d_p.tt.common import replicated_to_torch, to_mesh_activation
from models.demos.ernie45_d_p.tt.moe import TtRouter

TASK = "P2.8"


def dense_routing(idx, w, E):
    return torch.zeros(idx.shape[0], E).scatter(-1, idx.long(), w.float())


@mesh_1x4
@pytest.mark.parametrize("layer", [1, 14, 27])
def test_router(mesh_device, cfg, layer_weights, golden_2k, record, layer):
    E = cfg.moe_num_experts
    router = TtRouter(mesh_device, cfg, layer, layer_weights(layer))
    g = golden_2k.layer(0, layer)
    dense, idx, _ = router(to_mesh_activation(mesh_device, g["ffn_norm"].float()))
    got_dense = replicated_to_torch(dense)[0, 0].float()
    got_idx = replicated_to_torch(idx)[0, 0].long()
    want_dense = dense_routing(g["topk_idx"], g["topk_w"], E)
    got_sets = torch.zeros(got_idx.shape[0], E).scatter(-1, got_idx, 1.0)
    want_sets = (want_dense > 0).float()
    overlap = (got_sets * want_sets).sum(-1) / cfg.moe_k  # fraction of the 6 experts agreed per token
    exact = (overlap == 1).float().mean().item()
    frac = overlap.mean().item()
    print(f"L{layer}: expert-set agreement {frac:.5f}, tokens with identical top-6 {exact:.4f}")
    metrics.record(record.task, f"topk_set_match_frac_L{layer:02d}", frac)
    metrics.record(record.task, f"topk_exact_tokens_L{layer:02d}", exact)
    record(f"pcc_topk_w_L{layer:02d}", got_dense, want_dense, 0.99)
    assert frac >= 0.97
    record.check()
