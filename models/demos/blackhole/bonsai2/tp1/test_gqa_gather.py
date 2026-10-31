# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The reader's GQA gather: q/k carry H query heads, value head hv reads query head hv/G.

The fused recurrence used to GQA-expand q and k on the host with `repeat_interleave`, two
untilize/concat/tilize/copy chains per layer (~22 us at decode). The reader now reads query head
h/G for value head h, which costs one divide per core; this test pins that it is the SAME result as
the pre-expanded form.

Nothing in the existing suite covers gqa > 1 (test_fused_recurrent_gdn.py runs H == HV), so this is
the only guard on the change. Reference is torch: FLA's naive recurrence, with q/k expanded the way
the host used to do it.
"""
import pytest
import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.tests.gdn_fla_ref import (
    l2norm_fla,
    naive_recurrent_gated_delta_rule,
    naive_recurrent_per_token_state,
    pcc,
)
from models.experimental.gated_attention_gated_deltanet.tt.ttnn_delta_rule_ops import (
    fused_recurrent_gated_delta_rule_ttnn,
)

DK = DV = 128
SCALE = DK**-0.5


def _to_dev(mesh_device, t):
    return ttnn.from_torch(
        t,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _dev0(mesh_device, t, first=1):
    return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)).float()[:first]


def _inputs(B, T, H, G, seed):
    """q/k with H query heads, v with HV = H*G value heads (the post-conv, pre-expand decode forms)."""
    gen = torch.Generator().manual_seed(seed)
    HV = H * G
    return (
        torch.randn(B, T, H, DK, generator=gen, dtype=torch.float32),
        torch.randn(B, T, H, DK, generator=gen, dtype=torch.float32),
        torch.randn(B, T, HV, DV, generator=gen, dtype=torch.float32),
        torch.rand(B, T, HV, generator=gen, dtype=torch.float32),
        -torch.rand(B, T, HV, generator=gen, dtype=torch.float32) * 2.0,
    )


@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize("B,T,H,G", [(1, 1, 16, 3), (1, 1, 8, 4), (1, 4, 8, 3), (2, 1, 8, 2)])
def test_gqa_matches_torch(mesh_device, B, T, H, G):
    """q/k passed UNEXPANDED must give the same o and state as the expanded reference."""
    q, k, v, beta, g = _inputs(B, T, H, G, seed=B * 100 + T)
    HV = H * G
    s0 = torch.randn(B, HV, DK, DV, dtype=torch.float32) * 0.1

    # Reference: expand q/k the way the host used to, then FLA's naive recurrence.
    qe = q.repeat_interleave(G, dim=2)
    ke = k.repeat_interleave(G, dim=2)
    qn, kn = l2norm_fla(qe), l2norm_fla(ke)
    if T == 1:
        o_ref, h_ref = naive_recurrent_gated_delta_rule(
            qn, kn, v, beta, g, scale=SCALE, initial_state=s0, output_final_state=True
        )
    else:
        o_ref, h_per = naive_recurrent_per_token_state(qn, kn, v, beta, g, scale=SCALE, initial_state=s0)
        h_ref = h_per[:, -1]

    q_tt, k_tt, v_tt = (_to_dev(mesh_device, x) for x in (q, k, v))
    b_tt, g_tt, s_tt = _to_dev(mesh_device, beta), _to_dev(mesh_device, g), _to_dev(mesh_device, s0)
    o_tt, h_tt = fused_recurrent_gated_delta_rule_ttnn(
        q_tt, k_tt, v_tt, b_tt, g_tt, scale=SCALE, initial_state=s_tt, device=mesh_device
    )
    p_o = pcc(_dev0(mesh_device, o_tt, first=B), o_ref)  # first=B: a replicated B>1 output holds B copies
    p_h = pcc(_dev0(mesh_device, h_tt, first=B), h_ref)
    logger.info(f"B={B} T={T} H={H} G={G} (HV={HV}): PCC o={p_o:.6f} state={p_h:.6f}")
    assert p_o > 0.999, f"GQA gather broke the output: PCC {p_o} (B={B} T={T} H={H} G={G})"
    assert p_h > 0.999, f"GQA gather broke the state: PCC {p_h} (B={B} T={T} H={H} G={G})"


@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_gqa_equals_pre_expanded(mesh_device):
    """The unexpanded call and the pre-expanded call must agree -- the change must be invisible."""
    B, T, H, G = 1, 1, 16, 3
    q, k, v, beta, g = _inputs(B, T, H, G, seed=5)
    s0 = torch.randn(B, H * G, DK, DV, dtype=torch.float32) * 0.1
    qe = q.repeat_interleave(G, dim=2)
    ke = k.repeat_interleave(G, dim=2)

    def run(qq, kk):
        outs = fused_recurrent_gated_delta_rule_ttnn(
            _to_dev(mesh_device, qq),
            _to_dev(mesh_device, kk),
            _to_dev(mesh_device, v),
            _to_dev(mesh_device, beta),
            _to_dev(mesh_device, g),
            scale=SCALE,
            initial_state=_to_dev(mesh_device, s0),
            device=mesh_device,
        )
        return _dev0(mesh_device, outs[0]), _dev0(mesh_device, outs[1])

    o_g, h_g = run(q, k)  # gqa = 3, reader gathers
    o_e, h_e = run(qe, ke)  # gqa = 1, pre-expanded
    p_o, p_h = pcc(o_g, o_e), pcc(h_g, h_e)
    logger.info(f"gqa path vs pre-expanded: PCC o={p_o:.8f} state={p_h:.8f}")
    assert p_o > 0.99999, f"gqa path differs from the pre-expanded call: PCC {p_o}"
    assert p_h > 0.99999, f"gqa state differs from the pre-expanded call: PCC {p_h}"
