# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""One-chip (TP=1) routed-expert decode ops of the fused decode (tt/fused_decode/nd.py) against torch.

Random BFP8 expert weights at gpt-oss-20b shapes in the ND-sharded layout the TP=1 fused decode shares with prefill
(gate / up [1, E, 2880, 2880], down [1, E, 2880, 2880], one-tile-column shards), 4 routed experts. The torch reference
uses the device-rounded weights, so only the kernels' arithmetic is compared:
  gate|up  act_j = w_j * SwiGLU(x @ Wg_e + bg_e, x @ Wu_e + bu_e)        (e = the j-th routed expert)
  down     out = sum_j (act_j @ Wd_e + w_j bd_e), from the device act (isolates the down op)
A second pass with x = 0 checks the bias path alone (act and out are then functions of the biases only).

Run on one Blackhole P150 (or one chip of a QuietBox 2):
    pytest models/demos/gpt_oss/tests/unit/test_fused_decode_tp1_ops.py
"""

import os

import pytest
import torch

import ttnn

from ..test_factory import parametrize_mesh_with_fabric

H = I = 2880
LIMIT, ALPHA = 7.0, 1.702
EXPERTS = 8
IDS = (5, 2, 7, 0)


def _pcc(a, b):
    a, b = a.double().reshape(-1), b.double().reshape(-1)
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm() + 1e-30))


def _swiglu(gate, up):
    gate = gate.clamp(max=LIMIT)
    up = up.clamp(-LIMIT, LIMIT)
    return (up + 1) * gate * torch.sigmoid(ALPHA * gate)


@pytest.mark.skipif(os.environ.get("CI") == "true", reason="Needs a Blackhole device")
@parametrize_mesh_with_fabric([(1, 1)])
@pytest.mark.parametrize("zero_x", [False, True], ids=["weights", "bias_only"])
def test_nd_expert_ops(mesh_device, device_params, reset_seeds, zero_x):
    from models.demos.gpt_oss.tt.fused_decode.boundary import (
        flat_memory_config,
        flat_shape,
        flat_tiles,
        flat_to_tile_rows,
        tile_rows_to_flat,
    )
    from models.demos.gpt_oss.tt.fused_decode.nd import (
        NdExpertDown,
        NdExpertGateUp,
        nd_bias_tensor,
        nd_expert_memory_config,
    )

    if not ttnn.device.is_blackhole(mesh_device):
        pytest.skip("the fused decode is a Blackhole layout")
    mesh = mesh_device
    E, k = EXPERTS, len(IDS)
    torch.manual_seed(0)

    def nd_weight(t):
        d = ttnn.from_torch(
            t,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=nd_expert_memory_config(mesh, t.shape[2]),
        )
        return d, ttnn.to_torch(d).float().reshape(t.shape)

    w_gate, wg = nd_weight(torch.randn(1, E, H, I) * 0.02)
    w_up, wu = nd_weight(torch.randn(1, E, H, I) * 0.02)
    w_down, wd = nd_weight(torch.randn(1, E, I, H) * 0.02)
    b_gu = torch.randn(E, 2 * I) * 0.5  # packed [gate | up]
    b_dn = torch.randn(E, H) * 0.5
    bias_gu = nd_bias_tensor(mesh, b_gu)
    bias_dn = nd_bias_tensor(mesh, b_dn)

    x = torch.zeros(H) if zero_x else torch.randn(H)
    pages = flat_tiles(H)
    x_dev = ttnn.from_torch(
        flat_to_tile_rows(x.float(), pages).reshape(1, 1, 32, pages * 32),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=flat_memory_config(mesh, H),
    )
    w_sel = torch.softmax(torch.randn(k), 0).bfloat16().float()
    ids_t = torch.zeros(1, 32, dtype=torch.int32)
    ids_t[0, :k] = torch.tensor(IDS)
    sc_t = torch.zeros(1, 32)
    sc_t[0, :k] = w_sel
    l1 = ttnn.L1_MEMORY_CONFIG
    ids = ttnn.from_torch(ids_t, dtype=ttnn.uint16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, memory_config=l1)
    scores = ttnn.from_torch(sc_t, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, memory_config=l1)
    act = ttnn.from_torch(
        torch.zeros(k, I), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, memory_config=l1
    )
    out = ttnn.from_torch(
        torch.zeros(flat_shape(H)),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=flat_memory_config(mesh, H),
    )

    gate_up = NdExpertGateUp(mesh, H, I, k, LIMIT, ALPHA)
    down = NdExpertDown(mesh, H, I, k)
    gate_up(x_dev, ids, scores, w_gate, w_up, bias_gu, act)
    down(act, ids, scores, w_down, bias_dn, out)
    ttnn.synchronize_device(mesh)
    act_dev = ttnn.to_torch(act).float().reshape(k, I)
    out_dev = tile_rows_to_flat(ttnn.to_torch(out).reshape(32, -1), H).float()

    xb = x.bfloat16().float()
    ref_act = torch.stack(
        [w_sel[j] * _swiglu(xb @ wg[0, e] + b_gu[e, :I], xb @ wu[0, e] + b_gu[e, I:]) for j, e in enumerate(IDS)]
    )
    ref_out = sum(act_dev[j] @ wd[0, e] + w_sel[j] * b_dn[e] for j, e in enumerate(IDS))
    assert torch.isfinite(act_dev).all() and torch.isfinite(out_dev).all()
    pcc_act, pcc_out = _pcc(act_dev, ref_act), _pcc(out_dev, ref_out)
    print(f"ND gate|up PCC {pcc_act:.5f}, down PCC {pcc_out:.5f} ({len(gate_up.work)} / {len(down.work)} cores)")
    assert pcc_act > 0.999, f"gate|up PCC {pcc_act}"
    assert pcc_out > 0.999, f"down PCC {pcc_out}"
    # Every output column is produced by its own core: check them one by one (a wrong shard address shows up here).
    for c in range(I // 32):
        p = _pcc(act_dev[:, c * 32 : (c + 1) * 32], ref_act[:, c * 32 : (c + 1) * 32])
        assert p > 0.99, f"gate|up column {c}: PCC {p}"
    for c in range(H // 32):
        p = _pcc(out_dev[c * 32 : (c + 1) * 32], ref_out[c * 32 : (c + 1) * 32])
        assert p > 0.99, f"down column {c}: PCC {p}"
