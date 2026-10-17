# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""MiMo MoE decode against prefill on the same tokens, one Galaxy row (TP=8), BFP4 experts.

Each step takes 4 real users (layer-1 MoE inputs of the golden prompts) in rows 0-3 of a 32-row tile and runs
TtMiMoMoE.forward_decode (the streaming path: GateUpStream, DownStream, the compact row all-gather and ReduceExpand)
and TtMiMoMoE.forward (prefill) on the same tile with the token mask. Every reduction variant of the decode path
is checked:

- standalone (DownStream launched alone, all_gather, ReduceExpand): rows 0-3 match prefill, rows 4-31 are zero;
- with a residual: rows 0-3 are the residual plus the standalone output, rows 4-31 the residual;
- with the fused RMSNorm through the all_gather, and over the fabric with Down inside the reduce's launch (the
  full model's path): the same sum as the residual variant within a few bf16 ulps, its RMSNorm, and the two
  bit-identical.

Several steps run back to back and each decodes before its prefill, so an output that depends on what ran before
fails (an unwritten output buffer can otherwise hold the last freed tensor of the same size).
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.mimo_v2_6_flash.reference import model as ref
from models.experimental.mimo_v2_6_flash.tests.common import (
    GALAXY_DEVICE_PARAMS,
    cache_dir,
    dp_submeshes,
    golden_path,
    load_config_and_ckpt,
    pcc,
)
from models.experimental.mimo_v2_6_flash.tests.perf.moe_decode_common import golden_moe_inputs
from models.experimental.mimo_v2_6_flash.tt.common import TILE, MeshPlan
from models.experimental.mimo_v2_6_flash.tt.moe import TtMiMoMoE
from models.experimental.mimo_v2_6_flash.tt.moe_stream import zero_output

LAYER = 1
USERS = 4
STEPS = 4
RESIDUAL_STD = 0.5


def to_device(plan, t):
    return ttnn.from_torch(
        t.reshape(1, 1, *t.shape[-2:]).bfloat16(),
        device=plan.mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=plan.replicate(),
    )


def read(plan, t, free=True):
    out = plan.to_torch_replicated(t).float()[0, 0]
    if free:
        t.deallocate(True)
    return out


def read_normed(plan, result):
    """(sum, norm) of a fused-norm decode; the norm lands in the caller's persistent buffer, which stays."""
    y, h = result
    return read(plan, y), read(plan, h, free=False)


def ulps(a, b):
    """Largest |a - b| in bf16 ulps of the larger magnitude (both bf16-exact)."""
    mag = torch.maximum(a.abs(), b.abs()).clamp(min=2.0**-100)
    return ((a - b).abs() / torch.exp2(torch.floor(torch.log2(mag)) - 7)).max().item()


@pytest.mark.parametrize("device_params", [GALAXY_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
def test_moe_decode_vs_prefill(mesh_device):
    golden_path()  # skips without the golden file
    torch.manual_seed(0)
    cfg, ckpt = load_config_and_ckpt()
    assert cfg.is_moe(LAYER)
    inputs = golden_moe_inputs(cfg, ckpt)
    assert inputs.shape[0] >= STEPS * USERS

    plan = MeshPlan(dp_submeshes(mesh_device)[0])
    moe = TtMiMoMoE(
        plan,
        cfg,
        LAYER,
        ckpt.router(LAYER),
        ref.LazyExperts(ckpt, LAYER, torch.bfloat16),
        cache_dir=cache_dir("model_bfp4experts"),
        expert_dtype=ttnn.bfloat4_b,
    )
    moe.set_decode_users(USERS)
    assert moe.gate_up_stream is not None, "BFP4 experts with 4 users take the streaming decode path"
    red = moe.reduce_expand
    assert red.can_norm
    gamma = ckpt.norms(LAYER + 1)["input_layernorm"]  # the next layer's input norm, as in the model
    eps = cfg.layernorm_epsilon
    norm_out = zero_output(plan, cfg.hidden_size)
    norm = (red.compact_gamma(plan, gamma), norm_out, eps)
    mask = to_device(plan, (torch.arange(TILE) < USERS).float().reshape(TILE, 1))

    worst_pcc, worst_ulps = 1.0, 0.0
    for step in range(STEPS):
        x = torch.zeros(TILE, cfg.hidden_size)
        x[:USERS] = inputs[step * USERS : (step + 1) * USERS]
        x_tt = to_device(plan, x)
        r = (torch.randn(TILE, cfg.hidden_size) * RESIDUAL_STD).bfloat16().float()
        r_tt = to_device(plan, r)

        # Decode first: an output buffer the decode leaves unwritten then cannot hold the prefill result.
        moe.fabric_gather = False
        out = read(plan, moe.forward_decode(x_tt, mask))
        expected = read(plan, moe.forward(x_tt, mask))
        y = read(plan, moe.forward_decode(x_tt, mask, residual=r_tt))
        y_ag, h_ag = read_normed(plan, moe.forward_decode(x_tt, mask, residual=r_tt, norm=norm))
        moe.fabric_gather = True
        y_fab, h_fab = read_normed(plan, moe.forward_decode(x_tt, mask, residual=r_tt, norm=norm))
        moe.fabric_gather = False

        for name, t in (("out", out), ("y", y), ("y_ag", y_ag), ("h_ag", h_ag), ("y_fab", y_fab), ("h_fab", h_fab)):
            assert torch.isfinite(t).all(), f"step {step}: {name} is not finite"
        user_pcc = [pcc(out[u], expected[u]) for u in range(USERS)]
        # bf16(residual + the bf16-rounded sum); the fused reductions sum the chips with FPU adds instead of SFPU
        res_ulps = ulps(y[:USERS], (r[:USERS] + out[:USERS]).bfloat16().float())
        fused_ulps = ulps(y_ag[:USERS], y[:USERS])
        norm_pcc = pcc(h_ag[:USERS], ref.rms_norm(y_ag[:USERS], gamma, eps))
        logger.info(
            f"step {step}: decode vs prefill per user {[round(p, 5) for p in user_pcc]}; residual {res_ulps:g} ulp, "
            f"fused sum vs residual {fused_ulps:g} ulp, fused norm pcc {norm_pcc:.6f}"
        )
        worst_pcc, worst_ulps = min(worst_pcc, *user_pcc), max(worst_ulps, res_ulps, fused_ulps)
        assert torch.count_nonzero(out[USERS:]) == 0, f"step {step}: padding rows of the decode output are not zero"
        for name, t in (("y", y), ("y_ag", y_ag), ("y_fab", y_fab)):
            assert torch.equal(t[USERS:], r[USERS:]), f"step {step}: {name} rows 4-31 are not the residual's"
        assert res_ulps <= 2 and fused_ulps <= 4, f"step {step}"
        assert norm_pcc >= 0.9999, f"step {step}"
        # Down inside the fabric launch must give exactly what Down alone + all_gather gives.
        assert torch.equal(y_fab, y_ag) and torch.equal(h_fab, h_ag), f"step {step}: fabric and all_gather differ"
        for t in (x_tt, r_tt):
            t.deallocate(True)

    logger.info(f"worst decode vs prefill user pcc {worst_pcc:.6f}, worst residual / fused ulps {worst_ulps:g}")
    assert worst_pcc >= 0.999
