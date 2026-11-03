# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""MiMo MoE decode against prefill on the same tokens, one Galaxy row (TP=8), BFP4 experts.

Each step takes 4 (or 8: the deduplicated stream's 8-row tiles) real users (layer-1 MoE inputs of the golden prompts)
in the first rows of a 32-row tile and runs
TtMiMoMoE.forward_decode (the streaming path: GateUpStream, DownStream, the compact row all-gather and ReduceExpand)
and TtMiMoMoE.forward (prefill) on the same tile with the token mask. Every reduction variant of the decode path
is checked:

- standalone (DownStream launched alone, all_gather, ReduceExpand): rows 0-3 match prefill, rows 4-31 are zero;
- with a residual: rows 0-3 are the residual plus the standalone output, rows 4-31 the residual;
- with the fused RMSNorm through the all_gather, and over the fabric with Down inside the reduce's launch (the
  full model's path): the same sum as the residual variant within a few bf16 ulps, its RMSNorm, and the two
  bit-identical.

Several steps run back to back and each decodes before its prefill, so an output that depends on what ran before
fails (an unwritten output buffer can otherwise hold the last freed tensor of the same size). Besides the golden
steps (4 different tokens), two steps exercise the deduplicated stream (tt/moe_stream.py): users that select the
same experts (rows 0 / 2 and 1 / 3 equal), and a masked user (its row is zero, and it streams nothing). Every decode output
must be bit-identical on the 8 chips of the row (the fixed-order chip sum). Last, the fabric variant runs every step
again with no host read in between, so the chips drift up to a gather phase apart and both gather buffers are in
use, and must reproduce the per-step results bit for bit.

A user's decode output must not depend on the other users of the step (the DP replicas of the full model, and a
verify against a plain decode of the same token, rely on it): the deduplicated stream sums each user's experts by
ascending expert id, not in the order the step's users listed them. Every golden step runs again with its users
rotated through the rows, mixed with another step's users, and its last user alone, each row bit for bit its own.
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


def read(plan, t, free=True, same_on_chips=True):
    """Chip 0's copy [rows, cols]; ``same_on_chips``: every chip of the row must hold exactly the same tensor."""
    chips = [ttnn.to_torch(d).float()[0, 0] for d in ttnn.get_device_tensors(t)]
    assert len(chips) == plan.tp
    if same_on_chips:
        for c in range(1, len(chips)):
            assert torch.equal(chips[c], chips[0]), f"chip {c} differs from chip 0"
    if free:
        t.deallocate(True)
    return chips[0]


def read_normed(plan, result):
    """(sum, norm) of a fused-norm decode; the norm lands in the caller's persistent buffer, which stays."""
    y, h = result
    return read(plan, y), read(plan, h, free=False)


def ulps(a, b, *operands):
    """Largest |a - b| in bf16 ulps of the larger magnitude (both bf16-exact), or of the largest of ``operands``
    (the terms a and b are sums of: where they cancel, a few ulps of the terms are many ulps of the result)."""
    mag = torch.maximum(a.abs(), b.abs())
    for t in operands:
        mag = torch.maximum(mag, t.abs())
    mag = mag.clamp(min=2.0**-100)
    return ((a - b).abs() / torch.exp2(torch.floor(torch.log2(mag)) - 7)).max().item()


@pytest.mark.parametrize("users", [4, 8])
@pytest.mark.parametrize("device_params", [GALAXY_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
def test_moe_decode_vs_prefill(mesh_device, users):
    USERS = users
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
    assert moe.gate_up_stream is not None, f"BFP4 experts with {USERS} users take the streaming decode path"
    red = moe.reduce_expand
    assert red.can_norm
    gamma = ckpt.norms(LAYER + 1)["input_layernorm"]  # the next layer's input norm, as in the model
    eps = cfg.layernorm_epsilon
    norm_out = zero_output(plan, cfg.hidden_size)
    norm = (red.compact_gamma(plan, gamma), norm_out, eps)
    # Per step: the users' rows and the token mask's users.
    cases = [(inputs[step * USERS : (step + 1) * USERS], [1] * USERS) for step in range(STEPS)]
    pairs = torch.arange(USERS) % 2  # users 0, 2, ... and 1, 3, ... select the same experts
    cases.append((inputs[pairs], [1] * USERS))
    cases.append((inputs[USERS : 2 * USERS], [0 if u == 2 else 1 for u in range(USERS)]))  # user 2 masked

    worst_pcc, worst_ulps = 1.0, 0.0
    steps = []  # (x, r, mask, y_fab, h_fab) per step, for the back-to-back run
    for step, (rows, live) in enumerate(cases):
        x = torch.zeros(TILE, cfg.hidden_size)
        x[:USERS] = rows
        x_tt = to_device(plan, x)
        m = torch.zeros(TILE, 1)
        m[:USERS, 0] = torch.tensor(live, dtype=torch.float32)
        mask = to_device(plan, m)
        active = [u for u in range(USERS) if live[u]]
        r = (torch.randn(TILE, cfg.hidden_size) * RESIDUAL_STD).bfloat16().float()
        r_tt = to_device(plan, r)

        # Decode first: an output buffer the decode leaves unwritten then cannot hold the prefill result.
        moe.fabric_gather = False
        out = read(plan, moe.forward_decode(x_tt, mask))
        # prefill's ring all-reduce may round differently per chip
        expected = read(plan, moe.forward(x_tt, mask), same_on_chips=False)
        y = read(plan, moe.forward_decode(x_tt, mask, residual=r_tt))
        y_ag, h_ag = read_normed(plan, moe.forward_decode(x_tt, mask, residual=r_tt, norm=norm))
        moe.fabric_gather = True
        y_fab, h_fab = read_normed(plan, moe.forward_decode(x_tt, mask, residual=r_tt, norm=norm))
        moe.fabric_gather = False

        for name, t in (("out", out), ("y", y), ("y_ag", y_ag), ("h_ag", h_ag), ("y_fab", y_fab), ("h_fab", h_fab)):
            assert torch.isfinite(t).all(), f"step {step}: {name} is not finite"
        user_pcc = [pcc(out[u], expected[u]) for u in active]
        for u in set(range(USERS)) - set(active):
            assert torch.count_nonzero(out[u]) == 0 and torch.count_nonzero(expected[u]) == 0, f"step {step}: user {u}"
        # bf16(residual + the bf16-rounded sum); the fused reductions sum the chips with FPU adds instead of SFPU
        res_ulps = ulps(y[:USERS], (r[:USERS] + out[:USERS]).bfloat16().float())
        fused_ulps = ulps(y_ag[:USERS], y[:USERS], out[:USERS], r[:USERS])
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
        for t in (x_tt, r_tt, mask):
            t.deallocate(True)
        steps.append((x, r, m, y_fab, h_fab))

    logger.info(f"worst decode vs prefill user pcc {worst_pcc:.6f}, worst residual / fused ulps {worst_ulps:g}")
    assert worst_pcc >= 0.999

    # Every step's fabric decode enqueued before any read, each with its own norm buffer.
    moe.fabric_gather = True
    pending = []
    for x, r, m, _, _ in steps:
        x_tt, r_tt, h_buf = to_device(plan, x), to_device(plan, r), zero_output(plan, cfg.hidden_size)
        m_tt = to_device(plan, m)
        y_tt, h_tt = moe.forward_decode(x_tt, m_tt, residual=r_tt, norm=(norm[0], h_buf, eps))
        pending.append((y_tt, h_tt, x_tt, r_tt, m_tt))
    moe.fabric_gather = False
    for step, ((y_tt, h_tt, x_tt, r_tt, m_tt), (_, _, _, y_fab, h_fab)) in enumerate(zip(pending, steps)):
        y_b, h_b = read(plan, y_tt), read(plan, h_tt)
        assert torch.equal(y_b, y_fab) and torch.equal(h_b, h_fab), f"step {step}: back-to-back fabric run differs"
        for t in (x_tt, r_tt, m_tt):
            t.deallocate(True)
    logger.info(f"{len(steps)} fabric decodes back to back: bit-identical to the per-step runs on every chip")

    # Batch invariance: each golden step's users rotated through the rows, mixed with the next step's, and alone.
    def decode(rows, live):
        """The standalone decode output rows 0..USERS-1 of a step with ``rows`` and the users ``live`` unmasked."""
        x = torch.zeros(TILE, cfg.hidden_size)
        x[:USERS] = rows
        m = torch.zeros(TILE, 1)
        m[live, 0] = 1.0
        x_tt, m_tt = to_device(plan, x), to_device(plan, m)
        result = read(plan, moe.forward_decode(x_tt, m_tt))[:USERS]
        x_tt.deallocate(True)
        m_tt.deallocate(True)
        return result

    every, last = list(range(USERS)), USERS - 1
    batches = [inputs[step * USERS : (step + 1) * USERS] for step in range(STEPS)]
    outs = [decode(rows, every) for rows in batches]
    even = (torch.arange(USERS) % 2 == 0)[:, None]
    for step, (rows, out) in enumerate(zip(batches, outs)):
        for shift in range(1, USERS):
            rolled = decode(rows.roll(shift, 0), every)
            for u in every:
                assert torch.equal(rolled[(u + shift) % USERS], out[u]), f"step {step}: user {u} in row {u + shift}"
        nxt = (step + 1) % STEPS
        mixed = decode(torch.where(even, rows, batches[nxt]), every)
        for u in every:
            assert torch.equal(mixed[u], (out if u % 2 == 0 else outs[nxt])[u]), f"step {step}: user {u} mixed"
        alone = decode(rows, [last])
        assert torch.equal(alone[last], out[last]) and torch.count_nonzero(alone[:last]) == 0, f"step {step} alone"
    logger.info(f"{STEPS * (USERS + 1)} batch variants: every user's row bit-identical to its own step's")
