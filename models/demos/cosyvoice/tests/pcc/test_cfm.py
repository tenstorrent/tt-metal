# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""The flow-matching solver and the assembled flow stage.

The solver is checked in two independent ways before any device time:

* the **schedule** (t and dt per step) against the captured timesteps, and
* the **update rule** replayed against the captured per-step `dphi_dt`, with the
  network taken entirely out of the loop.

If both hold, then a device miss on `solve_euler` is the estimator drifting, not
the integration -- and since each step's input is the previous step's output,
that distinction is otherwise very hard to make.
"""
from __future__ import annotations

import os

import pytest
import torch

from models.demos.cosyvoice.tt.common import GOLDEN_DIR, as_torch, load_golden, pcc
from models.demos.cosyvoice.tt.flow.cfm import cosine_t_span, euler_steps
from models.demos.cosyvoice.tt.weights import default_weights_path

FLOW_WEIGHTS = default_weights_path().replace("hift_", "flow_")
CFG_RATE = 0.7

needs_l1_small = pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
needs_weights = pytest.mark.skipif(not os.path.exists(FLOW_WEIGHTS), reason="export flow weights first")
needs_golden = pytest.mark.skipif(
    not os.path.exists(os.path.join(GOLDEN_DIR, "flow.solve_euler.npz")), reason="generate goldens first"
)


# --------------------------------------------------------------------------
# host tier
# --------------------------------------------------------------------------
@needs_golden
def test_schedule_matches_the_captured_timesteps():
    """Cosine grid, and `dt` recomputed from it each step rather than held at 1/n."""
    span = cosine_t_span(10)
    want_span = as_torch(load_golden("flow.solve_euler")["call0.in_t_span"])
    assert torch.equal(span, want_span), (span, want_span)

    g = load_golden("flow.cfm_estimator")
    sched = euler_steps(span)
    for i, (t, _) in enumerate(sched):
        assert abs(t - float(as_torch(g[f"call{i}.in_t"])[0])) < 1e-7, i
    total = sum(dt for _, dt in sched)
    assert abs(total - 1.0) < 1e-6, f"the steps must span [0, 1] exactly, got {total}"
    dts = [dt for _, dt in sched]
    assert dts == sorted(dts), "a cosine grid widens monotonically; this one does not"


@needs_golden
def test_update_rule_replayed_against_captured_dphi_dt():
    """The solver with the network removed.

    Feeding the recorded `dphi_dt` back through the CFG combination and the Euler
    step must reproduce each following step's input exactly, and the last one must
    be `solve_euler`'s output. This isolates the integration from the model.
    """
    g = load_golden("flow.cfm_estimator")
    sched = euler_steps(cosine_t_span(10))
    x = as_torch(g["call0.in_x"])[:1]  # [1, 80, T]; both CFG rows are the same

    for i, (_, dt) in enumerate(sched):
        d = as_torch(g[f"call{i}.out_dphi_dt"])
        x = x + dt * ((1.0 + CFG_RATE) * d[:1] - CFG_RATE * d[1:])
        if i + 1 < len(sched):
            want = as_torch(g[f"call{i + 1}.in_x"])[:1]
            err = (x - want).abs().max()
            assert err < 2e-3, f"step {i}: diverged by {err}"

    want = as_torch(load_golden("flow.solve_euler")["call0.out_sample"])
    p = pcc(x, want)
    print(f"\n  replayed solver: PCC {p:.10f}  max|d| {(x - want).abs().max():.3e}")
    assert p >= 0.99999, p


@needs_golden
def test_injected_noise_is_the_solver_input():
    """`ConditionalCFM.forward` draws `z = randn_like(mu) * temperature` and hands it
    straight to `solve_euler`. The capture rewinds the RNG so the recorded draw is
    the one the reference actually used -- seeding a device RNG could never
    reproduce it, so `z` is injected."""
    z = as_torch(load_golden("flow.cfm")["call0.rng_z"])
    x0 = as_torch(load_golden("flow.solve_euler")["call0.in_x"])
    assert torch.equal(z, x0), (z - x0).abs().max()


@needs_golden
def test_prompt_is_dropped_at_the_end_not_the_start():
    """`feat[:, :, mel_len1:]` -- the solver runs over prompt *and* generated frames
    together and the prompt is discarded afterwards. Running it over the generated
    part alone would lose the conditioning the prompt provides through `cond`."""
    lr = load_golden("flow.length_regulator")
    mel_len1, mel_len2 = int(lr["call0.in_mel_len1"]), int(lr["call0.in_mel_len2"])
    sample = as_torch(load_golden("flow.solve_euler")["call0.out_sample"])
    assert sample.shape[2] == mel_len1 + mel_len2
    assert mel_len1 > 0 and mel_len2 > 0


@needs_golden
def test_mel_length_truncates_rather_than_rounds():
    from models.demos.cosyvoice.tt.flow.model import TtMaskedDiffWithXvec

    lr = load_golden("flow.length_regulator")
    token_len2 = as_torch(lr["call0.in_x2"]).shape[1]
    assert TtMaskedDiffWithXvec.mel_len_for(token_len2) == int(lr["call0.in_mel_len2"])


# --------------------------------------------------------------------------
# device tier
# --------------------------------------------------------------------------
@needs_weights
@needs_golden
@needs_l1_small
def test_device_solve_euler_matches_golden(device):
    """Ten Euler steps of the real UNet on device: 200 resnet blocks and 640
    transformer blocks of work, with the error from each step feeding the next."""
    import ttnn
    from models.demos.cosyvoice.tt.flow.cfm import TtConditionalCFM
    from models.demos.cosyvoice.tt.weights import WeightBag

    g = load_golden("flow.solve_euler")
    x0 = as_torch(g["call0.in_x"])  # [1, 80, T]
    mu = as_torch(g["call0.in_mu"])
    spks = as_torch(g["call0.in_spks"])  # [1, 80]
    cond = as_torch(g["call0.in_cond"])
    want = as_torch(g["call0.out_sample"])

    bag = WeightBag.load(FLOW_WEIGHTS)
    cfm = TtConditionalCFM(device, bag.sub("decoder"), inference_cfg_rate=CFG_RATE, n_timesteps=10)

    def cl(v):
        return ttnn.from_torch(
            v.permute(0, 2, 1).contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )

    out = cfm.solve_euler(
        cl(x0),
        cl(mu),
        ttnn.from_torch(spks.reshape(1, 1, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device),
        cl(cond),
    )
    got = ttnn.to_torch(out).float().permute(0, 2, 1)

    p = pcc(got, want)
    print(f"\n  solve_euler, 10 steps, T={x0.shape[2]}")
    print(f"  PCC {p:.10f}  max|d| {(got - want).abs().max():.3e}")
    assert got.shape == want.shape, (got.shape, want.shape)
    assert p >= 0.99, p


@needs_weights
@needs_golden
@needs_l1_small
def test_device_flow_tokens_to_mel(device):
    """The whole flow stage: semantic tokens in, mel out, nothing on the host.

    This is the flow stage's acceptance gate. It chains the token embedding, the
    6-block Conformer encoder, the projection, the length regulator and all ten
    solver steps -- every piece of the stage, in one graph.
    """
    import ttnn
    from models.demos.cosyvoice.tt.flow.model import TtMaskedDiffWithXvec
    from models.demos.cosyvoice.tt.weights import WeightBag

    emb_g = load_golden("flow.input_embedding")
    lr_g = load_golden("flow.length_regulator")
    cfm_g = load_golden("flow.cfm")
    spk_g = load_golden("flow.spk_embed_affine")

    tokens = torch.from_numpy(emb_g["call0.in_tokens"]).to(torch.int32)
    token_len1 = as_torch(lr_g["call0.in_x1"]).shape[1]
    mel_len1, mel_len2 = int(lr_g["call0.in_mel_len1"]), int(lr_g["call0.in_mel_len2"])
    prompt_feat = as_torch(cfm_g["call0.in_cond"])[:, :, :mel_len1].permute(0, 2, 1).contiguous()
    embedding = as_torch(spk_g["call0.in_x"]).reshape(1, 1, -1)
    z = as_torch(cfm_g["call0.rng_z"]).permute(0, 2, 1).contiguous()

    bag = WeightBag.load(FLOW_WEIGHTS)
    model = TtMaskedDiffWithXvec(device, bag, bag.meta)

    def dev(v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
        return ttnn.from_torch(v, dtype=dtype, layout=layout, device=device)

    out = model.inference(
        dev(tokens, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT),
        token_len1,
        mel_len1,
        mel_len2,
        dev(prompt_feat),
        dev(embedding),
        dev(z),
    )
    got = ttnn.to_torch(out).float().permute(0, 2, 1)
    want = as_torch(load_golden("flow.solve_euler")["call0.out_sample"])[:, :, mel_len1:]

    p = pcc(got, want)
    print(f"\n  flow: {tokens.shape[1]} tokens -> {mel_len2} mel frames (prompt {mel_len1} dropped)")
    print(f"  PCC {p:.10f}  max|d| {(got - want).abs().max():.3e}")
    assert got.shape == want.shape, (got.shape, want.shape)
    assert p >= 0.99, p
