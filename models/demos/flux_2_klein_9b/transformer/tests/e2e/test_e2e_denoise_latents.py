# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""CALL 2 -- `run_denoise_latents`: the real task, and Gate 2 / Gate 3 for it.

What Call 2 is
--------------
A diffusion transformer is only useful inside its sampler, so this is the head
that produces something a VAE decoder could consume: the flow-match Euler loop
around Call 1's forward, N steps, with the latents kept RESIDENT on device
between steps. Step i+1 consumes step i's real TT output -- no reference tensor
is injected at any joint, which is what makes the final PCC an end-to-end number
rather than N independent one-step numbers.

The horizon
-----------
A denoise loop has no stop token: `config.json` has no `eos_token_id` and no
model-specific stop id, and the checkpoint ships no `generation_config.json` and
no scheduler config (it is the `transformer` subfolder only), so neither the
stop-token rule nor the config-length rule has anything to read. N is therefore
chosen -- 4 by default, because this is the DISTILLED Klein variant
(`config._name_or_path = klein-9b-distilled-diffusers`), which is built for
few-step sampling -- and clamped to [1, 50] so the loop cannot run away. What is
NOT chosen is the schedule: the sigma list comes from Source A's own
`Flux2Pipeline` recipe (`np.linspace(1, 1/N, N)`, exponential time-shift with the
empirical mu computed from `image_seq_len`, trailing 0.0), and the SAME list is
consumed by the TT loop and by the HF golden loop, for the same N. That is
asserted below rather than assumed.

Gate 3 here is the final-latents PCC. Per-step PCC is printed for every step, so
a divergence that only shows up late in the loop is visible and is held to the
same threshold as the final number.
"""

from __future__ import annotations

import pytest
import torch

from models.common.utility_functions import comp_pcc
from models.demos.flux_2_klein_9b.transformer.tt import inputs as tt_inputs
from models.demos.flux_2_klein_9b.transformer.tt import pipeline as tt_pipeline
from models.demos.flux_2_klein_9b.transformer.tt import reference as tt_reference
from models.demos.flux_2_klein_9b.transformer.tt import stubs as tt_stubs


def test_the_two_sides_walk_one_schedule():
    """Host-only: the sigma list is Source A's, and there is exactly one of it."""
    steps = 4
    image_seq_len = 256
    sigmas = tt_inputs.sigma_schedule(steps, image_seq_len)

    assert len(sigmas) == steps + 1, sigmas
    assert sigmas[-1] == 0.0, "the scheduler appends a single trailing 0.0"
    assert all(sigmas[i] > sigmas[i + 1] for i in range(steps)), f"sigmas must decrease: {sigmas}"
    assert 0.0 < sigmas[0] <= 1.0, sigmas

    # The safety cap is the only bound on a loop with no stop token.
    assert len(tt_inputs.sigma_schedule(0, image_seq_len)) == tt_inputs.MIN_STEPS + 1
    assert len(tt_inputs.sigma_schedule(10_000, image_seq_len)) == tt_inputs.MAX_STEPS + 1
    print(f"[flux2] schedule N={steps}: {[round(s, 6) for s in sigmas]}", flush=True)


@pytest.mark.timeout(5400)
def test_gate_2_and_3_denoise_latents(flux2_pipeline, flux2_inputs, flux2_reference, flux2_device, flux2_shapes):
    """CALL 2: the whole Euler loop on device, then PCC against the golden loop."""
    pipe = flux2_pipeline
    depth = pipe.depth()
    steps = flux2_shapes["steps"]
    print(
        f"[flux2] Call 2: {steps} Euler steps at dual {depth['dual_layers']}/{pipe.full_dual_layers}, "
        f"single {depth['single_layers']}/{pipe.full_single_layers}",
        flush=True,
    )

    pipe.reset_invocations()
    result = tt_pipeline.run_denoise_latents(pipe, flux2_inputs, num_steps=steps)

    golden = tt_reference._hf_reference_denoise_latents(
        flux2_reference,
        flux2_inputs,
        num_inference_steps=steps,
        dual_layers=depth["dual_layers"],
        single_layers=depth["single_layers"],
    )

    # ---- one schedule, walked by both sides ----------------------------------
    assert result["sigmas"] == golden["sigmas"], (result["sigmas"], golden["sigmas"])
    assert len(result["per_step"]) == steps == len(golden["per_step"])

    # ---- Gate 2: every stub reached, `steps` times over ----------------------
    expected = {name: count * steps for name, count in pipe.expected_calls_per_step().items()}
    missing = sorted(name for name, count in pipe.invocations.items() if count == 0)
    assert not missing, f"graduated stubs never reached by the loop: {missing}"
    wrong = {
        name: (count, pipe.invocations[name]) for name, count in expected.items() if pipe.invocations[name] != count
    }
    assert not wrong, f"loop call counts differ from {steps} x routing.table (expected, actual): {wrong}"
    assert set(pipe.invocations) == set(tt_stubs.GRADUATED)
    print(f"[flux2] Gate 2: all 18 graduated stubs invoked {steps}x per routing.table", flush=True)

    # ---- Gate 3: per-step, then the final latents ---------------------------
    for index, (tt_step, hf_step) in enumerate(zip(result["per_step"], golden["per_step"])):
        tt_host = tt_pipeline.to_torch(tt_step, flux2_device).to(torch.float32)
        step_ok, step_pcc = comp_pcc(hf_step, tt_host, flux2_shapes["pcc"])
        print(f"step {index} PCC={step_pcc}", flush=True)
        assert step_ok, f"step {index} PCC {step_pcc} below the required {flux2_shapes['pcc']}"

    tt_latents = tt_pipeline.to_torch(result["latents"], flux2_device).to(torch.float32)
    assert tuple(tt_latents.shape) == tuple(golden["latents"].shape), (
        tuple(tt_latents.shape),
        tuple(golden["latents"].shape),
    )

    ok, achieved_pcc = comp_pcc(golden["latents"], tt_latents, flux2_shapes["pcc"])
    print(f"e2e PCC={achieved_pcc}", flush=True)
    assert ok, f"Call 2 e2e PCC {achieved_pcc} below the required {flux2_shapes['pcc']}"

    # The tensor the VAE decoder would consume, unpacked back onto its grid by
    # the position ids -- proof the loop's output really is a latent image and
    # not just a well-correlated vector.
    grid = tt_inputs.unpack_latents(tt_latents, flux2_inputs["img_ids"])
    meta = flux2_inputs["meta"]
    assert tuple(grid.shape) == (1, tt_inputs.LATENT_CHANNELS, meta["grid"][0], meta["grid"][1]), tuple(grid.shape)
    print(f"[flux2] Call 2 final latents {tuple(tt_latents.shape)} -> grid {tuple(grid.shape)}", flush=True)
