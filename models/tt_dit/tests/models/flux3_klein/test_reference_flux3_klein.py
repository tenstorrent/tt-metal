# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""CPU golden test: the vendored FLUX 3 Klein DiT replays a golden case's inputs step by step and its per-step
velocities and latents are compared with the recorded ones (PLAN.md section 4; layout in ``common.py``).

Skips unless ``FLUX3_KLEIN_GOLDENS`` names an existing directory and the case's checkpoint resolves. Each step
is teacher-forced from the golden ``x_t[i]`` so the per-step numbers measure the DiT alone; set
``FLUX3_KLEIN_GOLDEN_FREE_RUN=1`` to also walk the schedule from the noise and compare the final latents, and
``FLUX3_KLEIN_GOLDEN_STEPS=<n>`` to compare only the first ``n`` steps. The real model needs ~20 GB of host RAM
and minutes per step on CPU; ``FLUX3_KLEIN_REFERENCE_DEPTH`` shrinks it for smoke runs against a matching
random checkpoint (never for real goldens)."""

import os

import pytest
import torch
from loguru import logger

from ....pipelines.flux3_klein import packing as p
from ....pipelines.flux3_klein.scheduler import Flux3KleinScheduler
from ....reference.flux3_klein import checkpoint as ref_checkpoint
from ....reference.flux3_klein import dit as ref_dit
from ....reference.flux3_klein import pipeline as ref_pipeline
from ....reference.flux3_klein import sampling as ref_sampling
from .common import (
    GOLDENS_ENV,
    golden_cases,
    golden_index,
    golden_pcc_floor,
    goldens_root,
    load_golden_case,
    patch_reference_depth,
    pcc,
    reference_depth,
    require_checkpoint_dir,
)

FREE_RUN_ENV = "FLUX3_KLEIN_GOLDEN_FREE_RUN"
STEPS_ENV = "FLUX3_KLEIN_GOLDEN_STEPS"


def _pair(inputs, tokens: str, ids: str):
    return (inputs[tokens], inputs[ids]) if inputs[tokens].shape[0] else None


def _triples(golden):
    """The golden inputs as ``pack`` samples ([text, image_cond, image] triples): one sample for the flow map,
    [unconditional, conditional] for CFG. ``image_ids_scaled`` is what ``pack`` received (``image_ids`` holds the
    unscaled int64 latent ids of the VAE)."""
    inputs = golden.inputs
    cond = _pair(inputs, "cond_tokens", "cond_ids")
    image = (inputs["noise"].clone(), inputs["image_ids_scaled"])
    text = (inputs["text_tokens"], inputs["text_ids"])
    if golden.model == "distilled":
        return [[text, cond, image]]
    return [[golden.text_null(), cond, image], [text, cond, image]]


def _build_pack(golden):
    """The reference pack of the golden inputs."""
    return ref_pipeline.pack(_triples(golden))


def _load_dit(weights, model, monkeypatch):
    patch_reference_depth(monkeypatch, ref_dit, reference_depth())
    flow_map = ref_checkpoint.is_flow_map(str(weights))
    assert flow_map == (model == "distilled"), f"{weights} is {'a flow-map' if flow_map else 'a CFG'} checkpoint"
    with torch.device("meta"):
        dit = ref_dit.DiT(flow_map)
    return ref_checkpoint.load_checkpoint(dit, str(weights), "cpu", torch.bfloat16)


def _velocity(dit, packed, schedule, i, guidance):
    """One DiT evaluation exactly as sampling.py:37-41 / :50-54 perform it (state already in packed)."""
    t = schedule[i].item()
    packed.timesteps["image"].fill_(t)
    if dit.flow_map:
        target = schedule[i + 1].reshape(1).to(packed.data["image"].device)
        return dit(packed, timesteps_ref=target)
    unconditional, conditional = dit(packed).chunk(2)
    return unconditional + guidance * (conditional - unconditional)


def _set_state(packed, x, dit):
    packed.data["image"] = x if dit.flow_map else torch.cat((x, x))


def _state(packed, dit):
    return packed.data["image"] if dit.flow_map else packed.data["image"].chunk(2)[0]


@pytest.mark.timeout(14400)
@pytest.mark.skipif(goldens_root() is None, reason=f"{GOLDENS_ENV} not set or not a directory")
@pytest.mark.parametrize("case_dir", golden_cases(), ids=lambda path: f"{path.parent.name}/{path.name}")
def test_reference_matches_golden(case_dir, monkeypatch):
    golden = load_golden_case(case_dir)
    model = golden.model
    weights = require_checkpoint_dir(model, required=("model.safetensors",))
    index = golden_index()
    revision = index.get("weights", {}).get(model, {}).get("revision")
    if revision and revision not in weights.name:
        logger.warning(f"golden weights revision {revision} vs checkpoint {weights}")
    logger.info(f"{golden.name}: model {model}, upstream commit {index.get('reference', {}).get('upstream_commit')}")

    # The schedule and our scheduler agree with the golden grid.
    schedule = golden.inputs["schedule"]
    steps = golden.num_steps
    alpha = golden.alpha
    assert schedule.numel() == steps + 1
    assert torch.equal(schedule, ref_sampling.timesteps(steps, alpha))
    assert torch.equal(Flux3KleinScheduler(steps, alpha).schedule, schedule)
    assert torch.equal(golden.steps["t"], schedule[:-1]) and torch.equal(golden.steps["t_next"], schedule[1:])
    assert torch.equal(golden.x_t(0), golden.inputs["noise"])
    assert torch.equal(golden.inputs["image_ids_scaled"], ref_pipeline.scaled_ids(golden.inputs["image_ids"]))

    packed = _build_pack(golden)
    # Our host packing produces the same pack the reference ran with.
    ours = p.pack([p.Flux3KleinSample.from_reference_triple(triple) for triple in _triples(golden)])
    for m in ref_dit.MODES:
        assert torch.equal(ours.data[m], packed.data[m]) and torch.equal(ours.ids[m], packed.ids[m])
        assert ours.mode_indices[m] == packed.mode_indices[m]

    dit = _load_dit(weights, model, monkeypatch)
    guidance = golden.guidance
    floor = golden_pcc_floor()
    compare = min(steps, int(os.environ.get(STEPS_ENV, steps)))
    x_golden, v_golden = golden.x_t, golden.velocity

    worst = 1.0
    with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
        for i in range(compare):
            _set_state(packed, x_golden(i).clone(), dit)
            velocity = _velocity(dit, packed, schedule, i, guidance)
            dt = (schedule[i + 1] - schedule[i]).item()
            x_next = x_golden(i) + dt * velocity
            pcc_v, pcc_x = pcc(v_golden(i), velocity), pcc(x_golden(i + 1), x_next)
            max_abs = (x_golden(i + 1) - x_next).abs().max().item()
            logger.info(
                f"{golden.name} step {i:2d} t={schedule[i].item():.4f} pcc(v)={pcc_v:.6f} pcc(x)={pcc_x:.6f} max|dx|={max_abs:.4e}"
            )
            worst = min(worst, pcc_v)
            assert pcc_v >= floor, f"step {i}: velocity PCC {pcc_v:.6f} < {floor}"
            assert pcc_x >= floor, f"step {i}: latent PCC {pcc_x:.6f} < {floor}"

        if os.environ.get(FREE_RUN_ENV) == "1":
            _set_state(packed, golden.inputs["noise"].clone(), dit)
            for i in range(steps):
                velocity = _velocity(dit, packed, schedule, i, guidance)
                dt = (schedule[i + 1] - schedule[i]).item()
                _set_state(packed, _state(packed, dit) + dt * velocity, dit)
            final = pcc(x_golden(steps), _state(packed, dit))
            logger.info(f"{golden.name} free run: final latent pcc={final:.6f}")
            assert final >= floor
    logger.info(f"{golden.name}: worst velocity PCC over {compare} steps = {worst:.6f}")
