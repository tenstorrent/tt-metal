# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""TT action projections and flow loop against the torch reference on one Blackhole P100.

The expert attends over the reference's teacher-forced prefix cache, as in `test_expert`, so the velocity and the
actions differ from the reference only by the device action path.
"""

import pytest
import torch
from loguru import logger
from torch import nn

from models.experimental.alpamayo2_edge.reference import golden_cache
from models.experimental.alpamayo2_edge.reference.action import ActionInProj, action_to_traj, euler_sample
from models.experimental.alpamayo2_edge.reference.model import CotOut, ReferenceModel
from models.experimental.alpamayo2_edge.reference.text import TextModel
from models.experimental.alpamayo2_edge.reference.weights import load_weights
from models.experimental.alpamayo2_edge.tests._util import PCC, pcc, reference_cache
from models.experimental.alpamayo2_edge.tests.conftest import P100_MESH
from models.experimental.alpamayo2_edge.tt.action import (
    TtActionInProj,
    TtActionOutProj,
    action_features,
    action_velocity_tt,
    sample_actions_tt,
)
from models.experimental.alpamayo2_edge.tt.expert import TtExpert
from models.tt_dit.utils import tensor
from models.tt_dit.utils.check import assert_quality

SEED = 0


@pytest.fixture(scope="module")
def in_proj_weights(model_dir):
    return load_weights(model_dir, "expert.action_in_proj.")


@pytest.fixture(scope="module")
def out_proj_weights(model_dir):
    return load_weights(model_dir, "expert.action_out_proj.")


def _timestep(cfg, step: int, batch: int) -> torch.Tensor:
    """Flow time of Euler step `step`, [B, 1, 1], as `euler_sample` passes it."""
    return torch.linspace(0.0, 1.0, cfg.action.steps + 1)[step].view(1, 1, 1).expand(batch, 1, 1)


def _noise(cfg, batch: int) -> torch.Tensor:
    shape = (batch, cfg.action.n_waypoints, cfg.action.action_dim)
    return torch.randn(*shape, generator=torch.Generator().manual_seed(SEED))


def _reference_model(cfg, expert_weights, in_proj_weights, out_proj_weights) -> ReferenceModel:
    """The reference's expert path in fp32; the vision tower and LM are not needed over a given prefix cache."""
    expert = TextModel(cfg.expert, expert_weights, has_embedding=False)
    in_proj = ActionInProj(cfg.action, cfg.expert.hidden, in_proj_weights)
    out_proj = nn.Linear(cfg.expert.hidden, cfg.action.action_dim)
    out_proj.load_state_dict(out_proj_weights)
    for module in (expert, in_proj, out_proj):
        module.requires_grad_(False)
    return ReferenceModel(cfg, None, None, expert, in_proj, out_proj)


def _tt_action(mesh_device, cfg, expert_weights, in_proj_weights, out_proj_weights):
    expert = TtExpert(cfg.expert, mesh_device=mesh_device, lm_num_layers=cfg.text.layers)
    expert.load_torch_state_dict(TtExpert.convert_state(expert_weights))
    in_proj = TtActionInProj(cfg.action, cfg.expert.hidden, mesh_device=mesh_device)
    in_proj.load_torch_state_dict(in_proj_weights)
    out_proj = TtActionOutProj(cfg.expert.hidden, cfg.action.action_dim, mesh_device=mesh_device)
    out_proj.load_torch_state_dict(out_proj_weights)
    return expert, in_proj, out_proj


def test_action_features_rejects_bad_shapes(cfg, expect_error):
    n = cfg.action.n_waypoints
    with expect_error(ValueError, "expected 2 action dims"):
        action_features(torch.zeros(1, n, 3), torch.zeros(1, 1, 1), cfg.action)
    with expect_error(ValueError, "expected flow time"):
        action_features(torch.zeros(1, n, 2), torch.zeros(1, 1), cfg.action)


@P100_MESH
def test_in_proj(mesh_device, cfg, in_proj_weights):
    batch, n = 2, cfg.action.n_waypoints
    torch.manual_seed(SEED)
    x = torch.randn(batch, n, cfg.action.action_dim)
    t = _timestep(cfg, 3, batch)

    reference = ActionInProj(cfg.action, cfg.expert.hidden, in_proj_weights)
    with torch.no_grad():
        golden = reference(x, t)
        features = action_features(x, t, cfg.action)
        # The host features are exactly what the reference feeds its trunk.
        trunk = reference.encoder.trunk(features.flatten(0, 1)).reshape(batch, n, -1)
        torch.testing.assert_close(reference.norm(trunk), golden, rtol=0, atol=1e-5)

    in_proj = TtActionInProj(cfg.action, cfg.expert.hidden, mesh_device=mesh_device)
    in_proj.load_torch_state_dict(in_proj_weights)
    tt_features = tensor.from_torch(features, device=mesh_device)
    out = tensor.to_torch(in_proj.forward(tt_features))

    logger.info(
        f"bf16 features PCC {pcc(features, tensor.to_torch(tt_features)) * 100:.4f} %, "
        f"in-proj PCC {pcc(golden, out) * 100:.4f} %"
    )
    assert out.shape == golden.shape
    assert_quality(golden, out, pcc=PCC, ccc=PCC)


# Relative per-step velocity error injected into the reference flow by `test_flow_sensitivity`, below the TT
# velocity's on either weight set.
FLOW_NOISE = 0.004


def _reference_flow(cfg, inputs, prefix, model: ReferenceModel, noise, *, step_noise: float = 0.0) -> dict:
    """Euler-integrate the reference flow from `noise`, recording each step's input x and velocity.

    With `step_noise` > 0 each velocity gets seeded Gaussian noise of that fraction of its std before the update.
    """
    cot = CotOut(tokens=prefix["tokens"], offsets=prefix["offsets"], logits=None)
    cache = reference_cache(prefix)
    generator = torch.Generator().manual_seed(1)
    xs, vs = [], []

    def step(x, t):
        v = model.expert_velocity(cache, cot, inputs, x, t)
        xs.append(x.clone())
        vs.append(v)
        print(f"reference flow step {len(vs)}/{cfg.action.steps}", flush=True)
        return v + step_noise * v.std() * torch.randn(v.shape, generator=generator) if step_noise else v

    with torch.no_grad():
        actions = euler_sample(step, noise, cfg.action.steps)
    return {"x": xs, "v": vs, "actions": actions}


def _flow_key(inputs, prefix, noise, step_noise: float) -> dict:
    return {
        "input_ids": inputs.input_ids,
        "position_ids": inputs.position_ids,
        "patches": inputs.patches,
        "grid": inputs.image_grid_thw,
        "forced": prefix["tokens"],
        "k": prefix["k"],
        "v": prefix["v"],
        "noise": noise,
        "step_noise": step_noise,
    }


@pytest.fixture(scope="module")
def reference_model(cfg, expert_weights, in_proj_weights, out_proj_weights) -> ReferenceModel:
    return _reference_model(cfg, expert_weights, in_proj_weights, out_proj_weights)


@pytest.fixture(scope="module")
def reference_flow(cfg, sample_inputs, prefix, reference_model) -> dict:
    """The reference's 10-step flow from `_noise`: each step's input x and velocity, and the final actions."""
    noise = _noise(cfg, 1)
    return golden_cache.cached(
        "action_flow",
        _flow_key(sample_inputs, prefix, noise, 0.0),
        lambda: _reference_flow(cfg, sample_inputs, prefix, reference_model, noise),
        cfg=cfg,
    )


@pytest.fixture
def checkpoint_only(weights):
    if weights != "checkpoint":
        pytest.skip(
            f"the flow of random weights is chaotic: a {FLOW_NOISE:.1%} per-step velocity error already moves the "
            "reference's own 10-step actions below the PCC floor (test_flow_sensitivity), so the closed loop would "
            "measure the weights, not the port"
        )


def test_flow_sensitivity(weights, cfg, sample_inputs, prefix, reference_model, reference_flow):
    """How far a small per-step velocity error moves the reference's own 10-step actions.

    The trained flow absorbs it; the random one is chaotic and decorrelates, which is why `test_actions_10_steps`
    runs on the checkpoint only. If the random case passes the floor, that skip can go.
    """
    noise = _noise(cfg, 1)
    perturbed = golden_cache.cached(
        "action_flow",
        _flow_key(sample_inputs, prefix, noise, FLOW_NOISE),
        lambda: _reference_flow(cfg, sample_inputs, prefix, reference_model, noise, step_noise=FLOW_NOISE),
        cfg=cfg,
    )
    flow_pcc = pcc(reference_flow["actions"], perturbed["actions"])
    logger.info(f"{weights}: {FLOW_NOISE:.1%} per-step velocity noise -> 10-step actions PCC {flow_pcc * 100:.4f} %")
    if weights == "checkpoint":
        assert flow_pcc > 0.999
    else:
        assert flow_pcc < PCC, "the random-weight flow is no longer chaotic; drop the skip in test_actions_10_steps"


@P100_MESH
@pytest.mark.timeout(1500)
def test_velocity_every_step(
    mesh_device, cfg, expert_setup, expert_weights, in_proj_weights, out_proj_weights, reference_flow
):
    """The TT velocity at each Euler step's reference input, so a step's error cannot carry into the next."""
    cache, frame, prefix_len = expert_setup
    expert, in_proj, out_proj = _tt_action(mesh_device, cfg, expert_weights, in_proj_weights, out_proj_weights)
    velocities = []
    for step, (x, golden) in enumerate(zip(reference_flow["x"], reference_flow["v"])):
        t = _timestep(cfg, step, 1)
        v = action_velocity_tt(expert, in_proj, out_proj, cache=cache, frame=frame, x=x, t=t, acfg=cfg.action)
        logger.info(f"{prefix_len}-row prefix, step {step}: velocity PCC {pcc(golden, v) * 100:.4f} %")
        velocities.append(v)
    # Every step is measured and logged before the first failing one stops the test.
    for golden, v in zip(reference_flow["v"], velocities):
        assert v.dtype == torch.float32
        assert v.shape == golden.shape
        assert_quality(golden, v, pcc=PCC, ccc=PCC)


@P100_MESH
@pytest.mark.timeout(1500)
def test_actions_10_steps(
    checkpoint_only,
    mesh_device,
    cfg,
    sample_inputs,
    expert_setup,
    expert_weights,
    in_proj_weights,
    out_proj_weights,
    reference_flow,
):
    inputs = sample_inputs
    acfg = cfg.action
    cache, frame, prefix_len = expert_setup
    noise = _noise(cfg, 1)
    golden = reference_flow["actions"]

    expert, in_proj, out_proj = _tt_action(mesh_device, cfg, expert_weights, in_proj_weights, out_proj_weights)
    actions = sample_actions_tt(expert, in_proj, out_proj, cache=cache, frame=frame, noise=noise, acfg=acfg)

    delta = (actions - golden).abs().amax(dim=(0, 1))
    golden_xyz, _ = action_to_traj(golden, inputs.ego_history_xyz, inputs.ego_history_rot, acfg)
    xyz, _ = action_to_traj(actions, inputs.ego_history_xyz, inputs.ego_history_rot, acfg)
    position_error = torch.linalg.vector_norm(xyz - golden_xyz, dim=-1)
    logger.info(
        f"{cfg.expert.layers} layers, prefix {prefix_len}, {acfg.steps} steps: actions PCC "
        f"{pcc(golden, actions) * 100:.4f} %, max |delta| accel {delta[0]:.4f}, curvature {delta[1]:.4f} "
        f"(normalized); xyz max error {position_error.max():.4f} m at waypoint {int(position_error.argmax(-1)[0])}, "
        f"final-waypoint error {position_error[..., -1].max():.4f} m"
    )
    assert actions.dtype == torch.float32
    assert actions.shape == golden.shape
    assert_quality(golden, actions, pcc=PCC, ccc=PCC)
