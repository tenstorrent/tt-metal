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

from models.experimental.alpamayo2_edge.reference.action import ActionInProj, action_to_traj
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


@P100_MESH
@pytest.mark.timeout(1500)
def test_velocity_one_step(
    mesh_device, cfg, sample_inputs, prefix, expert_setup, expert_weights, in_proj_weights, out_proj_weights
):
    inputs = sample_inputs
    cache, frame, prefix_len = expert_setup
    noise = _noise(cfg, 1)
    t = _timestep(cfg, 0, 1)

    reference = _reference_model(cfg, expert_weights, in_proj_weights, out_proj_weights)
    cot = CotOut(tokens=prefix["tokens"], offsets=prefix["offsets"], logits=None)
    golden = reference.expert_velocity(reference_cache(prefix), cot, inputs, noise, t)

    expert, in_proj, out_proj = _tt_action(mesh_device, cfg, expert_weights, in_proj_weights, out_proj_weights)
    v = action_velocity_tt(expert, in_proj, out_proj, cache=cache, frame=frame, x=noise, t=t, acfg=cfg.action)

    logger.info(f"{cfg.expert.layers} layers, prefix {prefix_len}: velocity PCC {pcc(golden, v) * 100:.4f} %")
    assert v.dtype == torch.float32
    assert v.shape == golden.shape
    assert_quality(golden, v, pcc=PCC, ccc=PCC)


@P100_MESH
@pytest.mark.timeout(1500)
def test_actions_10_steps(
    mesh_device, cfg, sample_inputs, prefix, expert_setup, expert_weights, in_proj_weights, out_proj_weights
):
    inputs = sample_inputs
    acfg = cfg.action
    cache, frame, prefix_len = expert_setup
    noise = _noise(cfg, 1)

    reference = _reference_model(cfg, expert_weights, in_proj_weights, out_proj_weights)
    cot = CotOut(tokens=prefix["tokens"], offsets=prefix["offsets"], logits=None)
    golden = reference.sample_actions(reference_cache(prefix), cot, inputs, noise)

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
