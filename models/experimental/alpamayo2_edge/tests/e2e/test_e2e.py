# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""End-to-end TT pipeline against the torch reference on one Blackhole P100.

The teacher-forced tests force the same CoT and flow noise through both sides, so the trajectories differ only by
the device numerics of the vision tower, LM, expert and action projections.
"""

import os

import pytest
import torch
from loguru import logger
from tokenizers import Tokenizer

import ttnn
from models.experimental.alpamayo2_edge.reference import golden_cache
from models.experimental.alpamayo2_edge.reference.action import action_to_traj
from models.experimental.alpamayo2_edge.reference.forced_cot import driving_text_cot
from models.experimental.alpamayo2_edge.reference.model import TEMPERATURE, ReferenceModel
from models.experimental.alpamayo2_edge.tests._util import PCC, full_depth_only, pcc
from models.experimental.alpamayo2_edge.tests.conftest import P100_MESH
from models.experimental.alpamayo2_edge.tt.model import Alpamayo2EdgeTT
from models.tt_dit.utils.check import assert_quality

SEED = 0
# Max Euclidean waypoint error against the reference with the same CoT and noise: about twice the worst measured
# error (6.4 cm, at the last waypoint), rounded. Its fit on real driving data is checked in #66402.
POSITION_BOUND_M = 0.13
LM_HEAD_DTYPES = {"bf8": ttnn.bfloat8_b, "bf16": ttnn.bfloat16}
# What seed 0 samples on `synthetic_sample(0)`, the same CoT as the official stack: "<NOMINAL>\n" and the anchor.
FREE_RUNNING_COT = [124549, 11135, 3174, 4286, 1561, 135084]


def _noise(cfg, batch: int) -> torch.Tensor:
    shape = (batch, cfg.action.n_waypoints, cfg.action.action_dim)
    return torch.randn(*shape, generator=torch.Generator().manual_seed(SEED))


def _assert_finite(**tensors: torch.Tensor) -> None:
    for name, value in tensors.items():
        assert bool(torch.isfinite(value).all()), f"{name} has NaN or Inf"


def _position_error(golden_xyz: torch.Tensor, xyz: torch.Tensor) -> torch.Tensor:
    """Euclidean error in meters per waypoint, [..., N]."""
    return torch.linalg.vector_norm(xyz - golden_xyz, dim=-1)


def _reference_e2e(cfg, inputs, forced: torch.Tensor, noise: torch.Tensor) -> dict:
    """`ReferenceModel.infer` on noise row 0 with its stage outputs, then the flow from noise row 1 on the same CoT.

    Both rows share the prompt and the forced CoT, so row 1 needs only the flow and `action_to_traj` over the same
    prefix cache, exactly as `infer` runs them.
    """
    model = ReferenceModel.from_pretrained(cfg)
    captured = {}

    def progress(what):
        count = []

        def hook(_m, _a, _out):
            count.append(None)
            print(f"reference {what} {len(count)}", flush=True)

        return hook

    for block in model.vision.blocks:
        block.register_forward_hook(progress("vision block"))
    model.vision.register_forward_hook(lambda _m, _a, out: captured.update(vision=out))
    model.lm.text.register_forward_hook(progress("LM forward"))
    model.expert.register_forward_hook(progress("expert forward"))

    prefill, decode = model.prefill, model.decode

    def capture_prefill(inp):
        logits, cache = prefill(inp)
        captured.update(prefill_logits=logits, cache=cache)
        return logits, cache

    def capture_decode(*args, **kwargs):
        captured["cot"] = decode(*args, **kwargs)
        return captured["cot"]

    model.prefill, model.decode = capture_prefill, capture_decode
    first = model.infer(inputs, forced_cot=forced, noise=noise[:1])
    cot = captured["cot"]
    actions = model.sample_actions(captured["cache"], cot, inputs, noise[1:2])
    xyz, rot = action_to_traj(actions, inputs.ego_history_xyz, inputs.ego_history_rot, cfg.action)
    return {
        "vision": captured["vision"],
        "prefill_logits": captured["prefill_logits"],
        "cot_tokens": cot.tokens,
        "offsets": cot.offsets,
        "cot_logits": cot.logits,
        "actions": torch.cat([first.actions, actions]),
        "xyz": torch.cat([first.xyz, xyz]),
        "rot": torch.cat([first.rot, rot]),
    }


@pytest.fixture(scope="module")
def golden(cfg, sample_inputs):
    """Reference outputs for the forced CoT under noise rows 0 and 1 (`_noise(cfg, 2)`)."""
    inputs = sample_inputs
    forced = driving_text_cot(cfg)
    noise = _noise(cfg, 2)
    result = golden_cache.cached(
        "e2e_teacher_forced",
        {
            "input_ids": inputs.input_ids,
            "position_ids": inputs.position_ids,
            "patches": inputs.patches,
            "grid": inputs.image_grid_thw,
            "history": [inputs.ego_history_xyz, inputs.ego_history_rot],
            "forced": forced,
            "noise": noise,
        },
        lambda: _reference_e2e(cfg, inputs, forced, noise),
        cfg=cfg,
    )
    assert torch.equal(result["cot_tokens"], forced[None])
    return result


def _decode_quality(golden_logits: torch.Tensor, logits: torch.Tensor) -> dict:
    """Agreement of the per-step [1, T, V] logits with the reference, at the sampling temperature."""
    golden_logits, logits = golden_logits[0].double(), logits[0].double()
    golden_ids, ids = golden_logits.argmax(-1), logits.argmax(-1)
    disagree = (golden_ids != ids).nonzero().flatten().tolist()
    top2 = golden_logits.topk(2, dim=-1).values
    log_p = torch.log_softmax(golden_logits / TEMPERATURE, dim=-1)
    log_q = torch.log_softmax(logits / TEMPERATURE, dim=-1)
    kl = (log_p.exp() * (log_p - log_q)).sum(-1)
    return {
        "agreement": 1 - len(disagree) / golden_logits.shape[0],
        # Per disagreeing step: the reference's top-1 minus top-2 logit, and how far the TT logit of the
        # reference's choice falls below the TT argmax.
        "disagreements": [
            (
                t,
                float(top2[t, 0] - top2[t, 1]),
                float(logits[t, ids[t]] - logits[t, golden_ids[t]]),
            )
            for t in disagree
        ],
        "kl_mean": float(kl.mean()),
        "kl_max": float(kl.max()),
    }


def _log_decode_quality(label: str, quality: dict) -> None:
    logger.info(
        f"{label}: argmax agreement {quality['agreement'] * 100:.1f} %, "
        f"KL(ref, tt) at T={TEMPERATURE} mean {quality['kl_mean']:.3e} max {quality['kl_max']:.3e} nats"
    )
    for step, ref_margin, tt_margin in quality["disagreements"]:
        logger.info(
            f"{label}: step {step:3d} disagrees: reference top-1 - top-2 margin {ref_margin:.4f}, "
            f"TT argmax - TT logit of the reference choice {tt_margin:.4f}"
        )


@P100_MESH
@pytest.mark.timeout(1500)
def test_e2e_teacher_forced(mesh_device, cfg, sample_inputs, golden):
    inputs = sample_inputs
    forced = driving_text_cot(cfg)
    noise = _noise(cfg, 2)[:1]
    seq, steps, n = inputs.input_ids.shape[1], forced.shape[0], cfg.action.n_waypoints

    model = Alpamayo2EdgeTT.from_pretrained(mesh_device, cfg)
    run = model.run(inputs, forced_cot=forced, noise=noise, keep_intermediates=True)
    pred = run.prediction

    _assert_finite(vision=run.vision_embeds, logits=run.cot.logits, actions=pred.actions, xyz=pred.xyz, rot=pred.rot)
    assert torch.equal(pred.cot_tokens, forced[None])
    assert run.cot.offsets.tolist() == golden["offsets"].tolist() == [seq + steps]
    assert pred.xyz.shape == (1, n, 3)
    assert pred.rot.shape == (1, n, 3, 3)
    assert pred.actions.shape == (1, n, cfg.action.action_dim)

    step_pccs = torch.tensor([pcc(golden["cot_logits"][:, t], run.cot.logits[:, t]) for t in range(steps)])
    error = _position_error(golden["xyz"][:1], pred.xyz)[0]
    logger.info(
        f"S={seq}, {steps} forced CoT tokens, layers vision/LM/expert "
        f"{cfg.vision.layers}/{cfg.text.layers}/{cfg.expert.layers}; timings "
        + ", ".join(f"{k} {v:.1f} s" for k, v in run.timings.items())
    )
    logger.info(f"vision embeds PCC {pcc(golden['vision'], run.vision_embeds) * 100:.4f} %")
    logger.info(f"prefill last logits PCC {pcc(golden['prefill_logits'], run.cot.logits[:, 0]) * 100:.4f} %")
    logger.info(
        f"decode logits PCC min {step_pccs.min() * 100:.4f} % (step {int(step_pccs.argmin())}), "
        f"mean {step_pccs.mean() * 100:.4f} %"
    )
    logger.info(f"actions PCC {pcc(golden['actions'][:1], pred.actions) * 100:.4f} %")
    logger.info(
        f"xyz PCC {pcc(golden['xyz'][:1], pred.xyz) * 100:.6f} %, max position error {error.max():.4f} m "
        f"at waypoint {int(error.argmax())}, final waypoint {error[-1]:.4f} m"
    )
    _log_decode_quality("bf8 lm_head", _decode_quality(golden["cot_logits"], run.cot.logits))

    assert_quality(golden["vision"], run.vision_embeds, pcc=PCC)
    assert_quality(golden["prefill_logits"], run.cot.logits[:, 0], pcc=PCC)
    # Written so that a NaN PCC fails too.
    failing = (~(step_pccs >= PCC)).nonzero().flatten().tolist()
    assert not failing, f"decode steps {failing} have logits PCC below {PCC}: {step_pccs[failing].tolist()}"
    assert_quality(golden["actions"][:1], pred.actions, pcc=PCC, ccc=PCC)
    # Forward motion dominates the xyz PCC, so the trajectory gate is the position error.
    assert error.max() < POSITION_BOUND_M, f"max position error {error.max():.4f} m >= {POSITION_BOUND_M} m"


@P100_MESH
@pytest.mark.timeout(1500)
@pytest.mark.skipif(os.environ.get("ALPAMAYO_MEASURE") != "1", reason="a measurement; set ALPAMAYO_MEASURE=1 to run")
@pytest.mark.parametrize("lm_head", list(LM_HEAD_DTYPES))
def test_e2e_lm_head_decode_quality(mesh_device, cfg, sample_inputs, golden, lm_head):
    """A measurement, not a gate: teacher-forced decode agreement and CoT time with the lm_head in bf8 or bf16.

    The second run is timed, so the kernels are compiled and cached.
    """
    inputs = sample_inputs
    forced = driving_text_cot(cfg)
    noise = _noise(cfg, 2)[:1]

    model = Alpamayo2EdgeTT.from_pretrained(mesh_device, cfg, lm_head_dtype=LM_HEAD_DTYPES[lm_head])
    model.run(inputs, forced_cot=forced, noise=noise, keep_intermediates=True)
    run = model.run(inputs, forced_cot=forced, noise=noise, keep_intermediates=True)

    _assert_finite(logits=run.cot.logits, xyz=run.prediction.xyz)
    assert torch.equal(run.prediction.cot_tokens, forced[None])
    step_pccs = torch.tensor([pcc(golden["cot_logits"][:, t], run.cot.logits[:, t]) for t in range(forced.shape[0])])
    logger.info(
        f"{lm_head} lm_head: warm timings "
        + ", ".join(f"{k} {v:.2f} s" for k, v in run.timings.items())
        + f"; decode logits PCC min {step_pccs.min() * 100:.4f} %, mean {step_pccs.mean() * 100:.4f} %"
    )
    _log_decode_quality(f"{lm_head} lm_head", _decode_quality(golden["cot_logits"], run.cot.logits))


@P100_MESH
@pytest.mark.timeout(1500)
@full_depth_only
def test_e2e_free_running_smoke(mesh_device, cfg, sample_inputs):
    """Seeded sampling to the anchor gives a trajectory and the documented short CoT.

    Running out of tokens without an anchor is the other possible outcome of `infer`; it fails this test.
    """
    inputs = sample_inputs
    anchor = cfg.tokens.future_start
    model = Alpamayo2EdgeTT.from_pretrained(mesh_device, cfg)

    try:
        pred = model.infer(inputs, seed=SEED, max_new_tokens=256)
    except ValueError as error:
        pytest.fail(f"expected a trajectory with the CoT {FREE_RUNNING_COT}, got no anchor in 256 tokens: {error}")

    tokenizer = Tokenizer.from_file(str(cfg.model_dir / "tokenizer.json"))
    text = tokenizer.decode(pred.cot_tokens[0].tolist(), skip_special_tokens=False)
    logger.info(f"outcome: trajectory; sampled CoT ({pred.cot_tokens.shape[1]} tokens): {text!r}")
    logger.info(f"xyz waypoints 0, 31, 63: {pred.xyz[0, [0, 31, 63]].tolist()}")
    assert pred.xyz.shape == (1, cfg.action.n_waypoints, 3)
    assert pred.cot_tokens[0].tolist() == FREE_RUNNING_COT
    assert int(pred.cot_tokens[0, -1]) == anchor
    _assert_finite(actions=pred.actions, xyz=pred.xyz, rot=pred.rot)


@P100_MESH
@pytest.mark.timeout(1500)
def test_e2e_two_samples_smoke(mesh_device, cfg, sample_inputs, golden):
    """Two samples of one prompt: the repeated cache, two-row decode and two-row expert, one noise row each."""
    inputs = sample_inputs
    forced = driving_text_cot(cfg)
    noise = _noise(cfg, 2)
    n = cfg.action.n_waypoints

    model = Alpamayo2EdgeTT.from_pretrained(mesh_device, cfg)
    pred = model.infer(inputs, num_samples=2, forced_cot=forced, noise=noise)

    assert pred.xyz.shape == (2, n, 3)
    assert pred.rot.shape == (2, n, 3, 3)
    assert torch.equal(pred.cot_tokens, forced.expand(2, -1))
    _assert_finite(actions=pred.actions, xyz=pred.xyz, rot=pred.rot)
    for row in range(2):
        own = _position_error(golden["xyz"][row], pred.xyz[row]).max()
        other = _position_error(golden["xyz"][1 - row], pred.xyz[row]).max()
        logger.info(
            f"sample {row}: xyz PCC {pcc(golden['xyz'][row], pred.xyz[row]) * 100:.6f} %, "
            f"max position error {own:.4f} m (vs the other sample's reference {other:.4f} m)"
        )
        # Forward motion dominates the xyz PCC, so it barely separates the two noise draws; the error does.
        assert own < other, f"sample {row} is closer to the other sample's reference"
        assert own < POSITION_BOUND_M, f"sample {row}: max position error {own:.4f} m >= {POSITION_BOUND_M} m"
