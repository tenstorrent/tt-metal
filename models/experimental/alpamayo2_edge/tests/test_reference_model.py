# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import math
import os
import urllib.error
import urllib.request
from dataclasses import replace
from pathlib import Path

import einops
import pytest
import torch
from torch import nn

from models.experimental.alpamayo2_edge.reference import golden_cache
from models.experimental.alpamayo2_edge.reference.action import (
    ActionInProj,
    action_to_traj,
    estimate_t0_states,
    euler_sample,
    expert_mask,
    expert_position_ids,
)
from models.experimental.alpamayo2_edge.reference.config import num_layers_override
from models.experimental.alpamayo2_edge.reference.inputs import prepare_sample_inputs, synthetic_sample
from models.experimental.alpamayo2_edge.reference.model import ReferenceModel, make_forced_cot, process_logits
from models.experimental.alpamayo2_edge.reference.text import KVCache
from models.experimental.alpamayo2_edge.reference.weights import load_weights

# Plain text token ids for teacher-forced CoTs; none of them is a trajectory, EOS or anchor token.
TEXT = torch.tensor([1000, 2001, 3002, 4003, 5004, 6005, 7006, 8007])


@pytest.fixture(scope="module")
def small_cfg(cfg):
    layers = num_layers_override() or 2
    return replace(
        cfg,
        vision=replace(cfg.vision, layers=layers),
        text=replace(cfg.text, layers=layers),
        expert=replace(cfg.expert, layers=layers),
    )


@pytest.fixture(scope="module")
def model(small_cfg):
    return ReferenceModel.from_pretrained(small_cfg)


@pytest.fixture(scope="module")
def inputs(cfg):
    return prepare_sample_inputs(cfg, synthetic_sample(0))


@pytest.fixture(scope="module")
def prefilled(model, inputs):
    return model.prefill(inputs)


def _fresh(cache: KVCache) -> KVCache:
    """A copy that decode can extend and crop; cached tensors are never modified in place."""
    out = KVCache(len(cache.k))
    out.k, out.v = list(cache.k), list(cache.v)
    return out


def _anchor(cfg) -> torch.Tensor:
    return torch.tensor([cfg.tokens.future_start])


# ---------------------------------------------------------------------------------------------------------------------
# Upstream oracles: verbatim copies from NVlabs alpamayo2-edge (Apache-2.0), models/action_in_proj.py,
# models/expert_utils.py and diffusion/flow_matching.py.
# ---------------------------------------------------------------------------------------------------------------------


class _UpstreamRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return output.type_as(x) * self.weight


class _UpstreamMLPEncoder(nn.Module):
    def __init__(self, num_input_feats, num_enc_layers, hidden_size, outdim) -> None:
        super().__init__()
        enc_layers = [nn.Linear(num_input_feats, hidden_size), nn.SiLU()]
        for layer_idx in range(num_enc_layers):
            if layer_idx < num_enc_layers - 1:
                enc_layers.extend(
                    [_UpstreamRMSNorm(hidden_size, eps=1e-5), nn.Linear(hidden_size, hidden_size), nn.SiLU()]
                )
            else:
                enc_layers.extend([_UpstreamRMSNorm(hidden_size, eps=1e-5), nn.Linear(hidden_size, outdim)])
        self.trunk = nn.Sequential(*enc_layers)

    def forward(self, x):
        return self.trunk(x)


class _UpstreamFourierEncoderV2(nn.Module):
    def __init__(self, dim: int, max_freq: float = 100.0) -> None:
        super().__init__()
        half = dim // 2
        freqs = torch.logspace(0, math.log10(max_freq), steps=half)
        self.out_dim = dim
        self.register_buffer("freqs", freqs[None, :])

    def forward(self, x):
        arg = x[..., None] * self.freqs * 2 * torch.pi
        return torch.cat([torch.sin(arg), torch.cos(arg)], -1) * math.sqrt(2)


class _UpstreamPerWaypointActionInProjV2(nn.Module):
    def __init__(self, in_dims, out_dim, num_enc_layers=4, hidden_size=1024, max_freq=100.0, num_fourier_feats=20):
        super().__init__()
        self.sinus = nn.ModuleList(
            _UpstreamFourierEncoderV2(dim=num_fourier_feats, max_freq=max_freq) for _ in range(in_dims[-1])
        )
        self.timestep_fourier_encoder = _UpstreamFourierEncoderV2(dim=num_fourier_feats, max_freq=max_freq)
        num_input_feats = sum(encoder.out_dim for encoder in self.sinus) + self.timestep_fourier_encoder.out_dim
        self.encoder = _UpstreamMLPEncoder(num_input_feats, num_enc_layers, hidden_size, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x, timesteps):
        batch_size, n_waypoints, _ = x.shape
        x = x.float()
        timesteps = timesteps.float()
        action_feats = torch.cat([encoder(x[:, :, idx]) for idx, encoder in enumerate(self.sinus)], dim=-1)
        timestep_feats = self.timestep_fourier_encoder(timesteps[..., -1])
        timestep_feats = timestep_feats.repeat(1, n_waypoints, 1)
        features = torch.cat((action_feats, timestep_feats), dim=-1)
        features = features.to(self.encoder.trunk[0].weight.dtype)
        encoded = self.encoder(features.flatten(0, 1)).reshape(batch_size, n_waypoints, -1)
        return self.norm(encoded)


def _upstream_build_expert_pos_ids_and_attn_mask(
    offset, rope_deltas, kv_cache_seq_len, n_diffusion_tokens, b_star, device, prefix_mask=None
):
    position_ids = torch.arange(n_diffusion_tokens, device=device)
    position_ids = einops.repeat(position_ids, "l -> 3 b l", b=b_star).clone()
    position_ids += (rope_deltas + offset[:, None]).to(position_ids.device)

    attention_mask = torch.zeros(
        (b_star, 1, n_diffusion_tokens, kv_cache_seq_len + n_diffusion_tokens), dtype=torch.float32, device=device
    )
    min_value = torch.finfo(attention_mask.dtype).min
    cache_columns = torch.arange(kv_cache_seq_len, device=device)
    invalid_cache = cache_columns[None, :] >= offset[:, None]
    attention_mask[..., :kv_cache_seq_len].masked_fill_(invalid_cache[:, None, None, :], min_value)

    if prefix_mask is not None:
        input_mask = prefix_mask[:, None, None, :]
        attention_mask[:, :, :, : input_mask.shape[-1]] = torch.where(
            input_mask == 0, min_value, attention_mask[:, :, :, : input_mask.shape[-1]]
        )
    return position_ids, attention_mask


def _upstream_euler(x_dims, batch_size, step_fn, noise, inference_step):
    """`FlowMatching._euler` without guidance; `noise` replaces the `torch.randn(...) * temperature` draw."""
    x = noise
    time_steps = torch.linspace(0.0, 1.0, inference_step + 1, device=noise.device)
    n_dim = len(x_dims)
    for i in range(inference_step):
        dt = time_steps[i + 1] - time_steps[i]
        dt = dt.view(1, *[1] * n_dim).expand(batch_size, *[1] * n_dim)
        t_start = time_steps[i].view(1, *[1] * n_dim).expand(batch_size, *[1] * n_dim)
        v = step_fn(x=x, t=t_start)
        x = x + dt * v
    return x


# ---------------------------------------------------------------------------------------------------------------------
# NVlabs course action-space scenarios, ported from tests/test_course_action_space.py (Apache-2.0). The frozen
# float64 outputs live in tests/data/course_action_space_reference.safetensors, downloaded at test time.
# ---------------------------------------------------------------------------------------------------------------------

NVLABS_FIXTURE_URL = (
    "https://raw.githubusercontent.com/NVlabs/alpamayo2-edge/02f55a7aeabc1a2b9f7a80445c08e40bddc89f5e/"
    "tests/data/course_action_space_reference.safetensors"
)
HISTORY_STEPS, FUTURE_STEPS, DT = 20, 64, 0.1
COURSE_CFG = {
    "accel_mean": 0.02902694707164455,
    "accel_std": 0.6810426736454882,
    "curvature_mean": 0.0002692167976330542,
    "curvature_std": 0.026148280660833106,
    "accel_bounds": (-9.8, 9.8),
    "curvature_bounds": (-0.33, 0.33),
    "dt": DT,
    "n_waypoints": FUTURE_STEPS,
    "theta_ridge": 1e-8,
    "v_lambda": 1e-6,
    "v_ridge": 1e-4,
    "a_lambda": 1e-4,
    "a_ridge": 1e-4,
    "kappa_lambda": 1e-4,
    "kappa_ridge": 1e-4,
    "theta_lambda": 3e-6,
    "slip_gradient": 2.5e-3,
}
# NVlabs tolerances against the float64 reference; float32 is the inference precision.
TOLERANCES = {
    torch.float64: {"xyz": 1e-9, "yaw": 1e-9, "t0": 1e-9},
    torch.float32: {"xyz": 5e-4, "yaw": 1e-4, "t0": 5e-5},
}


def _hash_noise(index, seed):
    return 2.0 * torch.frac(torch.sin(index * 12.9898 + seed) * 43758.5453).abs() - 1.0


SCENARIOS = {
    "straight_cruise": dict(speed=lambda t: 15.0 + 0 * t, curvature=lambda t: 0 * t),
    "left_arc": dict(speed=lambda t: 10.0 + 0 * t, curvature=lambda t: 0.02 + 0 * t),
    "right_curve_accelerating": dict(speed=lambda t: 4.0 + 1.2 * t, curvature=lambda t: -0.04 + 0 * t),
    "pull_away_full_lock": dict(speed=lambda t: t.clamp(min=0.0), curvature=lambda t: 0.2 + 0 * t),
    "stop_and_go": dict(
        speed=lambda t: torch.where(t < 3.0, (2.0 - t).clamp(min=0.0), (1.5 * (t - 4.0)).clamp(min=0.0)),
        curvature=lambda t: 0.03 + 0 * t,
    ),
    "creep_course_fade": dict(speed=lambda t: 0.75 + 0 * t, curvature=lambda t: 0.1 + 0 * t),
    "reversing": dict(speed=lambda t: -2.0 + 0 * t, curvature=lambda t: 0.05 + 0 * t),
    "u_turn_wrap": dict(speed=lambda t: 5.0 + 0 * t, curvature=lambda t: torch.where(t > 0, 0.15, 0.0)),
    "lane_change": dict(speed=lambda t: 20.0 + 0 * t, curvature=lambda t: 0.004 * torch.sin(2 * math.pi * t / 4)),
    "history_pose_glitch": dict(speed=lambda t: 8.0 + 0 * t, curvature=lambda t: 0 * t, history_shift=0.1),
    "future_pose_glitch": dict(speed=lambda t: 8.0 + 0 * t, curvature=lambda t: 0.01 + 0 * t, future_shift=(30, 0.12)),
    "noisy_pitch_roll": dict(
        speed=lambda t: 12.0 + 0 * t,
        curvature=lambda t: -0.015 + 0 * t,
        noise=(0.02, 1e-3),
        pitch_roll=(0.02, -0.01),
    ),
}


def _rotation(yaw, pitch, roll):
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    cp, sp, cr, sr = math.cos(pitch), math.sin(pitch), math.cos(roll), math.sin(roll)
    zero, one = torch.zeros_like(yaw), torch.ones_like(yaw)
    rz = torch.stack([cy, -sy, zero, sy, cy, zero, zero, zero, one], -1).reshape(-1, 3, 3)
    ry = torch.tensor([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=yaw.dtype)
    rx = torch.tensor([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=yaw.dtype)
    return rz @ ry @ rx


def _scenario(
    seed, speed, curvature, history_shift=0.0, future_shift=None, noise=(0.0, 0.0), pitch_roll=(0.0, 0.0), substeps=10
):
    h = DT / substeps
    t = torch.arange(-(HISTORY_STEPS - 1) * substeps, FUTURE_STEPS * substeps + 1) * h
    t = t.double()
    v, kappa = speed(t).double(), curvature(t).double()
    course = torch.cat([t.new_zeros(1), torch.cumsum(kappa * v * h, 0)[:-1]])
    x = torch.cat([t.new_zeros(1), torch.cumsum(v * torch.cos(course) * h, 0)[:-1]])
    y = torch.cat([t.new_zeros(1), torch.cumsum(v * torch.sin(course) * h, 0)[:-1]])
    lateral_accel = kappa * v * v.abs()
    yaw = course + torch.where(v.abs() > 0.1, 0.4e-3 + 2.7e-3 * lateral_accel, 0.0)
    x, y, yaw = x[::substeps], y[::substeps], yaw[::substeps]
    index = torch.arange(len(x), dtype=torch.float64)
    x = x + noise[0] * _hash_noise(index, seed)
    y = y + noise[0] * _hash_noise(index, seed + 0.5)
    yaw = yaw + noise[1] * _hash_noise(index, seed + 0.25)
    t0 = HISTORY_STEPS - 1
    dx, dy = x - x[t0], y - y[t0]
    c, s = math.cos(yaw[t0]), math.sin(yaw[t0])
    xy = torch.stack([c * dx + s * dy, -s * dx + c * dy], -1)
    yaw = yaw - yaw[t0]
    if history_shift:
        xy[t0 - 1, 1] += history_shift
    if future_shift is not None:
        xy[t0 + 1 + future_shift[0], 1] += future_shift[1]
    xyz = torch.cat([xy, torch.zeros_like(xy[:, :1])], -1)
    rot = _rotation(yaw, *pitch_roll)
    return xyz[:HISTORY_STEPS], rot[:HISTORY_STEPS], xyz[HISTORY_STEPS:], rot[HISTORY_STEPS:]


def scenario_batch(dtype):
    parts = [_scenario(i, **kwargs) for i, kwargs in enumerate(SCENARIOS.values())]
    hist_xyz, hist_rot, fut_xyz, fut_rot = (torch.stack(p) for p in zip(*parts, strict=True))
    k = torch.arange(FUTURE_STEPS, dtype=torch.float32)
    b = torch.arange(len(SCENARIOS), dtype=torch.float32)[:, None]
    sampled = torch.stack([1.5 * torch.sin(0.1 * k + b), 2.0 * torch.cos(0.07 * k + 0.5 * b)], -1)
    batch = {"hist_xyz": hist_xyz, "hist_rot": hist_rot, "sampled_action": sampled}
    return {key: value.to(dtype) for key, value in batch.items()}


def _angle_close(actual, expected, atol, msg):
    difference = torch.remainder(actual - expected + math.pi, 2 * math.pi) - math.pi
    torch.testing.assert_close(difference, torch.zeros_like(difference), rtol=0, atol=atol, msg=msg)


def _nvlabs_fixture() -> Path:
    path = golden_cache.cache_dir() / "nvlabs" / "course_action_space_reference.safetensors"
    if not path.exists():
        try:
            with urllib.request.urlopen(NVLABS_FIXTURE_URL, timeout=60) as response:
                data = response.read()
        except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
            pytest.skip(
                f"no network for the NVlabs action-space fixture ({e}); fetch {NVLABS_FIXTURE_URL} into {path.parent}"
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_bytes(data)
        os.replace(tmp, path)
    return path


# ---------------------------------------------------------------------------------------------------------------------
# Action projection, flow matching, expert mask
# ---------------------------------------------------------------------------------------------------------------------


def test_fourier_width(cfg):
    in_proj = ActionInProj(cfg.action, cfg.expert.hidden)
    assert in_proj.in_features == 60
    assert in_proj.in_features == cfg.action.fourier_feats * (cfg.action.action_dim + 1)


def test_action_in_proj_matches_upstream(cfg, model_dir):
    weights = load_weights(model_dir, "expert.action_in_proj.")
    ours = ActionInProj(cfg.action, cfg.expert.hidden, weights)
    acfg = cfg.action
    upstream = _UpstreamPerWaypointActionInProjV2(
        [acfg.n_waypoints, acfg.action_dim],
        cfg.expert.hidden,
        num_enc_layers=acfg.in_proj_layers,
        hidden_size=acfg.in_proj_hidden,
        max_freq=acfg.max_freq,
        num_fourier_feats=acfg.fourier_feats,
    )
    missing, unexpected = upstream.load_state_dict({k: v.float() for k, v in weights.items()}, strict=False)
    assert all(k.endswith(".freqs") for k in missing) and not unexpected

    g = torch.Generator().manual_seed(0)
    x = torch.randn(3, acfg.n_waypoints, acfg.action_dim, generator=g)
    t = torch.rand(3, 1, 1, generator=g)
    with torch.no_grad():
        out = ours(x, t)
        assert out.shape == (3, acfg.n_waypoints, cfg.expert.hidden)
        assert torch.equal(out, upstream(x, t))


def test_euler_matches_upstream():
    g = torch.Generator().manual_seed(0)
    noise = torch.randn(2, 64, 2, generator=g)
    seen = []

    def step_fn(x, t):
        seen.append(t)
        return torch.sin(x) - (1 + t) * x

    ours = euler_sample(step_fn, noise, 10)
    ours_t = [t.clone() for t in seen]
    seen.clear()
    expected = _upstream_euler((64, 2), 2, step_fn, noise, 10)
    assert torch.equal(ours, expected)
    assert len(ours_t) == 10
    for got, want in zip(ours_t, seen):
        assert got.shape == (2, 1, 1) and torch.equal(got, want)
    assert ours_t[0].eq(0).all() and torch.allclose(ours_t[-1], torch.tensor(0.9))


def test_expert_mask():
    offsets = torch.tensor([5, 8])
    kv_len, n = 10, 3
    mask = expert_mask(offsets, torch.ones(2, 4, dtype=torch.long), kv_len, n)
    assert mask.shape == (2, 1, n, kv_len + n) and mask.dtype == torch.float32
    min_value = torch.finfo(torch.float32).min
    for row, offset in enumerate(offsets.tolist()):
        assert (mask[row, :, :, :offset] == 0).all()
        assert (mask[row, :, :, offset:kv_len] == min_value).all()
        assert (mask[row, :, :, kv_len:] == 0).all()

    # Left-padded prompt columns are masked too; positions continue each row's text after its anchor.
    prefix_mask = torch.tensor([[1, 1, 1, 1], [0, 0, 1, 1]])
    rope_deltas = torch.tensor([[-7], [3]])
    want_pos, want_mask = _upstream_build_expert_pos_ids_and_attn_mask(
        offsets, rope_deltas, kv_len, n, 2, offsets.device, prefix_mask
    )
    assert torch.equal(expert_mask(offsets, prefix_mask, kv_len, n), want_mask)
    assert torch.equal(expert_position_ids(offsets, rope_deltas, n), want_pos)


# ---------------------------------------------------------------------------------------------------------------------
# Action space
# ---------------------------------------------------------------------------------------------------------------------


def test_zero_action_continues_straight_at_constant_speed(cfg):
    acfg = cfg.action
    hist_xyz = torch.zeros(1, HISTORY_STEPS, 3)
    hist_xyz[0, :, 0] = torch.arange(-(HISTORY_STEPS - 1), 1, dtype=torch.float32) * 1.0  # 10 m/s at 0.1 s
    hist_rot = torch.eye(3).expand(1, HISTORY_STEPS, 3, 3)
    # Zero physical acceleration and curvature, normalized.
    zero = torch.tensor([-acfg.accel_mean / acfg.accel_std, -acfg.curvature_mean / acfg.curvature_std])
    actions = zero.expand(1, acfg.n_waypoints, 2)

    v0 = estimate_t0_states(hist_xyz, hist_rot, acfg)["v"]
    torch.testing.assert_close(v0, torch.tensor([10.0]), rtol=0, atol=1e-2)
    xyz, rot = action_to_traj(actions, hist_xyz, hist_rot, acfg)
    assert xyz.shape == (1, acfg.n_waypoints, 3) and rot.shape == (1, acfg.n_waypoints, 3, 3)
    steps = torch.arange(1, acfg.n_waypoints + 1, dtype=torch.float32)
    torch.testing.assert_close(xyz[0, :, 0], v0 * acfg.dt * steps, rtol=1e-5, atol=1e-4)
    torch.testing.assert_close(xyz[0, :, 1:], torch.zeros(acfg.n_waypoints, 2), rtol=0, atol=1e-4)
    torch.testing.assert_close(rot[0], torch.eye(3).expand_as(rot[0]), rtol=0, atol=1e-5)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_action_to_traj_matches_nvlabs_fixture(cfg, dtype):
    from safetensors.torch import load_file

    reference = load_file(_nvlabs_fixture())
    acfg = cfg.action
    for name, value in COURSE_CFG.items():
        assert getattr(acfg, name) == value, name

    tol = TOLERANCES[dtype]
    batch = scenario_batch(dtype)
    hist = batch["hist_xyz"], batch["hist_rot"]
    t0 = estimate_t0_states(*hist, acfg)
    for key in ("v", "theta"):
        torch.testing.assert_close(
            t0[key].double(), reference[f"course.t0_{key}"], rtol=0, atol=tol["t0"], msg=f"t0_{key}"
        )
    for name, action in (("roundtrip", reference["course.action"].to(dtype)), ("sampled", batch["sampled_action"])):
        xyz, rot = action_to_traj(action, *hist, acfg)
        assert xyz.dtype == dtype and rot.dtype == dtype
        torch.testing.assert_close(
            xyz.double(), reference[f"course.{name}_xyz"], rtol=0, atol=tol["xyz"], msg=f"{name}_xyz"
        )
        yaw = torch.atan2(rot[..., 1, 0], rot[..., 0, 0]).double()
        _angle_close(yaw, reference[f"course.{name}_yaw"], tol["yaw"], f"{name}_yaw")


# ---------------------------------------------------------------------------------------------------------------------
# Sampling and CoT decode
# ---------------------------------------------------------------------------------------------------------------------


def test_make_forced_cot(cfg):
    anchor = cfg.tokens.future_start
    assert make_forced_cot(torch.tensor([5, 6, anchor, 7, anchor]), anchor).tolist() == [5, 6, anchor]
    forced = make_forced_cot(torch.tensor([5, 6]), anchor)
    assert forced.tolist() == [5, 6, anchor] and forced.dtype == torch.long


def test_sampling_processors_match_hf(cfg):
    from transformers.generation.logits_process import (
        LogitsProcessorList,
        TemperatureLogitsWarper,
        TopPLogitsWarper,
    )

    tok = cfg.tokens
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(2, cfg.text.vocab, generator=g) * 4
    logits[:, tok.future_start] += 30.0  # inside top-p, so only a processor could ban it
    expected = logits.clone()
    expected[:, tok.history_id0 : tok.history_id0 + tok.history_vocab] = -torch.inf
    expected[:, tok.future_id0 : tok.future_id0 + tok.future_vocab] = -torch.inf
    expected[:, [tok.eos]] = -torch.inf
    expected = LogitsProcessorList([TemperatureLogitsWarper(0.6), TopPLogitsWarper(0.98)])(None, expected)

    raw = logits.clone()
    actual = process_logits(logits, tok)
    assert torch.equal(actual, expected)
    assert torch.isfinite(actual[:, tok.future_start]).all()
    assert torch.equal(logits, raw), "the raw logits must not be modified"


def test_no_anchor_raises(model, inputs, prefilled, expect_error):
    with expect_error(ValueError, "No <traj_future_start>"):
        model.decode(inputs, _fresh(prefilled[1]), forced=TEXT[None, :4], seed=0, max_new_tokens=4)


def test_forced_cot_without_anchor_raises_like_upstream(model, inputs, prefilled, expect_error):
    # Forced tokens running out before max_new_tokens is the same failure: a row never reached its anchor.
    with expect_error(ValueError, r"No <traj_future_start> token found in generated rows \[0\]"):
        model.decode(inputs, _fresh(prefilled[1]), forced=TEXT[None, :4], seed=0, max_new_tokens=256)


def test_sampled_decode_replays_under_teacher_forcing(cfg, model, inputs, prefilled, monkeypatch):
    tok = cfg.tokens
    anchor, prompt_len = tok.future_start, inputs.input_ids.shape[1]
    last_logits, prompt_cache = prefilled
    assert last_logits.shape == (1, cfg.text.vocab) and prompt_cache.seq_len == prompt_len

    # The truncated test model never emits the anchor by itself; force it on the sixth step.
    original, calls = model.lm.logits, []

    def logits_with_late_anchor(hidden):
        out = original(hidden)
        calls.append(None)
        if len(calls) == 6:
            out[..., anchor] += 1e4
        return out

    monkeypatch.setattr(model.lm, "logits", logits_with_late_anchor)
    cache = _fresh(prompt_cache)
    cot = model.decode(inputs, cache, seed=7)
    calls.clear()
    again = model.decode(inputs, _fresh(prompt_cache), seed=7)
    monkeypatch.undo()

    assert cot.tokens.shape == (1, 6) and cot.logits.shape == (1, 6, cfg.text.vocab)
    assert cot.tokens[0, -1] == anchor and (cot.tokens[0, :-1] != anchor).all()
    assert cot.offsets.tolist() == [prompt_len + 6] and cache.seq_len == prompt_len + 6
    assert torch.equal(again.tokens, cot.tokens), "seeded sampling must be reproducible"
    sampled = cot.tokens[0, :-1]
    assert not ((sampled >= tok.history_id0) & (sampled < tok.history_id0 + tok.history_vocab)).any()
    assert not ((sampled >= tok.future_id0) & (sampled < tok.future_id0 + tok.future_vocab)).any()
    assert (sampled != tok.eos).all()
    # The first decode step recomputes the last prompt position from the cache.
    torch.testing.assert_close(cot.logits[:, 0], last_logits, rtol=1e-4, atol=1e-4)

    forced = make_forced_cot(cot.tokens[0], anchor)
    assert torch.equal(forced, cot.tokens[0])
    replay_cache = _fresh(prompt_cache)
    replay = model.decode(inputs, replay_cache, forced=forced[None], seed=0)
    assert torch.equal(replay.tokens, cot.tokens) and torch.equal(replay.offsets, cot.offsets)
    assert torch.equal(replay.logits[:, :5], cot.logits[:, :5])
    for layer in range(len(cache.k)):
        assert torch.equal(replay_cache.k[layer], cache.k[layer]) and torch.equal(replay_cache.v[layer], cache.v[layer])


def test_anchor_on_last_allowed_token_is_cached(cfg, model, inputs, prefilled):
    forced = torch.cat([TEXT[:3], _anchor(cfg)])[None]
    prompt_len = inputs.input_ids.shape[1]
    at_limit, roomy = _fresh(prefilled[1]), _fresh(prefilled[1])
    cot = model.decode(inputs, at_limit, forced=forced, seed=0, max_new_tokens=4)
    ref = model.decode(inputs, roomy, forced=forced, seed=0)
    assert cot.offsets.tolist() == ref.offsets.tolist() == [prompt_len + 4]
    assert at_limit.seq_len == roomy.seq_len == prompt_len + 4
    for layer in range(len(at_limit.k)):
        assert torch.equal(at_limit.k[layer], roomy.k[layer]) and torch.equal(at_limit.v[layer], roomy.v[layer])


# ---------------------------------------------------------------------------------------------------------------------
# Expert and pipeline
# ---------------------------------------------------------------------------------------------------------------------


def test_velocity_deterministic(cfg, model, inputs, prefilled):
    cache = _fresh(prefilled[1])
    cot = model.decode(inputs, cache, forced=torch.cat([TEXT[:5], _anchor(cfg)])[None], seed=0)
    prefix = cache.seq_len
    x = torch.randn(1, cfg.action.n_waypoints, 2, generator=torch.Generator().manual_seed(1))
    v1 = model.expert_velocity(cache, cot, inputs, x, 0.3)
    assert cache.seq_len == prefix, "the expert must leave the CoT cache as it found it"
    v2 = model.expert_velocity(cache, cot, inputs, x, 0.3)
    assert v1.shape == (1, cfg.action.n_waypoints, 2) and torch.isfinite(v1).all()
    assert torch.equal(v1, v2)
    assert not torch.equal(v1, model.expert_velocity(cache, cot, inputs, x, 0.7))


def test_expert_ignores_text_after_each_rows_anchor(cfg, model, inputs, prefilled):
    anchor = _anchor(cfg)
    early = torch.cat([TEXT[:2], anchor, TEXT[2:4]])  # anchor at 2, then two tokens past it
    late = torch.cat([TEXT[4:8], anchor])
    prompt_len = inputs.input_ids.shape[1]
    pair_cache = _fresh(prefilled[1])
    pair_cache.repeat_interleave(2)
    pair = model.decode(inputs, pair_cache, forced=torch.stack([early, late]), seed=0)
    assert pair.offsets.tolist() == [prompt_len + 3, prompt_len + 5] and pair_cache.seq_len == prompt_len + 5

    x = torch.randn(2, cfg.action.n_waypoints, 2, generator=torch.Generator().manual_seed(2))
    v_pair = model.expert_velocity(pair_cache, pair, inputs, x, 0.4)
    for row, forced in ((0, early[:3]), (1, late)):
        cache = _fresh(prefilled[1])
        cot = model.decode(inputs, cache, forced=forced[None], seed=0)
        v = model.expert_velocity(cache, cot, inputs, x[row : row + 1], 0.4)
        torch.testing.assert_close(v_pair[row : row + 1], v, rtol=1e-4, atol=1e-5)


def test_infer_pipeline(cfg, model, inputs, prefilled):
    forced = torch.cat([TEXT[:5], _anchor(cfg)])
    noise = torch.randn(1, cfg.action.n_waypoints, 2, generator=torch.Generator().manual_seed(3))
    pred = model.infer(inputs, forced_cot=forced, noise=noise)
    n = cfg.action.n_waypoints
    assert pred.xyz.shape == (1, n, 3) and pred.rot.shape == (1, n, 3, 3) and pred.actions.shape == (1, n, 2)
    assert torch.equal(pred.cot_tokens, forced[None])
    assert torch.isfinite(pred.xyz).all() and torch.isfinite(pred.actions).all()
    torch.testing.assert_close(pred.rot @ pred.rot.transpose(-1, -2), torch.eye(3).expand_as(pred.rot))

    # infer is exactly prefill -> decode -> Euler over the expert -> unicycle decode.
    cache = _fresh(prefilled[1])
    cot = model.decode(inputs, cache, forced=forced[None], seed=0)
    actions = model.sample_actions(cache, cot, inputs, noise)
    assert torch.equal(actions, pred.actions)
    xyz, rot = action_to_traj(actions, inputs.ego_history_xyz, inputs.ego_history_rot, cfg.action)
    assert torch.equal(xyz, pred.xyz) and torch.equal(rot, pred.rot)


def _function_in(path: Path, source: str):
    """`compute` from `source`, compiled as if defined in the file `path`, which is written first."""
    path.write_text(source)
    namespace = {}
    exec(compile(source, str(path), "exec"), namespace)
    return namespace["compute"]


def test_golden_cache(cfg, tmp_path, monkeypatch, expect_error):
    monkeypatch.setenv("ALPAMAYO_CACHE_DIR", str(tmp_path / "cache"))
    calls = []

    def compute():
        calls.append(None)
        return {"value": torch.arange(3)}

    def run(key, run_cfg=cfg, fn=compute):
        """True when the call computed and stored a new entry."""
        before = len(list((tmp_path / "cache").glob("probe-*.pt")))
        golden_cache.cached("probe", key, fn, cfg=run_cfg)
        return len(list((tmp_path / "cache").glob("probe-*.pt"))) > before

    noise = torch.zeros(2000)
    assert run({"seed": 0, "noise": noise})
    assert not run({"seed": 0, "noise": noise.clone()}), "an identical key must hit the cache"
    assert torch.equal(
        golden_cache.cached("probe", {"seed": 0, "noise": noise}, compute, cfg=cfg)["value"], torch.arange(3)
    )
    assert len(calls) == 1

    # Every input that changes the result must change the key.
    assert run({"seed": 1, "noise": noise})
    nudged = noise.clone()
    nudged[1000] = 1e-7  # invisible in a tensor's printed form
    assert run({"seed": 0, "noise": nudged})
    assert run({"seed": 0, "noise": noise.double()})
    assert run({"seed": 0, "noise": noise.reshape(2, 1000)})
    for part in ("vision", "text", "expert"):
        deeper = replace(cfg, **{part: replace(getattr(cfg, part), layers=getattr(cfg, part).layers + 1)})
        assert run({"seed": 0, "noise": noise}, deeper), part

    # The checkpoint is keyed by content: a copy elsewhere hits, an edited config misses.
    copy = tmp_path / "checkpoint-copy"
    copy.mkdir()
    for name in ("config.json", "model.safetensors.index.json"):
        (copy / name).write_bytes((cfg.model_dir / name).read_bytes())
    assert not run({"seed": 0, "noise": noise}, replace(cfg, model_dir=copy))
    (copy / "config.json").write_bytes((cfg.model_dir / "config.json").read_bytes() + b"\n")
    assert run({"seed": 0, "noise": noise}, replace(cfg, model_dir=copy))

    # An edit to the reference sources or to the file that defines the golden function misses.
    sources = tmp_path / "reference"
    sources.mkdir()
    (sources / "model.py").write_text("A = 1\n")
    monkeypatch.setattr(golden_cache, "_REFERENCE_DIR", sources)
    assert run({"seed": 0, "noise": noise})
    assert not run({"seed": 0, "noise": noise})
    (sources / "model.py").write_text("A = 2\n")
    assert run({"seed": 0, "noise": noise})

    body = "import torch\n\n\ndef compute():\n    return {'value': torch.arange(3)}\n"
    test_file = tmp_path / "golden_fn.py"
    assert run({"seed": 0, "noise": noise}, fn=_function_in(test_file, body))
    assert not run({"seed": 0, "noise": noise}, fn=_function_in(test_file, body))
    assert run({"seed": 0, "noise": noise}, fn=_function_in(test_file, body + "# edited\n"))

    with expect_error(TypeError, "cannot key"):
        golden_cache.cached("probe", {"path": object()}, compute, cfg=cfg)
