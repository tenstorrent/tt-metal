# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Denoise loop: reference trajectory, trajectory-comparison harness, halt-trace telemetry,
the ttnn controller's tensor ownership, and the assembled device step (#47463/#47468/#48291).

The CPU sections are pure torch with a synthetic ``logits_fn`` oracle — no checkpoint / ttnn / HW.
"""

import json
import os

import pytest
import torch

import ttnn
from models.experimental.diffusion_gemma.config import DiffusionConfig
from models.experimental.diffusion_gemma.reference import sampling as S
from models.experimental.diffusion_gemma.reference.denoise_loop import denoise_block as ref_denoise_block
from models.experimental.diffusion_gemma.tests.trajectory_pcc import (
    _pearson,
    assert_trajectory_matches,
    compare_trajectories,
    sound_entropy_step_fidelity,
)
from models.experimental.diffusion_gemma.tt import denoise_loop as DL
from models.experimental.diffusion_gemma.tt.denoise_loop import (
    denoise_block,
    denoise_step,
    renoise,
    temperature_at_step,
)
from models.experimental.diffusion_gemma.tt.traced_denoise import _summarize_halt_trace
from tests.ttnn.utils_for_testing import assert_with_pcc


def _gen(seed=0):
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def _cfg(**kw):
    return DiffusionConfig(max_denoise_steps=8, entropy_stop_threshold=0.1, stable_steps_to_halt=1, **kw)


def _peaked_logits(batch, length, vocab, target):
    logits = torch.full((batch, length, vocab), -1e4)
    logits[..., target] = 1e4
    return logits


# --- reference denoise trajectory (#47463/#47468) -------------------------------


def test_halts_on_stable_low_entropy():
    batch, length, vocab = 1, 8, 32
    target = 7
    peaked = _peaked_logits(batch, length, vocab, target)  # constant, near-zero entropy
    init = S.random_canvas((batch, length), vocab, generator=_gen(1))

    traj = ref_denoise_block(lambda canvas, step: peaked, init, _cfg(), vocab)

    assert traj.halted
    assert traj.num_steps <= 3  # stable+low-entropy detected as soon as prev exists
    assert torch.equal(traj.committed, torch.full((batch, length), target))  # commit = clean argmax
    assert len(traj.per_step) == traj.num_steps


def test_runs_to_cap_when_never_converges():
    batch, length, vocab = 1, 8, 64
    # constant near-uniform logits: argmax is stable but entropy stays high -> never halts
    flat = torch.zeros(batch, length, vocab)

    traj = ref_denoise_block(
        lambda canvas, step: flat, S.random_canvas((batch, length), vocab, generator=_gen(2)), _cfg(), vocab
    )

    assert not traj.halted
    assert traj.num_steps == 8
    assert all(r.entropy_mean > 1.0 for r in traj.per_step)


def test_committed_equals_last_step_clean_argmax():
    batch, length, vocab = 2, 16, 48

    # logits depend on step so argmax shifts; never halts -> runs to cap
    def logits_fn(canvas, step):
        g = _gen(100 + step)
        return torch.randn(batch, length, vocab, generator=g)

    traj = ref_denoise_block(logits_fn, S.random_canvas((batch, length), vocab, generator=_gen(3)), _cfg(), vocab)

    assert torch.equal(traj.committed, traj.per_step[-1].argmax)


def test_determinism_with_injected_noise():
    batch, length, vocab = 1, 12, 40

    def logits_fn(canvas, step):
        return torch.randn(batch, length, vocab, generator=_gen(200 + step))

    def gumbel_fn(step):
        return S.sample_gumbel_noise((batch, length, vocab), generator=_gen(300 + step))

    def noise_fn(step):
        return torch.randint(0, vocab, (batch, length), generator=_gen(400 + step))

    init = S.random_canvas((batch, length), vocab, generator=_gen(5))
    a = ref_denoise_block(logits_fn, init.clone(), _cfg(), vocab, gumbel_noise_fn=gumbel_fn, noise_tokens_fn=noise_fn)
    b = ref_denoise_block(logits_fn, init.clone(), _cfg(), vocab, gumbel_noise_fn=gumbel_fn, noise_tokens_fn=noise_fn)

    assert a.num_steps == b.num_steps and a.halted == b.halted
    assert torch.equal(a.committed, b.committed)
    for ra, rb in zip(a.per_step, b.per_step):
        assert torch.equal(ra.argmax, rb.argmax)
        assert ra.num_accepted == rb.num_accepted


# --- trajectory comparison harness (#47468) ------------------------------------


def _peaked_traj(target=5, seed=1):
    batch, length, vocab = 1, 8, 32
    logits = _peaked_logits(batch, length, vocab, target)
    init = S.random_canvas((batch, length), vocab, generator=_gen(seed))
    return ref_denoise_block(lambda canvas, step: logits, init, _cfg(), vocab)


def _random_traj(seed):
    batch, length, vocab = 1, 12, 40

    def logits_fn(canvas, step):
        return torch.randn(batch, length, vocab, generator=_gen(seed * 1000 + step))

    init = S.random_canvas((batch, length), vocab, generator=_gen(seed))
    return ref_denoise_block(logits_fn, init, _cfg(), vocab)


def test_self_comparison_passes():
    traj = _peaked_traj()
    cmp = compare_trajectories(traj, traj)
    assert cmp.passed
    assert cmp.min_argmax_agreement == 1.0
    assert cmp.committed_match == 1.0
    assert cmp.entropy_trajectory_pcc == pytest.approx(1.0)
    # Decision-level fields (#47468): every per-step diff is perfect on self-compare.
    assert cmp.min_sampled_agreement == 1.0  # Gumbel-sampled ids
    assert cmp.min_accept_iou == 1.0  # accept-mask IoU
    assert cmp.min_canvas_agreement == 1.0  # renoised canvas
    assert cmp.min_entropy_pcc == pytest.approx(1.0)  # per-token entropy PCC


def test_entropy_abs_gate_catches_affine_error_pcc_misses():
    """PCC is invariant to a constant offset/scale, so a systematic entropy error
    (wrong log base / missing temperature) would pass PCC≈1 — the absolute gate
    must still catch it (finding #3). Only the entropy abs-err should fail here."""
    ref = _random_traj(seed=7)
    # candidate identical to ref on every decision, but each per-token entropy is offset by +0.5
    shifted_steps = [r._replace(entropy=r.entropy + 0.5) for r in ref.per_step]
    cand = ref._replace(per_step=shifted_steps)

    cmp = compare_trajectories(ref, cand)
    assert min(cmp.per_step_entropy_pcc) > 0.99  # PCC blind to the constant offset
    assert cmp.max_entropy_abs_err >= 0.49  # but the absolute gate sees it
    assert not cmp.passed  # so the comparison fails
    # every other decision class still matches (the failure is isolated to entropy magnitude)
    assert cmp.min_argmax_agreement == 1.0 and cmp.min_canvas_agreement == 1.0 and cmp.min_accept_iou == 1.0


def test_constant_mean_entropy_offset_fails_trajectory_pcc():
    ref = _peaked_traj()
    shifted_steps = [r._replace(entropy_mean=r.entropy_mean + 0.5) for r in ref.per_step]
    cand = ref._replace(per_step=shifted_steps)

    cmp = compare_trajectories(ref, cand)
    assert cmp.entropy_trajectory_pcc == 0.0
    assert not cmp.passed
    assert cmp.max_entropy_abs_err == 0.0
    assert cmp.min_argmax_agreement == 1.0


def test_decision_level_fields_distinguish_drifted_trajectories():
    """Distinct trajectories must fail on EVERY decision class — sampled, accept,
    canvas, per-token entropy — not just the clean argmax."""
    ref = _random_traj(seed=11)
    cand = _random_traj(seed=42)  # different logits AND different RNG -> different decisions
    cmp = compare_trajectories(ref, cand)
    assert not cmp.passed
    assert cmp.min_sampled_agreement < 1.0
    assert cmp.min_accept_iou < 1.0
    assert cmp.min_canvas_agreement < 1.0
    # entropy is a real-valued vector → some PCC is plausible by chance, but it should be far from 1.0
    # for genuinely independent random logits. Don't assert a hard bound on it here — the harness's
    # min_per_step_entropy_pcc threshold (0.99) catches drift in real use.


def test_assert_trajectory_matches_raises_on_mismatch(expect_error):
    ref = _random_traj(seed=3)
    cand = _random_traj(seed=88)
    with expect_error(AssertionError):
        assert_trajectory_matches(ref, cand)


# --- variance-gated entropy fidelity (#48291) ----------------------------------
# Raw per-step entropy PCC is ill-conditioned once a denoise step converges: the
# per-token entropy profile goes near-constant, its variance -> 0, and PCC is then
# dominated by rounding noise even when the absolute entropy error is negligible.
# sound_entropy_step_fidelity gates PCC only where the reference profile carries
# structure and falls back to an absolute tolerance where it does not.


def test_sound_entropy_passes_converged_step_where_raw_pcc_fails():
    # Near-constant entropy profile (converged step) + negligible absolute drift.
    torch.manual_seed(0)
    ref = 0.01 + 0.004 * torch.rand(256)  # std well below min_std=0.15
    cand = ref + 0.02 * (torch.rand(256) - 0.5)  # tiny bf16-scale rounding noise
    raw = _pearson(ref, cand)
    verdict = sound_entropy_step_fidelity(ref, cand)
    assert raw < 0.95, f"expected ill-conditioned raw PCC on a flat profile, got {raw}"
    assert verdict.mode == "abs" and verdict.passed
    assert verdict.max_abs < 0.5


def test_sound_entropy_fails_converged_step_with_real_divergence():
    # Near-constant profile but a genuinely large entropy error -> abs branch fails.
    ref = torch.full((256,), 0.01)
    cand = ref.clone()
    cand[10] += 2.0
    verdict = sound_entropy_step_fidelity(ref, cand)
    assert verdict.mode == "abs" and not verdict.passed


def test_sound_entropy_uses_pcc_on_structured_step():
    # High-variance early-step profile: correlated -> pass via the PCC branch.
    torch.manual_seed(1)
    ref = torch.rand(256) * 2.0  # std well above min_std
    cand = ref + 0.01 * (torch.rand(256) - 0.5)
    verdict = sound_entropy_step_fidelity(ref, cand)
    assert verdict.mode == "pcc" and verdict.passed and verdict.pcc >= 0.95


def test_sound_entropy_fails_structured_step_when_decorrelated():
    torch.manual_seed(2)
    ref = torch.rand(256) * 2.0
    cand = torch.rand(256) * 2.0  # unrelated structured profile
    verdict = sound_entropy_step_fidelity(ref, cand)
    assert verdict.mode == "pcc" and not verdict.passed


def test_sound_entropy_pcc_branch_catches_affine_offset():
    # Structured, PERFECTLY correlated, but a systematic +1.5-nat offset (wrong log
    # base / missing temperature). PCC ~= 1.0, but the absolute guard must fail it
    # on the PCC branch too (affine blindness fix).
    torch.manual_seed(3)
    ref = torch.rand(256) * 2.0
    cand = ref + 1.5
    assert _pearson(ref, cand) > 0.999
    verdict = sound_entropy_step_fidelity(ref, cand)
    assert verdict.mode == "pcc" and not verdict.passed and verdict.max_abs > 0.5


# --- halt-trace telemetry (#48291) ---------------------------------------------
# The traced controller computes ``(steps_run, mean_entropy, mismatch)`` every denoise step and
# used to keep only the ``halted`` boolean. That boolean cannot distinguish the three ways a
# block can burn all 48 steps, and they need opposite fixes:
#
# * the entropy floors structurally above the bar (content never converged),
# * the entropy misses the bar by a numerical hair,
# * the entropy clears the bar but the argmax never stops moving.

THRESHOLD = 0.005


def _trace(pairs):
    """Build a ``last_halt_trace`` from ``[(mean_entropy, mismatch), ...]`` in step order."""
    return [(index + 1, entropy, mismatch) for index, (entropy, mismatch) in enumerate(pairs)]


def test_halted_block_reports_no_blocking_gate():
    summary = _summarize_halt_trace(_trace([(0.9, 4.0), (0.004, 0.0)]), threshold=THRESHOLD)
    assert summary["halt_blocking_gate"] == "none"
    assert summary["halt_steps_both_gates"] == 1


def test_structural_entropy_floor_is_named_and_scaled():
    """The 0.14-0.51 nats regime: argmax goes stable, entropy is 30-100x the bar."""
    summary = _summarize_halt_trace(_trace([(0.6, 9.0)] + [(0.5, 0.0)] * 47), threshold=THRESHOLD)
    assert summary["halt_blocking_gate"] == "entropy"
    assert summary["halt_steps_mismatch_zero"] == 47
    assert summary["halt_steps_entropy_under_threshold"] == 0
    # The ratio is what separates this from a near-miss; 0.5 / 0.005 = 100x.
    assert summary["halt_entropy_floor_ratio"] == pytest.approx(100.0, rel=1e-3)
    assert summary["halt_entropy_margin_final"] > 0.4


def test_numerical_near_miss_is_distinguishable_from_the_floor():
    """Same blocking gate as the floor case, but the margin/ratio must expose the difference."""
    summary = _summarize_halt_trace(_trace([(0.6, 9.0)] + [(0.0051, 0.0)] * 47), threshold=THRESHOLD)
    assert summary["halt_blocking_gate"] == "entropy"
    assert summary["halt_entropy_floor_ratio"] == pytest.approx(1.02, rel=1e-2)
    assert 0.0 < summary["halt_entropy_margin_final"] < 0.001


def test_oscillating_argmax_is_attributed_to_the_mismatch_gate():
    summary = _summarize_halt_trace(_trace([(0.6, 9.0)] + [(0.001, 3.0)] * 47), threshold=THRESHOLD)
    assert summary["halt_blocking_gate"] == "mismatch"
    assert summary["halt_steps_entropy_under_threshold"] == 47
    assert summary["halt_steps_mismatch_zero"] == 0
    assert summary["halt_mismatch_final"] == 3.0


def test_gates_satisfied_on_different_steps_is_not_a_halt():
    """Both gates pass, never on the same step -- must not read as ``none`` (that would imply
    eval_halt should have fired) nor as a single-gate failure."""
    summary = _summarize_halt_trace(_trace([(0.5, 9.0), (0.001, 4.0), (0.4, 0.0), (0.001, 2.0)]), threshold=THRESHOLD)
    assert summary["halt_blocking_gate"] == "never_simultaneous"
    assert summary["halt_steps_both_gates"] == 0


def test_step_zero_pass_is_not_counted_as_eligible():
    """eval_halt requires ``step >= 1``, so a step-0-only pass is NOT a missed halt."""
    summary = _summarize_halt_trace(_trace([(0.001, 0.0), (0.5, 6.0), (0.5, 6.0)]), threshold=THRESHOLD)
    assert summary["halt_eligible_steps"] == 2
    assert summary["halt_steps_both_gates"] == 0
    assert summary["halt_blocking_gate"] == "both"


def test_empty_trace_is_reported_not_crashed():
    summary = _summarize_halt_trace([], threshold=THRESHOLD)
    assert summary["halt_trace_steps"] == 0
    assert summary["halt_blocking_gate"] == "none"


def test_summary_is_json_serializable_for_the_metric_channel():
    summary = _summarize_halt_trace(_trace([(0.5, 7.0)] * 3), threshold=THRESHOLD)
    assert json.loads(json.dumps(summary))["halt_trace_steps"] == 3


# --- ttnn controller tensor ownership ------------------------------------------


class _FakeTensor:
    def __init__(self, name):
        self.name = name
        self.deallocated = False

    def deallocate(self, force):
        assert force is True
        assert not self.deallocated, self.name
        self.deallocated = True


def _fake_step_result(canvas):
    return DL.TtDenoiseStepResult(
        canvas=canvas,
        accept_mask=_FakeTensor("accept"),
        entropy=_FakeTensor("entropy"),
        sampled=_FakeTensor("sampled"),
        argmax=_FakeTensor("argmax"),
    )


def _patch_host_readbacks(monkeypatch):
    monkeypatch.setattr(DL, "_ids_to_torch", lambda tensor: torch.ones(1, 1, dtype=torch.long))
    monkeypatch.setattr(DL, "_entropy_to_torch", lambda tensor: torch.zeros(1, 1, dtype=torch.float32))
    monkeypatch.setattr(DL, "_accept_to_torch", lambda tensor: torch.ones(1, 1, dtype=torch.bool))


def _halt_on_first_step_cfg():
    return DiffusionConfig(max_denoise_steps=1, entropy_stop_threshold=1.0, stable_steps_to_halt=0)


def test_to_host_torch_uses_first_device_tensor_for_mesh_readback(monkeypatch):
    class _FakeMeshDevice:
        def get_num_devices(self):
            return 4

    class _FakeMeshTensor:
        def device(self):
            return _FakeMeshDevice()

    mesh_tensor = _FakeMeshTensor()
    shard0 = _FakeTensor("shard0")
    calls = []

    class _FakeTtnn:
        @staticmethod
        def get_device_tensors(tensor):
            assert tensor is mesh_tensor
            return [shard0]

        @staticmethod
        def to_torch(tensor):
            calls.append(tensor.name)
            assert tensor is shard0
            return torch.tensor([7])

    monkeypatch.setattr(DL, "ttnn", _FakeTtnn)

    assert torch.equal(DL._to_host_torch(mesh_tensor), torch.tensor([7]))
    assert calls == ["shard0"]


@pytest.mark.parametrize(
    "gumbel_noise_fn,noise_tokens_fn",
    [
        (None, lambda step: _FakeTensor("noise")),
        (lambda step: _FakeTensor("gumbel"), None),
        (None, None),
    ],
)
def test_denoise_block_requires_injected_noise_hooks(gumbel_noise_fn, noise_tokens_fn, expect_error):
    with expect_error(ValueError, match="requires injected gumbel_noise_fn and noise_tokens_fn"):
        DL.denoise_block(
            lambda canvas, step: _FakeTensor("logits"),
            _FakeTensor("init-canvas"),
            DiffusionConfig(max_denoise_steps=1),
            gumbel_noise_fn=gumbel_noise_fn,
            noise_tokens_fn=noise_tokens_fn,
        )


def test_denoise_block_deallocates_consumed_injected_noise(monkeypatch):
    gumbel_noise = _FakeTensor("gumbel")
    noise_tokens = _FakeTensor("noise")
    logits = _FakeTensor("logits")
    init_canvas = _FakeTensor("init-canvas")
    result = _fake_step_result(_FakeTensor("next-canvas"))

    def fake_denoise_step(
        logits,
        *,
        temperature,
        entropy_budget,
        gumbel_noise,
        noise_tokens,
    ):
        assert logits.name == "logits"
        assert gumbel_noise is not None and not gumbel_noise.deallocated
        assert noise_tokens is not None and not noise_tokens.deallocated
        return result

    monkeypatch.setattr(DL, "denoise_step", fake_denoise_step)
    _patch_host_readbacks(monkeypatch)

    trajectory = DL.denoise_block(
        lambda canvas, step: logits,
        init_canvas,
        _halt_on_first_step_cfg(),
        gumbel_noise_fn=lambda step: gumbel_noise,
        noise_tokens_fn=lambda step: noise_tokens,
    )

    assert trajectory.halted
    assert gumbel_noise.deallocated
    assert noise_tokens.deallocated
    assert logits.deallocated


def test_denoise_block_allows_argmax_sampling_without_gumbel_tensor(monkeypatch):
    noise_tokens = _FakeTensor("noise")
    logits = _FakeTensor("logits")
    init_canvas = _FakeTensor("init-canvas")
    result = _fake_step_result(_FakeTensor("next-canvas"))

    def fake_denoise_step(
        logits,
        *,
        temperature,
        entropy_budget,
        gumbel_noise,
        noise_tokens,
    ):
        assert logits.name == "logits"
        assert gumbel_noise is None
        assert noise_tokens is not None and not noise_tokens.deallocated
        return result

    monkeypatch.setattr(DL, "denoise_step", fake_denoise_step)
    _patch_host_readbacks(monkeypatch)

    trajectory = DL.denoise_block(
        lambda canvas, step: logits,
        init_canvas,
        _halt_on_first_step_cfg(),
        gumbel_noise_fn=lambda step: None,
        noise_tokens_fn=lambda step: noise_tokens,
    )

    assert trajectory.halted
    assert noise_tokens.deallocated
    assert logits.deallocated


def test_denoise_block_allows_descriptor_gumbel_without_deallocate(monkeypatch):
    descriptor = object()
    noise_tokens = _FakeTensor("noise")
    logits = _FakeTensor("logits")
    init_canvas = _FakeTensor("init-canvas")
    result = _fake_step_result(_FakeTensor("next-canvas"))

    def fake_denoise_step(
        logits,
        *,
        temperature,
        entropy_budget,
        gumbel_noise,
        noise_tokens,
    ):
        assert logits.name == "logits"
        assert gumbel_noise is descriptor
        assert noise_tokens is not None and not noise_tokens.deallocated
        return result

    monkeypatch.setattr(DL, "denoise_step", fake_denoise_step)
    _patch_host_readbacks(monkeypatch)

    trajectory = DL.denoise_block(
        lambda canvas, step: logits,
        init_canvas,
        _halt_on_first_step_cfg(),
        gumbel_noise_fn=lambda step: descriptor,
        noise_tokens_fn=lambda step: noise_tokens,
    )

    assert trajectory.halted
    assert noise_tokens.deallocated
    assert logits.deallocated


def test_denoise_block_leaves_callback_owned_logits_for_self_conditioning(monkeypatch):
    gumbel_noise = _FakeTensor("gumbel")
    noise_tokens = _FakeTensor("noise")
    logits = _FakeTensor("logits")
    init_canvas = _FakeTensor("init-canvas")
    result = _fake_step_result(_FakeTensor("next-canvas"))

    class _StatefulLogits:
        prev_logits = None

        def __call__(self, canvas, step):
            del canvas, step
            self.prev_logits = logits
            return logits

        def owns_logits(self, value):
            return self.prev_logits is value

        def reset(self):
            self.prev_logits.deallocate(True)
            self.prev_logits = None

    logits_fn = _StatefulLogits()

    monkeypatch.setattr(DL, "denoise_step", lambda *args, **kwargs: result)
    _patch_host_readbacks(monkeypatch)

    trajectory = DL.denoise_block(
        logits_fn,
        init_canvas,
        _halt_on_first_step_cfg(),
        gumbel_noise_fn=lambda step: gumbel_noise,
        noise_tokens_fn=lambda step: noise_tokens,
    )

    assert trajectory.halted
    assert logits.deallocated


# --- assembled device denoise step / block (#47463) ----------------------------

requires_device = pytest.mark.skipif(
    os.environ.get("DG_RUN_DEVICE") != "1",
    reason="set DG_RUN_DEVICE=1 to run on a Tenstorrent device",
)


def _to_device(device, value, *, dtype=ttnn.float32):
    return ttnn.from_torch(value, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)


def _structured_logits(length: int, vocab_size: int):
    """Logits with stable argmax and well-separated entropy ordering."""
    logits = torch.full((1, length, vocab_size), -4.0, dtype=torch.float32)
    token_ids = torch.arange(length) % vocab_size
    sharpness = torch.linspace(0.25, 2.0, length)
    logits[0, torch.arange(length), token_ids] = sharpness
    logits += torch.randn_like(logits) * 1.0e-3
    return logits


def _budget_for_accept_count(entropy: torch.Tensor, count: int):
    sorted_entropy = torch.sort(entropy, dim=-1).values
    exclusive = torch.cumsum(sorted_entropy, dim=-1) - sorted_entropy
    return float((exclusive[0, count - 1] + exclusive[0, count]) / 2)


class _ResettableStaticLogits:
    def __init__(self, logits):
        self.logits = logits
        self.reset_calls = 0

    def __call__(self, canvas, step):
        return self.logits

    def reset(self):
        self.reset_calls += 1
        if self.logits is not None:
            self.logits.deallocate(True)
            self.logits = None


@requires_device
@pytest.mark.use_module_device
def test_single_denoise_step_matches_reference(device):
    torch.manual_seed(11)
    length = 256
    vocab_size = 256
    max_steps = 48
    step = 3
    temperature = temperature_at_step(step, max_steps, 0.8, 0.4)

    logits = _structured_logits(length, vocab_size)
    gumbel_noise = torch.zeros_like(logits)
    noise_tokens = torch.randint(0, vocab_size, (1, length), dtype=torch.long)
    ref_entropy = S.token_entropy(logits, temperature=temperature)
    accept_count = 96
    budget = _budget_for_accept_count(ref_entropy, accept_count)
    ref = S.denoise_step(
        logits,
        temperature=temperature,
        entropy_budget=budget,
        vocab_size=vocab_size,
        sampler=S.SAMPLER_GUMBEL,
        gumbel_noise=gumbel_noise,
        noise_tokens=noise_tokens,
        min_accept=0,
    )

    tt = denoise_step(
        _to_device(device, logits.unsqueeze(1)),
        temperature=temperature,
        entropy_budget=budget,
        gumbel_noise=_to_device(device, gumbel_noise.unsqueeze(1)),
        noise_tokens=_to_device(device, noise_tokens.view(1, 1, length, 1).to(torch.int32), dtype=ttnn.uint32),
    )

    out_entropy = ttnn.to_torch(tt.entropy).squeeze(1).squeeze(-1).float()
    out_accept = ttnn.to_torch(tt.accept_mask).squeeze(1).squeeze(1) > 0.5
    out_sampled = ttnn.to_torch(tt.sampled).squeeze(1).squeeze(-1).to(torch.long)
    out_argmax = ttnn.to_torch(tt.argmax).squeeze(1).squeeze(-1).to(torch.long)
    out_canvas = ttnn.to_torch(tt.canvas).squeeze(1).squeeze(-1).to(torch.long)

    passing, message = assert_with_pcc(ref.entropy.float(), out_entropy.float(), 0.99)
    assert passing, message
    assert torch.equal(out_accept, ref.accept_mask)
    assert torch.equal(out_sampled, ref.sampled)
    assert torch.equal(out_argmax, ref.argmax)
    assert torch.equal(out_canvas, ref.canvas)
    assert int(out_accept.sum()) == accept_count


@requires_device
@pytest.mark.use_module_device
def test_uint32_renoise_preserves_full_vocab_token_ids(device):
    sampled = torch.tensor([0, 1, 65535, 65536, 131071, 131072, 262143, 17], dtype=torch.int32).view(1, 1, 8, 1)
    noise_tokens = torch.tensor([262143, 131072, 131071, 65536, 65535, 1, 0, 2048], dtype=torch.int32).view(1, 1, 8, 1)
    accept = torch.tensor([1, 0, 1, 0, 1, 0, 1, 0], dtype=torch.float32).view(1, 1, 8, 1)
    ref = torch.where(accept.bool(), sampled, noise_tokens).view(1, 8).to(torch.long)

    out = renoise(
        _to_device(device, accept, dtype=ttnn.bfloat16),
        _to_device(device, sampled, dtype=ttnn.uint32),
        _to_device(device, noise_tokens, dtype=ttnn.uint32),
    )

    assert torch.equal(ttnn.to_torch(out).squeeze(1).squeeze(-1).to(torch.long), ref)


@requires_device
@pytest.mark.use_module_device
def test_multi_step_denoise_control_flow_smoke_matches_reference(device):
    """Synthetic controller smoke; real canvas->W2 logits cycling is covered in the integration suite."""
    torch.manual_seed(17)
    batch = 1
    length = 256
    vocab_size = 256
    max_steps = 4

    logits = _structured_logits(length, vocab_size)
    step0_temperature = temperature_at_step(0, max_steps, 0.8, 0.4)
    ref_entropy = S.token_entropy(logits, temperature=step0_temperature)
    budget = _budget_for_accept_count(ref_entropy, 96)
    cfg = DiffusionConfig(
        max_denoise_steps=max_steps,
        entropy_stop_threshold=10.0,
        stable_steps_to_halt=1,
        entropy_budget=budget,
    )
    init_canvas = torch.randint(0, vocab_size, (batch, length), dtype=torch.long)
    gumbel_noise = [torch.zeros_like(logits) for _ in range(max_steps)]
    noise_tokens = [torch.randint(0, vocab_size, (batch, length), dtype=torch.long) for _ in range(max_steps)]

    ref = ref_denoise_block(
        lambda canvas, step: logits,
        init_canvas,
        cfg,
        vocab_size,
        gumbel_noise_fn=lambda step: gumbel_noise[step],
        noise_tokens_fn=lambda step: noise_tokens[step],
    )

    tt_logits = _ResettableStaticLogits(_to_device(device, logits.unsqueeze(1)))
    tt_gumbel_noise = [_to_device(device, noise.unsqueeze(1)) for noise in gumbel_noise]
    tt_noise_tokens = [
        _to_device(device, noise.view(batch, 1, length, 1).to(torch.int32), dtype=ttnn.uint32) for noise in noise_tokens
    ]
    tt = denoise_block(
        tt_logits,
        _to_device(device, init_canvas.view(batch, 1, length, 1).to(torch.int32), dtype=ttnn.uint32),
        cfg,
        gumbel_noise_fn=lambda step: tt_gumbel_noise[step],
        noise_tokens_fn=lambda step: tt_noise_tokens[step],
    )

    comparison = compare_trajectories(ref, tt, max_entropy_abs_err_threshold=0.2)
    accept_flips = sum(int((ra.accept_mask != rb.accept_mask).sum()) for ra, rb in zip(ref.per_step, tt.per_step))
    assert comparison.passed, comparison
    assert ref.halted and tt.halted
    assert ref.num_steps == tt.num_steps == 2
    assert accept_flips == 0
    assert tt_logits.reset_calls == 1
