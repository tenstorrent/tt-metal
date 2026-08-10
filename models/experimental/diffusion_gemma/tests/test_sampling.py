# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Sampling tests: the torch reference, the ttnn ops, the vLLM params seam, and the
device Gumbel/entropy/argmax gates (#47463/#47468/#47472/#48291)."""

import os
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

import ttnn
from models.common.utility_functions import comp_pcc
from models.experimental.diffusion_gemma.reference import sampling as S
from models.experimental.diffusion_gemma.tt import sampling as TS
from models.experimental.diffusion_gemma.tt.denoise_loop import entropy_budget_accept
from models.experimental.diffusion_gemma.tt.sampling import argmax_last_dim
from models.experimental.diffusion_gemma.tt.sampling_params import (
    MODEL_CAPABILITIES,
    canvas_sample_from_params,
    canvas_sampling_config_from_params,
)
from tests.ttnn.utils_for_testing import assert_with_pcc

requires_device = pytest.mark.skipif(
    os.environ.get("DG_RUN_DEVICE") != "1",
    reason="set DG_RUN_DEVICE=1 to run on a Tenstorrent device",
)
requires_device_sfpi = pytest.mark.skipif(
    os.environ.get("DG_RUN_DEVICE") != "1",
    reason="set DG_RUN_DEVICE=1 to run on a Tenstorrent device (needs sfpi >= 7.60.0)",
)

# Canvas/row count shared by the device op harnesses below.
_SEQ = 256


def _gen(seed=0):
    g = torch.Generator()
    g.manual_seed(seed)
    return g


# --- torch reference sampling -------------------------------------------------
# Pure torch — no checkpoint / ttnn / hardware. These pin the exact semantics the
# device path must match, especially the entropy-budget acceptance scatter-back.


def test_temperature_schedule_endpoints_and_monotone():
    # HF reversed-step formula: step 0 == t_max (0.8); last step == t_min + (t_max-t_min)/N,
    # NOT exactly t_min (cur_step bottoms out at 1, not 0).
    assert S.temperature_at_step(0, 48, 0.8, 0.4) == pytest.approx(0.8)
    assert S.temperature_at_step(47, 48, 0.8, 0.4) == pytest.approx(0.4 + 0.4 * (1 / 48))
    assert S.temperature_at_step(24, 48, 0.8, 0.4) == pytest.approx(0.6)  # cur_step=24 -> midpoint
    ts = [S.temperature_at_step(i, 48, 0.8, 0.4) for i in range(48)]
    assert all(ts[i] >= ts[i + 1] for i in range(47))  # monotonically decreasing
    assert 0.4 < ts[24] < 0.8


def test_token_entropy_uniform_and_peaked():
    vocab = 64
    uniform = torch.zeros(1, 4, vocab)  # equal logits -> uniform softmax
    h = S.token_entropy(uniform)
    expected = torch.log(torch.tensor(float(vocab)))
    assert torch.allclose(h, torch.full_like(h, expected), atol=1e-5)

    peaked = torch.full((1, 4, vocab), -1e4)
    peaked[..., 0] = 1e4
    assert torch.all(S.token_entropy(peaked) < 1e-3)


def test_gumbel_max_zero_noise_is_argmax():
    logits = torch.randn(2, 8, 50, generator=_gen(1))
    out = S.gumbel_max_sample(logits, temperature=0.7, noise=torch.zeros_like(logits))
    assert torch.equal(out, logits.argmax(dim=-1))


def test_entropy_budget_accept_extremes_and_monotone():
    entropy = torch.tensor([[0.5, 0.1, 0.9, 0.2, 0.7]])
    assert S.entropy_budget_accept(entropy, budget=1e9).all()  # huge budget -> all
    assert not S.entropy_budget_accept(entropy, budget=-1e-6).any()  # HF default: no min accept

    acc = S.entropy_budget_accept(entropy, budget=0.0, min_accept=1)
    assert acc.sum().item() == 1 and bool(acc[0, 1])  # only the lowest-entropy pos (idx 1)

    prev = torch.zeros_like(entropy, dtype=torch.bool)
    for b in [0.0, 0.15, 0.35, 0.8, 1.5, 3.0]:  # increasing budget never un-accepts
        cur = S.entropy_budget_accept(entropy, budget=b, min_accept=1)
        assert torch.equal(cur | prev, cur)
        prev = cur


def test_entropy_budget_accept_exclusive_prefix_cutoff():
    # ascending order by value: idx 1(0.1), 3(0.2), 0(0.5), 4(0.7), 2(0.9)
    # EXCLUSIVE prefix (sum of strictly-more-confident) per sorted pos:
    #   idx1: 0.0 | idx3: 0.1 | idx0: 0.3 | idx4: 0.8 | idx2: 1.5
    entropy = torch.tensor([[0.5, 0.1, 0.9, 0.2, 0.7]])
    # budget 0.35: accept idx1(0.0), idx3(0.1), idx0(0.3) (<=0.35); reject idx4(0.8), idx2(1.5)
    acc = S.entropy_budget_accept(entropy, budget=0.35, min_accept=1)
    assert torch.equal(acc, torch.tensor([[True, True, False, True, False]]))


def test_acceptance_scatter_back_inverse_permutation():
    # The scatter-back the device path must replicate (#47463): accept decisions
    # taken in sorted-by-confidence order map to ORIGINAL canvas positions. Mirror
    # HF EntropyBoundSampler.accept_canvas exactly (exclusive-prefix cutoff).
    entropy = torch.rand(3, 17, generator=_gen(4))
    budget = 0.9
    acc = S.entropy_budget_accept(entropy, budget=budget, min_accept=1)

    sorted_e, idx = torch.sort(entropy, dim=-1)
    cum = torch.cumsum(sorted_e, dim=-1)
    accept_sorted = (cum - sorted_e) <= budget  # exclusive prefix
    ref = torch.zeros_like(entropy, dtype=torch.bool)
    for r in range(entropy.shape[0]):
        for c in range(entropy.shape[1]):
            ref[r, idx[r, c]] = accept_sorted[r, c]
    assert torch.equal(acc, ref)


def test_sample_canvas_multinomial_matches_argmax_when_peaked():
    # multinomial(softmax) of near-one-hot logits returns the peak token id.
    vocab = 50
    peaked = torch.full((2, 8, vocab), -1e4)
    peaked[..., 13] = 1e4
    out = S.sample_canvas(peaked, temperature=0.7, generator=_gen(11))
    assert out.shape == (2, 8)
    assert torch.equal(out, torch.full((2, 8), 13))
    # in-range for arbitrary logits
    rand = S.sample_canvas(torch.randn(1, 5, vocab, generator=_gen(12)), generator=_gen(13))
    assert int(rand.min()) >= 0 and int(rand.max()) < vocab


def test_renoise_keeps_accepted_replaces_rejected():
    vocab = 100
    tokens = torch.arange(5).view(1, 5)
    accept = torch.tensor([[True, False, True, False, True]])
    out = S.renoise(tokens, accept, vocab, noise_tokens=torch.full((1, 5), 99))
    assert torch.equal(out, torch.tensor([[0, 99, 2, 99, 4]]))

    out2 = S.renoise(tokens, accept, vocab, generator=_gen(5))  # random renoise in range
    assert int(out2.min()) >= 0 and int(out2.max()) < vocab
    assert out2[0, 0] == 0 and out2[0, 2] == 2 and out2[0, 4] == 4


def test_decision_dtype_red_lines_bf16_safe():
    """#47468 acceptance red lines for the diffusion DECISIONS under dtype.

    bf16 is the floor for the decision-critical ops: the entropy-budget accept mask
    and the clean-argmax commit must barely move at bf16, while bfp8 is NOT safe for
    them (entropy PCC ~0.74 measured on device -> accept flips; ttnn.argmax rejects
    bfp8). These bars are the harness's value: a regression that pushes decisions
    into bfp8, or otherwise perturbs them, trips here.
    """

    def bf16(x):
        return x.to(torch.bfloat16).float()

    # accept-mask flip rate at bf16 vs fp32, across budgets + varied entropy
    acc_flips = acc_tot = 0
    for seed in range(8):
        logits = torch.randn(1, 256, 2048, generator=_gen(seed)) * torch.linspace(0.3, 5, 256).view(1, 256, 1)
        for bound in [0.05, 0.1, 0.5]:
            ref = S.entropy_budget_accept(S.token_entropy(logits), bound, min_accept=0)
            got = S.entropy_budget_accept(S.token_entropy(bf16(logits)), bound, min_accept=0)
            acc_flips += int((ref != got).sum())
            acc_tot += ref.numel()
    acc_rate = acc_flips / acc_tot
    assert acc_rate <= 0.005, f"bf16 accept-mask flip rate {acc_rate:.4%} > 0.5% red line (measured ~0.13%)"

    # clean-argmax commit flip rate at bf16 vs fp32 (near-max ties may flip; bound generously)
    arg_flips = arg_tot = 0
    for seed in range(8):
        logits = torch.randn(1, 256, 2048, generator=_gen(100 + seed))
        arg_flips += int((logits.argmax(-1) != bf16(logits).argmax(-1)).sum())
        arg_tot += logits.shape[1]
    arg_rate = arg_flips / arg_tot
    assert arg_rate <= 0.03, f"bf16 commit-argmax flip rate {arg_rate:.4%} > 3% red line (measured ~1.3%)"


def test_random_canvas_in_range():
    canvas = S.random_canvas((2, 256), 262144, generator=_gen(7))
    assert canvas.shape == (2, 256)
    assert int(canvas.min()) >= 0 and int(canvas.max()) < 262144


def test_denoise_step_shapes_commit_argmax():
    batch, length, vocab = 2, 16, 64
    logits = torch.randn(batch, length, vocab, generator=_gen(6))
    res = S.denoise_step(
        logits,
        temperature=0.6,
        entropy_budget=0.5,
        vocab_size=vocab,
        gumbel_noise=torch.zeros_like(logits),  # -> sampled == argmax
    )
    assert res.canvas.shape == (batch, length)
    assert res.accept_mask.shape == (batch, length)
    assert res.entropy.shape == (batch, length)
    assert torch.equal(res.argmax, logits.argmax(dim=-1))  # commit value = clean argmax
    assert torch.equal(res.sampled, logits.argmax(dim=-1))  # zero noise
    # accepted canvas positions hold the (clean argmax) sample
    assert torch.equal(res.canvas[res.accept_mask], res.argmax[res.accept_mask])


# --- ttnn gumbel-noise helpers ------------------------------------------------


class _FakeDevice:
    shape = (1, 4)

    def __init__(self, num_devices):
        self._num_devices = num_devices

    def get_num_devices(self):
        return self._num_devices


class _FakeTensor:
    def __init__(self, name):
        self.name = name
        self.deallocated = False

    def deallocate(self, force):
        self.deallocated = force


_GUMBEL_NOISE_HELPERS = (
    TS.sample_gumbel_noise,
    TS.sample_gumbel_noise_with_permuted_vocab,
)
_VOCAB_AXIS_HELPERS = _GUMBEL_NOISE_HELPERS[1:]


def test_rand_mesh_mapper_replicates_over_flattened_mesh(monkeypatch):
    calls = {}

    class _FakeTtnn:
        class PlacementReplicate:
            pass

        class MeshShape:
            def __init__(self, shape):
                self.shape = shape

        class MeshMapperConfig:
            def __init__(self, *, placements, mesh_shape_override=None):
                calls["placements"] = placements
                calls["mesh_shape_override"] = mesh_shape_override

    monkeypatch.setattr(TS, "ttnn", _FakeTtnn)

    mapper = TS._rand_mesh_mapper(_FakeDevice(num_devices=4))

    assert isinstance(mapper, _FakeTtnn.MeshMapperConfig)
    assert len(calls["placements"]) == 1
    assert isinstance(calls["placements"][0], _FakeTtnn.PlacementReplicate)
    assert calls["mesh_shape_override"].shape == [4]


def test_rand_mesh_mapper_single_device_returns_none():
    assert TS._rand_mesh_mapper(_FakeDevice(num_devices=1)) is None


@pytest.mark.parametrize(
    "helpers,shape,seed,match",
    [
        pytest.param(_GUMBEL_NOISE_HELPERS, (1, 1, 32, 32), 0, "positive nonzero", id="seed-zero"),
        pytest.param(_GUMBEL_NOISE_HELPERS, (1, 1, 32, 32), -3, "positive nonzero", id="seed-negative"),
        pytest.param(_GUMBEL_NOISE_HELPERS, (), 47472, "shape", id="shape-empty"),
        pytest.param(_GUMBEL_NOISE_HELPERS, (1, 0, 32), 47472, "shape", id="shape-zero-dim"),
        pytest.param(_GUMBEL_NOISE_HELPERS, (1, -2, 32), 47472, "shape", id="shape-negative-dim"),
        pytest.param(_VOCAB_AXIS_HELPERS, (32,), 47472, "sample axis and a vocab axis", id="vocab-axis-only-vocab"),
        pytest.param(_VOCAB_AXIS_HELPERS, (1,), 47472, "sample axis and a vocab axis", id="vocab-axis-singleton"),
    ],
)
def test_gumbel_noise_helpers_reject_bad_arguments(helpers, shape, seed, match, expect_error):
    device = _FakeDevice(num_devices=1)

    for helper in helpers:
        with expect_error(ValueError, match=match):
            helper(shape, device=device, seed=seed)


def test_permuted_vocab_gumbel_noise_deallocates_pre_permute_tensor(monkeypatch):
    calls = {}
    raw = _FakeTensor("raw")
    permuted = _FakeTensor("permuted")
    reshaped = _FakeTensor("reshaped")

    class _FakeTtnn:
        TILE_LAYOUT = "tile"

        @staticmethod
        def rand(shape, **kwargs):
            calls["rand"] = (shape, kwargs)
            return raw

        @staticmethod
        def permute(tensor, order):
            calls["permute"] = (tensor, order)
            return permuted

        @staticmethod
        def reshape(tensor, shape):
            calls["reshape"] = (tensor, shape)
            return reshaped

    def fake_gumbel_from_uniform(tensor):
        calls["gumbel"] = tensor
        return "gumbel"

    monkeypatch.setattr(TS, "ttnn", _FakeTtnn)
    monkeypatch.setattr(TS, "_gumbel_from_uniform", fake_gumbel_from_uniform)

    out = TS.sample_gumbel_noise_with_permuted_vocab((2, 1, 4, 16), device="mesh", seed=47472, dtype="float32")

    assert out == "gumbel"
    # vocab (16) outermost, all non-vocab dims (2*1*4=8) collapsed into one tile-aligned axis;
    # no singleton axis lands in the tiled last-two dims (the 32x TILE-pad OOM fix).
    assert calls["rand"][0] == (16, 8)
    assert calls["permute"] == (raw, (1, 0))
    assert calls["reshape"] == (permuted, (2, 1, 4, 16))
    assert calls["gumbel"] is reshaped
    assert raw.deallocated is True
    assert permuted.deallocated is True
    assert reshaped.deallocated is False


# --- vLLM sampling-params seam -------------------------------------------------


@dataclass(frozen=True)
class DuckTypedTTSamplingParams:
    temperature: float | list[float]
    top_k: int | list[int]
    top_p: float | list[float]
    seed: int | list[int] | None = None


class _FakeLogits:
    shape = (2, 1, 4, 16)

    def device(self):
        return "mesh"


class _FakeNoise:
    def __init__(self, name):
        self.name = name
        self.deallocated = False

    def deallocate(self, force):
        self.deallocated = force


def test_canvas_sampling_params_defaults_and_capability():
    config = canvas_sampling_config_from_params(None, default_temperature=0.8, default_seed=47472)

    assert MODEL_CAPABILITIES["supports_sample_on_device"] is True
    assert config.temperature == pytest.approx(0.8)
    assert config.seed == 47472
    assert config.top_k is None
    assert config.top_p is None
    assert config.top_k_top_p_supported is False


def test_canvas_sampling_params_duck_type_vllm_fields():
    params = DuckTypedTTSamplingParams(
        temperature=[0.6, 0.7],
        top_k=[64, 32],
        top_p=[0.95, 0.9],
        seed=[1234, 5678],
    )

    config = canvas_sampling_config_from_params(params, default_temperature=0.8)

    assert config.temperature == pytest.approx(0.6)
    assert config.seed == 1234
    assert config.top_k == 64
    assert config.top_p == pytest.approx(0.95)
    assert config.top_k_top_p_supported is False


@pytest.mark.parametrize(
    "params,match",
    [
        pytest.param({"temperature": 0.0, "top_k": 1, "top_p": 1.0}, "temperature > 0", id="greedy-temperature"),
        pytest.param({"temperature": 0.8, "seed": 0}, "positive nonzero", id="seed-zero"),
        pytest.param({"temperature": 0.8, "seed": -1}, "positive nonzero", id="seed-negative"),
    ],
)
def test_canvas_sampling_params_rejects_bad_values(params, match, expect_error):
    with expect_error(ValueError, match=match):
        canvas_sampling_config_from_params(params, default_temperature=0.8)


@pytest.mark.parametrize(
    "params,kwargs,match",
    [
        pytest.param({"temperature": 0.8}, {}, "gumbel_noise or a sampling seed", id="no-noise-and-no-seed"),
    ],
)
def test_canvas_sample_from_params_rejects_bad_arguments(params, kwargs, match, expect_error):
    with expect_error(ValueError, match=match):
        canvas_sample_from_params(
            logits=None,
            sampling_params=params,
            default_temperature=0.8,
            **kwargs,
        )


def test_canvas_sample_from_params_defaults_to_vocab_innermost_rng(monkeypatch):
    """Default is the plain draw: for a (..., canvas, vocab) shape the permuted variant
    relocates the RNG degeneracy onto the canvas-position axis, which is what makes
    different positions collapse onto the same token."""
    calls = {}

    def fake_permuted_noise(shape, *, device, seed):
        calls["noise"] = (shape, device, seed)
        return "permuted-gumbel"

    def fake_canvas_sample(logits, temperature, gumbel_noise):
        calls["sample"] = (logits, temperature, gumbel_noise)
        return "samples"

    monkeypatch.setattr(TS, "sample_gumbel_noise", fake_permuted_noise)
    monkeypatch.setattr(TS, "canvas_sample", fake_canvas_sample)

    logits = _FakeLogits()
    out = canvas_sample_from_params(
        logits,
        {"temperature": 0.7, "seed": 47472},
        default_temperature=0.8,
    )

    assert out == "samples"
    assert calls["noise"] == (_FakeLogits.shape, "mesh", 47472)
    assert calls["sample"] == (logits, 0.7, "permuted-gumbel")


def test_canvas_sample_from_params_deallocates_generated_gumbel_noise(monkeypatch):
    calls = {}
    noise = _FakeNoise("permuted-gumbel")

    def fake_permuted_noise(shape, *, device, seed):
        calls["noise"] = (shape, device, seed)
        return noise

    def fake_canvas_sample(logits, temperature, gumbel_noise):
        calls["sample"] = (logits, temperature, gumbel_noise, gumbel_noise.deallocated)
        return "samples"

    monkeypatch.setattr(TS, "sample_gumbel_noise", fake_permuted_noise)
    monkeypatch.setattr(TS, "canvas_sample", fake_canvas_sample)

    logits = _FakeLogits()
    out = canvas_sample_from_params(
        logits,
        {"temperature": 0.7, "seed": 47472},
        default_temperature=0.8,
    )

    assert out == "samples"
    assert calls["noise"] == (_FakeLogits.shape, "mesh", 47472)
    assert calls["sample"] == (logits, 0.7, noise, False)
    assert noise.deallocated is True


def test_canvas_sample_from_params_deallocates_generated_gumbel_noise_on_failure(monkeypatch, expect_error):
    noise = _FakeNoise("permuted-gumbel")

    def fake_permuted_noise(shape, *, device, seed):
        return noise

    def fail_canvas_sample(logits, temperature, gumbel_noise):
        assert gumbel_noise is noise
        assert noise.deallocated is False
        raise RuntimeError("sampling failed")

    monkeypatch.setattr(TS, "sample_gumbel_noise", fake_permuted_noise)
    monkeypatch.setattr(TS, "canvas_sample", fail_canvas_sample)

    with expect_error(RuntimeError, match="sampling failed"):
        canvas_sample_from_params(
            _FakeLogits(),
            {"temperature": 0.7, "seed": 47472},
            default_temperature=0.8,
        )

    assert noise.deallocated is True


def test_canvas_sample_from_params_preserves_injected_gumbel_noise(monkeypatch):
    calls = {}
    noise = _FakeNoise("injected-gumbel")

    def fake_canvas_sample(logits, temperature, gumbel_noise):
        calls["sample"] = (logits, temperature, gumbel_noise)
        return "samples"

    monkeypatch.setattr(TS, "canvas_sample", fake_canvas_sample)

    logits = _FakeLogits()
    out = canvas_sample_from_params(
        logits,
        {"temperature": 0.7},
        default_temperature=0.8,
        gumbel_noise=noise,
    )

    assert out == "samples"
    assert calls["sample"] == (logits, 0.7, noise)
    assert noise.deallocated is False


# --- served gumbel default -----------------------------------------------------
# The default has moved twice (#48291): 2026-07-24 to `device`, 2026-07-25 back to `host` after a
# 4-seed A/B showed corrupted text, then back to `device` once the CAUSE was found — the Blackhole
# SFPU PRNG was a sliding window over one stream, so 64 of 256 canvas positions held a byte-identical
# copy of another position's noise. With the kernel advancing the window per element the same A/B is
# clean 4/4 at ~53.6 vs ~36.3 tokens/block/s. 2026-07-27 measured `host` as no repair at all (it
# drifts on the same prompts and costs 1.40x; the drift was the canvas attending the prefill pad
# keys), so the `host` SERVING mode was DELETED on 2026-07-28. The invariant is therefore not "the
# default is host" but "the default is only allowed to be a device RNG while the kernel-level
# independence gate holds" — that gate is
# tests/ttnn/nightly/unit_tests/operations/rand/test_rand_independence.py.

SUPPORTED_UPFRONT_MODES = {"device"}


def _generator_vllm():
    # `tt.generator_vllm` needs vllm (container-gated), so the skip stays inside the tests that
    # read it -- at module scope it would skip every test in this file.
    return pytest.importorskip("models.experimental.diffusion_gemma.tt.generator_vllm")


def test_served_default_is_device_for_throughput():
    """`device` is the default; the `host` alternative missed the throughput bar (~1.48x slower).

    If a non-device Gumbel source is ever reintroduced, the reason should be a NEW correctness
    finding, and the kernel gate in test_rand_independence.py is where to look first -- a
    regression there is what would make a device RNG unusable again.
    """
    GV = _generator_vllm()
    assert GV.DEFAULT_VLLM_GUMBEL_MODE == "device", (
        f"served Gumbel default is {GV.DEFAULT_VLLM_GUMBEL_MODE!r}, not 'device'. That is a "
        "throughput regression (~53.6 -> ~36.3 tokens/block/s) unless a correctness finding "
        "justifies it; see doc/decision_fidelity/degenerate_output_fix.md"
    )


def test_the_deleted_host_serving_mode_is_not_offered():
    """`host` was a SERVING mode and is gone; the torch-noise INJECTION harness is not.

    The distinction matters because both are spelled "host gumbel". Deleted: the per-step
    full-vocabulary torch draw offered as ``DG_VLLM_GUMBEL_MODE=host``. Kept: replaying a torch
    run's exact pre-computed noise onto the device so TT decisions are token-for-token comparable
    to a torch oracle -- ``reference/replay_hf_tt.py`` depends on ``make_host_gumbel_noise_fn``.
    """
    serving = pytest.importorskip("models.experimental.diffusion_gemma.tt.serving")
    generate = pytest.importorskip("models.experimental.diffusion_gemma.tt.generate")

    assert "host" not in serving.GUMBEL_MODES
    assert SUPPORTED_UPFRONT_MODES == {"device"}
    assert not hasattr(generate, "make_seeded_host_gumbel_noise_fn")
    assert not hasattr(generate, "_host_gumbel_prefetch_enabled")
    assert not hasattr(generate, "_host_gumbel_tensor")
    # The HF<->TT determinism harness is deliberately kept.
    assert hasattr(generate, "host_gumbel_noise_to_device")
    assert hasattr(generate, "make_host_gumbel_noise_fn")


def test_the_kernel_gate_this_default_depends_on_exists():
    """The device default is only defensible while the ttnn.rand independence gate is present."""
    GV = _generator_vllm()
    repo_root = Path(GV.__file__).resolve().parents[4]
    gate = repo_root / "tests" / "ttnn" / "nightly" / "unit_tests" / "operations" / "rand" / "test_rand_independence.py"
    assert gate.is_file(), (
        f"missing {gate}: the served device Gumbel default depends on the Blackhole ttnn.rand "
        "fix, and that gate is what keeps the fix from silently regressing"
    )
    text = gate.read_text()
    assert "test_rand_columns_are_distinct" in text, "the duplicate-column gate is gone"


def test_launcher_default_matches_the_module_default():
    """The GPQA launcher exports its own default; the two must not drift apart."""
    GV = _generator_vllm()
    script = Path(GV.__file__).resolve().parent.parent / "doc" / "optimize_perf" / "run_upfront_gpqa.sh"
    if not script.is_file():
        pytest.skip("run_upfront_gpqa.sh not present in this checkout")
    expected = f'DG_VLLM_GUMBEL_MODE="${{DG_VLLM_GUMBEL_MODE:-{GV.DEFAULT_VLLM_GUMBEL_MODE}}}"'
    assert (
        expected in script.read_text()
    ), f"launcher default disagrees with DEFAULT_VLLM_GUMBEL_MODE; expected {expected!r}"


# --- device: regenerated noise on the QB2 mesh ---------------------------------
# This is the only ``mesh_device`` test here, and it must stay AHEAD of the first ``device``-fixture
# test below: ``_device_module_impl`` is module-scoped (conftest.py:304), so once a ``device`` test
# has run, device 0 stays open for the rest of the module and opening a 4-device fabric mesh would
# fail (conftest.py:366).


def _mesh_1x4_with_fabric():
    """The (1, 4) FABRIC_1D parametrization, resolved without risking the rest of the file.

    ``parametrize_mesh_with_fabric`` enumerates devices while it builds the decorator and
    deliberately re-raises anything that is not a known discovery failure; here that would be a
    collection error for every host-only test in this module, so an unhealthy runner is degraded to
    the same ``device-unavailable`` skip the helper itself uses for discovery failures.
    """
    try:
        from models.demos.gemma4.tests.test_factory import parametrize_mesh_with_fabric

        return parametrize_mesh_with_fabric([(1, 4)])
    except (ImportError, RuntimeError) as exc:
        return pytest.mark.parametrize(
            "mesh_device, device_params",
            [
                pytest.param(
                    (1, 1),
                    {"fabric_config": None},
                    id="device-unavailable",
                    marks=pytest.mark.skip(reason=f"Mesh parametrization unavailable: {exc}"),
                )
            ],
            indirect=True,
        )


@requires_device
@pytest.mark.use_module_device
@_mesh_1x4_with_fabric()
def test_sample_gumbel_noise_runs_on_qb2_mesh(mesh_device):
    noise = TS.sample_gumbel_noise((1, 1, 32, 32), device=mesh_device, seed=47472)

    host_noise = ttnn.to_torch(ttnn.get_device_tensors(noise)[0])
    assert host_noise.shape == (1, 1, 32, 32)
    noise.deallocate(True)


# --- device: canvas sampling vs an injected gumbel reference -------------------


def _to_device(device, value, *, dtype=ttnn.float32):
    return ttnn.from_torch(value, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)


def _release(*tensors):
    # The device tests below share one module-scoped device, so the large N=4096 canvases have to be
    # handed back explicitly instead of relying on a per-file device close.
    for tensor in tensors:
        tensor.deallocate(True)


def _structured_logits_jittered(length: int, vocab_size: int):
    logits = torch.full((1, length, vocab_size), -2.0, dtype=torch.float32)
    base_ids = torch.arange(length) % vocab_size
    alt_ids = (base_ids + 17) % vocab_size
    logits[0, torch.arange(length), base_ids] = torch.linspace(0.5, 4.0, length)
    logits[0, torch.arange(length), alt_ids] = torch.linspace(0.25, 2.0, length)
    logits += torch.randn_like(logits) * 1.0e-3
    return logits


@requires_device
@pytest.mark.use_module_device
def test_canvas_sample_matches_injected_gumbel_reference(device):
    torch.manual_seed(23)
    length = 256
    vocab_size = 512
    temperature = S.temperature_at_step(step=5, num_steps=48, t_start=0.8, t_end=0.4)
    logits = _structured_logits_jittered(length, vocab_size)
    noise = S.sample_gumbel_noise(logits.shape, generator=torch.Generator().manual_seed(29))

    ref = S.gumbel_max_sample(logits, temperature, noise=noise)
    out = TS.canvas_sample(
        _to_device(device, logits),
        temperature,
        _to_device(device, noise),
    )

    assert torch.equal(ttnn.to_torch(out).squeeze(-1).to(torch.long), ref)


@requires_device
@pytest.mark.use_module_device
def test_canvas_sample_from_params_matches_injected_gumbel_reference(device):
    torch.manual_seed(37)
    length = 256
    vocab_size = 512
    temperature = S.temperature_at_step(step=11, num_steps=48, t_start=0.8, t_end=0.4)
    logits = _structured_logits_jittered(length, vocab_size)
    noise = S.sample_gumbel_noise(logits.shape, generator=torch.Generator().manual_seed(41))
    sampling_params = {"temperature": temperature, "top_k": 64, "top_p": 0.95, "seed": 41}
    config = canvas_sampling_config_from_params(sampling_params, default_temperature=0.8)
    assert config.top_k == 64
    assert config.top_p == 0.95
    assert config.top_k_top_p_supported is False

    ref = S.gumbel_max_sample(logits, temperature, noise=noise)
    out = canvas_sample_from_params(
        _to_device(device, logits),
        sampling_params,
        default_temperature=0.8,
        gumbel_noise=_to_device(device, noise),
    )

    assert torch.equal(ttnn.to_torch(out).squeeze(-1).to(torch.long), ref)


@requires_device
@pytest.mark.use_module_device
def test_temperature_scale_matches_reference(device):
    torch.manual_seed(31)
    logits = torch.randn(1, 256, 512, dtype=torch.float32)
    temperature = S.temperature_at_step(step=17, num_steps=48, t_start=0.8, t_end=0.4)
    out = TS.temperature_scale(_to_device(device, logits), temperature)

    passing, message = assert_with_pcc(logits / temperature, ttnn.to_torch(out), 0.9999)
    assert passing, message


# --- device: canvas sampling marginals -----------------------------------------
# SCOPE: these validate per-position **marginals**. ``_distribution_metrics`` averages over the
# sample axis, so correlation *between* canvas positions is invisible here -- every marginal can be
# correct while positions share noise. Independence across canvas positions is gated separately by
# the cross-position section below; do not read a pass here as evidence that the device Gumbel draw
# is IID (as of 2026-07-25 it is not).

DIST_NUM_SAMPLES = 4096
# Fixed-seed QB2 toy-vocab smoke margin for the regenerated-noise workarounds:
# observed max_top1_freq_error is ~0.03 at N=4096, so 0.05 catches large RNG
# regressions without pretending to validate the production 262144-vocab regime.
DIST_MAX_TOP1_FREQ_ERROR = 0.05
DIST_MAX_MEAN_KL = 0.05


def _structured_logits_repeated(num_samples: int, length: int, vocab_size: int):
    logits = torch.full((1, length, vocab_size), -1.5, dtype=torch.float32)
    top_ids = torch.arange(length) % vocab_size
    alt_ids = (top_ids + 5) % vocab_size
    logits[0, torch.arange(length), top_ids] = torch.linspace(0.75, 1.25, length)
    logits[0, torch.arange(length), alt_ids] = torch.linspace(0.25, 0.75, length)
    return logits.expand(num_samples, -1, -1).contiguous()


def _distribution_metrics(sample_ids, expected_probs):
    vocab_size = expected_probs.shape[-1]
    empirical = F.one_hot(sample_ids, num_classes=vocab_size).float().mean(dim=0)
    top_ids = expected_probs.argmax(dim=-1)
    top_expected = expected_probs.gather(-1, top_ids[:, None]).squeeze(-1)
    top_empirical = empirical.gather(-1, top_ids[:, None]).squeeze(-1)
    max_top1_freq_error = float((top_empirical - top_expected).abs().max())

    eps = 1.0e-4
    kl = (expected_probs * (expected_probs.clamp_min(eps).log() - empirical.clamp_min(eps).log())).sum(dim=-1)
    return max_top1_freq_error, float(kl.mean())


@requires_device
@pytest.mark.use_module_device
def test_canvas_sample_matches_torch_argmax_with_readback_device_noise(device):
    num_samples = 64
    length = 32
    vocab_size = 32
    temperature = 0.7
    logits = _structured_logits_repeated(num_samples, length, vocab_size)

    tt_logits = _to_device(device, logits)
    device_noise = TS.sample_gumbel_noise(logits.shape, device=device, seed=47472)
    samples = TS.canvas_sample(tt_logits, temperature, device_noise)

    host_noise = ttnn.to_torch(device_noise).float()
    sample_ids = ttnn.to_torch(samples).squeeze(-1).to(torch.long)
    _release(tt_logits, device_noise, samples)

    ref = torch.argmax(logits / temperature + host_noise, dim=-1)
    assert torch.equal(sample_ids, ref)


@requires_device
@pytest.mark.use_module_device
def test_canvas_sample_permuted_vocab_regenerated_noise_distribution(device):
    # One ttnn.rand call is still used, but vocab is generated as the outer axis
    # and permuted back to avoid the known innermost-vocab correlation.
    num_samples = DIST_NUM_SAMPLES
    length = 32
    vocab_size = 32
    temperature = 0.7
    logits = _structured_logits_repeated(num_samples, length, vocab_size)
    expected_probs = F.softmax(logits[0] / temperature, dim=-1)

    tt_logits = _to_device(device, logits)
    device_noise = TS.sample_gumbel_noise_with_permuted_vocab(logits.shape, device=device, seed=47472)
    samples = TS.canvas_sample(tt_logits, temperature, device_noise)
    sample_ids = ttnn.to_torch(samples).squeeze(-1).to(torch.long)
    _release(tt_logits, device_noise, samples)

    max_top1_freq_error, mean_kl = _distribution_metrics(sample_ids, expected_probs)
    print(
        f"\n[canvas sampling permuted-vocab dist] N={num_samples} "
        f"max_top1_freq_error={max_top1_freq_error:.4f} mean_kl={mean_kl:.4f}"
    )
    assert max_top1_freq_error < DIST_MAX_TOP1_FREQ_ERROR
    assert mean_kl < DIST_MAX_MEAN_KL


@requires_device
@pytest.mark.use_module_device
@pytest.mark.xfail(
    reason=(
        "QB2 ttnn.rand regenerated noise is currently not iid enough for W4 distributional canvas sampling: "
        "uniform-logit argmax histograms show empty/high buckets while torch-injected Gumbel noise is exact."
    ),
    strict=True,
)
def test_canvas_sample_regenerated_noise_distribution(device):
    num_samples = DIST_NUM_SAMPLES
    length = 32
    vocab_size = 32
    temperature = 0.7
    logits = _structured_logits_repeated(num_samples, length, vocab_size)
    expected_probs = F.softmax(logits[0] / temperature, dim=-1)

    tt_logits = _to_device(device, logits)
    device_noise = TS.sample_gumbel_noise(logits.shape, device=device, seed=47472)
    samples = TS.canvas_sample(tt_logits, temperature, device_noise)
    sample_ids = ttnn.to_torch(samples).squeeze(-1).to(torch.long)
    _release(tt_logits, device_noise, samples)

    max_top1_freq_error, mean_kl = _distribution_metrics(sample_ids, expected_probs)
    print(
        f"\n[canvas sampling dist] N={num_samples} max_top1_freq_error={max_top1_freq_error:.4f} "
        f"mean_kl={mean_kl:.4f}"
    )
    assert max_top1_freq_error < DIST_MAX_TOP1_FREQ_ERROR
    assert mean_kl < DIST_MAX_MEAN_KL


# --- device: cross-canvas-position correlation of the gumbel draw ---------------
# ``sample_gumbel_noise_with_permuted_vocab`` keeps the vocab axis off ``ttnn.rand``'s degenerate
# innermost axis, but it collapses every non-vocab axis into ONE trailing axis and draws
# ``ttnn.rand((vocab, inner))`` -- and for the production logits shape ``(1, 1, 256, vocab)`` that
# ``inner`` **is the 256 canvas positions**, so the last-dim correlation is relocated onto the
# canvas-position axis rather than removed (#48291). With correlated noise across positions one
# unlucky draw pushes the SAME token to the top at many positions at once instead of the independent
# rare flips IID Gumbel gives -- the observed degeneration texture, worst where the logits are
# flattest. Three arms make the numbers interpretable: ``host`` (torch Gumbel, the IID control that
# calibrates every metric), ``plain`` (``sample_gumbel_noise``, vocab-innermost -- the layout
# ``tt/generate.py`` actually ships, generate.py:652) and ``permuted``
# (``sample_gumbel_noise_with_permuted_vocab`` -- production until 2026-07-27, still reachable through
# the vLLM seam's ``use_vocab_permuted_noise=True``, and NOT IID: tt/sampling.py:329).

CANVAS_LEN = 256
# The production canvas/vocab geometry; PROBE_VOCAB keeps the default run cheap while staying
# far above the canvas length so the IID null is tight (off-diagonal |r| ~ 1/sqrt(vocab)).
PROD_VOCAB = 262144
PROBE_VOCAB = 16384


def _host_gumbel(shape, *, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    uniform = torch.rand(shape, generator=generator, dtype=torch.float32)
    return -torch.log(-torch.log(uniform.clamp_min(torch.finfo(torch.float32).tiny)))


def _draw(arm: str, shape, *, device, seed: int) -> torch.Tensor:
    """Return the arm's Gumbel noise as a host ``[canvas_len, vocab]`` matrix."""
    if arm == "host":
        return _host_gumbel(shape, seed=seed).reshape(shape[-2], shape[-1])
    generator = TS.sample_gumbel_noise_with_permuted_vocab if arm == "permuted" else TS.sample_gumbel_noise
    noise = generator(shape, device=device, seed=seed)
    host = ttnn.to_torch(noise).float()
    if hasattr(noise, "deallocate"):
        noise.deallocate(True)
    # A mesh draw comes back with the replicated devices concatenated; keep the first replica.
    return host.reshape(-1, shape[-1])[: shape[-2], :]


def _position_correlation(noise: torch.Tensor) -> dict:
    """Correlation ACROSS canvas positions, using the vocab axis as the sample axis."""
    vocab = noise.shape[1]
    corr = torch.corrcoef(noise.double())
    corr = torch.nan_to_num(corr, nan=0.0)
    off_diagonal = ~torch.eye(corr.shape[0], dtype=torch.bool)
    magnitudes = corr[off_diagonal].abs()
    # Under IID, off-diagonal r is ~N(0, 1/sqrt(vocab)); 5 sigma is a generous per-pair bound.
    sigma = 1.0 / (vocab**0.5)
    return {
        "max_abs_r": float(magnitudes.max()),
        "mean_abs_r": float(magnitudes.mean()),
        "sigma": sigma,
        "max_r_in_sigmas": float(magnitudes.max()) / sigma,
        "frac_pairs_over_5sigma": float((magnitudes > 5.0 * sigma).float().mean()),
    }


def _argmax_burst(noise: torch.Tensor) -> dict:
    """Flat-logits winner multiplicity: the deep-block regime, where noise decides alone.

    With flat logits the winner at each position is ``argmax`` of that position's noise row.
    Under IID over a vocab this large, 256 winners essentially never collide (expected
    collisions = C(256,2)/vocab), so any sizeable tie group is a cross-position dependency.
    """
    winners = noise.argmax(dim=-1)
    counts = torch.bincount(winners)
    max_multiplicity = int(counts.max())
    return {
        "distinct_winners": int((counts > 0).sum()),
        "num_positions": int(winners.numel()),
        "max_multiplicity": max_multiplicity,
        "positions_in_largest_group": max_multiplicity,
        "expected_collisions_iid": round(winners.numel() * (winners.numel() - 1) / 2 / noise.shape[1], 4),
    }


def _measure(arm: str, *, device, vocab: int, seed: int) -> dict:
    noise = _draw(arm, (1, 1, CANVAS_LEN, vocab), device=device, seed=seed)
    assert noise.shape == (CANVAS_LEN, vocab), noise.shape
    return {"arm": arm, "vocab": vocab, **_position_correlation(noise), **_argmax_burst(noise)}


def _report(label: str, stats: dict) -> None:
    print(
        f"[gumbel-pos-corr] {label:9s} vocab={stats['vocab']} "
        f"max|r|={stats['max_abs_r']:.5f} ({stats['max_r_in_sigmas']:.1f} sigma) "
        f"mean|r|={stats['mean_abs_r']:.5f} "
        f"pairs>5sigma={stats['frac_pairs_over_5sigma']:.4f} "
        f"distinct_winners={stats['distinct_winners']}/{stats['num_positions']} "
        f"max_multiplicity={stats['max_multiplicity']} "
        f"(iid expected collisions {stats['expected_collisions_iid']})"
    )


@requires_device
@pytest.mark.use_module_device
def test_host_control_calibrates_the_metrics():
    """The IID arm must pass both bounds, or the bounds are measuring the wrong thing."""
    stats = _measure("host", device=None, vocab=PROBE_VOCAB, seed=48291)
    _report("host", stats)
    assert stats["max_r_in_sigmas"] < 6.0
    assert stats["max_multiplicity"] <= 2
    assert stats["distinct_winners"] >= CANVAS_LEN - 2


@requires_device
@pytest.mark.use_module_device
def test_production_device_gumbel_is_independent_across_canvas_positions(device):
    """The gate the marginal test cannot express: no cross-position dependency."""
    stats = _measure("permuted", device=device, vocab=PROBE_VOCAB, seed=48291)
    _report("permuted", stats)
    assert stats["max_multiplicity"] <= 4, (
        "canvas positions are picking the same token far more often than IID noise allows -- "
        f"largest synchronized group is {stats['max_multiplicity']} positions: {stats}"
    )
    assert (
        stats["frac_pairs_over_5sigma"] < 0.01
    ), f"cross-position correlation is widespread, not a tail artefact: {stats}"


@requires_device
@pytest.mark.use_module_device
def test_shipped_plain_gumbel_is_independent_across_canvas_positions(device):
    """The SHIPPED layout's cross-position gate (was ``test_diagnostic_plain_path_for_comparison``).

    ``make_seeded_gumbel_noise_fn`` -- the serving denoise noise hook -- draws
    ``TS.sample_gumbel_noise``, vocab innermost, "deliberately" (tt/generate.py:652), and the
    253/256 figure in the comment above that draw IS this measurement. ``permuted`` has had no
    production caller since ``use_vocab_permuted_noise`` defaulted to False
    (tt/sampling_params.py:121), so without this row the #48291 section does not measure the layout
    production runs. It started life as a print-only comparison; it is a gate now because a
    ``ttnn.rand`` regression on the vocab-innermost row-stream assignment is exactly how the canvas
    collapse returns -- this layout already failed once with a different constant (offset 17, 96 of
    256 rows duplicated), and the surviving guards would all stay green through it: the permuted gate
    draws a different layout, ``test_canvas_sample_regenerated_noise_distribution`` is
    ``xfail(strict=True)`` so a worse plain path is still a "failure", and
    ``test_canvas_sample_matches_torch_argmax_with_readback_device_noise`` uses the read-back device
    noise as its own reference. Bounds are the same ones the permuted gate uses; they discriminate
    because ``doc/decision_fidelity/device_gumbel_restored.md`` measured plain at 157/256 distinct
    winners / max_mult 4 pre-fix against 253/256 / max_mult 2 post-fix.
    """
    stats = _measure("plain", device=device, vocab=PROBE_VOCAB, seed=48291)
    _report("plain", stats)
    assert stats["max_multiplicity"] <= 4, (
        "canvas positions are picking the same token far more often than IID noise allows on the "
        f"SHIPPED vocab-innermost draw -- largest synchronized group is {stats['max_multiplicity']} "
        f"positions: {stats}"
    )
    assert (
        stats["frac_pairs_over_5sigma"] < 0.01
    ), f"cross-position correlation is widespread on the shipped draw, not a tail artefact: {stats}"


@requires_device
@pytest.mark.use_module_device
@pytest.mark.skipif(
    os.environ.get("DG_GUMBEL_CORR_FULL_VOCAB") != "1",
    reason="set DG_GUMBEL_CORR_FULL_VOCAB=1 for the 262144-vocab production geometry",
)
def test_production_vocab_geometry(device):
    """The full 262144 vocab, all three arms, against the host IID control.

    ``plain`` is the real shipped geometry -- ``rand((1, 1, 256, 262144))`` via generate.py:652;
    ``rand((262144, 256))`` is the PERMUTED draw's rand shape, which is what this docstring used to
    claim was shipped. Both device arms are asserted so a full-vocab-only regression cannot hide
    behind the cheap PROBE_VOCAB gates above.
    """
    for arm in ("host", "plain", "permuted"):
        stats = _measure(arm, device=device if arm != "host" else None, vocab=PROD_VOCAB, seed=48291)
        _report(arm, stats)
        if arm != "host":
            assert stats["max_multiplicity"] <= 4, stats


# --- device: entropy + gumbel-max op harness -----------------------------------
# The #47468 harness extends PCC beyond logits to the diffusion *decisions*: per-position entropy
# and Gumbel-max argmax agreement, device-vs-torch and especially under bfp8, since small-probability
# drift can flip accept/renoise. Measured on QB2 (P150x4): entropy bf16 mean|Δ| ~0.09 on values ~7.6
# (accurate); entropy bfp8 mean|Δ| ~2.6 (**materially degraded** -- the headline finding: validate
# decisions, do not trust bfp8 probabilities); Gumbel-max argmax agreement ~0.98 at bf16, the ~2% gap
# being near-max ties flipping under logit quantization. Gumbel noise is drawn in torch and INJECTED
# into both paths, so argmax agreement is a token-for-token decision check, not a distributional one.

_DTYPES = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b, "fp32": ttnn.float32}
_VOCAB = 2048


def _varied_logits(seed=1):
    """Logits whose per-position entropy genuinely VARIES (low scale -> high
    entropy, high scale -> low entropy). A flat-entropy input makes PCC/agreement
    ill-conditioned, so vary the per-row temperature deliberately."""
    base = torch.randn(1, _SEQ, _VOCAB, generator=_gen(seed))
    scales = torch.linspace(0.2, 6.0, _SEQ).view(1, _SEQ, 1)
    return base * scales


def _to(t, device, dtype):
    return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)


@requires_device_sfpi
@pytest.mark.use_module_device
@pytest.mark.parametrize("temperature", [1.0, 0.6])
def test_token_entropy_bf16_accurate_and_bfp8_degrades(device, temperature):
    """ttnn −Σ p·log p matches torch in bf16; bfp8 is strictly worse (the #47468
    bfp8-drift finding). Uses varied-entropy logits so the metric is meaningful."""
    logits = _varied_logits()
    ref = S.token_entropy(logits, temperature=temperature)  # [1, 256]

    def err(dtype):
        out = ttnn.to_torch(TS.token_entropy(_to(logits, device, dtype), temperature=temperature)).squeeze(-1)
        assert torch.isfinite(out).all(), f"{dtype} entropy produced non-finite values (log(0) guard regressed?)"
        return (ref - out).abs(), comp_pcc(ref, out, 0.0)[1]

    bf16_d, bf16_pcc = err(ttnn.bfloat16)
    bfp8_d, bfp8_pcc = err(ttnn.bfloat8_b)
    print(
        f"\n[entropy T={temperature}] bf16: mean|Δ|={bf16_d.mean():.4f} max|Δ|={bf16_d.max():.3f} PCC={bf16_pcc:.5f}"
        f" | bfp8: mean|Δ|={bfp8_d.mean():.4f} max|Δ|={bfp8_d.max():.3f} PCC={bfp8_pcc:.5f}"
    )
    # bf16 path is accurate (mean abs err ~1% of the ~7.6 range; bound is generous vs measured ~0.09)
    assert bf16_d.mean() < 0.5, f"bf16 entropy mean|Δ|={bf16_d.mean():.4f} too high (expected ~0.09)"
    assert bf16_pcc >= 0.99, f"bf16 entropy PCC={bf16_pcc:.5f} < 0.99"
    # bfp8 is materially degraded — the harness's headline finding (decisions must be validated, not trusted)
    assert bfp8_d.mean() > 2.0 * bf16_d.mean(), "expected bfp8 entropy to be materially worse than bf16"


# Gumbel-max/argmax run on bf16 or fp32 — `ttnn.argmax` rejects bfp8 TILE inputs
# ("Only BFLOAT16, FLOAT32 are supported", assert.hpp). The canvas sampler therefore
# keeps logits at bf16+ for the argmax step (see test_gumbel_max_rejects_bfp8).
@requires_device_sfpi
@pytest.mark.use_module_device
@pytest.mark.parametrize("dtype_name", ["bf16", "fp32"])
def test_gumbel_max_argmax_agreement(device, dtype_name):
    """ttnn argmax(logits/T + injected_gumbel) agrees with torch under the SAME noise."""
    dtype = _DTYPES[dtype_name]
    temperature = 0.6
    logits = _varied_logits(seed=2)
    noise = S.sample_gumbel_noise((1, _SEQ, _VOCAB), generator=_gen(3))

    golden = S.gumbel_max_sample(logits, temperature, noise=noise)  # [1, 256] token ids
    out = ttnn.to_torch(TS.gumbel_max(_to(logits, device, dtype), temperature, _to(noise, device, dtype)))
    out = out.squeeze(-1).to(torch.long)

    agreement = float((out == golden).float().mean())
    print(f"\n[gumbel-max {dtype_name}] argmax agreement={agreement:.4f}")
    # ~0.99 measured (bf16); the gap is near-max ties flipping under logit quantization, not an op error.
    assert agreement >= 0.95, f"gumbel-max agreement {agreement:.4f} < 0.95 ({dtype_name})"


@requires_device_sfpi
@pytest.mark.use_module_device
@pytest.mark.parametrize("dtype_name", ["bf16", "fp32"])
def test_zero_noise_gumbel_is_argmax(device, dtype_name):
    """noise=0 -> argmax(logits) (temperature preserves argmax); a clean op-level check."""
    dtype = _DTYPES[dtype_name]
    logits = _varied_logits(seed=4)

    golden = logits.argmax(dim=-1)  # [1, 256]
    zero = torch.zeros(1, _SEQ, _VOCAB)
    out = ttnn.to_torch(TS.gumbel_max(_to(logits, device, dtype), 0.8, _to(zero, device, dtype)))
    out = out.squeeze(-1).to(torch.long)

    agreement = float((out == golden).float().mean())
    print(f"\n[zero-noise argmax {dtype_name}] agreement={agreement:.4f}")
    assert agreement >= 0.95, f"zero-noise argmax agreement {agreement:.4f} < 0.95 ({dtype_name})"


@requires_device_sfpi
@pytest.mark.use_module_device
def test_gumbel_max_rejects_bfp8(device, expect_error):
    """Document the op constraint: `ttnn.argmax` rejects bfp8 TILE inputs, so the
    Gumbel-max/argmax decision step must use bf16+ logits (entropy is fine in bfp8,
    but it shows large drift — see test_token_entropy_*)."""
    logits = _varied_logits(seed=5)
    with expect_error(RuntimeError, match="BFLOAT16, FLOAT32"):
        TS.gumbel_max(
            _to(logits, device, ttnn.bfloat8_b), 0.8, _to(torch.zeros(1, _SEQ, _VOCAB), device, ttnn.bfloat8_b)
        )


# --- device: entropy-budget acceptance chain -----------------------------------
# The #47463 plan's #1 unknown (risk R1): can sort-by-confidence + cumulative-entropy cutoff +
# **scatter-back to original canvas positions** run on device and reproduce the pure-torch reference?
# Validates ``ttnn.sort`` -> ``ttnn.cumsum`` -> ``ttnn.le`` -> ``ttnn.scatter`` against the oracle on
# real hardware. fp32 entropy/cumsum/threshold isolates the chain logic from bf16 drift; the scatter
# mask is bf16 (``ttnn.scatter`` rejects fp32+TILE, scatter.cpp:109). ``min_accept`` is omitted
# (host/slice op); the spike targets the sort/scatter mapping.


def _device_chain(device, entropy: torch.Tensor, budget: float) -> dict:
    """Run the acceptance chain on device; return every intermediate as torch."""
    ent = ttnn.from_torch(entropy.float(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    sorted_vals, sorted_idx = ttnn.sort(ent, dim=-1)  # ascending: most-confident first; idx uint16
    cum = ttnn.cumsum(sorted_vals, dim=-1)

    # EXCLUSIVE prefix (HF accept_canvas): position i accepts iff the sum over
    # *strictly more confident* positions stays <= budget, i.e. (cum - sorted_vals).
    # The most-confident position has an exclusive prefix of 0 -> always accepted.
    # (Inclusive cum <= budget wrongly drops the element that crosses the budget.)
    excl = ttnn.subtract(cum, sorted_vals)

    # tensor budget (unambiguous tensor-tensor compare; scalar overload misbehaved)
    budget_t = ttnn.full(list(entropy.shape), float(budget), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    accept_sorted = ttnn.le(excl, budget_t)  # exclusive prefix <= budget -> 1 / 0

    # ttnn.scatter rejects fp32+TILE (scatter.cpp:109); bf16+uint16+TILE is supported
    # (test_scatter.py:92). Mask is 0/1 -> exact in bf16. L<256 dodges issue #23407.
    accept_sorted_bf = ttnn.typecast(accept_sorted, ttnn.bfloat16)
    zeros = ttnn.typecast(ttnn.zeros_like(ent), ttnn.bfloat16)
    accept = ttnn.scatter(zeros, -1, sorted_idx, accept_sorted_bf)  # scatter-back to original positions

    return {
        "sorted_vals": ttnn.to_torch(sorted_vals).float(),
        "cum": ttnn.to_torch(cum).float(),
        "accept_sorted": ttnn.to_torch(accept_sorted_bf) > 0.5,
        "accept": ttnn.to_torch(accept) > 0.5,
    }


def _budget_for_fraction(entropy: torch.Tensor, frac: float) -> float:
    sorted_cum = torch.cumsum(torch.sort(entropy, dim=-1).values, dim=-1)
    k = int(frac * entropy.shape[-1])
    if k == 0:
        return float(sorted_cum[0, 0]) * 0.5
    if k >= entropy.shape[-1]:
        return float(sorted_cum[0, -1]) * 2.0
    return float((sorted_cum[0, k - 1] + sorted_cum[0, k]) / 2)


@requires_device_sfpi
@pytest.mark.use_module_device
@pytest.mark.parametrize("frac", [0.0, 0.3, 0.7, 1.0], ids=["accept~0", "accept~30", "accept~70", "accept~all"])
def test_entropy_budget_accept_matches_reference(device, frac):
    torch.manual_seed(7)
    batch, length = 1, 128
    entropy = torch.rand(batch, length) + torch.arange(length).float() * 1e-4
    budget = _budget_for_fraction(entropy, frac)

    ref = S.entropy_budget_accept(entropy, budget, min_accept=0)
    dev = _device_chain(device, entropy, budget)["accept"]

    assert dev.shape == ref.shape
    assert torch.equal(dev, ref), f"accept mask mismatch (frac={frac}): {int((dev != ref).sum())} of {length} differ"


@requires_device_sfpi
@pytest.mark.use_module_device
def test_production_entropy_budget_accept_guards_device_sort_at_canvas_256(device):
    torch.manual_seed(47463)
    batch, length = 1, 256
    entropy = torch.rand(batch, length) + torch.arange(length).float() * 1e-4
    budget = _budget_for_fraction(entropy, 0.375)
    ref = S.entropy_budget_accept(entropy, budget, min_accept=0)

    ent = ttnn.from_torch(entropy.float(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    dev_t = entropy_budget_accept(ent, budget)
    dev = ttnn.to_torch(dev_t) > 0.5

    try:
        assert int(dev.sum()) == int(ref.sum())
        assert torch.equal(dev, ref), f"production accept mask mismatch: {int((dev != ref).sum())} of {length} differ"
    finally:
        dev_t.deallocate(True)
        ent.deallocate(True)


@requires_device_sfpi
@pytest.mark.use_module_device
def test_production_entropy_budget_accept_uses_entropy_dtype_for_budget(device):
    torch.manual_seed(47464)
    batch, length = 1, 256
    entropy = torch.rand(batch, length) + torch.arange(length).float() * 1e-4
    entropy_bf16 = entropy.to(torch.bfloat16).float()
    budget = _budget_for_fraction(entropy_bf16, 0.5)
    ref = S.entropy_budget_accept(entropy_bf16, budget, min_accept=0)

    ent = ttnn.from_torch(entropy_bf16, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    dev_t = entropy_budget_accept(ent, budget)
    dev = ttnn.to_torch(dev_t) > 0.5

    try:
        assert int(dev.sum()) == int(ref.sum())
        assert torch.equal(
            dev, ref
        ), f"bf16 production accept mask mismatch: {int((dev != ref).sum())} of {length} differ"
    finally:
        dev_t.deallocate(True)
        ent.deallocate(True)


# --- device: topk / argmax across vocab-shard widths ---------------------------
# An independent DiffusionGemma implementation measured ``ttnn.topk`` returning a **garbage index
# and value** at a 32768-wide reduction (32 of 256 rows matching a torch control, with ``inf``
# values) while the same call was correct at width >= 49152; its workaround was to pad the shard to
# 49152 with ``-inf`` purely to obtain the index and to take the value from ``ttnn.max`` (reliable at
# any width). DiffusionGemma does not hit that width today -- the terminal argmax goes through
# ``tt.sampling.argmax_last_dim`` (``ttnn.argmax`` on ROW_MAJOR, not ``topk``), and the only
# ``ttnn.topk`` on the denoise path is the router's ``k=top_k`` over the 128-expert axis -- but the
# vocab shard IS a function of the mesh: V=262144 over tp=4 is 65536 (safe), over tp=8 exactly 32768.
#
# Measured on QB2 2026-07-27, and the cliff reproduces exactly as reported:
#     width 16384  index agreement 0.129
#     width 32768  index agreement 0.129   <- V/tp at tp=8
#     width 49152  index agreement 1.000
#     width 65536  index agreement 1.000   <- V/tp at tp=4, what we serve
#
# So ``ttnn.topk(k=1)`` must not be used on a vocab shard narrower than 49152 on this stack. These
# tests always assert the widths we serve today are exact, and for the widths we do not serve they
# record the agreement under ``-s`` so the reported cliff is a *finding*, not a red build on someone
# else's op.

# 65536 = V/tp at the tp=4 mesh we serve; 32768 = V/tp at tp=8, the reported cliff.
_SERVED_WIDTHS = (65536,)
_PROBE_WIDTHS = (16384, 32768, 49152, 65536)


def _logits(width, seed=7):
    """Row-varied logits with a unique maximum per row.

    A tie would make ``argmax`` legitimately implementation-defined, which would confound a
    correctness check, so the winner is nudged clear of the runner-up.
    """
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(1, 1, _SEQ, width, generator=g) * torch.linspace(0.5, 4.0, _SEQ).view(1, 1, _SEQ, 1)
    winners = torch.randint(0, width, (_SEQ,), generator=g)
    for row, col in enumerate(winners.tolist()):
        x[0, 0, row, col] = x[0, 0, row].max().item() + 1.0
    return x, winners


def _topk_index(tt_logits):
    _values, indices = ttnn.topk(tt_logits, 1, dim=-1, largest=True, sorted=False)
    out = ttnn.to_torch(indices).reshape(-1)[:_SEQ].to(torch.int64)
    _values.deallocate(True)
    indices.deallocate(True)
    return out


def _agreement(got, expected):
    return float((got == expected).float().mean().item())


@requires_device
@pytest.mark.use_module_device
@pytest.mark.parametrize("width", _PROBE_WIDTHS)
def test_topk_k1_index_across_shard_widths(device, width):
    """Record ``ttnn.topk(k=1)`` index agreement per width; assert exactness where we serve."""

    torch_logits, expected = _logits(width)
    tt_logits = ttnn.from_torch(torch_logits, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    try:
        got = _topk_index(tt_logits)
        max_value = ttnn.to_torch(ttnn.max(tt_logits, dim=-1, keepdim=True)).reshape(-1)[:_SEQ].float()
    finally:
        tt_logits.deallocate(True)

    agreement = _agreement(got, expected)
    finite = bool(torch.isfinite(max_value).all())
    print(f"[topk width sweep] width={width} index_agreement={agreement:.4f} max_all_finite={finite}")

    if width in _SERVED_WIDTHS:
        assert agreement == 1.0, (
            f"ttnn.topk(k=1) index is wrong at width {width}, which is the vocab shard "
            f"DiffusionGemma serves (V=262144 over tp=4): agreement {agreement:.4f}"
        )
        assert finite, f"ttnn.max produced a non-finite value at served width {width}"


@requires_device
@pytest.mark.use_module_device
def test_router_axis_topk_is_exact(device):
    """The 128-wide expert axis — the only ``ttnn.topk`` the denoise path actually runs.

    The selection is made **separable**: the 8 winners are lifted clear of the rest by a margin far
    wider than bf16 rounding. Without that, a plain random input puts the 8th and 9th values within
    a bf16 ulp of each other on a few rows and the op picks a different (equally valid) member —
    measured 0.9961 set overlap, which is tie-breaking, not an op error. Asserting exactness on a
    tie-free input tests the op; asserting it on a tied one tests nothing and fails intermittently.
    """

    num_experts, top_k = 128, 8
    g = torch.Generator().manual_seed(11)
    routing = torch.randn(1, 1, _SEQ, num_experts, generator=g).clamp(-3.0, 3.0)
    winners = torch.stack([torch.randperm(num_experts, generator=g)[:top_k] for _ in range(_SEQ)])
    for row in range(_SEQ):
        # Distinct, well-separated values (10.0, 11.0, ...) so neither the membership nor the order
        # can be decided by rounding.
        routing[0, 0, row, winners[row]] = torch.arange(10.0, 10.0 + top_k)

    tt_routing = ttnn.from_torch(routing, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    try:
        _values, indices = ttnn.topk(tt_routing, top_k, dim=-1, largest=True, sorted=True)
        got = ttnn.to_torch(indices).reshape(_SEQ, top_k).to(torch.int64)
        _values.deallocate(True)
        indices.deallocate(True)
    finally:
        tt_routing.deallocate(True)

    overlap = sum(len(set(got[i].tolist()) & set(winners[i].tolist())) for i in range(_SEQ)) / (_SEQ * top_k)
    print(f"[router topk] width={num_experts} k={top_k} set_overlap={overlap:.4f}")
    assert overlap == 1.0, f"router top-{top_k} over {num_experts} experts disagrees with torch: {overlap:.4f}"


@requires_device
@pytest.mark.use_module_device
def test_argmax_last_dim_matches_torch_at_served_width(device):
    """The op the terminal actually uses, at the width it actually runs on."""

    width = 65536
    torch_logits, expected = _logits(width, seed=13)
    tt_logits = ttnn.from_torch(torch_logits, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    try:
        got = ttnn.to_torch(argmax_last_dim(tt_logits)).reshape(-1)[:_SEQ].to(torch.int64)
    finally:
        tt_logits.deallocate(True)

    agreement = _agreement(got, expected)
    print(f"[argmax_last_dim] width={width} agreement={agreement:.4f}")
    assert agreement == 1.0, f"argmax_last_dim disagrees with torch at the served width: {agreement:.4f}"
