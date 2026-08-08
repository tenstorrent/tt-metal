# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""End-to-end model tests: teacher-forced forward and autoregressive generation."""

from dataclasses import replace

import pytest
import torch

from models.demos.time_series_transformer.reference.torch_reference import (
    compute_metrics,
    crps,
    generate_mean_reference,
    relative_error,
)
from models.demos.time_series_transformer.tt.model import TimeSeriesTransformer
from models.demos.time_series_transformer.tt.ops import to_torch

STACK_PCC_THRESHOLD = 0.99
ACCURACY_TOLERANCE = 0.05


@pytest.fixture(scope="module")
def model(device, config, hf_state):
    module = TimeSeriesTransformer(config, device=device)
    result = module.load_hf_state_dict(hf_state, strict=True)
    assert not result["missing_keys"], result["missing_keys"]
    return module


class TestForward:
    def test_hidden_states_pcc(self, model, goldens):
        output = model.forward(future_values=goldens.future_values, **goldens.inputs)

        _, _, encoder_pcc = compute_metrics(
            goldens.encoder_last_hidden_state, to_torch(output.encoder_last_hidden_state)
        )
        _, _, decoder_pcc = compute_metrics(
            goldens.decoder_last_hidden_state, to_torch(output.decoder_last_hidden_state)
        )

        assert encoder_pcc > STACK_PCC_THRESHOLD, f"encoder PCC {encoder_pcc:.6f}"
        assert decoder_pcc > STACK_PCC_THRESHOLD, f"decoder PCC {decoder_pcc:.6f}"

    def test_scaler_statistics_match(self, model, goldens):
        output = model.forward(future_values=goldens.future_values, **goldens.inputs)
        torch.testing.assert_close(output.loc, goldens.loc)
        torch.testing.assert_close(output.scale, goldens.scale)

    def test_embedder_weights_loaded(self, model, config):
        assert len(model.embedder_weights) == config.num_static_categorical_features
        assert model.embedder_weights[0].shape == (config.cardinality[0], config.embedding_dimension[0])


class TestGenerate:
    def test_output_shape(self, model, config, goldens):
        samples = 8
        output = model.generate(num_parallel_samples=samples, mode="sample", **goldens.inputs)
        assert output.shape == (
            goldens.inputs["past_values"].shape[0],
            samples,
            config.prediction_length,
        )
        assert torch.isfinite(output).all()

    def test_kv_cache_matches_uncached(self, device, config, hf_state, goldens):
        """The cached single-token path must reproduce the full-prefix decode."""
        cached = model_with(device, replace(config, use_kv_cache=True), hf_state)
        uncached = model_with(device, replace(config, use_kv_cache=False), hf_state)

        cached_output = cached.generate(num_parallel_samples=1, mode="mean", **goldens.inputs)
        uncached_output = uncached.generate(num_parallel_samples=1, mode="mean", **goldens.inputs)

        _, _, pcc = compute_metrics(uncached_output, cached_output)
        assert pcc > 0.999, f"KV-cache path diverged from full-prefix decode: PCC {pcc:.6f}"

    def test_mean_prediction_within_tolerance(self, model, hf_model, goldens):
        """Mean prediction must land within 5% MAE of the reference -- a Stage 1 gate.

        Both sides take the distribution mean at each step, so this measures model error
        rather than the Monte Carlo spread of two independent 100-sample rollouts.
        """
        reference = generate_mean_reference(hf_model, goldens.inputs)
        actual = model.generate(num_parallel_samples=1, mode="mean", **goldens.inputs).squeeze(1)

        assert actual.shape == reference.shape
        mae = float(torch.mean(torch.abs(actual - reference)))
        scale = float(torch.mean(torch.abs(reference)))
        assert (
            mae / scale < ACCURACY_TOLERANCE
        ), f"mean prediction MAE {mae:.4f} is {mae / scale * 100:.2f}% of reference magnitude {scale:.4f}"

    def test_sampled_mean_is_consistent(self, model, goldens):
        """A 100-sample rollout should track the reference's own 100-sample mean.

        Two independent Monte Carlo estimates of a heavy-tailed Student-t predictive mean
        differ by roughly sigma/sqrt(n) each, so this is held to a looser bound than the
        deterministic gate above.
        """
        reference_mean = goldens.generated.mean(dim=1)

        torch.manual_seed(0)
        actual_mean = model.generate(num_parallel_samples=100, mode="sample", **goldens.inputs).mean(dim=1)

        mae = float(torch.mean(torch.abs(actual_mean - reference_mean)))
        scale = float(torch.mean(torch.abs(reference_mean)))
        assert mae / scale < 0.15, f"sampled mean {mae / scale * 100:.2f}% off the reference sample mean"

    def test_crps_within_tolerance(self, model, goldens):
        """Probabilistic calibration must match the reference within 5%."""
        target = goldens.future_values

        torch.manual_seed(0)
        actual = model.generate(num_parallel_samples=100, mode="sample", **goldens.inputs)

        reference_crps = crps(goldens.generated, target)
        actual_crps = crps(actual, target)

        error = relative_error(reference_crps, actual_crps)
        assert (
            error < ACCURACY_TOLERANCE
        ), f"CRPS {actual_crps:.4f} vs reference {reference_crps:.4f} -- {error * 100:.2f}% off"

    def test_samples_are_dispersed(self, model, goldens):
        """Sampling must actually vary -- a collapsed sampler would still pass the mean gate."""
        torch.manual_seed(0)
        output = model.generate(num_parallel_samples=32, mode="sample", **goldens.inputs)
        assert float(output.std(dim=1).mean()) > 0.0

        deterministic = model.generate(num_parallel_samples=4, mode="mean", **goldens.inputs)
        spread = deterministic.std(dim=1).mean()
        assert float(spread) < 1e-3, "mean-mode trajectories should be identical across samples"


class TestTracedRollout:
    """The whole mean-mode rollout runs from one trace with the feedback closed on device."""

    def test_matches_stepped_path(self, device, config, hf_state, goldens):
        """Closing the loop on device must not change the forecast."""
        stepped = model_with(device, replace(config, use_trace=False), hf_state)
        unrolled = model_with(device, replace(config, use_trace=True), hf_state)
        try:
            expected = stepped.generate(num_parallel_samples=1, mode="mean", **goldens.inputs)
            actual = unrolled.generate(num_parallel_samples=1, mode="mean", **goldens.inputs)
        finally:
            unrolled.release_traces()

        _, _, pcc = compute_metrics(expected, actual)
        assert pcc > 0.999, f"unrolled rollout diverged from the stepped path: PCC {pcc:.6f}"

    def test_sampling_still_uses_the_stepped_path(self, device, config, hf_state, goldens):
        """Sampling cannot be closed on device -- a Student's t needs a host Gamma draw."""
        model = model_with(device, replace(config, use_trace=True), hf_state)
        try:
            model.generate(num_parallel_samples=4, mode="sample", **goldens.inputs)
            assert model._rollouts == {}, "sample mode must not use the mean-only rollout trace"
            assert model._runners, "sample mode should register a per-step runner"
        finally:
            model.release_traces()


class TestMemoryAndKernelOptions:
    """Optional runtime knobs must stay correct, whatever they cost."""

    @pytest.mark.parametrize("exact_softmax", [True, False], ids=["exact", "fused_kernel"])
    def test_softmax_modes_agree_end_to_end(self, device, config, hf_state, hf_model, goldens, exact_softmax):
        """Both softmax paths must clear the 5% mean-prediction gate.

        ttnn.softmax leaves attention rows a few percent off unity, which looks fatal in
        isolation. It is not: the error is close to a uniform per-row scale, and the layer norm
        after the residual removes it. This pins that down so the cheaper kernel cannot be
        swapped back out on the strength of the row-sum diagnostic alone.
        """
        model = model_with(device, replace(config, use_exact_softmax=exact_softmax), hf_state)
        reference = generate_mean_reference(hf_model, goldens.inputs)
        actual = model.generate(num_parallel_samples=1, mode="mean", **goldens.inputs).squeeze(1)

        mae = float(torch.mean(torch.abs(actual - reference)))
        relative = mae / float(reference.abs().mean())
        assert relative < ACCURACY_TOLERANCE, f"exact_softmax={exact_softmax}: relative MAE {relative * 100:.2f}%"

    def test_l1_residency_is_correct(self, device, config, hf_state, goldens):
        """``use_l1=True`` must produce the same forecast as the interleaved-DRAM default.

        It is off by default because it measures slower here -- a bias cannot be fused into a
        linear whose operands live in L1, so every projection pays an extra eltwise add -- but
        it has to stay correct for anyone who wants it on a larger configuration.
        """
        baseline = model_with(device, config, hf_state)
        l1_model = model_with(device, replace(config, use_l1=True), hf_state)

        expected = baseline.generate(num_parallel_samples=1, mode="mean", **goldens.inputs)
        actual = l1_model.generate(num_parallel_samples=1, mode="mean", **goldens.inputs)

        _, _, pcc = compute_metrics(expected, actual)
        assert pcc > 0.999, f"L1 path diverged from the DRAM default: PCC {pcc:.6f}"


def model_with(device, config, hf_state) -> TimeSeriesTransformer:
    module = TimeSeriesTransformer(config, device=device)
    module.load_hf_state_dict(hf_state, strict=True)
    return module
