# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""PCC tests for the probabilistic output head across all three distributions."""


from dataclasses import replace

import pytest
import torch

from models.demos.time_series_transformer.reference.torch_reference import (
    build_reference_model,
    compute_metrics,
    compute_pcc,
    generate_mean_reference,
    make_inputs,
    relative_error,
)
from models.demos.time_series_transformer.tt.config import config_from_hf, get_ttnn_dtype
from models.demos.time_series_transformer.tt.distribution import DistributionHead
from models.demos.time_series_transformer.tt.model import TimeSeriesTransformer
from models.demos.time_series_transformer.tt.ops import to_device, to_torch
from models.demos.time_series_transformer.tt.weights import substate

PCC_THRESHOLD = 0.999
NLL_TOLERANCE = 0.05


@pytest.fixture(scope="module")
def head(device, config, hf_state):
    dtype = get_ttnn_dtype(config.dtype)
    module = DistributionHead(config, device=device, dtype=dtype)
    result = module.load_hf_state_dict(substate(hf_state, "parameter_projection"), strict=True)
    assert not result["missing_keys"]
    return module


class TestParameterProjection:
    """HuggingFace's ParameterProjection.forward applies the domain map itself, so the
    reference parameters are already constrained -- compare against our mapped output."""

    def test_parameters_pcc(self, device, config, goldens, head):
        dtype = get_ttnn_dtype(config.dtype)
        hidden = to_device(goldens.decoder_last_hidden_state, device=device, dtype=dtype)
        actual = head.parameters(hidden)

        assert len(actual) == config.num_distribution_params == 3
        for index, expected in enumerate(goldens.distribution_params):
            got = to_torch(actual[index]).reshape(expected.shape)
            mse, mae, pcc = compute_metrics(expected, got)
            assert pcc > PCC_THRESHOLD, f"param {index} PCC {pcc:.6f} (mse={mse:.3e}, mae={mae:.3e})"

    def test_domain_map_constrains_parameters(self, device, config, goldens, head):
        """df > 2 and scale > 0 for every predicted timestep."""
        dtype = get_ttnn_dtype(config.dtype)
        hidden = to_device(goldens.decoder_last_hidden_state, device=device, dtype=dtype)
        df, _, scale = [to_torch(p) for p in head.parameters(hidden)]

        assert bool((df > 2.0).all()), "student-t degrees of freedom must exceed 2"
        assert bool((scale > 0.0).all()), "student-t scale must be positive"

    def test_raw_parameters_are_unconstrained(self, device, config, goldens, head):
        """The pre-domain-map projections should differ from the mapped ones."""
        dtype = get_ttnn_dtype(config.dtype)
        hidden = to_device(goldens.decoder_last_hidden_state, device=device, dtype=dtype)
        raw = [to_torch(p) for p in head.raw_parameters(hidden)]
        mapped = [to_torch(p) for p in head.domain_map(head.raw_parameters(hidden))]

        assert not torch.allclose(raw[0], mapped[0]), "df should be shifted by 2 + squareplus"
        torch.testing.assert_close(raw[1], mapped[1])  # loc passes through unchanged


class TestDomainMap:
    """On-device domain maps must agree with transformers' time_series_utils."""

    @pytest.mark.parametrize("kind", ["student_t", "normal", "negative_binomial"])
    def test_matches_huggingface(self, device, config, kind):
        from transformers.time_series_utils import NegativeBinomialOutput, NormalOutput, StudentTOutput

        outputs = {
            "student_t": StudentTOutput(),
            "normal": NormalOutput(),
            "negative_binomial": NegativeBinomialOutput(),
        }
        local_config = replace(config, distribution_output=kind)
        dtype = get_ttnn_dtype(local_config.dtype)
        head = DistributionHead(local_config, device=device, dtype=dtype)

        torch.manual_seed(0)
        num_params = local_config.num_distribution_params
        raw = [torch.randn(2, local_config.prediction_length, 1) * 2.0 for _ in range(num_params)]

        expected = outputs[kind].domain_map(*[r.clone() for r in raw])
        actual = head.domain_map([to_device(r, device=device, dtype=dtype) for r in raw])

        assert len(actual) == len(expected)
        for index, (want, got) in enumerate(zip(expected, actual)):
            got = to_torch(got).reshape(want.shape)
            pcc = compute_pcc(want, got)
            assert pcc > PCC_THRESHOLD, f"{kind} parameter {index} PCC {pcc:.6f}"
            torch.testing.assert_close(got, want, atol=1e-4, rtol=1e-4)


class TestMoments:
    def test_mean_matches_reference(self, device, config, goldens, head):
        """Device-computed mean of the affine-transformed distribution."""
        dtype = get_ttnn_dtype(config.dtype)
        hidden = to_device(goldens.decoder_last_hidden_state, device=device, dtype=dtype)
        parameters = head.parameters(hidden)

        loc = to_device(goldens.loc.reshape(-1, 1, 1), device=device, dtype=dtype)
        scale = to_device(goldens.scale.reshape(-1, 1, 1), device=device, dtype=dtype)
        actual = to_torch(head.mean(parameters, loc=loc, scale=scale)).reshape(goldens.distribution_mean.shape)

        mse, mae, pcc = compute_metrics(goldens.distribution_mean, actual)
        assert pcc > PCC_THRESHOLD, f"mean PCC {pcc:.6f} (mse={mse:.3e}, mae={mae:.3e})"

    def test_nll_within_tolerance(self, device, config, goldens, head):
        """Negative log-likelihood must land within 5% of the reference -- a Stage 1 gate."""
        dtype = get_ttnn_dtype(config.dtype)
        hidden = to_device(goldens.decoder_last_hidden_state, device=device, dtype=dtype)
        parameters = [to_torch(p) for p in head.parameters(hidden)]

        distribution = head.torch_distribution(parameters, loc=goldens.loc, scale=goldens.scale)
        actual_nll = float((-distribution.log_prob(goldens.future_values)).mean())
        expected_nll = float(goldens.nll.mean())

        error = relative_error(expected_nll, actual_nll)
        assert error < NLL_TOLERANCE, f"NLL {actual_nll:.4f} vs reference {expected_nll:.4f} -- {error * 100:.2f}% off"


class TestDistributionCoverage:
    """End-to-end parity for the heads the published checkpoint does not carry.

    The tourism checkpoint is Student's t only, so Normal and Negative Binomial are checked
    against a reference model built with the same geometry and a randomly-initialised head.
    """

    @pytest.mark.parametrize("distribution", ["student_t", "normal", "negative_binomial"])
    def test_generate_matches_reference(self, device, distribution):
        hf_model = build_reference_model(distribution)
        local_config = config_from_hf(hf_model.config, dtype="float32")
        assert local_config.distribution_output == distribution

        model = TimeSeriesTransformer(local_config, device=device)
        result = model.load_hf_state_dict(hf_model.state_dict(), strict=True)
        assert not result["missing_keys"]

        inputs = make_inputs(hf_model.config, batch=4)
        reference = generate_mean_reference(hf_model, inputs)
        actual = model.generate(num_parallel_samples=1, mode="mean", **inputs).squeeze(1)

        assert torch.isfinite(actual).all()
        mae = float(torch.mean(torch.abs(actual - reference)))
        relative = mae / max(float(reference.abs().mean()), 1e-8)
        assert relative < 0.05, f"{distribution}: mean prediction {relative * 100:.2f}% off reference"

    @pytest.mark.parametrize("distribution", ["normal", "negative_binomial"])
    def test_sampling_is_finite_and_dispersed(self, device, distribution):
        hf_model = build_reference_model(distribution)
        model = TimeSeriesTransformer(config_from_hf(hf_model.config, dtype="float32"), device=device)
        model.load_hf_state_dict(hf_model.state_dict(), strict=True)

        torch.manual_seed(0)
        samples = model.generate(num_parallel_samples=16, mode="sample", **make_inputs(hf_model.config, batch=2))
        assert torch.isfinite(samples).all()
        assert float(samples.std(dim=1).mean()) > 0.0


class TestDeviceSampling:
    """Normal draws are built on device from pre-generated noise.

    Stage 2 asks for the sampling operation itself on device. A Normal draw is
    ``loc + scale * z``, so the randomness can be generated in bulk and uploaded once, and the
    whole sampling rollout closes on device. Student's t needs a Gamma variate whose shape is
    the predicted ``df``, and the negative binomial a Poisson-Gamma pair; neither can be
    pre-generated, so both keep the host loop.
    """

    @pytest.mark.parametrize(
        "distribution, on_device",
        [("normal", True), ("student_t", False), ("negative_binomial", False)],
    )
    def test_which_heads_sample_on_device(self, device, config, distribution, on_device):
        head = DistributionHead(
            replace(config, distribution_output=distribution), device=device, dtype=get_ttnn_dtype(config.dtype)
        )
        assert head.supports_device_sampling is on_device

    def test_device_draw_matches_the_closed_form(self, device, config, hf_state, goldens):
        """``loc + scale * z`` on device must equal the same expression on the host."""
        local = replace(config, distribution_output="normal")
        head = DistributionHead(local, device=device, dtype=get_ttnn_dtype(local.dtype))
        head.load_hf_state_dict(substate(hf_state, "parameter_projection"), strict=False)

        hidden = to_device(goldens.decoder_last_hidden_state, device=device, dtype=get_ttnn_dtype(local.dtype))
        torch.manual_seed(0)
        noise_torch = torch.randn(*goldens.decoder_last_hidden_state.shape[:2], local.input_size)
        noise = to_device(noise_torch, device=device, dtype=get_ttnn_dtype(local.dtype))

        drawn = to_torch(head.sample_from_hidden(hidden, noise))

        loc, scale = [to_torch(p) for p in head.parameters(hidden)]
        expected = loc + scale * noise_torch
        torch.testing.assert_close(drawn.reshape(expected.shape), expected, atol=1e-3, rtol=1e-3)

    def test_sampling_rollout_stays_on_device(self, device, config, hf_state, hf_model):
        """A Normal sampling forecast must use the rollout trace, not the stepped path."""
        from models.demos.time_series_transformer.reference.torch_reference import build_reference_model

        reference = build_reference_model("normal")
        local = config_from_hf(reference.config, dtype="float32", use_trace=True)
        model = TimeSeriesTransformer(local, device=device)
        model.load_hf_state_dict(reference.state_dict(), strict=True)

        try:
            output = model.generate(num_parallel_samples=8, mode="sample", **make_inputs(reference.config, batch=2))
            assert model._trace_key is not None
            assert model._trace_key[0] == "rollout_sample", "Normal sampling should close on device"
            assert torch.isfinite(output).all()
            assert float(output.std(dim=1).mean()) > 0.0, "device draws must actually vary"
        finally:
            model.release_traces()
