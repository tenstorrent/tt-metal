# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Multivariate (``input_size > 1``) parity against HuggingFace.

The published checkpoint is univariate, so these run against reference models built with the
same geometry and a wider input. Shapes and element ordering are the whole point here: the lag
window flattens channel-major, and the distribution treats channels as an event dimension.
"""


import pytest
import torch

from models.demos.time_series_transformer.reference.torch_reference import (
    build_reference_model,
    compute_metrics,
    generate_mean_reference,
    make_inputs,
)
from models.demos.time_series_transformer.tt.config import config_from_hf
from models.demos.time_series_transformer.tt.inputs import get_lagged_subsequences, get_latest_lags
from models.demos.time_series_transformer.tt.model import TimeSeriesTransformer

CHANNELS = 3
BATCH = 2
ACCURACY_TOLERANCE = 0.05


@pytest.fixture(scope="module")
def multivariate_reference():
    return build_reference_model("student_t", input_size=CHANNELS)


def build(device, hf_model, **overrides):
    config = config_from_hf(hf_model.config, dtype="float32", **overrides)
    model = TimeSeriesTransformer(config, device=device)
    result = model.load_hf_state_dict(hf_model.state_dict(), strict=True)
    assert not result["missing_keys"]
    return model, config


class TestLagGather:
    """The fast single-row gather must equal the general one, element for element."""

    @pytest.mark.parametrize("channels", [1, CHANNELS])
    def test_latest_lags_matches_full_gather(self, channels):
        lags = (1, 2, 5, 11)
        torch.manual_seed(0)
        shape = (BATCH, 40) if channels == 1 else (BATCH, 40, channels)
        sequence = torch.randn(*shape)

        full = get_lagged_subsequences(sequence, subsequences_length=1, lags_sequence=lags, shift=1)
        expected = full.reshape(full.shape[0], full.shape[1], -1)
        actual = get_latest_lags(sequence, lags)

        assert actual.shape == expected.shape
        torch.testing.assert_close(actual, expected)


class TestMultivariateConfig:
    def test_feature_size_matches_huggingface(self, multivariate_reference):
        config = config_from_hf(multivariate_reference.config, dtype="float32")
        assert config.input_size == CHANNELS
        assert config.is_multivariate
        # HF counts log1p(|loc|) and log(scale) per channel.
        assert config.feature_size == multivariate_reference.config.feature_size


class TestMultivariateGenerate:
    def test_output_shape(self, device, multivariate_reference):
        model, config = build(device, multivariate_reference)
        inputs = make_inputs(multivariate_reference.config, batch=BATCH)
        samples = 4

        output = model.generate(num_parallel_samples=samples, mode="sample", **inputs)

        assert output.shape == (BATCH, samples, config.prediction_length, CHANNELS)
        assert torch.isfinite(output).all()

    def test_matches_reference(self, device, multivariate_reference):
        """Mean-mode rollout against the HuggingFace rollout, channel by channel."""
        model, _ = build(device, multivariate_reference)
        inputs = make_inputs(multivariate_reference.config, batch=BATCH)

        reference = generate_mean_reference(multivariate_reference, inputs)
        actual = model.generate(num_parallel_samples=1, mode="mean", **inputs).squeeze(1)

        assert actual.shape == reference.shape
        mse, mae, pcc = compute_metrics(reference, actual)
        relative = mae / max(float(reference.abs().mean()), 1e-8)
        assert relative < ACCURACY_TOLERANCE, f"multivariate MAE {relative * 100:.2f}% off (mse={mse:.3e})"

    def test_traced_matches_eager(self, device, multivariate_reference):
        """The device-side rollout keeps the channel axis in the right order."""
        eager, _ = build(device, multivariate_reference, use_trace=False)
        inputs = make_inputs(multivariate_reference.config, batch=BATCH)
        expected = eager.generate(num_parallel_samples=1, mode="mean", **inputs)

        traced, _ = build(device, multivariate_reference, use_trace=True)
        try:
            actual = traced.generate(num_parallel_samples=1, mode="mean", **inputs)
        finally:
            traced.release_traces()

        assert actual.shape == expected.shape
        _, _, pcc = compute_metrics(expected, actual)
        assert pcc > 0.999, f"traced multivariate rollout diverged: PCC {pcc:.6f}"

    def test_observed_mask_is_honoured(self, device, multivariate_reference):
        """Masking a channel must change the scaler statistics, and match HF when it does.

        The mask has to fall inside the last ``context_length`` steps: that slice is the only
        part the scaler reads, so masking earlier history is a no-op by construction.
        """
        model, config = build(device, multivariate_reference)
        inputs = make_inputs(multivariate_reference.config, batch=BATCH)

        masked = dict(inputs)
        mask = inputs["past_observed_mask"].clone()
        mask[:, -(config.context_length // 2) :, 0] = 0.0
        masked["past_observed_mask"] = mask

        reference = generate_mean_reference(multivariate_reference, masked)
        actual = model.generate(num_parallel_samples=1, mode="mean", **masked).squeeze(1)

        _, mae, _ = compute_metrics(reference, actual)
        relative = mae / max(float(reference.abs().mean()), 1e-8)
        assert relative < ACCURACY_TOLERANCE, f"masked multivariate MAE {relative * 100:.2f}% off"

        unmasked = model.generate(num_parallel_samples=1, mode="mean", **inputs).squeeze(1)
        assert not torch.allclose(actual, unmasked), "observed mask had no effect on the forecast"


class TestCorrelatedChannels:
    """Parity on channels that are correlated and differently scaled.

    ``make_inputs`` draws every channel i.i.d. from the same distribution, which makes the
    channels statistically interchangeable: a bug that transposed or rotated the channel axis
    would still produce a plausible forecast and still pass an aggregate PCC. These build a
    series where channel identity is unmistakable -- a shared driver, per-channel signs and
    magnitudes three orders of magnitude apart -- so any channel mixing shows up immediately.
    """

    @staticmethod
    def correlated_inputs(hf_config, *, seed: int = 11):
        inputs = make_inputs(hf_config, batch=BATCH, seed=seed)
        generator = torch.Generator().manual_seed(seed)
        length = inputs["past_values"].shape[1]

        driver = torch.rand(BATCH, length, generator=generator) * 10.0 + 20.0
        noise = torch.randn(BATCH, length, CHANNELS, generator=generator) * 0.05
        # Correlated with the driver, but each channel has its own sign and scale.
        weights = torch.tensor([1.0, 0.8, -0.5])
        offsets = torch.tensor([100.0, 10.0, 5000.0])
        scales = torch.tensor([10.0, 1.0, 500.0])
        values = offsets + scales * (driver.unsqueeze(-1) * weights + noise)

        inputs["past_values"] = values
        return inputs

    def test_matches_reference_channel_by_channel(self, device, multivariate_reference):
        model, _ = build(device, multivariate_reference)
        inputs = self.correlated_inputs(multivariate_reference.config)

        reference = generate_mean_reference(multivariate_reference, inputs)
        actual = model.generate(num_parallel_samples=1, mode="mean", **inputs).squeeze(1)
        assert actual.shape == reference.shape

        # Per channel, not pooled: pooling lets a large channel hide a small one.
        for channel in range(CHANNELS):
            want, got = reference[..., channel], actual[..., channel]
            _, mae, pcc = compute_metrics(want, got)
            relative = mae / max(float(want.abs().mean()), 1e-8)
            assert relative < ACCURACY_TOLERANCE, f"channel {channel} MAE {relative * 100:.2f}% off (PCC {pcc:.4f})"

    def test_permuting_channels_still_matches_reference(self, device, multivariate_reference):
        """Reorder the input channels and re-check parity against HuggingFace.

        The model is *not* permutation-equivariant -- the lag window is projected by a dense
        layer over the flattened (channel, lag) vector, so reordering the inputs legitimately
        changes the forecast. What must hold is that the TT model changes the same way the
        reference does: any channel the device mixed up would land on different projection
        weights here than it did unpermuted, and the two runs would disagree.
        """
        model, _ = build(device, multivariate_reference)
        inputs = self.correlated_inputs(multivariate_reference.config)
        permutation = [2, 0, 1]

        permuted = dict(inputs)
        permuted["past_values"] = inputs["past_values"][..., permutation]
        if inputs["past_observed_mask"].dim() == 3:
            permuted["past_observed_mask"] = inputs["past_observed_mask"][..., permutation]

        reference = generate_mean_reference(multivariate_reference, permuted)
        actual = model.generate(num_parallel_samples=1, mode="mean", **permuted).squeeze(1)

        for channel in range(CHANNELS):
            want, got = reference[..., channel], actual[..., channel]
            _, mae, _ = compute_metrics(want, got)
            relative = mae / max(float(want.abs().mean()), 1e-8)
            assert relative < ACCURACY_TOLERANCE, f"permuted channel {channel} MAE {relative * 100:.2f}% off"
