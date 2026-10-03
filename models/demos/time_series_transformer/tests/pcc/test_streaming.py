# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Online forecasting: a rolling window must agree with the equivalent batch forecast."""

from dataclasses import replace

import pytest
import torch

from models.demos.time_series_transformer.reference.torch_reference import compute_metrics, make_inputs
from models.demos.time_series_transformer.tt.model import TimeSeriesTransformer
from models.demos.time_series_transformer.tt.streaming import StreamingForecaster

BATCH = 2
STEPS = 5


@pytest.fixture(scope="module")
def streaming_model(device, config, hf_state):
    model = TimeSeriesTransformer(replace(config, use_trace=True), device=device)
    model.load_hf_state_dict(hf_state, strict=True)
    yield model
    model.release_traces()


def seeded(hf_config, *, extra: int):
    """A window plus ``extra`` further observations, so the two paths can be lined up."""
    generator = torch.Generator().manual_seed(7)
    past = int(hf_config.context_length) + int(max(hf_config.lags_sequence))
    total = past + extra
    values = torch.rand(BATCH, total, generator=generator) * 100 + 50
    time_features = torch.randn(BATCH, total, hf_config.num_time_features, generator=generator)
    future_time = torch.randn(BATCH, hf_config.prediction_length, hf_config.num_time_features, generator=generator)
    statics = make_inputs(hf_config, batch=BATCH)
    return values, time_features, future_time, statics, past


class TestStreamingForecaster:
    def test_matches_batch_forecast_after_updates(self, streaming_model, hf_model):
        """Rolling forward N steps must equal forecasting the window that ends at that step."""
        values, time_features, future_time, statics, past = seeded(hf_model.config, extra=STEPS)

        stream = StreamingForecaster(
            streaming_model,
            past_values=values[:, :past],
            past_time_features=time_features[:, :past],
            static_categorical_features=statics.get("static_categorical_features"),
            static_real_features=statics.get("static_real_features"),
        )
        for step in range(STEPS):
            stream.observe(values[:, past + step], time_features[:, past + step])

        streamed = stream.forecast(future_time, num_parallel_samples=1, mode="mean")

        # The same window presented directly.
        direct = streaming_model.generate(
            past_values=values[:, STEPS : past + STEPS],
            past_time_features=time_features[:, STEPS : past + STEPS],
            past_observed_mask=torch.ones(BATCH, past),
            future_time_features=future_time,
            static_categorical_features=statics.get("static_categorical_features"),
            static_real_features=statics.get("static_real_features"),
            num_parallel_samples=1,
            mode="mean",
        )

        assert streamed.shape == direct.shape
        _, _, pcc = compute_metrics(direct, streamed)
        assert pcc > 0.999, f"streaming forecast diverged from the batch forecast: PCC {pcc:.6f}"

    def test_window_length_is_fixed(self, streaming_model, hf_model):
        values, time_features, _, statics, past = seeded(hf_model.config, extra=STEPS)
        stream = StreamingForecaster(
            streaming_model,
            past_values=values[:, :past],
            past_time_features=time_features[:, :past],
            static_categorical_features=statics.get("static_categorical_features"),
            static_real_features=statics.get("static_real_features"),
        )
        for step in range(STEPS):
            stream.observe(values[:, past + step], time_features[:, past + step])
            assert stream.past_values.shape == (BATCH, past)
            assert stream.past_time_features.shape[1] == past
        assert stream.steps_observed == STEPS

    def test_trace_is_reused_across_updates(self, streaming_model, hf_model):
        """The point of a fixed window: forecasting again must not recapture."""
        values, time_features, future_time, statics, past = seeded(hf_model.config, extra=STEPS)
        stream = StreamingForecaster(
            streaming_model,
            past_values=values[:, :past],
            past_time_features=time_features[:, :past],
            static_categorical_features=statics.get("static_categorical_features"),
            static_real_features=statics.get("static_real_features"),
        )

        stream.forecast(future_time, mode="mean")
        captured = streaming_model._trace_runner
        assert captured is not None

        for step in range(STEPS):
            stream.observe(values[:, past + step], time_features[:, past + step])
            stream.forecast(future_time, mode="mean")
            assert streaming_model._trace_runner is captured, "streaming update forced a recapture"

    def test_rejects_a_wrongly_sized_window(self, streaming_model, hf_model, expect_error):
        values, time_features, _, statics, past = seeded(hf_model.config, extra=STEPS)
        with expect_error(ValueError, "exactly"):
            StreamingForecaster(
                streaming_model,
                past_values=values[:, : past - 1],
                past_time_features=time_features[:, : past - 1],
            )

    def test_observed_mask_streams_too(self, streaming_model, hf_model):
        """An unobserved arrival must reach the scaler like any other masked step."""
        values, time_features, future_time, statics, past = seeded(hf_model.config, extra=STEPS)

        def run(observed_flag: float) -> torch.Tensor:
            stream = StreamingForecaster(
                streaming_model,
                past_values=values[:, :past],
                past_time_features=time_features[:, :past],
                static_categorical_features=statics.get("static_categorical_features"),
                static_real_features=statics.get("static_real_features"),
            )
            for step in range(STEPS):
                stream.observe(
                    values[:, past + step],
                    time_features[:, past + step],
                    observed=torch.full((BATCH,), observed_flag),
                )
            return stream.forecast(future_time, mode="mean")

        assert not torch.allclose(run(1.0), run(0.0)), "streamed observed mask had no effect"
