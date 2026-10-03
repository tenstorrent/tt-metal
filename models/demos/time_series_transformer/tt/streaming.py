# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Online (streaming) forecasting.

Batch forecasting hands the model a fixed window and asks for one forecast. Streaming instead
keeps a rolling window that advances as observations arrive, so a forecast can be re-issued
after every new sample without rebuilding inputs from scratch.

The window length is fixed at ``past_length``, which is what makes this cheap on device: every
forecast presents the same shapes, so the captured trace is reused across updates rather than
recaptured. Advancing the window is a host-side roll of a few kilobytes.
"""

from __future__ import annotations

from typing import Optional

import torch

from .model import TimeSeriesTransformer


class StreamingForecaster:
    """A rolling window over one or more series, re-forecastable after each observation.

    ``past_values`` seeds the window and must be exactly ``past_length`` steps; each
    :meth:`observe` drops the oldest step and appends a new one.
    """

    def __init__(
        self,
        model: TimeSeriesTransformer,
        *,
        past_values: torch.Tensor,
        past_time_features: torch.Tensor,
        past_observed_mask: Optional[torch.Tensor] = None,
        static_categorical_features: Optional[torch.Tensor] = None,
        static_real_features: Optional[torch.Tensor] = None,
    ):
        config = model.config
        expected = config.past_length
        if past_values.shape[1] != expected:
            raise ValueError(f"Streaming window must be exactly {expected} steps, got {past_values.shape[1]}.")
        if past_time_features.shape[1] != expected:
            raise ValueError(f"Time features must cover {expected} steps, got {past_time_features.shape[1]}.")

        self.model = model
        self.past_values = past_values.clone()
        self.past_time_features = past_time_features.clone()
        self.past_observed_mask = (
            torch.ones_like(past_values) if past_observed_mask is None else past_observed_mask.clone()
        )
        self.static_categorical_features = static_categorical_features
        self.static_real_features = static_real_features
        self.steps_observed = 0

    @property
    def batch(self) -> int:
        return int(self.past_values.shape[0])

    def observe(
        self,
        value: torch.Tensor,
        time_feature: torch.Tensor,
        *,
        observed: Optional[torch.Tensor] = None,
    ) -> None:
        """Advance the window by one step.

        ``value`` is ``(batch,)`` for a univariate series or ``(batch, channels)`` otherwise;
        ``time_feature`` is ``(batch, num_time_features)``.
        """
        step_values = value.reshape(self.batch, 1, *self.past_values.shape[2:])
        step_mask = torch.ones_like(step_values) if observed is None else observed.reshape(step_values.shape)
        step_time = time_feature.reshape(self.batch, 1, self.past_time_features.shape[2])

        self.past_values = torch.cat((self.past_values[:, 1:], step_values), dim=1)
        self.past_observed_mask = torch.cat((self.past_observed_mask[:, 1:], step_mask), dim=1)
        self.past_time_features = torch.cat((self.past_time_features[:, 1:], step_time), dim=1)
        self.steps_observed += 1

    def window(self) -> dict[str, torch.Tensor]:
        """The current window, in the form :meth:`TimeSeriesTransformer.generate` expects."""
        inputs = {
            "past_values": self.past_values,
            "past_time_features": self.past_time_features,
            "past_observed_mask": self.past_observed_mask,
        }
        if self.static_categorical_features is not None:
            inputs["static_categorical_features"] = self.static_categorical_features
        if self.static_real_features is not None:
            inputs["static_real_features"] = self.static_real_features
        return inputs

    def forecast(
        self,
        future_time_features: torch.Tensor,
        *,
        num_parallel_samples: int = 1,
        mode: str = "mean",
    ) -> torch.Tensor:
        """Forecast from the current window; shapes match :meth:`generate`."""
        return self.model.generate(
            future_time_features=future_time_features,
            num_parallel_samples=num_parallel_samples,
            mode=mode,
            **self.window(),
        )


__all__ = ["StreamingForecaster"]
