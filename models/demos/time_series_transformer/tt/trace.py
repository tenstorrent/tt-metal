# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Trace capture and replay for autoregressive generation.

A decode step is roughly 95 TTNN ops on this model, so a forecast is bound by per-op dispatch
and host round-trips rather than arithmetic. These two runners are what removes that cost;
which one applies depends on whether the sampling feedback can be closed on device.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

import ttnn

from .ops import slice_last_step, to_device, to_torch

if TYPE_CHECKING:
    from .model import TimeSeriesTransformer


class TracedDecodeRunner:
    """Replays the decoder from a single captured trace, once per decode step.

    A decode step is roughly 95 TTNN ops on a model this small, so wall-clock is dominated by
    per-op dispatch rather than arithmetic: ~6.1 ms eager against ~0.65 ms replayed.

    tt-metal locks device allocations for as long as any trace exists, so capturing a
    separate trace per step is not viable -- the second capture allocates while the first is
    live. Instead one trace covers the whole prediction window: the decoder runs over all
    ``prediction_length`` positions behind a fixed causal mask, and the host reads out the
    column it needs. Positions past the current step contribute nothing because the mask
    hides them, so the untouched (zero) rows of the input buffer are harmless.

    That makes the traced path O(horizon^2) in arithmetic where the KV-cache path is O(n),
    but at d_model=26 with a 24-step horizon the recompute is far cheaper than the dispatch
    it saves. For long horizons the cached eager path in ``Decoder`` remains the right choice.
    """

    def __init__(self, model: "TimeSeriesTransformer", rows: int):
        self.model = model
        self.rows = rows
        self.device = model.device
        self.dtype = model.dtype
        config = model.config

        self.decoder_inputs = to_device(
            torch.zeros(rows, config.prediction_length, config.feature_size),
            device=self.device,
            dtype=self.dtype,
        )
        self.encoder_hidden = to_device(
            torch.zeros(rows, config.context_length, config.d_model),
            device=self.device,
            dtype=self.dtype,
        )
        self.parameters: Optional[ttnn.Tensor] = None
        self.trace_id: object = None
        self._capture()

    def _run(self) -> ttnn.Tensor:
        hidden = self.model.decoder(self.decoder_inputs, self.encoder_hidden)
        parameters = self.model.distribution_head.parameters(hidden)
        # Concatenate on device so each step costs one readback rather than one per parameter.
        return ttnn.concat(parameters, dim=-1)

    def _capture(self) -> None:
        # Compile kernels and populate the causal-mask and positional-embedding caches first:
        # a capture must not reach the host or allocate anything it has not already seen.
        self._run()
        ttnn.synchronize_device(self.device)

        self.trace_id = ttnn.begin_trace_capture(self.device, cq_id=0)
        self.parameters = self._run()
        ttnn.end_trace_capture(self.device, self.trace_id, cq_id=0)
        ttnn.synchronize_device(self.device)

    def prepare(self, encoder_hidden_states: ttnn.Tensor) -> None:
        """Point the trace at this rollout's encoder output, keeping the buffer address."""
        ttnn.copy(encoder_hidden_states, self.encoder_hidden)

    def step(self, position: int, decoder_inputs: torch.Tensor) -> list[torch.Tensor]:
        """Run the window and return the distribution parameters at ``position``."""
        host_tensor = ttnn.from_torch(decoder_inputs.contiguous(), dtype=self.dtype, layout=ttnn.TILE_LAYOUT)
        ttnn.copy_host_to_device_tensor(host_tensor, self.decoder_inputs)
        ttnn.execute_trace(self.device, self.trace_id, cq_id=0, blocking=False)
        stacked = to_torch(self.parameters)[:, position : position + 1, :]
        return list(torch.split(stacked, self.model.config.input_size, dim=-1))

    def release(self) -> None:
        if self.trace_id is not None:
            ttnn.release_trace(self.device, self.trace_id)
            self.trace_id = None
        self.parameters = None


class TracedRolloutRunner:
    """Captures an entire mean-mode rollout -- all ``prediction_length`` steps -- as one trace.

    :class:`TracedDecodeRunner` still pays a host round-trip per step, because the predicted
    value feeds into the next step's lag window. Measured, those round-trips are the bulk of
    the latency: 24 decoder steps replay in ~18 ms of device time but the stepped loop takes
    ~46 ms end to end.

    Closing the loop on device removes them. The trick is to unroll: with the loop written out
    at capture time, every step's lag offsets, sequence lengths and mask shapes are constants
    again, so the growing ``running`` and ``window`` tensors are perfectly traceable even
    though their shapes differ from step to step.

    Two details make it cheap. Gathering the lag window is a single matmul against a constant
    one-hot selector rather than one slice per lag. And because the running series is
    normalized, the value fed back is exactly the distribution's pre-affine mean -- the scaler
    statistics never have to reach the device.

    Mean mode only: sampling a Student's t needs a Gamma variate whose shape parameter is
    data-dependent, so it cannot be drawn on device or pre-generated.
    """

    def __init__(self, model: "TimeSeriesTransformer", rows: int):
        self.model = model
        self.rows = rows
        self.device = model.device
        self.dtype = model.dtype
        config = model.config
        self.num_lags = len(config.lags_sequence)

        # Written per rollout; the trace reads them at fixed addresses.
        self.channels = config.input_size
        # Channel-major: matmul against a selector then yields (rows, channels, lags), which
        # flattens to the channel-major/lag-minor order get_lagged_subsequences produces.
        self.running_init = to_device(
            torch.zeros(rows, self.channels, config.past_length), device=self.device, dtype=self.dtype
        )
        self.covariates = to_device(
            torch.zeros(rows, config.prediction_length, config.num_covariate_features),
            device=self.device,
            dtype=self.dtype,
        )
        # The encoder runs inside the trace too, so what crosses the boundary is its input
        # rather than its output: ~90 ops that would otherwise pay full host dispatch.
        self.encoder_inputs = to_device(
            torch.zeros(rows, config.context_length, config.feature_size),
            device=self.device,
            dtype=self.dtype,
        )

        # One selector per step: at step k the series is past_length + k long, and lag l sits
        # at index past_length + k - l.
        self.selectors = []
        for step in range(config.prediction_length):
            length = config.past_length + step
            selector = torch.zeros(length, self.num_lags)
            for column, lag in enumerate(config.lags_sequence):
                selector[length - lag, column] = 1.0
            self.selectors.append(to_device(selector, device=self.device, dtype=self.dtype))

        self.output: Optional[ttnn.Tensor] = None
        self.trace_id: object = None
        self._capture()

    def _rollout(self) -> ttnn.Tensor:
        config = self.model.config
        encoder_hidden = self.model.encoder(self.encoder_inputs)
        # The encoder output is constant across the forecast, so its cross-attention keys and
        # values are projected on the first step and reused by the remaining twenty-three.
        cross_caches = self.model.decoder.new_cross_caches()
        running = self.running_init
        window = None
        collected = []

        for step in range(config.prediction_length):
            lags = ttnn.matmul(running, self.selectors[step], dtype=self.dtype)
            lags = ttnn.reshape(lags, (self.rows, 1, self.channels * self.num_lags))
            covariate = ttnn.slice(self.covariates, [0, step, 0], [self.rows, step + 1, config.num_covariate_features])
            row = ttnn.concat([lags, covariate], dim=-1)

            window = row if window is None else ttnn.concat([window, row], dim=1)
            hidden = self.model.decoder(window, encoder_hidden, cross_caches=cross_caches)

            next_value = self.model.distribution_head.base_mean_from_hidden(slice_last_step(hidden))
            collected.append(next_value)
            # next_value is (rows, 1, channels); the running series is channel-major.
            appended = ttnn.permute(ttnn.reshape(next_value, (self.rows, 1, self.channels)), (0, 2, 1))
            running = ttnn.concat([running, appended], dim=-1)

        return ttnn.concat(collected, dim=1)

    def _capture(self) -> None:
        # Compile every kernel and fill the mask/positional caches before capturing.
        self._rollout()
        ttnn.synchronize_device(self.device)

        self.trace_id = ttnn.begin_trace_capture(self.device, cq_id=0)
        self.output = self._rollout()
        ttnn.end_trace_capture(self.device, self.trace_id, cq_id=0)
        ttnn.synchronize_device(self.device)

    def run(
        self,
        *,
        running: torch.Tensor,
        covariates: torch.Tensor,
        encoder_inputs: torch.Tensor,
    ) -> torch.Tensor:
        """Execute encoder and rollout together.

        Returns normalized predictions, ``(rows, horizon)`` or ``(rows, horizon, channels)``.
        ``running`` must be channel-major, ``(rows, channels, past_length)``.
        """
        staged = (
            (running, self.running_init),
            (covariates, self.covariates),
            (encoder_inputs, self.encoder_inputs),
        )
        for host_value, buffer in staged:
            uploaded = ttnn.from_torch(host_value.contiguous(), dtype=self.dtype, layout=ttnn.TILE_LAYOUT)
            ttnn.copy_host_to_device_tensor(uploaded, buffer)

        ttnn.execute_trace(self.device, self.trace_id, cq_id=0, blocking=False)
        horizon = self.model.config.prediction_length
        predictions = to_torch(self.output)
        if self.channels > 1:
            return predictions.reshape(self.rows, horizon, self.channels)
        return predictions.reshape(self.rows, horizon)

    def release(self) -> None:
        if self.trace_id is not None:
            ttnn.release_trace(self.device, self.trace_id)
            self.trace_id = None
        self.output = None


__all__ = ["TracedDecodeRunner", "TracedRolloutRunner"]
