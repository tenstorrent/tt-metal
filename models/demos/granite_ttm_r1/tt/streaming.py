# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Streaming (online) inference for Granite TTM-R1.

Wraps ``TtnnGraniteTTMModel`` in a rolling-window buffer so that new
observations can be appended one or more timesteps at a time and a forecast
is produced for each update — without re-loading the model or re-processing
the full history from scratch.

Usage example::

    from models.demos.granite_ttm_r1.tt.streaming import GraniteTTMStreamingForecaster

    forecaster = GraniteTTMStreamingForecaster(model, config, device)

    # Feed observations one timestep at a time
    for t, obs in enumerate(sensor_stream):
        # obs: torch.Tensor [num_channels]
        forecast = forecaster.step(obs.unsqueeze(0))  # [forecast_len, C]
        print(f"t={t}: next-{config.forecast_length}-step forecast = {forecast}")

    # Or feed a batch of new timesteps at once
    forecast = forecaster.step(new_block)   # new_block: [n_new, C]
"""

from __future__ import annotations

import torch

from models.demos.granite_ttm_r1.tt.config import GraniteTTMModelConfig


class GraniteTTMStreamingForecaster:
    """Rolling-window streaming forecaster for Granite TTM-R1.

    Maintains a circular buffer of the most recent ``context_length``
    timesteps.  On each call to :meth:`step`, new observations are appended
    and the oldest are dropped; then a full model forward pass produces the
    next forecast window.

    Args:
        model: A constructed ``TtnnGraniteTTMModel`` (eager or compiled).
        config: ``GraniteTTMModelConfig`` instance for the model.
        device: TTNN device handle.
        use_compiled: If ``True`` and ``model.compile()`` has been called,
            use ``model.execute_compiled()`` for lower-latency inference.
            Falls back to eager ``model()`` if not compiled.
        dtype: PyTorch dtype for the internal buffer (default ``float32``).
    """

    def __init__(
        self,
        model,
        config: GraniteTTMModelConfig,
        device,
        *,
        use_compiled: bool = True,
        dtype: torch.dtype = torch.float32,
    ):
        self._model = model
        self._config = config
        self._device = device
        self._use_compiled = use_compiled
        self._dtype = dtype

        # Rolling buffer: [context_length, num_channels]
        self._buffer = torch.zeros(
            config.context_length,
            config.num_channels,
            dtype=dtype,
        )

        # Track how many real observations have been fed (for cold-start info)
        self._n_observations: int = 0

    # ---------------------------------------------------------------------- #
    # Public API                                                               #
    # ---------------------------------------------------------------------- #

    def step(self, new_values: torch.Tensor) -> torch.Tensor:
        """Append new observations and return the updated forecast.

        Args:
            new_values: Tensor of shape ``[n_new, num_channels]`` or
                ``[num_channels]`` (single timestep).  ``n_new`` may be any
                positive integer ≤ ``context_length``.

        Returns:
            Forecast tensor of shape ``[forecast_len, num_channels]`` in the
            model's output dtype (bfloat16 on device, then converted to
            float32 on host).
        """
        # Normalise input to [n_new, C]
        if new_values.dim() == 1:
            new_values = new_values.unsqueeze(0)
        if new_values.dim() != 2 or new_values.shape[1] != self._config.num_channels:
            raise ValueError(
                f"new_values must be [n_new, {self._config.num_channels}], " f"got {list(new_values.shape)}"
            )

        n_new = new_values.shape[0]
        if n_new > self._config.context_length:
            # Keep only the most recent context_length timesteps
            new_values = new_values[-self._config.context_length :]
            n_new = self._config.context_length

        # Shift buffer left, append new values at the right
        self._buffer = torch.roll(self._buffer, shifts=-n_new, dims=0)
        self._buffer[-n_new:] = new_values.to(self._dtype)
        self._n_observations += n_new

        return self._run_inference()

    def reset(self, initial_history: torch.Tensor | None = None) -> None:
        """Reset the rolling buffer.

        Args:
            initial_history: Optional tensor of shape
                ``[context_length, num_channels]`` to seed the buffer.
                If ``None``, the buffer is zeroed.
        """
        if initial_history is not None:
            if initial_history.shape != (self._config.context_length, self._config.num_channels):
                raise ValueError(
                    f"initial_history must be [{self._config.context_length}, "
                    f"{self._config.num_channels}], got {list(initial_history.shape)}"
                )
            self._buffer = initial_history.to(self._dtype).clone()
            self._n_observations = self._config.context_length
        else:
            self._buffer = torch.zeros(
                self._config.context_length,
                self._config.num_channels,
                dtype=self._dtype,
            )
            self._n_observations = 0

    @property
    def n_observations(self) -> int:
        """Total number of real observations fed since construction or reset."""
        return self._n_observations

    @property
    def is_warmed_up(self) -> bool:
        """True once the buffer has been filled with at least context_length real observations."""
        return self._n_observations >= self._config.context_length

    # ---------------------------------------------------------------------- #
    # Internal                                                                 #
    # ---------------------------------------------------------------------- #

    def _run_inference(self) -> torch.Tensor:
        """Run a single forward pass over the current buffer contents."""
        import ttnn
        from models.demos.granite_ttm_r1.tt.common import to_torch_tensor

        # history: [1, context_length, num_channels]
        history = self._buffer.unsqueeze(0)

        if self._use_compiled and getattr(self._model, "_is_compiled", False):
            # Trace path: single host command
            out = self._model.execute_compiled(history)
        else:
            # Eager path
            ttnn_history = ttnn.from_torch(
                history,
                device=self._device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            out = self._model(ttnn_history, device=self._device)

        # Return [forecast_len, num_channels] on CPU as float32
        result = to_torch_tensor(out).float()
        # out shape is [1, forecast_len, C] → squeeze batch dim
        if result.dim() == 3 and result.shape[0] == 1:
            result = result.squeeze(0)
        return result
