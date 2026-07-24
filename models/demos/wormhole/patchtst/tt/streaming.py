# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

import ttnn
from models.demos.wormhole.patchtst.tt.common import TT_DTYPE
from models.demos.wormhole.patchtst.tt.model import PatchTSTTTNNModel, PreparedEncoderInput


class HostFullRerunPatchTSTStreamer:
    """Legacy host-driven rolling-window helper for parity/debug use.

    This path exists as the reference streaming baseline. Every step updates the host-side
    context window and reruns the full PatchTST forward pass on that latest slice.
    """

    def __init__(
        self,
        model: PatchTSTTTNNModel,
        initial_context: torch.Tensor,
        initial_observed_mask: torch.Tensor | None = None,
    ) -> None:
        self.model = model
        self.context_length = int(model.cfg.context_length)
        self.context = initial_context.detach().clone()
        self.observed_mask = (
            initial_observed_mask.detach().clone()
            if initial_observed_mask is not None
            else torch.ones_like(self.context, dtype=torch.float32)
        )

        if self.context.ndim != 3:
            raise ValueError(f"initial_context must be rank-3 [B, T, C], got shape={tuple(self.context.shape)}")
        if int(self.context.shape[1]) != self.context_length:
            raise ValueError(
                f"initial_context length ({self.context.shape[1]}) must equal configured context_length ({self.context_length})"
            )
        if self.observed_mask.shape != self.context.shape:
            raise ValueError(
                "initial_observed_mask shape must match initial_context shape, "
                f"got mask={tuple(self.observed_mask.shape)} context={tuple(self.context.shape)}"
            )

    def step(self, new_values: torch.Tensor, task: str = "forecast") -> torch.Tensor:
        if new_values.ndim != 3:
            raise ValueError(f"new_values must be rank-3 [B, T, C], got shape={tuple(new_values.shape)}")
        if int(new_values.shape[0]) != int(self.context.shape[0]):
            raise ValueError(
                f"Batch mismatch: new_values batch={new_values.shape[0]} does not match context batch={self.context.shape[0]}"
            )
        if int(new_values.shape[2]) != int(self.context.shape[2]):
            raise ValueError(
                "Channel mismatch: "
                f"new_values channels={new_values.shape[2]} does not match context channels={self.context.shape[2]}"
            )

        self.context = torch.cat([self.context, new_values], dim=1)[:, -self.context_length :, :].contiguous()
        new_observed = torch.ones_like(new_values, dtype=self.observed_mask.dtype)
        self.observed_mask = torch.cat([self.observed_mask, new_observed], dim=1)[
            :, -self.context_length :, :
        ].contiguous()

        output = self.model.forward(
            past_values=self.context,
            past_observed_mask=self.observed_mask,
            task=task,
        )
        prediction = output.prediction
        if not isinstance(prediction, torch.Tensor):
            raise TypeError("Legacy streaming baseline expects a tensor forecast prediction.")
        return prediction


class CachedForecastStreamer:
    """Stateful forecast-only streaming helper with persistent device buffers.

    This implementation is materially different from the legacy full-rerun helper:
    - it keeps a rolling raw context window on host,
    - it updates exact sliding-window normalization statistics incrementally,
    - it reuses a persistent device-side hidden-state buffer between steps,
    - it optionally replays a captured encoder/head trace on that stable buffer.

    It does *not* claim decoder-style autoregressive KV-cache semantics. PatchTST is an
    encoder-only sliding-window forecaster, so the cached state here is:
    - rolling raw context,
    - exact normalization statistics,
    - persistent pre-encoder hidden/input buffers,
    - optional traced replay state.
    The encoder itself is still executed every step on the latest context window.
    """

    def __init__(
        self,
        model: PatchTSTTTNNModel,
        initial_context: torch.Tensor,
        initial_observed_mask: torch.Tensor | None = None,
        *,
        use_trace: bool = True,
    ) -> None:
        self.model = model
        self.context_length = int(model.cfg.context_length)
        self.context = initial_context.detach().clone()
        self.observed_mask = (
            initial_observed_mask.detach().clone()
            if initial_observed_mask is not None
            else torch.ones_like(self.context, dtype=torch.float32)
        )
        self.use_trace = bool(use_trace)
        self.trace_id: int | None = None
        self.traced_output = None
        self.prepared: PreparedEncoderInput | None = None
        self._minimum_scale = float(model.ref_core.scaler.scaler.minimum_scale)
        self._sum = (self.context * self.observed_mask).sum(dim=1, keepdim=True).to(torch.float32)
        self._sum_sq = ((self.context.to(torch.float32) ** 2) * self.observed_mask).sum(dim=1, keepdim=True)
        self._count = self.observed_mask.sum(dim=1, keepdim=True).clamp_min(1.0).to(torch.float32)
        self._validate_inputs()
        self._initialize_runtime()

    def _validate_inputs(self) -> None:
        if self.model.reference.task != "forecast":
            raise ValueError("Cached streaming is only supported for the forecasting task.")
        if self.model.cfg.channel_mode != "independent":
            raise ValueError("Cached streaming is only supported for channel_mode='independent'.")
        if not bool(self.model.cfg.share_embedding):
            raise ValueError("Cached streaming is only supported for share_embedding=True.")
        if bool(self.model.ref_core.do_mask_input):
            raise ValueError("Cached streaming does not support masked-pretraining checkpoints.")
        if self.context.ndim != 3:
            raise ValueError(f"initial_context must be rank-3 [B, T, C], got shape={tuple(self.context.shape)}")
        if int(self.context.shape[1]) != self.context_length:
            raise ValueError(
                f"initial_context length ({self.context.shape[1]}) must equal configured context_length ({self.context_length})"
            )
        if self.observed_mask.shape != self.context.shape:
            raise ValueError(
                "initial_observed_mask shape must match initial_context shape, "
                f"got mask={tuple(self.observed_mask.shape)} context={tuple(self.context.shape)}"
            )
        if not torch.all(self.observed_mask == 1):
            raise ValueError("Cached streaming currently supports fully observed inputs only.")

    def _current_loc_scale(self) -> tuple[torch.Tensor, torch.Tensor]:
        denominator = self._count.clamp_min(1.0)
        loc = self._sum / denominator
        variance = (self._sum_sq / denominator) - torch.square(loc)
        variance = torch.clamp(variance, min=0.0)
        scale = torch.sqrt(variance + self._minimum_scale)
        return loc.to(torch.float32), scale.to(torch.float32)

    def _initialize_runtime(self) -> None:
        loc, scale = self._current_loc_scale()
        self.prepared = self.model.prepare_hidden_input(self.context, self.observed_mask)

        warm_output = self.model.forward_from_hidden_tt(self.prepared, task="forecast")
        warm_output.release()
        ttnn.synchronize_device(self.model.device)

        if self.use_trace:
            trace_id = ttnn.begin_trace_capture(self.model.device, cq_id=0)
            self.traced_output = self.model.forward_from_hidden_tt(self.prepared, task="forecast")
            ttnn.end_trace_capture(self.model.device, trace_id, cq_id=0)
            ttnn.synchronize_device(self.model.device)
            self.trace_id = trace_id

        # Copy the exact host-prepared hidden state into the persistent device buffer so the
        # cached path always starts from the same semantics as the eager reference preprocess.
        hidden_host, _, _ = self.model.prepare_hidden_input_host(self.context, self.observed_mask, loc=loc, scale=scale)
        hidden_host_tt = ttnn.from_torch(hidden_host, dtype=TT_DTYPE, layout=ttnn.TILE_LAYOUT)
        try:
            ttnn.copy_host_to_device_tensor(hidden_host_tt, self.prepared.hidden_state, 1)
            ttnn.synchronize_device(self.model.device)
        finally:
            ttnn.deallocate(hidden_host_tt)

    def close(self) -> None:
        if self.trace_id is not None:
            ttnn.release_trace(self.model.device, self.trace_id)
            self.trace_id = None
        if self.traced_output is not None:
            self.traced_output.release()
            self.traced_output = None
        if self.prepared is not None:
            self.prepared.release()
            self.prepared = None

    def _update_context_state(self, new_values: torch.Tensor) -> None:
        if new_values.ndim != 3:
            raise ValueError(f"new_values must be rank-3 [B, T, C], got shape={tuple(new_values.shape)}")
        if int(new_values.shape[0]) != int(self.context.shape[0]):
            raise ValueError(
                f"Batch mismatch: new_values batch={new_values.shape[0]} does not match context batch={self.context.shape[0]}"
            )
        if int(new_values.shape[2]) != int(self.context.shape[2]):
            raise ValueError(
                "Channel mismatch: "
                f"new_values channels={new_values.shape[2]} does not match context channels={self.context.shape[2]}"
            )
        step = int(new_values.shape[1])
        dropped = self.context[:, :step, :]
        self.context = torch.cat([self.context, new_values], dim=1)[:, -self.context_length :, :].contiguous()
        self._sum = (
            self._sum
            - dropped.sum(dim=1, keepdim=True).to(torch.float32)
            + new_values.sum(dim=1, keepdim=True).to(torch.float32)
        )
        self._sum_sq = (
            self._sum_sq
            - (dropped.to(torch.float32) ** 2).sum(dim=1, keepdim=True)
            + (new_values.to(torch.float32) ** 2).sum(dim=1, keepdim=True)
        )
        self._count = torch.full_like(self._count, float(self.context_length))

    def step(self, new_values: torch.Tensor) -> torch.Tensor:
        if self.prepared is None:
            raise RuntimeError("CachedForecastStreamer is closed.")

        self._update_context_state(new_values)
        loc, scale = self._current_loc_scale()
        hidden_host, _, _ = self.model.prepare_hidden_input_host(self.context, self.observed_mask, loc=loc, scale=scale)
        hidden_host_tt = ttnn.from_torch(hidden_host, dtype=TT_DTYPE, layout=ttnn.TILE_LAYOUT)
        try:
            ttnn.copy_host_to_device_tensor(hidden_host_tt, self.prepared.hidden_state, 1)
            ttnn.synchronize_device(self.model.device)
        finally:
            ttnn.deallocate(hidden_host_tt)

        if self.use_trace:
            if self.trace_id is None or self.traced_output is None:
                raise RuntimeError("CachedForecastStreamer trace state was not initialized.")
            ttnn.execute_trace(self.model.device, self.trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(self.model.device)
            prediction = ttnn.to_torch(ttnn.from_device(self.traced_output.prediction))
        else:
            output = self.model.forward_from_hidden_tt(self.prepared, task="forecast")
            try:
                ttnn.synchronize_device(self.model.device)
                prediction = ttnn.to_torch(ttnn.from_device(output.prediction))
            finally:
                output.release()
        return prediction * scale + loc


# Backward-compatible alias while downstream tests and helper imports migrate to the explicit names.
RollingPatchTSTStreamer = HostFullRerunPatchTSTStreamer
