# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Optional

import torch

import ttnn

from .config import TimeSeriesTransformerConfig, config_from_hf, get_memory_config, get_ttnn_dtype
from .distribution import DistributionHead
from .inputs import NetworkInputs, create_network_inputs, expand_features, get_lagged_subsequences, get_latest_lags
from .ops import slice_last_step, to_device, to_torch
from .state_io import extract_embedder_weights, load_checkpoint_config, load_checkpoint_state
from .trace import TracedDecodeRunner, TracedRolloutRunner
from .transformer import Decoder, Encoder
from .weights import LoadResult, merge_results, substate


@dataclass
class ForwardOutput:
    """Teacher-forced forward pass results."""

    encoder_last_hidden_state: ttnn.Tensor
    decoder_last_hidden_state: ttnn.Tensor
    loc: torch.Tensor
    scale: torch.Tensor
    static_feat: torch.Tensor


class TimeSeriesTransformer:
    """TTNN implementation of ``TimeSeriesTransformerForPrediction``."""

    def __init__(
        self,
        config: TimeSeriesTransformerConfig,
        *,
        device,
        seed: int = 0,
    ):
        self.config = config
        self.device = device
        self.dtype = get_ttnn_dtype(config.dtype)
        self.rng = torch.Generator().manual_seed(seed)

        memory_config = get_memory_config(config)
        self.encoder = Encoder(config, device=device, dtype=self.dtype)
        self.decoder = Decoder(config, device=device, dtype=self.dtype)
        self.distribution_head = DistributionHead(config, device=device, dtype=self.dtype, memory_config=memory_config)
        self.embedder_weights: list[torch.Tensor] = []
        self._runners: dict[int, TracedDecodeRunner] = {}
        self._rollouts: dict[int, TracedRolloutRunner] = {}

        if config.use_program_cache:
            device.enable_program_cache()

    # -- construction -------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str,
        *,
        device,
        seed: int = 0,
        **config_overrides,
    ) -> "TimeSeriesTransformer":
        """Build a model straight from a Hub id or local checkpoint directory."""
        hf_config = load_checkpoint_config(checkpoint)
        config = config_from_hf(hf_config, **config_overrides)
        model = cls(config, device=device, seed=seed)
        model.load_hf_state_dict(load_checkpoint_state(checkpoint), strict=True)
        return model

    def load_hf_state_dict(self, state: Mapping[str, torch.Tensor], *, strict: bool = True) -> LoadResult:
        self.embedder_weights = extract_embedder_weights(dict(state))
        return merge_results(
            [
                (
                    "model.encoder",
                    self.encoder.load_hf_state_dict(substate(state, "model.encoder"), strict=strict),
                ),
                (
                    "model.decoder",
                    self.decoder.load_hf_state_dict(substate(state, "model.decoder"), strict=strict),
                ),
                (
                    "parameter_projection",
                    self.distribution_head.load_hf_state_dict(substate(state, "parameter_projection"), strict=strict),
                ),
            ]
        )

    # -- input plumbing -----------------------------------------------------

    def create_inputs(
        self,
        *,
        past_values: torch.Tensor,
        past_time_features: torch.Tensor,
        past_observed_mask: Optional[torch.Tensor] = None,
        static_categorical_features: Optional[torch.Tensor] = None,
        static_real_features: Optional[torch.Tensor] = None,
        future_values: Optional[torch.Tensor] = None,
        future_time_features: Optional[torch.Tensor] = None,
    ) -> NetworkInputs:
        return create_network_inputs(
            self.config,
            past_values=past_values,
            past_time_features=past_time_features,
            past_observed_mask=past_observed_mask,
            static_categorical_features=static_categorical_features,
            static_real_features=static_real_features,
            future_values=future_values,
            future_time_features=future_time_features,
            embedder_weights=self.embedder_weights,
        )

    def encode(self, encoder_inputs: torch.Tensor) -> ttnn.Tensor:
        return self.encoder(to_device(encoder_inputs, device=self.device, dtype=self.dtype))

    # -- forward ------------------------------------------------------------

    def forward(self, **inputs) -> ForwardOutput:
        """Teacher-forced pass; ``future_values`` must be supplied."""
        network_inputs = self.create_inputs(**inputs)
        context = self.config.context_length
        encoder_hidden = self.encode(network_inputs.transformer_inputs[:, :context, ...])
        decoder_hidden = self.decoder(
            to_device(network_inputs.transformer_inputs[:, context:, ...], device=self.device, dtype=self.dtype),
            encoder_hidden,
        )
        return ForwardOutput(
            encoder_last_hidden_state=encoder_hidden,
            decoder_last_hidden_state=decoder_hidden,
            loc=network_inputs.loc,
            scale=network_inputs.scale,
            static_feat=network_inputs.static_feat,
        )

    # -- generation ---------------------------------------------------------

    def generate(
        self,
        *,
        past_values: torch.Tensor,
        past_time_features: torch.Tensor,
        future_time_features: torch.Tensor,
        past_observed_mask: Optional[torch.Tensor] = None,
        static_categorical_features: Optional[torch.Tensor] = None,
        static_real_features: Optional[torch.Tensor] = None,
        num_parallel_samples: Optional[int] = None,
        mode: str = "sample",
    ) -> torch.Tensor:
        """Autoregressively forecast ``prediction_length`` steps.

        Returns ``(batch, num_parallel_samples, prediction_length)``. ``mode='sample'`` draws
        from the predicted distribution; ``mode='mean'`` takes its mean, which is what the
        deterministic accuracy gates compare against.

        All ``num_parallel_samples`` trajectories are advanced as one batch. Which path runs
        depends on the mode: mean-mode replays the entire forecast from one trace with the
        feedback closed on device (:class:`TracedRolloutRunner`), while sampling steps through
        the decoder from the host because a Student's t draw cannot be produced on device
        (:class:`TracedDecodeRunner`). Without ``use_trace`` both fall back to eager decoding
        against a KV cache.
        """
        if mode not in ("sample", "mean"):
            raise ValueError(f"Unsupported generation mode: {mode}")
        samples = num_parallel_samples if num_parallel_samples is not None else self.config.num_parallel_samples
        config = self.config

        network_inputs = self.create_inputs(
            past_values=past_values,
            past_time_features=past_time_features,
            past_observed_mask=past_observed_mask,
            static_categorical_features=static_categorical_features,
            static_real_features=static_real_features,
            future_time_features=future_time_features,
        )
        loc, scale = network_inputs.loc, network_inputs.scale
        repeated_loc = loc.repeat_interleave(samples, dim=0)
        repeated_scale = scale.repeat_interleave(samples, dim=0)
        repeated_past = ((past_values - loc) / scale).repeat_interleave(samples, dim=0)

        features = expand_features(network_inputs.static_feat, future_time_features)
        repeated_features = features.repeat_interleave(samples, dim=0)
        rows = repeated_past.shape[0]

        # Mean mode has no host-side dependency inside the loop, so encoder and rollout both run
        # from a single trace with the feedback closed on device.
        if config.use_trace and mode == "mean":
            predictions = self._traced_rollout(rows).run(
                running=repeated_past,
                covariates=repeated_features,
                encoder_inputs=network_inputs.transformer_inputs.repeat_interleave(samples, dim=0),
            )
            forecast = repeated_loc + repeated_scale * predictions
            return forecast.reshape(past_values.shape[0], samples, config.prediction_length)

        encoder_hidden = self.encode(network_inputs.transformer_inputs)
        repeated_encoder_hidden = (
            ttnn.repeat_interleave(encoder_hidden, repeats=samples, dim=0) if samples > 1 else encoder_hidden
        )

        runner = self._traced_runner(rows) if config.use_trace else None
        if runner is not None:
            runner.prepare(repeated_encoder_hidden)
            caches = None
        else:
            caches = self.decoder.new_caches() if config.use_kv_cache else None

        # The traced window is a fixed prediction_length rows. Each step appends exactly one
        # row, so the window is carried across steps rather than rebuilt -- regenerating every
        # lag window each step would make the host side O(horizon^2) for no reason.
        window = (
            torch.zeros(repeated_past.shape[0], config.prediction_length, config.feature_size)
            if runner is not None
            else None
        )

        collected: list[torch.Tensor] = []
        for step in range(config.prediction_length):
            if runner is not None:
                window[:, step : step + 1] = self._build_decode_step(
                    repeated_past, repeated_features, step=step, use_cache=True
                )
                parameters = runner.step(step, window)
            else:
                decoder_input = self._build_decode_step(
                    repeated_past, repeated_features, step=step, use_cache=caches is not None
                )
                hidden = self.decoder(
                    to_device(decoder_input, device=self.device, dtype=self.dtype),
                    repeated_encoder_hidden,
                    caches=caches,
                )
                parameters = [to_torch(p) for p in self.distribution_head.parameters(slice_last_step(hidden))]

            next_sample = self._draw(parameters, loc=repeated_loc, scale=repeated_scale, mode=mode)
            repeated_past = torch.cat((repeated_past, (next_sample - repeated_loc) / repeated_scale), dim=1)
            collected.append(next_sample)

        return torch.cat(collected, dim=1).reshape(past_values.shape[0], samples, config.prediction_length)

    def _traced_runner(self, rows: int) -> TracedDecodeRunner:
        """Traces are shape-specific, so keep one runner per row count and reuse it."""
        runner = self._runners.get(rows)
        if runner is None:
            runner = TracedDecodeRunner(self, rows)
            self._runners[rows] = runner
        return runner

    def _traced_rollout(self, rows: int) -> TracedRolloutRunner:
        runner = self._rollouts.get(rows)
        if runner is None:
            runner = TracedRolloutRunner(self, rows)
            self._rollouts[rows] = runner
        return runner

    def release_traces(self) -> None:
        for runner in list(self._runners.values()) + list(self._rollouts.values()):
            runner.release()
        self._runners.clear()
        self._rollouts.clear()

    def _build_decode_step(
        self,
        repeated_past: torch.Tensor,
        repeated_features: torch.Tensor,
        *,
        step: int,
        use_cache: bool,
    ) -> torch.Tensor:
        """Assemble the decoder input for one step.

        With a KV cache only the newest token is needed. The right edge of a lagged window is
        pinned by the lag index, so the single-row gather here is exactly the last row of the
        full ``1 + step`` gather HuggingFace performs.
        """
        if use_cache:
            reshaped = get_latest_lags(repeated_past, self.config.lags_sequence)
            covariates = repeated_features[:, step : step + 1]
        else:
            lagged = get_lagged_subsequences(
                repeated_past,
                subsequences_length=step + 1,
                lags_sequence=self.config.lags_sequence,
                shift=1,
            )
            reshaped = lagged.reshape(lagged.shape[0], lagged.shape[1], -1)
            covariates = repeated_features[:, : step + 1]
        return torch.cat((reshaped, covariates), dim=-1)

    def _draw(
        self,
        parameters: list[torch.Tensor],
        *,
        loc: torch.Tensor,
        scale: torch.Tensor,
        mode: str,
    ) -> torch.Tensor:
        """Turn one step's distribution parameters into the next value.

        Parameters have already come back to the host -- the next lag window is built there
        regardless -- so the traced and untraced paths share this step exactly.
        """
        distribution = self.distribution_head.torch_distribution(parameters, loc=loc, scale=scale)
        value = distribution.mean if mode == "mean" else distribution.sample()
        return value.reshape(loc.shape[0], 1)


__all__ = ["ForwardOutput", "TimeSeriesTransformer"]
