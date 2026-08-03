# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Probabilistic output head: parameter projection, domain maps, sampling."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Optional

import torch

import ttnn

from .config import TimeSeriesTransformerConfig
from .ops import linear, squareplus
from .weights import LoadResult, load_tensors, upload

# torch.finfo(torch.float32).eps, matching HuggingFace's clamp_min on distribution scales.
FLOAT32_EPS = 1.1920928955078125e-07


class ParameterProjection:
    """One bias-carrying linear head per distribution parameter."""

    def __init__(
        self,
        d_model: int,
        out_dim: int,
        num_params: int,
        *,
        device,
        dtype: ttnn.DataType,
        memory_config: Optional[ttnn.MemoryConfig] = None,
        rng: Optional[torch.Generator] = None,
    ):
        self.device = device
        self.dtype = dtype
        self.memory_config = memory_config
        self.num_params = num_params
        self.out_dim = out_dim

        for index in range(num_params):
            if rng is None:
                weight = torch.zeros((out_dim, d_model), dtype=torch.float32)
            else:
                weight = torch.randn((out_dim, d_model), generator=rng, dtype=torch.float32) * 0.02
            bias = torch.zeros((out_dim,), dtype=torch.float32)
            setattr(self, f"proj_{index}_weight_torch", weight)
            setattr(self, f"proj_{index}_bias_torch", bias)
            setattr(self, f"proj_{index}_weight", upload(weight, device=device, dtype=dtype))
            setattr(self, f"proj_{index}_bias", upload(bias, device=device, dtype=dtype))

    def load_hf_state_dict(self, state: Mapping[str, torch.Tensor], *, strict: bool = True) -> LoadResult:
        mapping = tuple(
            (f"proj.{index}.{suffix}", f"proj_{index}_{suffix}")
            for index in range(self.num_params)
            for suffix in ("weight", "bias")
        )
        return load_tensors(
            self,
            state,
            mapping,
            device=self.device,
            dtype=self.dtype,
            strict=strict,
            label="parameter projection",
        )

    def project(self, hidden_states: ttnn.Tensor, index: int) -> ttnn.Tensor:
        """Apply a single parameter head."""
        return linear(
            hidden_states,
            getattr(self, f"proj_{index}_weight"),
            getattr(self, f"proj_{index}_bias"),
            dtype=self.dtype,
            memory_config=self.memory_config,
        )

    def __call__(self, hidden_states: ttnn.Tensor) -> list[ttnn.Tensor]:
        return [self.project(hidden_states, index) for index in range(self.num_params)]


class DistributionHead:
    """Projects decoder hidden states to distribution parameters and samples from them.

    The projection and domain map run on device. Drawing samples falls back to
    ``torch.distributions``: a Student-t variate needs a Gamma draw whose shape parameter is
    itself data-dependent, so it cannot be pre-generated and uploaded. Mean-mode decoding --
    what the MAE and NLL gates measure -- stays entirely on device.
    """

    def __init__(
        self,
        config: TimeSeriesTransformerConfig,
        *,
        device,
        dtype: ttnn.DataType,
        memory_config: Optional[ttnn.MemoryConfig] = None,
        rng: Optional[torch.Generator] = None,
    ):
        self.config = config
        self.device = device
        self.dtype = dtype
        self.projection = ParameterProjection(
            config.d_model,
            config.input_size,
            config.num_distribution_params,
            device=device,
            dtype=dtype,
            memory_config=memory_config,
            rng=rng,
        )

    def load_hf_state_dict(self, state: Mapping[str, torch.Tensor], *, strict: bool = True) -> LoadResult:
        return self.projection.load_hf_state_dict(state, strict=strict)

    # -- parameters ---------------------------------------------------------

    def raw_parameters(self, hidden_states: ttnn.Tensor) -> list[ttnn.Tensor]:
        return self.projection(hidden_states)

    def domain_map(self, raw: list[ttnn.Tensor]) -> list[ttnn.Tensor]:
        """Map unconstrained projections onto the distribution's parameter domain."""
        kind = self.config.distribution_output
        if kind == "student_t":
            df, loc, scale = raw
            return [
                ttnn.add(squareplus(df), 2.0),
                loc,
                ttnn.clamp(squareplus(scale), FLOAT32_EPS, float("inf")),
            ]
        if kind == "normal":
            loc, scale = raw
            return [loc, ttnn.clamp(squareplus(scale), FLOAT32_EPS, float("inf"))]
        if kind == "negative_binomial":
            total_count, logits = raw
            return [squareplus(total_count), logits]
        raise ValueError(f"Unsupported distribution_output: {kind}")

    def parameters(self, hidden_states: ttnn.Tensor) -> list[ttnn.Tensor]:
        return self.domain_map(self.raw_parameters(hidden_states))

    # -- moments ------------------------------------------------------------

    def base_mean(self, parameters: list[ttnn.Tensor]) -> ttnn.Tensor:
        """Mean of the distribution *before* the affine transform, on device.

        This is exactly the value the autoregressive loop feeds back: the running series is
        normalized, and ``(loc + scale * base_mean - loc) / scale`` collapses to ``base_mean``,
        so a mean-mode rollout never needs the scaler statistics on device.
        """
        kind = self.config.distribution_output
        if kind in ("student_t", "normal"):
            # Student-t with df > 2 (guaranteed by the domain map) has mean == its loc.
            return parameters[1] if kind == "student_t" else parameters[0]
        if kind == "negative_binomial":
            total_count, logits = parameters
            return ttnn.multiply(total_count, ttnn.exp(logits))
        raise ValueError(f"Unsupported distribution_output: {kind}")

    def base_mean_from_hidden(self, hidden_states: ttnn.Tensor) -> ttnn.Tensor:
        """Compute only what an autoregressive mean-mode rollout consumes.

        For Student's t and Normal the pre-affine mean is the loc parameter, and the domain map
        passes loc through untouched -- so the other projections and the whole domain map are
        dead weight inside the loop. Negative binomial genuinely needs both parameters.
        """
        kind = self.config.distribution_output
        if kind == "student_t":
            return self.projection.project(hidden_states, 1)
        if kind == "normal":
            return self.projection.project(hidden_states, 0)
        return self.base_mean(self.parameters(hidden_states))

    def mean(self, parameters: list[ttnn.Tensor], *, loc: ttnn.Tensor, scale: ttnn.Tensor) -> ttnn.Tensor:
        """Mean of the affine-transformed distribution, computed on device.

        ``loc``/``scale`` are the scaler statistics broadcast as ``(batch, 1, 1)``.
        """
        return ttnn.add(loc, ttnn.multiply(scale, self.base_mean(parameters)))

    # -- host-side distribution --------------------------------------------

    def torch_distribution(
        self,
        parameters: list[torch.Tensor],
        *,
        loc: Optional[torch.Tensor] = None,
        scale: Optional[torch.Tensor] = None,
    ) -> torch.distributions.Distribution:
        """Build the equivalent ``torch.distributions`` object for sampling and log-prob."""
        from transformers.time_series_utils import AffineTransformed

        kind = self.config.distribution_output
        squeezed = [p.squeeze(-1) if p.dim() == 3 and p.shape[-1] == 1 else p for p in parameters]

        if kind == "student_t":
            base = torch.distributions.StudentT(*squeezed)
        elif kind == "normal":
            base = torch.distributions.Normal(*squeezed)
        elif kind == "negative_binomial":
            base = torch.distributions.NegativeBinomial(total_count=squeezed[0], logits=squeezed[1])
        else:
            raise ValueError(f"Unsupported distribution_output: {kind}")

        if loc is None and scale is None:
            return base
        return AffineTransformed(base, loc=loc, scale=scale, event_dim=0)


__all__ = ["FLOAT32_EPS", "DistributionHead", "ParameterProjection"]
