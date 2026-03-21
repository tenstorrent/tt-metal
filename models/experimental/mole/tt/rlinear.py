# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import ttnn

from models.experimental.mole.reference.rlinear import RLinearExpert
from models.experimental.mole.tt.common import (
    TtExpertBase,
    apply_linear,
    apply_two_layer_mlp,
    extract_initial_marks,
    permute_temporal_router_mlp_to_channel_major,
    project_individual_channels,
    reduce_weighted_heads,
    reduce_weighted_heads_batch_major,
    register_trace_release_hook,
    temporal_gating,
    upload_linear,
    upload_vector,
    validate_time_marks,
    validate_timeseries_input,
)


class TtRLinearExpert(TtExpertBase):
    """TTNN implementation for both RLinear and RMLP experts.

    When the reference model has a ``temporal`` MLP (i.e. RMLP), its weights are
    uploaded and applied as a residual before projection.  Otherwise (RLinear)
    the temporal residual step is skipped.
    """

    def __init__(self, config, *, reference_model=None, runtime_options=None):
        ref = reference_model if reference_model is not None else RLinearExpert(config).eval()
        self._init_base(config, ref, runtime_options)
        self._has_temporal_mlp = hasattr(ref, "temporal") and ref.temporal is not None

    def _ensure_parameters(self, device) -> None:
        if self._cached_device is device and self._tt_parameters is not None:
            return

        up = lambda mod: upload_linear(mod, device=device, dtype=self.dtype, memory_config=self.parameter_memory_config)

        if self.config.individual:
            projection = [up(layer) for layer in self.reference_model.projection]
        else:
            projection = up(self.reference_model.projection)

        tp = permute_temporal_router_mlp_to_channel_major(
            self.reference_model.temporal_projection,
            channels=self.config.input_dim,
            num_predictions=self.config.t_dim,
        )
        up_t = lambda w, b: upload_linear(
            w, b, device=device, dtype=self.dtype, memory_config=self.parameter_memory_config
        )

        params = {
            "projection": projection,
            "temporal_projection": {
                "linear_1": up_t(tp["linear_1_weight"], tp["linear_1_bias"]),
                "linear_2": up_t(tp["linear_2_weight"], tp["linear_2_bias"]),
            },
        }
        if self._has_temporal_mlp:
            params["temporal"] = {
                "linear_1": up(self.reference_model.temporal[0]),
                "linear_2": up(self.reference_model.temporal[2]),
            }
        if self.reference_model.rev is not None:
            uv = lambda v: upload_vector(v, device=device, dtype=self.dtype, memory_config=self.parameter_memory_config)
            params["rev"] = {
                "affine_weight": uv(self.reference_model.rev.affine_weight),
                "affine_bias": uv(self.reference_model.rev.affine_bias),
            }
        self._tt_parameters = params
        self._cached_device = device
        register_trace_release_hook(device=device, hook=self._release_prediction_trace)

    def _normalize(self, input_tensor):
        mc = self.activation_memory_config
        mean = ttnn.sum(input_tensor, dim=2, keepdim=True, scalar=1.0 / self.config.seq_len, memory_config=mc)
        centered = ttnn.subtract(input_tensor, mean, memory_config=mc)
        variance = ttnn.multiply(centered, centered, memory_config=mc)
        variance = ttnn.sum(variance, dim=2, keepdim=True, scalar=1.0 / self.config.seq_len, memory_config=mc)
        stdev = ttnn.sqrt(ttnn.add(variance, self.config.revin_eps, memory_config=mc))
        normalized = ttnn.div(centered, stdev, memory_config=mc)

        if "rev" in self._tt_parameters:
            normalized = ttnn.multiply(normalized, self._tt_parameters["rev"]["affine_weight"], memory_config=mc)
            normalized = ttnn.add(normalized, self._tt_parameters["rev"]["affine_bias"], memory_config=mc)

        return normalized, mean, stdev

    def _denormalize(self, prediction, mean, stdev):
        mc = self.activation_memory_config
        if "rev" in self._tt_parameters:
            prediction = ttnn.subtract(prediction, self._tt_parameters["rev"]["affine_bias"], memory_config=mc)
            denom = ttnn.add(self._tt_parameters["rev"]["affine_weight"], self.config.revin_eps**2, memory_config=mc)
            prediction = ttnn.div(prediction, denom, memory_config=mc)
        prediction = ttnn.multiply(prediction, stdev, memory_config=mc)
        return ttnn.add(prediction, mean, memory_config=mc)

    def _project(self, input_tensor):
        mc = self.activation_memory_config
        if self.config.individual:
            return project_individual_channels(
                ttnn.permute(input_tensor, (0, 3, 1, 2)),
                self._tt_parameters["projection"],
                memory_config=mc,
            )
        return apply_linear(
            ttnn.permute(input_tensor, (0, 1, 3, 2)), self._tt_parameters["projection"], memory_config=mc
        )

    def _forward_outputs(self, input_tensor, input_marks, *, return_channelwise_weights=True):
        validate_timeseries_input(input_tensor, seq_len=self.config.seq_len, input_dim=self.config.input_dim)
        validate_time_marks(input_marks, seq_len=self.config.seq_len)
        self._ensure_parameters(input_tensor.device())

        batch_size = input_tensor.shape[1]
        mc = self.activation_memory_config
        normalized, mean, stdev = self._normalize(input_tensor)

        if self._has_temporal_mlp:
            t_in = ttnn.permute(normalized, (0, 3, 1, 2))
            t_res = apply_two_layer_mlp(t_in, self._tt_parameters["temporal"], memory_config=mc)
            normalized = ttnn.add(normalized, ttnn.permute(t_res, (0, 2, 3, 1)), memory_config=mc)

        projected = self._project(normalized)

        gating_logits = apply_two_layer_mlp(
            extract_initial_marks(input_marks), self._tt_parameters["temporal_projection"], memory_config=mc
        )
        gating_flat, gating_weights = temporal_gating(
            gating_logits,
            batch_size=batch_size,
            channels=self.config.input_dim,
            num_predictions=self.config.t_dim,
            return_channelwise_weights=return_channelwise_weights,
        )
        reduce_fn = reduce_weighted_heads if self.config.individual else reduce_weighted_heads_batch_major
        prediction = reduce_fn(
            projected,
            gating_flat,
            batch_size=batch_size,
            channels=self.config.input_dim,
            pred_len=self.config.pred_len,
            num_predictions=self.config.t_dim,
            memory_config=mc,
        )
        return self._denormalize(prediction, mean, stdev), gating_weights, gating_flat

    def forward(self, input_tensor, input_marks):
        prediction, _, _ = self._forward_outputs(input_tensor, input_marks, return_channelwise_weights=False)
        return prediction
