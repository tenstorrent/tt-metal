# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import ttnn

from models.experimental.mole.reference.rlinear import RLinearExpert
from models.experimental.mole.tt.common import (
    TtExpertBase,
    apply_linear,
    apply_two_layer_mlp,
    project_individual_channels,
    register_trace_release_hook,
    upload_linear,
    upload_vector,
    validate_timeseries_input,
)


class TtRLinearExpert(TtExpertBase):
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

        params = {
            "projection": projection,
        }
        if self._has_temporal_mlp:
            params["temporal"] = {
                "linear_1": up(self.reference_model.temporal[0]),
                "linear_2": up(self.reference_model.temporal[2]),
            }
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

    def _forward_normalized_prediction(self, normalized):
        mc = self.activation_memory_config
        if self._has_temporal_mlp:
            t_in = ttnn.permute(normalized, (0, 3, 1, 2))
            t_res = apply_two_layer_mlp(t_in, self._tt_parameters["temporal"], memory_config=mc)
            normalized = ttnn.add(normalized, ttnn.permute(t_res, (0, 2, 3, 1)), memory_config=mc)

        if self.config.individual:
            projected = project_individual_channels(
                ttnn.permute(normalized, (0, 3, 1, 2)),
                self._tt_parameters["projection"],
                memory_config=mc,
            )
            # project_individual_channels returns [1, channels, batch, pred_len]; map to [1, batch, pred_len, channels].
            return ttnn.permute(projected, (0, 2, 3, 1))
        projected = apply_linear(
            ttnn.permute(normalized, (0, 1, 3, 2)),
            self._tt_parameters["projection"],
            memory_config=mc,
        )
        return ttnn.permute(projected, (0, 1, 3, 2))

    def _denormalize(self, prediction, mean, stdev):
        mc = self.activation_memory_config
        if "rev" in self._tt_parameters:
            prediction = ttnn.subtract(prediction, self._tt_parameters["rev"]["affine_bias"], memory_config=mc)
            denom = ttnn.add(self._tt_parameters["rev"]["affine_weight"], self.config.revin_eps**2, memory_config=mc)
            prediction = ttnn.div(prediction, denom, memory_config=mc)
        prediction = ttnn.multiply(prediction, stdev, memory_config=mc)
        return ttnn.add(prediction, mean, memory_config=mc)

    def _forward_outputs(self, input_tensor, input_marks, *, return_channelwise_weights=True):
        validate_timeseries_input(input_tensor, seq_len=self.config.seq_len, input_dim=self.config.input_dim)
        self._ensure_parameters(input_tensor.device())
        normalized, mean, stdev = self._normalize(input_tensor)
        prediction = self._forward_normalized_prediction(normalized)
        return self._denormalize(prediction, mean, stdev), None, None

    def forward(self, input_tensor, input_marks):
        prediction, _, _ = self._forward_outputs(input_tensor, input_marks, return_channelwise_weights=False)
        return prediction
