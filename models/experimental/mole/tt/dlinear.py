# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import ttnn

from models.experimental.mole.reference.dlinear import DLinearExpert
from models.experimental.mole.tt.common import (
    TtExpertBase,
    apply_linear,
    apply_two_layer_mlp,
    extract_initial_marks,
    moving_average_1d,
    reduce_weighted_heads_batch_major,
    register_trace_release_hook,
    temporal_gating,
    upload_linear,
    validate_time_marks,
    validate_timeseries_input,
)


class TtDLinearExpert(TtExpertBase):
    def __init__(self, config, *, reference_model=None, runtime_options=None):
        ref = reference_model if reference_model is not None else DLinearExpert(config).eval()
        self._init_base(config, ref, runtime_options)

    def _ensure_parameters(self, device) -> None:
        if self._cached_device is device and self._tt_parameters is not None:
            return
        up = lambda mod: upload_linear(mod, device=device, dtype=self.dtype, memory_config=self.parameter_memory_config)
        self._tt_parameters = {
            "seasonal": up(self.reference_model.seasonal_projection),
            "trend": up(self.reference_model.trend_projection),
            "temporal": {
                "linear_1": up(self.reference_model.temporal_projection[0]),
                "linear_2": up(self.reference_model.temporal_projection[2]),
            },
        }
        self._cached_device = device
        register_trace_release_hook(device=device, hook=self._release_prediction_trace)

    def _forward_outputs(self, input_tensor, input_marks, *, return_channelwise_weights=True):
        validate_timeseries_input(input_tensor, seq_len=self.config.seq_len, input_dim=self.config.input_dim)
        validate_time_marks(input_marks, seq_len=self.config.seq_len)
        self._ensure_parameters(input_tensor.device())

        batch_size = input_tensor.shape[1]
        mc = self.activation_memory_config
        trend = moving_average_1d(input_tensor, self.config.moving_average_kernel_size, memory_config=mc)
        seasonal = ttnn.subtract(input_tensor, trend, memory_config=mc)

        seasonal = ttnn.permute(seasonal, (0, 1, 3, 2))
        trend = ttnn.permute(trend, (0, 1, 3, 2))
        combined = ttnn.add(
            apply_linear(seasonal, self._tt_parameters["seasonal"], memory_config=mc),
            apply_linear(trend, self._tt_parameters["trend"], memory_config=mc),
            memory_config=mc,
        )

        gating_logits = apply_two_layer_mlp(
            extract_initial_marks(input_marks), self._tt_parameters["temporal"], memory_config=mc
        )
        gating_flat, gating_weights = temporal_gating(
            gating_logits,
            batch_size=batch_size,
            channels=self.config.input_dim,
            num_predictions=self.config.t_dim,
            return_channelwise_weights=return_channelwise_weights,
        )
        prediction = reduce_weighted_heads_batch_major(
            combined,
            gating_flat,
            batch_size=batch_size,
            channels=self.config.input_dim,
            pred_len=self.config.pred_len,
            num_predictions=self.config.t_dim,
            memory_config=mc,
        )
        return prediction, gating_weights, gating_flat

    def forward(self, input_tensor, input_marks):
        prediction, _, _ = self._forward_outputs(input_tensor, input_marks, return_channelwise_weights=False)
        return prediction
