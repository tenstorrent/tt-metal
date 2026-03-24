# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import ttnn

from models.experimental.mole.reference.dlinear import DLinearExpert
from models.experimental.mole.tt.common import (
    TtExpertBase,
    apply_linear,
    moving_average_projection_matrix,
    register_trace_release_hook,
    upload_linear,
    validate_timeseries_input,
)


@dataclass
class _TtDLinearUploaded:
    moving_average: Any
    seasonal: Any
    trend: Any


class TtDLinearExpert(TtExpertBase):
    def __init__(self, config, *, reference_model=None, runtime_options=None):
        ref = reference_model if reference_model is not None else DLinearExpert(config).eval()
        self._init_base(config, ref, runtime_options)

    def _ensure_parameters(self, device) -> None:
        if self._cached_device is device and self._tt_parameters is not None:
            return
        up = lambda mod: upload_linear(mod, device=device, dtype=self.dtype, memory_config=self.parameter_memory_config)
        moving_average_weight = moving_average_projection_matrix(
            seq_len=self.config.seq_len,
            kernel_size=self.config.moving_average_kernel_size,
        )
        moving_average_bias = self.reference_model.seasonal_projection.bias.detach().new_zeros(self.config.seq_len)
        self._tt_parameters = _TtDLinearUploaded(
            moving_average=upload_linear(
                moving_average_weight,
                moving_average_bias,
                device=device,
                dtype=self.dtype,
                memory_config=self.parameter_memory_config,
            ),
            seasonal=up(self.reference_model.seasonal_projection),
            trend=up(self.reference_model.trend_projection),
        )
        self._cached_device = device
        register_trace_release_hook(device=device, hook=self._release_prediction_trace)

    def _forward_outputs(self, input_tensor, input_marks, *, return_channelwise_weights=True):
        validate_timeseries_input(input_tensor, seq_len=self.config.seq_len, input_dim=self.config.input_dim)
        self._ensure_parameters(input_tensor.device())

        mc = self.activation_memory_config
        # Match reference: moving average and seq_len projections run on [batch, channels, seq_len].
        input_channels_first = ttnn.permute(input_tensor, (0, 1, 3, 2))
        tp = self._tt_parameters
        trend = apply_linear(input_channels_first, tp.moving_average, memory_config=mc)
        seasonal = ttnn.subtract(input_channels_first, trend, memory_config=mc)
        prediction = ttnn.add(
            apply_linear(seasonal, tp.seasonal, memory_config=mc),
            apply_linear(trend, tp.trend, memory_config=mc),
            memory_config=mc,
        )
        prediction = ttnn.permute(prediction, (0, 1, 3, 2))
        return prediction, None, None

    def forward(self, input_tensor, input_marks):
        prediction, _, _ = self._forward_outputs(input_tensor, input_marks, return_channelwise_weights=False)
        return prediction
