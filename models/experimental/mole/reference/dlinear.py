# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn

from models.experimental.mole.reference.common import validate_model_inputs
from models.experimental.mole.reference.config import MoLEConfig
from models.experimental.mole.reference.model_outputs import MoLEForwardOutputs


class MovingAverage(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd, got {kernel_size}")
        self.kernel_size = kernel_size
        self.average_pool = nn.AvgPool1d(kernel_size=kernel_size, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"expected [batch, seq_len, channels], got {tuple(x.shape)}")

        pad = (self.kernel_size - 1) // 2
        left = x[:, :1, :].repeat(1, pad, 1)
        right = x[:, -1:, :].repeat(1, pad, 1)
        padded = torch.cat([left, x, right], dim=1)
        averaged = self.average_pool(padded.transpose(1, 2))
        return averaged.transpose(1, 2)


class SeriesDecomposition(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        self.moving_average = MovingAverage(kernel_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        trend = self.moving_average(x)
        seasonal = x - trend
        return seasonal, trend


class DLinearExpert(nn.Module):
    def __init__(self, config: MoLEConfig):
        super().__init__()
        self.config = config
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len
        self.decomposition = SeriesDecomposition(config.moving_average_kernel_size)
        self.Linear_Seasonal = nn.Linear(config.seq_len, config.pred_len)
        self.Linear_Trend = nn.Linear(config.seq_len, config.pred_len)

    @property
    def seasonal_projection(self) -> nn.Linear:
        return self.Linear_Seasonal

    @property
    def trend_projection(self) -> nn.Linear:
        return self.Linear_Trend

    def _forward_outputs(self, x: torch.Tensor, x_mark: torch.Tensor) -> MoLEForwardOutputs:
        validate_model_inputs(x, x_mark)
        seasonal_init, trend_init = self.decomposition(x)
        seasonal_init = seasonal_init.permute(0, 2, 1)
        trend_init = trend_init.permute(0, 2, 1)
        seasonal_output = self.Linear_Seasonal(seasonal_init)
        trend_output = self.Linear_Trend(trend_init)
        prediction = (seasonal_output + trend_output).permute(0, 2, 1)
        return MoLEForwardOutputs(
            prediction=prediction,
        )

    def forward(self, x: torch.Tensor, x_mark: torch.Tensor) -> torch.Tensor:
        return self._forward_outputs(x, x_mark).prediction
