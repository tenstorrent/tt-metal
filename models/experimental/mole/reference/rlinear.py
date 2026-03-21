# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn

from models.experimental.mole.reference.common import HeadDropout, RevIN, validate_model_inputs
from models.experimental.mole.reference.config import MoLEConfig
from models.experimental.mole.reference.model_outputs import MoLEForwardOutputs


class RLinearExpert(nn.Module):
    def __init__(self, config: MoLEConfig):
        super().__init__()
        self.config = config
        self.num_predictions = config.t_dim
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len
        self.channels = config.enc_in
        self.expected_time_features = 4 if config.freq.lower().endswith("h") else 5
        self.Linear = (
            nn.ModuleList(
                [nn.Linear(config.seq_len, config.pred_len * self.num_predictions) for _ in range(self.channels)]
            )
            if config.individual
            else nn.Linear(config.seq_len, config.pred_len * self.num_predictions)
        )
        self.dropout = nn.Dropout(config.drop)
        self.rev = RevIN(config.enc_in, eps=config.revin_eps) if not config.disable_rev else None
        self.Linear_Temporal = nn.Sequential(
            nn.Linear(self.expected_time_features, self.num_predictions * self.channels),
            nn.ReLU(),
            nn.Linear(self.num_predictions * self.channels, self.num_predictions * self.channels),
        )
        self.head_dropout = HeadDropout(config.head_dropout)

    @property
    def projection(self) -> nn.Module:
        return self.Linear

    @property
    def temporal_projection(self) -> nn.Sequential:
        return self.Linear_Temporal

    def _project(self, x: torch.Tensor) -> torch.Tensor:
        if self.config.individual:
            outputs = [layer(x[:, index, :]) for index, layer in enumerate(self.Linear)]
            return torch.stack(outputs, dim=1)
        return self.Linear(x)

    def _forward_outputs(self, x: torch.Tensor, x_mark: torch.Tensor) -> MoLEForwardOutputs:
        validate_model_inputs(x, x_mark, expected_time_features=self.expected_time_features)
        x_mark_initial = x_mark[:, 0]
        temporal_out = self.Linear_Temporal(x_mark_initial).reshape(-1, self.num_predictions, self.channels)
        temporal_out = self.head_dropout(temporal_out)
        temporal_out = torch.softmax(temporal_out, dim=1)

        x = self.rev(x, "norm") if self.rev else x
        x = self.dropout(x)
        projected = self._project(x.transpose(1, 2)).transpose(1, 2)
        pred_raw = (
            projected.permute(0, 2, 1)
            .reshape(-1, self.channels, self.pred_len, self.num_predictions)
            .permute(0, 3, 1, 2)
        )
        weighted = pred_raw * temporal_out.unsqueeze(-1)
        prediction = weighted.sum(dim=1).permute(0, 2, 1)
        prediction = self.rev(prediction, "denorm") if self.rev else prediction
        return MoLEForwardOutputs(
            prediction=prediction,
            gating_weights=temporal_out,
        )

    def forward(self, x: torch.Tensor, x_mark: torch.Tensor) -> torch.Tensor:
        return self._forward_outputs(x, x_mark).prediction
