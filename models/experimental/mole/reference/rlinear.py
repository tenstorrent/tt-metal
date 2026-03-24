# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn

from models.experimental.mole.reference.common import RevIN, validate_model_inputs
from models.experimental.mole.reference.config import MoLEConfig
from models.experimental.mole.reference.model_outputs import MoLEForwardOutputs


class RLinearExpert(nn.Module):
    def __init__(self, config: MoLEConfig):
        super().__init__()
        self.config = config
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len
        self.channels = config.enc_in
        self.Linear = (
            nn.ModuleList([nn.Linear(config.seq_len, config.pred_len) for _ in range(self.channels)])
            if config.individual
            else nn.Linear(config.seq_len, config.pred_len)
        )
        self.dropout = nn.Dropout(config.drop)
        self.rev = RevIN(config.enc_in, eps=config.revin_eps) if not config.disable_rev else None

    @property
    def projection(self) -> nn.Module:
        return self.Linear

    def _project(self, x: torch.Tensor) -> torch.Tensor:
        if self.config.individual:
            outputs = [layer(x[:, index, :]) for index, layer in enumerate(self.Linear)]
            return torch.stack(outputs, dim=1)
        return self.Linear(x)

    def _forward_outputs(self, x: torch.Tensor, x_mark: torch.Tensor) -> MoLEForwardOutputs:
        validate_model_inputs(x, x_mark)
        x = self.rev(x, "norm") if self.rev else x
        x = self.dropout(x)
        prediction = self._project(x.transpose(1, 2)).transpose(1, 2)
        prediction = self.rev(prediction, "denorm") if self.rev else prediction
        return MoLEForwardOutputs(
            prediction=prediction,
        )

    def forward(self, x: torch.Tensor, x_mark: torch.Tensor) -> torch.Tensor:
        return self._forward_outputs(x, x_mark).prediction
