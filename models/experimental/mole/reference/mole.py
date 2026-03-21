# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn

from models.experimental.mole.reference.config import MoLEConfig
from models.experimental.mole.reference.dlinear import DLinearExpert
from models.experimental.mole.reference.model_outputs import MoLEForwardOutputs
from models.experimental.mole.reference.rlinear import RLinearExpert
from models.experimental.mole.reference.rmlp import RMLPExpert


def create_reference_expert(config: MoLEConfig) -> nn.Module:
    if config.base_model_type == "dlinear":
        return DLinearExpert(config)
    if config.base_model_type == "rlinear":
        return RLinearExpert(config)
    if config.base_model_type == "rmlp":
        return RMLPExpert(config)
    raise ValueError(f"unsupported base_model_type: {config.base_model_type}")


class MixtureOfLinearExperts(nn.Module):
    def __init__(self, config: MoLEConfig):
        super().__init__()
        self.config = config
        self.model = create_reference_expert(config)

    def _forward_outputs(self, x: torch.Tensor, x_mark: torch.Tensor) -> MoLEForwardOutputs:
        return self.model._forward_outputs(x, x_mark)

    def forward(
        self,
        x: torch.Tensor,
        x_mark: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self._forward_outputs(x, x_mark)
        return outputs.prediction, outputs.gating_weights
