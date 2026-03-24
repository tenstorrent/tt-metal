# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn

from models.experimental.mole.reference.common import HeadDropout, validate_model_inputs
from models.experimental.mole.reference.config import MoLEConfig, replace_num_experts
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
        self.num_experts = config.num_experts
        self.expected_time_features = 4 if config.freq.lower().endswith("h") else 5
        expert_config = replace_num_experts(config, num_experts=1)
        self.experts = nn.ModuleList([create_reference_expert(expert_config) for _ in range(self.num_experts)])
        self.router = nn.Sequential(
            nn.Linear(self.expected_time_features, self.num_experts * config.input_dim),
            nn.ReLU(),
            nn.Linear(self.num_experts * config.input_dim, self.num_experts * config.input_dim),
        )
        self.head_dropout = HeadDropout(config.head_dropout)

    def _forward_outputs(self, x: torch.Tensor, x_mark: torch.Tensor) -> MoLEForwardOutputs:
        validate_model_inputs(x, x_mark, expected_time_features=self.expected_time_features)

        expert_predictions = torch.stack([expert(x, x_mark) for expert in self.experts], dim=1)
        gating_weights = self.router(x_mark[:, 0]).reshape(-1, self.num_experts, self.config.input_dim)
        gating_weights = self.head_dropout(gating_weights)
        gating_weights = torch.softmax(gating_weights, dim=1)
        prediction = (expert_predictions * gating_weights.unsqueeze(2)).sum(dim=1)
        return MoLEForwardOutputs(
            prediction=prediction,
            gating_weights=gating_weights,
            expert_predictions=expert_predictions,
        )

    def forward(
        self,
        x: torch.Tensor,
        x_mark: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self._forward_outputs(x, x_mark)
        return outputs.prediction, outputs.gating_weights
