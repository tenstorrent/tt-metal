# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class MoLEForwardOutputs:
    prediction: torch.Tensor
    gating_weights: torch.Tensor
