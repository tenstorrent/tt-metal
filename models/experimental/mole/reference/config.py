# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, replace
from typing import Literal


@dataclass(frozen=True)
class MoLEConfig:
    seq_len: int = 96
    pred_len: int = 24
    input_dim: int = 7
    base_model_type: Literal["dlinear", "rlinear", "rmlp"] = "dlinear"
    t_dim: int | None = None
    num_experts: int | None = None
    moving_average_kernel_size: int = 25
    revin_eps: float = 1e-5
    individual: bool = False
    freq: str = "h"
    drop: float = 0.1
    disable_rev: bool = False
    d_model: int = 512
    head_dropout: float = 0.0

    def __post_init__(self) -> None:
        resolved_num_experts = self.num_experts
        if resolved_num_experts is None:
            resolved_num_experts = self.t_dim if self.t_dim is not None else 4
        if self.num_experts is not None and self.t_dim is not None and self.num_experts != self.t_dim:
            raise ValueError(f"num_experts ({self.num_experts}) and t_dim ({self.t_dim}) must match")
        object.__setattr__(self, "num_experts", resolved_num_experts)
        # Keep t_dim as a compatibility alias for older call sites.
        object.__setattr__(self, "t_dim", resolved_num_experts)

    @property
    def enc_in(self) -> int:
        return self.input_dim

    @property
    def channel(self) -> int:
        return self.input_dim


def replace_num_experts(config: MoLEConfig, *, num_experts: int) -> MoLEConfig:
    return replace(config, num_experts=num_experts, t_dim=num_experts)
