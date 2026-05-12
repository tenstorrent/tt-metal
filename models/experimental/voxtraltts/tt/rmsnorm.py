# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Any

import torch
import ttnn
from models.common.rmsnorm import RMSNorm
from models.experimental.voxtraltts.utils.config_helpers import COMPUTE_KERNEL_CONFIG_VOXTRAL_ACOUSTIC
from models.tt_transformers.tt.common import Mode


class VoxtralAcousticRMSNorm(RMSNorm):
    """``models.common.rmsnorm.RMSNorm`` with acoustic FM HiFi4 kernel (no common API changes)."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.compute_kernel_config_hifi2 = COMPUTE_KERNEL_CONFIG_VOXTRAL_ACOUSTIC


class VoxtralTTRMSNorm:
    """Thin Voxtral adapter over shared TT RMSNorm implementation."""

    def __init__(
        self,
        device,
        dim: int,
        state_dict: dict[str, torch.Tensor],
        weight_key: str,
        eps: float = 1e-5,
        weight_dtype=ttnn.bfloat16,
        weight_cache_path: Path | None = None,
    ) -> None:
        # models.common.rmsnorm expects "<weight_key>.weight" in state_dict.
        if weight_key in state_dict and f"{weight_key}.weight" not in state_dict:
            state_dict = {**state_dict, f"{weight_key}.weight": state_dict[weight_key]}

        self.inner = RMSNorm(
            device=device,
            dim=dim,
            state_dict=state_dict,
            weight_key=weight_key,
            eps=eps,
            weight_dtype=weight_dtype,
            weight_cache_path=weight_cache_path,
            is_distributed=False,
        )

    def __call__(self, x: ttnn.Tensor, mode: Mode | str = Mode.DECODE, **kwargs: Any) -> ttnn.Tensor:
        return self.inner(x, mode=mode, **kwargs)
