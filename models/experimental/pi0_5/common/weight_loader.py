# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
PI0.5 weight loader.

Wraps `PI0WeightLoader` and patches `categorize_weights` to also pick up
the pi0.5 `time_mlp_in/out` projections (top-level in the safetensors).
"""

from pathlib import Path
from typing import Dict, Union

import torch

from models.experimental.pi0.common.weight_loader import (
    PI0Config,
    PI0WeightLoader,
    categorize_weights as _pi0_categorize,
)


def categorize_pi0_5_weights(state_dict: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, torch.Tensor]]:
    """Same buckets as PI0, plus `time_mlp_in/out` -> pi0_projections."""
    cat = _pi0_categorize(state_dict)
    for key, value in state_dict.items():
        if key.startswith("time_mlp_in") or key.startswith("time_mlp_out"):
            cat["pi0_projections"][key] = value
    return cat


class Pi0_5WeightLoader(PI0WeightLoader):
    """
    PI0 loader with the pi0.5 `time_mlp_*` categorization fix.

    pi0.5's `config.json` has many fields beyond what `PI0Config` accepts (it
    is the openpi policy config, not a model-architecture config), so we
    bypass the parent's strict JSON load and use the architectural defaults.
    """

    def __init__(self, model_path: Union[str, Path], cache_path=None):
        self.model_path = Path(model_path) if isinstance(model_path, str) else model_path
        self.cache_path = cache_path
        self.config = PI0Config()
        self._state_dict = None
        self._categorized = None

    @property
    def state_dict(self) -> Dict[str, torch.Tensor]:
        # Override: strip the leading "model." prefix that lerobot finetunes use
        # (`model.paligemma_with_expert...` -> `paligemma_with_expert...`). Base
        # `pi05_base` doesn't have that prefix; remapping is a no-op there.
        sd = super().state_dict
        if any(k.startswith("model.") for k in sd):
            return {k[len("model.") :] if k.startswith("model.") else k: v for k, v in sd.items()}
        return sd

    @property
    def categorized_weights(self) -> Dict[str, Dict[str, torch.Tensor]]:
        if self._categorized is None:
            self._categorized = categorize_pi0_5_weights(self.state_dict)
        return self._categorized
