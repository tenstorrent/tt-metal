# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Distribution strategy registry.

One stateless ``DistributionStrategy`` subclass per :class:`DistributionKind`.
Each strategy implements at most two methods:

* ``generate_face`` — per-face generation (called by the dispatch in
  ``generate_face`` and from the per-face loop in ``_generate_source_tensor``);
* ``generate_full_tensor`` — full-operand generation used when the strategy's
  ``short_circuit`` flag is True (bypasses the face loop).

Strategies that do not support a given mode raise ``NotImplementedError`` with
a clear message. The :data:`_STRATEGIES` dict maps every ``DistributionKind``
member to a single instance of its strategy, instantiated at import time.
"""

from typing import Dict, List, Optional, Protocol

import torch

from ...format_config import DataFormat
from ..spec import DistributionKind, StimuliSpec
from .constant import ConstantStrategy
from .deterministic import (
    GaussianLinspaceStrategy,
    LogUniformLinspaceStrategy,
    RampStrategy,
    SequentialStrategy,
)
from .random import GaussianStrategy, LogUniformStrategy, SawStrategy, UniformStrategy
from .structured import (
    CustomStrategy,
    FaceIdentityStrategy,
    IdentityStrategy,
    UlpSweepStrategy,
)


class DistributionStrategy(Protocol):
    """Protocol every distribution strategy implements.

    Concrete strategies are stateless. ``short_circuit`` controls which mode
    ``_generate_source_tensor`` uses:

    * ``False`` → per-face loop via :meth:`generate_face`;
    * ``True``  → single :meth:`generate_full_tensor` call (bypassing the loop).

    Strategies that genuinely support both modes (e.g. sequential / ramp /
    linspace variants) implement both methods. Strategies that only make sense
    in one mode raise ``NotImplementedError`` in the other.
    """

    short_circuit: bool

    def generate_face(
        self,
        spec: StimuliSpec,
        stimuli_format: DataFormat,
        face_r_dim: int,
        size: int,
        generator: Optional[torch.Generator],
    ) -> torch.Tensor: ...

    def generate_full_tensor(
        self,
        spec: StimuliSpec,
        stimuli_format: DataFormat,
        num_elements: int,
        input_dimensions: Optional[List[int]],
        generator: Optional[torch.Generator],
    ) -> torch.Tensor: ...


_STRATEGIES: Dict[DistributionKind, DistributionStrategy] = {
    DistributionKind.UNIFORM: UniformStrategy(),
    DistributionKind.GAUSSIAN: GaussianStrategy(),
    DistributionKind.LOG_UNIFORM: LogUniformStrategy(),
    DistributionKind.SAW: SawStrategy(),
    DistributionKind.RAMP: RampStrategy(),
    DistributionKind.GAUSSIAN_LINSPACE: GaussianLinspaceStrategy(),
    DistributionKind.LOG_UNIFORM_LINSPACE: LogUniformLinspaceStrategy(),
    DistributionKind.SEQUENTIAL: SequentialStrategy(),
    DistributionKind.IDENTITY: IdentityStrategy(),
    DistributionKind.FACE_IDENTITY: FaceIdentityStrategy(),
    DistributionKind.CUSTOM: CustomStrategy(),
    DistributionKind.ULP_SWEEP: UlpSweepStrategy(),
    DistributionKind.CONSTANT: ConstantStrategy(),
}


def lookup_strategy(kind: DistributionKind) -> DistributionStrategy:
    """Return the strategy registered for ``kind``."""
    return _STRATEGIES[kind]
