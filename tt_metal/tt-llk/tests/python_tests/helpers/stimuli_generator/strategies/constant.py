# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Constant-fill distribution strategy."""

from typing import List, Optional

import torch

from ...format_config import DataFormat
from ..spec import StimuliSpec
from ..utils import _get_dtype_for_format, _get_integer_bounds


class ConstantStrategy:
    """Fills every element with *spec.value*."""

    short_circuit = False

    def generate_face(
        self,
        spec: StimuliSpec,
        stimuli_format: DataFormat,
        face_r_dim: int,
        size: int,
        generator: Optional[torch.Generator],
    ) -> torch.Tensor:
        dtype = _get_dtype_for_format(stimuli_format)
        if stimuli_format.is_integer():
            int_min, int_max = _get_integer_bounds(stimuli_format)
            clamped = max(int_min, min(int(spec.value), int_max))
            return torch.full((size,), clamped, dtype=dtype)
        return torch.full((size,), spec.value, dtype=dtype)

    def generate_full_tensor(
        self,
        spec: StimuliSpec,
        stimuli_format: DataFormat,
        num_elements: int,
        input_dimensions: Optional[List[int]],
        generator: Optional[torch.Generator],
    ) -> torch.Tensor:
        raise NotImplementedError(
            "distribution='constant' is per-face only; use generate_face, "
            "not generate_full_tensor"
        )
