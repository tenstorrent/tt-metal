# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Public entry-point for the stimuli generator package.

Re-exports the symbols historically exposed under `helpers.stimuli_generator`
so all existing `from helpers.stimuli_generator import X` imports continue to
resolve. Internally the package is organised as:

    spec.py        — StimuliSpec, DistributionKind
    generator.py   — generate_face, generate_stimuli, _generate_source_tensor,
                     default_spec_for_format (orchestration only)
    utils.py       — shared helpers (intervals, format, tile/face count,
                     MX clamp, matmul/L1 layout, magnitude helpers)
    strategies/    — one stateless strategy per DistributionKind, looked up
                     via _STRATEGIES at dispatch time
"""

from .generator import (
    default_spec_for_format,
    generate_face,
    generate_stimuli,
)
from .spec import DistributionKind, StimuliSpec
from .utils import (
    apply_log_uniform_magnitudes,
    calculate_tile_and_face_counts,
    calculate_tile_and_face_counts_w_tile_dimensions,
    compute_safe_input_magnitude_range,
    convert_to_l1_view,
    format_elem_max,
    generate_face_matmul_data,
)

__all__ = [
    # spec.py
    "DistributionKind",
    "StimuliSpec",
    # generator.py
    "default_spec_for_format",
    "generate_face",
    "generate_stimuli",
    # utils.py
    "apply_log_uniform_magnitudes",
    "calculate_tile_and_face_counts",
    "calculate_tile_and_face_counts_w_tile_dimensions",
    "compute_safe_input_magnitude_range",
    "convert_to_l1_view",
    "format_elem_max",
    "generate_face_matmul_data",
]
