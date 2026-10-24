# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""FLUX.2 Klein DiT on a tt_dit mesh (see ``transformer_flux2_klein.py`` for the tensor contract)."""

from .block_flux2_klein import Flux2KleinSingleBlock, apply_exact_attention_profile
from .config_flux2_klein import FLUX2_KLEIN_SDPA_Q_CHUNK_MAX, FLUX2_KLEIN_SIZES, Flux2KleinConfig, Flux2KleinSizeRow
from .rope_flux2_klein import rope_cos_sin, rope_to_device
from .time_flux2_klein import Flux2KleinModulation, Flux2KleinTimeModulation
from .transformer_flux2_klein import Flux2KleinPlan, Flux2KleinSequenceLayout, Flux2KleinTransformer

__all__ = [
    "FLUX2_KLEIN_SDPA_Q_CHUNK_MAX",
    "FLUX2_KLEIN_SIZES",
    "Flux2KleinConfig",
    "Flux2KleinModulation",
    "Flux2KleinPlan",
    "Flux2KleinSequenceLayout",
    "Flux2KleinSingleBlock",
    "Flux2KleinSizeRow",
    "Flux2KleinTimeModulation",
    "Flux2KleinTransformer",
    "apply_exact_attention_profile",
    "rope_cos_sin",
    "rope_to_device",
]
