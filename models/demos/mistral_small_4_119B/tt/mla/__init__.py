# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Mistral Small 4 latent attention (MLA-style).

- :class:`TtMistral4MLA1D` / :class:`TtMistral4MLA2D`: eager PyTorch / HF parity (unit tests).
- :class:`MistralSmall4MLA1D` / :class:`MistralSmall4MLA2D`: ``ttnn`` stack aligned with DeepSeek ``MLA1D`` / ``MLA2D``.

See ``mla1d.py`` / ``mla2d.py``.
"""

from __future__ import annotations

from models.demos.mistral_small_4_119B.tt.mla.mla1d import (
    MLALoadResult,
    build_and_load_mla,
    load_ttmistral4_mla_from_sharded_safetensors,
    read_mla_tensors_from_sharded_checkpoint,
    MistralSmall4MLA1D,
    TtMistral4MLA1D,
    build_prefill_matmul_program_config,
    mistral4_hf_config_for_mla,
    pad_batch_to_dram_banks,
    pad_n_to_dram_banks,
)
from models.demos.mistral_small_4_119B.tt.mla.mla2d import (
    MistralSmall4MLA2D,
    TtMistral4MLA2D,
    build_and_load_mla2d,
    load_ttmistral4_mla2d_from_sharded_safetensors,
)

__all__ = [
    "MLALoadResult",
    "MistralSmall4MLA1D",
    "MistralSmall4MLA2D",
    "TtMistral4MLA1D",
    "TtMistral4MLA2D",
    "build_prefill_matmul_program_config",
    "mistral4_hf_config_for_mla",
    "pad_batch_to_dram_banks",
    "pad_n_to_dram_banks",
    "build_and_load_mla",
    "build_and_load_mla2d",
    "load_ttmistral4_mla2d_from_sharded_safetensors",
    "load_ttmistral4_mla_from_sharded_safetensors",
    "read_mla_tensors_from_sharded_checkpoint",
]
