# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""RMSNorm for Mistral-4 decoder blocks (HF-compatible PyTorch).

For ``ttnn`` RMSNorm (weights + prefill/decode configs), see :mod:`models.demos.mistral_small_4_119B.tt.rms_norm`.
"""

from __future__ import annotations

from transformers.models.mistral4.modeling_mistral4 import Mistral4RMSNorm


class TtMistral4RMSNorm(Mistral4RMSNorm):
    """Same math and parameter layout as HF :class:`Mistral4RMSNorm` (T5-style RMSNorm)."""


__all__ = ["TtMistral4RMSNorm"]
