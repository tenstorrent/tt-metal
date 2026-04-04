# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Gemma 4 E4B Text Model

Extends the base Transformer with Gemma 4 specific features:
- Final logit soft-capping: tanh(logits / 30) * 30
- Per-layer input gating (future)
- KV cache sharing (future)
- embed_tokens_per_layer (future)
"""

import ttnn
from models.tt_transformers.tt.model import Transformer


class TtGemma4TextModel(Transformer):
    """
    Gemma 4 E4B text model.

    This subclass adds Gemma 4 specific post-processing (logit soft-capping)
    and will be extended with per-layer input gating, KV cache sharing,
    and embed_tokens_per_layer in subsequent days.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Get softcapping value from config
        self.final_logit_softcapping = getattr(self.args, "final_logit_softcapping", None)

    def _apply_logit_softcapping(self, logits):
        """
        Apply logit soft-capping: tanh(logits / cap) * cap

        This bounds logits to the range (-cap, cap), preventing extreme values
        while preserving relative ordering.
        """
        if self.final_logit_softcapping is None:
            return logits

        cap = self.final_logit_softcapping
        # logits = tanh(logits / cap) * cap
        # Using ttnn ops: multiply by 1/cap, tanh, multiply by cap
        inv_cap = 1.0 / cap
        logits = ttnn.multiply(logits, inv_cap)
        logits = ttnn.tanh(logits)
        logits = ttnn.multiply(logits, cap)
        return logits

    def _apply_norm_and_lm_head(self, x):
        """Override to add logit soft-capping after LM head."""
        logits = super()._apply_norm_and_lm_head(x)
        logits = self._apply_logit_softcapping(logits)
        return logits
