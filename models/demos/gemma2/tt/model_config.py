# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Gemma-2 model configuration.

Gemma-2 (2B / 9B / 27B) is a plain dense text decoder (GQA + GeGLU + sandwich
RMSNorm + interleaved sliding-window attention). It therefore reuses the entire
generic ``tt_transformers`` stack (Transformer, attention, decoder, RMSNorm,
ScaledEmbedding, generator, sliding-window rope) and only overrides the two
Gemma-specific parameters that the generic path leaves unset:

* ``rms_norm_add_unit_offset`` - Gemma RMSNorm computes ``x_normed * (1 + weight)``
  (the checkpoint weights are ~0-centered). Without the +1 offset every norm
  multiplies activations by ~0 and decode collapses to a single repeated token.
* ``embed_scale`` - Gemma scales input embeddings by ``sqrt(hidden_size)``.

This mirrors how the dedicated Gemma-3 path sets the same two values in its own
``_set_model_specific_params`` override (see
``models/demos/multimodal/gemma3/tt/model_config.py``), keeping the shared
``tt_transformers`` configuration untouched.
"""

from models.tt_transformers.tt.model_config import ModelArgs as TTModelArgs


class ModelArgs(TTModelArgs):
    def _set_model_specific_params(self):
        # Gemma-2: RMSNorm uses (1 + weight); input embeddings scaled by sqrt(hidden).
        self.rms_norm_add_unit_offset = True
        self.embed_scale = self.dim**0.5
