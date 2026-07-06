# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `qwen2_v_l_text_model` (transformers
``Qwen2_5_VLTextModel``) — the LongCat-Image text encoder's decoder stack,
submodule ``text_encoder.model.language_model``.

This is EXACTLY the graduated `qwen2_v_l_model` computation minus the token
embedding: its real forward is driven by ``inputs_embeds`` [1,S,3584] (already
``embed_tokens(input_ids)``; ``input_ids`` is None on this path) rather than
``input_ids``. So it reuses the graduated native ``_TextEncoder`` port verbatim
(same bf16-weight/fp32-activation stack, the same critical precision fixes: exact
slice+concat GQA repeat instead of the bf16-rounding ``ttnn.repeat_interleave``,
manual fp32 softmax on the peaked late-layer rows, and bf16-limb 16-bit-emulated
SwiGLU MLP). Only ``__call__`` differs: it starts from the given
``inputs_embeds`` (no ``_embed`` step), runs the 28 decoder layers, applies the
final RMSNorm, and returns ``last_hidden_state`` [1,S,3584].

The submodule handed in IS the ``Qwen2_5_VLTextModel`` (the language_model);
``_TextEncoder.__init__`` expects ``te.model.language_model`` + ``te.lm_head``,
so a tiny shim exposes ``.model.language_model`` = this model and ``.lm_head`` =
None (unused on this path).
"""

from __future__ import annotations

from models.demos.vision.generative.longcat_image._stubs.qwen2_v_l_for_conditional_generation import F32, _TextEncoder


class _Shim:
    """Adapts a ``Qwen2_5_VLTextModel`` (the ``language_model`` itself) to the
    ``te.model.language_model`` / ``te.lm_head`` shape ``_TextEncoder.__init__``
    reads. ``lm_head`` is unused here (this path stops at last_hidden_state)."""

    class _Inner:
        def __init__(self, language_model):
            self.language_model = language_model

    def __init__(self, language_model):
        self.model = _Shim._Inner(language_model)
        self.lm_head = None

    def eval(self):
        return self


class _TextModel(_TextEncoder):
    def __call__(
        self, hidden_states=None, inputs_embeds=None, attention_mask=None, output_hidden_states=None, **_ignored
    ):
        # Driven by already-embedded `inputs_embeds`; accept the harness primary
        # via `hidden_states` as an alias in case it lands there.
        emb = inputs_embeds if inputs_embeds is not None else hidden_states
        if emb is None:
            raise ValueError("qwen2_v_l_text_model stub requires `inputs_embeds`")
        x = self._to_ttnn(emb, dtype=F32)  # [1,S,hidden] fp32
        S = int(x.shape[1])
        cos, sin = self._rope_tables(S)
        mask = self._causal_mask(attention_mask, S)

        for blk in self.lm.layers:
            x = self._layer(blk, x, cos, sin, mask, S)

        x = self._rmsnorm(x, self.lm.norm)  # final norm -> last_hidden_state
        return (x,)


def build(device, torch_module):
    """PCC-harness entry point: native TTNN Qwen2.5-VL language model (text path)."""
    return _TextModel(device, _Shim(torch_module))
