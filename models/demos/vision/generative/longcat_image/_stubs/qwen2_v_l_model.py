# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `qwen2_v_l_model` (transformers ``Qwen2_5_VLModel``) —
the LongCat-Image text encoder's inner model (Qwen2.5-VL 7B), submodule
``text_encoder.model``. Text-only path (no vision tower for text→image).

This is EXACTLY the `qwen2_v_l_for_conditional_generation` computation minus the
final ``lm_head`` — it returns ``last_hidden_state`` (the post-final-RMSNorm
hidden state, [1,S,3584]) instead of vocab logits. So it reuses the graduated
native ``_TextEncoder`` port verbatim (same bf16-weight/fp32-activation stack,
the same critical precision fixes: exact slice+concat GQA repeat instead of the
bf16-rounding ``ttnn.repeat_interleave``, manual fp32 softmax on the peaked
late-layer rows, and bf16-limb 16-bit-emulated SwiGLU MLP). Only ``__call__``
differs: it stops after the final norm and returns the hidden state.

The submodule handed in is the ``Qwen2_5_VLModel`` itself (it owns
``.language_model``); ``_TextEncoder.__init__`` expects ``te.model.language_model``
+ ``te.lm_head``, so a tiny shim exposes ``.model`` = this model and ``.lm_head``
= None (unused on this path).
"""

from __future__ import annotations

import ttnn
from models.demos.vision.generative.longcat_image._stubs.qwen2_v_l_for_conditional_generation import _TextEncoder


class _Shim:
    """Adapts a ``Qwen2_5_VLModel`` to the ``te.model.language_model`` / ``te.lm_head``
    shape that ``_TextEncoder.__init__`` reads. ``lm_head`` is unused here."""

    def __init__(self, qwen_model):
        self.model = qwen_model  # exposes .language_model
        self.lm_head = None

    def eval(self):
        return self


class _Model(_TextEncoder):
    def __call__(self, hidden_states=None, input_ids=None, attention_mask=None, output_hidden_states=None, **_ignored):
        if input_ids is None:
            raise ValueError("qwen2_v_l_model stub requires `input_ids`")
        S = int(input_ids.reshape(-1).shape[0])
        x = self._embed(input_ids)  # [1,S,hidden]
        # Free the 1.09 GB bf16 embedding table right after use (the fp32 weight
        # stack fills the card to the edge); it is unused for the rest of forward.
        if self._emb_w is not None:
            ttnn.deallocate(self._emb_w)
            self._emb_w = None
        cos, sin = self._rope_tables(S)
        mask = self._causal_mask(attention_mask, S)

        for blk in self.lm.layers:
            x = self._layer(blk, x, cos, sin, mask, S)

        x = self._rmsnorm(x, self.lm.norm)  # final norm -> last_hidden_state
        return (x,)


def build(device, torch_module):
    """PCC-harness entry point: native TTNN Qwen2.5-VL inner model (text path)."""
    return _Model(device, _Shim(torch_module))
