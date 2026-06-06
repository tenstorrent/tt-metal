# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Full dots.ocr text-decoder prefill in TP4 — a stack of decoder blocks.

This is the prefill body only (embeddings / final norm / LM head are out of
scope for this rebuild step). Input and output are the replicated hidden stream.
"""

import ttnn

from models.experimental.dots_ocr_tp4.tt.decoder_block import DotsOCRDecoderBlockTP4
from models.experimental.dots_ocr_tp4.tt.lm_head import DotsOCRLMHeadTP4


class DotsOCRPrefillModelTP4:
    def __init__(self, mesh_device, config, weight_dtype=ttnn.bfloat16):
        self.mesh_device = mesh_device
        self.config = config
        self.weight_dtype = weight_dtype
        self.layers = []
        self.head = None  # optional DotsOCRLMHeadTP4 (final norm + LM head)

    @classmethod
    def from_torch(
        cls, mesh_device, config, torch_layers, torch_norm=None, torch_lm_head=None, weight_dtype=ttnn.bfloat16
    ):
        """``torch_layers`` is an iterable of torch decoder blocks (e.g. a
        ``TorchDecoderStack.layers`` or HF ``model.model.layers``). Pass
        ``torch_norm`` (final RMSNorm) and ``torch_lm_head`` to also build the
        TP4 model head so ``forward`` can emit next-token logits/ids."""
        m = cls(mesh_device, config, weight_dtype=weight_dtype)
        for idx, torch_layer in enumerate(torch_layers):
            m.layers.append(
                DotsOCRDecoderBlockTP4.from_torch(
                    mesh_device, config, torch_layer, layer_idx=idx, weight_dtype=weight_dtype
                )
            )
        if torch_norm is not None and torch_lm_head is not None:
            m.head = DotsOCRLMHeadTP4.from_torch(
                mesh_device, config, torch_norm, torch_lm_head, weight_dtype=weight_dtype
            )
        return m

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """Decoder body only -> replicated hidden [B, S, H]."""
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def forward_with_head(self, x: ttnn.Tensor, last_token_only: bool = True, return_token: bool = True):
        """Full prefill: decoder body + final norm + LM head + argmax.

        Returns (logits, token_ids). Requires the head to have been built."""
        assert self.head is not None, "model head not built; pass torch_norm/torch_lm_head to from_torch"
        hidden = self.forward(x)
        return self.head.forward(hidden, last_token_only=last_token_only, return_token=return_token)
