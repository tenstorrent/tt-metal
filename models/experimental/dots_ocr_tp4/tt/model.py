# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Full dots.ocr text-decoder prefill in TP4 — a stack of decoder blocks.

This is the prefill body only (embeddings / final norm / LM head are out of
scope for this rebuild step). Input and output are the replicated hidden stream.
"""

import ttnn

from models.experimental.dots_ocr_tp4.tt.decoder_block import DotsOCRDecoderBlockTP4
from models.experimental.dots_ocr_tp4.tt.lm_head import DotsOCRLMHeadTP4
from models.experimental.tt_symbiote.core.module import TTNNModule


class DotsOCRPrefillModelTP4(TTNNModule):
    def __init__(self, mesh_device, config, weight_dtype=ttnn.bfloat16):
        super().__init__()
        self.mesh_device = mesh_device
        self.config = config
        self.weight_dtype = weight_dtype
        self.layers = []
        self.head = None  # optional DotsOCRLMHeadTP4 (final norm + LM head)

    def to_device(self, device):
        super().to_device(device)
        for layer in self.layers:
            layer.to_device(device)
        if self.head is not None:
            self.head.to_device(device)
        return self

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
        m.to_device(mesh_device)
        m._preprocessed_weight = True
        m._weights_on_device = True
        return m

    def forward(self, x: ttnn.Tensor, past_key_value=None, cache_position=None) -> ttnn.Tensor:
        """Decoder body -> replicated hidden [B, S, H].

        Pass ``past_key_value`` (paged KV cache) to fill it during prefill or to
        read/extend it during decode (seq_len==1). ``cache_position`` is the
        new-token position tensor used by the decode path."""
        for layer in self.layers:
            x = layer.forward(x, past_key_value=past_key_value, cache_position=cache_position)
        return x

    def forward_with_head(
        self, x: ttnn.Tensor, last_token_only: bool = True, return_token: bool = True, token_index=None
    ):
        """Full prefill: decoder body + final norm + LM head + argmax.

        Returns (logits, token_ids). Requires the head to have been built.
        ``token_index`` selects the sequence position to read (see the LM head;
        used to read the real last token when the input is right-padded)."""
        assert self.head is not None, "model head not built; pass torch_norm/torch_lm_head to from_torch"
        hidden = self.forward(x)
        return self.head.forward(
            hidden, last_token_only=last_token_only, return_token=return_token, token_index=token_index
        )

    def prefill_with_head(self, x: ttnn.Tensor, past_key_value, token_index, return_token: bool = True):
        """Prefill that FILLS the paged KV cache, then runs the head at the real
        last token. ``x`` may be right-padded to a tile multiple; pass the real
        last position as ``token_index``. Returns (logits, token_ids)."""
        assert self.head is not None, "model head not built"
        hidden = self.forward(x, past_key_value=past_key_value)
        return self.head.forward(hidden, last_token_only=True, return_token=return_token, token_index=token_index)

    def decode_with_head(self, x: ttnn.Tensor, past_key_value, cache_position, return_token: bool = True):
        """Single-token decode reading/extending the paged KV cache, then head.

        ``x`` is the new token's embedding [B, 1, H] (replicated); ``cache_position``
        is the int32 position tensor of that token. Returns (logits, token_ids)."""
        assert self.head is not None, "model head not built"
        hidden = self.forward(x, past_key_value=past_key_value, cache_position=cache_position)
        return self.head.forward(hidden, last_token_only=True, return_token=return_token, token_index=None)
