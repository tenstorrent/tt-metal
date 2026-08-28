# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native, tensor-parallel TTNN port of the dual-stream block stack
(`transformer_blocks`) of `black-forest-labs/FLUX.2-klein-9B (subfolder `transformer`)`.

`Flux2Transformer2DModel.transformer_blocks` is this DiT's encoder stack: the 8
`Flux2TransformerBlock`s that jointly attend over the image and text streams
before they are concatenated for the single-stream half of the model. The
planner pointed this component at
`models/tt_transformers/tt/multimodal/llama_vision_encoder.py`, whose structure
(a stack of identical blocks, run in sequence, with the residual stream carried
between them) is the same; only the block body differs, so this port keeps the
stack-level shape and replaces the block with `TtFlux2TransformerBlock` from
`_flux2_ttnn.py`.

Accepts either the whole `ModuleList` (runs every block in sequence, threading
both streams) or a single block -- the per-component PCC harness resolves a
`ModuleList` to its first element, so that is what it validates.

Tensor parallelism (TP=8): each block is column-parallel through the q/k/v and
feed-forward input projections and row-parallel + `all_reduce` through
`to_out` / `to_add_out` / `linear_out`; the residual stream stays replicated and
full-width between blocks, so the stack composes without any extra collective.
See `_flux2_ttnn.py` for the derivation.

The forward is pure ttnn: no torch math and no device->host readback.
"""

from __future__ import annotations

from models.demos.flux_2_klein_9b.transformer._stubs._flux2_ttnn import TtFlux2TransformerBlock


class TtEncoderStack:
    def __init__(self, device, torch_module) -> None:
        if torch_module is None:
            raise RuntimeError("encoder_stack needs the torch reference module for weights")
        self.device = device
        blocks = list(torch_module) if hasattr(torch_module, "__len__") else [torch_module]
        self.blocks = [TtFlux2TransformerBlock(device, b) for b in blocks]

    @classmethod
    def build(cls, device, torch_module):
        return cls(device, torch_module)

    def __call__(
        self,
        hidden_states,
        encoder_hidden_states=None,
        temb_mod_img=None,
        temb_mod_txt=None,
        image_rotary_emb=None,
        joint_attention_kwargs=None,
        **kwargs,
    ):
        for block in self.blocks:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb_mod_img=temb_mod_img,
                temb_mod_txt=temb_mod_txt,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )
        return encoder_hidden_states, hidden_states


def build(device, torch_module=None):
    return TtEncoderStack.build(device, torch_module)


def encoder_stack(device, torch_module=None):
    return TtEncoderStack.build(device, torch_module)
