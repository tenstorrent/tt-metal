# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Top-level Devstral-2 / Ministral3 model.

HF reference (``Ministral3Model`` + ``Ministral3ForCausalLM``)::

    h = embed_tokens(input_ids)
    for layer in layers:
        h = layer(h, ...)
    h = norm(h)
    # CausalLM-only:
    logits = lm_head(h)

Embedding is run on the host (cheap, vocab-bound; avoids embed-vocab-sized device weight on every
chip) and uploaded directly to L1. Activations stay in L1 between norm / linear / residual ops to
avoid DRAM tilize. Final norm runs on device. ``lm_head`` is optional and column-parallel.
"""

from __future__ import annotations

from typing import Optional

import torch
import ttnn

from models.experimental.devstral2_large.tt.model_args import Devstral2Args
from models.experimental.devstral2_large.tt.tt_ministral3_decoder_layer import TtDecoderLayer
from models.experimental.devstral2_large.tt.tt_ministral_rotary_emb import TtRotaryEmbedding
from models.experimental.devstral2_large.tt.tt_ministralrmsnorm import TtRMSNorm

__all__ = ["TtMinistral3Model", "TtMinistral3ForCausalLM"]


def _embed_host(
    input_ids: torch.Tensor,
    state_dict: dict,
    args: Devstral2Args,
    mesh_device,
    *,
    dtype: ttnn.DataType,
) -> ttnn.Tensor:
    """Look up ``embed_tokens.weight`` host-side and upload the resulting activation (replicated)."""
    embed_w = state_dict["model.embed_tokens.weight"].to(torch.bfloat16)
    h = torch.nn.functional.embedding(input_ids, embed_w)  # (B, S, dim)
    h = h.reshape(1, 1, -1, args.hidden_size)
    return ttnn.from_torch(
        h,
        device=mesh_device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


class TtMinistral3Model:
    """The Ministral3 decoder stack: embed → layers → final norm."""

    def __init__(
        self,
        args: Devstral2Args,
        mesh_device,
        state_dict: dict,
        tt_ccl,
        *,
        dtype: Optional[ttnn.DataType] = None,
        weight_cache_path: Optional[str] = None,
        num_layers: Optional[int] = None,
    ) -> None:
        self.args = args
        self.mesh_device = mesh_device
        self.state_dict = state_dict
        self.tt_ccl = tt_ccl
        self.dtype = dtype or args.weight_dtype
        self.num_layers = int(num_layers if num_layers is not None else args.num_hidden_layers)

        self.rotary_emb = TtRotaryEmbedding(args, mesh_device, dtype=self.dtype)
        self.layers = [
            TtDecoderLayer(
                args,
                mesh_device,
                state_dict,
                layer_idx=i,
                tt_ccl=tt_ccl,
                rotary_emb=self.rotary_emb,
                dtype=self.dtype,
                weight_cache_path=weight_cache_path,
            )
            for i in range(self.num_layers)
        ]
        self.norm = TtRMSNorm(args, mesh_device, state_dict, "model.norm.weight", dtype=self.dtype)

    def __call__(
        self,
        input_ids: torch.Tensor,
        *,
        mode: str = "prefill",
        start_pos: int = 0,
        current_pos_host: Optional[torch.Tensor] = None,
        user_id: int = 0,
    ) -> ttnn.Tensor:
        act_mem = self.args.get_activation_mem_config(mode, self.mesh_device)
        h = _embed_host(input_ids, self.state_dict, self.args, self.mesh_device, dtype=self.args.activation_dtype)
        for layer in self.layers:
            h = layer(
                h,
                mode=mode,
                start_pos=start_pos,
                current_pos_host=current_pos_host,
                user_id=user_id,
            )
        h = self.norm(h, memory_config=act_mem)
        return h

    def forward(self, *args, **kwargs):
        return self(*args, **kwargs)


class TtMinistral3ForCausalLM:
    """Causal LM: shares the base model and adds an optional column-parallel ``lm_head``."""

    def __init__(
        self,
        args: Devstral2Args,
        mesh_device,
        state_dict: dict,
        tt_ccl,
        **kwargs,
    ) -> None:
        self.model = TtMinistral3Model(args, mesh_device, state_dict, tt_ccl, **kwargs)
        self.args = args
        self.mesh_device = mesh_device

        # lm_head: ``(vocab_size, hidden_size)`` HF → ``(hidden_size, vocab_size)`` TT, colwise TP on out.
        lm_w_key = "lm_head.weight" if "lm_head.weight" in state_dict else "model.embed_tokens.weight"
        lm_w = state_dict[lm_w_key].to(torch.bfloat16).T.contiguous()  # (hidden, vocab)
        if args.cluster_axis == 1:
            dims = (None, -1)
        else:
            dims = (-1, None)
        self.lm_head = ttnn.from_torch(
            lm_w,
            device=mesh_device,
            dtype=args.weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=dims, mesh_shape=args.mesh_shape),
        )

    def __call__(self, *fwd_args, **fwd_kwargs) -> ttnn.Tensor:
        h = self.model(*fwd_args, **fwd_kwargs)
        logits = ttnn.linear(
            h,
            self.lm_head,
            dtype=self.args.activation_dtype,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        return logits

    def forward(self, *args, **kwargs):
        return self(*args, **kwargs)
