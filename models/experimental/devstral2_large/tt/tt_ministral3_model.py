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

``__call__`` takes ``input_ids`` and optional decode ``current_pos`` as ``ttnn.Tensor`` values only.
Weight upload uses host tensors at construction time; the forward path is device ops only.
"""

from __future__ import annotations

from typing import Optional

import ttnn

from models.experimental.devstral2_large.tt.model_args import Devstral2Args
from models.experimental.devstral2_large.tt.tt_ministral3_decoder_layer import TtDecoderLayer
from models.experimental.devstral2_large.tt.tt_ministral_rotary_emb import TtRotaryEmbedding
from models.experimental.devstral2_large.tt.tt_ministralrmsnorm import TtRMSNorm

__all__ = ["TtEmbedTokens", "TtMinistral3Model", "TtMinistral3ForCausalLM"]


class TtEmbedTokens:
    """Device ``embed_tokens`` via ``ttnn.embedding`` (replicated across the mesh)."""

    def __init__(
        self,
        args: Devstral2Args,
        mesh_device,
        embed_weight,
        *,
        dtype: ttnn.DataType,
    ) -> None:
        self.args = args
        self.mesh_device = mesh_device
        # ``ttnn.embedding`` weight layout: ``(vocab_size, hidden_size)``, row-major on device.
        self.weight = ttnn.from_torch(
            embed_weight,
            device=mesh_device,
            dtype=dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    def __call__(self, input_ids: ttnn.Tensor, *, memory_config: ttnn.MemoryConfig) -> ttnn.Tensor:
        return ttnn.embedding(
            input_ids,
            self.weight,
            layout=ttnn.TILE_LAYOUT,
            memory_config=memory_config,
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
        self.tt_ccl = tt_ccl
        self.dtype = dtype or args.weight_dtype
        self.num_layers = int(num_layers if num_layers is not None else args.num_hidden_layers)

        embed_w = state_dict["model.embed_tokens.weight"]
        self.embed_tokens = TtEmbedTokens(
            args,
            mesh_device,
            embed_w,
            dtype=self.dtype,
        )
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

    def _reshape_embeddings(self, hidden_states: ttnn.Tensor, input_ids: ttnn.Tensor, *, mode: str) -> ttnn.Tensor:
        """``(batch, seq, hidden)`` → ``(1, 1, seq, hidden)`` prefill or ``(1, 1, batch, hidden)`` decode."""
        hidden_size = self.args.hidden_size
        if mode == "decode":
            batch_size = int(input_ids.shape[0])
            return ttnn.reshape(hidden_states, (1, 1, batch_size, hidden_size))
        seq_len = int(input_ids.shape[-1])
        return ttnn.reshape(hidden_states, (1, 1, seq_len, hidden_size))

    def __call__(
        self,
        input_ids: ttnn.Tensor,
        *,
        mode: str = "prefill",
        start_pos: int = 0,
        current_pos: Optional[ttnn.Tensor] = None,
        user_id: int = 0,
    ) -> ttnn.Tensor:
        act_mem = self.args.get_activation_mem_config(mode, self.mesh_device)
        hidden_states = self.embed_tokens(input_ids, memory_config=act_mem)
        hidden_states = self._reshape_embeddings(hidden_states, input_ids, mode=mode)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                mode=mode,
                start_pos=start_pos,
                current_pos=current_pos,
                user_id=user_id,
            )
        return self.norm(hidden_states, memory_config=act_mem)

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

        lm_w_key = "lm_head.weight" if "lm_head.weight" in state_dict else "model.embed_tokens.weight"
        lm_w = state_dict[lm_w_key]
        if args.cluster_axis == 1:
            dims = (None, -1)
        else:
            dims = (-1, None)
        self.lm_head = ttnn.from_torch(
            lm_w.T.contiguous(),
            device=mesh_device,
            dtype=args.weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=dims, mesh_shape=args.mesh_shape),
        )

    def __call__(self, *fwd_args, **fwd_kwargs) -> ttnn.Tensor:
        hidden_states = self.model(*fwd_args, **fwd_kwargs)
        return ttnn.linear(
            hidden_states,
            self.lm_head,
            dtype=self.args.activation_dtype,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

    def forward(self, *args, **kwargs):
        return self(*args, **kwargs)
