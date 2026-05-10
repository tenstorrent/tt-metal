# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
``embed_tokens`` bring-up for Mistral Small 4 text (hub ``language_model.model.*``).

The full vocabulary table is **~1 GiB bf16**; it stays on **host** as torch tensors. The forward
path uses ``F.embedding`` for bit-exact parity with HF, then uploads token hidden states to the
mesh in ``[1, 1, S, H]`` TILE layout for chaining into :class:`TtMistral4DecoderLayerAttnPrefillBlock`.

A future step is a device-resident or sharded embedding table; this module exists to lock PCC
and wire shape/dtype into the decoder stack.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import ttnn

from models.common.lightweightmodule import LightweightModule
from models.experimental.mistral_small_4_119b.constants import TEXT_MODEL_EMBED_TOKENS_WEIGHT_KEY
from models.experimental.mistral_small_4_119b.tt.mistral4_self_attention import _torch_for_ttnn_upload


class TtMistral4EmbedTokensPrefill(LightweightModule):
    """
    Prefill token embedding: host ``F.embedding`` → TTNN ``[1, 1, S, H]`` replicated on ``device``.

    Args:
        device: ``MeshDevice`` or single device for output tensors.
        state_dict: Hub map containing :data:`~constants.TEXT_MODEL_EMBED_TOKENS_WEIGHT_KEY`.
    """

    def __init__(self, device, state_dict: dict):
        super().__init__()
        if TEXT_MODEL_EMBED_TOKENS_WEIGHT_KEY not in state_dict:
            raise KeyError(f"state_dict missing {TEXT_MODEL_EMBED_TOKENS_WEIGHT_KEY!r}")
        self.device = device
        self.weight = _torch_for_ttnn_upload(state_dict[TEXT_MODEL_EMBED_TOKENS_WEIGHT_KEY])
        if self.weight.ndim != 2:
            raise ValueError(f"embed weight must be 2D [V, H], got shape {tuple(self.weight.shape)}")

    def forward(self, input_ids: torch.Tensor) -> ttnn.Tensor:
        """
        Args:
            input_ids: ``[batch, seq]`` int64 (or int32) token ids on CPU.

        Returns:
            Hidden states ``[batch, 1, seq, hidden]`` bf16 TILE on ``device``.
        """
        if not isinstance(input_ids, torch.Tensor):
            raise TypeError("input_ids must be a torch.Tensor")
        if input_ids.dtype not in (torch.int64, torch.int32, torch.long):
            input_ids = input_ids.long()
        ids = input_ids.detach()
        if ids.device.type != "cpu":
            ids = ids.cpu()
        hidden = F.embedding(ids, self.weight)
        hidden_b1sh = hidden.unsqueeze(1)
        return ttnn.from_torch(
            hidden_b1sh,
            device=self.device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device=self.device),
        )
