# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn

from models.common.lightweightmodule import LightweightModule


def _to_device_rm(torch_tensor, device):
    """Upload a torch tensor to device in row-major bfloat16."""
    return ttnn.from_torch(
        torch_tensor.to(torch.bfloat16),
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        dtype=ttnn.bfloat16,
    )


class TtXttsTextEmbedding(LightweightModule):
    def __init__(self, state_dict, device):
        """Load text and positional embedding weights onto device."""
        super().__init__()
        self.device = device
        self.text_emb_weight = _to_device_rm(state_dict["gpt.text_embedding.weight"], device)
        self.text_pos_weight = _to_device_rm(state_dict["gpt.text_pos_embedding.emb.weight"], device)

    def forward(self, text_ids):
        """Embed text token ids with learned positional encodings."""
        seq = text_ids.shape[1]
        ids_tt = ttnn.from_torch(
            text_ids.to(torch.int32), layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device, dtype=ttnn.uint32
        )
        pos_tt = ttnn.arange(0, seq, 1, dtype=ttnn.uint32, device=self.device, layout=ttnn.ROW_MAJOR_LAYOUT)
        pos_tt = ttnn.reshape(pos_tt, (1, seq))

        tok = ttnn.to_layout(ttnn.embedding(ids_tt, self.text_emb_weight), ttnn.TILE_LAYOUT)
        ttnn.deallocate(ids_tt)
        pos = ttnn.to_layout(ttnn.embedding(pos_tt, self.text_pos_weight), ttnn.TILE_LAYOUT)
        ttnn.deallocate(pos_tt)
        emb = ttnn.add(tok, pos)
        ttnn.deallocate(tok)
        ttnn.deallocate(pos)
        return emb
