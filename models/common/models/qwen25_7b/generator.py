# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Greedy host-side decoding helper (no v1 ``Generator``): logits → ``argmax`` via
``to_torch_auto_compose`` for multi-device vocab shards.
"""

from __future__ import annotations

import torch

import ttnn
from models.common.auto_compose import to_torch_auto_compose
from models.common.models.qwen25_7b.model import Qwen25_7BTTT


def greedy_argmax_from_logits(logits: ttnn.Tensor, *, mesh_device: ttnn.MeshDevice) -> int:
    """Return global argmax token id from (possibly sharded) logits ``[1,1,B,V_local]``."""
    lt = to_torch_auto_compose(logits, device=mesh_device).float()
    # Flatten batch row to last dim for B=1
    if lt.dim() == 4:
        lt = lt[0, 0, 0]
    elif lt.dim() == 3:
        lt = lt[0, 0]
    return int(torch.argmax(lt).item())


def greedy_decode_one_step(model: Qwen25_7BTTT, token_id: int, *, current_pos: int) -> int:
    """Decode one token at ``current_pos``; returns next token id (greedy)."""
    tid = torch.tensor([[[[token_id]]]], dtype=torch.int32)
    x = ttnn.from_torch(
        tid,
        device=model.mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(model.mesh_device),
    )
    h = model.decode_forward(x, current_pos=current_pos)
    logits = model.lm_logits(h)
    return greedy_argmax_from_logits(logits, mesh_device=model.mesh_device)
