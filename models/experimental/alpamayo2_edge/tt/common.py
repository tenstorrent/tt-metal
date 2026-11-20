# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Small helpers shared by the TT modules."""

from __future__ import annotations

import torch

import ttnn
from models.tt_dit.encoders.transformer import TransformerContext


def round_up(x: int, multiple: int) -> int:
    return -(-x // multiple) * multiple


def single_device_ctx(mesh_device: ttnn.MeshDevice) -> TransformerContext:
    """A tt_dit transformer context with no tensor, sequence or FSDP parallelism."""
    return TransformerContext(device=mesh_device, tp_axis=None, ccl_manager=None, sp_axis=None, fsdp_axis=None)


def drop_unbuilt_layers(state: dict[str, torch.Tensor], prefix: str, built: int) -> None:
    """Removes the entries of `prefix`-indexed layers at index `built` and deeper, in place.

    A depth override builds fewer layers than the checkpoint holds; the deeper layers are not loaded.
    """
    for key in list(state):
        if key.startswith(prefix) and int(key[len(prefix) :].split(".", 1)[0]) >= built:
            del state[key]
