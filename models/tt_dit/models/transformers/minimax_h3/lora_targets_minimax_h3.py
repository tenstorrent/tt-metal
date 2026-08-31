# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""What a MiniMax-H3 adapter needs the generic loader to know.

Almost nothing, because the adapter is published against the diffusers checkpoint and this port
already renames its way onto that keyspace: ``attn.to_out.0`` -> ``to_out`` and ``ff.net.0.proj``
-> ``ff.ff1`` happen in the model's own ``_prepare_torch_state``, so routing an adapter tensor
lands it in the right place without a translation table. Two facts are left over.

**q/k/v fuse.** ``MiniMaxH3Attention`` builds one ``to_qkv``, so the three per-projection adapters
must be stacked into one. Order is q, k, v -- the order ``_interleave_heads`` concatenates them in.
Getting this wrong is not a crash; it silently swaps which heads receive which delta.

**AdaLN is not on the device.** With ``precomputed_adaln=True`` the pipeline projects every block's
``adaln_proj`` on host into a table and never builds the module, and the same is true of
``time_embedder`` and ``norm_out.linear`` (``transformer_minimax_h3.py:322``). Their adapters --
50 of the 362 low-rank pairs, plus six dense deltas -- therefore have no parameter to bind to and
are handed back for the AdaLN precompute to fold in. ``norm_out.norm`` is a real device parameter
and is deliberately *not* on this list.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from ....lora.apply import FusionGroup
from ....lora.route import named_modules
from .attention_minimax_h3 import MiniMaxH3Attention

if TYPE_CHECKING:
    from ....layers.module import Module

QKV_MEMBERS = ("to_q", "to_k", "to_v")

# Adapter-path prefixes the precomputed-AdaLN build resolves on host instead of on device.
MINIMAX_H3_HOST_PATHS = (
    "time_embedder.",
    "norm_out.linear",
    "adaln_proj.",
)


def minimax_h3_fusion_groups(model: Module) -> list[FusionGroup]:
    """One q/k/v group per attention in ``model``, discovered rather than enumerated.

    Derived from the built model so the transformer-block count, the token refiner's two blocks and
    any future depth change all stay covered without a second place to update.
    """
    return [
        FusionGroup(owner=path, members=QKV_MEMBERS)
        for path, module in named_modules(model)
        if isinstance(module, MiniMaxH3Attention)
    ]


def minimax_h3_host_paths(model: Module) -> tuple[str, ...]:
    """Adapter-path prefixes to defer to the host AdaLN fold, empty if the model holds them.

    Reading the flag off the model rather than assuming it keeps a projected-AdaLN build -- which
    the transformer parity test still exercises -- from silently dropping 40% of an adapter.
    """
    return MINIMAX_H3_HOST_PATHS if getattr(model, "precomputed_adaln", False) else ()


def is_host_path(path: str, host_paths: tuple[str, ...] = MINIMAX_H3_HOST_PATHS) -> bool:
    """Whether an adapter path targets a host-resolved parameter.

    Substring rather than prefix matching: the block-scoped ones arrive as
    ``transformer_blocks.7.adaln_proj.linear``, while ``time_embedder`` sits at the root.
    """
    return any(marker in path for marker in host_paths)
