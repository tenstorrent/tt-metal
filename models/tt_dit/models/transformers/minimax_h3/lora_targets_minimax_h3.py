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

import re
from dataclasses import replace
from typing import TYPE_CHECKING

import torch

from ....lora.apply import FusionGroup
from ....lora.keys import AdapterEntry
from ....lora.route import named_modules
from .attention_minimax_h3 import MiniMaxH3Attention

if TYPE_CHECKING:
    from collections.abc import Sequence

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


# ---------------------------------------------------------------------------- native keyspace

# MiniMax's own release, and every adapter trained against it, names the transformer differently
# from the diffusers conversion this port loads. Translating the adapter onto the diffusers
# keyspace up front keeps the rename out of the loader, the AdaLN fold and the target map, all of
# which already speak diffusers.
_ROOT_RENAMES = (
    ("token_refiner.blocks.", "token_refiner.refiner_blocks."),
    ("final_layer.adaln_proj.linear", "norm_out.linear"),
    ("time_embedder.proj_in", "time_embedder.linear_1"),
    ("time_embedder.proj_out", "time_embedder.linear_2"),
)
_LEAF_RENAMES = (
    ("attn.out_proj", "attn.to_out.0"),
    ("mlp.fc1", "ff.net.0.proj"),
    ("mlp.fc2", "ff.net.2"),
)
_NATIVE_BLOCK_PREFIX = re.compile(r"^blocks\.(\d+)\.")

# Leaves whose native spelling equals the diffusers one (`norm1`, `norm2`, `attn.norm_q`, ...) pass
# through untouched, so an unrecognised leaf is not an error here -- it surfaces as an unresolved
# routing key, which names the destination the model lacks rather than the rename that was missed.


def is_native_keyspace(entries: Sequence[AdapterEntry]) -> bool:
    """Whether ``entries`` are addressed against MiniMax's release rather than the diffusers one.

    Both spellings are block-scoped, so the discriminator has to be a prefix only one of them uses.
    """
    return any(entry.path.startswith(("blocks.", "final_layer.")) for entry in entries)


def translate_native_entries(entries: Sequence[AdapterEntry]) -> list[AdapterEntry]:
    """Rewrite a native-keyspace adapter onto the diffusers keyspace. A no-op for a diffusers one.

    Two of the rewrites move tensor rows rather than just a name, and both are silent when wrong:
    a mis-split ``qkv_proj`` delivers each projection's delta to the wrong one, and an unswapped
    ``fc1`` sends the gate's delta to the value half. Neither shows up as a shape error and neither
    is visible to output parity against a reference built the same way.
    """
    if not is_native_keyspace(entries):
        return list(entries)
    translated: list[AdapterEntry] = []
    for entry in entries:
        translated.extend(_translate(entry))
    return translated


def _rename(path: str) -> str:
    for native, diffusers in _ROOT_RENAMES:
        if path.startswith(native):
            path = f"{diffusers}{path[len(native):]}"
            break
    else:
        path = _NATIVE_BLOCK_PREFIX.sub(r"transformer_blocks.\1.", path)
    for native, diffusers in _LEAF_RENAMES:
        if path.endswith(native):
            return f"{path[: -len(native)]}{diffusers}"
    return path


def _map_out_rows(entry: AdapterEntry, transform) -> AdapterEntry:
    """Apply an output-dimension row transform to whichever tensor carries the delta's rows.

    ``A`` lives on the input side and is never touched; ``B`` and a dense delta both index the
    output dimension first, which is what every rename below permutes.
    """
    if entry.kind == "lora":
        return replace(entry, B=transform(entry.B))
    return replace(entry, delta=transform(entry.delta))


def _translate(entry: AdapterEntry) -> tuple[AdapterEntry, ...]:
    path = _rename(entry.path)

    if path.endswith(".attn.qkv_proj"):
        if entry.kind != "lora":
            msg = f"{entry.path}: a fused qkv adapter must be low-rank to split, got {entry.kind}"
            raise ValueError(msg)
        rows = entry.B.shape[0]
        if rows % 3:
            msg = f"{entry.path}: fused qkv has {rows} rows, not divisible by three"
            raise ValueError(msg)
        # Contiguous q|k|v, matching the released implementation's in-memory layout (its loader
        # de-interleaves the per-head grouping before the adapter ever sees it). The three land as
        # separate members of this attention's fusion group, so the head interleave and rotary
        # channel permutation stay in the model's own `_prepare_torch_state`.
        inner = rows // 3
        owner = path[: -len("qkv_proj")]
        return tuple(
            replace(entry, path=f"{owner}{member}", B=entry.B[i * inner : (i + 1) * inner].contiguous())
            for i, member in enumerate(QKV_MEMBERS)
        )

    if path.endswith(".ff.net.0.proj"):
        # The release stores the gated FFN's fused projection as [gate; value]; diffusers' SwiGLU
        # reads [value; gate]. Same swap the checkpoint conversion applies to the base weight.
        def swap(tensor):
            gate, value = tensor.chunk(2, dim=0)
            return torch.cat((value, gate), dim=0).contiguous()

        return (_map_out_rows(replace(entry, path=path), swap),)

    return (replace(entry, path=path),)
