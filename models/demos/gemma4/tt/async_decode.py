# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Gemma4-local async decode helpers (do not edit models/tt_transformers).

vLLM async scheduling can hand a one-step-stale host token while the device
trace buffer holds the authoritative sampled token. Nearest-bucket batch
changes and mesh-sharded decode buffers also make naive slot_remap gathers
raise IndexError. Keep the safe merge here so Gemma4 does not patch the
shared Generator.
"""

from __future__ import annotations

import torch


def merge_async_ahead_decode_tokens(
    host_toks: torch.Tensor,
    host_pos: torch.Tensor,
    dev_toks_full: torch.Tensor,
    dev_pos_full: torch.Tensor,
    slot_remap_local: torch.Tensor | None = None,
    prefilled_local: set[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    """Merge host decode inputs with the device's async-ahead token/pos buffers.

    Pure helper (no ttnn) so nearest-bucket / mesh-shard edge cases can be
    unit-tested without a device. Returns ``(tokens, positions, source)``
    where ``source`` is ``\"merged\"`` or ``\"host_fallback\"``.

    Device feedback may be **wider** than the host batch: non-PLI Gemma4
    always pads token/pos feedback to width 32 while nearest-bucket decode
    uses B∈{1,2,4,8,16,32}. Requiring ``dev_len == host_b`` forced
    ``host_fallback`` on every prefill→decode mode switch under concurrent
    async, restaging stale host tokens for continuing users (coherent but
    wrong GPQA finals). Accept ``dev_len >= host_b`` and take the front
    ``host_b`` slots; ``use_dev`` position matching still rejects unrelated
    rows after a true batch recomposition. Fall back only when the device
    buffer is narrower than the host batch, tok/pos lengths disagree, or
    ``slot_remap_local`` is OOB.
    """
    host_toks = host_toks.reshape(-1)
    host_pos = host_pos.reshape(-1).to(torch.int64)
    host_b = int(host_toks.shape[0])
    dev_toks_full = dev_toks_full.reshape(-1)
    dev_pos_full = dev_pos_full.reshape(-1).to(torch.int64)
    dev_len = int(dev_toks_full.shape[0])

    # Require each buffer independently wide enough. Do NOT require
    # tok_len == pos_len: non-PLI pads tokens to feedback width 32 while
    # sampling writeback can report a narrower logical shape on the token
    # buffer, and RoPE pos is always [1,32]. A strict equality check forced
    # host_fallback (stale host tokens) under async → "TheThe user user…"
    # doubling even at B=1.
    pos_len = int(dev_pos_full.shape[0])
    if dev_len < host_b or pos_len < host_b:
        return host_toks, host_pos, "host_fallback"

    dev_toks = dev_toks_full[:host_b]
    dev_pos = dev_pos_full[:host_b]
    if slot_remap_local is not None:
        remap_t = (
            (slot_remap_local if isinstance(slot_remap_local, torch.Tensor) else torch.tensor(slot_remap_local))
            .long()
            .reshape(-1)
        )
        if int(remap_t.numel()) != host_b or bool((remap_t < 0).any()) or bool((remap_t >= host_b).any()):
            return host_toks, host_pos, "host_fallback"
        dev_toks = dev_toks[remap_t]
        dev_pos = dev_pos[remap_t]

    use_dev = (dev_pos == host_pos) | (dev_pos == host_pos + 1)
    if prefilled_local:
        for slot in prefilled_local:
            if 0 <= slot < host_b:
                use_dev[slot] = False
    merged_toks = torch.where(use_dev, dev_toks.to(host_toks.dtype), host_toks)
    merged_pos = torch.where(use_dev, dev_pos, host_pos)
    return merged_toks, merged_pos, "merged"
