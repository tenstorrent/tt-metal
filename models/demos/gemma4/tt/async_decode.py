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

    if slot_remap_local is not None:
        # ``slot_remap_local`` entries are *device rows* (KV slots), so they are
        # bounded by the device buffer width, not by the host batch. Bounding
        # them by ``host_b`` rejected every legitimate mapping whenever the
        # device buffer is wider than the host batch (e.g. slot 1 with host_b=1),
        # silently forcing host_fallback.
        remap_t = (
            (slot_remap_local if isinstance(slot_remap_local, torch.Tensor) else torch.tensor(slot_remap_local))
            .long()
            .reshape(-1)
        )
        dev_rows = min(dev_len, pos_len)
        if int(remap_t.numel()) != host_b or bool((remap_t < 0).any()) or bool((remap_t >= dev_rows).any()):
            return host_toks, host_pos, "host_fallback"
        dev_toks = dev_toks_full[remap_t]
        dev_pos = dev_pos_full[remap_t]
    else:
        # No slot mapping: assume host row j is device row j and take the front
        # ``host_b`` rows. This is what keeps continuing users on device tokens
        # across nearest-bucket changes (requiring dev_len == host_b forced
        # host_fallback on every mode switch and restaged stale host tokens ->
        # "TheThe user user" doubling), so it is deliberately preserved.
        #
        # CAVEAT: the assumption is false when a request occupies a device slot
        # other than its host index -- seen on WH-T3K at max_num_seqs=1, where
        # every request is placed in slot 1 ("Prefilling User 1") while host_b=1,
        # so the front slice describes slot 0. The ``use_dev`` position guard
        # below is the only thing separating the rows, and it compares values,
        # not identities. The durable fix is for the caller to pass the real
        # per-row device slots (the plugin tracks them in ``_req_state_slot`` but
        # only forwards an identity remap on the non-lane path); until then the
        # async-decode capability gate in ``Gemma4Generator.decode_forward``
        # keeps this path off wherever async decode is disabled.
        dev_toks = dev_toks_full[:host_b]
        dev_pos = dev_pos_full[:host_b]

    use_dev = (dev_pos == host_pos) | (dev_pos == host_pos + 1)
    if prefilled_local:
        for slot in prefilled_local:
            if 0 <= slot < host_b:
                use_dev[slot] = False
    merged_toks = torch.where(use_dev, dev_toks.to(host_toks.dtype), host_toks)
    merged_pos = torch.where(use_dev, dev_pos, host_pos)
    return merged_toks, merged_pos, "merged"
