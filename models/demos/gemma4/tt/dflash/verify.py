# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Verify the DFlash drafter's candidates against the real target model, using Gemma4's
existing (already-built, EAGLE/MTP-drafter-decoupled) batch-dimension verify forward --
see models/demos/gemma4/docs/dflash_design.md section 3: "Gemma4's existing spec_decode.py
decouples verify/accept/commit from the drafter via a draft_fn seam -- reusable."

model.ttnn_verify_forward puts the K candidate tokens in the batch dimension at
consecutive positions with the real user's page-table row replicated K times -- this
reuses the ordinary batched-decode path, continuing from whatever KV cache the target's
own prefill already built up.

KNOWN GEMMA4 KERNEL BUG (real T3K, google/gemma-4-31B-it, root-caused via hardware A/B
testing plus static tracing -- see dflash_design.md): the FIRST decode-style write into
the paged KV cache after prefill (i.e. any call, verify or ordinary decode, whose query
position equals the prefill's committed length ``ctx_len``) returns a wrong, context-blind
answer WHENEVER ``ctx_len`` is not an exact multiple of 32. Root cause: prefill writes KV
in 32-row tiles; when the real prompt doesn't fill the last tile exactly, a pad-removal
branch in ``attention/prefill.py`` that is supposed to strip the leftover garbage-token
K/V from the tile's unused rows fails to fire (its guard condition is false exactly when
the valid content ends at the tile's own boundary). Decode's first ``paged_update_cache``
write into that same tile -- a different, single-row kernel from prefill's bulk tile
write -- reads that garbage on its first touch and produces a wrong result; the SAME tile
is fine on every subsequent touch, since by then decode itself has already written it once.
Confirmed on real hardware: identical prompt content at ctx_len=18 (not tile-aligned)
fails at position 18 and succeeds at position 19+ on the first attempt every time;
the SAME mechanism at ctx_len=32 (exactly tile-aligned) succeeds at position 32 on the
first attempt, no workaround needed. (An earlier workaround here routed the anchor
through ``ttnn_decode_forward`` instead of ``ttnn_verify_forward`` -- that does NOT help,
since ``ttnn_decode_forward`` hits the identical kernel bug at the identical position.)

This is a real defect in Gemma4's paged-attention prefill/decode write path, not specific
to DFlash or to the verify mechanism -- it affects any Gemma4 decode call immediately
following a non-tile-aligned prefill. The correct long-term fix is in
``attention/prefill.py``'s pad-removal condition. Until that lands, callers MUST ensure the
INITIAL prefill's committed length (``ctx_len``) is a multiple of 32 -- e.g. by padding the
real prompt with genuine, attended-to context up to the next 32-token boundary before
prefill (NOT with masked/ignored pad tokens -- padding that isn't actually attended to
leaves the same unused-tile-row garbage that causes the bug in the first place). This is a
ONE-TIME constraint on the FIRST decode-style touch to the cache: once any tile has been
written by a real decode/verify call, later ``dflash_verify`` calls landing anywhere in
that same tile -- even at a non-tile-aligned ``start_pos`` -- are fine, since every tile
beyond the (now tile-aligned) real content is uniform prefill padding, not a straddling
mix, and behaves like any other tile on its own first decode touch."""

from __future__ import annotations

import torch

import ttnn


def build_verify_inputs(mesh_device, page_table_torch: torch.Tensor, start_pos: int, block_size: int):
    """current_pos ([1,32] padded -- required for the tile-aligned RoPE embedding gather,
    cropped back to real batch internally) and current_pos_cache/page_table (both exactly
    block_size long, UNPADDED) -- mirrors spec_decode.py's _host_pos/_pos_tensors/_page_table
    exactly. x (candidate tokens) must ALSO stay unpadded [1, block_size]: the model derives
    its internal batch from x.shape[1], not from current_pos_cache -- padding x to 32 was
    a real, separate bug already found and fixed (a per-batch KV-write loop ran 32
    iterations against a 16-row page_table/current_pos_cache, indexing one row past the
    end at b=16)."""
    positions = list(range(start_pos, start_pos + block_size))

    pu = torch.zeros((1, 32), dtype=torch.int64)
    pu[0, :block_size] = torch.tensor(positions, dtype=torch.int64)
    pos_uint32 = ttnn.from_torch(
        pu,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint32,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    pos_int32 = ttnn.from_torch(
        torch.tensor(positions, dtype=torch.int32),
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.int32,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    user_row = page_table_torch[0:1] if page_table_torch.dim() > 1 else page_table_torch.unsqueeze(0)
    pt = user_row.repeat(block_size, 1).to(torch.int32)
    page_table_tt = ttnn.from_torch(
        pt,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.int32,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    return pos_uint32, pos_int32, page_table_tt


def dflash_verify(
    model, mesh_device, tt_kv_cache, page_table_torch: torch.Tensor, candidate_ids: list[int], start_pos: int
):
    """One verify "iteration": candidate_ids = [anchor_token, draft_1, ..., draft_{K-1}]
    (K total), positions [start_pos .. start_pos+K-1], verified in a single
    ttnn_verify_forward batch call. Returns (posterior, logits_torch) -- posterior[i] is
    the target's own greedy prediction AFTER consuming candidate_ids[i].

    The very FIRST dflash_verify call after prefill requires start_pos (== the prefill
    KV cache's committed length) to be a multiple of 32 -- see module docstring for the
    known kernel bug this sidesteps. Later calls, once real decode/verify writes have
    already touched every relevant tile, are not subject to this constraint."""
    block_size = len(candidate_ids)
    pos_uint32, pos_int32, page_table_tt = build_verify_inputs(mesh_device, page_table_torch, start_pos, block_size)

    x = torch.tensor(candidate_ids, dtype=torch.int64).reshape(1, block_size)
    x_tt = ttnn.from_torch(
        x,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint32,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    logits, hidden = model.ttnn_verify_forward(
        x=x_tt, current_pos=pos_uint32, current_pos_cache=pos_int32, page_table=page_table_tt, kv_cache=tt_kv_cache
    )
    ttnn.deallocate(hidden)

    is_mesh = hasattr(mesh_device, "shape")
    logits_torch = ttnn.to_torch(ttnn.get_device_tensors(logits)[0]) if is_mesh else ttnn.to_torch(logits)
    ttnn.deallocate(logits)
    logits_torch = logits_torch.float().reshape(1, -1, logits_torch.shape[-1])[:, :block_size, :]
    posterior = torch.argmax(logits_torch, dim=-1)
    return posterior, logits_torch


def greedy_accept_from_posterior(candidate_ids: list[int], posterior: torch.Tensor):
    """Exactly the reference's accept logic (dflash_generate): longest matching prefix
    between drafts (candidate_ids[1:]) and the target's own prediction at each position,
    plus the target's bonus/correction token at the point of divergence."""
    drafts = torch.tensor(candidate_ids[1:]).unsqueeze(0)
    acceptance_length = (drafts == posterior[:, :-1]).cumprod(dim=1).sum(dim=1)[0].item()
    bonus = posterior[0, acceptance_length].item()
    committed = candidate_ids[: acceptance_length + 1] + [bonus]
    return acceptance_length, bonus, committed
