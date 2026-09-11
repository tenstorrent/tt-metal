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
mix, and behaves like any other tile on its own first decode touch.

BUFFER REUSE: ``dflash_verify``'s position/candidate-id/page-table tensors are allocated
ONCE per generation session (``make_verify_buffers``) and refreshed in place every call
via ``ttnn.copy_host_to_device_tensor``, mirroring spec_decode.py's own
``_host_pos``/``_host_tokens`` + ``copy_host_to_device_tensor`` pattern -- not a fresh
``ttnn.from_torch(..., device=mesh_device)`` allocation each iteration. A fresh allocation
per call is exactly what a Metal trace cannot replay (a trace binds to fixed buffer
addresses captured once); reusing the same buffer is a prerequisite for eventually tracing
this loop, even though this call itself still runs eagerly."""

from __future__ import annotations

from dataclasses import dataclass

import torch

import ttnn
from models.demos.gemma4.tt.dflash.lm_head import argmax_last_dim


def _mesh_mapper(mesh_device):
    return ttnn.ReplicateTensorToMesh(mesh_device) if hasattr(mesh_device, "shape") else None


@dataclass
class DFlashVerifyBuffers:
    """Persistent device tensors for dflash_verify, reused across every verify call in a
    generation session. ``page_table`` is the one field that's genuinely CONSTANT for the
    whole session (same physical KV pages throughout a single-user run) -- built once and
    never refreshed again."""

    pos_uint32: ttnn.Tensor  # [1,32] uint32
    pos_int32: ttnn.Tensor  # [block_size] int32
    page_table: ttnn.Tensor  # [block_size, num_blocks] int32 -- constant
    candidate_ids: ttnn.Tensor  # [1,block_size] uint32
    block_size: int


def make_verify_buffers(mesh_device, page_table_torch: torch.Tensor, block_size: int) -> DFlashVerifyBuffers:
    mapper = _mesh_mapper(mesh_device)
    pos_uint32 = ttnn.from_torch(
        torch.zeros((1, 32), dtype=torch.int64),
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint32,
        mesh_mapper=mapper,
    )
    pos_int32 = ttnn.from_torch(
        torch.zeros((block_size,), dtype=torch.int32),
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.int32,
        mesh_mapper=mapper,
    )
    user_row = page_table_torch[0:1] if page_table_torch.dim() > 1 else page_table_torch.unsqueeze(0)
    pt = user_row.repeat(block_size, 1).to(torch.int32)
    page_table = ttnn.from_torch(
        pt, device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.int32, mesh_mapper=mapper
    )
    candidate_ids = ttnn.from_torch(
        torch.zeros((1, block_size), dtype=torch.int64),
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint32,
        mesh_mapper=mapper,
    )
    return DFlashVerifyBuffers(pos_uint32, pos_int32, page_table, candidate_ids, block_size)


def refresh_verify_positions(mesh_device, buffers: DFlashVerifyBuffers, start_pos: int) -> None:
    """Refresh buffers.pos_uint32/pos_int32 in place for a new start_pos -- the one part
    of verify's inputs that's genuinely dynamic every call (position values shift every
    iteration; there's no way around rebuilding this small host tensor each time, see
    verify.py's module docstring). Shared by dflash_verify (below) and generate.py's
    fused-iteration path, which builds its own candidate-ids tensor on device instead of
    going through buffers.candidate_ids."""
    mapper = _mesh_mapper(mesh_device)
    block_size = buffers.block_size
    positions = list(range(start_pos, start_pos + block_size))
    pu = torch.zeros((1, 32), dtype=torch.int64)
    pu[0, :block_size] = torch.tensor(positions, dtype=torch.int64)
    ttnn.copy_host_to_device_tensor(
        ttnn.from_torch(pu, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32, mesh_mapper=mapper), buffers.pos_uint32
    )
    ttnn.copy_host_to_device_tensor(
        ttnn.from_torch(
            torch.tensor(positions, dtype=torch.int32),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.int32,
            mesh_mapper=mapper,
        ),
        buffers.pos_int32,
    )


def run_verify_forward(model, mesh_device, tt_kv_cache, buffers: DFlashVerifyBuffers, x_tt: ttnn.Tensor):
    """Raw ttnn_verify_forward call against an ALREADY-ON-DEVICE candidate-ids tensor
    (``x_tt``, e.g. built via ttnn.concat rather than uploaded from a host list) --
    ``refresh_verify_positions`` must already have been called for this start_pos.
    Returns (logits, hidden) exactly as ttnn_verify_forward does; the caller owns both
    (deallocate hidden, argmax logits) -- this is the shared primitive dflash_verify and
    generate.py's fused-iteration path both build on."""
    return model.ttnn_verify_forward(
        x=x_tt,
        current_pos=buffers.pos_uint32,
        current_pos_cache=buffers.pos_int32,
        page_table=buffers.page_table,
        kv_cache=tt_kv_cache,
    )


def dflash_verify(
    model, mesh_device, tt_kv_cache, buffers: DFlashVerifyBuffers, candidate_ids: list[int], start_pos: int
):
    """One verify "iteration": candidate_ids = [anchor_token, draft_1, ..., draft_{K-1}]
    (K == buffers.block_size total), positions [start_pos .. start_pos+K-1], verified in a
    single ttnn_verify_forward batch call. Returns (posterior, None) -- posterior[i] is the
    target's own greedy prediction AFTER consuming candidate_ids[i]. Argmax runs ON
    DEVICE (argmax_last_dim); only the small [1,block_size] result is read to host, not
    the full-vocab logits -- the second return slot is kept for API compatibility with
    older callers that used the raw logits for debugging, but is no longer computed.

    The very FIRST dflash_verify call after prefill requires start_pos (== the prefill
    KV cache's committed length) to be a multiple of 32 -- see module docstring for the
    known kernel bug this sidesteps. Later calls, once real decode/verify writes have
    already touched every relevant tile, are not subject to this constraint."""
    block_size = buffers.block_size
    assert len(candidate_ids) == block_size, f"expected {block_size} candidate ids, got {len(candidate_ids)}"
    mapper = _mesh_mapper(mesh_device)

    refresh_verify_positions(mesh_device, buffers, start_pos)
    ttnn.copy_host_to_device_tensor(
        ttnn.from_torch(
            torch.tensor(candidate_ids, dtype=torch.int64).reshape(1, block_size),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.uint32,
            mesh_mapper=mapper,
        ),
        buffers.candidate_ids,
    )

    logits, hidden = run_verify_forward(model, mesh_device, tt_kv_cache, buffers, buffers.candidate_ids)
    ttnn.deallocate(hidden)

    vocab = logits.shape[-1]
    if logits.shape[2] != block_size:
        # Defensive: ttnn_verify_forward's docstring promises [1,1,K,vocab], but slice
        # down on device rather than assume, matching the reshape+slice the previous
        # host-side implementation did.
        sliced = ttnn.slice(logits, [0, 0, 0, 0], [1, 1, block_size, vocab])
        ttnn.deallocate(logits)
        logits = sliced
    posterior_tt = argmax_last_dim(logits, block_size)
    ttnn.deallocate(logits)

    is_mesh = hasattr(mesh_device, "shape")
    posterior_torch = (
        ttnn.to_torch(ttnn.get_device_tensors(posterior_tt)[0]) if is_mesh else ttnn.to_torch(posterior_tt)
    )
    ttnn.deallocate(posterior_tt)
    posterior = posterior_torch.reshape(1, -1)[:, :block_size].long()
    return posterior, None


def greedy_accept_from_posterior(candidate_ids: list[int], posterior: torch.Tensor):
    """Exactly the reference's accept logic (dflash_generate): longest matching prefix
    between drafts (candidate_ids[1:]) and the target's own prediction at each position,
    plus the target's bonus/correction token at the point of divergence.

    Genuinely host-side control flow, not a torch/ttnn-conversion candidate: deciding HOW
    MANY tokens to keep -- and therefore what the next iteration's inputs even look like --
    has to be visible to the host loop driving generation. This is the same point at which
    spec_decode.py's own traced implementation reads its trace's output back to host
    (``_ids_to_host``) after ``execute_trace`` -- a trace's replay ends here every
    iteration, it isn't blocked by it."""
    drafts = torch.tensor(candidate_ids[1:]).unsqueeze(0)
    acceptance_length = (drafts == posterior[:, :-1]).cumprod(dim=1).sum(dim=1)[0].item()
    bonus = posterior[0, acceptance_length].item()
    committed = candidate_ids[: acceptance_length + 1] + [bonus]
    return acceptance_length, bonus, committed
