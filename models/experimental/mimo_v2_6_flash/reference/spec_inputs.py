# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Packed inputs of a speculative step (no ttnn): what the host uploads for a verify step and for an MTP draft, the
per-chip constants, and torch mirrors of the device programs (tt/spec_inputs.py) that write the traces' inputs from
the upload. The device programs run the mirrors' ops one for one; the host tests check the mirrors against the host
functions that built every input before (TtMiMoModel.prepare_decode_inputs_host, TtMiMoMTP's host inputs).

A speculative step used to copy 13-14 tiny replicated tensors to the device per verify and per draft (~0.2 ms of host
time each on a Galaxy row, most of a step's host time). It now copies one int32 tensor, row-major [rows * B]: B-wide
rows, one after another (B = max_batch).

Verify: rows (tokens, positions, chain). The chain row is the accept trace's, not the verify step's: what the chain
step needs to draft behind the pass on device (reference/spec_chain.py), zeros when no draft chains behind it. Inside
the verify trace, before the decode step:

* tokens: row 0, as uint32 (the bits: token ids are non-negative);
* cur_pos: row 1;
* the global layers' owned positions: min(pos, own_cap), own_cap INT32_MAX on the slots the chip owns and -1 on the
  others. Positions are >= -1, so the minimum keeps an owned slot's position and makes the others -1
  (kv.owned_positions);
* serial position j: min(pos, slot_cap_j), slot_cap_j INT32_MAX at slot j and -1 elsewhere (serial_positions);
* the RoPE table index (with device RoPE tables): max(pos, 0), as uint32;
* the MoE token mask: min(pos, 0) + 1 (1 active, 0 idle), tilized, cast to bf16 and transposed into the
  [1, 1, B, 1] column. It is 0 / 1 before tilizing, so the tile's padding stays 0, as the host tensor's;
* the owner mask: the token mask times the chip's owned-slot column (kv.owner_mask).

MTP draft: rows (tokens, fill, step2, step3 cache positions, then the fill, step2, step3 RoPE indices). The plan
(reference.spec_decode.plan_mtp) and the ring cache positions (mtp_cache_position) are a few integers of host
arithmetic; what cost host time was building and copying a tensor for each. Inside the draft trace: tokens as
uint32, the cache positions as they are, serial fill positions min(fill, slot_cap_j), and the RoPE indices (the
true positions clamped at 0, as the host cos / sin were) as uint32 for the device RoPE tables. A draft chained behind
a verify (reference/spec_chain.py) finds this upload already written on device, by the accept trace's chain step.

The RoPE cos / sin themselves stay host inputs when the device tables do not exist (contexts past the tables' limit):
a verify then also copies its four, a draft its six.

MTP draft with every layer (MIMO_MTP_LAYERS=3, reference/spec_decode.py ``plan_mtp_layers``): which rows a draft runs
depends on the rows the last draft ran (the stash), which only the device keeps. So the upload describes the commit
instead of the passes (``pack_mtp_layers``, row 0 of the same buffer, the other rows zero): whether to draft, the
committed count m, the position A of the row taking committed token 0, the source's slots and the committed tokens.
The draft trace's plan step (reference/spec_chain.py ``plan_step``, kernels/spec_chain.cpp) derives the passes'
inputs from it and the device stash. pack_mtp's idle upload has a zero row 0: it is an idle description too.
"""

import torch

from .spec_decode import GROUP_SLOTS, LAYOUT, SPEC_SLOTS, mtp_cache_position

TILE = 32
INT32_MAX = 2**31 - 1
VERIFY_ROWS = 3  # tokens, positions, the chain row (reference/spec_chain.py)
MTP_ROWS = 7  # tokens; fill, step2, step3 cache positions; fill, step2, step3 RoPE indices
# pack_mtp_layers' row g, draft group g's commit: drafting (0 / 1), m, A, the source's slots [group slots] (-1 past m),
# committed tokens [group slots] (0 past m).
DESC_DRAFTING, DESC_COUNT, DESC_A, DESC_SLOTS = 0, 1, 2, 3


def desc_tokens(group_slots=GROUP_SLOTS):
    """A description's first committed-token word in draft groups of ``group_slots`` slots."""
    return DESC_SLOTS + group_slots


DESC_TOKENS = desc_tokens()
DESC_WORDS = DESC_TOKENS + GROUP_SLOTS


# ---------------------------------------------------------------------- constants (persistent device tensors)
def owned_slot_caps(max_batch, tp, rep):
    """int32 [tp, B]: INT32_MAX on the slots chip c owns (slot % rep == c % rep, kv.py), -1 on the others."""
    slots, chips = torch.arange(max_batch), torch.arange(tp)
    own = slots.reshape(1, -1) % rep == chips.reshape(-1, 1) % rep
    return torch.where(own, INT32_MAX, -1).to(torch.int32)


def owned_slot_columns(max_batch, tp, rep):
    """bf16 [tp, 1, 1, B, 1]: 1 on the slots chip c owns, as kv.owner_mask's layout."""
    return (owned_slot_caps(max_batch, tp, rep) >= 0).to(torch.bfloat16).reshape(tp, 1, 1, max_batch, 1)


def slot_caps(max_batch, slots=SPEC_SLOTS):
    """int32 [slots, B]: row j INT32_MAX at slot j, -1 elsewhere."""
    out = torch.full((slots, max_batch), -1, dtype=torch.int32)
    out[torch.arange(slots), torch.arange(slots)] = INT32_MAX
    return out


# ---------------------------------------------------------------------- uploads
def pack_verify(tokens, positions, chain=None):
    """tokens, positions [B] (-1 an idle slot), ``chain`` the chain row's B values (reference.spec_chain.chain_row;
    None: zeros, no draft chains behind the pass) -> int32 [VERIFY_ROWS * B]."""
    assert tokens.numel() == positions.numel()
    assert int(positions.min()) >= -1, "positions are >= -1 (-1 an idle slot)"
    chain = torch.zeros(positions.numel(), dtype=torch.int32) if chain is None else torch.as_tensor(chain)
    assert chain.numel() == positions.numel()
    return torch.cat(
        (tokens.reshape(-1).to(torch.int32), positions.reshape(-1).to(torch.int32), chain.reshape(-1).to(torch.int32))
    )


def pack_mtp(max_batch, tokens=None, plan=None):
    """int32 [MTP_ROWS * B] of one draft: ``tokens`` per slot (the tokens committed there), ``plan`` the
    reference.spec_decode.MtpInputs; None for an idle run (every position -1: no K / V written)."""
    out = torch.zeros(MTP_ROWS, max_batch, dtype=torch.int32)
    out[1:4] = -1
    if tokens is not None:
        out[0, : len(tokens)] = torch.tensor(tokens, dtype=torch.int32)
    if plan is not None:
        for row, values in enumerate((plan.fill, plan.step2, plan.step3)):
            out[1 + row, : len(values)] = torch.tensor([mtp_cache_position(p) for p in values], dtype=torch.int32)
            out[4 + row, : len(values)] = torch.tensor([max(p, 0) for p in values], dtype=torch.int32)
    return out.reshape(-1)


def mtp_layers_row(
    max_batch, source=None, committed_tokens=None, committed_positions=None, accepted=None, layout=LAYOUT
):
    """pack_mtp_layers' row 0, int32 values [max_batch]: the draft after the commit of ``accepted`` tokens
    ``committed_tokens`` at ``committed_positions`` (consecutive) of the lane's DraftSource ``source`` in ``layout``'s
    draft groups; None: idle."""
    gs = layout.group_slots
    assert max_batch >= desc_tokens(gs) + gs
    row = [0] * max_batch
    if source is None:
        return row
    m = int(accepted)
    assert 1 <= m <= len(source.slots) <= gs, (m, source)
    row[DESC_DRAFTING], row[DESC_COUNT], row[DESC_A] = 1, m, int(committed_positions[0]) - 1
    for c in range(gs):
        row[DESC_SLOTS + c] = source.slots[c] if c < m else -1
        row[desc_tokens(gs) + c] = int(committed_tokens[c]) if c < m else 0
    return row


def pack_mtp_layers(max_batch, source=None, committed_tokens=None, committed_positions=None, accepted=None):
    """int32 [MTP_ROWS * B] of a multi-layer draft for one commit: row ``source.group`` its ``mtp_layers_row``, the
    other rows zero (idle groups)."""
    commits = [] if source is None else [(source, committed_tokens, committed_positions, accepted)]
    return pack_mtp_commits(max_batch, commits)


def pack_mtp_commits(max_batch, commits, layout=LAYOUT):
    """int32 [MTP_ROWS * B] of a multi-layer draft for ``commits`` [(source, committed tokens, committed positions,
    accepted)], one per draft group of ``layout``: row g the description of group g's commit, zero for a group without
    one."""
    out = torch.zeros(MTP_ROWS, max_batch, dtype=torch.int32)
    for source, tokens, positions, m in commits:
        assert 0 <= source.group < layout.groups and not bool(out[source.group].any()), source
        out[source.group] = torch.tensor(mtp_layers_row(max_batch, source, tokens, positions, m, layout))
    return out.reshape(-1)


# ---------------------------------------------------------------------- torch mirrors of the device programs
def as_uint32(t):
    """ttnn.bitcast int32 -> uint32: the same bits (as int64 values here)."""
    return t.to(torch.int64) & 0xFFFFFFFF


def tile_padded(t):
    """A row-major tensor tilized (ttnn.to_layout to TILE): its last two dims zero-padded to whole tiles."""
    h, w = t.shape[-2], t.shape[-1]
    return torch.nn.functional.pad(t, (0, -w % TILE, 0, -h % TILE))


def derive_verify(packed, own_cap, own_col, caps, rope):
    """One chip's verify inputs from the upload ``packed`` int32 [VERIFY_ROWS * B] (the chain row is not read);
    ``own_cap`` int32 [B] and ``own_col`` bf16 [1, 1, B, 1] that chip's rows of owned_slot_caps / owned_slot_columns,
    ``caps`` slot_caps. Tile-layout outputs (``mask``, ``owner_mask``) come back padded to whole tiles. Returns
    {name: tensor}."""
    b = packed.numel() // VERIFY_ROWS
    pos = packed[b : 2 * b]
    out = {
        "tokens": as_uint32(packed[:b]).reshape(1, b),
        "cur_pos": pos,
        "pos_owned": torch.minimum(pos, own_cap),
        "serial": [torch.minimum(pos, cap) for cap in caps],
    }
    if rope:
        out["rope_pos"] = as_uint32(torch.maximum(pos, torch.zeros_like(pos))).reshape(1, b)
    active = torch.minimum(pos, torch.zeros_like(pos)) + torch.ones_like(pos)
    col = tile_padded(active.reshape(1, 1, 1, b)).to(torch.bfloat16).transpose(-2, -1)
    out["mask"] = col
    out["owner_mask"] = col * tile_padded(own_col)
    return out


def derive_mtp(packed, caps, rope):
    """The draft trace's position and token inputs from the upload ``packed`` int32 [MTP_ROWS * B]."""
    b = packed.numel() // MTP_ROWS
    rows = packed.reshape(MTP_ROWS, b)
    out = {
        "tokens": as_uint32(rows[0]).reshape(1, b),
        "fill": rows[1],
        "serial": [torch.minimum(rows[1], cap) for cap in caps],
        "step2": rows[2],
        "step3": rows[3],
    }
    if rope:
        out["rope_idx"] = [as_uint32(rows[4 + k]).reshape(1, b) for k in range(3)]
    return out
