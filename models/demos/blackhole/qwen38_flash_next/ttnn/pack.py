# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Multi-user packed prefill slab (P3d-2): the host side, no device import beyond the dtype names.

A pack is a bucket ``(S, G)`` with ``S`` in :data:`PACK_BUCKETS` and ``G = 2048 // S`` on the existing 2048-row slab state
(its buffers are shared by every bucket; one trace per bucket).  User ``u`` owns the rows ``[u*S, (u+1)*S)`` (its
"segment"): its ``P_u <= S`` prompt tokens first, the fixed pad token after them; an empty slot is an all-pad segment.  The
pack runs at position 0 of the chain's dense KV and compressed caches (the "virtual layout": user ``u`` at row base
``u*S``, compressed row base ``u*S/4``), so the KV and compressed-index writes of the slab need no change.

What differs from the single-user slab, per layer family:

* GDN: the chunk prims run with heads = 12 * G_launch user-major heads (at most :data:`GDN_LAUNCH_USERS` users per launch;
  S = 128 runs two launches of 8) from a zero initial state; the conv reader restarts its FIR window from a zero history
  tile at every segment start (``seg_chunks = S / 32``); pad rows are an identity update through the beta / g mask.
* QSA: no indexer scoring, mask, top-k or id expansion.  For S <= 1024 every complete block of a user is selected, so
  ``sparse_sdpa_tiled`` takes a per-bucket constant block-id list per user (global ids ``b0 .. b0 + S/4 - 1``, ASCENDING,
  then sentinels) and the GLOBAL positions (= the row index at position 0).  RoPE and the block-start RoPE rows use the
  LOCAL position of each row.
* PLE: the FIR history is zeroed at every segment start and one snapshot window is kept per user.
* Output: the G last-real-token rows are gathered by an exact 0/1 select, then the final mixer and LM head run on 32 rows
  (row u = user u).

This module holds the shapes, the constants and the host-written inputs so they can be tested without a device.  The
device state and the body are in ``ttnn/model.py`` (``allocate_pack_state``, ``forward_prefill_pack``,
``capture_prefill_pack``, ``run_pack``) and the layer families' ``forward_pack`` methods.

Landing sources (what the P3d-3 lane landing reads, all in PERSISTENT buffers after the replay; ``base_u = u * S``)
--------------------------------------------------------------------------------------------------------------------
Per GDN layer (35 of them, the pack state's ``layers[i]``):

* recurrent state: ``layers[i].states[l]`` fp32 ``[96, 128, 128]`` per device (head dim 0 is the user-major
  ``(user, value head)`` of launch ``l``); user ``u`` is launch ``l = u // 8`` (:func:`gdn_state_slot`), heads
  ``12 * (u % 8) .. + 12``.  Launches of fewer than 8 users write the first ``12 * users`` heads.
* last conv-input rows: ``layers[i].conv_rows`` bf16 ``[1, 1, 512, 2560]``; user ``u`` has rows ``32 u .. 32 u + 2``
  (oldest first: the q|k|v rows ``P_u - 3 .. P_u - 1`` of its segment; a row before the sequence start is exact zero, as
  is every row of an empty slot).  This is the layout of the GDN ``history`` tile (rows 0..2 of a ``[1,1,32,2560]`` tile).
  Row ``32 u + 3`` (P3d-3) is the row ``P_u - 4``: the fourth ring entry a single-path lane holds in its (dead) fixed-phase
  slot 0.

Per QSA layer (12): the KV rows and compressed rows are already in the chain's dense caches at the user's base
(``state.layers[i].attention.packed_kv_cache`` rows ``base_u .. base_u + P_u - 1``, ``compressed_index_cache`` rows
``base_u / 4 .. + P_u / 4`` complete blocks; the open block's compressed row mixes pad rows and is the lane's to rebuild from
the raw-key ring); the raw-key ring source and the staging-tile source are the slab chunk state's per-layer kept buffers:
``chunk_state.layers[i].attention.kept_kv`` ``[1, 1, 2048, 256]`` (the ``[v | k]`` rows; the staging tile is the 32-row tile
holding row ``base_u + P_u - 1``, i.e. tile ``(base_u + P_u - 1) // 32``) and ``kept_raw`` ``[1, 1, 2048, 128]`` (the raw
index keys; the ring holds the rows ``base_u + 4 * (P_u // 4) .. base_u + P_u - 1`` of the open block).

PLE (layer 1): ``pack_state.ple_snapshots[u]`` bf16 ``[1, 137, 4, 640]`` (hidden-sharded), the window
``[history (9 rows) | normalized (128 rows)]`` of the 128-row block that holds the user's last real row; the 9 history rows
before the first pad row are window rows ``o + 1 .. o + 9`` with ``o = (P_u - 1) % 128`` (:func:`ple_history_first_row`).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

PACK_ROWS = 2048
PACK_BUCKETS = (128, 256, 512, 1024)
PACK_MAX_USERS = PACK_ROWS // PACK_BUCKETS[0]  # 16
GDN_LAUNCH_USERS = 8  # 12 * 9 = 108 <= the 110-core cap of the scan; 8 keeps the nv = 1 form
BLOCK_ROWS = 4  # tokens per compressed block (QSA COMPRESS_RATIO)
BLOCK_TOPK = 512  # ids per row of the block-shared attention kernel
BLOCK_SENTINEL = 0xFFFFFFFF
VALUE_HEADS = 12
CHUNK = 32
PLE_BLOCK_ROWS = 128
PLE_HISTORY_ROWS = 9
CONV_HISTORY_ROWS = 3
CONV_RING_ROWS = (
    CONV_HISTORY_ROWS + 1
)  # the conv ring's four entries: the history rows and the slot the next token takes
CONV_SELECT_ROWS = CHUNK * PACK_MAX_USERS  # the stacked per-user history select: 32 rows per user slot, 16 slots
GATHER_SELECT_ROWS = 32
PACK_PAD_TOKEN_ID = 0  # qwen38_prefill_driver.CHUNK_PAD_TOKEN_ID


@dataclass(frozen=True)
class PackBucket:
    """One bucket: ``users`` slots of ``rows_per_user`` rows on the 2048-row slab."""

    rows_per_user: int  # S
    users: int  # G
    launches: tuple[tuple[int, int], ...]  # (first user, user count) per GDN prims launch

    @property
    def seg_chunks(self) -> int:
        return self.rows_per_user // CHUNK

    @property
    def blocks_per_user(self) -> int:
        return self.rows_per_user // BLOCK_ROWS

    @property
    def ple_blocks_per_user(self) -> int:
        return self.rows_per_user // PLE_BLOCK_ROWS


def pack_bucket(rows_per_user: int) -> PackBucket:
    if rows_per_user not in PACK_BUCKETS:
        raise ValueError(f"a pack bucket is one of {PACK_BUCKETS} rows per user, got {rows_per_user!r}")
    users = PACK_ROWS // rows_per_user
    launches = tuple((first, min(GDN_LAUNCH_USERS, users - first)) for first in range(0, users, GDN_LAUNCH_USERS))
    return PackBucket(rows_per_user, users, launches)


def bucket_for_length(prompt_tokens: int) -> int | None:
    """The smallest bucket row count holding ``prompt_tokens`` (the last prompt token included), or None past 1024."""

    if isinstance(prompt_tokens, bool) or type(prompt_tokens) is not int or prompt_tokens < 1:
        raise ValueError(f"a pack prompt has at least one token, got {prompt_tokens!r}")
    for rows in PACK_BUCKETS:
        if prompt_tokens <= rows:
            return rows
    return None


def gdn_state_slot(bucket: PackBucket, user: int) -> tuple[int, int]:
    """(launch index, first head of the user's 12 heads in that launch's state buffer)."""

    for index, (first, count) in enumerate(bucket.launches):
        if first <= user < first + count:
            return index, VALUE_HEADS * (user - first)
    raise ValueError(f"user {user} is not in the {bucket.users}-user pack")


def ple_history_first_row(prompt_tokens: int) -> int:
    """The first of the 9 history rows in a user's snapshot window ``[history | normalized]`` (137 rows)."""

    return (prompt_tokens - 1) % PLE_BLOCK_ROWS + 1


def check_lengths(bucket: PackBucket, lengths: Sequence[int]) -> list[int]:
    lengths = [int(n) for n in lengths]
    if len(lengths) > bucket.users:
        raise ValueError(f"a {bucket.users}-user pack takes at most {bucket.users} users, got {len(lengths)}")
    for n in lengths:
        if not 0 <= n <= bucket.rows_per_user:
            raise ValueError(f"a user holds 0..{bucket.rows_per_user} tokens in this bucket, got {n}")
    return lengths + [0] * (bucket.users - len(lengths))


def plan_packs(lengths: Sequence[int]) -> list[tuple[int, list[int]]]:
    """Group prompts (by index) into packs: each prompt takes the smallest bucket that holds it (a prompt past the largest
    bucket is left out: the caller's single path), each bucket's prompts are cut into packs of at most G users in order.
    Returns ``[(rows per user, [prompt indices])]`` ordered by bucket, then arrival."""

    by_bucket: dict[int, list[int]] = {}
    for index, n in enumerate(lengths):
        rows = bucket_for_length(int(n))
        if rows is not None:
            by_bucket.setdefault(rows, []).append(index)
    packs = []
    for rows in PACK_BUCKETS:
        indices = by_bucket.get(rows, [])
        users = PACK_ROWS // rows
        packs.extend((rows, indices[first : first + users]) for first in range(0, len(indices), users))
    return packs


# ------------------------------------------------------------------------------------------------ attention constants


def block_id_rows(bucket: PackBucket, *, rows: int = PACK_ROWS, ids: int = BLOCK_TOPK) -> torch.Tensor:
    """The bucket's constant block-id rows (int64 ``[rows, ids]``): every row of user ``u`` is
    ``[b0, b0 + 1, .., b0 + S/4 - 1, sentinel, ..]`` with ``b0 = u * S / 4`` (global ids, ascending).

    The block-shared kernel reads only the first ``min(ids, complete)`` entries of a row (``complete`` = (p + 1) >> 2 of the
    GLOBAL position p) and skips an id ``>= complete``; the visible blocks of row j sit at indices ``< b0 + complete_local``
    exactly when the list is ascending, and any other order silently gives wrong attention (P3d-1 probe P2)."""

    if rows != bucket.users * bucket.rows_per_user:
        raise ValueError(f"{rows} rows is not {bucket.users} x {bucket.rows_per_user}")
    per = bucket.blocks_per_user
    if per > ids:
        raise ValueError(f"{per} blocks per user exceed the {ids} ids of a row")
    host = torch.full((rows, ids), BLOCK_SENTINEL, dtype=torch.int64)
    for user in range(bucket.users):
        b0 = user * per
        host[user * bucket.rows_per_user : (user + 1) * bucket.rows_per_user, :per] = torch.arange(b0, b0 + per)
    assert_block_ids_ascending(host)
    return host


def assert_block_ids_ascending(host: torch.Tensor) -> None:
    """Raise unless every row is a strictly ascending run of ids followed only by sentinels.

    The kernel gives silently wrong attention for any other order; this is the host-side guard of the bucket constant.
    """

    if host.dim() != 2:
        raise ValueError(f"block ids must be [rows, ids], got {tuple(host.shape)}")
    ids = host.to(torch.int64)
    real = ids != BLOCK_SENTINEL
    # sentinels trail: no real id after a sentinel
    after_sentinel = torch.cumsum((~real).to(torch.int64), dim=1) > 0
    if bool((real & after_sentinel).any()):
        raise ValueError("block ids: a real id follows a sentinel")
    steps = ids[:, 1:] - ids[:, :-1]
    both = real[:, 1:] & real[:, :-1]
    if bool(((steps <= 0) & both).any()):
        raise ValueError("block ids must be strictly ascending (sparse_sdpa_tiled silently misattends otherwise)")


def local_positions(bucket: PackBucket) -> torch.Tensor:
    """The rotary positions of the rows (int64 ``[3, 2048]``): row ``u*S + j`` is position ``j`` on all three axes."""

    local = torch.arange(bucket.rows_per_user, dtype=torch.int64).repeat(bucket.users)
    return local.reshape(1, -1).expand(3, -1).contiguous()


def global_positions(rows: int = PACK_ROWS) -> torch.Tensor:
    return torch.arange(rows, dtype=torch.int64)


# ---------------------------------------------------------------------------------------------- host-written inputs


def pack_token_rows(
    bucket: PackBucket, users: Sequence[Sequence[int]], pad_token: int = PACK_PAD_TOKEN_ID
) -> list[list[int]]:
    """Per slot the ``S`` token ids: the prompt, then the pad token; a missing slot is all pad."""

    if len(users) > bucket.users:
        raise ValueError(f"a {bucket.users}-user pack takes at most {bucket.users} users, got {len(users)}")
    out = []
    for slot in range(bucket.users):
        tokens = [int(t) for t in users[slot]] if slot < len(users) else []
        if len(tokens) > bucket.rows_per_user:
            raise ValueError(f"user {slot} has {len(tokens)} tokens, the bucket holds {bucket.rows_per_user}")
        out.append(tokens + [int(pad_token)] * (bucket.rows_per_user - len(tokens)))
    return out


def pack_input_images(bucket: PackBucket, lengths: Sequence[int]) -> dict[str, object]:
    """The host images of every per-pack device input that depends on the users' lengths:

    * ``row_mask_c`` fp32 ``[12, 64, 32, 1]``: 1.0 for a real row, 0.0 for a pad row (the GDN prims' page layout);
    * ``conv_select`` bf16 ``[512, 2048]``: row ``32 u + i`` (i < 3) one-hot at row ``u*S + P_u - 3 + i`` (zero when that is
      before the sequence start, and for an empty slot): the user's last three conv-input rows; row ``32 u + 3`` one-hot at
      ``u*S + P_u - 4`` (the ring's fourth entry, same rule);
    * ``gather_select`` bf16 ``[32, 2048]``: row u one-hot at row ``u*S + P_u - 1`` (zero for an empty slot);
    * ``ple_take`` / ``ple_keep`` float ``[16]``: per 128-row block, take = 1 for the block that holds the user's last real
      row (and keep = 1 - take); a block of a user without rows never takes.
    """

    lengths = check_lengths(bucket, lengths)
    s = bucket.rows_per_user
    keep_rows = torch.zeros(PACK_ROWS)
    conv = torch.zeros(CONV_SELECT_ROWS, PACK_ROWS)
    gather = torch.zeros(GATHER_SELECT_ROWS, PACK_ROWS)
    take = torch.zeros(PACK_ROWS // PLE_BLOCK_ROWS)
    for user, n in enumerate(lengths):
        base = user * s
        keep_rows[base : base + n] = 1.0
        if n:
            for i in range(CONV_HISTORY_ROWS):
                src = n - CONV_HISTORY_ROWS + i
                if src >= 0:
                    conv[CHUNK * user + i, base + src] = 1.0
            if n >= CONV_RING_ROWS:
                conv[CHUNK * user + CONV_HISTORY_ROWS, base + n - CONV_RING_ROWS] = 1.0
            gather[user, base + n - 1] = 1.0
            take[(base + n - 1) // PLE_BLOCK_ROWS] = 1.0
    row_mask_c = keep_rows.reshape(1, PACK_ROWS // CHUNK, CHUNK, 1).expand(VALUE_HEADS, -1, -1, -1).contiguous()
    return {
        "row_mask_c": row_mask_c,
        "conv_select": conv.to(torch.bfloat16),
        "gather_select": gather.to(torch.bfloat16),
        "ple_take": take,
        "ple_keep": 1.0 - take,
    }


def landing_rows(bucket: PackBucket, lengths: Sequence[int]) -> list[dict[str, object]]:
    """Per slot the documented landing coordinates (see the module docstring); empty slots have ``tokens == 0``."""

    lengths = check_lengths(bucket, lengths)
    rows = []
    for user, n in enumerate(lengths):
        launch, head0 = gdn_state_slot(bucket, user)
        base = user * bucket.rows_per_user
        rows.append(
            {
                "user": user,
                "tokens": n,
                "kv_row_base": base,
                "compressed_row_base": base // BLOCK_ROWS,
                "last_row": base + n - 1 if n else None,
                "staging_tile": (base + n - 1) // CHUNK if n else None,
                "gdn_state_launch": launch,
                "gdn_state_head0": head0,
                "conv_row0": CHUNK * user,
                "ple_snapshot": user,
                "ple_history_first_row": ple_history_first_row(n) if n else None,
            }
        )
    return rows


# ----------------------------------------------------------------------------------------- landing (P3d-3) host images
#
# A user's landing writes its lane from the persistent sources above.  Every select below is an exact 0/1 matmul operand
# (bf16, fp32 accumulation: one nonzero term per output element), written to a persistent device buffer per landing, so one
# program serves every user, length and lane.  What a lane holds after a SINGLE-path prefill of ``P`` tokens (chunk prefill of
# the first ``P - 1`` tokens, hand-off, the last token as a forced decode step), which the landing reproduces:
#
# * GDN recurrent: the state after all ``P`` tokens (the pack's final state: pad rows are an identity update).
# * GDN ring (fixed phase, canonical slots): slot ``j`` holds the conv-input row of token ``P - 4 + j``, i.e. slot 1
#   = row ``P - 3``, slot 2 = ``P - 2``, slot 3 = ``P - 1`` (the history the next step reads, oldest first) and slot 0 = row
#   ``P - 4`` (dead: the next token overwrites it); a row before the sequence start is exact zero.  The position-dependent
#   ring phase of the generic state cancels in the canonical order (``lanes.canonical_ring_source``), so the source row of
#   slot ``j`` is ``conv_rows`` row ``32 u + (j - 1) % 4`` whatever ``P % 4`` is.
# * QSA KV: rows ``0 .. round_up(P, 32) - 1`` of the user's segment in the lane's pages (rows past ``P`` are dead).
# * QSA compressed index: the whole lane cache = the user's compressed rows with every row ``>= P // 4`` zeroed, zero past the
#   segment's ``S/4`` rows (a landed lane keeps nothing of its previous occupant; the
#   complete blocks are the only rows a decode reads; the open block's row is rewritten from the ring before it is scored).
# * QSA staging tile: the 32-row tile that holds position ``P - 1``, rows past ``P - 1`` zero.
# * QSA raw-key ring: row ``j <= (P - 1) % 4`` = the raw index key of position ``4 * ((P - 1) // 4) + j``, other rows zero.
# * PLE slots: slot ``k`` (0..8) = the normalized PLE row of position ``P - 9 + k`` (zero before the sequence start).


def conv_ring_source_row(slot: int) -> int:
    """The row of a user's 32-row ``conv_rows`` tile that lands in canonical ring slot ``slot`` (0..3)."""

    if not 0 <= slot < CONV_RING_ROWS:
        raise ValueError(f"a ring slot is 0..{CONV_RING_ROWS - 1}, got {slot}")
    return (slot - 1) % CONV_RING_ROWS


def conv_hit_image(lanes: int, assignments: Sequence[tuple[int, int]], slot: int) -> torch.Tensor:
    """bf16 ``[lanes, 512]``: row ``lane`` one-hot at ``conv_rows`` row ``32 user + conv_ring_source_row(slot)`` for every
    ``(lane, user)``; its matmul with ``conv_rows`` is the ring slot's new content of the landed lanes (other rows zero).
    """

    image = torch.zeros(lanes, CONV_SELECT_ROWS)
    for lane, user in assignments:
        if not 0 <= lane < lanes or not 0 <= user < PACK_MAX_USERS:
            raise ValueError(f"lane {lane} / user {user} out of range ({lanes} lanes, {PACK_MAX_USERS} users)")
        image[lane, CHUNK * user + conv_ring_source_row(slot)] = 1.0
    return image.to(torch.bfloat16)


def conv_keep_image(lanes: int, assignments: Sequence[tuple[int, int]]) -> torch.Tensor:
    """bf16 ``[lanes, 1]``: 0 for a landed lane, 1 for every other lane."""

    keep = torch.ones(lanes, 1)
    for lane, _user in assignments:
        keep[lane, 0] = 0.0
    return keep.to(torch.bfloat16)


def compressed_select_image(bucket: PackBucket, user: int, length: int) -> torch.Tensor:
    """bf16 ``[S/4, 512]``: row ``i < length // 4`` one-hot at compressed row ``user * S/4 + i`` of the pack's first 512 rows;
    the other rows are zero (the open block and everything past the prompt)."""

    check_lengths(bucket, [length])
    per = bucket.blocks_per_user
    image = torch.zeros(per, PACK_ROWS // BLOCK_ROWS)
    for i in range(length // BLOCK_ROWS):
        image[i, user * per + i] = 1.0
    return image.to(torch.bfloat16)


def staging_select_image(bucket: PackBucket, user: int, length: int) -> torch.Tensor:
    """bf16 ``[32, 2048]``: row ``i <= (length - 1) % 32`` one-hot at pack row ``user * S + 32 * ((length - 1) // 32) + i`` (the
    tile that holds position ``length - 1``); the later rows are zero."""

    if not 1 <= length <= bucket.rows_per_user:
        raise ValueError(f"a landed user holds 1..{bucket.rows_per_user} tokens, got {length}")
    image = torch.zeros(CHUNK, PACK_ROWS)
    tile = (length - 1) // CHUNK
    for i in range((length - 1) % CHUNK + 1):
        image[i, user * bucket.rows_per_user + CHUNK * tile + i] = 1.0
    return image.to(torch.bfloat16)


def raw_ring_select_image(bucket: PackBucket, user: int, length: int) -> torch.Tensor:
    """bf16 ``[32, 2048]``: row ``j <= (length - 1) % 4`` one-hot at pack row ``user * S + 4 * ((length - 1) // 4) + j`` (the
    raw index keys of the block that holds position ``length - 1``, up to it); the other rows are zero."""

    if not 1 <= length <= bucket.rows_per_user:
        raise ValueError(f"a landed user holds 1..{bucket.rows_per_user} tokens, got {length}")
    image = torch.zeros(CHUNK, PACK_ROWS)
    block = (length - 1) // BLOCK_ROWS
    for j in range((length - 1) % BLOCK_ROWS + 1):
        image[j, user * bucket.rows_per_user + BLOCK_ROWS * block + j] = 1.0
    return image.to(torch.bfloat16)


def ple_history_select_image(length: int) -> torch.Tensor:
    """bf16 ``[137, 32]``: column ``k < 9`` one-hot at window row ``ple_history_first_row(length) + k`` (the nine history rows
    before the first pad row); the other columns are zero."""

    if length < 1:
        raise ValueError(f"a landed user holds at least one token, got {length}")
    image = torch.zeros(PLE_HISTORY_ROWS + PLE_BLOCK_ROWS, CHUNK)
    first = ple_history_first_row(length)
    for k in range(PLE_HISTORY_ROWS):
        image[first + k, k] = 1.0
    return image.to(torch.bfloat16)


# ------------------------------------------------------------------------------------------------ device state records


@dataclass(frozen=True)
class GDNPackLayer:
    """One GDN layer's persistent landing sources: ``states[l]`` fp32 ``[96, 128, 128]`` per device (launch l's user-major
    final recurrent states) and ``conv_rows`` bf16 ``[1, 1, 512, 2560]`` (user u's last three conv-input rows at rows
    ``32 u .. 32 u + 2``)."""

    states: tuple
    conv_rows: object


@dataclass
class PackState:
    """The pack's persistent device buffers and constants, shared by every bucket (``Qwen38TTNNTextModel.allocate_pack_state``).

    ``chunk_state`` is the 2048-row slab chunk state whose buffers the pack runs on.  ``layers[i]`` is a
    :class:`GDNPackLayer` for a GDN layer and None for a QSA layer.  ``chunk`` / ``positions_row`` are the QSA inputs at
    position 0 (the zero position scalar, the compressed tile ids, the row index as the global query positions);
    ``block_ids[S]`` the bucket's constant ascending block-id rows.  ``zero_states[h]`` is the zero initial recurrent state of a
    launch of ``h`` heads.  The ``row_mask_c`` / ``conv_select`` / ``gather_select`` tensors and the PLE ``keep`` / ``take``
    scalars are host-written per pack (:func:`pack_input_images`)."""

    chunk_state: object
    buckets: dict
    layers: tuple
    zero_history: object
    zero_states: dict
    row_mask_c: object
    conv_select: object
    gather_select: object
    ple_zero_history: object
    ple_snapshots: tuple
    ple_keep: tuple
    ple_take: tuple
    chunk: object
    positions_row: object
    block_ids: dict
    zero_position: object
    gather_compute_config: object = None
    owned: tuple = ()

    def view(self, rows_per_user: int) -> "PackView":
        bucket = self.buckets[rows_per_user]
        return PackView(self, bucket, self.block_ids[rows_per_user])


@dataclass(frozen=True)
class PackView:
    """What the layers read for one call: the pack state with its bucket selected (attribute access passes through)."""

    state: PackState
    bucket: PackBucket
    block_ids: object

    def __getattr__(self, name):
        return getattr(self.state, name)
