# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Packed token masks of the fused MTP spec step (phase 3, design note 80 sections 3.1-3.2). Pure torch, no ttnn.

One mask row per engine row r = u*T + j (user u, candidate column j). The host composes grammar, allowed_token_ids,
bad_words and min_tokens into ONE row; the device edits the shard logits with it before the top-k trace.

Bit layout (the one xgrammar and the plugin use): int32 words, bit ``v % 32`` of word ``v // 32`` is 1 when token ``v`` is
allowed. The vocabulary is 248320 = 8 shards x 31040 = 8 x 970 words x 32 bits, so every shard holds whole words: shard ``d``
owns words d*970 .. d*970+969 (ids d*31040 ..). A row of -1 (all ones) means "no constraint". Ids a tokenizer does not use
(vocabulary padding) are disallowed: a narrower bitmask is padded with zero words, and ``clear_padding`` clears the bits
past the real vocabulary.

Masked logits are written as MASK_NEG (-2**99) on the device and on every host path (reference, walk, fallback). It is a
power of two, so it is exact in bf16: the device ``where`` writes a scalar into a bf16 tensor by truncation and the host
rounds to nearest, and the two agree bit for bit only for a bf16-exact value (design note 81 note A). It is finite
(no inf - inf in the stats trace and the accept body) and far below any real logit (adding it to a bf16 logit gives it
back exactly).

PERMUTED packing (design note 81, P0b): the device unpacks 0.335 ms with it (0.9 ms with the standard packing). In
the permuted words of shard ``s``, bit ``k`` of word ``w`` is token ``s * 31040 + k * 970 + w`` (the standard packing has
token ``s * 31040 + 32 * w + k``). The host repacks only the constrained rows (``repack_to_permuted``); the device input is
a contiguous int32 ``[8, 32, 970]`` (shard, row, word) sharded on dim 0.
"""

import numpy as np
import torch

WORD_BITS = 32
VOCAB = 248320
SHARDS = 8
SHARD_VOCAB = VOCAB // SHARDS  # 31040
SHARD_WORDS = SHARD_VOCAB // WORD_BITS  # 970
MASK_WORDS = VOCAB // WORD_BITS  # 7760
MASK_ROWS = 32  # one tile row: the largest engine row count of a bucket (B*T <= 32)
RING_DEPTH = 2
PERM_RING_DEPTH = 4  # permuted staging buffers: a step's host tensor stays alive until its event
MASK_NEG = -(2.0**99)  # what the device AND the host write at a disallowed token (raw logits, before the temperature)
MASK_KTH_GUARD = (
    -1e20
)  # the top-k tie rule ignores a k-th value below this: it is a masked value (or its quotient by T)

_BITS = torch.arange(WORD_BITS, dtype=torch.int32)


_BIT_IDX = np.arange(WORD_BITS)


def repack_to_permuted(std):
    """STANDARD-packed rows ``[R, 7760]`` (int32; token ``v`` = bit ``v % 32`` of word ``v // 32``) as PERMUTED rows
    ``[R, 8, 970]`` (int32): in shard ``s``, bit ``k`` of word ``w`` is token ``s * 31040 + k * 970 + w``. Numpy unpackbits /
    transpose / packbits (0.2 ms per row, design note 81)."""
    a = np.ascontiguousarray(torch.as_tensor(std).to(torch.int32).numpy())
    R = int(a.shape[0])
    if int(a.shape[-1]) != MASK_WORDS:
        raise ValueError(f"repack needs {MASK_WORDS} words, got {int(a.shape[-1])}")
    bits = np.unpackbits(
        a.view(np.uint8).reshape(R, SHARDS, SHARD_WORDS * 4), axis=-1, bitorder="little"
    )  # [R,8,31040]
    bits = bits.reshape(R, SHARDS, WORD_BITS, SHARD_WORDS).transpose(0, 1, 3, 2)  # [R,8,w,k]: token k*970+w
    packed = np.packbits(np.ascontiguousarray(bits), axis=-1, bitorder="little")  # [R,8,970,4] bytes
    return torch.from_numpy(
        packed.reshape(R, SHARDS, SHARD_WORDS * 4).view(np.int32).reshape(R, SHARDS, SHARD_WORDS).copy()
    )


def repack_from_permuted(perm):
    """The inverse of ``repack_to_permuted`` (tests): PERMUTED ``[R, 8, 970]`` or ``[R, 7760]`` rows -> STANDARD ``[R, 7760]``."""
    a = np.ascontiguousarray(torch.as_tensor(perm).to(torch.int32).numpy().reshape(-1, SHARDS, SHARD_WORDS))
    R = int(a.shape[0])
    bits = np.unpackbits(a.view(np.uint8).reshape(R, SHARDS, SHARD_WORDS * 4), axis=-1, bitorder="little")
    bits = bits.reshape(R, SHARDS, SHARD_WORDS, WORD_BITS).transpose(0, 1, 3, 2)  # [R,8,k,w]: token k*970+w
    packed = np.packbits(np.ascontiguousarray(bits).reshape(R, SHARDS, SHARD_VOCAB), axis=-1, bitorder="little")
    return torch.from_numpy(packed.view(np.int32).reshape(R, MASK_WORDS).copy())


def device_mask_input(perm_rows):
    """``[R, 8, 970]`` permuted rows -> the device input int32 ``[8, 32, 970]`` contiguous (shard, row, word); rows R.. are all
    ones. A new tensor; the model shards it on dim 0 (``ShardTensorToMesh(dim=0)``)."""
    out = torch.full((SHARDS, MASK_ROWS, SHARD_WORDS), -1, dtype=torch.int32)
    perm_rows = torch.as_tensor(perm_rows).to(torch.int32)
    out[:, : int(perm_rows.shape[0])] = perm_rows.permute(1, 0, 2)
    return out


def all_ones(*lead, words=MASK_WORDS):
    """The no-constraint mask ``[*lead, words]`` int32 (every word -1)."""
    return torch.full((*lead, words), -1, dtype=torch.int32)


def is_unconstrained(mask):
    """``[...]`` bool: the row is -1 in every word (no token is ruled out). ``mask`` ``[..., W]`` int32."""
    return (torch.as_tensor(mask) == -1).all(dim=-1)


def normalise_width(mask, words=MASK_WORDS):
    """``mask`` ``[..., W]`` padded with zero words (ids no grammar can name: disallowed) to ``words``; a wider one is an error."""
    mask = torch.as_tensor(mask).to(torch.int32)
    w = int(mask.shape[-1])
    if w > words:
        raise ValueError(f"mask has {w} words, the engine vocabulary has {words}")
    if w == words:
        return mask
    return torch.cat([mask, torch.zeros((*mask.shape[:-1], words - w), dtype=torch.int32)], dim=-1)


def unpack_allowed(mask, vocab=VOCAB):
    """``[..., vocab]`` bool, True where allowed, from the packed ``[..., W]`` mask. A mask narrower than ``vocab`` leaves its
    tail disallowed; a wider one is cut."""
    mask = torch.as_tensor(mask).to(torch.int32)
    allowed = ((mask.unsqueeze(-1) >> _BITS) & 1).bool().reshape(*mask.shape[:-1], -1)
    n = int(allowed.shape[-1])
    if n < vocab:
        allowed = torch.cat([allowed, torch.zeros((*allowed.shape[:-1], vocab - n), dtype=torch.bool)], dim=-1)
    return allowed[..., :vocab]


def pack_allowed(allowed, words=MASK_WORDS):
    """The packed ``[..., words]`` int32 mask of a ``[..., V]`` bool (True = allowed). Ids past V are disallowed."""
    allowed = torch.as_tensor(allowed).bool()
    v = int(allowed.shape[-1])
    if v > words * WORD_BITS:
        raise ValueError(f"{v} ids do not fit {words} words")
    pad = words * WORD_BITS - v
    if pad:
        allowed = torch.cat([allowed, torch.zeros((*allowed.shape[:-1], pad), dtype=torch.bool)], dim=-1)
    bits = allowed.reshape(*allowed.shape[:-1], words, WORD_BITS).to(torch.int64)
    packed = (bits << _BITS.to(torch.int64)).sum(dim=-1)  # 0 .. 2**32 - 1
    return torch.where(packed >= 2**31, packed - 2**32, packed).to(torch.int32)


def clear_padding(mask, vocab):
    """``mask`` with every bit of an id >= ``vocab`` (tokenizer size; the vocabulary padding) cleared. A new tensor."""
    mask = torch.as_tensor(mask).to(torch.int32)
    keep = pack_allowed(torch.ones(int(vocab), dtype=torch.bool), words=int(mask.shape[-1]))
    return mask & keep


def compose_masks(*masks):
    """AND of equally shaped packed masks (every class only clears bits, so the order does not matter)."""
    out = torch.as_tensor(masks[0]).to(torch.int32).clone()
    for m in masks[1:]:
        out &= torch.as_tensor(m).to(torch.int32)
    return out


def empty_rows(mask, vocab=VOCAB):
    """``[...]`` bool: no token below ``vocab`` is allowed (a row the engine and the plugin must refuse: no distribution)."""
    return ~unpack_allowed(mask, vocab).any(dim=-1)


def token_allowed(mask_rows, ids):
    """``[N, C]`` bool: is ``ids`` [N, C] allowed by ``mask_rows`` [N, W]. A bit lookup, no unpacking of the row."""
    ids = torch.as_tensor(ids).to(torch.int64)
    words = torch.as_tensor(mask_rows).to(torch.int32).gather(1, (ids >> 5).clamp(0, int(mask_rows.shape[-1]) - 1))
    inside = (ids >> 5) < int(mask_rows.shape[-1])
    return (((words >> (ids & 31).to(torch.int32)) & 1) == 1) & inside


def apply_to_logits(logits, mask, fill=float("-inf")):
    """``logits`` ``[N, V]`` with every disallowed token set to ``fill``; ``mask`` ``[N, W]``. A new tensor (the input is not
    changed); a row with no constraint is copied as it is."""
    logits = torch.as_tensor(logits)
    allowed = unpack_allowed(mask, int(logits.shape[-1]))
    return logits.masked_fill(~allowed, fill)


def shard_split(mask):
    """``[R, 8, 970]`` view of ``[R, 7760]``: row r, shard d = the words of vocabulary shard d (what one chip receives)."""
    mask = torch.as_tensor(mask)
    if int(mask.shape[-1]) != MASK_WORDS:
        raise ValueError(f"shard_split needs {MASK_WORDS} words, got {int(mask.shape[-1])}")
    return mask.reshape(*mask.shape[:-1], SHARDS, SHARD_WORDS)


def mask_to_engine_rows(mask, inv, B, T):
    """The plugin's ``mask`` ``[b, T, W]`` (runner slot order, b <= B live users) as the engine-row mask ``[B, T, W]``.
    Follows SpecEngine._to_rows: ``inv`` = ``_phys_perm[1]`` (None at the identity map, then the rows pass through). Users the
    plugin did not send (b < B) and every unconstrained row are all ones. Returns None for ``mask`` None."""
    if mask is None:
        return None
    mask = normalise_width(mask)
    if mask.dim() != 3 or int(mask.shape[1]) != T or int(mask.shape[0]) > B:
        raise ValueError(f"mask must be [<= {B}, {T}, W], got {tuple(mask.shape)}")
    if int(mask.shape[0]) < B:
        mask = torch.cat([mask, all_ones(B - int(mask.shape[0]), T)], dim=0)
    return mask if inv is None else mask[inv]


def mask_rows_flat(mask, B, T):
    """``mask`` ``[B, T, W]`` or ``[B*T, W]`` as ``[B*T, W]`` engine rows r = u*T + j; None stays None."""
    if mask is None:
        return None
    mask = torch.as_tensor(mask)
    flat = mask.reshape(-1, int(mask.shape[-1]))
    if int(flat.shape[0]) != B * T:
        raise ValueError(f"mask has {int(flat.shape[0])} rows, the step has {B * T}")
    return flat


class MaskStagingRing:
    """Two host buffers ``[MASK_ROWS, MASK_WORDS]`` int32 the device write reads from. The H2D copy of step N may still be
    queued when step N+1 is staged, so the buffers alternate (a ring of 2); the caller keeps buffer N alive until the event of
    step N. Pinned memory is used when the torch build supports it (no CUDA: plain host memory)."""

    def __init__(self, rows=MASK_ROWS, words=MASK_WORDS, depth=RING_DEPTH):
        self.bufs = [self._alloc(rows, words) for _ in range(depth)]
        self.next_index = 0

    @staticmethod
    def _alloc(rows, words):
        buf = all_ones(rows, words=words)
        try:
            return buf.pin_memory()
        except RuntimeError:  # no accelerator for pinning on this host
            return buf

    def stage(self, mask):
        """Copy ``mask`` ``[n, W]`` (engine rows) into the next buffer, rows n.. all ones; returns that buffer."""
        mask = torch.as_tensor(mask).to(torch.int32)
        n = int(mask.shape[0])
        buf = self.bufs[self.next_index]
        if n > int(buf.shape[0]) or int(mask.shape[1]) != int(buf.shape[1]):
            raise ValueError(f"cannot stage {tuple(mask.shape)} into {tuple(buf.shape)}")
        buf[:n] = mask
        buf[n:] = -1
        self.next_index = (self.next_index + 1) % len(self.bufs)
        return buf


class PermMaskStagingRing:
    """Host buffers of the DEVICE mask input: int32 ``[8, 32, 970]`` (shard, row, word, PERMUTED words), a ring of
    ``PERM_RING_DEPTH``. ``stage(rows, idx)`` fills a buffer with all ones, repacks only the constrained rows ``idx`` (a list of
    engine rows r < 32; ``rows`` = the standard-packed ``[R, 7760]`` engine-row mask) and returns it. The caller keeps the
    returned buffer (and the ttnn host tensor built from it) alive until the step's event: the buffer is reused PERM_RING_DEPTH
    stages later."""

    def __init__(self, depth=PERM_RING_DEPTH):
        self.bufs = [torch.full((SHARDS, MASK_ROWS, SHARD_WORDS), -1, dtype=torch.int32) for _ in range(depth)]
        self.next_index = 0

    def stage(self, rows, idx):
        rows = torch.as_tensor(rows)
        idx = [int(r) for r in idx]
        if any(not 0 <= r < MASK_ROWS or r >= int(rows.shape[0]) for r in idx):
            raise ValueError(f"constrained rows {idx} outside the {int(rows.shape[0])} mask rows (max {MASK_ROWS})")
        buf = self.bufs[self.next_index]
        buf.fill_(-1)
        if idx:
            sel = torch.as_tensor(idx, dtype=torch.int64)
            buf[:, sel] = repack_to_permuted(rows[sel]).permute(1, 0, 2)
        self.next_index = (self.next_index + 1) % len(self.bufs)
        return buf
