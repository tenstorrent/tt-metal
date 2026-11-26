# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Token masks of a deferred verify (vllm-tt-plugin ``supports_spec_device_mask``) for Qwen36MTPContractForCausalLM.

The plugin hands ``sample_deferred_decode`` one packed mask row per verify row and column: int32 ``[rows, T, W]``,
``W = ceil(V / 32)`` words, xgrammar's packing (bit ``v % 32`` of word ``v // 32`` allows token ``v``; a row of -1 words
constrains nothing). It is the grammar bitmask composed with ``allowed_token_ids``, ``bad_words`` and a pending
``min_tokens``. The sampler draws column j of a constrained user from the masked distribution, and a greedy constrained
user takes the masked argmax (a draw with top-k 1), so the plugin's id walk stays exact rejection sampling of the
masked target.

Device graph (``VerifyMask.apply_``, a trace of its own, replayed between the verify trace and the sampler trace): each
device holds the words of its own vocabulary shard of every verify row, in BIT-PLANE order: in shard s, bit k of word w
allows token ``s * 32 * Ws + k * Ws + w`` (Ws = W / shards). Shifting the row's ``[Ws]`` words right by 0..31 and taking
bit 0 then lays the shard's allowed flags out in token order, ``[32, Ws]`` = ``[32 * Ws]``, with no transpose on the
device. The flags become 0 or MASK_NEG and are added to the shard's logits in place. MASK_NEG = -2**99 is exact in
bfloat16 and far below any logit, so a masked logit becomes MASK_NEG and an allowed one keeps its bits. It stays finite:
the sampler's top-k pad (spec_sample.PAD_LOGIT = -1e30) still ranks below it, and a top-k that reaches past the allowed
tokens takes masked candidates of probability 0.

The host turns the plugin's packing into bit planes for the constrained engine rows only (``shard_bitplanes``); an
unconstrained row is staged as -1 words, which adds 0 everywhere.
"""

import numpy as np
import torch

import ttnn

WORD_BITS = 32
MASK_NEG = -(2.0**99)


def mask_words(vocab):
    """Words of one packed mask row of a ``vocab``-token vocabulary."""
    return -(-int(vocab) // WORD_BITS)


def shard_words(vocab, shards):
    """Words of one vocabulary shard's bit-plane row, or ValueError when the shards do not hold whole bit planes."""
    vocab, shards = int(vocab), int(shards)
    if vocab % (shards * WORD_BITS):
        raise ValueError(f"a vocabulary of {vocab} does not split into {shards} shards of whole 32-bit words")
    return vocab // shards // WORD_BITS


def normalise_rows(mask, vocab):
    """``mask`` int32 ``[..., W]`` with exactly ``mask_words(vocab)`` words: a narrower mask is padded with zero words
    (the ids past it no grammar names: disallowed), and the bits of ids at or past ``vocab`` are cleared. A wider mask
    raises ValueError. A new tensor."""
    mask = torch.as_tensor(mask).to(torch.int32)
    words = mask_words(vocab)
    width = int(mask.shape[-1])
    if width > words:
        raise ValueError(f"a mask of {width} words is wider than the {words} words of a {vocab}-token vocabulary")
    if width < words:
        mask = torch.cat([mask, torch.zeros((*mask.shape[:-1], words - width), dtype=torch.int32)], dim=-1)
    else:
        mask = mask.clone()
    tail = int(vocab) % WORD_BITS
    if tail:
        mask[..., -1] &= (1 << tail) - 1
    return mask


def unconstrained(mask):
    """``[...]`` bool: the row of ``mask`` ``[..., W]`` is -1 in every word."""
    return (torch.as_tensor(mask) == -1).all(dim=-1)


def allowed_tokens(mask, vocab):
    """``[..., vocab]`` bool, True where ``mask`` ``[..., W]`` (plugin packing) allows the token."""
    mask = torch.as_tensor(mask).to(torch.int32)
    bits = torch.arange(WORD_BITS, dtype=torch.int32)
    allowed = ((mask.unsqueeze(-1) >> bits) & 1).bool().reshape(*mask.shape[:-1], -1)
    if int(allowed.shape[-1]) < int(vocab):
        pad = torch.zeros((*allowed.shape[:-1], int(vocab) - int(allowed.shape[-1])), dtype=torch.bool)
        allowed = torch.cat([allowed, pad], dim=-1)
    return allowed[..., : int(vocab)]


def shard_bitplanes(rows, shards, words):
    """``rows`` int32 ``[R, shards * 32 * words / 32]`` (plugin packing, one row per vocabulary word group) as the device
    layout int32 ``[shards, R, words]``: in shard s, bit k of word w is token ``s * 32 * words + k * words + w``."""
    rows = np.ascontiguousarray(torch.as_tensor(rows).to(torch.int32).numpy())
    R = int(rows.shape[0])
    if int(rows.shape[-1]) != shards * words:
        raise ValueError(
            f"bit planes of {shards} x {words} words need rows of {shards * words} words, got {rows.shape}"
        )
    flags = np.unpackbits(rows.view(np.uint8).reshape(R, shards, 4 * words), axis=-1, bitorder="little")
    planes = flags.reshape(R, shards, WORD_BITS, words).transpose(0, 1, 3, 2)  # [R, s, w, k]
    packed = np.packbits(np.ascontiguousarray(planes), axis=-1, bitorder="little")  # [R, s, w, 4] bytes
    out = packed.reshape(R, shards, 4 * words).view(np.int32).reshape(R, shards, words)
    return torch.from_numpy(np.ascontiguousarray(out.transpose(1, 0, 2)))


def masked_logits(logits, mask, vocab):
    """Host reference of ``VerifyMask.apply_``: ``logits`` ``[R, vocab]`` with every token ``mask`` ``[R, W]`` rules out
    set to MASK_NEG, in bfloat16."""
    x = torch.as_tensor(logits).to(torch.bfloat16)
    neg = torch.tensor(MASK_NEG, dtype=torch.bfloat16)
    return torch.where(allowed_tokens(mask, vocab), x, neg)


class VerifyMask:
    """The device half: the persistent bit-plane input ``[1, rows, words]`` int32 ROW_MAJOR per device (shard d of the
    vocabulary on device d, or the whole vocabulary on every device with ``shards`` 1), and the trace-capturable
    ``apply_``. ``rows``: the verify's B * T rows; ``vocab``: the logits' width before sharding."""

    def __init__(self, mesh, rows, vocab, shards=1):
        self.mesh, self.rows, self.vocab, self.shards = mesh, int(rows), int(vocab), int(shards)
        self.words = shard_words(self.vocab, self.shards)
        self.width = WORD_BITS * self.words
        self.mask_in = ttnn.from_torch(
            torch.full((self.shards, self.rows, self.words), -1, dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            mesh_mapper=self._mapper(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.shifts = ttnn.from_torch(
            torch.arange(WORD_BITS, dtype=torch.int32).reshape(1, WORD_BITS, 1),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _mapper(self):
        if self.shards > 1:
            return ttnn.ShardTensorToMesh(self.mesh, dim=0)
        return ttnn.ReplicateTensorToMesh(self.mesh)

    def device_rows(self, mask):
        """The device input int32 ``[shards, rows, words]`` of the engine-row mask ``mask`` ``[R, W]`` (R <= rows,
        plugin packing, already ``normalise_rows``): the constrained rows' bit planes, -1 everywhere else."""
        mask = torch.as_tensor(mask).to(torch.int32)
        if int(mask.shape[0]) > self.rows:
            raise ValueError(f"a mask of {int(mask.shape[0])} rows for a verify of {self.rows} rows")
        out = torch.full((self.shards, self.rows, self.words), -1, dtype=torch.int32)
        idx = (~unconstrained(mask)).nonzero().reshape(-1)
        if idx.numel():
            out[:, idx] = shard_bitplanes(mask[idx], self.shards, self.words)
        return out

    def stage(self, mask):
        """Host -> device copy of ``mask`` (engine rows, ``device_rows``) into the persistent input (address kept)."""
        host = ttnn.from_torch(
            self.device_rows(mask), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=self._mapper()
        )
        ttnn.copy_host_to_device_tensor(host, self.mask_in)

    def apply_(self, x):
        """Edit ``x`` in place: bfloat16 TILE ``[1, 1, r, width]`` per device (r <= rows: the first r verify rows, a
        verify row bucket's), every masked token set to MASK_NEG."""
        r, width = int(x.shape[2]), int(x.shape[3])
        if width != self.width or not 1 <= r <= self.rows:
            raise ValueError(f"the mask edits [1, 1, <= {self.rows}, {self.width}] logits, got {tuple(x.shape)}")
        m = self.mask_in
        if r < self.rows:
            m = ttnn.slice(m, (0, 0, 0), (1, r, self.words))
        planes = ttnn.bitwise_right_shift(ttnn.reshape(m, (r, 1, self.words)), self.shifts)  # [r, 32, words]
        if r < self.rows:
            ttnn.deallocate(m)
        flags = ttnn.bitwise_and(planes, 1)
        ttnn.deallocate(planes)
        tiled = ttnn.to_layout(ttnn.reshape(flags, (1, 1, r, self.width)), ttnn.TILE_LAYOUT)
        ttnn.deallocate(flags)
        offset = ttnn.where(tiled, 0.0, MASK_NEG)
        ttnn.deallocate(tiled)
        ttnn.add_(x, offset)
        ttnn.deallocate(offset)

    def release(self):
        for t in (self.mask_in, self.shifts):
            ttnn.deallocate(t)
