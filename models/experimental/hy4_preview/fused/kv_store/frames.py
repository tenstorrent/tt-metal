# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Address arithmetic of the paged KV store: plain Python, no ttnn, no torch. ``kernels/kv_frames.hpp`` is the same
arithmetic for kernels; ``tests/test_frames_host.py`` holds the two against each other and against the contiguous
arithmetic the readers use today.

The storage (KV_STORE.md 1.2): a position p of a conversation lives on chip ``(p // 128) % 32`` at stripe row
``(p // 4096) * 128 + p % 128``. Per chip a cache tensor is ``[slots, 1, R, width]``, ND-sharded in 32-row shards
round-robin over the 8 DRAM banks, slot-major. When R is a multiple of 256 (8192 tokens) every slot starts on bank
0, so for stripe row r of any slot

    bank = (r // 32) % 8        the slot's shard in that bank = r // 256        row inside the shard = r % 32

A FRAME is 8192 tokens = 256 stripe rows per chip = exactly one shard in each of the 8 banks, for every layer. A
tensor holds, per bank, a sequence of shards of ``shard_bytes``; where the shard of (frame f, layer l) is in that
sequence is the tensor's GEOMETRY, two strides in shards:

    shard index = f * frame_stride + l * slot_stride

    frame-major pool  [frames * layers, 1, 256, w]  (one frame = one "user" of capacity 8192; KV_STORE.md 4)
        frame_stride = layers, slot_stride = 1,  f = the pool frame
    slot-major cache  [users * layers, 1, C * 256, w]  (today's layout, the prefill stack's; the workbench)
        frame_stride = 1, slot_stride = C,  f = user * layers * C + k  for the k-th 8192 tokens of ``user``

The same offset inside the bank holds on every bank and on every chip, so ONE table per decode row serves all
layers, banks and chips. A table entry is the byte offset of the frame's layer-0 shard inside a bank,
``f * frame_stride * shard_bytes`` ("frame base"); a kernel adds ``layer * slot_stride * shard_bytes`` (one value
per launch) and the row inside the shard:

    bank   = (r >> 5) & 7
    offset = table[r >> 8] + slot_bytes + (r & 31) * row_stride          (locate)

With the identity table of a slot-major cache this IS today's contiguous arithmetic, byte for byte, which is what
lets a reader take the table before the storage is re-allocated as a pool.
"""

from __future__ import annotations

from dataclasses import dataclass

CHIPS = 32  # chips a context is striped over (block-cyclic), linear chip = 4 * mesh row + mesh column
STRIPE_BLOCK = 128  # consecutive positions a chip holds per turn
PERIOD = STRIPE_BLOCK * CHIPS  # 4096 positions: 128 rows on every chip
SHARD_ROWS = 32  # rows of one ND shard
BANKS = 8  # DRAM banks of a Blackhole chip
FRAME_ROWS = SHARD_ROWS * BANKS  # 256 stripe rows per chip: one shard in every bank
FRAME_TOKENS = FRAME_ROWS * CHIPS  # 8192
TABLE_FRAMES = 128  # frames a decode row's table holds: 1,048,576 tokens
UNMAPPED = 0xFFFFFFFF  # table entry of a frame the conversation does not have


@dataclass(frozen=True)
class Kind:
    """One kind of layer storage. ``row_stride`` 0: the shard is not row-addressed (tiles), locate() gives the
    address of the 32-row block."""

    name: str
    layers: int  # layer slots of one conversation
    width: int  # last dimension of the cache tensor
    shard_bytes: int  # bytes of one 32-row shard in a bank
    row_stride: int  # bytes from one row to the next inside a shard

    @property
    def frame_bytes_per_bank(self) -> int:
        return self.layers * self.shard_bytes

    @property
    def frame_bytes_per_chip(self) -> int:
        return BANKS * self.frame_bytes_per_bank


# tencent/Hy4-preview on the prefill stack's storage (KV_STORE.md 1.1): 78 layers of packed latent rows (656 B:
# FP8 latent | FP32 scales | BF16 RoPE, one row per 704 B page) and 21 full-indexer layers of BFP8 index keys
# (128 dims: 4 tiles of 1088 B per 32-row block).
LATENT = Kind("latent", layers=78, width=656, shard_bytes=SHARD_ROWS * 704, row_stride=704)
INDEX = Kind("index", layers=21, width=128, shard_bytes=4 * 1088, row_stride=0)
KINDS = (LATENT, INDEX)
BYTES_PER_TOKEN_PER_CHIP = sum(k.frame_bytes_per_chip for k in KINDS) / FRAME_TOKENS  # 1,805.25


@dataclass(frozen=True)
class Geometry:
    """Where the shard of (frame, layer) is in a bank's share of one tensor, in shards (see the module docstring)."""

    kind: Kind
    frame_stride: int
    slot_stride: int

    @staticmethod
    def pool(kind: Kind) -> "Geometry":
        """Frame-major pool ``[frames * layers, 1, 256, width]``: frame f, layer l is slot ``f * layers + l``."""
        return Geometry(kind, frame_stride=kind.layers, slot_stride=1)

    @staticmethod
    def contiguous(kind: Kind, capacity_frames: int) -> "Geometry":
        """Slot-major cache ``[users * layers, 1, capacity_frames * 256, width]``: slot = user * layers + layer."""
        return Geometry(kind, frame_stride=1, slot_stride=capacity_frames)

    def frame_base(self, frame: int) -> int:
        """Table entry: byte offset inside a bank of the frame's layer-0 shard."""
        return frame * self.frame_stride * self.kind.shard_bytes

    def slot_bytes(self, layer: int) -> int:
        """What a kernel adds for its layer (rank, for the index keys)."""
        return layer * self.slot_stride * self.kind.shard_bytes

    def contiguous_frame(self, user: int, k: int) -> int:
        """The frame number of the k-th 8192 tokens of ``user`` in a slot-major cache (identity table entry)."""
        assert self.frame_stride == 1 and 0 <= k < self.slot_stride
        return user * self.kind.layers * self.slot_stride + k

    def identity_table(self, user: int, frames: int | None = None) -> list[int]:
        """The table under which locate() addresses ``user``'s contiguous slots of a slot-major cache."""
        n = self.slot_stride if frames is None else frames
        table = [self.frame_base(self.contiguous_frame(user, k)) for k in range(n)]
        return table + [UNMAPPED] * (TABLE_FRAMES - n)

    def shard_offset(self, frame: int, layer: int) -> int:
        return self.frame_base(frame) + self.slot_bytes(layer)

    def tensor_shape(self, frames_or_users: int) -> tuple[int, int, int, int]:
        """Per-chip shape of the tensor: a pool of ``frames`` frames, or a slot-major cache of ``users`` users."""
        if self.slot_stride == 1:
            return (frames_or_users * self.kind.layers, 1, FRAME_ROWS, self.kind.width)
        return (frames_or_users * self.kind.layers, 1, self.slot_stride * FRAME_ROWS, self.kind.width)

    def bytes_per_bank(self, frames_or_users: int) -> int:
        shape = self.tensor_shape(frames_or_users)
        return shape[0] * (shape[2] // FRAME_ROWS) * self.kind.shard_bytes


# ---------------------------------------------------------------------------------------------------------------
# The stripe: position <-> (chip, stripe row)
# ---------------------------------------------------------------------------------------------------------------
def chip_of(position: int) -> int:
    return (position // STRIPE_BLOCK) % CHIPS


def stripe_row(position: int) -> int:
    return (position // PERIOD) * STRIPE_BLOCK + position % STRIPE_BLOCK


def position_of(chip: int, row: int) -> int:
    return ((row // STRIPE_BLOCK) * CHIPS + chip) * STRIPE_BLOCK + row % STRIPE_BLOCK


def frames_for(tokens: int) -> int:
    """Frames a conversation of ``tokens`` positions needs."""
    return -(-tokens // FRAME_TOKENS)


def visible_rows(context: int, chip: int) -> int:
    """Stripe rows of ``chip`` that hold one of the positions [0, context): a prefix of the stripe."""
    in_block = min(max(context % PERIOD - chip * STRIPE_BLOCK, 0), STRIPE_BLOCK)
    return context // PERIOD * STRIPE_BLOCK + in_block


# ---------------------------------------------------------------------------------------------------------------
# THE function (kernels/kv_frames.hpp kv_frames::locate)
# ---------------------------------------------------------------------------------------------------------------
def locate(table, slot_bytes: int, row: int, row_stride: int) -> tuple[int, int]:
    """(DRAM bank, byte offset inside the tensor's allocation in that bank) of stripe row ``row`` of the
    conversation whose frame bases are ``table``, for the layer whose slot offset is ``slot_bytes``. With
    ``row_stride`` 0: of the 32-row block that holds the row (index keys)."""
    base = table[row >> 8]
    assert base != UNMAPPED, f"stripe row {row}: frame {row >> 8} is not mapped"
    return (row >> 5) & 7, (base + slot_bytes + (row & 31) * row_stride) & 0xFFFFFFFF


def locate_position(table, slot_bytes: int, position: int, row_stride: int) -> tuple[int, int, int]:
    """(chip, bank, offset) of a position (what the append does)."""
    return (chip_of(position), *locate(table, slot_bytes, stripe_row(position), row_stride))


def stream_block(table, slot_bytes: int, bank: int, j: int) -> int:
    """Byte offset, in ``bank``, of the j-th 32-row block of that bank's share of a slot (the index-key scan's
    stream: block j of bank b is stripe rows [(8 j + b) * 32, +32), frame j)."""
    return locate(table, slot_bytes, ((j * BANKS + bank) * SHARD_ROWS), 0)[1]


# ---------------------------------------------------------------------------------------------------------------
# Host views of a tensor's bytes (tests): the bank-major byte image <-> (frame, layer) shards
# ---------------------------------------------------------------------------------------------------------------
def slot_of_shard(geometry: Geometry, shard_index: int, frames_or_users: int) -> tuple[int, int]:
    """(tensor slot, shard of the slot in this bank) of a bank's ``shard_index``-th shard: the tensor is
    slot-major and a slot has ``rows / 256`` shards per bank."""
    per_slot = geometry.tensor_shape(frames_or_users)[2] // FRAME_ROWS
    return shard_index // per_slot, shard_index % per_slot
