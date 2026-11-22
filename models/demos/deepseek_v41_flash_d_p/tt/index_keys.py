# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""INDEX_K_SOURCE=served (P.76, owner 2026-11-18, item 3 of the slot-state page): the prefill indexer scores ONLY the
served index keys, Blaze decode's 192-byte rows (INDEX_K_SERVED=fp8_row192: e4m3 x 128 | fp32 scale | zeros), never a
private copy. A private copy (P.41's k8) is right only when this process wrote every earlier chunk of the slot; after
a store load, a d2p pull or a reused slot it held zeros or another user's keys. The served rows are what the engine
holds, whoever wrote them.

Each index-key cache (tt/compressor.TtServedRows, SP-sharded and TP-replicated: every chip holds its SP row's served
rows in the same local rows, i.e. the same block-cyclic order, as the ring score's k_local) decodes its served rows
with ttnn.bringup.unpack_fp8_rows (e4m3 x scale in fp32, rounded to nearest even onto bfloat8_b: bfp8(fp8(key))) into
a persistent bfloat8_b TILE buffer [1, 1, crows, 128] a chip, which the ring score (ttnn.bringup.
ring_indexer_score_dsa) reads exactly as it read k8. The buffers are not per user: one per (rows, source layer),
shared by every slot / state of the model (layer None: one per size, shared by every source of that size), made at
the state's build (``acquire``), freed with the last cache that holds it.

Per chunk at start (c = start // S, q = start % S, wr = W / r local rows a chip a W block): the chunk's keys sit in W
blocks c (and c + 1 when q != 0), local rows [c wr, hi), hi = (c + 1 + (q > 0)) wr. ``keys`` decodes [0, hi) (rows
past hi are later positions, causally masked, never read as keys of a visible position) unless ``prefetch`` already
decoded the history [0, c wr) in this chunk (TtDS41Model.prefetch_history at the first engram wait, as P.75's
history gather: the rows are complete before the chunk), when it decodes only [c wr, hi): the same bytes either way
(each row decodes on its own; a bfp8 group is 16 values of one row). A buffer records what it holds: (cache serial,
start, cache version, the model's chunk epoch); the epoch moves at every begin_chunk, so nothing decoded in an earlier
chunk is ever trusted (the engine may have written the slab between chunks). The re-index layers' memo hit (the same
cache, start and version within a chunk) reads the buffer as it is.
"""

from __future__ import annotations

import itertools

import ttnn

from .common import DRAM
from .cores import max_grid
from .settings import settings

_SERIAL = itertools.count(1)
_EPOCH = [0]
STATS = {"full": 0, "tail": 0, "prefetch": 0, "hit": 0}


def new_serial() -> int:
    return next(_SERIAL)


def new_chunk() -> None:
    """Every decoded row is stale from here on (tt/model.TtDS41Model.begin_chunk)."""
    _EPOCH[0] += 1


class KeyBuffer:
    """[1, 1, rows, 128] bfloat8_b TILE DRAM interleaved, replicated layout (each chip its own rows); ``users``: the
    caches holding it."""

    def __init__(self, mesh, rows: int, width: int):
        self.users = 0
        self.tensor = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, rows, width]), ttnn.bfloat8_b, ttnn.TILE_LAYOUT, mesh, DRAM
        )
        self.holds = None  # (serial, start, version, epoch, rows decoded [0, n))

    def free(self) -> None:
        if self.tensor.is_allocated():
            ttnn.deallocate(self.tensor)


_BUFFERS = {}  # (mesh id, rows, width, layer) -> KeyBuffer


def _key(cache) -> tuple:
    return (cache.mesh.id(), cache.crows, cache.width, cache.key_layer)


def acquire(cache) -> None:
    k = _key(cache)
    buf = _BUFFERS.get(k)
    if buf is None:
        buf = _BUFFERS[k] = KeyBuffer(cache.mesh, cache.crows, cache.width)
    buf.users += 1


def release(cache) -> None:
    k = _key(cache)
    buf = _BUFFERS.get(k)
    if buf is None:
        return
    buf.users -= 1
    if buf.users <= 0:
        buf.free()
        del _BUFFERS[k]


def _span(cache, start: int) -> tuple[int, int]:
    """(c wr, hi): the history rows complete before the chunk, and the end of the chunk's W blocks (local rows)."""
    g = cache.geo
    wr = g.w // cache.ratio
    c, q = divmod(int(start), g.chunk)
    return c * wr, min((c + 1 + (q > 0)) * wr, cache.crows)


# PACKED_HOST_FAST=on (PB.3K): the decodes on this core rectangle. unpack_fp8_rows' host cost is its per-core value
# slots (5 a core, 3 kernels: ~250-340 us a call on 110 cores of 32 chips, PB.3K census); its device time is small
# (a 192 B -> 128 B row map, ~4352 rows a chip at 139k). Each row decodes on its own: the same bytes on any cores.
DECODE_GRID = (4, 2)
# PACKED_LONG_HIDE=on (PB.6L): a decode of at least this many rows a chip runs on the whole compute grid (max_grid):
# at 1M a history decode is 65,024 rows a chip, 3.95 ms on 4 x 2 cores vs ~0.8 on 11 x 10 (PB.6L bench); at 139k
# (8,704 rows) the small rectangle's lower host cost wins
DECODE_WIDE_ROWS = 16384


def _decode(cache, buf: KeyBuffer, lo: int, hi: int) -> None:
    s = cache.served
    from .page_copy import dyn

    grid = max_grid()
    # MB.2 ("decode": POS_DYNAMIC=on): every range on the whole grid (cores past the band count idle): ONE program, not
    # one a band count. PB.MB: not PB.6L's small rectangle below DECODE_WIDE_ROWS either: that grid choice depends on
    # the history length, so the wide grid's program was first prepared on a long fresh history after warm-up
    # (test_no_rebuild 8192, resumed at 525k: 1 build, DS41-350)
    whole = dyn("decode")
    wide = settings.get("PACKED_LONG_HIDE") == "on" and hi - lo >= DECODE_WIDE_ROWS
    if settings.get("PACKED_HOST_FAST") == "on" and not wide and not whole:  # (every box's grid holds 4 x 4)
        gx, gy = (grid.x, grid.y) if grid is not None else DECODE_GRID
        grid = ttnn.CoreCoord(min(gx, DECODE_GRID[0]), min(gy, DECODE_GRID[1]))
    # PB.9X (COMMON_VALUE_SLOTS=on): the whole grid's per-run values as common runtime args (one patch a program,
    # not one a core: 0.335 -> 0.063 ms of host a call on the 8x4, 24 calls a packed chunk); the same bytes
    cv = whole and settings.get("COMMON_VALUE_SLOTS") == "on" and _has_common_values()
    ttnn.bringup.unpack_fp8_rows(
        s.tensor,
        buf.tensor,
        row_lo=lo,
        row_hi=hi,
        batch=s.batch,
        max_grid=grid,
        whole_grid=whole,
        **({"common_values": True} if cv else {}),
    )


def _has_common_values() -> bool:
    """The fp8_rows fork's common_values option (and prepare_program's common value slots) is built (its docstring
    names it: the registered op's wrapper hides the signature)."""
    return "common_values" in (getattr(ttnn.bringup.unpack_fp8_rows, "__doc__", "") or "")


def prefetch(cache, start: int) -> bool:
    """The history [0, c wr) of ``cache`` (complete before the chunk at ``start``) decoded into its buffer. False:
    nothing before the chunk (c = 0)."""
    buf = _BUFFERS[_key(cache)]
    lo, _ = _span(cache, start)
    buf.holds = None
    if lo == 0:
        return False
    _decode(cache, buf, 0, lo)
    buf.holds = (cache.serial, int(start), cache.version, _EPOCH[0], lo)
    STATS["prefetch"] += 1
    return True


def keys(cache, start: int) -> ttnn.Tensor:
    """The buffer holding ``cache``'s served keys of local rows [0, hi) decoded (the ring score's k_local; never
    freed by the caller). Called after the chunk's served write (the compressor runs first)."""
    buf = _BUFFERS[_key(cache)]
    lo, hi = _span(cache, start)
    h = buf.holds
    here = h is not None and h[0] == cache.serial and h[1] == int(start) and h[3] == _EPOCH[0]
    if here and h[2] == cache.version and h[4] >= hi:  # (decoded whole in this chunk at this version: a memo reader)
        STATS["hit"] += 1
        return buf.tensor
    if here and h[4] == lo and cache.version in (h[2], h[2] + 1):  # the history prefetched; the chunk wrote once since
        _decode(cache, buf, lo, hi)
        STATS["tail"] += 1
    else:
        _decode(cache, buf, 0, hi)
        STATS["full"] += 1
    buf.holds = (cache.serial, int(start), cache.version, _EPOCH[0], hi)
    return buf.tensor
