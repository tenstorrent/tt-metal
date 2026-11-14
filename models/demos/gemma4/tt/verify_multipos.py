# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""``GEMMA4_SPEC_VERIFY_MULTIPOS``: the serving verify's global layers on the causal multi-position paged decode SDPA
(``spec_multi_pos_tiles``) -- the device-free part (pure torch, no ttnn import, so the host tests load it alone).

The packed verify (``attention/decode.py: packed_decode_forward``) packs the ``H_local * P`` query rows of a record row
into one tile and runs a NON-causal paged SDPA over a page table as wide as the record's rung, with an additive mask
``[B, 1, H_local * P, S_k]`` that hides keys past each row's position. On a global layer every replay therefore reads
the rung's whole key span (and the S_k-wide mask), wherever the row really is: the verify cost follows the rung, not
the position. Under this knob a global layer instead runs the stock op CAUSAL with ``spec_multi_pos_tiles = Tg``:

* Q ``[1, B * G, Tg * 32, D]`` (``G = P / Tg`` groups per record row): candidate ``p = g * Tg + j`` of row ``b`` is row
  tile ``j`` of batch group ``b * G + g``, its ``H_local`` q heads in rows ``0 .. H_local - 1`` of that tile and zeros
  below (``multipos_q_layout`` emulates the device re-layout; ``multipos_out_rows`` the un-pad);
* ``cur_pos [B * P]`` int32, entry ``b * P + p = c_b + p`` (``multipos_positions``; ascending within every group by
  construction) -- candidate ``b * P + p`` attends keys ``[0, c_b + p]``;
* the page table of the layer with one row per group (``multipos_page_table``: row ``b * G + g`` = row ``b``);
* no mask: the op reads each group's KV once, up to the group's last position, and masks the tail per row tile.

Mask equivalence. A global layer has no window, no ring and no ``cache_position_modulo`` (checked per layer by
``multipos_layer_reason``), and the record's full-attention mask row ``h * P + p`` of row ``b`` is ``-1e9`` at key
``j`` iff ``j > c_b + p`` (``spec_decode._srv_masks_device``: ``gtz(iota - v_pos_rows)``), over ``j < S_k`` read
through column ``j // block`` of the same row's table. So the keys a candidate sees are exactly ``[0, c_b + p]``, the
causal bound; the candidates' own K / V are written (``packed_decode_forward`` step 5) before the attention in both
paths, and ``serving_width_for`` keeps ``c_b + P - 1 < S_k`` so every visible key sits inside the table.

``plan_verify_multipos`` picks ``Tg`` and the program config from the op's own circular-buffer arithmetic (every
Q-shaped CB scales with ``Tg`` row tiles, so at head_dim 512 the four candidates of a K=3 row in ONE group leave room
for two to four cores per group; two groups of two fit eleven, but read the row's KV twice) and an estimate of the
compute against the KV stream (``multipos_cost_ns``). The masked verify keeps all ``H_local * P`` rows of a record row
in ONE tile; here each candidate takes a tile of its own, so per key the multi-position call does ``P`` times the
matmul work of the packed one: it wins by reading only the keys up to the row, not the rung.
"""

from __future__ import annotations

import dataclasses
import os
from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from loguru import logger

VERIFY_MULTIPOS_ENV = "GEMMA4_SPEC_VERIFY_MULTIPOS"
#: pins of the planner (unset: derived): candidates per group, the K chunk (tokens), the cores-per-head cap
VERIFY_MULTIPOS_TG_ENV = "GEMMA4_SPEC_VERIFY_MULTIPOS_TG"
VERIFY_MULTIPOS_K_CHUNK_ENV = "GEMMA4_SPEC_VERIFY_MULTIPOS_K_CHUNK"
VERIFY_MULTIPOS_MAX_CORES_ENV = "GEMMA4_SPEC_VERIFY_MULTIPOS_MAX_CORES"

TILE = 32
BF16_TILE_BYTES = 2048
#: the op's dst tiles with its default compute config (HiFi2, fp32 dest accumulation off); the fixed-chunk granularity
#: defines are min(., DST_TILES) and must be powers of two
DST_TILES = 8
#: sdpa_decode_program_factory MAX_TREE_REDUCTION_ROUNDS: at most 2**6 cores per (batch group, KV head)
MAX_TREE_ROUNDS = 6
#: fixed K chunks (tokens) the planner tries per group size, in order; the in-kernel dynamic chunk (0) sizes its CBs at
#: four tiles in spec mode, the L1 of the 128-token chunk, so it is only taken when pinned
K_CHUNK_LADDER = (64, 32, 128)
DEFAULT_MAX_CORES = 32
#: The planner's cost estimate per 32-key tile of a layer (``multipos_cost_ns``), the larger of the compute on a
#: group's slowest core and the KV stream of every group from DRAM. Estimates, not measurements: a 32x32x32 bf16 tile
#: matmul at HiFi2 ~32 cycles at 1.35 GHz; the DRAM rate is what the 31B Galaxy (5,3) records' masked verify reads
#: per key (+21 ms over +98K keys of 10 global layers x 5 rows, bf16 K/V: ~0.13 us per row and key tile).
TILE_MATMUL_NS = 24.0
DRAM_BYTES_PER_NS = 450.0

_LOGGED = set()


def log_once(message: str):
    """One INFO line per distinct message."""
    if message not in _LOGGED:
        _LOGGED.add(message)
        logger.info(f"{VERIFY_MULTIPOS_ENV}: {message}")


def verify_multipos_enabled() -> bool:
    """The knob: ``GEMMA4_SPEC_VERIFY_MULTIPOS=1`` (default off)."""
    return os.environ.get(VERIFY_MULTIPOS_ENV, "0") == "1"


def multipos_layer_reason(
    *,
    is_sliding: bool,
    cache_position_modulo,
    kv_modulo,
    nkv_local: int,
    h_local: int,
    head_dim: int,
    cache_shape: Sequence[int],
) -> Optional[str]:
    """None when a layer's packed verify can take the multi-position SDPA, else why not. Static arguments only (the
    layer config, its cache's padded shape), so warmup, capture and replay decide alike: a global layer (no window),
    no ring (``cache_position_modulo`` / the verify's ``kv_modulo``: causal bounds are absolute positions), one local
    KV head, at most a tile of local q heads per candidate, and a cache allocated in the layer's own view
    ``[blocks, 1, block, head_dim]`` with a whole-tile block (the op checks the cache's own head count, not a
    geometry override)."""
    if is_sliding:
        return "sliding layer"
    if cache_position_modulo is not None or kv_modulo is not None:
        return f"ring cache (cache_position_modulo {cache_position_modulo}, verify modulo {kv_modulo})"
    if int(nkv_local) != 1:
        return f"{nkv_local} local KV heads (the op takes one)"
    if not 1 <= int(h_local) <= TILE:
        return f"{h_local} local q heads do not fit one {TILE}-row tile per candidate"
    shape = tuple(int(s) for s in cache_shape)
    if len(shape) != 4 or shape[1] != 1 or shape[3] != int(head_dim):
        return f"cache {shape} is not this layer's view [blocks, 1, block, {head_dim}] (a shared hybrid-group buffer)"
    if shape[2] % TILE:
        return f"block size {shape[2]} is not a multiple of {TILE}"
    return None


def multipos_positions(positions: Sequence[int], packed_p: int) -> torch.Tensor:
    """``[B * P]`` int32 cur_pos of one replay: entry ``b * P + p = c_b + p`` (``positions`` = the record's padded row
    positions ``c_b``, padding rows mirroring row 0, so no row is ever skipped with -1). The op's reader takes a
    group's KV span from its LAST entry and skips chunks before its FIRST, so each row's entries must be ascending:
    they are by construction, and checked here, at the upload (host, outside the trace)."""
    P = int(packed_p)
    cs = [int(c) for c in positions]
    if P < 1 or not cs:
        raise ValueError(f"multi-position verify needs P >= 1 and at least one row, got P={P}, {len(cs)} rows")
    if min(cs) < 0:
        raise ValueError(f"negative row position in {cs}: the records mirror row 0 for padding rows, never -1")
    if max(cs) + P - 1 >= 2**31:
        raise ValueError(f"position {max(cs) + P - 1} overflows the int32 cur_pos")
    out = torch.tensor([c + p for c in cs for p in range(P)], dtype=torch.int32)
    if not multipos_groups_ascending(out, P):
        raise ValueError(f"cur_pos {out.tolist()} is not ascending within each row of {P}")
    return out


def multipos_groups_ascending(cur_pos, tg: int) -> bool:
    """Whether every group of ``tg`` consecutive entries of ``cur_pos`` is non-decreasing (the op's contract)."""
    t = torch.as_tensor(cur_pos).reshape(-1, int(tg)).to(torch.int64)
    return bool((t[:, 1:] >= t[:, :-1]).all())


def multipos_page_table(table, groups: int) -> torch.Tensor:
    """``[B * G, cols]``: row ``b * G + g`` = row ``b`` of the record's ``[B, cols]`` verify table (one page-table row
    per batch group of the op; ``G = 1`` returns the table itself)."""
    t = torch.as_tensor(table)
    if t.dim() != 2:
        raise ValueError(f"page table must be [rows, blocks], got shape {tuple(t.shape)}")
    return t if int(groups) == 1 else t.repeat_interleave(int(groups), dim=0)


def multipos_q_layout(q: torch.Tensor, packed_p: int, tg: int) -> torch.Tensor:
    """Torch emulation of the device Q re-layout (``decode._multipos_verify_rows``), op for op:
    ``q [1, H, B * P, D]`` (the prefill-style split after RoPE, row ``b * P + p``) -> permute ``(0, 2, 1, 3)``
    ``[1, B * P, H, D]`` -> zero rows ``H .. 31`` -> reshape ``[1, B * G, Tg * 32, D]``."""
    _one, H, rows, D = (int(s) for s in q.shape)
    P, tg = int(packed_p), int(tg)
    if rows % P or P % tg or H > TILE:
        raise ValueError(f"q {tuple(q.shape)}: rows not a multiple of P={P}, P not of Tg={tg}, or H > {TILE}")
    c = q.permute(0, 2, 1, 3)
    c = F.pad(c, (0, 0, 0, TILE - H))
    return c.reshape(1, rows // tg, tg * TILE, D)


def multipos_out_rows(out: torch.Tensor, h_local: int) -> torch.Tensor:
    """Torch emulation of the device un-pad: the op output ``[1, B * G, Tg * 32, D]`` -> reshape
    ``[1, B * P, 32, D]`` -> rows ``0 .. H_local - 1`` = ``[1, B * P, H_local, D]``, the layout
    ``packed_decode_forward`` hands to the o_proj tail."""
    _one, groups, rows, D = (int(s) for s in out.shape)
    return out.reshape(1, groups * rows // TILE, TILE, D)[:, :, : int(h_local), :]


def spec_sdpa_tree_rounds(cores_per_head: int) -> int:
    """The factory's tree-reduction rounds: ceil(log2(cores)) for more than one core."""
    c = int(cores_per_head)
    return 0 if c <= 1 else (c - 1).bit_length()


def spec_sdpa_cb_tiles(tg: int, dht: int, sk_tiles: int, cores_per_head: int) -> int:
    """The spec-mode CB total on one core in 32x32 tiles (``sdpa_decode_program_factory``, ``PNHt = Tg``, ``vDHt = DHt``,
    the K / V chunks counted at full tiles): q (Tg x DHt), a one-tile tilize staging, K and V double-buffered
    (4 x Sk x DHt), mask and QK scores (2 x Tg x Sk), three out intermediates + out stats + out (5 x Tg x DHt), eleven
    statistics CBs (Tg each), three scalars, and the tree-reduction staging sized by ROUNDS in spec mode
    ((Tg x DHt + 2 Tg) x min(cores - 1, rounds))."""
    tg, dht, sk, cph = int(tg), int(dht), int(sk_tiles), int(cores_per_head)
    blocks = min(cph - 1, spec_sdpa_tree_rounds(cph))
    return tg * dht + 1 + 4 * sk * dht + 2 * tg * sk + 5 * tg * dht + 11 * tg + 3 + (tg * dht + 2 * tg) * blocks


def spec_sdpa_cb_bytes(tg, dht, sk_tiles, cores_per_head, kv_tile_bytes=BF16_TILE_BYTES, extra_bytes=0) -> int:
    """The CB total in bytes: every tile bf16 but the K / V chunks in the cache format, plus the untiled CBs
    (``extra_bytes``: the page-table row and the two cur_pos copies)."""
    tiles = spec_sdpa_cb_tiles(tg, dht, sk_tiles, cores_per_head)
    kv = 4 * int(sk_tiles) * int(dht)
    return (tiles - kv) * BF16_TILE_BYTES + kv * int(kv_tile_bytes) + int(extra_bytes)


def spec_sdpa_cores_per_head(grid_cores: int, groups: int, max_cores_per_head_batch: int) -> int:
    """The factory's cores per batch group (one KV head): ``min(grid, max x groups) // groups``, at least 1."""
    return max(1, min(int(grid_cores), int(max_cores_per_head_batch) * int(groups)) // max(1, int(groups)))


def multipos_extra_bytes(table_cols: int, cur_pos_entries: int, alignment: int = 64) -> int:
    """The untiled CBs of a call: one int32 page-table row and two copies of the int32 cur_pos row, DRAM-aligned."""

    def aligned(n):
        return -(-(int(n) * 4) // alignment) * alignment

    return aligned(table_cols) + 2 * aligned(cur_pos_entries)


def multipos_cost_ns(tg: int, sdpa_batch: int, cores_per_head: int, dht: int, kv_tile_bytes: int) -> float:
    """Estimated ns per 32-key tile of one layer: ``max(compute, dram)``. Compute: a group's ``Tg`` row tiles take
    ``2 x DHt`` tile matmuls per key tile (QK and PV) spread over its cores; DRAM: every group streams the row's K and
    V (``2 x DHt`` tiles per key tile), so a row split into G groups reads its KV G times."""
    compute = int(tg) * 2 * int(dht) * TILE_MATMUL_NS / max(1, int(cores_per_head))
    dram = int(sdpa_batch) * 2 * int(dht) * int(kv_tile_bytes) / DRAM_BYTES_PER_NS
    return max(compute, dram)


def _pow2(n: int) -> bool:
    return n >= 1 and (n & (n - 1)) == 0


@dataclasses.dataclass(frozen=True)
class MultiposPlan:
    """How one record's global layers run the multi-position SDPA: ``groups`` batch groups of ``tg`` candidates per
    record row (``batch * groups`` page-table rows), the fixed K chunk, the cores-per-head cap and what it gives, on
    the ``grid``, and the CB footprint against the budget."""

    batch: int
    packed_p: int
    tg: int
    groups: int
    k_chunk: int
    max_cores_per_head_batch: int
    cores_per_head: int
    grid: tuple
    head_dim: int
    h_local: int
    kv_tile_bytes: int
    extra_bytes: int
    cb_bytes: int
    budget_bytes: int
    fits: bool
    cost_ns: float = 0.0  # multipos_cost_ns per key tile of a layer
    note: str = ""

    @property
    def sdpa_batch(self) -> int:
        return self.batch * self.groups

    def describe(self) -> str:
        head = (
            f"{self.batch} rows x {self.packed_p} candidates as {self.sdpa_batch} groups of {self.tg} row tiles "
            f"({self.groups} per row), H_local {self.h_local}, head_dim {self.head_dim}, K/V tile {self.kv_tile_bytes} "
            f"B on a {self.grid[0]}x{self.grid[1]} grid"
        )
        if not self.fits:
            return head + f": no config fits ({self.note})"
        return head + (
            f": k_chunk {self.k_chunk}, max_cores_per_head_batch {self.max_cores_per_head_batch} "
            f"({self.cores_per_head} cores per group, {self.sdpa_batch * self.cores_per_head} active), CBs "
            f"{self.cb_bytes:,} of {self.budget_bytes:,} B, ~{self.cost_ns:.0f} ns per key tile"
            + (f" ({self.note})" if self.note else "")
        )


def _env_int(env, name):
    raw = str(env.get(name, "") or "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from None


def plan_verify_multipos(
    batch: int,
    packed_p: int,
    head_dim: int,
    h_local: int,
    grid,
    budget_bytes: int,
    kv_tile_bytes: int = BF16_TILE_BYTES,
    extra_bytes: int = 0,
    env=None,
) -> MultiposPlan:
    """The multi-position SDPA config for ``batch`` record rows of ``packed_p`` candidates at ``head_dim`` on ``grid``
    ``(x, y)``, against ``budget_bytes`` of CB space per core (``extra_bytes``: the untiled CBs,
    ``multipos_extra_bytes``).

    Per group size ``Tg`` (the divisors of P) the most cores per group that fit: K chunks from ``K_CHUNK_LADDER``
    that are a power of two in tiles with ``min(Tg x chunk tiles, 8)`` a power of two (the op's fixed-chunk
    granularity rules), the cores cap walked down from ``GEMMA4_SPEC_VERIFY_MULTIPOS_MAX_CORES`` (default 32) until
    the CBs fit, the ladder order breaking ties. Of those the lowest ``multipos_cost_ns`` wins, the larger group on a
    tie: one group per row reads the KV once but its ``Tg`` row tiles leave few cores the L1 for them (at head_dim 512
    four candidates fit two to four cores), more groups buy cores with KV re-reads. ``GEMMA4_SPEC_VERIFY_MULTIPOS_TG``
    / ``_K_CHUNK`` pin the group size / chunk (``0`` = the in-kernel dynamic chunk). Returns a plan with
    ``fits=False`` and the reason when nothing fits."""
    env = os.environ if env is None else env
    B, P, D, H = int(batch), int(packed_p), int(head_dim), int(h_local)
    gx, gy = (int(v) for v in grid)
    n = gx * gy
    budget = int(budget_bytes)
    if min(B, P, D, H, n) < 1 or D % TILE:
        raise ValueError(f"plan_verify_multipos needs positive sizes and a head_dim multiple of {TILE}")
    dht = D // TILE
    tg_pin = _env_int(env, VERIFY_MULTIPOS_TG_ENV)
    k_pin = _env_int(env, VERIFY_MULTIPOS_K_CHUNK_ENV)
    cores_pin = _env_int(env, VERIFY_MULTIPOS_MAX_CORES_ENV)
    cap_top = min(cores_pin if cores_pin else DEFAULT_MAX_CORES, 2**MAX_TREE_ROUNDS)

    shared = dict(batch=B, packed_p=P, grid=(gx, gy), head_dim=D, h_local=H, budget_bytes=budget)
    shared.update(kv_tile_bytes=int(kv_tile_bytes), extra_bytes=int(extra_bytes))

    def plan(tg, k=0, cap=0, cph=0, cb=0, fits=False, note=""):
        cost = multipos_cost_ns(tg, B * (P // tg), cph, dht, kv_tile_bytes) if fits else 0.0
        return MultiposPlan(
            tg=tg,
            groups=P // tg,
            k_chunk=k,
            max_cores_per_head_batch=cap,
            cores_per_head=cph,
            cb_bytes=cb,
            **shared,
            fits=fits,
            cost_ns=cost,
            note=note,
        )

    if tg_pin is not None and (tg_pin < 1 or P % tg_pin):
        return plan(max(1, tg_pin), note=f"{VERIFY_MULTIPOS_TG_ENV}={tg_pin} does not divide P={P}")
    tgs = [tg_pin] if tg_pin else [d for d in range(P, 0, -1) if P % d == 0]
    ladder = (k_pin,) if k_pin is not None else K_CHUNK_LADDER
    chosen, refused = None, []
    for tg in tgs:
        groups = B * (P // tg)
        if groups > n:
            refused.append(f"Tg {tg}: {groups} groups for {n} cores")
            continue
        best = None  # (cores per group, k, cap, cb) of the first ladder chunk with the most cores
        for k in ladder:
            if k < 0 or k % TILE:
                continue
            sk = k // TILE
            if sk:  # a fixed chunk: power-of-two tiles, and the granularity defines
                if not _pow2(sk) or not _pow2(min(tg * sk, DST_TILES)):
                    continue
                sk_cb = sk
            else:  # the dynamic chunk sizes its CBs at min(dst, 4) tiles in spec mode
                sk_cb = min(DST_TILES, 4)
            for cap in range(cap_top, 0, -1):
                cph = spec_sdpa_cores_per_head(n, groups, cap)
                if spec_sdpa_tree_rounds(cph) > MAX_TREE_ROUNDS:
                    continue
                cb = spec_sdpa_cb_bytes(tg, dht, sk_cb, cph, kv_tile_bytes, extra_bytes)
                if cb <= budget:
                    if best is None or cph > best[0]:
                        best = (cph, k, cap, cb)
                    break
        if best is None:
            refused.append(f"Tg {tg}: no K chunk / cores pair fits the CB budget")
            continue
        cph, k, cap, cb = best
        candidate = plan(tg, k, cap, cph, cb, True)
        if chosen is None or candidate.cost_ns < chosen.cost_ns:
            chosen = candidate
    if chosen is None:
        return plan(tgs[-1], note="; ".join(refused))
    return chosen


def multipos_reference_attention(q_mp, k_paged, v_paged, page_table, cur_pos, tg, block_size, scale=1.0):
    """fp32 torch semantics of the op call: ``q_mp [1, B * G, Tg * 32, D]``, paged caches ``[blocks, 1, block, D]``,
    ``page_table [B * G, cols]``, ``cur_pos [B * G * Tg]``: row tile ``j`` of group ``q`` attends keys
    ``[0, cur_pos[q * Tg + j]]`` read through row ``q`` of the table. Returns ``[1, B * G, Tg * 32, D]``."""
    _one, groups, rows, D = (int(s) for s in q_mp.shape)
    tg, bs = int(tg), int(block_size)
    out = torch.zeros(1, groups, rows, D, dtype=torch.float32)
    cur = torch.as_tensor(cur_pos).to(torch.int64).reshape(-1)
    for g in range(groups):
        for j in range(tg):
            pos = int(cur[g * tg + j])
            keys = torch.arange(pos + 1)
            blocks = torch.as_tensor(page_table)[g, keys // bs].to(torch.int64)
            k = k_paged[blocks, 0, keys % bs].float()
            v = v_paged[blocks, 0, keys % bs].float()
            q = q_mp[0, g, j * TILE : (j + 1) * TILE].float()
            p = torch.softmax((q @ k.T) * scale, dim=-1)
            out[0, g, j * TILE : (j + 1) * TILE] = p @ v
    return out
