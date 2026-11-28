# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""``GEMMA4_SPEC_VERIFY_PREFIX_LSE``: the serving verify's global layers as a causal decode over each row's prefix plus
the candidates' own keys, merged through their log-sum-exp -- the device-free part (pure torch, no ttnn import, so
the host tests load it alone).

The packed verify (``attention/decode.py: packed_decode_forward``) packs the ``H * P`` query rows of a record row
(head-major, row ``h * P + p``) into one tile and runs a NON-causal paged SDPA over the whole rung with an additive
mask as wide as the rung, so every replay reads the rung's whole KV span wherever the row really is. Row ``b``'s
candidates sit at ``c_b + p`` (``p = 0 .. P - 1``) and candidate ``p`` attends keys ``[0, c_b + p]``. Under this knob a
qualifying global layer splits those keys in two:

* prefix ``[0, c_b - 1]``: every candidate of the row sees all of it, so it is ONE causal paged decode at
  ``cur_pos = c_b - 1`` for all ``H * P`` packed rows of the row (``SdpaDecodeLse``: the stock decode kernels plus each
  row's softmax statistics m, l), reading the row's KV once and only up to the row. The op takes at most 32 q rows,
  so ``H * P > 32`` (8 local heads at P = 6) runs ``prefix_head_splits`` calls over head subsets;
* fresh ``c_b .. c_b + P - 1``: the candidates' own K / V, computed in this verify and taken before the KV write
  (``fresh_kv_rows``: per record row, keys padded to one 32-row tile with zeros, rounded through the cache's format so
  they equal the bytes the cache holds), ``scores = Q K^T * scale`` plus a CONSTANT causal mask (``fresh_causal_mask``:
  packed row ``(h, p)`` sees fresh key ``q`` iff ``q <= p``; the padded keys ``q >= P`` are always hidden), its own
  softmax statistics ``m_2 = max``, ``l_2 = sum exp(scores - m_2)``, and the unnormalised ``exp(.) @ V``;
* the merge per packed row in fp32 (``merge_prefix_fresh_torch`` is the device sequence in torch):
  ``lse_1 = scale * m_1 + ln l_1 + bias_b``, ``lse_2 = m_2 + ln l_2``, ``d = clamp(lse_1 - lse_2, +-LOGIT_CLAMP)``,
  ``O = sigmoid(d) * O_1 + sigmoid(-d) / l_2 * (exp(.) @ V)``.

Both parts are exact pieces of the masked row: the keys of a candidate are ``[0, c_b + p]``, the union of the prefix
and the fresh keys ``q <= p``, the masked verify writes the candidates' K / V before its attention as well, and the
global layers have no window, ring or ``cache_position_modulo`` (``prefix_lse_layer_reason``).

A row at ``c_b = 0`` has no prefix. Its prefix call runs at ``cur_pos = 0`` (``prefix_positions`` clamps: the op skips a
row only at -1 and then leaves its outputs untouched, i.e. garbage that may be NaN), so the outputs are always written
and finite, and the row's bias ``EMPTY_PREFIX_LSE`` (``prefix_bias``) puts its prefix lse ~3e4 below the fresh one: a
merge weight of sigmoid(-64) ~ 1.6e-28. The served records never verify a row at position 0 (a prompt has a token);
the eager and capture bodies run on zero inputs, where this keeps every value finite.

Per replay the record uploads ``[B]`` int32 prefix positions and the ``[1, B, R, 1]`` fp32 bias (``R = H_s * P``, the
q rows of one prefix call); the fresh mask is a constant of the shape, built before any capture. With every global
layer on this path the record builds no full-attention mask.

Pair split (a ``GEMMA4_SDPA_PAIR_SPLIT`` build: the global layers carry ``weights.pair``;
``GEMMA4_SPEC_VERIFY_PREFIX_LSE_PAIR=0`` keeps the 4-head verify). Each chip of a KV-head pair (mesh columns ``c`` and
``c ^ 1``) runs the verify on the pair view: the QKV of the KV group's ``2 * H`` q heads (the same Q, K and V on both
chips; K / V are the 4-head layout's, so the KV write is unchanged) and the group's o_proj rows. The prefix call runs
``pair_split=True``: chip ``c`` attends half ``c % 2`` of the row's prefix chunks (``pair_half_keys``; an empty half
writes O = 0, m = EMPTY_HALF_M, l = 1). The fresh part runs on both chips with identical values. Per packed row a
three-way merge (``pair_prefix_merge_torch``): the own half's lse, the partner's (the row all-gather as in
``decode.pair_merge``, times ``relu(-selector) = e_{c^1}`` and summed over the device dim: exactly the partner's value)
and the fresh lse, ``top`` the max of the three, ``a_i = exp(lse_i - top)``, ``S = (a_self + a_partner) + a_fresh``;
chip ``c`` feeds ``a_self / S * O_self + 0.5 * a_fresh / (S * l_2) * (exp(.) @ V)`` to the pair o_proj. The pair's two
products add up in the row all-reduce to ``(w_0 O_0 + w_1 O_1 + w_fresh O_fresh) @ o_proj``, exact: each chip adds half
of the identical fresh term. Under the split a chip reads half of each row's prefix KV.
"""

from __future__ import annotations

import dataclasses
import os
from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from loguru import logger

from models.demos.gemma4.tt.mesh_orientation import VERIFY_PREFIX_LSE_ENV, decode_flag
from models.demos.gemma4.tt.sdpa_pair_split import EMPTY_HALF_M, LOGIT_CLAMP, pair_selector

#: "0" keeps the 4-head verify on a GEMMA4_SDPA_PAIR_SPLIT build (default: the pair view where the layers have it)
VERIFY_PREFIX_LSE_PAIR_ENV = "GEMMA4_SPEC_VERIFY_PREFIX_LSE_PAIR"
#: the smallest record rung (S_k) at which a prefix that takes more than one call (h_local * P > 32 packed q rows: 8
#: pair heads at P = 6, the (1,5) / (5,5) records) replaces the masked verify; one-call records always take it
VERIFY_PREFIX_LSE_SPLIT_MIN_SK_ENV = "GEMMA4_SPEC_VERIFY_PREFIX_LSE_SPLIT_MIN_SK"
VERIFY_PREFIX_LSE_SPLIT_MIN_SK_DEFAULT = 131072
#: pins the prefix call's cores per (row, KV head); unset: GEMMA4_SDPA_MAX_CORES_PER_HEAD, else the stock default 16
VERIFY_PREFIX_LSE_MAX_CORES_ENV = "GEMMA4_SPEC_VERIFY_PREFIX_LSE_MAX_CORES"
#: the decode SDPA's K chunk knob, which the prefix call follows (attention/decode.py decode_forward)
DECODE_K_CHUNK_ENV = "GEMMA4_SDPA_DECODE_K_CHUNK"
DECODE_MAX_CORES_ENV = "GEMMA4_SDPA_MAX_CORES_PER_HEAD"

TILE = 32
#: SdpaDecodeLse's q: [1, B, Hq <= 32, D]
MAX_Q_ROWS = 32
#: SdpaDecodeLse's head dims
HEAD_DIMS = (256, 512)
#: the bias on the prefix lse of a row with no prefix (c_b = 0): finite, so no inf or NaN reaches the merge
EMPTY_PREFIX_LSE = -30000.0
#: the fresh mask's hidden entries (fp32, added to the raw scores)
FRESH_MASK_NEG = -1.0e9
DEFAULT_K_CHUNK = 64
#: ttnn.SDPAProgramConfig's default max_cores_per_head_batch (the plain decode's cap)
DEFAULT_MAX_CORES_PER_HEAD = 16
#: the DRAM page alignment the planner's stick sizes round to (Blackhole; Wormhole's 32 rounds to no more)
DRAM_ALIGNMENT = 64

_LOGGED = set()


def log_once(message: str):
    """One INFO line per distinct message."""
    if message not in _LOGGED:
        _LOGGED.add(message)
        logger.info(f"{VERIFY_PREFIX_LSE_ENV}: {message}")


def verify_prefix_lse_enabled() -> bool:
    """The knob: ``GEMMA4_SPEC_VERIFY_PREFIX_LSE`` through ``decode_flag`` (the env when set, else the build profile:
    on for a pair-qualified dense model on the (1,8) line, ``GALAXY_LINE_PAIR_ON``; off elsewhere)."""
    return decode_flag(VERIFY_PREFIX_LSE_ENV) == "1"


def verify_prefix_pair_enabled() -> bool:
    """``GEMMA4_SPEC_VERIFY_PREFIX_LSE_PAIR`` (default on): the verify's global layers take the pair view when the build
    gave them one (``GEMMA4_SDPA_PAIR_SPLIT``); ``0`` keeps the 4-head verify."""
    return os.environ.get(VERIFY_PREFIX_LSE_PAIR_ENV, "1") != "0"


def prefix_lse_rung_reason(splits: int, s_k: int, env=None) -> Optional[str]:
    """None when a record at rung ``s_k`` whose prefix takes ``splits`` calls should take the prefix + fresh verify,
    else why it keeps the masked one. Served 31B (1,8) boot tables (ms, plain..top of the rung): one call ((5,3), (8,3))
    is cheaper at every rung ((5,3) 8K 51.9..52.4 -> 50.7..52.2, 32K 57.4..58.5 -> 52.3..55.1); two calls ((1,5), (5,5))
    cost 1..3 ms more at the 8K / 32K rungs ((1,5) 8K 52.6..53.1 -> 55.6..56.7, which moved its served gain from 1.28x
    to 1.24x, under the adaptive policy's 1.25 entry, and halved multi-user short-context decode on the datasheet)
    and pay from the 131K rung ((1,5) 62.4..64.0 -> 58.0..62.8, (5,5) 86.2..93.1 -> 68.4..89.7)."""
    env = os.environ if env is None else env
    if str(env.get("GEMMA4_KV_PAIR_SHARD", "0")) == "1":
        return None  # pair-sharded global KV: only the prefix + fresh split reads it (the masked verify reads all)
    raw = str(env.get(VERIFY_PREFIX_LSE_SPLIT_MIN_SK_ENV, "") or "").strip()
    min_sk = int(raw) if raw else VERIFY_PREFIX_LSE_SPLIT_MIN_SK_DEFAULT
    if int(splits) > 1 and int(s_k) < min_sk:
        return f"{int(splits)} prefix calls pay only from S_k {min_sk} ({VERIFY_PREFIX_LSE_SPLIT_MIN_SK_ENV})"
    return None


def kv_dtype_name(dtype) -> Optional[str]:
    """``"bfloat16"`` / ``"bfloat8_b"`` for a ttnn DataType (or its name), None for any other cache format."""
    name = str(dtype or "").upper()
    if "BFLOAT8_B" in name:
        return "bfloat8_b"
    if "BFLOAT16" in name:
        return "bfloat16"
    return None


def prefix_lse_layer_reason(
    *,
    is_sliding: bool,
    cache_position_modulo,
    kv_modulo,
    nkv_local: int,
    head_dim: int,
    cache_shape: Sequence[int],
    is_kv_shared: bool = False,
) -> Optional[str]:
    """None when a layer's packed verify can take the prefix + fresh split, else why not. Static arguments only (the
    layer config, its cache's padded shape), so warmup, capture and replay decide alike: a global layer (no window),
    no ring (``cache_position_modulo`` / the verify's ``kv_modulo``: the causal bounds are absolute positions), its own
    K / V (not a KV-shared layer: the fresh part needs them), one local KV head, a head dim the op takes, and a cache
    the op can read as one KV head of ``head_dim`` through the stock geometry override: its own view
    ``[blocks, 1, block, head_dim]`` or a buffer shared with another layer type (vLLM's hybrid kv-cache groups give the
    31B's global layers the sliding view ``[blocks, 2, 64, 256]``), as long as the effective block
    (``prefix_effective_block``) is a whole number of tiles."""
    if is_sliding:
        return "sliding layer"
    if cache_position_modulo is not None or kv_modulo is not None:
        return f"ring cache (cache_position_modulo {cache_position_modulo}, verify modulo {kv_modulo})"
    if is_kv_shared:
        return "KV-shared layer (no K / V of its own for the fresh part)"
    if int(nkv_local) != 1:
        return f"{nkv_local} local KV heads (the op takes one)"
    if int(head_dim) not in HEAD_DIMS:
        return f"head_dim {head_dim} not in {HEAD_DIMS}"
    shape = tuple(int(s) for s in cache_shape)
    if len(shape) != 4:
        return f"cache {shape} is not [blocks, heads, block, dim]"
    block = prefix_effective_block(shape, head_dim)
    if block is None:
        return f"cache {shape} does not hold whole blocks of one {head_dim}-wide KV head (elements per block)"
    if block % TILE:
        return f"effective block {block} of cache {shape} (one {head_dim}-wide KV head) is not a multiple of {TILE}"
    return None


def prefix_effective_block(cache_shape: Sequence[int], head_dim: int, num_kv_heads: int = 1) -> Optional[int]:
    """The block size a layer of ``num_kv_heads`` x ``head_dim`` reads a paged cache of ``cache_shape`` with
    (``attention/operations.effective_block_size``: the elements per block are preserved across vLLM's shared
    hybrid-group views, ``heads * block * dim == num_kv_heads * eff_block * head_dim``), None when they do not divide.
    The 31B's global layer (one KV head of 512) on the sliding view ``[blocks, 2, 64, 256]``: 64."""
    _blocks, heads, block, dim = (int(s) for s in cache_shape)
    numerator, denominator = heads * block * dim, int(num_kv_heads) * int(head_dim)
    if denominator <= 0 or numerator % denominator:
        return None
    return numerator // denominator


def prefix_head_splits(h_local: int, packed_p: int) -> int:
    """The number of prefix calls a record row's ``h_local * P`` packed q rows take: the smallest divisor ``d`` of
    ``h_local`` with ``(h_local / d) * P <= 32`` (whole heads per call, head-major rows), 0 when even one head's ``P``
    rows exceed a tile."""
    h, p = int(h_local), int(packed_p)
    for d in range(1, h + 1):
        if h % d == 0 and (h // d) * p <= MAX_Q_ROWS:
            return d
    return 0


@dataclasses.dataclass(frozen=True)
class PrefixLseGeometry:
    """The static shape of one serving shape's prefix + fresh verify: ``batch`` record rows of ``packed_p``
    candidates, ``h_local`` q heads per chip in ``splits`` prefix calls of ``rows`` packed q rows each, the global
    layers' cache view (block, blocks, format), and ``pair``: the pair view (``h_local`` = the KV group's q heads, the
    prefix call split over the two chips of the pair)."""

    batch: int
    packed_p: int
    h_local: int
    head_dim: int
    splits: int
    block_size: int
    num_cache_blocks: int
    kv_dtype: str  # "bfloat16" / "bfloat8_b"
    layer_type: str = "full_attention"
    pair: bool = False

    @property
    def heads_per_call(self) -> int:
        return self.h_local // self.splits

    @property
    def rows(self) -> int:
        """Packed q rows of one prefix call: ``heads_per_call * P`` (head-major, row ``h * P + p``)."""
        return self.heads_per_call * self.packed_p


@dataclasses.dataclass(frozen=True)
class PrefixLsePlan:
    """One record's prefix call: the geometry, the verify table width it reads, the program knobs (grid, K chunk,
    cores-per-head cap and what it gives) and the CB footprint against the budget (the op's planner)."""

    geometry: PrefixLseGeometry
    table_cols: int
    grid: tuple
    k_chunk: int
    max_cores_per_head: int
    cores_per_head: int
    cb_bytes: int
    budget_bytes: int
    fits: bool
    note: str = ""

    @property
    def batch(self) -> int:
        return self.geometry.batch

    @property
    def packed_p(self) -> int:
        return self.geometry.packed_p

    @property
    def h_local(self) -> int:
        return self.geometry.h_local

    @property
    def head_dim(self) -> int:
        return self.geometry.head_dim

    @property
    def splits(self) -> int:
        return self.geometry.splits

    @property
    def rows(self) -> int:
        return self.geometry.rows

    @property
    def pair(self) -> bool:
        return self.geometry.pair

    def describe(self) -> str:
        g = self.geometry
        head = (
            f"{g.batch} rows x {g.packed_p} candidates, {g.h_local} heads of {g.head_dim} in {g.splits} prefix "
            f"call(s) of {g.rows} q rows{' (pair split)' if g.pair else ''}, {g.kv_dtype} cache (block "
            f"{g.block_size}), {self.table_cols}-column table on a {self.grid[0]}x{self.grid[1]} grid"
        )
        if not self.fits:
            return head + f": does not fit ({self.note})"
        return head + (
            f": k_chunk {self.k_chunk}, {self.cores_per_head} cores per row (cap {self.max_cores_per_head}), CBs "
            f"{self.cb_bytes:,} of {self.budget_bytes:,} B"
        )


def _env_int(env, name):
    raw = str(env.get(name, "") or "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from None


def prefix_program_knobs(head_dim: int, batch: int, kv_dtype: str, device_grid, env=None):
    """``(grid, k_chunk, max_cores_per_head)`` of the prefix call, the plain decode's choices
    (``decode_forward``): the (8, 4) grid for a lone row of a head_dim-512 layer over a bf16 cache, else the device
    grid; the K chunk of ``GEMMA4_SDPA_DECODE_K_CHUNK`` (64); the cores cap of
    ``GEMMA4_SPEC_VERIFY_PREFIX_LSE_MAX_CORES``, else ``GEMMA4_SDPA_MAX_CORES_PER_HEAD``, else 16."""
    env = os.environ if env is None else env
    gx, gy = (int(v) for v in device_grid)
    if int(head_dim) >= 512 and int(batch) == 1 and kv_dtype == "bfloat16":
        grid = (min(8, gx), min(4, gy))
    else:
        grid = (gx, gy)
    k_chunk = _env_int(env, DECODE_K_CHUNK_ENV)  # 0 (the stock dynamic chunk) is the op planner's to refuse
    cores = _env_int(env, VERIFY_PREFIX_LSE_MAX_CORES_ENV) or _env_int(env, DECODE_MAX_CORES_ENV)
    return grid, int(DEFAULT_K_CHUNK if k_chunk is None else k_chunk), int(cores or DEFAULT_MAX_CORES_PER_HEAD)


def _aligned(n_bytes: int, alignment: int = DRAM_ALIGNMENT) -> int:
    return -(-int(n_bytes) // alignment) * alignment


def plan_verify_prefix_lse(
    geometry: PrefixLseGeometry, table_cols: int, device_grid, budget_bytes: int, planner, env=None
) -> PrefixLsePlan:
    """The prefix call of one record (``table_cols``: its global verify table width) on ``device_grid`` against
    ``budget_bytes`` of CB space per core. ``planner`` is ``custom_ops/sdpa_decode_lse/planner.py`` (passed in: the
    module is pure python, its package imports ttnn): the op's own core plan and CB list for this geometry, the
    ``[B]`` int32 positions and the int32 table row as DRAM pages. ``fits=False`` with the reason when the geometry
    leaves the op's contract or its CBs exceed the budget."""
    g = geometry
    grid, k_chunk, cap = prefix_program_knobs(g.head_dim, g.batch, g.kv_dtype, device_grid, env)
    shared = dict(geometry=g, table_cols=int(table_cols), grid=grid, k_chunk=k_chunk, max_cores_per_head=cap)
    shared["budget_bytes"] = int(budget_bytes)
    if g.splits < 1:
        return PrefixLsePlan(**shared, cores_per_head=0, cb_bytes=0, fits=False, note="one head's P rows exceed a tile")
    try:
        lp = planner.plan(
            planner.Geometry(
                batch=g.batch,
                num_q_heads=g.rows,
                head_dim=g.head_dim,
                max_blocks=int(table_cols),
                block_size=g.block_size,
                num_cache_blocks=g.num_cache_blocks,
                kv_dtype=g.kv_dtype,
                k_chunk_size=k_chunk,
                scale=1.0,
                cur_pos_stick=_aligned(4 * g.batch),
                page_table_stick=_aligned(4 * int(table_cols)),
                cache_position_modulo=0,
                fp32_dest_acc_en=False,
                exp_approx_mode=False,
            ),
            grid,
            cap,
            pair_split=g.pair,
        )
    except ValueError as e:
        return PrefixLsePlan(**shared, cores_per_head=0, cb_bytes=0, fits=False, note=str(e))
    cb = int(lp.cb_total_bytes)
    fits = cb <= int(budget_bytes)
    note = "" if fits else f"CBs {cb:,} B over the {int(budget_bytes):,} B budget"
    return PrefixLsePlan(**shared, cores_per_head=int(lp.cores.cores_per_head), cb_bytes=cb, fits=fits, note=note)


# ── the per-replay inputs and the constant mask ────────────────────────────────────────────────────────────────────


def prefix_positions(positions: Sequence[int]) -> torch.Tensor:
    """``[B]`` int32 cur_pos of the prefix call of one replay: ``max(c_b - 1, 0)`` (``positions`` = the record's padded
    row positions ``c_b``, padding rows mirroring row 0). A row at 0 attends key 0 and is weighted out by its bias."""
    cs = [int(c) for c in positions]
    if not cs or min(cs) < 0:
        raise ValueError(f"prefix positions need at least one row and no negative position, got {cs}")
    if max(cs) >= 2**31:
        raise ValueError(f"position {max(cs)} overflows the int32 cur_pos")
    return torch.tensor([max(c - 1, 0) for c in cs], dtype=torch.int32)


def prefix_bias(positions: Sequence[int], rows: int) -> torch.Tensor:
    """``[1, B, rows, 1]`` fp32 added to the prefix lse of every packed q row: ``EMPTY_PREFIX_LSE`` for a row at
    ``c_b = 0`` (no prefix), 0 elsewhere."""
    cs = torch.tensor([int(c) for c in positions], dtype=torch.int64)
    bias = torch.where(cs == 0, EMPTY_PREFIX_LSE, 0.0).to(torch.float32)
    return bias.reshape(1, -1, 1, 1).expand(1, len(cs), int(rows), 1).contiguous()


def fresh_causal_mask(batch: int, heads: int, packed_p: int, neg: float = FRESH_MASK_NEG) -> torch.Tensor:
    """``[1, B, heads * P, 32]`` fp32: packed row ``h * P + p`` sees fresh key ``q`` (0) iff ``q <= p``, else ``neg``;
    the padded keys ``q >= P`` are hidden in every row. Independent of the positions: one constant per shape."""
    P, H = int(packed_p), int(heads)
    p = torch.arange(P).repeat(H)  # row h*P + p -> p
    q = torch.arange(TILE)
    rows = torch.where(q[None, :] <= p[:, None], 0.0, float(neg)).to(torch.float32)
    return rows.reshape(1, 1, H * P, TILE).expand(1, int(batch), H * P, TILE).contiguous()


# ── torch emulations of the device sequence (decode._fresh_kv_rows / _prefix_lse_verify_rows) ─────────────────────


def fresh_kv_rows(kv: torch.Tensor, batch: int, packed_p: int) -> torch.Tensor:
    """``decode._fresh_kv_rows`` in torch: ``[1, 1, >= B * P, D]`` (the prefill-style split's K or V, row ``b * P + p``,
    tile-padded tail) -> logical rows -> ``[1, B, P, D]`` -> zero rows ``P .. 31`` -> ``[1, B, 32, D]``."""
    B, P = int(batch), int(packed_p)
    D = int(kv.shape[-1])
    rows = kv[:, :, : B * P].reshape(1, B, P, D)
    return F.pad(rows, (0, 0, 0, TILE - P))


def prefix_q_rows(q: torch.Tensor, packed_p: int, splits: int):
    """The device Q layout in torch: ``[1, H, B * P, D]`` (the prefill-style split after RoPE) -> head-major
    ``[1, B, H * P, D]`` -> ``splits`` slices of ``[1, B, (H / splits) * P, D]`` (whole heads each)."""
    _one, H, rows, D = (int(s) for s in q.shape)
    P = int(packed_p)
    B = rows // P
    packed = q.reshape(1, H, B, P, D).permute(0, 2, 1, 3, 4).reshape(1, B, H * P, D)
    r = (H // int(splits)) * P
    return [packed[:, :, s * r : (s + 1) * r] for s in range(int(splits))]


def prefix_out_rows(parts, heads: int, packed_p: int) -> torch.Tensor:
    """The device un-pack in torch: the merged ``[1, B, R, D]`` per prefix call -> concat over the calls
    ``[1, B, H * P, D]`` -> ``[1, B * P, H, D]``, the rows the o_proj tail takes."""
    out = torch.cat(list(parts), dim=2)
    _one, B, _hp, D = (int(s) for s in out.shape)
    H, P = int(heads), int(packed_p)
    return out.reshape(1, B, H, P, D).permute(0, 1, 3, 2, 4).reshape(1, B * P, H, D)


def lse_from_stats(m: torch.Tensor, l: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """``scale * m + ln(l)`` in fp32 (the op's m is the raw-score max, l the row sum at that max)."""
    return float(scale) * m.float() + torch.log(l.float())


def fresh_attention_torch(q, k_fresh, v_fresh, fresh_mask, scale: float = 1.0):
    """The fresh part, op for op: ``q [1, B, R, D]``, ``k_fresh / v_fresh [1, B, 32, D]``, ``fresh_mask
    [1, B, R, 32]`` -> ``(o_un [1, B, R, D], m [1, B, R, 1], l [1, B, R, 1])`` fp32: scores = ``q k^T * scale +
    mask``, ``m`` their row max, ``e = exp(scores - m)``, ``l = sum e``, ``o_un = e @ v``."""
    scores = (q.float() @ k_fresh.float().transpose(-2, -1)) * float(scale) + fresh_mask.float()
    m = scores.amax(dim=-1, keepdim=True)
    e = torch.exp(scores - m)
    return e @ v_fresh.float(), m, e.sum(dim=-1, keepdim=True)


def merge_prefix_fresh_torch(o1, m1, l1, bias, o2_un, m2, l2, scale: float = 1.0) -> torch.Tensor:
    """The merge, op for op (fp32): ``lse_1 = scale * m_1 + ln l_1 + bias``, ``lse_2 = m_2 + ln l_2``,
    ``d = clamp(lse_1 - lse_2)``, ``O = sigmoid(d) * O_1 + sigmoid(-d) / l_2 * o2_un``. ``o1 [1, B, R, D]`` the
    prefix call's (normalised) output, ``m1 / l1 / bias / m2 / l2 [1, B, R, 1]``."""
    lse1 = lse_from_stats(m1, l1, scale) + bias.float()
    lse2 = m2.float() + torch.log(l2.float())
    d = (lse1 - lse2).clamp(-LOGIT_CLAMP, LOGIT_CLAMP)
    return o1.float() * torch.sigmoid(d) + o2_un.float() * (torch.sigmoid(-d) / l2.float())


def pair_half_keys(cur_pos: int, k_chunk: int, half: int):
    """The keys ``[lo, hi)`` pair half ``half`` of a row at ``cur_pos`` attends (``SdpaDecodeLse(pair_split=True)``,
    the mirror of ``planner.pair_key_range``): the row's ``n = ceil((cur_pos + 1) / k_chunk)`` chunks, half 0 the first
    ``ceil(n / 2)``, half 1 the rest (empty for a one-chunk row), clipped to ``cur_pos + 1``."""
    k = int(k_chunk)
    n = -(-(int(cur_pos) + 1) // k)
    first = (n + 1) // 2
    base, cnt = (0, first) if int(half) == 0 else (first, n - first)
    end = min((base + cnt) * k, int(cur_pos) + 1)
    return min(base * k, end), end


def prefix_attention_torch(
    q,
    k_cache,
    v_cache,
    page_table,
    cur_pos,
    block_size: int,
    scale: float = 1.0,
    dtype: Optional[torch.dtype] = None,
    half: Optional[int] = None,
    k_chunk: Optional[int] = None,
):
    """The prefix call's semantics (``SdpaDecodeLse`` over one KV head): ``q [1, B, R, D]``, paged caches
    ``[blocks, 1, block, D]``, ``page_table [B, cols]``, ``cur_pos [B]`` -> ``(o [1, B, R, D], m [1, B, R, 1],
    l [1, B, R, 1])``, row ``b`` attending keys ``[0, cur_pos[b]]`` through row ``b`` of the table; ``m`` the raw-score
    max (unscaled, as the op writes it), ``l = sum exp((s - m) * scale)``, ``o`` normalised by that ``l``. fp32 by
    default; ``dtype`` (bf16) rounds as the op's CBs do: the scores (so ``m`` is one of them, exactly), ``l`` and
    ``o``, each output consistent with the ``m`` / ``l`` it was normalised by. ``half`` (with ``k_chunk``): the pair
    split's half of the row (``pair_half_keys``); an empty half gives the op's sentinel O = 0, m = EMPTY_HALF_M,
    l = 1."""
    _one, B, R, D = (int(s) for s in q.shape)
    bs = int(block_size)

    def rnd(t):
        return t if dtype is None else t.to(dtype).float()

    o = torch.zeros(1, B, R, D)
    m = torch.zeros(1, B, R, 1)
    l = torch.zeros(1, B, R, 1)
    for b in range(B):
        lo, hi = (0, int(cur_pos[b]) + 1) if half is None else pair_half_keys(int(cur_pos[b]), k_chunk, half)
        if lo == hi:
            m[0, b], l[0, b] = EMPTY_HALF_M, 1.0
            continue
        keys = torch.arange(lo, hi)
        blocks = torch.as_tensor(page_table)[b, keys // bs].to(torch.int64)
        k = k_cache[blocks, 0, keys % bs].float()
        v = v_cache[blocks, 0, keys % bs].float()
        s = rnd(q[0, b].float() @ k.T)
        mb = s.amax(dim=-1, keepdim=True)
        e = torch.exp((s - mb) * float(scale))
        lb = rnd(e.sum(dim=-1, keepdim=True))
        o[0, b] = rnd((e @ v) / lb)
        m[0, b], l[0, b] = mb, lb
    return o, m, l


def pair_prefix_merge_torch(o, m, l, bias, o2_un, m2, l2, scale: float = 1.0) -> torch.Tensor:
    """The pair split's three-way merge on a mesh row, op for op (fp32): ``o [tp, B, R, D]`` each chip's half-normalised
    prefix output, ``m / l [tp, B, R, 1]`` its statistics, ``bias [1, B, R, 1]``, the fresh part ``o2_un / m2 / l2``
    (identical on every chip). Returns ``[tp, B, R, D]``, what each chip feeds its pair o_proj: ``lse_self`` (+ bias),
    ``lse_partner`` from the gathered row times ``relu(-selector)`` (the chip's selector is ``e_c - e_{c^1}``),
    ``top = max(lse_self, lse_partner, lse_fresh)``, ``a_i = exp(lse_i - top)``, ``S = (a_self + a_partner) + a_fresh``,
    ``a_self / S * O_self + 0.5 * a_fresh / (S * l_2) * o2_un``."""
    tp = int(o.shape[0])
    lse = lse_from_stats(m, l, scale) + bias.float()  # [tp, B, R, 1]
    partner = torch.relu(-pair_selector(tp).reshape(tp, tp))  # row c: e_{c^1}
    lse_p = torch.einsum("cj,jbrk->cbrk", partner, lse)
    lse_f = m2.float() + torch.log(l2.float())
    top = torch.maximum(torch.maximum(lse, lse_p), lse_f)
    a_s, a_p, a_f = torch.exp(lse - top), torch.exp(lse_p - top), torch.exp(lse_f - top)
    total = (a_s + a_p) + a_f
    return o.float() * (a_s / total) + o2_un.float() * ((a_f / (total * l2.float())) * 0.5)
