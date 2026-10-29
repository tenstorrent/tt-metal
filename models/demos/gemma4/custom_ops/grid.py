# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""The compute grid a custom op plans its work on, and the planners' shared rules.

Every op derives its work split (cores per row tile, column tiles per core, units per core, exchange groups)
from the device's ``compute_with_storage_grid_size`` -- never from a constant -- so the same kernels run on
Wormhole (8x8 worker grid) and Blackhole (13x10). ``GEMMA4_TEST_GRID=WxH`` (a test knob) caps the grid an op may
use, e.g. ``8x8`` on Blackhole exercises the Wormhole split on a Blackhole chip. The planners here are pure
python so a host test can check them for both grids.
"""

import dataclasses
import os

import ttnn

# The Blackhole worker L1 (the Python MeshDevice exposes no l1_size_per_core); the Blackhole-only ops size their
# resident CBs against it from the allocator base.
BLACKHOLE_L1_PER_CORE = 1536 * 1024


def compute_grid(device):
    """``(x, y)`` of the grid an op may use: the device grid, capped by ``GEMMA4_TEST_GRID`` when set."""
    g = device.compute_with_storage_grid_size()
    gx, gy = int(g.x), int(g.y)
    cap = os.environ.get("GEMMA4_TEST_GRID", "").strip().lower()
    if cap:
        cx, cy = (int(v) for v in cap.split("x"))
        gx, gy = min(gx, cx), min(gy, cy)
    return gx, gy


def grid_cores(grid) -> int:
    """Core count of a ``(x, y)`` grid."""
    return int(grid[0]) * int(grid[1])


def row_major_cores(grid, n):
    """The first ``n`` cores of the grid in row-major order (x fastest), as ``ttnn.CoreCoord``."""
    gx, gy = grid
    assert n <= gx * gy, f"{n} cores on a {gx}x{gy} grid"
    return [ttnn.CoreCoord(i % gx, i // gx) for i in range(n)]


def core_range_set(cores):
    return ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])


def plan_qkv_heads_units(grid, batch):
    """Work split of ``qkv_heads_norm_rope``: ``(n_cores, tensors_per_core)``. One (user, tensor) per core when
    the grid holds 3 B cores, else one user per core with the three tensors in sequence (needs B cores)."""
    gx, gy = grid
    cores = gx * gy
    if 3 * batch <= cores:
        return 3 * batch, 1
    if batch <= cores:
        return batch, 3
    raise ValueError(f"batch {batch} does not fit {cores} cores")


def plan_concat_heads_units(grid, num_heads, cols, heads_per_unit=None):
    """Work split of ``concat_heads_decode``: ``(n_cores, heads_per_unit, groups, per-core unit counts)``."""
    gx, gy = grid
    n_grid = gx * gy
    if heads_per_unit is None:
        heads_per_unit = next(hg for hg in range(1, num_heads + 1) if cols * ((num_heads + hg - 1) // hg) <= n_grid)
    heads_per_unit = max(1, min(num_heads, heads_per_unit))
    groups = (num_heads + heads_per_unit - 1) // heads_per_unit
    n_units = cols * groups
    n_cores = min(n_units, n_grid)
    per = [(n_units // n_cores) + (1 if i < n_units % n_cores else 0) for i in range(n_cores)]
    return n_cores, heads_per_unit, groups, per


NORM_MAX_CORES = 64  # an exchange group: two fp32 receive tiles of 32 columns
NORM_DEFAULT_CORES = 22  # the preferred group size (GEMMA4_FUSED_NORM_CORES caps it)


def plan_norm_split(grid_cores_, cols_t, nin=1, rt_in=1, cap=None):
    """The one work split of the fused norms (``fused_norm_decode``) for ``rt_in`` row tiles on a grid of
    ``grid_cores_`` cores: ``(N, per, groups, passes)`` -- ``N`` cores per row tile (a divisor of ``cols_t`` with
    ``per = cols_t / N <= 8 / nin`` tiles each so one fp32 dest window holds them), ``groups`` row tiles per pass
    (``min(rt_in, grid_cores_ // N)``) and ``passes = ceil(rt_in / groups)`` launches of the program. The split
    minimizes the passes, prefers a core count up to the cap (``cap``, else ``GEMMA4_FUSED_NORM_CORES``, default
    22) and then the largest count; it is derived from the grid, never from a constant: 2816 wide at 128 rows =
    11 cores x 4 row tiles in one pass on Wormhole's 8x8, 22 x 4 on Blackhole's 13x10. Raises ValueError when no
    split fits (the callers fall back to the composite path)."""
    if cap is None:
        cap = int(os.environ.get("GEMMA4_FUSED_NORM_CORES", str(NORM_DEFAULT_CORES)))
    cap = min(NORM_MAX_CORES, max(1, int(cap)))
    max_per = 8 // nin
    best = None
    for n in range(1, min(NORM_MAX_CORES, cols_t) + 1):
        if cols_t % n or cols_t // n > max_per:
            continue
        groups = min(rt_in, grid_cores_ // n)
        if groups < 1:
            continue
        passes = -(-rt_in // groups)
        key = (passes, 0 if n <= cap else 1, -n)
        if best is None or key < best[0]:
            best = (key, (n, cols_t // n, groups, passes))
    if best is None:
        raise ValueError(
            f"no core count up to {NORM_MAX_CORES} divides {cols_t} column tiles into <= {max_per} on "
            f"{grid_cores_} cores"
        )
    return best[1]


def plan_norm_cores(grid, cols_t, nin=1, cap=None):
    """The fused norms' core count for one row tile on ``grid`` (``plan_norm_split``'s ``N``)."""
    return plan_norm_split(grid_cores(grid), cols_t, nin, 1, cap)[0]


# ── Packed-verify decode SDPA: the K chunk and the cores per head that fit one core's L1 ────────────────────
# ``ttnn.transformer.(paged_)scaled_dot_product_attention_decode`` allocates every circular buffer on its whole
# core range, and program.cpp refuses a program whose static CBs exceed the 1.5 MB per-core L1. The packed verify
# puts ``rows`` positions of every local query head into the Q head dim (``PNH = h_local * rows`` rows, padded to
# tiles: ``PNHt``), so every Q-shaped CB scales with it; the 31B at TP4 (8 local q heads, 6 rows: PNHt 2 on the
# head_dim-512 global layers, 1 local KV head) overflowed at the op defaults (k_chunk 64, 16 cores per head:
# 1,884,544 B), as did its dFlash drafter (16 local heads x 17 rows: PNHt 9 on head_dim 128, 2 local KV heads:
# 2,362,752 B). The per-core tile count below is sdpa_decode_program_factory.cpp's CB list (bf16 full tiles,
# non-causal, mask on, no sink / sliding CB), in units of 2 KB tiles:
#   q (c_0, c_10)                      2 * PNHt * DHt
#   k, v (c_1, c_2, double-buffered)   4 * Sk * DHt
#   mask + qk scores (c_3, c_24)       2 * PNHt * Sk
#   out-shaped (c_25, c_26, c_23, c_16, c_20)   5 * PNHt * DHt
#   softmax statistics (11 CBs)        11 * PNHt
#   scale / identity (c_5, c_11, c_12) 3
#   cross-core flash reduction (c_19)  (PNHt * DHt + 2 * PNHt) * (cores_per_head - 1)
# with ``cores_per_head = min(grid, max_cores_per_head_batch * B * nkv) // B // nkv`` (the factory's core
# allocation). The planner walks a ladder of (k_chunk, max_cores_per_head_batch) from the op defaults, keeping the
# reduction width before the chunk (a narrower chunk costs one more chunk loop per K tile; fewer cores per head
# scan a longer K range each), and takes the first that fits ``SDPA_DECODE_L1_BUDGET_TILES``; the env knobs pin a
# chunk and the ladder walks the cores alone. The unpacked (plain) decode SDPA keeps its own config.
SDPA_DECODE_TILE_BYTES = 2048
SDPA_DECODE_L1_BYTES = 1536 * 1024
#: 15/16 of the L1 the program check enforces: the CB model above over-reads the real total by 0.5-7 % (measured
#: on the two overflowing programs), so this is the margin, not a tuning knob.
SDPA_DECODE_L1_BUDGET_TILES = 720
PACKED_SDPA_K_CHUNK_ENV = "GEMMA4_PACKED_SDPA_K_CHUNK"
PACKED_SDPA_Q_CHUNK_ENV = "GEMMA4_PACKED_SDPA_Q_CHUNK"
PACKED_SDPA_K_CHUNK_LEGACY_ENV = "GEMMA4_PV_K_CHUNK"  # the earlier experiment knob: read when the new one is unset
PACKED_SDPA_DEFAULT_K_CHUNK = 64
PACKED_SDPA_DEFAULT_Q_CHUNK = 32
PACKED_SDPA_DEFAULT_CORES_PER_HEAD = 16
#: (k_chunk, max_cores_per_head_batch) from the op defaults down: the chunk halves before the cores do.
PACKED_SDPA_LADDER = ((64, 16), (32, 16), (64, 8), (32, 8), (64, 4), (32, 4), (64, 2), (32, 2), (64, 1), (32, 1))


def sdpa_decode_cores_per_head(grid_cores_, batch, nkv, max_cores_per_head_batch):
    """The factory's cores per (batch row, KV head): ``min(grid, max * B * nkv) // B // nkv``, at least 1."""
    uncapped = min(int(grid_cores_), int(max_cores_per_head_batch) * int(batch) * int(nkv)) // max(1, int(batch))
    return max(1, uncapped // max(1, int(nkv)))


def sdpa_decode_cb_tiles(pnht, dht, sk_tiles, cores_per_head):
    """The decode SDPA's static CB total on one core, in 2 KB tiles (the model in the module comment)."""
    pnht, dht, sk, cph = int(pnht), int(dht), int(sk_tiles), int(cores_per_head)
    return (
        2 * pnht * dht  # q in + tilized q
        + 4 * sk * dht  # k + v, double-buffered
        + 2 * pnht * sk  # attention mask + qk scores
        + 5 * pnht * dht  # three out intermediates, out stats, out
        + 11 * pnht  # softmax statistics
        + 3  # scale, column identity, scale copy
        + (pnht * dht + 2 * pnht) * (cph - 1)  # cross-core flash reduction staging
    )


@dataclasses.dataclass(frozen=True)
class PackedSdpaPlan:
    """What the packed-verify decode SDPA runs with: the chunks, the cores-per-head cap, and the CB footprint."""

    k_chunk: int
    q_chunk: int
    max_cores_per_head_batch: int
    cores_per_head: int
    pnht: int
    tiles: int
    fits: bool

    @property
    def cb_bytes(self):
        return self.tiles * SDPA_DECODE_TILE_BYTES

    def describe(self):
        return (
            f"k_chunk {self.k_chunk}, q_chunk {self.q_chunk}, max_cores_per_head_batch {self.max_cores_per_head_batch} "
            f"({self.cores_per_head} cores/head), PNHt {self.pnht}, CBs {self.cb_bytes} B"
            + ("" if self.fits else " (over the L1 budget)")
        )


def _env_chunk(env, name, default, multiple=32):
    raw = str(env.get(name, "") or "").strip()
    if not raw:
        return default, False
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{name} must be an integer multiple of {multiple}, got {raw!r}") from None
    if value < multiple or value % multiple:
        raise ValueError(f"{name} must be a positive multiple of {multiple}, got {value}")
    return value, True


def plan_packed_sdpa(h_local, nkv_local, rows, head_dim, batch, grid_cores_, env=None, budget_tiles=None):
    """The packed-verify decode SDPA config for ``h_local`` query heads x ``rows`` packed positions per batch row,
    ``nkv_local`` KV heads, ``head_dim``, ``batch`` rows on a grid of ``grid_cores_`` cores: the first rung of
    ``PACKED_SDPA_LADDER`` whose CBs fit the budget. ``GEMMA4_PACKED_SDPA_K_CHUNK`` (or the earlier
    ``GEMMA4_PV_K_CHUNK``) pins the K chunk and ``GEMMA4_PACKED_SDPA_Q_CHUNK`` the Q chunk; unset, both derive.
    When no rung fits, the last (smallest) one is returned with ``fits=False`` for the caller to report."""
    env = os.environ if env is None else env
    budget = SDPA_DECODE_L1_BUDGET_TILES if budget_tiles is None else int(budget_tiles)
    h_local, nkv_local, rows, head_dim, batch = (int(v) for v in (h_local, nkv_local, rows, head_dim, batch))
    if min(h_local, nkv_local, rows, head_dim, batch, int(grid_cores_)) < 1:
        raise ValueError("plan_packed_sdpa needs positive heads, KV heads, rows, head_dim, batch and grid cores")
    if head_dim % 32:
        raise ValueError(f"plan_packed_sdpa needs a head_dim that is a multiple of 32, got {head_dim}")
    pnht = -(-(h_local * rows) // 32)
    dht = head_dim // 32
    q_chunk, _ = _env_chunk(env, PACKED_SDPA_Q_CHUNK_ENV, PACKED_SDPA_DEFAULT_Q_CHUNK)
    k_chunk, k_pinned = _env_chunk(env, PACKED_SDPA_K_CHUNK_ENV, PACKED_SDPA_DEFAULT_K_CHUNK)
    if not k_pinned:
        k_chunk, k_pinned = _env_chunk(env, PACKED_SDPA_K_CHUNK_LEGACY_ENV, PACKED_SDPA_DEFAULT_K_CHUNK)
    ladder = [(k, c) for k, c in PACKED_SDPA_LADDER if not k_pinned or k == k_chunk]
    if k_pinned and not ladder:
        ladder = [(k_chunk, c) for _k, c in PACKED_SDPA_LADDER if _k == PACKED_SDPA_DEFAULT_K_CHUNK]
    plan = None
    for k, cores in ladder:
        cph = sdpa_decode_cores_per_head(grid_cores_, batch, nkv_local, cores)
        tiles = sdpa_decode_cb_tiles(pnht, dht, k // 32, cph)
        plan = PackedSdpaPlan(k, q_chunk, cores, cph, pnht, tiles, tiles <= budget)
        if plan.fits:
            return plan
    return plan
