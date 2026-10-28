# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Cross-op weight prefetch for the 1D (mcast_in0) decode matmuls (QWEN36_WEIGHT_PREFETCH, default off).

The Tensor prefetcher (ttnn.experimental.start_tensor_prefetcher) runs one kernel per DRAM bank on the
Blackhole DRAM cores (DRISC). It streams a matmul's weight from GDDR straight into a GlobalCircularBuffer
(GCB) on the matmul's worker cores, over NOC0, off the command queue. The mcast_in0 1D matmul consumes
in1 from that GCB instead of reading DRAM itself (tt-metal dbe53bc2289 / #57456 and the GCB path before
it). Each request is captured into the decode trace next to its matmul, so a trace replay hands the
DRISC every request of the step at once: it fills the next matmul's GCB window while norms, CCLs,
attention and the GDN recurrence run, and keeps streaming while the matmul drains it.

Contract (per weight):
* the weight lives in DRAM in the receiver-contiguous layout: an NdShardSpec with one (K, per_core_N)
  shard per output worker, ROUND_ROBIN_1D over the DRAM banks, so shard i == ring position i == the
  i-th row-major worker of the matmul grid;
* the consuming matmul runs the same 1D mcast_in0 program config (one output block per worker: out_block_h/w
  default to per_core_M/N), with in1 fed from the GCB;
* every queued request is consumed by exactly one matmul, in queue order. All qwen36 compute runs on
  cq 0, and the request is captured into the same trace as its matmul (prefetch_and_linear).

The math is the matmul's own: the same compute kernel consumes the same K-blocks in the same order, so
outputs are expected to be byte-identical to the DRAM-reading 1D matmul. test_weight_prefetch.py checks
that on device.

Targets (QWEN36_WEIGHT_PREFETCH = comma list, or 1 = gateup):
* gateup   : MLP gate/up. Decode-only weights on TP (prefill uses the packed AGMM weight), so they are
             loaded in the receiver-contiguous layout INSTEAD of interleaved: no extra DRAM.
* down, gdn_in, gdn_out, attn_qkv, attn_wo : these weights are also read by prefill, so prefetch loads a
             second, receiver-contiguous copy for decode (extra DRAM per chip, logged at load).
* all      : every target above.

Knobs: QWEN36_WEIGHT_PREFETCH_GCB_KB (default 72): per-GCB L1 budget; each GCB holds at least two
K-block pages (the reader's streaming window) and as many whole pages as fit the budget. One GCB per
(grid width, receiver count, page size); the GCB is a lockstep L1 allocation, so every GCB costs its size
on every worker core.

Falls back (logged, never raises at load) when the Tensor prefetcher is unsupported (needs Blackhole with
firmware >= 19.12.0.0) or a weight's geometry does not fit the transport.
"""
import os
from dataclasses import dataclass

from loguru import logger

import ttnn

ENV = "QWEN36_WEIGHT_PREFETCH"
GCB_KB_ENV = "QWEN36_WEIGHT_PREFETCH_GCB_KB"
DEFAULT_GCB_KB = 72
TARGETS = ("gateup", "down", "gdn_in", "gdn_out", "attn_qkv", "attn_wo")
DEFAULT_TARGETS = frozenset({"gateup"})
TILE = 32
# The mcast_in0 GCB reader streams through a two-page window (matmul_device_operation.cpp
# kLookaheadMinResidentBlocks).
MIN_WINDOW_PAGES = 2


def parse_targets(value):
    """QWEN36_WEIGHT_PREFETCH value -> frozenset of targets. ''/0/off -> empty; 1/on -> DEFAULT_TARGETS."""
    v = (value or "").strip().lower()
    if v in ("", "0", "off", "false", "no"):
        return frozenset()
    if v in ("1", "on", "true", "yes"):
        return DEFAULT_TARGETS
    out = set()
    for tok in v.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok == "all":
            out.update(TARGETS)
        elif tok in TARGETS:
            out.add(tok)
        else:
            raise ValueError(f"{ENV}: unknown target {tok!r}; expected 0, 1, all or a comma list of {TARGETS}")
    return frozenset(out)


def enabled_targets():
    return parse_targets(os.environ.get(ENV, "0"))


def enabled(target):
    assert target in TARGETS, target
    return target in enabled_targets()


def gcb_budget_bytes():
    return max(0, int(os.environ.get(GCB_KB_ENV, str(DEFAULT_GCB_KB)) or 0)) * 1024


def tile_bytes(dtype):
    """Bytes per 32x32 tile of a DRAM weight dtype."""
    if dtype == ttnn.bfloat8_b:
        return 1088
    if dtype == ttnn.bfloat4_b:
        return 576
    if dtype == ttnn.bfloat16:
        return 2048
    raise ValueError(f"weight prefetch: unsupported weight dtype {dtype}")


@dataclass(frozen=True)
class Geometry:
    """How one (K, N) weight maps onto a 1D mcast_in0 program config's workers."""

    cols: int  # program-config grid width (ring position p -> core (p % cols, p // cols))
    rows: int
    k_tiles: int
    n_tiles: int
    per_core_N: int
    in0_block_w: int
    receivers: int  # output workers == GCB receivers == weight shards
    tile_bytes: int
    page_bytes: int  # one in1 K-block for one receiver
    block_count: int  # K-blocks per receiver per matmul (= k_tiles / in0_block_w)


def plan_geometry(k, n, cols, rows, per_core_N, in0_block_w, weight_tile_bytes):
    """Geometry for a per-device (k, n) weight on a (cols x rows) mcast_in0 grid, or (None, reason).

    The receiver-contiguous transport needs every output worker to own exactly per_core_N columns:
    n_tiles % per_core_N == 0 (no ragged last worker), and K a whole number of in0 blocks."""
    if k % TILE or n % TILE:
        return None, f"weight ({k}, {n}) is not tile aligned"
    k_tiles, n_tiles = k // TILE, n // TILE
    if per_core_N <= 0 or in0_block_w <= 0:
        return None, f"bad program config per_core_N={per_core_N} in0_block_w={in0_block_w}"
    if n_tiles % per_core_N:
        return None, f"N ({n_tiles} tiles) is not a whole number of per_core_N={per_core_N} blocks"
    if k_tiles % in0_block_w:
        return None, f"K ({k_tiles} tiles) is not a whole number of in0_block_w={in0_block_w} blocks"
    receivers = n_tiles // per_core_N
    if receivers > cols * rows:
        return None, f"{receivers} output workers exceed the {cols}x{rows} grid"
    return (
        Geometry(
            cols=cols,
            rows=rows,
            k_tiles=k_tiles,
            n_tiles=n_tiles,
            per_core_N=per_core_N,
            in0_block_w=in0_block_w,
            receivers=receivers,
            tile_bytes=weight_tile_bytes,
            page_bytes=in0_block_w * per_core_N * weight_tile_bytes,
            block_count=k_tiles // in0_block_w,
        ),
        None,
    )


def ring_core(pos, cols):
    """Logical (x, y) of ring position `pos`: the matmul places output block i on the i-th row-major core of
    its grid (num_cores_to_corerangeset_in_subcoregrids, start (0, 0) with no sub-device manager)."""
    return pos % cols, pos // cols


def bank_ring_positions(bank, receivers, num_banks):
    """Ring positions a DRAM bank feeds under ROUND_ROBIN_1D shard placement (shard s -> bank s % num_banks)."""
    return list(range(bank, receivers, num_banks))


def gcb_size_bytes(page_bytes, budget_bytes):
    """Whole pages that fit the budget, never fewer than the reader's two-page window."""
    return max(MIN_WINDOW_PAGES, budget_bytes // page_bytes) * page_bytes


def per_device_kn(torch_shape, shard_dim, tp):
    """Per-device (K, N) of a shard_w weight: torch [out, in] is transposed to [in, out] and split on `shard_dim`
    (-1 column-parallel, 0 row-parallel)."""
    out_f, in_f = int(torch_shape[-2]), int(torch_shape[-1])
    if shard_dim in (-1, 1):
        assert out_f % tp == 0, (out_f, tp)
        return in_f, out_f // tp
    assert shard_dim == 0, shard_dim
    assert in_f % tp == 0, (in_f, tp)
    return in_f // tp, out_f


def progcfg_problem(progcfg):
    """Why a 1D program config cannot consume from a DRAM-sender GCB, or None. The matmul op checks the same rules
    (matmul_device_operation.cpp: mcast_in0 only, one output block per worker, natural-order K-blocks)."""
    if not progcfg.mcast_in0 or progcfg.gather_in0:
        return "program config is not mcast_in0"
    if progcfg.stream_in1:
        return "program config streams in1 (gather-only mode)"
    if progcfg.out_block_h != progcfg.per_core_M or progcfg.out_block_w != progcfg.per_core_N:
        return "program config has more than one output block per worker"
    return None


def geometry_for(progcfg, k, n, dtype):
    """(Geometry, None) for a per-device (k, n) weight consumed by `progcfg`, or (None, reason). The registered matmul
    runs `progcfg` itself (its out_block_h/w already default to per_core_M/N), so the compute is unchanged."""
    why = progcfg_problem(progcfg)
    if why is not None:
        return None, why
    g = progcfg.compute_with_storage_grid_size
    return plan_geometry(k, n, int(g.x), int(g.y), int(progcfg.per_core_N), int(progcfg.in0_block_w), tile_bytes(dtype))


def recv_contig_memory_config(geom, num_banks):
    """DRAM NdShardSpec: one (K, per_core_N) shard per receiver, round-robin over the banks."""
    dram = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_banks - 1, 0))})
    nd = ttnn.NdShardSpec(
        ttnn.Shape([geom.k_tiles * TILE, geom.per_core_N * TILE]),
        dram,
        ttnn.ShardOrientation.ROW_MAJOR,
        ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
    )
    return ttnn.MemoryConfig(ttnn.BufferType.DRAM, nd)


def bank_to_receivers(geom, num_banks):
    pairs = []
    for b in range(num_banks):
        cores = [ring_core(p, geom.cols) for p in bank_ring_positions(b, geom.receivers, num_banks)]
        if cores:
            pairs.append(
                (b, ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in cores]))
            )
    return pairs


@dataclass
class _Entry:
    tag: str
    primary: object  # the weight call sites pass in (kept referenced so its id stays unique)
    twin: object  # receiver-contiguous weight the prefetcher streams (== primary for gate/up)
    progcfg: object
    gcb: object


class _MeshPrefetch:
    """Per-mesh state: the running Tensor prefetcher and its GCBs."""

    def __init__(self, mesh):
        self.mesh = mesh
        self.num_banks = int(mesh.dram_grid_size().x)
        self.gcbs = {}  # (cols, receivers, page_bytes) -> (gcb, size)
        self.twin_bytes = 0
        ttnn.experimental.start_tensor_prefetcher(mesh)
        logger.info(f"[weight_prefetch] Tensor prefetcher started ({self.num_banks} DRAM banks)")

    def gcb_for(self, geom):
        key = (geom.cols, geom.receivers, geom.page_bytes)
        hit = self.gcbs.get(key)
        if hit is None:
            size = gcb_size_bytes(geom.page_bytes, gcb_budget_bytes())
            try:
                gcb = ttnn.experimental.create_global_circular_buffer_for_tensor_prefetcher(
                    self.mesh, bank_to_receivers(geom, self.num_banks), size, support_multi_receiver_shards=False
                )
            except Exception as e:  # noqa: BLE001 - e.g. L1 or DRISC-arena exhaustion; remembered so the group's
                self.gcbs[key] = (None, f"GCB of {size} B failed: {e}")  # other layers do not retry it
                raise
            hit = self.gcbs[key] = (gcb, size)
            logger.info(
                f"[weight_prefetch] GCB #{len(self.gcbs)}: {geom.receivers} receivers on a {geom.cols}-wide grid, "
                f"{size} B ({size // geom.page_bytes} x {geom.page_bytes} B pages) of L1 per core"
            )
        if hit[0] is None:
            raise RuntimeError(hit[1])
        return hit[0]


_MESHES = {}  # id(mesh) -> _MeshPrefetch, or None when unsupported
_REGISTRY = {}  # id(primary weight) -> _Entry


def _mesh_state(mesh):
    key = id(mesh)
    if key not in _MESHES:
        state = None
        try:
            if not hasattr(ttnn.experimental, "tensor_prefetcher_matmul"):
                logger.warning(f"[weight_prefetch] {ENV} set but this ttnn has no tensor_prefetcher_matmul; disabled")
            elif not ttnn.experimental.is_tensor_prefetcher_supported(mesh):
                logger.warning(
                    f"[weight_prefetch] {ENV} set but the Tensor prefetcher is unsupported on this mesh "
                    "(needs Blackhole firmware >= 19.12.0.0); disabled"
                )
            else:
                state = _MeshPrefetch(mesh)
        except Exception as e:  # noqa: BLE001 - never fail model load over an opt-in perf path
            logger.warning(f"[weight_prefetch] Tensor prefetcher start failed ({e}); disabled")
            state = None
        _MESHES[key] = state
    return _MESHES[key]


def register(mesh, tag, primary, twin, base_progcfg, geom):
    """Route matmul_1d_decode/linear(primary, base_progcfg) through the prefetcher, streaming `twin`.
    Returns False (nothing registered) if the GCB or the weight<->matmul contract check fails."""
    state = _mesh_state(mesh)
    if state is None:
        return False
    try:
        gcb = state.gcb_for(geom)
        # The shared weight <-> matmul contract check (shard geometry, per_core_N, K blocking, GCB window) the matmul
        # itself runs; run it now so a mismatch falls back at load instead of hanging the first decode.
        bc = ttnn.experimental.tensor_prefetcher_block_count_for_matmul_1d(base_progcfg, twin, gcb)
        assert bc == geom.block_count, f"block_count {bc} != planned {geom.block_count}"
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[weight_prefetch] {tag}: not prefetched ({e})")
        return False
    _REGISTRY[id(primary)] = _Entry(tag, primary, twin, base_progcfg, gcb)
    return True


def load_prefetched(
    target, tag, mesh, torch_weight, shard_dim, dtype, cache_path, progcfg, tp, *, replaces_primary=False, shard_fn=None
):
    """Load `torch_weight` in the receiver-contiguous layout for `progcfg`'s matmul if `target` is enabled.

    Returns the loaded weight (registered for prefetch) or None (target off / unsupported / geometry mismatch).
    replaces_primary=True: the returned weight IS the decode weight (gate/up); the caller must load the interleaved
    weight itself when this returns None. Otherwise the caller registers it as the twin of its primary with
    attach_twin()."""
    if progcfg is None or not enabled(target) or _mesh_state(mesh) is None:
        return None
    k, n = per_device_kn(torch_weight.shape, shard_dim, tp)
    geom, why = geometry_for(progcfg, k, n, dtype)
    if geom is None:
        logger.warning(f"[weight_prefetch] {tag}: not prefetched ({why})")
        return None
    state = _mesh_state(mesh)
    if shard_fn is None:
        from models.demos.blackhole.qwen36.tt import tp_common as tpc

        shard_fn = tpc.shard_w
    rc_cache = f"{cache_path}.pf{geom.receivers}" if cache_path else None
    w = shard_fn(
        torch_weight,
        mesh,
        dim=shard_dim,
        memory_config=recv_contig_memory_config(geom, state.num_banks),
        cache_path=rc_cache,
        dtype=dtype,
    )
    if replaces_primary:
        if not register(mesh, tag, w, w, progcfg, geom):
            ttnn.deallocate(w)
            return None
        logger.info(f"[weight_prefetch] {tag}: prefetched, {geom.receivers} receivers, {geom.block_count} K-blocks")
    else:
        _PENDING[id(w)] = (tag, geom)
    return w


_PENDING = {}  # id(twin) -> (tag, geom) between load_prefetched and attach_twin


def attach_twin(mesh, primary, twin, progcfg):
    """Register a decode-only receiver-contiguous copy (from load_prefetched) for `primary`. Frees the twin and returns
    False when registration fails, so the primary keeps its plain DRAM path."""
    if twin is None:
        return False
    tag, geom = _PENDING.pop(id(twin))
    if not register(mesh, tag, primary, twin, progcfg, geom):
        ttnn.deallocate(twin)
        return False
    state = _mesh_state(mesh)
    extra = geom.k_tiles * geom.n_tiles * geom.tile_bytes
    state.twin_bytes += extra
    logger.info(
        f"[weight_prefetch] {tag}: prefetched via decode copy, {geom.receivers} receivers, {geom.block_count} "
        f"K-blocks, +{extra / 2**20:.1f} MiB/chip (decode copies so far {state.twin_bytes / 2**30:.2f} GiB/chip)"
    )
    return True


def lookup(weight, progcfg):
    e = _REGISTRY.get(id(weight))
    if e is None or e.primary is not weight:
        return None
    if progcfg is not e.progcfg:
        # The twin's layout and the GCB's receivers are tied to the registered program config.
        if e.twin is e.primary:
            raise RuntimeError(
                f"[weight_prefetch] {e.tag}: receiver-contiguous weight used with an unregistered program config; "
                f"unset {ENV} or route this call through the registered config"
            )
        return None
    return e


def linear(x, weight, program_config, **kwargs):
    """ttnn.linear(x, weight, program_config=...) that streams the weight through the Tensor prefetcher when it was
    registered for this program config (prefetch request + matmul from one call site, captured together)."""
    e = lookup(weight, program_config)
    if e is None:
        return ttnn.linear(x, weight, program_config=program_config, **kwargs)
    return ttnn.experimental.tensor_prefetcher_matmul.prefetch_and_linear(
        x, e.twin, global_cb=e.gcb, program_config=e.progcfg, **kwargs
    )


def is_registered(weight):
    return id(weight) in _REGISTRY


def _reset_for_tests():
    _MESHES.clear()
    _REGISTRY.clear()
    _PENDING.clear()


__all__ = [
    "ENV",
    "TARGETS",
    "parse_targets",
    "enabled",
    "plan_geometry",
    "geometry_for",
    "progcfg_problem",
    "load_prefetched",
    "attach_twin",
    "linear",
    "lookup",
    "is_registered",
]
