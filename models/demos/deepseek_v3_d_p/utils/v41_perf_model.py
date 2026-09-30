# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Theoretical per-chip performance and capacity model for DeepSeek-V4.1-Flash prefill.

Each operation of the semantic graph (``graph.md`` node ids N*, B*, F*, D*, V*) gets a lower bound for
one prefill chunk on one chip of an SP x TP mesh: total mathematical work divided evenly over the chips
that share it, mandatory DRAM traffic of the materialized graph (weights read once per chip, every
declared activation read once and written once), and the bottleneck-edge payload of its collectives.
Block types compose these operations under two scenarios. This is a lower bound, not a prediction of any
implementation; scheduling, grids and redundant traffic are deliberately ignored.

Distribution and placement follow the implemented V4.1 modules (``tt/v41/*``):
  * attention: row-parallel ``wq_a`` / ``wkv`` + TP all-reduce, column-parallel ``wq_b``, head->sequence
    all-to-all over TP around ``sparse_sdpa`` (every chip attends ``S/(sp*tp)`` queries with all heads),
    grouped ``wo_a`` + ``wo_b`` + TP reduce-scatter;
  * caches (``cache.V41PrefillState``): replicated on every chip; per chunk the window KV and the new
    compressed / index-K rows are SP all-gathered (no per-layer gather of the visible cache);
  * indexer (dev-spec D-C, C4 query split): every chip scores its ``S/(sp*tp)`` queries with all 32 heads
    against the replicated index-K; no score reduction;
  * MoE (``TtMoe`` via ``TtPrefillBlock._build_moe``): TP all-gather of the input, dispatch / combine along
    SP inside each TP column (a column holds ``experts / tp`` experts), reduce over top-k slots + TP
    reduce-scatter; dispatch buffer ``chunk * top-k`` rows (capacity factor = top-k, never drops tokens);
  * Engram (``engram.py``): packed 320 B rows; host table (rows uploaded per chunk) or device table
    (row-sharded over all chips, byte reduce-scatter SP -> TP).

Capability sources (Blackhole p150b; Galaxy chips assumed identical):
  * matrix engine 4096 FLOP/cycle/core at LoFi, divided by the fidelity phase count
    (``tech_reports/matrix_engine/matrix_engine.md``); elementwise 128 results/cycle/core (same).
  * 110 worker cores = compute_with_storage_grid 11x10, probed on this LoudBox (2026-09-28).
  * AICLK 1350 MHz: repository constant (``tests/didt/sweep_deepseek_v3_matmul_tune.py:55``); not a
    primary source — treat as an assumption.
  * DRAM 512 GB/s (``ttnn/core/operation.cpp:36-42``), 32 GB per chip (same comment).
  * Ethernet 25 GB/s per link per direction as currently enabled, 50 GB/s raw. Collectives are timed with the
    rate and fixed per-op latency each CCL op kind achieves standalone, calibrated on the LoudBox 2x4 at 2 links
    (``tests/v41/test_v41_ccl_calibration.py``, fit ``t = latency + edge_bytes / (rate x links)``, bead 8y7.9.3):
    all_gather_async 22 GB/s per link (88 % of the link), reduce_scatter_minimal_async 13, all_to_all_async_generic
    12 (TP) - 15 (SP) against the injection formula below; 12-20 us per op. Galaxy rings reuse the LoudBox rates
    (assumption: not calibrated there).
SFPU primitives (exp, rsqrt, sigmoid, softplus, sqrt, the router's top-6) have no documented throughput: they are
counted (elements), not timed. Exception: the indexer's top-k (``topk_large_indices``) is timed at its measured
per-chip element rate (``TOPK_ELEMENTS_PER_NS``, bead F10, LoudBox 2x4, 640 rows per chip), because at long context
it rivals the scoring FPU time. It runs on the Tensix math thread, so it adds to the op's FPU time
(``compute_ns = fpu_ns + sfpu_ns``). Rows are spread one per core: the rate is scaled by the row imbalance
ceil(rows / cores) / (rows / cores) (F10's convention). The top-k calls follow the implemented selection
(``tt/v41/indexer.py`` ``select``), including its width thresholds.

Scenarios (``SCENARIOS``, as DeepSeek-V3.2 / GLM ``tests/sparse_mla/test_sparse_mla_perf.py``): one 5120-token chunk
at start 0 (empty cache), 51200 (50k cached) and 512000 (0.5M cached). ``galaxy_slice`` scales chunk and start by
sp / 8 (V3.2's LoudBox per-chip Galaxy slice). V3.2 shards its caches block-cyclically over SP, so its slice keeps
Galaxy's per-chip cache depth; V4.1 replicates its caches on every chip, so the slice keeps Galaxy's per-chip query
count but only sp / 8 of its visible cache: it under-represents every cache-length term (indexer, top-k).

Bounds with two sides. Per op, ``dram_bytes`` is the optimistic (reuse-maximal) traffic; ``dram_bytes_cons`` (when
set) is the conservative traffic of an implementation without that reuse. Only ``sparse_attention`` has one: the
optimistic side reads the union of its queries' selected KV rows once, the conservative side reads each query's
``SLIDING_WINDOW + INDEX_TOPK`` (640) rows with no reuse across queries. A block's ``optimistic_ns`` =
max(sum compute, sum optimistic DRAM, sum CCL); ``conservative_ns`` = sum over ops of max(compute, conservative DRAM)
+ sum CCL. G2 target = max(2 x optimistic, conservative) (``g2_target_ns``).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from models.demos.deepseek_v3_d_p.reference.deepseek_v41_flash_config import DeepSeekV41FlashConfig as C
from models.demos.deepseek_v3_d_p.reference.deepseek_v41_flash_config import V41BlockType

# Physical bytes per element. Block-float tiles carry one shared exponent byte per 16 values. scaled_fp8 is the
# SCALED_FP8 KV row: 512 FP8 bytes + 4 fp32 scales = 528 B per 512 values (``cache.py``).
BYTES = {
    "fp32": 4.0,
    "bf16": 2.0,
    "bfp8": 1088 / 1024,
    "bfp4": 576 / 1024,
    "fp8": 1.0,
    "fp8_e8m0_32": 1 + 1 / 32,
    "scaled_fp8": 528 / 512,
}
FIDELITY_PHASES = {"LoFi": 1, "HiFi2": 2, "HiFi3": 3, "HiFi4": 4}
# Fidelity implied by the weight format of a matmul (in-tree convention; a §5 choice, not a law).
DEFAULT_FIDELITY = {"bfp4": "LoFi", "bfp8": "HiFi2", "bf16": "HiFi4", "fp32": "HiFi4"}
WINDOW_SLOT = 128  # carried window rows per KV tensor (``cache.WINDOW_SLOT``)
ENGRAM_PACKED_ROW_BYTES = 320  # ``engram.PACKED_WIDTH`` (160) uint16 containers: 256 FP8 values + 8 E8M0 scales, padded
TILE = 32
# ``topk_large_indices`` elements per ns per chip by k (F10 ``evidence/F10/targets_model.py``: k=512 from 3 points,
# linear in T, P = 128K-1M; k=2048 from one point at 256K, less certain). Measured, not a hardware capability.
TOPK_ELEMENTS_PER_NS = {512: 24.7, 2048: 7.1}
# Selection thresholds of the implemented indexer (``tt/v41/indexer.py``; duplicated because that module imports
# ttnn): the candidate source's own top-k runs among its candidate blocks above SUBSET_TOPK_MIN_WIDTH columns; a
# candidate index source masks its whole row up to DENSE_MASK_MAX_WIDTH and gathers its candidates' rows above it.
SUPERBLOCK = 32  # ``cache.SUPERBLOCK``
SUBSET_TOPK_MIN_WIDTH = 1 << 19
DENSE_MASK_MAX_WIDTH = 1 << 17
# Galaxy chunk starts of the V3.2 / GLM perf scenarios (``tests/sparse_mla/test_sparse_mla_perf.py``)
SCENARIOS = {"S0": 0, "S1": 51200, "S2": 512000}


@dataclass(frozen=True)
class Hardware:
    cores: int = 110
    clock_mhz: float = 1350.0
    matmul_flop_per_cycle_core: int = 4096  # LoFi
    eltwise_per_cycle_core: int = 128
    dram_bytes_per_ns: float = 512.0
    dram_capacity_bytes: float = 32e9
    # calibrated collective rate per link per direction and fixed latency per op, by collective kind (all_reduce =
    # reduce_scatter + all_gather, halo = all_gather); see the module docstring
    ccl: dict = field(default_factory=lambda: dict(BLACKHOLE_CCL))


@dataclass(frozen=True)
class CclRate:
    link_bytes_per_ns: float  # achieved per link, per direction (GB/s)
    latency_ns: float  # fixed per-op cost (launch, semaphores, first packet)


# LoudBox 2x4 fit at 2 links (``evidence/G2/ccl``): per-link rate = the 2-link fit / 2; latency = the fit's mean over
# the axes. all_to_all: TP 24.0 GB/s at 2 links (12.0 per link), SP 29.2 (14.6); the model takes the TP value (the
# attention a2a); the MoE dispatch / combine custom ops (modelled as SP all_to_all) were not calibrated standalone.
BLACKHOLE_CCL = {
    "all_gather": CclRate(link_bytes_per_ns=22.0, latency_ns=14_000.0),
    "reduce_scatter": CclRate(link_bytes_per_ns=13.0, latency_ns=18_000.0),
    "all_to_all": CclRate(link_bytes_per_ns=12.0, latency_ns=15_000.0),
}
BLACKHOLE_P150B = Hardware()


@dataclass(frozen=True)
class Layout:
    """SP shards tokens (mesh axis 0), TP shards hidden and heads (axis 1). ``ring`` marks wrapped axes."""

    sp: int
    tp: int
    links: int = 1
    sp_ring: bool = False
    tp_ring: bool = False

    @property
    def chips(self) -> int:
        return self.sp * self.tp


# links = ``tt/v41/ccl.fabric_num_links()``: every V4.1 collective (attention, norms, MoE) runs on 2 links on Blackhole
LOUDBOX_2X4 = Layout(sp=2, tp=4, links=2)
LOUDBOX_4X2 = Layout(sp=4, tp=2, links=2)
# The 32-chip Galaxy as a FABRIC_2D_TORUS_XY mesh (ring on both axes, ``tests/conftest.py`` torus-xy-8x4) viewed
# as SP8 x TP4 or SP4 x TP8 (both accepted by ``layout.V41MeshLayout``); 2 links = ``ccl.V41Collectives`` on
# Blackhole. The V4.1 collectives take each axis's topology from the opened fabric (``tt_ccl.per_axis_topology``),
# so both axes ring on the torus; ``sp_ring=tp_ring=False`` models an unwrapped (Linear) fabric.
GALAXY_8X4 = Layout(sp=8, tp=4, links=2, sp_ring=True, tp_ring=True)
GALAXY_4X8 = Layout(sp=4, tp=8, links=2, sp_ring=True, tp_ring=True)


@dataclass(frozen=True)
class Collective:
    kind: str  # all_gather | reduce_scatter | all_reduce | halo | all_to_all
    axis: str  # sp | tp | mesh
    shard_bytes: float  # per-chip shard (all_gather/reduce_scatter/all_reduce) or egress (halo/all_to_all)


@dataclass
class OpCost:
    node: str
    name: str
    matmul_flop: float = 0.0
    fidelity: str = "HiFi2"
    eltwise: float = 0.0
    sfpu: dict = field(default_factory=dict)
    dram_bytes: float = 0.0
    collectives: list = field(default_factory=list)
    # conservative DRAM bytes (no cross-query reuse); None = same as ``dram_bytes``
    dram_bytes_cons: float | None = None
    # timed top-k calls: (rows, width, k) per chip, at ``TOPK_ELEMENTS_PER_NS[k]``
    topk_calls: list = field(default_factory=list)
    fpu_ns: float = 0.0
    sfpu_ns: float = 0.0
    compute_ns: float = 0.0  # fpu_ns + sfpu_ns
    dram_ns: float = 0.0
    dram_cons_ns: float = 0.0
    ccl_ns: float = 0.0

    @property
    def roofline_ns(self) -> float:
        """Local compute/DRAM overlap (optimistic DRAM); collectives are a separate resource (see compose)."""
        return max(self.compute_ns, self.dram_ns)

    @property
    def roofline_cons_ns(self) -> float:
        """Local compute/DRAM overlap with the conservative DRAM traffic."""
        return max(self.compute_ns, self.dram_cons_ns)


@dataclass(frozen=True)
class Workload:
    """One prefill chunk: ``chunk`` tokens starting at absolute position ``start``."""

    chunk: int = 5120
    start: int = 0
    expert_dtype: str = "bfp8"
    dense_dtype: str = "bfp8"
    stream_dtype: str = "fp32"  # mHC residual streams between sublayers (in-tree V4 keeps fp32)
    kv_dtype: str = "bf16"  # compressed layers' KV tensors: "bf16" (BF16_RM) or "scaled_fp8" (SCALED_FP8)
    engram_tables: str = "host"  # "host" (rows uploaded per chunk) or "device" (row-sharded over all chips)

    @property
    def end(self) -> int:
        return self.start + self.chunk


def collective_ns(coll: Collective, layout: Layout, hw: Hardware) -> float:
    """Fixed per-op latency plus the bottleneck-edge payload over the calibrated per-direction link rate x links.

    Linear all-gather over n chips: the edge next to an end forwards n-1 shards in one direction;
    a bidirectional ring halves that. Reduce-scatter moves the same payload; all-reduce is a reduce-scatter
    then an all-gather (``V41Collectives.tp_all_reduce``). Halo sends one payload to a neighbor (all-gather rate).
    All-to-all (egress E per chip): E over the chip's ports on the axis (1 on a 2-chip line, else 2; the mesh
    variant counts both axes' ports). This injection bound ignores forwarding: the line-bisection payload
    (n/2)^2 E/(n-1) at the link rate is contradicted on TP 4 (measured 1.3-1.7x faster at 1 and 2 links), so
    the all-to-all rate is fitted against this formula instead.
    """
    if coll.axis == "mesh":
        n, ring = layout.chips, False
    else:
        n = layout.sp if coll.axis == "sp" else layout.tp
        ring = layout.sp_ring if coll.axis == "sp" else layout.tp_ring
    if n == 1 or coll.shard_bytes == 0:
        return 0.0

    def timed(kind: str, edge_bytes: float) -> float:
        rate = hw.ccl[kind]
        return rate.latency_ns + edge_bytes / (rate.link_bytes_per_ns * layout.links)

    edge_shards = (n - 1) / 2 if ring else (n - 1)
    if coll.kind in ("all_gather", "reduce_scatter"):
        return timed(coll.kind, edge_shards * coll.shard_bytes)
    if coll.kind == "all_reduce":
        return timed("reduce_scatter", edge_shards * coll.shard_bytes) + timed(
            "all_gather", edge_shards * coll.shard_bytes
        )
    if coll.kind == "halo":
        return timed("all_gather", coll.shard_bytes)
    if coll.kind == "all_to_all":
        if coll.axis == "mesh":
            ports = (2 if layout.sp > 1 else 0) + (2 if layout.tp > 1 else 0)
        else:
            ports = 1 if n == 2 and not ring else 2
        return timed("all_to_all", coll.shard_bytes / ports)
    raise ValueError(f"unknown collective kind {coll.kind}")


def _finish(op: OpCost, layout: Layout, hw: Hardware) -> OpCost:
    phases = FIDELITY_PHASES[op.fidelity]
    per_ns = hw.cores * hw.clock_mhz / 1000.0
    op.fpu_ns = (
        op.matmul_flop * phases / hw.matmul_flop_per_cycle_core + op.eltwise / hw.eltwise_per_cycle_core
    ) / per_ns
    op.sfpu_ns = sum(topk_ns(rows, width, k, hw) for rows, width, k in op.topk_calls)
    if op.topk_calls:
        op.sfpu["topk"] = op.sfpu.get("topk", 0) + sum(rows * width for rows, width, _ in op.topk_calls)
    op.compute_ns = op.fpu_ns + op.sfpu_ns
    op.dram_ns = op.dram_bytes / hw.dram_bytes_per_ns
    op.dram_cons_ns = (op.dram_bytes if op.dram_bytes_cons is None else op.dram_bytes_cons) / hw.dram_bytes_per_ns
    op.ccl_ns = sum(collective_ns(c, layout, hw) for c in op.collectives)
    return op


def topk_ns(rows: float, width: float, k: int, hw: Hardware = BLACKHOLE_P150B) -> float:
    """``topk_large_indices`` over ``rows`` x ``width`` on one chip: elements / measured rate (k <= 512 uses the
    k=512 rate), times the row imbalance of one row per core."""
    if rows <= 0 or width <= 0:
        return 0.0
    rate = TOPK_ELEMENTS_PER_NS[512 if k <= 512 else 2048]
    imbalance = -(-rows // hw.cores) / (rows / hw.cores)
    return rows * width / rate * imbalance


def _linear(node, name, tokens, k, n, wdtype, layout, *, k_sharded_tp, n_sharded_tp, act=("bf16", "bf16"), coll=()):
    """Per-chip cost of ``[tokens, k] @ [k, n]``: tokens are this chip's rows. Work and weights split
    over TP when either dimension is TP-sharded; otherwise every TP chip repeats the full product."""
    kk = k / layout.tp if k_sharded_tp else k
    nn = n / layout.tp if n_sharded_tp else n
    split = layout.tp if (k_sharded_tp or n_sharded_tp) else 1
    flop = 2 * tokens * k * n / split
    dram = k * n / split * BYTES[wdtype] + tokens * kk * BYTES[act[0]] + tokens * nn * BYTES[act[1]]
    return OpCost(
        node, name, matmul_flop=flop, fidelity=DEFAULT_FIDELITY[wdtype], dram_bytes=dram, collectives=list(coll)
    )


def block_ops(layer: int, w: Workload, layout: Layout, hw: Hardware = BLACKHOLE_P150B) -> list[OpCost]:
    """Operation costs of one backbone block (with its Engram at layers 1, 14) for one chunk on one chip."""
    btype = C.block_type(layer)
    ratio = C.compress_ratio(layer)
    s = w.chunk / layout.sp  # tokens per chip
    h, hc = C.EMB_SIZE, C.HC_MULT
    ht = h / layout.tp
    sd = BYTES[w.stream_dtype]
    ops: list[OpCost] = []

    def stream_bytes(copies):
        return s * copies * ht * sd

    # mHC around attention (B1, B2, B17) and FFN (B18, B23)
    for site in ("attn", "ffn"):
        mix = OpCost(
            "B1" if site == "attn" else "B18",
            f"hc_mixes_{site}",
            matmul_flop=2 * s * hc * h * (2 + hc) * hc / layout.tp,
            fidelity="HiFi4",
            eltwise=s * hc * ht * 2,
            sfpu={"rsqrt": s, "sigmoid": s * 2 * hc, "exp": s * hc * hc * C.HC_SINKHORN_ITERS},
            dram_bytes=stream_bytes(hc) + hc * h * (2 + hc) * hc / layout.tp * 4,
            collectives=[Collective("all_reduce", "tp", s * (2 + hc) * hc * 4)],
        )
        pre = OpCost(
            "B2" if site == "attn" else "B18",
            f"hc_pre_{site}",
            eltwise=s * hc * ht * 2,
            dram_bytes=stream_bytes(hc) + s * ht * 2,
        )
        post = OpCost(
            "B17" if site == "attn" else "B23",
            f"hc_post_{site}",
            eltwise=s * hc * ht * (2 * hc + 1),
            dram_bytes=stream_bytes(hc) * 2 + s * ht * 2,
        )
        norm = OpCost(
            "B3" if site == "attn" else "B19",
            f"{site}_norm",
            eltwise=s * ht * 3,
            sfpu={"rsqrt": s},
            dram_bytes=s * ht * 2 * 2,
            collectives=[Collective("all_gather", "tp", s * 32 * 4)],
        )
        ops += [mix, pre, norm]
        ops += _attention_ops(layer, btype, ratio, s, w, layout) if site == "attn" else _moe_ops(s, w, layout)
        ops.append(post)

    if layer in C.ENGRAM_LAYER_IDS:
        ops = engram_ops(s, w, layout) + ops
    return [_finish(op, layout, hw) for op in ops]


def _attention_ops(layer, btype, ratio, s, w: Workload, layout: Layout) -> list[OpCost]:
    h, d, heads, rd = C.EMB_SIZE, C.HEAD_DIM, C.NUM_ATTENTION_HEADS, C.QK_ROPE_HEAD_DIM
    q_lora, dense, idim = C.Q_LORA_RANK, w.dense_dtype, C.INDEX_HEAD_DIM
    kvb = BYTES[w.kv_dtype] if ratio else BYTES["bf16"]  # ratio-0 layers keep BF16 (cache.py)
    hl = heads / layout.tp
    qc = s / layout.tp  # queries per chip after the head->sequence all-to-all
    ops = [
        _linear(
            "B4",
            "wq_a",
            s,
            h,
            q_lora,
            dense,
            layout,
            k_sharded_tp=True,
            n_sharded_tp=False,
            coll=[Collective("all_reduce", "tp", s * q_lora * 2 / layout.tp)],
        ),
        OpCost("B4", "q_norm", eltwise=s * q_lora * 3, sfpu={"rsqrt": s}, dram_bytes=s * q_lora * 2 * 2),
        _linear("B4", "wq_b", s, q_lora, heads * d, dense, layout, k_sharded_tp=False, n_sharded_tp=True),
        OpCost("B4", "q_rope", eltwise=s * hl * rd * 3, dram_bytes=s * hl * d * 2 * 2),
        _linear(
            "B5",
            "wkv",
            s,
            h,
            d,
            dense,
            layout,
            k_sharded_tp=True,
            n_sharded_tp=False,
            coll=[Collective("all_reduce", "tp", s * d * 2 / layout.tp)],
        ),
        OpCost("B5", "kv_norm_rope_qdq", eltwise=s * d * 6, sfpu={"rsqrt": s}, dram_bytes=s * d * 2 * 2),
        # write_window: the chunk's window KV SP-gathered onto every chip (+ carry) in the layer's KV format
        OpCost(
            "B5",
            "window_kv_write",
            eltwise=w.chunk * d * (2 if kvb != 2 else 0),
            dram_bytes=w.chunk * d * 2 + (w.chunk + WINDOW_SLOT) * d * kvb + WINDOW_SLOT * d * kvb * 2,
            collectives=[Collective("all_gather", "sp", s * d * 2)],
        ),
    ]
    visible = 0
    if ratio:
        visible = w.end // ratio  # compressed rows visible to the last query of the chunk
        c_chip = s / ratio
        if btype in (V41BlockType.KV_INDEX_SOURCE, V41BlockType.CANDIDATE_SOURCE):
            wdt = "fp32" if ratio > 1 else "bf16"
            nproj = 2 if ratio > 1 else 1
            ops.append(
                OpCost(
                    "B6",
                    f"compressor_r{ratio}",
                    matmul_flop=2 * s * h * d * nproj / layout.tp,
                    fidelity="HiFi4",
                    eltwise=s * d * 3 * (ratio > 1) + c_chip * d * 3,
                    sfpu={"exp": s * d * (ratio > 1), "rsqrt": c_chip},
                    dram_bytes=h * d * nproj / layout.tp * BYTES[wdt] + s * h / layout.tp * 2 + c_chip * d * 2,
                    collectives=[Collective("all_reduce", "tp", s * d * nproj * 4 / layout.tp)],
                )
            )
            ops.append(
                OpCost(
                    "B7",
                    "index_keys",
                    matmul_flop=2 * c_chip * d * idim,  # wk replicated: every TP chip projects its SP rows
                    fidelity="HiFi4",
                    eltwise=c_chip * idim * 6,
                    dram_bytes=d * idim * 2 + c_chip * (d * 2 + idim * 2),
                )
            )
            # write_compressed: the chunk's new compressed KV and index-K rows SP-gathered onto every chip
            ops.append(
                OpCost(
                    "B8",
                    "compressed_kv_write",
                    eltwise=c_chip * (rd * 3 + d * 4),
                    dram_bytes=w.chunk / ratio * (d * 2 + d * kvb + idim * 2 * 2),
                    collectives=[Collective("all_gather", "sp", c_chip * (d + idim) * 2)],
                )
            )
        if btype in (V41BlockType.KV_INDEX_SOURCE, V41BlockType.CANDIDATE_SOURCE, V41BlockType.CANDIDATE_INDEX_SOURCE):
            ops += _indexer_ops(btype, s, visible, w, layout)
    selected = min(C.SLIDING_WINDOW, w.end) + (min(C.INDEX_TOPK, visible) if ratio else 0)
    a2a = [Collective("all_to_all", "tp", s * hl * d * 2 * (1 - 1 / layout.tp))] if layout.tp > 1 else []
    ops.append(OpCost("B14", "q_head_to_seq", dram_bytes=s * hl * d * 2 * 2, collectives=list(a2a)))
    # optimistic: every chip reads the KV rows its queries select once (their window span and the union of their
    # top-k); conservative: every query reads its own selected rows, no reuse across queries
    kv_rows = qc + C.SLIDING_WINDOW - 1 + (min(visible, qc * C.INDEX_TOPK) if ratio else 0)
    q_o_idx = qc * heads * d * 2 * 2 + qc * selected * 4
    ops.append(
        OpCost(
            "B14",
            "sparse_attention",
            matmul_flop=4 * qc * heads * selected * d,
            fidelity="HiFi2",
            eltwise=qc * heads * selected * 3,
            sfpu={"exp": qc * heads * selected},
            dram_bytes=q_o_idx + kv_rows * d * kvb,
            dram_bytes_cons=q_o_idx + qc * selected * d * kvb,
        )
    )
    ops.append(OpCost("B14", "o_seq_to_head", dram_bytes=s * hl * d * 2 * 2, collectives=list(a2a)))
    ops.append(OpCost("B15", "inverse_rope", eltwise=s * hl * rd * 3, dram_bytes=s * hl * d * 2 * 2))
    g_local = C.O_GROUPS / layout.tp
    ops.append(
        _linear(
            "B16",
            "wo_a",
            s,
            heads * d / C.O_GROUPS,
            C.O_GROUPS * C.O_LORA_RANK,
            dense,
            layout,
            k_sharded_tp=False,
            n_sharded_tp=True,
        )
    )
    ops[-1].matmul_flop = 2 * s * g_local * (heads * d / C.O_GROUPS) * C.O_LORA_RANK  # block-diagonal
    ops[-1].dram_bytes = (
        g_local * heads * d / C.O_GROUPS * C.O_LORA_RANK * BYTES[dense]
        + s * hl * d * 2
        + s * g_local * C.O_LORA_RANK * 2
    )
    ops.append(
        _linear(
            "B16",
            "wo_b",
            s,
            C.O_GROUPS * C.O_LORA_RANK,
            h,
            dense,
            layout,
            k_sharded_tp=True,
            n_sharded_tp=True,
            coll=[Collective("reduce_scatter", "tp", s * h * 2 / layout.tp)],
        )
    )
    return ops


def _indexer_ops(btype, s, visible, w: Workload, layout: Layout) -> list[OpCost]:
    """C4 query split: this chip's ``qi = S/(sp*tp)`` queries, all 32 heads, bf16 scores ``[qi, T]``."""
    ih, idim, q_lora, h = C.INDEX_N_HEADS, C.INDEX_HEAD_DIM, C.Q_LORA_RANK, C.EMB_SIZE
    qi = s / layout.tp
    t = max(-(-visible // TILE) * TILE, TILE)  # score width in whole tiles
    score = qi * t * 2
    ops = [
        _linear("B9", "index_wq_b", qi, q_lora, ih * idim, "bf16", layout, k_sharded_tp=False, n_sharded_tp=False),
        _linear(
            "B9",
            "index_weights_proj",
            s,
            h,
            ih,
            "bf16",
            layout,
            k_sharded_tp=True,
            n_sharded_tp=False,
            coll=[Collective("all_reduce", "tp", s * ih * 2 / layout.tp)],
        ),
        OpCost(
            "B9",
            "index_scores",
            matmul_flop=2 * qi * ih * idim * visible,
            fidelity="LoFi",  # FP4 q and k in the reference
            eltwise=qi * ih * visible * 3 + qi * t,
            # q once, index-K read + tiled copy + read, score written, visibility mask added (2 reads, 1 write)
            dram_bytes=qi * ih * idim * 2 + t * idim * 2 * 3 + score * 4,
        ),
    ]
    candidate_calls, row_calls = selection_topk_calls(btype, qi, t)
    if btype == V41BlockType.CANDIDATE_SOURCE:
        ops.append(
            OpCost(
                "B10",
                "candidate_select",
                eltwise=qi * t * 3,
                topk_calls=candidate_calls,
                # block max over 8 strided slices, block top-k, scatter, repeat_interleave, where -> published mask
                dram_bytes=score * 2 + score / C.CANDIDATE_BLOCK_SIZE * 4 + score * 2,
            )
        )
    if btype == V41BlockType.CANDIDATE_INDEX_SOURCE:
        ops.append(OpCost("B11", "candidate_mask", eltwise=qi * t, dram_bytes=score * 3))
    ops.append(OpCost("B12", "topk", topk_calls=row_calls, dram_bytes=score + qi * C.INDEX_TOPK * 4))
    return ops


def selection_topk_calls(btype, qi: float, t: int) -> tuple[list, list]:
    """The ``topk_large_indices`` calls (rows, width, k) of the implemented selection (``TtV41Indexer.select``) for
    ``qi`` query rows and a score ``t`` columns wide: (candidate-block calls of the candidate source, row top-k).

    Candidate blocks exist once a row has more than ``CANDIDATE_TOPK_BLOCKS`` blocks of 8. The candidate source
    ranks the block maxima of the whole row while it has at most 2048 superblocks of 32, else the superblock maxima
    and then the block maxima of the 2048 gathered superblocks (8192 blocks). Row top-k: direct over the row, except
    the candidate source above ``SUBSET_TOPK_MIN_WIDTH`` and a candidate index source above
    ``DENSE_MASK_MAX_WIDTH``, which rank the 2048 x 32 gathered columns."""
    kb, block, k = C.CANDIDATE_TOPK_BLOCKS, C.CANDIDATE_BLOCK_SIZE, C.INDEX_TOPK
    has_candidates = t // block > kb
    gathered = kb * SUPERBLOCK
    candidate_calls, row_calls = [], [(qi, t, k)]
    if btype == V41BlockType.CANDIDATE_SOURCE and has_candidates:
        nsb = t // SUPERBLOCK
        if nsb <= kb:
            candidate_calls = [(qi, t // block, kb)]
        else:
            candidate_calls = [(qi, nsb, kb), (qi, gathered // block, kb)]
        if t > SUBSET_TOPK_MIN_WIDTH:
            row_calls = [(qi, gathered, k)]
    if btype == V41BlockType.CANDIDATE_INDEX_SOURCE and has_candidates and t > DENSE_MASK_MAX_WIDTH:
        row_calls = [(qi, gathered, k)]
    return candidate_calls, row_calls


def moe_dispatch_rows(w: Workload, layout: Layout) -> int:
    """Rows of the per-chip dispatch buffer (``moe.init_helpers.compute_constants``): the column's chunk tokens
    times the capacity factor (top-k, ``tt/v41/moe.py``) plus one tile of alignment per further local expert."""
    experts_per_chip = C.NUM_ROUTED_EXPERTS // layout.chips
    return w.chunk * C.NUM_EXPERTS_PER_TOKEN + TILE * (experts_per_chip - 1)


def _moe_ops(s, w: Workload, layout: Layout) -> list[OpCost]:
    h, inter, e, k = C.EMB_SIZE, C.MOE_INTERMEDIATE_SIZE, C.NUM_ROUTED_EXPERTS, C.NUM_EXPERTS_PER_TOKEN
    picks_local = k / layout.tp  # expected top-k picks of a token that land in this chip's TP column
    routed_tokens = s * picks_local  # expert-token pairs computed per chip on average (balanced routing)
    egress = s * picks_local * (1 - 1 / layout.sp) * h * 2  # dispatch / combine along SP inside the column
    return [
        OpCost(
            "B20",
            "gate",
            matmul_flop=2 * s * h * e / layout.tp,
            fidelity="HiFi4",
            sfpu={"softplus": s * e / layout.tp, "sqrt": s * e / layout.tp, "topk_router": s * e / layout.tp},
            dram_bytes=h * e * 2 / layout.tp + s * h / layout.tp * 2 + s * e * 4 / layout.tp,
            collectives=[Collective("all_reduce", "tp", s * e * 4 / layout.tp)],
        ),
        OpCost(
            "B21",
            "moe_input_gather",
            dram_bytes=s * h / layout.tp * 2 + s * h * 2,
            collectives=[Collective("all_gather", "tp", s * h / layout.tp * 2)],
        ),
        OpCost(
            "B21",
            "dispatch",
            dram_bytes=routed_tokens * h * 2 * 2,
            collectives=[Collective("all_to_all", "sp", egress)],
        ),
        OpCost(
            "B21",
            "routed_experts",
            matmul_flop=2 * routed_tokens * h * inter * 3,
            fidelity=DEFAULT_FIDELITY[w.expert_dtype],
            eltwise=routed_tokens * inter * 4,
            sfpu={"sigmoid": routed_tokens * inter},
            dram_bytes=e / layout.chips * 3 * h * inter * BYTES[w.expert_dtype] + routed_tokens * h * 2 * 2,
        ),
        OpCost(
            "B21",
            "combine",
            eltwise=routed_tokens * h,
            dram_bytes=routed_tokens * h * 2 * 2,
            collectives=[Collective("all_to_all", "sp", egress)],
        ),
        OpCost(
            "B21",
            "routed_reduce",
            eltwise=s * k * h,
            dram_bytes=s * k * h * 2 + s * h * 2,
            collectives=[Collective("reduce_scatter", "tp", s * h * 2 / layout.tp)],
        ),
        OpCost(
            "B22",
            "shared_expert",
            matmul_flop=2 * s * h * inter * 3 / layout.tp,
            fidelity=DEFAULT_FIDELITY[w.dense_dtype],
            eltwise=s * inter * 4 / layout.tp,
            sfpu={"sigmoid": s * inter / layout.tp},
            dram_bytes=3 * h * inter / layout.tp * BYTES[w.dense_dtype] + s * h * 2 + s * h / layout.tp * 2,
            collectives=[Collective("reduce_scatter", "tp", s * h * 2 / layout.tp)],
        ),
    ]


def engram_ops(s, w: Workload, layout: Layout) -> list[OpCost]:
    """Engram at one layer (N3-N5) for ``s`` tokens per chip. N3 lookup per ``w.engram_tables``: host table
    (packed rows arrive over the host link, 1/chips of the lookups per chip; host time not modeled) or device
    table (every chip gathers all lookups from its row shard, byte sums reduce-scattered SP then TP); both
    decode on device and all-gather the rows over TP."""
    cols, hd, h, hc = (C.ENGRAM_MAX_NGRAM_SIZE - 1) * C.ENGRAM_N_HEADS, C.ENGRAM_HEAD_DIM, C.EMB_SIZE, C.HC_MULT
    lookups = w.chunk * cols
    per_chip = lookups / layout.chips
    decode = per_chip * ENGRAM_PACKED_ROW_BYTES * 2 + per_chip * hd * 2  # low/high bytes read, bf16 rows written
    gather = [Collective("all_gather", "tp", per_chip * hd * 2)]
    if w.engram_tables == "device":
        byte_sums = lookups * ENGRAM_PACKED_ROW_BYTES * 2  # low | high bytes as bf16, per chip before the scatter
        lookup = OpCost(
            "N3",
            "engram_lookup_device",
            eltwise=lookups * ENGRAM_PACKED_ROW_BYTES * 3,
            dram_bytes=lookups * ENGRAM_PACKED_ROW_BYTES * 2 + byte_sums * 3 + decode,
            collectives=[
                Collective("reduce_scatter", "sp", byte_sums / layout.sp),
                Collective("reduce_scatter", "tp", byte_sums / layout.chips),
            ]
            + gather,
        )
    else:
        lookup = OpCost("N3", "engram_lookup_host", eltwise=per_chip * hd * 4, dram_bytes=decode, collectives=gather)
    wkv = _linear(
        "N4", "engram_wkv", s, cols * hd, h * (hc + 1), w.dense_dtype, layout, k_sharded_tp=False, n_sharded_tp=True
    )
    gate = OpCost(
        "N5",
        "engram_gate_add",
        eltwise=s * hc * h / layout.tp * 8,
        sfpu={"rsqrt": 2 * s * hc, "sqrt": s * hc, "sigmoid": s * hc},
        dram_bytes=s * hc * h / layout.tp * (2 * BYTES[w.stream_dtype]) + s * h * (hc + 1) / layout.tp * 2,
        collectives=[Collective("all_gather", "tp", s * 32 * 4)],
    )
    return [lookup, wkv, gate]


def engram_host_upload_bytes_per_chunk(chunk: int) -> float:
    """Host -> device bytes per chunk per Engram layer with the host table: every lookup's packed row once."""
    return chunk * (C.ENGRAM_MAX_NGRAM_SIZE - 1) * C.ENGRAM_N_HEADS * ENGRAM_PACKED_ROW_BYTES


@dataclass
class BlockEstimate:
    layer: int
    block_type: str
    compute_ns: float
    dram_ns: float
    ccl_ns: float
    conservative_ns: float  # every op serialized: sum(max(compute, conservative dram)) + sum(ccl)
    optimistic_ns: float  # compute, DRAM and links each saturated in parallel across the block
    sfpu: dict
    ops: list
    sfpu_ns: float = 0.0  # timed top-k, included in compute_ns
    dram_cons_ns: float = 0.0

    @property
    def target_ns(self) -> float:
        return g2_target_ns(self)


def compose(ops: list[OpCost], layer: int = -1, block_type: str = "") -> BlockEstimate:
    compute = sum(o.compute_ns for o in ops)
    dram = sum(o.dram_ns for o in ops)
    ccl = sum(o.ccl_ns for o in ops)
    sfpu: dict = {}
    for o in ops:
        for key, value in o.sfpu.items():
            sfpu[key] = sfpu.get(key, 0) + value
    return BlockEstimate(
        layer=layer,
        block_type=block_type,
        compute_ns=compute,
        dram_ns=dram,
        ccl_ns=ccl,
        conservative_ns=sum(o.roofline_cons_ns for o in ops) + ccl,
        optimistic_ns=max(compute, dram, ccl),
        sfpu=sfpu,
        ops=ops,
        sfpu_ns=sum(o.sfpu_ns for o in ops),
        dram_cons_ns=sum(o.dram_cons_ns for o in ops),
    )


def g2_target_ns(e: BlockEstimate) -> float:
    """G2 layer target: max(2 x optimistic, conservative)."""
    return max(2 * e.optimistic_ns, e.conservative_ns)


def compose_block(layer: int, w: Workload, layout: Layout, hw: Hardware = BLACKHOLE_P150B) -> BlockEstimate:
    return compose(block_ops(layer, w, layout, hw), layer, C.block_type(layer).value)


# --- scenarios, op groups, full model ----------------------------------------------------------------------------
# Op group of every block op (by op name), for the per-segment view of a layer
OP_GROUPS = {
    "mhc": ("hc_mixes_attn", "hc_pre_attn", "hc_post_attn", "hc_mixes_ffn", "hc_pre_ffn", "hc_post_ffn"),
    "norms": ("attn_norm", "ffn_norm"),
    "attn.proj": ("wq_a", "q_norm", "wq_b", "q_rope", "wkv", "kv_norm_rope_qdq", "inverse_rope", "wo_a", "wo_b"),
    "attn.kv_write": ("window_kv_write", "compressor_r1", "compressor_r2", "index_keys", "compressed_kv_write"),
    "attn.a2a": ("q_head_to_seq", "o_seq_to_head"),
    "attn.sparse_sdpa": ("sparse_attention",),
    "attn.indexer": ("index_wq_b", "index_weights_proj", "index_scores", "candidate_select", "candidate_mask", "topk"),
    "moe.gate": ("gate",),
    "moe.dispatch_combine": ("moe_input_gather", "dispatch", "combine", "routed_reduce"),
    "moe.routed": ("routed_experts",),
    "moe.shared": ("shared_expert",),
    "engram": ("engram_lookup_host", "engram_lookup_device", "engram_wkv", "engram_gate_add"),
}
GROUP_OF = {name: group for group, names in OP_GROUPS.items() for name in names}
CHECKPOINT_LAYERS = (0, 2, 3, 20, 21, 24)  # the measured sharing schedule (one layer of every block type)


def scenario_workload(name: str, layout: Layout, *, galaxy_slice: bool = False, **kw) -> Workload:
    """The 5120-token chunk of scenario ``name`` (``SCENARIOS``); ``galaxy_slice`` scales chunk and start by sp / 8
    (V3.2's per-chip Galaxy slice on a smaller box, e.g. LoudBox 2x4: chunk 1280 at 0 / 12800 / 128000)."""
    start, chunk = SCENARIOS[name], 5120
    if galaxy_slice:
        assert (chunk * layout.sp) % 8 == 0 and (start * layout.sp) % 8 == 0
        chunk, start = chunk * layout.sp // 8, start * layout.sp // 8
    return Workload(chunk=chunk, start=start, **kw)


def group_estimates(ops: list[OpCost]) -> dict[str, BlockEstimate]:
    """Each op group's own composition (optimistic / conservative over the group's ops only)."""
    out = {}
    for group in OP_GROUPS:
        sel = [o for o in ops if GROUP_OF.get(o.name) == group]
        if sel:
            out[group] = compose(sel, block_type=group)
    unknown = [o.name for o in ops if o.name not in GROUP_OF]
    assert not unknown, f"ops without a group: {unknown}"
    return out


def chunk_estimate(
    w: Workload, layout: Layout, *, layers=None, dspark: bool = True, hw: Hardware = BLACKHOLE_P150B
) -> dict:
    """One chunk through ``layers`` (default: all backbone layers, i.e. the full model; + DSpark seeding): blocks
    run one after another, each composed on its own. Returns per-layer estimates, per block type (count, per-layer
    and summed optimistic / conservative / target, ns) and per op group (summed over layers, ns)."""
    layers = list(range(C.NUM_LAYERS)) if layers is None else list(layers)
    blocks = [compose_block(layer, w, layout, hw) for layer in layers]
    tail = compose(dspark_prefill_ops(w, layout, hw) if dspark else [], block_type="dspark_seed")
    by_type: dict = {}
    for b in blocks:
        t = by_type.setdefault(
            b.block_type, {"layers": [], "optimistic_ns": 0.0, "conservative_ns": 0.0, "target_ns": 0.0}
        )
        t["layers"].append(b.layer)
        t["optimistic_ns"] += b.optimistic_ns
        t["conservative_ns"] += b.conservative_ns
        t["target_ns"] += b.target_ns
    groups: dict = {}
    for b in blocks:
        for group, e in group_estimates(b.ops).items():
            g = groups.setdefault(group, {"optimistic_ns": 0.0, "conservative_ns": 0.0, "sfpu_ns": 0.0})
            g["optimistic_ns"] += e.optimistic_ns
            g["conservative_ns"] += e.conservative_ns
            g["sfpu_ns"] += e.sfpu_ns
    if dspark:
        groups["dspark_seed"] = {
            "optimistic_ns": tail.optimistic_ns,
            "conservative_ns": tail.conservative_ns,
            "sfpu_ns": 0.0,
        }
    total = {
        "optimistic_ns": sum(b.optimistic_ns for b in blocks) + tail.optimistic_ns,
        "conservative_ns": sum(b.conservative_ns for b in blocks) + tail.conservative_ns,
        "target_ns": sum(b.target_ns for b in blocks) + (g2_target_ns(tail) if dspark else 0.0),
    }
    return {"blocks": blocks, "tail": tail, "by_type": by_type, "groups": groups, "total": total}


def prefill_estimate(
    prompt_tokens: int, w: Workload, layout: Layout, *, dspark: bool = True, hw: Hardware = BLACKHOLE_P150B
) -> dict:
    """Full prefill of ``prompt_tokens`` in chunks of ``w.chunk`` (the last one padded): per-chunk block
    compositions summed, i.e. blocks run one after another and each block's scenarios apply within it."""
    chunks = -(-prompt_tokens // w.chunk)
    optimistic = conservative = 0.0
    per_chunk = []
    for i in range(chunks):
        wc = replace(w, start=i * w.chunk)
        blocks = [compose_block(layer, wc, layout, hw) for layer in range(C.NUM_LAYERS)]
        tail = (dspark_prefill_ops(wc, layout, hw) if dspark else []) + (
            final_ops(wc, layout, hw) if i == chunks - 1 else []
        )
        rest = compose(tail)
        o = sum(b.optimistic_ns for b in blocks) + rest.optimistic_ns
        c = sum(b.conservative_ns for b in blocks) + rest.conservative_ns
        per_chunk.append((wc.start, o, c))
        optimistic += o
        conservative += c
    return {"chunks": chunks, "optimistic_ns": optimistic, "conservative_ns": conservative, "per_chunk": per_chunk}


# --- capacity -------------------------------------------------------------------------------------------------
# Placement of each weight component (``tt/v41/*``): "tp" = split over TP, replicated over SP; "chips" = split
# over every chip; "replicated" = full copy per chip.
WEIGHT_PLACEMENT = {
    "attention": "tp",
    "mhc": "tp",
    "gate": "tp",
    "shared_expert": "tp",
    "routed_experts": "chips",
    "compressor": "tp",
    "index_keys": "replicated",
    "index_wq_b": "replicated",
    "index_weights_proj": "tp",
    "engram_wkv": "tp",
    "embedding": "tp",
    "lm_head": "tp",
    "dspark_main_proj": "tp",
    "dspark_wkv": "replicated",
}


def weight_bytes_per_layer(layer: int, w: Workload) -> dict:
    """Device bytes of one backbone layer's weights, by component (whole model, before placement)."""
    h, d, q_lora, idim = C.EMB_SIZE, C.HEAD_DIM, C.Q_LORA_RANK, C.INDEX_HEAD_DIM
    dense = BYTES[w.dense_dtype]
    attn = (
        h * q_lora
        + q_lora * C.NUM_ATTENTION_HEADS * d
        + h * d
        + C.NUM_ATTENTION_HEADS * d * C.O_LORA_RANK
        + C.O_GROUPS * C.O_LORA_RANK * h
    ) * dense
    out = {
        "attention": attn,
        "mhc": 2 * (2 + C.HC_MULT) * C.HC_MULT * C.HC_MULT * h * 4,
        "gate": C.NUM_ROUTED_EXPERTS * h * 2,
        "shared_expert": 3 * h * C.MOE_INTERMEDIATE_SIZE * dense,
        "routed_experts": C.NUM_ROUTED_EXPERTS * 3 * h * C.MOE_INTERMEDIATE_SIZE * BYTES[w.expert_dtype],
    }
    ratio = C.compress_ratio(layer)
    if layer in C.KV_SOURCE_LAYERS:
        out["compressor"] = h * d * (2 if ratio > 1 else 1) * (4 if ratio > 1 else 2)
        out["index_keys"] = d * idim * 2
    if layer in C.INDEX_SOURCE_LAYERS:
        out["index_wq_b"] = q_lora * C.INDEX_N_HEADS * idim * 2
        out["index_weights_proj"] = h * C.INDEX_N_HEADS * 2
    if layer in C.ENGRAM_LAYER_IDS:
        cols = (C.ENGRAM_MAX_NGRAM_SIZE - 1) * C.ENGRAM_N_HEADS
        out["engram_wkv"] = cols * C.ENGRAM_HEAD_DIM * h * (C.HC_MULT + 1) * dense
    return out


def model_weight_bytes(w: Workload, *, dspark: bool = True) -> dict:
    """Non-layer weights: bf16 embedding and LM head, DSpark prefill seeding (main_proj, 3 x wkv; bf16)."""
    out = {"embedding": C.VOCAB_SIZE * C.EMB_SIZE * 2, "lm_head": C.VOCAB_SIZE * C.EMB_SIZE * 2}
    if dspark:
        out["dspark_main_proj"] = len(C.DSPARK_TARGET_LAYER_IDS) * C.EMB_SIZE * C.EMB_SIZE * 2
        out["dspark_wkv"] = C.NUM_DSPARK_LAYERS * C.EMB_SIZE * C.HEAD_DIM * 2
    return out


def _placed(nbytes: float, placement: str, layout: Layout) -> float:
    return nbytes / {"tp": layout.tp, "chips": layout.chips, "replicated": 1}[placement]


def engram_table_bytes(layers: list[int] | None = None) -> float:
    """Packed Engram rows (``ENGRAM_PACKED_ROW_BYTES`` each) of the tables of ``layers`` (all by default):
    the host table's RAM and the device tables' total before sharding."""
    layers = list(C.ENGRAM_LAYER_IDS) if layers is None else layers
    rows = [n for layer, n in zip(C.ENGRAM_LAYER_IDS, C.ENGRAM_NUM_EMBEDDINGS) if layer in layers]
    return sum(n * ENGRAM_PACKED_ROW_BYTES for n in rows)


def engram_device_table_bytes_per_chip(layers: list[int], layout: Layout) -> float:
    """``TtV41EngramTable``: each chip holds ceil(rows / chips) packed rows plus one zero row, per table."""
    rows = [n for layer, n in zip(C.ENGRAM_LAYER_IDS, C.ENGRAM_NUM_EMBEDDINGS) if layer in layers]
    return sum((-(-n // layout.chips) + 1) * ENGRAM_PACKED_ROW_BYTES for n in rows)


def kv_cache_bytes(max_seq_len: int, w: Workload, layers: list[int] | None = None) -> float:
    """Per-chip bytes of ``V41PrefillState`` (replicated on every chip) for requests up to ``max_seq_len``: per KV
    source one KV tensor (window slot + chunk scratch + compressed rows, in the KV format) and its bf16 index-K;
    one bf16 ratio-0 scratch tensor if any SWA-only layer is present; one window carry per layer."""
    layers = list(range(C.NUM_LAYERS)) if layers is None else layers
    d, idim = C.HEAD_DIM, C.INDEX_HEAD_DIM

    def row(ratio):
        return d * (BYTES[w.kv_dtype] if ratio else BYTES["bf16"])

    total = 0.0
    for layer in layers:
        ratio = C.compress_ratio(layer)
        if layer in C.KV_SOURCE_LAYERS:
            compressed = max_seq_len // ratio
            total += (WINDOW_SLOT + w.chunk + compressed) * row(ratio) + compressed * idim * 2
        total += WINDOW_SLOT * row(ratio)
    if any(C.compress_ratio(layer) == 0 for layer in layers):
        total += (WINDOW_SLOT + w.chunk) * row(0)
    return total


def activation_bytes_per_chip(w: Workload, layout: Layout, max_seq_len: int) -> dict:
    """Largest transient per-chip DRAM of one chunk, by phase (phases do not overlap; peak = streams + max).

    streams: fp32 mHC streams in / out / mixed (3 live copies). moe: gathered input, dispatch buffer +
    metadata, routed expert output (same rows), combine output ``[S/sp, k, H]``, and one expert's
    intermediates at the worst per-expert count (the whole chunk). indexer (at a full ``max_seq_len`` context,
    ratio-1 index sources): index-K tiled copy, bf16 score, visibility mask, masked score and the published
    candidate mask ``[S/(sp*tp), T]``. engram: the device-table lookup's gathered rows and byte tensors (host
    table: the chunk's packed upload and decoded rows)."""
    s, h, k = w.chunk / layout.sp, C.EMB_SIZE, C.NUM_EXPERTS_PER_TOKEN
    rows = moe_dispatch_rows(w, layout)
    streams = 3 * s * C.HC_MULT * h / layout.tp * BYTES[w.stream_dtype]
    moe = s * h * 2 + rows * (h * 2 + 3 * 4) + rows * h * 2 + s * k * h * 2 + w.chunk * C.MOE_INTERMEDIATE_SIZE * 2 * 3
    qi = s / layout.tp
    t = max_seq_len  # ratio-1 visible rows at the end of the context
    indexer = t * C.INDEX_HEAD_DIM * 2 + 4 * qi * t * 2
    cols = (C.ENGRAM_MAX_NGRAM_SIZE - 1) * C.ENGRAM_N_HEADS
    lookups = w.chunk * cols
    if w.engram_tables == "device":
        engram = lookups * ENGRAM_PACKED_ROW_BYTES * (1 + 2 * 2 + 2)  # gathered uint16, low/high int32, bf16 sums
    else:
        engram = lookups / layout.chips * ENGRAM_PACKED_ROW_BYTES + lookups / layout.sp * C.ENGRAM_HEAD_DIM * 2
    engram += s * cols * C.ENGRAM_HEAD_DIM * 2 + s * (C.HC_MULT + 1) * h / layout.tp * 2  # rows + wkv output
    out = {"streams": streams, "moe": moe, "indexer": indexer, "engram": engram}
    out["peak"] = streams + max(moe, indexer, engram)
    return out


def capacity_per_chip(
    layers: list[int],
    w: Workload,
    layout: Layout,
    *,
    context_tokens: int,
    dspark: bool = True,
) -> dict:
    """Per-chip DRAM bytes by component for ``layers`` (+ embedding, head, DSpark seeding), Engram tables per
    ``w.engram_tables``, state for ``context_tokens`` and the peak chunk transient."""
    per: dict = {}
    for layer in layers:
        for name, nbytes in weight_bytes_per_layer(layer, w).items():
            per[name] = per.get(name, 0.0) + _placed(nbytes, WEIGHT_PLACEMENT[name], layout)
    for name, nbytes in model_weight_bytes(w, dspark=dspark).items():
        per[name] = _placed(nbytes, WEIGHT_PLACEMENT[name], layout)
    per["weights"] = sum(per.values())
    engram_layers = [layer for layer in layers if layer in C.ENGRAM_LAYER_IDS]
    per["engram_tables"] = (
        engram_device_table_bytes_per_chip(engram_layers, layout) if w.engram_tables == "device" else 0.0
    )
    per["caches"] = kv_cache_bytes(context_tokens, w, layers)
    if dspark:
        per["caches"] += C.NUM_DSPARK_LAYERS * C.SLIDING_WINDOW * C.HEAD_DIM * 2  # bf16 DSpark rings
    per["activations"] = activation_bytes_per_chip(w, layout, context_tokens)["peak"]
    per["total"] = per["weights"] + per["engram_tables"] + per["caches"] + per["activations"]
    return per


def final_ops(w: Workload, layout: Layout, hw: Hardware = BLACKHOLE_P150B) -> list[OpCost]:
    """F1 final collapse and F2 norm over the chunk; F3 LM head (bf16, vocab split over TP) on the last token."""
    s, ht, hc = w.chunk / layout.sp, C.EMB_SIZE / layout.tp, C.HC_MULT
    ops = [
        OpCost(
            "F1", "final_collapse", eltwise=s * hc * ht * 2, dram_bytes=s * hc * ht * BYTES[w.stream_dtype] + s * ht * 2
        ),
        OpCost(
            "F2",
            "final_norm",
            eltwise=s * ht * 3,
            sfpu={"rsqrt": s},
            dram_bytes=s * ht * 2 * 2,
            collectives=[Collective("all_gather", "tp", s * 32 * 4)],
        ),
        # one token on the SP row that holds it; the hidden is gathered over TP, each TP chip owns vocab / tp
        OpCost(
            "F3",
            "lm_head_last_token",
            matmul_flop=2 * C.EMB_SIZE * C.VOCAB_SIZE / layout.tp,
            fidelity="HiFi4",
            dram_bytes=C.EMB_SIZE * C.VOCAB_SIZE / layout.tp * 2 + C.VOCAB_SIZE / layout.tp * 4,
            collectives=[Collective("all_gather", "tp", ht * 2)],
        ),
    ]
    return [_finish(op, layout, hw) for op in ops]


def dspark_prefill_ops(w: Workload, layout: Layout, hw: Hardware = BLACKHOLE_P150B) -> list[OpCost]:
    """N6 taps (3 layers, whole chunk, SP-gathered) and D1/D2 over the chunk's last ``min(128, chunk)`` rows:
    main_proj bf16 15360 -> 5120 (row-parallel, TP all-reduce) + main_norm, then per DSpark layer the replicated
    bf16 wkv 5120 -> 512, kv_norm, RoPE, FP8 QDQ into the ring. Runs every chunk (``dspark.seed``)."""
    s, h, d, hc = w.chunk / layout.sp, C.EMB_SIZE, C.HEAD_DIM, C.HC_MULT
    rows = min(C.SLIDING_WINDOW, w.chunk)
    taps = len(C.DSPARK_TARGET_LAYER_IDS)
    ops = [
        OpCost(
            "N6",
            "dspark_taps",
            eltwise=taps * s * hc * h / layout.tp,
            dram_bytes=taps * s * (hc * BYTES[w.stream_dtype] + 2) * h / layout.tp + taps * w.chunk * h / layout.tp * 2,
            collectives=[Collective("all_gather", "sp", taps * s * h / layout.tp * 2)],
        ),
        _linear(
            "D1",
            "main_proj",
            rows,
            taps * h,
            h,
            "bf16",
            layout,
            k_sharded_tp=True,
            n_sharded_tp=False,
            act=("bf16", "fp32"),
            coll=[Collective("all_reduce", "tp", rows * h * 4 / layout.tp)],
        ),
        OpCost("D1", "main_norm", eltwise=rows * h * 3, sfpu={"rsqrt": rows}, dram_bytes=rows * h * 2 * 2),
    ]
    for i in range(C.NUM_DSPARK_LAYERS):
        ops.append(_linear("D2", f"dspark{i}_wkv", rows, h, d, "bf16", layout, k_sharded_tp=False, n_sharded_tp=False))
        ops.append(
            OpCost(
                "D2",
                f"dspark{i}_kv_norm_rope_qdq",
                eltwise=rows * d * 6,
                sfpu={"rsqrt": rows},
                dram_bytes=rows * d * (2 + 2),
            )
        )
    return [_finish(op, layout, hw) for op in ops]


def vision_ops(
    image_tokens: int, layout: Layout, hw: Hardware = BLACKHOLE_P150B, weight_dtype: str = "bf16"
) -> list[OpCost]:
    """V1-V3 for one image of ``image_tokens`` aligner outputs (= patches / 9), work split over all chips.

    ViT: patch_embed 588 -> 1024, 32 x (RMSNorm, qkv 1024 -> 3072, bidirectional attention 16 x 64,
    wo, SwiGLU 1024 -> 2 x 2816 -> 1024), aligner 9216 -> 5120 -> 5120. V5 merge is a row copy (negligible).
    """
    p = image_tokens * C.VISION_DOWNSAMPLE_RATIO**2
    dim, inter, layers = C.VISION_DIM, C.VISION_INTER_DIM, C.VISION_N_LAYERS
    n = layout.chips
    wb = BYTES[weight_dtype]
    fid = DEFAULT_FIDELITY[weight_dtype]
    per_layer_w = dim * 3 * dim + dim * dim + dim * 2 * inter + inter * dim
    ops = [
        OpCost(
            "V1",
            "patch_embed",
            matmul_flop=2 * p * 588 * dim / n,
            fidelity=fid,
            dram_bytes=588 * dim * wb + p * (588 + dim) * 2 / n,
        ),
        OpCost(
            "V2",
            "vit_blocks",
            matmul_flop=layers * (2 * p * per_layer_w + 4 * p * p * dim) / n,
            fidelity=fid,
            eltwise=layers * p * dim * 12 / n,
            sfpu={"exp": layers * C.VISION_N_HEADS * p * p / n, "sigmoid": layers * p * inter / n},
            dram_bytes=layers * per_layer_w * wb + layers * p * dim * 2 * 8 / n,
        ),
        OpCost(
            "V3",
            "aligner",
            matmul_flop=2 * image_tokens * (9 * dim * C.EMB_SIZE + C.EMB_SIZE * C.EMB_SIZE) / n,
            fidelity=fid,
            sfpu={"gelu": image_tokens * C.EMB_SIZE / n},
            dram_bytes=(9 * dim * C.EMB_SIZE + C.EMB_SIZE * C.EMB_SIZE) * wb
            + image_tokens * (9 * dim + 2 * C.EMB_SIZE) * 2 / n,
        ),
    ]
    return [_finish(op, layout, hw) for op in ops]
