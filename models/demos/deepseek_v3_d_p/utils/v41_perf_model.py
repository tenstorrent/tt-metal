# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Theoretical per-chip performance and capacity model for DeepSeek-V4.1-Flash prefill.

Each operation of the semantic graph (``graph.md`` node ids N*, B*, F*, D*, V*) gets a lower bound for
one prefill chunk on one chip of an SP x TP mesh: total mathematical work divided evenly over all
chips, mandatory DRAM traffic of the materialized graph (weights read once per chip, every declared
activation read once and written once), and the bottleneck-edge payload of its collectives. Block
types compose these operations under two scenarios. This is a lower bound, not a prediction of any
implementation; scheduling, grids and redundant traffic are deliberately ignored.

Capability sources (Blackhole p150b):
  * matrix engine 4096 FLOP/cycle/core at LoFi, divided by the fidelity phase count
    (``tech_reports/matrix_engine/matrix_engine.md``); elementwise 128 results/cycle/core (same).
  * 110 worker cores = compute_with_storage_grid 11x10, probed on this LoudBox (2026-09-28).
  * AICLK 1350 MHz: repository constant (``tests/didt/sweep_deepseek_v3_matmul_tune.py:55``); not a
    primary source — treat as an assumption.
  * DRAM 512 GB/s (``ttnn/core/operation.cpp:36-42``), 32 GB per chip (same comment).
  * Ethernet 25 GB/s per link per direction as currently enabled, 50 GB/s raw; 619 ns per Fabric2D hop
    (``ttnn/cpp/ttnn/operations/ccl/ccl_common.cpp:2003-2026``; the hop latency is an empirical estimate).
SFPU primitives (exp, rsqrt, sigmoid, softplus, sqrt, topk) have no documented throughput: they are
counted, not timed, so compute time is FPU time only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from models.demos.deepseek_v3_d_p.reference.deepseek_v41_flash_config import DeepSeekV41FlashConfig as C
from models.demos.deepseek_v3_d_p.reference.deepseek_v41_flash_config import V41BlockType

# Physical bytes per element. Block-float tiles carry one shared exponent byte per 16 values.
BYTES = {"fp32": 4.0, "bf16": 2.0, "bfp8": 1088 / 1024, "bfp4": 576 / 1024, "fp8": 1.0, "fp8_e8m0_32": 1 + 1 / 32}
FIDELITY_PHASES = {"LoFi": 1, "HiFi2": 2, "HiFi3": 3, "HiFi4": 4}
# Fidelity implied by the weight format of a matmul (in-tree convention; a §5 choice, not a law).
DEFAULT_FIDELITY = {"bfp4": "LoFi", "bfp8": "HiFi2", "bf16": "HiFi4", "fp32": "HiFi4"}


@dataclass(frozen=True)
class Hardware:
    cores: int = 110
    clock_mhz: float = 1350.0
    matmul_flop_per_cycle_core: int = 4096  # LoFi
    eltwise_per_cycle_core: int = 128
    dram_bytes_per_ns: float = 512.0
    dram_capacity_bytes: float = 32e9
    link_bytes_per_ns: float = 25.0  # per link, per direction
    hop_latency_ns: float = 619.0


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


LOUDBOX_2X4 = Layout(sp=2, tp=4, links=1)
LOUDBOX_4X2 = Layout(sp=4, tp=2, links=1)
GALAXY_8X4 = Layout(sp=8, tp=4, links=2, sp_ring=True, tp_ring=True)


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
    compute_ns: float = 0.0
    dram_ns: float = 0.0
    ccl_ns: float = 0.0

    @property
    def roofline_ns(self) -> float:
        """Local compute/DRAM overlap; collectives are a separate resource (see compose)."""
        return max(self.compute_ns, self.dram_ns)


@dataclass(frozen=True)
class Workload:
    """One prefill chunk: ``chunk`` tokens starting at absolute position ``start``."""

    chunk: int = 5120
    start: int = 0
    expert_dtype: str = "bfp8"
    dense_dtype: str = "bfp8"
    stream_dtype: str = "fp32"  # mHC residual streams between sublayers (in-tree V4 keeps fp32)
    kv_dtype: str = "bf16"
    indexer_heads_over_tp: bool = True  # False: replicate the indexer per TP chip (no score reduction)

    @property
    def end(self) -> int:
        return self.start + self.chunk


def collective_ns(coll: Collective, layout: Layout, hw: Hardware) -> float:
    """Bottleneck-edge payload over per-direction link bandwidth plus hop latency.

    Linear all-gather over n chips: the edge next to an end forwards n-1 shards in one direction;
    a bidirectional ring halves that. Reduce-scatter moves the same payload; all-reduce is both.
    Halo sends one payload to a neighbor. All-to-all is bounded by the per-chip egress spread over the
    chip's mesh ports (a lower bound that ignores multi-hop forwarding load).
    """
    if coll.axis == "mesh":
        n, ring = layout.chips, False
    else:
        n = layout.sp if coll.axis == "sp" else layout.tp
        ring = layout.sp_ring if coll.axis == "sp" else layout.tp_ring
    if n == 1 or coll.shard_bytes == 0:
        return 0.0
    bw = hw.link_bytes_per_ns * layout.links
    if coll.kind in ("all_gather", "reduce_scatter", "all_reduce"):
        edge_shards = (n - 1) / 2 if ring else (n - 1)
        hops = n // 2 if ring else n - 1
        passes = 2 if coll.kind == "all_reduce" else 1
        return passes * (edge_shards * coll.shard_bytes / bw + hops * hw.hop_latency_ns)
    if coll.kind == "halo":
        return coll.shard_bytes / bw + hw.hop_latency_ns
    if coll.kind == "all_to_all":
        ports = (2 if layout.sp > 1 else 0) + (2 if layout.tp > 1 else 0)
        return coll.shard_bytes / (bw * ports) + hw.hop_latency_ns
    raise ValueError(f"unknown collective kind {coll.kind}")


def _finish(op: OpCost, layout: Layout, hw: Hardware) -> OpCost:
    phases = FIDELITY_PHASES[op.fidelity]
    per_ns = hw.cores * hw.clock_mhz / 1000.0
    op.compute_ns = (
        op.matmul_flop * phases / hw.matmul_flop_per_cycle_core + op.eltwise / hw.eltwise_per_cycle_core
    ) / per_ns
    op.dram_ns = op.dram_bytes / hw.dram_bytes_per_ns
    op.ccl_ns = sum(collective_ns(c, layout, hw) for c in op.collectives)
    return op


def _linear(node, name, tokens, k, n, wdtype, layout, *, k_sharded_tp, n_sharded_tp, act=("bf16", "bf16"), coll=()):
    """Per-chip cost of ``[tokens, k] @ [k, n]``: tokens are this chip's SP shard. Work and weights split
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
    """Operation costs of one backbone block for one chunk on one chip."""
    btype = C.block_type(layer)
    ratio = C.compress_ratio(layer)
    s = w.chunk / layout.sp  # tokens per chip
    h, hc = C.EMB_SIZE, C.HC_MULT
    ht = h / layout.tp
    sd = BYTES[w.stream_dtype]
    dense = w.dense_dtype
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
        if site == "attn":
            ops += [mix, pre, norm]
            ops += _attention_ops(layer, btype, ratio, s, w, layout)
            ops.append(post)
        else:
            ops += [mix, pre, norm]
            ops += _moe_ops(s, w, layout)
            ops.append(post)

    if layer in C.ENGRAM_LAYER_IDS:
        ops = engram_ops(s, w, layout) + ops
    return [_finish(op, layout, hw) for op in ops]


def _attention_ops(layer, btype, ratio, s, w: Workload, layout: Layout) -> list[OpCost]:
    h, d, heads, rd = C.EMB_SIZE, C.HEAD_DIM, C.NUM_ATTENTION_HEADS, C.QK_ROPE_HEAD_DIM
    q_lora, dense, kvb = C.Q_LORA_RANK, w.dense_dtype, BYTES[w.kv_dtype]
    hl = heads / layout.tp
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
        OpCost(
            "B5",
            "kv_norm_rope_qdq",
            eltwise=s * d * 6,
            sfpu={"rsqrt": s},
            dram_bytes=s * d * 2 * 2,
            collectives=[Collective("halo", "sp", (C.SLIDING_WINDOW - 1) * d * kvb)],
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
                    matmul_flop=2 * c_chip * d * C.INDEX_HEAD_DIM / layout.tp,
                    fidelity="HiFi4",
                    eltwise=c_chip * C.INDEX_HEAD_DIM * 6,
                    dram_bytes=d * C.INDEX_HEAD_DIM * 2 + c_chip * (d * 2 + C.INDEX_HEAD_DIM * kvb),
                )
            )
            ops.append(
                OpCost(
                    "B8",
                    "compressed_kv_write",
                    eltwise=c_chip * (rd * 3 + d * 4),
                    dram_bytes=c_chip * d * (2 + kvb),
                )
            )
        if btype in (V41BlockType.KV_INDEX_SOURCE, V41BlockType.CANDIDATE_SOURCE, V41BlockType.CANDIDATE_INDEX_SOURCE):
            ops += _indexer_ops(btype, s, visible, w, layout)
        # every compressed layer gathers the visible compressed KV of its source along SP
        ops.append(
            OpCost(
                "B14",
                "compressed_kv_gather",
                dram_bytes=visible * d * kvb,
                collectives=[Collective("all_gather", "sp", visible / layout.sp * d * kvb)],
            )
        )
    selected = min(C.SLIDING_WINDOW, w.end) + (min(C.INDEX_TOPK, visible) if ratio else 0)
    ops.append(
        OpCost(
            "B14",
            "sparse_attention",
            matmul_flop=4 * s * hl * selected * d,
            fidelity="HiFi2",
            eltwise=s * hl * selected * 3,
            sfpu={"exp": s * hl * selected},
            # q and o once, the chunk's KV rows (window halo + chunk + visible compressed) once, indices once
            dram_bytes=s * hl * d * 2 * 2 + (s + C.SLIDING_WINDOW - 1 + visible) * d * kvb + s * selected * 4,
        )
    )
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
    ih, idim, q_lora, h = C.INDEX_N_HEADS, C.INDEX_HEAD_DIM, C.Q_LORA_RANK, C.EMB_SIZE
    kvb = BYTES[w.kv_dtype]
    split = layout.tp if w.indexer_heads_over_tp else 1  # compute split over TP only if heads are split
    heads_local = ih / split
    ops = [
        _linear(
            "B9",
            "index_wq_b",
            s,
            q_lora,
            ih * idim,
            w.dense_dtype,
            layout,
            k_sharded_tp=False,
            n_sharded_tp=w.indexer_heads_over_tp,
        ),
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
            coll=[Collective("all_reduce", "tp", s * ih * 4 / layout.tp)],
        ),
        OpCost(
            "B9",
            "index_k_gather",
            dram_bytes=visible * idim * kvb,
            collectives=[Collective("all_gather", "sp", visible / layout.sp * idim * kvb)],
        ),
        OpCost(
            "B9",
            "index_scores",
            matmul_flop=2 * s * heads_local * idim * visible,
            fidelity="LoFi",  # FP4 q and k in the reference
            eltwise=s * heads_local * visible * 3,
            dram_bytes=s * heads_local * idim * 2 + visible * idim * kvb + s * visible * 4,
            collectives=[Collective("all_reduce", "tp", s * visible * 4 / layout.tp)]
            if w.indexer_heads_over_tp
            else [],
        ),
    ]
    if btype == V41BlockType.CANDIDATE_SOURCE:
        ops.append(
            OpCost(
                "B10",
                "candidate_select",
                eltwise=s * visible,
                sfpu={"topk": s},
                dram_bytes=s * visible * 4 + s * visible,
            )
        )
    if btype == V41BlockType.CANDIDATE_INDEX_SOURCE:
        ops.append(OpCost("B11", "candidate_mask", eltwise=s * visible, dram_bytes=s * visible * (4 + 1 + 4)))
    ops.append(OpCost("B12", "topk", sfpu={"topk": s}, dram_bytes=s * visible * 4 + s * C.INDEX_TOPK * 4))
    return ops


def _moe_ops(s, w: Workload, layout: Layout) -> list[OpCost]:
    h, inter, e, k = C.EMB_SIZE, C.MOE_INTERMEDIATE_SIZE, C.NUM_ROUTED_EXPERTS, C.NUM_EXPERTS_PER_TOKEN
    chips = layout.chips
    tokens_src = s / layout.tp  # tokens dispatched by each chip of the TP row
    routed_tokens = s * k / layout.tp  # expert-token pairs computed per chip on average
    ops = [
        OpCost(
            "B20",
            "gate",
            matmul_flop=2 * s * h * e / layout.tp,
            fidelity="HiFi4",
            sfpu={"softplus": s * e / layout.tp, "sqrt": s * e / layout.tp, "topk": s / layout.tp},
            dram_bytes=h * e * 4 / layout.tp + s * h / layout.tp * 2 + s * e * 4 / layout.tp,
            collectives=[Collective("all_reduce", "tp", s * e * 4 / layout.tp)],
        ),
        OpCost(
            "B21",
            "dispatch",
            dram_bytes=tokens_src * h * 2 + routed_tokens * h * 2,
            collectives=[Collective("all_to_all", "mesh", tokens_src * k * (1 - 1 / chips) * h * 2)],
        ),
        OpCost(
            "B21",
            "routed_experts",
            matmul_flop=2 * routed_tokens * h * inter * 3,
            fidelity=DEFAULT_FIDELITY[w.expert_dtype],
            eltwise=routed_tokens * inter * 4,
            sfpu={"sigmoid": routed_tokens * inter},
            dram_bytes=e / chips * 3 * h * inter * BYTES[w.expert_dtype] + routed_tokens * h * 2 * 2,
        ),
        OpCost(
            "B21",
            "combine",
            eltwise=routed_tokens * h,
            dram_bytes=routed_tokens * h * 2 + tokens_src * h * 2,
            collectives=[Collective("all_to_all", "mesh", tokens_src * k * (1 - 1 / chips) * h * 2)],
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
    return ops


def engram_ops(s, w: Workload, layout: Layout, device_tables: bool = False) -> list[OpCost]:
    """Engram at one layer (N3-N5). Host lookup: rows arrive over the host link (bytes reported in
    ``host_bytes`` via the lookup op's dram field only when tables live on device)."""
    cols, hd, h, hc = (C.ENGRAM_MAX_NGRAM_SIZE - 1) * C.ENGRAM_N_HEADS, C.ENGRAM_HEAD_DIM, C.EMB_SIZE, C.HC_MULT
    row_bytes = hd * BYTES["fp8_e8m0_32"]
    lookup = OpCost(
        "N3",
        "engram_lookup",
        eltwise=s * cols * hd / layout.tp,
        dram_bytes=(s * cols * row_bytes / layout.tp if device_tables else 0) + s * cols * hd * 2 / layout.tp,
    )
    wkv = _linear(
        "N4", "engram_wkv", s, cols * hd, h * (hc + 1), w.dense_dtype, layout, k_sharded_tp=False, n_sharded_tp=True
    )
    gate = OpCost(
        "N5",
        "engram_gate_add",
        eltwise=s * hc * h / layout.tp * 8,
        sfpu={"rsqrt": 2 * s * hc, "sqrt": s * hc, "sigmoid": s * hc},
        dram_bytes=s * hc * h / layout.tp * (2 * BYTES[w.stream_dtype]) + s * h * (hc + 1) / layout.tp * 2,
        collectives=[Collective("all_reduce", "tp", s * hc * 4 * 2 / layout.tp)],
    )
    return [lookup, wkv, gate]


def engram_host_bytes_per_token() -> float:
    """Bytes per token per Engram layer if the host looks up and dequantizes rows to bf16 (N3 on host)."""
    return (C.ENGRAM_MAX_NGRAM_SIZE - 1) * C.ENGRAM_N_HEADS * C.ENGRAM_HEAD_DIM * 2


@dataclass
class BlockEstimate:
    layer: int
    block_type: str
    compute_ns: float
    dram_ns: float
    ccl_ns: float
    conservative_ns: float  # every op serialized: sum(max(compute, dram)) + sum(ccl)
    optimistic_ns: float  # compute, DRAM and links each saturated in parallel across the block
    sfpu: dict
    ops: list


def compose_block(layer: int, w: Workload, layout: Layout, hw: Hardware = BLACKHOLE_P150B) -> BlockEstimate:
    ops = block_ops(layer, w, layout, hw)
    compute = sum(o.compute_ns for o in ops)
    dram = sum(o.dram_ns for o in ops)
    ccl = sum(o.ccl_ns for o in ops)
    sfpu: dict = {}
    for o in ops:
        for key, value in o.sfpu.items():
            sfpu[key] = sfpu.get(key, 0) + value
    return BlockEstimate(
        layer=layer,
        block_type=C.block_type(layer).value,
        compute_ns=compute,
        dram_ns=dram,
        ccl_ns=ccl,
        conservative_ns=sum(o.roofline_ns for o in ops) + ccl,
        optimistic_ns=max(compute, dram, ccl),
        sfpu=sfpu,
        ops=ops,
    )


def weight_bytes_per_layer(layer: int, w: Workload) -> dict:
    """Device bytes of one backbone layer's weights, by component (whole mesh, before sharding)."""
    h, d, q_lora = C.EMB_SIZE, C.HEAD_DIM, C.Q_LORA_RANK
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
        "gate": C.NUM_ROUTED_EXPERTS * h * 4,
        "shared_expert": 3 * h * C.MOE_INTERMEDIATE_SIZE * dense,
        "routed_experts": C.NUM_ROUTED_EXPERTS * 3 * h * C.MOE_INTERMEDIATE_SIZE * BYTES[w.expert_dtype],
    }
    ratio = C.compress_ratio(layer)
    if layer in C.KV_SOURCE_LAYERS:
        out["compressor"] = h * d * (2 if ratio > 1 else 1) * (4 if ratio > 1 else 2) + d * C.INDEX_HEAD_DIM * 2
    if layer in C.INDEX_SOURCE_LAYERS:
        out["indexer"] = q_lora * C.INDEX_N_HEADS * C.INDEX_HEAD_DIM * dense + h * C.INDEX_N_HEADS * 2
    if layer in C.ENGRAM_LAYER_IDS:
        cols = (C.ENGRAM_MAX_NGRAM_SIZE - 1) * C.ENGRAM_N_HEADS
        out["engram_wkv"] = cols * C.ENGRAM_HEAD_DIM * h * (C.HC_MULT + 1) * dense
    return out


def engram_table_bytes(layers: list[int] | None = None) -> float:
    """FP8 rows + E8M0 scales of the Engram tables of ``layers`` (all Engram layers by default)."""
    layers = list(C.ENGRAM_LAYER_IDS) if layers is None else layers
    rows = [n for layer, n in zip(C.ENGRAM_LAYER_IDS, C.ENGRAM_NUM_EMBEDDINGS) if layer in layers]
    return sum(n * C.ENGRAM_HEAD_DIM * BYTES["fp8_e8m0_32"] for n in rows)


def capacity_per_chip(
    layers: list[int], w: Workload, layout: Layout, *, engram_on_device: bool, context_tokens: int = 0
) -> dict:
    """Per-chip DRAM bytes: attention/dense weights replicated over SP and split over TP; routed experts
    and Engram tables split over every chip; caches split over SP (token-sharded)."""
    per = {"dense": 0.0, "routed_experts": 0.0, "engram_tables": 0.0, "embed_head": 0.0, "caches": 0.0}
    for layer in layers:
        wb = weight_bytes_per_layer(layer, w)
        per["routed_experts"] += wb.pop("routed_experts") / layout.chips
        per["dense"] += sum(wb.values()) / layout.tp
    per["embed_head"] = 2 * C.VOCAB_SIZE * C.EMB_SIZE * 2 / layout.chips
    if engram_on_device:
        per["engram_tables"] = engram_table_bytes(layers) / layout.chips
    per["caches"] = kv_cache_bytes(context_tokens, w, layers) / layout.sp
    per["total"] = sum(per.values())
    return per


def kv_cache_bytes(tokens: int, w: Workload, layers: list[int] | None = None) -> float:
    """Compressed KV + index-K caches of the KV sources in ``layers`` plus the window rings of all layers."""
    layers = list(range(C.NUM_LAYERS)) if layers is None else layers
    kvb = BYTES[w.kv_dtype]
    total = 0.0
    for layer in layers:
        if layer in C.KV_SOURCE_LAYERS:
            total += tokens // C.compress_ratio(layer) * (C.HEAD_DIM + C.INDEX_HEAD_DIM) * kvb
        total += C.SLIDING_WINDOW * C.HEAD_DIM * kvb
    return total


def final_ops(w: Workload, layout: Layout, hw: Hardware = BLACKHOLE_P150B) -> list[OpCost]:
    """F1 final collapse and F2 norm over the chunk; F3 fp32 LM head over the last token only."""
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
        # one token; vocab split over every chip, hidden gathered to the owner first
        OpCost(
            "F3",
            "lm_head_last_token",
            matmul_flop=2 * C.EMB_SIZE * C.VOCAB_SIZE / layout.chips,
            fidelity="HiFi4",
            dram_bytes=C.EMB_SIZE * C.VOCAB_SIZE / layout.chips * 2 + C.VOCAB_SIZE / layout.chips * 4,
            collectives=[Collective("all_gather", "tp", ht * 2)],
        ),
    ]
    return [_finish(op, layout, hw) for op in ops]


def dspark_prefill_ops(w: Workload, layout: Layout, hw: Hardware = BLACKHOLE_P150B) -> list[OpCost]:
    """N6 taps (3 layers, whole chunk) and D1/D2 over the last ``min(128, chunk)`` rows: main_proj FP8
    15360 -> 5120 + main_norm once, then per DSpark layer wkv 5120 -> 512, kv_norm, RoPE, FP8 QDQ."""
    s, h, d, hc = w.chunk / layout.sp, C.EMB_SIZE, C.HEAD_DIM, C.HC_MULT
    rows = min(C.SLIDING_WINDOW, w.chunk)  # held by the last SP rank(s); modeled on one chip
    taps = len(C.DSPARK_TARGET_LAYER_IDS)
    ops = [
        OpCost(
            "N6",
            "dspark_taps",
            eltwise=taps * s * hc * h / layout.tp,
            dram_bytes=taps * s * (hc * BYTES[w.stream_dtype] + 2) * h / layout.tp,
        ),
        _linear(
            "D1",
            "main_proj",
            rows,
            taps * h,
            h,
            w.dense_dtype,
            layout,
            k_sharded_tp=True,
            n_sharded_tp=False,
            coll=[Collective("all_reduce", "tp", rows * h * 2 / layout.tp)],
        ),
        OpCost("D1", "main_norm", eltwise=rows * h * 3, sfpu={"rsqrt": rows}, dram_bytes=rows * h * 2 * 2),
    ]
    for i in range(C.NUM_DSPARK_LAYERS):
        ops.append(
            _linear("D2", f"dspark{i}_wkv", rows, h, d, w.dense_dtype, layout, k_sharded_tp=False, n_sharded_tp=True)
        )
        ops.append(
            OpCost(
                "D2",
                f"dspark{i}_kv_norm_rope_qdq",
                eltwise=rows * d * 6,
                sfpu={"rsqrt": rows},
                dram_bytes=rows * d * (2 + BYTES[w.kv_dtype]),
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
