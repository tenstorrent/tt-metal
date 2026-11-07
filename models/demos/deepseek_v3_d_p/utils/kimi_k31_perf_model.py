# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Kimi K3.1 prefill performance model: DSA operation models and TTFT targets on 8x4 PP4.

K3.1 is K3 with the attention core of its 24 full-attention layers changed from dense MLA to DSA (indexer top-2048
+ sparse attention). The MLA projections, KDA, MoE and AttnRes are unchanged, so this model composes:

  * **unchanged K3 work, measured**: K3's traced PP4 CI run on four 8x4 Galaxies (``perf_baselines.json``
    ``kimi_k3/sc4/8kw``: retire throughput and TTFT per probe). No theoretical MoE / AttnRes model exists in
    tree and K3's KDA models are per operation, so the measured stage is the canonical "existing K3 model"; and
  * **DSA attention core, modelled**: per-chip operation bounds below, replacing K3's dense MLA attention core, whose
    cost is taken from the same measurement (its growth with the prefix; see ``dense_mla_core_ns``).

Operation models (one DSA layer, one chunk, one chip; conventions of the DeepSeek-V4.1 model ``v41_perf_model.py``,
branch ``mvasilijevic/deepseek-v41-prefill``, and of GLM-5.3's sparse MLA, whose code K3.1 shares):

  * placement: SP8 x TP4. The indexer and sparse attention split the chunk's queries over all 32 chips
    (``chunk / 32`` = 160 queries per chip, all heads; GLM-5.3 ring indexer and heads->sequence reshard). The index-K
    and KVPE caches are both deduped over all 32 chips (``PREFILL_TP_SHARD_KV``; ``indexer.py`` "KV dedup") and a DSA
    layer gathers the whole visible prefix of each onto every chip (the fused ring indexer's full-mesh gather; GLM-5.3
    full-indexer layers: ``mla.py`` ``_sparse_kv_gather_buffer``; the per-row push is only for shared layers, which
    K3.1 does not have).
  * compute: matrix 4096 FLOP/cycle/core at LoFi divided by the fidelity phases; elementwise 128 per cycle per core
    (``tech_reports/matrix_engine/matrix_engine.md``); 120 cores (Galaxy compute grid 12x10); 1.35 GHz (repository
    constant, an assumption). SFPU primitives (exp) are counted, not timed. Exceptions, implementation measurements
    of K3.1 on this 8x4 (ao3i.2.2, stage 2, real weights; indicative, TDP_LIMIT_MAX=75 W), not capabilities:
      - the indexer top-k (``topk_large_indices``, k=2048, 160 rows per chip) runs on its own sub-device concurrently
        with the KVPE gather and emits no profiler record. Its time is bounded on the chip timeline by the region from
        the KVPE write's end to ``sparse_sdpa``'s start, which also holds the gather. ``TOPK_NS`` is that region at
        prefix 0 (5,120 scores per row: <= 128.4 us, gather 39 us), established there only. On the 128K prompt the
        region outlasts the gather by 37-96 us per layer (stage 0: 49-94) at chunks 0-11 and 20-24, by 10-40 us at
        chunks 12-19, and by ~10 us only at 1M (4,768 vs 4,734 us): that tail is either the top-k (then the top-k is ~
        the region, ~645 us at 117,760, 5x ``TOPK_NS``) or a post-gather wait; the dumps cannot separate them. So the
        top-k's rate is not identified (>= 35.7 elements/ns per chip at 1M, the only depth where the gather certainly
        hides it) and is not modelled; a standalone ``topk_large_indices`` at 160 x 5,120 and 160 x 1,064,960 on 120
        cores would measure it. The concurrent term max(top-k, gather) covers the measured region from the 6th chunk on
        only through the gather model's 1.5-2x slack; at chunks 0-4 the region exceeds it by 1.01-1.24x. This replaces
        V4.1's LoudBox rate (7.1 elements/ns per chip), which the 1M bound contradicts;
      - the indexer score program (fused with the index-K ring all-gather) has a fixed per-op cost
        (``INDEX_SCORE_FIXED_NS``) that the collective's 14 us latency below omits: every chip pays a floor of
        197.7 us at prefix 0 (197.9 at 5,120) at HiFi2 while its modelled work nearly doubles; the constant is that
        floor less the conservative model of the fused pair there (46.6 us). At LoFi the floor was 128.2 us (model
        36.7 us), so the fixed cost depends on the fidelity (cause not established; tt_metal_tracker-ao3i.2.7). The 30-80 us chip spread above the floor (waiting on
        ring peers) is not a fixed cost and is not modelled.
  * DRAM: 512 GB/s per chip (``ttnn/core/operation.cpp``). Optimistic traffic reads each selected KV row once per
    chip (union of the queries' selections); conservative traffic reads each query's rows with no reuse.
  * fabric: optimistic = per-chip ingress bound at 25 GB/s per link per direction x 2 links x 4 torus ports; this
    equals the spanning-tree link-load bound of an all-gather on the 8x4 torus. Conservative = the in-tree
    collectives at the rate and latency calibrated on LoudBox 2x4 (V4.1 ``BLACKHOLE_CCL``; Galaxy not calibrated):
    the KVPE gather as one 32-chip bidirectional ring, all-to-all over the TP ring's two ports.

Layer composition (``dsa_core``), as V4.1: optimistic = max(sum compute, sum optimistic DRAM, sum fabric) (all
resources saturated in parallel) + the fixed per-op costs; conservative = sum over ops of max(compute, conservative
DRAM) + fixed cost + calibrated fabric, except the top-k and the KVPE gather, which run concurrently on separate
sub-devices and count once, as the longer of the two; target = max(2 x ideal + measured, conservative), where ideal is the optimistic bound
without the measured terms (top-k, fixed costs): V4.1's G2 target (2 x optimistic) doubles only modelled ideal work,
never an already measured time.

Pipeline composition (``ttft``): PP4 stages 0-23 / 24-47 / 48-71 / 72-92, each with 6 attention layers.
Chunks retire every ``period(P)``, the busiest stage's time at prefix P, so TTFT = sum of the prompt's chunk periods
+ a pipeline fill, the measured TTFT of K3's first chunk minus its period (``K3Measured.fill_ns``, 0.70 s). On K3's
measurement this reproduces the first chunk exactly and the deeper probes within a few percent. K3.1's period is K3's measured first-chunk period with that stage's 6 dense MLA cores
swapped for 6 DSA cores at the chunk's prefix.

What this does not model: K3.1's KDA / MoE / AttnRes / projections are assumed to cost what K3's do (same code and
dimensions; K3's anchor ran real weights and real routing, so a synthetic-weight run differs in its MoE load); the
fill is assumed unchanged; the dense core's efficiency comes from K3's prefix slope, not a measurement at prefix 0.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from models.demos.deepseek_v3_d_p.reference.kimi_k3_1_config import KimiK31Config as C
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import MlaKvCacheGeometry

PERF_BASELINES = Path(__file__).parents[2] / "common" / "prefill" / "runners" / "ci" / "perf_baselines.json"

CHUNK = 5120  # K3 prefill chunk (``tests/kimi_k3/test_prefill_perf.py``), the CI probes' unit
FIDELITY_PHASES = {"LoFi": 1, "HiFi2": 2, "HiFi3": 3, "HiFi4": 4}
BYTES = {"bf16": 2.0, "bfp8": 1088 / 1024, "int32": 4.0}
KVPE_ROW_BYTES = MlaKvCacheGeometry(latent_dim=C.KV_LORA_RANK, rope_dim=C.QK_ROPE_HEAD_DIM).packed_row_bytes
# Measured on K3.1 8x4 (see module docstring), ns: the top-k's identifiable time (bound at 5,120 scores per row; rate
# not identified), and the fused indexer score's fixed cost (chip floor at prefix 0 minus its conservative model).
TOPK_NS = 128_400
INDEX_SCORE_FIXED_NS = 197_700 - 46_600
STAGES = ((0, 24), (24, 48), (48, 72), (72, 93))  # PP4 layer ranges, half-open, 0-based


@dataclass(frozen=True)
class CclRate:
    link_bytes_per_ns: float  # achieved per link per direction
    latency_ns: float  # fixed per-op cost


@dataclass(frozen=True)
class Galaxy8x4:
    sp: int = 8
    tp: int = 4
    links: int = 2
    torus_ports: int = 4
    cores: int = 120
    clock_ghz: float = 1.35
    matmul_flop_per_cycle_core: int = 4096  # LoFi
    eltwise_per_cycle_core: int = 128
    dram_bytes_per_ns: float = 512.0
    link_bytes_per_ns: float = 25.0  # per link per direction, as enabled
    # V4.1 LoudBox 2x4 calibration at 2 links, reused for Galaxy (assumption)
    all_gather: CclRate = CclRate(22.0, 14_000.0)
    all_to_all: CclRate = CclRate(12.0, 15_000.0)

    @property
    def chips(self) -> int:
        return self.sp * self.tp


GALAXY = Galaxy8x4()


@dataclass
class OpCost:
    """One operation on one chip. ``*_ns`` are filled by ``_finish``; ``fabric_cons_ns`` is the calibrated path."""

    name: str
    matmul_flop: float = 0.0
    fidelity: str = "HiFi2"
    eltwise: float = 0.0
    sfpu: dict = field(default_factory=dict)
    fixed_ns: float = 0.0  # measured per-op cost, added as is (not a resource bound)
    dram_bytes: float = 0.0
    dram_bytes_cons: float | None = None
    fabric_ns: float = 0.0
    fabric_cons_ns: float = 0.0
    fpu_ns: float = 0.0
    topk_ns: float = 0.0  # measured top-k time (TOPK_NS), not derived from work
    compute_ns: float = 0.0  # fpu_ns + topk_ns
    dram_ns: float = 0.0
    dram_cons_ns: float = 0.0

    @property
    def roofline_cons_ns(self) -> float:
        return max(self.compute_ns, self.dram_cons_ns) + self.fixed_ns


def _finish(op: OpCost, hw: Galaxy8x4) -> OpCost:
    per_ns = hw.cores * hw.clock_ghz
    fpu = op.matmul_flop * FIDELITY_PHASES[op.fidelity] / hw.matmul_flop_per_cycle_core
    fpu += op.eltwise / hw.eltwise_per_cycle_core
    op.fpu_ns = fpu / per_ns
    op.compute_ns = op.fpu_ns + op.topk_ns
    op.dram_ns = op.dram_bytes / hw.dram_bytes_per_ns
    op.dram_cons_ns = (op.dram_bytes if op.dram_bytes_cons is None else op.dram_bytes_cons) / hw.dram_bytes_per_ns
    return op


def causal_pairs(prefix: int, queries: int, cap: int | None = None) -> int:
    """Sum over ``queries`` consecutive queries at positions prefix..prefix+queries-1 of the keys each attends:
    every earlier token and itself, at most ``cap``."""
    first, last = prefix + 1, prefix + queries
    if cap is None or last <= cap:
        return (first + last) * queries // 2
    if first >= cap:
        return cap * queries
    below = cap - first  # queries that see fewer than cap keys
    return (first + cap - 1) * below // 2 + cap * (queries - below)


def _gather_fabric(received_bytes: float, chips: int, hw: Galaxy8x4) -> tuple[float, float]:
    """(optimistic, calibrated) time for every chip to receive ``received_bytes`` it does not hold."""
    if received_bytes <= 0:
        return 0.0, 0.0
    ingress = hw.link_bytes_per_ns * hw.links * hw.torus_ports
    shard = received_bytes / (chips - 1)  # each other chip's share
    edge = (chips - 1) / 2 * shard  # bidirectional ring
    calibrated = hw.all_gather.latency_ns + edge / (hw.all_gather.link_bytes_per_ns * hw.links)
    return received_bytes / ingress, calibrated


def dsa_ops(prefix: int, chunk: int = CHUNK, hw: Galaxy8x4 = GALAXY) -> list[OpCost]:
    """Per-chip operations of one DSA layer's attention core for the chunk at ``prefix`` (tokens already cached).

    Covers what DSA adds or replaces relative to K3's dense MLA core; the MLA projections (q/kv LoRA, wkv_b
    absorption, output gate, o_proj) are unchanged and stay in the measured K3 stage.
    """
    assert chunk % hw.chips == 0, f"chunk {chunk} must split over {hw.chips} chips"
    s = chunk // hw.sp  # tokens per SP rank (projections)
    q = chunk // hw.chips  # queries per chip (indexer, top-k, sparse attention)
    visible = prefix + chunk  # keys visible to the chunk's last query
    ih, idim, topk = C.INDEX_N_HEADS, C.INDEX_HEAD_DIM, C.INDEX_TOPK
    heads, latent, rope = C.NUM_ATTENTION_HEADS, C.KV_LORA_RANK, C.QK_ROPE_HEAD_DIM
    bf16, bfp8 = BYTES["bf16"], BYTES["bfp8"]
    # Query positions on this chip: the worst chip holds the chunk's last q queries (most visible keys).
    last = prefix + chunk - q

    ops = []
    # Indexer projections: wq_b on this chip's queries (qr is MLA's, already present), wk + LayerNorm and
    # weights_proj on the SP rank's tokens, TP-sharded over the hidden dim (+ TP all-reduce, negligible bytes).
    ops.append(
        OpCost(
            "index_proj",
            matmul_flop=2 * q * C.Q_LORA_RANK * ih * idim + 2 * s * C.EMB_SIZE * (idim + ih) / hw.tp,
            fidelity="HiFi2",
            eltwise=s * idim * 6,
            sfpu={"rsqrt": s},
            dram_bytes=(C.Q_LORA_RANK * ih * idim + C.EMB_SIZE * (idim + ih) / hw.tp) * bf16
            + q * ih * idim * bf16
            + s * idim * bf16,
        )
    )
    # Index-K prefix (bfp8): every chip receives the visible keys it does not own (cache deduped over all chips) and
    # writes them to its gather buffer once; the indexer score counts their read.
    received = visible * idim * bfp8 * (hw.chips - 1) / hw.chips
    opt, cal = _gather_fabric(received, hw.chips, hw)
    ops.append(OpCost("index_k_gather", dram_bytes=received, fabric_ns=opt, fabric_cons_ns=cal))
    # Indexer score: relu(q.k) per head, weighted head sum, causal, bfp8 at HiFi2 (K3.1's config; GLM-5.3 runs LoFi).
    # Scores are materialized in bf16 for the top-k (the in-tree graph: indexer_score -> topk_large_indices).
    pairs = causal_pairs(last, q)
    width = -(-visible // 32) * 32
    scores = q * width * bf16
    ops.append(
        OpCost(
            "index_score",
            matmul_flop=2 * pairs * ih * idim,
            fidelity="HiFi2",
            eltwise=pairs * ih * 3,  # relu, weight multiply, head-sum add
            dram_bytes=visible * idim * bfp8 + q * ih * idim * bf16 + scores,
            fixed_ns=INDEX_SCORE_FIXED_NS,
        )
    )
    ops.append(
        OpCost(
            "index_topk",
            topk_ns=TOPK_NS,
            sfpu={"topk": q * width},
            dram_bytes=scores + q * topk * BYTES["int32"],
        )
    )
    # KVPE prefix: every chip receives the visible rows it does not own (cache deduped over all chips) and writes
    # them to its gather buffer once; sparse attention counts their read.
    received = visible * KVPE_ROW_BYTES * (hw.chips - 1) / hw.chips
    opt, cal = _gather_fabric(received, hw.chips, hw)
    ops.append(OpCost("kvpe_gather", dram_bytes=received, fabric_ns=opt, fabric_cons_ns=cal))
    # Heads -> sequence reshard of q (latent + rope, bf16) before and the latent output after sparse attention,
    # over the TP ring (K3's ring_mla runs head-sharded and needs neither).
    q_bytes = s * heads / hw.tp * (latent + rope) * bf16 * (hw.tp - 1) / hw.tp
    o_bytes = s * heads / hw.tp * latent * bf16 * (hw.tp - 1) / hw.tp
    for name, egress in (("q_head_to_seq", q_bytes), ("o_seq_to_head", o_bytes)):
        ops.append(
            OpCost(
                name,
                dram_bytes=2 * egress * hw.tp / (hw.tp - 1),
                fabric_ns=egress / (2 * hw.links * hw.link_bytes_per_ns),
                fabric_cons_ns=hw.all_to_all.latency_ns + egress / (2 * hw.links * hw.all_to_all.link_bytes_per_ns),
            )
        )
    # Sparse attention over min(visible, top-k) selected latent rows: q.k over latent+rope, p.v over latent.
    sel = causal_pairs(last, q, cap=topk)
    q_o = q * heads * ((latent + rope) + latent) * bf16 + q * topk * BYTES["int32"]
    ops.append(
        OpCost(
            "sparse_attention",
            matmul_flop=2 * sel * heads * (2 * latent + rope),
            fidelity="HiFi2",
            eltwise=sel * heads * 3,
            sfpu={"exp": sel * heads},
            dram_bytes=q_o + min(visible, q * topk) * KVPE_ROW_BYTES,
            dram_bytes_cons=q_o + sel * KVPE_ROW_BYTES,
        )
    )
    return [_finish(op, hw) for op in ops]


@dataclass(frozen=True)
class CoreEstimate:
    ideal_ns: float  # max(sum FPU, sum DRAM, sum fabric): modelled work only
    measured_ns: float  # measured top-k time (TOPK_NS) + fixed per-op costs
    optimistic_ns: float  # max(sum FPU + top-k, sum DRAM, sum fabric) + fixed costs
    conservative_ns: float
    ops: tuple

    @property
    def target_ns(self) -> float:
        return max(2 * self.ideal_ns + self.measured_ns, self.conservative_ns)


def dsa_core(prefix: int, chunk: int = CHUNK, hw: Galaxy8x4 = GALAXY) -> CoreEstimate:
    """One DSA layer's attention core on one chip for the chunk at ``prefix``."""
    ops = dsa_ops(prefix, chunk, hw)
    fpu, topk, fixed = (sum(getattr(o, a) for o in ops) for a in ("fpu_ns", "topk_ns", "fixed_ns"))
    dram, fabric = sum(o.dram_ns for o in ops), sum(o.fabric_ns for o in ops)
    cons = {o.name: o.roofline_cons_ns + o.fabric_cons_ns for o in ops}
    # The top-k and the KVPE gather run concurrently (separate sub-devices): one term, the longer of the two.
    concurrent = max(cons.pop("index_topk"), cons.pop("kvpe_gather"))
    conservative = sum(cons.values()) + concurrent
    optimistic = max(fpu + topk, dram, fabric) + fixed
    return CoreEstimate(max(fpu, dram, fabric), topk + fixed, optimistic, conservative, tuple(ops))


# --- measured K3 anchor ----------------------------------------------------------------------------------------


def _probe_chunk(label: str) -> int:
    """CI probe label -> chunk index (``summarize_ci_run.py``: label = tokens through the chunk / 1024, or Mi)."""
    tokens = int(label[1:-1]) * (1048576 if label.endswith("M") else 1024)
    return round(tokens / CHUNK) - 1


@dataclass(frozen=True)
class K3Measured:
    """K3 sc4 (PP4 on four 8x4 Galaxies, 93 layers, traced) by probe chunk index: retire throughput, chunk latency
    and TTFT (seconds, tok/s, ms as in ``perf_baselines.json``)."""

    throughput_tok_s: dict
    chunk_time_ms: dict
    ttft_s: dict

    @classmethod
    def load(cls, path: Path = PERF_BASELINES, power: str = "8kw") -> "K3Measured":
        entry = json.loads(path.read_text())["baselines"]["kimi_k3"]["sc4"][power]
        pick = lambda metric: {_probe_chunk(lbl): v[metric] for lbl, v in entry.items() if metric in v}
        return cls(pick("throughput_tok_s"), pick("chunk_time_ms"), pick("ttft_s"))

    def period_ns(self, d: int) -> float:
        """Retire period of chunk ``d``: the busiest stage's time, linearly interpolated between probes."""
        probes = sorted(self.throughput_tok_s)
        assert probes[0] <= d <= probes[-1], f"chunk {d} outside the measured probes {probes}"
        hi = next(p for p in probes if p >= d)
        lo = max(p for p in probes if p <= d)
        at = lambda p: CHUNK / self.throughput_tok_s[p] * 1e9
        return at(lo) if hi == lo else at(lo) + (at(hi) - at(lo)) * (d - lo) / (hi - lo)

    def fill_ns(self) -> float:
        """Pipeline fill: the first chunk's TTFT minus its period. ``summarize_ci_run.py`` measures both chunk_time and
        TTFT from the request's first layer ack, so chunk 0's chunk_time is its TTFT."""
        return self.chunk_time_ms[0] * 1e6 - self.period_ns(0)


def dense_mla_core_ns(prefix: int, k3: K3Measured, chunk: int = CHUNK) -> float:
    """K3's dense MLA core per layer for the chunk at ``prefix``, from K3's measured prefix growth.

    The retire period is the busiest stage's time; it grows with the prefix only through that stage's 6 MLA cores
    (KDA, MoE, AttnRes and the projections do not depend on it). The core's work is proportional to its causal
    (query, key) pairs, and one more prefix token adds ``chunk`` pairs, so the core costs the measured per-token slope
    (period slope from the first to the deepest probe, per layer) times ``causal_pairs / chunk``.
    """
    probes = sorted(k3.throughput_tok_s)
    d0, d1 = probes[0], probes[-1]
    slope = (k3.period_ns(d1) - k3.period_ns(d0)) / ((d1 - d0) * chunk) / max(stage_attention_layers())
    return slope * causal_pairs(prefix, chunk) / chunk


# --- K3.1 pipeline -----------------------------------------------------------------------------------------------


def stage_attention_layers() -> list[int]:
    ids = set(C.mla_layer_ids())
    return [sum(1 for layer in range(lo, hi) if layer in ids) for lo, hi in STAGES]


SCENARIOS = ("optimistic", "target", "conservative", "k3")


@dataclass(frozen=True)
class Ttft:
    prompt_tokens: int
    chunks: int
    scenario: str
    ttft_s: float
    first_period_ms: float
    last_period_ms: float

    @property
    def prefill_tok_s(self) -> float:
        return self.prompt_tokens / self.ttft_s


def ttft(prompt_tokens: int, scenario: str = "target", k3: K3Measured | None = None, hw: Galaxy8x4 = GALAXY) -> Ttft:
    """TTFT on PP4 8x4 for a ``prompt_tokens`` prompt in 5120-token chunks (the last one padded).

    TTFT = sum of the chunks' retire periods + K3's measured pipeline fill. A period is K3's measured first-chunk
    period with the busiest stage's 6 dense MLA cores swapped for attention cores at the chunk's prefix: DSA cores
    under ``scenario`` (``optimistic``, ``conservative``, ``target``), or dense MLA cores (``k3``: the self-check
    that reproduces K3's measured TTFT). ``k3`` is the anchor (default: the CI baselines).
    """
    assert scenario in SCENARIOS, f"unknown scenario {scenario}"
    k3 = K3Measured.load() if k3 is None else k3
    assert min(k3.throughput_tok_s) == 0, "the anchor is the first chunk"
    layers = max(stage_attention_layers())
    base = k3.period_ns(0) - layers * dense_mla_core_ns(0, k3)

    def core(prefix: int) -> float:
        if scenario == "k3":
            return dense_mla_core_ns(prefix, k3)
        e = dsa_core(prefix, hw=hw)
        return {"optimistic": e.optimistic_ns, "conservative": e.conservative_ns, "target": e.target_ns}[scenario]

    chunks = -(-prompt_tokens // CHUNK)
    periods = [base + layers * core(c * CHUNK) for c in range(chunks)]
    total = sum(periods) + k3.fill_ns()
    return Ttft(prompt_tokens, chunks, scenario, total / 1e9, periods[0] / 1e6, periods[-1] / 1e6)
