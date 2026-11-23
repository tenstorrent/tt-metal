# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""GLM-5.4 prefill performance model: the operation models of the nodes GLM-5.4 adds to the GLM-5.3 graph, and
their composition into per-stage time per 5120-token chunk on 4 stages of 8x4 (SP8 x TP4, one Galaxy each).

GLM-5.4 is GLM-5.3 with layers 26-53 (0-indexed) run twice on shared weights, a damp gate after each pass, and in
pass 2 no indexer and a two-bank sparse MLA (pass-1 KV latents at the pass-1 top-k + a 128-token window of pass-2
latents). Stages: G0 = layers 0-25, G1 = loop pass 1 (26-53) + gate 1, G2 = loop pass 2 (26-53, weight copy) +
gate 2, G3 = layers 54-77 (MTP excluded on both models). Node names (G1.5, G2.6, ...) are those of the semantic graph
(tt-work/artifacts/glm54/semantic-graph.md, bead tt_metal_tracker-q171.2.1).

How a stage time is built (``stage_times``):

  * **unchanged GLM-5.3 work, measured** (``Glm53Anchor``): GLM-5.3's layers 0-25 / 26-53 / 54-77 cut by layer acks
    out of one traced 78-layer run on one 8x4 Galaxy (each layer fed its real predecessor), per chunk at the probe
    depth. No theoretical MoE or projection model exists in tree, so the unchanged work is the measured span itself.
  * **G2's base is G1's minus its indexers**: pass 2 has no indexer, so all 28 layers are reuse layers (sparse KV
    push at the injected top-k). G2's base = G1's measured time - 7 x ``full_extra_ms``, the measured extra cost of a
    full-indexer layer (indexer + KVPE prefix gather) over a reuse layer at the same depth.
  * **new nodes, modelled** (the functions below): theoretical per-chip bounds against aggregate hardware capability,
    plus per-op launch and collective start-up costs, which are measurements (labelled as such), not capabilities.

Graph node -> model: G1.2 / G2.3 ``keep_pass_input``; G1.5 + G1.6 ``g1_collect``; G1.7 / G2.12 ``damp_gate``;
G1.8 / G2.1 / X.8 ``d2d_transfer``; G2.2 ``bank_a_page_copy``; G2.4 + G2.8 (no indexer, push at the injected set) the
G2 base; G2.5 (pass-2 latent to the window instead of the KVPE slot) cost-neutral, its write replaces the reuse
layer's; G2.6 ``sp_halo``; G2.7 ``window_carry``; G2.9 ``two_bank_attention_delta``. G2.11 (ack ids) and G2.14
(weight copy) add no per-chunk work.

Hardware (``Galaxy8x4``), per chip: 120 Tensix cores at 1.35 GHz (Blackhole AICLK, #61846; grid an assumption);
matrix engine 4096 FLOP/cycle/core at LoFi, divided by the fidelity phases, and 128 eltwise results/cycle/core
(``tech_reports/matrix_engine/matrix_engine.md``); DRAM 512 GB/s (``ttnn/core/operation.cpp``); Ethernet 25 GB/s
per link per direction (Blackhole Galaxy links run at half the 50 GB/s line rate,
``tech_reports/CCLs/CCL_bandwidth.md``). SFPU primitives (rsqrt, sigmoid, exp) are counted in ``OpBound.sfpu``, not
timed. Collectives use the Blackhole Galaxy kernel-time fits of ``CCL_bandwidth.md`` (start-up + bytes through the
busiest link at the measured peak rate): measured fits, not capabilities.

Scenarios (``OPTIMISTIC``, ``CONSERVATIVE``) bracket what the code does not yet fix:
  * launch cost per small op: 3 us (a small copy inside a traced Galaxy layer call advanced it 2.5-3 us) or 6.7 us
    (traced small-op wall time on LoudBox 2x4); knowledge note "Small data-movement chains cost launches not bytes";
  * SP halo (G2.6): the 127 predecessor rows point to point, or the graph's default, an SP all-gather of the chunk's
    pass-2 latent (5120 rows);
  * sparse-attention window reads: each window row read once per chip (queries on a chip are consecutive positions,
    so their windows overlap) or once per query (no reuse);
  * G1 top-k sets (G1.5): kept resident (no copy) or copied into a send buffer;
  * G1->G2 D2D (X.8): overlapped with compute (a pipeline resource of its own) or serialized into G2's period.

Pipeline (``pipeline``): one chunk retires per period = the slowest stage; throughput = 5120 / period. Stage times
exclude D2D, as do the stage-by-stage b06u02 measurements the met / unmet rule compares (no SC4: user decision
2026-11-23), so ``slowest_stage_ms``, ``ratio_vs_glm53`` and ``tok_s`` are b06u02-comparable. ``sc4_period_ms`` adds
the G1->G2 D2D (a pipeline resource of its own when overlapped; its extra segments on G1 and G2 when serialized), and
``sc4_tok_s`` rescales it by GLM-5.3's ratio of SC4 CI period to b06u02 slowest stage at the same probe (context
only: on the full78 anchor b06u02 runs 2-18% faster than SC4 CI, growing with depth, so it is not a constant
machine factor). ``g2_d2d_headroom_ms`` is how much serialized D2D G2 absorbs before it, not G1, paces the pipeline.

Tolerance (``TOLERANCE``): G0, G1 and G3 are >= 99% measured GLM-5.3 work, so they carry the anchor's chunk-to-chunk
spread (~3%) and the difference between a 78-layer run's slice and a stage run on its own rank (not measured;
stage-alone runs of the offset stages were rejected, they feed embeddings to the first layer): +-10%. G2 adds the
full-indexer subtraction (the full78 ack-delta estimate exceeds the isolated sparse-MLA CI measurement, #66182, by
0.6-1.0 ms per layer at 50k-500k, i.e. <= 7 ms or ~4% of G2) and new nodes whose implementation does not exist yet: +-15%. A measured stage
outside its band is a deviation to attribute; beyond 25% it is a big gap (knowledge note "Performance models are
estimates with a tolerance band not oracles").

Not modelled: MTP; the decode window publication (deferred); the window storage format other than scaled FP8 (a bf16
joint table would change push bytes, q171.2.5 decides); host-side gaps between chunks; any interference of the D2D
service with on-mesh fabric traffic.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from models.demos.deepseek_v3_d_p.reference.glm_5_3_config import GLM53Config as C
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import MlaKvCacheGeometry

CHUNK = 5120
# CI probe labels (summarize_ci_run.py): tokens prefilled through probe chunk 0 / 9 / 19 / 50 / 101 / 203.
PROBES = ("5k", "50k", "100k", "255k", "510k", "1020k")
STAGES = ("G0", "G1", "G2", "G3")
LOOP_LAYERS = 28  # 0-indexed 26..53
LOOP_FULL_LAYERS = 7  # full-indexer layers among 26..53: 26, 30, ..., 50
WINDOW = 128  # pass-2 window, positions p-127..p (A3)
TOLERANCE = {"G0": 0.10, "G1": 0.10, "G2": 0.15, "G3": 0.10}
FIDELITY_PHASES = {"LoFi": 1, "HiFi2": 2, "HiFi3": 3, "HiFi4": 4}

_KV_GEOMETRY = MlaKvCacheGeometry(latent_dim=C.KV_LORA_RANK, rope_dim=C.QK_ROPE_HEAD_DIM)
KVPE_PAGE_BYTES = -(-_KV_GEOMETRY.packed_row_bytes // 64) * 64  # 656 logical bytes in a 64 B-aligned 704 B page
HIDDEN_BYTES = 2  # bf16 activations and gate weights
INDEX_BYTES = 4  # uint32 top-k indices


@dataclass(frozen=True)
class Galaxy8x4:
    sp: int = 8
    tp: int = 4
    cores: int = 120
    clock_ghz: float = 1.35
    matmul_flop_per_cycle_core: int = 4096  # LoFi
    eltwise_per_cycle_core: int = 128
    dram_bytes_per_ns: float = 512.0
    link_bytes_per_ns: float = 25.0  # per link per direction

    @property
    def chips(self) -> int:
        return self.sp * self.tp

    def ccl_ns(self, kind: str, topology: str, n: int, input_bytes: float) -> float:
        """Blackhole Galaxy kernel-time fit of ``CCL_bandwidth.md`` (input_bytes per device, ns)."""
        fits = {  # (start-up us, per-step us, step count, bytes factor, GB/s)
            ("all_gather", "line"): (3.7, 1.3, n - 1, n - 1, 48.0),
            ("all_gather", "ring"): (3.7, 1.6, n / 2, n - 1, 95.0),
            ("all_reduce", "line"): (4.4, 4.4, n - 1, 2 * (n - 1) / n, 44.0),
        }
        start, step, steps, factor, gb_s = fits[(kind, topology)]
        return (start + step * steps) * 1e3 + factor * input_bytes / gb_s


GALAXY = Galaxy8x4()


@dataclass(frozen=True)
class Scenario:
    name: str
    launch_ns: float
    halo: str  # "p2p" | "sp_all_gather"
    window_reuse: bool
    topk_copy: bool
    d2d: str  # "overlapped" | "serialized"


OPTIMISTIC = Scenario("optimistic", 3_000.0, "p2p", True, False, "overlapped")
CONSERVATIVE = Scenario("conservative", 6_700.0, "sp_all_gather", False, True, "serialized")
SCENARIOS = {s.name: s for s in (OPTIMISTIC, CONSERVATIVE)}


@dataclass(frozen=True)
class OpBound:
    """One graph node on one chip for one chunk (one layer for per-layer nodes).

    ``compute_ns`` (FPU), ``dram_ns`` and ``fabric_ns`` are theoretical and assumed to overlap; ``fixed_ns``
    (collective start-up and launch costs, measured) adds."""

    name: str
    compute_ns: float = 0.0
    dram_ns: float = 0.0
    fabric_ns: float = 0.0
    fixed_ns: float = 0.0
    matmul_flop: float = 0.0
    eltwise: float = 0.0
    dram_bytes: float = 0.0
    sfpu: dict = field(default_factory=dict)

    @property
    def time_ns(self) -> float:
        return max(self.compute_ns, self.dram_ns, self.fabric_ns) + self.fixed_ns


def _op(
    name: str,
    *,
    matmul_flop: float = 0.0,
    fidelity: str = "HiFi2",
    eltwise: float = 0.0,
    dram_bytes: float = 0.0,
    fabric_ns: float = 0.0,
    fixed_ns: float = 0.0,
    sfpu: dict | None = None,
) -> OpBound:
    per_ns = GALAXY.cores * GALAXY.clock_ghz
    cycles = matmul_flop * FIDELITY_PHASES[fidelity] / GALAXY.matmul_flop_per_cycle_core
    cycles += eltwise / GALAXY.eltwise_per_cycle_core
    return OpBound(
        name,
        compute_ns=cycles / per_ns,
        dram_ns=dram_bytes / GALAXY.dram_bytes_per_ns,
        fabric_ns=fabric_ns,
        fixed_ns=fixed_ns,
        matmul_flop=matmul_flop,
        eltwise=eltwise,
        dram_bytes=dram_bytes,
        sfpu=sfpu or {},
    )


# --- operation models of the new nodes (per chip) -----------------------------------------------------------------


def damp_gate(sc: Scenario, width: str = "scalar") -> OpBound:
    """G1.7 / G2.12: ``x_k = x + sigmoid(RMSNorm(x) @ W + b) * (h - x)`` on this chip's ``[640, 6144 / tp]`` shard.

    RMSNorm: sum of squares (mul + add per element), TP all-reduce of the per-row partial sums, rsqrt, two scaling
    multiplies. ``width="scalar"`` (W ``[6144, 1]``): local dot product + TP all-reduce of the per-row partial;
    ``"channel"`` (W ``[6144, 6144]``): TP all-gather of the normed rows, column-parallel matmul. Then sigmoid and
    the lerp (subtract, multiply, add). Mandatory DRAM: read x and h, write x_k, read the chip's weight shard; the
    intermediates stay on chip. Launches: norm, gate projection, sigmoid, lerp (one each, as separate ops) plus each
    collective.
    """
    assert width in ("scalar", "channel"), f"unknown gate width {width}"
    rows, cols = CHUNK // GALAXY.sp, C.EMB_SIZE // GALAXY.tp
    elems = rows * cols
    act = elems * HIDDEN_BYTES
    eltwise = 4 * elems + 3 * elems  # norm: square, sum, two scalings; lerp: subtract, multiply, add
    fabric = GALAXY.ccl_ns("all_reduce", "line", GALAXY.tp, rows * 4)  # norm partial sums, fp32
    if width == "scalar":
        matmul = 2 * elems
        weight = cols * HIDDEN_BYTES
        betas = rows
        fabric += GALAXY.ccl_ns("all_reduce", "line", GALAXY.tp, rows * 4)
    else:
        matmul = 2 * rows * C.EMB_SIZE * cols
        weight = C.EMB_SIZE * cols * HIDDEN_BYTES
        betas = elems
        fabric += GALAXY.ccl_ns("all_gather", "line", GALAXY.tp, act)
    return _op(
        f"damp_gate[{width}]",
        matmul_flop=matmul,
        eltwise=eltwise,
        dram_bytes=3 * act + weight,  # read x and h, write x_k, read the weight shard
        # the collectives sit between the ops, so their time adds rather than overlapping DRAM
        fixed_ns=fabric + (4 + 2) * sc.launch_ns,
        sfpu={"rsqrt": rows, "sigmoid": betas},
    )


def keep_pass_input(sc: Scenario) -> OpBound:
    """G1.2 / G2.3: copy the pass input x0 / x1 (``[640, 1536]`` bf16 per chip) into a persistent buffer, because the
    traced D2D drain refills the input buffer every chunk. One copy per chunk."""
    act = (CHUNK // GALAXY.sp) * (C.EMB_SIZE // GALAXY.tp) * HIDDEN_BYTES
    return _op("keep_pass_input", dram_bytes=2 * act, fixed_ns=sc.launch_ns)


def two_bank_attention_delta(sc: Scenario) -> OpBound:
    """G2.9: what one pass-2 attention core adds over a GLM-5.3 reuse layer's, per layer.

    The bank-A select (G2.8) is the reuse layer's push at the injected top-k, so it is not counted again. Added:
      * window table: this chip's 160 queries need window rows [q0 - 127, q_last]: 127 carry or halo rows plus the
        chunk's own rows, copied into the compact key table (read + write ``(160 + 127)`` pages);
      * index list: window slots first, then the top-k, then the sentinels: read 2048, write 2176 indices per query;
      * ``sparse_sdpa`` over k = 2176 instead of 2048: 128 more keys per query and head (q.k over 576, p.v over 512),
        their exp, and their reads (once per chip when ``window_reuse``, once per query otherwise).
    Launches: table build and index-list build.
    """
    q = CHUNK // GALAXY.chips
    heads, latent, rope = C.NUM_ATTENTION_HEADS, C.KV_LORA_RANK, C.QK_ROPE_HEAD_DIM
    pairs = q * WINDOW
    table_rows = q + WINDOW - 1
    window_reads = table_rows if sc.window_reuse else pairs
    dram = 2 * table_rows * KVPE_PAGE_BYTES
    dram += q * (C.INDEX_TOPK + C.INDEX_TOPK + WINDOW) * INDEX_BYTES
    dram += window_reads * KVPE_PAGE_BYTES + q * WINDOW * INDEX_BYTES
    return _op(
        "two_bank_attention_delta",
        matmul_flop=2 * pairs * heads * (2 * latent + rope),
        eltwise=pairs * heads * 3,  # scale, running max/sum update
        dram_bytes=dram,
        fixed_ns=2 * sc.launch_ns,
        sfpu={"exp": pairs * heads},
    )


def sp_halo(sc: Scenario) -> OpBound:
    """G2.6: the first 127 window rows of SP shard s come from shard s-1 of the same chunk, per layer.

    ``p2p``: 127 pass-2 latent pages from the predecessor (a 2-device line all-gather's cost, one hop).
    ``sp_all_gather``: the graph's default, an SP all-gather of the chunk's 640-row shards (8-device ring).
    The receiver writes what arrives."""
    rows = CHUNK // GALAXY.sp
    if sc.halo == "p2p":
        sent = (WINDOW - 1) * KVPE_PAGE_BYTES
        fabric = GALAXY.ccl_ns("all_gather", "line", 2, sent)
        received = sent
    else:
        fabric = GALAXY.ccl_ns("all_gather", "ring", GALAXY.sp, rows * KVPE_PAGE_BYTES)
        received = (GALAXY.sp - 1) * rows * KVPE_PAGE_BYTES
    return _op(f"sp_halo[{sc.halo}]", dram_bytes=2 * received, fixed_ns=fabric)


def window_carry(sc: Scenario) -> OpBound:
    """G2.7: keep the chunk's last 127 pass-2 latent rows per loop layer per user (read + write), one copy. The
    next chunk's read of the carry is in ``two_bank_attention_delta``'s table build."""
    return _op("window_carry", dram_bytes=2 * (WINDOW - 1) * KVPE_PAGE_BYTES, fixed_ns=sc.launch_ns)


def bank_a_page_copy(sc: Scenario) -> OpBound:
    """G2.2 option (a): copy the 160 pass-1 KVPE pages this chip stored per layer on G1 from the received D2D segment
    into the same (user, layer) slot positions of bank A, per layer. Replaces nothing: pass 2's own latent goes to
    the window (G2.5), and its write is costed as the reuse layer's KV write it replaces."""
    pages = CHUNK // GALAXY.chips
    return _op("bank_a_page_copy", dram_bytes=2 * pages * KVPE_PAGE_BYTES, fixed_ns=sc.launch_ns)


def g1_collect(sc: Scenario) -> OpBound:
    """G1.5 + G1.6, per chunk: slice the 160 KVPE pages stored per loop layer into the send buffer (28 copies), and
    keep the 7 top-k sets ``[160, 2048]`` u32 (resident, or copied into the send buffer when ``topk_copy``)."""
    q = CHUNK // GALAXY.chips
    dram = LOOP_LAYERS * 2 * q * KVPE_PAGE_BYTES
    launches = LOOP_LAYERS
    if sc.topk_copy:
        dram += LOOP_FULL_LAYERS * 2 * q * C.INDEX_TOPK * INDEX_BYTES
        launches += LOOP_FULL_LAYERS
    return _op("g1_collect", dram_bytes=dram, fixed_ns=launches * sc.launch_ns)


def d2d_payload_bytes(boundary: str) -> float:
    """Bytes per chunk over a stage boundary: the activation everywhere; G1->G2 adds the pass-1 KVPE rows of the 28
    loop layers (option a, TP-deduped pages) and the 7 top-k sets."""
    act = CHUNK * C.EMB_SIZE * HIDDEN_BYTES
    if boundary != "G1->G2":
        return act
    return act + LOOP_LAYERS * CHUNK * KVPE_PAGE_BYTES + LOOP_FULL_LAYERS * CHUNK * C.INDEX_TOPK * INDEX_BYTES


SEAM_GB_S = 16 * GALAXY.link_bytes_per_ns  # q171.3.2's SC4 seam: 16 cables x 25 GB/s per direction (assumption)


def d2d_transfer(boundary: str, seam_gb_s: float = SEAM_GB_S) -> OpBound:
    """X.8: one chunk's payload over the inter-Galaxy seam at ``seam_gb_s`` (decimal GB/s = bytes/ns, per direction;
    pass q171.3.2's measured effective rate when known). Per chip, the sender reads and the receiver writes its 1/32
    share. Socket set-up, relay hops inside each Galaxy and flow control are not modelled."""
    payload = d2d_payload_bytes(boundary)
    return _op(f"d2d[{boundary}]", dram_bytes=payload / GALAXY.chips, fabric_ns=payload / seam_gb_s)


# --- composition --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Glm53Anchor:
    """Measured GLM-5.3 costs by probe label (ms per chunk at the probe depth).

    Stage time is ``period_ms``: from the ack of the layer before the stage (the previous chunk's retire for the first
    stage) to the stage's last layer ack, median over chunks d-4..d+4. It spans every layer of the stage, including the
    first layer's work before its KV-write ack (a full-indexer layer's indexer); first-to-last ack (CI chunk_time) does
    not. Measure GLM-5.4 stages the same way.

    ``stage_ms``: GLM-5.3 layers ``G0`` (0-25), ``G1`` (26-53), ``G3`` (54-77); ``full_extra_ms``: a full-indexer
    layer's cost over a reuse layer's; ``glm53_slowest_ms``: GLM-5.3's slowest stage on its own production split
    (the met / unmet reference); ``sc4_ci_tok_s``: GLM-5.3's SC4 CI throughput (optional, context)."""

    stage_ms: dict
    full_extra_ms: dict
    glm53_slowest_ms: dict
    sc4_ci_tok_s: dict = field(default_factory=dict)
    source: dict = field(default_factory=dict)

    def __post_init__(self):
        assert set(self.stage_ms) == {
            "G0",
            "G1",
            "G3",
        }, f"anchor stages must be G0, G1, G3, got {sorted(self.stage_ms)}"

    @classmethod
    def load(cls, path: str | Path) -> "Glm53Anchor":
        d = json.loads(Path(path).read_text())
        return cls(d["stage_ms"], d["full_extra_ms"], d["glm53_slowest_ms"], d.get("sc4_ci_tok_s", {}), d["source"])


@dataclass(frozen=True)
class StageTime:
    stage: str
    base_ms: float  # measured GLM-5.3 work (G2: G1's minus 7 indexers)
    nodes: dict  # new node -> ms per chunk (all layers)

    @property
    def total_ms(self) -> float:
        return self.base_ms + sum(self.nodes.values())

    @property
    def band_ms(self) -> tuple:
        tol = TOLERANCE[self.stage]
        return self.total_ms * (1 - tol), self.total_ms * (1 + tol)


def stage_times(probe: str, anchor: Glm53Anchor, sc: Scenario, gate_width: str = "scalar") -> dict:
    """GLM-5.4 stage times for the chunk at ``probe`` without D2D, in the anchor's ``period_ms`` definition (the
    stage's whole layer span), as the b06u02 measurements the met / unmet rule compares."""
    ms = lambda op: op.time_ns / 1e6
    gate = ms(damp_gate(sc, gate_width))
    keep = ms(keep_pass_input(sc))
    per_layer = {
        "two_bank_attention_delta": two_bank_attention_delta(sc),
        "sp_halo": sp_halo(sc),
        "window_carry": window_carry(sc),
        "bank_a_page_copy": bank_a_page_copy(sc),
    }
    g2_nodes = {name: LOOP_LAYERS * ms(op) for name, op in per_layer.items()}
    g2_nodes.update(keep_pass_input=keep, damp_gate=gate)
    s = anchor.stage_ms
    return {
        "G0": StageTime("G0", s["G0"][probe], {}),
        "G1": StageTime(
            "G1", s["G1"][probe], {"keep_pass_input": keep, "damp_gate": gate, "g1_collect": ms(g1_collect(sc))}
        ),
        "G2": StageTime("G2", s["G1"][probe] - LOOP_FULL_LAYERS * anchor.full_extra_ms[probe], g2_nodes),
        "G3": StageTime("G3", s["G3"][probe], {}),
    }


@dataclass(frozen=True)
class PipelinePrediction:
    probe: str
    scenario: str
    stages: dict  # stage -> StageTime, each stage alone (no D2D)
    d2d_ms: dict  # boundary -> ms per chunk at the seam rate
    sc4_period_ms: float  # with D2D: overlapped -> max(stages, D2D); serialized -> the G1->G2 extra on G1 and G2
    sc4_pacing: str
    glm53_slowest_ms: float
    sc4_ci_tok_s: float | None

    @property
    def slowest_stage(self) -> str:
        return max(self.stages, key=lambda k: self.stages[k].total_ms)

    @property
    def slowest_stage_ms(self) -> float:
        return self.stages[self.slowest_stage].total_ms

    @property
    def tok_s(self) -> float:
        """Composed from the stages alone: one chunk per slowest stage (b06u02 stage-by-stage)."""
        return CHUNK / (self.slowest_stage_ms / 1e3)

    @property
    def ratio_vs_glm53(self) -> float:
        """GLM-5.4 slowest stage / GLM-5.3 slowest stage, both stage by stage (met when <= 1.54)."""
        return self.slowest_stage_ms / self.glm53_slowest_ms

    @property
    def g2_d2d_headroom_ms(self) -> float:
        """How much serialized G1->G2 D2D time G2 absorbs before it, not G1, sets the period."""
        return self.stages["G1"].total_ms - self.stages["G2"].total_ms

    @property
    def sc4_tok_s(self) -> float | None:
        """SC4 period rescaled by GLM-5.3's SC4 CI period over its b06u02 slowest stage at this probe (context)."""
        if self.sc4_ci_tok_s is None:
            return None
        glm53_sc4_period_ms = CHUNK / self.sc4_ci_tok_s * 1e3
        return CHUNK / (self.sc4_period_ms * glm53_sc4_period_ms / self.glm53_slowest_ms / 1e3)


def pipeline(
    probe: str,
    anchor: Glm53Anchor,
    sc: Scenario,
    gate_width: str = "scalar",
    seam_gb_s: float = SEAM_GB_S,
) -> PipelinePrediction:
    """Per-stage times, D2D, and the composed period and throughput of the 4-stage pipeline at ``probe``.

    The activation hop exists on every GLM-5.3 boundary too and stays inside the SC4 rescaling; in the serialized
    scenario only G1->G2's extra segments (KVPE rows and top-k sets) add to G1's and G2's periods."""
    assert probe in PROBES, f"unknown probe {probe}"
    stages = stage_times(probe, anchor, sc, gate_width)
    d2d = {b: d2d_transfer(b, seam_gb_s).time_ns / 1e6 for b in ("G0->G1", "G1->G2", "G2->G3")}
    periods = {k: v.total_ms for k, v in stages.items()}
    if sc.d2d == "overlapped":
        periods.update(d2d)
    else:
        extra = d2d["G1->G2"] - d2d["G0->G1"]
        periods["G1"] += extra
        periods["G2"] += extra
    pacing = max(periods, key=periods.get)
    return PipelinePrediction(
        probe,
        sc.name,
        stages,
        d2d,
        periods[pacing],
        pacing,
        anchor.glm53_slowest_ms[probe],
        anchor.sc4_ci_tok_s.get(probe),
    )


def report(anchor: Glm53Anchor, gate_width: str = "scalar", seam_gb_s: float = SEAM_GB_S) -> str:
    """Markdown table of every probe under both scenarios."""
    lines = [
        "| probe | scenario | G0 ms | G1 ms | G2 ms | G3 ms | slowest | GLM-5.3 slowest ms | ratio (met <= 1.54) | "
        "tok/s (stages) | G1-G2 ms | D2D G1->G2 ms | SC4 pacing | SC4-scaled tok/s |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for probe in PROBES:
        for sc in (OPTIMISTIC, CONSERVATIVE):
            p = pipeline(probe, anchor, sc, gate_width, seam_gb_s)
            st = " | ".join(f"{p.stages[s].total_ms:.1f}" for s in STAGES)
            sc4 = f"{p.sc4_tok_s:,.0f}" if p.sc4_tok_s else "-"
            lines.append(
                f"| {probe} | {sc.name} | {st} | {p.slowest_stage} | {p.glm53_slowest_ms:.1f} | {p.ratio_vs_glm53:.3f} | "
                f"{p.tok_s:,.0f} | {p.g2_d2d_headroom_ms:.1f} | {p.d2d_ms['G1->G2']:.2f} | {p.sc4_pacing} | {sc4} |"
            )
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("anchor", help="Glm53Anchor JSON")
    ap.add_argument("--gate-width", default="scalar", choices=("scalar", "channel"))
    ap.add_argument("--seam-gb-s", type=float, default=SEAM_GB_S, help="G1->G2 D2D effective rate, GB/s")
    args = ap.parse_args()
    print(report(Glm53Anchor.load(args.anchor), args.gate_width, args.seam_gb_s))
