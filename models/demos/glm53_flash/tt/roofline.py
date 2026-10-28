# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only roofline of the GLM-5.3-Flash traced decode step on the 8x4 Blackhole Galaxy (no device, no ttnn).

Layout modelled (``tt/mesh.py``): TP4 over the 4 columns x DP8 over the 8 rows, ``T`` users per row (``8 T`` users on
the mesh), every per-chip activation tile-padded to ``Tp = 32`` rows. Routed experts in the hybrid placement (72 experts
per column, the 2048-wide intermediate split over the 8 rows: 256 per chip; ``GLM53_MOE_LAYOUT=ep`` = 9 full experts per
chip with the fabric dispatch / combine). mHC residual streams ``[Tp, 4 x 4096]`` fp32 on every chip.

Per op class the module computes the bytes moved (weights, activations, KDA / latent / index state, the active experts'
slices with the measured routing), the FLOPs, the bytes through the busiest ethernet link, and the time floors under the
chip peaks in ``CHIP`` (every number with its source); it pairs them with the measured per-op device times of the
2026-10-21 tracy profiles (``_logs/mesh/report_prof*``, chip (0,0) of one traced replay) and the 2026-10-26 e2e steps,
and classifies every op: ``DRAM`` (within 1.6x of the DRAM floor), ``compute``, ``CCL`` (ethernet / collective
latency), ``launch/gap`` (fixed per-op cost dominates), ``L1/NoC`` (far below the DRAM rate on few cores or with small
accesses), ``data-movement`` (layout ops to erase by fusion). Floors are per chip; the step is the sum over the layer
types (31 KDA+MoE, 11 DSA+MoE, 3 KDA+dense, embedding + LM head).

    python -m models.demos.glm53_flash.tt.roofline --users-per-row 4
    python -m models.demos.glm53_flash.tt.roofline --users-per-row 32 --layer "KDA+MoE" --markdown
    python -m models.demos.glm53_flash.tt.roofline --users-per-row 4 --spec-k 3 --accepted 2.0 --markdown

Speculative decode (``--spec-k K``, ``tt/spec_decode.py``): one iteration = V (the 45 layers + head on ``R = B (K + 1)``
candidate rows per chip, dense-range DSA, ``K + 1`` kda_decode launches, the MTP level 1 on all R rows) + the host accept
(``--host-ms``) + C (the KDA replay + blends on the backup state per launch, the index maintenance per candidate) + D
(MTP levels 2 .. K on B rows); ``--accepted`` tokens per iteration give the tok/s per user. Policy ``K = min(K_ctx,
32 // B - 1)`` (``spec_policy_k``), off above one tile row.

Everything here is a model: the ``proj`` measured entries are projections where no profile exists (DSA layers at 32
users/row, the hybrid MoE at 4 users/row on the 8x4), marked as such in the ``src`` column.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass, field, replace

from models.demos.glm53_flash.tt.config import GLM53FlashConfig

TILE = 32
MB = 1e6
KDA_LAYERS = 34
DSA_LAYERS = 11
DENSE_LAYERS = 3  # first_k_dense_replace: layers 0-2 (KDA + dense MLP)

# ----------------------------------------------------------------------------------------------- chip numbers
# value, source. Ranges where the peak is uncertain: the DRAM floor is given at ``dram_gbs`` (achievable, the ratio /
# bound classification) and at ``dram_gbs_spec``; the CCL floor at the Galaxy's measured link rate and at the rated one.
CHIP = {
    "dram_gbs_spec": (
        512.0,
        "tt-metal tech_reports/CCLs/CCL_bandwidth.md: Blackhole 8 GDDR6 channels x ~64 GB/s = ~512 GB/s (device "
        "spec; tt-perf-report's 'DRAM %' denominator; a per-bank measurement of 64 GB/s agrees)",
    ),
    "dram_gbs": (
        400.0,
        "achievable: the best decode-shaped matmuls of the 2026-10-21 profiles reach 393-412 GB/s (kda qkv 393-395, "
        "DSA stem 405, indexer query 412: report_prof_ep32_ep_b4 / report_prof2_l3_b4_stream); 1D mcast configs "
        "200-330 GB/s (tt/matmul_configs.py)",
    ),
    "expert_stream_gbs": (
        326.0,
        "doc/mesh/expert_kernel_sweep.md: expert_stream op 69 active = 231 MB in 707 us = 326 GB/s (64 %), 28 active "
        "= 93.6 MB in 341 us = 274 GB/s (54 %): ~110 us of fixed per-block cost",
    ),
    "kda_state_gbs": (
        250.0,
        "decode ledger section 5: kda_decode moves the fp32 state twice at ~250 GB/s (64 MB in 260 us at 32 users/row)",
    ),
    "clock_ghz": (1.35, "tech_reports/GEMM_FLOPS/GEMM_FLOPS.md: Blackhole 1.35 GHz"),
    "worker_cores": (
        120,
        "tt-perf-report header of the profiles: 'Architecture: blackhole, Worker cores: 120' (13x10 grid, 120 usable "
        "on these P150 Galaxy chips; GEMM_FLOPS.md quotes 13x10 = 130 for a P150 card)",
    ),
    "tflops_per_core": (
        {"LoFi": 5.4, "HiFi2": 2.7, "HiFi3": 1.8, "HiFi4": 1.35},
        "GEMM_FLOPS.md: 2*8*16*16 muladds per cycle at 1.35 GHz = 5.4 TFLOPS per matrix engine at LoFi, / fidelity "
        "(HiFi2 2.7, HiFi4 1.35); tt-perf-report uses the same per-core peaks for its 'FLOPs %'",
    ),
    "elementwise_tflops_per_core": (
        0.128 * 1.35,
        "tech_reports/matrix_engine/matrix_engine.md: elementwise 8*16 ops per cycle = 0.128 TFLOPS per core at 1 GHz "
        "(x1.35 for Blackhole); fp32 elementwise goes through the SFPU and is slower: lower bound only",
    ),
    "eth_link_gbs_rated": (50.0, "CCL_bandwidth.md: Blackhole 400 Gbps = 50 GB/s per link per direction, ~48 payload"),
    "eth_link_gbs": (
        25.0,
        "CCL_bandwidth.md: 'Blackhole Galaxy links currently run at half their rated speed' (25 GB/s per link per "
        "direction); doc/mesh/README.md ep32 analysis: 2 x 24.2 GB/s on the row axis",
    ),
    "links_per_axis": (2, "models/common/modules/tt_ccl.py get_num_links: BHGLX (2, 2) links per axis"),
    "ccl_fit": (
        # collective -> (fixed us, us per hop, GB/s through the busiest link, bottleneck-bytes factor of N)
        {
            "all_gather": (3.7, 1.3, 48.0, "N-1"),
            "reduce_scatter": (1.3, 3.0, 40.0, "(N-1)/N"),
            "all_reduce": (4.4, 4.4, 44.0, "2(N-1)/N"),
            "all_to_all": (1.8, 1.4, 33.0, "floor(N/2)ceil(N/2)/N"),
        },
        "CCL_bandwidth.md 'Kernel time' fits for Blackhole LINE topologies (the measured Galaxy at 25 GB/s per link; "
        "the fits' link count is the machine's 2 per axis, so this module scales their GB/s by link_gbs/25 x links/2); "
        "ring fits exist only for N = 8 (all_gather 95 GB/s, reduce_scatter 82, all_reduce 86, all_to_all 68)",
    ),
    "hop_latency_us": ({"1d": 0.515, "2d": 0.619}, "CCL_bandwidth.md per-hop forwarding latency, Blackhole"),
    "op_floor_us": (
        1.0,
        "smallest device ops in the profiles: Slice 0.84 us, Typecast 1.1 us (report_prof2_l3_b4_stream); one program "
        "launch inside a trace cannot go below ~1 us",
    ),
    "op_gap_us": (
        0.46,
        "decode ledger section 3: median in-trace op-to-op gap (0.46 us; 1.0-1.5 after single-core ops)",
    ),
    "noc_gbs_per_core": (
        86.0,
        "ASSUMED (not measured in this tree): 64 B/cycle x 1.35 GHz per NoC port; only used to bound L1-resident "
        "elementwise work. Measured small-access rates: ~14 GB/s per core with 1 KB tile reads (expert_kernel_sweep.md)",
    ),
}

# expert_stream op cost model (doc/mesh/expert_kernel_sweep.md: 28 active = 93.5 MB in 341 us, 69 active = 231 MB in 707 us)
EXPERT_FIXED_US = 92.0
EXPERT_GBS = 376.0

BYTES = {"bf16": 2.0, "fp32": 4.0, "bfp8": 1088 / 1024, "bfp4": 576 / 1024, "u16": 2.0, "u32": 4.0}


def chip(name):
    return CHIP[name][0]


# ----------------------------------------------------------------------------------------------- layout
@dataclass(frozen=True)
class Layout:
    """The decode configuration the floors are computed for (one chip of the 8x4; the step waits for the slowest)."""

    rows: int = 8
    cols: int = 4
    users_per_row: int = 4
    moe: str = "hybrid"  # hybrid | ep
    expert_path: str = (
        "block"  # hybrid at T <= 32: block (sparse tuned block: the no-knob e2e path) | stream (GLM53_STREAM_EXPERTS=1)
    )
    expert_w13_dtype: str = "bfp8"
    expert_w2_dtype: str = "bfp8"
    isl: int = 16  # context position of every user (index scoring and SDPA state terms)
    routing: str = "measured"  # measured (layer-4 real routing samples) | uniform (Monte-Carlo, seeded)
    active_experts: float | None = None  # override: active experts per chip (hybrid: of 72; ep: of 9)
    dram_gbs: float = chip("dram_gbs")
    link_gbs: float = chip("eth_link_gbs")
    links: int = chip("links_per_axis")
    topology: str = "line"  # line | ring (CCL fits)
    mhc_pre_block: bool = True  # A1 fused pre-block (GLM53_MHC_PRE_BLOCK, on by default since 2026-10-28)
    dsa_state_fused: bool = False  # B5 (GLM53_DSA_DECODE_STATE=1, off by default)
    # speculative decode (tt/spec_decode.py): K drafts per user, R = B (K + 1) candidate rows per chip in the verify
    spec_k: int = 0
    dsa_index: bool = True  # False: dense-range rows (verify / MTP chain): no index scoring, selection or gather
    accepted: float = 2.0  # tokens committed per iteration (1 + mean accepted drafts)
    host_ms: float = 2.0  # host accept: readback of the R ids + input refresh per iteration

    @property
    def users(self) -> int:
        return self.rows * self.users_per_row

    @property
    def tp(self) -> int:
        return TILE  # tile-padded rows per chip for every T <= 32

    @property
    def verify_rows(self) -> int:
        """R = B (K + 1) candidate rows per chip (spec_decode.candidate_rows); refused above one tile row."""
        r = self.users_per_row * (self.spec_k + 1)
        if r > TILE:
            raise ValueError(f"B={self.users_per_row} K={self.spec_k}: B (K + 1) = {r} exceeds {TILE} rows")
        return r

    @property
    def rows_per_chip(self) -> int:
        """Activation rows per chip: the verify's R under speculation, else the users per row."""
        return self.verify_rows if self.spec_k else self.users_per_row

    @property
    def point(self) -> str:
        """Which measured column applies: 32 users (4/row) or 256 users (32/row); other T interpolate the state terms."""
        return "256u" if self.users_per_row >= 32 else "32u"


def spec_policy_k(users_per_row: int, k_ctx: int) -> int:
    """The draft length the serving policy picks (``spec_decode.spec_draft_len``): ``K = min(K_ctx, 32 // B - 1)``, 0 =
    plain decode (at ``B (K + 1) > 32`` the verify leaves one tile row and the weight-bandwidth-bound regime)."""
    return max(0, min(int(k_ctx), TILE // users_per_row - 1))


# ----------------------------------------------------------------------------------------------- floors
def dram_us(nbytes: float, gbs: float) -> float:
    return nbytes / (gbs * 1e3)


def compute_us(flops: float, fidelity: str = "HiFi2", cores: int | None = None) -> float:
    cores = chip("worker_cores") if cores is None else cores
    return flops / (chip("tflops_per_core")[fidelity] * 1e12 * cores) * 1e6


def ccl_us(collective: str, n: int, bytes_in: float, *, links=None, link_gbs=None, topology="line") -> float:
    """Blackhole kernel-time fit (CCL_bandwidth.md) for ``n`` devices on one axis, ``bytes_in`` per device, scaled to
    ``links`` x ``link_gbs`` (the fits were measured with 2 links at 25 GB/s per link per direction)."""
    links = chip("links_per_axis") if links is None else links
    link_gbs = chip("eth_link_gbs") if link_gbs is None else link_gbs
    fixed, per_hop, gbs, _ = chip("ccl_fit")[collective]
    if topology == "ring" and n == 8:
        fixed, per_hop, gbs = {
            "all_gather": (3.7, 1.6, 95.0),
            "reduce_scatter": (1.3, 2.1, 82.0),
            "all_reduce": (4.4, 3.9, 86.0),
            "all_to_all": (1.8, 2.2, 68.0),
        }[collective]
        hops = n / 2
    else:
        hops = n - 1
    factor = {
        "all_gather": n - 1,
        "reduce_scatter": (n - 1) / n,
        "all_reduce": 2 * (n - 1) / n,
        "all_to_all": (n // 2) * ((n + 1) // 2) / n,
    }[collective]
    eff_gbs = gbs * (link_gbs / 25.0) * (links / 2)
    return fixed + per_hop * hops + factor * bytes_in / (eff_gbs * 1e3)


def expected_active_experts(
    n_experts: int, top_k: int, tokens: int, group_size: int, *, trials: int = 64, seed: int = 0
) -> tuple[float, float]:
    """Uniform-routing Monte-Carlo: (mean distinct experts per group, mean over trials of the max over the groups) when
    ``tokens`` tokens each pick ``top_k`` distinct experts of ``n_experts`` split into groups of ``group_size``."""
    rng = random.Random(seed)
    groups = n_experts // group_size
    means, maxes = 0.0, 0.0
    for _ in range(trials):
        hit = [set() for _ in range(groups)]
        for _t in range(tokens):
            for e in rng.sample(range(n_experts), top_k):
                hit[e // group_size].add(e)
        counts = [len(h) for h in hit]
        means += sum(counts) / groups
        maxes += max(counts)
    return means / trials, maxes / trials


# measured active-expert counts (real routing samples, doc/mesh/expert_kernel_sweep.md): hybrid column of 72 at 32 and
# 256 gathered tokens; EP32 chip of 9: the worst chip of a column (per-chip expert-stage spread 33-502 us at 32 users =
# ~0-6 experts, 112-661 us at 256 = ~1.4-8 of 9)
MEASURED_ACTIVE = {("hybrid", 32): 28.0, ("hybrid", 256): 58.0, ("ep", 32): 6.0, ("ep", 256): 8.0}


def active_experts(cfg: GLM53FlashConfig, lay: Layout) -> tuple[float, str]:
    """Active experts per chip whose weights the step reads (the slowest chip of the step), with the source. The measured
    real-routing samples (32 and 256 gathered tokens) are interpolated log-linearly in the token count; outside them the
    uniform Monte-Carlo's ratio extrapolates; ``routing="uniform"`` uses the Monte-Carlo alone."""
    if lay.active_experts is not None:
        return float(lay.active_experts), "override"
    tokens = lay.rows * lay.rows_per_chip  # every row's rows are gathered (hybrid) or dispatched (EP32)
    e, k = cfg.n_routed_experts, cfg.num_experts_per_tok
    group = e // lay.cols if lay.moe == "hybrid" else e // (lay.rows * lay.cols)
    mc = lambda n: expected_active_experts(e, k, n, group)
    if lay.routing == "measured":
        lo, hi = MEASURED_ACTIVE[(lay.moe, 32)], MEASURED_ACTIVE[(lay.moe, 256)]
        if tokens in (32, 256):
            return (lo if tokens == 32 else hi), "measured (layer-4 real routing)"
        if 32 < tokens < 256:
            f = (math.log2(tokens) - 5) / 3
            return lo + (hi - lo) * f, "measured 32 / 256-token samples, log-linear interpolation"
        ref, ref_tokens = (lo, 32) if tokens < 32 else (hi, 256)
        est = min(float(group), ref * mc(tokens)[1] / mc(ref_tokens)[1])
        return est, f"measured {ref_tokens}-token sample scaled by the uniform Monte-Carlo ratio"
    mean, mx = mc(tokens)
    return mx, f"uniform Monte-Carlo max over groups (mean {mean:.1f})"


# ----------------------------------------------------------------------------------------------- op rows
@dataclass
class OpRow:
    key: str
    module: str
    op: str
    kind: str = "compute"  # compute | layout | ccl | generic | state | gaps
    n_ops: int = 1
    dram_bytes: float = 0.0  # bytes read + written per chip
    flops: float = 0.0
    fidelity: str = "HiFi2"
    cores: int = 120
    ccl: tuple | None = None  # (collective, n_devices, bytes_in per device)
    measured: dict = field(default_factory=dict)  # point -> (us, src)
    gap_us: float = 0.0  # in-trace gap before the op (measured)
    bound_hint: str | None = None
    action_id: str = "-"
    action: str = "keep"
    risk: str = "-"
    after: dict = field(default_factory=dict)  # point -> expected us after the action
    steps: list = field(default_factory=list)  # [(action_id, {point: after_us}, risk, text)] applied in order
    note: str = ""

    # filled by evaluate()
    floor_us: float = 0.0
    floor_spec_us: float = 0.0
    floors: dict = field(default_factory=dict)
    bound: str = ""
    measured_us: float | None = None
    measured_src: str = ""
    ratio: float | None = None
    after_us: float | None = None


def _w(k: int, n: int, dtype: str) -> float:
    return k * n * BYTES[dtype]


def _act(tp: int, k: int, n: int, in_dtype="bf16", out_dtype="bf16") -> float:
    return tp * (k * BYTES[in_dtype] + n * BYTES[out_dtype])


def _ccl_bytes(collective: str, n: int, bytes_in: float) -> float:
    """Bytes through the busiest link (CCL_bandwidth.md)."""
    factor = {
        "all_gather": n - 1,
        "reduce_scatter": (n - 1) / n,
        "all_reduce": 2 * (n - 1) / n,
        "all_to_all": (n // 2) * ((n + 1) // 2) / n,
    }[collective]
    return factor * bytes_in


def mhc_wrap_rows(cfg: GLM53FlashConfig, lay: Layout, wrap: str) -> list[OpRow]:
    """One mHC wrap (pre-block + post-mix) around a sublayer; measured ep_b4 / ep_b32 ids 248-263 / 271 (identical at
    every batch: the streams are tile-padded)."""
    tp, n, h = lay.tp, cfg.hc_mult, cfg.hidden_size
    nc = n * h
    x_bytes = tp * nc * BYTES["fp32"]  # 2 MB residual streams
    fn_bytes = nc * TILE * BYTES["fp32"]  # fn_k [16384, 32] fp32 (24 real columns)
    h_bytes = tp * h * BYTES["bf16"]
    m = f"MHC_{wrap}"
    rows: list[OpRow] = []
    if lay.mhc_pre_block:
        rows.append(
            OpRow(
                "mhc.pre_block_fused",
                m,
                "mhc_pre_block fused generic op (projection, norms, Sinkhorn, h_mix, input norm)",
                kind="generic",
                dram_bytes=x_bytes + fn_bytes + h_bytes,
                flops=2 * tp * nc * TILE,
                fidelity="HiFi4",
                measured={
                    "32u": (101.0, "A1 chip test 2026-10-21 (203 -> 101 us per wrap); 22-layer chain 37.7 -> 33.3 ms"),
                    "256u": (101.0, "same (batch-independent)"),
                },
                action_id="D13",
                action="column-shard the streams (DeepSeek distributed-norm style): elementwise work / 4",
                risk="H",
                after={"32u": 40.0, "256u": 40.0},
            )
        )
    else:
        rows += [
            OpRow(
                "mhc.proj",
                m,
                "K-split projection matmul [1,32,32,512]x[1,32,512,32] fp32 HiFi4",
                dram_bytes=x_bytes + fn_bytes + 32 * tp * TILE * 4,
                flops=2 * tp * nc * TILE,
                fidelity="HiFi4",
                cores=32,
                measured={"32u": (12.9, "ep_b4 #248"), "256u": (12.7, "ep_b32 #245")},
                steps=[
                    (
                        "A1",
                        {"32u": 101.0, "256u": 101.0},
                        "M",
                        "fuse the 18 pre-block ops into mhc_pre_block (measured 101 us per wrap vs 168: this row carries "
                        "the fused op's cost, the other pre-block rows go to 0)",
                    ),
                    (
                        "A4",
                        {"32u": 57.0, "256u": 57.0},
                        "M",
                        "per-row Sinkhorn inside the fused op (one token row per core, 32 cores: 52 -> ~8 us)",
                    ),
                ],
            ),
            OpRow(
                "mhc.xx",
                m,
                "x*x fp32 (2 MB in, 2 MB out)",
                dram_bytes=2 * x_bytes,
                flops=tp * nc,
                measured={"32u": (23.2, "ep_b4 #249"), "256u": (23.8, "ep_b32 #246")},
                bound_hint="DRAM",
                action_id="A1",
                action="fused pre-block: the square never leaves DST",
                risk="M",
                after={"32u": 0.0, "256u": 0.0},
            ),
            OpRow(
                "mhc.ssp",
                m,
                "sum of squares over W (reduce, 32 cores)",
                dram_bytes=x_bytes,
                flops=tp * nc,
                cores=32,
                measured={"32u": (10.4, "ep_b4 #250"), "256u": (10.4, "ep_b32 #247")},
                action_id="A1",
                action="fused pre-block",
                risk="M",
                after={"32u": 0.0, "256u": 0.0},
            ),
            OpRow(
                "mhc.ksum",
                m,
                "exact K-sum chains: 2 x (transpose, reduce, transpose) + fill_pad on [1,32,32,32] fp32",
                kind="layout",
                n_ops=7,
                dram_bytes=4 * 32 * tp * TILE * 4,
                cores=32,
                measured={"32u": (24.9, "ep_b4 #251-257"), "256u": (24.7, "ep_b32 #248-254")},
                action_id="A1",
                action="fused pre-block (A9 alone: fast_reduce_nc, -10 ops)",
                risk="M",
                after={"32u": 0.0, "256u": 0.0},
            ),
            OpRow(
                "mhc.rsqrt_mix",
                m,
                "rs = rsqrt(ss + eps*nC); mixes = mixes_un * rs (2 tiny elementwise ops)",
                n_ops=2,
                dram_bytes=4 * tp * TILE * 4,
                measured={"32u": (12.3, "ep_b4 #258-259"), "256u": (12.5, "ep_b32 #255-256")},
                bound_hint="launch/gap",
                action_id="A1",
                action="fused pre-block",
                risk="M",
                after={"32u": 0.0, "256u": 0.0},
            ),
            OpRow(
                "mhc.sinkhorn",
                m,
                "mhc_split_sinkhorn: 20 iterations on ONE core for the token tile",
                dram_bytes=2 * tp * TILE * 4,
                flops=cfg.hc_sinkhorn_iters * 2 * tp * (n * n) * 4,
                fidelity="HiFi4",
                cores=1,
                measured={"32u": (52.1, "ep_b4 #260"), "256u": (52.1, "ep_b32 #257")},
                bound_hint="L1/NoC",
                action_id="A1",
                action="inside the fused pre-block (its 101 us include this single-core chain: A4 on the mhc.proj row)",
                risk="M",
                after={"32u": 0.0, "256u": 0.0},
                note="single-core serial SFPU chain; the fused A1 op keeps it (101 us total per wrap)",
            ),
            OpRow(
                "mhc.hmix",
                m,
                "h_mix generic op: h = sum_i pre_i x_i (fp32 -> fp32 h [32,4096])",
                kind="generic",
                dram_bytes=x_bytes + tp * h * 4,
                flops=2 * tp * nc,
                measured={"32u": (17.9, "ep_b4 #261"), "256u": (18.0, "ep_b32 #258")},
                action_id="A1",
                action="fused pre-block (reads x once; writes h bf16 sharded)",
                risk="M",
                after={"32u": 0.0, "256u": 0.0},
            ),
            OpRow(
                "mhc.norm",
                m,
                "A3 sharded input norm: typecast + interleaved_to_sharded + rms_norm (32 cores) + sharded_to_interleaved",
                n_ops=4,
                dram_bytes=tp * h * 4 + 3 * h_bytes,
                measured={
                    "32u": (14.0, "A3 2026-10-26: traced wrap 246 -> 210 us (was typecast 3.3 + 1-core rms_norm 46.4)"),
                    "256u": (14.0, "same"),
                },
                bound_hint="launch/gap",
                action_id="A1",
                action="fused pre-block writes the normed bf16 h directly (norm folded)",
                risk="M",
                after={"32u": 0.0, "256u": 0.0},
                note="A3 validated 2026-10-26 (default GLM53_MHC_NORM=sharded)",
            ),
        ]
    rows.append(
        OpRow(
            "mhc.post_mix",
            m.replace("PRE", "POST"),
            "mhc_post_mix v2 generic op: exact fp32 residual update x += post * y (2 MB in, 2 MB out, y bf16)",
            kind="generic",
            dram_bytes=2 * x_bytes + h_bytes,
            flops=2 * tp * nc * 2,
            measured={"32u": (34.3, "ep_b4 #271"), "256u": (34.0, "ep_b32 #268")},
            action_id="P1",
            action="tune the generic op's access pattern (125 GB/s today: resident operands, 8 tiles in flight per core) "
            "or keep the streams L1-resident between wraps (D13)",
            risk="L-M",
            after={"32u": 20.0, "256u": 20.0},
        )
    )
    return rows


def kda_rows(cfg: GLM53FlashConfig, lay: Layout) -> list[OpRow]:
    """KDA attention sublayer per chip (16 of 64 heads, K = V = 128): measured ep_b4 #264-270 / ep_b32 #261-267."""
    tp, t = lay.tp, lay.users_per_row
    h = cfg.hidden_size
    hl = cfg.kda_num_heads // lay.cols
    k = cfg.kda_head_dim
    qkv_n = 3 * hl * k  # 6144
    gate_mid = 288  # [f_a | g_a | b] columns
    gate_out = 2 * hl * k + 32  # [f | og | b_logits] 4128
    state_bytes = hl * k * k * BYTES["fp32"]  # 1 MB per user per chip
    conv_bytes = (cfg.kda_conv_kernel - 1) * qkv_n * BYTES["bf16"]
    m = "ATTN_KDA"
    kda_meas = {"32u": (74.9, "ep_b4 #267 (gap 34.2 before it)"), "256u": (259.7, "ep_b32 #264 (gap 34.3)")}
    if t not in (4, 32):  # two-point fit of the measured op: 48.6 us fixed + 6.6 us per user
        kda_meas = {lay.point: (48.6 + 6.6 * t, "fit of ep_b4 / ep_b32: 48.6 + 6.6 us x users")}
    if lay.spec_k:
        return kda_rows_verify(cfg, lay)
    return [
        OpRow(
            "kda.qkv",
            m,
            "qkv matmul [32,4096]x[4096,6144] bfp8 HiFi2 (1D mcast, 96 cores)",
            dram_bytes=_w(h, qkv_n, "bfp8") + _act(tp, h, qkv_n),
            flops=2 * tp * h * qkv_n,
            cores=96,
            measured={"32u": (65.2, "ep_b4 #264, 393 GB/s"), "256u": (65.3, "ep_b32 #261")},
            action_id="-",
            action="keep (at the achievable DRAM rate); only bfp4 weights would cut it (accuracy-gated)",
            risk="-",
        ),
        OpRow(
            "kda.gate1",
            m,
            "gate matmul 1 [32,4096]x[4096,288] bf16 (9 cores)",
            dram_bytes=_w(h, gate_mid, "bf16") + _act(tp, h, gate_mid),
            flops=2 * tp * h * gate_mid,
            cores=9,
            measured={"32u": (14.1, "ep_b4 #265, 187 GB/s"), "256u": (14.0, "ep_b32 #262")},
            bound_hint="L1/NoC",
            action_id="K1",
            action="append the 288 gate columns to the qkv weight (one [4096, 6432] matmul at ~390 GB/s)",
            risk="L",
            after={"32u": 7.0, "256u": 7.0},
        ),
        OpRow(
            "kda.gate2",
            m,
            "gate matmul 2 block-diagonal [32,288]x[288,4128] bf16 -> fp32 (65 cores)",
            dram_bytes=_w(gate_mid, gate_out, "bf16") + _act(tp, gate_mid, gate_out, "bf16", "fp32"),
            flops=2 * tp * gate_mid * gate_out,
            fidelity="HiFi4",
            cores=65,
            measured={"32u": (9.4, "ep_b4 #266"), "256u": (9.1, "ep_b32 #263")},
            action_id="K2",
            action="fold the block-diagonal [f_b | g_b] expansion into kda_decode (per-head [18 x 128] weights in L1)",
            risk="M",
            after={"32u": 0.0, "256u": 0.0},
        ),
        OpRow(
            "kda.decode",
            m,
            f"kda_decode generic op: conv, norms, gates, delta rule, gated norm; fp32 state [{t},16,128,128] read + written",
            kind="state",
            dram_bytes=2 * t * state_bytes + 2 * t * conv_bytes + _act(tp, qkv_n, hl * k) + tp * gate_out * 4,
            flops=t * hl * (6 * k * k + 10 * k),
            fidelity="HiFi4",
            cores=80,
            measured=kda_meas,
            gap_us=34.2,
            action_id="K3",
            action="cut the ~49 us fixed cost + 34 us pre-op gap (runtime-arg volume of 80 cores x 3 RISCs: common "
            "args + per-core index; kernel prologue); the per-user slope (6.6 us = 303 GB/s) is already at DRAM rate",
            risk="M",
            after={"32u": 40.0, "256u": 215.0},
            note="B6 rectangular ranges did not move the 34 us gap (e2e 83.08 vs 83.07 ms)",
        ),
        OpRow(
            "kda.o_proj",
            m,
            "o_proj matmul [32,2048]x[2048,4096] bfp8 (22 cores)",
            dram_bytes=_w(hl * k, h, "bfp8") + _act(tp, hl * k, h),
            flops=2 * tp * hl * k * h,
            cores=22,
            measured={"32u": (27.1, "ep_b4 #268, 348 GB/s"), "256u": (26.1, "ep_b32 #265")},
        ),
        OpRow(
            "kda.all_reduce",
            m,
            "TP all-reduce over the 4 columns of [32,4096] bf16 (reduce_scatter + all_gather composite)",
            kind="ccl",
            n_ops=2,
            ccl=("all_reduce", lay.cols, tp * h * BYTES["bf16"]),
            measured={"32u": (51.0, "ep_b4 #269-270 (RS 38.7 + AG 12.3)"), "256u": (51.7, "ep_b32 #266-267")},
            action_id="C2",
            action="the MoE all-reduce on the same shape takes 30 us: the extra ~20 us is column skew / input placement "
            "(RS waits for the slowest column after kda_decode); B7 gather form measured 46 us (no gain) -> keep ccl; "
            "D13 column-sharded mHC would turn it into RS + AG of the streams (neutral)",
            risk="L",
            after={"32u": 30.0, "256u": 30.0},
        ),
    ]


def kda_rows_verify(cfg: GLM53FlashConfig, lay: Layout) -> list[OpRow]:
    """The KDA sublayer of a verify step (``KdaSpecState.verify_attention``): the projections on all R rows (tile-padded:
    as the plain rows), the state backed up (``ttnn.copy`` of the recurrent + conv), then K + 1 sequential ``kda_decode``
    launches on the B-row candidate slabs, concat, o_proj, all-reduce."""
    b, k = lay.users_per_row, lay.spec_k
    hl, kd = cfg.kda_num_heads // lay.cols, cfg.kda_head_dim
    state_bytes = hl * kd * kd * BYTES["fp32"]
    conv_bytes = (cfg.kda_conv_kernel - 1) * 3 * hl * kd * BYTES["bf16"]
    plain = kda_rows(cfg, replace(lay, spec_k=0))
    out = []
    for r in plain:
        if r.key != "kda.decode":
            out.append(r)
            continue
        out.append(
            OpRow(
                "kda.spec_backup",
                r.module,
                f"backup S_0: ttnn.copy of the recurrent [{b},16,128,128] fp32 + conv history (verify)",
                kind="state",
                n_ops=2,
                dram_bytes=2 * b * (state_bytes + conv_bytes),
                measured={
                    lay.point: (2.0 + 2 * b * (state_bytes + conv_bytes) / 250e3, "proj: copy at 250 GB/s + 2 us")
                },
                action_id="SP1",
                action="write the commit's selected state inside kda_decode's writer (per-user destination select): the "
                "backup copy, the replay and the 6 blend ops per launch collapse into the launches themselves",
                risk="M",
                after={lay.point: 2.0 + 2 * b * (state_bytes + conv_bytes) / 250e3},
            )
        )
        out.append(
            OpRow(
                "kda.decode",
                r.module,
                f"{k + 1} sequential kda_decode launches on the candidate slabs ([{b}] users each; state in place)",
                kind="state",
                n_ops=k + 1,
                dram_bytes=(k + 1)
                * (2 * b * (state_bytes + conv_bytes) + 32 * (3 * hl * kd * 2 + (2 * hl * kd + 32) * 4)),
                flops=(k + 1) * b * hl * (6 * kd * kd + 10 * kd),
                fidelity="HiFi4",
                cores=80,
                measured={
                    lay.point: (
                        (k + 1) * (48.6 + 6.6 * b),
                        f"fit of ep_b4 / ep_b32 per launch: {k + 1} x (48.6 + 6.6 x {b})",
                    )
                },
                gap_us=34.2 * (k + 1),
                action_id="K3",
                action="the fixed 49 us + 34 us gap per launch is paid K + 1 times: K3 (runtime-arg volume, prologue) matters "
                "K + 1 times more under speculation",
                risk="M",
                after={lay.point: (k + 1) * (14.0 + 6.6 * b)},
            )
        )
    return out


def dsa_rows(cfg: GLM53FlashConfig, lay: Layout) -> list[OpRow]:
    """DSA (MLA + DeepSeek sparse index) attention per chip (16 of 64 heads): measured on the (2,4) block at 4 users/row
    (report_prof2_l3_b4_stream #438-487: identical per-chip TP4 shapes); no profile exists at 32 users/row, the state
    terms there are projections (``proj``). ``dsa_index=False`` (the verify / MTP chain rows of the dense range): the
    index query, scoring, selection and pool gather are dropped, the state maintenance is the per-candidate latent
    append + the raw k / gate row caches, and the flash-MLA decode runs on R rows over the users' pages."""
    tp, t = lay.tp, lay.rows_per_chip
    if not lay.dsa_index or lay.spec_k:  # the verify never scores / selects / gathers: its index work is the commit's
        return dsa_rows_dense(cfg, lay)
    h = cfg.hidden_size
    hl = cfg.num_attention_heads // lay.cols
    lq, l, dq, dv = cfg.q_lora_rank, cfg.kv_lora_rank, cfg.qk_head_dim, cfg.v_head_dim
    hi, di = cfg.index_n_heads, cfg.index_head_dim
    stem_n = lq + l + 2 * di + hi  # q_a | kv_a | k | g | w = 2336
    sel = cfg.index_max_selected_tokens  # 2051 selected tokens per user
    sel_pad = -(-sel // TILE) * TILE
    pools = max(1, math.ceil(lay.isl / cfg.index_kpool))
    m = "ATTN_DSA"
    point = lay.point
    scale = t / 4.0  # state ops measured at 4 users/row
    rows = [
        OpRow(
            "dsa.stem",
            m,
            "decode stem matmul [32,4096]x[4096,2336] bf16 (q_a | kv_a | k | g | w; 37 cores)",
            dram_bytes=_w(h, stem_n, "bf16") + _act(tp, h, stem_n),
            flops=2 * tp * h * stem_n,
            cores=37,
            measured={"32u": (48.3, "blk_l3_b4 #438, 405 GB/s"), "256u": (48.3, "same shape")},
            action_id="D1",
            action="q_a part in bfp8 (kv_a stays bf16: it is the cached latent): -5.9 MB",
            risk="M (accuracy)",
            after={"32u": 34.0, "256u": 34.0},
        ),
    ]
    if lay.dsa_state_fused:
        rows.append(
            OpRow(
                "dsa.state_fused",
                m,
                "dsa_decode_state program: q_a norm, latent append, ring + pool maintenance in place (B5)",
                kind="state",
                dram_bytes=tp * stem_n * 2 + t * (l + 4 * di + 2 * di) * 2 * 2,
                measured={
                    "32u": (40.0, "proj: B5 single-chip 75 us at B=32; e2e -1.2 ms/11 layers at 64 K vs the composite"),
                    "256u": (75.0, "B5 single-chip measurement at B=32"),
                },
                action_id="B5",
                action="validated path pending the pool-block diagnostic; keep",
                risk="M",
            )
        )
    else:
        rows.append(
            OpRow(
                "dsa.state_composite",
                m,
                "state maintenance composite: 5 stem slices, 3 single-core norms, 4 x (interleaved_to_sharded + "
                "paged_update_cache), pooling (typecast x2, add, softmax, mul, fill_pad, reduce, typecast)",
                kind="layout",
                n_ops=29,
                dram_bytes=tp * stem_n * 2 * 3 + t * (l + 6 * di) * 2 * 4,
                measured={
                    "32u": (100.3, "blk_l3_b4 #439-468 (20.4 us of gaps besides)"),
                    "256u": (100.3 + 12.0 * (scale - 1), "proj: the [B,P,D] pooling ops grow with B"),
                },
                gap_us=20.4,
                action_id="B5",
                action="GLM53_DSA_DECODE_STATE=1: one dsa_decode_state program (29 -> 1 op; measured -109 us/layer e2e)",
                risk="M",
                after={"32u": 40.0, "256u": 75.0},
            )
        )
    rows += [
        OpRow(
            "dsa.q_b",
            m,
            "q_b matmul [32,1536]x[1536,4096] bfp8 (22 cores)"
            + (" folded with Wkb1: [1536, 8192]" if lay.dsa_state_fused else ""),
            dram_bytes=_w(lq, hl * (l if lay.dsa_state_fused else dq), "bfp8") + _act(tp, lq, hl * dq),
            flops=2 * tp * lq * hl * dq,
            cores=22,
            measured={
                "32u": (
                    35.0 if lay.dsa_state_fused else 20.3,
                    "blk_l3_b4 #445" if not lay.dsa_state_fused else "proj 13.4 MB at 390 GB/s",
                ),
                "256u": (35.0 if lay.dsa_state_fused else 20.3, "same shape"),
            },
            note="the q fold (B5) doubles the weight (6.7 -> 13.4 MB, +17 us floor) but removes 34.5 us of ops",
        ),
    ]
    if not lay.dsa_state_fused:
        rows += [
            OpRow(
                "dsa.q_view",
                m,
                "reshape q -> [1,T,H,Dq] (real copy 11 us) + transpose -> [1,H,T,Dq]",
                kind="layout",
                n_ops=2,
                dram_bytes=4 * tp * hl * dq * 2,
                measured={"32u": (14.5, "blk_l3_b4 #469-470"), "256u": (14.5, "same shape")},
                action_id="B5",
                action="free head-major view (T <= 32: the packed tile sequence IS the head-major sequence)",
                risk="L",
                after={"32u": 0.0, "256u": 0.0},
            ),
            OpRow(
                "dsa.absorb_q",
                m,
                "absorb_q bmm b=16 [32,256]x[256,512] bfp8 (16 cores)",
                dram_bytes=hl * _w(dq, l, "bfp8") + hl * _act(tp, dq, l),
                flops=2 * tp * hl * dq * l,
                cores=16,
                measured={"32u": (20.0, "blk_l3_b4 #471, 144 GB/s"), "256u": (20.0, "same shape")},
                bound_hint="L1/NoC",
                action_id="B5",
                action="fold into q_b (W_qb1 = Wqb . Wkb1 per head): one matmul; or keep the bmm on 64 cores",
                risk="L",
                after={"32u": 0.0, "256u": 0.0},
            ),
        ]
    rows += [
        OpRow(
            "dsa.idx_query",
            m,
            "indexer query matmul [32,1536]x[1536,4096] bf16 (64 cores)",
            dram_bytes=_w(lq, hi * di, "bf16") + _act(tp, lq, hi * di),
            flops=2 * tp * lq * hi * di,
            cores=64,
            measured={"32u": (31.4, "blk_l3_b4 #472, 412 GB/s"), "256u": (31.4, "same shape")},
            action_id="D2",
            action="bfp8 indexer query weight (-6 MB) if the selected-pool sets stay identical (A/B on the index)",
            risk="M (selection)",
            after={"32u": 18.0, "256u": 18.0},
        ),
        OpRow(
            "dsa.idx_view",
            m,
            "reshape q_idx (real copy 19.8 us) + 2 transposes (scorer layout)",
            kind="layout",
            n_ops=3,
            dram_bytes=6 * tp * hi * di * 2,
            measured={
                "32u": (0.0 if lay.dsa_state_fused else 27.7, "blk_l3_b4 #473-475"),
                "256u": (0.0 if lay.dsa_state_fused else 27.7, "same shape"),
            },
            action_id="B5",
            action="free view (tt/indexer.py free_view, part of B5)",
            risk="L",
            after={"32u": 0.0, "256u": 0.0},
        ),
        OpRow(
            "dsa.index_score",
            m,
            f"index_score_decode generic op over {pools} pooled keys per user (B x pools x 128 bf16)",
            kind="state",
            dram_bytes=t * pools * di * 2 + tp * hi * di * 2 + t * pools * 4,
            flops=2 * t * hi * pools * di,
            cores=min(120, 12 * max(1, t // 4)),
            measured={
                "32u": (10.7, "blk_l3_b4 #476 (ISL 16: launch-bound)"),
                "256u": (12.0, "proj: cores scale with B"),
            },
            note="at 64 K: 4 MiB of pooled keys per user, ~90 GB/s (doc/mesh/sparse_1m.md): the long-context floor",
        ),
        OpRow(
            "dsa.topk_rows",
            m,
            "topk_rows generic op (512 pools of the scores)",
            dram_bytes=t * pools * 4 * 2,
            measured={"32u": (4.7, "blk_l3_b4 #477"), "256u": (6.0, "proj")},
            bound_hint="launch/gap",
        ),
        OpRow(
            "dsa.pool_gather",
            m,
            f"paged_pool_gather: {sel} selected latent rows x 512 bf16 per user into the compact cache (128 B pieces)",
            kind="state",
            dram_bytes=2 * t * sel * l * 2,
            measured={
                "32u": (136.4, "blk_l3_b4 #478 (gap 14.7); README 61/183/339 us at 1/4/8 users: ~39 us per user"),
                "256u": (22.0 + 39.0 * t, "proj: linear in users (39 us/user + 22)"),
            },
            gap_us=14.7,
            bound_hint="L1/NoC",
            action_id="B8",
            action="(a) dense regime (position <= 2050): attend the paged cache directly, no gather; (b) pool-major "
            "cache pages: one 4 KB run per pool instead of 32 x 128 B pieces",
            risk="M",
            after={"32u": 0.0, "256u": 0.0},
            note="after = B8a at ISL <= 2050; B8b at long context: ~40 us at 4 users, ~250 at 32 (floor x 1.5)",
        ),
        OpRow(
            "dsa.sdpa_layout",
            m,
            "transpose q_lat -> q_dec + transpose o_lat (SDPA head-major layout)",
            kind="layout",
            n_ops=2,
            dram_bytes=4 * tp * hl * l * 2,
            measured={"32u": (8.7, "blk_l3_b4 #479, #481"), "256u": (8.7, "same shape")},
            action_id="B5",
            action="free views with the packed [1,1,B,H*L] layout",
            risk="L",
            after={"32u": 0.0, "256u": 0.0},
        ),
        OpRow(
            "dsa.sdpa",
            m,
            f"paged flash-MLA decode over the compact cache ({sel_pad} rows x 512 bf16 per user; 12 cores at B=4)",
            kind="state",
            dram_bytes=t * sel_pad * l * 2 + 2 * tp * hl * l * 2,
            flops=2 * 2 * t * hl * sel_pad * l,
            cores=min(120, 12 * max(1, t // 4)),
            measured={
                "32u": (72.9, "blk_l3_b4 #480"),
                "256u": (
                    max(72.9, dram_us(t * sel_pad * l * 2, lay.dram_gbs)),
                    "proj: max(73 us, DRAM floor of the B x 2.1 MB read)",
                ),
            },
            action_id="D3",
            action="B <= 8: more cores per head-batch (k_chunk 32 -> 128 over 2080 rows); dense regime: attend the "
            "paged cache (B8a) so no compact copy is read twice",
            risk="L-M",
            after={"32u": 40.0, "256u": max(72.9, dram_us(t * sel_pad * l * 2, lay.dram_gbs))},
        ),
    ]
    if not lay.dsa_state_fused:
        rows += [
            OpRow(
                "dsa.absorb_o",
                m,
                "absorb_o bmm b=16 [32,512]x[512,256] bfp8 (16 cores)",
                dram_bytes=hl * _w(l, dv, "bfp8") + hl * _act(tp, l, dv),
                flops=2 * tp * hl * l * dv,
                cores=16,
                measured={"32u": (19.2, "blk_l3_b4 #482, 150 GB/s"), "256u": (19.2, "same shape")},
                bound_hint="L1/NoC",
                action_id="D4",
                action="keep as a bmm on 64 cores (the o fold W_o2 [8192,4096] costs +18 MB = +45 us of DRAM floor for "
                "the 31 us of ops it removes: roofline-negative at every batch) -> GLM53_DSA_FOLD=q only",
                risk="L",
                after={"32u": 8.0, "256u": 8.0},
            ),
            OpRow(
                "dsa.o_view",
                m,
                "transpose heads + reshape -> [1,1,T,H*Dv] (8 us copy)",
                kind="layout",
                n_ops=2,
                dram_bytes=4 * tp * hl * dv * 2,
                measured={"32u": (11.5, "blk_l3_b4 #483-484"), "256u": (11.5, "same shape")},
                action_id="B5",
                action="free view",
                risk="L",
                after={"32u": 0.0, "256u": 0.0},
            ),
        ]
    rows += [
        OpRow(
            "dsa.o_proj",
            m,
            "o_proj matmul [32,4096]x[4096,4096] bfp8 (64 cores)"
            + (" folded with Wkb2: [8192, 4096]" if lay.dsa_state_fused else ""),
            dram_bytes=_w(hl * (l if lay.dsa_state_fused else dv), h, "bfp8") + _act(tp, hl * dv, h),
            flops=2 * tp * hl * dv * h,
            cores=64,
            measured={
                "32u": (
                    90.0 if lay.dsa_state_fused else 43.7,
                    "blk_l3_b4 #485, 396 GB/s" if not lay.dsa_state_fused else "proj 35.7 MB at 396 GB/s",
                ),
                "256u": (90.0 if lay.dsa_state_fused else 43.7, "same shape"),
            },
        ),
        OpRow(
            "dsa.all_reduce",
            m,
            "TP all-reduce over the 4 columns of [32,4096] bf16",
            kind="ccl",
            n_ops=2,
            ccl=("all_reduce", lay.cols, tp * h * BYTES["bf16"]),
            measured={"32u": (39.5, "blk_l3_b4 #486-487 (RS 27.5 + AG 12.0)"), "256u": (39.5, "same shape")},
            action_id="C2",
            action="as the KDA all-reduce: column skew; keep ccl",
            risk="L",
            after={"32u": 30.0, "256u": 30.0},
        ),
    ]
    return rows


def dsa_rows_dense(cfg: GLM53FlashConfig, lay: Layout) -> list[OpRow]:
    """Dense-range DSA rows (verify / MTP chain): see ``dsa_rows``. The projections are the plain rows at the same rows
    per chip (tile-padded: shape-independent for R <= 32); the index maintenance of the committed rows is charged to
    the commit (``commit_rows``: ``spec.index_commit``), not here."""
    t = lay.rows_per_chip
    l, hl = cfg.kv_lora_rank, cfg.num_attention_heads // lay.cols
    keys_per_row = lay.isl + lay.spec_k + 1
    plain = dsa_rows(cfg, replace(lay, dsa_index=True, spec_k=0, users_per_row=t))
    out = []
    for r in plain:
        if r.key in ("dsa.idx_query", "dsa.idx_view", "dsa.index_score", "dsa.topk_rows", "dsa.pool_gather"):
            continue
        if r.key in ("dsa.state_composite", "dsa.state_fused"):
            out.append(
                OpRow(
                    "dsa.spec_append",
                    r.module,
                    f"per-candidate latent append (paged_update_cache x{t // lay.users_per_row if lay.spec_k else 1}) + the raw k / gate row copies "
                    "for the commit; no ring / pool writes",
                    kind="layout",
                    n_ops=4 + (3 if lay.spec_k else 1),
                    dram_bytes=6 * t * (l + 2 * cfg.index_head_dim) * 2,
                    measured={lay.point: (20.0 + 4.0 * (lay.spec_k + 1), "proj: 5-8 small ops at 3-5 us")},
                    bound_hint="launch/gap",
                )
            )
            continue
        if r.key == "dsa.sdpa":
            nbytes = t * keys_per_row * l * 2 + 2 * 32 * hl * l * 2
            floor = dram_us(nbytes, lay.dram_gbs)
            out.append(
                OpRow(
                    "dsa.sdpa",
                    r.module,
                    f"paged flash-MLA decode: {t} rows over the users' pages ({keys_per_row} keys each at ISL {lay.isl})",
                    kind="state",
                    dram_bytes=nbytes,
                    flops=2 * 2 * t * hl * keys_per_row * l,
                    cores=min(120, 12 * max(1, t // 4)),
                    measured={lay.point: (max(72.9, floor * 3.0), "proj: max(73 us at B=4, 3x the DRAM floor)")},
                    bound_hint="launch/gap" if t * keys_per_row * l * 2 < 4e6 else "L1/NoC",
                )
            )
            continue
        out.append(r)
    return out


def moe_rows(cfg: GLM53FlashConfig, lay: Layout) -> list[OpRow]:
    """Routed + shared MoE sublayer per chip. Hybrid: router, row gathers, the chip's expert slices over the 8T gathered
    tokens, reduce-scatter back, shared expert, TP all-reduce. EP32: router, dispatch glue + all_to_all_dispatch, 9
    full experts, reduce-scatter combine, shared expert, TP all-reduce."""
    tp, t = lay.tp, lay.rows_per_chip
    h, e, k = cfg.hidden_size, cfg.n_routed_experts, cfg.num_experts_per_tok
    i_full = cfg.moe_intermediate_size
    n_tok = lay.rows * t  # gathered / dispatched tokens on the mesh
    tok_pad = lay.rows * tp  # the gathered tensor is [rows x 32, H] whatever T
    act_n, src = active_experts(cfg, lay)
    point = lay.point
    w13, w2 = lay.expert_w13_dtype, lay.expert_w2_dtype
    rows = [
        OpRow(
            "moe.router_gate",
            "MOE_ROUTER",
            "gate matmul [32,4096]x[4096,288] fp32 weights, sigmoid fused (9 cores)",
            dram_bytes=_w(h, e, "fp32") + _act(tp, h, e, "bf16", "fp32"),
            flops=2 * tp * h * e,
            fidelity="HiFi4",
            cores=9,
            measured={"32u": (17.8, "ep_b4 #288"), "256u": (17.8, "ep_b32 #285")},
            action_id="R1",
            action="widen the grid (9 -> 36 cores, per_core_N 1 -> the 1D config's 280 GB/s -> ~12 us)",
            risk="L",
            after={"32u": 13.0, "256u": 13.0},
        ),
        OpRow(
            "moe.router_topk",
            "MOE_ROUTER",
            "router_topk generic op (exact top-8 of 288, w bf16 / idx u32)",
            kind="generic",
            dram_bytes=tp * e * 4 + tp * k * 8,
            cores=32,
            measured={"32u": (12.1, "ep_b4 #289"), "256u": (13.8, "ep_b32 #286")},
            bound_hint="launch/gap",
        ),
    ]
    if lay.moe == "hybrid":
        slice_bytes = h * (i_full // lay.rows) * (2 * BYTES[w13] + BYTES[w2])  # one expert's chip slice
        full_bytes = (e // lay.cols) * slice_bytes
        compact_gather = t * h * 2  # [T, H] bf16 per row (no tile padding)
        if t < TILE:
            gx = {
                "32u": (
                    95.0,
                    "proj: blk_l4_b4 #383-387 all_broadcast 24.9 + untilize/concat/tilize glue 18 on 2 rows -> 8 rows",
                ),
            }
            gr = {"32u": (25.0, "proj: blk_l4_b4 #388-392 routing broadcast 5.8 + glue 22 on 2 rows")}
            glue_ops = 8
        else:
            gx = {"256u": (101.7, "hyb_b32 #279 all_gather_async [32,4096] bf16 over 8 rows")}
            gr = {"256u": (15.3, "hyb_b32 #280 routing [32,288] fp32")}
            glue_ops = 0
        rows += [
            OpRow(
                "moe.gather_x",
                "MOE_HYB_GATHER",
                f"all_gather_rows of x [{tp if t >= TILE else t},4096] bf16 over the 8 rows"
                + (" (T=4: all_broadcast + untilize/concat/tilize)" if t < TILE else ""),
                kind="ccl",
                n_ops=1 + glue_ops // 2,
                ccl=("all_gather", lay.rows, compact_gather),
                measured=gx,
                action_id="H1",
                action="T < 32: one all_gather_async on the compact row-major [T, H] (32 KB per chip), no pads / tilize",
                risk="L-M",
                after={"32u": 35.0, "256u": 101.7},
                note="floor = the compact gather; the padded [32,4096] gather moves 8x the bytes at T=4",
            ),
            OpRow(
                "moe.gather_routing",
                "MOE_HYB_GATHER",
                "all_gather_rows of the dense routing [T,288] fp32 over the 8 rows",
                kind="ccl",
                n_ops=1 + glue_ops // 2,
                ccl=("all_gather", lay.rows, t * e * 4),
                measured=gr,
                action_id="H1",
                action="gather the top-k lists (T x 8 x 8 B) instead of the dense [T,288] fp32 routing",
                risk="L",
                after={"32u": 15.0, "256u": 15.0},
            ),
        ]
        if t < TILE and (t != 4 or lay.spec_k):
            # no measured point (or a speculative layout): the stream op's two-point cost model from the sweep's 28-active
            # (93.5 MB, 341 us) and 69-active (231 MB, 707 us) chip measurements: 92 us fixed + bytes at 376 GB/s
            act_bytes = act_n * slice_bytes
            ex_meas = {
                "32u": (
                    EXPERT_FIXED_US + act_bytes / (EXPERT_GBS * 1e3) + 10.0,
                    f"model: stream op {EXPERT_FIXED_US:.0f} us + {act_bytes / MB:.0f} MB at {EXPERT_GBS:.0f} GB/s "
                    "+ selector matmul 8 + x copy 2 (the 351 us stream row at 28 active)",
                )
            }
            ex_ops = 4
            ex_steps = [
                ("E1", {"32u": 50.0 + act_bytes / 400e3}, "M", "expert_stream fixed cost (~92 -> ~50 us) at 400 GB/s"),
                ("C12", {"32u": 50.0 + act_bytes * (2 * 0.53 + 1) / 3 / 400e3}, "M (accuracy)", "bfp4 gate|up experts"),
            ]
        elif t < TILE:
            if lay.expert_path == "stream":
                ex_meas = {
                    "32u": (
                        351.0,
                        "expert_kernel_sweep.md: stream op 341 us at 28 active (8x4 slice, chip test) + selector matmul 8 + x copy 2",
                    )
                }
                ex_ops = 4
                ex_steps = []
            else:
                ex_meas = {
                    "32u": (
                        683.0,
                        "expert_kernel_sweep.md: sparse tuned block 683 us at 28 active, 13 ops (the no-knob path; e2e 83.1 ms = 68.7 model + 42 x 0.33)",
                    )
                }
                ex_ops = 13
                ex_steps = [
                    (
                        "E0",
                        {"32u": 351.0},
                        "L",
                        "GLM53_STREAM_EXPERTS=1: the expert_stream op (341 vs 683 us at 28 active; block test PCC 0.99997); confirm with a layer-4 tracy of the 8x4 hybrid at 4 users/row",
                    )
                ]
            ex_steps += [
                (
                    "E1",
                    {"32u": 250.0},
                    "M",
                    "expert_stream fixed cost (~110 us: launch x2, x copy, routing scans, DST fill): fold the selector matmul + x copy into the op, pipeline the DST fill",
                ),
                (
                    "C12",
                    {"32u": 180.0},
                    "M (accuracy)",
                    "bfp4 gate|up experts (GLM53_EXPERT_W13_DTYPE=bfp4): -1/3 of the expert bytes, eval-gated",
                ),
            ]
        else:
            ex_meas = {
                "256u": (
                    1198.0,
                    "hyb_b32 #281-286 dense layout: 3 matmuls 89 + 317 + 307 + 329 + 2 binary 91 + 65 (all 72 experts, 252 GB/s, 122 TFLOPs)",
                ),
            }
            ex_ops = 6
            ex_steps = [
                (
                    "E1",
                    {"256u": 779.0},
                    "M",
                    "stream dispatch form (expert_kernel_sweep.md section 10: 779 us at 58 active, weights read once, no dense compute of the non-routed pairs); the down op's expert-parallel split next (~-400 us)",
                ),
                ("C12", {"256u": 600.0}, "M (accuracy)", "bfp4 gate|up experts: -1/3 of the expert bytes, eval-gated"),
            ]
        routed_pairs = n_tok * k / lay.cols  # (token, expert) pairs on this column
        dense_flops = 2 * n_tok * (e // lay.cols) * 3 * h * (i_full // lay.rows)
        routed_flops = 2 * routed_pairs * 3 * h * (i_full // lay.rows)
        rows.append(
            OpRow(
                "moe.experts",
                "MOE_HYB_EXPERTS",
                f"routed experts: {act_n:.0f} of {e // lay.cols} active slices x {slice_bytes / MB:.2f} MB ({w13} gate|up, {w2} down) "
                f"for {n_tok} gathered tokens" + (" [dense layout reads all 72]" if t >= TILE else ""),
                kind="generic",
                n_ops=ex_ops,
                dram_bytes=act_n * slice_bytes + 2 * tok_pad * h * 2,
                flops=routed_flops,
                cores=96,
                measured=ex_meas,
                steps=ex_steps,
                note=f"active experts: {src}; routed FLOPs {routed_flops / 1e9:.1f} GFLOP; the dense layout reads all "
                f"{full_bytes / MB:.0f} MB and computes {dense_flops / 1e9:.0f} GFLOP (358 us at the HiFi2 peak)",
            )
        )
        rs_in = tok_pad * h * 2  # [8 x 32, H] bf16 = 2 MB per chip whatever T
        rows.append(
            OpRow(
                "moe.rs_rows",
                "MOE_HYB_RS",
                f"reduce_scatter_rows of the expert partials [{tok_pad},4096] bf16 over the 8 rows"
                + (" (T=4: 8 pads/slices + RS composite)" if t < TILE else ""),
                kind="ccl",
                n_ops=1 + (9 if t < TILE else 0),
                ccl=("reduce_scatter", lay.rows, rs_in if t >= TILE else lay.rows * t * h * 2),
                measured={
                    "32u": (
                        162.0,
                        "proj: hyb_b32 #287 RS 128.5 on the same [256,4096] shape + blk_l4_b4 #397-406 glue 34",
                    ),
                    "256u": (128.5, "hyb_b32 #287 ReduceScatterMinimalAsync"),
                },
                action_id="H2",
                action="T < 32: write the expert output compact ([rows x T, H] = 256 KB) and all_reduce_rows it (every "
                "row slices its T rows): 8x fewer bytes than the padded reduce-scatter, no pads / slices",
                risk="M",
                after={"32u": 50.0, "256u": 128.5},
                note="the measured RS runs at ~2x the CCL fit (128 vs 66 us): num_links / chunks_per_sync sweep",
            )
        )
    else:  # EP32
        full_expert = h * i_full * (2 * BYTES[w13] + BYTES[w2])  # 26.7 MB bfp8
        disp_in = t * k * (h * 2 + 2 * k)  # [T x k] rows of x (bf16) + routing weight tail, per chip
        rows += [
            OpRow(
                "ep.dispatch_glue",
                "MOE_EP_DISPATCH",
                "untilize x, untilize w, concat [x|w], typecast idx u32->u16, untilize idx (row-major L1 inputs)",
                kind="layout",
                n_ops=5,
                dram_bytes=4 * tp * h * 2,
                measured={"32u": (10.1, "ep_b4 #290-294"), "256u": (23.9, "ep_b32 #287-291")},
                action_id="C10",
                action="router_topk writes w bf16 / idx u16 ROW_MAJOR directly (-3 ops)",
                risk="L",
                after={"32u": 4.0, "256u": 10.0},
            ),
            OpRow(
                "ep.a2a_dispatch",
                "MOE_EP_DISPATCH",
                f"all_to_all_dispatch along the 8 rows: {t} tokens x 8 experts x 8.2 KB per chip",
                kind="ccl",
                ccl=("all_to_all", lay.rows, disp_in),
                measured={"32u": (67.2, "ep_b4 #295"), "256u": (93.2, "ep_b32 #292")},
            ),
            OpRow(
                "ep.experts",
                "MOE_EP_EXPERTS",
                f"expert_stream dispatch ops: {act_n:.1f} of {e // (lay.rows * lay.cols)} full experts x {full_expert / MB:.1f} MB on the slowest chip of the column",
                kind="generic",
                n_ops=2,
                dram_bytes=act_n * full_expert + 2 * t * k * h * 2,
                flops=2 * t * k * 3 * h * i_full,
                cores=96,
                measured={
                    "32u": (
                        218.0,
                        "ep_b4 #296-297 chip 0 (109 + 109); per-chip spread 33-502 us, absorbed by the combine row",
                    ),
                    "256u": (441.4, "ep_b32 #293-294 chip 0 (224 + 217); spread 112-661"),
                },
                action_id="E2",
                action="placement / replication of hot experts across the column's chips (the skew the combine waits "
                "for IS the cost); bfp4 gate|up",
                risk="M-H",
                after={"32u": 218.0, "256u": 441.4},
                note=f"active experts: {src}; the floor is the slowest chip's read, the measured is chip 0's",
            ),
            OpRow(
                "ep.combine_rs",
                "MOE_EP_COMBINE",
                "reduce_scatter_rows of the zero-padded expert output [256,4096] bf16 (C1 rs combine, default at T % 32 == 0; "
                "T=4 needs GLM53_EP_RS_REMAP=1)",
                kind="ccl",
                ccl=("reduce_scatter", lay.rows, tok_pad * h * 2),
                measured={
                    "32u": (
                        459.0,
                        "ep_b4 #298-303 op path: combine 418.7 (incl. the senders' skew) + untilize/clone/tilize/fast_reduce/slice 40; the e2e op-combine step 71.2 ms agrees",
                    ),
                    "256u": (
                        990.0,
                        "e2e-derived: EP32 rs 123.4 ms vs hybrid dense 115.0 -> EP MoE sublayer ~1.64 ms/layer, minus the chip-0 rows 0.65 = ~1.0 ms for the combine stage incl. the column's expert skew (661 vs 441 us) and the RS",
                    ),
                },
                steps=[
                    (
                        "C1",
                        {"32u": 300.0},
                        "L",
                        "GLM53_EP_RS_REMAP=1 for T < 32 (validated 2026-10-26: 67.68 vs 71.2 ms = -84 us/layer); the RS itself is ~130 us, the rest is the senders' skew",
                    ),
                    (
                        "E2",
                        {"32u": 160.0, "256u": 300.0},
                        "M-H",
                        "hot-expert placement / replication across the column's chips so the slowest sender is near the mean (the RS floor is 68 us)",
                    ),
                ],
            ),
        ]
    rows += [
        OpRow(
            "moe.shared_w13",
            "MOE_SHARED",
            "shared expert gate|up matmul [32,4096]x[4096,1024] bfp8 (TP4 slice, 16 cores)",
            dram_bytes=_w(h, 2 * i_full // lay.cols, "bfp8") + _act(tp, h, 2 * i_full // lay.cols),
            flops=2 * tp * h * 2 * i_full // lay.cols,
            cores=16,
            measured={"32u": (16.9, "ep_b4 #304"), "256u": (16.5, "ep_b32 #300")},
            action_id="S1",
            action="fold the 2 slices + clamped_silu_glu into the matmul epilogue (gate/up column-interleaved blocks)",
            risk="L",
            after={"32u": 14.0, "256u": 14.0},
        ),
        OpRow(
            "moe.shared_glue",
            "MOE_SHARED",
            "slice gate, slice up, clamped_silu_glu",
            kind="layout",
            n_ops=3,
            dram_bytes=6 * tp * (i_full // lay.cols) * 2,
            measured={"32u": (8.0, "ep_b4 #305-307"), "256u": (7.8, "ep_b32 #301-303")},
            action_id="S1",
            action="fused into the w13 matmul",
            risk="L",
            after={"32u": 0.0, "256u": 0.0},
        ),
        OpRow(
            "moe.shared_w2",
            "MOE_SHARED",
            "shared expert down matmul [32,512]x[512,4096] bfp8 (22 cores)",
            dram_bytes=_w(i_full // lay.cols, h, "bfp8") + _act(tp, i_full // lay.cols, h),
            flops=2 * tp * (i_full // lay.cols) * h,
            cores=22,
            measured={"32u": (9.1, "ep_b4 #308"), "256u": (9.3, "ep_b32 #304")},
        ),
        OpRow(
            "moe.add",
            "MOE",
            "add routed + shared partials",
            dram_bytes=3 * tp * h * 2,
            measured={"32u": (3.6, "ep_b4 #309"), "256u": (3.5, "ep_b32 #305")},
            bound_hint="launch/gap",
            action_id="S1",
            action="fold into the shared w2 matmul (fused add) or the all-reduce input",
            risk="L",
            after={"32u": 0.0, "256u": 0.0},
        ),
        OpRow(
            "moe.all_reduce",
            "MOE",
            "TP all-reduce over the 4 columns of [32,4096] bf16",
            kind="ccl",
            n_ops=2,
            ccl=("all_reduce", lay.cols, tp * h * BYTES["bf16"]),
            measured={
                "32u": (29.7, "ep_b4 #310-311 (RS 11.7 + AG 18.0)"),
                "256u": (28.7, "ep_b32 #306-307; hyb_b32 27.4"),
            },
        ),
    ]
    return rows


def dense_mlp_rows(cfg: GLM53FlashConfig, lay: Layout) -> list[OpRow]:
    """Layers 0-2: dense MLP (intermediate 12288, TP4 -> 3072 per chip). No decode profile of a dense layer exists; the
    measured entries are the matmul sweep's per-shape times (tests/test_matmul_sweep.py, 2026-10-20)."""
    tp = lay.tp
    h, i_loc = cfg.hidden_size, cfg.intermediate_size // lay.cols
    return [
        OpRow(
            "mlp.w13",
            "MLP_DENSE",
            "dense w13 matmul [32,4096]x[4096,6144] bfp8",
            dram_bytes=_w(h, 2 * i_loc, "bfp8") + _act(tp, h, 2 * i_loc),
            flops=2 * tp * h * 2 * i_loc,
            cores=96,
            measured={"32u": (76.1, "matmul sweep 4096x6144 (1D config)"), "256u": (76.1, "same")},
            action_id="S1",
            action="the kda qkv runs the same shape at 65 us: apply its config",
            risk="L",
            after={"32u": 66.0, "256u": 66.0},
        ),
        OpRow(
            "mlp.glue",
            "MLP_DENSE",
            "slice gate, slice up, clamped_silu_glu",
            kind="layout",
            n_ops=3,
            dram_bytes=6 * tp * i_loc * 2,
            measured={"32u": (12.0, "proj (shared expert glue x 3 in width)"), "256u": (12.0, "proj")},
            action_id="S1",
            action="fused into the w13 matmul",
            risk="L",
            after={"32u": 0.0, "256u": 0.0},
        ),
        OpRow(
            "mlp.w2",
            "MLP_DENSE",
            "dense w2 matmul [32,3072]x[3072,4096] bfp8",
            dram_bytes=_w(i_loc, h, "bfp8") + _act(tp, i_loc, h),
            flops=2 * tp * i_loc * h,
            cores=24,
            measured={"32u": (43.9, "matmul sweep 3072x4096"), "256u": (43.9, "same")},
        ),
        OpRow(
            "mlp.all_reduce",
            "MLP_DENSE",
            "TP all-reduce over the 4 columns of [32,4096] bf16",
            kind="ccl",
            n_ops=2,
            ccl=("all_reduce", lay.cols, tp * h * BYTES["bf16"]),
            measured={"32u": (29.7, "as the MoE all-reduce"), "256u": (28.7, "same")},
        ),
    ]


def head_rows(cfg: GLM53FlashConfig, lay: Layout) -> list[OpRow]:
    """Once per step: embedding gather, final collapse / norm, the vocab-split LM head and its column gather, argmax."""
    tp = lay.tp
    h, v = cfg.hidden_size, cfg.vocab_size
    v_loc = v // lay.cols  # 38720
    return [
        OpRow(
            "embed.gather",
            "EMBED",
            "embedding gather [T] -> [1,T,4096] bf16 widened to the fp32 streams",
            dram_bytes=tp * h * (2 + 4 * cfg.hc_mult),
            measured={"32u": (20.0, "proj"), "256u": (20.0, "proj")},
            bound_hint="launch/gap",
        ),
        OpRow(
            "head.norm_cast",
            "HEAD",
            "stream collapse (folded into the last post-block), final rms_norm, cast to bf16",
            n_ops=2,
            dram_bytes=tp * h * (4 + 2 + 2),
            measured={"32u": (15.0, "proj"), "256u": (15.0, "proj")},
            bound_hint="launch/gap",
        ),
        OpRow(
            "head.matmul",
            "HEAD",
            "LM head matmul [32,4096]x[4096,38720] bfp8 (the column's vocab block)",
            dram_bytes=_w(h, v_loc, "bfp8") + _act(tp, h, v_loc),
            flops=2 * tp * h * v_loc,
            cores=120,
            measured={"32u": (432.0, "proj: 168.5 MB at 390 GB/s"), "256u": (432.0, "proj")},
        ),
        OpRow(
            "head.all_gather_cols",
            "HEAD",
            "all_gather_cols of the logits [32,38720] bf16 over the 4 columns",
            kind="ccl",
            ccl=("all_gather", lay.cols, tp * v_loc * 2),
            measured={"32u": (170.0, "proj: CCL fit"), "256u": (170.0, "proj")},
            action_id="HD1",
            action="argmax per column on the local block + all_gather of 32 (value, id) pairs (-7.4 MB on the link)",
            risk="L",
            after={"32u": 15.0, "256u": 15.0},
        ),
        OpRow(
            "head.argmax",
            "HEAD",
            "argmax over the vocab",
            dram_bytes=tp * v * 2,
            measured={"32u": (15.0, "proj"), "256u": (15.0, "proj")},
            bound_hint="launch/gap",
        ),
    ]


def gaps_row(layer: str, n_ops: int, lay: Layout) -> OpRow:
    meas = {
        "KDA+MoE": (
            85.0 if lay.moe == "hybrid" else 64.0,
            "ledger section 3: 64 us (EP32) / 85 us (hybrid block) of in-trace gaps",
        ),
        "DSA+MoE": (90.0, "ledger section 3: ~90 us (block, hybrid)"),
        "KDA+dense": (40.0, "proj: 0.46 us x ops + the 34 us kda_decode gap"),
        "head": (5.0, "proj"),
    }[layer]
    return OpRow(
        "layer.gaps",
        "TRACE",
        f"in-trace op-to-op gaps ({n_ops} ops; the 34 us before kda_decode and 14.7 us before paged_pool_gather counted on those ops)",
        kind="gaps",
        n_ops=n_ops,
        measured={"32u": meas, "256u": meas},
        action_id="A1",
        action="fewer ops: every erased op saves its gap (0.46 us median, 1-1.5 us after single-core ops)",
        risk="-",
    )


LAYER_COUNTS = {"KDA+MoE": KDA_LAYERS - DENSE_LAYERS, "DSA+MoE": DSA_LAYERS, "KDA+dense": DENSE_LAYERS, "head": 1}


def layer_rows(cfg: GLM53FlashConfig, lay: Layout, layer: str) -> list[OpRow]:
    if layer == "KDA+MoE":
        rows = (
            mhc_wrap_rows(cfg, lay, "PRE_ATTN")
            + kda_rows(cfg, lay)
            + mhc_wrap_rows(cfg, lay, "PRE_FFN")
            + moe_rows(cfg, lay)
        )
    elif layer == "DSA+MoE":
        rows = (
            mhc_wrap_rows(cfg, lay, "PRE_ATTN")
            + dsa_rows(cfg, lay)
            + mhc_wrap_rows(cfg, lay, "PRE_FFN")
            + moe_rows(cfg, lay)
        )
    elif layer == "KDA+dense":
        rows = (
            mhc_wrap_rows(cfg, lay, "PRE_ATTN")
            + kda_rows(cfg, lay)
            + mhc_wrap_rows(cfg, lay, "PRE_FFN")
            + dense_mlp_rows(cfg, lay)
        )
    elif layer == "head":
        rows = head_rows(cfg, lay)
    else:
        raise ValueError(layer)
    n_ops = sum(r.n_ops for r in rows if r.kind != "gaps")
    rows.append(gaps_row(layer, n_ops, lay))
    return [evaluate(r, lay) for r in rows]


# ----------------------------------------------------------------------------------------------- evaluation
def evaluate(r: OpRow, lay: Layout) -> OpRow:
    """Fill the floors, the measured value for the layout's point, the ratio and the bound classification."""
    floors = {}
    if r.kind == "gaps":
        floors["gap"] = chip("op_gap_us") * max(1, r.n_ops)
    else:
        floors["dram"] = dram_us(r.dram_bytes, lay.dram_gbs)
        floors["dram_spec"] = dram_us(r.dram_bytes, chip("dram_gbs_spec"))
        floors["compute"] = compute_us(r.flops, r.fidelity) if r.flops else 0.0
        if r.ccl is not None:
            coll, n, bytes_in = r.ccl
            floors["ccl"] = ccl_us(coll, n, bytes_in, links=lay.links, link_gbs=lay.link_gbs, topology=lay.topology)
            floors["ccl_rated"] = ccl_us(
                coll, n, bytes_in, links=lay.links, link_gbs=chip("eth_link_gbs_rated"), topology=lay.topology
            )
            floors["ccl_link_bytes"] = _ccl_bytes(coll, n, bytes_in)
        floors["launch"] = chip("op_floor_us") * r.n_ops
    r.floors = floors
    r.floor_us = max(v for k, v in floors.items() if k in ("dram", "compute", "ccl", "launch", "gap"))
    r.floor_spec_us = (
        max(v for k, v in floors.items() if k in ("dram_spec", "compute", "ccl_rated", "launch", "gap"))
        if r.kind != "gaps"
        else r.floor_us
    )
    meas = r.measured.get(lay.point)
    if meas is None and r.measured:  # a single-point measurement reused (shape-independent ops)
        meas = next(iter(r.measured.values()))
    if meas is not None:
        r.measured_us, r.measured_src = float(meas[0]), meas[1]
        r.ratio = (r.measured_us / r.floor_us) if r.floor_us > 0 else None
    if not r.steps and r.action_id != "-":
        r.steps = [(r.action_id, dict(r.after), r.risk, r.action)]
    if r.steps:
        last = r.steps[-1]
        r.action_id, r.risk, r.action = last[0], last[2], last[3]
        r.after_us = r.measured_us
        for _aid, after, _risk, _text in r.steps:
            r.after_us = after.get(lay.point, r.after_us)
    else:
        r.after_us = r.measured_us
    r.bound = classify(r)
    return r


def classify(r: OpRow) -> str:
    if r.kind == "gaps":
        return "launch/gap"
    if r.kind == "layout":
        return "data-movement"
    if r.kind == "ccl":
        return "ethernet/CCL"
    if r.bound_hint:
        return r.bound_hint
    f = r.floors
    if r.measured_us is None:
        return "DRAM" if f["dram"] >= f["compute"] else "compute"
    if f["compute"] > f["dram"] and f["compute"] >= 0.5 * r.measured_us:
        return "compute"
    if r.floor_us < 5.0 and r.measured_us < 20.0:
        return "launch/gap"
    if r.ratio is not None and r.ratio <= 1.6:
        return "DRAM"
    if r.cores <= 32:
        return "L1/NoC"
    return "DRAM (under-utilized)"


# ----------------------------------------------------------------------------------------------- tables
def layer_table(cfg: GLM53FlashConfig, lay: Layout, layer: str) -> list[dict]:
    out = []
    for r in layer_rows(cfg, lay, layer):
        out.append(
            {
                "layer": layer,
                "module": r.module,
                "key": r.key,
                "op": r.op,
                "kind": r.kind,
                "n_ops": r.n_ops,
                "MB": round(r.dram_bytes / MB, 3),
                "GFLOP": round(r.flops / 1e9, 4),
                "fidelity": r.fidelity if r.flops else "",
                "ccl_link_MB": round(r.floors.get("ccl_link_bytes", 0.0) / MB, 3),
                "floor_us": round(r.floor_us, 1),
                "floor_spec_us": round(r.floor_spec_us, 1),
                "floor_dram_us": round(r.floors.get("dram", 0.0), 1),
                "floor_compute_us": round(r.floors.get("compute", 0.0), 2),
                "floor_ccl_us": round(r.floors.get("ccl", 0.0), 1),
                "measured_us": r.measured_us,
                "measured_src": r.measured_src,
                "gap_us": r.gap_us,
                "ratio": round(r.ratio, 2) if r.ratio is not None else None,
                "bound": r.bound,
                "action_id": r.action_id,
                "action": r.action,
                "risk": r.risk,
                "after_us": r.after_us,
                "note": r.note,
            }
        )
    return out


def layer_summary(cfg: GLM53FlashConfig, lay: Layout, layer: str) -> dict:
    rows = layer_rows(cfg, lay, layer)
    floor = sum(r.floor_us for r in rows)
    floor_spec = sum(r.floor_spec_us for r in rows)
    meas = sum(r.measured_us or 0.0 for r in rows)
    after = sum((r.after_us if r.after_us is not None else (r.measured_us or 0.0)) for r in rows)
    by_module: dict[str, dict] = {}
    for r in rows:
        d = by_module.setdefault(r.module, {"floor_us": 0.0, "measured_us": 0.0, "after_us": 0.0, "ops": 0})
        d["floor_us"] += r.floor_us
        d["measured_us"] += r.measured_us or 0.0
        d["after_us"] += r.after_us if r.after_us is not None else (r.measured_us or 0.0)
        d["ops"] += r.n_ops if r.kind != "gaps" else 0
    return {
        "layer": layer,
        "count": LAYER_COUNTS[layer],
        "ops": sum(r.n_ops for r in rows if r.kind != "gaps"),
        "floor_us": floor,
        "floor_spec_us": floor_spec,
        "measured_us": meas,
        "after_us": after,
        "ratio": meas / floor if floor else None,
        "modules": {
            k: {kk: round(vv, 1) if isinstance(vv, float) else vv for kk, vv in v.items()} for k, v in by_module.items()
        },
    }


# e2e traced decode steps (ms) to compare the step model with (models/demos/glm53_flash/doc/mesh/README.md)
MEASURED_STEPS_MS = {
    "32u": {
        "hybrid 2026-10-26 (A3 sharded norm, B6 ranges, distinct ids)": 80.55,
        "hybrid 2026-10-21 (distinct ids)": 83.1,
        "hybrid 2026-10-21 (same token on every slot: 2 active experts)": 69.8,
        "EP32 op combine 2026-10-21 (distinct ids)": 71.2,
        "EP32 rs-remap 2026-10-26 (distinct ids)": 67.68,
    },
    "256u": {
        "hybrid dense 2026-10-21 (distinct ids)": 115.0,
        "EP32 rs combine 2026-10-22 (distinct ids)": 123.4,
        "EP32 op combine 2026-10-21 (distinct ids)": 139.2,
    },
}

TARGETS_MS = {"32u": 1000 / 33.0, "256u": 1000 / 19.0}  # 33 / 19 tok/s per user


def step_table(cfg: GLM53FlashConfig, lay: Layout) -> dict:
    layers = {name: layer_summary(cfg, lay, name) for name in LAYER_COUNTS}
    floor_ms = sum(s["floor_us"] * s["count"] for s in layers.values()) / 1e3
    floor_spec_ms = sum(s["floor_spec_us"] * s["count"] for s in layers.values()) / 1e3
    meas_ms = sum(s["measured_us"] * s["count"] for s in layers.values()) / 1e3
    after_ms = sum(s["after_us"] * s["count"] for s in layers.values()) / 1e3
    return {
        "layout": lay.__dict__ | {"users": lay.users, "point": lay.point},
        "layers": layers,
        "step_floor_ms": floor_ms,
        "step_floor_spec_ms": floor_spec_ms,
        "step_model_measured_ms": meas_ms,
        "step_after_ms": after_ms,
        "step_measured_e2e_ms": MEASURED_STEPS_MS.get(lay.point, {}),
        "target_ms": TARGETS_MS.get(lay.point),
        "tok_s_per_user_ceiling": 1000.0 / floor_ms,
        "tok_s_per_user_ceiling_spec": 1000.0 / floor_spec_ms,
        "tok_s_aggregate_ceiling": lay.users * 1000.0 / floor_ms,
        "tok_s_per_user_after": 1000.0 / after_ms,
    }


def ranked_actions(cfg: GLM53FlashConfig, lay: Layout) -> list[dict]:
    """Actions ranked by ms saved per step (measured - expected after, x layer count), merged by action id."""
    acc: dict[str, dict] = {}
    for layer, count in LAYER_COUNTS.items():
        for r in layer_rows(cfg, lay, layer):
            if not r.steps or r.measured_us is None:
                continue
            prev = r.measured_us
            for aid, after, risk, text in r.steps:
                cur = after.get(lay.point, prev)
                d = acc.setdefault(aid, {"action_id": aid, "ms": 0.0, "ops": set(), "risk": risk, "action": text})
                d["ms"] += (prev - cur) * count / 1e3
                d["ops"].add(r.key)
                prev = cur
    out = sorted(acc.values(), key=lambda d: -d["ms"])
    for d in out:
        d["ops"] = sorted(d["ops"])
        d["ms"] = round(d["ms"], 2)
    return out


# ----------------------------------------------------------------------------------------------- printing
def _fmt(v, nd=1, markdown=False):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v).replace("|", "\\|") if markdown else str(v)


def format_layer_table(rows: list[dict], markdown: bool = False) -> str:
    cols = [
        "key",
        "n_ops",
        "MB",
        "GFLOP",
        "floor_us",
        "floor_spec_us",
        "measured_us",
        "ratio",
        "bound",
        "action_id",
        "after_us",
        "measured_src",
    ]
    lines = []
    if markdown:
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("|" + "---|" * len(cols))
        for r in rows:
            lines.append("| " + " | ".join(_fmt(r[c], 3 if c in ("MB", "GFLOP") else 1, True) for c in cols) + " |")
    else:
        w = {c: max(len(c), *(len(_fmt(r[c], 3 if c in ("MB", "GFLOP") else 1)) for r in rows)) for c in cols}
        lines.append("  ".join(c.ljust(w[c]) for c in cols))
        for r in rows:
            lines.append("  ".join(_fmt(r[c], 3 if c in ("MB", "GFLOP") else 1).ljust(w[c]) for c in cols))
    return "\n".join(lines)


def format_step(step: dict, markdown: bool = False) -> str:
    lay = step["layout"]
    lines = [
        f"layout: {lay['rows']}x{lay['cols']}, {lay['users_per_row']} users/row = {lay['users']} users, moe={lay['moe']} "
        f"({lay['expert_path']}), experts {lay['expert_w13_dtype']}/{lay['expert_w2_dtype']}, ISL {lay['isl']}, "
        f"DRAM {lay['dram_gbs']:.0f} GB/s (spec {chip('dram_gbs_spec'):.0f}), links {lay['links']} x {lay['link_gbs']:.0f} GB/s ({lay['topology']})",
        "",
    ]
    hdr = ["layer", "count", "ops", "floor_us", "floor_spec_us", "model_measured_us", "ratio", "after_us"]
    body = [
        [
            name,
            s["count"],
            s["ops"],
            round(s["floor_us"], 1),
            round(s["floor_spec_us"], 1),
            round(s["measured_us"], 1),
            round(s["ratio"], 2) if s["ratio"] else None,
            round(s["after_us"], 1),
        ]
        for name, s in step["layers"].items()
    ]
    if markdown:
        lines.append("| " + " | ".join(hdr) + " |")
        lines.append("|" + "---|" * len(hdr))
        lines += ["| " + " | ".join(_fmt(v, 1, True) for v in row) + " |" for row in body]
    else:
        lines.append("  ".join(h.ljust(18) for h in hdr))
        lines += ["  ".join(_fmt(v).ljust(18) for v in row) for row in body]
    lines += [
        "",
        f"step floor: {step['step_floor_ms']:.2f} ms at {lay['dram_gbs']:.0f} GB/s ({step['step_floor_spec_ms']:.2f} ms at the spec peaks) "
        f"-> ceiling {step['tok_s_per_user_ceiling']:.1f} tok/s per user ({step['tok_s_per_user_ceiling_spec']:.1f} at spec), "
        f"{step['tok_s_aggregate_ceiling']:.0f} aggregate; target {step['target_ms']:.1f} ms",
        f"step model (sum of the measured / projected ops): {step['step_model_measured_ms']:.2f} ms; after the listed actions: "
        f"{step['step_after_ms']:.2f} ms = {step['tok_s_per_user_after']:.1f} tok/s per user",
        "e2e measured steps: " + ", ".join(f"{k}: {v} ms" for k, v in step["step_measured_e2e_ms"].items()),
    ]
    return "\n".join(lines)


def format_actions(actions: list[dict], markdown: bool = False, top: int = 12) -> str:
    hdr = ["rank", "action", "ms/step", "risk", "ops", "what"]
    rows = [
        [i + 1, a["action_id"], a["ms"], a["risk"], ", ".join(a["ops"]), a["action"]]
        for i, a in enumerate(actions[:top])
    ]
    if markdown:
        return "\n".join(
            ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
            + ["| " + " | ".join(_fmt(v, 2, True) for v in r) + " |" for r in rows]
        )
    return "\n".join("  ".join(_fmt(v, 2) for v in r) for r in [hdr] + rows)


# ----------------------------------------------------------------------------------------------- speculative decode
MTP_GLUE_OPS = 5  # embedding gather, enorm, hnorm, typecast, concat


def mtp_block_rows(cfg: GLM53FlashConfig, lay: Layout, rows: int) -> list[OpRow]:
    """One pass of the MTP block (checkpoint layer 45, ``tt/mtp.py``) on ``rows`` rows per chip: glue (enorm / hnorm /
    eh_proj [8192, 4096] bfp8, replicated), its own DSA (dense range, own latent cache) and 288-expert MoE, the
    shared_head norm and the shared LM head (logits + greedy). No mHC wrap (plain residuals)."""
    sub = replace(lay, users_per_row=rows, spec_k=0, dsa_index=False)
    h, tp = cfg.hidden_size, lay.tp
    glue = [
        OpRow(
            "mtp.glue",
            "MTP_GLUE",
            "embedding gather of the next token, enorm, hnorm, typecast, concat [e | h]",
            n_ops=MTP_GLUE_OPS,
            dram_bytes=tp * h * (2 + 2 + 4 + 2 + 4 + 4),
            measured={lay.point: (18.0, "proj: 5 small ops")},
            bound_hint="launch/gap",
        ),
        OpRow(
            "mtp.eh_proj",
            "MTP_GLUE",
            "eh_proj matmul [32,8192]x[8192,4096] bfp8 (replicated, no collective)",
            dram_bytes=_w(2 * h, h, "bfp8") + _act(tp, 2 * h, h),
            flops=2 * tp * 2 * h * h,
            cores=96,
            measured={lay.point: (95.0, "proj: 35.7 MB at 390 GB/s")},
        ),
    ]
    tail = [
        OpRow(
            "mtp.shared_norm",
            "MTP_HEAD",
            "shared_head.norm (rms_norm, bf16)",
            dram_bytes=3 * tp * h * 2,
            measured={lay.point: (7.0, "proj: sharded rms_norm")},
            bound_hint="launch/gap",
        )
    ]
    head = [r for r in head_rows(cfg, sub) if r.key != "embed.gather"]
    rows_ = glue + dsa_rows(cfg, sub) + moe_rows(cfg, sub) + tail + head
    for r in rows_:
        r.module = "MTP_" + r.module if not r.module.startswith("MTP_") else r.module
    return [evaluate(r, sub) for r in rows_]


def commit_rows(cfg: GLM53FlashConfig, lay: Layout) -> list[OpRow]:
    """The commit graph C per iteration (``KdaSpecState.commit`` x 34 KDA layers, ``DsaSpecState.commit`` x 11 DSA
    layers), as rows already multiplied by their layer counts."""
    b, k = lay.users_per_row, lay.spec_k
    hl, kd = cfg.kda_num_heads // lay.cols, cfg.kda_head_dim
    state_bytes = hl * kd * kd * BYTES["fp32"]
    conv_bytes = (cfg.kda_conv_kernel - 1) * 3 * hl * kd * BYTES["bf16"]
    n_kda, n_dsa = KDA_LAYERS, DSA_LAYERS
    launches = n_kda * (k + 1)
    blend_bytes = 7 * b * state_bytes  # in-place multiply (r+w), multiply (r+w), in-place add (2r+w) on the fp32 state
    maint = 100.3 + 12.0 * (b / 4 - 1) if not lay.dsa_state_fused else 40.0 + 10.0 * (b / 4 - 1)
    rows = [
        OpRow(
            "spec.kda_replay",
            "COMMIT_KDA",
            f"{k + 1} kda_decode launches per KDA layer on the backup from the cached projections ({launches} launches)",
            kind="state",
            n_ops=launches,
            dram_bytes=launches * 2 * b * (state_bytes + conv_bytes),
            flops=launches * b * hl * (6 * kd * kd + 10 * kd),
            fidelity="HiFi4",
            cores=80,
            measured={lay.point: (launches * (48.6 + 6.6 * b), f"fit per launch: {launches} x (48.6 + 6.6 x {b})")},
            gap_us=34.2 * launches,
            steps=[
                (
                    "K3",
                    {lay.point: launches * (14.0 + 6.6 * b)},
                    "M",
                    "kda_decode fixed cost + pre-op gap (paid per launch)",
                ),
                (
                    "SP1",
                    {lay.point: 0.0},
                    "M",
                    "no replay: the verify's launches write the selected user's state in place (per-user destination select in the writer)",
                ),
            ],
        ),
        OpRow(
            "spec.kda_blend",
            "COMMIT_KDA",
            f"per launch: live = live * keep_j + backup * take_j on the fp32 state (3 full-state elementwise ops, {7 * b:.0f} MB) + 3 conv blends",
            n_ops=6 * launches,
            dram_bytes=launches * (blend_bytes + 7 * b * conv_bytes),
            measured={
                lay.point: (
                    launches * (blend_bytes / 175e3 + 3 * 1.5),
                    "proj: fp32 elementwise at 175 GB/s (the mHC x*x rate) + 3 tiny ops",
                )
            },
            bound_hint="DRAM (under-utilized)",
            steps=[
                (
                    "SP2",
                    {lay.point: launches * (2 * b * state_bytes / 300e3 + 1.5)},
                    "L",
                    "one masked copy per launch (backup -> live for take_j users) instead of 3 full-state ops: 7 -> 2 x B MB",
                ),
                ("SP1", {lay.point: 0.0}, "M", "blend inside kda_decode's writer"),
            ],
        ),
        OpRow(
            "spec.index_commit",
            "COMMIT_DSA",
            f"{k + 1} x the pooled-key index maintenance per DSA layer (ring row + pool rewrite per candidate, rejected rows to dead ring rows)",
            kind="layout",
            n_ops=n_dsa * (k + 1) * (1 if lay.dsa_state_fused else 29),
            dram_bytes=n_dsa * (k + 1) * b * (cfg.kv_lora_rank + 6 * cfg.index_head_dim) * 2 * 4,
            measured={
                lay.point: (
                    n_dsa * (k + 1) * maint,
                    f"{n_dsa} x {k + 1} x the state maintenance ({maint:.0f} us: blk_l3_b4 composite at B=4"
                    + (", B5 program)" if lay.dsa_state_fused else ")"),
                )
            },
            steps=[]
            if lay.dsa_state_fused
            else [
                (
                    "B5",
                    {lay.point: n_dsa * (k + 1) * (40.0 + 10.0 * (b / 4 - 1))},
                    "M",
                    "dsa_decode_state program for the commit rows (29 -> 1 op per candidate)",
                )
            ],
        ),
    ]
    return [evaluate(r, lay) for r in rows]


def spec_iteration(cfg: GLM53FlashConfig, lay: Layout) -> dict:
    """One speculative iteration at B users per row with K = ``lay.spec_k`` drafts: V (verify: the 45 layers + head on
    R = B (K + 1) rows, then the MTP level 1 on all R rows), host accept, C (commit) and D (MTP levels 2 .. K on B rows);
    floors and modelled times in ms, tokens per iteration and the tok/s per user they imply."""
    assert lay.spec_k >= 1, "spec_k >= 1"
    b, k = lay.users_per_row, lay.spec_k
    if (
        b < TILE
    ):  # one expert cost basis for V, D and the plain step: the stream op (E0; the T=4 block op has no R-row point)
        lay = replace(lay, expert_path="stream")
    r = lay.verify_rows
    v_layers = {name: layer_summary(cfg, lay, name) for name in LAYER_COUNTS}
    v_floor = sum(s["floor_us"] * s["count"] for s in v_layers.values())
    v_model = sum(s["measured_us"] * s["count"] for s in v_layers.values())
    v_after = sum(s["after_us"] * s["count"] for s in v_layers.values())
    mtp_v = mtp_block_rows(cfg, lay, r)
    mtp_d = mtp_block_rows(cfg, lay, b) if k > 1 else []
    onehot = (k - 1) * 10.0
    c_rows = commit_rows(cfg, lay)
    f = lambda rows: sum(x.floor_us for x in rows)
    mm = lambda rows: sum(x.measured_us or 0.0 for x in rows)
    aa = lambda rows: sum(x.after_us if x.after_us is not None else (x.measured_us or 0.0) for x in rows)
    phases = {
        "V main model (R rows)": (v_floor, v_model, v_after),
        "V MTP level 1 (R rows)": (f(mtp_v), mm(mtp_v), aa(mtp_v)),
        "host accept": (lay.host_ms * 1e3, lay.host_ms * 1e3, lay.host_ms * 1e3),
        "C commit (KDA replay + blends, index)": (f(c_rows), mm(c_rows), aa(c_rows)),
        "D MTP levels 2..K (B rows)": (
            (k - 1) * f(mtp_d) + onehot,
            (k - 1) * mm(mtp_d) + onehot,
            (k - 1) * aa(mtp_d) + onehot,
        ),
    }
    tot = [sum(p[i] for p in phases.values()) / 1e3 for i in range(3)]
    plain = step_table(cfg, replace(lay, spec_k=0))
    verify = [(v_floor + f(mtp_v)) / 1e3, (v_model + mm(mtp_v)) / 1e3, (v_after + aa(mtp_v)) / 1e3]
    return {
        "B": b,
        "K": k,
        "R": r,
        "phases_ms": {name: tuple(x / 1e3 for x in v) for name, v in phases.items()},
        "verify_ms": tuple(verify),  # V: the 45 layers + head on R rows + MTP level 1 on R rows (floor / model / after)
        "verify_ratio_model": verify[1] / plain["step_model_measured_ms"],  # the R-row cost ratio over the plain step
        "verify_ratio_floor": verify[0] / plain["step_floor_ms"],
        "iteration_floor_ms": tot[0],
        "design_floor_ms": (
            tot[0] * 1e3 - sum(x.floor_us for x in c_rows if x.key in ("spec.kda_replay", "spec.kda_blend"))
        )
        / 1e3,  # SP1: the verify's launches commit in place, no replay / blend rows
        "iteration_model_ms": tot[1],
        "iteration_after_ms": tot[2],
        "accepted": lay.accepted,
        "tok_s_per_user_floor": lay.accepted * 1e3 / tot[0],
        "tok_s_per_user_model": lay.accepted * 1e3 / tot[1],
        "tok_s_per_user_after": lay.accepted * 1e3 / tot[2],
        "plain_step_model_ms": plain["step_model_measured_ms"],
        "plain_step_floor_ms": plain["step_floor_ms"],
        "tok_s_per_user_plain_model": 1e3 / plain["step_model_measured_ms"],
        "tokens_needed_for_33": 33.0 * tot[1] / 1e3,
        "tokens_needed_for_33_floor": 33.0 * tot[0] / 1e3,
        # accepted tokens per iteration a target tok/s per user needs: model / floor / after-actions iteration
        "tokens_needed": {
            tgt: (tgt * tot[1] / 1e3, tgt * tot[0] / 1e3, tgt * tot[2] / 1e3) for tgt in SPEC_TARGETS_TOK_S
        },
        "speedup_model": (lay.accepted * 1e3 / tot[1]) / (1e3 / plain["step_model_measured_ms"]),
        "commit_rows": c_rows,
        "mtp_rows": mtp_v,
    }


def spec_sweep(cfg: GLM53FlashConfig, lay: Layout, ks=(1, 2, 3, 5)) -> list[dict]:
    out = []
    for k in ks:
        if lay.users_per_row * (k + 1) > TILE:
            continue
        out.append(spec_iteration(cfg, replace(lay, spec_k=k)))
    return out


SPEC_TARGETS_TOK_S = (30.0, 33.0, 50.0)


def spec_scenarios(
    cfg: GLM53FlashConfig,
    lay: Layout,
    plain_steps=(80.0, 45.0, 33.0),
    accepted=(1.5, 2.0, 2.5),
    targets=SPEC_TARGETS_TOK_S,
) -> dict:
    """tok/s per user for a plain step of ``plain_steps`` ms and ``accepted`` tokens per iteration: the verify is the
    plain step x the model's R-row cost ratio (``verify_ratio_model``), the host accept is fixed, and C + D scale with
    the plain step like the verify (the same op classes: kda_decode launches, elementwise state ops, one MTP pass per
    level). ``table_design`` puts C + D at their after-actions values instead (SP1: the verify's launches commit in
    place, B5 index program): the commit no longer scales with anything. ``tokens_needed[target][plain]`` = the
    accepted tokens per iteration a target tok/s per user needs at that plain step. The elasticities at the central
    cell say which assumption dominates."""
    it = spec_iteration(cfg, lay)
    ph = it["phases_ms"]
    v = ph["V main model (R rows)"][1] + ph["V MTP level 1 (R rows)"][1]
    cd = ph["C commit (KDA replay + blends, index)"][1] + ph["D MTP levels 2..K (B rows)"][1]
    cd_design = ph["C commit (KDA replay + blends, index)"][2] + ph["D MTP levels 2..K (B rows)"][2]
    plain_model = it["plain_step_model_ms"]
    host = lay.host_ms

    def iteration(plain, host_ms=host, cd_scale=1.0, design=False):
        scale = plain / plain_model
        return v * scale + host_ms + (cd_design if design else cd * scale * cd_scale)

    table = {p: {a: a * 1e3 / iteration(p) for a in accepted} for p in plain_steps}
    table_design = {p: {a: a * 1e3 / iteration(p, design=True) for a in accepted} for p in plain_steps}
    tokens_needed = {t: {p: t * iteration(p) / 1e3 for p in plain_steps} for t in targets}
    tokens_needed_design = {t: {p: t * iteration(p, design=True) / 1e3 for p in plain_steps} for t in targets}
    p0, a0 = plain_steps[len(plain_steps) // 2], accepted[len(accepted) // 2]
    base = a0 * 1e3 / iteration(p0)
    eps = 0.1
    elast = {
        "plain step": ((a0 * 1e3 / iteration(p0 * (1 + eps))) / base - 1) / eps,
        "accepted tokens": ((a0 * (1 + eps) * 1e3 / iteration(p0)) / base - 1) / eps,
        "host overhead": ((a0 * 1e3 / iteration(p0, host_ms=host * (1 + eps))) / base - 1) / eps,
        "commit + chain cost": ((a0 * 1e3 / iteration(p0, cd_scale=1 + eps)) / base - 1) / eps,
    }
    dominant = max(elast, key=lambda n: abs(elast[n]))
    return {
        "table": table,
        "table_design": table_design,
        "tokens_needed": tokens_needed,
        "tokens_needed_design": tokens_needed_design,
        "center": (p0, a0, base),
        "elasticities": elast,
        "dominant": dominant,
        "v_fraction": v / (v + cd),
        "verify_ratio": it["verify_ratio_model"],
        "iteration_at": {p: iteration(p) for p in plain_steps},
        "iteration_design_at": {p: iteration(p, design=True) for p in plain_steps},
    }


def format_spec(cfg: GLM53FlashConfig, lay: Layout, markdown: bool = False) -> str:
    lines = []
    sw = spec_sweep(cfg, lay)
    hdr = [
        "K",
        "R",
        "V main",
        "V MTP1",
        "host",
        "C",
        "D",
        "verify step model ms (x plain)",
        "iteration floor (design floor) / model / after ms",
        "tok/s/u floor / model / after",
        "plain model",
        "speedup",
        "tokens for 33 tok/s",
    ]
    rows = []
    for it in sw:
        ph = it["phases_ms"]
        rows.append(
            [
                it["K"],
                it["R"],
                f"{ph['V main model (R rows)'][1]:.1f}",
                f"{ph['V MTP level 1 (R rows)'][1]:.1f}",
                f"{ph['host accept'][1]:.1f}",
                f"{ph['C commit (KDA replay + blends, index)'][1]:.1f}",
                f"{ph['D MTP levels 2..K (B rows)'][1]:.1f}",
                f"{it['verify_ms'][1]:.1f} ({it['verify_ratio_model']:.2f}x)",
                f"{it['iteration_floor_ms']:.1f} ({it['design_floor_ms']:.1f}) / {it['iteration_model_ms']:.1f} / {it['iteration_after_ms']:.1f}",
                f"{it['tok_s_per_user_floor']:.1f} / {it['tok_s_per_user_model']:.1f} / {it['tok_s_per_user_after']:.1f}",
                f"{it['plain_step_model_ms']:.1f} ms = {it['tok_s_per_user_plain_model']:.1f}",
                f"{it['speedup_model']:.2f}x",
                f"{it['tokens_needed_for_33']:.1f} ({it['tokens_needed_for_33_floor']:.1f} at floor)",
            ]
        )
    lines.append(
        f"speculative decode at {lay.users_per_row} users/row, accepted {lay.accepted} tokens per iteration, host {lay.host_ms} ms (ms per iteration, model = sum of measured / projected rows):"
    )
    if markdown:
        lines += ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)] + [
            "| " + " | ".join(_fmt(v, 1, True) for v in r) + " |" for r in rows
        ]
    else:
        lines += ["  ".join(hdr)] + ["  ".join(_fmt(v) for v in r) for r in rows]
    if lay.spec_k:
        it = spec_iteration(cfg, lay)
        lines.append("")
        lines.append(f"K={lay.spec_k} commit rows (per iteration, all layers):")
        for r in it["commit_rows"]:
            lines.append(
                f"  {r.key}: {r.measured_us / 1e3:.2f} ms model, floor {r.floor_us / 1e3:.2f}, after {r.after_us / 1e3:.2f} ({' -> '.join(s[0] for s in r.steps)}); {r.op}"
            )
        sc = spec_scenarios(cfg, lay)
        lines.append("")
        lines.append(
            f"tok/s per user vs plain step x accepted tokens per iteration (verify = plain step x {sc['verify_ratio']:.2f} "
            f"(the R = {it['R']}-row cost ratio), host {lay.host_ms} ms fixed; C + D scaled with the plain step):"
        )
        acc = list(next(iter(sc["table"].values())).keys())
        tgts = list(sc["tokens_needed"].keys())
        h2 = (
            ["plain step ms", "verify ms", "iteration ms"]
            + [f"accepted {a}" for a in acc]
            + [f"tokens for {t:.0f} tok/s" for t in tgts]
        )
        tables = [("today's commit graph", "table", "iteration_at", "tokens_needed")]
        tables.append(
            (
                "SP1 + B5 design (C + D at their after values)",
                "table_design",
                "iteration_design_at",
                "tokens_needed_design",
            )
        )
        for title, tk, ik, nk in tables:
            rows2 = [
                [f"{p:.0f}", f"{p * sc['verify_ratio']:.1f}", f"{sc[ik][p]:.1f}"]
                + [f"{sc[tk][p][a]:.1f}" for a in acc]
                + [f"{sc[nk][t][p]:.2f}" for t in tgts]
                for p in sc[tk]
            ]
            lines.append(f"{title}:")
            if markdown:
                lines += ["| " + " | ".join(h2) + " |", "|" + "---|" * len(h2)] + [
                    "| " + " | ".join(r) + " |" for r in rows2
                ]
            else:
                lines += ["  ".join(h2)] + ["  ".join(r) for r in rows2]
            lines.append("")
        e = sc["elasticities"]
        lines.append(
            f"elasticities at the centre (plain {sc['center'][0]:.0f} ms, accepted {sc['center'][1]}: {sc['center'][2]:.1f} tok/s/u): "
            + ", ".join(f"{k} {v:+.2f}" for k, v in e.items())
            + f" -> dominant: {sc['dominant']} (V is {sc['v_fraction'] * 100:.0f} % of the device time)"
        )
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--users-per-row", type=int, default=4)
    p.add_argument("--rows", type=int, default=8)
    p.add_argument("--cols", type=int, default=4)
    p.add_argument("--moe", choices=["hybrid", "ep"], default="hybrid")
    p.add_argument("--expert-path", choices=["block", "stream"], default="block")
    p.add_argument("--expert-dtype", choices=["bfp8", "bfp4"], default="bfp8", help="gate|up dtype")
    p.add_argument("--expert-down-dtype", choices=["bfp8", "bfp4"], default="bfp8")
    p.add_argument("--isl", type=int, default=16)
    p.add_argument("--routing", choices=["measured", "uniform"], default="measured")
    p.add_argument("--active-experts", type=float, default=None)
    p.add_argument("--dram-gbs", type=float, default=chip("dram_gbs"))
    p.add_argument("--link-gbs", type=float, default=chip("eth_link_gbs"))
    p.add_argument("--links", type=int, default=chip("links_per_axis"))
    p.add_argument("--topology", choices=["line", "ring"], default="line")
    p.add_argument("--mhc-pre-block", action="store_true", help="A1 fused pre-block on")
    p.add_argument("--dsa-state-fused", action="store_true", help="B5 on")
    p.add_argument(
        "--spec-k",
        type=int,
        default=0,
        help="speculative decode: K drafts per user (0 = plain; the policy picks min(K_ctx, 32 // B - 1): spec_policy_k)",
    )
    p.add_argument(
        "--accepted", type=float, default=2.0, help="tokens committed per iteration (1 + mean accepted drafts)"
    )
    p.add_argument("--host-ms", type=float, default=2.0, help="host accept overhead per iteration (ms)")
    p.add_argument("--layer", choices=list(LAYER_COUNTS), default=None, help="print this layer type's op table")
    p.add_argument("--all-layers", action="store_true")
    p.add_argument("--markdown", action="store_true")
    p.add_argument("--json", action="store_true")
    p.add_argument(
        "--snapshot", default=None, help="HF snapshot dir (default: the cached checkpoint, else the built-in config)"
    )
    a = p.parse_args(argv)
    cfg = load_config(a.snapshot)
    lay = Layout(
        rows=a.rows,
        cols=a.cols,
        users_per_row=a.users_per_row,
        moe=a.moe,
        expert_path=a.expert_path,
        expert_w13_dtype=a.expert_dtype,
        expert_w2_dtype=a.expert_down_dtype,
        isl=a.isl,
        routing=a.routing,
        active_experts=a.active_experts,
        dram_gbs=a.dram_gbs,
        link_gbs=a.link_gbs,
        links=a.links,
        topology=a.topology,
        mhc_pre_block=a.mhc_pre_block,
        dsa_state_fused=a.dsa_state_fused,
        accepted=a.accepted,
        host_ms=a.host_ms,
    )
    if a.spec_k:
        spec_lay = replace(lay, spec_k=a.spec_k)
        if a.json:
            it = spec_iteration(cfg, spec_lay)
            it = {k: v for k, v in it.items() if k not in ("commit_rows", "mtp_rows")}
            print(
                json.dumps(
                    {
                        "spec": it,
                        "sweep": [
                            {k: v for k, v in s.items() if k not in ("commit_rows", "mtp_rows")}
                            for s in spec_sweep(cfg, lay)
                        ],
                        "scenarios": spec_scenarios(cfg, spec_lay),
                    },
                    indent=1,
                    default=str,
                )
            )
            return
        print("\n## speculative decode\n")
        print(format_spec(cfg, spec_lay, a.markdown))
        return
    step = step_table(cfg, lay)
    if a.json:
        print(json.dumps({"step": step, "actions": ranked_actions(cfg, lay)}, indent=1, default=str))
        return
    layers = list(LAYER_COUNTS) if a.all_layers else ([a.layer] if a.layer else [])
    for name in layers:
        print(f"\n## {name} (x{LAYER_COUNTS[name]}), {lay.users} users\n")
        print(format_layer_table(layer_table(cfg, lay, name), a.markdown))
    print("\n## step\n")
    print(format_step(step, a.markdown))
    print("\n## ranked actions (ms saved per step at this point)\n")
    print(format_actions(ranked_actions(cfg, lay), a.markdown))


def load_config(snapshot: str | None = None) -> GLM53FlashConfig:
    """The checkpoint's config when the HF cache has it, else the dataclass defaults with the 45-layer schedule."""
    try:
        return GLM53FlashConfig.from_snapshot(snapshot)
    except (FileNotFoundError, OSError, KeyError):
        kda, dsa = "linear_attention", "deepseek_sparse_attention"
        layer_types = tuple(dsa if i % 4 == 3 else kda for i in range(45))
        mlp = tuple("dense" if i < 3 else "sparse" for i in range(45))
        return GLM53FlashConfig(layer_types=layer_types, mlp_layer_types=mlp)


if __name__ == "__main__":
    main()
