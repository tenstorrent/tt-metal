# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""(Cgt, Ceq) exact-count engine microbenchmark driver
(``sources/cgtceq_perf.cpp``, Blackhole only).

HONESTY GUARD: this bench prices the Gate-2 correctness ORACLE only
(RADIX_BUCKET_GPU.md IMPL-2). Nothing here is a claimed speedup; the numbers
close dependency-map open dep #1 (rendezvous cost) and give the dual-RISC
histogram alternative an honest SFPU-side comparator.

Three test groups, one module (one CSV schema — all variants share the single
``CGTCEQ_PARAMS`` parameter class, the TOPK_MICRO_OP precedent):

  test_cgtceq_stream_additivity
      Streamed pipeline arms (none / single / dual / ctrl_load / ctrl_swap)
      under L1_TO_L1 + UNPACK_ISOLATE + MATH_ISOLATE, two-point slope over
      TILE_CNT {16, 64}. Gate-2 question: L1_TO_L1(single) - L1_TO_L1(none)
      ?= MATH_ISOLATE(single), i.e. is the count fully additive to the ~3.94
      cyc/vec fp32 unpack_to_dest floor. Timing-only (bodies are
      byte-identical to the checked arms below).

  test_cgtceq_rendezvous
      The 3x3 fold-depth x ordering-primitive crossing, plus the plain-rate
      subtraction partner. Every segment counts KNOWN data (2 tiles unpacked
      in INIT) and the next threshold depends on the count the RISC just read
      back through memory-mapped Dst; the kernel reports an XOR checksum of
      the per-segment counts and this driver simulates the same automaton
      exactly — any count mismatch anywhere in the chain fails the test.
      cycles/decision = (slope(rendezvous) - slope(rate)) * 64 over
      ITER_COUNT {512, 2048}; the arithmetic lives in cgtceq_analysis.py.

  test_cgtceq_bisect
      TRISC1-driven <=16-decision bisection to the exact K-th threshold over
      Dst-resident rows (1 row = 1 tile = 1024 bf16-pattern fp32 words).
      The driver simulates the kernel's exact bisection (same ord<->key maps,
      same early-exit rule) against a sign-magnitude golden and asserts every
      reported field: found threshold, Cgt, Ceq, decision count, exit mode,
      and the invariant Cgt < K <= Cgt + Ceq (CERT) / Cgt == K (VALIDSET).
      Per-row decisions/cycles are appended to CGTCEQ_ROWS_OUT
      (default /tmp/cgtceq_bisect_rows.txt) for p50/p95 aggregation.

READING THE NUMBERS: raw zone cycles live in
perf_data/test_cgtceq_perf/test_cgtceq_perf.csv after the consumer phase;
run python_tests/cgtceq_analysis.py to get the additivity table, the 3x3
rendezvous cycles/decision matrix, and the bisection p50/p95s.

Run order discipline (sfpu_count_above lesson): run
test_profiler_overhead.py first; read ctrl_load (~1.0 cyc/vec) and ctrl_swap
(~2.0) before believing any other arm.

Scale knob: CGTCEQ_BISECT_GROUPS (default 4) seed-groups per random-ish
distribution; 34 gives the >=100 rows/distribution the report asks for.
"""

import os
import struct
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from conftest import blackhole_only
from helpers.device_io import read_from_device
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import (
    ELEMENTS_PER_TILE,
    CountAboveGolden,
    sign_magnitude_order_key,
)
from helpers.llk_params import DestAccumulation, PerfRunType
from helpers.param_config import parametrize
from helpers.perf.core import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import BuildMode, TestConfig
from helpers.test_variant_parameters import (
    LOOP_FACTOR,
    NUM_FACES,
    RELU_CONFIG,
    TILE_COUNT,
    TemplateParameter,
    generate_input_dim,
)

# --- kernel-mirrored constants -------------------------------------------------

ARM_IDS = {
    "stream_none": 0,
    "stream_single": 1,
    "stream_dual": 2,
    "ctrl_load": 3,
    "ctrl_swap": 4,
    "rate": 5,
    "rendezvous": 6,
    "bisect": 7,
}

DIAG_MAGIC = 0xC67C0DE1
VECTORS_PER_SEGMENT = 64
SRC_SLOTS = 16  # L1 stimulus ring; must match cgtceq_perf.cpp
RES_SLOTS = 16

_FP32 = InputOutputFormat(DataFormat.Float32, DataFormat.Float32)
_U32 = InputOutputFormat(DataFormat.UInt32, DataFormat.UInt32)

_TILE_COUNTS = [16, 64]  # streamed two-point slope (multiples of the 4-tile Dest block)
_ITER_COUNTS = [512, 2048]  # rate/rendezvous two-point slope

_BISECT_GROUPS = int(os.environ.get("CGTCEQ_BISECT_GROUPS", "4"))
_ROWS_OUT = Path(os.environ.get("CGTCEQ_ROWS_OUT", "/tmp/cgtceq_bisect_rows.txt"))

_RUN_COUNT = 5


# --- the single parameter class (one CSV schema for the whole module) ----------


@dataclass
class CGTCEQ_PARAMS(TemplateParameter):
    """All compile-time knobs of cgtceq_perf.cpp, bundled so every variant in
    this module emits the same CSV columns (assert_single_schema is a hard
    gate). All emitted as #define: the kernel guards each with #ifndef and a
    constexpr would silently leave the fallback in place for every variant.

    ``arm``/``dist``/``seed`` are emitted as comments where the kernel does
    not read them — they exist to enter the variant hash and the report row.
    """

    arm: str = "rate"
    fold: int = 0
    sync: int = 0
    iters: int = 512
    thr_bits: int = 0x3F800000
    thr2_bits: int = 0x3F000000
    k: int = 32
    rows: int = 3
    dist: str = "na"
    seed: int = 0

    def convert_to_cpp(self) -> str:
        return "\n".join(
            [
                f"#define CGTCEQ_ARM {ARM_IDS[self.arm]}",
                f"#define ITER_COUNT {self.iters}",
                f"#define THR_BITS 0x{self.thr_bits:08X}",
                f"#define THR2_BITS 0x{self.thr2_bits:08X}",
                f"#define FOLD_DEPTH {self.fold}",
                f"#define SYNC_PRIM {self.sync}",
                f"#define BISECT_K {self.k}",
                f"#define BISECT_ROWS {self.rows}",
                f"// arm = {self.arm}",
                f"// dist = {self.dist}",
                f"// seed = {self.seed}",
            ]
        )


# --- bit-pattern helpers --------------------------------------------------------


def bf16_bits(x: float) -> int:
    """bf16 bit pattern of x (truncation of the fp32 pattern)."""
    return struct.unpack("<I", struct.pack("<f", float(x)))[0] >> 16


def bf16_pattern_fp32(x: float) -> int:
    """fp32 bit pattern whose low 16 mantissa bits are zero (an exact bf16)."""
    return bf16_bits(x) << 16


def key_to_ord(k: int) -> int:
    """Kernel's monotone 16-bit sign-magnitude -> unsigned order map."""
    return ((~k) & 0xFFFF) if (k & 0x8000) else (k | 0x8000)


def ord_to_key(m: int) -> int:
    return (m & 0x7FFF) if (m & 0x8000) else ((~m) & 0xFFFF)


def _count_above(values_bits, thr_bits: int) -> int:
    # Direct instantiation (not get_golden_generator) so the golden also runs
    # under --compile-producer paths that swap in the dummy generator.
    return CountAboveGolden()(values_bits, thr_bits)


def _bf16_ladder(n: int) -> list:
    """n distinct, exactly-representable bf16 magnitudes in [1, 256)."""
    vals = []
    for e in range(8):
        for m in range(128):
            vals.append((2.0**e) * (1.0 + m / 128.0))
    return vals[:n]


# --- rendezvous stimulus + automaton golden -------------------------------------


def _rendezvous_stimulus():
    """2 tiles (2048 words) of distinct bf16-pattern fp32 values, plus two
    thresholds with an ODD and an EVEN strict-above count so the kernel's
    data-dependent threshold automaton actually alternates."""
    g = torch.Generator().manual_seed(23)
    ladder = _bf16_ladder(1024)
    perm = torch.randperm(1024, generator=g).tolist()
    vals = [ladder[perm[i]] for i in range(1024)] + [
        -ladder[perm[i]] for i in range(1024)
    ]
    bits = [bf16_pattern_fp32(v) for v in vals]

    thr_odd = None
    thr_even = None
    for v in sorted(set(vals)):
        c = _count_above(bits, bf16_pattern_fp32(v))
        if thr_odd is None and (c & 1) == 1 and c > 0:
            thr_odd = bf16_pattern_fp32(v)
        if thr_even is None and (c & 1) == 0 and c > 0:
            thr_even = bf16_pattern_fp32(v)
        if thr_odd is not None and thr_even is not None:
            break
    assert thr_odd is not None and thr_even is not None
    return bits, thr_odd, thr_even


def _simulate_rendezvous(bits, thr: int, thr2: int, segments: int):
    """Mirror of the kernel's segment automaton: per segment count strictly
    above the current threshold over tiles 0..1, fold checksum, pick the next
    threshold by count parity."""
    c_by_thr = {thr: _count_above(bits, thr), thr2: _count_above(bits, thr2)}
    checksum = 0
    t = thr
    last = 0
    for seg in range(segments):
        c = c_by_thr[t]
        checksum ^= (c + seg) & 0xFFFFFFFF
        last = c
        t = thr2 if (c & 1) else thr
    return checksum, last


# --- bisection stimuli + exact kernel simulation ---------------------------------


def _row_random(rng: torch.Generator, seed_shift: int) -> list:
    g = torch.Generator().manual_seed(1000 + seed_shift)
    vals = torch.randn(ELEMENTS_PER_TILE, generator=g) * 8.0
    return [bf16_pattern_fp32(float(v)) for v in vals]


def _row_clustered(rng, seed_shift: int) -> list:
    """All values inside one binade (RadiK's adversarial case for value-space
    bucketing; key-space bisection still makes 1 bit/decision progress)."""
    g = torch.Generator().manual_seed(2000 + seed_shift)
    m = torch.randint(0, 128, (ELEMENTS_PER_TILE,), generator=g)
    return [bf16_pattern_fp32(1.0 + int(x) / 128.0) for x in m]


def _row_allequal(rng, seed_shift: int) -> list:
    return [bf16_pattern_fp32(3.0)] * ELEMENTS_PER_TILE


def _row_kstraddle(rng, seed_shift: int) -> list:
    """24 distinct values above a 16-deep tie block: counts jump 24 -> 40, so
    K in {31,32,33} cannot early-exit and must take the certified-tie path."""
    ladder = _bf16_ladder(1024)
    top = [200.0 + i for i in range(24)]
    ties = [100.0] * 16
    rest = ladder[: ELEMENTS_PER_TILE - len(top) - len(ties)]
    rest = [v / 512.0 for v in rest]  # keep them all below the tie value
    return [bf16_pattern_fp32(v) for v in (top + ties + rest)]


def _row_ties_at_threshold(rng, seed_shift: int) -> list:
    """31 above 2.0, five copies of 2.0, rest below: K=32 lands ON the tie."""
    top = [10.0 + i for i in range(31)]
    ties = [2.0] * 5
    rest = [1.0 + (i % 100) / 128.0 for i in range(ELEMENTS_PER_TILE - 36)]
    return [bf16_pattern_fp32(v) for v in (top + ties + rest)]


def _row_allneg(rng, seed_shift: int) -> list:
    g = torch.Generator().manual_seed(3000 + seed_shift)
    vals = -(torch.rand(ELEMENTS_PER_TILE, generator=g) * 100.0 + 0.5)
    return [bf16_pattern_fp32(float(v)) for v in vals]


def _row_specials(rng, seed_shift: int) -> list:
    """+-0 / +-Inf / NaNs / denormals in the mix (the Gate-3 specials list).
    SFPGT's total order -NaN < -Inf < ... < -0 < +0 < ... < +Inf < +NaN
    handles all of them; the golden uses the same order."""
    POS_INF = 0x7F800000
    NEG_INF = 0xFF800000
    POS_NAN = 0x7FC00000
    NEG_NAN = 0xFFC00000
    POS_ZERO = 0x00000000
    NEG_ZERO = 0x80000000
    DENORM = 0x00010000  # bf16 denormal pattern << 16
    fixed = (
        [POS_INF] * 10
        + [NEG_INF] * 10
        + [POS_NAN] * 10
        + [NEG_NAN] * 10
        + [POS_ZERO] * 5
        + [NEG_ZERO] * 5
        + [DENORM] * 10
    )
    g = torch.Generator().manual_seed(4000 + seed_shift)
    vals = torch.randn(ELEMENTS_PER_TILE - len(fixed), generator=g) * 4.0
    return fixed + [bf16_pattern_fp32(float(v)) for v in vals]


_DISTS = {
    "random": _row_random,
    "clustered": _row_clustered,
    "allequal": _row_allequal,
    "kstraddle": _row_kstraddle,
    "ties": _row_ties_at_threshold,
    "allneg": _row_allneg,
    "specials": _row_specials,
}


def _simulate_bisect(row_bits: list, k: int):
    """Exact mirror of the kernel's TRISC1 bisection (same maps, same
    early-exit rule, same certification), computed with the sign-magnitude
    golden. Returns (found_thr_bits, m_star, cgt, ceq, decisions, exit_mode).
    """
    lo, hi = 0, 0xFFFF
    decisions = 0
    while lo < hi:
        mid = lo + ((hi - lo) >> 1)
        thr_bits = ord_to_key(mid) << 16
        c = _count_above(row_bits, thr_bits)
        decisions += 1
        if c == k:
            return thr_bits, mid, c, 0, decisions, 2  # VALIDSET
        if c < k:
            hi = mid
        else:
            lo = mid + 1
    key = ord_to_key(lo)
    thr_bits = key << 16
    cgt = _count_above(row_bits, thr_bits)
    if lo > 0:
        cge = _count_above(row_bits, ord_to_key(lo - 1) << 16)
    else:
        cge = len(row_bits)
    decisions += 1  # the dual certification count
    return thr_bits, lo, cgt, cge - cgt, decisions, 1  # CERT


# --- common PerfConfig plumbing ---------------------------------------------------


def _runtimes(tile_cnt: int):
    return [
        TILE_COUNT(tile_cnt),
        LOOP_FACTOR(1),
        RELU_CONFIG(0),
        NUM_FACES(num_faces=4),
    ]


def _write_stimuli(configuration) -> None:
    """Write variant_stimuli to device L1 before running.

    Bring-up root cause (2026-08-16): ``TestConfig.run()`` writes
    ``variant_stimuli`` to L1, but ``PerfConfig.run()`` overrides run() and
    never does — the perf flow is timing-only by design. Every self-checking
    arm here counts REAL data, so the driver must do the write itself (the
    L1 buffers persist across the perf run loop; nothing else touches them).
    """
    if TestConfig.BUILD_MODE == BuildMode.PRODUCE:
        return
    configuration.variant_stimuli.write(TestConfig.TENSIX_LOCATION)


def _read_diag(configuration, num_words: int) -> list:
    raw = read_from_device(
        TestConfig.TENSIX_LOCATION,
        configuration.variant_stimuli.buf_res_addr,
        num_bytes=4 * num_words,
    )
    return [
        int.from_bytes(bytes(raw[4 * i : 4 * i + 4]), "little")
        for i in range(num_words)
    ]


# ==================================================================================
# (i) additivity: streamed count arms
# ==================================================================================


@pytest.mark.perf
@blackhole_only
@parametrize(
    arm=["stream_none", "stream_single", "stream_dual", "ctrl_load", "ctrl_swap"],
    tile_cnt=_TILE_COUNTS,
)
def test_cgtceq_stream_additivity(perf_report, arm, tile_cnt):
    ladder = _bf16_ladder(ELEMENTS_PER_TILE)
    one_tile = torch.tensor(
        [bf16_pattern_fp32(v) for v in ladder], dtype=torch.int64
    ).view(torch.int64)
    src_A = one_tile.to(torch.int32).view(torch.float32).repeat(SRC_SLOTS)
    src_B = torch.zeros(ELEMENTS_PER_TILE, dtype=torch.float32)

    configuration = PerfConfig(
        "sources/cgtceq_perf.cpp",
        _FP32,
        run_types=[
            PerfRunType.L1_TO_L1,
            PerfRunType.UNPACK_ISOLATE,
            PerfRunType.MATH_ISOLATE,
        ],
        templates=[
            generate_input_dim([32, 32], [32, 32]),
            CGTCEQ_PARAMS(
                arm=arm,
                # Thresholds mid-ladder so the compare has real work; timing
                # is data-independent either way.
                thr_bits=bf16_pattern_fp32(16.0),
                thr2_bits=bf16_pattern_fp32(15.9375),
                dist="stream",
            ),
        ],
        runtimes=_runtimes(tile_cnt),
        variant_stimuli=StimuliConfig(
            src_A,
            _FP32.input_format,
            src_B,
            _FP32.input_format,
            _FP32.output_format,
            tile_count_A=SRC_SLOTS,
            tile_count_B=1,
            tile_count_res=RES_SLOTS,
        ),
        dest_acc=DestAccumulation.Yes,
        unpack_to_dest=True,
    )

    _write_stimuli(configuration)
    configuration.run(perf_report, run_count=_RUN_COUNT)


# ==================================================================================
# (ii) rendezvous: fold x ordering, self-checked against the automaton golden
# ==================================================================================

_RDV_BITS, _RDV_THR, _RDV_THR2 = None, None, None


def _rdv_stimulus_cached():
    global _RDV_BITS, _RDV_THR, _RDV_THR2
    if _RDV_BITS is None:
        _RDV_BITS, _RDV_THR, _RDV_THR2 = _rendezvous_stimulus()
    return _RDV_BITS, _RDV_THR, _RDV_THR2


def _rdv_cases():
    cases = []
    for iters in _ITER_COUNTS:
        cases.append(("rate", 0, 0, iters))
        for fold in (0, 1, 2):
            for sync in (0, 1, 2):
                cases.append(("rendezvous", fold, sync, iters))
    return cases


@pytest.mark.perf
@blackhole_only
@pytest.mark.parametrize(
    "arm, fold, sync, iters",
    [pytest.param(*c, id=f"{c[0]}-f{c[1]}-s{c[2]}-i{c[3]}") for c in _rdv_cases()],
)
def test_cgtceq_rendezvous(perf_report, arm, fold, sync, iters):
    bits, thr, thr2 = _rdv_stimulus_cached()
    src_A = torch.tensor(bits, dtype=torch.int64)
    src_B = torch.zeros(ELEMENTS_PER_TILE, dtype=torch.int64)

    configuration = PerfConfig(
        "sources/cgtceq_perf.cpp",
        _U32,
        # Same run-type triple as the stream tests: every variant in this module
        # must emit identical CSV columns (assert_single_schema fired on the
        # first bring-up run when this was MATH_ISOLATE-only). The fill arms
        # ignore PERF_RUN_TYPE in kernel code, so L1_TO_L1 / UNPACK_ISOLATE are
        # redundant-but-harmless re-measurements; the analysis reads only
        # mean(MATH_ISOLATE) for the rendezvous slopes.
        run_types=[
            PerfRunType.L1_TO_L1,
            PerfRunType.UNPACK_ISOLATE,
            PerfRunType.MATH_ISOLATE,
        ],
        templates=[
            generate_input_dim([32, 32], [32, 32]),
            CGTCEQ_PARAMS(
                arm=arm,
                fold=fold,
                sync=sync,
                iters=iters,
                thr_bits=thr,
                thr2_bits=thr2,
                dist="rdv",
            ),
        ],
        runtimes=_runtimes(1),
        variant_stimuli=StimuliConfig(
            src_A,
            _U32.input_format,
            src_B,
            _U32.input_format,
            _U32.output_format,
            tile_count_A=2,
            tile_count_B=1,
            tile_count_res=1,
        ),
        dest_acc=DestAccumulation.Yes,
        unpack_to_dest=True,
        disable_format_inference=True,
        compile_time_formats=True,
    )

    _write_stimuli(configuration)
    configuration.run(perf_report, run_count=_RUN_COUNT)

    if TestConfig.BUILD_MODE == BuildMode.PRODUCE:
        return
    if arm != "rendezvous":
        return

    # --- exactness gate: any dropped/mis-read count anywhere fails here ---
    d = _read_diag(configuration, 16)
    segments = iters // VECTORS_PER_SEGMENT
    exp_checksum, exp_last = _simulate_rendezvous(bits, thr, thr2, segments)
    print(
        f"\n  [rdv] fold={fold} sync={sync} iters={iters}: "
        f"magic=0x{d[0]:08X} segs={d[4]} checksum=0x{d[6]:08X} "
        f"(exp 0x{exp_checksum:08X}) last={d[8]} (exp {exp_last}) flags=0x{d[7]:X}"
    )
    print(
        "  [rdv] probe r0w0=0x{:08X} r64w0=0x{:08X} | cnt_t0_neg0={} (exp 1024) "
        "cnt_t1_neg0={} (exp 0) cnt_2t_neg0={} (exp 1024) cnt_t1_thr={} (exp 1023) "
        "scratch0=0x{:08X} | thr=0x{:08X}".format(
            d[9], d[10], d[11], d[12], d[13], d[14], d[15], thr
        )
    )
    assert d[0] == DIAG_MAGIC, "kernel never reached the diagnostics dump"
    assert d[7] == 0, (
        f"rendezvous flags 0x{d[7]:X}: 0x1=semaphore poll timeout, "
        f"0x2=sentinel poll timeout, 0x4=sentinel survived into the read. "
        f"A wrong lane-map model (R0_WORD) or a dead ordering primitive "
        f"reads out here instead of hanging."
    )
    assert d[4] == segments
    assert d[8] == exp_last, f"last segment count {d[8]} != golden {exp_last}"
    assert d[6] == exp_checksum, (
        f"per-segment count checksum mismatch: device 0x{d[6]:08X}, golden "
        f"0x{exp_checksum:08X} — the counting/fold/order/read chain dropped "
        f"or corrupted at least one count."
    )


# ==================================================================================
# (iii) bisection to the exact K-th threshold, exact-golden checked
# ==================================================================================


def _bisect_cases():
    cases = []
    for dist in ("random", "clustered", "allequal"):
        for grp in range(_BISECT_GROUPS):
            cases.append((dist, grp, 32, 0))  # sync=0 for the broad sweep
    for k in (31, 32, 33):
        cases.append(("kstraddle", 0, k, 0))
    cases.append(("ties", 0, 32, 0))
    cases.append(("allneg", 0, 32, 0))
    cases.append(("specials", 0, 32, 0))
    # In-situ ordering-primitive comparison on one distribution.
    cases.append(("random", 0, 32, 1))
    cases.append(("random", 0, 32, 2))
    return cases


_BISECT_ROWS_N = 3  # rows (tiles) per variant; +1 scratch tile = 4-tile Dest


@pytest.mark.perf
@blackhole_only
@pytest.mark.parametrize(
    "dist, group, k, sync",
    [pytest.param(*c, id=f"{c[0]}-g{c[1]}-k{c[2]}-s{c[3]}") for c in _bisect_cases()],
)
def test_cgtceq_bisect(perf_report, dist, group, k, sync):
    rows = [
        _DISTS[dist](None, group * _BISECT_ROWS_N + r) for r in range(_BISECT_ROWS_N)
    ]
    src_A = torch.tensor([b for row in rows for b in row], dtype=torch.int64)
    src_B = torch.zeros(ELEMENTS_PER_TILE, dtype=torch.int64)

    configuration = PerfConfig(
        "sources/cgtceq_perf.cpp",
        _U32,
        # Full run-type triple for CSV-schema homogeneity (see rendezvous note).
        run_types=[
            PerfRunType.L1_TO_L1,
            PerfRunType.UNPACK_ISOLATE,
            PerfRunType.MATH_ISOLATE,
        ],
        templates=[
            generate_input_dim([32, 32], [32, 32]),
            CGTCEQ_PARAMS(
                arm="bisect",
                fold=0,  # interior probes read 1 word (the R0 shape)
                sync=sync,
                k=k,
                rows=_BISECT_ROWS_N,
                dist=dist,
                seed=group,
            ),
        ],
        runtimes=_runtimes(1),
        variant_stimuli=StimuliConfig(
            src_A,
            _U32.input_format,
            src_B,
            _U32.input_format,
            _U32.output_format,
            tile_count_A=_BISECT_ROWS_N,
            tile_count_B=1,
            tile_count_res=1,
        ),
        dest_acc=DestAccumulation.Yes,
        unpack_to_dest=True,
        disable_format_inference=True,
        compile_time_formats=True,
    )

    _write_stimuli(configuration)
    # Same run_count as the other groups: a single run emits no std(...)
    # stat columns and splits the module CSV into two schemas (hard gate).
    configuration.run(perf_report, run_count=_RUN_COUNT)

    if TestConfig.BUILD_MODE == BuildMode.PRODUCE:
        return

    d = _read_diag(configuration, 16 + 8 * _BISECT_ROWS_N)
    assert d[0] == DIAG_MAGIC, "kernel never reached the diagnostics dump"
    assert d[7] == 0, f"rendezvous flags 0x{d[7]:X} during bisection"
    assert d[4] == _BISECT_ROWS_N and d[5] == k

    _ROWS_OUT.parent.mkdir(parents=True, exist_ok=True)
    for r in range(_BISECT_ROWS_N):
        rec = d[16 + 8 * r : 16 + 8 * r + 8]
        found_thr, m_star, cgt, ceq, decisions, exit_mode, cycles, ok = rec
        g_thr, g_m, g_cgt, g_ceq, g_dec, g_mode = _simulate_bisect(rows[r], k)

        print(
            f"\n  [bisect] {dist} g{group} k={k} s{sync} row{r}: "
            f"thr=0x{found_thr:08X} m*={m_star} Cgt={cgt} Ceq={ceq} "
            f"dec={decisions} mode={exit_mode} cyc={cycles} ok={ok} | "
            f"golden thr=0x{g_thr:08X} Cgt={g_cgt} Ceq={g_ceq} "
            f"dec={g_dec} mode={g_mode}"
        )

        assert ok == 1, f"row {r}: kernel-side invariant check failed"
        assert exit_mode == g_mode, f"row {r}: exit mode {exit_mode} != {g_mode}"
        assert (
            found_thr == g_thr
        ), f"row {r}: found threshold 0x{found_thr:08X} != golden 0x{g_thr:08X}"
        assert cgt == g_cgt, f"row {r}: Cgt {cgt} != golden {g_cgt}"
        if exit_mode == 1:
            assert ceq == g_ceq, f"row {r}: Ceq {ceq} != golden {g_ceq}"
            assert cgt < k <= cgt + ceq, f"row {r}: invariant violated"
        else:
            assert cgt == k
            # A VALIDSET threshold defines an exactly-K top set; re-check it
            # against the golden count independently of the kernel's own count.
            assert _count_above(rows[r], found_thr) == k
        assert decisions == g_dec and decisions <= 17

        with open(_ROWS_OUT, "a") as f:
            f.write(f"{dist},{group},{k},{sync},{r},{decisions},{cycles},{exit_mode}\n")


# ==================================================================================
# host-only sanity: the python order maps mirror the kernel's
# ==================================================================================


@pytest.mark.perf
def test_cgtceq_keymap_anchors():
    """No device. ord<->key must be inverse bijections and monotone against
    the sign-magnitude order of the corresponding fp32 patterns."""
    for m in (0, 1, 0x7FFE, 0x7FFF, 0x8000, 0x8001, 0xFFFE, 0xFFFF):
        assert key_to_ord(ord_to_key(m)) == m
    # -NaN(max payload) lowest, +NaN(max payload) highest, -0 just below +0.
    assert key_to_ord(0xFFFF) == 0x0000
    assert key_to_ord(0x8000) == 0x7FFF
    assert key_to_ord(0x0000) == 0x8000
    assert key_to_ord(0x7FFF) == 0xFFFF
    # Monotone against the golden's 32-bit sign-magnitude key on patterns<<16.
    ms = list(range(0, 0x10000, 257)) + [0xFFFF]
    keys32 = sign_magnitude_order_key(
        torch.tensor([ord_to_key(m) << 16 for m in ms], dtype=torch.int64)
    ).tolist()
    assert keys32 == sorted(keys32), "ord_to_key must be monotone in sign-mag order"
