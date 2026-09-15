# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Table-driven single-op SFPU accuracy harness with result plotting — Quasar.

Sibling of test_sfpu_plot.py (the WH/BH harness). It reuses that file's
plotting / stats code (plot_and_print and friends) unchanged and only swaps
the device side: each Case runs through the same Quasar kernel the functional
suite drives (sources/quasar/eltwise_unary_sfpu_quasar_test.cpp), with the
data route (Dest / SFPU / packer formats, unpack-to-Dest vs FPU datacopy)
resolved by resolve_quasar_sfpu_variant exactly as in
quasar/test_eltwise_unary_sfpu_quasar.py.

Each entry in CASES (right after the Case dataclass) sweeps one SFPU op over an
input domain and emits a multi-panel accuracy plot + stats summary (golden vs
hardware: signed ULP error, relative error, ULP CDF, per-bin percentiles,
monotonicity) to tests/python_tests/_plot_output/qsr/sfpu_<id>.png.

To add a test, append a Case(...) to CASES — see "HOW TO ADD A TEST" above
CASES. Run one op (from tests/python_tests, Quasar simulator):

    CHIP_ARCH=quasar pytest --run-simulator quasar/test_sfpu_plot_quasar.py -k Exp -s
"""

import math
import time
from dataclasses import dataclass, replace
from typing import List, Optional, Tuple

import numpy as np
import pytest
import torch
from helpers.chip_architecture import ChipArchitecture
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import (
    TILE_DIMENSIONS,
    UnarySFPUGolden,
    get_golden_generator,
)
from helpers.llk_params import (
    ApproximationMode,
    DataCopyType,
    DestAccumulation,
    DestSync,
    ImpliedMathFormat,
    MathOperation,
    PerfRunType,
    UnpackerEngine,
    format_dict,
)
from helpers.logger import logger
from helpers.param_config import (
    DEST_SYNC_TILE_LIMITS,
    QuasarSfpuVariant,
    resolve_quasar_sfpu_variant,
)
from helpers.sfpu_domains import _SFPU_UNDEFINED_RANGES, Operand
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import DistributionKind, StimuliSpec, generate_stimuli
from helpers.stimuli_generator.strategies.structured import (
    _enumerate_representable,
    ulp_sweep_value_count,
)
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    APPROX_MODE,
    DATA_COPY_TYPE,
    DEST_INDEX,
    DEST_SYNC,
    IMPLIED_MATH_FORMAT,
    LOOP_FACTOR,
    MATH_OP,
    NUM_FACES,
    PERF_RUN_TYPE,
    TEST_FACE_DIMS,
    TILE_COUNT,
    TYPECAST_FORMATS,
    UNPACKER_ENGINE_SEL,
)
from helpers.tile_constants import MAX_NUM_FACES
from helpers.utils import passed_test

# Plotting, stats, interval shading and downsampling are shared with the WH/BH
# harness so both archs produce identical figures for the same data.
from test_sfpu_plot import (
    FMT_SHORT,
    allowed_intervals_for,
    arch_title_suffix,
    plot_and_print,
    plot_output_dir,
)

QUASAR_KERNEL = "sources/quasar/eltwise_unary_sfpu_quasar_test.cpp"

# tests/python_tests/_plot_output/qsr — the same per-arch layout as the WH/BH
# harness (wh / bh), anchored to the file so the runner's chdir into quasar/
# does not move it.
PLOT_OUTPUT_DIR = plot_output_dir(ChipArchitecture.QUASAR)

BF16 = InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b)
FP16 = InputOutputFormat(DataFormat.Float16, DataFormat.Float16)
FP32 = InputOutputFormat(DataFormat.Float32, DataFormat.Float32)

# Ops the Quasar unary SFPU kernel dispatches with a real approximate kernel
# (mirrors quasar/test_eltwise_unary_sfpu_quasar.py). Every other op only has
# ApproximationMode.No; asking for approx on them is a Case mistake.
QUASAR_APPROX_CAPABLE_OPS = (
    MathOperation.Exp,
    MathOperation.Gelu,
    MathOperation.Reciprocal,
    MathOperation.Rsqrt,
)


# ---------------------------------------------------------------------------
# Stress harness — table-driven SFPU accuracy plots
# ---------------------------------------------------------------------------
#
# HOW TO ADD A TEST
#   Append one Case(...) to the CASES list below — that is the whole workflow:
#
#       Case(op=MathOperation.Exp, spec=StimuliSpec.ramp(low=-10.0, high=10.0))
#
#   Choose the input domain with a StimuliSpec (ramp / uniform / ulp_sweep — the
#   same API the rest of the suite uses) and the numeric format with fmt=:
#
#       fmt=BF16   bfloat16 in/out   (default; 16-bit Dest, judge accuracy in ULPs)
#       fmt=FP16   float16  in/out   (16-bit Dest)
#       fmt=FP32   float32  in/out   (auto-selects the fp32 Dest route;
#                                     judge accuracy in abs/rel error — fp32 ULP
#                                     is not meaningful, see the notes inside
#                                     plot_and_print)
#
#   Any InputOutputFormat works as long as Quasar can execute it: the route
#   (Dest width, unpack-to-Dest vs FPU datacopy, packer conversion) is picked by
#   resolve_quasar_sfpu_variant, and a Case the hardware cannot run fails
#   loudly at run time.
#
#   Run one op:   CHIP_ARCH=quasar pytest --run-simulator quasar/test_sfpu_plot_quasar.py -k Exp -s
#   Run all:      CHIP_ARCH=quasar pytest --run-simulator quasar/test_sfpu_plot_quasar.py -s
#
#   Each case writes _plot_output/qsr/sfpu_<id>.png and prints a stats
#   summary, then asserts the hardware result matches golden.
#   Set expect_pass=False to keep the run green while exploring a known-inaccurate op.
#
# Optional Case fields (advanced — sensible defaults cover the common cases):
#   StimuliSpec(intervals=[(lo, hi), ...])  sweep disjoint bands (e.g. either
#                                           side of a singularity)
#   approx_mode             ApproximationMode.Yes on QUASAR_APPROX_CAPABLE_OPS only
#   dest_acc                override the format-derived Dest width (fp32 Dest for
#                           a 16-bit format, for example)
#   dest_sync / implied_math_format  Quasar-only kernel knobs (default Half / Yes,
#                           the perf-mode pins of the functional suite)
#   extra_undefined_ranges  override the red "undefined domain" plot shading
#   name                    custom test id — the plot filename and the -k selector
#                           (defaults to "<Op>-<fmt>", or "<Op>-<fmt>-approx" when
#                           approx_mode=Yes)
#   input_dimensions        sample-point count (defaults to [32, 32] = 1024, one
#                           tile). The Quasar kernel runs every tile in one Dest
#                           section, so this is capped by Dest capacity: 8 tiles
#                           ([32, 256]) for a 16-bit Dest, 4 for fp32 at
#                           DestSync.Half (double at DestSync.Full).
#   batch_tiles             ulp_sweep only: tiles per device run when a sweep
#                           exceeds Dest capacity (defaults to the capacity).


@dataclass
class Case:
    """One SFPU stress configuration. Add a Case to CASES to add a test."""

    op: MathOperation
    spec: StimuliSpec
    fmt: InputOutputFormat = BF16
    expect_pass: bool = True
    name: Optional[str] = None
    approx_mode: ApproximationMode = ApproximationMode.No
    # Advanced overrides — defaults are derived from `fmt`.
    dest_acc: Optional[DestAccumulation] = None
    dest_sync: DestSync = DestSync.Half
    implied_math_format: ImpliedMathFormat = ImpliedMathFormat.Yes
    input_dimensions: Optional[List[int]] = None
    extra_undefined_ranges: Optional[List[Tuple[float, float]]] = None
    # ulp_sweep only: sweep a range too large for one run in batches of this
    # many tiles (offset walks the range). None = Dest capacity.
    batch_tiles: Optional[int] = None

    @property
    def test_id(self) -> str:
        if self.name:
            return self.name
        short = FMT_SHORT.get(self.fmt.output_format, self.fmt.output_format.name)
        approx = "-approx" if self.approx_mode == ApproximationMode.Yes else ""
        return f"{self.op.name}-{short}{approx}"


# ---------------------------------------------------------------------------
# CASES — add rows here.
# ---------------------------------------------------------------------------
# The eight nonlinear ops of the functional suite and the input domain each is
# swept over. Every op is run in bf16, fp16 and fp32 (in == out).
_OP_DOMAINS = {
    # -100 crosses the fp32 exp underflow (x < ~-87.3 -> 0); 75 overflows fp16
    # (exp(x) > 65504 for x > ~11.09), so the fp16 case reports those as inf.
    MathOperation.Exp: StimuliSpec.ramp(low=-100.0, high=75.0),
    MathOperation.Gelu: StimuliSpec.ramp(low=-9.0, high=9.0),
    MathOperation.Relu: StimuliSpec.ramp(low=-5.0, high=5.0),
    # 1/x sampled on both sides of 0 but never on the singularity at 0.
    MathOperation.Reciprocal: StimuliSpec.uniform(
        intervals=[(-100.0, -0.01), (0.01, 100.0)]
    ),
    MathOperation.Sqrt: StimuliSpec.ramp(low=0.0, high=100.0),
    MathOperation.Tanh: StimuliSpec.ramp(low=-5.0, high=5.0),
    MathOperation.Sigmoid: StimuliSpec.ramp(low=-8.0, high=8.0),
    MathOperation.Silu: StimuliSpec.ramp(low=-5.0, high=5.0),
}

# fp32 runs on the fp32 Dest route with 4 tiles (4096 points, the fp32 Dest
# capacity at DestSync.Half) so its fine grid is sampled more densely than the
# single-tile default that is plenty for the 16-bit formats.
_FP32_INPUT_DIMS = [32, 32 * 4]


def _approx_modes(op: MathOperation):
    """Both kernels for an op that has a LUT (approx) path, else exact only.

    The LUT cases are the "-approx" ids. Note that for a 16-bit Dest the exp
    kernel takes the LUT path in either mode, so Exp-bf16 and Exp-bf16-approx
    are the same run; only the fp32 Dest distinguishes them.
    """
    if op in QUASAR_APPROX_CAPABLE_OPS:
        return (ApproximationMode.No, ApproximationMode.Yes)
    return (ApproximationMode.No,)


CASES = [
    Case(
        op=op,
        spec=spec,
        fmt=fmt,
        approx_mode=approx,
        input_dimensions=(
            _FP32_INPUT_DIMS if fmt.output_format == DataFormat.Float32 else None
        ),
    )
    for op, spec in _OP_DOMAINS.items()
    for fmt in (BF16, FP16, FP32)
    for approx in _approx_modes(op)
] + [
    # exhaustive demo: every bf16 value in [0.01, 10] through the approx kernel
    # (~1.3k values, auto-sized to whole tiles: a single 2-tile run, no batching).
    Case(
        op=MathOperation.Reciprocal,
        spec=StimuliSpec.ulp_sweep(low=0.01, high=10.0),
        approx_mode=ApproximationMode.Yes,
        name="Reciprocal-bf16-approx-exhaustive",
    ),
    # batched fp32 demo: every fp32 value in [1.0, 1.002] (~16.8k values, 17
    # tiles) swept in 4-tile batches — the fp32 Dest capacity — and joined.
    # Exercises the batching loop on the simulator in a few runs. The WH/BH
    # file's full-octave [1.0, 2.0] sweep (2^23 values) would be 2048 runs of
    # 4 tiles here — about 25 minutes on the simulator and over
    # _MAX_SWEEP_BATCHES, so run_case rejects it.
    Case(
        op=MathOperation.Reciprocal,
        spec=StimuliSpec.ulp_sweep(low=1.0, high=1.002),
        fmt=FP32,
        name="Reciprocal-fp32-exhaustive-batched",
    ),
    # Diagnostic-only example (uncomment to explore a known-inaccurate op without failing the run):
    # Case(op=MathOperation.Gelu, spec=StimuliSpec.ramp(low=-5.0, high=5.0),
    #      approx_mode=ApproximationMode.Yes, expect_pass=False),
]


# One 32x32 tile = 1024 elements.
_TILE_ELEMENTS = TILE_DIMENSIONS[0] * TILE_DIMENSIONS[1]

_MAX_SWEEP_BATCHES = 512


def _dest_capacity_tiles(dest_sync: DestSync, dest_acc: DestAccumulation) -> int:
    """Tiles one Quasar kernel run may hold in Dest (all tiles share one section)."""
    return DEST_SYNC_TILE_LIMITS[dest_sync] // (
        2 if dest_acc == DestAccumulation.Yes else 1
    )


def _ulp_sweep_dims(
    stimuli_format: DataFormat,
    low: float,
    high: float,
    capacity_tiles: int,
) -> List[int]:
    """input_dimensions ([rows, cols]) with enough tiles for every value in
    [low, high], capped at the Dest capacity (batching handles the rest)."""
    n = int(_enumerate_representable(stimuli_format, low, high).numel())
    tiles = min(capacity_tiles, max(1, math.ceil(n / _TILE_ELEMENTS)))
    return [TILE_DIMENSIONS[0], TILE_DIMENSIONS[1] * tiles]


def _resolve_variant(case: Case, dest_acc: DestAccumulation) -> QuasarSfpuVariant:
    """The Quasar data route for this Case; fails loudly if there is none."""
    variant = resolve_quasar_sfpu_variant(case.op, case.fmt, dest_acc)
    if variant is None:
        raise ValueError(
            f"{case.test_id}: Quasar cannot execute "
            f"{case.fmt.input_format.name} -> {case.fmt.output_format.name} "
            f"with dest_acc={dest_acc.name}"
        )
    return variant


def run_case(case: Case) -> bool:
    """Run one Case end-to-end: stimuli -> golden -> Quasar kernel -> plot + stats.

    Returns whether the hardware result matched golden (passed_test) and writes
    _plot_output/qsr/sfpu_<id>.png.
    """
    formats = case.fmt
    mathop = case.op
    spec = case.spec

    if (
        case.approx_mode == ApproximationMode.Yes
        and mathop not in QUASAR_APPROX_CAPABLE_OPS
    ):
        raise ValueError(
            f"{case.test_id}: {mathop.name} has no approximate kernel on Quasar; "
            "use ApproximationMode.No"
        )

    # Format-derived default: a 32-bit output needs the fp32 Dest. The route
    # itself (and therefore unpack_to_dest) comes from the resolver, not from
    # the WH/BH "32-bit input + dest_acc" rule.
    dest_acc = case.dest_acc
    if dest_acc is None:
        dest_acc = (
            DestAccumulation.Yes
            if formats.output_format.is_32_bit() or formats.input_format.is_32_bit()
            else DestAccumulation.No
        )
    variant = _resolve_variant(case, dest_acc)
    unpack_to_dest = variant.unpack_to_dest
    capacity_tiles = _dest_capacity_tiles(case.dest_sync, dest_acc)
    logger.info(
        "[{}] route: unpack_to_dest={} unpack_dst={} sfpu={}/{} pack_src={} "
        "dest_acc={} (Dest capacity {} tiles)",
        case.test_id,
        unpack_to_dest,
        variant.unpack_dst.name,
        variant.sfpu_src.name,
        variant.sfpu_dst.name,
        variant.pack_src.name,
        dest_acc.name,
        capacity_tiles,
    )

    plot_path = str(PLOT_OUTPUT_DIR / f"sfpu_{case.test_id}.png")
    torch_format = format_dict[formats.output_format]
    generate_golden = get_golden_generator(UnarySFPUGolden)

    def _run_chunk(chunk_spec, dims):
        """Run one chunk on the device and return (input, golden, hardware result).

        Builds the stimuli, computes the torch golden, runs the kernel, reads the
        result back. Called once for a normal run, or once per batch when batching.
        """
        src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
            stimuli_format_A=formats.input_format,
            input_dimensions_A=dims,
            spec_A=chunk_spec,
            stimuli_format_B=formats.input_format,
            input_dimensions_B=dims,
        )
        if tile_cnt_A > capacity_tiles:
            raise ValueError(
                f"{case.test_id}: {tile_cnt_A} tiles exceed the Quasar Dest "
                f"capacity of {capacity_tiles} tiles ({case.dest_sync.name}, "
                f"dest_acc={dest_acc.name}); shrink input_dimensions"
            )
        golden = generate_golden(
            mathop, src_A, formats.output_format, dest_acc, formats.input_format, dims
        )
        # Same template / runtime list as quasar/test_eltwise_unary_sfpu_quasar.py
        # for a non-typecast op (create_test_or_perf_config adds PERF_RUN_TYPE
        # there; here it is spelled out).
        configuration = TestConfig(
            QUASAR_KERNEL,
            formats,
            templates=[
                PERF_RUN_TYPE(PerfRunType.L1_TO_L1),
                MATH_OP(mathop=mathop),
                APPROX_MODE(case.approx_mode),
                IMPLIED_MATH_FORMAT(case.implied_math_format),
                DATA_COPY_TYPE(DataCopyType.A2D),
                UNPACKER_ENGINE_SEL(
                    UnpackerEngine.UnpDest if unpack_to_dest else UnpackerEngine.UnpA
                ),
                DEST_SYNC(case.dest_sync),
                TYPECAST_FORMATS(),
            ],
            runtimes=[
                TILE_COUNT(tile_cnt_A),
                NUM_FACES(MAX_NUM_FACES),
                TEST_FACE_DIMS(),
                DEST_INDEX(0),
                LOOP_FACTOR(1),
            ],
            variant_stimuli=StimuliConfig(
                src_A,
                formats.input_format,
                src_B,
                formats.input_format,
                formats.output_format,
                tile_count_A=tile_cnt_A,
                tile_count_B=tile_cnt_B,
                tile_count_res=tile_cnt_A,
                num_faces=MAX_NUM_FACES,
            ),
            dest_acc=dest_acc,
            unpack_to_dest=unpack_to_dest,
        )
        variant.apply_formats(configuration.formats_config)
        res = torch.tensor(configuration.run().result, dtype=torch_format)
        return src_A, golden, res

    # Pick batching for a ulp_sweep. Use batch_tiles if set; otherwise batch at
    # Dest capacity when the range does not fit one run, so it can't silently
    # truncate. If input_dimensions is set, use that size as-is (no batching).
    is_ulp = spec.distribution == DistributionKind.ULP_SWEEP
    batch_tiles = case.batch_tiles
    if is_ulp:
        if batch_tiles is not None and not 1 <= batch_tiles <= capacity_tiles:
            raise ValueError(
                f"{case.test_id}: batch_tiles={batch_tiles} must be between 1 and "
                f"the Quasar Dest capacity of {capacity_tiles} tiles"
            )
        total = ulp_sweep_value_count(formats.input_format, spec.low, spec.high)
        # Reject a range that would take too many device runs. Skipped when
        # input_dimensions is set — that is a single, quick run.
        if case.input_dimensions is None:
            run_tiles = batch_tiles if batch_tiles is not None else capacity_tiles
            run_values = run_tiles * _TILE_ELEMENTS
            num_runs = math.ceil(total / run_values)
            if num_runs > _MAX_SWEEP_BATCHES:
                raise ValueError(
                    f"{case.test_id}: ulp_sweep [{spec.low}, {spec.high}] has "
                    f"{total:,} values = {num_runs:,} runs of {run_tiles} tiles, "
                    f"over the {_MAX_SWEEP_BATCHES}-run limit — narrow the range "
                    f"(at most {run_values * _MAX_SWEEP_BATCHES:,} values)."
                )
        if (
            batch_tiles is None
            and case.input_dimensions is None
            and total > capacity_tiles * _TILE_ELEMENTS
        ):
            batch_tiles = capacity_tiles
            logger.info(
                "ulp_sweep [{}, {}] has {} values — too many for one run, "
                "auto-batching at {} tiles (Dest capacity)",
                spec.low,
                spec.high,
                total,
                batch_tiles,
            )

    if is_ulp and batch_tiles is not None:
        # Sweep a range too large for one run in fixed-size batches (offset walks
        # the range) and join them. Every batch is the same size, so the kernel is
        # compiled once and reused.
        batch_values = batch_tiles * _TILE_ELEMENTS
        batch_dims = [TILE_DIMENSIONS[0], TILE_DIMENSIONS[1] * batch_tiles]
        num_batches = max(1, math.ceil(total / batch_values))
        logger.info(
            "ulp_sweep batched: {} values -> {} batch(es) of {} tiles ({} values each)",
            total,
            num_batches,
            batch_tiles,
            batch_values,
        )
        src_parts, golden_parts, res_parts = [], [], []
        start = time.perf_counter()
        for k in range(num_batches):
            s, g, r = _run_chunk(replace(spec, offset=k * batch_values), batch_dims)
            # Full batches are all real; the last is zero-padded at the tail, so
            # keep only its real values — padding is not test data.
            real = min(batch_values, total - k * batch_values)
            src_parts.append(s[:real])
            golden_parts.append(g[:real])
            res_parts.append(r[:real])
            logger.info(
                "  batch {}/{} done ({:.1f}s elapsed)",
                k + 1,
                num_batches,
                time.perf_counter() - start,
            )
        src_A = torch.cat(src_parts)
        golden_tensor = torch.cat(golden_parts)
        res_tensor = torch.cat(res_parts)
        logger.info(
            "ulp_sweep batched: {} values swept in {:.1f}s",
            src_A.numel(),
            time.perf_counter() - start,
        )
    else:
        if case.input_dimensions is not None:
            input_dimensions = case.input_dimensions
        elif is_ulp:
            input_dimensions = _ulp_sweep_dims(
                formats.input_format, spec.low, spec.high, capacity_tiles
            )
        else:
            input_dimensions = [32, 32]
        src_A, golden_tensor, res_tensor = _run_chunk(spec, input_dimensions)
        if is_ulp:
            # Drop the trailing zero-padding a ulp_sweep adds when the range has
            # fewer values than the tensor holds — it is not test data.
            real = min(total, input_dimensions[0] * input_dimensions[1])
            src_A = src_A[:real]
            golden_tensor = golden_tensor[:real]
            res_tensor = res_tensor[:real]

    sort_idx = torch.argsort(src_A.to(torch.float32))
    x = src_A.to(torch.float32)[sort_idx].numpy()
    y_golden = golden_tensor.to(torch.float32)[sort_idx].numpy()
    y_hw = res_tensor.to(torch.float32)[sort_idx].numpy()

    allowed_intervals = allowed_intervals_for(spec, x)
    # extra_undefined_ranges, when supplied (even as an empty list), fully
    # overrides the registry — letting callers inject custom asymptote bands or
    # suppress the registry's red shading entirely.
    if case.extra_undefined_ranges is not None:
        undefined_ranges = list(case.extra_undefined_ranges)
    else:
        undefined_ranges = list(
            _SFPU_UNDEFINED_RANGES.get(mathop, {}).get(Operand.A, [])
        )

    # ULP/eps spacing in plot_and_print is taken from the format the compared
    # values live in: golden and hw are produced in output_format.
    title_suffix = arch_title_suffix(ChipArchitecture.QUASAR)
    if case.approx_mode == ApproximationMode.Yes:
        title_suffix += " approx"
    route = "unpack→Dest" if unpack_to_dest else "SrcA→FPU→Dest"
    param_line = (
        f"{case.test_id}  |  in {formats.input_format.name} → Dest "
        f"{variant.unpack_dst.name} ({route}) → SFPU {variant.sfpu_src.name}/"
        f"{variant.sfpu_dst.name} → pack {variant.pack_src.name} → out "
        f"{formats.output_format.name}\n"
        f"dest_acc={dest_acc.name}  |  approx={case.approx_mode.name}  |  "
        f"{case.dest_sync.name}  |  implied_math_format={case.implied_math_format.name}"
        f"  |  {spec.distribution.name.lower()}, {x.size} points"
    )
    plot_and_print(
        mathop,
        formats.output_format,
        x,
        y_golden,
        y_hw,
        plot_path,
        title_suffix=title_suffix,
        allowed_intervals=allowed_intervals,
        undefined_ranges=undefined_ranges,
        param_line=param_line,
    )

    test_passed = passed_test(golden_tensor, res_tensor, formats.output_format)
    logger.info("passed_test: {}", test_passed)

    # Bit-distance ULP measurement — reinterpret each result in its OWN format's
    # integer width and take |golden_bits - hw_bits|; the max across finite
    # samples is the worst-case ULP error.
    torch_out = format_dict[formats.output_format]
    if formats.output_format == DataFormat.Float32:
        torch_int, np_int = torch.int32, np.int32
    else:  # Float16_b / Float16 — 2-byte formats reinterpreted as int16
        torch_int, np_int = torch.int16, np.int16
    golden_native = golden_tensor.to(torch_out).contiguous()
    hw_native = res_tensor.to(torch_out).contiguous()
    valid_mask = (
        torch.isfinite(golden_native) & torch.isfinite(hw_native) & (golden_native != 0)
    ).numpy()
    if valid_mask.any():
        gb = golden_native.view(torch_int).numpy()[valid_mask].astype(np.int64)
        rb = hw_native.view(torch_int).numpy()[valid_mask].astype(np.int64)
        int_min = np.iinfo(np_int).min
        gb = np.where(gb < 0, int_min - gb, gb)
        rb = np.where(rb < 0, int_min - rb, rb)
        max_ulp = int(np.abs(gb - rb).max())
        logger.info(
            "[{}] max ULP across {} finite nonzero-golden samples: {}",
            mathop.name,
            int(valid_mask.sum()),
            max_ulp,
        )
    else:
        logger.warning("[{}] no finite samples for ULP measurement", mathop.name)

    return test_passed


@pytest.mark.quasar
@pytest.mark.accuracy
@pytest.mark.parametrize("case", CASES, ids=[c.test_id for c in CASES])
def test_sfpu_stress_quasar(case: Case):
    passed = run_case(case)
    if case.expect_pass:
        assert passed, f"{case.test_id}: result did not match golden"
