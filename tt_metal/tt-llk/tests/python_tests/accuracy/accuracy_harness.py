# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math
import shutil
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from helpers.accuracy_metrics import compute_pointwise_metrics
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import (
    TILE_DIMENSIONS,
    UnarySFPUGolden,
    get_golden_generator,
)
from helpers.llk_params import (
    ApproximationMode,
    BlocksCalculationAlgorithm,
    DestAccumulation,
    FastMode,
    MathOperation,
    format_dict,
)
from helpers.param_config import get_num_blocks_and_num_tiles_in_block
from helpers.sfpu_domains import for_op
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import DistributionKind, StimuliSpec, generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    APPROX_MODE,
    CLAMP_NEGATIVE,
    FAST_MODE,
    MATH_OP,
    NUM_BLOCKS,
    NUM_TILES_IN_BLOCK,
    TILE_COUNT,
    DestSync,
    generate_input_dim,
)

_THIS_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = _THIS_DIR / "_csv_output"
SHARD_DIR = OUTPUT_DIR / "_shards"

# ── CSV schema ──────────────────────────
CSV_COLUMNS: List[str] = [
    "op",
    "input_format",
    "output_format",
    "chip_arch",
    "distribution",
    "intervals",
    "seed",
    "sample_index",
    "x",
    "golden",
    "hw",
    "approx_mode",
    "fast_mode",
    "dest_acc",
    "signed_error",
    "rel_error",
    "signed_ulp_error",
    "is_finite_hw",
    "is_finite_golden",
]

# Float precision on disk: bf16/fp16 outputs carry <=11 mantissa bits, so 7
# significant digits is lossless-enough and far shorter than full float64 repr.
FLOAT_FORMAT = "%.7g"

_ARCH_ABBR = {"wormhole": "wh", "blackhole": "bh", "quasar": "qsr"}
_FMT_ABBR = {
    "Float32": "fp32",
    "Float16": "fp16",
    "Float16_b": "bf16",
    "Bfp8_b": "bfp8",
    "Bfp4_b": "bfp4",
    "Bfp2_b": "bfp2",
    "Tf32": "tf32",
}

# How rows are ordered inside each final per-op CSV.
MERGE_SORT_COLS = [
    "op",
    "input_format",
    "output_format",
    "approx_mode",
    "fast_mode",
    "dest_acc",
    "x",
]

# How many input points each op's curve is sampled at.
DEFAULT_SWEEP_POINTS = 2048


def variant_name(
    op: MathOperation,
    in_fmt: DataFormat,
    out_fmt: DataFormat,
    approx: ApproximationMode,
    fast: FastMode,
    dest: DestAccumulation,
) -> str:
    """Stable shard stem / `variant_name` CSV value for one matrix cell."""
    return (
        f"{op.name.lower()}__{in_fmt.name}_{out_fmt.name}__"
        f"approx{int(approx == ApproximationMode.Yes)}_"
        f"fast{int(fast == FastMode.Yes)}_"
        f"dest{int(dest == DestAccumulation.Yes)}"
    )


def build_sweep_spec(
    op: MathOperation,
    in_fmt: DataFormat,
    distribution: DistributionKind = DistributionKind.RAMP,
    seed: int = None,
) -> StimuliSpec:
    """Build the input sweep over the op's defined domain.

    for_op already returns the op's safe per-format domain (within the region
    where the op is defined); this just applies the chosen *distribution*.

    *distribution* defaults to RAMP (deterministic, sorted — clean curves).
    Pass *seed* for reproducible random distributions.
    """
    spec = for_op(op, in_fmt, distribution_a=distribution).spec_A
    if seed is not None:
        spec.seed = seed
    return spec


def sweep_input_dimensions(points: int = DEFAULT_SWEEP_POINTS) -> List[int]:
    """Tensor shape [32, 32*K] that holds at least *points* sweep values.

    A Tensix tile is 32x32 = 1024 elements, and test inputs must be whole tiles
    (each dim a multiple of 32). The sweep is just a flat list of values, so we
    keep the height at one tile (32 rows) and grow only the width in tile-sized
    steps: K = ceil(points / 1024) tiles laid in a row -> [32, 32*K].
    """
    k = max(1, math.ceil(points / (32 * 32)))
    return [32, 32 * k]


def rows_dataframe(
    *,
    op_name: str,
    in_fmt: str,
    out_fmt: str,
    chip_arch: str,
    distribution: str,
    intervals: str,
    seed: str,
    approx: str,
    fast: str,
    dest: str,
    x,
    golden,
    hw,
    out_fmt_enum_name: str,
) -> "pd.DataFrame":
    """Build the CSV rows for one variant (one op + format + config).

    Takes the raw sweep arrays (x, golden, hw), sorts them by x, computes the
    error columns (via accuracy_metrics), and returns a DataFrame in CSV_COLUMNS
    order — ready to hand to write_shard.
    """
    x = np.asarray(x, dtype=np.float64)
    golden = np.asarray(golden, dtype=np.float64)
    hw = np.asarray(hw, dtype=np.float64)

    order = np.argsort(x, kind="stable")
    x, golden, hw = x[order], golden[order], hw[order]

    out_fmt_enum = DataFormat[out_fmt_enum_name]
    m = compute_pointwise_metrics(x, golden, hw, out_fmt_enum)

    n = len(x)
    df = pd.DataFrame(
        {
            "op": op_name,
            "input_format": _FMT_ABBR.get(in_fmt, in_fmt),
            "output_format": _FMT_ABBR.get(out_fmt, out_fmt),
            "chip_arch": _ARCH_ABBR.get(chip_arch, chip_arch),
            "distribution": distribution,
            "intervals": intervals,
            "seed": seed,
            "sample_index": np.arange(n, dtype=np.int64),
            "x": x,
            "golden": golden,
            "hw": hw,
            "approx_mode": approx,
            "fast_mode": fast,
            "dest_acc": dest,
            "signed_error": m["signed_error"],
            "rel_error": m["rel_error"],
            "signed_ulp_error": m["signed_ulp_error"],
            "is_finite_hw": np.where(m["is_finite_hw"], "T", "F"),
            "is_finite_golden": np.where(m["is_finite_golden"], "T", "F"),
        }
    )
    return df[CSV_COLUMNS]


def write_shard(df: "pd.DataFrame", variant: str) -> Path:
    """Write one variant's DataFrame to SHARD_DIR/{variant}.csv.

    *variant* is the unique shard stem (it is no longer a CSV column, so it is
    passed explicitly).
    """
    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    shard_path = SHARD_DIR / f"{variant}.csv"
    df.to_csv(shard_path, index=False, float_format=FLOAT_FORMAT)
    return shard_path


def merge_shards() -> List[Path]:
    """Merge current-run shards into one sorted CSV per op (overwrite).

    Reads only the shards present in SHARD_DIR, groups by op, sorts by
    MERGE_SORT_COLS, and overwrites OUTPUT_DIR/{op}.csv from scratch. Never
    reads a pre-existing final CSV.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    shard_files = sorted(SHARD_DIR.glob("*.csv")) if SHARD_DIR.exists() else []
    if not shard_files:
        return []

    frames = [pd.read_csv(f) for f in shard_files]
    combined = pd.concat(frames, ignore_index=True)

    written: List[Path] = []
    for op_name, group in combined.groupby("op", sort=True):
        ordered = group.sort_values(MERGE_SORT_COLS, kind="stable")
        out_path = OUTPUT_DIR / f"{op_name}.csv"
        ordered.to_csv(out_path, index=False, float_format=FLOAT_FORMAT)
        written.append(out_path)
    return written


def clear_shards() -> None:
    """Remove the shard dir so a run starts with no stale shards."""
    if SHARD_DIR.exists():
        shutil.rmtree(SHARD_DIR)
    SHARD_DIR.mkdir(parents=True, exist_ok=True)


def run_case(
    op: MathOperation,
    formats: InputOutputFormat,
    approx_mode: ApproximationMode,
    fast_mode: FastMode,
    dest_acc: DestAccumulation,
    *,
    points: int = DEFAULT_SWEEP_POINTS,
    distribution: DistributionKind = DistributionKind.RAMP,
    seed: int = None,
) -> Path:
    """Measure one (op, format, config) on hardware and write its shard CSV.

    The full pipeline for one variant:
      1. build the input sweep for this op (deterministic RAMP by default;
         pass *distribution* for another, and *seed* for reproducible random ones),
      2. compute the torch "golden" output,
      3. compile + run the op on the device to get the "hw" output,
      4. compute per-element errors and write a shard via rows_dataframe/write_shard.

    Returns the shard path.
    """
    torch.manual_seed(0)

    spec = build_sweep_spec(op, formats.input_format, distribution, seed)
    input_dimensions = sweep_input_dimensions(points)

    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        spec_A=spec,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )

    generate_golden = get_golden_generator(UnarySFPUGolden)
    golden_tensor = generate_golden(
        op,
        src_A,
        formats.output_format,
        dest_acc,
        formats.input_format,
        input_dimensions,
    )

    num_blocks, num_tiles_in_block = get_num_blocks_and_num_tiles_in_block(
        DestSync.Half,
        dest_acc,
        formats,
        input_dimensions,
        TILE_DIMENSIONS,
        BlocksCalculationAlgorithm.Standard,
    )

    configuration = TestConfig(
        "sources/eltwise_unary_sfpu_test.cpp",
        formats,
        templates=[
            generate_input_dim(input_dimensions, input_dimensions),
            APPROX_MODE(approx_mode),
            FAST_MODE(fast_mode),
            CLAMP_NEGATIVE(True),
            MATH_OP(mathop=op),
        ],
        runtimes=[
            TILE_COUNT(tile_cnt_A),
            NUM_BLOCKS(num_blocks),
            NUM_TILES_IN_BLOCK(num_tiles_in_block),
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
        ),
        dest_acc=dest_acc,
        unpack_to_dest=(
            formats.input_format.is_32_bit() and dest_acc == DestAccumulation.Yes
        ),
    )

    res_from_L1 = configuration.run().result
    assert len(res_from_L1) == len(golden_tensor), (
        f"{op.name}: result length {len(res_from_L1)} != golden "
        f"{len(golden_tensor)}"
    )

    torch_format = format_dict[formats.output_format]
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    x_np = src_A.to(torch.float32).numpy()
    golden_np = golden_tensor.to(torch.float32).numpy()
    hw_np = res_tensor.to(torch.float32).numpy()

    # Sanity asserts (no threshold gating).
    assert not np.all(~np.isfinite(hw_np)), f"{op.name}: all HW outputs non-finite"
    golden_has_signal = np.any(np.isfinite(golden_np) & (golden_np != 0.0))
    if golden_has_signal:
        assert np.any(hw_np != 0.0), f"{op.name}: HW returned all zeros (no-op?)"

    domain = spec.intervals if spec.intervals is not None else [(spec.low, spec.high)]
    intervals_str = str(domain)
    vname = variant_name(
        op,
        formats.input_format,
        formats.output_format,
        approx_mode,
        fast_mode,
        dest_acc,
    )
    df = rows_dataframe(
        op_name=op.name.lower(),
        in_fmt=formats.input_format.name,
        out_fmt=formats.output_format.name,
        chip_arch=str(TestConfig.CHIP_ARCH.name).lower(),
        distribution=(
            distribution.value
            if isinstance(distribution, DistributionKind)
            else "custom"
        ),
        intervals=intervals_str,
        seed="" if seed is None else str(seed),
        approx=str(int(approx_mode == ApproximationMode.Yes)),
        fast=str(int(fast_mode == FastMode.Yes)),
        dest=str(int(dest_acc == DestAccumulation.Yes)),
        x=x_np,
        golden=golden_np,
        hw=hw_np,
        out_fmt_enum_name=formats.output_format.name,
    )
    return write_shard(df, vname)
