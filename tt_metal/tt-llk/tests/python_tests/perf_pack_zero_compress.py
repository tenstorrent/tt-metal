# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""What a compression-enabled PACR costs on Blackhole, in cycles.

WHY
---
``sources/pack_zero_compress_test.cpp`` proved on silicon that the packer can do
threshold + compaction with ZERO SFPU instructions, and that the compressed
stream decodes back bit-exactly. It never priced it. The whole
packer-resident-selection thesis rests on the inferred claim that a
compression-enabled pack costs the same as a plain pack -- if that holds, the
pack side runs at roughly 1 cycle per 32-element vector, 2.8x under
``_topk_xl_merge_``'s measured 2.844 cyc/vector. This file measures it.

WHAT IS MEASURED
----------------
PACK_ISOLATE. The work is on the pack thread: unpack and math fill Dest once in
the INIT zone and do nothing inside TILE_LOOP, which issues PACK_ITER_COUNT tile
packs from the same Dest tile.

Cycles per 32-element vector comes from a two-point slope over PACK_ITER_COUNT:

    cyc_per_vector = (mean@hi - mean@lo) / (hi - lo) / 32

One 32x32 tile is 1024 datums = 32 vectors of 32. The subtraction cancels the
~30-cycle START_PERF_MEASURE marker pair and every one-time cost inside the zone
(packer hw-configure, MOP programming, the compression config writes, the
wait-for-math-done). What survives is the marginal cost of one more tile pack --
exactly what a tile loop pays. LOOP_FACTOR / TILE_COUNT are deliberately NOT
passed, so ``postprocess_tile_loop``'s divisor defaults to 1 and the .post.csv
carries raw zone cycles; the arithmetic is done downstream, not by the harness.

CONTROLS
--------
The measurement is a DELTA, so the control is the same kernel, same Dest, same
16 PACRs, same L1 destination, differing ONLY in the two config-register writes
that enable compression. Everything the absolute number depends on (MOP issue
rate, per-PACR fixed cost, RISC loop overhead, L1 arbitration) is common-mode and
cancels.

Two independent tripwires make an accidental no-op detectable:

  * density 1024 with compress on vs off. With no zeroes to elide, compression
    cannot shrink anything -- it GROWS the tile from 2048 B to 2624 B (32
    four-bit counters per 32 datums, plus the row-start index array). So this
    pair separates "cost tracks bytes written" from "cost tracks datums read".
  * the kernel dumps ``PackerTileSize`` -- the packer's own report of the bytes it
    emitted for the last tile -- into buffer_Res outside every timed zone. The
    expected value of every row is known independently from the correctness probe
    (topk32 -> 384 B, dense+compress -> 2624 B, dense -> 2048 B), so a build where
    the compression enable silently failed is caught by data, not by eyeballing
    timings.

ARMS
----
  plain   Dest already holds the sparse pattern (the zeroes are in the data).
          Prices compression alone, as a function of how many bytes come out.
  relu    Dest holds a DENSE ladder of distinct values and the packer's
          MIN_THRESHOLD_RELU zeroes the sub-threshold ones: the full
          filter+compact-in-one-PACR configuration.
          CAVEAT, and it is a big one: MIN_THRESHOLD_RELU cannot express a
          negative threshold (WormholeB0/.../Packers/ReLU.md:41 -- a signbit
          threshold is explicitly UndefinedBehavior), and it reinterprets the
          datum as a float, so it cannot be combined with a fused
          value|index INT32 sort key. This arm is valid ONLY for non-negative,
          index-free data.
  fused32 32-bit datums (Int32 pack format), Dest holding the fused
          [bf16 value | u16 index] word that ckernel_sfpu_topk_xl.h uses, and 0
          for non-survivors. This is the arrangement that actually competes: the
          SFPU does the compare (1.003 cyc/vector, already measured; SFPGT
          handles negative thresholds, which the packer RELU cannot), the packer
          elides the zeroed words, and survivors carry their own indices so no
          run-length position decode is needed. It only works if the packer's
          zero test operates on the raw 32-bit datum -- which no doc states.
          test_pack_compress_int32.py checks correctness; this arm prices it.
"""

from dataclasses import dataclass

import pytest
import torch
from conftest import blackhole_only
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import PackGolden
from helpers.llk_params import DestAccumulation, PackerReluType, PerfRunType
from helpers.param_config import parametrize
from helpers.perf.core import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import (
    NUM_FACES,
    RELU_CONFIG,
    TemplateParameter,
    generate_input_dim,
)

TILE_DATUMS = 1024

# 64 B reserved before the data stream for the row-start-index array. The rss sweep
# in test_pack_zero_compress.py showed 4 (in 16 B units) is enough for 16 compression
# rows; it is a pure address offset, not a per-datum cost.
ROW_START_SECTION_SIZE = 4

# Two-point slope. 16 tile packs is already ~15x the marker pair at the cheapest
# plausible rate, and 64 keeps a 5-repeat run short on a shared device.
_ITER_COUNTS = [16, 64]

_ARMS = ["plain", "relu", "fused32"]

_BF16 = InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b)
_INT32 = InputOutputFormat(DataFormat.Int32, DataFormat.Int32)


@dataclass
class PACK_COMPRESS(TemplateParameter):
    """Compression knobs, plus the arm/density labels.

    arm and density are emitted into build.h even though the kernel never reads
    them: they have to be part of the variant hash (otherwise two densities share
    one ELF and one report row) and they have to be report columns (otherwise the
    CSV cannot tell the rows apart).
    """

    compress_en: bool = False
    row_start_section_size: int = 0
    downsample_mask: int = 0
    pack_iter_count: int = 16
    cycle_output: bool = True
    arm: str = "plain"
    density: int = 0

    def convert_to_cpp(self) -> str:
        return "\n".join(
            [
                f"constexpr bool COMPRESS_EN = {str(self.compress_en).lower()};",
                f"constexpr std::uint32_t ROW_START_SECTION_SIZE = {self.row_start_section_size};",
                f"constexpr std::uint32_t DOWNSAMPLE_MASK = {self.downsample_mask};",
                f"constexpr std::uint32_t PACK_ITER_COUNT = {self.pack_iter_count};",
                f"constexpr bool CYCLE_OUTPUT = {str(self.cycle_output).lower()};",
                f"constexpr std::uint32_t STIMULUS_DENSITY = {self.density}; // label only",
                f"// arm = {self.arm}",
            ]
        )


def _scatter(num_survivors, seed=7):
    """Survivor positions, scattered -- the realistic top-k case, and the same seed
    and selection the correctness probe's ``topk32`` pattern used, so the emitted
    byte counts here are comparable to the numbers already on record."""
    if num_survivors <= 0:
        return []
    if num_survivors >= TILE_DATUMS:
        return list(range(TILE_DATUMS))
    g = torch.Generator().manual_seed(seed)
    return sorted(torch.randperm(TILE_DATUMS, generator=g)[:num_survivors].tolist())


def _bf16_ladder(n=TILE_DATUMS):
    """n distinct, exactly-representable bfloat16 values in [1, 256)."""
    vals = []
    for e in range(8):
        for m in range(128):
            vals.append((2.0**e) * (1.0 + m / 128.0))
    return vals[:n]


def _stimulus(arm, density):
    """(src_A, relu_config) for one arm/density."""
    if arm == "plain":
        t = torch.zeros(TILE_DATUMS, dtype=torch.bfloat16)
        for k, i in enumerate(_scatter(density)):
            t[i] = float((k % 250) + 1)
        return t, 0

    if arm == "relu":
        # Dense distinct values; the packer does the zeroing. The threshold is picked
        # so exactly `density` datums survive.
        vals = _bf16_ladder()
        g = torch.Generator().manual_seed(11)
        perm = torch.randperm(TILE_DATUMS, generator=g).tolist()
        t = torch.tensor(
            [vals[perm[i]] for i in range(TILE_DATUMS)], dtype=torch.bfloat16
        )
        threshold = vals[TILE_DATUMS - density]
        relu = PackGolden.generate_relu_config(
            PackerReluType.MinThresholdRelu, threshold, DataFormat.Float16_b
        )
        return t, relu

    # fused32: [bf16 value (high 16) | u16 index (low 16)], 0 for non-survivors.
    t = torch.zeros(TILE_DATUMS, dtype=torch.int32)
    for k, i in enumerate(_scatter(density)):
        val_bits = int(
            torch.tensor([float((k % 250) + 1)], dtype=torch.bfloat16)
            .view(torch.uint16)
            .item()
        )
        # index+1, so a survivor at position 0 is never an all-zero word (which the
        # packer would elide as a hole).
        t[i] = (val_bits << 16) | ((i + 1) & 0xFFFF)
    return t, 0


def _formats(arm):
    return [_INT32] if arm == "fused32" else [_BF16]


def _densities(arm):
    # 0 and 1024 are the two ends that make the byte-count tripwire work; 32 is the
    # top-k case that matters; 128 and 512 fill the middle so a byte-count dependence
    # shows up as a trend rather than as two points. The relu arm needs a strictly
    # positive threshold, so density 0 and 1024 are not expressible there.
    if arm == "relu":
        return [32, 128]
    return [0, 32, 128, 512, 1024]


@pytest.mark.perf
@blackhole_only
@parametrize(
    arm=_ARMS,
    formats=_formats,
    density=_densities,
    compress=[False, True],
    iter_count=_ITER_COUNTS,
)
def test_perf_pack_zero_compress(
    perf_report, arm, formats, density, compress, iter_count
):
    src_A, relu_config = _stimulus(arm, density)
    is_32bit = arm == "fused32"
    src_B = torch.zeros_like(src_A)

    configuration = PerfConfig(
        "sources/pack_zero_compress_perf.cpp",
        formats,
        run_types=[PerfRunType.PACK_ISOLATE],
        templates=[
            generate_input_dim([32, 32], [32, 32]),
            PACK_COMPRESS(
                compress_en=compress,
                row_start_section_size=ROW_START_SECTION_SIZE if compress else 0,
                downsample_mask=0,
                pack_iter_count=iter_count,
                cycle_output=True,
                arm=arm,
                density=density,
            ),
        ],
        runtimes=[
            RELU_CONFIG(relu_config),
            NUM_FACES(num_faces=4),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=1,
            tile_count_B=1,
            tile_count_res=1,
        ),
        dest_acc=DestAccumulation.Yes if is_32bit else DestAccumulation.No,
        unpack_to_dest=is_32bit,
    )

    configuration.run(perf_report, run_count=5)
