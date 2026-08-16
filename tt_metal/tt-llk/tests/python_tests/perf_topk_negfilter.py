# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""What does a NEGATIVE-threshold Top-K filter cost end to end on Blackhole?

THE QUESTION
------------
The zero-SFPU Top-K path -- packer ``MIN_THRESHOLD_RELU`` doing the compare,
zero-compression doing the compaction -- costs 4.175 cyc/vector end to end
against ``_topk_xl_merge_``'s 6.930 in the same kernel. It cannot express a
negative threshold (``Packers/ReLU.md:41``); measured on silicon in
``test_topk_negfilter.py``, the packer compares against ``|Threshold|`` instead,
so signed data needs an SFPU fallback. This driver prices that fallback on the
basis that matters.

WHY L1_TO_L1 IS THE BASIS
-------------------------
PACK overlaps the unpacker but the SFPU does not -- math and unpack both drive
the Dst register file -- so SFPU work ADDS to the ~3.94 cyc/vector unpack floor
rather than hiding under it. An isolate cannot see that. L1_TO_L1 timestamps
unpack's ZONE_START against pack's ZONE_END, i.e. the whole three-thread
pipeline; MATH_ISOLATE is carried alongside as the component number.

THE ARMS
--------
    none       SFPU off -- stream + compressed pack. The floor.
    ctrlload   CONTROL: replay+MOP stream of plain SFPLOAD, 1 instr/vector.
    ctrlswap   CONTROL / tripwire: same with SFPSWAP, documented at 2 backend
               cycles. MUST be ~2.0x ctrlload under MATH_ISOLATE or the run is
               invalid.
    mask1      CALIBRATION: the published 1-macro/vector MaskStore probe. NOT a
               usable filter (it stores SFPGT's -1/0 mask and destroys the
               values) -- it is carried so the negfilter number is a difference
               of differences against a known point.
    negfilter  THE CANDIDATE: 2 macros/vector, value-preserving, bit-exact.
    xlmerge    THE COMPETITION: ``_topk_xl_merge_<512, false, true>`` on the
               math thread of this same pipeline.
    relucomp   THE REFERENCE: the zero-SFPU packer-resident path (positive
               threshold), i.e. what the win looks like for unsigned data.

READING THE NUMBER
------------------
``postprocess_tile_loop``'s divisor is pinned to 1, so the .post.csv carries RAW
zone cycles. Cycles per 32-element vector is a two-point slope over the streamed
tile count:

    cyc_per_vector = (mean@hi - mean@lo) / (hi - lo) / 32

The subtraction cancels the marker pair, the hw-configure, the replay-buffer
load, the MOP programming and the pipeline fill/drain. One 32x32 tile is 1024
datums = 32 vectors of 32.
"""

import struct
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
    LOOP_FACTOR,
    NUM_FACES,
    RELU_CONFIG,
    TILE_COUNT,
    TemplateParameter,
    generate_input_dim,
)

TILE_DATUMS = 1024

# Ring sizes for the L1 buffers. Powers of two so the kernel can index with a
# mask; 16 keeps the working set small while still being large enough that
# consecutive packs never hit the same line.
SRC_SLOTS = 16
RES_SLOTS = 16

# 64 B reserved before the data stream for the row-start index array.
ROW_START_SECTION_SIZE = 4

# Two-point slope over the streamed tile count. Both are multiples of the 4-tile
# Dest block (fp32 Dest, DstSync::SyncHalf).
_TILE_COUNTS = [16, 64]

_FORMATS = InputOutputFormat(DataFormat.Float32, DataFormat.Float32)

# How many of the 1024 datums survive. 32 is the top-k case that matters.
# Compression cost was measured FLAT in density, so this steers correctness and
# emitted size, not the rate.
_SURVIVORS = 32


@dataclass
class NF_PARAMS(TemplateParameter):
    """Compile-time knobs, plus the arm label.

    ``arm`` is emitted even though the kernel never reads it: it has to be part
    of the variant hash (otherwise two arms would share one ELF and one report
    row) and it has to be a report column (otherwise the CSV cannot tell the
    rows apart).
    """

    filter_arm: int = 0
    compress_en: bool = False
    row_start_section_size: int = 0
    downsample_mask: int = 0
    thr_bits: int = 0
    arm: str = "none"
    xl_merge_en: bool = False

    def convert_to_cpp(self) -> str:
        return "\n".join(
            [
                # A #define, not a constexpr: the kernel guards the topk_xl
                # include itself with #if, and only the preprocessor can gate an
                # #include.
                f"#define XL_MERGE_EN {1 if self.xl_merge_en else 0}",
                f"constexpr std::uint32_t FILTER_ARM = {self.filter_arm};",
                f"constexpr bool COMPRESS_EN = {str(self.compress_en).lower()};",
                f"constexpr std::uint32_t ROW_START_SECTION_SIZE = {self.row_start_section_size};",
                f"constexpr std::uint32_t DOWNSAMPLE_MASK = {self.downsample_mask};",
                f"constexpr std::uint32_t THR_BITS = {self.thr_bits}u;",
                f"constexpr std::uint32_t SRC_SLOTS = {SRC_SLOTS};",
                f"constexpr std::uint32_t RES_SLOTS = {RES_SLOTS};",
                f"// arm = {self.arm}",
            ]
        )


def bf16_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0] >> 16


def bf16_ladder(n=TILE_DATUMS):
    """n distinct, exactly-representable bfloat16 magnitudes in [1, 256)."""
    vals = []
    for e in range(8):
        for m in range(128):
            vals.append((2.0**e) * (1.0 + m / 128.0))
    return vals[:n]


def fused_tile(signed: bool):
    """One dense tile of fused FP32 sort keys plus the threshold that lets
    exactly ``_SURVIVORS`` of them through.

    Word layout is ckernel_sfpu_topk_xl.h's: [bf16 value | u16 (index+1)].

    ``signed=True`` makes every value NEGATIVE and the threshold negative, which
    is the case the packer cannot express. ``signed=False`` is the published
    all-positive stimulus, which the packer-resident ``relucomp`` arm needs.
    """
    ladder = sorted(bf16_ladder())
    g = torch.Generator().manual_seed(11)
    perm = torch.randperm(TILE_DATUMS, generator=g).tolist()

    if signed:
        values = [-ladder[perm[i]] for i in range(TILE_DATUMS)]
        # Survivors are the ones closest to zero; the index field only ever adds
        # magnitude, so the datum whose value IS the threshold falls below it.
        threshold = -ladder[_SURVIVORS]
    else:
        values = [ladder[perm[i]] for i in range(TILE_DATUMS)]
        threshold = ladder[TILE_DATUMS - _SURVIVORS]

    words = [(bf16_bits(v) << 16) | ((i + 1) & 0xFFFF) for i, v in enumerate(values)]
    tensor = torch.tensor(
        [w - (1 << 32) if w >= (1 << 31) else w for w in words], dtype=torch.int32
    ).view(torch.float32)
    return tensor, threshold


# arm -> (filter_arm, xl_merge_en, relu_en, signed_stimulus)
_ARMS = {
    "none": (0, False, False, True),
    # The per-tile SFPU envelope alone (sfpu_start/done + drain SFPNOPs, no MOP
    # body). Every SFPU arm pays it, so it is what stands between a raw slope and
    # a comparable issue rate.
    "ctrlenv": (5, False, False, True),
    "ctrlload": (3, False, False, True),
    "ctrlswap": (4, False, False, True),
    "mask1": (1, False, False, True),
    "negfilter": (2, False, False, True),
    "xlmerge": (0, True, False, True),
    "relucomp": (0, False, True, False),
}


@pytest.mark.perf
@blackhole_only
@parametrize(
    arm=sorted(_ARMS),
    tile_cnt=_TILE_COUNTS,
)
def test_perf_topk_negfilter(perf_report, arm, tile_cnt):
    filter_arm, xl_merge_en, relu_en, signed = _ARMS[arm]

    one_tile, threshold = fused_tile(signed)
    src_A = one_tile.repeat(SRC_SLOTS)
    src_B = torch.zeros(TILE_DATUMS, dtype=torch.float32)

    relu_config = (
        PackGolden.generate_relu_config(
            PackerReluType.MinThresholdRelu, threshold, DataFormat.Float32
        )
        if relu_en
        else 0
    )
    if relu_config >= (1 << 31):
        relu_config -= 1 << 32

    thr_bits = bf16_bits(threshold) << 16

    configuration = PerfConfig(
        "sources/topk_negfilter_perf.cpp",
        _FORMATS,
        run_types=[
            PerfRunType.L1_TO_L1,
            PerfRunType.UNPACK_ISOLATE,
            PerfRunType.MATH_ISOLATE,
            PerfRunType.PACK_ISOLATE,
            PerfRunType.L1_CONGESTION,
        ],
        templates=[
            generate_input_dim([32, 32], [32, 32]),
            NF_PARAMS(
                filter_arm=filter_arm,
                compress_en=True,
                row_start_section_size=ROW_START_SECTION_SIZE,
                downsample_mask=0,
                thr_bits=thr_bits,
                arm=arm,
                xl_merge_en=xl_merge_en,
            ),
        ],
        runtimes=[
            TILE_COUNT(tile_cnt),
            LOOP_FACTOR(1),
            RELU_CONFIG(relu_config),
            NUM_FACES(num_faces=4),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            _FORMATS.input_format,
            src_B,
            _FORMATS.input_format,
            _FORMATS.output_format,
            tile_count_A=SRC_SLOTS,
            tile_count_B=1,
            # One slot past the ring for the packer's PackerTileSize dump, which
            # is what proves the compression config write actually took effect
            # in THIS build.
            tile_count_res=RES_SLOTS + 1,
        ),
        dest_acc=DestAccumulation.Yes,
        unpack_to_dest=True,
    )

    configuration.run(perf_report, run_count=5)
