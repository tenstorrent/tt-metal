# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""End-to-end steady state of a pipelined Top-K filter pass on Blackhole.

WHAT THIS ANSWERS
-----------------
Two isolate numbers were measured earlier this session:

    SFPU MaskStore macro (Load+SFPGT+SFPSTORE)   1.003 cyc / 32-elem vector
    PACK 32-bit datums, zero-compression on      1.648 cyc / 32-elem vector

and ``_topk_xl_merge_`` at K=512 fused is 2.844. The SFPU and the packer are
separate backend ports off the same frontend mux, so a kernel running both
SHOULD steady-state at max(1.003, 1.648) = 1.648, not at the sum 2.651. Neither
isolate can distinguish those. This driver runs the real pipeline over a
multi-tile stream and reads the answer off the wall clock.

THE 2x2 (plus the ReLU arm)
---------------------------
SFPU_EN and COMPRESS_EN are independent compile-time flags, so every effect is a
difference of differences on ONE kernel, ONE stimulus, ONE session:

    base       sfpu=0 compress=0 relu=off   plain streamed pack
    comp       sfpu=0 compress=1 relu=off   compression alone
    sfpu       sfpu=1 compress=0 relu=off   the SFPU filter alone
    sfpucomp   sfpu=1 compress=1 relu=off   both -> max() or sum()?
    relu       sfpu=0 compress=0 relu=on    packer threshold alone
    relucomp   sfpu=0 compress=1 relu=on    THE CANDIDATE: threshold + compaction
                                            in one PACR, zero SFPU instructions

``relucomp`` is the arm that matters. ``test_pack_compress_arms.py`` established
on silicon that a dense tile of fused FP32 sort keys
``[bf16 value (high 16) | u16 index (low 16)]`` packs 4096 B -> 640 B with
MIN_THRESHOLD_RELU + zero-compression enabled -- the same emitted size the
SFPU-prezeroed Int32 arm produces. The fused word is a well-formed FP32 whose
FP32 ordering is exactly "by value, ties broken by index", so the packer's own
float compare is the right compare and the SFPU is not needed at all. The one
restriction is that the threshold must be non-negative
(``Packers/ReLU.md:41``); negative DATA is fine.

RUN TYPES, AND WHY
------------------
L1_TO_L1 is the primary measurement: it timestamps unpack's ZONE_START against
pack's ZONE_END, i.e. the whole three-thread pipeline, which is the only run
type that can tell max from sum. L1_CONGESTION is the cross-check -- it times
each thread's own zone while all three threads run, so L1_CONGESTION[PACK] is
the packer's own view of the same steady state and L1_CONGESTION[UNPACK] shows
whether the unpacker (not the packer) is the real limiter.

READING THE NUMBER
------------------
LOOP_FACTOR and TILE_COUNT are deliberately left at 1 in
``postprocess_tile_loop``'s divisor (TILE_COUNT is passed as a runtime parameter
that the kernel loops on, but the report's ``tile_cnt`` column is pinned to 1),
so the .post.csv carries RAW zone cycles. Cycles per 32-element vector is a
two-point slope over the streamed tile count:

    cyc_per_vector = (mean@hi - mean@lo) / (hi - lo) / 32

The subtraction cancels the marker pair, the packer/unpacker hw-configure, the
replay-buffer load, the MOP programming, and the pipeline fill/drain -- every
one-time cost. One 32x32 tile is 1024 datums = 32 vectors of 32.
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
    LOOP_FACTOR,
    NUM_FACES,
    RELU_CONFIG,
    TILE_COUNT,
    TemplateParameter,
    generate_input_dim,
)

TILE_DATUMS = 1024

# Ring sizes for the L1 buffers. Powers of two so the kernel can index with a
# mask; 16 keeps the working set small (16 * 4 KB in, 17 * 4 KB out) while still
# being large enough that consecutive packs never hit the same line.
SRC_SLOTS = 16
RES_SLOTS = 16

# 64 B reserved before the data stream for the row-start index array. The rss
# sweep in test_pack_zero_compress.py showed 4 (16 B units) is enough for the 16
# compression rows a Default-MOP tile pack produces; it is a pure address offset,
# not a per-datum cost.
ROW_START_SECTION_SIZE = 4

# Two-point slope over the streamed tile count. Both are multiples of the 4-tile
# Dest block (fp32 Dest, DstSync::SyncHalf), so neither point pays a partial
# block the other does not.
_TILE_COUNTS = [16, 64]

_FORMATS = InputOutputFormat(DataFormat.Float32, DataFormat.Float32)

# How many of the 1024 datums are meant to survive the threshold. 32 is the
# top-k case that matters. Compression cost was measured FLAT in density, so
# this steers correctness/emitted-size, not the rate.
_SURVIVORS = 32


@dataclass
class PIPE_PARAMS(TemplateParameter):
    """Compile-time knobs, plus the arm label.

    ``arm`` is emitted even though the kernel never reads it: it has to be part
    of the variant hash (otherwise two arms would share one ELF and one report
    row) and it has to be a report column (otherwise the CSV cannot tell the
    rows apart).
    """

    sfpu_en: bool = False
    compress_en: bool = False
    row_start_section_size: int = 0
    downsample_mask: int = 0
    thr_bits: int = 0
    arm: str = "base"
    xl_merge_en: bool = False

    def convert_to_cpp(self) -> str:
        return "\n".join(
            [
                # A #define, not a constexpr: the kernel guards the topk_xl
                # include itself with #if, and only the preprocessor can gate an
                # #include. The experimental SFPU trees must not all be pulled in
                # at once.
                f"#define XL_MERGE_EN {1 if self.xl_merge_en else 0}",
                f"constexpr bool SFPU_EN = {str(self.sfpu_en).lower()};",
                f"constexpr bool COMPRESS_EN = {str(self.compress_en).lower()};",
                f"constexpr std::uint32_t ROW_START_SECTION_SIZE = {self.row_start_section_size};",
                f"constexpr std::uint32_t DOWNSAMPLE_MASK = {self.downsample_mask};",
                f"constexpr std::uint32_t THR_BITS = {self.thr_bits}u;",
                f"constexpr std::uint32_t SRC_SLOTS = {SRC_SLOTS};",
                f"constexpr std::uint32_t RES_SLOTS = {RES_SLOTS};",
                f"// arm = {self.arm}",
            ]
        )


def bf16_ladder(n=TILE_DATUMS):
    """n distinct, exactly-representable bfloat16 values in [1, 256)."""
    vals = []
    for e in range(8):
        for m in range(128):
            vals.append((2.0**e) * (1.0 + m / 128.0))
    return vals[:n]


def fused_dense_tile(seed=11):
    """One dense tile of fused FP32 sort keys, plus the bf16 threshold that lets
    exactly _SURVIVORS of them through.

    Word layout is ckernel_sfpu_topk_xl.h's: [bf16 value | u16 (index+1)].
    """
    vals = bf16_ladder()
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(TILE_DATUMS, generator=g).tolist()
    words = []
    for i in range(TILE_DATUMS):
        vb = int(
            torch.tensor([vals[perm[i]]], dtype=torch.bfloat16)
            .view(torch.uint16)
            .item()
        )
        words.append((vb << 16) | ((i + 1) & 0xFFFF))
    threshold = vals[TILE_DATUMS - _SURVIVORS]
    return torch.tensor(words, dtype=torch.int32).view(torch.float32), threshold


# arm -> (sfpu_en, compress_en, relu_en, xl_merge_en)
#
# xlmerge is the competition, run in this kernel rather than quoted from
# another: same stream, same unpacker, same packer, same session. It is the only
# way to see whether _topk_xl_merge_'s SFPU time hides under the unpacker or
# adds to it -- which is exactly the question an isolate cannot answer.
_ARMS = {
    "base": (False, False, False, False),
    "comp": (False, True, False, False),
    "sfpu": (True, False, False, False),
    "sfpucomp": (True, True, False, False),
    "relu": (False, False, True, False),
    "relucomp": (False, True, True, False),
    "xlmerge": (False, False, False, True),
    "xlmergecomp": (False, True, True, True),
}


@pytest.mark.perf
@blackhole_only
@parametrize(
    arm=sorted(_ARMS),
    tile_cnt=_TILE_COUNTS,
)
def test_perf_topk_pipeline(perf_report, arm, tile_cnt):
    sfpu_en, compress_en, relu_en, xl_merge_en = _ARMS[arm]

    one_tile, threshold = fused_dense_tile()
    src_A = one_tile.repeat(SRC_SLOTS)
    src_B = torch.zeros(TILE_DATUMS, dtype=torch.float32)

    relu_config = (
        PackGolden.generate_relu_config(
            PackerReluType.MinThresholdRelu, threshold, DataFormat.Float32
        )
        if relu_en
        else 0
    )

    # The SFPU compares the fused FP32 word against the same threshold the packer
    # ReLU would use, so both arms select the same survivors. SFPGT orders by the
    # sign-magnitude total order, which on well-formed FP32 agrees with the
    # packer's float compare.
    thr_bits = (
        int(torch.tensor([threshold], dtype=torch.bfloat16).view(torch.uint16).item())
        << 16
    )

    configuration = PerfConfig(
        "sources/topk_pipeline_perf.cpp",
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
            PIPE_PARAMS(
                sfpu_en=sfpu_en,
                compress_en=compress_en,
                row_start_section_size=ROW_START_SECTION_SIZE if compress_en else 0,
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
            # is what proves the compression config write actually took effect in
            # THIS build rather than being inferred from a timing that a silently
            # failed write would reproduce exactly.
            tile_count_res=RES_SLOTS + 1,
        ),
        dest_acc=DestAccumulation.Yes,
        unpack_to_dest=True,
    )

    configuration.run(perf_report, run_count=5)
