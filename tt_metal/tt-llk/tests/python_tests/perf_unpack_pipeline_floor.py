# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""End-to-end floor for a streamed Top-K filter pass once the unpack-to-Dest
handshake is removed and Dest is double-buffered.

WHERE THIS COMES FROM
---------------------
``perf_unpack_ceiling.py`` established two things on this Blackhole, with the
SFPLOAD/SFPSWAP tripwires at 0.995 / 1.995:

    RawDest          1.004 cyc/vector   (127.5 B/cycle -- the unpacker's real rate)
    LlkDest          3.855 cyc/vector   (33.2 B/cycle -- the stock LLK path)
    RawDestSfpu      unpack 1.001, math 1.317, on DISJOINT Dest tiles -> max()
    RawDestSfpuSame  unpack 2.348, math 1.442, on the SAME Dest tiles  -> sum()

So the unpacker is not the limiter and never was, and the SFPU/unpack
serialisation is a same-Dest-region dependency, not a Dest port.

WHAT THIS ADDS
--------------
The two isolate run types cannot make an end-to-end claim -- for that the packer
has to be in the measurement. This driver runs the full three-thread pipeline
under L1_TO_L1 with the packer doing the shipping filter (MIN_THRESHOLD_RELU +
zero-compression, the ``relucomp`` arm of ``perf_topk_pipeline.py``, which needs
no SFPU at all) and the raw unpack stream feeding it.

``pack_dest_tile`` is the double-buffering knob, and it is the whole experiment:

    pack_dest_tile = 4   packer reads Dest tiles 4..7, unpacker writes 0..3
                         -> disjoint. Predicted steady state max(1.004, 1.25).
    pack_dest_tile = 0   packer reads the tiles the unpacker is writing
                         -> predicted sum, mirroring RawDestSfpuSame.

``LlkDest`` is carried as the in-kernel baseline so the win is a difference of
differences inside one binary, one stimulus, one session, rather than a
cross-kernel comparison against ``perf_topk_pipeline.py``'s 4.175.

The packer is free running -- no ``_llk_packer_wait_for_math_done_`` -- exactly
as ``PACK_ISOLATE`` already runs it in ``perf_topk_pipeline.py``. The packer's
rate was measured flat in survivor density there, so the emitted bytes are
correct-by-construction for the rate even though the Dest contents are whatever
the unpack stream happened to leave. ``PackerTileSize`` is dumped outside every
timed zone so a silently-failed compression config write cannot masquerade as
"compression is free".

READING THE NUMBER
------------------
Two-point slope over TILE_CNT; the report column is already per-tile, so

    cyc_per_tile   = (mean@hi * hi - mean@lo * lo) / (hi - lo)
    cyc_per_vector = cyc_per_tile / 32
"""

import pytest
import torch
from conftest import blackhole_only
from helpers.format_config import DataFormat
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
    generate_input_dim,
)
from perf_unpack_ceiling import (
    _FMT_F32,
    RES_SLOTS,
    SRC_SLOTS,
    TILE_DATUMS,
    UNP_PARAMS,
    UnpArm,
    _fused_dense_tile,
)

# 64 B reserved before the data stream for the row-start index array; the rss
# sweep in test_pack_zero_compress.py showed 4 (16 B units) covers the 16
# compression rows a Default-MOP tile pack produces. Pure address offset.
ROW_START_SECTION_SIZE = 4

_TILE_COUNTS = [16, 64]

# (unp_arm, pack_dest_tile) -> label. LlkDest is only run at pack_dest_tile 0
# because its unpacker writes wherever the math thread's mailbox tells it to,
# which is Dest tiles 0..3 -- there is no disjoint variant of the stock path.
_ARMS = {
    "llk_pack": (UnpArm.LlkDest, 0),
    "raw_pack_same": (UnpArm.RawDest, 0),
    "raw_pack_split": (UnpArm.RawDest, 4),
}


@pytest.mark.perf
@blackhole_only
@parametrize(
    arm=sorted(_ARMS),
    tile_cnt=_TILE_COUNTS,
)
def test_perf_unpack_pipeline_floor(perf_report, arm, tile_cnt):
    unp_arm, pack_dest_tile = _ARMS[arm]

    one_tile, threshold = _fused_dense_tile()
    src_A = one_tile.repeat(SRC_SLOTS)
    src_B = torch.zeros(TILE_DATUMS, dtype=torch.float32)

    # MIN_THRESHOLD_RELU on a fused FP32 sort key [bf16 value | u16 index]: the
    # word is a well-formed FP32 whose FP32 ordering is exactly "by value, ties
    # broken by index", so the packer's own float compare is the right compare
    # and the SFPU is not needed. The threshold must be non-negative
    # (Packers/ReLU.md:41); negative DATA is fine.
    relu_config = PackGolden.generate_relu_config(
        PackerReluType.MinThresholdRelu, threshold, DataFormat.Float32
    )

    configuration = PerfConfig(
        "sources/unpack_ceiling_perf.cpp",
        _FMT_F32,
        # L1_TO_L1 is the claim; the two isolates are the decomposition. Under
        # UNPACK_ISOLATE the packer is skipped and under PACK_ISOLATE both the
        # unpack and math threads are, so the three rows add up to a full
        # accounting of the same binary.
        run_types=[
            PerfRunType.L1_TO_L1,
            PerfRunType.UNPACK_ISOLATE,
            PerfRunType.PACK_ISOLATE,
        ],
        templates=[
            generate_input_dim([32, 32], [32, 32]),
            UNP_PARAMS(
                unp_arm=unp_arm,
                sfpu_dest_tile=0,
                throttle_mode=2,
                thr_bits=0,
                pack_en=True,
                compress_en=True,
                row_start_section_size=ROW_START_SECTION_SIZE,
                pack_dest_tile=pack_dest_tile,
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
            _FMT_F32.input_format,
            src_B,
            _FMT_F32.input_format,
            _FMT_F32.output_format,
            tile_count_A=SRC_SLOTS,
            tile_count_B=1,
            # One slot past the ring for the PackerTileSize dump.
            tile_count_res=RES_SLOTS + 1,
        ),
        dest_acc=DestAccumulation.Yes,
        unpack_to_dest=True,
    )

    configuration.run(perf_report, run_count=5)
