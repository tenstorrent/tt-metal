# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Blackhole: is 3.938 cyc/vector the unpacker's ceiling, or the LLK handshake?

WHAT THIS ATTACKS
-----------------
``perf_topk_pipeline.py``'s UNPACK_ISOLATE row is 126.0 cycles per 32x32 FP32
tile (3.9375 cyc / 32-element vector, 32.5 B/cycle) and has been treated as a
hardware floor for a streamed Top-K. Two independent readings of the ISA docs
say it should not be:

* ``UNPACR_Regular.md:640-652`` -- the unpacker fetches L1 at x1/x2/x4 =
  16/32/64 B per cycle, selected by ``THCON_SEC[i].REG2_Throttle_mode``. LLK
  programs 2 (x4 = 64 B/cycle) at ``cunpack_common.h:892``. So 32.5 B/cycle is
  HALF the applicable documented ceiling. (The 128 B/cycle figure lives only
  inside the ``!UnpackToDst`` SrcA-burst branch at ``UNPACR_Regular.md:315-328``
  and is not an unpack-to-Dest number.)
* the per-tile LLK bracket around the 4-UNPACR MOP is a full cross-thread
  lockstep: ``mailbox_read(MathThreadId)`` (blocking), a ``SEMWAIT`` on a
  ``max=1`` semaphore, ``STALLWAIT(STALL_UNPACK, TRISC_CFG)``, four
  ``cfg_reg_rmw_tensix`` and, on the math side, a full
  ``STALLWAIT(STALL_SYNC, MATH|WAIT_SFPU)`` drain of every in-flight FPU and
  SFPU instruction BEFORE every single tile.

The second point also predicts the serialisation the branch has been living
with: ``perf_topk_pipeline.py``'s UNPACK_ISOLATE moves 126 -> 176 cycles/tile
when the math thread runs the SFPU filter. The unpacker is not losing a Dest
port; it is parked in ``mailbox_read`` behind a math thread that the handshake
requires to be idle first.

THE ARMS
--------
``Raw*`` arms hoist the entire bracket out of the loop -- the MOP body is
byte-identical to the LLK's, only the handshake is gone.

    LlkDest          stock path, reproduces 3.938 in this kernel
    LlkDestSfpu      stock path + SFPU filter, reproduces 5.438
    RawDest          handshake removed -> the unpacker->Dest ceiling
    RawDestSfpu      RawDest + SFPU filter on DISJOINT Dest tiles (4..7)
    RawDestSfpuSame  RawDest + SFPU filter on the SAME Dest tiles (0..3)
    SfpuOnly         SFPU filter alone
    CtrlLoad         SFPLOAD tripwire, MUST be 1.000
    CtrlSwap         SFPSWAP tripwire, MUST be 2.000 (else the run is invalid)
    LlkSrcA          stock `_llk_unpack_A_` to SrcA, format-swept
    RawSrcA          MOP-only stream to SrcA, no DVALID, format-swept

RawDest and RawSrcA additionally sweep ``Throttle_mode`` in {2, 3}: Blackhole
decodes 3 as x8, and whether that reaches the Dest write path is exactly the
kind of thing the docs decline to say.

WHY BOTH ISOLATE RUN TYPES ON THE OVERLAP ARMS
----------------------------------------------
In this kernel ``PERF_RUN_TYPE`` selects which thread's zone the report exposes,
not which threads run: in the Raw arms the unpack thread streams and the math
thread runs its SFPU body under BOTH isolate run types, with no synchronisation
between them. So ``mean(UNPACK_ISOLATE)`` answers "does the SFPU slow the
unpacker" and ``mean(MATH_ISOLATE)`` answers "does the unpacker slow the SFPU",
from one binary. Compare each against the solo arms (RawDest, SfpuOnly).

RawDestSfpuSame vs RawDestSfpu is the structural question. If concurrency
appears only when the SFPU works a disjoint Dest region, the limit is a
same-region dependency (and Dest double-buffering is the fix). If both
serialise, the Dest register file has one arbitrated port and it is not.

READING THE NUMBER
------------------
Two-point slope over TILE_CNT, exactly as ``perf_topk_pipeline.py``: the report
column is already per-tile, so

    cyc_per_tile   = (mean@hi * hi - mean@lo * lo) / (hi - lo)
    cyc_per_vector = cyc_per_tile / 32
    bytes_per_cyc  = tile_bytes / cyc_per_tile

The subtraction cancels the ~30-cycle marker pair, the hw-configure, the MOP
programming and the replay-buffer load.
"""

from dataclasses import dataclass
from enum import Enum

import pytest
import torch
from conftest import blackhole_only
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import DestAccumulation, PerfRunType
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

# Stimulus ring. 16 slots of one tile each, matching perf_topk_pipeline.py, so
# the L1 working set (and therefore any L1 bank behaviour) is the same.
SRC_SLOTS = 16
RES_SLOTS = 16

# Two-point slope. Both multiples of the 4-tile FP32 Dest block, so neither
# point pays a partial block the other does not.
_TILE_COUNTS = [16, 64]


class UnpArm(Enum):
    """Arm id. The value IS the integer the kernel's ``UNP_ARM`` preprocessor
    comparison expects, so it is emitted verbatim."""

    LlkDest = 0
    LlkDestSfpu = 1
    RawDest = 2
    RawDestSfpu = 3
    RawDestSfpuSame = 4
    SfpuOnly = 5
    CtrlLoad = 6
    CtrlSwap = 7
    LlkSrcA = 8
    RawSrcA = 9


# Arms whose unpack path requires 32-bit in and out (should_unpack_to_dest,
# cunpack_common.h:244). Also the value handed to PerfConfig(unpack_to_dest=),
# which the Python format-inference model uses to pick unpack_A_dst.
_DEST_ARMS = {
    UnpArm.LlkDest,
    UnpArm.LlkDestSfpu,
    UnpArm.RawDest,
    UnpArm.RawDestSfpu,
    UnpArm.RawDestSfpuSame,
}

# Dest tile the SFPU filter macro walks. 4 is disjoint from the raw stream's
# tiles 0..3 (the kernel wraps its Dest ADC every MAX_TILES_DEST = 4 tiles);
# 0 collides with it deliberately.
_SFPU_DEST_TILE = {
    UnpArm.RawDestSfpu: 4,
    UnpArm.RawDestSfpuSame: 0,
    UnpArm.LlkDestSfpu: 0,
    UnpArm.SfpuOnly: 0,
}

# Throttle_mode sweep. Only the Raw arms can carry it: the stock LLK arms would
# have their value overwritten by _llk_unpack_hw_configure_ on every call.
_THROTTLE_MODES = {
    UnpArm.RawDest: [2, 3],
    UnpArm.RawSrcA: [2, 3],
}

# Format sweep. The Dest arms are pinned to Float32 -- unpack-to-dest is gated
# on 32-bit in AND out, and a non-32-bit format would silently fall through to
# the SrcA path and measure the wrong thing. The SrcA arms sweep three widths so
# B/cycle can be read against datum size.
_FMT_F32 = InputOutputFormat(DataFormat.Float32, DataFormat.Float32)
_FMT_F16B = InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b)
_FMT_BFP8 = InputOutputFormat(DataFormat.Bfp8_b, DataFormat.Bfp8_b)

_SRCA_FORMATS = [_FMT_F32, _FMT_F16B, _FMT_BFP8]


@dataclass
class UNP_PARAMS(TemplateParameter):
    """Compile-time knobs.

    ``unp_arm`` MUST be emitted as a ``#define``: the kernel guards it with
    ``#ifndef UNP_ARM`` and falls back to 0, so a ``constexpr`` would leave every
    swept arm compiling as LlkDest while still hashing to a distinct variant id
    -- ten identical rows and no error anywhere.
    """

    unp_arm: UnpArm = UnpArm.LlkDest
    sfpu_dest_tile: int = 0
    throttle_mode: int = 2
    thr_bits: int = 0
    # Packer side. Off in this driver; perf_unpack_pipeline_floor.py turns it on.
    pack_en: bool = False
    compress_en: bool = False
    row_start_section_size: int = 0
    pack_dest_tile: int = 0

    def convert_to_cpp(self) -> str:
        return "\n".join(
            [
                # A #define, not a constexpr: the kernel's PACK_EN guard has to
                # be visible to the preprocessor because it gates whole blocks
                # that reference packer-only runtime parameters.
                f"#define PACK_EN {1 if self.pack_en else 0}",
                f"#define UNP_ARM {self.unp_arm.value}",
                f"constexpr std::uint32_t SFPU_DEST_TILE = {self.sfpu_dest_tile};",
                f"constexpr std::uint32_t PACK_DEST_TILE = {self.pack_dest_tile};",
                f"constexpr std::uint32_t THROTTLE_MODE = {self.throttle_mode};",
                f"constexpr std::uint32_t THR_BITS = {self.thr_bits}u;",
                f"constexpr bool COMPRESS_EN = {str(self.compress_en).lower()};",
                f"constexpr std::uint32_t ROW_START_SECTION_SIZE = {self.row_start_section_size};",
                f"constexpr std::uint32_t SRC_SLOTS = {SRC_SLOTS};",
                f"constexpr std::uint32_t RES_SLOTS = {RES_SLOTS};",
                f"// arm = {self.unp_arm.name}",
            ]
        )


def _bf16_ladder(n=TILE_DATUMS):
    """n distinct, exactly-representable bfloat16 values in [1, 256)."""
    vals = []
    for e in range(8):
        for m in range(128):
            vals.append((2.0**e) * (1.0 + m / 128.0))
    return vals[:n]


def _fused_dense_tile(seed=11):
    """One dense tile of fused FP32 sort keys plus a threshold letting 32 through.

    Same stimulus and same word layout ``[bf16 value | u16 (index+1)]`` as
    ``perf_topk_pipeline.py``, so the SFPU filter arm here is doing exactly the
    work the pipeline test measured at 1.3125 cyc/vector.
    """
    vals = _bf16_ladder()
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
    threshold = vals[TILE_DATUMS - 32]
    return torch.tensor(words, dtype=torch.int32).view(torch.float32), threshold


def _formats_for(arm: UnpArm):
    """Only the two SrcA arms sweep width.

    Everything else is pinned to Float32: the Dest arms because unpack-to-dest
    is gated on 32-bit in AND out and a narrower format would silently fall
    through to the SrcA path, and the SFPU/control arms because their Dst walk
    is an INT32 load/store of the fused ``[bf16 | u16]`` key.
    """
    return _SRCA_FORMATS if arm in (UnpArm.LlkSrcA, UnpArm.RawSrcA) else [_FMT_F32]


@pytest.mark.perf
@blackhole_only
@parametrize(
    unp_arm=list(UnpArm),
    formats=lambda unp_arm: _formats_for(unp_arm),
    throttle_mode=lambda unp_arm: _THROTTLE_MODES.get(unp_arm, [2]),
    tile_cnt=_TILE_COUNTS,
)
def test_perf_unpack_ceiling(perf_report, unp_arm, formats, throttle_mode, tile_cnt):
    one_tile, threshold = _fused_dense_tile()

    # The stimulus is only meaningful for the FP32 arms; the SrcA format sweep
    # cares about tile SIZE, not content, and every arm here is data-independent
    # (fixed loop bounds, fixed compare network).
    if formats.input_format == DataFormat.Float32:
        src_A = one_tile.repeat(SRC_SLOTS)
    else:
        src_A = torch.rand(TILE_DATUMS * SRC_SLOTS, dtype=torch.float32)
    src_B = torch.zeros(TILE_DATUMS, dtype=torch.float32)

    thr_bits = (
        int(torch.tensor([threshold], dtype=torch.bfloat16).view(torch.uint16).item())
        << 16
    )

    configuration = PerfConfig(
        "sources/unpack_ceiling_perf.cpp",
        formats,
        # UNPACK_ISOLATE exposes the unpack thread's zone, MATH_ISOLATE the math
        # thread's. In the Raw arms BOTH threads work under BOTH, which is what
        # makes the overlap question answerable from one binary. L1_TO_L1 and
        # PACK_ISOLATE are omitted: the packer does no work here, so an
        # unpack-start-to-pack-end span would be meaningless.
        run_types=[
            PerfRunType.UNPACK_ISOLATE,
            PerfRunType.MATH_ISOLATE,
        ],
        templates=[
            generate_input_dim([32, 32], [32, 32]),
            UNP_PARAMS(
                unp_arm=unp_arm,
                sfpu_dest_tile=_SFPU_DEST_TILE.get(unp_arm, 0),
                throttle_mode=throttle_mode,
                thr_bits=thr_bits,
            ),
        ],
        runtimes=[
            TILE_COUNT(tile_cnt),
            LOOP_FACTOR(1),
            # Emitted even though the packer is disabled here: the pack kernel
            # reads RELU_CONFIG unconditionally, and under --speed-of-light a
            # runtime parameter only exists as a constexpr if it is listed.
            RELU_CONFIG(0),
            NUM_FACES(num_faces=4),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=SRC_SLOTS,
            tile_count_B=1,
            tile_count_res=RES_SLOTS + 1,
        ),
        dest_acc=DestAccumulation.Yes,
        unpack_to_dest=unp_arm in _DEST_ARMS,
    )

    configuration.run(perf_report, run_count=5)
