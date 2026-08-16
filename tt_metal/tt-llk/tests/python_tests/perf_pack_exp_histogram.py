# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""What the packer's exponent histogram costs on Blackhole, in cycles.

WHY
---
``sources/pack_exp_histogram_test.cpp`` proved on silicon that the Blackhole packer
maintains a 32-bin exponent histogram, that ``CLREXPHIST`` resets it, and that
``SETDMAREG`` modes 6/7/9 read it back. If it is FREE, a top-k threshold search gets
its bucket counts on a unit that is otherwise idle during SFPU work, at zero SFPU
issue slots -- which is the whole point, since counting on the SFPU is pinned at
1.998 cyc per 32-element vector by the single shared SFPU issue port.

WHAT IS MEASURED
----------------
PACK_ISOLATE. Unpack and math fill Dest once in the INIT zone and do nothing inside
TILE_LOOP, which issues PACK_ITER_COUNT tile packs from that one Dest tile.

Cycles per 32-element vector comes from a two-point slope over PACK_ITER_COUNT:

    cyc_per_vector = (mean@hi - mean@lo) / (hi - lo) / 32

One 32x32 tile is 1024 datums = 32 vectors of 32. The subtraction cancels the marker
pair and every one-time cost inside the zone (packer hw-configure, MOP programming,
the SETC16, the wait-for-math-done). What survives is the marginal cost of one more
tile pack.

ARMS
----
  off  ENABLE_ACC_STATS_Enable = 0 on all three threads. Baseline.
  on   ENABLE_ACC_STATS_Enable = 1 on the pack thread.
  clr  Same, plus one CLREXPHIST inside the loop before every tile pack -- what a real
       per-tile threshold search must do, since the counters were measured to
       accumulate across packs when it is omitted.

TRIPWIRE
--------
The kernel dumps the first 16 histogram bins (SETDMAREG mode 6) and PackerTileSize into
buffer_Res outside every timed zone. An "on" row whose histogram bytes are all zero is a
build where the enable silently failed, and would otherwise read as "the histogram is
free".
"""

from dataclasses import dataclass

import pytest
import torch
from conftest import blackhole_only
from helpers.device_io import read_from_device
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import DestAccumulation, PerfRunType
from helpers.param_config import parametrize
from helpers.perf.core import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import BuildMode, TestConfig
from helpers.test_variant_parameters import (
    NUM_FACES,
    RELU_CONFIG,
    TemplateParameter,
    generate_input_dim,
)

TILE_DATUMS = 1024

# Two-point slope. 16 tile packs is already ~15x the marker pair at the cheapest
# plausible rate, and 64 keeps a 5-repeat run short on a shared device.
_ITER_COUNTS = [16, 64]

_ARMS = ["off", "on", "clr"]

_BF16 = InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b)


@dataclass
class HIST_PERF(TemplateParameter):
    """Histogram knobs, plus the arm label.

    ``arm`` is emitted into build.h even though the kernel never reads it: it has to be
    part of the variant hash and it has to be a report column, otherwise two arms share
    one ELF and one report row.
    """

    hist_en: bool = False
    clr_per_tile: bool = False
    pack_iter_count: int = 16
    cycle_output: bool = True
    downsample_mask: int = 0
    arm: str = "off"

    def convert_to_cpp(self) -> str:
        return "\n".join(
            [
                f"constexpr bool HIST_EN = {str(self.hist_en).lower()};",
                f"constexpr bool CLR_PER_TILE = {str(self.clr_per_tile).lower()};",
                f"constexpr std::uint32_t PACK_ITER_COUNT = {self.pack_iter_count};",
                f"constexpr bool CYCLE_OUTPUT = {str(self.cycle_output).lower()};",
                f"constexpr std::uint32_t DOWNSAMPLE_MASK = {self.downsample_mask};",
                f"// arm = {self.arm}",
            ]
        )


def _stimulus():
    """A spread of exponents, so the histogram has real work to do rather than hitting
    one bin. 1..250 is exact in bfloat16 and spans exponents 127..134."""
    g = torch.Generator().manual_seed(7)
    return (
        torch.randint(1, 251, (TILE_DATUMS,), generator=g)
        .to(torch.bfloat16)
        .contiguous()
    )


@pytest.mark.perf
@blackhole_only
@parametrize(
    arm=_ARMS,
    formats=[_BF16],
    iter_count=_ITER_COUNTS,
)
def test_perf_pack_exp_histogram(perf_report, arm, formats, iter_count):
    src_A = _stimulus()
    src_B = torch.zeros_like(src_A)

    configuration = PerfConfig(
        "sources/pack_exp_histogram_perf.cpp",
        formats,
        run_types=[PerfRunType.PACK_ISOLATE],
        templates=[
            generate_input_dim([32, 32], [32, 32]),
            HIST_PERF(
                hist_en=(arm != "off"),
                clr_per_tile=(arm == "clr"),
                pack_iter_count=iter_count,
                cycle_output=True,
                downsample_mask=0,
                arm=arm,
            ),
        ],
        runtimes=[
            RELU_CONFIG(0),
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
        dest_acc=DestAccumulation.No,
    )

    configuration.run(perf_report, run_count=5)

    # Tripwire. The kernel writes PackerTileSize and the first 16 histogram bins to
    # buffer_Res[0] outside every timed zone. "cost is exactly zero" is precisely what a
    # silently-failed enable would also produce, so read it back and prove the counters
    # were live during the timed packs.
    if TestConfig.BUILD_MODE == BuildMode.PRODUCE:
        return
    raw = read_from_device(
        TestConfig.TENSIX_LOCATION,
        configuration.variant_stimuli.buf_res_addr,
        num_bytes=32,
    )
    d = [int.from_bytes(bytes(raw[4 * i : 4 * i + 4]), "little") for i in range(8)]
    bins = [(d[4 + w] >> (8 * k)) & 0xFF for w in range(4) for k in range(4)]
    print(
        f"\n  [tripwire] arm={arm} iter={iter_count} sentinel=0x{d[0]:08X} "
        f"packed_units={d[1]} hist_en={d[2]} bins0_15={bins}"
    )
    assert d[0] == 0xC0DEBA5E, "kernel did not reach the diagnostic dump"
    assert d[1] == 128, f"pack did not emit a full bf16 tile: {d[1]} units"
    # INIT clears the counters in every arm, so "off" means "stayed empty".
    if arm == "off":
        assert sum(bins) == 0, f"histogram live in the OFF arm: {bins}"
    else:
        assert sum(bins) > 0, f"histogram DEAD in the {arm} arm: {bins}"
