# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Test for tt-metal#60241 item 8: pack untilize must not write L1 outside its output.
# The result buffer is placed 32 B into a 64 B block, so the first and last packer writes end
# partway through a 64 B block. Sentinel bytes before and after the result buffer must survive.
import pytest

# Module import, not `from ... import test_*`, so pytest does not collect the base test here too.
import quasar.test_pack_untilize_quasar as base
from helpers.format_config import DataFormat
from helpers.llk_params import DestAccumulation, PerfRunType
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from ttexalens.tt_exalens_lib import read_from_device, write_to_device

L1_BLOCK_BYTES = 64
SENTINEL = 0x5A


def _align_down(addr: int) -> int:
    return addr - addr % L1_BLOCK_BYTES


def _align_up(addr: int) -> int:
    return _align_down(addr + L1_BLOCK_BYTES - 1)


class SentinelStimuliConfig(StimuliConfig):
    """Result buffer starts 32 B into a 64 B block, with sentinel bytes on both sides."""

    last = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        SentinelStimuliConfig.last = self

    def _calculate_tile_sizes(self):
        super()._calculate_tile_sizes()
        # One whole sentinel block, then the result 32 B into the next block.
        self.buf_res_addr = _align_up(self.buf_res_addr) + L1_BLOCK_BYTES + 32

    def sentinel_regions(self):
        start = self.buf_res_addr
        end = start + self.result_buffer_num_bytes
        return [
            (_align_down(start) - L1_BLOCK_BYTES, start),
            (end, _align_up(end) + L1_BLOCK_BYTES),
        ]

    def write(self, location: str = "0,0"):
        super().write(location)
        for lo, hi in self.sentinel_regions():
            write_to_device(location, lo, bytes([SENTINEL]) * (hi - lo))


# Float16_b, 16-bit Dest, first dest sync / dimensions for each tile shape. 1x32 and 2x32 take the
# strided (PACR_STRIDE) path with 32 B writes; 32x32 is the non-strided control.
# Plain names so `-k` can select them (it rejects commas).
COMBINATIONS = {
    name: next(
        c
        for c in base.generate_pack_untilize_combinations(
            input_output_formats([DataFormat.Float16_b])
        )
        if c[1] == DestAccumulation.No and tuple(c[4]) == tile_dims
    )
    for name, tile_dims in [
        ("tile1x32", (1, 32)),
        ("tile2x32", (2, 32)),
        ("tile32x32", (32, 32)),
    ]
}


@pytest.mark.quasar
@parametrize(tile_shape=list(COMBINATIONS))
def test_pack_untilize_sentinel_quasar(tile_shape):
    # A lone parametrize argument arrives wrapped in a 1-tuple.
    (tile_shape,) = tile_shape
    base.test_pack_untilize_quasar(
        COMBINATIONS[tile_shape],
        [PerfRunType.L1_TO_L1],
        1,
        stimuli_config_cls=SentinelStimuliConfig,
    )

    stimuli = SentinelStimuliConfig.last
    for lo, hi in stimuli.sentinel_regions():
        data = bytes(
            read_from_device(TestConfig.TENSIX_LOCATION, lo, num_bytes=hi - lo)
        )
        clobbered = [hex(lo + i) for i, b in enumerate(data) if b != SENTINEL]
        assert not clobbered, (
            f"pack untilize wrote outside its output [{hex(stimuli.buf_res_addr)}, "
            f"{hex(stimuli.buf_res_addr + stimuli.result_buffer_num_bytes)}): "
            f"{len(clobbered)} sentinel bytes changed, first at {clobbered[0]}"
        )
