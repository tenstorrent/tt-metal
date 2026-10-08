# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Perf sweep of the Blackhole SDPA weighted reduce (sources/sdpa_weighted_reduce_perf.cpp, the api header's path as the
DSA indexer runs it): per-chunk unpack transactions against one transaction for all chunks of a DEST section, and per-row
packs against one pack setup for the section's rows (weighted_reduce_pack_block); the figures are per chunk.
"""

from dataclasses import dataclass

import pytest
from conftest import skip_for_quasar, skip_for_wormhole
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import DestAccumulation, PerfRunType
from helpers.param_config import parametrize
from helpers.perf.core import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import LOOP_FACTOR, TILE_COUNT, TemplateParameter

pytestmark = [skip_for_wormhole, skip_for_quasar]

BF16 = DataFormat.Float16_b


@dataclass
class WEIGHTED_REDUCE_PERF(TemplateParameter):
    chunks_per_section: int = 1
    block_unpack: bool = False
    block_pack: bool = False

    def convert_to_cpp(self) -> str:
        return "\n".join(
            [
                f"constexpr std::uint32_t NUM_CHUNKS = {self.chunks_per_section};",
                f"constexpr bool BLOCK = {str(self.block_unpack).lower()};",
                f"constexpr bool BLOCK_PACK = {str(self.block_pack).lower()};",
            ]
        )


@pytest.mark.perf
@parametrize(chunks=[1, 4, 8, 16], block=[False, True], block_pack=[False, True])
def test_perf_sdpa_weighted_reduce(perf_report, chunks, block, block_pack):
    configuration = PerfConfig(
        "sources/sdpa_weighted_reduce_perf.cpp",
        InputOutputFormat(BF16, BF16),
        run_types=[
            PerfRunType.L1_TO_L1,
            PerfRunType.UNPACK_ISOLATE,
            PerfRunType.MATH_ISOLATE,
            PerfRunType.PACK_ISOLATE,
        ],
        templates=[
            WEIGHTED_REDUCE_PERF(
                chunks_per_section=chunks, block_unpack=block, block_pack=block_pack
            )
        ],
        runtimes=[TILE_COUNT(chunks), LOOP_FACTOR(64)],
        variant_stimuli=StimuliConfig(
            None,
            BF16,
            None,
            BF16,
            BF16,
            tile_count_A=1,
            tile_count_B=chunks,
            tile_count_res=1,
        ),
        dest_acc=DestAccumulation.No,
    )
    configuration.run(perf_report)
