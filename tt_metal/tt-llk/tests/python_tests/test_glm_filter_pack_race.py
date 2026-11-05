# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Single-core repro of the tt-blaze glm_sparse_read_sdpa tilizer filter losing its mask (tt-blaze#6224).

Run with TT_LLK_DISABLE_ASSERTS=1 (tt-metal kernels build without LLK_ASSERT). Expected on this tt-metal
pin (44d66500): reps=2 with the mailbox switch (sem_switch=False) fails (reps=2,tail=False gives an
all-zero mask); reps=1, and every sem_switch=True case (tt-metal PR #57747's switch), pass. With asserts
on, every case passes.

One UInt32 index tile goes through the tilizer's fp32-DEST filter window (see
sources/glm_filter_pack_race_test.cpp). The packed mask must hold exactly one
nonzero word per index that belongs to bank 6, with value (index & 0x3FFF) + 1.
On the failing silicon the packer writes an all-zero mask instead.
"""

from dataclasses import dataclass

import pytest
import torch
from conftest import skip_for_quasar
from helpers.chip_architecture import ChipArchitecture, get_chip_architecture
from helpers.format_config import DataFormat
from helpers.llk_params import DestAccumulation
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import TemplateParameter

pytestmark = [skip_for_quasar]

ELEMENTS = 1024
MY_BANK = 6
BANK_SHIFT = 14
WITHIN_MASK = 0x3FFF
NUM_MATCHES = 80


@dataclass
class RS_KNOBS(TemplateParameter):
    remap: bool
    sort: bool
    per_tile_reconfig: bool
    pack_delay: int
    drain: bool
    sem_switch: bool = False
    unpack_delay: int = 0
    math_delay: int = 0
    reps: int = 1
    tail: bool = False

    def convert_to_cpp(self) -> str:
        lines = [
            f"constexpr bool RS_REMAP = {str(self.remap).lower()};",
            f"constexpr bool RS_SORT = {str(self.sort).lower()};",
            f"constexpr bool RS_PER_TILE_RECONFIG = {str(self.per_tile_reconfig).lower()};",
            f"constexpr std::uint32_t RS_PACK_DELAY = {self.pack_delay};",
            f"constexpr std::uint32_t RS_UNPACK_DELAY = {self.unpack_delay};",
            f"constexpr std::uint32_t RS_MATH_DELAY = {self.math_delay};",
            f"constexpr std::uint32_t RS_REPS = {self.reps};",
            f"constexpr bool RS_TAIL = {str(self.tail).lower()};",
        ]
        if self.drain:
            lines.append("#define RS_DRAIN_BEFORE_RELEASE 1")
        if self.sem_switch:
            lines.append("#define RS_SEM_SWITCH 1")
        return "\n".join(lines)


def _indices(seed: int) -> torch.Tensor:
    """Bank-addressed indices like the indexer's top-k output: NUM_MATCHES in bank 6, the rest in other banks."""
    g = torch.Generator().manual_seed(seed)
    other_banks = torch.tensor([b for b in range(8) if b != MY_BANK])
    banks = other_banks[torch.randint(0, len(other_banks), (ELEMENTS,), generator=g)]
    banks[torch.randperm(ELEMENTS, generator=g)[:NUM_MATCHES]] = MY_BANK
    within = torch.randint(0, 4096, (ELEMENTS,), generator=g)
    return ((banks << BANK_SHIFT) | within).to(torch.int64)


def _expected(indices: torch.Tensor) -> list[int]:
    hit = ((indices >> BANK_SHIFT) & 0xF) == MY_BANK
    return sorted(((indices[hit] & WITHIN_MASK) + 1).tolist())


@parametrize(
    formats=input_output_formats([DataFormat.UInt32], same=True),
    remap=[True, False],
    drain=[False],
    per_tile_reconfig=[True],
    sort=[True],
    pack_delay=[0],
    sem_switch=[False, True],
    unpack_delay=[0],
    math_delay=[0],
    reps=[1, 2],
    tail=[False, True],
)
def test_glm_filter_pack_race(
    formats, remap, drain, per_tile_reconfig, sort, pack_delay, sem_switch, unpack_delay, math_delay, reps, tail
):
    if get_chip_architecture() != ChipArchitecture.BLACKHOLE:
        pytest.skip("Blackhole-only repro")

    src_A = _indices(seed=6224)
    src_B = torch.zeros(ELEMENTS, dtype=torch.int64)

    configuration = TestConfig(
        "sources/glm_filter_pack_race_test.cpp",
        formats,
        templates=[
            RS_KNOBS(
                remap, sort, per_tile_reconfig, pack_delay, drain, sem_switch, unpack_delay, math_delay, reps, tail
            )
        ],
        runtimes=[],
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
        unpack_to_dest=True,
        dest_acc=DestAccumulation.No,
    )

    res = [int(v) & 0xFFFFFFFF for v in configuration.run().result[:ELEMENTS]]
    got = sorted(v for v in res if v != 0)
    want = _expected(src_A)
    print(
        f"RSRACE remap={remap} drain={drain} per_tile_reconfig={per_tile_reconfig} sort={sort} "
        f"delay={pack_delay} sem_switch={sem_switch} unpack_delay={unpack_delay} reps={reps} tail={tail} first={[hex(v) for v in res[:4]]} nonzero={len(got)} expected={len(want)} match={got == want}"
    )
    assert got == want, (
        f"mask holds {len(got)} nonzero words, expected {len(want)}; "
        f"first words {[hex(v) for v in res[:8]]}"
    )
