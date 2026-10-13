# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""llk_analysis transpose family headroom bit dump (Blackhole), not part of the tree. sources/transpose_blk_bits_x.cpp runs
transpose_tile's unpacker transpose per tile (form 0, main), as one block call per DEST section (form 1) and alternating
(form 4) on the same stimuli: every bf16 and fp16 pattern once (64 tiles), seeded Bfp8_b and Int8. The raw result bytes go
to $BITDUMP_DIR/<case>_f<form>.bin (one test per case and form; bits_cmp.py compares the forms byte for byte) and form 0
prints the content check (result words equal to the transposed input words)."""

import os
from dataclasses import dataclass

import pytest
import torch
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import TransposeGolden, get_golden_generator
from helpers.llk_params import DestAccumulation
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import BuildMode, TestConfig
from helpers.test_variant_parameters import NUM_FACES, TILE_COUNT, TemplateParameter

OUT = os.environ.get("BITDUMP_DIR", "/tmp/bitdump_transpose")
_RAW = {}
_real_collect = StimuliConfig.collect_results


@dataclass
class UNPACK_FORM(TemplateParameter):
    unpack_form: int

    def convert_to_cpp(self) -> str:
        return f"constexpr std::uint32_t UNPACK_FORM = {self.unpack_form};"


@dataclass
class SECTION_TILES(TemplateParameter):
    section_tiles: int

    def convert_to_cpp(self) -> str:
        return f"constexpr std::uint32_t SECTION_TILES = {self.section_tiles};"


def _collect(self, location="0,0"):
    _RAW["last"] = self.collect_raw_result_bytes(location)
    return _real_collect(self, location)


@pytest.fixture(autouse=True)
def _patch(monkeypatch):
    monkeypatch.setattr(StimuliConfig, "collect_results", _collect)


F = DataFormat
NO, YES = DestAccumulation.No, DestAccumulation.Yes
CASES = {
    "bf16_s8": (F.Float16_b, NO, 8, True),
    "bf16acc_s4": (F.Float16_b, YES, 4, True),
    "fp16_s8": (F.Float16, NO, 8, True),
    "bfp8_s8": (F.Bfp8_b, NO, 8, False),
    "int8acc_s4": (F.Int8, YES, 4, False),
}
DIMS = [256, 256]


@pytest.mark.parametrize("form", [0, 1, 4])
@pytest.mark.parametrize("case", list(CASES.keys()))
def test_transpose_blk_bits_x(case, form):
    fmt, acc, section, exhaustive = CASES[case]
    formats = InputOutputFormat(fmt, fmt)
    torch.manual_seed(1234)
    src_A, tile_cnt, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=fmt, input_dimensions_A=DIMS, stimuli_format_B=fmt, input_dimensions_B=DIMS
    )
    if exhaustive:
        src_A = torch.arange(src_A.numel(), dtype=torch.int64).remainder(65536).to(torch.int16).view(src_A.dtype).reshape(src_A.shape)

    configuration = TestConfig(
        "sources/transpose_blk_bits_x.cpp",
        formats,
        templates=[UNPACK_FORM(form), SECTION_TILES(section)],
        runtimes=[TILE_COUNT(tile_cnt), NUM_FACES()],
        variant_stimuli=StimuliConfig(
            src_A,
            fmt,
            src_B,
            fmt,
            fmt,
            tile_count_A=tile_cnt,
            tile_count_B=tile_cnt_B,
            tile_count_res=tile_cnt,
        ),
        dest_acc=acc,
        unpack_to_dest=False,
    )
    outcome = configuration.run()
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"{case}_f{form}.bin"), "wb") as f:
        f.write(_RAW.pop("last"))
    with open(os.path.join(OUT, f"{case}_in{form}.bin"), "wb") as f:
        f.write(src_A.contiguous().view(torch.uint8).numpy().tobytes())

    if exhaustive and form == 0:
        t = get_golden_generator(TransposeGolden)
        golden = t.transpose_faces_multi_tile(src_A, fmt, num_tiles=tile_cnt, tilize=False, input_dimensions=DIMS)
        golden = t.transpose_within_faces_multi_tile(golden, fmt, num_tiles=tile_cnt, untilize=False, input_dimensions=DIMS)
        res = torch.tensor(outcome.result, dtype=golden.dtype)
        diff = int((res.view(torch.int16) != golden.view(torch.int16)).sum())
        with open(os.path.join(OUT, f"{case}_content.txt"), "w") as f:
            f.write(f"content {case}: {res.numel()} words, {diff} differ from the transposed input words\n")
