# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import re
import subprocess
from dataclasses import dataclass

import pytest
import torch
from conftest import blackhole_only
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import (
    ApproximationMode,
    DestAccumulation,
    MathOperation,
    format_dict,
)
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import TemplateParameter
from test_eltwise_unary_sfpu import build_eltwise_unary_sfpu_config

pytestmark = blackhole_only


@pytest.mark.parametrize(
    "input_format,output_format",
    [
        (DataFormat.Float16, DataFormat.Float16),
        pytest.param(
            DataFormat.Float16, DataFormat.Float16_b, marks=pytest.mark.nightly
        ),
        pytest.param(DataFormat.Float16, DataFormat.Float32, marks=pytest.mark.nightly),
        pytest.param(DataFormat.Float32, DataFormat.Float16, marks=pytest.mark.nightly),
        # FP8 in L1 expands to Float16 in registers on Blackhole.
        pytest.param(
            DataFormat.Fp8_e4m3, DataFormat.Float16, marks=pytest.mark.nightly
        ),
        pytest.param(
            DataFormat.Fp8_e4m3, DataFormat.Float16_b, marks=pytest.mark.nightly
        ),
        pytest.param(
            DataFormat.Fp8_e4m3, DataFormat.Float32, marks=pytest.mark.nightly
        ),
        pytest.param(
            DataFormat.Fp8_e4m3, DataFormat.Fp8_e4m3, marks=pytest.mark.nightly
        ),
    ],
)
@pytest.mark.parametrize(
    "dest_acc",
    [
        DestAccumulation.No,
        pytest.param(DestAccumulation.Yes, marks=pytest.mark.nightly),
    ],
)
@pytest.mark.parametrize(
    "dimensions", [[32, 32], pytest.param([128, 256], marks=pytest.mark.nightly)]
)
def test_sfpu_fp16_dest(input_format, output_format, dest_acc, dimensions):
    # Add1 requires numerical loads/stores, unlike sign-bit-only Abs/Neg.
    # Both operands and the expected result are exactly representable.
    count = dimensions[0] * dimensions[1]
    values = torch.full((count,), 1.5, dtype=format_dict[input_format])
    tiles = count // 1024
    config = build_eltwise_unary_sfpu_config(
        InputOutputFormat(input_format, output_format),
        dest_acc,
        ApproximationMode.No,
        MathOperation.Add1,
        dimensions,
        values,
        values,
        tiles,
        tiles,
    )
    assert config.dest_acc == dest_acc
    result = torch.tensor(config.run().result, dtype=format_dict[output_format])
    torch.testing.assert_close(result, torch.full_like(result, 2.5), rtol=0, atol=0)


@dataclass
class SfpuDestFormat(TemplateParameter):
    raw_srcb: int
    writer: int
    dest_format: str
    wait_probe: bool = False
    switch_banks: bool = False
    poison_cache: bool = False
    initial_srcb: int = 255

    def convert_to_cpp(self):
        return (
            f"[[maybe_unused]] constexpr unsigned INITIAL_SRCB = {self.initial_srcb};\n"
            f"[[maybe_unused]] constexpr bool POISON_CACHE = {str(self.poison_cache).lower()};\n"
            f"constexpr unsigned RAW_SRCB = {self.raw_srcb};\n"
            f"constexpr unsigned FORMAT_WRITER = {self.writer};\n"
            f"[[maybe_unused]] constexpr bool SWITCH_BANKS = {str(self.switch_banks).lower()};\n"
            f"[[maybe_unused]] constexpr bool WAIT_PROBE = {str(self.wait_probe).lower()};\n"
            f"[[maybe_unused]] constexpr auto DEST_FORMAT = DataFormat::{self.dest_format};"
        )


@pytest.mark.parametrize(
    "writer",
    [0, *[pytest.param(i, marks=pytest.mark.nightly) for i in range(1, 5)]],
    ids=["init", "both", "both-skip", "srcb", "srcb-skip"],
)
@pytest.mark.parametrize(
    "raw_srcb", [26, 10, 9, 24], ids=["Fp8_e4m3", "Lf8", "UInt16", "UInt32"]
)
def test_sfpu_raw_srcb_format(raw_srcb, writer):
    dest_format = "Float16" if raw_srcb in (26, 10) else "Float16_b"
    values = torch.zeros(1024, dtype=torch.int32)
    config = TestConfig(
        "sources/sfpu_dest_format_test.cpp",
        InputOutputFormat(DataFormat.Int32, DataFormat.Int32),
        templates=[SfpuDestFormat(raw_srcb, writer, dest_format)],
        variant_stimuli=StimuliConfig(
            values,
            DataFormat.Int32,
            values,
            DataFormat.Int32,
            DataFormat.Int32,
            tile_count_B=1,
            tile_count_res=2,
        ),
    )
    result = [int(x) & 0xFFFFFFFF for x in config.run().result]
    expected_format = 1 if raw_srcb in (26, 10) else 5
    assert (
        result[1024] == expected_format
    ), f"raw SrcB {raw_srcb}: SFPU format {result[1024]}, expected {expected_format}; Add1 bits {result[0]:08x}"
    assert result[0] == 0x40200000


@pytest.mark.parametrize(
    "writer", [1, 2, 3, 4], ids=["both", "both-skip", "srcb", "srcb-skip"]
)
@pytest.mark.parametrize("dest_format,raw_srcb", [("Float16", 5), ("Float16_b", 1)])
def test_sfpu_reconfig_drains_pending_work(writer, dest_format, raw_srcb):
    values = torch.zeros(1024, dtype=torch.int32)
    config = TestConfig(
        "sources/sfpu_dest_format_test.cpp",
        InputOutputFormat(DataFormat.Int32, DataFormat.Int32),
        templates=[SfpuDestFormat(raw_srcb, writer, dest_format, True)],
        variant_stimuli=StimuliConfig(
            values,
            DataFormat.Int32,
            values,
            DataFormat.Int32,
            DataFormat.Int32,
            tile_count_B=1,
            tile_count_res=2,
            twos_complement=True,
        ),
    )
    selector_index, stall_index, rmw2_index, rmw3_index, opcode_mask_index = (
        1024,
        1026,
        1027,
        1028,
        1029,
    )
    result = [int(x) & 0xFFFFFFFF for x in config.run().result]
    assert result[selector_index] == raw_srcb
    assert result[0] == 0x40200000

    # Timing can hide a missing drain; check the executed reconfig function, excluding startup.
    elf_path = (
        TestConfig.ARTEFACTS_DIR / config.test_name / config.variant_id / "elf/math.elf"
    )
    disassembly = subprocess.check_output(
        [
            TestConfig.OBJDUMP,
            "-d",
            "--disassemble=configure_math_format_for_test",
            str(elf_path),
        ],
        text=True,
    )
    words = [
        int(word, 16)
        for word in re.findall(
            r"^\s*[0-9a-f]+:\s+([0-9a-f]{8})\b", disassembly, re.MULTILINE
        )
    ]
    instructions = [(w >> 2) | ((w & 3) << 30) for w in words]
    stalls = [i for i, insn in enumerate(instructions) if insn == result[stall_index]]
    writes = [
        i
        for i, insn in enumerate(instructions)
        if (insn & result[opcode_mask_index])
        in (result[rmw2_index], result[rmw3_index])
    ]
    assert (
        stalls and writes and min(stalls) < min(writes)
    ), "SrcB reconfig must drain WAIT_SFPU before its first RMWCIB"


@pytest.mark.parametrize(
    "poison_cache", [False, True], ids=["bank-switch", "cleanup-external-write"]
)
def test_sfpu_format_cache_tracks_configuration_banks(poison_cache):
    if TestConfig.TEST_TARGET.run_simulator and not poison_cache:
        pytest.skip("ttsim does not implement configuration-bank-1 MMIO reads")
    values = torch.zeros(1024, dtype=torch.int32)
    config = TestConfig(
        "sources/sfpu_dest_format_test.cpp",
        InputOutputFormat(DataFormat.Int32, DataFormat.Int32),
        templates=[
            SfpuDestFormat(
                1,
                4,
                "Float16",
                switch_banks=not poison_cache,
                poison_cache=poison_cache,
            )
        ],
        variant_stimuli=StimuliConfig(
            values,
            DataFormat.Int32,
            values,
            DataFormat.Int32,
            DataFormat.Int32,
            tile_count_B=1,
            tile_count_res=2,
        ),
    )
    result = [int(x) & 0xFFFFFFFF for x in config.run().result]
    expected = 5 if poison_cache else 1
    assert (
        result[1024] == expected
    ), "stale SFPU format cache after bank switch or cleanup"
    if not poison_cache:
        assert result[0] == 0x40200000


@pytest.mark.parametrize(
    "raw_srcb",
    [9, 6, 7, 15, 4, 0],
    ids=["UInt16", "Bfp8_b", "Bfp4_b", "Bfp2_b", "Tf32", "Float32"],
)
@pytest.mark.parametrize(
    "writer", [3, pytest.param(4, marks=pytest.mark.nightly)], ids=["srcb", "srcb-skip"]
)
def test_sfpu_same_family_reconfig(raw_srcb, writer):
    values = torch.zeros(1024, dtype=torch.int32)
    config = TestConfig(
        "sources/sfpu_dest_format_test.cpp",
        InputOutputFormat(DataFormat.Int32, DataFormat.Int32),
        templates=[SfpuDestFormat(raw_srcb, writer, "Float16_b", initial_srcb=5)],
        variant_stimuli=StimuliConfig(
            values,
            DataFormat.Int32,
            values,
            DataFormat.Int32,
            DataFormat.Int32,
            tile_count_B=1,
            tile_count_res=2,
        ),
    )
    result = [int(x) & 0xFFFFFFFF for x in config.run().result]
    assert result[1024] == 5
    assert result[0] == 0x40200000


@pytest.mark.nightly
@pytest.mark.parametrize(
    "srca,dest_format,srcb", [(1, "Float16_b", 5), (5, "Float16", 1)]
)
def test_sfpu_srca_reconfig_preserves_srcb_family(srca, dest_format, srcb):
    values = torch.zeros(1024, dtype=torch.int32)
    config = TestConfig(
        "sources/sfpu_dest_format_test.cpp",
        InputOutputFormat(DataFormat.Int32, DataFormat.Int32),
        templates=[SfpuDestFormat(srca, 5, dest_format, initial_srcb=srcb)],
        variant_stimuli=StimuliConfig(
            values,
            DataFormat.Int32,
            values,
            DataFormat.Int32,
            DataFormat.Int32,
            tile_count_B=1,
            tile_count_res=2,
        ),
    )
    result = [int(x) & 0xFFFFFFFF for x in config.run().result]
    assert result[1024] == srcb
    assert result[0] == 0x40200000
