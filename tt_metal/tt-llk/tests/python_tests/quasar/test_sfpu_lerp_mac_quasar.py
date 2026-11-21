# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import subprocess
from collections import Counter

import pytest
import torch
from helpers.format_config import DataFormat
from helpers.golden_generators import TernarySFPUGolden, get_golden_generator
from helpers.llk_params import (
    DataCopyType,
    DestAccumulation,
    MathOperation,
    UnpackerEngine,
    format_dict,
)
from helpers.param_config import (
    QuasarSfpuVariant,
    generate_quasar_sfpu_format_variants,
    input_output_formats,
    parametrize,
)
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    DATA_COPY_TYPE,
    DEST_INDEX,
    DEST_SYNC,
    NUM_FACES,
    SFPU_TERNARY_OP,
    TEST_FACE_DIMS,
    TILE_COUNT,
    UNPACKER_ENGINE_SEL,
)
from helpers.utils import passed_test

TILE_ELEMENTS = 32 * 32
NUM_FACES_PER_TILE = 4
OPERAND_TILES = 3
# One operand triple per Dest section. Three sections walk the Dest banks 0 -> 1 -> 0, and every
# section after the first reruns the op's init over the state the previous section left behind.
NUM_SECTIONS = 3

# mac_init records two Dest rows per replay: six loads, two multiply-adds, a rounding pair when
# Dest is 16-bit, and two stores. ckernel_sfpu_mac.h states the same counts as mac_replay_len.
_MAC_RECORD_BODY = {
    DestAccumulation.Yes: Counter(sfpload=6, sfpmad=2, sfpstore=2),
    DestAccumulation.No: Counter(sfpload=6, sfpmad=2, sfpstochrnd=2, sfpstore=2),
}


def get_valid_formats_dest_acc():
    """16-bit Dest and 32-bit Dest, the latter staged both by Unpack-to-Dest and by FPU datacopy."""
    formats = input_output_formats(
        [DataFormat.Float16_b, DataFormat.Float32], same=True
    )
    return generate_quasar_sfpu_format_variants(
        MathOperation.SfpuLerp, formats, full_format_route_sweep=True
    )


def _operands(mathop: MathOperation, torch_format) -> tuple:
    """Three operands of NUM_SECTIONS tiles each, every element distinct from its neighbours.

    The kernels write the result over operand tile 0, and they pair Dest rows. The ranges keep
    the result away from the first operand and from any other row's result by far more than the
    comparison tolerance, so a row left unprocessed, or computed from another row or face, fails.
    """
    count = NUM_SECTIONS * TILE_ELEMENTS
    generator = torch.Generator().manual_seed(65376)

    def uniform(low, high):
        values = low + (high - low) * torch.rand(count, generator=generator)
        return values.to(torch_format)

    src_a = uniform(-8.0, 8.0)
    src_b = uniform(-8.0, 8.0)
    if mathop == MathOperation.SfpuLerp:
        # Weight: inside (0, 1) and extrapolating past 1.
        src_c = uniform(0.25, 1.75)
    else:
        # Addend of mac.
        src_c = uniform(16.0, 32.0)
    return src_a, src_b, src_c


def _section_interleaved(src_a, src_b, src_c) -> torch.Tensor:
    """buffer_A layout: for each section its three operand tiles, back to back."""
    tiles = []
    for section in range(NUM_SECTIONS):
        window = slice(section * TILE_ELEMENTS, (section + 1) * TILE_ELEMENTS)
        tiles += [src_a[window], src_b[window], src_c[window]]
    return torch.cat(tiles)


def _mac_record_windows(math_elf) -> list:
    """Mnemonics of every recording replay window in the math ELF.

    A recording replay captures the next ``length`` Tensix instructions; RISC-V instructions the
    compiler schedules among them are not part of the window.
    """
    objdump = (TestConfig.TOOL_PATH / "riscv-tt-elf-objdump").absolute()
    listing = subprocess.run(
        [str(objdump), "-d", "--no-show-raw-insn", str(math_elf)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    tensix = []
    for line in listing.splitlines():
        fields = line.split("\t")
        if len(fields) < 2:
            continue
        mnemonic = fields[1].strip()
        if mnemonic.startswith(("tt", "sfp")):
            tensix.append((mnemonic, fields[2].strip() if len(fields) > 2 else ""))

    windows = []
    for index, (mnemonic, operands) in enumerate(tensix):
        if mnemonic != "ttreplay":
            continue
        # ttreplay start, length, last, set_mutex, execute_while_loading, load_mode
        fields = [int(field, 0) for field in operands.split(",")]
        if fields[-1] != 1:
            continue
        length = fields[1]
        body = [name for name, _ in tensix[index + 1 : index + 1 + length]]
        windows.append((length, body))
    return windows


@pytest.mark.quasar
@parametrize(
    mathop=[MathOperation.SfpuLerp, MathOperation.SfpuMac],
    formats_dest_acc=get_valid_formats_dest_acc(),
)
def test_sfpu_lerp_mac_quasar(mathop, formats_dest_acc):
    """
    lerp(a, b, c) = a + c * (b - a) and mac(a, b, c) = a * b + c on Quasar, as the Compute API runs
    them: operands in Dest tiles 0, 1, 2, init after the operands are staged, all four faces, result
    over tile 0.

    Both kernels handle two Dest rows per iteration and advance Dest from the second row's store, so
    the check is on every element of every face, with data that tells the rows and faces apart. For
    mac the recorded replay body is also read back from the math ELF: its length is a hand-kept
    constant that must equal what the compiler emits for the body.
    """
    format_variant = formats_dest_acc
    assert isinstance(format_variant, QuasarSfpuVariant)
    formats = format_variant.formats
    dest_acc = format_variant.dest_acc
    unpack_to_dest = format_variant.unpack_to_dest

    src_a, src_b, src_c = _operands(mathop, format_dict[formats.input_format])
    src_A = _section_interleaved(src_a, src_b, src_c)
    tile_cnt_A = NUM_SECTIONS * OPERAND_TILES

    generate_golden = get_golden_generator(TernarySFPUGolden)
    golden_tensor = generate_golden(
        mathop, src_a, src_b, src_c, 0, formats.output_format
    )

    configuration = TestConfig(
        "sources/quasar/sfpu_lerp_mac_quasar_test.cpp",
        formats,
        templates=[
            SFPU_TERNARY_OP(mathop),
            DATA_COPY_TYPE(DataCopyType.A2D),
            UNPACKER_ENGINE_SEL(
                UnpackerEngine.UnpDest if unpack_to_dest else UnpackerEngine.UnpA
            ),
            DEST_SYNC(),
        ],
        runtimes=[
            TILE_COUNT(tile_cnt_A),
            NUM_FACES(NUM_FACES_PER_TILE),
            TEST_FACE_DIMS(),
            DEST_INDEX(0),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            torch.zeros_like(src_a),
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=NUM_SECTIONS,
            tile_count_res=NUM_SECTIONS,
            num_faces=NUM_FACES_PER_TILE,
        ),
        unpack_to_dest=unpack_to_dest,
        dest_acc=dest_acc,
    )
    format_variant.apply_formats(configuration.formats_config)

    res_from_L1 = configuration.run().result

    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    res_tensor = torch.tensor(res_from_L1, dtype=format_dict[formats.output_format])
    assert passed_test(
        golden_tensor, res_tensor, formats.output_format
    ), "Assert against golden failed"

    if mathop == MathOperation.SfpuMac:
        math_elf = (
            TestConfig.ARTEFACTS_DIR
            / configuration.test_name
            / configuration.variant_id
            / "elf"
            / "math.elf"
        )
        expected = _MAC_RECORD_BODY[dest_acc]
        windows = _mac_record_windows(math_elf)
        assert windows, f"no recording replay in {math_elf}"
        for length, body in windows:
            assert (
                Counter(body) == expected and body[-1] == "sfpstore"
            ), f"mac_init records {length} instructions, but they are not its two-row body: {body}"
