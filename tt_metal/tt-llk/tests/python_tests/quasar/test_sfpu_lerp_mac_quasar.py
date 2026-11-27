# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import struct
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
    MATH_OP,
    NUM_FACES,
    TEST_FACE_DIMS,
    TILE_COUNT,
    UNPACKER_ENGINE_SEL,
)
from helpers.utils import passed_test

TILE_ELEMENTS = 32 * 32
NUM_FACES_PER_TILE = 4
OPERAND_TILES = 3
# One operand triple per Dest section: three sections walk the Dest banks 0 -> 1 -> 0 and rerun
# the op's init over the state the previous section left.
NUM_SECTIONS = 3

# Quasar opcodes (top byte of the instruction word, ckernel_ops.h) the mac recording is made of.
_TENSIX_MNEMONICS = {
    0x04: "ttreplay",
    0x70: "sfpload",
    0x72: "sfpstore",
    0x84: "sfpmad",
    0x8E: "sfpstochrnd",
}
# MAC_REPLAY_SLOT of ckernel_sfpu_mac.h.
_MAC_REPLAY_SLOT = 0
# mac_init's two-row replay body; ckernel_sfpu_mac.h states the same counts as mac_replay_len.
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
    """Three operands of NUM_SECTIONS tiles each, with ranges that keep a row left unprocessed, or
    computed from another row or face, far outside the comparison tolerance."""
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


def _tensix_instructions(math_elf) -> list:
    """Tensix instruction words of the ELF's executable sections, in address order.

    In the RISC-V stream a Tensix instruction is its word rotated left by two, which leaves the low
    two bits other than the 0b11 of a 32-bit RISC-V instruction.
    """
    data = math_elf.read_bytes()
    (shoff,) = struct.unpack_from("<I", data, 0x20)
    shentsize, shnum = struct.unpack_from("<HH", data, 0x2E)
    words = []
    for index in range(shnum):
        _, sh_type, sh_flags, _, offset, size = struct.unpack_from(
            "<6I", data, shoff + index * shentsize
        )
        if sh_type == 1 and sh_flags & 0x4:  # SHT_PROGBITS, SHF_EXECINSTR
            words += struct.unpack_from(f"<{size // 4}I", data, offset)
    return [(word >> 2) | ((word & 0x3) << 30) for word in words if word & 0x3 != 0x3]


def _mac_record_windows(math_elf) -> list:
    """(length, mnemonics) of every recording mac_init makes in the math ELF: a load-mode replay at
    MAC_REPLAY_SLOT that does not execute while loading. It captures the next ``length`` Tensix
    instructions."""
    tensix = _tensix_instructions(math_elf)
    windows = []
    for index, word in enumerate(tensix):
        if _TENSIX_MNEMONICS.get(word >> 24) != "ttreplay":
            continue
        # TTREPLAY: load_mode is bit 0, execute_while_loading bit 1, the length bits 4..13 and the
        # start slot bits 14..23. A recording the compiler adds on its own executes while loading.
        load_mode, execute_while_loading = word & 0x1, (word >> 1) & 0x1
        start = (word >> 14) & 0x3FF
        if not load_mode or execute_while_loading or start != _MAC_REPLAY_SLOT:
            continue
        length = (word >> 4) & 0x3FF
        body = [
            _TENSIX_MNEMONICS.get(body_word >> 24, f"opcode {body_word >> 24:#04x}")
            for body_word in tensix[index + 1 : index + 1 + length]
        ]
        windows.append((length, body))
    return windows


@pytest.mark.quasar
@parametrize(
    mathop=[MathOperation.SfpuLerp, MathOperation.SfpuMac],
    formats_dest_acc=get_valid_formats_dest_acc(),
)
def test_sfpu_lerp_mac_quasar(mathop, formats_dest_acc):
    """lerp and mac as the Compute API runs them: operands in Dest tiles 0, 1, 2, init after they
    are staged, result over tile 0. mac's recorded replay body is also read back from the math ELF,
    since its length is a hand-kept constant."""
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
            MATH_OP(mathop=mathop),
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
    # A 16-bit Dest result is exact: the kernels round to nearest even, and one step off is inside
    # the default tolerance.
    assert passed_test(
        golden_tensor,
        res_tensor,
        formats.output_format,
        max_ulp=0 if formats.output_format == DataFormat.Float16_b else None,
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
