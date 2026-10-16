# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""A mid-kernel dest-acc switch must leave the packer reading Dst the way a kernel built in that mode does.

The kernel is built in one dest-acc mode and switched to the other with ``_llk_set_fp32_dest_acc_``.
The enable arm (ELWMUL) pins rounding of a 32-bit Dst to bf16. The disable arm (A2D datacopy) pins the identity path
of a 16-bit Dst, which keeps NaN.
"""

from dataclasses import dataclass

import numpy as np
import pytest
import torch
from helpers.chip_architecture import ChipArchitecture, get_chip_architecture
from helpers.format_config import DataFormat
from helpers.llk_params import (
    BroadcastType,
    DataCopyType,
    DestAccumulation,
    DestSync,
    EltwiseBinaryReuseDestType,
    MathFidelity,
    MathOperation,
    PerfRunType,
    Tilize,
    Transpose,
)
from helpers.pack import pack_bfp16
from helpers.param_config import input_output_formats
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    ACC_TO_DEST,
    BROADCAST_TYPE,
    DATA_COPY_TYPE,
    DEST_INDEX,
    DEST_SYNC,
    LOOP_FACTOR,
    MATH_FIDELITY,
    MATH_OP,
    NUM_BLOCKS,
    NUM_FACES,
    NUM_FACES_C_DIM,
    NUM_FACES_R_DIM,
    NUM_TILES_IN_BLOCK,
    PERF_RUN_TYPE,
    REUSE_DEST_TYPE,
    TEST_FACE_DIMS,
    TILE_COUNT,
    TILIZE,
    UNPACK_TRANS_FACES,
    UNPACK_TRANS_WITHIN_FACE,
    TemplateParameter,
    generate_input_dim,
)

TILE = [32, 32]
N = TILE[0] * TILE[1]
SEED = 54381

# bf16 bit patterns placed in the first datums of A. B holds 1.0 at the same indices.
SPECIALS_A = [
    0x7FC0,  # +NaN
    0xFFC0,  # -NaN
    0x7F80,  # +inf
    0xFF80,  # -inf
    0x8000,  # -0
    0x0000,  # +0
    0x0001,  # smallest bf16 denormal
    0x0080,  # smallest bf16 normal
]
BF16_ONE = 0x3F80


@dataclass
class MID_KERNEL_DATACOPY(TemplateParameter):
    """Selects the A2D datacopy in the kernel instead of ELWMUL."""

    datacopy: bool = False

    def convert_to_cpp(self) -> str:
        return f"constexpr bool MID_KERNEL_DATACOPY = {str(self.datacopy).lower()};"


def _int16_from_bits(bits):
    return torch.tensor(
        [b - 0x10000 if b >= 0x8000 else b for b in bits], dtype=torch.int16
    )


def _random_normals_in_one_two(gen):
    # Exponent of 1.0 with a random 7 bit mantissa, so the exact product needs up to 16 mantissa bits.
    mantissa = torch.randint(0, 128, (N,), generator=gen, dtype=torch.int32)
    return (BF16_ONE | mantissa).to(torch.int16)


def _make_stimuli():
    gen = torch.Generator().manual_seed(SEED)
    a_bits = _random_normals_in_one_two(gen)
    b_bits = _random_normals_in_one_two(gen)
    a_bits[: len(SPECIALS_A)] = _int16_from_bits(SPECIALS_A)
    b_bits[: len(SPECIALS_A)] = BF16_ONE
    return a_bits.view(torch.bfloat16), b_bits.view(torch.bfloat16)


def _sent_bits(t):
    # pack_bfp16 serialises through float32, which quiets NaN payloads, so read back what reaches L1.
    sent = np.frombuffer(pack_bfp16(t), dtype=np.int16).copy()
    return torch.from_numpy(sent)


def _stimuli_config(formats, src_A, src_B):
    return StimuliConfig(
        src_A,
        formats.input_format,
        src_B,
        formats.input_format,
        formats.output_format,
        tile_count_A=1,
        tile_count_B=1,
        tile_count_res=1,
        num_faces=4,
        face_r_dim=16,
        tile_dimensions=TILE,
        use_dense_tile_dimensions=True,
    )


def _run_bits(configuration):
    configuration.run()
    raw = configuration.variant_stimuli.collect_raw_result_bytes(
        TestConfig.TENSIX_LOCATION
    )
    return torch.from_numpy(np.frombuffer(raw, dtype=np.int16)[:N].copy())


def _run_toggled(formats, dest_acc, datacopy, src_A, src_B):
    configuration = TestConfig(
        "sources/fp32_dest_acc_mid_kernel_pack_test.cpp",
        formats,
        templates=[MID_KERNEL_DATACOPY(datacopy)],
        variant_stimuli=_stimuli_config(formats, src_A, src_B),
        dest_acc=dest_acc,
    )
    return _run_bits(configuration)


def _run_native_datacopy(formats, dest_acc, src_A, src_B):
    # Same parameters test_pack_identity_exhaustive.py uses for an A2D datacopy, cut to one tile.
    configuration = TestConfig(
        "sources/eltwise_unary_datacopy_test.cpp",
        formats,
        templates=[
            generate_input_dim(TILE, TILE),
            TILIZE(Tilize.No),
            DATA_COPY_TYPE(DataCopyType.A2D),
            PERF_RUN_TYPE(PerfRunType.L1_TO_L1),
        ],
        runtimes=[
            DEST_INDEX(0),
            TILE_COUNT(1),
            NUM_FACES(4),
            NUM_BLOCKS(1),
            NUM_TILES_IN_BLOCK(1),
            LOOP_FACTOR(1),
        ],
        variant_stimuli=_stimuli_config(formats, src_A, src_B),
        dest_acc=dest_acc,
    )
    return _run_bits(configuration)


def _run_native_elwmul(formats, dest_acc, src_A, src_B):
    # Same templates test_eltwise_binary.py uses for one ELWMUL HiFi4 tile without broadcast.
    configuration = TestConfig(
        "sources/eltwise_binary_test.cpp",
        formats,
        templates=[
            MATH_FIDELITY(MathFidelity.HiFi4),
            BROADCAST_TYPE(BroadcastType.None_),
            MATH_OP(mathop=MathOperation.Elwmul),
            DEST_SYNC(DestSync.Half),
            REUSE_DEST_TYPE(reuse_dest_type=EltwiseBinaryReuseDestType.NONE),
            ACC_TO_DEST(False),
            PERF_RUN_TYPE(PerfRunType.L1_TO_L1),
        ],
        runtimes=[
            generate_input_dim(TILE, TILE, tile_dimensions=TILE),
            UNPACK_TRANS_FACES(Transpose.No),
            UNPACK_TRANS_WITHIN_FACE(Transpose.No),
            TILE_COUNT(1),
            NUM_TILES_IN_BLOCK(
                1, input_num_tiles_in_block=1, output_num_tiles_in_block=1
            ),
            NUM_BLOCKS(1, input_num_blocks=1, output_num_blocks=1),
            NUM_FACES_R_DIM(2),
            NUM_FACES_C_DIM(2),
            TEST_FACE_DIMS(face_r_dim=16),
            LOOP_FACTOR(1),
        ],
        variant_stimuli=_stimuli_config(formats, src_A, src_B),
        dest_acc=dest_acc,
    )
    return _run_bits(configuration)


def _is_nan(bits):
    b = bits.to(torch.int32)
    return ((b & 0x7F80) == 0x7F80) & ((b & 0x007F) != 0)


def _is_normal(bits):
    exp = (bits.to(torch.int32) >> 7) & 0xFF
    return (exp != 0) & (exp != 0xFF)


def _describe_mismatch(what, a, b, expected, got, considered):
    differ = (expected != got) & considered
    idx = torch.nonzero(differ).flatten().tolist()
    lines = [f"{what}: {len(idx)} of {int(considered.sum())} datums differ"]
    for i in idx[:8]:
        lines.append(
            f"  [{i:4d}] a=0x{int(a[i]) & 0xFFFF:04X} b=0x{int(b[i]) & 0xFFFF:04X}"
            f" expected=0x{int(expected[i]) & 0xFFFF:04X} got=0x{int(got[i]) & 0xFFFF:04X}"
        )
    return len(idx), "\n".join(lines)


@pytest.mark.skipif(
    get_chip_architecture()
    not in (ChipArchitecture.BLACKHOLE, ChipArchitecture.WORMHOLE),
    reason="_llk_set_fp32_dest_acc_ exists on Blackhole and Wormhole only.",
)
@pytest.mark.parametrize(
    "built_dest_acc",
    [
        pytest.param(DestAccumulation.No, id="enable"),
        pytest.param(DestAccumulation.Yes, id="disable"),
    ],
)
def test_fp32_dest_acc_mid_kernel_pack(built_dest_acc):
    formats = input_output_formats([DataFormat.Float16_b])[0]
    run_dest_acc = (
        DestAccumulation.Yes
        if built_dest_acc == DestAccumulation.No
        else DestAccumulation.No
    )

    src_A, src_B = _make_stimuli()
    a_bits = _sent_bits(src_A)
    b_bits = _sent_bits(src_B)
    everything = torch.ones(N, dtype=torch.bool)

    # The disable arm datacopies A, since ELWMUL does not leave a NaN in Dst for the packer to lose.
    datacopy = run_dest_acc == DestAccumulation.No
    got = _run_toggled(formats, built_dest_acc, datacopy, src_A, src_B)
    if datacopy:
        native = _run_native_datacopy(formats, run_dest_acc, src_A, src_B)
        native_name = "eltwise_unary_datacopy_test A2D"
    else:
        native = _run_native_elwmul(formats, run_dest_acc, src_A, src_B)
        native_name = "eltwise_binary_test ELWMUL HiFi4"

    failures = []
    if datacopy:
        # Report a NaN that came back as anything else; datums that stayed NaN count as matching.
        count, message = _describe_mismatch(
            "NaN sent but not NaN in the toggled kernel's output",
            a_bits,
            b_bits,
            a_bits,
            torch.where(_is_nan(got), a_bits, got),
            _is_nan(a_bits),
        )
        if count:
            failures.append(message)
    else:
        # bf16 x bf16 is exact in float32, so the 16 dropped bits decide the rounding.
        a = a_bits.view(torch.bfloat16).float()
        b = b_bits.view(torch.bfloat16).float()
        product = a * b
        rounded = product.to(torch.bfloat16).view(torch.int16)
        # The packer rounds exact ties away from zero (measured on P300, 3 of 1017 datums), so ties are left out.
        tie = (product.view(torch.int32) & 0xFFFF) == 0x8000
        # Non-normal datums are left to the native oracle.
        considered = (
            _is_normal(a_bits) & _is_normal(b_bits) & _is_normal(rounded) & ~tie
        )
        count, message = _describe_mismatch(
            "toggled kernel vs fp32 product rounded to nearest even bf16",
            a_bits,
            b_bits,
            rounded,
            got,
            considered,
        )
        if count:
            failures.append(message)

    count, message = _describe_mismatch(
        f"toggled kernel vs {native_name} built with dest_acc={run_dest_acc.name}",
        a_bits,
        b_bits,
        native,
        got,
        everything,
    )
    if count:
        failures.append(message)

    if failures:
        print("\n" + "\n".join(failures))
    assert not failures, "\n".join(failures)
