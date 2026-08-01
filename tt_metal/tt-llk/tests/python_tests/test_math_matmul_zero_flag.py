# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Zero-flag isolation probe for the FPU matmul (tt-metal#49924 / the ttsim modelling gap).
#
# Background: copy_init() folds in _configure_unary_preserve_zero_flag_state_(), which sets the Src
# zero-substitution flag ALU_ACC_CTRL_Zero_Flag_disabled_src=1 ("keep denormals") to preserve bf16
# -0.0 through a datacopy. That flag is a sticky math-ALU config. The FPU matmul (MVMUL) *reads* the
# flag, but per the ISA it only affects denormal flushing on SrcA/SrcB. Matmul is the one FP op whose
# LLK init never re-establishes the flag's baseline (reduce/transpose/datacopy all do), so a
# copy_init immediately followed by a matmul leaves the matmul running with the flag set.
#
# Two tests isolate the flag as the ONLY variable (identical stimuli, flag 0 vs 1):
#   * test_matmul_ignores_src_zero_flag       — NORMAL bf16 inputs. Proves the normal path is
#     unaffected (bit-identical output on silicon).
#   * test_matmul_src_zero_flag_denormals     — DENORMAL Src inputs. Characterizes what the flag
#     actually does for MVMUL (the case the ISA calls "not fully characterized").
#
# On ttsim (--run-simulator) the flag=1 run raises
#     UnsupportedFunctionality: tensix_matmul_op: ALU_ACC_CTRL_Zero_Flag_disabled_src=1 ... not modeled
# which is exactly the sanity/ttnn failure this reproduces in one kernel.

import pytest
import torch
from helpers.format_config import DataFormat
from helpers.golden_generators import MatmulGolden, get_golden_generator
from helpers.llk_params import (
    DestAccumulation,
    DestSync,
    MathFidelity,
    StochasticRounding,
    Transpose,
    format_dict,
)
from helpers.matmul_sweep import sweep_matmul
from helpers.param_config import input_output_formats
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import convert_to_l1_view, generate_face_matmul_data
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    CRK_TILE_DIMM,
    DEST_INDEX,
    DEST_SYNC,
    IN_TILE_DIMS,
    MATH_FIDELITY,
    NUM_BLOCKS,
    NUM_FACES,
    NUM_TILES_IN_BLOCK,
    PARTIAL_FACE,
    SET_SRC_ZERO_FLAG,
    STOCHASTIC_ROUNDING,
    THROTTLE_LEVEL,
    TILE_COUNT,
    UNPACK_TRANS_FACES,
    UNPACK_TRANS_WITHIN_FACE,
)
from helpers.tilize_untilize import tilize_block
from helpers.utils import passed_test

# One representative, normal-valued bf16 config: Float16_b in/out, no dest-acc, half dest-sync.
# Pick the first single-tile combination so the probe is as small and unambiguous as possible.
_BF16_COMBOS = sweep_matmul(
    input_output_formats([DataFormat.Float16_b]),
    [DestAccumulation.No],
    [StochasticRounding.No],
    [DestSync.Half],
    math_matmul=True,
)
_MATMUL_CONFIG = next(
    (c for c in _BF16_COMBOS if c.tile_dimensions.tile_cnt == 1),
    _BF16_COMBOS[0],
)
_MATH_FIDELITY = MathFidelity.HiFi4
_THROTTLE = 0
_NUM_BLOCKS = 1


def _make_stimuli(matmul_config, math_fidelity):
    """Generate the matmul stimuli + golden ONCE. Both flag runs must share identical inputs — the
    stimuli generators are random, so regenerating per run would compare two different matmuls."""
    formats = matmul_config.formats
    td = matmul_config.tile_dimensions
    flc = matmul_config.face_layout_config

    assert (
        flc.unpack_transpose_faces == Transpose.No
    ), "isolation probe intentionally uses the non-transpose config"

    # Normal-valued stimuli (no denormals), so the zero-flag's only documented effect (denormal
    # flush) cannot come into play — any difference between the two runs would be a real matmul
    # dependence on the flag.
    in0 = generate_face_matmul_data(
        num_faces=flc.num_faces_in0,
        stimuli_format=formats.input_format,
        input_dimensions=td.in0_dimensions,
        is_matrix_A=True,
        face_r_dim=(td.in0_tile_r_dim if td.in0_tile_r_dim < 16 else 16),
    )
    in1 = generate_face_matmul_data(
        num_faces=flc.num_faces_in1,
        stimuli_format=formats.input_format,
        input_dimensions=td.in1_dimensions,
        is_matrix_A=False,
    )

    generate_golden = get_golden_generator(MatmulGolden)
    golden = generate_golden(
        in0,
        in1,
        formats.output_format,
        math_fidelity,
        input_A_dimensions=td.in0_dimensions,
        input_B_dimensions=td.in1_dimensions,
        tilize=True,
        input_A_format=formats.input_format,
        input_B_format=formats.input_format,
    )
    golden_l1 = convert_to_l1_view(
        golden,
        (td.in0_dimensions[0], td.in1_dimensions[1]),
        tile_dimensions=[td.in0_tile_r_dim, td.in1_tile_c_dim],
    )

    in0_l1, in1_l1 = _tilize_to_l1(matmul_config, in0, in1)
    return in0_l1, in1_l1, golden_l1


def _tilize_to_l1(matmul_config, in0, in1):
    formats = matmul_config.formats
    td = matmul_config.tile_dimensions
    tilized_in0 = tilize_block(in0, dimensions=td.in0_dimensions, stimuli_format=formats.input_format)
    tilized_in1 = tilize_block(in1, dimensions=td.in1_dimensions, stimuli_format=formats.input_format)
    in0_l1 = convert_to_l1_view(
        tilized_in0, td.in0_dimensions, tile_dimensions=[td.in0_tile_r_dim, td.in0_tile_c_dim]
    )
    in1_l1 = convert_to_l1_view(
        tilized_in1, td.in1_dimensions, tile_dimensions=[td.in1_tile_r_dim, td.in1_tile_c_dim]
    )
    return in0_l1, in1_l1


def _run(matmul_config, math_fidelity, throttle, tilized_in0_l1_view, tilized_in1_l1_view, zero_flag):
    """Build + run math_matmul_test.cpp once with the given (pre-generated) stimuli."""
    formats = matmul_config.formats
    td = matmul_config.tile_dimensions
    flc = matmul_config.face_layout_config

    configuration = TestConfig(
        "sources/math_matmul_test.cpp",
        formats,
        templates=[
            STOCHASTIC_ROUNDING(matmul_config.stochastic_rnd),
            MATH_FIDELITY(math_fidelity),
            THROTTLE_LEVEL(throttle),
            DEST_SYNC(matmul_config.dest_sync),
            SET_SRC_ZERO_FLAG(zero_flag),
        ],
        runtimes=[
            TILE_COUNT(td.tile_cnt),
            NUM_BLOCKS(_NUM_BLOCKS),
            NUM_TILES_IN_BLOCK(td.tile_cnt),
            NUM_FACES(flc.num_faces, flc.num_faces_in0, flc.num_faces_in1),
            UNPACK_TRANS_FACES(flc.unpack_transpose_faces),
            UNPACK_TRANS_WITHIN_FACE(flc.unpack_transpose_faces),
            PARTIAL_FACE(
                partial_a=flc.partial_face_in0,
                partial_face_pack=flc.partial_face_pack,
                partial_b=flc.partial_face_in1,
                partial_face_math=flc.partial_face_math,
            ),
            CRK_TILE_DIMM(td.ct_dim, td.rt_dim, td.kt_dim),
            IN_TILE_DIMS(td.in0_tile_r_dim, td.in0_tile_c_dim, td.in1_tile_r_dim, td.in1_tile_c_dim),
            DEST_INDEX(matmul_config.dst_index),
        ],
        variant_stimuli=StimuliConfig(
            tilized_in0_l1_view.flatten(),
            formats.input_format,
            tilized_in1_l1_view.flatten(),
            formats.input_format,
            formats.output_format,
            tile_count_A=td.tile_cnt_in0,
            tile_count_B=td.tile_cnt_in1,
            tile_count_res=td.tile_cnt * _NUM_BLOCKS,
        ),
        dest_acc=matmul_config.dest_acc,
    )
    return configuration.run().result


@pytest.mark.nightly
def test_matmul_ignores_src_zero_flag():
    """A bf16 FPU matmul must produce identical output whether the Src zero-substitution flag is 0 or
    1 for NORMAL inputs. Bit-exact equality on silicon proves the normal path is unaffected."""
    in0_l1, in1_l1, golden_l1 = _make_stimuli(_MATMUL_CONFIG, _MATH_FIDELITY)

    res_default = _run(_MATMUL_CONFIG, _MATH_FIDELITY, _THROTTLE, in0_l1, in1_l1, zero_flag=False)
    res_flag_set = _run(_MATMUL_CONFIG, _MATH_FIDELITY, _THROTTLE, in0_l1, in1_l1, zero_flag=True)

    torch_format = format_dict[_MATMUL_CONFIG.formats.output_format]

    # Sanity: the baseline matmul is actually correct (both runs being wrong-but-equal would be a
    # false positive).
    assert passed_test(
        torch.tensor(golden_l1, dtype=torch_format),
        torch.tensor(res_default, dtype=torch_format),
        output_data_format=_MATMUL_CONFIG.formats.output_format,
    ), "baseline (zero-flag=0) matmul did not match golden"

    a = torch.as_tensor(res_default)
    b = torch.as_tensor(res_flag_set)
    assert a.shape == b.shape, f"length mismatch: {a.shape} vs {b.shape}"
    assert torch.equal(
        a, b
    ), "matmul output changed when ALU_ACC_CTRL_Zero_Flag_disabled_src was set — the flag is NOT a no-op for matmul"


# --- Denormal characterization -------------------------------------------------------------------
# bf16 = 1 sign / 8 exp / 7 mantissa. 0x0040 is a denormal (2^-127); 0x7880 is a normal (2^114) —
# deliberately well below bf16 max (2^128) to avoid any overflow/saturation confound. Their product
# is 2^-13, and a single-tile K=32 matmul accumulates 32 * 2^-13 = 2^-8 = 0.00390625 — a normal,
# exactly-representable bf16. So:
#   * if the FPU flushes the denormal Src (flag=0): each product is 0 -> result 0.0
#   * if it keeps it            (flag=1):           each product is 2^-13 -> result 0.00390625
# Comparing the two device outputs therefore *characterizes* whether the flag affects MVMUL at all.
_DENORMAL_BITS = 0x0040
_LARGE_BITS = 0x7880


def _bf16(bits):
    return torch.tensor([bits], dtype=torch.uint16).view(torch.bfloat16)[0]


def _make_denormal_stimuli(matmul_config):
    """Same layout as _make_stimuli, but in0 is a constant bf16 denormal and in1 a constant large
    normal (constants make the face/tile layout irrelevant, so only the values differ)."""
    formats = matmul_config.formats
    td = matmul_config.tile_dimensions
    flc = matmul_config.face_layout_config

    in0 = generate_face_matmul_data(
        num_faces=flc.num_faces_in0,
        stimuli_format=formats.input_format,
        input_dimensions=td.in0_dimensions,
        is_matrix_A=True,
        face_r_dim=(td.in0_tile_r_dim if td.in0_tile_r_dim < 16 else 16),
    )
    in1 = generate_face_matmul_data(
        num_faces=flc.num_faces_in1,
        stimuli_format=formats.input_format,
        input_dimensions=td.in1_dimensions,
        is_matrix_A=False,
    )
    # Overwrite with the constants (preserves the exact denormal / large bit patterns end-to-end —
    # tilize/l1_view are pure data reshapes, no arithmetic).
    in0 = torch.full_like(in0, _bf16(_DENORMAL_BITS))
    in1 = torch.full_like(in1, _bf16(_LARGE_BITS))
    return _tilize_to_l1(matmul_config, in0, in1)


@pytest.mark.nightly
def test_matmul_src_zero_flag_denormals():
    """Characterize whether ALU_ACC_CTRL_Zero_Flag_disabled_src affects MVMUL for *denormal* Src
    inputs (the case the ISA calls 'not fully characterized'). Prints both device outputs; a
    difference means the flag matters for matmul, equality means it is a no-op even for denormals."""
    in0_l1, in1_l1 = _make_denormal_stimuli(_MATMUL_CONFIG)

    res_default = _run(_MATMUL_CONFIG, _MATH_FIDELITY, _THROTTLE, in0_l1, in1_l1, zero_flag=False)
    res_flag_set = _run(_MATMUL_CONFIG, _MATH_FIDELITY, _THROTTLE, in0_l1, in1_l1, zero_flag=True)

    a = torch.as_tensor(res_default).flatten()
    b = torch.as_tensor(res_flag_set).flatten()
    print(f"\n[denormal] flag=0 (flush) first 4: {a[:4].tolist()}")
    print(f"[denormal] flag=1 (keep)  first 4: {b[:4].tolist()}")

    # HW-characterized behavior (Wormhole n150 / Blackhole p150b both behave the same):
    #   flag=0 -> the FPU flushes the denormal Src operands -> matmul output is exactly 0.
    #   flag=1 -> it keeps them             -> matmul output is nonzero (~2^-8 for these stimuli).
    # i.e. the flag genuinely affects MVMUL for denormals — it is NOT a no-op. This matches the ISA's
    # FlushDenormals=!bit model (the same one ttsim already implements for MOVA2D/ELWADD), so ttsim
    # can and should model it for tensix_matmul_op instead of aborting. It also means a copy_init that
    # leaves the keep-denormals state set going into a matmul changes the denormal result on silicon.
    assert torch.count_nonzero(a) == 0, f"flag=0 (flush) should zero denormal contributions, got {a[:4].tolist()}"
    assert torch.count_nonzero(b) == b.numel(), f"flag=1 (keep) should keep denormals nonzero, got {b[:4].tolist()}"
    assert not torch.equal(a, b), "the zero-flag must change the denormal matmul result"
