# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Three-stage tilize→matmul repro tests for BH fast-tilize conv2d integration.
Test 1: Pure matmul (both inputs pre-tilized by Python)
Test 2: Standard tilize A on compute + pre-tilized B → matmul
Test 3: Fast-tilize A on compute + pre-tilized B → matmul
"""

import pytest
import torch
from helpers.chip_architecture import ChipArchitecture, get_chip_architecture
from helpers.format_config import DataFormat
from helpers.golden_generators import MathFidelity, MatmulGolden, get_golden_generator
from helpers.llk_params import DestAccumulation, format_dict
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    CRK_TILE_DIMM,
    LOOP_FACTOR,
    MATH_FIDELITY,
    NUM_FACES,
    TILE_COUNT,
    generate_input_dim,
)
from helpers.tilize_untilize import tilize_block
from helpers.utils import passed_test

TILE_R = 32
TILE_C = 32


def make_stimuli(formats, act_dims, wt_dims, stimulus_type):
    """Generate stimuli based on type: 'ones_A', 'ones_B', or 'random'."""
    if stimulus_type == "ones_A":
        src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
            stimuli_format_A=formats.input_format,
            input_dimensions_A=act_dims,
            stimuli_format_B=formats.input_format,
            input_dimensions_B=wt_dims,
            const_face=True,
            const_value_A=1,
        )
    elif stimulus_type == "ones_B":
        src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
            stimuli_format_A=formats.input_format,
            input_dimensions_A=act_dims,
            stimuli_format_B=formats.input_format,
            input_dimensions_B=wt_dims,
            const_face=True,
            const_value_B=1,
        )
    elif stimulus_type == "const_3_7":
        src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
            stimuli_format_A=formats.input_format,
            input_dimensions_A=act_dims,
            stimuli_format_B=formats.input_format,
            input_dimensions_B=wt_dims,
            const_face=True,
            const_value_A=3,
            const_value_B=7,
        )
    else:  # random
        src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
            stimuli_format_A=formats.input_format,
            input_dimensions_A=act_dims,
            stimuli_format_B=formats.input_format,
            input_dimensions_B=wt_dims,
        )
    return src_A, tile_cnt_A, src_B, tile_cnt_B


# ============================================================
# Test 1: Pure matmul (both inputs pre-tilized)
# Uses existing matmul_test.cpp — known working
# ============================================================
@parametrize(
    formats=[*input_output_formats([DataFormat.Float16_b], same=True)],
    dest_acc=[DestAccumulation.No],
    kt_dim=[2, 4],
    stimulus=["ones_A", "ones_B", "random"],
)
def test_1_matmul_pretilized(formats, dest_acc, kt_dim, stimulus):
    if get_chip_architecture() != ChipArchitecture.BLACKHOLE:
        pytest.skip("BH only")

    act_dims = [TILE_R, kt_dim * TILE_C]
    wt_dims = [kt_dim * TILE_R, TILE_C]

    src_A, tile_cnt_A, src_B, tile_cnt_B = make_stimuli(
        formats, act_dims, wt_dims, stimulus
    )

    golden = get_golden_generator(MatmulGolden)(
        src_A,
        src_B,
        formats.output_format,
        MathFidelity.HiFi4,
        input_A_dimensions=act_dims,
        input_B_dimensions=wt_dims,
        tilize=True,
        input_A_format=formats.input_format,
        input_B_format=formats.input_format,
    )

    # Tilize both inputs (matmul reads tilized tiles from L1)
    tilized_A = tilize_block(src_A, act_dims, formats.input_format).flatten()
    tilized_B = tilize_block(src_B, wt_dims, formats.input_format).flatten()

    cfg = TestConfig(
        "sources/matmul_test.cpp",
        formats,
        templates=[MATH_FIDELITY(MathFidelity.HiFi4)],
        runtimes=[
            NUM_FACES(),
            TILE_COUNT(1),  # 1 output tile
            CRK_TILE_DIMM(1, 1, kt_dim),
        ],
        variant_stimuli=StimuliConfig(
            tilized_A,
            formats.input_format,
            tilized_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=1,
        ),
        dest_acc=dest_acc,
    )

    res = cfg.run().result
    res_tensor = torch.tensor(res, dtype=format_dict[formats.output_format])
    assert passed_test(golden, res_tensor, formats.output_format)


# ============================================================
# Test 2a: Standard tilize only — verify tilize output in L1
# ============================================================
@parametrize(
    formats=[*input_output_formats([DataFormat.Float16_b], same=True)],
    dest_acc=[DestAccumulation.No],
    kt_dim=[2, 4],
    stimulus=["ones_A", "ones_B", "random"],
)
def test_2a_std_tilize_only(formats, dest_acc, kt_dim, stimulus):
    if get_chip_architecture() != ChipArchitecture.BLACKHOLE:
        pytest.skip("BH only")

    act_dims = [TILE_R, kt_dim * TILE_C]
    wt_dims = [kt_dim * TILE_R, TILE_C]

    src_A, tile_cnt_A, src_B, tile_cnt_B = make_stimuli(
        formats, act_dims, wt_dims, stimulus
    )

    from helpers.golden_generators import TILE_DIMENSIONS, TilizeGolden
    from helpers.llk_params import DestSync
    from helpers.param_config import get_num_blocks_and_num_tiles_in_block
    from helpers.test_variant_parameters import NUM_BLOCKS, NUM_TILES_IN_BLOCK

    golden = get_golden_generator(TilizeGolden)(src_A, act_dims, formats.output_format)
    num_blocks, num_tiles_in_block = get_num_blocks_and_num_tiles_in_block(
        DestSync.Half, None, formats, act_dims, TILE_DIMENSIONS
    )

    cfg = TestConfig(
        "sources/unpack_tilize_test.cpp",
        formats,
        templates=[],
        runtimes=[
            generate_input_dim(act_dims, act_dims),
            TILE_COUNT(tile_cnt_A),
            NUM_FACES(4),
            NUM_BLOCKS(num_blocks),
            NUM_TILES_IN_BLOCK(num_tiles_in_block),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=tile_cnt_A,
        ),
        dest_acc=dest_acc,
    )

    res = cfg.run().result
    res_tensor = torch.tensor(res, dtype=format_dict[formats.output_format])
    assert len(res_tensor) == len(golden), f"Size: {len(res_tensor)} vs {len(golden)}"
    assert passed_test(golden, res_tensor, formats.output_format, print_pcc=True)


# ============================================================
# Test 2b: Standard tilize A on compute + pre-tilized B → matmul
# A is row-major in buffer_A, hardware tilizes to buffer_Res
# B is pre-tilized in buffer_B
# ============================================================
@parametrize(
    formats=[*input_output_formats([DataFormat.Float16_b], same=True)],
    dest_acc=[DestAccumulation.No],
    kt_dim=[2, 4],
    stimulus=["ones_A", "ones_B", "random"],
)
def test_2_std_tilize_matmul(formats, dest_acc, kt_dim, stimulus):
    if get_chip_architecture() != ChipArchitecture.BLACKHOLE:
        pytest.skip("BH only")

    act_dims = [TILE_R, kt_dim * TILE_C]
    wt_dims = [kt_dim * TILE_R, TILE_C]

    src_A, tile_cnt_A, src_B, tile_cnt_B = make_stimuli(
        formats, act_dims, wt_dims, stimulus
    )

    # Golden: row-major matmul, then tilize to match pack output
    from helpers.tilize_untilize import tilize

    golden_rm = get_golden_generator(MatmulGolden)(
        src_A,
        src_B,
        formats.output_format,
        MathFidelity.HiFi4,
        input_A_dimensions=act_dims,
        input_B_dimensions=wt_dims,
    )
    golden = tilize(golden_rm).to(format_dict[formats.output_format])

    # B pre-tilized for matmul
    tilized_B = tilize_block(src_B, wt_dims, formats.input_format).flatten()

    cfg = TestConfig(
        "sources/std_tilize_matmul_repro.cpp",
        formats,
        templates=[],
        runtimes=[
            generate_input_dim(act_dims, wt_dims, block_ct_dim=kt_dim),
            TILE_COUNT(1),
            LOOP_FACTOR(1),
            NUM_FACES(4),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            tilized_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=kt_dim + 1,  # kt_dim tilized A tiles + 1 matmul output
        ),
        dest_acc=dest_acc,
        compile_time_formats=True,
    )

    res = cfg.run().result
    tile_size = TILE_R * TILE_C
    matmul_result = res[kt_dim * tile_size :]  # skip tilized A tiles
    assert len(matmul_result) == len(
        golden
    ), f"Size: {len(matmul_result)} vs {len(golden)}"
    res_tensor = torch.tensor(matmul_result, dtype=format_dict[formats.output_format])
    assert passed_test(golden, res_tensor, formats.output_format, print_pcc=True)


# ============================================================
# Test 3a: Fast-tilize only — verify output matches standard tilize
# ============================================================
@parametrize(
    formats=[*input_output_formats([DataFormat.Float16_b], same=True)],
    dest_acc=[DestAccumulation.No],
    kt_dim=[2, 4],
    stimulus=["ones_A", "ones_B", "random"],
)
def test_3a_fast_tilize_only(formats, dest_acc, kt_dim, stimulus):
    if get_chip_architecture() != ChipArchitecture.BLACKHOLE:
        pytest.skip("BH only")

    act_dims = [TILE_R, kt_dim * TILE_C]
    wt_dims = [kt_dim * TILE_R, TILE_C]

    src_A, tile_cnt_A, src_B, tile_cnt_B = make_stimuli(
        formats, act_dims, wt_dims, stimulus
    )

    from helpers.golden_generators import TilizeGolden

    golden = get_golden_generator(TilizeGolden)(src_A, act_dims, formats.output_format)

    cfg = TestConfig(
        "sources/fast_tilize_only_repro.cpp",
        formats,
        templates=[],
        runtimes=[
            generate_input_dim(act_dims, wt_dims, block_ct_dim=kt_dim),
            TILE_COUNT(1),
            LOOP_FACTOR(1),
            NUM_FACES(4),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=kt_dim,
        ),
        dest_acc=dest_acc,
        compile_time_formats=True,
    )

    res = cfg.run().result
    res_tensor = torch.tensor(res, dtype=format_dict[formats.output_format])
    assert len(res_tensor) == len(golden), f"Size: {len(res_tensor)} vs {len(golden)}"
    assert passed_test(golden, res_tensor, formats.output_format, print_pcc=True)


# ============================================================
# Test 3a_metal: Fast-tilize only — Metal API call pattern
# Same as 3a but uses init_unit_dim=(full_dim>=4)?4:full_dim
# and per-fill decomposition matching tilize.h BH path
# ============================================================
@parametrize(
    formats=[*input_output_formats([DataFormat.Float16_b], same=True)],
    dest_acc=[DestAccumulation.No],
    kt_dim=[2, 4],
    stimulus=["ones_A", "ones_B", "random"],
)
def test_3a_metal_api_pattern(formats, dest_acc, kt_dim, stimulus):
    if get_chip_architecture() != ChipArchitecture.BLACKHOLE:
        pytest.skip("BH only")

    act_dims = [TILE_R, kt_dim * TILE_C]
    wt_dims = [kt_dim * TILE_R, TILE_C]

    src_A, tile_cnt_A, src_B, tile_cnt_B = make_stimuli(
        formats, act_dims, wt_dims, stimulus
    )

    from helpers.golden_generators import TilizeGolden

    golden = get_golden_generator(TilizeGolden)(src_A, act_dims, formats.output_format)

    cfg = TestConfig(
        "sources/fast_tilize_metal_api_repro.cpp",
        formats,
        templates=[],
        runtimes=[
            generate_input_dim(act_dims, wt_dims, block_ct_dim=kt_dim),
            TILE_COUNT(1),
            LOOP_FACTOR(1),
            NUM_FACES(4),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=kt_dim,
        ),
        dest_acc=dest_acc,
        compile_time_formats=True,
    )

    res = cfg.run().result
    res_tensor = torch.tensor(res, dtype=format_dict[formats.output_format])
    assert len(res_tensor) == len(golden), f"Size: {len(res_tensor)} vs {len(golden)}"
    assert passed_test(golden, res_tensor, formats.output_format, print_pcc=True)


# ============================================================
# Test 3b: Fast-tilize A on compute + pre-tilized B → matmul
# A is row-major in buffer_A, fast-tilize to buffer_Res
# B is pre-tilized in buffer_B
# ============================================================
@parametrize(
    formats=[*input_output_formats([DataFormat.Float16_b], same=True)],
    dest_acc=[DestAccumulation.No],
    kt_dim=[2, 4],
    stimulus=["ones_A", "ones_B", "random"],
)
def test_3b_fast_tilize_matmul(formats, dest_acc, kt_dim, stimulus):
    if get_chip_architecture() != ChipArchitecture.BLACKHOLE:
        pytest.skip("BH only")

    act_dims = [TILE_R, kt_dim * TILE_C]
    wt_dims = [kt_dim * TILE_R, TILE_C]

    src_A, tile_cnt_A, src_B, tile_cnt_B = make_stimuli(
        formats, act_dims, wt_dims, stimulus
    )

    # Golden: row-major matmul, then tilize to match pack output
    from helpers.tilize_untilize import tilize

    golden_rm = get_golden_generator(MatmulGolden)(
        src_A,
        src_B,
        formats.output_format,
        MathFidelity.HiFi4,
        input_A_dimensions=act_dims,
        input_B_dimensions=wt_dims,
    )
    golden = tilize(golden_rm).to(format_dict[formats.output_format])

    # B pre-tilized for matmul
    tilized_B = tilize_block(src_B, wt_dims, formats.input_format).flatten()

    cfg = TestConfig(
        "sources/fast_tilize_matmul_repro.cpp",
        formats,
        templates=[],
        runtimes=[
            generate_input_dim(act_dims, wt_dims, block_ct_dim=kt_dim),
            TILE_COUNT(1),
            LOOP_FACTOR(1),
            NUM_FACES(4),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            tilized_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=kt_dim + 1,  # kt_dim tilized A tiles + 1 matmul output
        ),
        dest_acc=dest_acc,
        compile_time_formats=True,
    )

    res = cfg.run().result
    tile_size = TILE_R * TILE_C
    matmul_result = res[kt_dim * tile_size :]  # skip tilized A tiles
    assert len(matmul_result) == len(
        golden
    ), f"Size: {len(matmul_result)} vs {len(golden)}"
    res_tensor = torch.tensor(matmul_result, dtype=format_dict[formats.output_format])
    assert passed_test(golden, res_tensor, formats.output_format, print_pcc=True)


# ============================================================
# Helper for multi-iteration tilize → matmul tests
# ============================================================
def _run_tilize_matmul_niter(
    source_file,
    formats,
    dest_acc,
    kt_dim,
    num_iters,
    stimulus,
):
    row_dims = [TILE_R, kt_dim * TILE_C]
    wt_dims = [kt_dim * TILE_R, TILE_C]

    # Generate num_iters different A rows and 1 B
    src_A_rows = []
    src_A0, tile_cnt_A0, src_B, tile_cnt_B = make_stimuli(
        formats, row_dims, wt_dims, stimulus
    )
    src_A_rows.append(src_A0)
    for i in range(1, num_iters):
        if stimulus == "ones_A":
            src_A_rows.append(torch.full_like(src_A0, float(i + 1)))
        else:
            a, _, _, _ = generate_stimuli(
                stimuli_format_A=formats.input_format,
                input_dimensions_A=row_dims,
                stimuli_format_B=formats.input_format,
                input_dimensions_B=wt_dims,
            )
            src_A_rows.append(a)

    src_A_combined = torch.cat([a.flatten() for a in src_A_rows])

    from helpers.tilize_untilize import tilize

    goldens = []
    for src_Ai in src_A_rows:
        g_rm = get_golden_generator(MatmulGolden)(
            src_Ai,
            src_B,
            formats.output_format,
            MathFidelity.HiFi4,
            input_A_dimensions=row_dims,
            input_B_dimensions=wt_dims,
        )
        goldens.append(tilize(g_rm).to(format_dict[formats.output_format]))

    tilized_B = tilize_block(src_B, wt_dims, formats.input_format).flatten()
    tile_count_res = num_iters * (kt_dim + 1)

    cfg = TestConfig(
        source_file,
        formats,
        templates=[],
        runtimes=[
            generate_input_dim(row_dims, wt_dims, block_ct_dim=kt_dim),
            TILE_COUNT(1),
            LOOP_FACTOR(num_iters),
            NUM_FACES(4),
        ],
        variant_stimuli=StimuliConfig(
            src_A_combined,
            formats.input_format,
            tilized_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A0 * num_iters,
            tile_count_B=tile_cnt_B,
            tile_count_res=tile_count_res,
        ),
        dest_acc=dest_acc,
        compile_time_formats=True,
    )

    res = cfg.run().result
    tile_size = TILE_R * TILE_C

    for i in range(num_iters):
        base = i * (kt_dim + 1)
        mm_result = res[(base + kt_dim) * tile_size : (base + kt_dim + 1) * tile_size]
        assert len(mm_result) == len(
            goldens[i]
        ), f"Iter {i}: size {len(mm_result)} vs {len(goldens[i])}"
        res_tensor = torch.tensor(mm_result, dtype=format_dict[formats.output_format])
        ok = passed_test(goldens[i], res_tensor, formats.output_format, print_pcc=True)
        assert ok, f"Iter {i} FAILED"


# ============================================================
# Test 4: Multi-iteration standard tilize → matmul
# ============================================================
@parametrize(
    formats=[*input_output_formats([DataFormat.Float16_b], same=True)],
    dest_acc=[DestAccumulation.No],
    kt_dim=[2, 4],
    num_iters=[2],
    stimulus=["ones_A", "ones_B", "random"],
)
def test_4_std_tilize_matmul_niter(formats, dest_acc, kt_dim, num_iters, stimulus):
    if get_chip_architecture() != ChipArchitecture.BLACKHOLE:
        pytest.skip("BH only")
    _run_tilize_matmul_niter(
        "sources/std_tilize_matmul_2iter_repro.cpp",
        formats,
        dest_acc,
        kt_dim,
        num_iters,
        stimulus,
    )


# ============================================================
# Test 5: Multi-iteration fast tilize → matmul
# Same as test_4 but using fast tilize. This is the conv2d pattern.
# ============================================================
@parametrize(
    formats=[*input_output_formats([DataFormat.Float16_b], same=True)],
    dest_acc=[DestAccumulation.No],
    kt_dim=[2, 4],
    num_iters=[2],
    stimulus=["ones_A", "ones_B", "random"],
)
def test_5_fast_tilize_matmul_niter(formats, dest_acc, kt_dim, num_iters, stimulus):
    if get_chip_architecture() != ChipArchitecture.BLACKHOLE:
        pytest.skip("BH only")
    _run_tilize_matmul_niter(
        "sources/fast_tilize_matmul_2iter_repro.cpp",
        formats,
        dest_acc,
        kt_dim,
        num_iters,
        stimulus,
    )


# ============================================================
# Test 6: Multi-iteration standard tilize → matmul with accumulation
# Conv2d pattern: iter 0 = tilize+mm, iter 1+ = tilize + reload + mm(accumulate)
# Golden = sum of all A_row[i] × B
# ============================================================
def _run_tilize_matmul_accum(
    source_file,
    formats,
    dest_acc,
    kt_dim,
    num_iters,
    stimulus,
):
    row_dims = [TILE_R, kt_dim * TILE_C]
    wt_dims = [kt_dim * TILE_R, TILE_C]

    src_A_rows = []
    src_A0, tile_cnt_A0, src_B, tile_cnt_B = make_stimuli(
        formats, row_dims, wt_dims, stimulus
    )
    src_A_rows.append(src_A0)
    # All iterations use the same A data so golden = num_iters × (A × B)
    for i in range(1, num_iters):
        src_A_rows.append(src_A0.clone())

    src_A_combined = torch.cat([a.flatten() for a in src_A_rows])

    # Golden: accumulated matmul = sum of A_row[i] × B
    from helpers.tilize_untilize import tilize

    accumulated = None
    for src_Ai in src_A_rows:
        g_rm = get_golden_generator(MatmulGolden)(
            src_Ai,
            src_B,
            formats.output_format,
            MathFidelity.HiFi4,
            input_A_dimensions=row_dims,
            input_B_dimensions=wt_dims,
        )
        if accumulated is None:
            accumulated = g_rm.float()
        else:
            accumulated = accumulated + g_rm.float()
    golden = tilize(accumulated.to(format_dict[formats.output_format])).to(
        format_dict[formats.output_format]
    )

    tilized_B = tilize_block(src_B, wt_dims, formats.input_format).flatten()

    # Result layout: buffer_Res[0..KT-1] = tilized A scratch, buffer_Res[KT] = accumulated result
    tile_count_res = kt_dim + 1

    cfg = TestConfig(
        source_file,
        formats,
        templates=[],
        runtimes=[
            generate_input_dim(row_dims, wt_dims, block_ct_dim=kt_dim),
            TILE_COUNT(1),
            LOOP_FACTOR(num_iters),
            NUM_FACES(4),
        ],
        variant_stimuli=StimuliConfig(
            src_A_combined,
            formats.input_format,
            tilized_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A0 * num_iters,
            tile_count_B=tile_cnt_B,
            tile_count_res=tile_count_res,
        ),
        dest_acc=dest_acc,
        compile_time_formats=True,
    )

    res = cfg.run().result
    tile_size = TILE_R * TILE_C

    # Dump ALL result buffer tiles for diagnostics
    face_size = 256
    one_iter_matmul = None  # value of one A×B iteration for reference
    for t in range(kt_dim + 1):
        tile_data = res[t * tile_size : (t + 1) * tile_size]
        tile_tensor = torch.tensor(tile_data, dtype=format_dict[formats.output_format])
        label = f"Res[{t}]" if t < kt_dim else f"Res[{t}] (ACCUM RESULT)"
        face_vals = []
        for f in range(4):
            face = tile_tensor[f * face_size : (f + 1) * face_size]
            face_vals.append(torch.unique(face).tolist())
        print(f"  {label}: faces={face_vals}")

    mm_result = res[kt_dim * tile_size : (kt_dim + 1) * tile_size]
    assert len(mm_result) == len(golden), f"Size: {len(mm_result)} vs {len(golden)}"
    res_tensor = torch.tensor(mm_result, dtype=format_dict[formats.output_format])

    # Read L1 probes from last 3 words of Res[KT-1] tile
    diag_base = (kt_dim - 1) * tile_size + tile_size - 3
    probe_pre = res[diag_base]
    probe_post_tilize = res[diag_base + 1]
    probe_post_mm = res[diag_base + 2]
    print(
        f"  L1_PROBE (last iter): pre_tilize=0x{int(probe_pre):04x} post_tilize_pack=0x{int(probe_post_tilize):04x} post_mm_pack=0x{int(probe_post_mm):04x}"
    )
    # bf16: 0x4040=3.0, 0x40e0=7.0
    for name, val in [
        ("pre_tilize", probe_pre),
        ("post_tilize_pack", probe_post_tilize),
        ("post_mm_pack", probe_post_mm),
    ]:
        decoded = (
            "3.0" if int(val) == 0x4040 else "7.0" if int(val) == 0x40E0 else f"{val}"
        )
        print(f"    {name} = {decoded}")

    # Per-face comparison
    print(f"  RESULT[0:4] = {res_tensor[:4].tolist()}")
    print(f"  GOLDEN[0:4] = {golden[:4].tolist()}")
    for f in range(4):
        face_r = res_tensor[f * face_size : (f + 1) * face_size]
        face_g = golden[f * face_size : (f + 1) * face_size]
        r_unique = torch.unique(face_r).tolist()
        g_unique = torch.unique(face_g).tolist()
        match = "OK" if r_unique == g_unique else "MISMATCH"
        print(f"  Face {f}: result={r_unique}, golden={g_unique} [{match}]")

    ok = passed_test(golden, res_tensor, formats.output_format, print_pcc=True)
    assert ok, "Accumulated matmul FAILED"


@parametrize(
    formats=[*input_output_formats([DataFormat.Float16_b], same=True)],
    dest_acc=[DestAccumulation.Yes],
    kt_dim=[2, 4],
    num_iters=[1, 2, 3, 8],
    stimulus=["ones_A", "const_3_7", "random"],
)
def test_6_std_tilize_matmul_accum(formats, dest_acc, kt_dim, num_iters, stimulus):
    if get_chip_architecture() != ChipArchitecture.BLACKHOLE:
        pytest.skip("BH only")
    _run_tilize_matmul_accum(
        "sources/std_tilize_matmul_accum_repro.cpp",
        formats,
        dest_acc,
        kt_dim,
        num_iters,
        stimulus,
    )


# ============================================================
# Test 7: Multi-iteration fast tilize → matmul with accumulation
# Same as test_6 but using fast tilize. Should reproduce conv2d bug.
# ============================================================
@parametrize(
    formats=[*input_output_formats([DataFormat.Float16_b], same=True)],
    dest_acc=[DestAccumulation.Yes],
    kt_dim=[2, 4],
    num_iters=[1, 2, 3, 8],
    stimulus=["ones_A", "const_3_7", "random"],
)
def test_7_fast_tilize_matmul_accum(formats, dest_acc, kt_dim, num_iters, stimulus):
    if get_chip_architecture() != ChipArchitecture.BLACKHOLE:
        pytest.skip("BH only")
    _run_tilize_matmul_accum(
        "sources/fast_tilize_matmul_accum_repro.cpp",
        formats,
        dest_acc,
        kt_dim,
        num_iters,
        stimulus,
    )
