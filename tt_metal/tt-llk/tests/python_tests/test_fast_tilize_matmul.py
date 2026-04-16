# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Minimal tilize→matmul repro for BH fast-tilize conv2d integration issue.
Flow: fast_tilize(kt_dim-tile row-major activation) → matmul(act × weights) → pack result
Matches conv2d failure case: bf16, SyncHalf, kt_dim=2.
kt_dim=4 variant validates the test is correct (4-aligned should work).
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
    LOOP_FACTOR,
    NUM_FACES,
    TILE_COUNT,
    generate_input_dim,
)
from helpers.utils import passed_test

TILE_R = 32
TILE_C = 32


@parametrize(
    formats=[*input_output_formats([DataFormat.Float16_b], same=True)],
    dest_acc=[DestAccumulation.No],
    kt_dim=[1, 2],
    tilize_mode=["standard", "fast"],
)
def test_fast_tilize_matmul(formats, dest_acc, kt_dim, tilize_mode):
    if get_chip_architecture() != ChipArchitecture.BLACKHOLE:
        pytest.skip("BH only")

    # Activation: 1 tile row × kt_dim tile cols (row-major, will be tilized)
    act_dims = [TILE_R, kt_dim * TILE_C]
    # Weights: kt_dim tile rows × 1 tile col (already tilized)
    wt_dims = [kt_dim * TILE_R, TILE_C]

    # Use sequential A for deterministic, traceable values. Weights = all 1s.
    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=act_dims,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=wt_dims,
    )

    # Golden uses row-major A and B
    golden = get_golden_generator(MatmulGolden)(
        src_A,
        src_B,
        formats.output_format,
        MathFidelity.HiFi4,
        input_A_dimensions=act_dims,
        input_B_dimensions=wt_dims,
    )

    source_file = (
        "sources/tilize_matmul_test.cpp"
        if tilize_mode == "standard"
        else "sources/fast_tilize_matmul_test.cpp"
    )

    # Tilize weights for L1 — matmul unpack reads tilized tiles from buffer_B.
    # src_A stays row-major (fast-tilize hardware tilizes it).
    from helpers.tilize_untilize import tilize_block

    src_B_tilized = tilize_block(src_B, wt_dims, formats.input_format).flatten()

    cfg = TestConfig(
        source_file,
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
            src_B_tilized,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=kt_dim + 1,  # kt_dim tilized tiles + 1 matmul output tile
        ),
        dest_acc=dest_acc,
        compile_time_formats=True,
    )

    res = cfg.run().result
    tile_size = TILE_R * TILE_C

    # --- Diagnostic: verify tilize output against standalone fast-tilize golden ---
    if tilize_mode == "fast":
        from helpers.golden_generators import TilizeGolden

        tilize_golden = get_golden_generator(TilizeGolden)(
            src_A, act_dims, formats.output_format
        )
        for t in range(kt_dim):
            tile_data = res[t * tile_size : (t + 1) * tile_size]
            tile_tensor = torch.tensor(
                tile_data, dtype=format_dict[formats.output_format]
            )
            tile_golden = tilize_golden[t * tile_size : (t + 1) * tile_size]
            tilize_ok = passed_test(tile_golden, tile_tensor, formats.output_format)
            print(
                f"  Tilize tile {t}: {'PASS' if tilize_ok else 'FAIL'} "
                f"(first 4 vals: res={tile_data[:4]}, golden={tile_golden[:4].tolist()})"
            )

    matmul_result = res[kt_dim * tile_size :]
    assert len(matmul_result) == len(
        golden
    ), f"Result size: {len(matmul_result)} vs golden: {len(golden)}"
    res_tensor = torch.tensor(matmul_result, dtype=format_dict[formats.output_format])

    # --- Diagnostic: print per-face averages ---
    import numpy as np

    g = golden.float().numpy().flatten()
    r = res_tensor.float().numpy().flatten()
    if np.std(g) > 0 and np.std(r) > 0:
        pcc_val = np.corrcoef(g, r)[0, 1]
    else:
        pcc_val = 0.0
    # Reshape to 32x32 tile and print per-face average
    r_tile = np.array(r).reshape(32, 32) if len(r) == 1024 else None
    g_tile = np.array(g).reshape(32, 32) if len(g) == 1024 else None
    if r_tile is not None:
        print(f"  Matmul PCC: {pcc_val:.6f}")
        for fr in range(2):
            for fc in range(2):
                rv = r_tile[fr * 16 : (fr + 1) * 16, fc * 16 : (fc + 1) * 16].mean()
                gv = g_tile[fr * 16 : (fr + 1) * 16, fc * 16 : (fc + 1) * 16].mean()
                print(f"    Face[{fr},{fc}]: result={rv:.3f}, golden={gv:.3f}")
    else:
        print(f"  Matmul PCC: {pcc_val:.6f} (first 4: res={r[:4]}, golden={g[:4]})")

    assert passed_test(golden, res_tensor, formats.output_format)
