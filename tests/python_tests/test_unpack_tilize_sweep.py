# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from helpers.chip_architecture import ChipArchitecture, get_chip_architecture
from helpers.device import collect_results, write_stimuli_to_l1
from helpers.format_config import DataFormat
from helpers.golden_generators import TilizeGolden, get_golden_generator
from helpers.llk_params import (
    DestAccumulation,
    NarrowTile,
    StochasticRounding,
    Transpose,
    format_dict,
)
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import run_test
from helpers.utils import passed_test


@parametrize(
    test_name="unpack_tilize_sweep_test",
    formats=input_output_formats(
        [
            DataFormat.Float32,
            DataFormat.Float16,
            DataFormat.Float16_b,
            DataFormat.Bfp8_b,
        ]
    ),
    stoch_rnd_type=[
        StochasticRounding.No,
        StochasticRounding.Fpu,
        StochasticRounding.Pack,
        StochasticRounding.All,
    ],
    transpose=[Transpose.No],
    narrow_tile=[NarrowTile.No],
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    num_faces=[4, 2, 1],
    input_dimensions=[[32, 32], [64, 64], [32, 64], [32, 128], [128, 32]],
)
def test_unpack_tilize_comprehensive(
    test_name,
    formats,
    stoch_rnd_type,
    transpose,
    narrow_tile,
    dest_acc,
    num_faces,
    input_dimensions,
):
    """Comprehensive parameter sweep test for unpack_tilize operation."""

    # Get architecture for architecture-specific skips
    arch = get_chip_architecture()

    # Wormhole unpack_tilize has 0-loop for num_faces=1
    # File: tt_llk_wormhole_b0/llk_lib/llk_unpack_tilize.h:220
    # num_loops = num_faces / 2 → when num_faces=1, this is 1/2=0 (integer division)
    # Result: for (n=0; n<0; n++) never executes, no data unpacked, packer timeout
    if arch == ChipArchitecture.WORMHOLE and num_faces == 1:
        pytest.skip(
            "Wormhole LLK: num_loops = num_faces/2 = 0 when num_faces=1 "
            "(tt_llk_wormhole_b0/llk_lib/llk_unpack_tilize.h:220)"
        )

    # BFP8_b input format not supported by tilize unpacker
    # Tilize unpacker cannot correctly read row-major BFP8_b data with shared exponents
    # Note: BFP8_b input works in regular unpack mode (test_eltwise_unary_datacopy with tilize_en=false)
    if formats.input_format == DataFormat.Bfp8_b:
        pytest.skip(
            "BFP8_b input format not supported by tilize unpacker: "
            "cannot read row-major BFP8_b shared exponent data"
        )

    # Blackhole BFP8_b output fails for num_faces=1,2 due to hardcoded tilize packer values
    # Root cause: llk_pack.h has hardcoded MOP_OUTER_LOOP=2, PACK_INTF_SEL for 4-face tiles
    # The packer tries to process 2+ faces even when only 1-2 faces exist, corrupting BFP8_b output
    if (
        arch == ChipArchitecture.BLACKHOLE
        and formats.output_format == DataFormat.Bfp8_b
        and num_faces in [1, 2]
    ):
        pytest.skip(
            "Blackhole BFP8_b output fails for num_faces=1,2: tilize packer has hardcoded "
            "MOP_OUTER_LOOP=2 and PACK_INTF_SEL values that don't adapt to tiny tiles"
        )

    # Bfp8_b output + Stochastic Rounding Pack/All causes output corruption (value -508 becomes 0)
    if formats.output_format == DataFormat.Bfp8_b and stoch_rnd_type in [
        StochasticRounding.Pack,
        StochasticRounding.All,
    ]:
        pytest.skip(
            "Bfp8_b output with StochasticRounding.Pack/All causes the resulting value to be 0 when input is -508"
        )

    # Determine unpack_to_dest based on format
    unpack_to_dest = formats.input_format in [DataFormat.Int32, DataFormat.UInt32]

    # Generate test data (tilize is unary, only src_A is used)
    src_A, _, tile_cnt = generate_stimuli(
        formats.input_format,
        formats.input_format,
        input_dimensions=input_dimensions,
    )

    # Create dummy src_B for write_stimuli_to_l1 (required by interface but not used)
    src_B = torch.full((1024 * tile_cnt,), 0)

    torch_format = format_dict[formats.output_format]

    # Generate golden reference using TilizeGolden model
    tilize_function = get_golden_generator(TilizeGolden)
    golden_tensor = tilize_function(
        src_A,
        input_dimensions,
        formats.output_format,
        num_faces,
    )

    golden_tensor = golden_tensor.to(torch_format)

    # Build test config
    test_config = {
        "formats": formats,
        "testname": test_name,
        "tile_cnt": tile_cnt,
        "input_A_dimensions": input_dimensions,
        "input_B_dimensions": input_dimensions,
        "unpack_to_dest": unpack_to_dest,
        "stochastic_rnd": stoch_rnd_type,
        "unpack_transpose_faces": Transpose.No,
        "unpack_transpose_within_face": transpose,
        "dest_acc": dest_acc,
        "num_faces": num_faces,
        "narrow_tile": narrow_tile,
    }

    # Write stimuli to L1
    res_address = write_stimuli_to_l1(
        test_config,
        src_A,
        src_B,
        formats.input_format,
        formats.input_format,
        tile_count_A=tile_cnt,
        tile_count_B=tile_cnt,
        num_faces=num_faces,
    )

    # Execute the kernel
    run_test(test_config)

    # Collect results from L1 memory
    res_from_L1 = collect_results(
        formats, tile_count=tile_cnt, address=res_address, num_faces=num_faces
    )

    # Verify result size matches expected
    assert len(res_from_L1) == len(
        golden_tensor
    ), f"Result size mismatch: got {len(res_from_L1)}, expected {len(golden_tensor)}"

    # Convert to tensor for comparison
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    # Verify results match golden reference
    test_passed = passed_test(golden_tensor, res_tensor, formats.output_format)

    assert test_passed
