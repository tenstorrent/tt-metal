# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import pytest
from fast_untilize_common import (
    FAST_UNTILIZE_CT_DIMS,
    FAST_UNTILIZE_RT_DIMS,
    fast_untilize_dest_acc_modes,
    fast_untilize_formats,
)
from helpers.chip_architecture import ChipArchitecture, get_chip_architecture
from helpers.constraints import get_valid_dest_accumulation_modes
from helpers.data_format_inference import (
    dest_acc_for,
    infer_data_formats,
    unpacks_to_dest,
)
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import (
    TILE_DIMENSIONS,
    UntilizeGolden,
    get_golden_generator,
)
from helpers.llk_params import (
    BlocksCalculationAlgorithm,
    DestAccumulation,
    DestSync,
    PackUntilizeBenchmark,
    PerfRunType,
)
from helpers.param_config import (
    get_num_blocks_and_num_tiles_in_block,
    input_output_formats,
    max_dest_tiles,
    parametrize,
)
from helpers.perf.core import create_test_or_perf_config
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import StimuliSpec, generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    DEST_SYNC,
    LOOP_FACTOR,
    NUM_FACES,
    PACK_UNTILIZE_BENCHMARK,
    TILE_DST_CT_OFFSET,
    generate_input_dim,
)
from helpers.utils import passed_test, result_tensor

STANDARD_PACK_UNTILIZE_FORMATS = input_output_formats(
    [
        DataFormat.Float16_b,
        DataFormat.Float32,
        DataFormat.Int32,
        DataFormat.Bfp8_b,
        DataFormat.Fp8_e4m3,
    ]
)
FUNCTIONAL_PACK_UNTILIZE_DIMS = [(2, 2), (1, 4), (4, 4), (1, 2)]


@dataclass(frozen=True)
class PackUntilizeCase:
    formats: InputOutputFormat
    dest_acc: DestAccumulation
    benchmark: PackUntilizeBenchmark
    full_rt_dim: int
    full_ct_dim: int
    compile_time_formats: bool


def _valid_pack_untilize_formats(formats):
    if formats.output_format == DataFormat.Bfp8_b:
        return False
    if (formats.input_format == DataFormat.Int32) ^ (
        formats.output_format == DataFormat.Int32
    ):
        return False
    return True


def pack_untilize_cases():
    """Ordered union of valid functional, standard-perf, and baseline-perf rows."""
    cases = []

    def append(case):
        if case not in cases:
            cases.append(case)

    for formats in STANDARD_PACK_UNTILIZE_FORMATS:
        if not _valid_pack_untilize_formats(formats):
            continue
        for dest_acc in get_valid_dest_accumulation_modes(formats):
            for rt_dim, ct_dim in FUNCTIONAL_PACK_UNTILIZE_DIMS:
                append(
                    PackUntilizeCase(
                        formats,
                        dest_acc,
                        PackUntilizeBenchmark.Functional,
                        rt_dim,
                        ct_dim,
                        False,
                    )
                )

    for formats in STANDARD_PACK_UNTILIZE_FORMATS:
        if not _valid_pack_untilize_formats(formats):
            continue
        for rt_dim in range(1, 9):
            for ct_dim in range(1, 9):
                append(
                    PackUntilizeCase(
                        formats,
                        dest_acc_for(formats.input_format),
                        PackUntilizeBenchmark.Standard,
                        rt_dim,
                        ct_dim,
                        False,
                    )
                )

    if get_chip_architecture() == ChipArchitecture.BLACKHOLE:
        for formats in fast_untilize_formats():
            for dest_acc in fast_untilize_dest_acc_modes(formats):
                for rt_dim in FAST_UNTILIZE_RT_DIMS:
                    for ct_dim in FAST_UNTILIZE_CT_DIMS:
                        append(
                            PackUntilizeCase(
                                formats,
                                dest_acc,
                                PackUntilizeBenchmark.FastUntilizeBaseline,
                                rt_dim,
                                ct_dim,
                                True,
                            )
                        )
    return cases


def _largest_divisor_at_most(value, cap):
    for candidate in range(min(value, cap), 0, -1):
        if value % candidate == 0:
            return candidate
    return 1


def pack_untilize_block_ct_dim(case):
    if case.benchmark == PackUntilizeBenchmark.FastUntilizeBaseline:
        return _largest_divisor_at_most(case.full_ct_dim, max_dest_tiles(case.dest_acc))

    max_block_dim = 4 if case.formats.input_format.is_32_bit() else 8
    return _largest_divisor_at_most(case.full_ct_dim, max_block_dim)


@parametrize(
    case=pack_untilize_cases,
    #  TODO add DestSync::Full tests when we have a solution for the static_assert in _llk_pack_untilize_init_ that requires block_ct_dim to be less or equal to 8,
    #  which is currently a limitation for testing DestSync::Full with the Untilize blocks calculation algorithm.
    dest_sync=[DestSync.Half],
    tile_dst_ct_offset=[0],  # Non-zero offsets are tracked in #1449
)
def test_pack_untilize(
    case,
    dest_sync,
    tile_dst_ct_offset,
    *,
    is_perf=False,
    perf_report=None,
    run_types=None,
    loop_factor=1,
    block_ct_dim=None,
):
    formats = case.formats
    dest_acc = case.dest_acc
    benchmark = case.benchmark
    compile_time_formats = case.compile_time_formats
    input_dimensions = [case.full_rt_dim * 32, case.full_ct_dim * 32]
    if run_types is None:
        run_types = [PerfRunType.L1_TO_L1]

    if not is_perf and loop_factor != 1:
        raise ValueError("Functional pack-untilize runs require loop_factor=1")
    if is_perf and perf_report is None:
        raise ValueError("perf_report must be provided when is_perf=True")

    if TestConfig.WITH_COVERAGE and input_dimensions == [64, 512]:
        pytest.skip(
            "Skipping large dimension test in coverage mode, check issue: #1063 on TT-LLK repo"
        )

    if get_chip_architecture() != ChipArchitecture.BLACKHOLE and (
        formats.input_format == DataFormat.Fp8_e4m3
        or formats.output_format == DataFormat.Fp8_e4m3
    ):
        pytest.skip(
            "Pack Untilize does not support Fp8_e4m3 on non-Blackhole architectures"
        )

    if formats.output_format == DataFormat.Bfp8_b:
        pytest.skip("Pack Untilize does not support Bfp8_b format")

    if (formats.input_format == DataFormat.Int32) ^ (
        formats.output_format == DataFormat.Int32
    ):
        pytest.skip("Pack Untilize does not support mixing Int32 with other formats")

    data_formats = infer_data_formats(
        formats.input_format,
        formats.output_format,
        dest_acc,
        False,
    )

    # Handling a hardware limitation: cannot convert 8-bit exponent datums to Float16 without storing them as intermediate Float32 in dest register.
    # For wormhole architecture, gasket cannot perform this conversion and packer takes input Float32 (from dest register) converting to Float16_A.
    # For blackhole architecture, gasket is able to convert Float32 to Float16_A before packing (reduces work on packer).`
    if (
        formats.input_format == DataFormat.Float16
        and data_formats.pack_src.is_32_bit()
        and dest_acc == DestAccumulation.No
    ):
        pytest.skip(
            "Due to hardware limitation, cannot convert 8-bit exponent datums to Float16 without storing them as intermediate Float32 in dest register. Therefore using dest_acc=No is not supported in this case."
        )

    # TODO: Checkout issue #1405 on tt-llk.
    if (
        get_chip_architecture() == ChipArchitecture.WORMHOLE
        and formats.input_format
        in (DataFormat.Float16_b, DataFormat.Float16, DataFormat.Bfp8_b)
        and formats.output_format == DataFormat.Float32
        and dest_acc == DestAccumulation.No
        and input_dimensions == [64, 512]
    ):
        pytest.skip("Wormhole pack_untilize does not support this format combination.")

    sfpu_false_spec = StimuliSpec.uniform(low=0.0, high=1.0)
    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        spec_A=sfpu_false_spec,
        spec_B=sfpu_false_spec,
    )

    generate_golden = get_golden_generator(UntilizeGolden)

    golden_tensor = generate_golden(src_A, formats.output_format, input_dimensions)

    functional_unpack_to_dest = unpacks_to_dest(formats.input_format, dest_acc)

    # _llk_pack_untilize_init_ has a static_assert that checks if block_ct_dim is less or equal to 8.
    # TODO: Update this logic to accept more than 8 tiles per block if the static_assert changes in the future.
    if block_ct_dim is None:
        block_ct_dim = (
            pack_untilize_block_ct_dim(case)
            if benchmark != PackUntilizeBenchmark.Functional
            else get_num_blocks_and_num_tiles_in_block(
                dest_sync,
                dest_acc,
                formats,
                input_dimensions,
                TILE_DIMENSIONS,
                BlocksCalculationAlgorithm.Untilize,
            )[1]
        )

    test_config_kwargs = {
        "test_name": "sources/pack_untilize_test.cpp",
        "formats": formats,
        "templates": [
            generate_input_dim(
                input_dimensions,
                input_dimensions,
                block_ct_dim,
            ),
            DEST_SYNC(dest_sync),
            TILE_DST_CT_OFFSET(tile_dst_ct_offset),
            PACK_UNTILIZE_BENCHMARK(benchmark),
            NUM_FACES(4),
            LOOP_FACTOR(loop_factor),
        ],
        "runtimes": [],
        "variant_stimuli": StimuliConfig(
            None if is_perf else src_A,
            formats.input_format,
            None if is_perf else src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=tile_cnt_A,
            sfpu=False,
        ),
        "dest_acc": dest_acc,
        "unpack_to_dest": (
            # Match perf_fast_untilize: its Float32 baseline unpacks through
            # SrcA and performs A2D datacopy instead of unpacking to Dest.
            False
            if benchmark == PackUntilizeBenchmark.FastUntilizeBaseline
            else functional_unpack_to_dest
        ),
        "compile_time_formats": compile_time_formats,
    }

    configuration = create_test_or_perf_config(
        is_perf=is_perf,
        run_types=run_types,
        test_config_kwargs=test_config_kwargs,
    )
    if is_perf:
        configuration.run(perf_report)
        return

    res_from_L1 = configuration.run().result

    res_tensor = result_tensor(res_from_L1, formats.output_format, golden_tensor)

    assert passed_test(
        golden_tensor, res_tensor, formats.output_format
    ), "Assert against golden failed"
