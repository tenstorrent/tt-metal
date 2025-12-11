# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from helpers.constraints import (
    get_valid_dest_accumulation_modes,
    get_valid_math_fidelities,
)
from helpers.device import (
    collect_results,
    write_stimuli_to_l1,
)
from helpers.format_config import DataFormat
from helpers.golden_generators import (
    BroadcastGolden,
    EltwiseBinaryGolden,
    get_golden_generator,
)
from helpers.llk_params import (
    BroadcastType,
    ImpliedMathFormat,
    MathOperation,
    format_dict,
)
from helpers.param_config import (
    generate_unary_input_dimensions,
    input_output_formats,
    parametrize,
)
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import BootMode, run_test
from helpers.utils import passed_test


@pytest.mark.quasar
@parametrize(
    test_name="eltwise_binary_broadcast_quasar_test",
    formats=input_output_formats(
        [
            DataFormat.Float16_b,
            DataFormat.Float16,
        ],
    ),
    dest_acc=lambda formats: get_valid_dest_accumulation_modes(formats),
    mathop=[
        MathOperation.Elwadd,
        MathOperation.Elwsub,
        MathOperation.Elwmul,
    ],
    broadcast_type=[
        BroadcastType.Column,
        BroadcastType.Row,
        BroadcastType.Scalar,
    ],
    math_fidelity=lambda formats, mathop: get_valid_math_fidelities(formats, mathop),
    implied_math_format=[
        ImpliedMathFormat.No,
        ImpliedMathFormat.Yes,
    ],
    input_dimensions=lambda dest_acc: generate_unary_input_dimensions(dest_acc),
)
def test_eltwise_binary_broadcast_quasar(
    test_name,
    formats,
    dest_acc,
    mathop,
    broadcast_type,
    math_fidelity,
    implied_math_format,
    input_dimensions,
    boot_mode=BootMode.DEFAULT,
):

    src_A, src_B, tile_cnt = generate_stimuli(
        formats.input_format,
        formats.input_format,
        input_dimensions=input_dimensions,
    )

    generate_broadcast_golden = get_golden_generator(BroadcastGolden)
    bcast_src_B_tensor = generate_broadcast_golden(
        broadcast_type,
        src_B,
        formats.output_format,
        num_faces=4,
        tile_cnt=tile_cnt,
        face_r_dim=16,
    )

    generate_golden = get_golden_generator(EltwiseBinaryGolden)
    golden_tensor = generate_golden(
        mathop,
        src_A,
        bcast_src_B_tensor,
        formats.output_format,
        math_fidelity,
    )

    test_config = {
        "formats": formats,
        "testname": test_name,
        "mathop": mathop,
        "math_fidelity": math_fidelity,
        "implied_math_format": implied_math_format,
        "dest_acc": dest_acc,
        "input_A_dimensions": input_dimensions,
        "input_B_dimensions": input_dimensions,
        "unpack_to_dest": False,
        "broadcast_type": broadcast_type,
        "tile_cnt": tile_cnt,
    }

    res_address = write_stimuli_to_l1(
        test_config,
        src_A,
        src_B,
        formats.input_format,
        formats.input_format,
        tile_count_A=tile_cnt,
        tile_count_B=tile_cnt,
        num_faces=4,
    )

    run_test(test_config, boot_mode=boot_mode)

    res_from_L1 = collect_results(
        formats, tile_count=tile_cnt, address=res_address, num_faces=4
    )

    assert len(res_from_L1) == len(golden_tensor)

    torch_format = format_dict[formats.output_format]
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    assert passed_test(golden_tensor, res_tensor, formats.output_format)
