# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from helpers import *


torch.set_printoptions(linewidth=500, sci_mode=False, precision=2, threshold=10000)


def generate_golden(operand1, data_format):

    A_untilized = untilize(operand1, data_format)
    return A_untilized.flatten()


full_sweep = False
all_format_combos = generate_format_combinations(
    [DataFormat.Float16_b, DataFormat.Float16], all_same=True
)  # Generate format combinations with all formats being the same (flag set to True), refer to `param_config.py` for more details.
all_params = generate_params(["pack_untilize_test"], all_format_combos)
param_ids = generate_param_ids(all_params)


@pytest.mark.parametrize("testname, formats", clean_params(all_params), ids=param_ids)
def test_pack_untilize(testname, formats):

    src_A, src_B = generate_stimuli(formats.unpack_A_src, formats.unpack_B_src)
    src_A = torch.cat(
        [
            torch.full((256,), i, dtype=format_dict[formats.unpack_A_src])
            for i in range(1, 5)
        ]
    )
    src_B = torch.full((1024,), 0)

    golden_tensor = generate_golden(src_A, formats.pack_dst)

    write_stimuli_to_l1(src_A, src_B, formats.unpack_A_src, formats.unpack_B_src)

    test_config = {
        "formats": formats,
        "testname": testname,
    }

    make_cmd = generate_make_command(test_config)
    run_shell_command(f"cd .. && {make_cmd}")

    run_elf_files(testname)
    wait_for_tensix_operations_finished()

    res_from_L1 = collect_results(
        formats, tensor_size=len(src_A)
    )  # Bug patchup in (unpack.py): passing formats struct to check unpack_src with pack_dst and distinguish when input and output formats have different exponent widths then reading from L1 changes
    assert len(res_from_L1) == len(golden_tensor)

    res_tensor = torch.tensor(
        res_from_L1,
        dtype=(
            format_dict[formats.pack_dst]
            if formats.pack_dst in [DataFormat.Float16, DataFormat.Float16_b]
            else torch.bfloat16
        ),
    )

    if formats.pack_dst in [DataFormat.Float16_b, DataFormat.Float16]:
        atol = 0.1
        rtol = 0.05
    elif formats.pack_dst == DataFormat.Bfp8_b:
        atol = 0.1
        rtol = 0.2

    for i in range(len(golden_tensor)):
        assert torch.isclose(
            golden_tensor[i], res_tensor[i], rtol=rtol, atol=atol
        ), f"Failed at index {i} with values {golden_tensor[i]} and {res_from_L1[i]}"

    _, pcc = compare_pcc(golden_tensor, res_tensor, pcc=0.99)
    assert pcc > 0.98
