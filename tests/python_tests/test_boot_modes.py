# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from conftest import skip_for_wormhole
from helpers.device import BootMode
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.hardware_controller import HardwareController
from helpers.llk_params import DestAccumulation, MathFidelity
from helpers.param_config import parametrize
from test_matmul import test_matmul as run_matmul


@skip_for_wormhole
@parametrize(
    boot_mode=[BootMode.BRISC, BootMode.TRISC, BootMode.EXALENS],
)
def test_boot_modes(boot_mode):
    test_name = "matmul_test"
    math_fidelity = MathFidelity.LoFi
    format_dest_acc_and_dims = (
        InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b),
        DestAccumulation.No,
        ([32, 32], [32, 32]),
    )

    HardwareController().reset_card()

    run_matmul(test_name, math_fidelity, format_dest_acc_and_dims, boot_mode[0])
