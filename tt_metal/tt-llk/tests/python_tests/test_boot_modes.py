# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from conftest import skip_for_blackhole, skip_for_wormhole
from helpers.device import BootMode
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.hardware_controller import HardwareController
from helpers.llk_params import DestAccumulation, MathFidelity
from helpers.param_config import parametrize
from test_matmul import test_matmul as run_matmul


# TODO Skip for all until per-tensix reset is available
@skip_for_wormhole
@skip_for_blackhole
@parametrize(
    boot_mode=[BootMode.BRISC, BootMode.TRISC, BootMode.EXALENS],
)
def test_boot_modes(
    boot_mode,
):
    math_fidelity = MathFidelity.LoFi
    HardwareController().reset_card()

    run_matmul(
        math_fidelity,
        InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b),
        DestAccumulation.No,
        1,
        1,
        1,
        boot_mode=boot_mode[0],
    )
