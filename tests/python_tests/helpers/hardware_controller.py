# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from .chip_architecture import ChipArchitecture, get_chip_architecture
from .target_config import TestTargetConfig
from .utils import run_shell_command


class HardwareController:
    """
    This class is used for issuing reset commands to TT cards.
    """

    def __init__(self):
        self.chip_architecture = get_chip_architecture()

    def reset_card(self):
        test_target = TestTargetConfig()
        if test_target.run_simulator:
            print("Running under simulator, unable to reset")
            return

        if self.chip_architecture == ChipArchitecture.BLACKHOLE:
            print("Resetting BH card")
            run_shell_command("tt-smi -r")
        elif self.chip_architecture == ChipArchitecture.WORMHOLE:
            print("Resetting WH card")
            run_shell_command("tt-smi -r")
        else:
            raise ValueError("Unknown chip architecture")
