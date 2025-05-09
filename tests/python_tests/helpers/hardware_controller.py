# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from .chip_architecture import ChipArchitecture, get_chip_architecture
from .utils import run_shell_command


class HardwareController:
    """
    This class is used for issuing reset commands to TT cards.
    """

    def __init__(self):
        self.chip_architecture = get_chip_architecture()

    def reset_card(self):
        if self.chip_architecture == ChipArchitecture.BLACKHOLE:
            print("Resetting BH card")
            run_shell_command("tt-smi -r 0")
        elif self.chip_architecture == ChipArchitecture.WORMHOLE:
            print("Resetting WH card")
            run_shell_command("tt-smi -r 0")
        else:
            raise ValueError("Unknown chip architecture")
