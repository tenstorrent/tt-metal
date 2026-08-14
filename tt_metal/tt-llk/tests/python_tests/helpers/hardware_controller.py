# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from .chip_architecture import ChipArchitecture, get_chip_architecture
from .logger import logger
from .target_config import TestTargetConfig
from .utils import run_shell_command

# Callers reset precisely when the card has stopped behaving, so the reset is
# not entitled to assume it will return on its own.
RESET_TIMEOUT_SECONDS = 180


class HardwareController:
    """
    This class is used for issuing reset commands to TT cards.
    """

    def __init__(self):
        self.chip_architecture = get_chip_architecture()

    def reset_card(self):
        test_target = TestTargetConfig()
        if test_target.run_simulator:
            logger.info("Running under simulator, unable to reset")
            return

        if self.chip_architecture == ChipArchitecture.BLACKHOLE:
            logger.info("Resetting BH card")
            run_shell_command("tt-smi -r", timeout=RESET_TIMEOUT_SECONDS)
        elif self.chip_architecture == ChipArchitecture.WORMHOLE:
            logger.info("Resetting WH card")
            run_shell_command("tt-smi -r", timeout=RESET_TIMEOUT_SECONDS)
        else:
            raise ValueError("Unknown chip architecture")
