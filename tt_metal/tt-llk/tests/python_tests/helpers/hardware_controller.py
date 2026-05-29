# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from .chip_architecture import ChipArchitecture, get_chip_architecture
from .logger import logger
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
            logger.info("Running under simulator, unable to reset")
            return

        if self.chip_architecture not in (
            ChipArchitecture.BLACKHOLE,
            ChipArchitecture.WORMHOLE,
        ):
            raise ValueError("Unknown chip architecture")

        logger.info("Resetting {} card via 'tt-smi -r'", self.chip_architecture.name)
        # Capture stdout (normally swallowed) so a failed/partial reset on CI
        # surfaces tt-smi's own diagnostics in the logs instead of failing
        # blind. run_shell_command raises on a non-zero exit, folding both
        # stdout and stderr into the message.
        result = run_shell_command("tt-smi -r", capture_stdout=True)
        output = (result.stdout or "").strip()
        if output:
            logger.info("'tt-smi -r' output:\n{}", output)
