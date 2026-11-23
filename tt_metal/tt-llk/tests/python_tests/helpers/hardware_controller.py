# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import tt_umd

from .chip_architecture import ChipArchitecture, get_chip_architecture
from .logger import logger
from .target_config import TestTargetConfig


class HardwareController:
    """
    This class is used for issuing reset commands to TT cards.
    """

    def __init__(self):
        self.chip_architecture = get_chip_architecture()

    @staticmethod
    def _warm_reset():
        # Warm reset through UMD, then verify with topology discovery (retries).
        if not tt_umd.WarmResetWithRecovery.warm_reset(max_attempts=3):
            raise RuntimeError("UMD warm reset failed")

    def reset_card(self):
        test_target = TestTargetConfig()
        if test_target.run_simulator:
            logger.info("Running under simulator, unable to reset")
            return

        if self.chip_architecture == ChipArchitecture.BLACKHOLE:
            logger.info("Resetting BH card")
            self._warm_reset()
        elif self.chip_architecture == ChipArchitecture.WORMHOLE:
            logger.info("Resetting WH card")
            self._warm_reset()
        else:
            raise ValueError("Unknown chip architecture")
