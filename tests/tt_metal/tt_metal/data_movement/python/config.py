# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os
import sys
from loguru import logger


class DataMovementConfig:
    def __init__(self):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.test_info_path = os.path.join(self.script_dir, "test_mappings", "test_information.yaml")
        self.test_bounds_path = os.path.join(self.script_dir, "test_mappings", "test_bounds.yaml")
        self.test_type_attributes_path = os.path.join(self.script_dir, "test_mappings", "test_type_attributes.yaml")

    def get_arch(self, arch_name=None, test_bounds=None):
        """Get architecture from command line argument or environment variable."""
        if arch_name:
            return arch_name

        arch = os.environ.get("ARCH_NAME", None)
        if arch is None:
            logger.warning("ARCH_NAME environment variable is not set, defaulting to 'blackhole'.")
            return "blackhole"
        elif test_bounds and arch not in test_bounds.keys():
            logger.error(f"ARCH_NAME '{arch}' is not recognized.")
            sys.exit(1)
        return arch
