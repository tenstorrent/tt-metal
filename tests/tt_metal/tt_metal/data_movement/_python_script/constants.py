#!/usr/bin/env python3

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# RISC-V processor types
RISCV_PROCESSORS = ["riscv_1", "riscv_0"]

# Architecture-specific settings
NOC_WIDTHS = {
    "wormhole_b0": 32,
    "blackhole": 64,
    # Add other architectures as needed
}

# Test categories
MULTICAST_SCHEMES_TEST_IDS = [100, 101]

# Plot configuration
DEFAULT_PLOT_WIDTH = 12
DEFAULT_PLOT_HEIGHT = 6
DEFAULT_COMMENT_HEIGHT_RATIO = 0.2

# Kernel mapping
RISC_TO_KERNEL_MAP = {
    "riscv_1": "Receiver",
    "riscv_0": "Sender",
}

# Output directory default
DEFAULT_OUTPUT_DIR = "tests/tt_metal/tt_metal/data_movement"
