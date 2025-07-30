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

# Test category-specific attributes
# These attributes are dynamically extracted from the test metadata
# and are used to enrich the aggregated statistics for each test category.
TEST_TYPE_ATTRIBUTES = {
    "multicast_schemes": {
        "test_ids": [100, 101],
        "attributes": {
            "NoC Index": "noc_index",
            "Multicast Scheme Type": "multicast_scheme_number",
            "Subordinate Grid Size X": "sub_grid_size_x",
            "Subordinate Grid Size Y": "sub_grid_size_y",
        },
    },
}

# Plot configuration
DEFAULT_PLOT_WIDTH = 12
DEFAULT_PLOT_HEIGHT = 6
DEFAULT_COMMENT_HEIGHT_RATIO = 0.2

# Kernel mapping
RISC_TO_KERNEL_MAP = {
    ## TO-DO: Make this programmable since this isn't consistent for all tests
    ## Maybe also put this in the yaml file
    "riscv_1": "Receiver",
    "riscv_0": "Sender",
}

# Output directory default
DEFAULT_OUTPUT_DIR = "tests/tt_metal/tt_metal/data_movement/data"
