#!/usr/bin/env python3
"""
Configuration for Blackhole Galaxy (32-device) telemetry benchmarking.

This configuration file allows flexible testing on Galaxy systems with:
- 32 Blackhole devices arranged in 4 trays (8 devices each)
- Configurable device counts to avoid system-wide impact
- Optimized polling frequencies for performance testing
"""

import os
from pathlib import Path

# Galaxy-specific paths (update these for your system)
GALAXY_PATHS = {
    "TT_METAL_HOME": "/localdev/kkfernandez/tt-metal",
    "TELEMETRY_BIN": "/localdev/kkfernandez/tt-telemetry/build_Release/bin/tt_telemetry_server",
    "FSD_PATH": "/localdev/kkfernandez/fsd.textproto",
}

# Device count configurations for Galaxy
# Start small to avoid system-wide corruption, scale up as needed
DEVICE_COUNTS_GALAXY = {
    "minimal": [2, 4],  # Safest - minimal impact
    "reduced": [2, 4, 8],  # Single tray testing
    "extended": [2, 4, 8, 16],  # Two tray testing
    "full": [2, 4, 8, 16, 32],  # Full galaxy (use with caution)
}

# Mesh shapes for multi-device tests on Galaxy
MESH_SHAPES_GALAXY = {
    2: (1, 2),  # 1x2 mesh - 2 devices
    4: (2, 2),  # 2x2 mesh - 4 devices (half row in tray)
    8: (2, 4),  # 2x4 mesh - 8 devices (full tray)
    16: (4, 4),  # 4x4 mesh - 16 devices (two trays)
    32: (4, 8),  # 4x8 mesh - 32 devices (full galaxy)
}

# Polling frequencies optimized for performance testing
# Removed 60s and 10s as they are too slow to show meaningful impact
POLLING_FREQUENCIES_GALAXY = {
    "quick": ["1s", "10ms", "100us"],  # 3 key points
    "reduced": ["5s", "1s", "100ms", "10ms", "1ms", "100us"],  # 6 frequencies
    "full": ["5s", "1s", "500ms", "100ms", "50ms", "10ms", "5ms", "1ms", "500us", "100us"],  # 10 frequencies
}

# CCL operations to test
CCL_OPERATIONS_GALAXY = {
    "quick": ["AllGather"],  # Fastest test
    "reduced": ["AllGather", "AllReduce"],  # Key operations
    "full": ["AllGather", "ReduceScatter", "AllReduce"],  # All operations
}

# Tensor sizes for testing (in elements per dimension)
TENSOR_SIZES_GALAXY = {
    "small": (1, 1, 1024, 1024),  # 2MB - minimal memory
    "medium": (1, 1, 8192, 8192),  # 128MB - standard test
    "large": (1, 1, 16384, 16384),  # 512MB - stress test
}

# Test presets for different scenarios
TEST_PRESETS = {
    "quick_validation": {
        "device_counts": DEVICE_COUNTS_GALAXY["minimal"],
        "polling_frequencies": POLLING_FREQUENCIES_GALAXY["quick"],
        "ccl_operations": CCL_OPERATIONS_GALAXY["quick"],
        "tensor_size": TENSOR_SIZES_GALAXY["small"],
        "iterations": 10,
        "description": "Quick 15-minute validation test",
    },
    "safe_comprehensive": {
        "device_counts": DEVICE_COUNTS_GALAXY["reduced"],
        "polling_frequencies": POLLING_FREQUENCIES_GALAXY["reduced"],
        "ccl_operations": CCL_OPERATIONS_GALAXY["reduced"],
        "tensor_size": TENSOR_SIZES_GALAXY["medium"],
        "iterations": 50,
        "description": "2-hour comprehensive test (single tray)",
    },
    "full_galaxy": {
        "device_counts": DEVICE_COUNTS_GALAXY["full"],
        "polling_frequencies": POLLING_FREQUENCIES_GALAXY["full"],
        "ccl_operations": CCL_OPERATIONS_GALAXY["full"],
        "tensor_size": TENSOR_SIZES_GALAXY["medium"],
        "iterations": 100,
        "description": "Full galaxy test (use with caution)",
    },
}


def get_preset(preset_name: str = "safe_comprehensive"):
    """Get test configuration preset."""
    if preset_name not in TEST_PRESETS:
        print(f"Available presets: {list(TEST_PRESETS.keys())}")
        raise ValueError(f"Unknown preset: {preset_name}")
    return TEST_PRESETS[preset_name]


def setup_galaxy_environment():
    """Set up environment variables for Galaxy testing."""
    for key, value in GALAXY_PATHS.items():
        os.environ[key] = value

    # Verify paths exist
    for name, path in GALAXY_PATHS.items():
        if not Path(path).exists():
            print(f"Warning: {name} path does not exist: {path}")

    print("Galaxy environment configured:")
    print(f"  TT_METAL_HOME: {GALAXY_PATHS['TT_METAL_HOME']}")
    print(f"  FSD_PATH: {GALAXY_PATHS['FSD_PATH']}")
    print(f"  TELEMETRY_BIN: {GALAXY_PATHS['TELEMETRY_BIN']}")


if __name__ == "__main__":
    print("Blackhole Galaxy Configuration")
    print("=" * 50)
    print("Available test presets:")
    for name, config in TEST_PRESETS.items():
        print(f"  {name}: {config['description']}")
    print("\nDevice configurations:")
    for name, counts in DEVICE_COUNTS_GALAXY.items():
        print(f"  {name}: {counts}")
    print("\nMesh shapes supported:")
    for devices, shape in MESH_SHAPES_GALAXY.items():
        print(f"  {devices} devices: {shape[0]}x{shape[1]} mesh")
