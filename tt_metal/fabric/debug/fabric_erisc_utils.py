# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Utility Functions for ERISC Fabric Debugging

This module provides helper functions for the ERISC debugging toolkit, including:
- Stream register address calculation for different architectures
- Device architecture detection and normalization
- Fabric stream register group management
- Core coordinate parsing and validation

These utilities abstract away architecture-specific details and provide a clean
interface for the main dumper script.
"""

from fabric_erisc_constants import (
    STREAM_REGISTER_INDICES,
    FABRIC_STREAM_GROUPS,
    ARCHITECTURE_MAPPING,
    NOC_OVERLAY_START_ADDR,
    NOC_STREAM_REG_SPACE_SIZE,
    DEFAULT_FABRIC_STREAM_GROUP,
    DEFAULT_FABRIC_REGISTER,
)


def get_stream_reg_address(stream_id: int, reg_name: str, arch: str) -> int:
    """
    Calculate the physical address of a stream register.

    Stream registers are organized in the NOC overlay space with each stream
    having its own 4KB register space. Within each stream's space, register
    indices are architecture-specific.

    Args:
        stream_id: Stream ID (typically 0-31)
        reg_name: Register name (BUF_SPACE_AVAILABLE, BUF_SIZE, BUF_SPACE_UPDATE)
        arch: Architecture string (wormhole, blackhole)

    Returns:
        Physical address of the register (32-bit aligned)

    Raises:
        KeyError: If architecture or register name is not supported
        ValueError: If stream_id is out of valid range
    """
    if not (0 <= stream_id <= 31):
        raise ValueError(f"Stream ID {stream_id} out of valid range [0-31]")

    if arch not in STREAM_REGISTER_INDICES:
        available_archs = list(STREAM_REGISTER_INDICES.keys())
        raise KeyError(f"Unsupported architecture '{arch}'. Available: {available_archs}")

    if reg_name not in STREAM_REGISTER_INDICES[arch]:
        available_regs = list(STREAM_REGISTER_INDICES[arch].keys())
        raise KeyError(f"Unsupported register '{reg_name}' for architecture '{arch}'. Available: {available_regs}")

    reg_index = STREAM_REGISTER_INDICES[arch][reg_name]
    # Address calculation: base + (stream * space_size) + (register_index * 4)
    return NOC_OVERLAY_START_ADDR + (stream_id * NOC_STREAM_REG_SPACE_SIZE) + (reg_index << 2)


def normalize_architecture(arch_str: str) -> str:
    """
    Normalize device architecture string to internal representation.

    Device architecture strings can vary (e.g., "wormhole_b0", "WORMHOLE", "blackhole")
    but we need consistent internal names for register lookups.

    Args:
        arch_str: Architecture string from device (case insensitive)

    Returns:
        Normalized architecture string (wormhole, blackhole)

    Raises:
        KeyError: If architecture is not supported
    """
    if not arch_str:
        raise KeyError("Empty architecture string provided")

    arch_lower = arch_str.lower().strip()

    for known_arch, normalized_arch in ARCHITECTURE_MAPPING.items():
        if known_arch.lower() in arch_lower:
            return normalized_arch

    available_archs = list(ARCHITECTURE_MAPPING.keys())
    raise KeyError(f"Unsupported architecture '{arch_str}'. Supported: {available_archs}")


def get_fabric_stream_addresses(group_name: str = None, reg_name: str = None, arch: str = "wormhole") -> dict:
    """
    Get all fabric stream register addresses for a specific group and register type.

    This function is the main interface for getting fabric flow control register
    addresses for debugging. It handles the mapping from logical stream groups
    to physical register addresses.

    Args:
        group_name: Fabric stream group name (default: all_fabric_free_slots)
        reg_name: Register name (default: BUF_SPACE_AVAILABLE)
        arch: Architecture string (default: wormhole)

    Returns:
        Dictionary mapping stream_id -> physical_address

    Raises:
        KeyError: If group_name is not valid or architecture is unsupported
    """
    # Use defaults if not specified
    if group_name is None:
        group_name = DEFAULT_FABRIC_STREAM_GROUP
    if reg_name is None:
        reg_name = DEFAULT_FABRIC_REGISTER

    # Validate group name
    if group_name not in FABRIC_STREAM_GROUPS:
        available_groups = list(FABRIC_STREAM_GROUPS.keys())
        raise KeyError(f"Unknown fabric stream group '{group_name}'. Available: {available_groups}")

    group = FABRIC_STREAM_GROUPS[group_name]
    addresses = {}

    # Calculate address for each stream in the group
    for stream_id in group["stream_ids"]:
        try:
            addresses[stream_id] = get_stream_reg_address(stream_id, reg_name, arch)
        except (KeyError, ValueError) as e:
            # Re-raise with more context about which stream failed
            raise type(e)(f"Failed to calculate address for stream {stream_id} in group '{group_name}': {e}")

    return addresses


def detect_device_architecture(device) -> str:
    """
    Detect device architecture from ttexalens device object.

    Uses multiple detection methods in order of preference:
    1. Direct architecture attribute (_arch)
    2. Device type inspection
    3. Fallback to wormhole (most common)

    Args:
        device: ttexalens device object

    Returns:
        Normalized architecture string (wormhole, blackhole)
    """
    # Method 1: Try direct architecture attribute
    if hasattr(device, "_arch") and device._arch:
        try:
            return normalize_architecture(device._arch)
        except KeyError:
            print(f"Warning: Device {device.id} has unknown architecture '{device._arch}', trying fallback methods")

    # Method 2: Try type checking (requires architecture-specific imports)
    try:
        from ttexalens.hw.tensix.blackhole.blackhole import BlackholeDevice
        from ttexalens.hw.tensix.wormhole.wormhole import WormholeDevice

        device_type = type(device)
        if device_type == WormholeDevice:
            return "wormhole"
        elif device_type == BlackholeDevice:
            return "blackhole"
        else:
            print(f"Warning: Unknown device type {device_type.__name__} for device {device.id}")
    except ImportError as e:
        print(f"Warning: Could not import device type classes for architecture detection: {e}")

    # Method 3: Final fallback with warning
    print(f"Warning: Could not detect architecture for device {device.id}, defaulting to wormhole")
    return "wormhole"


def parse_core_key(core_key: str) -> tuple:
    """
    Parse core_key string into (x, y) coordinates.

    Core keys are generated by the dumper in the format "core_x_y" where
    x and y are the logical coordinates of the ethernet core.

    Args:
        core_key: Core key in format "core_x_y" (e.g., "core_0_5")

    Returns:
        Tuple of (x, y) coordinates as integers

    Raises:
        ValueError: If core_key format is invalid or coordinates are not integers
    """
    if not isinstance(core_key, str):
        raise ValueError(f"Core key must be a string, got {type(core_key)}")

    # Expected format: "core_x_y"
    parts = core_key.split("_")
    if len(parts) != 3 or parts[0] != "core":
        raise ValueError(f"Invalid core_key format '{core_key}'. Expected format: 'core_x_y'")

    try:
        x = int(parts[1])
        y = int(parts[2])
        return (x, y)
    except ValueError as e:
        raise ValueError(f"Invalid coordinate values in core_key '{core_key}': {e}")
