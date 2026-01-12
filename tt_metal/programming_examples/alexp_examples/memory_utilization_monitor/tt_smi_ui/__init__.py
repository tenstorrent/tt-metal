# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
TT-SMI: Tenstorrent System Management Interface

Python UI for monitoring Tenstorrent devices with beautiful, interactive display.
"""

__version__ = "0.1.0"

from .core import (
    get_devices,
    update_telemetry,
    update_memory,
    cleanup_dead_processes,
    format_bytes,
)

__all__ = [
    "get_devices",
    "update_telemetry",
    "update_memory",
    "cleanup_dead_processes",
    "format_bytes",
]
