# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""
ComfyUI Bridge Server for Tenstorrent Hardware

Provides a Unix socket bridge between ComfyUI frontend and tt-metal SDXL backend.
Uses shared memory for efficient tensor transfer between processes.
"""

__version__ = "0.1.0"

from .server import ComfyUIBridgeServer
from .protocol import receive_message, send_message, send_error, send_success
from .handlers import OperationHandler

__all__ = [
    "ComfyUIBridgeServer",
    "OperationHandler",
    "receive_message",
    "send_message",
    "send_error",
    "send_success",
]
