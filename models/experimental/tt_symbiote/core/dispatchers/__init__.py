# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Dispatcher implementations for TTNN operation handling."""

from models.experimental.tt_symbiote.core.dispatchers.dispatcher_config import (
    get_active_dispatcher,
    set_dispatcher,
    list_available_dispatchers,
    register_dispatcher,
)

__all__ = [
    "get_active_dispatcher",
    "set_dispatcher",
    "list_available_dispatchers",
    "register_dispatcher",
]
