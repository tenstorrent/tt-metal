# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging

from vllm.logger import init_logger

_VLLM_LOGGER_PREFIX = "vllm."


def init_tt_logger(name: str) -> logging.Logger:
    """Initialize a TT plugin logger under vLLM's configured logger tree."""
    if name.startswith(_VLLM_LOGGER_PREFIX):
        return init_logger(name)
    return init_logger(f"{_VLLM_LOGGER_PREFIX}{name}")
