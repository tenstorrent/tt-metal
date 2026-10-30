# SPDX-License-Identifier: Apache-2.0
"""Host-only opt-in selection; CLI cache sizes must match this worker setting."""
import os

DEFAULT_BLOCK_SIZE = 4096
SUPPORTED_BLOCK_SIZES = (2048, 4096)
BLOCK_SIZE_ENV = "QWEN38_SCHEDULER_BLOCK_SIZE"


def scheduler_block_size():
    value = os.environ.get(BLOCK_SIZE_ENV, str(DEFAULT_BLOCK_SIZE))
    if value not in tuple(map(str, SUPPORTED_BLOCK_SIZES)):
        raise ValueError(f"{BLOCK_SIZE_ENV} must be2048 or4096, got {value!r}")
    return int(value)
