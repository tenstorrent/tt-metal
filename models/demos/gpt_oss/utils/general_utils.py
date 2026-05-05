# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
General utilities for the GPT-OSS demo.
"""

from loguru import logger


def get_cache_file_name(tensor_cache_path, name):
    if not tensor_cache_path:
        return None
    cache_path = f"{tensor_cache_path}/{name}"
    # TEMP: ttnn from_torch_and_dump logs cache hits/misses at DEBUG, which CI
    # suppresses, leaving cold-start runs looking like a 20-min hang. Heartbeat
    # at INFO so progress through layers/weights is visible. Drop once cache
    # cold-start performance is back to acceptable on CI runners.
    logger.info(f"gpt_oss cache: {cache_path}")
    return cache_path
