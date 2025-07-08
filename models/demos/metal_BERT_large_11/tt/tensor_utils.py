# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Callable, Optional

from loguru import logger

import ttnn


def load_or_compute_and_cache(
    cache_path: Optional[str],
    compute_func: Callable[[], ttnn.Tensor],
    device: ttnn.Device,
    mem_config: ttnn.MemoryConfig,
) -> ttnn.Tensor:
    """
    Attempts to load a tensor from cache_path using ttnn.load_tensor.
    If cache_path is None or loading fails, it calls compute_func,
    converts the result using convert_func, places it on the specified device and mem_config,
    and caches the result if cache_path was provided.

    Args:
        cache_path (Optional[str]): Path (as string) to load from/save to.
        compute_func (Callable[[], ttnn.Tensor]): Function that computes and returns the tensor as a ttnn.Tensor
                                                  (handling dtype and layout).
        device (ttnn.Device): Target device for the final ttnn.Tensor.
        mem_config (ttnn.MemoryConfig): Target memory config for the final ttnn.Tensor.

    Returns:
        ttnn.Tensor: The loaded or computed tensor on the specified device and memory configuration.
    """
    tensor = None
    if cache_path:
        try:
            tensor = ttnn.load_tensor(cache_path, device=device)
            logger.info(f"Loaded tensor from cache: {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to load tensor from cache: {cache_path}. Error: {e}")
            tensor = None

    if tensor is None:
        tensor = compute_func()
        if cache_path:
            logger.info(f"Dumping tensor to cache: {cache_path}")
            ttnn.dump_tensor(cache_path, tensor)

    return tensor.to(device, mem_config)
