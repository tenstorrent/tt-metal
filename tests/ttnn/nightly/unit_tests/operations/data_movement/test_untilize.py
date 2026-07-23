# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import pytest
from loguru import logger

from tests.ttnn.utils_for_testing import assert_equal


@pytest.mark.parametrize("mem_config", [ttnn.DRAM_MEMORY_CONFIG])
def test_untilize_with_padded_input(mem_config, device):
    """Regression test: untilize on a padded TILE tensor must discard padding.

    Issue #36765: untilize kept the padded values in the buffer (padded_shape
    stayed larger than logical_shape), silently corrupting downstream ops. After
    the fix, untilize routes to untilize_with_unpadding so the output buffer holds
    only the logical data (padded_shape == logical_shape).
    """
    torch_tensor = torch.randn(1, 1, 33, 33, dtype=torch.bfloat16)
    tt_tensor = ttnn.from_torch(torch_tensor, layout=ttnn.TILE_LAYOUT, memory_config=mem_config, device=device)

    logger.debug(f"Input logical_shape: {tt_tensor.shape}")
    logger.debug(f"Input padded_shape:  {tt_tensor.padded_shape}")

    untilized = ttnn.untilize(tt_tensor)

    logger.debug(f"After untilize:")
    logger.debug(f"  logical_shape: {untilized.shape}")
    logger.debug(f"  padded_shape:  {untilized.padded_shape}")
    logger.debug(f"  layout:        {untilized.layout}")

    if untilized.padded_shape == untilized.shape:
        torch_output = ttnn.to_torch(untilized)
        logger.debug(f"  to_torch() shape: {torch_output.shape}")
        logger.debug("   No padding in output buffer (clean data)")
    else:
        raise AssertionError("Output has padding in buffer")

    assert_equal(torch_tensor, ttnn.to_torch(untilized))
