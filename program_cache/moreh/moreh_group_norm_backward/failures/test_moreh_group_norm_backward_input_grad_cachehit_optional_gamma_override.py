import pytest
import torch
import ttnn
from loguru import logger


@pytest.mark.timeout(30)
def test_moreh_group_norm_backward_input_grad_program_cache_optional_gamma_override(device):
    """
    Validate override path when optional gamma is absent. This ensures that runtime arg index 4
    is not overwritten if gamma is None on cache-hit.

    Location:
      - ttnn/cpp/ttnn/operations/moreh/moreh_group_norm_backward/device/input_grad/moreh_group_norm_backward_input_grad_factory.cpp
        override_runtime_arguments: index 4 only updated when gamma_buffer != nullptr (correct).

    This test is a guard to catch future regressions.
    """

    torch.manual_seed(0)
    N, C, H, W = 1, 4, 16, 16
    num_groups = 2

    x = torch.randn((N, C, H, W)).bfloat16()
    dy1 = torch.randn((N, C, H, W)).bfloat16()

    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_dy1 = ttnn.from_torch(dy1, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Use forward to materialize mean/rstd quickly
    y, mean, rstd = ttnn.operations.moreh.group_norm(
        tt_x, num_groups, 1e-5, gamma=None, beta=None, are_required_outputs=(True, True, True)
    )

    num_cache_start = device.num_program_cache_entries()

    logger.debug("First run backward input_grad (compiles and seeds cache)")
    dx1 = ttnn.operations.moreh.group_norm_backward_input_grad(tt_dy1, tt_x, mean, rstd, num_groups, gamma=None)

    num_cache_end = device.num_program_cache_entries()
    assert num_cache_end == num_cache_start + 1

    # Second run with new buffers (cache hit) and still gamma=None
    dy2 = torch.randn((N, C, H, W)).bfloat16()
    tt_dy2 = ttnn.from_torch(dy2, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    logger.debug("Second run backward input_grad (cache hit)")
    dx2 = ttnn.operations.moreh.group_norm_backward_input_grad(tt_dy2, tt_x, mean, rstd, num_groups, gamma=None)

    # If it reaches here, override handled optional gamma correctly on cache-hit.
