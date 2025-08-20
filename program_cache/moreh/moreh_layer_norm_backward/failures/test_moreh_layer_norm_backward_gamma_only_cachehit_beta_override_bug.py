import pytest
import torch
import ttnn
from loguru import logger


@pytest.mark.timeout(30)
def test_moreh_layer_norm_backward_program_cache_gamma_only_triggers_beta_override_bug(device):
    """
    Exposes a program-cache override bug in layer_norm_backward gamma_beta_grad factory:
      - In override_runtime_arguments, index 1 (beta_grad address) is guarded by `if (gamma_grad_buffer != nullptr)`
        instead of checking `beta_grad_buffer`. With gamma_grad present and beta_grad absent, this dereferences null.

    Location:
      - ttnn/cpp/ttnn/operations/moreh/moreh_layer_norm_backward/device/moreh_layer_norm_backward_gamma_beta_grad_program_factory.cpp
        lines ~298-304

    Expected failure mode:
      - Crash or hang on the second run (cache-hit) when only gamma_grad is requested.
    """

    torch.manual_seed(0)

    # Shapes
    N, C, H, W = 1, 4, 16, 16
    normalized_dims = 1
    eps = 1e-5

    # Input and grads
    x1 = torch.randn((N, C, H, W)).bfloat16()
    dy1 = torch.randn((N, C, H, W)).bfloat16()

    tt_x1 = ttnn.from_torch(x1, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_dy1 = ttnn.from_torch(dy1, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Compute mean/rstd via forward to get correct shapes
    y, mean, rstd = ttnn.operations.moreh.layer_norm(
        tt_x1,
        normalized_dims,
        eps,
        gamma=None,
        beta=None,
        are_required_outputs=(True, True, True),
    )

    # Preallocate only gamma_grad (beta_grad absent)
    gamma_shape = [1, 1, 1, C]
    gamma_grad_buf1 = ttnn.from_torch(torch.empty(gamma_shape).bfloat16(), dtype=ttnn.bfloat16, device=device)

    num_cache_start = device.num_program_cache_entries()
    logger.debug("First run backward gamma-only (compile and seed cache)")
    _, gamma_grad_out1, _ = ttnn.operations.moreh.layer_norm_backward(
        tt_dy1,
        tt_x1,
        mean,
        rstd,
        normalized_dims,
        gamma=None,
        input_grad=None,
        gamma_grad=gamma_grad_buf1,
        beta_grad=None,
    )
    num_cache_end = device.num_program_cache_entries()
    assert num_cache_end == num_cache_start + 1

    # Second run with new buffers to hit cache; still no beta_grad provided
    dy2 = torch.randn((N, C, H, W)).bfloat16()
    tt_dy2 = ttnn.from_torch(dy2, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    gamma_grad_buf2 = ttnn.from_torch(torch.empty(gamma_shape).bfloat16(), dtype=ttnn.bfloat16, device=device)

    logger.debug("Second run backward gamma-only (cache hit; bug expected)")
    _, gamma_grad_out2, _ = ttnn.operations.moreh.layer_norm_backward(
        tt_dy2,
        tt_x1,
        mean,
        rstd,
        normalized_dims,
        gamma=None,
        input_grad=None,
        gamma_grad=gamma_grad_buf2,
        beta_grad=None,
    )
