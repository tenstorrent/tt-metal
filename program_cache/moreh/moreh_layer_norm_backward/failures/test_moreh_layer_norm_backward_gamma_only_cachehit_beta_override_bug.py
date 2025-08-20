import pytest
import torch
import torch.nn.functional as F
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

    # Input and grads (use small integer range for numerical stability, float32 baseline)
    x_f32 = torch.randint(-2, 3, (N, C, H, W), dtype=torch.float32)
    dy1_f32 = torch.randint(-2, 3, (N, C, H, W), dtype=torch.float32)

    tt_x1 = ttnn.from_torch(x_f32, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_dy1 = ttnn.from_torch(dy1_f32, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Run forward to get device-produced mean/rstd and preallocated buffers
    # Run forward on device to generate mean/rstd with exact kernel layout/shape
    mean_rstd_shape = [N, C, H]
    tt_y = ttnn.from_torch(
        torch.empty_like(x_f32, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    tt_mean = ttnn.from_torch(
        torch.full(mean_rstd_shape, float("nan"), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    tt_rstd = ttnn.from_torch(
        torch.full(mean_rstd_shape, float("nan"), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    tt_y, tt_mean, tt_rstd = ttnn.operations.moreh.layer_norm(
        tt_x1,
        normalized_dims,
        eps,
        gamma=None,
        beta=None,
        output=tt_y,
        mean=tt_mean,
        rstd=tt_rstd,
    )

    # Provide gamma so only gamma_grad is requested; leave beta and beta_grad absent
    gamma = torch.randn((W,), dtype=torch.float32)
    tt_gamma = ttnn.from_torch(gamma, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Preallocate only gamma_grad (beta_grad absent). Shape matches last normalized dims
    gamma_shape = [W]
    gamma_grad_buf1 = ttnn.from_torch(
        torch.empty(gamma_shape).bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )

    # Optionally provide input_grad buffer (not used in this test) to match reference tests
    tt_input_grad1 = ttnn.from_torch(
        torch.empty((N, C, H, W), dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )

    num_cache_start = device.num_program_cache_entries()
    logger.debug("First run backward gamma-only (compile and seed cache)")
    _, gamma_grad_out1, _ = ttnn.operations.moreh.layer_norm_backward(
        tt_dy1,
        tt_x1,
        tt_mean,
        tt_rstd,
        normalized_dims,
        gamma=tt_gamma,
        input_grad=tt_input_grad1,
        gamma_grad=gamma_grad_buf1,
        beta_grad=None,
    )
    num_cache_end = device.num_program_cache_entries()
    assert num_cache_end > num_cache_start

    # Check gamma_grad correctness against PyTorch for first run
    x_t = x_f32.detach().clone().requires_grad_(True)
    dy_t = dy1_f32.detach().clone()
    gamma_t = gamma.detach().clone().requires_grad_(True)
    y_t = F.layer_norm(x_t, [W], weight=gamma_t, bias=None, eps=eps)
    y_t.backward(dy_t)
    expected_gamma_grad1 = gamma_t.grad.to(torch.bfloat16)
    actual_gamma_grad1 = ttnn.to_torch(gamma_grad_out1, dtype=torch.bfloat16)
    assert torch.allclose(expected_gamma_grad1, actual_gamma_grad1, rtol=0.1, atol=0.5)

    # Second run with new buffers to hit cache; still no beta_grad provided
    dy2 = torch.randint(-2, 3, (N, C, H, W), dtype=torch.float32)
    tt_dy2 = ttnn.from_torch(dy2, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    gamma_grad_buf2 = ttnn.from_torch(
        torch.empty(gamma_shape).bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    tt_input_grad2 = ttnn.from_torch(
        torch.empty((N, C, H, W), dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )

    logger.debug("Second run backward gamma-only (cache hit; bug expected)")
    _, gamma_grad_out2, _ = ttnn.operations.moreh.layer_norm_backward(
        tt_dy2,
        tt_x1,
        tt_mean,
        tt_rstd,
        normalized_dims,
        gamma=tt_gamma,
        input_grad=tt_input_grad2,
        gamma_grad=gamma_grad_buf2,
        beta_grad=None,
    )

    # Check gamma_grad correctness against PyTorch for second run
    x_t2 = x_f32.detach().clone().requires_grad_(True)
    dy_t2 = dy2.detach().clone()
    gamma_t2 = gamma.detach().clone().requires_grad_(True)
    y_t2 = F.layer_norm(x_t2, [W], weight=gamma_t2, bias=None, eps=eps)
    y_t2.backward(dy_t2)
    expected_gamma_grad2 = gamma_t2.grad.to(torch.bfloat16)
    actual_gamma_grad2 = ttnn.to_torch(gamma_grad_out2, dtype=torch.bfloat16)
    assert torch.allclose(expected_gamma_grad2, actual_gamma_grad2, rtol=0.1, atol=0.5)
