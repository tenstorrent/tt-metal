import pytest
import torch
import ttnn
import torch.nn.functional as F
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

    # Use float32 host tensors for ground-truth; convert to TT for device runs
    x_f32 = torch.randn((N, C, H, W), dtype=torch.float32)
    dy1_f32 = torch.randn((N, C, H, W), dtype=torch.float32)

    # Compute mean/rstd on host to avoid accumulation error from device bf16 forward
    x_view = x_f32.view(N, num_groups, -1)
    mean = x_view.mean(dim=-1, keepdim=True)
    var = ((x_view - mean) ** 2).mean(dim=-1, keepdim=False)
    rstd = (var + 1e-5).rsqrt()
    mean = mean.view(1, 1, N, num_groups)
    rstd = rstd.view(1, 1, N, num_groups)

    # Convert to device tensors
    tt_x = ttnn.from_torch(x_f32, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_dy1 = ttnn.from_torch(dy1_f32, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_mean = ttnn.from_torch(mean, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_rstd = ttnn.from_torch(rstd, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    logger.debug("First run backward input_grad with gamma present (compiles and seeds cache)")
    gamma = torch.randn((1, 1, 1, C), dtype=torch.float32)
    tt_gamma = ttnn.from_torch(gamma, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    dx1, _, _ = ttnn.operations.moreh.group_norm_backward(
        tt_dy1,
        tt_x,
        tt_mean,
        tt_rstd,
        num_groups,
        are_required_outputs=(True, False, False),
        gamma=tt_gamma,
    )

    # Torch baseline for gamma present
    x_torch = x_f32.detach().clone().requires_grad_(True)
    dy_torch = dy1_f32.detach().clone()
    gamma_torch = gamma.view(-1)
    y_torch = F.group_norm(x_torch, num_groups, gamma_torch, None, 1e-5)
    y_torch.backward(dy_torch)
    dx1_expected = x_torch.grad.to(torch.bfloat16)
    dx1_actual = ttnn.to_torch(dx1, dtype=torch.bfloat16)
    assert torch.allclose(dx1_expected, dx1_actual, rtol=0.1, atol=0.1)
    cache_after_first = device.num_program_cache_entries()

    # Second run toggles gamma from None -> present. This MUST be a cache miss (new program)
    # because compile-time args differ when gamma_has_value toggles.
    dy2 = torch.randn((N, C, H, W), dtype=torch.float32)
    tt_dy2 = ttnn.from_torch(dy2, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    logger.debug("Second run backward input_grad (gamma absent)")
    dx2, _, _ = ttnn.operations.moreh.group_norm_backward(
        tt_dy2,
        tt_x,
        tt_mean,
        tt_rstd,
        num_groups,
        are_required_outputs=(True, False, False),
        gamma=None,
    )

    # Torch baseline for gamma absent
    x_torch2 = x_f32.detach().clone().requires_grad_(True)
    dy_torch2 = dy2.detach().clone()
    y_torch2 = F.group_norm(x_torch2, num_groups, None, None, 1e-5)
    y_torch2.backward(dy_torch2)
    dx2_expected = x_torch2.grad.to(torch.bfloat16)
    dx2_actual = ttnn.to_torch(dx2, dtype=torch.bfloat16)
    assert torch.allclose(dx2_expected, dx2_actual, rtol=0.1, atol=0.1)
    # Verify correct behavior: toggling gamma present -> None should be a cache miss
    cache_after_second = device.num_program_cache_entries()
    assert cache_after_second == cache_after_first + 1

    # Third run back to gamma=None: again should be a cache miss (different compile-time args)
    dy3 = torch.randn((N, C, H, W), dtype=torch.float32)
    tt_dy3 = ttnn.from_torch(dy3, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    logger.debug("Third run backward input_grad (gamma absent again)")
    dx3, _, _ = ttnn.operations.moreh.group_norm_backward(
        tt_dy3,
        tt_x,
        tt_mean,
        tt_rstd,
        num_groups,
        are_required_outputs=(True, False, False),
        gamma=None,
    )

    # Torch baseline for gamma=None again
    x_torch3 = x_f32.detach().clone().requires_grad_(True)
    dy_torch3 = dy3.detach().clone()
    y_torch3 = F.group_norm(x_torch3, num_groups, None, None, 1e-5)
    y_torch3.backward(dy_torch3)
    dx3_expected = x_torch3.grad.to(torch.bfloat16)
    dx3_actual = ttnn.to_torch(dx3, dtype=torch.bfloat16)
    assert torch.allclose(dx3_expected, dx3_actual, rtol=0.1, atol=0.1)

    # Program cache checks after correctness: ensure gamma=None path reuses program
    cache_before = device.num_program_cache_entries()
    dy4 = torch.randn((N, C, H, W), dtype=torch.float32)
    tt_dy4 = ttnn.from_torch(dy4, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    dx4, _, _ = ttnn.operations.moreh.group_norm_backward(
        tt_dy4,
        tt_x,
        tt_mean,
        tt_rstd,
        num_groups,
        are_required_outputs=(True, False, False),
        gamma=None,
    )
    _ = ttnn.to_torch(dx4, dtype=torch.bfloat16)
    cache_after = device.num_program_cache_entries()
    assert cache_after == cache_before
