import torch
import ttnn

torch.manual_seed(1337)

with ttnn.manage_device(device_id=0) as device:
    device.enable_program_cache()

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )

    # dim = 24 * 32
    dim = 2**12
    # dim = 16
    x_torch = torch.randn((32, 1, 4096, dim), dtype=torch.bfloat16)
    # x_torch = torch.randn((12, 1, 4096, dim), dtype=torch.bfloat16)
    # x_torch = torch.randn((1, 1, 1, dim), dtype=torch.bfloat16)
    expected_rms = torch.sqrt(torch.sum(x_torch**2, dim=-1, keepdim=True) / dim + 1e-5)
    gamma_torch = torch.randn((1, 1, 1, dim), dtype=torch.bfloat16)
    expected_result = x_torch * gamma_torch / expected_rms

    x = ttnn.from_torch(x_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    gamma = ttnn.from_torch(gamma_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    x_original = ttnn.rms_norm(x, weight=gamma, epsilon=1e-5, compute_kernel_config=compute_kernel_config)
    tmp = ttnn.to_torch(x_original)

    def composite_rms_norm(x):
        x_2 = x * x
        x_2_sum = ttnn.sum(x_2, dim=3)
        x_2_sum = ttnn.div(x_2_sum, dim)
        x_2_sum_sqrt = ttnn.sqrt(x_2_sum + 1e-5)
        x_2_sum_sqrt_inv = ttnn.reciprocal(x_2_sum_sqrt)
        composite_result = x * x_2_sum_sqrt_inv * gamma
        return composite_result

    composite_result = composite_rms_norm(x)
    composite_result = ttnn.to_torch(composite_result)

    x_new, rms = ttnn.experimental.rmsnorm_fw(x, gamma, True, 1e-5)
    ttnn.synchronize_device(device)

    x_new, _ = ttnn.experimental.rmsnorm_fw(x, gamma, False, 1e-5)
    ttnn.synchronize_device(device)

    x_new, rms = ttnn.experimental.rmsnorm_fw(x, gamma, True, 1e-5)
    ttnn.synchronize_device(device)

    from time import time

    start = time()
    for _ in range(25):
        e = ttnn.rms_norm(x, weight=gamma, epsilon=1e-5, compute_kernel_config=compute_kernel_config)
        ttnn.synchronize_device(device)
    original_total = time() - start

    start = time()
    for _ in range(25):
        e, _ = ttnn.experimental.rmsnorm_fw(x, gamma, False, 1e-5)
        ttnn.synchronize_device(device)
    new_total = time() - start

    start_time = time()
    for _ in range(25):
        e = composite_rms_norm(x)
        ttnn.synchronize_device(device)
    composite_total = time() - start_time

    print("timing in ms: ")
    print(" Original:", (original_total / 25) * 1000)
    print(" Ours:", (new_total / 25) * 1000)
    print(" Composite:", (composite_total / 25) * 1000)

    x_original_tt = ttnn.to_torch(x_original)
    x_new_tt = ttnn.to_torch(x_new)
    x_tt = ttnn.to_torch(x)

    def mse_diff(a, b):
        return torch.mean((a - b) ** 2)

    print("mse diff")
    print("TT:", mse_diff(x_original_tt, expected_result))
    print("Ours:", mse_diff(x_new_tt, expected_result))
    print("Composite:", mse_diff(composite_result, expected_result))

    print("rms diff:", mse_diff(ttnn.to_torch(rms), expected_rms))
