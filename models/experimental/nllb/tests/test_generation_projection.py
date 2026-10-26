# SPDX-FileCopyrightText: © 2026 Abror Shopulatov

# SPDX-License-Identifier: Apache-2.0

"""TT projection shape check; no learned CPU inference."""

import torch, ttnn

from models.experimental.nllb.tt.backend import Backend


def check(device):
    b = Backend.__new__(Backend)
    b.device = device
    b.kernel = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False
    )
    g = torch.Generator().manual_seed(121)
    w = b.upload(torch.randn(96, 1024, generator=g) * 0.01)
    for n in (2, 31, 32, 33, 63, 64):
        x = b.upload(torch.randn(1, 1, ((n + 31) // 32) * 32, 1024, generator=g) * 0.1)
        whole = ttnn.linear(x, w, transpose_b=True, dtype=ttnn.bfloat16, compute_kernel_config=b.kernel)
        last = ttnn.slice(x, (0, 0, n - 1, 0), (1, 1, n, 1024), pad_value=0.0)
        out = ttnn.linear(last, w, transpose_b=True, dtype=ttnn.bfloat16, compute_kernel_config=b.kernel)
        a = ttnn.to_torch(whole).float()[0, 0, n - 1]
        z = ttnn.to_torch(out).float()[0, 0, 0]
        ttnn.synchronize_device(device)
        print(
            "PROJECTION",
            n,
            "shape",
            list(last.shape),
            "exact",
            torch.equal(a, z),
            "max_abs",
            float((a - z).abs().max()),
            flush=True,
        )
        assert torch.equal(a, z)


def test_projection(nllb_component_runner):
    nllb_component_runner("generation_projection")


if __name__ == "__main__":
    torch.set_num_threads(1)
    d = ttnn.open_device(device_id=0)
    try:
        check(d)
    finally:
        ttnn.close_device(d)
