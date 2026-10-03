import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tests.test_mhc_microbench import traced_ms

D, T = 5120, 4


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {
                "l1_small_size": 16384,
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "trace_region_size": 100_000_000,
            },
            id="ring",
        )
    ],
    indirect=True,
)
@torch.no_grad()
def test_mb2(mesh_device):
    md = mesh_device
    rep = ttnn.ReplicateTensorToMesh(md)
    up = lambda t, dt=ttnn.float32: ttnn.from_torch(
        t, device=md, dtype=dt, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=rep
    )
    hifi4 = ttnn.init_device_compute_kernel_config(
        md.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True
    )
    hifi2 = ttnn.init_device_compute_kernel_config(
        md.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True, packer_l1_acc=True
    )
    xt = up(torch.randn(T, 1, 4, D))
    xs = up(torch.randn(4, 1, T, D))
    xp = up(torch.randn(1, 1, T, 4 * D))
    fnp = up(torch.randn(1, 1, 4 * D, 24) * 0.02)
    fnS = up(torch.randn(4, 1, D, 24) * 0.02)
    ones = up(torch.ones(1, 1, 4 * D, 32))
    onesS = up(torch.ones(4, 1, D, 32))
    r = {}
    r["reshape xt->packed"] = traced_ms(md, lambda: ttnn.reshape(xt, [1, 1, T, 4 * D]))
    r["packed matmul hifi4"] = traced_ms(md, lambda: ttnn.matmul(xp, fnp, compute_kernel_config=hifi4))
    r["packed matmul hifi2"] = traced_ms(md, lambda: ttnn.matmul(xp, fnp, compute_kernel_config=hifi2))
    r["stream-major batched matmul"] = traced_ms(md, lambda: ttnn.matmul(xs, fnS, compute_kernel_config=hifi4))
    m = ttnn.matmul(xs, fnS, compute_kernel_config=hifi4)
    r["sum dim0 of [4,1,T,24]"] = traced_ms(md, lambda: ttnn.sum(m, dim=0, keepdim=True))
    r["x*x packed"] = traced_ms(md, lambda: ttnn.multiply(xp, xp))
    x2 = ttnn.multiply(xp, xp)
    r["sum last dim packed"] = traced_ms(md, lambda: ttnn.sum(x2, dim=-1, keepdim=True))
    r["mean last dim packed"] = traced_ms(md, lambda: ttnn.mean(x2, dim=-1, keepdim=True))
    r["sumsq via matmul ones"] = traced_ms(md, lambda: ttnn.matmul(x2, ones, compute_kernel_config=hifi4))
    r["x*x + sum stream-major"] = traced_ms(md, lambda: ttnn.sum(ttnn.multiply(xs, xs), dim=-1, keepdim=True))
    r["x*x + matmul ones stream-major"] = traced_ms(
        md, lambda: ttnn.matmul(ttnn.multiply(xs, xs), onesS, compute_kernel_config=hifi4)
    )
    pre = up(torch.rand(T, 1, 1, 4))
    comb = up(torch.rand(T, 1, 4, 4))
    post = up(torch.rand(T, 1, 4, 1))
    y = up(torch.randn(T, 1, 1, D))
    r["tok-major collapse matmul"] = traced_ms(md, lambda: ttnn.matmul(pre, xt, compute_kernel_config=hifi4))

    def expand():
        return ttnn.add(ttnn.matmul(comb, xt, compute_kernel_config=hifi4), ttnn.multiply(post, y))

    r["tok-major expand (3 ops)"] = traced_ms(md, expand)
    for k, v in r.items():
        print(f"MB2 {k:40s} {v:7.3f}")
