# Scratch, not for merge: a spread of ttnn ops at LoFi and HiFi4 so their compute kernels are JIT-built in each arm's cache.
import pytest
import torch
import ttnn

FID = ["LoFi", "HiFi4"]


def _cfg(device, fid):
    return ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=getattr(ttnn.MathFidelity, fid), fp32_dest_acc_en=False, math_approx_mode=False)


def _t(device, *shape, dt=ttnn.bfloat16):
    torch.manual_seed(0)
    return ttnn.from_torch(torch.randn(*shape), dtype=dt, layout=ttnn.TILE_LAYOUT, device=device)


@pytest.mark.parametrize("fid", FID)
@pytest.mark.parametrize("op", ["layer_norm", "rms_norm", "softmax", "scale_mask_softmax", "sdpa", "matmul", "sum",
                                "add_fpu", "add_sfpu", "mul_bcast_row", "relu", "exp", "transpose", "untilize", "concat"])
def test_k(device, op, fid):
    c = _cfg(device, fid)
    x = _t(device, 1, 1, 256, 1024)
    if op == "layer_norm":
        out = ttnn.layer_norm(x, weight=_t(device, 1, 1, 1, 1024), compute_kernel_config=c)
    elif op == "rms_norm":
        out = ttnn.rms_norm(x, weight=_t(device, 1, 1, 1, 1024), compute_kernel_config=c)
    elif op == "softmax":
        out = ttnn.softmax(x, dim=-1, compute_kernel_config=c)
    elif op == "scale_mask_softmax":
        out = ttnn.scale_mask_softmax(x, 0.125, _t(device, 1, 1, 32, 1024), compute_kernel_config=c)
    elif op == "sdpa":
        q, k, v = (_t(device, 1, 8, 256, 64) for _ in range(3))
        out = ttnn.transformer.scaled_dot_product_attention(q, k, v, is_causal=True, compute_kernel_config=c)
    elif op == "matmul":
        out = ttnn.matmul(x, _t(device, 1, 1, 1024, 256), compute_kernel_config=c)
    elif op == "sum":
        out = ttnn.sum(x, dim=-1, compute_kernel_config=c)
    elif op == "add_fpu":
        out = ttnn.add(x, _t(device, 1, 1, 256, 1024), fast_and_approximate_mode=True)
    elif op == "add_sfpu":
        out = ttnn.add(x, _t(device, 1, 1, 256, 1024))
    elif op == "mul_bcast_row":
        out = ttnn.multiply(x, _t(device, 1, 1, 1, 1024), fast_and_approximate_mode=True)
    elif op == "relu":
        out = ttnn.relu(x)
    elif op == "exp":
        out = ttnn.exp(x)
    elif op == "transpose":
        out = ttnn.transpose(x, -2, -1)
    elif op == "untilize":
        out = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
    else:
        out = ttnn.concat([x, x], dim=-1)
    ttnn.to_torch(out)
