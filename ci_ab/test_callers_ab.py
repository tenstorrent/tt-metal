# Scratch, not for merge. Callers of the pack thread (matmul, conv2d, binary_ng, unary, reduce, norms, softmax, transpose, concat, tilize,
# untilize, copy), one op call per test call; each output's sha1 goes to $EB_BITS_OUT as "<arm> <case> <sha1>".
import hashlib
import os
import sys

import pytest
import torch
import ttnn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_cls_ab as cls

BF, F32, B8 = ttnn.bfloat16, ttnn.float32, ttnn.bfloat8_b
T, R = ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT
DRAM = ttnn.DRAM_MEMORY_CONFIG


def dev_t(device, shape, dtype=BF, layout=T, seed=1, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    return ttnn.from_torch((torch.rand(shape, generator=g) - 0.5) * scale, dtype=dtype, layout=layout, device=device, memory_config=DRAM)


def sha(t):
    x = ttnn.to_torch(t).contiguous()
    b = x.view(torch.int16) if x.dtype == torch.bfloat16 else x
    return hashlib.sha1(b.numpy().tobytes()).hexdigest()[:16]


def fp32acc():
    return ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True)


def conv(device, c, k, h, w, dt=BF, wdt=BF):
    g = torch.Generator().manual_seed(2)
    x = (torch.rand((1, h, w, c), generator=g) - 0.5)
    wt = (torch.rand((k, c, 3, 3), generator=g) - 0.5) * 0.1
    tin = ttnn.from_torch(x.reshape(1, 1, h * w, c), dtype=dt, layout=R, device=device, memory_config=DRAM)
    tw = ttnn.from_torch(wt, dtype=wdt)
    out = ttnn.conv2d(input_tensor=tin, weight_tensor=tw, device=device, in_channels=c, out_channels=k, batch_size=1, input_height=h,
                      input_width=w, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), conv_config=ttnn.Conv2dConfig(weights_dtype=wdt))
    return out[0] if isinstance(out, (tuple, list)) else out


CASES = {
    "mm_2048x1024x4096": lambda d: ttnn.matmul(dev_t(d, (1, 1, 2048, 1024)), dev_t(d, (1, 1, 1024, 4096), seed=2)),
    "mm_bfp8w_2048x1024x4096": lambda d: ttnn.matmul(dev_t(d, (1, 1, 2048, 1024)), dev_t(d, (1, 1, 1024, 4096), B8, seed=2)),
    "mm_32x4096x4096": lambda d: ttnn.matmul(dev_t(d, (1, 1, 32, 4096)), dev_t(d, (1, 1, 4096, 4096), seed=2)),
    "mm_fp32acc_1024x4096x1024": lambda d: ttnn.matmul(dev_t(d, (1, 1, 1024, 4096)), dev_t(d, (1, 1, 4096, 1024), seed=2), compute_kernel_config=fp32acc()),
    "mm_transpose_a_1024x512x1024": lambda d: ttnn.matmul(dev_t(d, (1, 1, 512, 1024)), dev_t(d, (1, 1, 512, 1024), seed=2), transpose_a=True),
    "linear_bias_gelu_2048x1024x4096": lambda d: ttnn.linear(dev_t(d, (1, 1, 2048, 1024)), dev_t(d, (1, 1, 1024, 4096), seed=2),
                                                              bias=dev_t(d, (1, 1, 1, 4096), seed=3), activation="gelu"),
    "mm_f32_512x512x512": lambda d: ttnn.matmul(dev_t(d, (1, 1, 512, 512), F32), dev_t(d, (1, 1, 512, 512), F32, seed=2)),
    "conv_320x320x64x64": lambda d: conv(d, 320, 320, 64, 64),
    "conv_640x640x32x32_f32": lambda d: conv(d, 640, 640, 32, 32, F32, F32),
    "conv_1280x1280x16x16": lambda d: conv(d, 1280, 1280, 16, 16),
    "add_2048x2048": lambda d: ttnn.add(dev_t(d, (1, 1, 2048, 2048)), dev_t(d, (1, 1, 2048, 2048), seed=2)),
    "add_bcast_row": lambda d: ttnn.add(dev_t(d, (1, 1, 2048, 2048)), dev_t(d, (1, 1, 1, 2048), seed=2)),
    "mul_f32_1024": lambda d: ttnn.multiply(dev_t(d, (1, 1, 1024, 1024), F32), dev_t(d, (1, 1, 1024, 1024), F32, seed=2)),
    "sub_bfp8_2048": lambda d: ttnn.subtract(dev_t(d, (1, 1, 2048, 2048), B8), dev_t(d, (1, 1, 2048, 2048), B8, seed=2)),
    "add_scalar_2048": lambda d: ttnn.add(dev_t(d, (1, 1, 2048, 2048)), 1.5),
    "exp_2048": lambda d: ttnn.exp(dev_t(d, (1, 1, 2048, 2048))),
    "gelu_2048": lambda d: ttnn.gelu(dev_t(d, (1, 1, 2048, 2048))),
    "relu_f32_1024": lambda d: ttnn.relu(dev_t(d, (1, 1, 1024, 1024), F32)),
    "sum_w_2048": lambda d: ttnn.sum(dev_t(d, (1, 1, 2048, 2048)), dim=-1),
    "mean_h_2048": lambda d: ttnn.mean(dev_t(d, (1, 1, 2048, 2048)), dim=-2),
    "max_w_2048": lambda d: ttnn.max(dev_t(d, (1, 1, 2048, 2048)), dim=-1),
    "sum_nc_8x1x1024x1024": lambda d: ttnn.sum(dev_t(d, (8, 1, 1024, 1024)), dim=0),
    "layernorm_2048x4096": lambda d: ttnn.layer_norm(dev_t(d, (1, 1, 2048, 4096)), weight=dev_t(d, (1, 1, 1, 4096), seed=2), bias=dev_t(d, (1, 1, 1, 4096), seed=3)),
    "layernorm_plain_2048x4096": lambda d: ttnn.layer_norm(dev_t(d, (1, 1, 2048, 4096))),
    "rmsnorm_32x1024": lambda d: ttnn.rms_norm(dev_t(d, (1, 1, 32, 1024)), weight=dev_t(d, (1, 1, 1, 1024), seed=2)),
    "rmsnorm_2048x4096": lambda d: ttnn.rms_norm(dev_t(d, (1, 1, 2048, 4096)), weight=dev_t(d, (1, 1, 1, 4096), seed=2)),
    "softmax_2048x2048": lambda d: ttnn.softmax(dev_t(d, (1, 1, 2048, 2048)), dim=-1),
    "softmax_f32_1024": lambda d: ttnn.softmax(dev_t(d, (1, 1, 1024, 1024), F32), dim=-1),
    "transpose_wh_2048": lambda d: ttnn.transpose(dev_t(d, (1, 1, 2048, 2048)), -2, -1),
    "transpose_wh_f32_1024": lambda d: ttnn.transpose(dev_t(d, (1, 1, 1024, 1024), F32), -2, -1),
    "transpose_hc_8x64x512x64": lambda d: ttnn.transpose(dev_t(d, (1, 64, 512, 64)), 1, 2),
    "permute_0213_bf16": lambda d: ttnn.permute(dev_t(d, (1, 32, 256, 128)), (0, 2, 1, 3)),
    "concat_w_2x_2048": lambda d: ttnn.concat([dev_t(d, (1, 1, 2048, 1024)), dev_t(d, (1, 1, 2048, 1024), seed=2)], dim=-1),
}
for name in cls.CASES:
    CASES["cls_" + name] = (lambda n: (lambda d: cls.run_case(d, n)))(name)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("case", list(CASES))
def test_caller(device, case, request):
    out = CASES[case](device)
    ttnn.synchronize_device(device)
    p = os.environ.get("EB_BITS_OUT")
    if p:
        with open(p, "a") as f:
            f.write(f"{os.environ.get('EB_ARM', '?')} {case} {sha(out)}\n")
