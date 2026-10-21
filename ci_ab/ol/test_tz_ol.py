# Scratch op-level cases (not for merge): ttnn.tilize, tilize_with_val_padding and conv3d on the tilize_block paths #60672
# and its fixes change. Every call appends "<case> <sha256 of the output>" to $TT_METAL_CACHE/sha.txt (one cache per arm).
import hashlib
import os

import pytest
import torch
import ttnn

CASES = {
    "uint8_512x1024": (ttnn.uint8, (1, 1, 512, 1024)),
    "uint8_2048x64": (ttnn.uint8, (1, 1, 2048, 64)),
    "uint8_32x128256": (ttnn.uint8, (1, 1, 32, 128256)),
    "uint8_1024x96": (ttnn.uint8, (1, 1, 1024, 96)),
    "int32_512x1024": (ttnn.int32, (1, 1, 512, 1024)),
    "fp32_512x1024": (ttnn.float32, (1, 1, 512, 1024)),
    "bf16_4096x32": (ttnn.bfloat16, (1, 1, 4096, 32)),
    "bf16_4096x1024": (ttnn.bfloat16, (1, 1, 4096, 1024)),
    "bf16_to_bfp8_2048x64": (ttnn.bfloat16, (1, 1, 2048, 64)),
    "bf16_to_bfp8_4096x1024": (ttnn.bfloat16, (1, 1, 4096, 1024)),
    "bf16_to_bfp8_512x96": (ttnn.bfloat16, (1, 1, 512, 96)),
    "conv2d_rn50_l2_3x3": None,
    "conv2d_rn50_l3_1x1": None,
    "matmul_2k2k2k_bf16_hifi2": None,
    "matmul_2k2k4k_bfp8_lofi": None,
    "sum_rm_w_bf16_2048x1024": None,
    "sum_rm_h_bf16_2048x1024": None,
    "sdpa_decode_b32_s2k": None,
}
_CW = {}


def _ckc(device, fid):
    return ttnn.WormholeComputeKernelConfig(math_fidelity=fid, math_approx_mode=False, fp32_dest_acc_en=False,
                                            packer_l1_acc=True, dst_full_sync_en=False)


def _conv2d(device, case):
    b, cin, cout, h, w, kh, p = (8, 128, 128, 28, 28, 3, 1) if "3x3" in case else (8, 256, 1024, 14, 14, 1, 0)
    g = torch.Generator().manual_seed(7)
    x = ttnn.from_torch(torch.randn(1, 1, b * h * w, cin, generator=g), dtype=ttnn.bfloat16,
                        layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    if case not in _CW:
        _CW[case] = ttnn.from_torch(torch.randn(cout, cin, kh, kh, generator=g), dtype=ttnn.bfloat16)
    o, [_CW[case], _] = ttnn.conv2d(
        input_tensor=x, weight_tensor=_CW[case], bias_tensor=None, return_weights_and_bias=True, in_channels=cin,
        out_channels=cout, batch_size=b, input_height=h, input_width=w, kernel_size=(kh, kh), stride=(1, 1),
        padding=(p, p), dilation=(1, 1), groups=1, device=device, conv_config=ttnn.Conv2dConfig(weights_dtype=ttnn.bfloat8_b),
        compute_config=_ckc(device, ttnn.MathFidelity.LoFi), dtype=ttnn.bfloat16)
    return ttnn.to_torch(o)


def _matmul(device, case):
    n, wdt, fid = (2048, ttnn.bfloat16, ttnn.MathFidelity.HiFi2) if "bf16" in case else (4096, ttnn.bfloat8_b, ttnn.MathFidelity.LoFi)
    g = torch.Generator().manual_seed(7)
    a = ttnn.from_torch(torch.randn(1, 1, 2048, 2048, generator=g), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    w = ttnn.from_torch(torch.randn(1, 1, 2048, n, generator=g), dtype=wdt, layout=ttnn.TILE_LAYOUT, device=device)
    gr = device.compute_with_storage_grid_size()
    o = ttnn.matmul(a, w, compute_kernel_config=_ckc(device, fid), core_grid=ttnn.CoreGrid(y=gr.y, x=gr.x))
    return ttnn.to_torch(o)


def _record(case, out):
    p = os.path.join(os.environ.get("TT_METAL_CACHE", "/tmp"), "sha.txt")
    with open(p, "a") as f:
        f.write(f"{case} {hashlib.sha256(out.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()[:16]}\n")


def _conv3d(device):
    # LTX-2 audio vocoder conv (models/tt_dit/layers/audio_ops.py): (B, T, 1, 1, C) fp32 row-major
    c_in, c_out, k, t_len, blocking = 384, 384, 7, 256, (128, 128, 16, 1, 1)
    g = torch.Generator().manual_seed(7)
    x = torch.randn((1, t_len, 1, 1, c_in), generator=g)
    w = torch.randn((c_out, c_in, k, 1, 1), generator=g) * 0.05
    b = torch.randn((1, c_out), generator=g)
    tx = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    tw = ttnn.from_torch(w, dtype=ttnn.float32, pad_value=0)
    tw = ttnn.experimental.prepare_conv3d_weights(weight_tensor=tw, C_in_block=blocking[0], device=device)
    tb = ttnn.from_torch(b, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, pad_value=0)
    cfg = ttnn.Conv3dConfig(
        weights_dtype=ttnn.float32, output_layout=ttnn.ROW_MAJOR_LAYOUT, C_in_block=blocking[0],
        C_out_block=blocking[1], T_out_block=blocking[2], H_out_block=blocking[3], W_out_block=blocking[4],
        compute_with_storage_grid_size=device.compute_with_storage_grid_size())
    cc = ttnn.init_device_compute_kernel_config(device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
                                                fp32_dest_acc_en=True, packer_l1_acc=False)
    y = ttnn.experimental.conv3d(
        input_tensor=tx, weight_tensor=tw, bias_tensor=tb, device=device, config=cfg, output_channels=c_out,
        kernel_size=(k, 1, 1), stride=(1, 1, 1), padding=(k // 2, 0, 0), padding_mode="zeros",
        dtype=ttnn.float32, compute_kernel_config=cc)
    return ttnn.to_torch(y)


def _sum_rm(device, case):
    g = torch.Generator().manual_seed(7)
    x = ttnn.from_torch(torch.randn(1, 1, 2048, 1024, generator=g).to(torch.bfloat16), dtype=ttnn.bfloat16,
                        layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    return ttnn.to_torch(ttnn.sum(x, dim=-1 if "_w_" in case else -2))


def _sdpa_decode(device, case):
    import numpy as np
    b, s, nh, nkv, d = 32, 2048, 8, 1, 128
    g = torch.Generator().manual_seed(7)
    mk = lambda sh, dt: ttnn.from_torch(torch.randn(sh, generator=g), dtype=dt, layout=ttnn.TILE_LAYOUT, device=device,
                                        memory_config=ttnn.DRAM_MEMORY_CONFIG)
    q = mk((1, b, nh, d), ttnn.bfloat16)
    k, v = mk((b, nkv, s, d), ttnn.bfloat8_b), mk((b, nkv, s, d), ttnn.bfloat8_b)
    pos = np.linspace(0, s - 1, b, dtype=np.int32).tolist()
    pc = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=(8, 8), q_chunk_size=32, k_chunk_size=256, exp_approx_mode=False)
    o = ttnn.transformer.scaled_dot_product_attention_decode(
        q, k, v, cur_pos=pos, scale=d ** -0.5, program_config=pc,
        compute_kernel_config=_ckc(device, ttnn.MathFidelity.HiFi2), memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return ttnn.to_torch(o)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("case", list(CASES))
def test_tz(device, case):
    if case.startswith("sum_rm"):
        _record(case, _sum_rm(device, case))
        return
    if case.startswith("sdpa_decode"):
        _record(case, _sdpa_decode(device, case))
        return
    if case.startswith("conv2d"):
        _record(case, _conv2d(device, case))
        return
    if case.startswith("matmul"):
        _record(case, _matmul(device, case))
        return
    dt, shape = CASES[case]
    if case.startswith("c3d"):
        _record(case, _conv3d(device))
        return
    g = torch.Generator().manual_seed(7)
    if dt == ttnn.uint8:
        t = torch.randint(0, 256, shape, generator=g, dtype=torch.uint8)
    elif dt in (ttnn.int32, ttnn.uint32):
        t = torch.randint(0, 2**20, shape, generator=g, dtype=torch.int32)
    else:
        t = torch.randn(shape, generator=g).to(torch.float32 if dt == ttnn.float32 else torch.bfloat16)
    if case.startswith("twvp"):
        padded = list(shape[:-2]) + [(shape[-2] + 31) // 32 * 32, (shape[-1] + 31) // 32 * 32]
        x = ttnn.from_torch(t, dtype=dt, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        out = ttnn.to_torch(ttnn.tilize_with_val_padding(x, padded, 0, use_multicore=True))
        assert torch.equal(out[..., : shape[-2], : shape[-1]].to(t.dtype), t)
        assert not out[..., shape[-2] :, :].any() and not out[..., :, shape[-1] :].any()
        _record(case, out)
        return
    if "_hs_" in case:
        m = ttnn.create_sharded_memory_config(
            shape, core_grid=ttnn.CoreGrid(y=8, x=8), strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR)
        x = ttnn.from_torch(t, dtype=dt, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=m)
        out = ttnn.to_torch(ttnn.tilize(x, memory_config=m, use_multicore=True))
        assert torch.equal(out.to(t.dtype), t)
        _record(case, out)
        return
    x = ttnn.from_torch(t, dtype=dt, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    if case.startswith("bf16_to_bfp8"):
        y = ttnn.tilize(x, dtype=ttnn.bfloat8_b, use_multicore=True)
        yt = ttnn.to_torch(y).float()
        assert torch.allclose(yt, t.float(), rtol=0.05, atol=0.05)
        _record(case, yt)
        return
    out = ttnn.to_torch(ttnn.tilize(x, use_multicore=True))
    assert torch.equal(out.to(t.dtype), t)
    _record(case, out)
