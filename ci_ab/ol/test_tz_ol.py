# Scratch op-level cases (not for merge): ttnn.tilize, tilize_with_val_padding and conv3d on the tilize_block paths #60672
# and its fixes change. Every call appends "<case> <sha256 of the output>" to $TT_METAL_CACHE/sha.txt (one cache per arm).
import hashlib
import os

import pytest
import torch
import ttnn

CASES = {
    "int32_512x1024": (ttnn.int32, (1, 1, 512, 1024)),
    "uint32_512x1024": (ttnn.uint32, (1, 1, 512, 1024)),
    "fp32_512x1024": (ttnn.float32, (1, 1, 512, 1024)),
    "int32_2048x64": (ttnn.int32, (1, 1, 2048, 64)),
    "int32_32x128256": (ttnn.int32, (1, 1, 32, 128256)),
    "fp32_hs_4096x256": (ttnn.float32, (1, 1, 4096, 256)),
    "twvp_int32_30x1000": (ttnn.int32, (1, 1, 30, 1000)),
    "twvp_fp32_1x8x30x250": (ttnn.float32, (1, 8, 30, 250)),
    "c3d_fp32_384_k7": (ttnn.float32, None),
    "bf16_4096x32": (ttnn.bfloat16, (1, 1, 4096, 32)),
    "bf16_to_bfp8_2048x64": (ttnn.bfloat16, (1, 1, 2048, 64)),
}


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


@pytest.mark.parametrize("case", list(CASES))
def test_tz(device, case):
    dt, shape = CASES[case]
    if case.startswith("c3d"):
        _record(case, _conv3d(device))
        return
    g = torch.Generator().manual_seed(7)
    if dt in (ttnn.int32, ttnn.uint32):
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
        assert torch.allclose(ttnn.to_torch(y).float(), t.float(), rtol=0.05, atol=0.05)
        return
    out = ttnn.to_torch(ttnn.tilize(x, use_multicore=True))
    assert torch.equal(out.to(t.dtype), t)
    _record(case, out)
