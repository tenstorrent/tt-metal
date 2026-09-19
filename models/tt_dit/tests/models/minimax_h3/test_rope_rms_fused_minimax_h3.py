# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""RoPE with the q/k RMS normalisation folded in (`rotary_embedding_llama(..., rms_norm_eps=eps)`) against today's
`ttnn.rms_norm` -> `rotary_embedding_llama` pair at the H3 VAE decoder's shape, one chip: both against a float64 reference
(the fused form must be at least as accurate), both timed."""

import time

import pytest
import torch
from loguru import logger

import ttnn

from ....utils.mochi import get_rot_transformation_mat
from ....utils.tensor import bf16_tensor

SINGLE_DEVICE = [pytest.param((1, 1), {"l1_small_size": 65536}, id="single_device")]
HEADS, SEQ, HEAD_DIM, EPS = 32, 1824, 64, 1e-5


def _timed(mesh_device, fn, n=15):
    fn()
    ttnn.synchronize_device(mesh_device)
    best = float("inf")
    for _ in range(n):
        ttnn.synchronize_device(mesh_device)
        mark = time.perf_counter()
        out = fn()
        ttnn.synchronize_device(mesh_device)
        best = min(best, time.perf_counter() - mark)
    return out, best * 1e3


def _reference(x, cos, sin, trans, eps):
    """float64: RMS over the head dim, then x*cos + (x @ trans_mat per 32-column tile) * sin."""
    x = x.double()
    xn = x / torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    t = trans.double()  # (32, 32) applied to each 32-wide column tile
    rot = (xn.reshape(*xn.shape[:-1], HEAD_DIM // 32, 32) @ t).reshape(xn.shape)
    return xn * cos.double() + rot * sin.double()


@pytest.mark.timeout(900)
@pytest.mark.parametrize("view", ["heads", "batch"])
@pytest.mark.parametrize(("mesh_device", "device_params"), SINGLE_DEVICE, indirect=["mesh_device", "device_params"])
def test_rope_rms_fused(mesh_device, view):
    torch.manual_seed(0)
    x = torch.randn(1, HEADS, SEQ, HEAD_DIM) * 2.0
    cos = torch.randn(1, 1, SEQ, HEAD_DIM).clamp(-1, 1)
    sin = torch.randn(1, 1, SEQ, HEAD_DIM).clamp(-1, 1)
    trans_t = get_rot_transformation_mat()
    trans_t = trans_t.reshape(32, 32) if trans_t.numel() == 1024 else trans_t.reshape(-1, 32)[:32]
    dev = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, device=mesh_device, layout=ttnn.TILE_LAYOUT)  # noqa: E731
    x_dev, cos_dev, sin_dev = dev(x), dev(cos), dev(sin)
    trans = bf16_tensor(get_rot_transformation_mat(), device=mesh_device)
    if view == "batch":
        shp = ttnn.Shape([HEADS, 1, SEQ, HEAD_DIM])
        x_dev = ttnn.reshape(x_dev, shp, shp)
    rms_cfg = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=True
    )
    rope_cfg = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
    )

    def today():
        n = ttnn.rms_norm(x_dev, epsilon=EPS, compute_kernel_config=rms_cfg)
        return ttnn.experimental.rotary_embedding_llama(n, cos_dev, sin_dev, trans, compute_kernel_config=rope_cfg)

    def fused():
        return ttnn.experimental.rotary_embedding_llama(
            x_dev, cos_dev, sin_dev, trans, compute_kernel_config=rope_cfg, rms_norm_eps=EPS
        )

    out_t, t_today = _timed(mesh_device, today)
    out_f, t_fused = _timed(mesh_device, fused)
    x_bf16 = ttnn.to_torch(ttnn.from_torch(x, dtype=ttnn.bfloat16)).float()  # what the device saw
    ref = _reference(x_bf16.reshape(1, HEADS, SEQ, HEAD_DIM), ttnn.to_torch(cos_dev).float(), ttnn.to_torch(sin_dev).float(), ttnn.to_torch(trans).float().reshape(-1, 32)[:32], EPS)
    got_t = ttnn.to_torch(out_t).float().reshape(1, HEADS, SEQ, HEAD_DIM).double()
    got_f = ttnn.to_torch(out_f).float().reshape(1, HEADS, SEQ, HEAD_DIM).double()
    scale = ref.std()
    rmse_t = float((got_t - ref).pow(2).mean().sqrt() / scale)
    rmse_f = float((got_f - ref).pow(2).mean().sqrt() / scale)
    diff_tf = float((got_t - got_f).abs().max())
    logger.info(
        f"ROPE_RMS view={view}: rms_norm+rope {t_today:.3f} ms, fused {t_fused:.3f} ms (saves {t_today - t_fused:.3f} ms); "
        f"rel-RMSE vs float64: today {rmse_t:.3e}, fused {rmse_f:.3e}; max |today - fused| {diff_tf:.3e}"
    )
    assert torch.isfinite(got_f).all()
    assert rmse_f <= 1.05 * rmse_t, f"fused RMS+RoPE is less accurate than the pair: {rmse_f:.3e} vs {rmse_t:.3e}"
