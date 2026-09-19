# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""C-06 first experiment, one chip: the decoder's q/k rms_norm alone, rotary_embedding_llama alone, the pair back
to back, and RoPE with the input viewed as (heads, 1, S, D) so the factory parallelises over 96 cores instead of
57. Min-of-N wall per op. Not a gate; it sizes the fused-RMS-prologue idea and the free view.
    pytest models/tt_dit/tests/models/minimax_h3/tools/rope_rms_probe.py -q -s
"""
import time

import pytest
import torch
from loguru import logger

import ttnn

from .....utils.mochi import get_rot_transformation_mat
from .....utils.tensor import bf16_tensor

SINGLE_DEVICE = [pytest.param((1, 1), {"l1_small_size": 65536}, id="single_device")]
HEADS, SEQ, HEAD_DIM = 32, 1824, 64
REPEATS = 20


def _timed(mesh_device, fn, n=REPEATS):
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


@pytest.mark.parametrize(("mesh_device", "device_params"), SINGLE_DEVICE, indirect=["mesh_device", "device_params"])
def test_rope_rms_probe(mesh_device):
    torch.manual_seed(0)
    dev = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, device=mesh_device, layout=ttnn.TILE_LAYOUT)  # noqa: E731
    x = dev(torch.randn(1, HEADS, SEQ, HEAD_DIM))
    cos = dev(torch.randn(1, 1, SEQ, HEAD_DIM))
    sin = dev(torch.randn(1, 1, SEQ, HEAD_DIM))
    trans = bf16_tensor(get_rot_transformation_mat(), device=mesh_device)
    rms_cfg = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=True
    )
    rope_cfg = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )
    rms = lambda t: ttnn.rms_norm(t, epsilon=1e-5, compute_kernel_config=rms_cfg)  # noqa: E731
    rope = lambda t: ttnn.experimental.rotary_embedding_llama(t, cos, sin, trans, compute_kernel_config=rope_cfg)  # noqa: E731

    _, t_rms = _timed(mesh_device, lambda: rms(x))
    out_rope, t_rope = _timed(mesh_device, lambda: rope(x))
    _, t_pair = _timed(mesh_device, lambda: rope(rms(x)))
    logger.info(f"C06_PROBE rms_norm {t_rms:.3f} ms | rope {t_rope:.3f} ms | rms->rope {t_pair:.3f} ms (1,32,1824,64)")

    # The free view: heads on the batch axis.
    xv = ttnn.reshape(x, (HEADS, 1, SEQ, HEAD_DIM))
    try:
        out_v, t_rope_v = _timed(mesh_device, lambda: rope(xv))
        same = torch.equal(ttnn.to_torch(out_v).reshape(1, HEADS, SEQ, HEAD_DIM), ttnn.to_torch(out_rope))
        logger.info(f"C06_PROBE rope on (32,1,1824,64) view {t_rope_v:.3f} ms vs {t_rope:.3f} ms; bit-identical {same}")
        _, t_rms_v = _timed(mesh_device, lambda: rms(xv))
        logger.info(f"C06_PROBE rms_norm on (32,1,1824,64) view {t_rms_v:.3f} ms vs {t_rms:.3f} ms")
    except Exception as err:  # noqa: BLE001
        logger.info(f"C06_PROBE view rejected: {type(err).__name__}: {str(err)[:200]}")
