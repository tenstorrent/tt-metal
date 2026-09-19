# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""SDPA writing the concat-heads layout itself (`output_concat_heads=True`) against SDPA + `nlp_concat_heads`, at the
H3 VAE decoder's shape (32 heads, 1824 padded / 1797 logical tokens, head dim 64), one chip: bit-identical, both timed."""

import time

import pytest
import torch
from loguru import logger

import ttnn

SINGLE_DEVICE = [pytest.param((1, 1), {"l1_small_size": 65536}, id="single_device")]
HEADS, SEQ, VALID, HEAD_DIM = 32, 1824, 1797, 64


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


@pytest.mark.timeout(900)
@pytest.mark.parametrize("valid_len", [VALID, SEQ], ids=["view1797", "full1824"])
@pytest.mark.parametrize(("mesh_device", "device_params"), SINGLE_DEVICE, indirect=["mesh_device", "device_params"])
def test_sdpa_concat_heads_matches(mesh_device, valid_len):
    torch.manual_seed(0)
    dev = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, device=mesh_device, layout=ttnn.TILE_LAYOUT)  # noqa: E731
    q, k, v = (dev(torch.randn(1, HEADS, SEQ, HEAD_DIM)) for _ in range(3))
    padded = ttnn.Shape([1, HEADS, SEQ, HEAD_DIM])
    if valid_len != SEQ:
        logical = ttnn.Shape([1, HEADS, valid_len, HEAD_DIM])
        q, k, v = (ttnn.reshape(t, logical, padded) for t in (q, k, v))
    grid = mesh_device.compute_with_storage_grid_size()
    cfg = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=grid, q_chunk_size=128, k_chunk_size=128, exp_approx_mode=False)
    kcfg = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=False
    )
    dim = HEADS * HEAD_DIM

    def chain():
        out = ttnn.transformer.scaled_dot_product_attention(
            q, k, v, attn_mask=None, is_causal=False, program_config=cfg, compute_kernel_config=kcfg
        )
        if out.shape[-2] != SEQ:
            out = ttnn.reshape(out, padded, padded)
        return ttnn.reshape(ttnn.experimental.nlp_concat_heads(out), (1, SEQ, dim))

    def fused():
        out = ttnn.transformer.scaled_dot_product_attention(
            q, k, v, attn_mask=None, is_causal=False, program_config=cfg, compute_kernel_config=kcfg, output_concat_heads=True
        )
        full = ttnn.Shape([1, 1, SEQ, dim])
        if out.shape[-2] != SEQ:
            out = ttnn.reshape(out, full, full)
        return ttnn.reshape(out, (1, SEQ, dim))

    ref, t_chain = _timed(mesh_device, chain)
    got, t_fused = _timed(mesh_device, fused)
    ref_t, got_t = ttnn.to_torch(ref), ttnn.to_torch(got)
    rows = valid_len  # rows past the logical length are pad
    same = torch.equal(got_t[:, :rows], ref_t[:, :rows])
    n_diff = int((got_t[:, :rows] != ref_t[:, :rows]).sum())
    logger.info(
        f"SDPA_CONCAT valid={valid_len}: sdpa+concat_heads {t_chain:.3f} ms, sdpa(output_concat_heads) {t_fused:.3f} ms "
        f"(saves {t_chain - t_fused:.3f} ms); valid rows bit-identical {same} ({n_diff} differ)"
    )
    assert tuple(got.shape) == (1, SEQ, dim)
    assert same
