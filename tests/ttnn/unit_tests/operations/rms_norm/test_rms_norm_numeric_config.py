# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 1 pins: 16-bit DEST accumulation (fp32_dest_acc_en=False) and bfloat8_b on every kernel path
the precision matrix (TILE / interleaved) does not reach.

  * WIDTH_SHARDED (R3, zero-copy shards): bf16 and bf8b input at both DEST widths — the gather / rstd-mcast
    payload stride is the DEST-width page (ACC_TILE_BYTES), so the collective must agree on it end to end.
  * ROW_MAJOR input at 16-bit DEST with a ROW_MAJOR gamma: the fp32 RM gamma case is the one where the
    start-up gamma stick block (Wc * 4 KiB) is LARGER than the cb_normed allocation it aliases
    (B * Wc * 2 KiB at B = 1) — the hosting CB must be sized as the max of the two.
  * bf8b without gamma (D packs straight into a Bfp8_b output CB).
  * {float32, fp32_dest_acc_en=False} keeps refusing with ExcludedCell.

The golden suite's tolerances apply: PCC 0.995 (bf16) / 0.99 (bf8b) and relative RMS <= 0.04 / 0.10.
"""

import pytest
import torch
import ttnn

from ttnn.operations._op_contract import ExcludedCell
from ttnn.operations.rms_norm import rms_norm

TORCH_DTYPE = {ttnn.bfloat16: torch.bfloat16, ttnn.bfloat8_b: torch.bfloat16, ttnn.float32: torch.float32}
GATES = {ttnn.bfloat16: (0.995, 0.04), ttnn.bfloat8_b: (0.99, 0.10)}  # (pcc, rel_rms)
DTYPE_IDS = {ttnn.bfloat16: "bf16", ttnn.bfloat8_b: "bfp8"}


def _torch_rms_norm(x, gamma=None, epsilon=1e-6):
    xf = x.to(torch.float32)
    y = xf * torch.rsqrt(torch.mean(xf * xf, dim=-1, keepdim=True) + epsilon)
    return y * gamma.to(torch.float32).reshape(-1) if gamma is not None else y


def _config(fp32_acc, fidelity=ttnn.MathFidelity.HiFi2):
    return ttnn.ComputeConfigDescriptor(math_fidelity=fidelity, fp32_dest_acc_en=fp32_acc, math_approx_mode=False)


def _check(ttnn_out, expected, shape, layout, dtype):
    assert list(ttnn_out.shape) == list(shape)
    assert ttnn_out.layout == layout and ttnn_out.dtype == dtype
    actual = ttnn.to_torch(ttnn_out).to(torch.float32)
    e, a = expected.flatten(), actual.flatten()
    pcc = torch.corrcoef(torch.stack([e, a]))[0, 1].item()
    rel_rms = ((a - e).pow(2).mean().sqrt() / e.pow(2).mean().sqrt()).item()
    pcc_gate, rms_gate = GATES[dtype]
    print(f"\nNUMERIC shape={tuple(shape)} dtype={dtype} pcc={pcc:.6f} rel_rms={rel_rms:.5f}")
    assert torch.isfinite(actual).all()
    assert pcc >= pcc_gate, f"PCC {pcc:.6f} < {pcc_gate}"
    assert rel_rms <= rms_gate, f"rel RMS {rel_rms:.5f} > {rms_gate}"


def _inputs(shape, dtype, with_gamma=True, seed=3):
    torch.manual_seed(seed)
    x = torch.randn(shape, dtype=torch.float32).to(TORCH_DTYPE[dtype])
    g = torch.randn(shape[-1], dtype=torch.float32).to(TORCH_DTYPE[dtype]) if with_gamma else None
    return x, g


@pytest.mark.parametrize("fp32_acc", [True, False], ids=["fp32_acc", "bf16_acc"])
@pytest.mark.parametrize("dtype", list(DTYPE_IDS), ids=list(DTYPE_IDS.values()))
def test_rms_norm_width_sharded_numeric_config(device, dtype, fp32_acc):
    """R3: W=2048 over 8 x 1 cores ([32, 256] shards); the collective payload follows the DEST width."""
    if device.compute_with_storage_grid_size().x < 8:
        pytest.skip("needs a compute grid at least 8 cores wide")
    shape = (1, 1, 32, 2048)
    x, g = _inputs(shape, dtype)
    sharded = ttnn.create_sharded_memory_config(
        shape=(32, 256),
        core_grid=ttnn.CoreGrid(x=8, y=1),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    ttnn_x = ttnn.from_torch(x, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=sharded)
    ttnn_g = ttnn.from_torch(g.reshape(1, 1, 1, -1), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    expected = _torch_rms_norm(ttnn.to_torch(ttnn_x), ttnn.to_torch(ttnn_g))

    out = rms_norm(ttnn_x, gamma=ttnn_g, memory_config=ttnn_x.memory_config(), compute_kernel_config=_config(fp32_acc))
    assert out.memory_config().memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED
    _check(out, expected, shape, ttnn.TILE_LAYOUT, dtype)


@pytest.mark.parametrize("gamma_dtype", [ttnn.bfloat16, ttnn.float32], ids=["bf16_gamma", "fp32_gamma"])
@pytest.mark.parametrize(
    "shape", [pytest.param((2, 4, 128, 512), id="R1_row_split"), pytest.param((1, 1, 32, 4096), id="R2_w_split")]
)
def test_rms_norm_row_major_16bit_dest(device, shape, gamma_dtype):
    """ROW_MAJOR x and ROW_MAJOR gamma under 16-bit DEST (T_acc = 2 KiB); fp32 gamma is the aliased-CB max case."""
    dtype = ttnn.bfloat16
    x, g = _inputs(shape, dtype)
    g = g.to(TORCH_DTYPE[gamma_dtype])
    ttnn_x = ttnn.from_torch(x, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_g = ttnn.from_torch(g.reshape(1, 1, 1, -1), dtype=gamma_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    expected = _torch_rms_norm(x, g)

    out = rms_norm(ttnn_x, gamma=ttnn_g, compute_kernel_config=_config(False))
    _check(out, expected, shape, ttnn.ROW_MAJOR_LAYOUT, dtype)


@pytest.mark.parametrize("fp32_acc", [True, False], ids=["fp32_acc", "bf16_acc"])
def test_rms_norm_bf8b_no_gamma(device, fp32_acc):
    shape, dtype = (4, 128, 512), ttnn.bfloat8_b
    x, _ = _inputs(shape, dtype, with_gamma=False)
    ttnn_x = ttnn.from_torch(x, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    expected = _torch_rms_norm(ttnn.to_torch(ttnn_x))

    out = rms_norm(ttnn_x, compute_kernel_config=_config(fp32_acc))
    _check(out, expected, shape, ttnn.TILE_LAYOUT, dtype)


def test_rms_norm_rejects_fp32_with_16bit_dest(device, expect_error):
    """{float32, fp32_dest_acc_en=False} is a permanent EXCLUSION — refused before any dispatch."""
    x = torch.randn(1, 1, 32, 64, dtype=torch.float32)
    ttnn_x = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    with expect_error(ExcludedCell, "fp32_dest_acc_en"):
        rms_norm(ttnn_x, compute_kernel_config=_config(False))
