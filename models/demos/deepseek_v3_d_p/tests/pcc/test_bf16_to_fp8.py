# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Standalone accuracy test for ttnn.experimental.deepseek_prefill.bf16_to_fp8.

The op takes a BF16 TILE-layout tensor and returns a UINT8 TILE-layout tensor whose
bytes are Fp8_e4m3-encoded values (the same trick used by the dispatch op's FP8 path).
Workflow:
  - Input x is bf16 on both sides.
  - Device path: bf16 -> fp8 (uint8 bytes) -> view(float8_e4m3fn) -> float32.
  - Reference: x.to(float32) — lossless bf16 widening, no fp8 quantization.
  - Compare via PCC (~0.99 threshold) — measures fp8 quantization noise relative
    to the lossless input.
"""

import pytest
import torch
from loguru import logger

import ttnn


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    """Pearson correlation coefficient over flattened float32 views."""
    af = a.flatten().to(torch.float32)
    bf = b.flatten().to(torch.float32)
    af = af - af.mean()
    bf = bf - bf.mean()
    denom = torch.sqrt((af * af).sum() * (bf * bf).sum())
    if denom == 0:
        return 1.0 if torch.equal(a, b) else 0.0
    return (af * bf).sum().item() / denom.item()


@pytest.mark.parametrize(
    "rows, cols",
    [
        (32, 32),  # 1 tile
        (32, 256),  # 1 row band, 8 tiles wide
        (64, 128),  # 2x4 tiles
        (128, 7168),  # representative deepseek hidden_dim slice
    ],
    ids=["1tile", "2tile", "3tile", "4tile"],
)
@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param(
            (8, 1),
            {},
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 1), topology="linear"),
            id="8x1",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_bf16_to_fp8(mesh_device, rows, cols):
    torch.manual_seed(42)

    num_devices = mesh_device.get_num_devices()

    # Mix of "easy" values (representable exactly in fp8_e4m3) and random values, so
    # we exercise both the bit-exact path and the rounded path.
    x = torch.randn((rows, cols), dtype=torch.bfloat16) * 2.0

    logger.info(f"input shape={tuple(x.shape)} dtype={x.dtype}")

    tt_in = ttnn.from_torch(
        x,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    tt_out = ttnn.experimental.deepseek_prefill.bf16_to_fp8(tt_in)

    # Replicated input means each device produces the same output; concat stacks all
    # `num_devices` copies along dim 0 — reshape to (num_devices, rows, cols) and verify
    # each replica independently.
    out_all = ttnn.to_torch(tt_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    out_all = out_all.reshape(num_devices, rows, cols)
    assert out_all.dtype == torch.uint8, f"expected uint8 output, got {out_all.dtype}"

    ref_fp32 = x.to(torch.float32)

    for dev_idx in range(num_devices):
        # Decode device fp8 bytes -> float32 for numeric-domain comparison against the
        # original bf16 input (also widened to float32). PCC here measures fp8 quantization
        # noise relative to the lossless input, not packer-vs-torch rounding agreement.
        out_fp32 = out_all[dev_idx].view(torch.float8_e4m3fn).to(torch.float32)

        pcc = _pcc(out_fp32, ref_fp32)
        max_abs_err = (out_fp32 - ref_fp32).abs().max().item()

        logger.info(f"device={dev_idx} PCC={pcc:.6f}  max_abs_err={max_abs_err:.6f}")

        # ~0.99 reflects fp8_e4m3's ~3-bit mantissa precision against float32 reference.
        assert pcc > 0.99, f"device {dev_idx} PCC too low: {pcc}"
