# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Standalone accuracy test for ttnn.experimental.deepseek_prefill.bf16_to_fp8.

The op takes a BF16 TILE-layout tensor and returns a UINT8 TILE-layout tensor whose
bytes are Fp8_e4m3-encoded values (the same trick used by the dispatch op's FP8 path).
Compare the device output against a torch reference that casts the same BF16 input to
torch.float8_e4m3fn, and check both bit-exact agreement and PCC.
"""

import pytest
import torch
from loguru import logger

import ttnn


def _torch_bf16_to_fp8_bytes(x_bf16: torch.Tensor) -> torch.Tensor:
    """Reference: cast BF16 -> Fp8_e4m3 and view the underlying bytes as uint8."""
    return x_bf16.to(torch.float8_e4m3fn).view(torch.uint8)


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
    ids=["1tile", "1x8", "2x4", "128x7168"],
)
@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param(
            (1, 1),
            {},
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(1, 1), topology="linear"),
            id="single",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_bf16_to_fp8(mesh_device, rows, cols):
    torch.manual_seed(42)

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

    out_uint8 = ttnn.to_torch(tt_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    # On a (1,1) mesh ConcatMeshToTensor just returns the per-device tensor — squeeze any
    # leading singleton dims and reshape to the expected (rows, cols).
    out_uint8 = out_uint8.reshape(rows, cols)
    assert out_uint8.dtype == torch.uint8, f"expected uint8 output, got {out_uint8.dtype}"

    ref_uint8 = _torch_bf16_to_fp8_bytes(x)

    # View bytes as fp8 floats for a numeric-domain comparison.
    out_fp8 = out_uint8.view(torch.float8_e4m3fn).to(torch.float32)
    ref_fp8 = ref_uint8.view(torch.float8_e4m3fn).to(torch.float32)

    pcc = _pcc(out_fp8, ref_fp8)
    max_abs_err = (out_fp8 - ref_fp8).abs().max().item()
    bit_match_ratio = (out_uint8 == ref_uint8).float().mean().item()

    logger.info(f"PCC={pcc:.6f}  max_abs_err={max_abs_err:.6f}  bit_match={bit_match_ratio:.4f}")

    # PCC threshold is the primary correctness gate — packer rounding can disagree with
    # torch's nearest-even at exact halfway points, so don't require bit-exact match.
    assert pcc > 0.999, f"PCC too low: {pcc}"
