# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Unit test for ``pack_latents_device`` (device-side Qwen latent packing).

Validates that the device op reproduces the host torch ``_pack_latents``
(``view -> permute(0,2,4,1,3,5) -> reshape``) within bf16 tolerance on a
replicated 1x1 mesh.
"""

import pytest
import torch

import ttnn

from ....utils import tensor as tensor_utils


def _pack_latents_torch(
    latents: torch.Tensor, batch_size: int, num_channels: int, height: int, width: int
) -> torch.Tensor:
    latents = latents.view(batch_size, num_channels, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    return latents.reshape(batch_size, (height // 2) * (width // 2), num_channels * 4)


@pytest.mark.parametrize(
    "mesh_device",
    [
        pytest.param((1, 1), id="1x1"),
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    ("batch_size", "num_channels", "height", "width", "input_rank"),
    [
        pytest.param(1, 16, 64, 64, 4, id="b1c16_64x64_4d"),
        pytest.param(1, 16, 64, 64, 5, id="b1c16_64x64_5d"),
        pytest.param(2, 16, 128, 128, 4, id="b2c16_128x128_4d"),
        pytest.param(2, 16, 128, 128, 5, id="b2c16_128x128_5d"),
    ],
)
def test_pack_latents_device(
    mesh_device: ttnn.MeshDevice,
    batch_size: int,
    num_channels: int,
    height: int,
    width: int,
    input_rank: int,
) -> None:
    torch.manual_seed(0)

    latents_4d = torch.randn(batch_size, num_channels, height, width, dtype=torch.float32)

    if input_rank == 5:
        latents_input = latents_4d.unsqueeze(2).contiguous()
    else:
        latents_input = latents_4d.contiguous()

    expected = _pack_latents_torch(latents_4d.clone(), batch_size, num_channels, height, width)

    tt_input = tensor_utils.from_torch(latents_input, device=mesh_device, on_host=False)
    tt_packed = tensor_utils.pack_latents_device(tt_input, batch_size, num_channels, height, width)

    got = tensor_utils.to_torch(tt_packed)

    assert list(got.shape) == list(
        expected.shape
    ), f"shape mismatch: got {tuple(got.shape)} expected {tuple(expected.shape)}"

    atol = 2e-2
    rtol = 2e-2
    diff = (got.to(torch.float32) - expected.to(torch.float32)).abs()
    max_abs = diff.max().item()
    assert torch.allclose(
        got.to(torch.float32), expected.to(torch.float32), atol=atol, rtol=rtol
    ), f"pack_latents_device mismatch: max_abs={max_abs}"
