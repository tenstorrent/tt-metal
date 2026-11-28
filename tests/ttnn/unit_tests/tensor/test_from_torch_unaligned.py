# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""``ttnn.from_torch`` accepts a torch tensor whose data is not aligned for its element type.

Tensors loaded from a safetensors file can view the mmap at an odd byte offset (SDXL inpainting UNet, DeepSeek-R1
BF16 norms). ``from_torch`` copies such data instead of borrowing it.
"""

import pytest
import torch

import ttnn


def _unaligned(values: torch.Tensor) -> torch.Tensor:
    """A copy of ``values`` whose data starts one byte past an aligned address."""
    nbytes = values.numel() * values.element_size()
    buffer = bytearray(nbytes + 1)
    out = torch.frombuffer(buffer, dtype=values.dtype, offset=1, count=values.numel()).reshape(values.shape)
    out.copy_(values)
    assert out.data_ptr() % values.element_size() != 0, "the test tensor must be unaligned"
    return out


@pytest.mark.parametrize(
    "torch_dtype, ttnn_dtype",
    [(torch.float32, ttnn.float32), (torch.bfloat16, ttnn.bfloat16), (torch.int32, ttnn.int32)],
)
def test_from_torch_unaligned_round_trips(torch_dtype, ttnn_dtype):
    expected = torch.arange(2 * 64, dtype=torch.float32).reshape(2, 64).to(torch_dtype)
    source = _unaligned(expected)

    tt_tensor = ttnn.from_torch(source, dtype=ttnn_dtype, layout=ttnn.ROW_MAJOR_LAYOUT)

    assert torch.equal(ttnn.to_torch(tt_tensor), expected)
