# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Regression markers for the ttnn::typecast(uint32 → bf16) bug that bit
moe_group_op's d(scores) backward.

Direct uint32 → bf16 typecast in this checkout produces 2^31 for input
value 1 (bf16 bit pattern 0x4F00) instead of 1.0. The fix in
moe_group_op.cpp goes through float32 first.

If `test_typecast_uint32_to_bf16_directly_is_broken` ever flips to passing,
ttnn has fixed the bug — drop the float32 detour in
`ttml::ops::moe_group_op` and update the reference memory.
"""

from __future__ import annotations

import pytest
import torch

try:
    import ttnn
    import ttml

    _AVAILABLE = True
except Exception:
    _AVAILABLE = False


def _src_uint32():
    device = ttml.autograd.AutoContext.get_instance().get_device()
    return ttnn.from_torch(
        torch.tensor([0, 1], dtype=torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )


@pytest.mark.skipif(not _AVAILABLE, reason="ttml/ttnn not importable")
@pytest.mark.requires_device
def test_typecast_uint32_to_bf16_directly_is_broken():
    """Regression marker: direct uint32→bf16 typecast does NOT recover [0, 1].
    Currently observed to emit [0.0, 2^31]."""
    direct = ttnn.typecast(_src_uint32(), ttnn.bfloat16)
    vals = ttnn.to_torch(direct).float().tolist()
    assert vals != [0.0, 1.0], (
        f"ttnn::typecast(uint32→bf16) returned {vals} — the bug appears to be "
        "fixed. Drop the fp32 detour in moe_group_op.cpp and update "
        "memory/reference_ttnn_typecast_uint32_bf16_broken.md."
    )


@pytest.mark.skipif(not _AVAILABLE, reason="ttml/ttnn not importable")
@pytest.mark.requires_device
def test_typecast_uint32_via_fp32_to_bf16_is_correct():
    """Workaround: uint32 → float32 → bf16 round-trip recovers [0, 1]."""
    via_fp32 = ttnn.typecast(ttnn.typecast(_src_uint32(), ttnn.float32), ttnn.bfloat16)
    vals = ttnn.to_torch(via_fp32).float().tolist()
    assert vals == [0.0, 1.0], f"expected [0.0, 1.0], got {vals}"
