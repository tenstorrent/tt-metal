# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Demonstrates what ttnn.pad does to a ROW-MAJOR host tensor's alignment.

Key point (see conv2d_alignment_explained.html):
  - pad keeps the LOGICAL shape unchanged (still width 125)
  - pad grows the PHYSICAL/padded shape (width -> 128) and writes real zeros
  - for row-major, that physical growth is recorded as alignment[-1] = padded width
"""

import torch
import ttnn


def _dump(name, t):
    print(f"\n{name}")
    print("  on device :", ttnn.is_tensor_storage_on_device(t))
    print("  layout    :", t.layout)
    print("  logical   :", list(t.shape))
    print("  padded    :", list(t.padded_shape))
    print("  alignment :", list(t.spec.alignment))


def test_host_pad_alignment():
    # 1) ROW-MAJOR bfloat16 tensor on HOST (no device= -> stays on host), inner width = 125.
    torch_x = torch.randn(1, 1, 2, 125, dtype=torch.bfloat16)
    t = ttnn.from_torch(torch_x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

    assert not ttnn.is_tensor_storage_on_device(t), "expected a HOST tensor"
    _dump("BEFORE pad  (width 125, unpadded):", t)

    before_alignment = list(t.spec.alignment)

    # 2) Pad the inner width 125 -> 128, still on HOST (no to_device between).
    t_padded = ttnn.pad(t, padding=[(0, 0), (0, 0), (0, 0), (0, 3)], value=0.0)

    assert not ttnn.is_tensor_storage_on_device(t_padded), "pad should have stayed on HOST"
    _dump("AFTER  pad  (width 125 -> 128):", t_padded)

    after_alignment = list(t_padded.spec.alignment)

    # --- what the demo proves ---------------------------------------------------
    # logical width is UNCHANGED by pad (still 125), physical/padded width grew to 128
    assert list(t.shape)[-1] == 125
    assert list(t_padded.shape)[-1] == 125, "pad must NOT change the logical shape"
    assert list(t_padded.padded_shape)[-1] == 128, "pad grows the physical/padded width to 128"

    # the innermost alignment jumps from 1 (no padding) to 128 (the full padded row width)
    assert before_alignment[-1] == 1, f"unpadded RM width-alignment should be 1, got {before_alignment}"
    assert after_alignment[-1] == 128, f"padded RM width-alignment should be 128, got {after_alignment}"

    print("\nSUMMARY: width alignment", before_alignment[-1], "->", after_alignment[-1])
    print("         (logical width stayed 125; only physical width + alignment changed)\n")
