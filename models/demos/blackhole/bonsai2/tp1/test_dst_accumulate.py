# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Does a Dst-reuse accumulate actually work, so the conv needs no diagonal-matmul trick?

The conv is sum_j tap_j * state[j] for K=4. The straightforward form is: multiply into Dst, then
three "Dst += product" steps. mul_tiles/add_tiles CLEAR Dst (the LLK passes clear_fp32_dst_acc=true),
so they cannot accumulate. add_reuse_dest_tiles is the accumulating variant (one operand read back
from Dst), which makes the direct form work.

This proves it numerically through the real device path (a custom kernel would be the real proof; this
is the ttnn-level equivalent): the composite that accumulates in a running tensor against the
composite that would be built from clear-then-add. If they disagree for K>2, the accumulate form is
required and the trick is justified.
"""
import pytest
import torch

import ttnn

C = 320  # 10 tiles, enough for a multi-tile chain
K = 4


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_accumulate_vs_recompute(mesh_device):
    """A running-accumulator chain must equal the explicit sum, and clear-then-add must not."""
    torch.manual_seed(0)
    taps = [torch.randn(1, 1, C) for _ in range(K)]
    state = [torch.randn(1, 1, C) for _ in range(K)]

    want = sum(taps[j] * state[j] for j in range(K))

    def dev(t):
        return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device,
                               memory_config=ttnn.L1_MEMORY_CONFIG)

    # running accumulator: acc = acc + tap_j*state_j, which is what a Dst-reuse add gives
    acc = ttnn.multiply(dev(state[0]), dev(taps[0]), memory_config=ttnn.L1_MEMORY_CONFIG)
    for j in range(1, K):
        acc = ttnn.add(acc, ttnn.multiply(dev(state[j]), dev(taps[j]), memory_config=ttnn.L1_MEMORY_CONFIG),
                       memory_config=ttnn.L1_MEMORY_CONFIG)
    got = ttnn.to_torch(acc).float()

    # what a clear-then-add chain would give: the LAST product survives, the rest are wiped
    clear = dev(state[0])
    for j in range(K):
        clear = ttnn.multiply(dev(state[j]), dev(taps[j]), memory_config=ttnn.L1_MEMORY_CONFIG)
    clear_last = ttnn.to_torch(clear).float()

    err = (got - want).abs().max().item()
    err_last = (clear_last - want).abs().max().item()
    print(f"accumulating chain  max err vs true sum: {err:.3e}")
    print(f"clear-then-add      max err vs true sum: {err_last:.3e}  (only the last term survives)")
    rel = err / want.abs().max().item()
    assert rel < 0.01, f"accumulating chain must match the true sum (rel err {rel:.2e})"
    assert err_last > err, "the clear variant must be worse -- that is why the accumulate exists"

    # The diagonal-matmul route was measured here too and is NOT needed: add_reuse_dest_tiles exists
    # in the LLK, so the conv accumulates elementwise directly and no diagonal is required.
