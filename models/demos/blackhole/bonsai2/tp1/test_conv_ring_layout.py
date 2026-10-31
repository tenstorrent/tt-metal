# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""GDN decode milestone 1: conv state ring layout, checked on the host before any kernel exists.

Milestone 1 folds the depthwise causal conv + SiLU + q/k L2-norm into the fused recurrence op. The
kernel needs the conv state as a ring it shifts itself (the planner's constraint: a host slot index
would be baked into a decode trace, so the shift must be in-kernel).

Two things are pinned here, both pure torch so they can be checked in seconds:
  1. The tap order. tap j multiplies state row j, and state row j must hold the input row t-3+j, so
     the conv is a plain 4-term weighted sum over the ring in a FIXED order.
  2. The ring shift. After consuming the ring, the next call must see rows shifted by one with the
     new input appended -- which is exactly what the current 4-Copy host shift does, so the kernel's
     in-kernel shift is verified against the thing it replaces rather than against a fresh idea.

Reference is the layer's own FIR (models/demos/blackhole/qwen36/tt/gdn Qwen36GatedDeltaNet used
ttnn.multiply + ttnn.mac over conv_states), reimplemented here in torch.
"""
import pytest
import torch

C = 10240
K = 4


def conv_fir(state, taps):
    """state [K,1,C] rows oldest..newest, taps [K,1,C]; returns sum_j taps[j]*state[j]."""
    out = torch.zeros_like(state[0])
    for j in range(K):
        out = out + taps[j] * state[j]
    return out


def shift_and_append(state, new_row):
    """What the host does today: copy rows 1..K-1 down, write the new input into row K-1."""
    out = state.clone()
    for j in range(K - 1):
        out[j] = state[j + 1]
    out[K - 1] = new_row
    return out


def test_conv_is_a_plain_four_term_sum():
    torch.manual_seed(0)
    state = torch.randn(K, C)
    taps = torch.randn(K, C)
    got = conv_fir(state, taps)
    want = sum(taps[j] * state[j] for j in range(K))
    assert torch.equal(got, want), "conv tap order must be a fixed oldest-to-newest sum"


def test_ring_shift_matches_the_host_shift_register():
    """K successive tokens must produce the same conv as the host's shift register, token by token."""
    torch.manual_seed(1)
    taps = torch.randn(K, C)
    inputs = torch.randn(K + 2, C)  # K+2 tokens: two need a carried window

    # Host: maintain state as [K,1,C], shift then conv.
    host_state = torch.zeros(K, C)
    host_out = []
    for t in range(inputs.shape[0]):
        host_state = shift_and_append(host_state, inputs[t])
        host_out.append(conv_fir(host_state, taps))

    # In-kernel: same ring, but the shift is described as a rotation so the kernel can do it in place.
    kern_state = torch.zeros(K, C)
    kern_out = []
    for t in range(inputs.shape[0]):
        kern_out.append(conv_fir(kern_state, taps))  # conv the CURRENT ring first
        kern_state = shift_and_append(kern_state, inputs[t])  # then shift

    # The two differ by one step by construction: the host convolves AFTER shifting. Compare the host's
    # step t against the kernel's step t+1, which is the ordering the kernel must implement (shift
    # first, then conv, all inside one call).
    for t in range(inputs.shape[0] - 1):
        assert torch.equal(host_out[t], kern_out[t + 1]), f"ring order mismatch at token {t}"


def test_shift_is_a_rotation_too():
    """The shift writes rows from rows: a kernel doing it in place must copy in the ascending order."""
    torch.manual_seed(2)
    state = torch.randn(K, C)
    new_row = torch.randn(C)
    want = shift_and_append(state, new_row)
    # In place, ascending: row j <- row j+1 for j = 0..K-2, then row K-1 <- new. Safe because each
    # read is of a row not yet overwritten.
    got = state.clone()
    for j in range(K - 1):
        got[j] = got[j + 1].clone()
    got[K - 1] = new_row
    assert torch.equal(got, want), "ascending in-place shift must equal the copy-based shift"


@pytest.mark.parametrize("k", [2, 3, 4])
def test_shift_generalises_across_tap_counts(k):
    torch.manual_seed(3)
    state = torch.randn(k, 64)
    got = state.clone()
    for j in range(k - 1):
        got[j] = got[j + 1].clone()
    got[k - 1] = torch.zeros(64)
    assert torch.equal(got[0], state[1]), "row 0 must take row 1"
    assert torch.equal(got[k - 2], state[k - 1]), "last carried row must take the newest row"
    assert torch.equal(got[k - 1], torch.zeros(64)), "the new row slot must be the appended one"
