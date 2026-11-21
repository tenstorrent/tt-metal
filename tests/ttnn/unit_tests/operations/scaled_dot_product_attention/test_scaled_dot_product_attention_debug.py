# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic debug tests of the streamed regime (two key chunks, online rescale). DO NOT DELETE.

Found: row_sum_block copied alpha with copy_tile(rescale, ...) after copy_init(row_sum). On Quasar the unpack
buffer descriptor is baked at the op init, so the copy read l_old again: l = l_old * l_old + rowsum. With
Q = 0, V = 1 and two chunks of 32 keys: l = 32 * 32 + 32 = 1056 instead of 64, output 64 / 1056 = 0.0605
instead of 1.

Hand-calculated expectations (S = 64, D = 32, scale 1, not causal, k_chunk_tiles = 1):
  A  Q = 0, V = 1               -> every score 0, m = 0, alpha = 1, l = 64, O = 64, out = 1
  B  Q = 0, V[j, :] = j         -> uniform weights, out = mean(j) = 31.5
  C  Q = 1, K[j] = 1/32 for j >= 32 (score 1 in chunk 1: the max grows, alpha = e^-1 on chunk 1)
  D  Q = 1, K[j] = 1/32 for j < 32 (score 1 in chunk 0: m stays 1, alpha = 1, P1 = e^-1)
"""

import pytest
import torch

import ttnn
from eval.specs.scaled_dot_product_attention.spec import reference
from ttnn.operations.scaled_dot_product_attention import run_attention

S, D = 64, 32


def _cases():
    ones = torch.ones(1, 1, S, D)
    zeros = torch.zeros(1, 1, S, D)
    v_rows = torch.arange(S, dtype=torch.float32).reshape(1, 1, S, 1).repeat(1, 1, 1, D)
    k_late = torch.zeros(1, 1, S, D)
    k_late[..., 32:, :] = 1.0 / 32
    k_early = torch.zeros(1, 1, S, D)
    k_early[..., :32, :] = 1.0 / 32
    return {
        "A_uniform_ones": (zeros, zeros, ones),
        "B_uniform_rows": (zeros, zeros, v_rows),
        "C_max_grows": (ones, k_late, v_rows),
        "D_max_first": (ones, k_early, v_rows),
    }


@pytest.mark.parametrize("threads", [(1, 1, 1), (4, 4, 2)], ids=lambda p: "R%dC%dW%d" % p)
@pytest.mark.parametrize("case", list(_cases()))
def test_two_chunks_deterministic(device, case, threads):
    q, k, v = (t.bfloat16() for t in _cases()[case])
    put = lambda t: ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)  # noqa: E731
    out = ttnn.to_torch(run_attention(put(q), put(k), put(v), is_causal=False, scale=1.0, k_chunk_tiles=1,
                                      threads=threads)).double()  # fmt: skip
    ref = reference(q, k, v, is_causal=False, scale=1.0)
    # bfloat16 output rounding of values up to ~40 is 0.125
    assert torch.allclose(out, ref, atol=0.15, rtol=0.01), (out - ref).abs().max()
