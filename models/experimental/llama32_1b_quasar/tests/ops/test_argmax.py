# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Per-op test: ``ttnn.argmax``.

Model call site (modules/sampling/sampling_1d.py):
  * L284  _sample_argmax — ttnn.argmax(x_untilized, dim=-1, output_tensor=..., keepdim=False)
          where ``x_untilized = ttnn.untilize(logits, use_multicore=True)`` (L283).

The force-argmax decode path picks the max-logit token id per row. Input is the
untilized (ROW_MAJOR) logits tensor [1, 1, batch, VOCAB]. Reference is
torch.argmax(dim=-1); we assert an exact integer index match (mirrors the
known-good module test test_sampling_1d.py:test_force_argmax).

Note on width: exact argmax over the full VOCAB (128256) in bfloat16 is
tie-dominated, so this op-level test uses a moderate width for a deterministic
reference (the sampler module test uses vocab_size=1024 for the same reason).
The full-VOCAB argmax path is exercised end-to-end in the sampling module tests.
"""

import pytest
import torch

import ttnn
from models.experimental.llama32_1b_quasar.tests.ops import op_utils as U


@U.with_default_mesh()
@pytest.mark.parametrize("width", [pytest.param(w, id=f"w{w}") for w in (1024, 2048)])
@pytest.mark.parametrize("batch", [pytest.param(b, id=f"b{b}") for b in U.DECODE_BATCHES])
def test_argmax(ttnn_mesh_device, reset_seeds, batch, width):
    mesh = ttnn_mesh_device
    shape = (1, 1, batch, width)

    x_torch = U.torch_rand(shape)
    x = U.to_tt(x_torch, mesh)

    # Mirror the model: untilize before argmax (sampling_1d.py:283-289).
    x_untilized = ttnn.untilize(x, use_multicore=True)
    out = ttnn.argmax(x_untilized, dim=-1, keepdim=False)

    # ttnn.argmax -> uint32, torch.argmax -> int64; normalize before compare.
    got = U.from_tt(out, mesh).flatten()[:batch].long()
    ref = x_torch.float().argmax(dim=-1).flatten()[:batch].long()

    assert torch.equal(got, ref), f"argmax mismatch:\n  got: {got[:8]}\n  ref: {ref[:8]}"
