# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Per-op test: ``ttnn.topk``.

Model call sites (modules/sampling/sampling_1d.py):
  * L530  _topk_single_device — per half-vocab shard, k=cfg.max_top_k, dim=-1,
          indices_tensor=<local indices>
  * L568  _topk_multi_device  — per-device shard (padded to a power of 2),
          k=cfg.max_top_k, dim=-1, indices_tensor=<local indices>

``cfg.max_top_k`` defaults to 32 (Sampling1DConfig). The collected demo cases
(tests/modules/sampling/test_sampling_1d.py:_list_collected_sampling_cases) use
k ∈ {1, 10} at the sampler level; the topk op itself is always driven at
max_top_k. We exercise k ∈ {1, 10, 32}.

The single-device path splits the full vocab (VOCAB=128256) in half and runs
topk over each ~64k shard. Full-vocab topk over bfloat16 is dominated by ties,
so this op-level test uses tile-aligned power-of-2 widths for a stable reference
(the padding-to-power-of-2 mirrors _topk_multi_device:559). ``indices_tensor`` is
omitted here — the op returns fresh 0..W-1 indices, which is the reference space.

Reference: ``torch.topk``. We assert the selected *values* match by PCC and that
the selected *index sets* overlap (bfloat16 ties can reorder equal values).
"""

import pytest
import torch

import ttnn
from models.experimental.llama32_1b_quasar.tests.ops import op_utils as U

# max_top_k default (Sampling1DConfig); k=1/10 are the demo-collected sampler ks.
_MAX_TOP_K = 32


@U.with_default_mesh()
@pytest.mark.parametrize("width", [pytest.param(w, id=f"w{w}") for w in (1024, 2048)])
@pytest.mark.parametrize("k", [pytest.param(k, id=f"k{k}") for k in (1, 10, _MAX_TOP_K)])
@pytest.mark.parametrize("batch", [pytest.param(b, id=f"b{b}") for b in U.DECODE_BATCHES])
def test_topk(ttnn_mesh_device, reset_seeds, batch, k, width):
    mesh = ttnn_mesh_device
    shape = (1, 1, batch, width)

    x_torch = U.torch_rand(shape)
    x = U.to_tt(x_torch, mesh)

    # Match the model's call: pass a persistent uint16 local-index tensor (TILE layout),
    # exercising the indices_tensor overload used at sampling_1d.py:530-536 / 568-574.
    # With arange indices the returned idxs are positional, so the torch.topk reference holds.
    idx_torch = torch.arange(width, dtype=torch.int32).view(1, 1, 1, width).expand(1, 1, batch, width).contiguous()
    indices_tensor = U.to_tt(idx_torch, mesh, dtype=ttnn.uint16)

    values, indices = ttnn.topk(x, k=k, dim=-1, indices_tensor=indices_tensor)

    # torch reference on the same bfloat16 values the device saw.
    ref_vals, ref_idxs = torch.topk(x_torch.float(), k=k, dim=-1)  # [1,1,batch,k]

    # Read back and slice per-row to the requested k (device output may be tile-padded).
    dev_vals = U.from_tt(values, mesh).reshape(1, 1, batch, -1)[..., :k]
    dev_idxs = U.from_tt(indices, mesh).reshape(1, 1, batch, -1)[..., :k].long()

    for b in range(batch):
        # Values: the k selected values (sorted desc) should match the reference per row.
        got_v = torch.sort(dev_vals[0, 0, b].float(), descending=True).values
        exp_v = torch.sort(ref_vals[0, 0, b].float(), descending=True).values
        assert torch.allclose(got_v, exp_v, atol=0.05, rtol=0.05), f"batch {b}: top-{k} values {got_v} vs {exp_v}"

        # Indices: set overlap per row (allow a small slack for bfloat16 tie reordering).
        got_i = set(dev_idxs[0, 0, b].tolist())
        exp_i = set(ref_idxs[0, 0, b].tolist())
        overlap = len(got_i & exp_i)
        assert overlap >= max(1, k - 2), f"batch {b}: top-{k} index overlap {overlap} (got {got_i} vs {exp_i})"
