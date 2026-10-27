# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""TopKRouter fused path (topk_router_gpt), the one decode high-throughput takes on Wormhole.

Single device, synthetic 128-expert config, no HF weights. Guards two things:
- the returned indices / weights are still allocated (the model used to slice them to [B, k] and deallocate the op
  outputs; once the op reported [B, k] itself the slice aliased its input and the deallocate freed the result);
- the outputs are [32, k] in both logical and padded width, so fused_decode's zero-copy reshape to [32, 1, 1, k]
  keeps one page per row (the dispatch op asserts indices pages == hidden-state pages under watcher).
"""
import types

import pytest
import torch

import ttnn
from models.demos.gpt_oss.tt.topk import TopKRouter

from ..test_factory import parametrize_mesh_with_fabric


@parametrize_mesh_with_fabric([(1, 1)])
def test_topk_router_fused_decode(mesh_device, device_params):
    if ttnn.device.is_blackhole(mesh_device):
        pytest.skip("TopKRouter uses the fused topk_router_gpt op on Wormhole only")
    torch.manual_seed(0)
    hidden, experts, k, tokens = 2880, 128, 4, 32
    cfg = types.SimpleNamespace(hidden_size=hidden, num_local_experts=experts, num_experts_per_tok=k)
    weight = torch.randn(experts, hidden) * 0.02
    bias = torch.randn(experts) * 0.1
    router = TopKRouter(mesh_device, cfg, {"weight": weight, "bias": bias})
    assert router.use_fused_op

    x = torch.randn(1, 1, tokens, hidden) * 0.1
    tt_x = ttnn.from_torch(x, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    indices, weights = router(tt_x, use_throughput_experts=True)

    assert indices.is_allocated() and weights.is_allocated(), "router returned deallocated tensors"
    for t in (indices, weights):
        assert tuple(t.shape) == (tokens, k)
        assert tuple(t.padded_shape) == (tokens, k)
        # Same zero-copy reshape fused_decode does; it must keep one page per row.
        view = ttnn.reshape(t, (tokens, 1, 1, k))
        assert view.buffer_num_pages() == tokens

    idx = ttnn.to_torch(ttnn.get_device_tensors(indices)[0]).long().reshape(tokens, k)
    wts = ttnn.to_torch(ttnn.get_device_tensors(weights)[0]).float().reshape(tokens, k)
    logits = x.reshape(tokens, hidden).bfloat16().float() @ weight.bfloat16().float().T + bias.bfloat16().float()
    selected = logits.gather(1, idx)
    torch.testing.assert_close(
        selected.sort(-1, descending=True).values, logits.topk(k, -1).values, atol=0.02, rtol=0.02
    )
    torch.testing.assert_close(wts, selected.softmax(-1), atol=0.01, rtol=0.05)
