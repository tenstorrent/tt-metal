# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Expert-sorted prefill MoE under SKEWED routing: exercises the hot-expert branch of
models/demos/gpt_oss/tt/experts/prefill.py (_sorted_moe_plan / _sorted_moe_forward) and compares against the
transformers reference. The uniform random routing of the experts unit test never produces hot experts, so the
hot/cold partition and its cost-model cap selection are only covered here."""


import pytest
import torch
from loguru import logger

import ttnn
from models.demos.gpt_oss.tests.test_factory import TestFactory, parametrize_mesh_with_fabric
from models.demos.gpt_oss.tests.unit import test_modules as tm
from models.demos.gpt_oss.tt.experts import prefill as experts_prefill

# One expert takes ~60% of the tokens, two more ~25% / ~12%: the hottest three exceed any cap the plan can pick
# (<= 256 rows per expert per 1024-token split), so they must go through the dense hot-expert group.
HOT_EXPERTS = {3: 0.60, 7: 0.25, 11: 0.12}


def _skewed_routing(seq_len, num_experts, top_k, seed=1):
    g = torch.Generator().manual_seed(seed)
    router_indices = torch.zeros(seq_len, top_k, dtype=torch.long)
    dense = torch.zeros(seq_len, num_experts)
    by_position = torch.zeros(seq_len, top_k)
    for t in range(seq_len):
        forced = [e for e, p in HOT_EXPERTS.items() if torch.rand(1, generator=g).item() < p]
        others = [e for e in torch.randperm(num_experts, generator=g).tolist() if e not in forced]
        idx = torch.tensor((forced + others)[:top_k])
        w = torch.rand(top_k, generator=g)
        w = w / w.sum()
        router_indices[t] = idx
        dense[t, idx] = w
        by_position[t] = w
    return router_indices, dense, by_position


@pytest.mark.timeout(1200)
@pytest.mark.parametrize("seq_len", [1024, 4096], ids=["prefill_1024", "prefill_4096"])
@parametrize_mesh_with_fabric([(1, 8)])
def test_experts_prefill_skewed_routing(mesh_device, device_params, seq_len):
    setup = TestFactory.setup_test(mesh_device, use_real_weights=False)
    config = setup["config"]
    if config.num_local_experts < experts_prefill._SORTED_MOE_MIN_EXPERTS:
        pytest.skip("the expert-sorted prefill path is only used for models with >= 64 experts (gpt-oss-120b)")
    reference_layer = tm.setup_reference_layer(setup, layer_idx=0)
    decoder_layer = tm.setup_decoder_layer(setup, reference_layer, 1, seq_len, layer_idx=0)

    router_indices, routing_dense, routing_by_position = _skewed_routing(
        seq_len, config.num_local_experts, config.num_experts_per_tok
    )
    counts = (routing_dense > 0).sum(0)
    logger.info(f"routed tokens per expert: top-4 {counts.topk(4).values.tolist()}, mean {counts.float().mean():.1f}")
    hidden = torch.randn(1, seq_len, config.hidden_size)
    with torch.no_grad():
        reference = reference_layer.mlp.experts.eval()(
            hidden.reshape(-1, config.hidden_size), router_indices=router_indices, routing_weights=routing_by_position
        )

    mapper = ttnn.ShardTensor2dMesh(dims=(None, None), mesh_shape=mesh_device.shape, mesh_device=mesh_device)
    tt_hidden = ttnn.from_torch(
        hidden.reshape(1, 1, seq_len, -1),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=mapper,
    )
    tt_routing = ttnn.from_torch(
        routing_dense, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=mapper
    )
    experts_prefill.LAST_SORTED_MOE_PLAN.clear()
    tt_out = decoder_layer.mlp.experts(hidden_states=tt_hidden, topk_expert_weights=tt_routing, is_decode=False)
    composer = ttnn.ConcatMesh2dToTensor(mesh_device, dims=(-2, -1), mesh_shape=tuple(mesh_device.shape))
    out = (
        ttnn.to_torch(tt_out, mesh_composer=composer)[..., :seq_len, : config.hidden_size].reshape(seq_len, -1).float()
    )

    plan = dict(experts_prefill.LAST_SORTED_MOE_PLAN)
    logger.info(f"sorted-MoE plan of the last split: {plan}")
    assert plan.get("hot", 0) >= 1, f"expected the hot-expert branch to run, plan was {plan}"
    pcc = torch.corrcoef(torch.stack([out.flatten(), reference.flatten().float()]))[0, 1].item()
    logger.info(f"skewed-routing experts prefill S={seq_len}: PCC {pcc:.5f}")
    # Same bound as the experts unit test with uniform routing (bf8 activations/weights): ~0.983 measured.
    assert pcc > 0.97, f"PCC {pcc:.5f} vs the transformers reference"
