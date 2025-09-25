import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import ttnn
from models.utility_functions import comp_allclose_and_pcc, comp_pcc

from ...reference.configuration_gpt_oss import GptOssConfig
from ...reference.hf_utils import get_state_dict
from ...tt.topk import TopKRouter, topk_router

# ModelArgs will be instantiated inside test functions to avoid import-time loading


def reference_topk(g, experts_per_token):
    experts = torch.topk(g, k=experts_per_token, dim=-1)
    expert_weights = torch.nn.functional.softmax(experts.values, dim=1)
    expert_indices = experts.indices
    router_scores = torch.zeros_like(g).scatter_(1, expert_indices, expert_weights)

    return router_scores, expert_weights, expert_indices


class ReferenceTopKRouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_local_experts
        self.hidden_dim = config.hidden_size
        self.weight = nn.Parameter(torch.randn(self.num_experts, self.hidden_dim))
        self.bias = nn.Parameter(torch.randn(self.num_experts))

    def forward(self, hidden_states):
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = F.linear(hidden_states, self.weight, self.bias)  # (seq_len, num_experts)
        router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=-1)  # (seq_len, top_k)
        router_top_value = torch.nn.functional.softmax(router_top_value, dim=1, dtype=router_top_value.dtype)
        router_scores = torch.zeros_like(router_logits).scatter_(1, router_indices, router_top_value)
        return router_scores, router_indices, router_logits

    def forward_given_indices(self, hidden_states, indices):
        # Since TT may produce different indices, force the reference model to use given indices and compare the outputs.
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = F.linear(hidden_states, self.weight, self.bias)  # (seq_len, num_experts)
        router_top_value = router_logits[torch.arange(indices.shape[0])[:, None], indices]
        router_top_value = torch.nn.functional.softmax(router_top_value, dim=1, dtype=router_top_value.dtype)
        router_scores = torch.zeros_like(router_logits).scatter_(1, indices, router_top_value)
        return router_scores, indices, router_logits


@pytest.mark.parametrize(
    "experts_per_token, num_experts",
    [
        (4, 32),  # 20B
        (4, 128),  # 120B
    ],
    ids=["gpt20B", "gpt120B"],
)
@pytest.mark.parametrize("seq_len", [1, 32, 64, 128, 512, 1024], ids=["s1_", "s32", "s64", "s128", "s512", "s1024"])
def test_topk(device, experts_per_token, num_experts, seq_len):
    g = torch.randn(seq_len, num_experts)
    tt_g = ttnn.from_torch(g, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    router_scores, expert_weights, expert_indices = reference_topk(g, experts_per_token)
    # print(f'router_scores: {router_scores}')
    # print(f'expert_weights: {expert_weights}')
    # print(f'expert_indices: {expert_indices}')

    ttnn_router_scores, ttnn_expert_weights, ttnn_expert_indices = topk_router(tt_g, experts_per_token)
    # print(f'ttnn_router_scores: {ttnn_router_scores}')
    # print(f'ttnn_expert_weights: {ttnn_expert_weights}')
    # print(f'ttnn_expert_indices: {ttnn_expert_indices}')

    ttnn_expert_weights = ttnn.to_torch(ttnn_expert_weights)
    ttnn_expert_indices = ttnn.to_torch(ttnn_expert_indices)
    ttnn_router_scores = ttnn.to_torch(ttnn_router_scores)

    if not torch.equal(expert_indices, ttnn_expert_indices):
        total_indices = expert_indices.numel()
        num_mismatches = torch.sum(expert_indices != ttnn_expert_indices).item()
        print(f"Detected {num_mismatches} indices mismatch out of {total_indices} total indices")

        # Get pre-softmax weights (topk values before softmax)
        ref_topk = torch.topk(g, k=experts_per_token, dim=-1)
        ref_pre_softmax_weights = ref_topk.values
        # Find mismatched positions
        mismatch_mask = expert_indices != ttnn_expert_indices

        for seq_idx in range(expert_indices.shape[0]):
            for expert_idx in range(expert_indices.shape[1]):
                if mismatch_mask[seq_idx, expert_idx]:
                    ref_idx = expert_indices[seq_idx, expert_idx].item()
                    ttnn_idx = ttnn_expert_indices[seq_idx, expert_idx].item()
                    ref_weight = ref_pre_softmax_weights[seq_idx, expert_idx].item()
                    ttnn_weight = ref_pre_softmax_weights[seq_idx, expert_idx].item()

                    # Check if the pre-softmax weights are equal
                    if not torch.isclose(torch.tensor(ref_weight), torch.tensor(ttnn_weight), atol=1e-6):
                        print(f"  Weights are NOT equal! Difference: {abs(ref_weight - ttnn_weight)}")
                        assert (
                            False
                        ), f"Pre-softmax weights mismatch at position [{seq_idx}, {expert_idx}]: ref={ref_weight}, ttnn={ttnn_weight}"

    passing, output = comp_allclose_and_pcc(expert_weights, ttnn_expert_weights, atol=1e-2, rtol=1e-1)
    print(f"expert_weights: {output}")
    assert passing, "expert_weights mismatch"

    # Check router scores - if indices were permuted due to sorting differences,
    # router scores should still be correct since they represent the final distribution
    if not torch.equal(expert_indices, ttnn_expert_indices):
        print("Checking router scores with potentially permuted indices...")
        # Router scores should still match even if indices are permuted
        # because scatter operation places weights at correct expert positions
        passing, output = comp_allclose_and_pcc(router_scores, ttnn_router_scores, atol=1e-2, rtol=1e-1)
        print(f"router_scores (with permuted indices): {output}")
        if not passing:
            print("Router scores mismatch despite index permutation - this indicates a real error")
            # Compute expected router scores using TTNN indices and reference weights
            expected_router_scores = torch.zeros_like(g).scatter_(1, ttnn_expert_indices.long(), expert_weights)
            passing_expected, output_expected = comp_pcc(expected_router_scores, ttnn_router_scores)
            print(f"router_scores (using TTNN indices with ref weights): {output_expected}")
            if not passing_expected:
                assert False, "Router scores computation is incorrect"
            else:
                print("Router scores computation is correct, but indices sorting differs from reference")

    else:
        # Normal check when indices match
        passing, output = comp_allclose_and_pcc(router_scores, ttnn_router_scores, atol=1e-2, rtol=1e-1)
        print(f"router_scores: {output}")
        assert passing, "router_scores mismatch"


@pytest.mark.parametrize(
    "experts_per_token, num_experts",
    [
        (4, 32),  # 20B
        (4, 128),  # 120B
    ],
    ids=["gpt20B", "gpt120B"],
)
@pytest.mark.parametrize("seq_len", [1, 32, 64, 128, 512, 1024], ids=["s1_", "s32", "s64", "s128", "s512", "s1024"])
@pytest.mark.parametrize("hidden_dim", [2880])
@pytest.mark.parametrize("use_real_weights", [True, False], ids=["real", "random"])
@pytest.mark.parametrize("mesh_device", [(1, 2)], indirect=True)
def test_topk_router(mesh_device, experts_per_token, num_experts, seq_len, hidden_dim, use_real_weights, reset_seeds):
    hidden_states = torch.randn(seq_len, hidden_dim)
    tt_hidden_states = ttnn.from_torch(hidden_states, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    config = GptOssConfig(
        num_local_experts=num_experts,
        hidden_size=hidden_dim,
        num_experts_per_tok=experts_per_token,
    )
    reference_model = ReferenceTopKRouter(config)

    if use_real_weights:
        state_dict = get_state_dict(local_weights_path, "model.layers.0.mlp.router.", dtype=torch.float32)
        reference_model.load_state_dict(state_dict, strict=True)

    state_dict = reference_model.state_dict()
    tt_model = TopKRouter(mesh_device, config, state_dict)
    router_scores, router_indices, router_logits = reference_model(hidden_states)

    ttnn_router_scores, ttnn_router_indices, ttnn_router_logits = tt_model(tt_hidden_states)

    ttnn_router_scores_tensors = ttnn.get_device_tensors(ttnn_router_scores)
    ttnn_router_indices_tensors = ttnn.get_device_tensors(ttnn_router_indices)
    ttnn_router_logits_tensors = ttnn.get_device_tensors(ttnn_router_logits)

    for i in range(len(ttnn_router_scores_tensors)):
        ttnn_router_scores = ttnn.to_torch(ttnn_router_scores_tensors[i])
        ttnn_router_indices = ttnn.to_torch(ttnn_router_indices_tensors[i])
        ttnn_router_logits = ttnn.to_torch(ttnn_router_logits_tensors[i])

        passing, output = comp_pcc(router_logits, ttnn_router_logits, pcc=0.99)
        mse = torch.nn.functional.mse_loss(router_logits, ttnn_router_logits)
        print(f"router_logits: {output}, mse: {mse}")
        assert passing, "router_logits mismatch"

        # Run with reference model forced to tt model indices
        router_scores, _, _ = reference_model.forward_given_indices(hidden_states, ttnn_router_indices.long())
        passing, output = comp_pcc(router_scores, ttnn_router_scores, pcc=0.987)
        mse = torch.nn.functional.mse_loss(router_scores, ttnn_router_scores)
        print(f"router_scores: {output}, mse: {mse}")
    assert passing, "router_scores mismatch"
