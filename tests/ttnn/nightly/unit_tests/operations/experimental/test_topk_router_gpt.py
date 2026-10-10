# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Test for the fused topk_router_gpt operation.

The fused op computes:  matmul + bias → logits → topk(k=4) → softmax
Outputs: (indices_rm, weights_rm) both in ROW_MAJOR format
  - indices_rm: [B, k] uint16 expert indices
  - weights_rm: [B, k] bf16 softmax weights
"""

import pytest
import torch
import torch.nn.functional as F
import ttnn
from loguru import logger

from models.common.utility_functions import comp_pcc

PCC_THRESHOLD = 0.95

# (B, K=hidden_dim, N=num_experts, TOP_K)
# B is within 1..32 and N=128. Outputs retain B logical rows over a full physical tile.
# K must be divisible by 32 (tile size).
TEST_SHAPES = [
    (32, 2880, 128, 4),  # production shape
    (1, 2880, 128, 4),
    (9, 2880, 128, 4),
    (16, 2880, 128, 4),
    (31, 2880, 128, 4),
    (32, 64, 128, 4),  # small hidden_dim edge case
    (32, 4096, 128, 4),  # large hidden_dim
]


@pytest.fixture(autouse=True)
def require_router_topology(device):
    workers = device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)
    if len(workers) < 8:
        pytest.skip("fused router requires at least eight DRAM-aligned workers (P150 or Wormhole)")


def run_fused_op(device, torch_input, torch_weight, torch_bias, B, K, N, k=4):
    """Run the fused op and return the torch result."""
    torch_bias_bcast = torch_bias.expand(B, N).contiguous()
    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    tt_weight = ttnn.from_torch(torch_weight, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    tt_bias = ttnn.from_torch(torch_bias_bcast, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

    indices_rm, weights_rm = ttnn.experimental.topk_router_gpt(
        tt_input,
        weight_tensor=tt_weight,
        bias_tensor=tt_bias,
        k=k,
        num_experts=N,
    )

    indices = ttnn.to_torch(indices_rm)[:B, :k].long()
    weights = ttnn.to_torch(weights_rm)[:B, :k].float()
    return weights, indices


@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {
                "dispatch_core_axis": ttnn.DispatchCoreAxis.ROW,
            },
            id="dispatch_row",
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "B, K, N, TOP_K",
    TEST_SHAPES,
)
def test_topk_router_gpt_deterministic(device, B, K, N, TOP_K):
    """Known logits → verify topk indices are correct.

    Use a weight that produces known logits: all zeros except bias = arange.
    Top-4 should be cols 127,126,125,124.
    """
    torch_input = torch.zeros(B, K, dtype=torch.bfloat16)
    torch_weight = torch.zeros(K, N, dtype=torch.bfloat16)
    torch_bias = torch.arange(N, dtype=torch.float32).unsqueeze(0).to(torch.bfloat16)

    weights_tt, indices_tt = run_fused_op(device, torch_input, torch_weight, torch_bias, B, K, N, TOP_K)

    ref_logits = torch_bias.float().expand(B, N)
    ref_vals, ref_idxs = torch.topk(ref_logits, TOP_K, dim=-1)
    ref_weights = F.softmax(ref_vals, dim=-1)

    logger.info(f"  Expected indices row 0: {ref_idxs[0].tolist()}")
    logger.info(f"  TT       indices row 0: {indices_tt[0].tolist()}")

    idx_match = (indices_tt == ref_idxs).all().item()
    _pcc_passed, pcc_val = comp_pcc(ref_weights, weights_tt)
    logger.info(f"  Indices exact match: {idx_match}")
    logger.info(f"  Weight PCC: {pcc_val}")

    assert idx_match, f"Indices mismatch: expected {ref_idxs[0].tolist()}, got {indices_tt[0].tolist()}"
    assert pcc_val >= 0.99, f"Weight PCC {pcc_val} below threshold 0.99"


@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {
                "dispatch_core_axis": ttnn.DispatchCoreAxis.ROW,
            },
            id="dispatch_row",
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "B, K, N, TOP_K",
    TEST_SHAPES,
)
@pytest.mark.parametrize("seed", [42, 99], ids=["seed_42", "seed_99"])
def test_topk_router_gpt_random_matmul(device, B, K, N, TOP_K, seed):
    """Random matmul + bias → verify topk + softmax accuracy."""
    torch.manual_seed(seed)
    torch_input = (torch.randn(B, K) * 0.1).to(torch.bfloat16)
    torch_weight = (torch.randn(K, N) * 0.01).to(torch.bfloat16)
    torch_bias = (torch.randn(1, N) * 0.1).to(torch.bfloat16)

    weights_tt, indices_tt = run_fused_op(device, torch_input, torch_weight, torch_bias, B, K, N, TOP_K)

    ref_logits = (torch_input.float() @ torch_weight.float() + torch_bias.float()).to(torch.bfloat16).float()
    ref_vals, ref_idxs = torch.topk(ref_logits, TOP_K, dim=-1)
    ref_weights = F.softmax(ref_vals, dim=-1)

    logger.info(f"  Ref indices row 0: {ref_idxs[0].tolist()}")
    logger.info(f"  TT  indices row 0: {indices_tt[0].tolist()}")

    idx_match_count = (indices_tt == ref_idxs).sum().item()
    total_indices = B * TOP_K
    idx_match_pct = idx_match_count / total_indices * 100
    logger.info(f"  Index match: {idx_match_count}/{total_indices} ({idx_match_pct:.1f}%)")

    _pcc_passed, pcc_val = comp_pcc(ref_weights, weights_tt)
    logger.info(f"  Weight PCC: {pcc_val}")

    # Verify softmax properties
    row_sums = weights_tt.sum(dim=-1)
    all_positive = (weights_tt > 0).all().item()
    logger.info(f"  Weight row sums - min: {row_sums.min():.4f}, max: {row_sums.max():.4f}")
    logger.info(f"  All weights positive: {all_positive}")

    # Thresholds are intentionally loose because bf16 matmul accumulation
    # produces slightly different logit values than float32 reference, causing
    # topk tie-breaking to differ for tokens with near-equal expert scores.
    assert idx_match_pct >= 90.0, f"Index match {idx_match_pct:.1f}% below threshold 90%"
    assert pcc_val >= PCC_THRESHOLD, f"Weight PCC {pcc_val} below threshold {PCC_THRESHOLD}"
    assert all_positive, "Some weights are not positive"


@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {
                "dispatch_core_axis": ttnn.DispatchCoreAxis.ROW,
            },
            id="dispatch_row",
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "B, K, N, TOP_K",
    TEST_SHAPES,
)
def test_topk_router_gpt_program_cache(device, B, K, N, TOP_K):
    """Cache hits must rebind the input, weight, bias and output addresses."""
    torch.manual_seed(42)

    spacers = []
    num_entries_before = device.num_program_cache_entries()
    for i in range(3):
        # A growing live allocation moves every tensor run_fused_op allocates to a new address on each cache
        # hit, and fresh data per iteration makes a stale address show up as an accuracy failure.
        spacers.append(ttnn.from_torch(torch.zeros((32, 32 * (i + 1))), layout=ttnn.TILE_LAYOUT, device=device))
        torch_input = (torch.randn(B, K) * 0.1).to(torch.bfloat16)
        torch_weight = (torch.randn(K, N) * 0.01).to(torch.bfloat16)
        # Keep the selected experts well separated: this test checks buffer
        # rebinding, so BF16 ties must not make a correct cache hit flaky.
        # The random-matmul test above retains unconstrained numerical inputs.
        torch_bias = torch.full((1, N), -8.0, dtype=torch.bfloat16)
        selected = torch.randperm(N)[:TOP_K]
        torch_bias[0, selected] = torch.arange(TOP_K, dtype=torch.bfloat16)
        weights_tt, indices_tt = run_fused_op(device, torch_input, torch_weight, torch_bias, B, K, N, TOP_K)

        ref_logits = (torch_input.float() @ torch_weight.float() + torch_bias.float()).to(torch.bfloat16).float()
        ref_vals, ref_idxs = torch.topk(ref_logits, TOP_K, dim=-1)
        _pcc_passed, pcc_val = comp_pcc(F.softmax(ref_vals, dim=-1), weights_tt)
        assert torch.equal(indices_tt.long(), ref_idxs), f"iteration {i}: cached output selected the wrong experts"
        assert pcc_val >= PCC_THRESHOLD, f"iteration {i}: weight PCC {pcc_val} below threshold {PCC_THRESHOLD}"

    assert device.num_program_cache_entries() - num_entries_before == 1


@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {
                "dispatch_core_axis": ttnn.DispatchCoreAxis.ROW,
            },
            id="dispatch_row",
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "B, K, N, TOP_K",
    TEST_SHAPES,
)
def test_topk_router_gpt_dtype_verification(device, B, K, N, TOP_K):
    """Verify output dtypes are uint16 for indices and bfloat16 for weights."""
    torch.manual_seed(42)
    torch_input = (torch.randn(B, K) * 0.1).to(torch.bfloat16)
    torch_weight = (torch.randn(K, N) * 0.01).to(torch.bfloat16)
    torch_bias = (torch.randn(1, N) * 0.1).to(torch.bfloat16)

    torch_bias_bcast = torch_bias.expand(B, N).contiguous()
    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    tt_weight = ttnn.from_torch(torch_weight, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    tt_bias = ttnn.from_torch(torch_bias_bcast, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

    indices_rm, weights_rm = ttnn.experimental.topk_router_gpt(
        tt_input,
        weight_tensor=tt_weight,
        bias_tensor=tt_bias,
        k=TOP_K,
        num_experts=N,
    )

    logger.info(f"  indices_rm dtype: {indices_rm.dtype}")
    logger.info(f"  weights_rm dtype: {weights_rm.dtype}")

    assert tuple(indices_rm.shape) == (B, TOP_K)
    assert tuple(weights_rm.shape) == (B, TOP_K)
    assert indices_rm.padded_shape[0] == 32
    assert weights_rm.padded_shape[0] == 32
    assert indices_rm.dtype == ttnn.uint16, f"Expected uint16 dtype for indices, got {indices_rm.dtype}"
    assert weights_rm.dtype == ttnn.bfloat16, f"Expected bfloat16 dtype for weights, got {weights_rm.dtype}"


@pytest.mark.parametrize("batch", [1, 9, 32])
@pytest.mark.parametrize("memory_config", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG], ids=["dram", "l1"])
@pytest.mark.parametrize(
    "device_params", [{"dispatch_core_axis": ttnn.DispatchCoreAxis.ROW, "trace_region_size": 4000000}], indirect=True
)
def test_topk_router_gpt_fresh_buffers_and_trace(device, batch, memory_config):
    """Fresh operand addresses and replayed inputs must preserve each logical row."""
    hidden, experts, top_k = 2880, 128, 4
    operands = []
    cache_entries = None

    def upload(value):
        return ttnn.from_torch(
            value, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config
        )

    def check(output, x, weight, bias):
        indices, scores = (ttnn.to_torch(tensor) for tensor in output)
        assert tuple(indices.shape) == (batch, top_k)
        assert tuple(scores.shape) == (batch, top_k)
        indices = indices.long()
        logits = x.float() @ weight.float() + bias.float()
        selected = logits.gather(1, indices)
        best = logits.topk(top_k, dim=-1).values
        # Equal or nearly equal BF16 logits can exchange ranks; their values and
        # normalized routing weights must still agree with the dense reference.
        torch.testing.assert_close(selected, best, atol=0.01, rtol=0.02)
        torch.testing.assert_close(scores.float(), selected.softmax(-1), atol=0.005, rtol=0.02)

    for seed in (37, 83):
        torch.manual_seed(seed)
        x = (torch.randn(batch, hidden) * 0.1).bfloat16()
        weight = (torch.randn(hidden, experts) * 0.01).bfloat16()
        bias = (torch.randn(batch, experts) * 0.1).bfloat16()
        tensors = [upload(value) for value in (x, weight, bias)]
        if operands:
            assert all(new.buffer_address() != old.buffer_address() for new, old in zip(tensors, operands[-1]))
        operands.append(tensors)
        output = ttnn.experimental.topk_router_gpt(
            tensors[0], weight_tensor=tensors[1], bias_tensor=tensors[2], k=top_k, num_experts=experts
        )
        check(output, x, weight, bias)
        if cache_entries is None:
            cache_entries = device.num_program_cache_entries()
        else:
            assert device.num_program_cache_entries() == cache_entries

    trace = ttnn.begin_trace_capture(device, cq_id=0)
    captured = ttnn.experimental.topk_router_gpt(
        tensors[0], weight_tensor=tensors[1], bias_tensor=tensors[2], k=top_k, num_experts=experts
    )
    ttnn.end_trace_capture(device, trace, cq_id=0)
    try:
        for shift in (1, 3):
            changed_x = x.roll(shift, dims=1).contiguous()
            host = ttnn.from_torch(changed_x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
            ttnn.copy_host_to_device_tensor(host, tensors[0])
            ttnn.execute_trace(device, trace, cq_id=0, blocking=True)
            check(captured, changed_x, weight, bias)
    finally:
        ttnn.release_trace(device, trace)
