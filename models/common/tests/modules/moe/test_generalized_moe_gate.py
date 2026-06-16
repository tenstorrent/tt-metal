# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Standalone unit test for the C++ op ``ttnn.experimental.deepseek.moe.generalized_moe_gate``.

Exercises the device op directly against the inlined PyTorch ``_generalized_golden`` reference, so the
op can be validated in isolation without running the full ``MoEGate`` module. Modeled on
``models/demos/deepseek_v3_b1/tests/unit_tests/test_deepseek_moe_gate.py``.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import skip_for_blackhole


def _generalized_golden(
    input_tensor, bias_tensor, eps=1e-20, scaling_factor=2.5, enable_sigmoid=False, topk=8, output_softmax=False
):
    """PyTorch reference for the *ungrouped* generalized MoE gate: rank by the bias-corrected score, take
    the global top-`topk`, gather the UNBIASED score at those experts, normalize (softmax-over-selected if
    output_softmax else linear), scale. ``input_tensor``/``bias_tensor``: [batch, n_group, group_size]."""
    batch = input_tensor.shape[0]
    scores = torch.sigmoid(input_tensor) if enable_sigmoid else input_tensor
    bias_scores = scores + bias_tensor
    _, topk_indices = torch.topk(bias_scores.reshape(batch, -1), topk, dim=-1, sorted=True)
    topk_scores = torch.gather(scores.reshape(batch, -1), dim=-1, index=topk_indices)
    if output_softmax:
        # Subtract the per-row max before exp: numerically stable and mathematically identical
        # (softmax is shift-invariant). Mirrors the in-kernel max-subtraction so this reference stays
        # valid for RAW router logits (score_func="softmax"), not just inputs squashed to [0, 1].
        topk_scores = topk_scores - topk_scores.max(dim=-1, keepdim=True).values
        weights = torch.exp(topk_scores)
    else:
        weights = topk_scores
    return weights / (torch.sum(weights, dim=-1, keepdim=True) + eps) * scaling_factor, topk_indices


@skip_for_blackhole("Skipped for now. BH performance verification will be tracked in a follow-up PR.")
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("output_softmax", [False, True])
@pytest.mark.parametrize("topk", [8, 6, 4])
@pytest.mark.parametrize("enable_sigmoid", [True, False])
@pytest.mark.parametrize("seed", [42, 201, 512])
def test_generalized_moe_gate(device, batch_size, enable_sigmoid, seed, topk, output_softmax):
    """Test the generalized MoE gate C++ op on a 32x32 tile against the golden reference (top-`topk`,
    linear-normalize or softmax-over-selected)."""

    # Tensor dimensions — full 32x32 tile, logical 32x32 per shard.
    input_shape = (batch_size, 8, 32)
    reshaped_input_shape = (batch_size, 16, 16)
    input_shard_shape = (32, 32)
    input_tile = ttnn.Tile(input_shard_shape)
    output_shape = (batch_size, 1, 16)
    output_shard_shape = (32, 32)
    output_tile = ttnn.Tile(output_shard_shape)

    logger.info(f"Testing generalized MoE gate with input shape {input_shape}")

    # Create input PyTorch tensor with random values.
    torch.manual_seed(seed)
    torch_input = (2 * torch.rand(input_shape, dtype=torch.bfloat16)) - 1  # ~[-1, 1]
    if enable_sigmoid:
        pass  # the op sigmoids internally -> scores land in [0, 1]; a raw [-1, 1] input is fine
    elif output_softmax:
        # RAW-LOGIT SOFTMAX path: score_func="softmax" feeds the op UNBOUNDED router logits, NOT [0, 1].
        # Scale past the bf16 exp range (exp overflows ~88) so this exercises the in-kernel max-subtraction:
        # without it exp() saturates to inf -> nan/zero weights; with it the result is the correct softmax.
        # (Real logits are smaller, but the test deliberately overflows to guard the fix; the squash-to-[0,1]
        # used by the linear path below would hide this case entirely.)
        torch_input = torch_input * 100.0  # ~[-100, 100]
    else:
        # LINEAR-renorm path: keep scores in [0, 1] so the (Σ + eps) denominator stays well-conditioned.
        torch_input = torch.sigmoid(torch_input)
    torch_bias = (2 * torch.rand(input_shape, dtype=torch.bfloat16)) - 1
    eps = 1e-20
    scaling_factor = 2.5

    # Reference output. (Only the golden indices are used — scores are validated tie-robustly below
    # against the device's OWN selection, not the golden's scores, so the golden scores are unused here.)
    _, top8_indices = _generalized_golden(
        torch_input, torch_bias, eps, scaling_factor, enable_sigmoid, topk, output_softmax
    )

    grid = device.compute_with_storage_grid_size()
    core_grid = ttnn.num_cores_to_corerangeset(
        batch_size,
        ttnn.CoreCoord(grid.x, grid.y),
        row_wise=True,
    )
    input_shard_spec = ttnn.ShardSpec(
        core_grid,
        input_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    input_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, input_shard_spec)

    output_shard_spec = ttnn.ShardSpec(
        core_grid,
        output_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    output_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, output_shard_spec)

    # Input values — sharded on a single core per batch.
    reshaped_input = torch.reshape(torch_input, reshaped_input_shape)
    ttnn_input = ttnn.from_torch(
        reshaped_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=input_mem_config,
        tile=input_tile,
    )

    # Bias is transposed before upload (the kernel expects the transposed layout).
    reshaped_bias = torch.transpose(torch.reshape(torch_bias, reshaped_input_shape), -2, -1)
    ttnn_bias = ttnn.from_torch(
        reshaped_bias,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=input_mem_config,
        tile=input_tile,
    )

    # Transposed routing indices: 0..255 laid out as (16,16) then transposed.
    torch_input_indices = torch.arange(reshaped_input_shape[1] * reshaped_input_shape[2], dtype=torch.int32)
    torch_input_indices = torch_input_indices.unsqueeze(0).expand(reshaped_input_shape[0], -1)
    torch_input_indices = torch_input_indices.reshape(reshaped_input_shape)
    torch_input_indices = torch.transpose(torch_input_indices, -2, -1).to(torch.uint16)
    ttnn_input_indices = ttnn.from_torch(
        torch_input_indices,
        dtype=ttnn.uint16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=input_mem_config,
        tile=input_tile,
    )

    # Preallocated output buffers (filled in place by the op).
    ttnn_output = ttnn.from_torch(
        torch.zeros(output_shape, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=output_mem_config,
        tile=output_tile,
    )
    ttnn_output_indices = ttnn.from_torch(
        torch.zeros(output_shape, dtype=torch.uint16),
        dtype=ttnn.uint16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=output_mem_config,
        tile=output_tile,
    )

    logger.info("Running generalized MoE gate operation...")
    ttnn_result, ttnn_result_indices = ttnn.experimental.deepseek.moe.generalized_moe_gate(
        ttnn_input,
        bias_tensor=ttnn_bias,
        input_indices_tensor=ttnn_input_indices,
        output_tensor=ttnn_output,
        output_indices_tensor=ttnn_output_indices,
        eps=eps,
        scaling_factor=scaling_factor,
        enable_sigmoid=enable_sigmoid,
        topk=topk,
        output_softmax=output_softmax,
    )

    # Convert back to torch and keep the top-`topk` slots (ranks 0..topk-1 sit in the first topk cols;
    # the dropped ranks topk..7 are zeroed by the kernel).
    output_torch = ttnn.to_torch(ttnn_result)[:, 0, :topk]
    output_indices_torch = ttnn.to_torch(ttnn_result_indices)[:, 0, :topk]

    # The op does not guarantee a stable order across ties, so sort both by index
    # before comparing (same approach as the reference unit test).
    sorted_output_indices_torch, i = torch.sort(output_indices_torch, dim=-1)
    sorted_output_torch = torch.gather(output_torch, dim=-1, index=i)

    top8_indices = torch.sort(top8_indices, dim=-1).values

    # bf16 produces many equal bias-corrected values, so the exact top-8 *indices* are ambiguous at
    # the rank-8 cutoff (genuine ties — e.g. two experts with identical bf16 bias fight for the last
    # slot, and torch.topk vs the device break it differently). A strict index match is the wrong
    # check. Validate tie-robustly:
    #   (1) the device's selected experts form a VALID top-8 by the bias-corrected ranking key
    #       (same sorted key multiset as the golden), and
    #   (2) the normalized scores are self-consistent with the device's own selection.
    ranking = torch.sigmoid(torch_input) if enable_sigmoid else torch_input
    bias_key = (ranking + torch_bias).reshape(batch_size, -1).float()
    raw_scores = ranking.reshape(batch_size, -1).float()
    dev_idx = sorted_output_indices_torch.long()
    gold_idx = top8_indices.long()

    logger.info(f"dev_idx=\n{dev_idx}\ngold_idx=\n{gold_idx}")
    assert dev_idx.min() >= 0 and dev_idx.max() < 256, f"device produced out-of-range expert id:\n{dev_idx}"

    dev_key = torch.gather(bias_key, dim=-1, index=dev_idx).sort(dim=-1).values
    gold_key = torch.gather(bias_key, dim=-1, index=gold_idx).sort(dim=-1).values
    # bf16 ranks by a coarse key: near ±100 (the raw-logit path) the bf16 cell is ~0.5 wide, so experts
    # whose float keys differ by up to ~0.5 round to the SAME bf16 key — they are genuinely tied to the
    # device and it may break the tie either way (the float32 golden breaks it differently). That gap
    # scales with magnitude, so loosen the cutoff tolerance for the ×100 raw-logit path; the squashed
    # [0,1] paths keep the tight 1e-2. A real mis-selection is off by ≫1 and still fails.
    key_atol = 1.0 if (output_softmax and not enable_sigmoid) else 1e-2
    assert torch.allclose(dev_key, gold_key, atol=key_atol), (
        f"Device selection is not a valid top-8 by bias key.\n dev_idx={dev_idx}\n gold_idx={gold_idx}"
        f"\n dev_key={dev_key}\n gold_key={gold_key}"
    )

    dev_sel = torch.gather(raw_scores, dim=-1, index=dev_idx)
    # Consistency check vs the device's OWN selection: softmax-over-selected when output_softmax, else linear.
    if output_softmax:
        # Max-subtract before exp (matches the kernel; stable for raw logits, identical for [0, 1]).
        weights = torch.exp(dev_sel - dev_sel.max(dim=-1, keepdim=True).values)
    else:
        weights = dev_sel
    expected_norm = weights / (weights.sum(dim=-1, keepdim=True) + eps) * scaling_factor
    assert torch.allclose(
        sorted_output_torch.float(), expected_norm, atol=1e-2, rtol=1e-4
    ), "Normalized scores are not consistent with the device's own top-8 selection"


@skip_for_blackhole("Skipped for now. BH performance verification will be tracked in a follow-up PR.")
@pytest.mark.parametrize("output_softmax", [False, True])
@pytest.mark.parametrize("topk", [8, 6, 4])
@pytest.mark.parametrize("enable_sigmoid", [True, False])
@pytest.mark.parametrize("seed", [42, 201, 512])
def test_generalized_moe_gate_512_global(device, enable_sigmoid, seed, topk, output_softmax):
    """512-expert true GLOBAL top-8 (A2 combine). Each of the 2 blocks produces a re-mergeable top-8
    RUN (idx made global via +b*256), stashed to L1; the combine places run0 at {0,2} and run1 at
    {4,6} and finalizes -> the global top-8 over all 512 experts (indices 0-511). GMG_DIAG_BLOCK must
    be UNSET in the kernel. Input layout = slice (each 256-block -> face0 of its own 32x32 tile)."""
    num_experts = 512
    num_blocks = num_experts // 256
    batch_size = 1
    eps, scaling_factor = 1e-20, 2.5
    tile = ttnn.Tile((32, 32))

    torch.manual_seed(seed)
    torch_input = (2 * torch.rand((batch_size, num_experts), dtype=torch.bfloat16)) - 1  # ~[-1, 1]
    if enable_sigmoid:
        pass  # the op sigmoids internally -> scores land in [0, 1]; a raw [-1, 1] input is fine
    elif output_softmax:
        # RAW-LOGIT SOFTMAX path: score_func="softmax" feeds the op UNBOUNDED router logits, NOT [0, 1].
        # Scale past the bf16 exp range (exp overflows ~88) to exercise the in-kernel max-subtraction
        # across the 512-expert combine path (without it exp() saturates to inf -> nan/zero weights).
        torch_input = torch_input * 100.0  # ~[-100, 100]
    else:
        # LINEAR-renorm path: keep scores in [0, 1] so the (Σ + eps) denominator stays well-conditioned.
        torch_input = torch.sigmoid(torch_input)
    torch_bias = (2 * torch.rand((batch_size, num_experts), dtype=torch.bfloat16)) - 1

    # Golden: flatten (batch, 512) -> true global top-`topk` (indices 0-511). Only the golden INDICES are
    # used (selection check below); scores are validated against the device's OWN selection, not the
    # golden's, because a bf16 tie at the cutoff can pick different-but-valid experts (see the score check).
    _, gold_idx = _generalized_golden(
        torch_input, torch_bias, eps, scaling_factor, enable_sigmoid, topk, output_softmax
    )
    scores_all = (torch.sigmoid(torch_input) if enable_sigmoid else torch_input).float()
    bias_key = scores_all + torch_bias.float()  # bias-corrected ranking key, (batch, 512)

    logits_blocks = torch_input.reshape(batch_size, num_blocks, 16, 16)
    # bias uploaded transposed within each (16,16) block (kernel expects the transposed layout).
    bias_blocks = torch.transpose(torch_bias.reshape(batch_size, num_blocks, 16, 16), -2, -1).contiguous()

    grid = device.compute_with_storage_grid_size()
    core_grid = ttnn.num_cores_to_corerangeset(batch_size, ttnn.CoreCoord(grid.x, grid.y), row_wise=True)

    def mem(shard):
        return ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(core_grid, shard, ttnn.ShardOrientation.ROW_MAJOR),
        )

    multi, one = (num_blocks * 32, 32), (32, 32)
    ttnn_input = ttnn.from_torch(
        logits_blocks, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem(multi), tile=tile
    )
    ttnn_bias = ttnn.from_torch(
        bias_blocks, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem(multi), tile=tile
    )
    # input_indices: one tile per block, holding that block's GLOBAL expert ids (block b = arange + b*256),
    # transposed per block (kernel expects the transposed layout). The pipeline tracks global ids directly.
    ar = torch.arange(256, dtype=torch.int32).reshape(1, 1, 16, 16)
    offs = (torch.arange(num_blocks, dtype=torch.int32) * 256).reshape(1, num_blocks, 1, 1)
    idx_blocks = torch.transpose(ar + offs, -2, -1).contiguous().to(torch.uint16)  # (1, num_blocks, 16, 16)
    ttnn_input_indices = ttnn.from_torch(
        idx_blocks, dtype=ttnn.uint16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem(multi), tile=tile
    )
    out_shape = (batch_size, 1, 16)
    ttnn_output = ttnn.from_torch(
        torch.zeros(out_shape, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=mem(one),
        tile=tile,
    )
    ttnn_output_indices = ttnn.from_torch(
        torch.zeros(out_shape, dtype=torch.uint16),
        dtype=ttnn.uint16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=mem(one),
        tile=tile,
    )

    res_scores, res_idx = ttnn.experimental.deepseek.moe.generalized_moe_gate(
        ttnn_input,
        bias_tensor=ttnn_bias,
        input_indices_tensor=ttnn_input_indices,
        output_tensor=ttnn_output,
        output_indices_tensor=ttnn_output_indices,
        eps=eps,
        scaling_factor=scaling_factor,
        enable_sigmoid=enable_sigmoid,
        topk=topk,
        output_softmax=output_softmax,
    )

    dev_idx = ttnn.to_torch(res_idx)[:, 0, :topk].to(torch.int64)
    dev_scores = ttnn.to_torch(res_scores)[:, 0, :topk].float()
    logger.info(f"512 global (topk={topk}): dev_idx={dev_idx}  gold_idx={gold_idx}")

    # Indices: dev must be a valid GLOBAL top-`topk` (tie-robust: compare the gathered bias-keys, sorted).
    dev_key = torch.gather(bias_key, -1, dev_idx).sort(-1).values
    gold_key = torch.gather(bias_key, -1, gold_idx.to(torch.int64)).sort(-1).values
    # bf16 ranks by a coarse key: near ±100 (the raw-logit path) the bf16 cell is ~0.5 wide, so experts
    # whose float keys differ by up to ~0.5 round to the SAME bf16 key — they are genuinely tied to the
    # device and it may break the tie either way. Loosen the cutoff tolerance for the ×100 raw-logit path
    # (the gap scales with magnitude); the squashed [0,1] paths keep the tight 1e-2.
    key_atol = 1.0 if (output_softmax and not enable_sigmoid) else 1e-2
    assert torch.allclose(dev_key, gold_key, atol=key_atol), (
        f"512 global not a valid top-{topk}.\n dev_idx={dev_idx}\n gold_idx={gold_idx}\n"
        f" dev_key={dev_key}\n gold_key={gold_key}"
    )

    # Scores: validate against the device's OWN selection (selection-agnostic, like the 256 test). At ±100
    # the device and golden may break a bf16 tie toward DIFFERENT (equally valid) experts, so comparing to
    # the golden's scores is wrong; recompute the expected softmax/linear weights over the device's selected
    # raw scores instead. The selection check above already confirmed those experts are a valid top-`topk`.
    dev_sel = torch.gather(scores_all, -1, dev_idx)
    if output_softmax:
        # Max-subtract before exp (matches the kernel; stable for raw logits, identical for [0, 1]).
        w = torch.exp(dev_sel - dev_sel.max(-1, keepdim=True).values)
    else:
        w = dev_sel
    expected = w / (w.sum(-1, keepdim=True) + eps) * scaling_factor
    assert torch.allclose(
        dev_scores.sort(-1).values, expected.sort(-1).values, atol=2e-2
    ), f"512 normalized scores not consistent with device selection.\n dev={dev_scores}\n expected={expected}"
