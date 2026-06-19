# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Standalone unit test for the C++ op ``ttnn.experimental.deepseek.moe.generalized_moe_gate``.

Exercises the device op directly against inlined PyTorch references, so the op can be validated in isolation
without running the full ``MoEGate`` module. Covers all three of its modes:
  - ungrouped global top-k, 256 experts (``test_generalized_moe_gate``, vs ``_generalized_golden``);
  - ungrouped global top-k, 512 experts via the 2-block combine (``test_generalized_moe_gate_512_global``);
  - DeepSeek grouped gate via ``grouped=True`` (``test_generalized_moe_gate_grouped``, vs ``_grouped_golden``) —
    the path the standalone ``deepseek_moe_gate`` op used to own.
Modeled on ``models/demos/deepseek_v3_b1/tests/unit_tests/test_deepseek_moe_gate.py``.
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


def _grouped_golden(input_tensor, bias_tensor, eps=1e-20, scaling_factor=2.5, enable_sigmoid=True):
    """PyTorch reference for the DeepSeek *grouped* gate (the op's ``grouped=True`` path): 256 experts as
    8 groups × 32; per group take the top-2 bias-corrected scores and sum them, pick the top-4 groups by
    that sum, then the top-8 (by bias-corrected score) over those 4 groups (128 experts), gather the
    UNBIASED score at the chosen experts, linearly renormalize and scale. ``input_tensor``/``bias_tensor``:
    [batch, 8, 32]. Returns (scores[batch, 8], global_indices[batch, 8])."""
    row_offsets = torch.arange(input_tensor.shape[-2]) * input_tensor.shape[-1]  # group g -> base id g*32
    batch_idx = torch.arange(input_tensor.shape[0]).unsqueeze(-1)
    scores = torch.sigmoid(input_tensor) if enable_sigmoid else input_tensor
    bias_scores = scores + bias_tensor
    sorted_bias, sorted_indices = torch.sort(bias_scores, dim=-1, descending=True)
    sorted_scores = torch.gather(scores, dim=-1, index=sorted_indices)
    sorted_indices = sorted_indices + row_offsets.view(1, -1, 1)  # local -> global expert id
    top2_sum = sorted_bias[:, :, 0] + sorted_bias[:, :, 1]  # per-group top-2-sum
    _, sorted_top2_indices = torch.sort(top2_sum, dim=-1, descending=True)  # rank groups by it
    top4_values = sorted_bias[batch_idx, sorted_top2_indices[:, :4]].flatten(1)  # top-4 groups' bias scores
    top4_scores = sorted_scores[batch_idx, sorted_top2_indices[:, :4]].flatten(1)
    top4_indices = sorted_indices[batch_idx, sorted_top2_indices[:, :4]].flatten(1)
    _, top8_pos = torch.topk(top4_values, 8, dim=-1, sorted=True)  # top-8 of the 128 by bias score
    top8_scores = torch.gather(top4_scores, dim=-1, index=top8_pos)  # UNBIASED scores at the chosen experts
    top8_indices = torch.gather(top4_indices, dim=-1, index=top8_pos)
    return top8_scores / (torch.sum(top8_scores, dim=-1, keepdim=True) + eps) * scaling_factor, top8_indices


@skip_for_blackhole("Skipped for now. BH performance verification will be tracked in a follow-up PR.")
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("output_softmax", [False, True])
@pytest.mark.parametrize("topk", [8, 6, 4])
@pytest.mark.parametrize("enable_sigmoid", [True, False])
@pytest.mark.parametrize("seed", [42, 201])
# logit_scale only matters on the raw-logit softmax path (see input gen): 1.0 = small/realistic regime,
# 100.0 = past the bf16 exp ceiling (overflow stress). Other paths ignore it and run once (scale 1.0).
@pytest.mark.parametrize("logit_scale", [1.0, 100.0])
def test_generalized_moe_gate(device, batch_size, enable_sigmoid, seed, topk, output_softmax, logit_scale):
    """Test the generalized MoE gate C++ op on a 32x32 tile against the golden reference (top-`topk`,
    linear-normalize or softmax-over-selected)."""
    raw_logit_softmax = output_softmax and not enable_sigmoid  # the only path logit_scale affects
    if logit_scale != 1.0 and not raw_logit_softmax:
        pytest.skip("logit_scale only varies the raw-logit softmax path")

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
        # SOFTMAX path (score_func="softmax", enable_sigmoid=False). logit_scale sweeps two regimes:
        #   1.0 -> sigmoid to [0, 1]: the original small-magnitude coverage (benign, well inside the exp
        #          range; also confirms the max-subtraction does not regress this case).
        #   100 -> raw ~[-100, 100]: UNBOUNDED router logits past the bf16 exp ceiling (~88), which exercises
        #          the in-kernel max-subtraction — without it exp() saturates to inf -> nan/zero weights.
        torch_input = torch.sigmoid(torch_input) if logit_scale == 1.0 else torch_input * logit_scale
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
    # bf16 ranks by a coarse key whose cell width scales with magnitude (ULP ≈ 2^-8·|key|): at ±100 it is
    # ~0.5, so experts whose float keys differ by up to ~0.5 round to the SAME bf16 key — genuinely tied to
    # the device, which may break the tie differently than the float32 golden. So scale the cutoff tolerance
    # with the logit magnitude: tight 1e-2 at the small/[0,1] scales, ~1.0 at ×100. A real mis-selection is
    # off by ≫ that and still fails. (Non-raw paths run only at scale 1.0, so they keep the tight 1e-2.)
    key_atol = 1e-2 * max(1.0, logit_scale)
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
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("output_softmax", [False, True])
@pytest.mark.parametrize("topk", [8, 6, 4])
@pytest.mark.parametrize("enable_sigmoid", [True, False])
@pytest.mark.parametrize("seed", [42, 201])
# logit_scale only matters on the raw-logit softmax path: 1.0 = small/realistic, 100.0 = overflow stress.
@pytest.mark.parametrize("logit_scale", [1.0, 100.0])
def test_generalized_moe_gate_512_global(device, batch_size, enable_sigmoid, seed, topk, output_softmax, logit_scale):
    """512-expert true GLOBAL top-8 (A2 combine). Each of the 2 blocks produces a re-mergeable top-8
    RUN (idx made global via +b*256), stashed to L1; the combine places run0 at {0,2} and run1 at
    {4,6} and finalizes -> the global top-8 over all 512 experts (indices 0-511). GMG_DIAG_BLOCK must
    be UNSET in the kernel. Input layout = slice (each 256-block -> face0 of its own 32x32 tile)."""
    raw_logit_softmax = output_softmax and not enable_sigmoid  # the only path logit_scale affects
    if logit_scale != 1.0 and not raw_logit_softmax:
        pytest.skip("logit_scale only varies the raw-logit softmax path")
    num_experts = 512
    num_blocks = num_experts // 256
    eps, scaling_factor = 1e-20, 2.5
    tile = ttnn.Tile((32, 32))

    torch.manual_seed(seed)
    torch_input = (2 * torch.rand((batch_size, num_experts), dtype=torch.bfloat16)) - 1  # ~[-1, 1]
    if enable_sigmoid:
        pass  # the op sigmoids internally -> scores land in [0, 1]; a raw [-1, 1] input is fine
    elif output_softmax:
        # SOFTMAX path (score_func="softmax", enable_sigmoid=False), across the 512 combine. logit_scale:
        #   1.0 -> sigmoid to [0, 1]: original small-magnitude coverage (also confirms max-sub doesn't regress).
        #   100 -> raw ~[-100, 100]: UNBOUNDED logits past the bf16 exp ceiling (~88) -> exercises max-sub.
        torch_input = torch.sigmoid(torch_input) if logit_scale == 1.0 else torch_input * logit_scale
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
    # The ids are batch-independent (arange + block offset), but the shard grid has one core per batch row,
    # so replicate to batch_size — otherwise rows >0 get an unfilled (zero) index shard and route on id 0.
    idx_blocks = idx_blocks.expand(batch_size, -1, -1, -1).contiguous()  # (batch_size, num_blocks, 16, 16)
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
    # bf16 ranks by a coarse key whose cell width scales with magnitude (ULP ≈ 2^-8·|key|): at ±100 it is
    # ~0.5, so experts whose float keys differ by up to ~0.5 round to the SAME bf16 key — genuinely tied to
    # the device, which may break the tie differently than the float32 golden. Scale the cutoff tolerance
    # with the logit magnitude: tight 1e-2 at the small/[0,1] scales, ~1.0 at ×100. A real mis-selection is
    # off by ≫ that and still fails. (Non-raw paths run only at scale 1.0, so they keep the tight 1e-2.)
    key_atol = 1e-2 * max(1.0, logit_scale)
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
    # Position-aligned, NOT sorted independently: dev_scores[i] and expected[i] both correspond to expert
    # dev_idx[i], so they must match elementwise. Sorting each side separately would only check the weight
    # multiset and would pass even if the kernel paired the right weights with the wrong ids — a real MoE
    # bug, since combine applies weight[i] to expert dev_idx[i].
    assert torch.allclose(
        dev_scores, expected, atol=2e-2
    ), f"512 normalized scores not consistent with device selection.\n dev={dev_scores}\n expected={expected}"


@skip_for_blackhole("Skipped for now. BH performance verification will be tracked in a follow-up PR.")
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("enable_sigmoid", [True, False])
@pytest.mark.parametrize("seed", [42, 201])
def test_generalized_moe_gate_grouped(device, batch_size, enable_sigmoid, seed):
    """DeepSeek GROUPED gate via ``generalized_moe_gate(grouped=True)``: 256 experts = 8 groups × 32 ->
    top-2-sum per group -> top-4 groups -> top-8, linear renorm + scale. Confirms the unified op's grouped
    path (compiled with GMG_UNGROUPED_TOP8=0) matches the grouped golden — the path the standalone
    ``deepseek_moe_gate`` op used to own. grouped fixes top-8 + linear renorm, so topk / output_softmax are
    not swept (the op rejects other values in that mode)."""
    eps, scaling_factor = 1e-20, 2.5
    input_shape = (batch_size, 8, 32)
    reshaped_input_shape = (batch_size, 16, 16)
    shard = (32, 32)
    tile = ttnn.Tile(shard)
    out_shape = (batch_size, 1, 16)

    torch.manual_seed(seed)
    torch_input = (2 * torch.rand(input_shape, dtype=torch.bfloat16)) - 1  # ~[-1, 1]
    if not enable_sigmoid:
        # No in-op sigmoid: keep scores in [0, 1] so the (Σ + eps) linear-renorm denominator is well-conditioned.
        torch_input = torch.sigmoid(torch_input)
    torch_bias = (2 * torch.rand(input_shape, dtype=torch.bfloat16)) - 1

    # Golden INDICES only — scores are validated against the device's OWN selection below (tie-robust).
    _, gold_idx = _grouped_golden(torch_input, torch_bias, eps, scaling_factor, enable_sigmoid)

    grid = device.compute_with_storage_grid_size()
    core_grid = ttnn.num_cores_to_corerangeset(batch_size, ttnn.CoreCoord(grid.x, grid.y), row_wise=True)
    mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(core_grid, shard, ttnn.ShardOrientation.ROW_MAJOR),
    )

    # Same single-256-block device layout as the ungrouped 256 test — only grouped=True differs in the call.
    ttnn_input = ttnn.from_torch(
        torch.reshape(torch_input, reshaped_input_shape),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=mem,
        tile=tile,
    )
    # Bias is transposed within each (16,16) block before upload (the kernel expects the transposed layout).
    reshaped_bias = torch.transpose(torch.reshape(torch_bias, reshaped_input_shape), -2, -1)
    ttnn_bias = ttnn.from_torch(
        reshaped_bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem, tile=tile
    )
    # Transposed routing indices: 0..255 laid out as (16,16) then transposed.
    torch_input_indices = torch.arange(reshaped_input_shape[1] * reshaped_input_shape[2], dtype=torch.int32)
    torch_input_indices = torch_input_indices.unsqueeze(0).expand(reshaped_input_shape[0], -1)
    torch_input_indices = torch_input_indices.reshape(reshaped_input_shape)
    torch_input_indices = torch.transpose(torch_input_indices, -2, -1).to(torch.uint16)
    ttnn_input_indices = ttnn.from_torch(
        torch_input_indices, dtype=ttnn.uint16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem, tile=tile
    )
    ttnn_output = ttnn.from_torch(
        torch.zeros(out_shape, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=mem,
        tile=tile,
    )
    ttnn_output_indices = ttnn.from_torch(
        torch.zeros(out_shape, dtype=torch.uint16),
        dtype=ttnn.uint16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=mem,
        tile=tile,
    )

    logger.info("Running generalized MoE gate (grouped=True) ...")
    res_scores, res_idx = ttnn.experimental.deepseek.moe.generalized_moe_gate(
        ttnn_input,
        bias_tensor=ttnn_bias,
        input_indices_tensor=ttnn_input_indices,
        output_tensor=ttnn_output,
        output_indices_tensor=ttnn_output_indices,
        eps=eps,
        scaling_factor=scaling_factor,
        enable_sigmoid=enable_sigmoid,
        topk=8,
        output_softmax=False,
        grouped=True,
    )

    output_torch = ttnn.to_torch(res_scores)[:, 0, :8]
    output_indices_torch = ttnn.to_torch(res_idx)[:, 0, :8]
    # Sort by index so the device's (tie-arbitrary) order lines up with the golden for the score check.
    sorted_idx, i = torch.sort(output_indices_torch, dim=-1)
    sorted_scores = torch.gather(output_torch, dim=-1, index=i)

    ranking = torch.sigmoid(torch_input) if enable_sigmoid else torch_input
    bias_key = (ranking + torch_bias).reshape(batch_size, -1).float()  # bias-corrected ranking key (256)
    raw_scores = ranking.reshape(batch_size, -1).float()  # UNBIASED scores
    dev_idx = sorted_idx.long()
    gold_idx = torch.sort(gold_idx, dim=-1).values.long()

    logger.info(f"grouped: dev_idx=\n{dev_idx}\ngold_idx=\n{gold_idx}")
    assert dev_idx.min() >= 0 and dev_idx.max() < 256, f"device produced out-of-range expert id:\n{dev_idx}"

    # (1) Selection: the device's chosen experts match the GROUPED golden's by bias-corrected key (sorted
    #     multiset) — NOT a global top-8, the grouped golden's own selection. Tie-robust: a bf16 tie at the
    #     group / rank-8 boundary may swap near-equal-key experts, so compare key VALUES not index positions;
    #     a real grouping/wiring bug shifts a key by >> the tolerance.
    dev_key = torch.gather(bias_key, -1, dev_idx).sort(-1).values
    gold_key = torch.gather(bias_key, -1, gold_idx).sort(-1).values
    assert torch.allclose(dev_key, gold_key, atol=1e-2), (
        f"grouped selection not consistent with the grouped golden.\n dev_idx={dev_idx}\n gold_idx={gold_idx}\n"
        f" dev_key={dev_key}\n gold_key={gold_key}"
    )

    # (2) Scores: self-consistent with the device's OWN selection — linear renorm of the UNBIASED scores at
    #     the experts the device picked, scaled. Position-aligned (both indexed by the sorted dev ids).
    dev_sel = torch.gather(raw_scores, -1, dev_idx)
    expected = dev_sel / (dev_sel.sum(-1, keepdim=True) + eps) * scaling_factor
    assert torch.allclose(
        sorted_scores.float(), expected, atol=1e-2, rtol=1e-4
    ), f"grouped normalized scores not consistent with device selection.\n dev={sorted_scores}\n expected={expected}"
