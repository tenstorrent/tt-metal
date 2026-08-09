# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Test for DeepSeek V3-like MoE architecture (PyTorch reference implementation).

This test validates the full MoE dispatch -> expert -> combine -> weighted sum flow:
1. Tokens are dispatched to expert buffers based on router indices
2. Routed experts (FFN networks) process their assigned tokens
3. Expert outputs are combined back to original token positions
4. Gate weights are applied to each expert contribution (split connection)
5. Shared expert output is added to the final result

Configuration:
- 24 routed experts (each is an FFN with gate_proj, up_proj, down_proj)
- num_experts_per_tok = 4 (each token routes to 4 experts)
- 1 shared expert (same FFN structure as routed experts)
- Dispatch group size = 4
- Experts initialized with random weights for flow verification
"""

import pytest
import torch
from loguru import logger

from models.demos.deepseek_v3_d_p.reference.deepseek_v3_config import DeepSeekV3Config
from models.demos.deepseek_v3_d_p.reference.tt.moe.moe import TorchMoe
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import (
    E4M3_MAX,
    FP8_SCALE_BLOCK,
    ExpertMapping,
    compute_constants,
    create_gate_weights,
    create_shared_expert_weights,
    create_torch_expert_weights,
    get_gate_outputs,
    initialize_test_inputs,
)
from tests.ttnn.utils_for_testing import comp_pcc


def test_compute_constants_reserves_local_expert_tile_padding():
    """A raw 256-token budget needs 736 rows when it spans 16 experts."""
    experts_per_chip, _, dispatch_buffer_rows, max_tokens_per_expert = compute_constants(
        seq_len_per_chip=64,
        num_routed_experts=64,
        num_experts_per_tok=1,
        num_devices=4,
        dispatch_group_size=4,
        dispatch_buffer_capacity_factor=1,
    )

    assert experts_per_chip == 16
    assert max_tokens_per_expert == 256
    assert dispatch_buffer_rows == 736


# dispatch_buffer_capacity_factor below is ceil(N/2) of the most conservative
# integer N such that dgs*seq*N >= theoretical worst-case dispatch buffer.
# Real traffic never approaches the worst case, so half-capacity is sufficient.
# Exception — gate rows on short sequences: the dispatch buffer (dgs*seq*cf rows/chip) must
# also fit the TILE_SIZE-aligned expert regions, i.e. >= experts_per_chip * TILE_SIZE rows
# even for near-empty experts. 64 experts/chip (256 total / dgs=4) need 2048 rows, so the
# 256-expert gate row uses cf=16 (4*32*16 = 2048) while 64-expert rows fit in cf=8.
# fp8 rows need emb_dim divisible by 128 (one scale per block).
@pytest.mark.parametrize(
    "seq_len_per_chip, emb_dim, hidden_dim, num_routed_experts, num_experts_per_tok, dispatch_group_size, dispatch_buffer_capacity_factor, use_gate, model_id, layer_idx, compressed_fp8_dispatch",
    [
        # fmt: off
        pytest.param(32, 64, 128, 24, 4, 4, 2, False, None, None, False, id="random-weights"),
        pytest.param(32, 224, 64, 64, 8, 4, 8, True, None, None, False, id="random-weights-gate"),
        pytest.param(32, 224, 64, 256, 8, 4, 16, True, None, None, False, id="random-weights-gate-256"),
        pytest.param(32, 256, 128, 24, 4, 4, 2, False, None, None, True, id="random-weights-fp8"),
        pytest.param(32, 256, 64, 64, 8, 4, 8, True, None, None, True, id="random-weights-gate-fp8"),
        pytest.param(32,DeepSeekV3Config.EMB_SIZE,DeepSeekV3Config.MOE_INTERMEDIATE_SIZE,DeepSeekV3Config.NUM_ROUTED_EXPERTS,DeepSeekV3Config.NUM_EXPERTS_PER_TOKEN,4,8,False,"deepseek-ai/DeepSeek-V3",3,False,id="hf-weights",marks=pytest.mark.slow,
        ),
        # fmt: on
    ],
)
def test_moe(
    seq_len_per_chip,
    emb_dim,
    hidden_dim,
    num_routed_experts,
    num_experts_per_tok,
    dispatch_group_size,
    dispatch_buffer_capacity_factor,
    use_gate,
    model_id,
    layer_idx,
    compressed_fp8_dispatch,
    expect_error,
):
    """
    Test TorchMoe module with and without integrated gate.

    Without gate: pre-computed weights/indices are passed to forward.
    With gate: gate_weights are passed to TorchMoe, forward only takes x.
    Can run with random weights (fast) or real HF weights (slow).

    With compressed_fp8_dispatch: validates the fp8 dispatch semantics (the executable spec of
    TtMoe's fp8 path) — routing/gate/shared expert untouched, metadata tail carries each token's
    per-128-block fp32 scales bit-exact, the dispatched buffer holds the e4m3 round-trip values,
    and the final output stays within PCC 0.99 of a bf16 baseline run.
    """
    torch.manual_seed(42)
    use_hf_weights = model_id is not None

    logger.debug(f"\n{'='*60}")
    label = "HF Weights" if use_hf_weights else ("Gate" if use_gate else "Random Weights")
    logger.debug(f"TorchMoe Test ({label})")
    if use_hf_weights:
        logger.debug(f"Model: {model_id}, Layer: {layer_idx}")
    logger.debug(f"{'='*60}\n")

    # Compute derived constants
    (
        experts_per_chip,
        metadata_len,
        max_dispatch_buffer_token_size,
        max_dispatched_tokens_per_expert,
    ) = compute_constants(
        seq_len_per_chip,
        num_routed_experts,
        num_experts_per_tok,
        dispatch_group_size,
        dispatch_group_size,
        dispatch_buffer_capacity_factor,
        emb_dim=emb_dim,
        fp8_scaled_input=compressed_fp8_dispatch,
    )

    # Create expert dispatch table
    expert_dispatch_table = ExpertMapping.create_dispatch_table(
        num_routed_experts=num_routed_experts,
        dispatch_group_size=dispatch_group_size,
        num_dispatch_groups=1,
    )

    # Create weights
    if use_hf_weights:
        routed_expert_weights = None
        shared_expert_weights = None
        gate_weights_dict = None
    else:
        logger.debug("Creating random weights for experts...")
        routed_expert_weights = create_torch_expert_weights(num_routed_experts, emb_dim, hidden_dim)
        shared_expert_weights = create_shared_expert_weights(emb_dim, hidden_dim)
        if use_gate:
            gate_weights_dict = create_gate_weights(num_routed_experts, emb_dim, dtype=torch.float32)
        else:
            gate_weights_dict = None

    # Prepare gate inputs (pre-computed weights/indices) when not using gate
    if not use_gate:
        x, weights, indices = initialize_test_inputs(
            dispatch_group_size=dispatch_group_size,
            seq_len_per_chip=seq_len_per_chip,
            emb_dim=emb_dim,
            num_routed_experts=num_routed_experts,
            num_experts_per_tok=num_experts_per_tok,
            max_dispatched_tokens_per_expert=max_dispatched_tokens_per_expert,
        )
        expert_offsets, expert_token_counts, expert_region_offsets, _ = get_gate_outputs(
            indices,
            dispatch_group_size,
            num_routed_experts,
            experts_per_chip,
            seq_len_per_chip,
            num_experts_per_tok,
            expert_dispatch_table=expert_dispatch_table,
        )
    else:
        x = torch.randn(dispatch_group_size, seq_len_per_chip, emb_dim, dtype=torch.float32)

    # Create TorchMoe module. A builder, so the fp8 branch below can construct an identical
    # bf16 baseline (fp8 changes only metadata_len — the scale tail — and the flag itself).
    logger.debug(f"Creating MoE{' with HF weights from ' + model_id if use_hf_weights else ' with random weights'}...")

    def build_moe(fp8):
        return TorchMoe(
            dispatch_group_size=dispatch_group_size,
            experts_per_chip=experts_per_chip,
            num_routed_experts=num_routed_experts,
            num_experts_per_tok=num_experts_per_tok,
            metadata_len=metadata_len if fp8 else 3,
            max_dispatched_tokens_per_expert=max_dispatched_tokens_per_expert,
            max_dispatch_buffer_token_size=max_dispatch_buffer_token_size,
            seq_len_per_chip=seq_len_per_chip,
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
            expert_dispatch_table=expert_dispatch_table,
            model_id=model_id,
            layer_idx=layer_idx,
            routed_expert_weights=routed_expert_weights,
            shared_expert_weights=shared_expert_weights,
            gate_weights=gate_weights_dict,
            # TorchMoe's gate requires these; pinned to the neutral values (single expert group,
            # no group limiting, unit routing scale) so gate rows exercise plain top-k routing.
            n_expert_groups=1 if use_gate else None,
            n_limited_groups=1 if use_gate else None,
            route_scale=1.0 if use_gate else None,
            compressed_fp8_dispatch=fp8,
        )

    moe = build_moe(compressed_fp8_dispatch)

    if use_hf_weights:
        logger.debug("Weight shapes (first routed expert):")
        logger.debug(f"  gate_proj: {moe.routed_experts[0].gate_proj.shape}")
        logger.debug(f"  up_proj: {moe.routed_experts[0].up_proj.shape}")
        logger.debug(f"  down_proj: {moe.routed_experts[0].down_proj.shape}")

    # Test without intermediates (only for random weights without gate - faster)
    if not use_hf_weights and not use_gate:
        logger.debug("Testing forward pass without intermediates...")
        final_output, intermediates = moe(
            x, weights, indices, expert_offsets, expert_token_counts, expert_region_offsets, return_intermediates=False
        )
        assert intermediates is None, "Expected no intermediates when return_intermediates=False"
        assert final_output.shape == x.shape, f"Expected output shape {x.shape}, got {final_output.shape}"
        logger.debug(f"Output shape: {final_output.shape}")
        logger.debug(f"Output sum (abs): {final_output.abs().sum().item():.4f}")

    # Test with intermediates
    logger.debug("\nTesting forward pass with intermediates...")
    if use_gate:
        final_output_2, intermediates = moe(x, return_intermediates=True)
    else:
        final_output_2, intermediates = moe(
            x, weights, indices, expert_offsets, expert_token_counts, expert_region_offsets, return_intermediates=True
        )
    assert intermediates is not None, "Expected intermediates when return_intermediates=True"

    # Verify intermediates shapes
    logger.debug("Intermediate shapes:")
    logger.debug(f"  dispatched_buffer: {intermediates.dispatched_buffer.shape}")
    logger.debug(f"  metadata: {intermediates.metadata.shape}")
    logger.debug(f"  expert_outputs: {intermediates.expert_outputs.shape}")
    logger.debug(f"  shared_output: {intermediates.shared_output.shape}")
    logger.debug(f"  combined_output: {intermediates.combined_output.shape}")
    logger.debug(f"  routed_output: {intermediates.routed_output.shape}")

    assert intermediates.dispatched_buffer.shape == (
        1,
        dispatch_group_size,
        max_dispatch_buffer_token_size,
        emb_dim,
    )
    assert intermediates.shared_output.shape == (dispatch_group_size, seq_len_per_chip, emb_dim)
    assert intermediates.combined_output.shape == (
        dispatch_group_size,
        seq_len_per_chip,
        num_experts_per_tok,
        emb_dim,
    )
    assert intermediates.routed_output.shape == (dispatch_group_size, seq_len_per_chip, emb_dim)

    # Gate-specific checks
    if use_gate:
        assert intermediates.gate_scores is not None, "Expected gate_scores in intermediates"
        assert intermediates.gate_indices is not None, "Expected gate_indices in intermediates"
        logger.debug(f"Gate scores shape: {intermediates.gate_scores.shape}")
        logger.debug(f"Gate indices shape: {intermediates.gate_indices.shape}")

    # Verify both runs produce same output (only for random weights without gate)
    if not use_hf_weights and not use_gate:
        assert torch.allclose(
            final_output, final_output_2
        ), "Outputs should be identical regardless of return_intermediates"

    assert final_output_2.shape == x.shape, f"Expected output shape {x.shape}, got {final_output_2.shape}"

    # Verify no NaN/Inf
    assert not torch.isnan(final_output_2).any(), "Final output contains NaN values"
    assert not torch.isinf(final_output_2).any(), "Final output contains Inf values"
    assert not torch.isnan(intermediates.shared_output).any(), "Shared expert output contains NaN"
    assert not torch.isnan(intermediates.routed_output).any(), "Routed expert output contains NaN"

    logger.debug(
        f"\nOutput stats - min: {final_output_2.min().item():.4f}, max: {final_output_2.max().item():.4f}, mean: {final_output_2.mean().item():.4f}"
    )

    # FP8 dispatch semantics (the executable spec of TtMoe's fp8 path)
    if compressed_fp8_dispatch:
        logger.debug("\nValidating fp8 dispatch semantics against a bf16 baseline run...")
        if use_gate:
            final_bf16, im_bf16 = build_moe(False)(x, return_intermediates=True)
        else:
            final_bf16, im_bf16 = build_moe(False)(
                x,
                weights,
                indices,
                expert_offsets,
                expert_token_counts,
                expert_region_offsets,
                return_intermediates=True,
            )

        # Compression must only touch the dispatched values — routing, gate, and shared expert
        # consume the original x
        assert torch.equal(
            intermediates.metadata[..., :3], im_bf16.metadata[..., :3]
        ), "routing metadata fields 0-2 changed"
        assert torch.equal(intermediates.shared_output, im_bf16.shared_output), "shared expert must consume original x"
        if use_gate:
            assert torch.equal(intermediates.gate_indices, im_bf16.gate_indices), "gate must consume original x"
            assert torch.equal(intermediates.gate_scores, im_bf16.gate_scores), "gate must consume original x"

        # Expected quantization, restated independently of the reference helper
        # (same formula as the cast op unit test's _ref_scale)
        num_scale_blocks = emb_dim // FP8_SCALE_BLOCK
        blocks = x.float().view(dispatch_group_size, seq_len_per_chip, num_scale_blocks, FP8_SCALE_BLOCK)
        exp_scales = blocks.abs().amax(dim=-1).clamp(min=1e-4) / E4M3_MAX
        exp_roundtrip = (
            (blocks / exp_scales.unsqueeze(-1)).to(torch.float8_e4m3fn).float() * exp_scales.unsqueeze(-1)
        ).view(dispatch_group_size, seq_len_per_chip, emb_dim)

        # Every dispatched slot: metadata tail carries the source token's scales bit-exact, and
        # the buffer holds the e4m3 round-trip values (what per_token_cast_back returns on device).
        # With num_dispatch_groups=1, metadata field 0 (linearized mesh coord) is the source chip.
        checked = 0
        for chip in range(dispatch_group_size):
            for slot in range(max_dispatch_buffer_token_size):
                if intermediates.metadata[0, chip, slot, 1] < 0:
                    continue  # unfilled slot (torch reference initializes metadata to -1)
                src = int(intermediates.metadata[0, chip, slot, 0])
                tok = int(intermediates.metadata[0, chip, slot, 1])
                tail = intermediates.metadata[0, chip, slot, 3:].contiguous().view(torch.float32)
                assert torch.equal(tail, exp_scales[src, tok]), f"scale tail mismatch at chip={chip}, slot={slot}"
                assert torch.equal(
                    intermediates.dispatched_buffer[0, chip, slot], exp_roundtrip[src, tok]
                ), f"dispatched buffer is not the e4m3 round-trip of its source token at chip={chip}, slot={slot}"
                checked += 1
        expected_slots = dispatch_group_size * seq_len_per_chip * num_experts_per_tok
        assert checked == expected_slots, f"expected {expected_slots} dispatched slots, validated {checked}"
        logger.debug(f"Scale tail + round-trip values bit-exact on all {checked} dispatched slots")

        # End-to-end: the fp8 path really engaged, and its noise is bounded
        assert not torch.equal(intermediates.dispatched_buffer, im_bf16.dispatched_buffer), "fp8 path did not engage"
        passed, pcc = comp_pcc(final_bf16.float(), final_output_2.float(), 0.99)
        logger.debug(f"final_output PCC (fp8 vs bf16): {pcc}")
        assert passed, f"final_output PCC {pcc} below 0.99"

        # Constructor gate: metadata_len must be sized for the scale tail
        with expect_error(AssertionError, "metadata_len"):
            TorchMoe(
                dispatch_group_size=dispatch_group_size,
                experts_per_chip=experts_per_chip,
                num_routed_experts=num_routed_experts,
                num_experts_per_tok=num_experts_per_tok,
                metadata_len=3,
                max_dispatched_tokens_per_expert=max_dispatched_tokens_per_expert,
                max_dispatch_buffer_token_size=max_dispatch_buffer_token_size,
                seq_len_per_chip=seq_len_per_chip,
                emb_dim=emb_dim,
                hidden_dim=hidden_dim,
                expert_dispatch_table=expert_dispatch_table,
                compressed_fp8_dispatch=True,
            )

    logger.debug("\n" + "=" * 60)
    logger.debug(f"TorchMoe Test ({label}) PASSED!")
    logger.debug("=" * 60)
