# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""CPU contract tests for the published Laguna-XS-2.1 DFlash checkpoint."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tt.dflash_reference import (  # noqa: E402
    DEFAULT_DFLASH_SNAPSHOT,
    LagunaDFlashCheckpoint,
    LayerContextKV,
    apply_neox_rope,
    build_proposal_block,
    causal_sliding_attention,
    evaluate_dflash_draft_argmax_accuracy,
    expected_checkpoint_shapes,
    retain_dflash_context_window,
    rms_norm,
    split_fused_qkv,
)

SNAPSHOT = Path(os.environ.get("LAGUNA_DFLASH_SNAPSHOT", DEFAULT_DFLASH_SNAPSHOT))
HAS_CHECKPOINT = (SNAPSHOT / "config.json").is_file() and (SNAPSHOT / "model.safetensors").is_file()
requires_checkpoint = pytest.mark.skipif(
    not HAS_CHECKPOINT,
    reason="published Laguna DFlash checkpoint is not present in the Hugging Face cache",
)


def test_draft_argmax_accuracy_requires_exact_unique_ids_and_exact_tie_membership():
    reference = torch.tensor(
        [
            [4.0, 3.0, 2.0],
            [8.0, 8.0, 7.0],
        ],
        dtype=torch.bfloat16,
    )
    tt = torch.tensor(
        [
            [4.0, 3.0, 2.0],
            [7.0, 8.5, 7.5],
        ],
        dtype=torch.bfloat16,
    )
    accuracy = evaluate_dflash_draft_argmax_accuracy(tt, reference)
    assert accuracy.passed and accuracy.non_tied_exact and accuracy.tied_membership
    assert not accuracy.literal_exact
    assert accuracy.tt_ids == (0, 1) and accuracy.reference_ids == (0, 0)
    assert accuracy.tied_rows == (1,) and accuracy.tied_maximum_ids == ((0, 1),)

    non_tied_mismatch = tt.clone()
    non_tied_mismatch[0] = torch.tensor([3.0, 5.0, 2.0], dtype=torch.bfloat16)
    accuracy = evaluate_dflash_draft_argmax_accuracy(non_tied_mismatch, reference)
    assert not accuracy.passed and not accuracy.non_tied_exact and accuracy.tied_membership

    outside_tie = tt.clone()
    outside_tie[1] = torch.tensor([7.0, 7.5, 9.0], dtype=torch.bfloat16)
    accuracy = evaluate_dflash_draft_argmax_accuracy(outside_tie, reference)
    assert not accuracy.passed and accuracy.non_tied_exact and not accuracy.tied_membership


def test_draft_argmax_accuracy_rejects_non_raw_or_invalid_logits(expect_error):
    logits = torch.ones((2, 3), dtype=torch.bfloat16)
    with expect_error(ValueError, "shapes differ"):
        evaluate_dflash_draft_argmax_accuracy(logits, logits[:1])
    with expect_error(TypeError, "raw BF16"):
        evaluate_dflash_draft_argmax_accuracy(logits.float(), logits)
    invalid = logits.clone()
    invalid[0, 0] = torch.nan
    with expect_error(ValueError, "finite"):
        evaluate_dflash_draft_argmax_accuracy(invalid, logits)


@requires_checkpoint
def test_published_config_and_checkpoint_layout():
    checkpoint = LagunaDFlashCheckpoint(SNAPSHOT)
    config = checkpoint.config
    checkpoint.validate_layout()

    assert config.num_hidden_layers == 5
    assert config.hidden_size == 2048
    assert config.intermediate_size == 8192
    assert (config.num_attention_heads, config.num_key_value_heads, config.head_dim) == (64, 8, 128)
    assert (config.q_size, config.kv_size, config.fused_qkv_size) == (8192, 1024, 10240)
    assert config.sliding_window == 512
    assert config.block_size == 16
    assert config.max_speculative_tokens == 15
    assert config.mask_token_id == 12
    assert config.target_layer_ids == (1, 13, 25, 33, 39)
    assert config.aux_hidden_state_layer_ids == (2, 14, 26, 34, 40)

    shapes = checkpoint.tensor_shapes()
    assert shapes == expected_checkpoint_shapes(config)
    # Embedding and output projection are intentionally shared with the target.
    assert "embed_tokens.weight" not in shapes
    assert "lm_head.weight" not in shapes


@requires_checkpoint
def test_parallel_proposal_block_geometry(expect_error):
    config = LagunaDFlashCheckpoint(SNAPSHOT).config
    block = build_proposal_block(
        config,
        bonus_token_id=37,
        last_valid_position=1000,
    )
    assert block.input_ids.tolist() == [37] + [12] * 15
    assert block.positions.tolist() == list(range(1001, 1017))
    assert block.sample_indices.tolist() == list(range(1, 16))
    assert block.sample_positions.tolist() == list(range(1002, 1017))

    short = build_proposal_block(
        config,
        bonus_token_id=37,
        last_valid_position=1000,
        num_speculative_tokens=3,
    )
    assert short.input_ids.tolist() == [37, 12, 12, 12]
    assert short.sample_positions.tolist() == [1002, 1003, 1004]
    with expect_error(ValueError, r"\[1, 15\]"):
        build_proposal_block(config, bonus_token_id=37, last_valid_position=1000, num_speculative_tokens=16)


def test_neox_rope_and_causal_sliding_window_semantics():
    x = torch.arange(2 * 2 * 8, dtype=torch.float32).reshape(2, 2, 8) / 10
    rotated = apply_neox_rope(x, torch.tensor([0, 7]), theta=500_000.0)
    torch.testing.assert_close(rotated[0], x[0])
    torch.testing.assert_close(rotated.float().norm(dim=-1), x.float().norm(dim=-1), rtol=1e-6, atol=1e-6)

    # Zero Q/K makes the attention probability uniform over visible values.
    # With window=3, q(pos=3) sees values at positions [1, 2, 3], while
    # q(pos=4) sees [2, 3, 4].  The second query must not leak into the first.
    context = LayerContextKV(
        key=torch.zeros(3, 1, 1),
        value=torch.tensor([1.0, 2.0, 3.0]).reshape(3, 1, 1),
        positions=torch.tensor([0, 1, 2]),
    )
    output = causal_sliding_attention(
        torch.zeros(2, 1, 1),
        torch.zeros(2, 1, 1),
        torch.tensor([4.0, 5.0]).reshape(2, 1, 1),
        torch.tensor([3, 4]),
        context,
        sliding_window=3,
    )
    torch.testing.assert_close(output.flatten(), torch.tensor([3.0, 4.0]))


@requires_checkpoint
def test_context_retention_is_exactly_the_useful_511_row_tail(expect_error):
    config = LagunaDFlashCheckpoint(SNAPSHOT).config
    hidden = torch.arange(600 * 3, dtype=torch.float32).reshape(600, 3)
    positions = torch.arange(900, 1500)
    retained, retained_positions = retain_dflash_context_window(config, hidden, positions)
    assert retained.shape == (511, 3)
    assert retained_positions.tolist() == list(range(989, 1500))
    torch.testing.assert_close(retained, hidden[-511:])

    with expect_error(ValueError, "contiguous"):
        retain_dflash_context_window(config, hidden[:3], torch.tensor([4, 6, 7]))


@requires_checkpoint
@torch.inference_mode()
def test_real_checkpoint_layer_zero_numeric_contract():
    """Exercise every Laguna DFlash primitive with real BF16 checkpoint data.

    A single layer is sufficient to cover fused-QKV splitting, per-head Q/K
    norms, RoPE, causal SWA, softplus head gates, dense SwiGLU, both residual
    additions, and final RMSNorm while keeping this test well below one second
    on the bringup host.
    """

    model = LagunaDFlashCheckpoint(SNAPSHOT).load_reference(layer_indices=(0,))
    config = model.config
    hidden = config.hidden_size

    # Integer-derived inputs avoid RNG/version drift in the BF16 fingerprint.
    aux = (((torch.arange(2 * 5 * hidden) % 97) - 48).float() / 2400).reshape(2, 5, hidden)
    query = (((torch.arange(3 * hidden) % 83) - 41).float() / 2100).reshape(3, hidden)
    aux = aux.to(torch.bfloat16)
    query = query.to(torch.bfloat16)
    context_positions = torch.tensor([509, 510])
    query_positions = torch.tensor([511, 512, 513])

    combined = model.combine_aux_hidden_states(aux)
    context_kv = model.precompute_context_kv(combined, context_positions)
    output = model.forward_query_embeddings(query, query_positions, context_kv)

    assert combined.shape == (2, 2048)
    assert context_kv[0].key.shape == (2, 8, 128)
    assert context_kv[0].value.shape == (2, 8, 128)
    assert output.shape == (3, 2048)
    assert combined.dtype == context_kv[0].key.dtype == output.dtype == torch.bfloat16
    assert torch.isfinite(combined).all() and torch.isfinite(output).all()

    # Independently project and split layer-0 fused QKV to lock row ordering.
    normalized_context = rms_norm(
        combined,
        model.weights["layers.0.input_layernorm.weight"],
        config.rms_norm_eps,
    )
    fused = F.linear(normalized_context, model.weights["layers.0.self_attn.qkv_proj.weight"])
    q, k, v = split_fused_qkv(fused, config)
    assert (q.shape[-1], k.shape[-1], v.shape[-1]) == (8192, 1024, 1024)
    torch.testing.assert_close(torch.cat((q, k, v), dim=-1), fused, rtol=0, atol=0)

    # The gate is positive and one scalar per query head before broadcasting.
    normalized_query = rms_norm(
        query,
        model.weights["layers.0.input_layernorm.weight"],
        config.rms_norm_eps,
    )
    head_gate = F.softplus(F.linear(normalized_query, model.weights["layers.0.self_attn.g_proj.weight"]).float())
    assert head_gate.shape == (3, 64)
    assert bool((head_gate > 0).all())

    # BF16 golden values catch changes in aux-slice order, norms, rotation,
    # attention masking/gating, residual order, or SwiGLU semantics.
    torch.testing.assert_close(
        combined[0, :8].float(),
        torch.tensor(
            [-0.052978515625, -0.291015625, -0.39453125, 0.6796875, 1.2734375, -2.28125, 0.1494140625, 0.59375]
        ),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        context_kv[0].key[0, 0, :8].float(),
        torch.tensor(
            [-0.60546875, -0.396484375, 0.55078125, 1.71875, -1.328125, -1.421875, -0.07666015625, -0.83984375]
        ),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        output[0, :8].float(),
        torch.tensor([1.21875, -0.484375, -0.353515625, 1.2265625, 0.007659912109375, 1.34375, -1.453125, 0.32421875]),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(output.float().norm(), torch.tensor(78.38668060302734), rtol=1e-6, atol=1e-6)

    # Shared target embedding/LM-head plumbing: neither tensor is draft-owned.
    target_embedding = (((torch.arange(13 * hidden) % 31) - 15).float() / 1000).reshape(13, hidden)
    target_lm_head = (((torch.arange(17 * hidden) % 29) - 14).float() / 1000).reshape(17, hidden)
    target_embedding = target_embedding.to(torch.bfloat16)
    target_lm_head = target_lm_head.to(torch.bfloat16)
    block = build_proposal_block(
        config,
        bonus_token_id=7,
        last_valid_position=510,
        num_speculative_tokens=2,
    )
    embeddings = model.embed_input_ids(block.input_ids, target_embedding)
    torch.testing.assert_close(embeddings, target_embedding[block.input_ids], rtol=0, atol=0)
    logits = model.proposal_logits(
        block,
        target_embedding_weight=target_embedding,
        target_lm_head_weight=target_lm_head,
        context_aux_hidden_states=aux,
        context_positions=context_positions,
    )
    assert logits.shape == (2, 17)
    assert torch.isfinite(logits).all()


@requires_checkpoint
@torch.inference_mode()
def test_real_checkpoint_full_five_layer_one_round_contract():
    """Load every published layer and execute the exact anchor+15 proposal."""

    model = LagunaDFlashCheckpoint(SNAPSHOT).load_reference()
    config = model.config
    hidden = config.hidden_size
    assert model.layer_indices == (0, 1, 2, 3, 4)

    aux = (((torch.arange(2 * 5 * hidden) % 97) - 48).float() / 2400).reshape(2, 5, hidden)
    aux = aux.to(torch.bfloat16)
    context_positions = torch.tensor([123, 124])
    target_embedding = (((torch.arange(13 * hidden) % 31) - 15).float() / 1000).reshape(13, hidden)
    target_lm_head = (((torch.arange(19 * hidden) % 29) - 14).float() / 1000).reshape(19, hidden)
    target_embedding = target_embedding.to(torch.bfloat16)
    target_lm_head = target_lm_head.to(torch.bfloat16)
    block = build_proposal_block(config, bonus_token_id=7, last_valid_position=124)

    context_states = model.combine_aux_hidden_states(aux)
    contexts = model.precompute_context_kv(context_states, context_positions)
    query = model.embed_input_ids(block.input_ids, target_embedding)
    hidden_states = model.forward_query_embeddings(query, block.positions, contexts)
    traced_hidden, layer_outputs = model.forward_query_embeddings_with_layer_outputs(
        query,
        block.positions,
        contexts,
    )
    logits = model.proposal_logits(
        block,
        target_embedding_weight=target_embedding,
        target_lm_head_weight=target_lm_head,
        context_aux_hidden_states=aux,
        context_positions=context_positions,
    )

    assert tuple(contexts) == (0, 1, 2, 3, 4)
    assert len(layer_outputs) == 5
    assert all(stage.shape == (16, hidden) and stage.dtype == torch.bfloat16 for stage in layer_outputs)
    torch.testing.assert_close(traced_hidden, hidden_states, rtol=0, atol=0)
    assert hidden_states.shape == (16, hidden)
    assert logits.shape == (15, 19)
    assert hidden_states.dtype == logits.dtype == torch.bfloat16
    assert torch.isfinite(hidden_states).all() and torch.isfinite(logits).all()
    torch.testing.assert_close(logits, model.compute_logits(hidden_states[1:16], target_lm_head), rtol=0, atol=0)
