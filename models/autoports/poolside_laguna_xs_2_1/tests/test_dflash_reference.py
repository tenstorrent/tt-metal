# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""CPU contract tests for the published Laguna DFlash checkpoints.

Every checkpoint-backed test runs once per published draft (``tt/model_spec.py``
``DFLASH_MODELS``: Laguna-XS-2.1-DFlash and Laguna-S-2.1-DFlash) whose snapshot is in
the Hugging Face cache, independent of ``TT_LAGUNA_MODEL``: the reference is pure CPU
and owns no target weights.  ``LAGUNA_DFLASH_SNAPSHOT`` replaces the selected
checkpoint's snapshot only.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tt.dflash_reference import (  # noqa: E402
    DFLASH_TARGET_LAYER_IDS,
    LagunaDFlashCheckpoint,
    LagunaDFlashConfig,
    LayerContextKV,
    apply_neox_rope,
    build_proposal_block,
    causal_sliding_attention,
    dflash_snapshot_path,
    evaluate_dflash_draft_argmax_accuracy,
    expected_checkpoint_shapes,
    published_dflash_config,
    retain_dflash_context_window,
    rms_norm,
    split_fused_qkv,
)
from tt.model_spec import DFLASH_MODELS, MODEL_ID  # noqa: E402

XS = "poolside/Laguna-XS-2.1"
S = "poolside/Laguna-S-2.1"


def _snapshot(model_id: str) -> Path:
    override = os.environ.get("LAGUNA_DFLASH_SNAPSHOT")
    if override and model_id == MODEL_ID:
        return Path(override)
    return dflash_snapshot_path(model_id)


def _has_checkpoint(model_id: str) -> bool:
    snapshot = _snapshot(model_id)
    return (snapshot / "config.json").is_file() and (snapshot / "model.safetensors").is_file()


# One parameter per published draft; a draft whose snapshot is absent skips.
CHECKPOINTS = [
    pytest.param(
        model_id,
        id=model_id.split("/")[-1],
        marks=pytest.mark.skipif(
            not _has_checkpoint(model_id),
            reason=f"published {DFLASH_MODELS[model_id].repo_id} checkpoint is not present in the Hugging Face cache",
        ),
    )
    for model_id in sorted(DFLASH_MODELS)
]


@dataclass(frozen=True)
class _Expected:
    """Independently written published values plus locked BF16 layer-0 fingerprints."""

    layers: int
    hidden: int
    intermediate: int
    heads: tuple[int, int, int]
    sizes: tuple[int, int, int]
    target_layer_ids: tuple[int, ...]
    aux_layer_ids: tuple[int, ...]
    max_position_embeddings: int
    combined: tuple[float, ...]
    key: tuple[float, ...]
    output: tuple[float, ...]
    output_norm: float


EXPECTED = {
    XS: _Expected(
        layers=5,
        hidden=2048,
        intermediate=8192,
        heads=(64, 8, 128),
        sizes=(8192, 1024, 10240),
        target_layer_ids=(1, 13, 25, 33, 39),
        aux_layer_ids=(2, 14, 26, 34, 40),
        max_position_embeddings=262144,
        combined=(-0.052978515625, -0.291015625, -0.39453125, 0.6796875, 1.2734375, -2.28125, 0.1494140625, 0.59375),
        key=(-0.60546875, -0.396484375, 0.55078125, 1.71875, -1.328125, -1.421875, -0.07666015625, -0.83984375),
        output=(1.21875, -0.484375, -0.353515625, 1.2265625, 0.007659912109375, 1.34375, -1.453125, 0.32421875),
        output_norm=78.38668060302734,
    ),
    # Laguna-S-2.1-DFlash revision 1334981872c58f5023614307b4536f8b23c262e5 (config.json).  The
    # fingerprint was locked from this reference on 2026-10-01 with the same integer-derived
    # inputs as XS; it guards against drift, while test_draft_proposals_track_real_target_greedy gives the
    # independent evidence that the S wiring is right.
    S: _Expected(
        layers=6,
        hidden=3072,
        intermediate=12288,
        heads=(72, 8, 128),
        sizes=(9216, 1024, 11264),
        target_layer_ids=(1, 10, 19, 29, 38, 47),
        aux_layer_ids=(2, 11, 20, 30, 39, 48),
        max_position_embeddings=1048576,
        combined=(
            -1.0234375,
            0.92578125,
            -0.119140625,
            0.546875,
            0.455078125,
            0.01373291015625,
            -1.09375,
            -0.1474609375,
        ),
        key=(-0.51171875, 1.6796875, 0.04541015625, -0.54296875, -1.625, -0.90625, 0.78515625, -0.52734375),
        output=(-0.66015625, -0.80078125, 0.322265625, 1.3984375, 0.46875, -0.10888671875, 0.671875, 0.0888671875),
        output_norm=95.99884796142578,
    ),
}


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


@pytest.mark.parametrize("model_id", sorted(DFLASH_MODELS), ids=lambda m: m.split("/")[-1])
def test_spec_matches_independent_published_values(model_id):
    """``model_spec.DFLASH_MODELS`` against the values copied from each published config.json."""

    expected = EXPECTED[model_id]
    spec = DFLASH_MODELS[model_id]
    assert spec.repo_id == f"{model_id}-DFlash"
    assert spec.num_draft_layers == expected.layers
    assert (spec.hidden_size, spec.intermediate_size) == (expected.hidden, expected.intermediate)
    assert (spec.num_attention_heads, spec.num_key_value_heads, spec.head_dim) == expected.heads
    assert spec.target_layer_ids == expected.target_layer_ids
    assert spec.aux_hidden_state_layer_ids == expected.aux_layer_ids
    assert spec.max_position_embeddings == expected.max_position_embeddings
    assert (spec.vocab_size, spec.sliding_window, spec.block_size, spec.mask_token_id) == (100352, 512, 16, 12)

    config = published_dflash_config(model_id)
    assert config.target_model_id == model_id
    assert (config.q_size, config.kv_size, config.fused_qkv_size) == expected.sizes
    assert config.num_aux_hidden_states == len(expected.target_layer_ids)


def test_selected_checkpoint_defaults_follow_tt_laguna_model():
    spec = DFLASH_MODELS[MODEL_ID]
    assert DFLASH_TARGET_LAYER_IDS == spec.target_layer_ids
    assert dflash_snapshot_path().parts[-3:] == (
        "models--" + spec.repo_id.replace("/", "--"),
        "snapshots",
        spec.revision,
    )


def test_validation_rejects_cross_checkpoint_target_ids(expect_error):
    """S geometry with the XS capture layers (and vice versa) is not a published draft."""

    import dataclasses

    xs = published_dflash_config(XS)
    s = published_dflash_config(S)
    with expect_error(ValueError, "unexpected Laguna DFlash target layer IDs"):
        dataclasses.replace(
            s,
            target_layer_ids=(1, 13, 25, 33, 39, 47),
            aux_hidden_state_layer_ids=(2, 14, 26, 34, 40, 48),
        ).validate()
    with expect_error(ValueError, "any published Laguna DFlash geometry"):
        dataclasses.replace(xs, num_hidden_layers=6, layer_types=("sliding_attention",) * 6).validate()


@pytest.mark.parametrize("model_id", CHECKPOINTS)
def test_published_config_and_checkpoint_layout(model_id):
    expected = EXPECTED[model_id]
    snapshot = _snapshot(model_id)
    checkpoint = LagunaDFlashCheckpoint(snapshot)
    config = checkpoint.config
    checkpoint.validate_layout()

    assert config.num_hidden_layers == expected.layers
    assert config.hidden_size == expected.hidden
    assert config.intermediate_size == expected.intermediate
    assert (config.num_attention_heads, config.num_key_value_heads, config.head_dim) == expected.heads
    assert (config.q_size, config.kv_size, config.fused_qkv_size) == expected.sizes
    assert config.sliding_window == 512
    assert config.block_size == 16
    assert config.max_speculative_tokens == 15
    assert config.mask_token_id == 12
    assert config.max_position_embeddings == expected.max_position_embeddings
    assert config.target_layer_ids == expected.target_layer_ids
    assert config.aux_hidden_state_layer_ids == expected.aux_layer_ids
    assert config.target_model_id == model_id
    # The spec-built config is the downloaded config, field for field.
    assert config == published_dflash_config(model_id)
    with (snapshot / "config.json").open(encoding="utf-8") as config_file:
        raw = json.load(config_file)
    assert raw["dflash_config"]["num_target_layers"] == DFLASH_MODELS[model_id].num_target_layers

    shapes = checkpoint.tensor_shapes()
    assert shapes == expected_checkpoint_shapes(config)
    # Embedding and output projection are intentionally shared with the target.
    assert "embed_tokens.weight" not in shapes
    assert "lm_head.weight" not in shapes


@pytest.mark.parametrize("model_id", CHECKPOINTS)
def test_parallel_proposal_block_geometry(model_id, expect_error):
    config = LagunaDFlashCheckpoint(_snapshot(model_id)).config
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


@pytest.mark.parametrize("model_id", CHECKPOINTS)
def test_context_retention_is_exactly_the_useful_511_row_tail(model_id, expect_error):
    config = LagunaDFlashCheckpoint(_snapshot(model_id)).config
    hidden = torch.arange(600 * 3, dtype=torch.float32).reshape(600, 3)
    positions = torch.arange(900, 1500)
    retained, retained_positions = retain_dflash_context_window(config, hidden, positions)
    assert retained.shape == (511, 3)
    assert retained_positions.tolist() == list(range(989, 1500))
    torch.testing.assert_close(retained, hidden[-511:])

    with expect_error(ValueError, "contiguous"):
        retain_dflash_context_window(config, hidden[:3], torch.tensor([4, 6, 7]))


@pytest.mark.parametrize("model_id", CHECKPOINTS)
@torch.inference_mode()
def test_real_checkpoint_layer_zero_numeric_contract(model_id):
    """Exercise every Laguna DFlash primitive with real BF16 checkpoint data.

    A single layer is sufficient to cover fused-QKV splitting, per-head Q/K
    norms, RoPE, causal SWA, softplus head gates, dense SwiGLU, both residual
    additions, and final RMSNorm while keeping this test well below one second
    on the bringup host.
    """

    expected = EXPECTED[model_id]
    model = LagunaDFlashCheckpoint(_snapshot(model_id)).load_reference(layer_indices=(0,))
    config = model.config
    hidden = config.hidden_size
    num_aux = config.num_aux_hidden_states

    # Integer-derived inputs avoid RNG/version drift in the BF16 fingerprint.
    aux = (((torch.arange(2 * num_aux * hidden) % 97) - 48).float() / 2400).reshape(2, num_aux, hidden)
    query = (((torch.arange(3 * hidden) % 83) - 41).float() / 2100).reshape(3, hidden)
    aux = aux.to(torch.bfloat16)
    query = query.to(torch.bfloat16)
    context_positions = torch.tensor([509, 510])
    query_positions = torch.tensor([511, 512, 513])

    combined = model.combine_aux_hidden_states(aux)
    context_kv = model.precompute_context_kv(combined, context_positions)
    output = model.forward_query_embeddings(query, query_positions, context_kv)

    assert combined.shape == (2, expected.hidden)
    assert context_kv[0].key.shape == (2, 8, 128)
    assert context_kv[0].value.shape == (2, 8, 128)
    assert output.shape == (3, expected.hidden)
    assert combined.dtype == context_kv[0].key.dtype == output.dtype == torch.bfloat16
    assert torch.isfinite(combined).all() and torch.isfinite(output).all()
    # The flattened [tokens, num_aux * hidden] form is the same contract.
    torch.testing.assert_close(model.combine_aux_hidden_states(aux.reshape(2, -1)), combined, rtol=0, atol=0)

    # Independently project and split layer-0 fused QKV to lock row ordering.
    normalized_context = rms_norm(
        combined,
        model.weights["layers.0.input_layernorm.weight"],
        config.rms_norm_eps,
    )
    fused = F.linear(normalized_context, model.weights["layers.0.self_attn.qkv_proj.weight"])
    q, k, v = split_fused_qkv(fused, config)
    assert (q.shape[-1], k.shape[-1], v.shape[-1]) == (expected.sizes[0], 1024, 1024)
    torch.testing.assert_close(torch.cat((q, k, v), dim=-1), fused, rtol=0, atol=0)

    # The gate is positive and one scalar per query head before broadcasting.
    normalized_query = rms_norm(
        query,
        model.weights["layers.0.input_layernorm.weight"],
        config.rms_norm_eps,
    )
    head_gate = F.softplus(F.linear(normalized_query, model.weights["layers.0.self_attn.g_proj.weight"]).float())
    assert head_gate.shape == (3, expected.heads[0])
    assert bool((head_gate > 0).all())

    # BF16 golden values catch changes in aux-slice order, norms, rotation,
    # attention masking/gating, residual order, or SwiGLU semantics.
    torch.testing.assert_close(combined[0, :8].float(), torch.tensor(expected.combined), rtol=0, atol=0)
    torch.testing.assert_close(context_kv[0].key[0, 0, :8].float(), torch.tensor(expected.key), rtol=0, atol=0)
    torch.testing.assert_close(output[0, :8].float(), torch.tensor(expected.output), rtol=0, atol=0)
    torch.testing.assert_close(output.float().norm(), torch.tensor(expected.output_norm), rtol=1e-6, atol=1e-6)

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


@pytest.mark.parametrize("model_id", CHECKPOINTS)
@torch.inference_mode()
def test_real_checkpoint_full_draft_one_round_contract(model_id):
    """Load every published layer and execute the exact anchor+15 proposal."""

    expected = EXPECTED[model_id]
    model = LagunaDFlashCheckpoint(_snapshot(model_id)).load_reference()
    config = model.config
    hidden = config.hidden_size
    num_aux = config.num_aux_hidden_states
    all_layers = tuple(range(expected.layers))
    assert model.layer_indices == all_layers

    aux = (((torch.arange(2 * num_aux * hidden) % 97) - 48).float() / 2400).reshape(2, num_aux, hidden)
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

    assert tuple(contexts) == all_layers
    assert len(layer_outputs) == expected.layers
    assert all(stage.shape == (16, hidden) and stage.dtype == torch.bfloat16 for stage in layer_outputs)
    torch.testing.assert_close(traced_hidden, hidden_states, rtol=0, atol=0)
    assert hidden_states.shape == (16, hidden)
    assert logits.shape == (15, 19)
    assert hidden_states.dtype == logits.dtype == torch.bfloat16
    assert torch.isfinite(hidden_states).all() and torch.isfinite(logits).all()
    torch.testing.assert_close(logits, model.compute_logits(hidden_states[1:16], target_lm_head), rtol=0, atol=0)


def _capture_path(model_id: str) -> Path:
    from models.autoports.poolside_laguna_xs_2_1.tests.dflash_acceptance import default_capture_path
    from models.autoports.poolside_laguna_xs_2_1.tt.model_spec import SUPPORTED_MODELS

    if os.environ.get("LAGUNA_DFLASH_TARGET_CAPTURE") and model_id != MODEL_ID:
        return Path("/nonexistent")  # the override names the selected checkpoint's capture only
    return default_capture_path(SUPPORTED_MODELS[model_id])


# Real-target qualification: opt-in by the presence of a capture from tests/gen_dflash_target_capture.py
# (CPU, layer-streamed; ~5 min for S on the bring-up host).
CAPTURED_CHECKPOINTS = [
    pytest.param(
        model_id,
        id=model_id.split("/")[-1],
        marks=pytest.mark.skipif(
            not (_has_checkpoint(model_id) and _capture_path(model_id).is_file()),
            reason=(
                f"no real-target capture for {model_id}; run TT_LAGUNA_MODEL={model_id} python -m "
                "models.autoports.poolside_laguna_xs_2_1.tests.gen_dflash_target_capture"
            ),
        ),
    )
    for model_id in sorted(DFLASH_MODELS)
]


@pytest.mark.parametrize("model_id", CAPTURED_CHECKPOINTS)
@torch.inference_mode()
def test_draft_proposals_track_real_target_greedy(model_id):
    """The published draft, fed real target states at its published layers, predicts the target.

    The synthetic fingerprints above lock arithmetic but cannot show the draft reads the right
    target layers.  Here every continuation position of the readiness AIME24 sequence is one
    served-style round on real layer-streamed target states.  Measured on 2026-10-01 (99 anchors),
    first-proposal agreement / mean teacher-forced acceptance:

    * S: 0.869 / 2.64 at the published layers; reversed slice order 0.061 / 0.07; layer 47 in
      every slice 0.051 / 0.05.
    * XS (the hardware-qualified draft, as calibration): 0.848 / 2.00; reversed 0.111 / 0.15;
      layer 39 in every slice 0.101 / 0.10.
    """

    from models.autoports.poolside_laguna_xs_2_1.tests.dflash_acceptance import (
        load_target_embedding_and_lm_head,
        teacher_forced_acceptance,
    )

    capture = torch.load(_capture_path(model_id))
    assert capture["model"] == model_id
    # The streamed target reproduces the stored readiness reference before it is trusted. S's
    # reference came from the same fp32 layer-streamed HF code (1.00); XS's from a whole-model
    # run (0.98).
    assert capture["readiness_top1_agreement"] >= 0.95
    reference = LagunaDFlashCheckpoint(_snapshot(model_id)).load_reference()
    embedding, lm_head = load_target_embedding_and_lm_head(model_id)
    prompt_len = int(capture["prompt_len"])
    anchors = range(prompt_len, int(capture["token_ids"].numel()) - 1)
    ids = DFLASH_MODELS[model_id].target_layer_ids

    published = teacher_forced_acceptance(
        reference, capture, ids, target_embedding=embedding, target_lm_head=lm_head, anchors=anchors
    )
    reversed_order = teacher_forced_acceptance(
        reference, capture, tuple(reversed(ids)), target_embedding=embedding, target_lm_head=lm_head, anchors=anchors
    )
    print(
        f"DFLASH_REAL_TARGET {model_id} anchors={len(anchors)} first={published.first_rate:.3f} "
        f"mean_accepted={published.mean_accepted:.3f} reversed_first={reversed_order.first_rate:.3f} "
        f"reversed_mean_accepted={reversed_order.mean_accepted:.3f}"
    )
    assert published.first_rate >= 0.75
    assert published.mean_accepted >= 1.75
    assert reversed_order.first_rate <= 0.25
    assert published.first_rate - reversed_order.first_rate >= 0.5


def test_published_config_from_json_round_trip(tmp_path):
    """``from_json`` of a spec-shaped config.json reproduces ``published_dflash_config`` for both drafts."""

    for model_id, spec in DFLASH_MODELS.items():
        config = published_dflash_config(model_id)
        raw = {
            "attention_bias": False,
            "head_dim": spec.head_dim,
            "hidden_act": "silu",
            "hidden_size": spec.hidden_size,
            "intermediate_size": spec.intermediate_size,
            "max_position_embeddings": spec.max_position_embeddings,
            "num_attention_heads": spec.num_attention_heads,
            "num_hidden_layers": spec.num_draft_layers,
            "num_key_value_heads": spec.num_key_value_heads,
            "rms_norm_eps": 1e-6,
            "rope_theta": 500000.0,
            "sliding_window": spec.sliding_window,
            "vocab_size": spec.vocab_size,
            "layer_types": ["sliding_attention"] * spec.num_draft_layers,
            "gating": "per-head",
            "architectures": ["DFlashLagunaForCausalLM"],
            "draft_vocab_size": spec.vocab_size,
            "torch_dtype": "bfloat16",
            "eagle_aux_hidden_state_layer_ids": list(spec.aux_hidden_state_layer_ids),
            "dflash_config": {
                "block_size": spec.block_size,
                "mask_token_id": spec.mask_token_id,
                "num_target_layers": spec.num_target_layers,
                "target_layer_ids": list(spec.target_layer_ids),
                "causal": True,
            },
            "num_experts": 0,
        }
        path = tmp_path / f"{spec.num_draft_layers}.json"
        path.write_text(json.dumps(raw))
        assert LagunaDFlashConfig.from_json(path) == config
