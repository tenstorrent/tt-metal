# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests of the vendored reference against the upstream HF implementation.

Two levels, both exercised here because a divergence at either one invalidates the other:

  backbone   input_ids -> last_hidden_state, compared layer by layer against upstream.
             Near perfect is the bar (PCC ~ 1.0, max-abs 0.0); missing it is a regression, and
             the thresholds are a sanity check rather than a knob.
  pipeline   text -> embedding, the full stack: tokenizer, task prefix, all 12 blocks,
             mean pool, Matryoshka truncation, L2 normalize. test_model_card_similarity is the
             headline, reproducing the published number end to end in a single scalar.

Needs the checkpoint and a warm HF cache or network.
"""

import pytest
import torch

from models.common.metrics import compute_max_abs_error, compute_pcc
from models.experimental.nomic_embed_text_v2_moe.common import (
    MODEL_CARD,
    PARITY,
    TOKENIZER,
    capture_hidden_states,
    layer_ladder_paths,
    random_input_ids,
)
from models.experimental.nomic_embed_text_v2_moe.reference import embedding, postprocessing, preprocessing
from models.experimental.nomic_embed_text_v2_moe.reference.hf_reference import (
    RemoteCodeResolutionError,
    assert_resolved_from_remote_code,
    hf_layer_ladder,
    hf_forward,
)
from models.experimental.nomic_embed_text_v2_moe.reference.preprocessing import NomicPromptPrefix

pytestmark = pytest.mark.needs_weights

MATRYOSHKA_DIMS = [768, 512, 256, 128]


def reference_ladder(model, config, input_ids, attention_mask) -> dict[str, torch.Tensor]:
    paths = layer_ladder_paths(config.num_hidden_layers)
    captures, handles = capture_hidden_states(model, paths)
    try:
        with torch.no_grad():
            model(input_ids, attention_mask=attention_mask)
    finally:
        for handle in handles:
            handle.remove()
    return captures


# Oracle integrity. Everything below compares against hf_model, so these run first.


def test_hf_model_came_from_remote_code(hf_model):
    """A bare AutoModel.from_pretrained resolves to the native v1.5 class, discards the expert
    weights and does not raise. If this fires, the golden reference has been downgraded."""
    assert type(hf_model).__module__.startswith("transformers_modules")
    assert [key for key in hf_model.state_dict() if "experts" in key]


def test_remote_code_guard_rejects_a_native_class(expect_error):
    class Impostor:
        pass

    with expect_error(RemoteCodeResolutionError, "native transformers"):
        assert_resolved_from_remote_code(Impostor(), "AutoModel")


def test_tokenizer_identity_and_special_tokens(tokenizer):
    """AutoTokenizer is safe even though AutoModel is not: tokenizer_config.json's explicit
    tokenizer_class outranks the nomic_bert model-type mapping. This is the canary on that."""
    assert "XLMRoberta" in type(tokenizer).__name__
    assert tokenizer.pad_token_id == TOKENIZER.pad_token_id
    assert tokenizer.bos_token_id == TOKENIZER.bos_token_id
    assert tokenizer.eos_token_id == TOKENIZER.eos_token_id
    assert len(tokenizer) == TOKENIZER.length
    assert tokenizer.model_max_length == preprocessing.MAX_SEQ_LENGTH


def test_vocab_size_exceeds_tokenizer_length(config, tokenizer):
    """The embedding table is padded past the tokenizer, so its trailing rows are unreachable.
    Size it from the checkpoint, not the tokenizer."""
    assert config.vocab_size > len(tokenizer)


# Backbone parity: input_ids -> last_hidden_state.


@pytest.mark.parametrize(
    "batch,seqlen,pad_lengths",
    [
        (1, 8, None),
        (2, 24, [0, 7]),
        (3, 17, [0, 3, 11]),
    ],
)
def test_end_to_end_parity(config, reference_model, hf_model, batch, seqlen, pad_lengths):
    input_ids, attention_mask = random_input_ids(batch, seqlen, config, seed=seqlen, pad_lengths=pad_lengths)

    with torch.no_grad():
        ours = reference_model(input_ids, attention_mask=attention_mask)
    theirs = hf_forward(hf_model, input_ids, attention_mask)

    assert ours.shape == theirs.shape
    assert compute_pcc(ours, theirs) > PARITY.pcc
    assert compute_max_abs_error(ours, theirs) < PARITY.max_abs


def test_layer_ladder_parity(config, reference_model, hf_model):
    """Per-layer comparison localises the first divergence to a single block."""
    input_ids, attention_mask = random_input_ids(2, 24, config, seed=0, pad_lengths=[0, 7])
    paths = layer_ladder_paths(config.num_hidden_layers)

    ours = reference_ladder(reference_model, config, input_ids, attention_mask)
    theirs = hf_layer_ladder(hf_model, input_ids, attention_mask)

    assert len(paths) == config.num_ladder_points

    failures = [
        f"{path}: pcc={compute_pcc(ours[path], theirs[path]):.9f} max_abs={compute_max_abs_error(ours[path], theirs[path]):.3e}"
        for path in paths
        if compute_pcc(ours[path], theirs[path]) <= PARITY.pcc
        or compute_max_abs_error(ours[path], theirs[path]) >= PARITY.max_abs
    ]
    assert not failures, "first divergence at " + failures[0]


def test_moe_layers_carry_the_largest_activations(config, reference_model):
    """MoE blocks produce larger activations than the dense ones, so a relative tolerance
    calibrated on layer 0 would be far too loose from layer 1 onwards."""
    input_ids, attention_mask = random_input_ids(2, 24, config, seed=0)
    captures = reference_ladder(reference_model, config, input_ids, attention_mask)

    absmax = {path: float(tensor.abs().max()) for path, tensor in captures.items()}
    moe_peak = max(absmax[f"encoder.layers.{i}"] for i in config.moe_layers)

    assert moe_peak > absmax["encoder.layers.0"]
    assert all(torch.isfinite(tensor).all() for tensor in captures.values())


# Negative controls. Each implements the plausible wrong choice and asserts it differs, because
# for this architecture none of them raises: the wrong expert view typechecks, and the wrong
# bias placement scores PCC 0.9999991.


def moe_layer(config, reference_model):
    """The first MoE block's mlp, plus a fixed input and its routing.

    Args:
        config: NomicMoEConfig, read for hidden_size and moe_layers.
        reference_model: The loaded reference model.

    Returns:
        tuple: (moe, hidden, top_weights, top_experts) for the first MoE layer.
    """
    moe = reference_model.encoder.layers[config.moe_layers[0]].mlp
    torch.manual_seed(0)
    hidden = torch.randn(2, 24, config.hidden_size)
    _weights, top_weights, top_experts = moe.router(hidden)
    return moe, hidden, top_weights, top_experts


def test_top_two_weights_are_not_renormalized(config, reference_model):
    """Renormalizing the top-2 weights is what Mixtral and Switch do, and is wrong here.

    moe_normalize_expert_weights is false, so the two weights are used as the softmax produced
    them and sum to less than 1, attenuating the MoE branch against the residual.
    """
    moe, hidden, top_weights, top_experts = moe_layer(config, reference_model)

    assert (top_weights.sum(dim=-1) < 0.999).all(), "top-2 weights unexpectedly sum to 1"

    with torch.no_grad():
        correct = moe.experts(hidden, top_weights, top_experts)
        renormalized = moe.experts(hidden, top_weights / top_weights.sum(dim=-1, keepdim=True), top_experts)

    assert compute_max_abs_error(correct, renormalized) > 1.0


def test_shared_bias_is_added_once_after_the_weighted_sum(config, reference_model):
    """Folding the shared bias into the per-expert loop is invisible to PCC.

    Doing so scales the bias by the routed-weight sum, an offset of (sum(w) - 1) * bias. That
    offset is nearly constant and PCC mean-centres, so this class of bug must be gated on
    max-abs instead.
    """
    moe, hidden, top_weights, top_experts = moe_layer(config, reference_model)

    with torch.no_grad():
        correct = moe.experts(hidden, top_weights, top_experts)
        routed_sum = top_weights.sum(dim=-1).reshape(hidden.shape[0], hidden.shape[1], 1)
        bias_inside_loop = correct + (routed_sum - 1.0) * moe.experts.bias

    assert compute_pcc(correct, bias_inside_loop) > 0.999, "PCC unexpectedly separated these"
    assert compute_max_abs_error(correct, bias_inside_loop) > 1e-2


def test_w2_transposed_view_typechecks_but_is_garbage(config, reference_model):
    """Viewing w2 as (E, H, F) instead of (E, F, H) succeeds and produces noise.

    The element count is symmetric in F and H, so the view is legal and every downstream matmul
    typechecks. Nothing raises; only the numbers say it is wrong.
    """
    experts = reference_model.encoder.layers[config.moe_layers[0]].mlp.experts.mlp
    num_experts, ffn, hidden_size = experts.expert_shape

    torch.manual_seed(0)
    x = torch.randn(8, ffn)
    with torch.no_grad():
        correct = x @ experts.w2.view(num_experts, ffn, hidden_size)[0]
        transposed = x @ experts.w2.view(num_experts, hidden_size, ffn)[0].T

    assert correct.shape == transposed.shape == (8, hidden_size)
    assert abs(compute_pcc(correct, transposed)) < 0.2


def test_dense_forward_matches_the_ragged_loop(config, reference_model):
    """The dense all-experts formulation must equal upstream's ragged gather loop.

    dense_forward is what the TTNN port implements, because the loop's nonzero/index_add_ are
    data-dependent and have no device equivalent. Nothing else checks the two agree, so a
    divergence here would reach the port as a silent wrong answer.
    """
    moe = reference_model.encoder.layers[config.moe_layers[0]].mlp
    torch.manual_seed(0)
    hidden = torch.randn(2, 24, config.hidden_size)

    _weights, top_weights, top_experts = moe.router(hidden)
    ragged = moe.experts(hidden, top_weights, top_experts)
    dense = moe.experts.dense_forward(hidden, moe.router.dense_weights(top_weights, top_experts))

    assert dense.shape == ragged.shape
    assert compute_pcc(dense, ragged) > PARITY.pcc
    assert compute_max_abs_error(dense, ragged) < PARITY.max_abs


@pytest.mark.parametrize("batch,seqlen", [(1, 1), (1, 4), (2, 8), (3, 17), (1, 512)])
def test_runs_on_small_inputs(config, reference_model, batch, seqlen):
    input_ids, attention_mask = random_input_ids(batch, seqlen, config, seed=batch * 1000 + seqlen)
    with torch.no_grad():
        out = reference_model(input_ids, attention_mask=attention_mask)

    assert out.shape == (batch, seqlen, config.hidden_size)
    assert torch.isfinite(out).all()

    embeddings = postprocessing.l2_normalize(postprocessing.mean_pool(out, attention_mask))
    assert embeddings.shape == (batch, config.hidden_size)
    assert torch.isfinite(embeddings).all()


# Upstream behaviours the reference deliberately does not reproduce.


def test_upstream_requires_an_attention_mask_but_the_reference_does_not(config, reference_model, hf_model):
    """Upstream calls get_extended_attention_mask unconditionally and raises on None."""
    input_ids, attention_mask = random_input_ids(1, 8, config, seed=3)

    with torch.no_grad():
        ours = reference_model(input_ids)
    assert ours.shape == (1, 8, config.hidden_size)

    raised = None
    try:
        with torch.no_grad():
            hf_model(input_ids=input_ids, attention_mask=None)
    except Exception as exc:  # noqa: BLE001 - the exception type is what is under observation
        raised = exc
    assert raised is not None, "upstream unexpectedly accepted attention_mask=None"

    with torch.no_grad():
        ours_masked = reference_model(input_ids, attention_mask=attention_mask)
    assert compute_pcc(ours_masked, hf_forward(hf_model, input_ids, attention_mask)) > PARITY.pcc


def test_upstream_matryoshka_dim_slices_the_sequence_axis(config, hf_model):
    """Upstream slices sequence_output[:, :matryoshka_dim], dropping tokens rather than
    features. postprocessing.py truncates after pooling instead."""
    seqlen = 10
    input_ids, attention_mask = random_input_ids(2, seqlen, config, seed=5)
    with torch.no_grad():
        out = hf_model(input_ids=input_ids, attention_mask=attention_mask, matryoshka_dim=256).last_hidden_state

    assert out.shape == (2, seqlen, config.hidden_size)


# Full pipeline parity: text -> embedding.


def test_model_card_similarity(config, reference_model, tokenizer):
    embeddings = embedding.encode(
        reference_model, tokenizer, list(MODEL_CARD.sentences), prompt_prefix=NomicPromptPrefix.PASSAGE
    )

    assert embeddings.shape == (len(MODEL_CARD.sentences), config.hidden_size)
    similarity = float(embeddings[0] @ embeddings[1])
    assert abs(similarity - MODEL_CARD.cosine_similarity) < MODEL_CARD.tolerance, f"got {similarity:.6f}"


def test_pipeline_matches_hf_backbone(reference_model, hf_model, tokenizer):
    """Identical pipeline on both sides, so any difference is attributable to the backbone."""
    texts = ["search_document: the quick brown fox", "search_document: el zorro marron rapido"]
    ours = embedding.encode(reference_model, tokenizer, texts)
    theirs = embedding.encode(hf_model, tokenizer, texts)

    assert compute_pcc(ours, theirs) > PARITY.pcc
    assert compute_max_abs_error(ours, theirs) < 1e-5


def test_embeddings_are_unit_norm(reference_model, tokenizer):
    embeddings = embedding.encode(
        reference_model, tokenizer, ["one", "two", "three"], prompt_prefix=NomicPromptPrefix.QUERY
    )
    norms = embeddings.norm(dim=-1)
    torch.testing.assert_close(norms, torch.ones_like(norms), rtol=1e-5, atol=1e-5)


def test_task_prefix_changes_the_embedding(reference_model, tokenizer):
    """The prefixes are trained-in, not decoration."""
    text = ["how tall is the eiffel tower"]
    query = embedding.encode(reference_model, tokenizer, text, prompt_prefix=NomicPromptPrefix.QUERY)
    passage = embedding.encode(reference_model, tokenizer, text, prompt_prefix=NomicPromptPrefix.PASSAGE)
    bare = embedding.encode(reference_model, tokenizer, text, prompt_prefix=None)

    assert float(query[0] @ passage[0]) < 0.999
    assert float(query[0] @ bare[0]) < 0.999


def test_per_query_prefixes_match_encoding_each_query_alone(reference_model, tokenizer):
    """Asymmetric retrieval: a search query and its candidate documents, embedded in one batch."""
    queries = ["blackhole", "Tenstorrent's architecture"]
    prefixes = [NomicPromptPrefix.QUERY, NomicPromptPrefix.PASSAGE]

    together = embedding.encode(reference_model, tokenizer, queries, prompt_prefix=prefixes)
    separate = [
        embedding.encode(reference_model, tokenizer, [query], prompt_prefix=prefix)[0]
        for query, prefix in zip(queries, prefixes)
    ]

    torch.testing.assert_close(together, torch.stack(separate), rtol=1e-5, atol=1e-5)


def test_prompt_prefix_rejects_a_task_name_string(expect_error):
    """The MTEB task names are enum member names, not values, so a bare string is not one."""
    with expect_error(ValueError, "not a valid NomicPromptPrefix"):
        preprocessing.apply_prompt(["hola"], "passage")


def test_per_query_prefixes_must_cover_every_query(expect_error):
    with expect_error(ValueError, "1 prompt prefixes for 2 queries"):
        preprocessing.apply_prompt(["hola", "mundo"], [NomicPromptPrefix.QUERY])


@pytest.mark.parametrize("dim", MATRYOSHKA_DIMS)
def test_matryoshka_truncation(reference_model, tokenizer, dim):
    embeddings = embedding.encode(
        reference_model, tokenizer, ["hola mundo"], prompt_prefix=NomicPromptPrefix.PASSAGE, matryoshka_dim=dim
    )

    assert embeddings.shape == (1, dim)
    torch.testing.assert_close(embeddings.norm(dim=-1), torch.ones(1), rtol=1e-5, atol=1e-5)


def test_matryoshka_order_is_cosine_invariant(reference_model, tokenizer):
    """Truncate-then-normalize and normalize-then-truncate differ in norm but not direction, so
    the order is a free choice for the TTNN port."""
    texts = ["search_document: alpha", "search_document: beta"]
    encoded = preprocessing.tokenize(tokenizer, texts)
    with torch.no_grad():
        hidden = reference_model(encoded["input_ids"], attention_mask=encoded["attention_mask"])

    pooled = postprocessing.mean_pool(hidden, encoded["attention_mask"])
    dim = 256

    truncate_first = postprocessing.l2_normalize(pooled[..., :dim])
    normalize_first = postprocessing.l2_normalize(pooled)[..., :dim]

    assert abs(float(normalize_first.norm(dim=-1)[0]) - 1.0) > 0.05
    torch.testing.assert_close(truncate_first.norm(dim=-1), torch.ones(len(texts)), rtol=1e-5, atol=1e-5)

    torch.testing.assert_close(
        postprocessing.cosine_similarity_matrix(truncate_first, truncate_first),
        postprocessing.cosine_similarity_matrix(normalize_first, normalize_first),
        rtol=1e-5,
        atol=1e-5,
    )


def test_mean_pool_excludes_padding(reference_model, tokenizer):
    """A short text's embedding must not depend on how long its batch-mates are."""
    short = ["hello"]
    ragged = ["hello", "a considerably longer sentence that forces the batch to pad the first one"]

    alone = embedding.encode(reference_model, tokenizer, short, prompt_prefix=NomicPromptPrefix.PASSAGE)
    batched = embedding.encode(reference_model, tokenizer, ragged, prompt_prefix=NomicPromptPrefix.PASSAGE)

    assert float(alone[0] @ batched[0]) > 0.9999
    assert compute_max_abs_error(alone[0], batched[0]) < 1e-3


def test_mean_pool_is_not_cls_pooling(reference_model, tokenizer):
    """1_Pooling/config.json sets mean pooling and disables CLS."""
    encoded = preprocessing.tokenize(tokenizer, ["search_document: a sentence with several distinct tokens in it"])
    with torch.no_grad():
        hidden = reference_model(encoded["input_ids"], attention_mask=encoded["attention_mask"])

    mean = postprocessing.l2_normalize(postprocessing.mean_pool(hidden, encoded["attention_mask"]))
    cls = postprocessing.l2_normalize(hidden[:, 0])
    assert float(mean[0] @ cls[0]) < 0.999


def test_multilingual_pairs_are_closer_than_unrelated_ones(reference_model, tokenizer):
    texts = ["the cat sits on the mat", "el gato se sienta en la alfombra", "quarterly revenue exceeded forecasts"]
    embeddings = embedding.encode(reference_model, tokenizer, texts, prompt_prefix=NomicPromptPrefix.PASSAGE)

    translation = float(embeddings[0] @ embeddings[1])
    unrelated = float(embeddings[0] @ embeddings[2])
    assert translation > unrelated, f"translation {translation:.4f} not above unrelated {unrelated:.4f}"
