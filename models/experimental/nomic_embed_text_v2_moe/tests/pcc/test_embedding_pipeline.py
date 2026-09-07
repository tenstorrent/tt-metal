# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Embedding pipeline: tokenizer, task prefixes, pooling, Matryoshka truncation, L2 normalize.

test_model_card_similarity is the headline check. It exercises the whole stack in one number:
tokenizer, prefix, all blocks including the MoE ones, pooling and normalization.

Needs the checkpoint and a warm HF cache or network.
"""

import pytest
import torch

from models.experimental.nomic_embed_text_v2_moe.common import (
    BOS_TOKEN_ID,
    EOS_TOKEN_ID,
    MODEL_CARD_SENTENCES,
    MODEL_CARD_SIMILARITY,
    MODEL_CARD_TOLERANCE,
    PAD_TOKEN_ID,
    PARITY_PCC,
    TOKENIZER_LENGTH,
    max_abs_diff,
    pcc,
    random_input_ids,
)
from models.experimental.nomic_embed_text_v2_moe.reference import pipeline

pytestmark = pytest.mark.needs_weights

MATRYOSHKA_DIMS = [768, 512, 256, 128]


def test_tokenizer_identity_and_special_tokens(tokenizer):
    """AutoTokenizer is safe even though AutoModel is not: tokenizer_config.json's explicit
    tokenizer_class outranks the nomic_bert model-type mapping. This is the canary on that."""
    assert "XLMRoberta" in type(tokenizer).__name__
    assert tokenizer.pad_token_id == PAD_TOKEN_ID
    assert tokenizer.bos_token_id == BOS_TOKEN_ID
    assert tokenizer.eos_token_id == EOS_TOKEN_ID
    assert len(tokenizer) == TOKENIZER_LENGTH
    assert tokenizer.model_max_length == pipeline.MAX_SEQ_LENGTH


def test_vocab_size_exceeds_tokenizer_length(config, tokenizer):
    """The embedding table is padded past the tokenizer, so its trailing rows are unreachable.
    Size it from the checkpoint, not the tokenizer."""
    assert config.vocab_size > len(tokenizer)


def test_model_card_similarity(config, reference_model, tokenizer):
    embeddings = pipeline.encode(reference_model, tokenizer, list(MODEL_CARD_SENTENCES), prompt_name="passage")

    assert embeddings.shape == (len(MODEL_CARD_SENTENCES), config.hidden_size)
    similarity = float(embeddings[0] @ embeddings[1])
    assert abs(similarity - MODEL_CARD_SIMILARITY) < MODEL_CARD_TOLERANCE, f"got {similarity:.6f}"


def test_pipeline_matches_hf_backbone(reference_model, hf_model, tokenizer):
    texts = ["search_document: the quick brown fox", "search_document: el zorro marrón rápido"]
    ours = pipeline.encode(reference_model, tokenizer, texts)
    theirs = pipeline.encode(hf_model, tokenizer, texts)

    assert pcc(ours, theirs) > PARITY_PCC
    assert max_abs_diff(ours, theirs) < 1e-5


def test_embeddings_are_unit_norm(reference_model, tokenizer):
    embeddings = pipeline.encode(reference_model, tokenizer, ["one", "two", "three"], prompt_name="query")
    norms = embeddings.norm(dim=-1)
    torch.testing.assert_close(norms, torch.ones_like(norms), rtol=1e-5, atol=1e-5)


def test_task_prefix_changes_the_embedding(reference_model, tokenizer):
    """The prefixes are trained-in, not decoration."""
    text = ["how tall is the eiffel tower"]
    query = pipeline.encode(reference_model, tokenizer, text, prompt_name="query")
    passage = pipeline.encode(reference_model, tokenizer, text, prompt_name="passage")
    bare = pipeline.encode(reference_model, tokenizer, text, prompt_name=None)

    assert float(query[0] @ passage[0]) < 0.999
    assert float(query[0] @ bare[0]) < 0.999


@pytest.mark.parametrize("dim", MATRYOSHKA_DIMS)
def test_matryoshka_truncation(reference_model, tokenizer, dim):
    embeddings = pipeline.encode(reference_model, tokenizer, ["hola mundo"], prompt_name="passage", matryoshka_dim=dim)

    assert embeddings.shape == (1, dim)
    torch.testing.assert_close(embeddings.norm(dim=-1), torch.ones(1), rtol=1e-5, atol=1e-5)


def test_matryoshka_order_is_cosine_invariant(reference_model, tokenizer):
    """Truncate-then-normalize and normalize-then-truncate differ in norm but not direction, so
    the order is a free choice for the TTNN port."""
    texts = ["search_document: alpha", "search_document: beta"]
    encoded = pipeline.tokenize(tokenizer, texts)
    with torch.no_grad():
        hidden = reference_model(encoded["input_ids"], attention_mask=encoded["attention_mask"])

    pooled = pipeline.mean_pool(hidden, encoded["attention_mask"])
    dim = 256

    truncate_first = pipeline.l2_normalize(pooled[..., :dim])
    normalize_first = pipeline.l2_normalize(pooled)[..., :dim]

    assert abs(float(normalize_first.norm(dim=-1)[0]) - 1.0) > 0.05
    torch.testing.assert_close(truncate_first.norm(dim=-1), torch.ones(len(texts)), rtol=1e-5, atol=1e-5)

    torch.testing.assert_close(
        pipeline.cosine_similarity_matrix(truncate_first, truncate_first),
        pipeline.cosine_similarity_matrix(normalize_first, normalize_first),
        rtol=1e-5,
        atol=1e-5,
    )


def test_mean_pool_excludes_padding(reference_model, tokenizer):
    """A short text's embedding must not depend on how long its batch-mates are."""
    short = ["hello"]
    ragged = ["hello", "a considerably longer sentence that forces the batch to pad the first one"]

    alone = pipeline.encode(reference_model, tokenizer, short, prompt_name="passage")
    batched = pipeline.encode(reference_model, tokenizer, ragged, prompt_name="passage")

    assert float(alone[0] @ batched[0]) > 0.9999
    assert max_abs_diff(alone[0], batched[0]) < 1e-3


def test_mean_pool_is_not_cls_pooling(reference_model, tokenizer):
    """1_Pooling/config.json sets mean pooling and disables CLS."""
    encoded = pipeline.tokenize(tokenizer, ["search_document: a sentence with several distinct tokens in it"])
    with torch.no_grad():
        hidden = reference_model(encoded["input_ids"], attention_mask=encoded["attention_mask"])

    mean = pipeline.l2_normalize(pipeline.mean_pool(hidden, encoded["attention_mask"]))
    cls = pipeline.l2_normalize(hidden[:, 0])
    assert float(mean[0] @ cls[0]) < 0.999


@pytest.mark.parametrize("batch,seqlen", [(1, 1), (1, 4), (2, 8), (3, 17), (1, 512)])
def test_runs_on_small_inputs(config, reference_model, batch, seqlen):
    input_ids, attention_mask = random_input_ids(batch, seqlen, config, seed=batch * 1000 + seqlen)
    with torch.no_grad():
        out = reference_model(input_ids, attention_mask=attention_mask)

    assert out.shape == (batch, seqlen, config.hidden_size)
    assert torch.isfinite(out).all()

    pooled = pipeline.pool_and_normalize(out, attention_mask)
    assert pooled.shape == (batch, config.hidden_size)
    assert torch.isfinite(pooled).all()


def test_multilingual_pairs_are_closer_than_unrelated_ones(reference_model, tokenizer):
    texts = ["the cat sits on the mat", "el gato se sienta en la alfombra", "quarterly revenue exceeded forecasts"]
    embeddings = pipeline.encode(reference_model, tokenizer, texts, prompt_name="passage")

    translation = float(embeddings[0] @ embeddings[1])
    unrelated = float(embeddings[0] @ embeddings[2])
    assert translation > unrelated, f"translation {translation:.4f} not above unrelated {unrelated:.4f}"
