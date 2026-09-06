# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The published embedding pipeline: tokenizer, prefixes, pooling, Matryoshka, L2.

Requires the network (or a warm HF cache) and the checkpoint.

The headline assertion is the model card's own worked example -- cosine 0.9118 between
"Hello!" and "¡Hola!" under the `passage` prefix. That single number exercises the whole
stack end to end: tokenizer, prefix, all 12 blocks including the six MoE ones, mean
pooling and normalisation. If any of them is wrong it moves.
"""

import pytest
import torch

from models.experimental.nomic_embed_text_v2_moe.common import (
    MODEL_CARD_SENTENCES,
    MODEL_CARD_SIMILARITY,
    max_abs_diff,
    pcc,
)
from models.experimental.nomic_embed_text_v2_moe.reference import pipeline

pytestmark = pytest.mark.needs_weights


def test_tokenizer_identity_and_special_tokens(tokenizer):
    """`AutoTokenizer` is safe here even though `AutoModel` is not.

    The `nomic_bert` model-type mapping would point at a BERT tokenizer, but
    `tokenizer_config.json`'s explicit `tokenizer_class` outranks it. This is the canary on
    that precedence -- if a future transformers release flips it, tokenisation would change
    silently and every embedding with it.
    """
    assert "XLMRoberta" in type(tokenizer).__name__
    assert tokenizer.pad_token_id == 1
    assert tokenizer.bos_token_id == 0
    assert tokenizer.eos_token_id == 2
    assert len(tokenizer) == 250002
    assert tokenizer.model_max_length == pipeline.MAX_SEQ_LENGTH


def test_vocab_size_exceeds_tokenizer_length(config, tokenizer):
    """The embedding table is padded past the tokenizer: 250048 rows vs 250002 tokens.

    `pad_vocab_size_multiple` is 64 and 250002 is not a multiple of it. The extra rows are
    unreachable; the TTNN port must size the table from the checkpoint, not the tokenizer.
    """
    assert config.vocab_size == 250048
    assert config.vocab_size > len(tokenizer)
    assert config.vocab_size % 64 == 0


def test_model_card_similarity(reference_model, tokenizer):
    """The card prints 0.9118 for these two sentences under the `passage` prefix."""
    embeddings = pipeline.encode(reference_model, tokenizer, list(MODEL_CARD_SENTENCES), prompt_name="passage")

    assert embeddings.shape == (2, 768)
    similarity = float(embeddings[0] @ embeddings[1])
    assert abs(similarity - MODEL_CARD_SIMILARITY) < 1e-4, f"got {similarity:.6f}, card says {MODEL_CARD_SIMILARITY}"


def test_pipeline_matches_hf_backbone(reference_model, hf_model, tokenizer):
    texts = ["search_document: the quick brown fox", "search_document: el zorro marrón rápido"]
    ours = pipeline.encode(reference_model, tokenizer, texts)
    theirs = pipeline.encode(hf_model, tokenizer, texts)
    assert pcc(ours, theirs) > 0.9999999
    assert max_abs_diff(ours, theirs) < 1e-5


def test_embeddings_are_unit_norm(reference_model, tokenizer):
    embeddings = pipeline.encode(reference_model, tokenizer, ["one", "two", "three"], prompt_name="query")
    norms = embeddings.norm(dim=-1)
    torch.testing.assert_close(norms, torch.ones_like(norms), rtol=1e-5, atol=1e-5)


def test_task_prefix_changes_the_embedding(reference_model, tokenizer):
    """The prefixes are load-bearing, not decoration -- the model was trained with them."""
    text = ["how tall is the eiffel tower"]
    query = pipeline.encode(reference_model, tokenizer, text, prompt_name="query")
    passage = pipeline.encode(reference_model, tokenizer, text, prompt_name="passage")
    bare = pipeline.encode(reference_model, tokenizer, text, prompt_name=None)

    assert float(query[0] @ passage[0]) < 0.999
    assert float(query[0] @ bare[0]) < 0.999
    assert pipeline.PROMPTS["query"] == "search_query: "
    assert pipeline.PROMPTS["passage"] == "search_document: "


@pytest.mark.parametrize("dim", [768, 512, 256, 128])
def test_matryoshka_truncation(reference_model, tokenizer, dim):
    embeddings = pipeline.encode(reference_model, tokenizer, ["hola mundo"], prompt_name="passage", matryoshka_dim=dim)
    assert embeddings.shape == (1, dim)
    torch.testing.assert_close(embeddings.norm(dim=-1), torch.ones(1), rtol=1e-5, atol=1e-5)


def test_matryoshka_order_is_cosine_invariant(reference_model, tokenizer):
    """The lemma: truncate-then-normalize and normalize-then-truncate differ only in norm.

    They produce vectors of different length (1.0 vs ~0.57 at d=256) but identical
    direction, and the model's declared similarity function is cosine. So the order is a
    free choice for the TTNN port -- worth pinning explicitly rather than leaving as a
    thing someone has to rederive.
    """
    texts = ["search_document: alpha", "search_document: beta"]
    encoded = pipeline.tokenize(tokenizer, texts)
    with torch.no_grad():
        hidden = reference_model(encoded["input_ids"], attention_mask=encoded["attention_mask"])

    pooled = pipeline.mean_pool(hidden, encoded["attention_mask"])
    dim = 256

    truncate_then_normalize = pipeline.l2_normalize(pooled[..., :dim])
    normalize_then_truncate = pipeline.l2_normalize(pooled)[..., :dim]

    # Different norms...
    assert abs(float(normalize_then_truncate.norm(dim=-1)[0]) - 1.0) > 0.05
    torch.testing.assert_close(truncate_then_normalize.norm(dim=-1), torch.ones(2), rtol=1e-5, atol=1e-5)

    # ...but identical direction, so identical cosine.
    a = pipeline.cosine_similarity_matrix(truncate_then_normalize, truncate_then_normalize)
    b = pipeline.cosine_similarity_matrix(normalize_then_truncate, normalize_then_truncate)
    torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)


def test_mean_pool_excludes_padding(reference_model, tokenizer):
    """Pooling must be mask-weighted; `<pad>` carries a non-zero embedding.

    A ragged batch is the real test: the short text's embedding must not depend on how long
    its batch-mates are.
    """
    short = ["hello"]
    ragged = ["hello", "a considerably longer sentence that forces the batch to pad the first one"]

    alone = pipeline.encode(reference_model, tokenizer, short, prompt_name="passage")
    batched = pipeline.encode(reference_model, tokenizer, ragged, prompt_name="passage")

    assert float(alone[0] @ batched[0]) > 0.9999
    assert max_abs_diff(alone[0], batched[0]) < 1e-3


def test_mean_pool_is_not_cls_pooling(reference_model, tokenizer):
    """`1_Pooling/config.json` sets mean pooling and disables CLS. Confirm they differ."""
    texts = ["search_document: a sentence with several distinct tokens in it"]
    encoded = pipeline.tokenize(tokenizer, texts)
    with torch.no_grad():
        hidden = reference_model(encoded["input_ids"], attention_mask=encoded["attention_mask"])

    mean = pipeline.l2_normalize(pipeline.mean_pool(hidden, encoded["attention_mask"]))
    cls = pipeline.l2_normalize(hidden[:, 0])
    assert float(mean[0] @ cls[0]) < 0.999


@pytest.mark.parametrize("batch,seqlen", [(1, 1), (1, 4), (2, 8), (3, 17), (1, 512)])
def test_runs_on_small_inputs(config, reference_model, batch, seqlen):
    """Issue #54917's acceptance criterion, on real weights across the shape range."""
    from models.experimental.nomic_embed_text_v2_moe.common import random_input_ids

    input_ids, attention_mask = random_input_ids(batch, seqlen, config, seed=batch * 1000 + seqlen)
    with torch.no_grad():
        out = reference_model(input_ids, attention_mask=attention_mask)

    assert out.shape == (batch, seqlen, config.hidden_size)
    assert torch.isfinite(out).all()

    pooled = pipeline.pool_and_normalize(out, attention_mask)
    assert pooled.shape == (batch, config.hidden_size)
    assert torch.isfinite(pooled).all()


def test_multilingual_pairs_are_closer_than_unrelated_ones(reference_model, tokenizer):
    """A sanity check with semantic content: translation pairs should beat unrelated text."""
    texts = ["the cat sits on the mat", "el gato se sienta en la alfombra", "quarterly revenue exceeded forecasts"]
    embeddings = pipeline.encode(reference_model, tokenizer, texts, prompt_name="passage")

    translation = float(embeddings[0] @ embeddings[1])
    unrelated = float(embeddings[0] @ embeddings[2])
    assert translation > unrelated, f"translation pair {translation:.4f} !> unrelated {unrelated:.4f}"
