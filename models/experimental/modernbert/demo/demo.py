# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""ModernBERT demos on Tenstorrent hardware.

Two demos, both comparing TTNN against the HuggingFace reference side by side:

  test_modernbert_mlm_demo        fills in masked tokens
  test_modernbert_embedding_demo  semantic similarity over sentence embeddings

Run:
    pytest --disable-warnings models/experimental/modernbert/demo/demo.py
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.modernbert.common import load_config, load_tokenizer, load_torch_mlm_model, load_torch_model
from models.experimental.modernbert.reference.modernbert import ModernBertForMaskedLM, ModernBertModel
from models.experimental.modernbert.tt.modernbert_head import TtnnModernBertLMHead
from models.experimental.modernbert.tt.modernbert_model import TtnnModernBertModel
from models.experimental.modernbert.tt.weights import prepare_mlm_weights, prepare_weights

SEQ_LEN = 256

MASKED_SENTENCES = [
    "The capital of France is [MASK].",
    "Tenstorrent builds [MASK] accelerators for machine learning.",
    "Water freezes at zero degrees [MASK].",
    "Python is a popular programming [MASK].",
]

SIMILARITY_SENTENCES = [
    "A man is playing a guitar on stage.",
    "A musician performs with his guitar in front of an audience.",
    "The stock market closed lower on Tuesday.",
    "Equity indices finished the session down on Tuesday.",
]


def _pad_to(encoded, seq_len, pad_id):
    """Right-pad a tokenizer batch to the fixed sequence length the TTNN model is built for.

    The tokenizer has already padded every row to the longest sentence in the batch,
    so its attention mask - not the row width - is what marks real tokens.
    """
    src_ids, src_mask = encoded["input_ids"], encoded["attention_mask"]
    n = min(src_ids.shape[1], seq_len)
    ids = torch.full((src_ids.shape[0], seq_len), pad_id, dtype=torch.long)
    mask = torch.zeros((src_ids.shape[0], seq_len), dtype=torch.long)
    ids[:, :n] = src_ids[:, :n]
    mask[:, :n] = src_mask[:, :n]
    return ids, mask


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_modernbert_mlm_demo(device):
    """Fill masked tokens and show TTNN predictions next to the reference."""
    config = load_config()
    tokenizer = load_tokenizer()
    hf = load_torch_mlm_model()
    ref = ModernBertForMaskedLM(config)
    ref.load_state_dict(hf.state_dict(), strict=True)
    ref.eval()

    encoded = tokenizer(MASKED_SENTENCES, return_tensors="pt", padding=True)
    ids, attention_mask = _pad_to(encoded, SEQ_LEN, config.pad_token_id)

    params = prepare_mlm_weights(ref, device)
    encoder = TtnnModernBertModel(params["model"], config, device, SEQ_LEN, attention_mask=attention_mask)
    head = TtnnModernBertLMHead(params, config)

    tt_ids = ttnn.from_torch(ids.to(torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    tt_logits = ttnn.to_torch(head(encoder(tt_ids))).float()

    with torch.no_grad():
        hf_logits = hf(input_ids=ids, attention_mask=attention_mask).logits
    tt_logits = tt_logits.reshape(hf_logits.shape)

    logger.info("ModernBERT masked-token predictions (TTNN vs HuggingFace)")
    agree = total = 0
    for row, sentence in enumerate(MASKED_SENTENCES):
        pos = (ids[row] == tokenizer.mask_token_id).nonzero()
        if pos.numel() == 0:
            continue
        p = int(pos[0, 0])
        tt_tok = tokenizer.decode([int(tt_logits[row, p].argmax())]).strip()
        hf_tok = tokenizer.decode([int(hf_logits[row, p].argmax())]).strip()
        total += 1
        agree += tt_tok == hf_tok
        logger.info(f"  {sentence}")
        logger.info(f"      ttnn -> {tt_tok!r}   reference -> {hf_tok!r}")

    logger.info(f"top-1 agreement: {agree}/{total}")
    assert total > 0, "no masked positions found"


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_modernbert_embedding_demo(device):
    """Downstream sanity check: mean-pooled embedding cosine similarity.

    Sentences 0/1 and 2/3 are paraphrase pairs; the cross pairs are unrelated.

    Caveat worth stating plainly: base ModernBERT is a masked-LM checkpoint, not a
    fine-tuned sentence-embedding model, so raw mean-pooled cosine similarities
    cluster high (~0.88-0.97) and separate paraphrases from unrelated text only
    weakly. Producing well-separated embeddings needs a fine-tune such as
    sentence-transformers.

    The claim under test here is therefore agreement with the reference, not
    embedding quality: the TTNN similarity matrix must reproduce the reference's
    to within a small delta, and both must rank the paraphrase pairs above the
    unrelated ones.
    """
    config = load_config()
    tokenizer = load_tokenizer()
    hf = load_torch_model()
    ref = ModernBertModel(config)
    ref.load_state_dict(hf.state_dict(), strict=True)
    ref.eval()

    encoded = tokenizer(SIMILARITY_SENTENCES, return_tensors="pt", padding=True)
    ids, attention_mask = _pad_to(encoded, SEQ_LEN, config.pad_token_id)

    params = prepare_weights(ref, device)
    model = TtnnModernBertModel(params, config, device, SEQ_LEN, attention_mask=attention_mask)
    tt_ids = ttnn.from_torch(ids.to(torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    tt_hidden = ttnn.to_torch(model(tt_ids)).float()

    with torch.no_grad():
        hf_hidden = hf(input_ids=ids, attention_mask=attention_mask).last_hidden_state
    tt_hidden = tt_hidden.reshape(hf_hidden.shape)

    def mean_pool(h):
        m = attention_mask.unsqueeze(-1).float()
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
        return torch.nn.functional.normalize(pooled, dim=-1)

    tt_emb, hf_emb = mean_pool(tt_hidden), mean_pool(hf_hidden.float())
    tt_sim = tt_emb @ tt_emb.T
    hf_sim = hf_emb @ hf_emb.T

    logger.info("Cosine similarity (TTNN / reference)")
    for i in range(len(SIMILARITY_SENTENCES)):
        row = "  ".join(f"{tt_sim[i, j]:.3f}/{hf_sim[i, j]:.3f}" for j in range(len(SIMILARITY_SENTENCES)))
        logger.info(f"  s{i}: {row}")

    # paraphrase pairs must score above unrelated pairs, in both implementations
    for sim, name in ((hf_sim, "reference"), (tt_sim, "ttnn")):
        assert sim[0, 1] > sim[0, 2], f"{name}: paraphrase pair 0-1 not above unrelated pair 0-2"
        assert sim[2, 3] > sim[1, 2], f"{name}: paraphrase pair 2-3 not above unrelated pair 1-2"

    # A per-element deviation of d can close a margin by at most 2d, so bounding the
    # delta at half the reference's margin is what makes the TTNN ranking above a
    # consequence of numerical agreement rather than a coincidence.
    ref_margin = min((hf_sim[0, 1] - hf_sim[0, 2]).item(), (hf_sim[2, 3] - hf_sim[1, 2]).item())
    delta = (tt_sim - hf_sim).abs().max().item()
    logger.info(f"max |ttnn - reference| similarity delta: {delta:.4f} (reference margin {ref_margin:.4f})")
    assert delta < ref_margin / 2, f"similarity matrix deviates from reference by {delta:.4f}"
