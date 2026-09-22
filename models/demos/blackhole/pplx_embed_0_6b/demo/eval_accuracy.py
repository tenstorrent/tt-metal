# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Accuracy evaluation for pplx-embed-v1-0.6B with optional architectural skips.

Evaluates embedding quality on:
  - STS-B: Spearman correlation between model cosine similarities and human scores
  - SciFact: nDCG@10 on a small information retrieval benchmark (paper Table 9)

Supports testing architectural modifications to validate their accuracy impact
before applying them for performance on TT devices:
  --skip-rope       Skip rotary position embeddings (all layers)
  --skip-qk-norm    Skip Q/K RMSNorm (all layers)

Examples:
  # Baseline
  python models/demos/blackhole/pplx_embed_0_6b/demo/eval_accuracy.py

  # Skip RoPE
  python models/demos/blackhole/pplx_embed_0_6b/demo/eval_accuracy.py --skip-rope

  # Skip QK norm
  python models/demos/blackhole/pplx_embed_0_6b/demo/eval_accuracy.py --skip-qk-norm

  # Both skips
  python models/demos/blackhole/pplx_embed_0_6b/demo/eval_accuracy.py --skip-rope --skip-qk-norm

  # SciFact nDCG@10 (retrieval, as in pplx-embed paper Table 9)
  python models/demos/blackhole/pplx_embed_0_6b/demo/eval_accuracy.py --dataset scifact

  # Both benchmarks
  python models/demos/blackhole/pplx_embed_0_6b/demo/eval_accuracy.py --dataset both
"""

import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from scipy.stats import spearmanr

MODEL_NAME = "perplexity-ai/pplx-embed-v1-0.6b"


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def load_stsb():
    """Load STS-B test set. Returns (sentences1, sentences2, scores_normalized)."""
    from datasets import load_dataset

    ds = load_dataset("mteb/stsbenchmark-sts", split="test")
    scores = [s / 5.0 for s in ds["score"]]
    return list(ds["sentence1"]), list(ds["sentence2"]), scores


def load_scifact():
    """Load SciFact retrieval benchmark.

    Returns (queries_dict, corpus_dict, qrels) where:
      queries_dict: {query_id: text}
      corpus_dict:  {corpus_id: text}
      qrels:        {query_id: {corpus_id: relevance_score}}
    """
    from datasets import load_dataset

    corpus_ds = load_dataset("mteb/scifact", "corpus", split="corpus")
    queries_ds = load_dataset("mteb/scifact", "queries", split="queries")
    default_ds = load_dataset("mteb/scifact", "default", split="test")

    corpus = {}
    for row in corpus_ds:
        text = row.get("title", "")
        if text and row["text"]:
            text += " "
        text += row["text"]
        corpus[row["_id"]] = text.strip()

    queries = {row["_id"]: row["text"] for row in queries_ds}

    qrels = {}
    for row in default_ds:
        qid = str(row["query-id"])
        cid = str(row["corpus-id"])
        score = row["score"]
        qrels.setdefault(qid, {})[cid] = score

    return queries, corpus, qrels


# ---------------------------------------------------------------------------
# Model loading & modification
# ---------------------------------------------------------------------------


def load_model(skip_rope=False, skip_qk_norm=False):
    """Load pplx-embed model on CPU with optional architectural skips.

    Uses the Qwen3 base architecture (identical to pplx-embed) with weights
    loaded directly from pplx-embed safetensors.  SDPA is monkey-patched to
    use bidirectional (non-causal) attention matching pplx-embed's design.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    model = _load_pplx_model()
    model.eval()

    if skip_rope:
        _apply_skip_rope(model)
        logger.info("Applied: skip RoPE (identity rotation)")
    if skip_qk_norm:
        _apply_skip_qk_norm(model)
        logger.info("Applied: skip Q/K norm (identity)")

    return model, tokenizer


def _load_pplx_model():
    """Load pplx-embed weights into Qwen3 architecture with bidirectional attention."""
    import functools
    import json

    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    config_path = hf_hub_download(MODEL_NAME, "config.json")
    safetensor_path = hf_hub_download(MODEL_NAME, "model.safetensors")

    with open(config_path) as f:
        cfg = json.load(f)

    from transformers import Qwen3Config, Qwen3Model

    qwen3_config = Qwen3Config(
        hidden_size=cfg["hidden_size"],
        num_hidden_layers=cfg["num_hidden_layers"],
        num_attention_heads=cfg["num_attention_heads"],
        num_key_value_heads=cfg["num_key_value_heads"],
        intermediate_size=cfg["intermediate_size"],
        vocab_size=cfg["vocab_size"],
        max_position_embeddings=cfg.get("max_position_embeddings", 32768),
        rms_norm_eps=cfg.get("rms_norm_eps", 1e-6),
        rope_theta=cfg.get("rope_theta", 1000000.0),
        head_dim=cfg.get("head_dim", 128),
        tie_word_embeddings=cfg.get("tie_word_embeddings", True),
    )

    model = Qwen3Model(qwen3_config)

    raw_sd = load_file(safetensor_path)
    result = model.load_state_dict(raw_sd, strict=False)
    if result.unexpected_keys:
        logger.warning(f"Unexpected keys when loading weights: {result.unexpected_keys[:5]}...")
    if result.missing_keys:
        logger.debug(f"Missing keys (expected for base model): {result.missing_keys[:5]}...")
    logger.info(f"Loaded pplx-embed weights into Qwen3Model ({cfg['num_hidden_layers']} layers)")

    # Monkey-patch SDPA globally to force bidirectional (non-causal) attention.
    # pplx-embed uses is_causal=False; Qwen3 defaults to causal.
    original_sdpa = torch.nn.functional.scaled_dot_product_attention

    @functools.wraps(original_sdpa)
    def _bidirectional_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kwargs):
        return original_sdpa(
            query, key, value, attn_mask=None, dropout_p=dropout_p, is_causal=False, scale=scale, **kwargs
        )

    torch.nn.functional.scaled_dot_product_attention = _bidirectional_sdpa
    logger.info("Patched SDPA for bidirectional attention")

    return model


def _get_layers(model):
    """Get transformer layers from model (handles both custom pplx and standard Qwen3)."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "layers"):
        return model.layers
    raise AttributeError("Cannot find transformer layers")


def _get_attn(layer):
    """Get attention module from a decoder layer."""
    for attr in ("self_attn", "attention"):
        if hasattr(layer, attr):
            return getattr(layer, attr)
    raise AttributeError("Cannot find attention module in layer")


def _apply_skip_rope(model):
    """Replace rotary embeddings with identity (cos=1, sin=0)."""
    layers = _get_layers(model)

    class IdentityRotary(torch.nn.Module):
        def __init__(self, original):
            super().__init__()
            self.original = original

        def forward(self, x, position_ids=None, **kwargs):
            cos, sin = self.original(x, position_ids=position_ids, **kwargs)
            return torch.ones_like(cos), torch.zeros_like(sin)

    for layer in layers:
        attn = _get_attn(layer)
        if hasattr(attn, "rotary_emb"):
            attn.rotary_emb = IdentityRotary(attn.rotary_emb)

    logger.info(f"  Replaced rotary_emb in {len(layers)} layers")


def _apply_skip_qk_norm(model):
    """Replace Q/K RMSNorm with identity."""
    layers = _get_layers(model)

    class IdentityNorm(torch.nn.Module):
        def forward(self, x):
            return x

    count = 0
    for layer in layers:
        attn = _get_attn(layer)
        if hasattr(attn, "q_norm"):
            attn.q_norm = IdentityNorm()
            count += 1
        if hasattr(attn, "k_norm"):
            attn.k_norm = IdentityNorm()

    logger.info(f"  Replaced q_norm/k_norm in {count} layers")


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------


@torch.no_grad()
def encode_texts(texts, model, tokenizer, batch_size=32, max_length=512):
    """Encode texts with mean pooling (pplx-embed style)."""
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        outputs = model(**inputs)

        if hasattr(outputs, "last_hidden_state"):
            hidden = outputs.last_hidden_state
        elif isinstance(outputs, tuple):
            hidden = outputs[0]
        else:
            hidden = outputs

        mask = inputs["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        all_embs.append(pooled.cpu())

        if (i // batch_size) % 20 == 0 and i > 0:
            logger.info(f"    encoded {i}/{len(texts)} texts...")

    return torch.cat(all_embs, dim=0)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def eval_stsb(embs1, embs2, gold_scores):
    """Spearman correlation on STS-B."""
    cos_sims = F.cosine_similarity(embs1, embs2, dim=-1).numpy()
    corr, pval = spearmanr(cos_sims, np.array(gold_scores))
    return float(corr), float(pval)


def eval_retrieval(query_embs, query_ids, corpus_embs, corpus_ids, qrels, k=10):
    """Compute mean nDCG@k over all queries with relevance judgments."""
    query_embs_n = F.normalize(query_embs, dim=-1)
    corpus_embs_n = F.normalize(corpus_embs, dim=-1)

    sim_matrix = query_embs_n @ corpus_embs_n.T

    corpus_id_to_idx = {cid: idx for idx, cid in enumerate(corpus_ids)}

    ndcg_scores = []
    for qi, qid in enumerate(query_ids):
        qid_str = str(qid)
        if qid_str not in qrels or not qrels[qid_str]:
            continue

        qrel = qrels[qid_str]
        scores = sim_matrix[qi].numpy()
        top_k_indices = np.argsort(-scores)[:k]

        rel_at_rank = [float(qrel.get(str(corpus_ids[ci]), 0)) for ci in top_k_indices]

        dcg = sum(r / np.log2(rank + 2) for rank, r in enumerate(rel_at_rank))

        ideal_rels = sorted(qrel.values(), reverse=True)[:k]
        idcg = sum(float(r) / np.log2(rank + 2) for rank, r in enumerate(ideal_rels))
        if idcg == 0:
            continue

        ndcg_scores.append(dcg / idcg)

    return float(np.mean(ndcg_scores)) if ndcg_scores else 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="pplx-embed-v1-0.6B accuracy eval")
    parser.add_argument("--skip-rope", action="store_true", help="Skip RoPE in all layers")
    parser.add_argument("--skip-qk-norm", action="store_true", help="Skip Q/K RMSNorm in all layers")
    parser.add_argument("--batch-size", type=int, default=32, help="Encoding batch size")
    parser.add_argument("--max-length", type=int, default=512, help="Max token length")
    parser.add_argument("--dataset", default="stsb", choices=["stsb", "scifact", "both"])
    args = parser.parse_args()

    config_parts = []
    if args.skip_rope:
        config_parts.append("skip-rope")
    if args.skip_qk_norm:
        config_parts.append("skip-qk-norm")
    config_label = "+".join(config_parts) if config_parts else "baseline"

    logger.info(f"Config: [{config_label}]")

    model, tokenizer = load_model(args.skip_rope, args.skip_qk_norm)

    results = {}

    if args.dataset in ("stsb", "both"):
        logger.info("--- STS-B Evaluation ---")
        s1, s2, scores = load_stsb()
        all_texts = s1 + s2
        logger.info(f"  {len(s1)} pairs, {len(all_texts)} texts to encode")

        t0 = time.perf_counter()
        all_embs = encode_texts(all_texts, model, tokenizer, args.batch_size, args.max_length)
        enc_time = time.perf_counter() - t0
        logger.info(f"  Encoding time: {enc_time:.1f}s ({enc_time / len(all_texts) * 1000:.1f}ms/text)")

        embs1 = all_embs[: len(s1)]
        embs2 = all_embs[len(s1) :]
        corr, pval = eval_stsb(embs1, embs2, scores)
        results["stsb_spearman"] = corr
        results["stsb_pval"] = pval
        logger.info(f"  STS-B Spearman: {corr:.4f} (p={pval:.2e})")

    if args.dataset in ("scifact", "both"):
        logger.info("--- SciFact Retrieval Evaluation ---")
        try:
            queries_dict, corpus_dict, qrels = load_scifact()
            query_ids = list(queries_dict.keys())
            query_texts = [queries_dict[qid] for qid in query_ids]
            corpus_ids = list(corpus_dict.keys())
            corpus_texts = [corpus_dict[cid] for cid in corpus_ids]
            logger.info(f"  {len(query_ids)} queries, {len(corpus_ids)} corpus docs")

            t0 = time.perf_counter()
            query_embs = encode_texts(query_texts, model, tokenizer, args.batch_size, args.max_length)
            corpus_embs = encode_texts(corpus_texts, model, tokenizer, args.batch_size, args.max_length)
            enc_time = time.perf_counter() - t0
            logger.info(f"  Encoding time: {enc_time:.1f}s")

            ndcg = eval_retrieval(query_embs, query_ids, corpus_embs, corpus_ids, qrels, k=10)
            results["scifact_ndcg10"] = ndcg
            logger.info(f"  SciFact nDCG@10: {ndcg:.4f}")
        except Exception as e:
            logger.error(f"SciFact eval failed: {e}")

    logger.info("")
    logger.info("=" * 60)
    logger.info(f"  pplx-embed-v1-0.6B Accuracy [{config_label}]")
    logger.info("=" * 60)
    for key, val in results.items():
        if "pval" not in key:
            logger.info(f"  {key:20s}: {val:.4f}")
    logger.info(f"  {'skip_rope':20s}: {args.skip_rope}")
    logger.info(f"  {'skip_qk_norm':20s}: {args.skip_qk_norm}")
    logger.info("=" * 60)

    return results


if __name__ == "__main__":
    main()
