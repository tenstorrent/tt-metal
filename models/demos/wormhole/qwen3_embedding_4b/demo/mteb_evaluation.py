# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""MTEB (Multilingual) evaluation for Qwen3-Embedding-4B on Tenstorrent hardware.

Runs both the HuggingFace reference model and the TT-accelerated model on the
same MTEB datasets, then displays a comparison table with TT/HF accuracy ratio.

Published MTEB (Multilingual) scores from the model card are included for
context.  The HF reference run on our subset should closely match the published
per-dataset scores, validating that our evaluation pipeline is correct.

Supported task types (auto-detected from dataset columns):
  - Retrieval:  ArguAna, etc.           -> recall@k, nDCG@k
  - STS:        STS-Benchmark, etc.     -> Spearman correlation

Examples:
    # Default: run ArguAna retrieval + STS-Benchmark, both HF and TT
    MESH_DEVICE=P150 python models/demos/wormhole/qwen3_embedding_4b/demo/mteb_evaluation.py

    # Retrieval only, full dataset
    MESH_DEVICE=P150 python .../mteb_evaluation.py --datasets mteb/ArguAna --max-samples 0

    # Quick subset
    MESH_DEVICE=P150 python .../mteb_evaluation.py --max-samples 100

    # Skip HF reference (TT-only, faster)
    MESH_DEVICE=P150 python .../mteb_evaluation.py --skip-hf-reference

4B-specific notes:
  - hidden_size=2560 (vs 1024 for 0.6B), 36 layers (vs 28)
  - bs=1 activations fit in L1; bs>=11 spills to DRAM
  - bs=32 uses the full 130-core (13x10) matmul grid
"""

from __future__ import annotations

import argparse
import os
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from loguru import logger
from scipy.stats import spearmanr
from tqdm import tqdm

import ttnn
from models.demos.wormhole.qwen3_embedding_4b.demo._common import (
    MODEL_NAME,
    apply_recommended_env,
    build_single_device_model,
    generate_synthetic_inputs,
)

DEFAULT_BATCH_SIZE = 32
DEFAULT_SEQ_LEN = 512
DEFAULT_QUERY_INSTRUCTION = (
    "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"
)

DEFAULT_DATASETS = ["mteb/ArguAna", "mteb/stsbenchmark-sts"]

PUBLISHED_MTEB_SCORES = {
    "0.6B": {
        "Mean (Task)": 64.33,
        "Mean (Type)": 56.00,
        "Bitext Mining": 72.22,
        "Classification": 66.83,
        "Clustering": 52.33,
        "Instruction Retrieval": 5.09,
        "Multilabel Classification": 24.59,
        "Pair Classification": 80.83,
        "Reranking": 61.41,
        "Retrieval": 64.64,
        "STS": 76.17,
    },
    "4B": {
        "Mean (Task)": 69.45,
        "Mean (Type)": 60.86,
        "Bitext Mining": 79.36,
        "Classification": 72.33,
        "Clustering": 57.15,
        "Instruction Retrieval": 11.56,
        "Multilabel Classification": 26.77,
        "Pair Classification": 85.05,
        "Reranking": 65.08,
        "Retrieval": 69.60,
        "STS": 80.86,
    },
    "8B": {
        "Mean (Task)": 70.58,
        "Mean (Type)": 61.69,
        "Bitext Mining": 80.89,
        "Classification": 74.00,
        "Clustering": 57.65,
        "Instruction Retrieval": 10.06,
        "Multilabel Classification": 28.66,
        "Pair Classification": 86.40,
        "Reranking": 65.63,
        "Retrieval": 70.88,
        "STS": 81.08,
    },
}

DATASET_URL_FALLBACKS = {
    "mteb/arguana": {
        "test": {
            "builder_name": "json",
            "data_files": {"test": "https://huggingface.co/datasets/mteb/arguana/resolve/main/qrels/test.jsonl"},
        },
        "corpus": {
            "builder_name": "json",
            "data_files": {"corpus": "https://huggingface.co/datasets/mteb/arguana/resolve/main/corpus.jsonl"},
        },
        "queries": {
            "builder_name": "json",
            "data_files": {"queries": "https://huggingface.co/datasets/mteb/arguana/resolve/main/queries.jsonl"},
        },
    },
    "mteb/stsbenchmark-sts": {
        "test": {
            "builder_name": "json",
            "data_files": {"test": "https://huggingface.co/datasets/mteb/stsbenchmark-sts/resolve/main/test.jsonl.gz"},
        },
    },
}


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def _load_mteb_dataset(dataset_name: str, split: str = "test", max_samples: int | None = None):
    logger.info(f"Loading dataset {dataset_name} ({split=})")
    normalized = dataset_name.lower()
    try:
        dataset = load_dataset(dataset_name, split=split)
    except Exception as exc:
        fallback = DATASET_URL_FALLBACKS.get(normalized, {}).get(split)
        if fallback is None:
            raise
        logger.warning(f"Primary load failed ({type(exc).__name__}). Using fallback.")
        dataset = load_dataset(fallback["builder_name"], data_files=fallback["data_files"], split=split)

    if max_samples is not None and max_samples > 0:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    logger.info(f"Loaded {len(dataset)} samples")
    return dataset


def _first_present(example: dict, fields: Sequence[str]) -> str | None:
    for field in fields:
        if field in example and example[field] is not None:
            value = str(example[field]).strip()
            if value:
                return value
    return None


def _detect_task_type(dataset) -> str:
    columns = set(dataset.column_names)
    if {"sentence1", "sentence2"}.issubset(columns):
        return "sts"
    if (
        {"query-id", "corpus-id"}.issubset(columns)
        or {"query_id", "corpus_id"}.issubset(columns)
        or {"query", "corpus"}.issubset(columns)
    ):
        return "retrieval"
    raise ValueError(f"Unsupported dataset format with columns: {sorted(columns)}")


def _extract_retrieval_samples(dataset_name: str, dataset, query_instruction: str):
    queries, documents, relevant_doc_ids = [], [], []
    columns = set(dataset.column_names)

    if {"query-id", "corpus-id"}.issubset(columns) or {"query_id", "corpus_id"}.issubset(columns):
        corpus_dataset = _load_mteb_dataset(dataset_name, split="corpus")
        queries_dataset = _load_mteb_dataset(dataset_name, split="queries")

        doc_dict: dict[str, str] = {}
        for ex in corpus_dataset:
            doc_id = _first_present(ex, ("_id", "docid", "id", "doc_id", "corpus-id", "corpus_id"))
            title = _first_present(ex, ("title",))
            body = _first_present(ex, ("text", "content", "passage"))
            doc_text = " ".join(p for p in (title, body) if p)
            if doc_id and doc_text and doc_id not in doc_dict:
                doc_dict[doc_id] = doc_text

        query_dict: dict[str, str] = {}
        for ex in queries_dataset:
            qid = _first_present(ex, ("_id", "queryid", "id", "query_id", "query-id"))
            qtxt = _first_present(ex, ("text", "query", "question"))
            if qid and qtxt and qid not in query_dict:
                query_dict[qid] = f"{query_instruction}{qtxt}"

        seen_corpus, seen_query = [], []
        q2rel: dict[str, list[str]] = {}
        for ex in dataset:
            qid = _first_present(ex, ("query-id", "query_id"))
            cid = _first_present(ex, ("corpus-id", "corpus_id"))
            if not qid or not cid:
                continue
            if qid not in seen_query:
                seen_query.append(qid)
            if cid not in seen_corpus:
                seen_corpus.append(cid)
            if float(ex.get("score", 0.0)) > 0:
                q2rel.setdefault(qid, []).append(cid)

        corpus_ids = [c for c in seen_corpus if c in doc_dict]
        documents = [doc_dict[c] for c in corpus_ids]
        cid2idx = {c: i for i, c in enumerate(corpus_ids)}

        for qid in seen_query:
            qt = query_dict.get(qid)
            if not qt:
                continue
            queries.append(qt)
            rel = q2rel.get(qid, [])
            first = next((d for d in rel if d in cid2idx), None)
            relevant_doc_ids.append(cid2idx[first] if first else -1)

    elif {"query", "corpus"}.issubset(columns):
        doc_dict = {}
        for ex in dataset:
            corpus = ex["corpus"]
            if isinstance(corpus, dict):
                for did, dtxt in corpus.items():
                    did = str(did)
                    if did not in doc_dict:
                        doc_dict[did] = str(dtxt)

        corpus_ids = sorted(doc_dict)
        documents = [doc_dict[c] for c in corpus_ids]
        cid2idx = {c: i for i, c in enumerate(corpus_ids)}

        for ex in dataset:
            queries.append(f"{query_instruction}{ex['query']}")
            rel = ex.get("relevant_docs", [])
            if isinstance(rel, list) and rel:
                relevant_doc_ids.append(cid2idx.get(str(rel[0]), -1))
            elif rel:
                relevant_doc_ids.append(cid2idx.get(str(rel), -1))
            else:
                relevant_doc_ids.append(-1)
    else:
        raise ValueError(f"Unsupported retrieval format: {sorted(columns)}")

    logger.info(f"Prepared retrieval: {len(queries)} queries, {len(documents)} documents")
    return queries, documents, relevant_doc_ids


def _extract_sts_samples(dataset_name: str, dataset):
    s1, s2, scores = [], [], []
    for ex in dataset:
        if "sentence1" not in ex or "sentence2" not in ex:
            continue
        s1.append(str(ex["sentence1"]))
        s2.append(str(ex["sentence2"]))
        scores.append(float(ex.get("score", ex.get("label", 0.0))))
    logger.info(f"Prepared STS: {len(s1)} sentence pairs")
    return s1, s2, scores


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _normalize(embeddings: torch.Tensor) -> torch.Tensor:
    return F.normalize(embeddings, p=2, dim=1)


def _recall_at_k(sims: np.ndarray, rel: Sequence[int], k: int) -> float:
    scores = []
    for i, rid in enumerate(rel):
        if rid < 0:
            continue
        top_k = np.argsort(sims[i])[::-1][:k]
        scores.append(1.0 if rid in top_k else 0.0)
    return float(np.mean(scores)) if scores else 0.0


def _ndcg_at_k(sims: np.ndarray, rel: Sequence[int], k: int) -> float:
    scores = []
    for i, rid in enumerate(rel):
        if rid < 0:
            continue
        top_k = np.argsort(sims[i])[::-1][:k]
        if rid in top_k:
            rank = int(np.where(top_k == rid)[0][0]) + 1
            scores.append(1.0 / np.log2(rank + 1))
        else:
            scores.append(0.0)
    return float(np.mean(scores)) if scores else 0.0


# ---------------------------------------------------------------------------
# HF reference encoding (last-token pooling, matches model card)
# ---------------------------------------------------------------------------


def _last_token_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    left_padding = mask[:, -1].sum() == mask.shape[0]
    if left_padding:
        return hidden[:, -1]
    seq_lens = mask.sum(dim=1) - 1
    return hidden[torch.arange(hidden.shape[0], device=hidden.device), seq_lens.long()]


def _build_hf_reference():
    from transformers import AutoModel, AutoTokenizer

    logger.info(f"Loading HF reference model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left")
    model = AutoModel.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16).eval()
    if torch.cuda.is_available():
        model = model.cuda()
    logger.info(f"HF model loaded on {model.device}")
    return model, tokenizer


def _encode_hf(
    texts: list[str],
    model,
    tokenizer,
    seq_len: int,
    batch_size: int,
    *,
    desc: str = "HF encoding",
) -> torch.Tensor:
    all_embeddings = []
    for start in tqdm(range(0, len(texts), batch_size), desc=desc, leave=False):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(batch, padding=True, truncation=True, max_length=seq_len, return_tensors="pt")
        encoded = {k: v.to(model.device) for k, v in encoded.items()}
        with torch.no_grad():
            outputs = model(**encoded)
        emb = _last_token_pool(outputs.last_hidden_state, encoded["attention_mask"])
        all_embeddings.append(emb.float().cpu())
    return torch.cat(all_embeddings, dim=0)


# ---------------------------------------------------------------------------
# TT model encoding
# ---------------------------------------------------------------------------


def _build_tt_model(device, batch_size: int, seq_len: int):
    os.environ.setdefault("HF_MODEL", MODEL_NAME)
    apply_recommended_env(batched_l1=batch_size <= 10)

    generator, model_args, kv_caches, page_table = build_single_device_model(
        device, batch_size=batch_size, seq_len=seq_len
    )
    tokenizer = model_args.tokenizer

    logger.info("Compiling TT model (warmup forward)...")
    warmup_ids, warmup_lens = generate_synthetic_inputs(tokenizer, batch_size, seq_len)
    generator.prefill_forward_text(
        warmup_ids,
        page_table=page_table,
        kv_cache=kv_caches,
        prompt_lens=warmup_lens,
        enable_trace=True,
        return_hidden_states=True,
        warmup_prefill=True,
    )
    logger.info("TT model compiled")
    return generator, tokenizer, kv_caches, page_table


def _encode_ttnn(
    texts: list[str],
    generator,
    tokenizer,
    page_table,
    kv_caches,
    batch_size: int,
    seq_len: int,
    *,
    desc: str = "TT encoding",
) -> torch.Tensor:
    orig_pad = tokenizer.padding_side
    tokenizer.padding_side = "right"

    all_embeddings = []
    for start in tqdm(range(0, len(texts), batch_size), desc=desc, leave=False):
        batch = list(texts[start : start + batch_size])
        n_real = len(batch)
        while len(batch) < batch_size:
            batch.append(batch[-1])

        encoded = tokenizer(batch, padding="max_length", max_length=seq_len, truncation=True, return_tensors="pt")
        prompt_lens = [max(1, int(x)) for x in encoded["attention_mask"].sum(dim=1).tolist()]

        generator.prev_page_table = None
        emb = generator.prefill_forward_text(
            encoded["input_ids"],
            page_table=page_table,
            kv_cache=kv_caches,
            prompt_lens=prompt_lens,
            enable_trace=True,
            return_hidden_states=True,
            warmup_prefill=False,
        )
        ttnn.synchronize_device(generator.model[0].mesh_device)

        if isinstance(emb, torch.Tensor):
            all_embeddings.append(emb[:n_real].float().cpu())
        else:
            all_embeddings.append(torch.tensor(emb[:n_real], dtype=torch.float32))

    tokenizer.padding_side = orig_pad
    return torch.cat(all_embeddings, dim=0)


# ---------------------------------------------------------------------------
# Per-dataset evaluation
# ---------------------------------------------------------------------------


def _eval_retrieval(queries, documents, rel_ids, encode_fn, *, batch_size, seq_len, label):
    logger.info(f"[{label}] Encoding {len(queries)} queries + {len(documents)} documents...")
    q_emb = encode_fn(queries, batch_size=batch_size, seq_len=seq_len, desc=f"{label} queries")
    d_emb = encode_fn(documents, batch_size=batch_size, seq_len=seq_len, desc=f"{label} docs")
    sims = torch.mm(_normalize(q_emb), _normalize(d_emb).t()).numpy()
    return {
        "recall@10": _recall_at_k(sims, rel_ids, k=10) * 100,
        "ndcg@10": _ndcg_at_k(sims, rel_ids, k=10) * 100,
        "recall@100": _recall_at_k(sims, rel_ids, k=100) * 100,
        "ndcg@100": _ndcg_at_k(sims, rel_ids, k=100) * 100,
    }


def _eval_sts(s1, s2, gold, encode_fn, *, batch_size, seq_len, label):
    logger.info(f"[{label}] Encoding {len(s1)} sentence pairs...")
    e1 = encode_fn(s1, batch_size=batch_size, seq_len=seq_len, desc=f"{label} sent1")
    e2 = encode_fn(s2, batch_size=batch_size, seq_len=seq_len, desc=f"{label} sent2")
    cos = (_normalize(e1) * _normalize(e2)).sum(dim=1).numpy()
    sp, _ = spearmanr(cos, gold)
    return {"spearman": float(sp) * 100}


def _embedding_alignment(ref: torch.Tensor, cand: torch.Tensor) -> float:
    return float((_normalize(ref) * _normalize(cand)).sum(dim=1).mean().item())


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------


def run_mteb_evaluation(
    device,
    *,
    dataset_names: list[str],
    batch_size: int = DEFAULT_BATCH_SIZE,
    seq_len: int = DEFAULT_SEQ_LEN,
    max_samples: int | None = 100,
    query_instruction: str = DEFAULT_QUERY_INSTRUCTION,
    skip_hf_reference: bool = False,
):
    generator, tt_tokenizer, kv_caches, page_table = _build_tt_model(device, batch_size, seq_len)

    def tt_encode(texts, *, batch_size, seq_len, desc):
        return _encode_ttnn(texts, generator, tt_tokenizer, page_table, kv_caches, batch_size, seq_len, desc=desc)

    hf_model, hf_tokenizer = (None, None)
    if not skip_hf_reference:
        hf_model, hf_tokenizer = _build_hf_reference()

    def hf_encode(texts, *, batch_size, seq_len, desc):
        return _encode_hf(texts, hf_model, hf_tokenizer, seq_len, min(batch_size, 8), desc=desc)

    results = []

    for ds_name in dataset_names:
        logger.info(f"\n{'='*70}")
        logger.info(f"Dataset: {ds_name}")
        logger.info(f"{'='*70}")

        test_ds = _load_mteb_dataset(ds_name, split="test", max_samples=max_samples)
        task_type = _detect_task_type(test_ds)

        if task_type == "retrieval":
            queries, docs, rel_ids = _extract_retrieval_samples(ds_name, test_ds, query_instruction)
            tt_metrics = _eval_retrieval(
                queries, docs, rel_ids, tt_encode, batch_size=batch_size, seq_len=seq_len, label="TT"
            )
            hf_metrics, alignment = None, None
            if not skip_hf_reference:
                hf_metrics = _eval_retrieval(
                    queries, docs, rel_ids, hf_encode, batch_size=batch_size, seq_len=seq_len, label="HF"
                )
                q_tt = tt_encode(
                    queries[: min(64, len(queries))], batch_size=batch_size, seq_len=seq_len, desc="align-q"
                )
                q_hf = hf_encode(
                    queries[: min(64, len(queries))], batch_size=batch_size, seq_len=seq_len, desc="align-q"
                )
                alignment = _embedding_alignment(q_hf, q_tt)
            results.append(
                {
                    "dataset": ds_name,
                    "task_type": "Retrieval",
                    "metric_name": "nDCG@10",
                    "tt": tt_metrics["ndcg@10"],
                    "hf": hf_metrics["ndcg@10"] if hf_metrics else None,
                    "alignment": alignment,
                    "tt_details": tt_metrics,
                    "hf_details": hf_metrics,
                }
            )

        elif task_type == "sts":
            s1, s2, gold = _extract_sts_samples(ds_name, test_ds)
            tt_metrics = _eval_sts(s1, s2, gold, tt_encode, batch_size=batch_size, seq_len=seq_len, label="TT")
            hf_metrics, alignment = None, None
            if not skip_hf_reference:
                hf_metrics = _eval_sts(s1, s2, gold, hf_encode, batch_size=batch_size, seq_len=seq_len, label="HF")
                e_tt = tt_encode(s1[: min(64, len(s1))], batch_size=batch_size, seq_len=seq_len, desc="align")
                e_hf = hf_encode(s1[: min(64, len(s1))], batch_size=batch_size, seq_len=seq_len, desc="align")
                alignment = _embedding_alignment(e_hf, e_tt)
            results.append(
                {
                    "dataset": ds_name,
                    "task_type": "STS",
                    "metric_name": "Spearman",
                    "tt": tt_metrics["spearman"],
                    "hf": hf_metrics["spearman"] if hf_metrics else None,
                    "alignment": alignment,
                    "tt_details": tt_metrics,
                    "hf_details": hf_metrics,
                }
            )

    _print_comparison_table(results, skip_hf_reference)
    return results


def _print_comparison_table(results: list[dict], skip_hf: bool):
    model_size = "4B"
    published = PUBLISHED_MTEB_SCORES.get(model_size, {})

    logger.info("")
    logger.info("=" * 90)
    logger.info(f"  MTEB Evaluation Summary  —  Qwen3-Embedding-{model_size}")
    logger.info("=" * 90)

    if not skip_hf:
        header = (
            f"{'Dataset':<30} {'Task':<12} {'Metric':<10} {'Published':>10} {'HF (ours)':>10} {'TT':>10} {'TT/HF':>8}"
        )
    else:
        header = f"{'Dataset':<30} {'Task':<12} {'Metric':<10} {'Published':>10} {'TT':>10}"
    logger.info(header)
    logger.info("-" * len(header))

    for r in results:
        pub = published.get(r["task_type"], None)
        pub_str = f"{pub:.2f}" if pub is not None else "—"
        tt_str = f"{r['tt']:.2f}"
        if not skip_hf and r["hf"] is not None:
            hf_str = f"{r['hf']:.2f}"
            ratio = r["tt"] / r["hf"] if r["hf"] != 0 else float("nan")
            ratio_str = f"{ratio:.4f}"
            logger.info(
                f"  {r['dataset']:<28} {r['task_type']:<12} {r['metric_name']:<10} {pub_str:>10} {hf_str:>10} {tt_str:>10} {ratio_str:>8}"
            )
        else:
            logger.info(f"  {r['dataset']:<28} {r['task_type']:<12} {r['metric_name']:<10} {pub_str:>10} {tt_str:>10}")

    if not skip_hf:
        alignments = [r["alignment"] for r in results if r["alignment"] is not None]
        if alignments:
            logger.info("-" * len(header))
            logger.info(f"  Mean embedding alignment (cosine) TT vs HF: {sum(alignments)/len(alignments):.4f}")

    logger.info("=" * 90)

    logger.info("")
    logger.info("Published MTEB (Multilingual) scores from model card:")
    logger.info(f"  {'Category':<28} {'0.6B':>8} {'4B':>8} {'8B':>8}")
    logger.info(f"  {'-'*52}")
    for cat in [
        "Mean (Task)",
        "Mean (Type)",
        "Bitext Mining",
        "Classification",
        "Clustering",
        "Instruction Retrieval",
        "Multilabel Classification",
        "Pair Classification",
        "Reranking",
        "Retrieval",
        "STS",
    ]:
        s06 = PUBLISHED_MTEB_SCORES["0.6B"].get(cat, 0)
        s4 = PUBLISHED_MTEB_SCORES["4B"].get(cat, 0)
        s8 = PUBLISHED_MTEB_SCORES["8B"].get(cat, 0)
        logger.info(f"  {cat:<28} {s06:>8.2f} {s4:>8.2f} {s8:>8.2f}")
    logger.info("")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args():
    parser = argparse.ArgumentParser(description="MTEB evaluation for Qwen3-Embedding-4B: TT vs HF comparison")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument("--max-samples", type=int, default=100, help="Max samples per dataset (0 = all)")
    parser.add_argument("--query-instruction", default=DEFAULT_QUERY_INSTRUCTION)
    parser.add_argument("--skip-hf-reference", action="store_true")
    parser.add_argument("--device-id", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    max_samples = args.max_samples if args.max_samples > 0 else None

    logger.info(f"Opening device {args.device_id}...")
    device = ttnn.open_device(
        device_id=args.device_id,
        l1_small_size=32768,
        trace_region_size=200_000_000,
        num_command_queues=1,
    )
    try:
        run_mteb_evaluation(
            device,
            dataset_names=args.datasets,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            max_samples=max_samples,
            query_instruction=args.query_instruction,
            skip_hf_reference=args.skip_hf_reference,
        )
    finally:
        ttnn.close_device(device)
