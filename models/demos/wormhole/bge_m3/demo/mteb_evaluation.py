# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Dense-only MTEB evaluation for the BGE-M3 generator wrapper.

Examples:
    TT_VISIBLE_DEVICES=0 python models/demos/wormhole/bge_m3/demo/mteb_evaluation.py
    TT_VISIBLE_DEVICES=0 python models/demos/wormhole/bge_m3/demo/mteb_evaluation.py --dataset-name mteb/stsbenchmark-sts
"""

from __future__ import annotations

import argparse
import os
from typing import Sequence

import numpy as np
import torch
from datasets import load_dataset
from loguru import logger
from scipy.stats import spearmanr
from tqdm import tqdm
from transformers import AutoModelForCausalLM

import ttnn
from models.demos.wormhole.bge_m3.demo.generator_vllm import BgeM3ForEmbedding

QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
DEFAULT_MODEL_NAME = "BAAI/bge-m3"
DEFAULT_SEQUENCE_LENGTH = 8192
DEFAULT_PER_CHIP_BATCH_SIZES = (8, 16, 32)
# Temporarily disable HF reference evaluation so this script can be used for TT-only device testing.
ENABLE_HF_REFERENCE = False
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


def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1)
    return summed / counts


def _crop_hidden_states(last_hidden_state: torch.Tensor, seq_len: int) -> torch.Tensor:
    return last_hidden_state[:, :seq_len, :]


def _normalize_embeddings(embeddings: torch.Tensor) -> torch.Tensor:
    return embeddings / torch.norm(embeddings, dim=1, keepdim=True).clamp(min=1e-12)


def _mean_embedding_alignment(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    similarity = (_normalize_embeddings(reference) * _normalize_embeddings(candidate)).sum(dim=1)
    return float(similarity.mean().item())


def _load_mteb_dataset(dataset_name: str, split: str = "test", max_samples: int | None = None):
    logger.info(f"Loading dataset {dataset_name} ({split=})")
    dataset_name_normalized = dataset_name.lower()

    try:
        dataset = load_dataset(dataset_name, split=split)
    except Exception as exc:
        fallback = DATASET_URL_FALLBACKS.get(dataset_name_normalized, {}).get(split)
        if fallback is None:
            raise

        logger.warning(
            "Primary load_dataset({}, split={!r}) failed with {}. Falling back to {} via explicit data_files.",
            dataset_name,
            split,
            type(exc).__name__,
            fallback["builder_name"],
        )
        dataset = load_dataset(
            fallback["builder_name"],
            data_files=fallback["data_files"],
            split=split,
        )

    if max_samples is not None:
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
    if {"query-id", "corpus-id"}.issubset(columns) or {"query_id", "corpus_id"}.issubset(columns):
        return "retrieval"
    if {"query", "corpus"}.issubset(columns):
        return "retrieval"
    raise ValueError(f"Unsupported dataset format with columns: {sorted(columns)}")


def _extract_retrieval_samples(dataset_name: str, dataset):
    queries: list[str] = []
    documents: list[str] = []
    relevant_doc_ids: list[int] = []
    columns = set(dataset.column_names)

    if {"query-id", "corpus-id"}.issubset(columns) or {"query_id", "corpus_id"}.issubset(columns):
        corpus_dataset = _load_mteb_dataset(dataset_name, split="corpus")
        queries_dataset = _load_mteb_dataset(dataset_name, split="queries")

        doc_dict: dict[str, str] = {}
        for example in corpus_dataset:
            doc_id = _first_present(example, ("_id", "docid", "id", "doc_id", "corpus-id", "corpus_id"))
            title = _first_present(example, ("title",))
            body = _first_present(example, ("text", "content", "passage"))
            doc_text = " ".join(part for part in (title, body) if part)
            if doc_id is not None and doc_text and doc_id not in doc_dict:
                doc_dict[doc_id] = doc_text

        query_dict: dict[str, str] = {}
        for example in queries_dataset:
            query_id = _first_present(example, ("_id", "queryid", "id", "query_id", "query-id"))
            query_text = _first_present(example, ("text", "query", "question"))
            if query_id is not None and query_text and query_id not in query_dict:
                query_dict[query_id] = f"{QUERY_INSTRUCTION}{query_text}"

        seen_corpus_ids: list[str] = []
        seen_query_ids: list[str] = []
        query_to_relevant_docs: dict[str, list[str]] = {}
        for example in dataset:
            query_id = _first_present(example, ("query-id", "query_id"))
            corpus_id = _first_present(example, ("corpus-id", "corpus_id"))
            score = float(example.get("score", 0.0))
            if query_id is None or corpus_id is None:
                continue
            if query_id not in seen_query_ids:
                seen_query_ids.append(query_id)
            if corpus_id not in seen_corpus_ids:
                seen_corpus_ids.append(corpus_id)
            if score > 0:
                query_to_relevant_docs.setdefault(query_id, []).append(corpus_id)

        corpus_ids = [corpus_id for corpus_id in seen_corpus_ids if corpus_id in doc_dict]
        documents = [doc_dict[corpus_id] for corpus_id in corpus_ids]
        corpus_id_to_idx = {corpus_id: idx for idx, corpus_id in enumerate(corpus_ids)}

        for query_id in seen_query_ids:
            query_text = query_dict.get(query_id)
            if query_text is None:
                continue
            queries.append(query_text)
            relevant = query_to_relevant_docs.get(query_id, [])
            first_relevant = next((doc_id for doc_id in relevant if doc_id in corpus_id_to_idx), None)
            relevant_doc_ids.append(corpus_id_to_idx[first_relevant] if first_relevant is not None else -1)

    elif {"query", "corpus"}.issubset(columns):
        doc_dict: dict[str, str] = {}
        for example in dataset:
            corpus = example["corpus"]
            if not isinstance(corpus, dict):
                continue
            for doc_id, doc_text in corpus.items():
                doc_id = str(doc_id)
                if doc_id not in doc_dict:
                    doc_dict[doc_id] = str(doc_text)

        corpus_ids = sorted(doc_dict)
        documents = [doc_dict[corpus_id] for corpus_id in corpus_ids]
        corpus_id_to_idx = {corpus_id: idx for idx, corpus_id in enumerate(corpus_ids)}

        for example in dataset:
            queries.append(f"{QUERY_INSTRUCTION}{str(example['query'])}")
            relevant = example.get("relevant_docs", [])
            if isinstance(relevant, list) and relevant:
                relevant_doc_ids.append(corpus_id_to_idx.get(str(relevant[0]), -1))
            elif relevant:
                relevant_doc_ids.append(corpus_id_to_idx.get(str(relevant), -1))
            else:
                relevant_doc_ids.append(-1)

    else:
        raise ValueError(f"Unsupported retrieval dataset format with columns: {sorted(columns)}")

    if not queries or not documents:
        raise ValueError(f"Failed to extract retrieval samples from {dataset_name}")

    logger.info(f"Prepared retrieval dataset with {len(queries)} queries and {len(documents)} documents")
    return queries, documents, relevant_doc_ids


def _extract_sts_samples(dataset_name: str, dataset):
    sentences1: list[str] = []
    sentences2: list[str] = []
    gold_scores: list[float] = []

    for example in dataset:
        if "sentence1" not in example or "sentence2" not in example:
            continue
        sentences1.append(str(example["sentence1"]))
        sentences2.append(str(example["sentence2"]))
        if "score" in example:
            gold_scores.append(float(example["score"]))
        elif "label" in example:
            gold_scores.append(float(example["label"]))
        else:
            gold_scores.append(0.0)

    if not sentences1:
        raise ValueError(f"Failed to extract STS samples from {dataset_name}")

    logger.info(f"Prepared STS dataset with {len(sentences1)} sentence pairs")
    return sentences1, sentences2, gold_scores


def _load_reference_backbone(model_name: str):
    logger.info(f"Loading HF reference model {model_name}")
    reference_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).eval()
    return reference_model.roberta if hasattr(reference_model, "roberta") else reference_model


def _encode_dense_hf(
    texts: Sequence[str],
    host_batch_size: int,
    model_args,
    backbone,
    *,
    desc: str,
) -> torch.Tensor:
    embeddings = []
    for start in tqdm(range(0, len(texts), host_batch_size), desc=desc):
        batch = list(texts[start : start + host_batch_size])
        encoded = model_args.encode_prompts(batch)
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        token_type_ids = encoded.get("token_type_ids", torch.zeros_like(input_ids))

        with torch.no_grad():
            last_hidden_state = backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=None,
                return_dict=True,
            ).last_hidden_state.to(torch.float32)

        embeddings.append(_mean_pool(last_hidden_state, attention_mask).cpu())

    return torch.cat(embeddings, dim=0)


def _encode_dense_ttnn(
    texts: Sequence[str],
    host_batch_size: int,
    model_args,
    generator_model: BgeM3ForEmbedding,
    *,
    desc: str,
) -> torch.Tensor:
    embeddings = []
    for start in tqdm(range(0, len(texts), host_batch_size), desc=desc):
        batch = list(texts[start : start + host_batch_size])
        encoded = model_args.encode_prompts(batch)
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        token_type_ids = encoded.get("token_type_ids", torch.zeros_like(input_ids))
        seq_len = input_ids.shape[1]

        last_hidden_state = generator_model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        last_hidden_state = _crop_hidden_states(last_hidden_state, seq_len).to(torch.float32)
        embeddings.append(_mean_pool(last_hidden_state, attention_mask).cpu())

    return torch.cat(embeddings, dim=0)


def _calculate_recall_at_k(similarities: np.ndarray, relevant_doc_ids: Sequence[int], k: int) -> float:
    recall_scores = []
    for i, relevant_id in enumerate(relevant_doc_ids):
        if relevant_id < 0:
            continue
        top_k_indices = np.argsort(similarities[i])[::-1][:k]
        recall_scores.append(1.0 if relevant_id in top_k_indices else 0.0)
    return float(np.mean(recall_scores)) if recall_scores else 0.0


def _calculate_ndcg_at_k(similarities: np.ndarray, relevant_doc_ids: Sequence[int], k: int) -> float:
    ndcg_scores = []
    for i, relevant_id in enumerate(relevant_doc_ids):
        if relevant_id < 0:
            continue
        top_k_indices = np.argsort(similarities[i])[::-1][:k]
        if relevant_id in top_k_indices:
            rank = int(np.where(top_k_indices == relevant_id)[0][0]) + 1
            ndcg_scores.append(1.0 / np.log2(rank + 1))
        else:
            ndcg_scores.append(0.0)
    return float(np.mean(ndcg_scores)) if ndcg_scores else 0.0


def _build_generator_model(
    device,
    model_name: str,
    sequence_length: int,
    host_batch_size: int,
    tt_data_parallel: int,
) -> BgeM3ForEmbedding:
    generator_model = BgeM3ForEmbedding(
        device=device,
        max_batch_size=host_batch_size,
        max_seq_len=sequence_length,
        tt_data_parallel=tt_data_parallel,
        dtype=ttnn.bfloat8_b,
        model_name=model_name,
    )
    generator_model._initialize_model()
    return generator_model


def _evaluate_retrieval_batch(
    device,
    model_name: str,
    sequence_length: int,
    queries: Sequence[str],
    documents: Sequence[str],
    relevant_doc_ids: Sequence[int],
    per_chip_batch_size: int,
    tt_data_parallel: int,
    backbone,
):
    host_batch_size = per_chip_batch_size * tt_data_parallel
    logger.info(
        "Evaluating retrieval with per_chip_batch_size={} host_batch_size={} tt_data_parallel={}",
        per_chip_batch_size,
        host_batch_size,
        tt_data_parallel,
    )

    generator_model = _build_generator_model(device, model_name, sequence_length, host_batch_size, tt_data_parallel)
    model_args = (
        generator_model.model_args_list[0]
        if generator_model.model_args_list is not None
        else generator_model.model_args
    )
    assert model_args is not None

    ttnn_query_embeddings = _encode_dense_ttnn(
        queries, host_batch_size, model_args, generator_model, desc=f"TTNN queries bs={host_batch_size}"
    )
    ttnn_doc_embeddings = _encode_dense_ttnn(
        documents, host_batch_size, model_args, generator_model, desc=f"TTNN docs bs={host_batch_size}"
    )

    ttnn_similarities = torch.mm(
        _normalize_embeddings(ttnn_query_embeddings), _normalize_embeddings(ttnn_doc_embeddings).t()
    ).numpy()

    if ENABLE_HF_REFERENCE:
        ref_query_embeddings = _encode_dense_hf(
            queries, host_batch_size, model_args, backbone, desc=f"HF queries bs={host_batch_size}"
        )
        ref_doc_embeddings = _encode_dense_hf(
            documents, host_batch_size, model_args, backbone, desc=f"HF docs bs={host_batch_size}"
        )
        ref_similarities = torch.mm(
            _normalize_embeddings(ref_query_embeddings), _normalize_embeddings(ref_doc_embeddings).t()
        ).numpy()
        ref_metrics = {
            "recall@10": _calculate_recall_at_k(ref_similarities, relevant_doc_ids, k=10),
            "ndcg@10": _calculate_ndcg_at_k(ref_similarities, relevant_doc_ids, k=10),
            "recall@100": _calculate_recall_at_k(ref_similarities, relevant_doc_ids, k=100),
            "ndcg@100": _calculate_ndcg_at_k(ref_similarities, relevant_doc_ids, k=100),
        }
        alignment = {
            "queries": _mean_embedding_alignment(ref_query_embeddings, ttnn_query_embeddings),
            "documents": _mean_embedding_alignment(ref_doc_embeddings, ttnn_doc_embeddings),
        }
    else:
        logger.info("HF reference evaluation is disabled; running TT-only retrieval batch.")
        ref_metrics = {
            "recall@10": float("nan"),
            "ndcg@10": float("nan"),
            "recall@100": float("nan"),
            "ndcg@100": float("nan"),
        }
        alignment = {"queries": float("nan"), "documents": float("nan")}

    result = {
        "per_chip_batch_size": per_chip_batch_size,
        "host_batch_size": host_batch_size,
        "pytorch": ref_metrics,
        "ttnn": {
            "recall@10": _calculate_recall_at_k(ttnn_similarities, relevant_doc_ids, k=10),
            "ndcg@10": _calculate_ndcg_at_k(ttnn_similarities, relevant_doc_ids, k=10),
            "recall@100": _calculate_recall_at_k(ttnn_similarities, relevant_doc_ids, k=100),
            "ndcg@100": _calculate_ndcg_at_k(ttnn_similarities, relevant_doc_ids, k=100),
        },
        "alignment": alignment,
    }

    logger.info(
        "Retrieval bs/chip={} host_bs={} query_align={:.4f} doc_align={:.4f}",
        per_chip_batch_size,
        host_batch_size,
        result["alignment"]["queries"],
        result["alignment"]["documents"],
    )
    logger.info(
        "HF  recall@10={:.4f} ndcg@10={:.4f} recall@100={:.4f} ndcg@100={:.4f}",
        result["pytorch"]["recall@10"],
        result["pytorch"]["ndcg@10"],
        result["pytorch"]["recall@100"],
        result["pytorch"]["ndcg@100"],
    )
    logger.info(
        "TT  recall@10={:.4f} ndcg@10={:.4f} recall@100={:.4f} ndcg@100={:.4f}",
        result["ttnn"]["recall@10"],
        result["ttnn"]["ndcg@10"],
        result["ttnn"]["recall@100"],
        result["ttnn"]["ndcg@100"],
    )

    return result


def _evaluate_sts_batch(
    device,
    model_name: str,
    sequence_length: int,
    sentences1: Sequence[str],
    sentences2: Sequence[str],
    gold_scores: Sequence[float],
    per_chip_batch_size: int,
    tt_data_parallel: int,
    backbone,
):
    host_batch_size = per_chip_batch_size * tt_data_parallel
    logger.info(
        "Evaluating STS with per_chip_batch_size={} host_batch_size={} tt_data_parallel={}",
        per_chip_batch_size,
        host_batch_size,
        tt_data_parallel,
    )

    generator_model = _build_generator_model(device, model_name, sequence_length, host_batch_size, tt_data_parallel)
    model_args = (
        generator_model.model_args_list[0]
        if generator_model.model_args_list is not None
        else generator_model.model_args
    )
    assert model_args is not None

    ttnn_emb1 = _encode_dense_ttnn(
        sentences1, host_batch_size, model_args, generator_model, desc=f"TTNN sentence1 bs={host_batch_size}"
    )
    ttnn_emb2 = _encode_dense_ttnn(
        sentences2, host_batch_size, model_args, generator_model, desc=f"TTNN sentence2 bs={host_batch_size}"
    )

    ttnn_similarities = (_normalize_embeddings(ttnn_emb1) * _normalize_embeddings(ttnn_emb2)).sum(dim=1).numpy()

    ttnn_spearman, _ = spearmanr(ttnn_similarities, gold_scores)
    if ENABLE_HF_REFERENCE:
        ref_emb1 = _encode_dense_hf(
            sentences1, host_batch_size, model_args, backbone, desc=f"HF sentence1 bs={host_batch_size}"
        )
        ref_emb2 = _encode_dense_hf(
            sentences2, host_batch_size, model_args, backbone, desc=f"HF sentence2 bs={host_batch_size}"
        )
        ref_similarities = (_normalize_embeddings(ref_emb1) * _normalize_embeddings(ref_emb2)).sum(dim=1).numpy()
        ref_spearman, _ = spearmanr(ref_similarities, gold_scores)
        alignment = {
            "sentence1": _mean_embedding_alignment(ref_emb1, ttnn_emb1),
            "sentence2": _mean_embedding_alignment(ref_emb2, ttnn_emb2),
        }
    else:
        logger.info("HF reference evaluation is disabled; running TT-only STS batch.")
        ref_spearman = float("nan")
        alignment = {"sentence1": float("nan"), "sentence2": float("nan")}

    result = {
        "per_chip_batch_size": per_chip_batch_size,
        "host_batch_size": host_batch_size,
        "pytorch": {"spearman": float(ref_spearman)},
        "ttnn": {"spearman": float(ttnn_spearman)},
        "alignment": alignment,
    }

    logger.info(
        "STS bs/chip={} host_bs={} sent1_align={:.4f} sent2_align={:.4f}",
        per_chip_batch_size,
        host_batch_size,
        result["alignment"]["sentence1"],
        result["alignment"]["sentence2"],
    )
    logger.info(
        "HF  spearman={:.4f} | TT spearman={:.4f} | delta={:+.4f}",
        result["pytorch"]["spearman"],
        result["ttnn"]["spearman"],
        result["ttnn"]["spearman"] - result["pytorch"]["spearman"],
    )

    return result


def run_mteb_evaluation(
    device,
    *,
    dataset_name: str,
    model_name: str = DEFAULT_MODEL_NAME,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    max_samples: int | None = 100,
    per_chip_batch_sizes: Sequence[int] = DEFAULT_PER_CHIP_BATCH_SIZES,
    tt_data_parallel: int = 2,
):
    test_dataset = _load_mteb_dataset(dataset_name, split="test", max_samples=max_samples)
    task_type = _detect_task_type(test_dataset)
    backbone = _load_reference_backbone(model_name) if ENABLE_HF_REFERENCE else None

    logger.info(
        "Running {} evaluation for {} with sequence_length={} and per-chip batch sizes {}",
        task_type,
        dataset_name,
        sequence_length,
        list(per_chip_batch_sizes),
    )

    if tt_data_parallel != device.get_num_devices():
        raise ValueError(
            f"Current BGE-M3 generator DP path expects tt_data_parallel == num_devices. "
            f"Got tt_data_parallel={tt_data_parallel}, num_devices={device.get_num_devices()}"
        )

    if task_type == "retrieval":
        queries, documents, relevant_doc_ids = _extract_retrieval_samples(dataset_name, test_dataset)
        results = [
            _evaluate_retrieval_batch(
                device,
                model_name,
                sequence_length,
                queries,
                documents,
                relevant_doc_ids,
                per_chip_batch_size,
                tt_data_parallel,
                backbone,
            )
            for per_chip_batch_size in per_chip_batch_sizes
        ]
    else:
        sentences1, sentences2, gold_scores = _extract_sts_samples(dataset_name, test_dataset)
        results = [
            _evaluate_sts_batch(
                device,
                model_name,
                sequence_length,
                sentences1,
                sentences2,
                gold_scores,
                per_chip_batch_size,
                tt_data_parallel,
                backbone,
            )
            for per_chip_batch_size in per_chip_batch_sizes
        ]

    logger.info("=" * 80)
    logger.info("Batch sweep summary")
    logger.info("=" * 80)
    for result in results:
        if task_type == "retrieval":
            logger.info(
                "bs/chip={} host_bs={} | HF r@10={:.4f} TT r@10={:.4f} | HF ndcg@10={:.4f} TT ndcg@10={:.4f}",
                result["per_chip_batch_size"],
                result["host_batch_size"],
                result["pytorch"]["recall@10"],
                result["ttnn"]["recall@10"],
                result["pytorch"]["ndcg@10"],
                result["ttnn"]["ndcg@10"],
            )
        else:
            logger.info(
                "bs/chip={} host_bs={} | HF spearman={:.4f} TT spearman={:.4f}",
                result["per_chip_batch_size"],
                result["host_batch_size"],
                result["pytorch"]["spearman"],
                result["ttnn"]["spearman"],
            )

    return {"task_type": task_type, "results": results}


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-name", default="mteb/ArguAna")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--sequence-length", type=int, default=DEFAULT_SEQUENCE_LENGTH)
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--tt-data-parallel", type=int, default=2)
    parser.add_argument("--mesh-rows", type=int, default=1)
    parser.add_argument("--mesh-cols", type=int, default=2)
    parser.add_argument("--per-chip-batch-sizes", type=int, nargs="+", default=list(DEFAULT_PER_CHIP_BATCH_SIZES))
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    logger.info(f"TT_VISIBLE_DEVICES={os.environ.get('TT_VISIBLE_DEVICES', '<unset>')}")

    original_default_device = ttnn.GetDefaultDevice()
    with ttnn.create_mesh_device(mesh_shape=ttnn.MeshShape(args.mesh_rows, args.mesh_cols)) as mesh_device:
        logger.info(
            "Opened mesh device with num_devices={} and grid={}",
            mesh_device.get_num_devices(),
            mesh_device.compute_with_storage_grid_size(),
        )
        try:
            ttnn.SetDefaultDevice(mesh_device)
            run_mteb_evaluation(
                mesh_device,
                dataset_name=args.dataset_name,
                model_name=args.model_name,
                sequence_length=args.sequence_length,
                max_samples=args.max_samples,
                per_chip_batch_sizes=args.per_chip_batch_sizes,
                tt_data_parallel=args.tt_data_parallel,
            )
        finally:
            ttnn.SetDefaultDevice(original_default_device)
