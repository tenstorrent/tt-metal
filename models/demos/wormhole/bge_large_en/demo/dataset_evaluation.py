# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch
import transformers
from datasets import load_dataset
from loguru import logger
from scipy.stats import spearmanr
from tqdm import tqdm

import ttnn
from models.demos.bge_large_en.common import load_torch_model
from models.demos.bge_large_en.runner.performant_runner import BGEPerformantRunner
from models.demos.sentence_bert.reference.sentence_bert import BertModel, custom_extended_mask
from models.demos.wormhole.bge_large_en.ttnn.common import BGE_L1_SMALL_SIZE, BGE_SEQ_LENGTH

# BGE models work best with instruction prefix for retrieval tasks
RETRIEVAL_INSTRUCTION = "Represent this sentence for searching relevant passages: "


def pad_inputs_for_ttnn(input_ids, attention_mask, token_type_ids, extended_mask, position_ids, target_batch_size):
    """
    Pad inputs to match the target batch size required by TTNN model.

    Args:
        input_ids: Input token IDs tensor [batch_size, seq_len]
        attention_mask: Attention mask tensor [batch_size, seq_len]
        token_type_ids: Token type IDs tensor [batch_size, seq_len]
        extended_mask: Extended attention mask tensor [batch_size, 1, 1, seq_len]
        position_ids: Position IDs tensor [1, seq_len]
        target_batch_size: Target batch size to pad to

    Returns:
        Tuple of padded tensors and the actual batch size (before padding)
    """
    actual_batch_size = input_ids.shape[0]

    if actual_batch_size >= target_batch_size:
        return input_ids, attention_mask, token_type_ids, extended_mask, position_ids, actual_batch_size

    # Pad by repeating the last sample
    pad_size = target_batch_size - actual_batch_size
    last_sample_idx = actual_batch_size - 1

    # Pad input_ids, attention_mask, token_type_ids
    input_ids_padded = torch.cat(
        [input_ids, input_ids[last_sample_idx : last_sample_idx + 1].repeat(pad_size, 1)], dim=0
    )
    attention_mask_padded = torch.cat(
        [attention_mask, attention_mask[last_sample_idx : last_sample_idx + 1].repeat(pad_size, 1)], dim=0
    )
    token_type_ids_padded = torch.cat(
        [token_type_ids, token_type_ids[last_sample_idx : last_sample_idx + 1].repeat(pad_size, 1)], dim=0
    )

    # Pad extended_mask (shape: [batch_size, 1, 1, seq_len])
    extended_mask_padded = torch.cat(
        [extended_mask, extended_mask[last_sample_idx : last_sample_idx + 1].repeat(pad_size, 1, 1, 1)], dim=0
    )

    # position_ids doesn't need padding (it's [1, seq_len])

    return (
        input_ids_padded,
        attention_mask_padded,
        token_type_ids_padded,
        extended_mask_padded,
        position_ids,
        actual_batch_size,
    )


def load_mteb_dataset(dataset_name="mteb/ArguAna", split="test", max_samples=None):
    """
    Load an MTEB dataset from Hugging Face.

    Args:
        dataset_name: Name of the dataset (e.g., "mteb/ArguAna", "mteb/stsbenchmark-sts")
        split: Dataset split to use ("test", "dev", "train")
        max_samples: Maximum number of samples to use (None for all)

    Returns:
        Dataset object
    """
    logger.info(f"Loading dataset: {dataset_name} (split: {split})")
    try:
        # Load dataset from HuggingFace
        dataset = load_dataset(dataset_name, split=split)

        if max_samples is not None:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        logger.info(f"Loaded {len(dataset)} samples from {dataset_name}")
        return dataset
    except Exception as e:
        logger.error(f"Failed to load dataset {dataset_name}: {e}")
        logger.info("Note: Make sure 'datasets' library is installed: pip install datasets")
        raise


def evaluate_retrieval_task(
    device,
    model_name,
    sequence_length,
    dataset_name,
    max_samples,
    batch_size,
    model_location_generator,
):
    """
    Evaluate BGE model on a retrieval task (e.g., ArguAna).

    For retrieval tasks, we compute embeddings for queries and documents,
    then calculate retrieval metrics like nDCG@10, Recall@10, etc.
    """
    batch_size = batch_size * device.get_num_devices()
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    config = transformers.BertConfig.from_pretrained(model_name)

    # Set attention implementation if not set (required for reference model)
    if not hasattr(config, "_attn_implementation") or config._attn_implementation is None:
        config._attn_implementation = "eager"

    # Load dataset
    dataset = load_mteb_dataset(dataset_name, split="test", max_samples=max_samples)

    # Load reference PyTorch model
    logger.info("Loading PyTorch reference model...")
    reference_module = BertModel(config).to(torch.bfloat16)
    reference_module = load_torch_model(
        reference_module, target_prefix="", model_location_generator=model_location_generator
    )
    reference_module.eval()

    # Initialize TTNN model (will be created on first batch)
    ttnn_module = None
    ttnn_initialized_batch_size = None

    # Extract queries and documents based on dataset format
    queries = []
    documents = []
    relevant_doc_ids = []

    logger.info(f"Dataset columns: {dataset.column_names}")

    if "query-id" in dataset.column_names and "corpus-id" in dataset.column_names:
        # ArguAna test format: query-id, corpus-id, score
        # Need to load corpus and queries separately
        logger.info("Detected ArguAna test format (query-id, corpus-id, score)")

        try:
            # Try to load corpus and queries from separate splits
            # MTEB datasets typically have "corpus" and "queries" splits
            # Note: We load full corpus/queries (no max_samples limit) since we need all documents/queries
            # for proper evaluation, even if test set is limited
            corpus_dataset = load_dataset(dataset_name, split="corpus")
            queries_dataset = load_dataset(dataset_name, split="queries")

            logger.info(f"Loaded corpus with {len(corpus_dataset)} documents")
            logger.info(f"Loaded queries with {len(queries_dataset)} queries")
            logger.info(f"Corpus columns: {corpus_dataset.column_names}")
            logger.info(f"Queries columns: {queries_dataset.column_names}")

            # Build corpus dictionary - try different field name variations
            doc_dict = {}
            for example in corpus_dataset:
                # Try different possible field names for document ID
                doc_id = None
                for id_field in ["_id", "docid", "id", "doc_id", "corpus-id"]:
                    if id_field in example:
                        doc_id = str(example[id_field])
                        break

                # Try different possible field names for document text
                doc_text = None
                for text_field in ["text", "title", "content", "passage"]:
                    if text_field in example:
                        doc_text = str(example[text_field])
                        break

                if doc_id and doc_text and doc_id not in doc_dict:
                    doc_dict[doc_id] = RETRIEVAL_INSTRUCTION + doc_text

            # Build queries dictionary - try different field name variations
            query_dict = {}
            for example in queries_dataset:
                # Try different possible field names for query ID
                query_id = None
                for id_field in ["_id", "queryid", "id", "query_id", "query-id"]:
                    if id_field in example:
                        query_id = str(example[id_field])
                        break

                # Try different possible field names for query text
                query_text = None
                for text_field in ["text", "query", "question"]:
                    if text_field in example:
                        query_text = str(example[text_field])
                        break

                if query_id and query_text and query_id not in query_dict:
                    query_dict[query_id] = RETRIEVAL_INSTRUCTION + query_text

            # Process test set to get query-document pairs
            query_to_relevant_docs = {}
            for example in dataset:
                query_id = str(example.get("query-id", example.get("query_id", "")))
                corpus_id = str(example.get("corpus-id", example.get("corpus_id", "")))
                score = float(example.get("score", 0.0))

                if query_id not in query_to_relevant_docs:
                    query_to_relevant_docs[query_id] = []

                # Only consider positive pairs (score > 0)
                if score > 0:
                    query_to_relevant_docs[query_id].append(corpus_id)

            # Create ordered lists
            # Get all unique query IDs from test set
            unique_query_ids = sorted(set([str(ex.get("query-id", ex.get("query_id", ""))) for ex in dataset]))
            unique_corpus_ids = sorted(set([str(ex.get("corpus-id", ex.get("corpus_id", ""))) for ex in dataset]))

            # Build documents list (all documents that appear in test set)
            documents = [doc_dict.get(cid, "") for cid in unique_corpus_ids if cid in doc_dict]

            # Create mapping from corpus_id to index
            corpus_id_to_idx = {cid: idx for idx, cid in enumerate(unique_corpus_ids)}

            # Build queries list and relevant_doc_ids together to maintain alignment
            # Only include queries that exist in query_dict
            for qid in unique_query_ids:
                if qid in query_dict:
                    queries.append(query_dict[qid])
                    # Map relevant documents for this query
                    if qid in query_to_relevant_docs and query_to_relevant_docs[qid]:
                        # Get the first relevant document ID
                        rel_doc_id = query_to_relevant_docs[qid][0]
                        relevant_doc_ids.append(corpus_id_to_idx.get(rel_doc_id, 0))
                    else:
                        relevant_doc_ids.append(0)

            logger.info(f"Found {len(queries)} queries and {len(documents)} documents from test set")

        except Exception as e:
            logger.warning(f"Failed to load corpus/queries separately: {e}")
            logger.info("Falling back to reconstructing from test set...")
            # Fallback: reconstruct from test set (limited - only documents/queries in test set)
            doc_dict = {}
            query_dict = {}
            query_to_relevant_docs = {}

            # Collect all unique documents and queries from test set
            for example in dataset:
                query_id = str(example.get("query-id", example.get("query_id", "")))
                corpus_id = str(example.get("corpus-id", example.get("corpus_id", "")))
                score = float(example.get("score", 0.0))

                # Note: In this fallback, we don't have the actual text, so we can't do proper evaluation
                # This is a limitation - we'd need the full corpus/queries datasets
                if query_id not in query_dict:
                    query_dict[query_id] = f"query_{query_id}"  # Placeholder
                if corpus_id not in doc_dict:
                    doc_dict[corpus_id] = f"document_{corpus_id}"  # Placeholder

                if query_id not in query_to_relevant_docs:
                    query_to_relevant_docs[query_id] = []
                if score > 0:
                    query_to_relevant_docs[query_id].append(corpus_id)

            # Create ordered lists
            unique_query_ids = sorted(query_dict.keys())
            unique_corpus_ids = sorted(doc_dict.keys())

            documents = [RETRIEVAL_INSTRUCTION + doc_dict[cid] for cid in unique_corpus_ids]

            corpus_id_to_idx = {cid: idx for idx, cid in enumerate(unique_corpus_ids)}

            # Build queries and relevant_doc_ids together to maintain alignment
            for qid in unique_query_ids:
                queries.append(RETRIEVAL_INSTRUCTION + query_dict[qid])
                if qid in query_to_relevant_docs and query_to_relevant_docs[qid]:
                    rel_doc_id = query_to_relevant_docs[qid][0]
                    relevant_doc_ids.append(corpus_id_to_idx.get(rel_doc_id, 0))
                else:
                    relevant_doc_ids.append(0)

    elif "query" in dataset.column_names and "corpus" in dataset.column_names:
        # ArguAna format: query, corpus (dict), relevant_docs (list)
        logger.info("Detected ArguAna format (retrieval task)")
        doc_dict = {}

        # First pass: collect all documents from corpus
        for example in dataset:
            if isinstance(example["corpus"], dict):
                for doc_id, doc_text in example["corpus"].items():
                    doc_id_str = str(doc_id)
                    if doc_id_str not in doc_dict:
                        doc_dict[doc_id_str] = RETRIEVAL_INSTRUCTION + str(doc_text)

        # Second pass: extract queries and relevant docs
        for example in dataset:
            queries.append(RETRIEVAL_INSTRUCTION + str(example["query"]))
            if "relevant_docs" in example and example["relevant_docs"]:
                if isinstance(example["relevant_docs"], list) and len(example["relevant_docs"]) > 0:
                    rel_doc = str(example["relevant_docs"][0])
                else:
                    rel_doc = str(example["relevant_docs"]) if example["relevant_docs"] else "0"
                # Map document ID to index
                try:
                    relevant_doc_ids.append(int(rel_doc))
                except (ValueError, TypeError):
                    relevant_doc_ids.append(0)
            else:
                relevant_doc_ids.append(0)

        # Create documents list in order (assuming numeric IDs)
        if doc_dict:
            try:
                max_doc_id = max([int(k) for k in doc_dict.keys() if k.isdigit()], default=0)
                documents = [doc_dict.get(str(i), "") for i in range(max_doc_id + 1)]
            except ValueError:
                # If IDs are not numeric, use the dict values directly
                documents = list(doc_dict.values())
                # Remap relevant_doc_ids to indices
                id_to_idx = {doc_id: idx for idx, doc_id in enumerate(doc_dict.keys())}
                relevant_doc_ids = [id_to_idx.get(str(rid), 0) for rid in relevant_doc_ids]

    elif "sentence1" in dataset.column_names and "sentence2" in dataset.column_names:
        # STS format: sentence1, sentence2, score
        logger.info("Detected STS format (semantic similarity task)")
        for idx, example in enumerate(dataset):
            queries.append(RETRIEVAL_INSTRUCTION + str(example["sentence1"]))
            documents.append(RETRIEVAL_INSTRUCTION + str(example["sentence2"]))
            relevant_doc_ids.append(idx)  # Each query's relevant doc is at its index
    else:
        raise ValueError(f"Unsupported dataset format. Columns: {dataset.column_names}")

    logger.info(f"Processing {len(queries)} queries and {len(documents)} documents")

    # Generate embeddings for queries (PyTorch)
    logger.info("Generating query embeddings with PyTorch model...")
    ref_query_embeddings = []
    for i in tqdm(range(0, len(queries), batch_size), desc="PyTorch queries"):
        batch_queries = queries[i : i + batch_size]
        encoded_input = tokenizer(
            batch_queries,
            padding="max_length",
            max_length=sequence_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoded_input["input_ids"]
        attention_mask = encoded_input["attention_mask"]
        extended_mask = custom_extended_mask(attention_mask, dtype=torch.bfloat16)
        token_type_ids = encoded_input["token_type_ids"]
        position_ids = torch.arange(0, input_ids.shape[-1], dtype=torch.int64).unsqueeze(dim=0)

        with torch.no_grad():
            embeddings = reference_module(
                input_ids,
                extended_attention_mask=extended_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
            ).post_processed_output
            ref_query_embeddings.append(embeddings.cpu())

    ref_query_embeddings = torch.cat(ref_query_embeddings, dim=0)

    # Generate embeddings for documents (PyTorch)
    logger.info("Generating document embeddings with PyTorch model...")
    ref_doc_embeddings = []
    for i in tqdm(range(0, len(documents), batch_size), desc="PyTorch documents"):
        batch_docs = documents[i : i + batch_size]
        encoded_input = tokenizer(
            batch_docs,
            padding="max_length",
            max_length=sequence_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoded_input["input_ids"]
        attention_mask = encoded_input["attention_mask"]
        extended_mask = custom_extended_mask(attention_mask, dtype=torch.bfloat16)
        token_type_ids = encoded_input["token_type_ids"]
        position_ids = torch.arange(0, input_ids.shape[-1], dtype=torch.int64).unsqueeze(dim=0)

        with torch.no_grad():
            embeddings = reference_module(
                input_ids,
                extended_attention_mask=extended_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
            ).post_processed_output
            ref_doc_embeddings.append(embeddings.cpu())

    ref_doc_embeddings = torch.cat(ref_doc_embeddings, dim=0)

    # Generate embeddings for queries (TTNN)
    logger.info("Generating query embeddings with TTNN model...")
    ttnn_query_embeddings = []
    for i in tqdm(range(0, len(queries), batch_size), desc="TTNN queries"):
        batch_queries = queries[i : i + batch_size]
        encoded_input = tokenizer(
            batch_queries,
            padding="max_length",
            max_length=sequence_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoded_input["input_ids"]
        attention_mask = encoded_input["attention_mask"]
        extended_mask = custom_extended_mask(attention_mask, dtype=torch.bfloat16)
        token_type_ids = encoded_input["token_type_ids"]
        position_ids = torch.arange(0, input_ids.shape[-1], dtype=torch.int64).unsqueeze(dim=0)

        if ttnn_module is None:
            # Initialize model with first batch (may need padding if batch_size < 8 per device)
            per_device_batch_size = 8
            target_batch_size = per_device_batch_size * device.get_num_devices()
            if input_ids.shape[0] < target_batch_size:
                input_ids, attention_mask, token_type_ids, extended_mask, position_ids, _ = pad_inputs_for_ttnn(
                    input_ids, attention_mask, token_type_ids, extended_mask, position_ids, target_batch_size
                )
            ttnn_initialized_batch_size = input_ids.shape[0]

            ttnn_module = BGEPerformantRunner(
                device=device,
                model_location_generator=model_location_generator,
                input_ids=input_ids,
                extended_mask=extended_mask,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                model_name=model_name,
            )
            ttnn_module._capture_bge_trace_2cqs()

        # Pad inputs if current batch is smaller than initialized batch size
        actual_batch_size = input_ids.shape[0]
        if actual_batch_size < ttnn_initialized_batch_size:
            (
                input_ids,
                attention_mask,
                token_type_ids,
                extended_mask,
                position_ids,
                actual_batch_size,
            ) = pad_inputs_for_ttnn(
                input_ids, attention_mask, token_type_ids, extended_mask, position_ids, ttnn_initialized_batch_size
            )

        ttnn_out = ttnn_module.run(input_ids, token_type_ids, position_ids, extended_mask, attention_mask)
        embeddings = ttnn.to_torch(
            ttnn_out, dtype=torch.float32, mesh_composer=ttnn_module.runner_infra.output_mesh_composer
        )
        # Slice to only keep actual samples (remove padding)
        embeddings = embeddings[:actual_batch_size]
        ttnn_query_embeddings.append(embeddings.cpu())

    ttnn_query_embeddings = torch.cat(ttnn_query_embeddings, dim=0)

    # Generate embeddings for documents (TTNN)
    logger.info("Generating document embeddings with TTNN model...")
    ttnn_doc_embeddings = []
    for i in tqdm(range(0, len(documents), batch_size), desc="TTNN documents"):
        batch_docs = documents[i : i + batch_size]
        encoded_input = tokenizer(
            batch_docs,
            padding="max_length",
            max_length=sequence_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoded_input["input_ids"]
        attention_mask = encoded_input["attention_mask"]
        extended_mask = custom_extended_mask(attention_mask, dtype=torch.bfloat16)
        token_type_ids = encoded_input["token_type_ids"]
        position_ids = torch.arange(0, input_ids.shape[-1], dtype=torch.int64).unsqueeze(dim=0)

        # Pad inputs if current batch is smaller than initialized batch size
        actual_batch_size = input_ids.shape[0]
        if actual_batch_size < ttnn_initialized_batch_size:
            (
                input_ids,
                attention_mask,
                token_type_ids,
                extended_mask,
                position_ids,
                actual_batch_size,
            ) = pad_inputs_for_ttnn(
                input_ids, attention_mask, token_type_ids, extended_mask, position_ids, ttnn_initialized_batch_size
            )

        ttnn_out = ttnn_module.run(input_ids, token_type_ids, position_ids, extended_mask, attention_mask)
        embeddings = ttnn.to_torch(
            ttnn_out, dtype=torch.float32, mesh_composer=ttnn_module.runner_infra.output_mesh_composer
        )
        # Slice to only keep actual samples (remove padding)
        embeddings = embeddings[:actual_batch_size]
        ttnn_doc_embeddings.append(embeddings.cpu())

    ttnn_doc_embeddings = torch.cat(ttnn_doc_embeddings, dim=0)

    # Calculate retrieval metrics
    logger.info("Calculating retrieval metrics...")

    # Normalize embeddings for cosine similarity
    ref_query_emb = ref_query_embeddings / torch.norm(ref_query_embeddings, dim=1, keepdim=True)
    ref_doc_emb = ref_doc_embeddings / torch.norm(ref_doc_embeddings, dim=1, keepdim=True)
    ttnn_query_emb = ttnn_query_embeddings / torch.norm(ttnn_query_embeddings, dim=1, keepdim=True)
    ttnn_doc_emb = ttnn_doc_embeddings / torch.norm(ttnn_doc_embeddings, dim=1, keepdim=True)

    # Calculate similarity scores
    ref_similarities = torch.mm(ref_query_emb, ref_doc_emb.t()).numpy()
    ttnn_similarities = torch.mm(ttnn_query_emb, ttnn_doc_emb.t()).numpy()

    # Calculate metrics
    def calculate_recall_at_k(similarities, relevant_doc_ids, k=10):
        """Calculate Recall@K metric."""
        recall_scores = []
        for i, relevant_id in enumerate(relevant_doc_ids):
            if relevant_id < 0:
                continue
            top_k_indices = np.argsort(similarities[i])[::-1][:k]
            recall = 1.0 if relevant_id in top_k_indices else 0.0
            recall_scores.append(recall)
        return np.mean(recall_scores) if recall_scores else 0.0

    def calculate_ndcg_at_k(similarities, relevant_doc_ids, k=10):
        """Calculate nDCG@K metric."""
        ndcg_scores = []
        for i, relevant_id in enumerate(relevant_doc_ids):
            if relevant_id < 0:
                continue
            top_k_indices = np.argsort(similarities[i])[::-1][:k]
            if relevant_id in top_k_indices:
                rank = np.where(top_k_indices == relevant_id)[0][0] + 1
                dcg = 1.0 / np.log2(rank + 1)
                idcg = 1.0 / np.log2(2)  # Ideal DCG for rank 1
                ndcg = dcg / idcg
            else:
                ndcg = 0.0
            ndcg_scores.append(ndcg)
        return np.mean(ndcg_scores) if ndcg_scores else 0.0

    # Calculate metrics for both models
    ref_recall_10 = calculate_recall_at_k(ref_similarities, relevant_doc_ids, k=10)
    ref_ndcg_10 = calculate_ndcg_at_k(ref_similarities, relevant_doc_ids, k=10)
    ref_recall_100 = calculate_recall_at_k(ref_similarities, relevant_doc_ids, k=100)
    ref_ndcg_100 = calculate_ndcg_at_k(ref_similarities, relevant_doc_ids, k=100)

    ttnn_recall_10 = calculate_recall_at_k(ttnn_similarities, relevant_doc_ids, k=10)
    ttnn_ndcg_10 = calculate_ndcg_at_k(ttnn_similarities, relevant_doc_ids, k=10)
    ttnn_recall_100 = calculate_recall_at_k(ttnn_similarities, relevant_doc_ids, k=100)
    ttnn_ndcg_100 = calculate_ndcg_at_k(ttnn_similarities, relevant_doc_ids, k=100)

    # Print results
    logger.info("=" * 80)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 80)
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Number of queries: {len(queries)}")
    logger.info(f"Number of documents: {len(documents)}")
    logger.info("")
    logger.info("PyTorch Reference Model Metrics:")
    logger.info(f"  Recall@10:  {ref_recall_10:.4f}")
    logger.info(f"  nDCG@10:    {ref_ndcg_10:.4f}")
    logger.info(f"  Recall@100: {ref_recall_100:.4f}")
    logger.info(f"  nDCG@100:   {ref_ndcg_100:.4f}")
    logger.info("")
    logger.info("TTNN Model Metrics:")
    logger.info(f"  Recall@10:  {ttnn_recall_10:.4f}")
    logger.info(f"  nDCG@10:    {ttnn_ndcg_10:.4f}")
    logger.info(f"  Recall@100: {ttnn_recall_100:.4f}")
    logger.info(f"  nDCG@100:   {ttnn_ndcg_100:.4f}")
    logger.info("")
    logger.info("Difference (TTNN - PyTorch):")
    logger.info(f"  Recall@10:  {ttnn_recall_10 - ref_recall_10:+.4f}")
    logger.info(f"  nDCG@10:    {ttnn_ndcg_10 - ref_ndcg_10:+.4f}")
    logger.info(f"  Recall@100: {ttnn_recall_100 - ref_recall_100:+.4f}")
    logger.info(f"  nDCG@100:   {ttnn_ndcg_100 - ref_ndcg_100:+.4f}")
    logger.info("=" * 80)

    # Cleanup
    if ttnn_module is not None:
        ttnn_module.release()

    return {
        "pytorch": {
            "recall@10": ref_recall_10,
            "ndcg@10": ref_ndcg_10,
            "recall@100": ref_recall_100,
            "ndcg@100": ref_ndcg_100,
        },
        "ttnn": {
            "recall@10": ttnn_recall_10,
            "ndcg@10": ttnn_ndcg_10,
            "recall@100": ttnn_recall_100,
            "ndcg@100": ttnn_ndcg_100,
        },
    }


def evaluate_semantic_similarity_task(
    device,
    model_name,
    sequence_length,
    dataset_name,
    max_samples,
    batch_size,
    model_location_generator,
):
    """
    Evaluate BGE model on a semantic similarity task (e.g., STSBenchmark).

    For semantic similarity tasks, we compute embeddings for sentence pairs,
    then calculate Spearman correlation between cosine similarity and gold scores.
    """
    batch_size = batch_size * device.get_num_devices()
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    config = transformers.BertConfig.from_pretrained(model_name)

    # Set attention implementation if not set (required for reference model)
    if not hasattr(config, "_attn_implementation") or config._attn_implementation is None:
        config._attn_implementation = "eager"

    # Load dataset
    dataset = load_mteb_dataset(dataset_name, split="test", max_samples=max_samples)

    # Load reference PyTorch model
    logger.info("Loading PyTorch reference model...")
    reference_module = BertModel(config).to(torch.bfloat16)
    reference_module = load_torch_model(
        reference_module, target_prefix="", model_location_generator=model_location_generator
    )
    reference_module.eval()

    # Initialize TTNN model (will be created on first batch)
    ttnn_module = None
    ttnn_initialized_batch_size = None

    # Extract sentence pairs and gold scores
    sentences1 = []
    sentences2 = []
    gold_scores = []

    for example in dataset:
        if "sentence1" in example and "sentence2" in example:
            sentences1.append(RETRIEVAL_INSTRUCTION + str(example["sentence1"]))
            sentences2.append(RETRIEVAL_INSTRUCTION + str(example["sentence2"]))
            if "score" in example:
                gold_scores.append(float(example["score"]))
            elif "label" in example:
                gold_scores.append(float(example["label"]))
            else:
                gold_scores.append(0.0)

    logger.info(f"Processing {len(sentences1)} sentence pairs")

    # Generate embeddings for sentence1 (PyTorch)
    logger.info("Generating embeddings for sentence1 with PyTorch model...")
    ref_emb1 = []
    for i in tqdm(range(0, len(sentences1), batch_size), desc="PyTorch sentence1"):
        batch = sentences1[i : i + batch_size]
        encoded_input = tokenizer(
            batch, padding="max_length", max_length=sequence_length, truncation=True, return_tensors="pt"
        )
        input_ids = encoded_input["input_ids"]
        attention_mask = encoded_input["attention_mask"]
        extended_mask = custom_extended_mask(attention_mask, dtype=torch.bfloat16)
        token_type_ids = encoded_input["token_type_ids"]
        position_ids = torch.arange(0, input_ids.shape[-1], dtype=torch.int64).unsqueeze(dim=0)

        with torch.no_grad():
            embeddings = reference_module(
                input_ids,
                extended_attention_mask=extended_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
            ).post_processed_output
            ref_emb1.append(embeddings.cpu())

    ref_emb1 = torch.cat(ref_emb1, dim=0)

    # Generate embeddings for sentence2 (PyTorch)
    logger.info("Generating embeddings for sentence2 with PyTorch model...")
    ref_emb2 = []
    for i in tqdm(range(0, len(sentences2), batch_size), desc="PyTorch sentence2"):
        batch = sentences2[i : i + batch_size]
        encoded_input = tokenizer(
            batch, padding="max_length", max_length=sequence_length, truncation=True, return_tensors="pt"
        )
        input_ids = encoded_input["input_ids"]
        attention_mask = encoded_input["attention_mask"]
        extended_mask = custom_extended_mask(attention_mask, dtype=torch.bfloat16)
        token_type_ids = encoded_input["token_type_ids"]
        position_ids = torch.arange(0, input_ids.shape[-1], dtype=torch.int64).unsqueeze(dim=0)

        with torch.no_grad():
            embeddings = reference_module(
                input_ids,
                extended_attention_mask=extended_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
            ).post_processed_output
            ref_emb2.append(embeddings.cpu())

    ref_emb2 = torch.cat(ref_emb2, dim=0)

    # Generate embeddings for sentence1 (TTNN)
    logger.info("Generating embeddings for sentence1 with TTNN model...")
    ttnn_emb1 = []
    for i in tqdm(range(0, len(sentences1), batch_size), desc="TTNN sentence1"):
        batch = sentences1[i : i + batch_size]
        encoded_input = tokenizer(
            batch, padding="max_length", max_length=sequence_length, truncation=True, return_tensors="pt"
        )
        input_ids = encoded_input["input_ids"]
        attention_mask = encoded_input["attention_mask"]
        extended_mask = custom_extended_mask(attention_mask, dtype=torch.bfloat16)
        token_type_ids = encoded_input["token_type_ids"]
        position_ids = torch.arange(0, input_ids.shape[-1], dtype=torch.int64).unsqueeze(dim=0)

        if ttnn_module is None:
            # Initialize model with first batch (may need padding if batch_size < 8 per device)
            per_device_batch_size = 8
            target_batch_size = per_device_batch_size * device.get_num_devices()
            if input_ids.shape[0] < target_batch_size:
                input_ids, attention_mask, token_type_ids, extended_mask, position_ids, _ = pad_inputs_for_ttnn(
                    input_ids, attention_mask, token_type_ids, extended_mask, position_ids, target_batch_size
                )
            ttnn_initialized_batch_size = input_ids.shape[0]

            ttnn_module = BGEPerformantRunner(
                device=device,
                model_location_generator=model_location_generator,
                input_ids=input_ids,
                extended_mask=extended_mask,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                model_name=model_name,
            )
            ttnn_module._capture_bge_trace_2cqs()

        # Pad inputs if current batch is smaller than initialized batch size
        actual_batch_size = input_ids.shape[0]
        if actual_batch_size < ttnn_initialized_batch_size:
            (
                input_ids,
                attention_mask,
                token_type_ids,
                extended_mask,
                position_ids,
                actual_batch_size,
            ) = pad_inputs_for_ttnn(
                input_ids, attention_mask, token_type_ids, extended_mask, position_ids, ttnn_initialized_batch_size
            )

        ttnn_out = ttnn_module.run(input_ids, token_type_ids, position_ids, extended_mask, attention_mask)
        embeddings = ttnn.to_torch(
            ttnn_out, dtype=torch.float32, mesh_composer=ttnn_module.runner_infra.output_mesh_composer
        )
        # Slice to only keep actual samples (remove padding)
        embeddings = embeddings[:actual_batch_size]
        ttnn_emb1.append(embeddings.cpu())

    ttnn_emb1 = torch.cat(ttnn_emb1, dim=0)

    # Generate embeddings for sentence2 (TTNN)
    logger.info("Generating embeddings for sentence2 with TTNN model...")
    ttnn_emb2 = []
    for i in tqdm(range(0, len(sentences2), batch_size), desc="TTNN sentence2"):
        batch = sentences2[i : i + batch_size]
        encoded_input = tokenizer(
            batch, padding="max_length", max_length=sequence_length, truncation=True, return_tensors="pt"
        )
        input_ids = encoded_input["input_ids"]
        attention_mask = encoded_input["attention_mask"]
        extended_mask = custom_extended_mask(attention_mask, dtype=torch.bfloat16)
        token_type_ids = encoded_input["token_type_ids"]
        position_ids = torch.arange(0, input_ids.shape[-1], dtype=torch.int64).unsqueeze(dim=0)

        # Pad inputs if current batch is smaller than initialized batch size
        actual_batch_size = input_ids.shape[0]
        if actual_batch_size < ttnn_initialized_batch_size:
            (
                input_ids,
                attention_mask,
                token_type_ids,
                extended_mask,
                position_ids,
                actual_batch_size,
            ) = pad_inputs_for_ttnn(
                input_ids, attention_mask, token_type_ids, extended_mask, position_ids, ttnn_initialized_batch_size
            )

        ttnn_out = ttnn_module.run(input_ids, token_type_ids, position_ids, extended_mask, attention_mask)
        embeddings = ttnn.to_torch(
            ttnn_out, dtype=torch.float32, mesh_composer=ttnn_module.runner_infra.output_mesh_composer
        )
        # Slice to only keep actual samples (remove padding)
        embeddings = embeddings[:actual_batch_size]
        ttnn_emb2.append(embeddings.cpu())

    ttnn_emb2 = torch.cat(ttnn_emb2, dim=0)

    # Calculate cosine similarities
    logger.info("Calculating cosine similarities...")

    # Normalize embeddings
    ref_emb1_norm = ref_emb1 / torch.norm(ref_emb1, dim=1, keepdim=True)
    ref_emb2_norm = ref_emb2 / torch.norm(ref_emb2, dim=1, keepdim=True)
    ttnn_emb1_norm = ttnn_emb1 / torch.norm(ttnn_emb1, dim=1, keepdim=True)
    ttnn_emb2_norm = ttnn_emb2 / torch.norm(ttnn_emb2, dim=1, keepdim=True)

    # Calculate cosine similarity for each pair
    ref_similarities = (ref_emb1_norm * ref_emb2_norm).sum(dim=1).numpy()
    ttnn_similarities = (ttnn_emb1_norm * ttnn_emb2_norm).sum(dim=1).numpy()

    # Calculate Spearman correlation
    ref_spearman, _ = spearmanr(ref_similarities, gold_scores)
    ttnn_spearman, _ = spearmanr(ttnn_similarities, gold_scores)

    # Print results
    logger.info("=" * 80)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 80)
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Number of sentence pairs: {len(sentences1)}")
    logger.info("")
    logger.info("PyTorch Reference Model Metrics:")
    logger.info(f"  Spearman Correlation: {ref_spearman:.4f}")
    logger.info("")
    logger.info("TTNN Model Metrics:")
    logger.info(f"  Spearman Correlation: {ttnn_spearman:.4f}")
    logger.info("")
    logger.info("Difference (TTNN - PyTorch):")
    logger.info(f"  Spearman Correlation: {ttnn_spearman - ref_spearman:+.4f}")
    logger.info("=" * 80)

    # Cleanup
    if ttnn_module is not None:
        ttnn_module.release()

    return {
        "pytorch": {"spearman": ref_spearman},
        "ttnn": {"spearman": ttnn_spearman},
    }


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": BGE_L1_SMALL_SIZE, "trace_region_size": 6434816, "num_command_queues": 2}],
    indirect=True,
)
@pytest.mark.parametrize(
    "model_name, sequence_length, dataset_name, max_samples, batch_size",
    [
        ("BAAI/bge-large-en-v1.5", BGE_SEQ_LENGTH, "mteb/ArguAna", 100, 8),
    ],
)
def test_bge_retrieval_evaluation(
    device, model_name, sequence_length, dataset_name, max_samples, batch_size, model_location_generator
):
    """
    Evaluate BGE model on MTEB retrieval task (ArguAna).

    This test downloads the ArguAna dataset from Hugging Face and evaluates
    both PyTorch and TTNN models on retrieval metrics (Recall@K, nDCG@K).

    Reference: https://huggingface.co/BAAI/bge-large-en-v1.5
    """
    return evaluate_retrieval_task(
        device=device,
        model_name=model_name,
        sequence_length=sequence_length,
        dataset_name=dataset_name,
        max_samples=max_samples,
        batch_size=batch_size,
        model_location_generator=model_location_generator,
    )


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": BGE_L1_SMALL_SIZE, "trace_region_size": 6434816, "num_command_queues": 2}],
    indirect=True,
)
@pytest.mark.parametrize(
    "model_name, sequence_length, dataset_name, max_samples, batch_size",
    [
        ("BAAI/bge-large-en-v1.5", BGE_SEQ_LENGTH, "mteb/stsbenchmark-sts", 100, 8),
    ],
)
def test_bge_sts_evaluation(
    device, model_name, sequence_length, dataset_name, max_samples, batch_size, model_location_generator
):
    """
    Evaluate BGE model on MTEB semantic similarity task (STSBenchmark).

    This test downloads the STSBenchmark dataset from Hugging Face and evaluates
    both PyTorch and TTNN models using Spearman correlation.

    Reference: https://huggingface.co/BAAI/bge-large-en-v1.5
    """
    return evaluate_semantic_similarity_task(
        device=device,
        model_name=model_name,
        sequence_length=sequence_length,
        dataset_name=dataset_name,
        max_samples=max_samples,
        batch_size=batch_size,
        model_location_generator=model_location_generator,
    )


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": BGE_L1_SMALL_SIZE, "trace_region_size": 6434816, "num_command_queues": 2}],
    indirect=True,
)
@pytest.mark.parametrize(
    "model_name, sequence_length, dataset_name, max_samples, batch_size",
    [
        ("BAAI/bge-large-en-v1.5", BGE_SEQ_LENGTH, "mteb/ArguAna", 100, 8),
    ],
)
def test_bge_retrieval_evaluation_dp(
    mesh_device, model_name, sequence_length, dataset_name, max_samples, batch_size, model_location_generator
):
    """
    Evaluate BGE model on MTEB retrieval task (ArguAna) with data parallelism.
    """
    return evaluate_retrieval_task(
        device=mesh_device,
        model_name=model_name,
        sequence_length=sequence_length,
        dataset_name=dataset_name,
        max_samples=max_samples,
        batch_size=batch_size,
        model_location_generator=model_location_generator,
    )
