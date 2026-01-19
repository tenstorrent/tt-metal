# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os
from itertools import cycle, islice

import numpy as np
import pytest
import torch
import transformers
from loguru import logger
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

import ttnn
from models.demos.bge_large_en.runner.performant_runner import BGEPerformantRunner
from models.demos.sentence_bert.reference.sentence_bert import custom_extended_mask
from models.demos.wormhole.bge_large_en.ttnn.common import BGE_L1_SMALL_SIZE, BGE_SEQ_LENGTH

# Instruction prefix recommended for BGE retrieval tasks
RETRIEVAL_INSTRUCTION = "Represent this sentence for searching relevant passages: "


def compute_ttnn_embeddings(
    sentences, model_name, device, model_location_generator, batch_size=8, use_instruction=True
):
    logger.info("Loading tokenizer...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

    # Add instruction prefix for retrieval tasks
    if use_instruction:
        sentences = [RETRIEVAL_INSTRUCTION + s for s in sentences]

    all_embeddings = []
    all_sentences = []
    sentence_bert_module = None

    for i in tqdm(range(0, len(sentences), batch_size), desc="Batches"):
        batch_sentences = sentences[i : i + batch_size]
        logger.info(f"Encoding batch {i//batch_size + 1} with {len(batch_sentences)} sentences...")

        # If batch is smaller than batch_size, repeat to fill batch_size
        orig_batch_size = len(batch_sentences)
        if orig_batch_size < batch_size:
            batch_sentences = list(islice(cycle(batch_sentences), batch_size))

        encoded_input = tokenizer(
            batch_sentences, padding="max_length", max_length=BGE_SEQ_LENGTH, truncation=True, return_tensors="pt"
        )
        input_ids = encoded_input["input_ids"]
        attention_mask = encoded_input["attention_mask"]
        extended_mask = custom_extended_mask(attention_mask, dtype=torch.bfloat16)
        token_type_ids = encoded_input["token_type_ids"]
        position_ids = torch.arange(0, input_ids.shape[-1], dtype=torch.int64).unsqueeze(dim=0)

        if sentence_bert_module is None:
            logger.info("Initializing trace...")
            sentence_bert_module = BGEPerformantRunner(
                device=device,
                model_location_generator=model_location_generator,
                input_ids=input_ids,
                extended_mask=extended_mask,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                model_name=model_name,
            )
            sentence_bert_module._capture_bge_trace_2cqs()

        ttnn_output = sentence_bert_module.run(input_ids, token_type_ids, position_ids, extended_mask, attention_mask)
        logger.info("Running inference on TTNN model for current batch...")
        ttnn_output = ttnn.to_torch(ttnn_output, mesh_composer=sentence_bert_module.runner_infra.output_mesh_composer)

        # Always slice to the original batch size (before padding) to match number of sentences
        all_embeddings.append(ttnn_output[:orig_batch_size])
        all_sentences.extend(sentences[i : i + orig_batch_size])

    all_embeddings = torch.cat(all_embeddings, dim=0)
    logger.info("All embeddings computed.")
    return all_embeddings, all_sentences, sentence_bert_module


def load_knowledge_base(kb_file="knowledge_base.txt"):
    kb_sentences = []
    filepath = os.path.join(os.path.dirname(__file__), kb_file)

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                # Remove inline comments
                for comment_token in ["#", "//"]:
                    if comment_token in line:
                        line = line.split(comment_token, 1)[0].strip()
                if line:
                    kb_sentences.append(line)
    except FileNotFoundError:
        logger.error(f"Knowledge base file not found: {filepath}")
        raise
    except Exception as e:
        logger.error(f"Error loading knowledge base: {e}")
        raise

    return kb_sentences


def run_interactive_demo_inference(device, model_name, sequence_length, batch_size, kb_file, model_location_generator):
    batch_size = batch_size * device.get_num_devices()
    logger.info(f"Loading knowledge base from {kb_file}...")
    kb_sentences = load_knowledge_base(kb_file)

    # Compute embeddings for knowledge base (without instruction prefix for documents)
    kb_embeddings, kb_sentences, model_instance = compute_ttnn_embeddings(
        kb_sentences, model_name, device, model_location_generator, batch_size, use_instruction=False
    )

    logger.info("\n" + "=" * 80)
    logger.info("BGE-Large-EN-v1.5 Interactive Semantic Search Demo")
    logger.info("=" * 80)
    logger.info(f"Knowledge base loaded: {len(kb_sentences)} entries")
    logger.info("Note: Queries are automatically prefixed with retrieval instruction")
    logger.info("=" * 80 + "\n")

    # Initialize tokenizer once (outside the loop for efficiency)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

    while True:
        logger.info("Ready for semantic search. Please enter your query ('exit' to quit):")
        query = input("Query: ").strip()

        if query.lower() in {"exit", "quit"}:
            logger.info("Exiting interactive demo.")
            break

        # Add instruction prefix to query (recommended for BGE retrieval)
        query_with_instruction = RETRIEVAL_INSTRUCTION + query

        # Pad query to batch_size by repeating (required for TTNN model)
        batch_sentences = list(islice(cycle([query_with_instruction]), batch_size))
        encoded_input = tokenizer(
            batch_sentences, padding="max_length", max_length=sequence_length, truncation=True, return_tensors="pt"
        )
        input_ids = encoded_input["input_ids"]
        attention_mask = encoded_input["attention_mask"]
        extended_mask = custom_extended_mask(attention_mask, dtype=torch.bfloat16)
        token_type_ids = encoded_input["token_type_ids"]
        position_ids = torch.arange(0, input_ids.shape[-1], dtype=torch.int64).unsqueeze(dim=0)

        ttnn_output = model_instance.run(input_ids, token_type_ids, position_ids, extended_mask, attention_mask)
        logger.info("Computing semantic similarity...")
        query_embeddings = ttnn.to_torch(ttnn_output, mesh_composer=model_instance.runner_infra.output_mesh_composer)

        # Slice to get only the first embedding (the actual query, not the padded duplicates)
        query_embedding = query_embeddings[0:1]
        similarities = cosine_similarity(query_embedding.detach().cpu().numpy(), kb_embeddings.detach().cpu().numpy())[
            0
        ]

        # Get top 3 results
        top_indices = np.argsort(similarities)[-3:][::-1]

        logger.info("\n" + "-" * 80)
        logger.info(f"Query: {query}")
        logger.info("-" * 80)
        for rank, idx in enumerate(top_indices, 1):
            logger.info(f"#{rank} (score: {similarities[idx]:.4f}): {kb_sentences[idx]}")
        logger.info("-" * 80 + "\n")


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": BGE_L1_SMALL_SIZE, "trace_region_size": 6434816, "num_command_queues": 2}],
    indirect=True,
)
@pytest.mark.parametrize(
    "model_name, sequence_length, batch_size, kb_file",
    [("BAAI/bge-large-en-v1.5", BGE_SEQ_LENGTH, 8, "knowledge_base.txt")],
)
def test_interactive_demo_inference(device, model_name, sequence_length, batch_size, kb_file, model_location_generator):
    return run_interactive_demo_inference(
        device, model_name, sequence_length, batch_size, kb_file, model_location_generator
    )


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": BGE_L1_SMALL_SIZE, "trace_region_size": 6434816, "num_command_queues": 2}],
    indirect=True,
)
@pytest.mark.parametrize(
    "model_name, sequence_length, device_batch_size, kb_file",
    [("BAAI/bge-large-en-v1.5", BGE_SEQ_LENGTH, 8, "knowledge_base.txt")],
)
def test_interactive_demo_inference_dp(
    mesh_device, model_name, sequence_length, device_batch_size, kb_file, model_location_generator
):
    return run_interactive_demo_inference(
        mesh_device, model_name, sequence_length, device_batch_size, kb_file, model_location_generator
    )
