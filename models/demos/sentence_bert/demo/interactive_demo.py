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
from models.demos.sentence_bert.reference.sentence_bert import custom_extended_mask
from models.demos.sentence_bert.runner.performant_runner import SentenceBERTPerformantRunner


def compute_ttnn_embeddings(sentences, model_name, device, batch_size=8):
    logger.info("Loading tokenizer...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    all_embeddings = []
    all_sentences = []
    num_batches = (len(sentences) + batch_size - 1) // batch_size
    sentence_bert_module = None
    for i in tqdm(range(0, len(sentences), batch_size), desc="Batches"):
        batch_sentences = sentences[i : i + batch_size]
        logger.info(f"Encoding batch {i//batch_size + 1} with {len(batch_sentences)} sentences...")
        # If batch is smaller than batch_size, repeat to fill batch_size
        orig_batch_size = len(batch_sentences)
        if orig_batch_size < batch_size:
            # Repeat sentences as needed to fill the batch, even if original batch is very small
            batch_sentences = list(islice(cycle(batch_sentences), batch_size))
        encoded_input = tokenizer(
            batch_sentences, padding="max_length", max_length=384, truncation=True, return_tensors="pt"
        )
        input_ids = encoded_input["input_ids"]
        attention_mask = encoded_input["attention_mask"]
        extended_mask = custom_extended_mask(attention_mask, dtype=torch.bfloat16)
        token_type_ids = encoded_input["token_type_ids"]
        position_ids = torch.arange(0, input_ids.shape[-1], dtype=torch.int64).unsqueeze(dim=0)
        if sentence_bert_module is None:
            logger.info("initialising trace..")
            sentence_bert_module = SentenceBERTPerformantRunner(
                device=device,
                input_ids=input_ids,
                extended_mask=extended_mask,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
            )
            sentence_bert_module._capture_sentencebert_trace_2cqs()

        ttnn_output = sentence_bert_module.run(input_ids, token_type_ids, position_ids, extended_mask, attention_mask)
        logger.info("Running inference on TTNN model for current batch...")
        ttnn_output = ttnn.to_torch(ttnn_output)
        # Always slice to the original batch size (before padding)embeddings = embeddings[:orig_batch_size]
        all_embeddings.append(ttnn_output)
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


@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 24576, "trace_region_size": 6434816, "num_command_queues": 2}], indirect=True
)
@pytest.mark.parametrize(
    "model_name, sequence_length, batch_size,kb_file",
    [("emrecan/bert-base-turkish-cased-mean-nli-stsb-tr", 384, 8, "knowledge_base.txt")],
)
def test_interactive_demo_inference(device, model_name, sequence_length, batch_size, kb_file):
    logger.info(f"Loading knowledge base from {kb_file}...")
    kb_sentences = load_knowledge_base(kb_file)
    kb_embeddings, kb_sentences, model_instance = compute_ttnn_embeddings(kb_sentences, model_name, device)
    # Example query (in Turkish): "Siparişim ne zaman teslim edilir?"
    # English translation: "When will my order be delivered?"
    # This matches the knowledge base entry about order delivery times. i.e Siparişim ne zaman kargoya verilecek?
    while 1:
        logger.info("Ready for semantic search. Please enter your query ('exit' to quit the demo):")
        query = input("Query: ").strip()
        if query.lower() in {"exit", "quit"}:
            logger.info("Exiting interactive demo.")
            break
        orig_batch_size = len([query])
        if orig_batch_size < batch_size:
            # Repeat sentences as needed to fill the batch, even if original batch is very small
            batch_sentences = list(islice(cycle([query]), batch_size))
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        encoded_input = tokenizer(
            batch_sentences, padding="max_length", max_length=sequence_length, truncation=True, return_tensors="pt"
        )
        input_ids = encoded_input["input_ids"]
        attention_mask = encoded_input["attention_mask"]
        extended_mask = custom_extended_mask(attention_mask, dtype=torch.bfloat16)
        token_type_ids = encoded_input["token_type_ids"]
        position_ids = torch.arange(0, input_ids.shape[-1], dtype=torch.int64).unsqueeze(dim=0)
        ttnn_output = model_instance.run(input_ids, token_type_ids, position_ids, extended_mask, attention_mask)
        logger.info("Running inference on TTNN model for current batch...")
        query_embeddings = ttnn.to_torch(ttnn_output)
        logger.info("Computing cosine similarities...")
        similarities = cosine_similarity(query_embeddings.detach().cpu().numpy(), kb_embeddings.detach().cpu().numpy())[
            0
        ]
        top_idx = np.argmax(similarities)
        logger.info(f"\tQuery: {query}")
        logger.info(f"Best match: {kb_sentences[top_idx]}")
        logger.info(f"Similarity score: {similarities[top_idx]:.4f}")
