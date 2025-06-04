# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from models.experimental.sentence_bert.reference.sentence_bert import custom_extended_mask
import transformers
import torch
import ttnn
import pytest
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
from loguru import logger
from tqdm import tqdm
from itertools import islice, cycle
from models.experimental.sentence_bert.tests.sentence_bert_e2e_performant import SentenceBERTrace2CQ


def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def compute_ttnn_embeddings(sentences, model_name, device, batch_size=8):
    # transformers_model = transformers.AutoModel.from_pretrained(model_name).eval()
    # config = transformers.BertConfig.from_pretrained(model_name)
    logger.info("Loading tokenizer...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    all_embeddings = []
    all_sentences = []
    num_batches = (len(sentences) + batch_size - 1) // batch_size
    sentence_bert_trace_2cq = None
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
        if sentence_bert_trace_2cq is None:
            logger.info("initialising trace..")
            sentence_bert_trace_2cq = SentenceBERTrace2CQ()

            sentence_bert_trace_2cq.initialize_sentence_bert_trace_2cqs_inference(
                device,
                input_ids,
                extended_mask,
                token_type_ids,
                position_ids,
                device_batch_size=batch_size,
                weight_dtype=ttnn.bfloat8_b,
                sequence_length=384,
            )

        ttnn_output = sentence_bert_trace_2cq.run(input_ids, iter=0)
        print("ttnn output is", ttnn_output.shape, ttnn_output.layout)
        logger.info("Running inference on TTNN model for current batch...")
        ttnn_output = ttnn.to_torch(ttnn_output).squeeze(dim=1)
        embeddings = mean_pooling(ttnn_output, attention_mask)
        # Always slice to the original batch size (before padding)embeddings = embeddings[:orig_batch_size]
        all_embeddings.append(embeddings)
        all_sentences.extend(sentences[i : i + orig_batch_size])
    all_embeddings = torch.cat(all_embeddings, dim=0)
    logger.info("All embeddings computed.")
    return all_embeddings, all_sentences, sentence_bert_trace_2cq


@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 24576, "trace_region_size": 6434816, "num_command_queues": 2}], indirect=True
)
def test_semantic_search_with_ttnn(device, use_program_cache):
    model_name = "emrecan/bert-base-turkish-cased-mean-nli-stsb-tr"
    kb_file = "knowledge_base.txt"
    logger.info(f"Loading knowledge base from {kb_file}...")
    kb_sentences = []
    filepath = os.path.join(os.path.dirname(__file__), kb_file)
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # Remove inline comments (e.g., anything after # or //)
            for comment_token in ["#", "//"]:
                if comment_token in line:
                    line = line.split(comment_token, 1)[0].strip()
            # Skip empty lines
            if line:
                kb_sentences.append(line)
    logger.info(f"Loaded {len(kb_sentences)} sentences from knowledge base.")
    kb_embeddings, kb_sentences, model_instance = compute_ttnn_embeddings(kb_sentences, model_name, device)
    # Example query (in Turkish): "Siparişim ne zaman teslim edilir?"
    # English translation: "When will my order be delivered?"
    # This matches the knowledge base entry about order delivery times. i.e Siparişim ne zaman kargoya verilecek?
    while 1:
        logger.info("Ready for semantic search. Please enter your query:")
        query = input("Query: ").strip()
        orig_batch_size = len([query])
        if orig_batch_size < 8:
            # Repeat sentences as needed to fill the batch, even if original batch is very small
            batch_sentences = list(islice(cycle([query]), 8))
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        encoded_input = tokenizer(
            batch_sentences, padding="max_length", max_length=384, truncation=True, return_tensors="pt"
        )
        input_ids = encoded_input["input_ids"]
        attention_mask = encoded_input["attention_mask"]
        ttnn_output = model_instance.run(input_ids, iter=0)
        print("ttnn output is", ttnn_output.shape, ttnn_output.layout)
        logger.info("Running inference on TTNN model for current batch...")
        ttnn_output = ttnn.to_torch(ttnn_output).squeeze(dim=1)
        query_embeddings = mean_pooling(ttnn_output, attention_mask)
        # query_embeddings, _ = compute_ttnn_embeddings([query], model_name, device)
        logger.info("Computing cosine similarities...")
        similarities = cosine_similarity(query_embeddings.detach().cpu().numpy(), kb_embeddings.detach().cpu().numpy())[
            0
        ]
        top_idx = np.argmax(similarities)
        logger.info(f"Best match: {kb_sentences[top_idx]}")
        logger.info(f"Similarity score: {similarities[top_idx]:.4f}")
        print(f"\tQuery: {query}")
        print(f"\tBest match: {kb_sentences[top_idx]}")
        print(f"\tSimilarity score: {similarities[top_idx]:.4f}")
