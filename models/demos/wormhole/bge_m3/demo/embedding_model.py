# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""BGE-M3 demo: sample input, load model, PCC/cosine validation, return sparse + colbert scores from BgeM3Model."""
import numpy as np
import torch
from loguru import logger
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForCausalLM

import ttnn
from models.common.auto_compose import to_torch_auto_compose
from models.common.utility_functions import comp_pcc
from models.demos.wormhole.bge_m3.demo.m3_scores import compute_score
from models.demos.wormhole.bge_m3.tt.model import BgeM3Model
from models.demos.wormhole.bge_m3.tt.model_config import ModelArgs

INPUTS = [
    "Artificial intelligence is transforming how we interact with technology.",
    "AI is changing the way humans use computers and machines.",
    "Machine learning algorithms are revolutionizing data analysis.",
    "Deep learning networks can process complex patterns in data.",
    "Neural networks mimic the human brain's structure and function.",
    "Natural language processing enables computers to understand text.",
    "Computer vision allows machines to interpret visual information.",
    "The weather is sunny today with clear blue skies.",
]


def _to_ttnn_ids(ids: torch.Tensor, device, dtype=ttnn.uint32) -> ttnn.Tensor:
    return ttnn.from_torch(
        ids.to(torch.int32),
        device=device,
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )


def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool last hidden state by attention mask (for dense/sentence embedding)."""
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1)
    return summed / counts


def _mean_pairwise_cosine_similarity(sentence_embeddings: torch.Tensor) -> float:
    cosine_sim_matrix = cosine_similarity(sentence_embeddings.detach().cpu().numpy())
    if cosine_sim_matrix.shape[0] < 2:
        return 1.0
    upper_triangle = cosine_sim_matrix[np.triu_indices_from(cosine_sim_matrix, k=1)]
    return float(upper_triangle.mean()) if upper_triangle.size else 1.0


def _mean_embedding_alignment(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    cross_similarity = cosine_similarity(reference.detach().cpu().numpy(), candidate.detach().cpu().numpy())
    return float(np.diag(cross_similarity).mean())


def run_bge_demo_inference(
    device,
    sentence_pairs,
    model_name="BAAI/bge-m3",
    sequence_length=8192,
):
    """Run inference: PCC/cosine validation (ref vs TT), then return sparse + colbert scores from BgeM3Model only."""
    sentence_pairs = list(sentence_pairs)
    if not sentence_pairs:
        raise ValueError("sentence_pairs must be a non-empty list of (query, passage) tuples.")
    queries = [p[0] for p in sentence_pairs]
    n_pairs = len(sentence_pairs)

    model_args = ModelArgs(
        mesh_device=device,
        max_batch_size=n_pairs,
        max_seq_len=sequence_length,
        hf_model_name=model_name,
    )
    reference_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).eval()
    backbone = reference_model.roberta if hasattr(reference_model, "roberta") else reference_model
    ttnn_model = BgeM3Model(
        args=model_args,
        mesh_device=device,
        dtype=ttnn.bfloat8_b,
        state_dict=reference_model.state_dict(),
    )

    # Cosine similarity block: same as reference — encode queries, run ref + TT, comp_pcc, mean_pool, log
    encoded_input = model_args.encode_prompts(queries)
    input_ids = encoded_input["input_ids"]
    attention_mask = encoded_input["attention_mask"]
    token_type_ids = encoded_input.get("token_type_ids", torch.zeros_like(input_ids))
    logger.info(f"Running demo with batch_size={len(queries)} and seq_len={input_ids.shape[1]}")

    with torch.no_grad():
        hf_last_hidden_state = backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=None,
            return_dict=True,
        ).last_hidden_state.to(torch.float32)
    hf_sentence_embeddings = _mean_pool(hf_last_hidden_state, attention_mask)

    tt_input_ids = _to_ttnn_ids(input_ids, device=device)
    tt_attention_mask = _to_ttnn_ids(attention_mask, device=device)
    tt_token_type_ids = _to_ttnn_ids(token_type_ids, device=device)
    ttnn_output = ttnn_model(
        input_ids=tt_input_ids,
        attention_mask=tt_attention_mask,
        token_type_ids=tt_token_type_ids,
    )
    ttnn_last_hidden_state = to_torch_auto_compose(ttnn_output).squeeze(1).to(torch.float32)
    ttnn_sentence_embeddings = _mean_pool(ttnn_last_hidden_state, attention_mask)

    hidden_state_passing, hidden_state_pcc = comp_pcc(hf_last_hidden_state, ttnn_last_hidden_state, 0.94)
    assert hidden_state_passing, f"Hidden state PCC check failed: {hidden_state_pcc}"

    hf_mean_similarity = _mean_pairwise_cosine_similarity(hf_sentence_embeddings)
    tt_mean_similarity = _mean_pairwise_cosine_similarity(ttnn_sentence_embeddings)
    embedding_alignment = _mean_embedding_alignment(hf_sentence_embeddings, ttnn_sentence_embeddings)

    logger.info(f"Mean Cosine Similarity for Reference Model (PyTorch): {hf_mean_similarity:.4f}")
    logger.info(f"Mean Cosine Similarity for TTNN Model: {tt_mean_similarity:.4f}")
    logger.info(f"Mean Embedding Alignment (reference vs TTNN): {embedding_alignment:.4f}")

    similarity_diff = abs(hf_mean_similarity - tt_mean_similarity)
    logger.info(f"Hidden state PCC passed: {hidden_state_pcc}")
    logger.info(f"Cosine similarity delta: {similarity_diff:.4f}")

    tolerance = 0.02
    assert (
        similarity_diff < tolerance
    ), f"Cosine similarities differ by {similarity_diff:.4f}, which exceeds tolerance of {tolerance}"

    # Call compute_score and return sparse and colbert scores
    scores = compute_score(device, sentence_pairs, ttnn_model, model_args, _to_ttnn_ids, to_torch_auto_compose)
    # logger.info(f"Sparse scores: {scores['sparse']}")
    # logger.info(f"ColBERT scores: {scores['colbert']}")
    for i, (q, p) in enumerate(sentence_pairs):
        logger.info(f"  pair {i}: sparse={scores['sparse'][i]:.4f}, colbert={scores['colbert'][i]:.4f}")
    return {"sparse": scores["sparse"], "colbert": scores["colbert"]}


if __name__ == "__main__":
    sentence_pairs = [(INPUTS[i], INPUTS[(i + 1) % len(INPUTS)]) for i in range(len(INPUTS))]
    device = ttnn.open_device(device_id=0)
    original_default_device = ttnn.GetDefaultDevice()
    try:
        ttnn.SetDefaultDevice(device)
        run_bge_demo_inference(device, sentence_pairs)
    finally:
        ttnn.SetDefaultDevice(original_default_device)
        ttnn.close_device(device)
