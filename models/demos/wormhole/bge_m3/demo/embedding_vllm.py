# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""BGE-M3 generator-based demo on a single N300-style 1x2 mesh.

Run with a single PCIe-visible board, for example:
    TT_VISIBLE_DEVICES=0 python models/demos/wormhole/bge_m3/demo/embedding_vllm.py
"""

import os

import numpy as np
import torch
from loguru import logger
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForCausalLM

import ttnn
from models.common.auto_compose import to_torch_auto_compose
from models.common.utility_functions import comp_pcc
from models.demos.wormhole.bge_m3.demo.generator_vllm import BgeM3ForEmbedding
from models.demos.wormhole.bge_m3.demo.m3_scores import compute_score

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


def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool last hidden state by attention mask."""
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


def _crop_hidden_states(last_hidden_state: torch.Tensor, seq_len: int) -> torch.Tensor:
    return last_hidden_state[:, :seq_len, :]


def _compute_scores_with_replicas(generator_model, sentence_pairs, to_torch_fn):
    score_models = generator_model.models if generator_model.models is not None else [generator_model.model]
    score_model_args_list = (
        generator_model.model_args_list if generator_model.model_args_list is not None else [generator_model.model_args]
    )
    score_submeshes = generator_model.submeshes if generator_model.submeshes is not None else [generator_model.device]

    sparse_scores = []
    colbert_scores = []
    start = 0
    while start < len(sentence_pairs):
        for score_model, score_model_args, score_submesh in zip(score_models, score_model_args_list, score_submeshes):
            if start >= len(sentence_pairs):
                break

            chunk_size = min(score_model_args.max_batch_size, len(sentence_pairs) - start)
            chunk_pairs = sentence_pairs[start : start + chunk_size]

            def _to_ttnn_ids(ids: torch.Tensor, score_device) -> ttnn.Tensor:
                return generator_model._to_ttnn_ids(ids, device=score_device)

            chunk_scores = compute_score(
                score_submesh,
                chunk_pairs,
                score_model,
                score_model_args,
                _to_ttnn_ids,
                to_torch_fn,
            )
            sparse_scores.extend(float(score) for score in chunk_scores["sparse"])
            colbert_scores.extend(float(score) for score in chunk_scores["colbert"])
            start += chunk_size

    return {"sparse": sparse_scores, "colbert": colbert_scores}


def run_bge_vllm_demo_inference(
    device,
    sentence_pairs,
    model_name="BAAI/bge-m3",
    sequence_length=8192,
):
    """Run generator-backed inference with HF validation and score logging."""
    sentence_pairs = list(sentence_pairs)
    if not sentence_pairs:
        raise ValueError("sentence_pairs must be a non-empty list of (query, passage) tuples.")

    queries = [pair[0] for pair in sentence_pairs]
    n_pairs = len(sentence_pairs)

    generator_model = BgeM3ForEmbedding(
        device=device,
        max_batch_size=n_pairs,
        max_seq_len=sequence_length,
        tt_data_parallel=device.get_num_devices(),
        dtype=ttnn.bfloat8_b,
        model_name=model_name,
    )
    # Initialize once so the dense path and sparse/ColBERT score path reuse the same TT model + tokenizer.
    generator_model._initialize_model()
    model_args = (
        generator_model.model_args_list[0]
        if generator_model.model_args_list is not None
        else generator_model.model_args
    )
    assert model_args is not None

    reference_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).eval()
    backbone = reference_model.roberta if hasattr(reference_model, "roberta") else reference_model

    encoded_input = model_args.encode_prompts(queries)
    input_ids = encoded_input["input_ids"]
    attention_mask = encoded_input["attention_mask"]
    token_type_ids = encoded_input.get("token_type_ids", torch.zeros_like(input_ids))
    seq_len = input_ids.shape[1]
    logger.info(f"Running generator demo with batch_size={len(queries)} and seq_len={seq_len}")

    with torch.no_grad():
        hf_last_hidden_state = backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=None,
            return_dict=True,
        ).last_hidden_state.to(torch.float32)

    hf_sentence_embeddings = _mean_pool(hf_last_hidden_state, attention_mask)

    ttnn_last_hidden_state = generator_model.forward(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
    )
    ttnn_last_hidden_state = _crop_hidden_states(ttnn_last_hidden_state, seq_len).to(torch.float32)
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

    scores = _compute_scores_with_replicas(generator_model, sentence_pairs, to_torch_auto_compose)

    for i, (_query, _passage) in enumerate(sentence_pairs):
        logger.info(f"  pair {i}: sparse={scores['sparse'][i]:.4f}, colbert={scores['colbert'][i]:.4f}")

    return {"sparse": scores["sparse"], "colbert": scores["colbert"]}


if __name__ == "__main__":
    sentence_pairs = [(INPUTS[i], INPUTS[(i + 1) % len(INPUTS)]) for i in range(len(INPUTS))]
    logger.info(f"TT_VISIBLE_DEVICES={os.environ.get('TT_VISIBLE_DEVICES', '<unset>')}")

    original_default_device = ttnn.GetDefaultDevice()
    with ttnn.create_mesh_device(mesh_shape=ttnn.MeshShape(1, 2)) as mesh_device:
        logger.info(
            "Opened mesh device with num_devices={} and grid={}",
            mesh_device.get_num_devices(),
            mesh_device.compute_with_storage_grid_size(),
        )
        try:
            ttnn.SetDefaultDevice(mesh_device)
            run_bge_vllm_demo_inference(mesh_device, sentence_pairs)
        finally:
            ttnn.SetDefaultDevice(original_default_device)
