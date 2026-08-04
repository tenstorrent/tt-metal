# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Pytest checks for `generator_vllm.py`, aligned with the vLLM BGE-M3 reference tests."""

import pytest
import torch
import torch.nn.functional as F

import ttnn
from models.demos.wormhole.bge_m3.demo.generator_vllm import BgeM3ForEmbedding
from models.demos.wormhole.bge_m3.demo.m3_scores import (
    compute_colbert_score_torch,
    compute_dense_score_torch,
    compute_sparse_score_torch,
)

MODEL_NAME = "BAAI/bge-m3"
MAX_MODEL_LEN = 512

# Example and references from tests/pcc/test_reference_vllm.py
sentences_1 = ["What is BGE M3?", "Definition of BM25"]
sentences_2 = [
    "BGE M3 is an embedding model supporting dense retrieval, " "lexical matching and multi-vector interaction.",
    "BM25 is a bag-of-words retrieval function that ranks a set "
    "of documents based on the query terms appearing in each document",
]

similarity_reference = torch.tensor([[0.6259, 0.3474], [0.3309, 0.6734]], dtype=torch.float32)
lexical_score_reference = [0.19554901123046875, 0.0]
colbert_score_reference = [0.7797, 0.4620]
corner_case_token_id = 2673
corner_case_token_weight = 0.26710861921310425


def _require_single_device(device) -> None:
    if hasattr(device, "get_num_devices") and device.get_num_devices() != 1:
        raise ValueError("BGE-M3 generator tests currently expect a single device")


def _resolve_model_name(model_name, model_location_generator):
    if model_location_generator is None:
        return model_name
    return str(model_location_generator(model_name))


def _build_generator_model(
    device,
    model_name: str,
    sequence_length: int,
    max_batch_size: int,
) -> tuple[BgeM3ForEmbedding, object]:
    tt_data_parallel = device.get_num_devices() if hasattr(device, "get_num_devices") else 1
    generator_model = BgeM3ForEmbedding(
        device=device,
        max_batch_size=max_batch_size,
        max_seq_len=sequence_length,
        tt_data_parallel=tt_data_parallel,
        dtype=ttnn.bfloat8_b,
        model_name=model_name,
        # Match BGE-M3 reference dense semantics from flag_embedding_model.py.
        sentence_pooling_method="cls",
        return_dense=True,
        return_sparse=True,
        return_colbert=True,
    )
    generator_model._initialize_model()
    model_args = (
        generator_model.model_args_list[0]
        if generator_model.model_args_list is not None
        else generator_model.model_args
    )
    assert model_args is not None
    return generator_model, model_args


def _run_generator_embeddings(
    generator_model: BgeM3ForEmbedding,
    model_args,
    sentences: list[str],
) -> dict[str, torch.Tensor]:
    encoded_input = model_args.encode_prompts(sentences)
    input_ids = encoded_input["input_ids"]
    attention_mask = encoded_input["attention_mask"]
    token_type_ids = encoded_input.get("token_type_ids", torch.zeros_like(input_ids))
    seq_len = input_ids.shape[1]

    outputs = generator_model.forward(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
    )
    dense_vecs = outputs["dense_vecs"][: len(sentences)].to(torch.float32)
    sparse_vecs = outputs["sparse_vecs"][: len(sentences)].to(torch.float32)
    colbert_vecs = outputs["colbert_vecs"][: len(sentences), : seq_len - 1].to(torch.float32)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "dense_vecs": dense_vecs,
        "dense_vecs_norm": F.normalize(dense_vecs, dim=-1),
        "sparse_vecs": sparse_vecs,
        "colbert_vecs": colbert_vecs,
        "colbert_vecs_norm": F.normalize(colbert_vecs, dim=-1),
    }


def _load_reference_outputs(device, model_name, sequence_length, model_location_generator):
    _require_single_device(device)
    resolved_model_name = _resolve_model_name(model_name, model_location_generator)
    max_batch_size = max(len(sentences_1), len(sentences_2), 1)
    generator_model, model_args = _build_generator_model(device, resolved_model_name, sequence_length, max_batch_size)
    return {
        "sentences_1": _run_generator_embeddings(generator_model, model_args, sentences_1),
        "sentences_2": _run_generator_embeddings(generator_model, model_args, sentences_2),
        "corner_case": _run_generator_embeddings(generator_model, model_args, ["Hi"]),
    }


@pytest.mark.parametrize("model_name, sequence_length", [(MODEL_NAME, MAX_MODEL_LEN)])
def test_bge_m3_vllm_dense_embedding(device, model_name, sequence_length, model_location_generator):
    outputs = _load_reference_outputs(device, model_name, sequence_length, model_location_generator)
    similarity = compute_dense_score_torch(
        outputs["sentences_1"]["dense_vecs_norm"],
        outputs["sentences_2"]["dense_vecs_norm"],
    )
    assert torch.allclose(similarity, similarity_reference, rtol=0.01)


@pytest.mark.parametrize("model_name, sequence_length", [(MODEL_NAME, MAX_MODEL_LEN)])
def test_bge_m3_vllm_sparse_embedding(device, model_name, sequence_length, model_location_generator):
    outputs = _load_reference_outputs(device, model_name, sequence_length, model_location_generator)
    sparse_cross_scores = compute_sparse_score_torch(
        outputs["sentences_1"]["sparse_vecs"],
        outputs["sentences_2"]["sparse_vecs"],
    )
    sparse_self_scores = compute_sparse_score_torch(
        outputs["sentences_1"]["sparse_vecs"][:1],
        outputs["sentences_1"]["sparse_vecs"][1:2],
    )

    lexical_score_1_0_x_2_0 = float(sparse_cross_scores[0, 0])
    assert lexical_score_1_0_x_2_0 == pytest.approx(lexical_score_reference[0], rel=0.01)

    lexical_score_1_0_x_1_1 = float(sparse_self_scores[0, 0])
    assert lexical_score_1_0_x_1_1 == pytest.approx(lexical_score_reference[1], rel=0.01)


@pytest.mark.xfail(reason="Single-token sparse weight drifts under TT bfloat8_b precision", strict=False)
@pytest.mark.parametrize("model_name, sequence_length", [(MODEL_NAME, MAX_MODEL_LEN)])
def test_bge_m3_vllm_sparse_embedding_corner_case(device, model_name, sequence_length, model_location_generator):
    outputs = _load_reference_outputs(device, model_name, sequence_length, model_location_generator)
    corner_sparse_weight = float(outputs["corner_case"]["sparse_vecs"][0, corner_case_token_id])
    assert corner_sparse_weight == pytest.approx(corner_case_token_weight, rel=0.01)


@pytest.mark.parametrize("model_name, sequence_length", [(MODEL_NAME, MAX_MODEL_LEN)])
def test_bge_m3_vllm_multi_vector(device, model_name, sequence_length, model_location_generator):
    outputs = _load_reference_outputs(device, model_name, sequence_length, model_location_generator)
    colbert_scores = compute_colbert_score_torch(
        outputs["sentences_1"]["colbert_vecs_norm"],
        outputs["sentences_2"]["colbert_vecs_norm"],
        q_mask=outputs["sentences_1"]["attention_mask"],
    )

    colbert_score_1_0_x_2_0 = float(colbert_scores[0, 0])
    assert colbert_score_1_0_x_2_0 == pytest.approx(colbert_score_reference[0], rel=0.01)

    colbert_score_1_0_x_2_1 = float(colbert_scores[0, 1])
    assert colbert_score_1_0_x_2_1 == pytest.approx(colbert_score_reference[1], rel=0.01)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
