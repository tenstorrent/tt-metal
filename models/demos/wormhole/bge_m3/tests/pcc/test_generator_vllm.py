# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""PCC tests for `BgeM3ForEmbedding` (dense, sparse, ColBERT) with fixed reference tensors."""

import pytest
import torch
import torch.nn.functional as F
from loguru import logger
from ttnn.device import is_blackhole as ttnn_is_blackhole

import ttnn
from models.demos.wormhole.bge_m3.demo.generator_vllm import BgeM3ForEmbedding
from models.demos.wormhole.bge_m3.demo.m3_scores import (
    compute_colbert_score_torch,
    compute_dense_score_torch,
    compute_sparse_score_torch,
)

MODEL_NAME = "BAAI/bge-m3"
MAX_MODEL_LEN = 512

# Example queries/documents and fixed reference values for the score checks.
# lexical_score_reference and colbert_score_reference are the upstream FlagEmbedding BGE-M3
# README values (fp16 GPU run, verbatim). similarity_reference and corner_case_token_weight come
# from the float32 torch reference (tests/pcc/test_reference_vllm.py at the time); sentences_2[1]
# reads "Definition" where the README has "Defination", so row 1 of the similarity matrix differs
# from the README. None of these is a device capture. Do not fix a failure below by editing them:
# they are ground truth, not a record of past TT output. The tolerances are the lever.
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


def _vllm_dense_similarity_allclose_kwargs(device) -> dict[str, float]:
    """Dense cosine similarity matrix vs fixed reference (matched on Wormhole). Blackhole drifts slightly more."""
    if ttnn_is_blackhole(device):
        return {"rtol": 0.035, "atol": 1e-2}
    # rtol 0.03 with atol 1e-3 matches the bf8 noise floor. The measured spread
    # reaches 1.9%, and a relative-only bound rejects a reference value of 0.
    return {"rtol": 0.03, "atol": 1e-3}


def _vllm_score_rel_tolerance(device) -> float:
    """Sparse / ColBERT scalar scores vs fixed reference: pytest.approx(..., rel=...).

    Wormhole: 0.02. The heads run at bfloat8_b against fp16 references. Measured deviations
    were 1.006 %-1.588 % on 2026-08-18 (#53042, run 31794258337) and 1.31 %-1.36 % since the
    accurate SDPA exponential (#57180) and the exact SDPA reciprocal (#56292) landed: lexical
    0.1928863525390625 vs 0.19554901123046875, ColBERT 0.4559256434440613 vs 0.462, identical
    on three nightly runs. The previous 0.01 was tuned to the legacy kernels' one-sided bias
    and left no margin. These tests pad the prompts to 32 tokens at batch 2, so on Wormhole
    they take the accurate-exponential SDPA branch with fp32 dest; a change to that padding or
    to _sdpa_exp_approx moves these scalars again.
    """
    return 0.025 if ttnn_is_blackhole(device) else 0.02


def _vllm_corner_sparse_weight_rel(device) -> float:
    """Single-token sparse weight under BF8; noisier than batched paths.

    The device emits this scalar with bf16 precision: one ulp at 0.26 is 2^-9 = 0.00195,
    0.73 % of the reference. On Wormhole the "Hi" weight measured 0.2578125 (-3.48 %) with the
    exact SDPA reciprocal from #56292 and 0.27734375 (+3.83 %) with the approximate
    exponential, so a 4 % band would sit within one ulp of both edges. 0.06 leaves at least
    two ulps on either side. Blackhole keeps 0.04.
    """
    return 0.04 if ttnn_is_blackhole(device) else 0.06


def _log_score(label: str, measured: float, reference: float) -> None:
    """Log the measured value against its reference so passing runs leave a margin trail in CI."""
    if reference == 0.0:
        margin = f"abs diff {abs(measured - reference):.3e}"
    else:
        margin = f"{(measured - reference) / reference * 100:+.3f} % vs reference"
    logger.info(f"{label}: measured {measured!r} vs reference {reference!r} -> {margin}")


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
    # `BgeM3ForEmbedding._pad_inputs` and the dense / ColBERT scoring
    # helpers expect the raw 2D boolean keep-mask `[B, S]`, so request the
    # 2D form from `encode_prompts` instead of its default 4D additive.
    encoded_input = model_args.encode_prompts(sentences, attention_mask_4d=False)
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
    for i in range(similarity.shape[0]):
        for k in range(similarity.shape[1]):
            _log_score(f"dense similarity[{i}][{k}]", float(similarity[i, k]), float(similarity_reference[i, k]))
    assert torch.allclose(similarity, similarity_reference, **_vllm_dense_similarity_allclose_kwargs(device))


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

    rel = _vllm_score_rel_tolerance(device)
    lexical_score_1_0_x_2_0 = float(sparse_cross_scores[0, 0])
    _log_score("lexical cross score", lexical_score_1_0_x_2_0, lexical_score_reference[0])
    assert lexical_score_1_0_x_2_0 == pytest.approx(lexical_score_reference[0], rel=rel)

    lexical_score_1_0_x_1_1 = float(sparse_self_scores[0, 0])
    _log_score("lexical self score", lexical_score_1_0_x_1_1, lexical_score_reference[1])
    # The reference is exactly 0, so a relative bound gives no tolerance.
    assert lexical_score_1_0_x_1_1 == pytest.approx(lexical_score_reference[1], rel=rel, abs=1e-3)


@pytest.mark.parametrize("model_name, sequence_length", [(MODEL_NAME, MAX_MODEL_LEN)])
def test_bge_m3_vllm_sparse_embedding_corner_case(device, model_name, sequence_length, model_location_generator):
    outputs = _load_reference_outputs(device, model_name, sequence_length, model_location_generator)
    corner_sparse_weight = float(outputs["corner_case"]["sparse_vecs"][0, corner_case_token_id])
    _log_score("corner-case sparse weight", corner_sparse_weight, corner_case_token_weight)
    assert corner_sparse_weight == pytest.approx(
        corner_case_token_weight,
        rel=_vllm_corner_sparse_weight_rel(device),
    )


@pytest.mark.parametrize("model_name, sequence_length", [(MODEL_NAME, MAX_MODEL_LEN)])
def test_bge_m3_vllm_multi_vector(device, model_name, sequence_length, model_location_generator):
    outputs = _load_reference_outputs(device, model_name, sequence_length, model_location_generator)
    colbert_scores = compute_colbert_score_torch(
        outputs["sentences_1"]["colbert_vecs_norm"],
        outputs["sentences_2"]["colbert_vecs_norm"],
        q_mask=outputs["sentences_1"]["attention_mask"],
    )

    rel = _vllm_score_rel_tolerance(device)
    colbert_score_1_0_x_2_0 = float(colbert_scores[0, 0])
    _log_score("colbert score [0][0]", colbert_score_1_0_x_2_0, colbert_score_reference[0])
    assert colbert_score_1_0_x_2_0 == pytest.approx(colbert_score_reference[0], rel=rel)

    colbert_score_1_0_x_2_1 = float(colbert_scores[0, 1])
    _log_score("colbert score [0][1]", colbert_score_1_0_x_2_1, colbert_score_reference[1])
    assert colbert_score_1_0_x_2_1 == pytest.approx(colbert_score_reference[1], rel=rel)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
