# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch
from loguru import logger
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForCausalLM

import ttnn
from models.common.auto_compose import to_torch_auto_compose
from models.common.utility_functions import comp_pcc
from models.demos.wormhole.bge_m3.demo.generator_vllm import BgeM3ForEmbedding
from models.demos.wormhole.bge_m3.tt.common import create_tt_model

DEFAULT_MODEL_NAME = "BAAI/bge-m3"
DEFAULT_SEQUENCE_LENGTH = 8192
DEFAULT_TT_DTYPE = ttnn.bfloat8_b

inputs = [
    [
        "Artificial intelligence is transforming how we interact with technology.",
        "AI is changing the way humans use computers and machines.",
        "Machine learning algorithms are revolutionizing data analysis.",
        "Deep learning networks can process complex patterns in data.",
        "Neural networks mimic the human brain's structure and function.",
        "Natural language processing enables computers to understand text.",
        "Computer vision allows machines to interpret visual information.",
        "The weather is sunny today with clear blue skies.",
    ]
]


def _require_single_device(device) -> None:
    if hasattr(device, "get_num_devices") and device.get_num_devices() != 1:
        raise ValueError("BGE-M3 demo currently expects a single device")


def _resolve_model_name(model_name, model_location_generator):
    if model_location_generator is None:
        return model_name
    return str(model_location_generator(model_name, download_if_ci_v2=True, ci_v2_timeout_in_s=1800))


def _to_ttnn_ids(ids: torch.Tensor, device, dtype=ttnn.uint32) -> ttnn.Tensor:
    return ttnn.from_torch(
        ids.to(torch.int32),
        device=device,
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )


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


def _load_reference_hidden_states(
    resolved_model_name: str,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    token_type_ids: torch.Tensor,
):
    reference_model = AutoModelForCausalLM.from_pretrained(resolved_model_name, torch_dtype=torch.bfloat16).eval()
    backbone = reference_model.roberta if hasattr(reference_model, "roberta") else reference_model

    with torch.no_grad():
        reference_hidden_states = backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=None,
            return_dict=True,
        ).last_hidden_state.to(torch.float32)

    return reference_hidden_states


def _log_embedding_comparison(
    reference_hidden_states: torch.Tensor,
    tt_hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    pool_fn,
    path_name: str,
):
    reference_sentence_embeddings = pool_fn(reference_hidden_states, attention_mask)
    tt_sentence_embeddings = pool_fn(tt_hidden_states, attention_mask)

    hidden_state_passing, hidden_state_pcc = comp_pcc(reference_hidden_states, tt_hidden_states, 0.94)
    assert hidden_state_passing, f"{path_name} hidden state PCC check failed: {hidden_state_pcc}"

    reference_mean_similarity = _mean_pairwise_cosine_similarity(reference_sentence_embeddings)
    tt_mean_similarity = _mean_pairwise_cosine_similarity(tt_sentence_embeddings)
    embedding_alignment = _mean_embedding_alignment(reference_sentence_embeddings, tt_sentence_embeddings)

    logger.info(f"{path_name} reference hidden-state shape: {tuple(reference_hidden_states.shape)}")
    logger.info(f"{path_name} TT hidden-state shape: {tuple(tt_hidden_states.shape)}")
    logger.info(f"{path_name} pooled embedding shape: {tuple(tt_sentence_embeddings.shape)}")
    logger.info(f"{path_name} hidden state PCC: {hidden_state_pcc}")
    logger.info(f"{path_name} mean cosine similarity (PyTorch): {reference_mean_similarity:.4f}")
    logger.info(f"{path_name} mean cosine similarity (TTNN): {tt_mean_similarity:.4f}")
    logger.info(f"{path_name} mean embedding alignment: {embedding_alignment:.4f}")

    similarity_diff = abs(reference_mean_similarity - tt_mean_similarity)
    tolerance = 0.02
    assert (
        similarity_diff < tolerance
    ), f"{path_name} cosine similarities differ by {similarity_diff:.4f}, exceeding tolerance {tolerance}"
    logger.info(f"{path_name} cosine similarities are close (difference: {similarity_diff:.4f})")

    return tt_sentence_embeddings


def run_bge_demo_inference(device, inputs, model_name, sequence_length, model_location_generator):
    _require_single_device(device)
    resolved_model_name = _resolve_model_name(model_name, model_location_generator)

    model_args, tt_model, _ = create_tt_model(
        mesh_device=device,
        max_batch_size=len(inputs),
        max_seq_len=sequence_length,
        dtype=DEFAULT_TT_DTYPE,
        state_dict=None,
        hf_model_name=resolved_model_name,
    )

    encoded_input = model_args.encode_prompts(inputs)
    input_ids = encoded_input["input_ids"]
    attention_mask = encoded_input["attention_mask"]
    token_type_ids = encoded_input.get("token_type_ids", torch.zeros_like(input_ids))
    seq_len = input_ids.shape[1]

    reference_hidden_states = _load_reference_hidden_states(
        resolved_model_name,
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
    )

    tt_output = tt_model(
        input_ids=_to_ttnn_ids(input_ids, device=device),
        attention_mask=_to_ttnn_ids(attention_mask, device=device),
        token_type_ids=_to_ttnn_ids(token_type_ids, device=device),
    )
    tt_hidden_states = to_torch_auto_compose(tt_output, device=device)
    if tt_hidden_states.dim() == 4 and tt_hidden_states.shape[1] == 1:
        tt_hidden_states = tt_hidden_states.squeeze(1)
    tt_hidden_states = _crop_hidden_states(tt_hidden_states, seq_len).to(torch.float32)

    return _log_embedding_comparison(
        reference_hidden_states,
        tt_hidden_states,
        attention_mask,
        pool_fn=BgeM3ForEmbedding._pool_embeddings,
        path_name="create_tt_model",
    )


def run_bge_vllm_demo(device, inputs, model_name, sequence_length, model_location_generator):
    resolved_model_name = _resolve_model_name(model_name, model_location_generator)

    generator_model = BgeM3ForEmbedding(
        device=device,
        max_batch_size=len(inputs),
        max_seq_len=sequence_length,
        dtype=DEFAULT_TT_DTYPE,
        model_name=resolved_model_name,
    )
    generator_model._initialize_model()
    model_args = generator_model.model_args
    assert model_args is not None

    encoded_input = model_args.encode_prompts(inputs)
    input_ids = encoded_input["input_ids"]
    attention_mask = encoded_input["attention_mask"]
    token_type_ids = encoded_input.get("token_type_ids", torch.zeros_like(input_ids))
    seq_len = input_ids.shape[1]

    reference_hidden_states = _load_reference_hidden_states(
        resolved_model_name,
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
    )

    tt_hidden_states = generator_model.forward(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=None,
    )
    tt_hidden_states = _crop_hidden_states(tt_hidden_states, seq_len).to(torch.float32)

    return _log_embedding_comparison(
        reference_hidden_states,
        tt_hidden_states,
        attention_mask,
        pool_fn=generator_model._pool_embeddings,
        path_name="vllm_generator",
    )


@pytest.mark.parametrize("inputs", inputs)
@pytest.mark.parametrize("model_name, sequence_length", [(DEFAULT_MODEL_NAME, DEFAULT_SEQUENCE_LENGTH)])
def test_bge_demo_inference(device, inputs, model_name, sequence_length, model_location_generator):
    run_bge_demo_inference(device, inputs, model_name, sequence_length, model_location_generator)


@pytest.mark.parametrize("inputs", inputs)
@pytest.mark.parametrize("model_name, sequence_length", [(DEFAULT_MODEL_NAME, DEFAULT_SEQUENCE_LENGTH)])
def test_bge_vllm_demo(device, inputs, model_name, sequence_length, model_location_generator):
    run_bge_vllm_demo(device, inputs, model_name, sequence_length, model_location_generator)
