# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import math
import time

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
from models.demos.wormhole.bge_m3.tt.model_config import get_padded_sequence_length

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
    return str(model_location_generator(model_name))


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


def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1)
    return summed / counts


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


def _log_pooled_embedding_comparison(
    reference_sentence_embeddings: torch.Tensor,
    tt_sentence_embeddings: torch.Tensor,
    *,
    path_name: str,
):
    reference_mean_similarity = _mean_pairwise_cosine_similarity(reference_sentence_embeddings)
    tt_mean_similarity = _mean_pairwise_cosine_similarity(tt_sentence_embeddings)
    embedding_alignment = _mean_embedding_alignment(reference_sentence_embeddings, tt_sentence_embeddings)

    logger.info(f"{path_name} pooled embedding shape: {tuple(tt_sentence_embeddings.shape)}")
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
        pool_fn=_mean_pool,
        path_name="create_tt_model",
    )


def run_bge_vllm_demo(device, inputs, model_name, sequence_length, model_location_generator):
    resolved_model_name = _resolve_model_name(model_name, model_location_generator)

    generator_model = BgeM3ForEmbedding(
        device=device,
        max_batch_size=32,
        max_seq_len=sequence_length,
        dtype=DEFAULT_TT_DTYPE,
        model_name=resolved_model_name,
        sentence_pooling_method="mean",
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
    reference_sentence_embeddings = _mean_pool(reference_hidden_states, attention_mask)

    outputs = generator_model.forward(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=None,
    )
    tt_sentence_embeddings = outputs["dense_vecs"][: len(inputs)].to(torch.float32)

    return _log_pooled_embedding_comparison(
        reference_sentence_embeddings,
        tt_sentence_embeddings,
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


def _build_synthetic_inputs(tokenizer, isl: int, batch_size: int):
    # Use an arbitrary in-vocab non-special token id so the model executes a
    # realistic prefill instead of hitting the pad-token fast path.
    vocab_size = getattr(tokenizer, "vocab_size", 250002)
    special_ids = set(tokenizer.all_special_ids) if hasattr(tokenizer, "all_special_ids") else set()
    fill_id = next((tid for tid in range(5, vocab_size) if tid not in special_ids), 5)

    input_ids = torch.full((batch_size, isl), fill_value=fill_id, dtype=torch.int64)
    attention_mask = torch.ones((batch_size, isl), dtype=torch.int64)
    token_type_ids = torch.zeros((batch_size, isl), dtype=torch.int64)
    return input_ids, attention_mask, token_type_ids


def run_bge_benchmark(
    device,
    model_name: str,
    isl: int,
    max_concurrency: int,
    num_requests: int,
    model_location_generator,
    *,
    warmup_iters: int = 1,
):
    resolved_model_name = _resolve_model_name(model_name, model_location_generator)
    padded_isl = get_padded_sequence_length(isl)

    generator_model = BgeM3ForEmbedding(
        device=device,
        max_batch_size=max_concurrency,
        max_seq_len=max(padded_isl, isl),
        dtype=DEFAULT_TT_DTYPE,
        model_name=resolved_model_name,
        sentence_pooling_method="mean",
    )
    generator_model._initialize_model()

    input_ids, attention_mask, token_type_ids = _build_synthetic_inputs(generator_model.tokenizer, isl, max_concurrency)

    for _ in range(warmup_iters):
        generator_model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=None,
        )
    ttnn.synchronize_device(device)

    num_batches = max(1, math.ceil(num_requests / max_concurrency))
    effective_num_requests = num_batches * max_concurrency

    batch_latencies_s: list[float] = []
    wall_start = time.perf_counter()
    for _ in range(num_batches):
        batch_start = time.perf_counter()
        generator_model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=None,
        )
        ttnn.synchronize_device(device)
        batch_latencies_s.append(time.perf_counter() - batch_start)
    wall_total_s = time.perf_counter() - wall_start

    e2el_ms = (sum(batch_latencies_s) / len(batch_latencies_s)) * 1000.0
    tput_prefill_tps = (effective_num_requests * isl) / wall_total_s
    req_tput_rps = effective_num_requests / wall_total_s

    metrics = {
        "ISL": isl,
        "Padded ISL": padded_isl,
        "Max Concurrency": max_concurrency,
        "Num Requests": effective_num_requests,
        "Tput Prefill (TPS)": tput_prefill_tps,
        "E2EL (ms)": e2el_ms,
        "Req Tput (RPS)": req_tput_rps,
    }

    header = (
        f"{'ISL':>8} | {'Max Concurrency':>15} | {'Num Requests':>12} | "
        f"{'Tput Prefill (TPS)':>20} | {'E2EL (ms)':>12} | {'Req Tput (RPS)':>15}"
    )
    row = (
        f"{isl:>8} | {max_concurrency:>15} | {effective_num_requests:>12} | "
        f"{tput_prefill_tps:>20.2f} | {e2el_ms:>12.2f} | {req_tput_rps:>15.2f}"
    )
    logger.info("BGE-M3 benchmark results:\n" + header + "\n" + "-" * len(header) + "\n" + row)

    return metrics


@pytest.fixture(scope="session")
def bge_benchmark_results():
    results: list[dict] = []
    yield results

    if not results:
        return

    header = (
        f"{'ISL':>8} | {'Max Concurrency':>15} | {'Num Requests':>12} | "
        f"{'Tput Prefill (TPS)':>20} | {'E2EL (ms)':>12} | {'Req Tput (RPS)':>15}"
    )
    separator = "-" * len(header)
    rows = [
        f"{m['ISL']:>8} | {m['Max Concurrency']:>15} | {m['Num Requests']:>12} | "
        f"{m['Tput Prefill (TPS)']:>20.2f} | {m['E2EL (ms)']:>12.2f} | {m['Req Tput (RPS)']:>15.2f}"
        for m in results
    ]
    logger.info("BGE-M3 benchmark summary:\n" + header + "\n" + separator + "\n" + "\n".join(rows) + "\n" + separator)


@pytest.mark.parametrize(
    "isl, max_concurrency, num_requests",
    [
        (32, 1, 10),
        (64, 1, 10),
        (128, 1, 10),
        (256, 1, 10),
        (512, 1, 10),
        (32, 25, 50),
        (64, 25, 50),
        (128, 25, 50),
        (256, 25, 50),
        (512, 25, 50),
    ],
)
@pytest.mark.parametrize("model_name", [DEFAULT_MODEL_NAME])
def test_bge_benchmark(
    device,
    model_name,
    isl,
    max_concurrency,
    num_requests,
    model_location_generator,
    bge_benchmark_results,
):
    metrics = run_bge_benchmark(
        device=device,
        model_name=model_name,
        isl=isl,
        max_concurrency=max_concurrency,
        num_requests=num_requests,
        model_location_generator=model_location_generator,
    )
    bge_benchmark_results.append(metrics)
