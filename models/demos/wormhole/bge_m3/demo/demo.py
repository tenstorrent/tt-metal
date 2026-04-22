# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
BGE-M3 demo tests: create_tt_model path, vLLM-style generator path, and throughput benchmark.

Kept in sync with ``demo.py`` patterns; uses standard ASCII whitespace only (NBSP breaks linters).
"""

import math
import time
from collections.abc import Callable

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
DEFAULT_SEQUENCE_LENGTH = 512
DEFAULT_TT_DTYPE = ttnn.bfloat8_b

# PCC gate for full hidden-state tensor vs HF reference (matches PCC tests).
HIDDEN_STATE_PCC_THRESHOLD = 0.94
# Max allowed gap between PyTorch vs TT mean pairwise cosine similarity on pooled embeddings.
COSINE_MEAN_DIFF_TOLERANCE = 0.02

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


# ----------------------------------------------------------------------
# Trace + 2-command-queue benchmark runner
# ----------------------------------------------------------------------
# The non-traced ``BgeM3ForEmbedding`` path is dispatch-bound on Wormhole: every encoder
# forward pays per-op Python/dispatch overhead, and per-call ``ttnn.from_torch`` allocates
# fresh input buffers. ``BgeM3PerformantRunner`` removes that overhead by:
#   1. Pre-allocating persistent device input tensors and refreshing them via
#      ``ttnn.copy_host_to_device_tensor``.
#   2. Capturing the full encoder forward once with ``ttnn.begin_trace_capture`` and
#      replaying it via ``ttnn.execute_trace``.
#   3. Overlapping host->device input copies (CQ 1) with compute (CQ 0) using event
#      handshakes.
# Pattern mirrors ``models/demos/sentence_bert/runner/performant_runner.py``.

# Wormhole encoder mask uses the same large-negative additive value as the model's
# internal builder (see ``BgeM3Model._ADDITIVE_MASKED_VALUE``).
_ADDITIVE_MASKED_VALUE = -100000.0


class BgeM3PerformantRunner:
    """Trace-and-2CQ runner exposing a ``run()`` interface for fixed-shape forwards."""

    def __init__(
        self,
        *,
        device: ttnn.Device,
        model_name: str,
        batch_size: int,
        sequence_length: int,
        dtype=ttnn.bfloat8_b,
        model_location_generator=None,
    ) -> None:
        self.device = device
        self.batch_size = batch_size
        self.sequence_length = sequence_length

        del model_location_generator  # Resolution happens inside the generator (HF cache).

        self.generator = BgeM3ForEmbedding(
            device=device,
            max_batch_size=batch_size,
            max_seq_len=sequence_length,
            dtype=dtype,
            model_name=model_name,
            sentence_pooling_method="mean",
        )
        self.generator._initialize_model()
        self.model = self.generator.model
        self.tokenizer = self.generator.tokenizer
        self.model_args = self.generator.model_args
        self.pad_token_id = int(self.model_args.pad_token_id)

        self._build_default_host_inputs()
        self._allocate_device_inputs()
        self._capture_trace()

    def run(
        self,
        *,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> ttnn.Tensor:
        """
        Replay the encoder trace once. ``None`` inputs reuse the cached default host
        tensors. Returns the on-device encoder output (held in the trace's allocator slot).
        """
        in_ids_host = self._to_host_id_tensor(input_ids) if input_ids is not None else self.input_ids_host
        tok_ids_host = (
            self._to_host_id_tensor(token_type_ids) if token_type_ids is not None else self.token_type_ids_host
        )
        pos_ids_host = self._to_host_id_tensor(position_ids) if position_ids is not None else self.position_ids_host
        att_mask_host = (
            self._to_host_additive_mask_tensor(attention_mask)
            if attention_mask is not None
            else self.attention_mask_host
        )

        ttnn.wait_for_event(1, self.op_event)
        ttnn.copy_host_to_device_tensor(in_ids_host, self.input_ids_dev, 1)
        ttnn.copy_host_to_device_tensor(tok_ids_host, self.token_type_ids_dev, 1)
        ttnn.copy_host_to_device_tensor(pos_ids_host, self.position_ids_dev, 1)
        ttnn.copy_host_to_device_tensor(att_mask_host, self.attention_mask_dev, 1)
        self.write_event = ttnn.record_event(self.device, 1)
        ttnn.wait_for_event(0, self.write_event)
        self.op_event = ttnn.record_event(self.device, 0)
        ttnn.execute_trace(self.device, self.tid, cq_id=0, blocking=False)
        return self.output_dev

    def release(self) -> None:
        ttnn.release_trace(self.device, self.tid)

    @staticmethod
    def build_position_ids(input_ids: torch.Tensor, pad_token_id: int) -> torch.Tensor:
        """XLM-Roberta-compatible padding-aware position IDs (matches HF + on-device builder)."""
        mask = (input_ids != pad_token_id).to(torch.int64)
        incremental_indices = torch.cumsum(mask, dim=1) * mask
        return (incremental_indices + pad_token_id).to(torch.int64)

    @staticmethod
    def build_additive_attention_mask(attention_mask: torch.Tensor) -> torch.Tensor:
        """Build a rank-4 ``[B, 1, 1, S]`` bf16 additive ``{0.0, -100000.0}`` mask from a HF ``{0, 1}`` 2-D mask."""
        keep = attention_mask.to(torch.bfloat16)
        additive = (1.0 - keep) * _ADDITIVE_MASKED_VALUE
        return additive.unsqueeze(1).unsqueeze(1)

    def _build_default_host_inputs(self) -> None:
        batch, seq = self.batch_size, self.sequence_length
        vocab_size = getattr(self.tokenizer, "vocab_size", 250002)
        special_ids = set(self.tokenizer.all_special_ids) if hasattr(self.tokenizer, "all_special_ids") else set()
        fill_id = next((tid for tid in range(5, vocab_size) if tid not in special_ids), 5)

        default_input_ids = torch.full((batch, seq), fill_value=fill_id, dtype=torch.int64)
        default_attention_mask = torch.ones((batch, seq), dtype=torch.int64)
        default_token_type_ids = torch.zeros((batch, seq), dtype=torch.int64)
        default_position_ids = self.build_position_ids(default_input_ids, self.pad_token_id)
        default_additive_mask = self.build_additive_attention_mask(default_attention_mask)

        self.input_ids_host = self._to_host_id_tensor(default_input_ids)
        self.token_type_ids_host = self._to_host_id_tensor(default_token_type_ids)
        self.position_ids_host = self._to_host_id_tensor(default_position_ids)
        self.attention_mask_host = self._to_host_additive_mask_tensor_from_additive(default_additive_mask)

    def _to_host_id_tensor(self, ids: torch.Tensor) -> ttnn.Tensor:
        return ttnn.from_torch(ids.to(torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)

    def _to_host_additive_mask_tensor(self, attention_mask: torch.Tensor) -> ttnn.Tensor:
        return self._to_host_additive_mask_tensor_from_additive(self.build_additive_attention_mask(attention_mask))

    def _to_host_additive_mask_tensor_from_additive(self, additive_mask: torch.Tensor) -> ttnn.Tensor:
        return ttnn.from_torch(additive_mask.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    def _allocate_device_inputs(self) -> None:
        dram = ttnn.DRAM_MEMORY_CONFIG
        self.input_ids_dev = self.input_ids_host.to(self.device, dram)
        self.token_type_ids_dev = self.token_type_ids_host.to(self.device, dram)
        self.position_ids_dev = self.position_ids_host.to(self.device, dram)
        self.attention_mask_dev = self.attention_mask_host.to(self.device, dram)

    def _model_forward(self) -> ttnn.Tensor:
        return self.model(
            input_ids=self.input_ids_dev,
            attention_mask=self.attention_mask_dev,
            token_type_ids=self.token_type_ids_dev,
            position_ids=self.position_ids_dev,
        )

    def _h2d_all(self) -> None:
        ttnn.copy_host_to_device_tensor(self.input_ids_host, self.input_ids_dev, 1)
        ttnn.copy_host_to_device_tensor(self.token_type_ids_host, self.token_type_ids_dev, 1)
        ttnn.copy_host_to_device_tensor(self.position_ids_host, self.position_ids_dev, 1)
        ttnn.copy_host_to_device_tensor(self.attention_mask_host, self.attention_mask_dev, 1)

    def _capture_trace(self) -> None:
        self.op_event = ttnn.record_event(self.device, 0)

        # Pass 1: warm up program cache (JIT) and force op state to settle.
        ttnn.wait_for_event(1, self.op_event)
        self._h2d_all()
        self.write_event = ttnn.record_event(self.device, 1)
        ttnn.wait_for_event(0, self.write_event)
        self.op_event = ttnn.record_event(self.device, 0)
        out = self._model_forward()
        ttnn.deallocate(out)

        # Pass 2: optimized run (program cache hot).
        ttnn.wait_for_event(1, self.op_event)
        self._h2d_all()
        self.write_event = ttnn.record_event(self.device, 1)
        ttnn.wait_for_event(0, self.write_event)
        self.op_event = ttnn.record_event(self.device, 0)
        out = self._model_forward()
        ttnn.deallocate(out)

        # Pass 3: capture. Persistent ``self.*_dev`` buffers stay alive for the runner's
        # lifetime; the trace records reads from those fixed addresses, and ``run()``
        # updates them in place via CQ-1 H2D copies between trace replays. No staging
        # tensors / post-capture ``allocate_tensor_on_device`` round-trip needed (those are
        # only required when the model's input is sharded and an in-trace reshard
        # materializes it, like sentence_bert).
        ttnn.wait_for_event(1, self.op_event)
        self._h2d_all()
        self.write_event = ttnn.record_event(self.device, 1)
        ttnn.wait_for_event(0, self.write_event)
        self.op_event = ttnn.record_event(self.device, 0)

        self.tid = ttnn.begin_trace_capture(self.device, cq_id=0)
        self.output_dev = self._model_forward()
        ttnn.end_trace_capture(self.device, self.tid, cq_id=0)
        ttnn.synchronize_device(self.device)


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
    cross_similarity = cosine_similarity(
        reference.detach().cpu().numpy(),
        candidate.detach().cpu().numpy(),
    )
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
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    token_type_ids: torch.Tensor,
) -> torch.Tensor:
    reference_model = AutoModelForCausalLM.from_pretrained(
        resolved_model_name,
        torch_dtype=torch.bfloat16,
    ).eval()
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


def _assert_mean_cosine_diff_below_tolerance(
    reference_mean_similarity: float,
    tt_mean_similarity: float,
    path_name: str,
) -> None:
    similarity_diff = abs(reference_mean_similarity - tt_mean_similarity)
    assert similarity_diff < COSINE_MEAN_DIFF_TOLERANCE, (
        f"{path_name} cosine similarities differ by {similarity_diff:.4f}, "
        f"exceeding tolerance {COSINE_MEAN_DIFF_TOLERANCE}"
    )
    logger.info(
        f"{path_name} cosine similarities are close (difference: {similarity_diff:.4f})",
    )


def _log_embedding_comparison(
    reference_hidden_states: torch.Tensor,
    tt_hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    pool_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    path_name: str,
) -> torch.Tensor:
    reference_sentence_embeddings = pool_fn(reference_hidden_states, attention_mask)
    tt_sentence_embeddings = pool_fn(tt_hidden_states, attention_mask)

    hidden_state_passing, hidden_state_pcc = comp_pcc(
        reference_hidden_states,
        tt_hidden_states,
        HIDDEN_STATE_PCC_THRESHOLD,
    )
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

    _assert_mean_cosine_diff_below_tolerance(reference_mean_similarity, tt_mean_similarity, path_name)

    return tt_sentence_embeddings


def _log_pooled_embedding_comparison(
    reference_sentence_embeddings: torch.Tensor,
    tt_sentence_embeddings: torch.Tensor,
    *,
    path_name: str,
) -> torch.Tensor:
    reference_mean_similarity = _mean_pairwise_cosine_similarity(reference_sentence_embeddings)
    tt_mean_similarity = _mean_pairwise_cosine_similarity(tt_sentence_embeddings)
    embedding_alignment = _mean_embedding_alignment(reference_sentence_embeddings, tt_sentence_embeddings)

    logger.info(f"{path_name} pooled embedding shape: {tuple(tt_sentence_embeddings.shape)}")
    logger.info(f"{path_name} mean cosine similarity (PyTorch): {reference_mean_similarity:.4f}")
    logger.info(f"{path_name} mean cosine similarity (TTNN): {tt_mean_similarity:.4f}")
    logger.info(f"{path_name} mean embedding alignment: {embedding_alignment:.4f}")

    _assert_mean_cosine_diff_below_tolerance(reference_mean_similarity, tt_mean_similarity, path_name)

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
    # In-vocab non-special token so the model runs a realistic prefill (not pad-token fast path).
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
    warmup_iters: int = 3,
):
    """
    Trace + 2CQ benchmark. The runner captures the encoder forward once and replays it
    with overlapping H2D copies; each timed batch is one ``execute_trace`` plus one
    device synchronize.
    """
    resolved_model_name = _resolve_model_name(model_name, model_location_generator)
    padded_isl = get_padded_sequence_length(isl)

    runner = BgeM3PerformantRunner(
        device=device,
        model_name=resolved_model_name,
        batch_size=max_concurrency,
        sequence_length=max(padded_isl, isl),
        dtype=DEFAULT_TT_DTYPE,
        model_location_generator=model_location_generator,
    )

    input_ids, attention_mask, token_type_ids = _build_synthetic_inputs(
        runner.tokenizer,
        isl,
        max_concurrency,
    )
    position_ids = BgeM3PerformantRunner.build_position_ids(input_ids, runner.pad_token_id)

    for _ in range(warmup_iters):
        runner.run(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )
    ttnn.synchronize_device(device)

    num_batches = max(1, math.ceil(num_requests / max_concurrency))
    effective_num_requests = num_batches * max_concurrency

    batch_latencies_s: list[float] = []
    wall_start = time.perf_counter()
    for _ in range(num_batches):
        batch_start = time.perf_counter()
        runner.run(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )
        ttnn.synchronize_device(device)
        batch_latencies_s.append(time.perf_counter() - batch_start)
    wall_total_s = time.perf_counter() - wall_start

    runner.release()

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
    logger.info(
        "BGE-M3 benchmark summary:\n" + header + "\n" + separator + "\n" + "\n".join(rows) + "\n" + separator,
    )


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 79104, "trace_region_size": 23887872, "num_command_queues": 2}],
    indirect=True,
)
@pytest.mark.parametrize(
    "isl, max_concurrency, num_requests",
    [
        (32, 1, 10),
        (64, 1, 10),
        (128, 1, 10),
        (256, 1, 10),
        (512, 1, 10),
        # Batch-25 sweep: num_requests bumped to 50 (= 2 batches) so we time more
        # than a single trace replay per ISL.
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
