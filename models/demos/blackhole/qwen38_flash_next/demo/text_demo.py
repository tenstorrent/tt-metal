# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Qwen3.8-Flash-Next end-to-end text generation on a 1x4 Blackhole mesh at batch 1: the tiered CI demo.

The model is the vLLM adapter's ``Qwen38ForCausalLM`` on the shared ``mesh_device`` fixture, prefill over a raw-tokenized
128-token prompt, greedy host argmax over the full-vocabulary logits, 50 decode steps.  ``traced_128`` asserts the token
count and emits the benchmark JSON (under ``CI=true``); ``determinism_128`` generates twice and asserts identical tokens.

Run:  pytest models/demos/blackhole/qwen38_flash_next/demo/text_demo.py -v -s -k traced_128
Env:  MODEL_WEIGHTS_DIR (or HF_MODEL, resolved through the local Hugging Face cache), QWEN38_CACHE_ROOT,
      TT_METAL_TRACE_ALLOC_TRACKING=1 before ttnn is imported; QWEN38_BF4_CORPUS and QWEN38_BF4_CORPUS_VERIFICATION
      when the BF4 expert cache under QWEN38_CACHE_ROOT is not populated yet.
"""

import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from loguru import logger

from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen38_flash_next.chat import TOKENIZER_SIZE, Qwen38OfficialChatTemplate
from models.demos.blackhole.qwen38_flash_next.tools.qwen38_vllm import Qwen38ForCausalLM
from models.demos.blackhole.qwen38_flash_next.ttnn.builder import RESIDENT_CONTEXT_HEADROOM
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import FABRIC_CONFIG, MESH_SHAPE
from models.demos.utils.llm_demo_utils import create_benchmark_data
from models.perf.benchmarking_utils import BenchmarkProfiler
from models.tt_transformers.tt.model_config import determine_device_name

MODEL_NAME = "Qwen/Qwen3.8-Flash-Next"  # normalized by the benchmark writer to the model_targets.yaml key
MAX_SEQ_LEN = 32_768 - RESIDENT_CONTEXT_HEADROOM  # the 32k resident context's limit, vLLM's --max-model-len
PROMPT_FILE = Path(__file__).parent / "sample_prompts" / "input_data_questions_prefill_128.json"
# The adapter's mesh parameters (the vLLM launch's tt config); the chain allocates its traces itself.
DEVICE_PARAMS = {
    "l1_small_size": 24576,
    "num_command_queues": 2,
    "fabric_config": FABRIC_CONFIG["config"],
    "reliability_mode": FABRIC_CONFIG["reliability_mode"],
    "fabric_tensix_config": FABRIC_CONFIG["fabric_tensix_config"],
    "trace_region_size": 0,
}


def generate(model: Qwen38ForCausalLM, prompt_ids: list[int], max_generated_tokens: int, profiler: BenchmarkProfiler):
    """Prefill, then greedy decode for the fixed count (no stop on EOS); returns (tokens, ttft_s, step_seconds)."""

    profiler.start("inference_prefill")
    started = time.perf_counter()
    logits, _rope_deltas = model.prefill_forward(
        tokens=torch.tensor([prompt_ids], dtype=torch.int32),
        page_table=None,
        kv_cache=None,
        enable_trace=True,
        prompt_lens=[len(prompt_ids)],
        start_pos=[0],
        empty_slots=[0],
    )
    assert torch.isfinite(logits[..., :TOKENIZER_SIZE]).all(), "non-finite prefill logits"
    generated = [int(logits.argmax())]
    ttft = time.perf_counter() - started
    profiler.end("inference_prefill")

    profiler.start("inference_decode")
    step_seconds = []
    for step in range(1, max_generated_tokens):
        started = time.perf_counter()
        logits = model.decode_forward(
            tokens=torch.tensor([[generated[-1]]], dtype=torch.int32),
            start_pos=[len(prompt_ids) + step - 1],
            page_table=None,
            kv_cache=None,
            enable_trace=True,
            read_from_device=True,
        )
        assert torch.isfinite(logits[..., :TOKENIZER_SIZE]).all(), f"non-finite decode logits at step {step}"
        generated.append(int(logits.argmax()))
        step_seconds.append(time.perf_counter() - started)
    profiler.end("inference_decode")
    model.release_request(0)
    return generated, ttft, step_seconds


@run_for_blackhole()
@pytest.mark.timeout(1200)  # a cold JIT cache builds and warms the chain in about four minutes
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize(
    "seqlen, max_generated_tokens, repeat_runs",
    [
        pytest.param(128, 50, 1, id="traced_128"),
        pytest.param(128, 50, 2, id="determinism_128"),
    ],
)
def test_demo_text(mesh_device, seqlen, max_generated_tokens, repeat_runs, tmp_path, record_property):
    if os.environ.get("MODEL_WEIGHTS_DIR"):
        checkpoint = Path(os.environ["MODEL_WEIGHTS_DIR"])
    elif os.environ.get("HF_MODEL"):  # the snapshot in the local Hugging Face cache, offline
        from huggingface_hub import snapshot_download

        checkpoint = Path(snapshot_download(os.environ["HF_MODEL"], local_files_only=True))
    else:
        pytest.fail("set MODEL_WEIGHTS_DIR to the checkpoint directory or HF_MODEL to its Hugging Face id")
    tokenizer = Qwen38OfficialChatTemplate(checkpoint).tokenizer
    with open(PROMPT_FILE) as handle:
        prompt_ids = tokenizer(json.load(handle)[0]["prompt"])["input_ids"]
    while len(prompt_ids) < seqlen:  # the sample prompt is about 128 tokens: repeat, then clip to seqlen exactly
        prompt_ids = prompt_ids + prompt_ids
    prompt_ids = prompt_ids[:seqlen]

    mesh_device.enable_program_cache()
    profiler = BenchmarkProfiler()
    profiler.start("run")
    profiler.start("compile_prefill")
    model = Qwen38ForCausalLM.initialize_vllm_model(
        SimpleNamespace(_name_or_path=str(checkpoint)), mesh_device, max_batch_size=1, max_seq_len=MAX_SEQ_LEN
    )
    profiler.end("compile_prefill")
    logger.info(f"model ready in {profiler.get_duration('compile_prefill'):.1f} s")

    try:
        runs = [generate(model, prompt_ids, max_generated_tokens, profiler) for _ in range(repeat_runs)]
    finally:
        model.release_persistent_capture()
    profiler.end("run")

    report_path = tmp_path / "fixed-length-generation.json"
    report_path.write_text(
        json.dumps(
            {
                "schema": "qwen38-fixed-length-demo/v1",
                "route": "full logits to host greedy argmax",
                "ignore_eos": True,
                "prompt_token_ids": prompt_ids,
                "warmup": "model construction and resident trace warmup",
                "runs": [
                    {"generated_token_ids": tokens, "ttft_s": ttft, "decode_step_seconds": steps}
                    for tokens, ttft, steps in runs
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    record_property("fixed_length_generation_report", str(report_path))
    generated, ttft, step_seconds = runs[0]
    decode_ms = 1000.0 * sum(step_seconds) / len(step_seconds)
    decode_tok_s = 1000.0 / decode_ms
    logger.info(
        f"prompt {len(prompt_ids)} tokens: ttft {ttft:.3f} s, decode {decode_ms:.2f} ms/token = {decode_tok_s:.2f} t/s"
    )
    logger.info(f"generated: {tokenizer.decode(generated)!r}")
    assert len(generated) == max_generated_tokens, f"{len(generated)} != {max_generated_tokens}"
    for run, (tokens, _ttft, _steps) in enumerate(runs[1:], start=1):
        assert tokens == generated, f"run {run} differs from run 0:\n{generated}\n{tokens}"

    measurements = {
        "compile_prefill": profiler.get_duration("compile_prefill"),
        "prefill_t/s": len(prompt_ids) / ttft,
        "prefill_time_to_token": ttft,
        "decode_t/s": decode_tok_s,
        "decode_t/s/u": decode_tok_s,
    }
    benchmark_data = create_benchmark_data(profiler, measurements, {"inference_prefill": 0, "inference_decode": 1}, {})
    benchmark_data.save_partial_run_json(
        profiler,
        run_type="demo",
        ml_model_name=MODEL_NAME,
        ml_model_type="llm",
        device_name=determine_device_name(mesh_device),
        batch_size=1,
        input_sequence_length=seqlen,
        output_sequence_length=len(generated),
    )
