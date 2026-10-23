# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Qwen3.8-Flash-Next end-to-end text generation on a 1x4 Blackhole mesh at batch 1, parametrized by input length.

The model is the vLLM adapter's ``Qwen38ForCausalLM`` on the shared ``mesh_device`` fixture.  The prompt is a chat turn
(thinking disabled) holding a War and Peace excerpt with a pass key hidden in the middle, built to exactly ``seqlen``
tokens (clipped so the context limit holds the prompt and the generated tokens).  Greedy: the device's greedy token
(token_row_io) through a greedy ``sampling_params``, or, with QWEN38_DEMO_HOST_ARGMAX=1, a host argmax over the full
vocabulary logits; 128 decode steps.  Each run asserts the token count and that the decoded text holds the pass key;
``determinism_128`` generates twice and asserts identical tokens.  Under ``CI=true`` the benchmark JSON is written.

Run:  pytest models/demos/blackhole/qwen38_flash_next/demo/text_demo.py -v -s -k "traced_128 and not traced_128k"
Env:  QWEN38_DEMO_HOST_ARGMAX=1 (the A/B route: full logits, host argmax), MODEL_WEIGHTS_DIR (or HF_MODEL, resolved
      through the local Hugging Face cache), QWEN38_CACHE_ROOT (also holds the downloaded book under demo-context), TT_METAL_TRACE_ALLOC_TRACKING=1 before ttnn is imported; QWEN38_BF4_CORPUS and
      QWEN38_BF4_CORPUS_VERIFICATION when the BF4 expert cache under QWEN38_CACHE_ROOT is not populated yet.
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
from models.demos.blackhole.qwen38_flash_next.demo.long_prompt import build_prompt_ids, load_context_text
from models.demos.blackhole.qwen38_flash_next.tools import hardware_profiles
from models.demos.blackhole.qwen38_flash_next.tools.qwen38_chat_session import resolve_route
from models.demos.blackhole.qwen38_flash_next.tools.qwen38_vllm import Qwen38ForCausalLM
from models.demos.blackhole.qwen38_flash_next.ttnn.builder import RESIDENT_CONTEXT_HEADROOM
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import FABRIC_CONFIG, MESH_SHAPE
from models.demos.utils.llm_demo_utils import create_benchmark_data
from models.perf.benchmarking_utils import BenchmarkProfiler
from models.tt_transformers.tt.model_config import determine_device_name

MODEL_NAME = "Qwen/Qwen3.8-Flash-Next"  # normalized by the benchmark writer to the model_targets.yaml key
PROMPT_FILE = Path(__file__).parent / "sample_prompts" / "long_context_needle.json"
CONTEXTS = (32_768, 65_536, 131_072, 262_144)  # the resident contexts the demo uses
MAX_SEQ_LEN = CONTEXTS[0] - RESIDENT_CONTEXT_HEADROOM  # the 32k resident context's limit (tests/test_model_reuse_tp.py)
# The adapter's mesh parameters (the vLLM launch's tt config); the chain allocates its traces itself.
DEVICE_PARAMS = {
    "l1_small_size": 24576,
    "num_command_queues": 2,
    "fabric_config": FABRIC_CONFIG["config"],
    "reliability_mode": FABRIC_CONFIG["reliability_mode"],
    "fabric_tensix_config": FABRIC_CONFIG["fabric_tensix_config"],
    "trace_region_size": 0,
}


@pytest.fixture
def device_params(request):
    """The shared fixture's mesh parameters plus the physical line order the fused line kernels need."""
    params = dict(request.param)
    profile = hardware_profiles.resolve_hardware_profile(os.environ.get("QWEN38_HARDWARE_PROFILE"))
    descriptor = hardware_profiles.mesh_graph_descriptor_path(profile)
    if descriptor is not None and "TT_MESH_GRAPH_DESC_PATH" not in os.environ:
        os.environ["TT_MESH_GRAPH_DESC_PATH"] = str(descriptor.resolve())
    if tuple(profile.system_mesh_local_shape) != tuple(MESH_SHAPE):
        pytest.fail(
            f"hardware profile {profile.lane} has the physical mesh shape {tuple(profile.system_mesh_local_shape)}, "
            f"not {tuple(MESH_SHAPE)}; it needs the server's reshape path, which the demo does not support yet"
        )
    resolved, _derivation = resolve_route(profile)
    params["physical_device_ids"] = list(resolved.route)
    logger.info(f"hardware profile {profile.lane} route {resolved.route}")
    return params


ROUTE_DEVICE = "device greedy token (token_row_io)"
ROUTE_HOST = "full logits to host greedy argmax"


def host_argmax_route() -> bool:
    return os.environ.get("QWEN38_DEMO_HOST_ARGMAX") == "1"


def greedy_sampling_params() -> SimpleNamespace:
    """The plugin's ``TTSamplingParams`` for one greedy row: per-row lists, no penalty, no logprobs, no seed."""

    return SimpleNamespace(
        temperature=[0.0],
        top_k=[1],
        top_p=[0.0],
        presence_penalty=[0.0],
        frequency_penalty=[0.0],
        repetition_penalty=[1.0],
        seed=[None],
        num_logprobs=[0],
        enable_log_probs=[False],
    )


def generate(model: Qwen38ForCausalLM, prompt_ids: list[int], max_generated_tokens: int, profiler: BenchmarkProfiler):
    """Prefill, then greedy decode for the fixed count (no stop on EOS); returns (tokens, ttft_s, step_seconds)."""

    host_argmax = host_argmax_route()
    sampling = {} if host_argmax else {"sampling_params": greedy_sampling_params()}
    profiler.start("inference_prefill")
    started = time.perf_counter()
    out, _rope_deltas = model.prefill_forward(
        tokens=torch.tensor([prompt_ids], dtype=torch.int32),
        page_table=None,
        kv_cache=None,
        enable_trace=True,
        prompt_lens=[len(prompt_ids)],
        start_pos=[0],
        empty_slots=[0],
        **sampling,
    )
    if host_argmax:
        assert torch.isfinite(out[..., :TOKENIZER_SIZE]).all(), "non-finite prefill logits"
        generated = [int(out.argmax())]
    else:
        generated = [int(out.reshape(-1)[0])]
    ttft = time.perf_counter() - started
    profiler.end("inference_prefill")

    profiler.start("inference_decode")
    step_seconds = []
    for step in range(1, max_generated_tokens):
        started = time.perf_counter()
        out = model.decode_forward(
            tokens=torch.tensor([[generated[-1]]], dtype=torch.int32),
            start_pos=[len(prompt_ids) + step - 1],
            page_table=None,
            kv_cache=None,
            enable_trace=True,
            read_from_device=True,
            **sampling,
        )
        if host_argmax:
            assert torch.isfinite(out[..., :TOKENIZER_SIZE]).all(), f"non-finite decode logits at step {step}"
            generated.append(int(out.argmax()))
        else:
            generated.append(int(out.reshape(-1)[0]))
        step_seconds.append(time.perf_counter() - started)
    profiler.end("inference_decode")
    model.release_request(0)
    return generated, ttft, step_seconds


@run_for_blackhole()
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize(
    "seqlen, max_generated_tokens, repeat_runs",
    [
        pytest.param(128, 128, 1, id="traced_128", marks=pytest.mark.timeout(2400)),
        pytest.param(256, 128, 1, id="traced_256", marks=pytest.mark.timeout(2400)),
        pytest.param(1024, 128, 1, id="traced_1k", marks=pytest.mark.timeout(2400)),
        pytest.param(2048, 128, 1, id="traced_2k", marks=pytest.mark.timeout(2400)),
        pytest.param(4096, 128, 1, id="traced_4k", marks=pytest.mark.timeout(2400)),
        pytest.param(8192, 128, 1, id="traced_8k", marks=pytest.mark.timeout(2400)),
        pytest.param(16384, 128, 1, id="traced_16k", marks=pytest.mark.timeout(2400)),
        pytest.param(32768, 128, 1, id="traced_32k", marks=pytest.mark.timeout(3000)),
        pytest.param(65536, 128, 1, id="traced_64k", marks=pytest.mark.timeout(3600)),
        pytest.param(131072, 128, 1, id="traced_128k", marks=pytest.mark.timeout(5400)),
        pytest.param(262144, 128, 1, id="traced_256k", marks=pytest.mark.timeout(9000)),
        pytest.param(128, 128, 2, id="determinism_128", marks=pytest.mark.timeout(2400)),
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
    if not os.environ.get("QWEN38_CACHE_ROOT"):
        pytest.fail("set QWEN38_CACHE_ROOT to the cache root (the book download goes to <root>/demo-context)")

    allocated_context = min(context for context in CONTEXTS if context >= seqlen)
    max_seq_len = allocated_context - RESIDENT_CONTEXT_HEADROOM
    prompt_len = min(seqlen, max_seq_len - max_generated_tokens)
    cache_label = os.environ.get("QWEN38_CACHE_LABEL")
    if cache_label and not cache_label.startswith(f"c{allocated_context}-"):
        pytest.fail(
            f"QWEN38_CACHE_LABEL={cache_label!r} must start with 'c{allocated_context}-' for seqlen {seqlen}; "
            "leave it unset to use the adapter default"
        )

    # Build the prompt first so a bad download or tokenizer problem fails before the model loads.
    template = Qwen38OfficialChatTemplate(checkpoint)
    tokenizer = template.tokenizer
    config = json.loads(PROMPT_FILE.read_text(encoding="utf-8"))
    context_text = load_context_text(config, Path(os.environ["QWEN38_CACHE_ROOT"]) / "demo-context")
    prompt_ids, prompt_info = build_prompt_ids(
        tokenizer,
        lambda content: template.render([{"role": "user", "content": content}], enable_thinking=False).text,
        config,
        context_text,
        prompt_len,
    )

    mesh_device.enable_program_cache()
    profiler = BenchmarkProfiler()
    profiler.start("run")
    profiler.start("compile_prefill")
    model = Qwen38ForCausalLM.initialize_vllm_model(
        SimpleNamespace(_name_or_path=str(checkpoint)), mesh_device, max_batch_size=1, max_seq_len=max_seq_len
    )
    profiler.end("compile_prefill")
    logger.info(f"model ready in {profiler.get_duration('compile_prefill'):.1f} s")

    route = ROUTE_HOST if host_argmax_route() else ROUTE_DEVICE
    logger.info(f"route: {route}")
    try:
        runs = [generate(model, prompt_ids, max_generated_tokens, profiler) for _ in range(repeat_runs)]
    finally:
        model.release_persistent_capture()
    profiler.end("run")

    generated, ttft, step_seconds = runs[0]
    decoded = tokenizer.decode(generated, skip_special_tokens=True)
    pass_key_found = config["pass_key"] in decoded
    prefill_tok_s = prompt_len / ttft
    decode_ms = 1000.0 * sum(step_seconds) / len(step_seconds)
    decode_tok_s_u = 1000.0 / decode_ms
    logger.info(
        f"ISL {prompt_len} (ctx {allocated_context}) TTFT {ttft:.3f} s | prefill {prefill_tok_s:.0f} tok/s | "
        f"decode {decode_ms:.2f} ms/tok ({decode_tok_s_u:.2f} t/s/u) | pass key {'found' if pass_key_found else 'MISSING'}"
    )
    logger.info(f"generated: {decoded!r}")

    report_path = tmp_path / "fixed-length-generation.json"
    report_path.write_text(
        json.dumps(
            {
                "schema": "qwen38-fixed-length-demo/v1",
                "route": route,
                "ignore_eos": True,
                "seqlen": seqlen,
                "prompt_len": prompt_len,
                "allocated_context": allocated_context,
                "prompt_info": prompt_info,
                "prompt_token_ids": prompt_ids,
                "warmup": "model construction and resident trace warmup",
                "runs": [
                    {"generated_token_ids": tokens, "ttft_s": run_ttft, "decode_step_seconds": steps}
                    for tokens, run_ttft, steps in runs
                ],
                "decoded_text": decoded,
                "pass_key_found": pass_key_found,
                "decode_ms_per_token": decode_ms,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    record_property("fixed_length_generation_report", str(report_path))

    assert len(generated) == max_generated_tokens, f"{len(generated)} != {max_generated_tokens}"
    assert len(set(generated)) > 1, f"degenerate output: one token repeated: {generated[:8]}"
    assert pass_key_found, f"pass key {config['pass_key']} not in the output: {decoded[:300]!r}"
    for run, (tokens, _ttft, _steps) in enumerate(runs[1:], start=1):
        assert tokens == generated, f"run {run} differs from run 0:\n{generated}\n{tokens}"

    if os.environ.get("CI") == "true":
        measurements = {
            "compile_prefill": profiler.get_duration("compile_prefill"),
            "prefill_t/s": prefill_tok_s,
            "prefill_time_to_token": ttft,
            "decode_t/s": decode_tok_s_u,
            "decode_t/s/u": decode_tok_s_u,
        }
        benchmark_data = create_benchmark_data(
            profiler, measurements, {"inference_prefill": 0, "inference_decode": 1}, {}
        )
        benchmark_data.save_partial_run_json(
            profiler,
            run_type="demo",
            ml_model_name=MODEL_NAME,
            ml_model_type="llm",
            device_name=determine_device_name(mesh_device),
            batch_size=1,
            input_sequence_length=prompt_len,
            output_sequence_length=len(generated),
        )
