# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Qwen3.8-Flash-Next end-to-end text generation on a 1xTP_SIZE Blackhole mesh (1x4 by default, 1x8 with MESH_DEVICE=P150x8) at batch 1, parametrized by input length.

The model is the vLLM adapter's ``Qwen38ForCausalLM`` on the shared ``mesh_device`` fixture.  The prompt is a chat turn
(thinking disabled) holding a War and Peace excerpt with a pass key hidden in the middle, built to exactly ``seqlen``
tokens (clipped so the context limit holds the prompt and the generated tokens).  Greedy: the device's greedy token
(token_row_io) through a greedy ``sampling_params``, or, with QWEN38_DEMO_HOST_ARGMAX=1, a host argmax over the full
vocabulary logits; 128 decode steps.  Each run asserts the token count and that the decoded text holds the pass key;
``determinism_128`` generates twice and asserts identical tokens.  Under ``CI=true`` the benchmark JSON is written.

Run:  pytest models/demos/blackhole/qwen38_flash_next/demo/text_demo.py -v -s -k "traced_128 and not traced_128k"
Env:  QWEN38_DEMO_HOST_ARGMAX=1 (the A/B route: full logits, host argmax), HF_MODEL (a checkpoint directory, or a
      Hugging Face repo id in the local hub cache; default Qwen/Qwen3.8-Flash-Next), MESH_DEVICE (P150x4 or P150x8;
      it selects the hardware profile), TT_CACHE_PATH (also holds the downloaded book under demo-context),
      TT_METAL_TRACE_ALLOC_TRACKING=1 before ttnn is imported; QWEN38_BF4_CORPUS and
      QWEN38_BF4_CORPUS_VERIFICATION when the BF4 expert cache under TT_CACHE_PATH is not populated yet.
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
from models.demos.blackhole.qwen38_flash_next.checkpoint import checkpoint_directory
from models.demos.blackhole.qwen38_flash_next.demo.long_prompt import (
    build_prompt_ids,
    build_user_prompt_ids,
    load_context_text,
)
from models.demos.blackhole.qwen38_flash_next.tools import hardware_profiles
from models.demos.blackhole.qwen38_flash_next.tools.qwen38_batch_decode import (
    Qwen38PlainLaneDecoder,
    chain_warm_hook,
    lane_page_count,
)
from models.demos.blackhole.qwen38_flash_next.tools.qwen38_chat_session import resolve_route
from models.demos.blackhole.qwen38_flash_next.tools.qwen38_vllm import Qwen38ForCausalLM
from models.demos.blackhole.qwen38_flash_next.ttnn.builder import RESIDENT_CONTEXT_HEADROOM
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import FABRIC_CONFIG, MESH_SHAPE
from models.demos.blackhole.qwen38_flash_next.ttnn.qsa import Qwen38LanePaging
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
    profile = hardware_profiles.resolve_hardware_profile(hardware_profiles.mesh_device_profile_name("p150-x8"))
    hardware_profiles.export_mesh_graph_descriptor(profile)
    if tuple(profile.system_mesh_local_shape) != tuple(MESH_SHAPE):
        pytest.fail(
            f"hardware profile {profile.lane} has the physical mesh shape {tuple(profile.system_mesh_local_shape)}, "
            f"not {tuple(MESH_SHAPE)}; it needs the server's reshape path, which the demo does not support yet"
        )
    if not profile.derive_route:
        logger.info(f"hardware profile {profile.lane} opens the default system mesh without physical device ids")
        return params
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
    try:
        checkpoint = checkpoint_directory()
    except FileNotFoundError as error:
        pytest.fail(str(error))
    if not os.environ.get("TT_CACHE_PATH"):
        pytest.fail("set TT_CACHE_PATH to the cache root (the book download goes to <root>/demo-context)")

    allocated_context = min(context for context in CONTEXTS if context >= seqlen)
    max_seq_len = allocated_context - RESIDENT_CONTEXT_HEADROOM
    prompt_len = min(seqlen, max_seq_len - max_generated_tokens)

    # Build the prompt first so a bad download or tokenizer problem fails before the model loads.
    template = Qwen38OfficialChatTemplate(checkpoint)
    tokenizer = template.tokenizer
    config = json.loads(PROMPT_FILE.read_text(encoding="utf-8"))
    context_text = load_context_text(config, Path(os.environ["TT_CACHE_PATH"]) / "demo-context")
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


BATCH_PAGE_SIZE = 64
BATCH_EXACT_PREFIX = 8
DRIFT_WINDOW = 16


def batch_pool_pages(prompt_lens: list[int], max_generated_tokens: int, page_size: int = BATCH_PAGE_SIZE) -> int:
    """Pool size: the scratch page 0, every user's pages for its prompt and generated tokens, and 8 spare."""

    return 1 + sum(lane_page_count(p, max_generated_tokens, page_size) for p in prompt_lens) + 8


@run_for_blackhole()
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize(
    "seqlen, batch_size",
    [
        pytest.param(128, 8, id="batched_128_b8", marks=pytest.mark.timeout(5400)),
        pytest.param(128, 32, id="batched_128_b32", marks=pytest.mark.timeout(5400)),
        pytest.param(1024, 8, id="batched_1k_b8", marks=pytest.mark.timeout(5400)),
        pytest.param(1024, 32, id="batched_1k_b32", marks=pytest.mark.timeout(5400)),
        pytest.param(4096, 8, id="batched_4k_b8", marks=pytest.mark.timeout(5400)),
        pytest.param(4096, 32, id="batched_4k_b32", marks=pytest.mark.timeout(7200)),
        pytest.param(8192, 8, id="batched_8k_b8", marks=pytest.mark.timeout(5400)),
        pytest.param(16384, 8, id="batched_16k_b8", marks=pytest.mark.timeout(5400)),
        pytest.param(32768, 8, id="batched_32k_b8", marks=pytest.mark.timeout(7200)),
        pytest.param(65536, 8, id="batched_64k_b8", marks=pytest.mark.timeout(9000)),
    ],
)
def test_demo_text_batched(mesh_device, seqlen, batch_size, tmp_path, record_property):
    """B users, each with its own prompt (same ISL, own book excerpt and pass key), decoded together on the plain lane
    body with paged KV: all users are admitted, then 128 tokens per user (t0 from the prefill + 127 batched steps)."""

    max_generated_tokens = 128
    try:
        checkpoint = checkpoint_directory()
    except FileNotFoundError as error:
        pytest.fail(str(error))
    if not os.environ.get("TT_CACHE_PATH"):
        pytest.fail("set TT_CACHE_PATH to the cache root (the book download goes to <root>/demo-context)")

    allocated_context = min(context for context in CONTEXTS if context >= seqlen)
    max_seq_len = allocated_context - RESIDENT_CONTEXT_HEADROOM
    prompt_len = min(seqlen, max_seq_len - max_generated_tokens)

    template = Qwen38OfficialChatTemplate(checkpoint)
    tokenizer = template.tokenizer
    config = json.loads(PROMPT_FILE.read_text(encoding="utf-8"))
    context_text = load_context_text(config, Path(os.environ["TT_CACHE_PATH"]) / "demo-context")
    built = [
        build_user_prompt_ids(
            tokenizer,
            lambda content: template.render([{"role": "user", "content": content}], enable_thinking=False).text,
            config,
            context_text,
            prompt_len,
            user,
        )
        for user in range(batch_size)
    ]
    prompts = [ids for ids, _ in built]
    infos = [info for _, info in built]
    keys = [info["pass_key"] for info in infos]
    assert len(set(keys)) == batch_size and len({tuple(p) for p in prompts}) == batch_size

    max_blocks = -(-allocated_context // BATCH_PAGE_SIZE)
    paging = Qwen38LanePaging(
        page_size=BATCH_PAGE_SIZE,
        num_pages=batch_pool_pages([len(p) for p in prompts], max_generated_tokens),
        max_blocks=-(-max_blocks // 8) * 8,
    )
    mesh_device.enable_program_cache()
    profiler = BenchmarkProfiler()
    profiler.start("run")
    started = time.perf_counter()
    decoder = Qwen38PlainLaneDecoder(batch_size, paging=paging, decode_steps=max_generated_tokens)
    logger.info(
        f"B={batch_size} ISL {prompt_len} (ctx {allocated_context}) indexer form {decoder.indexer_form}; paged KV page "
        f"size {BATCH_PAGE_SIZE}, {paging.num_pages} pages = {paging.pool_rows} pool rows vs "
        f"{batch_size * allocated_context} dense rows; pass keys {keys}"
    )
    with chain_warm_hook(decoder.prepare):
        vm = Qwen38ForCausalLM.initialize_vllm_model(
            SimpleNamespace(_name_or_path=str(checkpoint)), mesh_device, max_batch_size=1, max_seq_len=max_seq_len
        )
    model_ready_s = time.perf_counter() - started
    logger.info(f"model ready in {model_ready_s:.1f} s (lane prepare {decoder.timing.get('allocate_s', 0):.1f} s)")

    try:
        references, ref_ttft, ref_steps = [], [], []
        for user, prompt in enumerate(prompts):  # B=1 references before the lane capture
            tokens, ttft, steps = generate(vm, prompt, max_generated_tokens, profiler)
            references.append(tokens)
            ref_ttft.append(ttft)
            ref_steps.append(sum(steps) / len(steps))
            logger.info(f"reference user {user}: ttft {ttft:.3f} s, {1000 * ref_steps[-1]:.2f} ms/tok, {tokens[:8]}...")
        decoder.capture(vm)
        logger.info(f"lane decoder captured: {decoder.timing}")

        admit_started = time.perf_counter()
        user_prefill_s, user_import_s, firsts = [], [], []
        for user, prompt in enumerate(prompts):
            firsts.append(decoder.admit(user, prompt))
            user_prefill_s.append(decoder.last_admit["prefill_s"])
            user_import_s.append(decoder.last_admit["import_s"])
        admission_s = time.perf_counter() - admit_started

        outputs = [[t] for t in firsts]
        step_ms = []
        current = list(firsts)
        for _ in range(max_generated_tokens - 1):
            step_started = time.perf_counter()
            current = decoder.step(current)
            step_ms.append(1000.0 * (time.perf_counter() - step_started))
            for user in range(batch_size):
                outputs[user].append(current[user])
    finally:
        decoder.release()
        vm.release_persistent_capture()
    profiler.end("run")

    mean_ms = sum(step_ms) / len(step_ms)
    first_ms = sum(step_ms[:DRIFT_WINDOW]) / DRIFT_WINDOW
    last_ms = sum(step_ms[-DRIFT_WINDOW:]) / DRIFT_WINDOW
    decode_tok_s_u = 1000.0 / mean_ms
    aggregate_tok_s = batch_size * decode_tok_s_u
    mean_ttft = sum(user_prefill_s) / batch_size
    ref_ms = 1000.0 * sum(ref_steps) / batch_size
    results = []
    for user in range(batch_size):
        ref, out = references[user], outputs[user]
        diffs = [i for i in range(max_generated_tokens) if out[i] != ref[i]]
        decoded = tokenizer.decode(out, skip_special_tokens=True)
        leaks = [other for other in range(batch_size) if other != user and keys[other] in decoded]
        results.append(
            {
                "user": user,
                "pass_key": keys[user],
                "exact_match": not diffs,
                "first_diff_index": diffs[0] if diffs else None,
                "prefix_match": out[:BATCH_EXACT_PREFIX] == ref[:BATCH_EXACT_PREFIX],
                "own_key_found": keys[user] in decoded,
                "leaked_keys_of_users": leaks,
                "ttft_s": user_prefill_s[user],
                "import_s": user_import_s[user],
                "reference_ttft_s": ref_ttft[user],
                "tokens": out,
                "reference_tokens": ref,
                "decoded": decoded,
            }
        )
        logger.info(
            f"user {user}: exact={'yes' if not diffs else 'no'} first_diff={results[-1]['first_diff_index']} "
            f"own key {'found' if results[-1]['own_key_found'] else 'MISSING'} leaks {leaks}"
        )
    exact = sum(r["exact_match"] for r in results)
    logger.info(
        f"B={batch_size} ISL {prompt_len} (ctx {allocated_context}, indexer {decoder.indexer_form}) | mean TTFT "
        f"{mean_ttft:.3f} s/user | admission {admission_s:.1f} s (prefill {sum(user_prefill_s):.1f} s, import "
        f"{sum(user_import_s):.1f} s) | decode step {mean_ms:.2f} ms (first16 {first_ms:.2f}, last16 {last_ms:.2f}) | "
        f"{decode_tok_s_u:.2f} t/s/u | aggregate {aggregate_tok_s:.1f} tok/s | B=1 reference {ref_ms:.2f} ms/tok | "
        f"exact {exact}/{batch_size} | own key {sum(r['own_key_found'] for r in results)}/{batch_size} | model ready "
        f"{model_ready_s:.1f} s"
    )

    report_path = tmp_path / "batched-generation.json"
    report_path.write_text(
        json.dumps(
            {
                "schema": "qwen38-batched-demo/v1",
                "batch_size": batch_size,
                "seqlen": seqlen,
                "prompt_len": prompt_len,
                "allocated_context": allocated_context,
                "indexer_form": decoder.indexer_form,
                "page_size": BATCH_PAGE_SIZE,
                "num_pages": paging.num_pages,
                "prompt_infos": infos,
                "model_ready_s": model_ready_s,
                "admission_s": admission_s,
                "admission_prefill_s": sum(user_prefill_s),
                "admission_import_s": sum(user_import_s),
                "mean_ttft_s": mean_ttft,
                "decode_step_ms": step_ms,
                "decode_step_ms_mean": mean_ms,
                "decode_step_ms_first16": first_ms,
                "decode_step_ms_last16": last_ms,
                "decode_tok_s_u": decode_tok_s_u,
                "aggregate_decode_tok_s": aggregate_tok_s,
                "reference_decode_ms_per_token": ref_ms,
                "exact_users": exact,
                "users": results,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    record_property("batched_generation_report", str(report_path))

    for r in results:
        user = r["user"]
        assert r["prefix_match"], f"user {user}: first {BATCH_EXACT_PREFIX} tokens differ from the B=1 reference"
        assert r["own_key_found"], f"user {user}: pass key {r['pass_key']} not in {r['decoded'][:300]!r}"
        assert not r["leaked_keys_of_users"], f"user {user}: holds the pass keys of users {r['leaked_keys_of_users']}"

    if os.environ.get("CI") == "true":
        measurements = {
            "compile_prefill": model_ready_s,
            "prefill_time_to_token": mean_ttft,
            "decode_t/s": aggregate_tok_s,
            "decode_t/s/u": decode_tok_s_u,
        }
        benchmark_data = create_benchmark_data(profiler, measurements, {}, {})
        benchmark_data.save_partial_run_json(
            profiler,
            run_type="demo",
            ml_model_name=MODEL_NAME,
            ml_model_type="llm",
            device_name=determine_device_name(mesh_device),
            batch_size=batch_size,
            input_sequence_length=prompt_len,
            output_sequence_length=max_generated_tokens,
        )
