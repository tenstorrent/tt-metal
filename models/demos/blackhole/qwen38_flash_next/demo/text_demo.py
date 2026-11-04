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
from models.demos.blackhole.qwen38_flash_next.chat import EOS_TOKEN_IDS, TOKENIZER_SIZE, Qwen38OfficialChatTemplate
from models.demos.blackhole.qwen38_flash_next.checkpoint import checkpoint_directory
from models.demos.blackhole.qwen38_flash_next.demo.long_prompt import (
    _BOOK_TOKENS,
    build_prompt_ids,
    build_user_prompt_ids,
    load_context_text,
)
from models.demos.blackhole.qwen38_flash_next.demo.mtp_demo import open_mtp_chain, run_mtp_lanes, run_mtp_single
from models.demos.blackhole.qwen38_flash_next.tools import hardware_profiles
from models.demos.blackhole.qwen38_flash_next.tools.qwen38_batch_decode import (
    Qwen38PlainLaneDecoder,
    chain_warm_hook,
    lane_page_count,
)
from models.demos.blackhole.qwen38_flash_next.tools.qwen38_chat_session import (
    Qwen38ChatChainError,
    Qwen38ChatSession,
    resolve_route,
)
from models.demos.blackhole.qwen38_flash_next.tools.qwen38_lanes_session import Qwen38LanesSession
from models.demos.blackhole.qwen38_flash_next.tools.qwen38_vllm import Qwen38ForCausalLM, plain_lanes_max_batch
from models.demos.blackhole.qwen38_flash_next.ttnn.builder import RESIDENT_CONTEXT_HEADROOM
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import FABRIC_CONFIG, MESH_SHAPE, fabric_config
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
    params["fabric_config"] = fabric_config()["config"]
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


def generate(
    model: Qwen38ForCausalLM,
    prompt_ids: list[int],
    max_generated_tokens: int,
    profiler: BenchmarkProfiler,
    stop_ids: tuple[int, ...] = (),
):
    """Prefill, then greedy decode for the fixed count (no stop on EOS unless ``stop_ids`` is given; the stop id is kept
    as the last token); returns (tokens, ttft_s, step_seconds)."""

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
    lean = not host_argmax and os.environ.get("QWEN38_HOST_LEAN", "1") == "1"
    if lean:  # one input tensor for the whole loop: decode_forward reads the id at once and keeps no reference
        step_tokens = torch.empty((1, 1), dtype=torch.int32)
        step_image = step_tokens.numpy()
    for step in range(1, max_generated_tokens):
        if generated[-1] in stop_ids:
            break
        started = time.perf_counter()
        if lean:
            step_image[0, 0] = generated[-1]
        out = model.decode_forward(
            tokens=step_tokens if lean else torch.tensor([[generated[-1]]], dtype=torch.int32),
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
        elif lean:
            generated.append(out.item())
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
        if (
            os.environ.get("QWEN38_ZERO_DEAD_ROWS_CHECK") == "1"
        ):  # after the traced replays: the pinned buffers' dead rows
            from models.demos.blackhole.qwen38_flash_next.ttnn.fused import program as fused_program

            dead = fused_program.check_dead_rows(mesh_device)
            logger.info(f"ZERO_DEAD_ROWS_CHECK {dead} unpinned_in_capture={fused_program.UNPINNED_IN_CAPTURE}")
            assert dead and all(d == 0 for d, _ in dead.values()), dead
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


MTP_DRAFTS = 4
CHAT_MAX_NEW_TOKENS = 512  # the chat prompts stop at EOS; the cap only guards a runaway
CHAT_WARMUP_TOKENS = 48
CHAT_PROMPT_FILE = Path(__file__).with_name("chat_prompts.json")  # the 4 prompts of the MTP chat bench (client2.py)
CHAT_NAMES = ("P1", "P2", "P3", "P4")  # passage summary, GSM8K, Fibonacci function, transformer explanation


def build_chat_prompts(checkpoint):
    """(template, [(name, ids)], warm-up ids): the bench's chat messages through the model's chat template with thinking
    off, which is what the chat server renders for them.  The warm-up prompt is P3 (as the bench's)."""

    template = Qwen38OfficialChatTemplate(checkpoint)
    texts = json.loads(CHAT_PROMPT_FILE.read_text(encoding="utf-8"))["prompts"]

    def ids_of(name):
        rendered = template.render([{"role": "user", "content": texts[name]}], enable_thinking=False)
        return [int(token) for token in rendered.input_ids[0].tolist()]

    return template, [(name, ids_of(name)) for name in CHAT_NAMES], ids_of("P3")


def chat_row(name, isl, ttft_s, tokens, decode_s, passes, accepted, finish):
    return {
        "prompt": name,
        "isl": isl,
        "ttft_s": ttft_s,
        "generated_tokens": tokens,
        "decode_s": decode_s,
        "decode_ms_per_token": 1000.0 * decode_s / tokens,
        "tok_s": tokens / decode_s,
        "passes": passes,
        "accepted_drafts": accepted,
        "tokens_per_pass": tokens / passes,
        "finish": finish,
    }


def chat_totals(rows):
    tokens = sum(row["generated_tokens"] for row in rows)
    decode_s = sum(row["decode_s"] for row in rows)
    passes = sum(row["passes"] for row in rows)
    return {
        "generated_tokens": tokens,
        "decode_s": decode_s,
        "decode_ms_per_token": 1000.0 * decode_s / tokens,
        "tok_s": tokens / decode_s,
        "passes": passes,
        "accepted_drafts": sum(row["accepted_drafts"] for row in rows),
        "tokens_per_pass": tokens / passes,
    }


def log_chat_table(label, rows, totals):
    logger.info(f"{label}: prompt ISL TTFT_s tokens decode_s ms/tok tok/s passes accepted tok/pass finish")
    for row in rows:
        logger.info(
            f"{label}: {row['prompt']} {row['isl']} {row['ttft_s']:.3f} {row['generated_tokens']} {row['decode_s']:.4f} "
            f"{row['decode_ms_per_token']:.2f} {row['tok_s']:.1f} {row['passes']} {row['accepted_drafts']} "
            f"{row['tokens_per_pass']:.2f} {row['finish']}"
        )
    logger.info(
        f"{label}: ALL tokens={totals['generated_tokens']} decode_s={totals['decode_s']:.4f} "
        f"ms/tok={totals['decode_ms_per_token']:.2f} tok/s={totals['tok_s']:.1f} passes={totals['passes']} "
        f"accepted={totals['accepted_drafts']} tok/pass={totals['tokens_per_pass']:.2f}"
    )


def write_chat_report(tmp_path, record_property, schema, extra):
    report_path = tmp_path / "fixed-length-generation.json"
    report_path.write_text(json.dumps({"schema": schema, **extra}, indent=2) + "\n", encoding="utf-8")
    record_property("fixed_length_generation_report", str(report_path))


@run_for_blackhole()
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize(
    "case",
    [
        pytest.param("chat_mtp4", id="chat_mtp4", marks=pytest.mark.timeout(3600)),
        pytest.param("chat_mtp3", id="chat_mtp3", marks=pytest.mark.timeout(3600)),
        pytest.param("determinism", id="determinism_128_mtp4", marks=pytest.mark.timeout(2400)),
    ],
)
def test_demo_text_mtp(mesh_device, case, tmp_path, record_property):
    """Batch 1 with MTP speculative decode (4 drafts, greedy): the chat server's chain and session without the server.
    chat_mtp4: the 4 chat prompts one after another (session reset before each), each until EOS or 512 tokens, after
    one uncounted warm-up prompt.  determinism_128_mtp4: the first chat prompt twice, 128 tokens, equal streams."""

    drafts = 3 if case == "chat_mtp3" else MTP_DRAFTS
    try:
        checkpoint = checkpoint_directory()
    except FileNotFoundError as error:
        pytest.fail(str(error))

    max_seq_len = CONTEXTS[0] - RESIDENT_CONTEXT_HEADROOM
    template, prompts, warmup_ids = build_chat_prompts(checkpoint)
    tokenizer = template.tokenizer
    stop_ids = tuple(EOS_TOKEN_IDS)

    mesh_device.enable_program_cache()
    started = time.perf_counter()
    try:
        chain = open_mtp_chain(mesh_device, checkpoint, max_seq_len, drafts)
    except Qwen38ChatChainError as error:
        pytest.skip(f"MTP admission refused at context {CONTEXTS[0]}: {error}")
    logger.info(f"MTP chain ready in {time.perf_counter() - started:.1f} s")
    try:
        session = Qwen38ChatSession(chain, template, prefill_mode="chunked")
        run_mtp_single(session, warmup_ids, CHAT_WARMUP_TOKENS, stop_ids=stop_ids)
        if case == "determinism":
            runs = [run_mtp_single(session, prompts[0][1], 128, stop_ids=stop_ids) for _ in range(2)]
            for index, run in enumerate(runs):
                logger.info(
                    f"determinism run {index}: {len(run['tokens'])} tokens, TTFT {run['ttft_s']:.3f} s, decode "
                    f"{run['decode_ms_per_token']:.2f} ms/tok, tokens/pass {run['chain_tokens_per_pass']:.3f}"
                )
            write_chat_report(
                tmp_path,
                record_property,
                "qwen38-fixed-length-demo/v1-mtp-chat",
                {
                    "drafts": drafts,
                    "prompt": prompts[0][0],
                    "runs": [
                        {"generated_token_ids": r["tokens"], **{k: v for k, v in r.items() if k != "tokens"}}
                        for r in runs
                    ],
                },
            )
            assert not session.poisoned
            assert runs[0]["tokens"] == runs[1]["tokens"], "run 1 differs from run 0"
            assert 8 < len(runs[0]["tokens"]) <= 128 and len(set(runs[0]["tokens"])) > 1
            return
        runs = []
        for name, ids in prompts:
            runs.append(run_mtp_single(session, ids, CHAT_MAX_NEW_TOKENS, stop_ids=stop_ids))
            logger.info(f"{name}: {tokenizer.decode(runs[-1]['tokens'], skip_special_tokens=True)!r}")
        assert not session.poisoned
    finally:
        chain.close()

    # tokens/pass as the chat bench counts it: generated tokens over the chain's passes (the bootstrap pass included)
    rows = [
        chat_row(
            name,
            len(ids),
            run["ttft_s"],
            run["generated_tokens"],
            run["decode_s"],
            run["chain_passes"],
            run["chain_accepted_drafts"],
            run["finish"],
        )
        for (name, ids), run in zip(prompts, runs)
    ]
    totals = chat_totals(rows)
    log_chat_table(f"chat_mtp{drafts}", rows, totals)
    write_chat_report(
        tmp_path,
        record_property,
        "qwen38-chat-demo/v1-mtp",
        {
            "drafts": drafts,
            "max_new_tokens": CHAT_MAX_NEW_TOKENS,
            "stop_ids": list(stop_ids),
            "rows": rows,
            "totals": totals,
            "runs": [
                {
                    "prompt": name,
                    "generated_token_ids": run["tokens"],
                    "decoded": tokenizer.decode(run["tokens"], skip_special_tokens=True),
                }
                for (name, _ids), run in zip(prompts, runs)
            ],
        },
    )
    for row, run in zip(rows, runs):
        assert row["finish"] in ("stop", "length"), (row["prompt"], row["finish"])
        assert row["generated_tokens"] > 8 and len(set(run["tokens"])) > 1, (row["prompt"], run["tokens"][:8])
    assert 1.0 < totals["tokens_per_pass"] <= drafts + 1, totals["tokens_per_pass"]


@run_for_blackhole()
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("case", [pytest.param("chat_plain", id="chat_plain", marks=pytest.mark.timeout(3600))])
def test_demo_text_chat_plain(mesh_device, case, tmp_path, record_property):
    """The same 4 chat prompts as chat_mtp4 on the normal batch-1 demo chain without MTP: each until EOS or 512 tokens,
    after one uncounted warm-up prompt, so the MTP speedup is measured on the same prompts."""

    try:
        checkpoint = checkpoint_directory()
    except FileNotFoundError as error:
        pytest.fail(str(error))

    max_seq_len = CONTEXTS[0] - RESIDENT_CONTEXT_HEADROOM
    template, prompts, warmup_ids = build_chat_prompts(checkpoint)
    tokenizer = template.tokenizer
    stop_ids = tuple(EOS_TOKEN_IDS)

    mesh_device.enable_program_cache()
    profiler = BenchmarkProfiler()
    model = Qwen38ForCausalLM.initialize_vllm_model(
        SimpleNamespace(_name_or_path=str(checkpoint)), mesh_device, max_batch_size=1, max_seq_len=max_seq_len
    )
    runs = []
    try:
        generate(model, warmup_ids, CHAT_WARMUP_TOKENS, profiler, stop_ids)
        for name, ids in prompts:
            runs.append(generate(model, ids, CHAT_MAX_NEW_TOKENS, profiler, stop_ids))
            logger.info(f"{name}: {tokenizer.decode(runs[-1][0], skip_special_tokens=True)!r}")
    finally:
        model.release_persistent_capture()

    rows = []
    for (name, ids), (tokens, ttft, steps) in zip(prompts, runs):
        finish = "stop" if tokens[-1] in stop_ids else "length"
        # one pass per decode step; the first token comes from the prefill, so the stream has len(steps) + 1 tokens
        rows.append(chat_row(name, len(ids), ttft, len(tokens), sum(steps), len(steps), 0, finish))
    totals = chat_totals(rows)
    log_chat_table("chat_plain", rows, totals)
    write_chat_report(
        tmp_path,
        record_property,
        "qwen38-chat-demo/v1-plain",
        {
            "max_new_tokens": CHAT_MAX_NEW_TOKENS,
            "stop_ids": list(stop_ids),
            "rows": rows,
            "totals": totals,
            "runs": [
                {
                    "prompt": name,
                    "generated_token_ids": tokens,
                    "decoded": tokenizer.decode(tokens, skip_special_tokens=True),
                }
                for (name, _ids), (tokens, _ttft, _steps) in zip(prompts, runs)
            ],
        },
    )
    for row, (tokens, _ttft, _steps) in zip(rows, runs):
        assert row["generated_tokens"] > 8 and len(set(tokens)) > 1, (row["prompt"], tokens[:8])


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
        pytest.param(128, None, id="batched_128_bmax", marks=pytest.mark.timeout(5400)),
        pytest.param(256, None, id="batched_256_bmax", marks=pytest.mark.timeout(5400)),
        pytest.param(1024, None, id="batched_1k_bmax", marks=pytest.mark.timeout(5400)),
        pytest.param(2048, None, id="batched_2k_bmax", marks=pytest.mark.timeout(5400)),
        pytest.param(4096, None, id="batched_4k_bmax", marks=pytest.mark.timeout(7200)),
        pytest.param(8192, None, id="batched_8k_bmax", marks=pytest.mark.timeout(7200)),
        pytest.param(16384, None, id="batched_16k_bmax", marks=pytest.mark.timeout(9000)),
        pytest.param(32768, None, id="batched_32k_bmax", marks=pytest.mark.timeout(10800)),
        pytest.param(65536, None, id="batched_64k_bmax", marks=pytest.mark.timeout(14400)),
        pytest.param(131072, None, id="batched_128k_bmax", marks=pytest.mark.timeout(21600)),
        pytest.param(262144, None, id="batched_256k_bmax", marks=pytest.mark.timeout(28800)),
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
    bmax_record = None
    if batch_size is None:
        bmax_record = plain_lanes_max_batch(prompt_len, max_generated_tokens, allocated_context, BATCH_PAGE_SIZE)
        batch_size = bmax_record["B"]
        # each user reads its own non-overlapping excerpt, so the book text also caps B
        render = lambda content: template.render([{"role": "user", "content": content}], enable_thinking=False).text
        _, info0 = build_prompt_ids(tokenizer, render, config, context_text, prompt_len)
        book_cap = len(_BOOK_TOKENS[context_text]) // info0["context_tokens"]
        bmax_record["book_cap"] = book_cap
        if book_cap < batch_size:
            batch_size = book_cap
            bmax_record["B"] = book_cap
            bmax_record["limiting"] = "book_text"
        if os.environ.get("QWEN38_BMAX_OVERRIDE"):  # the model is an estimate: a real out-of-memory B is retried lower
            batch_size = int(os.environ["QWEN38_BMAX_OVERRIDE"])
            bmax_record["override"] = batch_size
        logger.info(f"max batch B={batch_size} limited by {bmax_record['limiting']}: {bmax_record}")
        record_property("bmax_record", json.dumps(bmax_record))
        assert batch_size >= 1, f"no lane fits at ISL {prompt_len}, context {allocated_context}: {bmax_record}"

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
                "bmax_record": bmax_record,
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


MTP_LANES = 8
MTP_LANE_DRAFTS = 3  # 8 lanes x (3 + 1) rows fill the 32-row tile; 4 drafts do not fit


@run_for_blackhole()
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize(
    "device_params", [{**DEVICE_PARAMS, "num_command_queues": 1}], indirect=True
)  # the lanes' one-queue form
@pytest.mark.parametrize(
    "seqlen",
    [
        pytest.param(128, id="batched_128_b8_mtp3", marks=pytest.mark.timeout(5400)),
        pytest.param(1024, id="batched_1k_b8_mtp3", marks=pytest.mark.timeout(5400)),
        pytest.param(4096, id="batched_4k_b8_mtp3", marks=pytest.mark.timeout(5400)),
        pytest.param(8192, id="batched_8k_b8_mtp3", marks=pytest.mark.timeout(5400)),
        pytest.param(16384, id="batched_16k_b8_mtp3", marks=pytest.mark.timeout(5400)),
        pytest.param(32768, id="batched_32k_b8_mtp3", marks=pytest.mark.timeout(7200)),
        pytest.param(65536, id="batched_64k_b8_mtp3", marks=pytest.mark.timeout(9000)),
    ],
)
def test_demo_text_batched_mtp(mesh_device, seqlen, tmp_path, record_property):
    """8 users, each with its own prompt, decoded together by the MTP lane scheduler (3 drafts): 128 tokens per user
    (token 1 from the admission, the rest from the passes).  Each prompt is first run alone on the same chain as the
    reference for the lane's first tokens."""

    batch_size, drafts, max_generated_tokens = MTP_LANES, MTP_LANE_DRAFTS, 128
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

    mesh_device.enable_program_cache()
    started = time.perf_counter()
    lanes_session = Qwen38LanesSession(lanes=batch_size, drafts=drafts, marker=logger.info)
    chain = None
    try:
        try:
            chain = open_mtp_chain(mesh_device, checkpoint, max_seq_len, drafts, lanes_session=lanes_session)
        except Qwen38ChatChainError as error:
            pytest.skip(f"MTP lanes admission refused at context {allocated_context}: {error}")
        model_ready_s = time.perf_counter() - started
        logger.info(f"MTP lanes chain ready in {model_ready_s:.1f} s; pass keys {keys}")
        session = Qwen38ChatSession(chain, template, prefill_mode="chunked")
        references = []
        for user, prompt in enumerate(prompts):  # single-stream runs on the same chain form, before the lane capture
            reference = run_mtp_single(session, prompt, max_generated_tokens)
            references.append(reference)
            logger.info(
                f"reference user {user}: ttft {reference['ttft_s']:.3f} s, {reference['decode_ms_per_token']:.2f} "
                f"ms/tok, tokens/pass {reference['tokens_per_pass']:.3f}, {reference['tokens'][:8]}..."
            )
        session.reset()
        lanes_session.capture(chain, session)
        run = run_mtp_lanes(lanes_session, prompts, drafts, max_generated_tokens)
        assert not session.poisoned
    except BaseException:  # the mesh close after a failed lane run can take minutes: keep the cause in the log
        logger.exception("MTP lanes run failed")
        raise
    finally:
        if chain is not None:
            lanes_session.close()
            chain.close()

    results = []
    for user, row in enumerate(run["lanes_rows"]):
        out, ref = row["tokens"], references[user]["tokens"]
        diffs = [i for i in range(min(len(out), len(ref))) if out[i] != ref[i]]
        decoded = tokenizer.decode(out, skip_special_tokens=True)
        leaks = [other for other in range(batch_size) if other != user and keys[other] in decoded]
        results.append(
            {
                **{k: v for k, v in row.items() if k != "tokens"},
                "user": user,
                "pass_key": keys[user],
                "own_key_found": keys[user] in decoded,
                "leaked_keys_of_users": leaks,
                "exact_match": len(out) == len(ref) and not diffs,
                "first_diff_index": diffs[0] if diffs else None,
                "prefix_match": out[:BATCH_EXACT_PREFIX] == ref[:BATCH_EXACT_PREFIX],
                "tokens": out,
                "reference_tokens": ref,
                "reference_ttft_s": references[user]["ttft_s"],
                "decoded": decoded,
            }
        )
        logger.info(
            f"user {user}: {len(out)} tokens {row['finish']} ttft own {row['ttft_own_prefill_s']:.3f} s / from batch "
            f"start {row['ttft_from_batch_start_s']:.3f} s | passes {row['passes']} tokens/pass "
            f"{row['tokens_per_pass']:.3f} | vs single K={drafts}: exact={'yes' if results[-1]['exact_match'] else 'no'} "
            f"first_diff={results[-1]['first_diff_index']} | own key {'found' if results[-1]['own_key_found'] else 'MISSING'} "
            f"leaks {leaks}"
        )
    exact = sum(r["exact_match"] for r in results)
    mean_ttft = sum(r["ttft_own_prefill_s"] for r in results) / batch_size
    logger.info(
        f"B={batch_size} MTP k={drafts} ISL {prompt_len} (ctx {allocated_context}) | mean TTFT {mean_ttft:.3f} s/user "
        f"(last lane {max(r['ttft_from_batch_start_s'] for r in results):.3f} s from batch start) | pass "
        f"{run['pass_ms_mean']:.2f} ms (median {run['pass_ms_median']:.2f}) | decode "
        f"{run['decode_ms_per_token_per_user']:.2f} ms/tok/user | {run['per_user_tok_s']:.2f} t/s/u | aggregate "
        f"{run['aggregate_tok_s']:.1f} tok/s | tokens/pass {run['tokens_per_pass_mean']:.3f} | exact vs single "
        f"{exact}/{batch_size} | own key {sum(r['own_key_found'] for r in results)}/{batch_size} | model ready "
        f"{model_ready_s:.1f} s"
    )

    report_path = tmp_path / "batched-generation.json"
    report_path.write_text(
        json.dumps(
            {
                "schema": "qwen38-batched-demo/v1-mtp",
                "batch_size": batch_size,
                "drafts": drafts,
                "seqlen": seqlen,
                "prompt_len": prompt_len,
                "allocated_context": allocated_context,
                "prompt_infos": infos,
                "model_ready_s": model_ready_s,
                "lanes_admission": lanes_session.admission,
                "mean_ttft_own_prefill_s": mean_ttft,
                "pass_ms": run["pass_ms"],
                "pass_ms_mean": run["pass_ms_mean"],
                "pass_ms_median": run["pass_ms_median"],
                "decode_window_s": run["decode_window_s"],
                "decode_ms_per_token_per_user": run["decode_ms_per_token_per_user"],
                "decode_tok_s_u": run["per_user_tok_s"],
                "aggregate_decode_tok_s": run["aggregate_tok_s"],
                "tokens_per_pass_mean": run["tokens_per_pass_mean"],
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
        assert (
            r["finish"] == "length" and len(r["tokens"]) == max_generated_tokens
        ), f"user {user}: {len(r['tokens'])} tokens, finish {r['finish']}"
        assert r["own_key_found"], f"user {user}: pass key {r['pass_key']} not in {r['decoded'][:300]!r}"
        assert not r["leaked_keys_of_users"], f"user {user}: holds the pass keys of users {r['leaked_keys_of_users']}"
        assert (
            r["passes"] > 0 and 1.0 < r["tokens_per_pass"] <= drafts + 1
        ), f"user {user}: passes {r['passes']} tokens/pass {r['tokens_per_pass']}"
        assert r["prefix_match"], f"user {user}: first {BATCH_EXACT_PREFIX} tokens differ from the single-stream run"


@run_for_blackhole()
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize(
    "device_params", [{**DEVICE_PARAMS, "num_command_queues": 1}], indirect=True
)  # the lanes' one-queue form
@pytest.mark.parametrize("case", [pytest.param("chat_b8_mtp3", id="chat_b8_mtp3", marks=pytest.mark.timeout(5400))])
def test_demo_text_batched_mtp_chat(mesh_device, case, tmp_path, record_property):
    """8 lanes at 3 drafts: the 4 chat prompts twice (lane i gets prompt i % 4), greedy, each lane stops at its own EOS
    or at 512 tokens (the lane scheduler's per-ticket stop).  Each prompt is first run alone at 3 drafts on the same
    chain form (batch-1 reference, after one warm-up prompt).  Lanes 0-3 and 4-7 must be equal (same prompts); the
    equality with the batch-1 run is reported, not asserted."""

    batch_size, drafts = MTP_LANES, MTP_LANE_DRAFTS
    try:
        checkpoint = checkpoint_directory()
    except FileNotFoundError as error:
        pytest.fail(str(error))

    max_seq_len = CONTEXTS[0] - RESIDENT_CONTEXT_HEADROOM
    template, chat_prompts, warmup_ids = build_chat_prompts(checkpoint)
    tokenizer = template.tokenizer
    stop_ids = tuple(EOS_TOKEN_IDS)
    lane_prompts = [chat_prompts[user % len(chat_prompts)] for user in range(batch_size)]

    mesh_device.enable_program_cache()
    started = time.perf_counter()
    lanes_session = Qwen38LanesSession(lanes=batch_size, drafts=drafts, marker=logger.info)
    chain = None
    try:
        try:
            chain = open_mtp_chain(mesh_device, checkpoint, max_seq_len, drafts, lanes_session=lanes_session)
        except Qwen38ChatChainError as error:
            pytest.skip(f"MTP lanes admission refused at context {CONTEXTS[0]}: {error}")
        logger.info(f"MTP lanes chain ready in {time.perf_counter() - started:.1f} s")
        session = Qwen38ChatSession(chain, template, prefill_mode="chunked")
        run_mtp_single(session, warmup_ids, CHAT_WARMUP_TOKENS, stop_ids=stop_ids)
        references = {}
        for name, ids in chat_prompts:  # batch-1 runs on the same chain form, before the lane capture
            references[name] = run_mtp_single(session, ids, CHAT_MAX_NEW_TOKENS, stop_ids=stop_ids)
            logger.info(
                f"batch-1 K={drafts} {name}: {references[name]['generated_tokens']} tokens {references[name]['finish']}, "
                f"tokens/pass {references[name]['tokens_per_pass']:.3f}"
            )
        session.reset()
        lanes_session.capture(chain, session)
        run = run_mtp_lanes(
            lanes_session, [ids for _name, ids in lane_prompts], drafts, CHAT_MAX_NEW_TOKENS, stop_ids=stop_ids
        )
        assert not session.poisoned
    except BaseException:  # the mesh close after a failed lane run can take minutes: keep the cause in the log
        logger.exception("MTP lanes chat run failed")
        raise
    finally:
        if chain is not None:
            lanes_session.close()
            chain.close()

    results = []
    for user, row in enumerate(run["lanes_rows"]):
        name, ids = lane_prompts[user]
        out, ref = row["tokens"], references[name]["tokens"]
        diffs = [i for i in range(min(len(out), len(ref))) if out[i] != ref[i]]
        first_diff = diffs[0] if diffs else (None if len(out) == len(ref) else min(len(out), len(ref)))
        results.append(
            {
                **{k: v for k, v in row.items() if k != "tokens"},
                "user": user,
                "prompt": name,
                "isl": len(ids),
                "tok_s": (row["committed"] - 1) / row["decode_s"],
                "equal_to_batch1_k3": first_diff is None,
                "first_diff_index_vs_batch1": first_diff,
                "batch1_tokens": len(ref),
                "tokens": out,
                "decoded": tokenizer.decode(out, skip_special_tokens=True),
            }
        )
        logger.info(
            f"lane {user} {name}: ISL {len(ids)} | TTFT own {row['ttft_own_prefill_s']:.3f} s / from batch start "
            f"{row['ttft_from_batch_start_s']:.3f} s | {row['committed']} tokens {row['finish']} | decode "
            f"{row['decode_s']:.3f} s | {results[-1]['tok_s']:.2f} tok/s | passes {row['passes']} tokens/pass "
            f"{row['tokens_per_pass']:.3f} | vs batch-1 K={drafts}: equal={results[-1]['equal_to_batch1_k3']} "
            f"first_diff={first_diff} (batch-1 {len(ref)} tokens)"
        )
    logger.info(
        f"chat_b8_mtp3: total tokens {run['total_generated_tokens']} | decode window {run['decode_window_s']:.3f} s | "
        f"aggregate {run['aggregate_all_tokens_tok_s']:.1f} tok/s (token 1 of each lane counted; "
        f"{run['aggregate_tok_s']:.1f} without it) | mean per-user {run['mean_per_user_tok_s']:.2f} tok/s | pass "
        f"{run['pass_ms_mean']:.2f} ms (median {run['pass_ms_median']:.2f}) | scheduler passes {run['scheduler_passes']}"
    )
    write_chat_report(
        tmp_path,
        record_property,
        "qwen38-chat-demo/v1-mtp-lanes",
        {
            "drafts": drafts,
            "lanes": batch_size,
            "max_new_tokens": CHAT_MAX_NEW_TOKENS,
            "stop_ids": list(stop_ids),
            "batch1_references": {
                name: {k: v for k, v in ref.items() if k != "tokens"} | {"tokens": ref["tokens"]}
                for name, ref in references.items()
            },
            "run": {k: v for k, v in run.items() if k not in ("lanes_rows", "pass_ms")},
            "lanes_rows": results,
        },
    )

    for r in results:
        assert r["finish"] in ("stop", "length"), (r["user"], r["finish"])
        assert r["passes"] > 0 and 1.0 < r["tokens_per_pass"] <= drafts + 1, (
            r["user"],
            r["passes"],
            r["tokens_per_pass"],
        )
    for user in range(batch_size // 2):
        a, b = results[user], results[user + batch_size // 2]
        assert a["tokens"] == b["tokens"], f"lanes {user} and {user + batch_size // 2} (same prompt) differ"
