# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Multi-user regression sweep for GPT-OSS on single-row meshes (e.g. 8x Blackhole P150, 1x8).

One pytest case per batch size (1, 2, 4, 8, 16, 32): the model is built for that batch, then every
(input length, output length) pair of the tt-inference-server benchmark sweep
(reference_config/benchmarking/benchmark_config.py::BENCHMARK_ISL_OSL_PAIRS) that fits the per-batch
context budget is run the way the benchmark client does it: all users prefilled, then exactly OSL
decode steps with no EOS stopping. The context budget mirrors the server's concurrency cap
(``max_tokens_all_users // (isl + osl)``): a batch of B users gets ``min(64K, 512K // B)`` tokens of
context each, so long inputs are swept at small batch sizes only.

Per run the sweep records prefill time, decode step statistics (mean / p50 / p99, first traced step
separately), per-user and aggregate tokens/s, and checks the outputs:
  * every generated token id is a valid vocab id and the text decodes;
  * outputs are not degenerate (no long runs of one token, reasonable token diversity);
  * for the 128-token QA prompts, the answer keyword appears in the generated text for >= 75% of users;
  * for the long prompts (same prompt for every user) the users' outputs are compared with each other,
    which doubles as a cross-user isolation signal (reported, not asserted: greedy decode is not
    bit-reproducible on device and near-tie tokens can flip).

Results are appended to generated/gpt_oss_multi_user_regression/<model>_<mesh>.jsonl and printed as a
Markdown table at the end of each case.

    export HF_MODEL=/path/to/gpt-oss-120b   # or gpt-oss-20b
    pytest models/demos/gpt_oss/tests/test_multi_user_regression.py -k 1x8                # all batch sizes
    pytest models/demos/gpt_oss/tests/test_multi_user_regression.py -k "1x8 and batch32"   # one batch size
    GPT_OSS_REGRESSION_PAIRS="128:128,1024:128" pytest ... -k 1x8                          # subset of pairs
"""

import json
import os
import re
import statistics
import subprocess
import time
from datetime import datetime
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.common.sampling import SamplingParams
from models.demos.gpt_oss.demo.text_demo import prepare_gpt_oss_generator_args
from models.demos.gpt_oss.tests.test_factory import TestFactory, parametrize_mesh_with_fabric
from models.tt_transformers.demo.simple_text_demo import load_inputs
from models.tt_transformers.tt.common import get_padded_prefill_len, preprocess_inputs_prefill
from models.tt_transformers.tt.generator import Generator

# tt-inference-server BENCHMARK_ISL_OSL_PAIRS, minus the points this box cannot host (>= 64K per user on
# a single-row mesh, see text_demo's OOM skip) and the 10000-token point (no matching sample prompt).
ISL_OSL_PAIRS = [
    (128, 128),
    (128, 1024),
    (1024, 128),
    (2048, 128),
    (4096, 128),
    (8192, 128),
    (8192, 1024),
    (16384, 128),
    (32768, 128),
]
BATCH_SIZES = [1, 2, 4, 8, 16, 32]
# Total KV tokens the sweep may allocate per mesh (512K tokens = 8192 blocks of 64 = ~2.6 GB per device on
# the 120B at batch >= 8; smaller batches are capped at 64K per user and allocate less). Mirrors the server's
# context-capped concurrency: a batch of B gets min(64K, 512K // B) tokens of context each.
TOTAL_KV_TOKENS = int(os.getenv("GPT_OSS_REGRESSION_KV_TOKENS", 512 * 1024))  # env override for smaller DRAM budgets
MAX_CONTEXT_PER_USER = 64 * 1024
BLOCK_SIZE = 64

SAMPLE_PROMPTS = "models/tt_transformers/demo/sample_prompts"
PROMPT_FILES = {
    128: "models/demos/gpt_oss/demo/sample_prompts/input_data_questions_prefill_128.json",
    1024: f"{SAMPLE_PROMPTS}/input_data_long_1k.json",
    2048: f"{SAMPLE_PROMPTS}/input_data_long_2k.json",
    4096: f"{SAMPLE_PROMPTS}/input_data_long_4k.json",
    8192: f"{SAMPLE_PROMPTS}/input_data_long_8k.json",
    16384: f"{SAMPLE_PROMPTS}/input_data_long_16k.json",
    32768: f"{SAMPLE_PROMPTS}/input_data_long_32k.json",
}

# Expected answer keywords for the 32 QA prompts (any keyword matching, case-insensitive, counts).
QA_KEYWORDS = [
    ["one natural", "one moon", "1 moon", "single moon", "one natural satellite"],  # moons of Earth
    ["blue"],  # colour of the sky
    ["bell"],  # telephone
    ["7"],  # sqrt(49)
    ["366"],  # leap year
    ["leonardo", "da vinci"],  # Mona Lisa
    ["paris"],  # capital of France
    ["30"],  # days in June
    ["giraffe"],  # tallest mammal
    ["fleming"],  # penicillin
    ["100"],  # boiling point (°C)
    ["six", "6"],  # hexagon sides
    ["venus", "mercury"],  # closest planet (both defensible answers)
    ["herbert"],  # Dune
    ["hydrogen"],  # atomic number one
    ["1000", "1,000", "thousand"],  # metres in a kilometre
    ["lion"],  # king of beasts
    ["einstein"],  # relativity
    ["spanish", "castilian"],  # main language in Spain
    ["three", "3"],  # octopus hearts
    ["carbon dioxide", "co2", "co₂"],  # gas breathed out
    ["beethoven"],  # Fifth Symphony
    ["pacific"],  # largest ocean
    ["365"],  # orbit period
    ["guitar"],  # six strings
    ["armstrong"],  # first on the Moon
    ["orange", "kiwi", "citrus", "guava", "strawberr", "lemon", "papaya", "blackcurrant", "acerola"],  # vitamin C
    ["eight", "8"],  # planets
    ["0", "32"],  # freezing point (°C / °F)
    [
        "franklin",
        "faraday",
        "thales",
        "gilbert",
        "volta",
        "no single",
        "not a single",
        "not discovered by",
    ],  # electricity
    ["chameleon", "octopus", "cuttlefish", "squid", "flounder"],  # colour-changing animal
    ["eight", "8"],  # spider legs
]

_PROMPT_CACHE = {}


def _stop_token_ids(tokenizer):
    """Harmony end-of-generation tokens: <|return|> (eos), <|call|>, <|endoftext|>. NOT <|end|>, which only closes
    the analysis channel before the final answer."""
    ids = {tokenizer.eos_token_id}
    for tok in ("<|return|>", "<|call|>", "<|endoftext|>"):
        tid = tokenizer.convert_tokens_to_ids(tok)
        if isinstance(tid, int) and tid >= 0 and tid != getattr(tokenizer, "unk_token_id", -1):
            ids.add(tid)
    return ids


def _channel_token_id(tokenizer):
    ids = tokenizer.convert_tokens_to_ids(["<|channel|>"])
    return int(ids[0]) if ids and ids[0] is not None and ids[0] != tokenizer.unk_token_id else 200005


def _truncate_at_stop(tokens, stop_ids):
    """Tokens up to (excluding) the first end-of-generation token; the sweep keeps decoding past it for timing."""
    for i, t in enumerate(tokens):
        if t in stop_ids:
            return tokens[:i]
    return tokens


def _keyword_present(text, keyword):
    """Whole-token match so that '7' does not match '49' or '2017' and '0' does not match '100'."""
    return re.search(r"(?<![\w.])" + re.escape(keyword) + r"(?![\w])", text) is not None


def _prompts_for(isl, batch):
    """Batch prompts for a nominal input length (the 128 case are distinct QA prompts, the rest one long prompt)."""
    key = (isl, batch)
    if key not in _PROMPT_CACHE:
        prompts, _ = load_inputs(PROMPT_FILES[isl], batch, instruct=False)
        _PROMPT_CACHE[key] = list(prompts)[:batch]
    return _PROMPT_CACHE[key]


# Debug knob: GPT_OSS_REGRESSION_DECODE_TRACE=0 runs decode eagerly (slower, but isolates trace-replay issues).
DECODE_TRACE = os.getenv("GPT_OSS_REGRESSION_DECODE_TRACE", "1") not in ("0", "false", "False")


def _selected_pairs():
    env = os.getenv("GPT_OSS_REGRESSION_PAIRS")
    if not env:
        return ISL_OSL_PAIRS
    return [tuple(int(x) for x in p.split(":")) for p in env.split(",") if p.strip()]


def _clear_kv_caches(models):
    for m in models:
        for layer in m.layers:
            k_cache, v_cache = layer.self_attn.layer_past
            ttnn.mul(k_cache, 0, output_tensor=k_cache)
            ttnn.mul(v_cache, 0, output_tensor=v_cache)


def _degenerate(tokens, max_run=48, min_unique_ratio=0.15):
    """Heuristic garbage detector: long runs of one token or almost no token diversity."""
    if len(tokens) < 32:
        return False
    run = best = 1
    for a, b in zip(tokens, tokens[1:]):
        run = run + 1 if a == b else 1
        best = max(best, run)
    unique_ratio = len(set(tokens)) / len(tokens)
    return best > max_run or unique_ratio < min_unique_ratio


def _percentile(values, pct):
    if not values:
        return float("nan")
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(pct / 100 * (len(ordered) - 1)))))
    return ordered[idx]


def _git_rev():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:  # pragma: no cover - best effort metadata
        return "unknown"


# Thermal guard. On this P150x8 box the hottest board reaches ~88 C during sustained prefill and throttles its AI
# clock to 800 MHz; a traced decode step launched while a device is deep-throttled has deadlocked inside paged SDPA
# decode (watcher: writer stuck in a NOC write barrier, other devices waiting in the next CCL). Set
# GPT_OSS_REGRESSION_COOLDOWN_C=<max asic temperature> to wait (up to COOLDOWN_TIMEOUT_S) for all boards to cool
# below it before each prefill and before the first decode step of a pair.
COOLDOWN_C = float(os.getenv("GPT_OSS_REGRESSION_COOLDOWN_C", "0"))
COOLDOWN_TIMEOUT_S = float(os.getenv("GPT_OSS_REGRESSION_COOLDOWN_TIMEOUT_S", "180"))


def _board_telemetry():
    """Returns (max_temp_C, min_aiclk_MHz) across boards via tt-smi, or (None, None) if unavailable."""
    try:
        out = subprocess.run(["tt-smi", "-s", "--snapshot_no_tty"], capture_output=True, text=True, timeout=60).stdout
        data = json.loads(out[out.find("{") :])
        temps, clocks = [], []
        for dev in data.get("device_info", []):
            tel = dev.get("telemetry", {})
            temps.append(float(str(tel.get("asic_temperature", "nan")).strip()))
            clocks.append(int(str(tel.get("aiclk", "0")).strip()))
        return (max(temps) if temps else None, min(clocks) if clocks else None)
    except Exception as e:  # pragma: no cover - telemetry is best effort
        logger.warning(f"tt-smi telemetry unavailable: {e}")
        return None, None


def _wait_for_cooldown(stage):
    """Blocks until every board is below COOLDOWN_C (or the timeout passes). Returns the last max temperature."""
    if COOLDOWN_C <= 0:
        return None
    t0 = time.perf_counter()
    max_t, _clk = _board_telemetry()
    if max_t is not None and max_t > COOLDOWN_C:
        logger.info(f"   cooldown before {stage}: waiting, max board temperature {max_t} C > {COOLDOWN_C} C")
    while max_t is not None and max_t > COOLDOWN_C and time.perf_counter() - t0 < COOLDOWN_TIMEOUT_S:
        time.sleep(5)
        max_t, _clk = _board_telemetry()
    waited = time.perf_counter() - t0
    if waited > 6:
        logger.info(f"   cooldown before {stage}: waited {waited:.0f}s, max board temperature now {max_t} C")
    return max_t


def _prefill(generator, models, tt_kv_cache, page_table, input_tokens, decoding_pos, enable_trace=True):
    """One timed prefill of all users on a cleared KV cache. warmup_prefill=False is the GPT-OSS path (as in
    text_demo): it skips the generic batch-1 warm-up sweep and hoists the decode-trace allocations ahead of
    the first trace capture (see Generator.prefill_forward_text and tt-metal #52176)."""
    _clear_kv_caches(models)
    generator.prev_page_table = None
    ttnn.synchronize_device(models[0].mesh_device)  # keep the (async) cache clear out of the prefill timing
    t0 = time.perf_counter()
    logits = generator.prefill_forward_text(
        input_tokens,
        page_table=page_table,
        kv_cache=tt_kv_cache,
        prompt_lens=decoding_pos,
        enable_trace=enable_trace,
        warmup_prefill=False,
    )
    ttnn.synchronize_device(models[0].mesh_device)
    return logits, time.perf_counter() - t0


def _precompile_prefill_lengths(generator, models, model_args, tt_kv_cache, page_table, tokenizer, isls, max_seq_len):
    """Compile every prefill length (one user, eager, no trace) BEFORE any trace is captured.

    Programs and persistent tensors (e.g. the cached RoPE slices) created while a trace is live may be placed
    in a trace's freed intermediate address range and be overwritten by a later replay -- tt-metal's
    TT_METAL_TRACE_ALLOC_TRACKING flags exactly this, and the symptom is a garbage prefill or a device hang
    some pairs later. tt_transformers avoids it with warmup_model_prefill, which GPT-OSS disables (#52176),
    so the sweep does its own eager warm-up up front. Returns {padded_len: seconds}."""
    compile_times = {}
    for isl in sorted(isls):
        prompts = _prompts_for(isl, 1)
        input_tokens, _e, decoding_pos, _l = preprocess_inputs_prefill(
            prompts, tokenizer, model_args, instruct=False, max_generated_tokens=1, max_prefill_len=max_seq_len
        )
        input_tokens = torch.stack(input_tokens).view(1, -1)
        padded_len = get_padded_prefill_len(int(max(decoding_pos)))
        if padded_len in compile_times:
            continue
        _, compile_times[padded_len] = _prefill(
            generator, models, tt_kv_cache, page_table[:1], input_tokens, decoding_pos[:1], enable_trace=False
        )
        logger.info(f"   pre-compiled prefill length {padded_len} in {compile_times[padded_len]:.1f}s (eager, 1 user)")
    return compile_times


def _run_pair(
    generator,
    models,
    model_args,
    tt_kv_cache,
    page_table,
    tokenizer,
    prompts,
    isl,
    osl,
    sampling,
    max_seq_len,
    warmed_lengths,
    compile_times=None,
):
    """Prefill all users then decode exactly `osl` tokens (no EOS stop). Returns (metrics dict, token lists).

    The first time a padded prefill length is seen in a batch case, an untimed pass absorbs program
    compilation (and, for the very first pair, the decode-trace and prefill-trace capture) so that the
    reported prefill time is the steady-state number the demo also reports."""
    batch = len(prompts)
    input_tokens, _encoded, decoding_pos, prefill_lens = preprocess_inputs_prefill(
        prompts, tokenizer, model_args, instruct=False, max_generated_tokens=osl, max_prefill_len=max_seq_len
    )
    input_tokens = torch.stack(input_tokens).view(batch, -1)
    encoded_len = int(max(decoding_pos))
    padded_len = get_padded_prefill_len(encoded_len)

    compile_time = compile_times.get(padded_len) if compile_times else None
    if padded_len not in warmed_lengths:
        # Program compilation is per shape, so one user is enough to warm a new padded length. The very first
        # pair still warms the whole batch: that call also prepares the decode trace, whose persistent page
        # table must have the full batch shape.
        warm_users = batch if not warmed_lengths else 1
        _, compile_time = _prefill(
            generator,
            models,
            tt_kv_cache,
            page_table[:warm_users],
            input_tokens[:warm_users],
            decoding_pos[:warm_users],
        )
        warmed_lengths.add(padded_len)
    temp_before_prefill = _wait_for_cooldown("prefill")
    logits, prefill_time = _prefill(generator, models, tt_kv_cache, page_table, input_tokens, decoding_pos)
    temp_after_prefill, clk_after_prefill = _board_telemetry() if COOLDOWN_C > 0 else (None, None)
    temp_before_decode = _wait_for_cooldown("decode")

    out_tok = torch.argmax(logits, dim=-1)  # [B, 1]
    outputs = [[int(out_tok[b])] for b in range(batch)]
    current_pos = torch.tensor(decoding_pos)

    step_times = []
    for _ in range(osl - 1):
        t1 = time.perf_counter()
        out_tok, _ = generator.decode_forward(
            out_tok,
            current_pos,
            enable_trace=DECODE_TRACE,
            page_table=page_table,
            kv_cache=tt_kv_cache,
            sampling_params=sampling,
        )
        step_times.append(time.perf_counter() - t1)
        if len(step_times) == 1:
            logger.info(f"   first decode step (pos {int(current_pos.max())}) done in {1000 * step_times[0]:.1f} ms")
        current_pos += 1
        for b in range(batch):
            outputs[b].append(int(out_tok[b]))

    steady = step_times[1:] if len(step_times) > 1 else step_times
    mean_step = statistics.fmean(steady) if steady else float("nan")
    metrics = {
        "batch": batch,
        "isl_nominal": isl,
        "isl_encoded": encoded_len,
        "isl_padded": padded_len,
        "osl": osl,
        "prefill_compile_s": round(compile_time, 2) if compile_time is not None else None,
        "prefill_total_s": round(prefill_time, 4),
        "prefill_per_user_ms": round(1000 * prefill_time / batch, 1),
        # TTFT: prefill is sequential per user on a single-row mesh and the first token is the argmax of each user's
        # prefill logits, so user k (1-based) sees k * per-user prefill: first user, mean over users, last user.
        "ttft_first_user_ms": round(1000 * prefill_time / batch, 1),
        "ttft_mean_user_ms": round(1000 * prefill_time / batch * (batch + 1) / 2, 1),
        "ttft_last_user_ms": round(1000 * prefill_time, 1),
        "first_decode_step_ms": round(1000 * step_times[0], 2) if step_times else None,
        "decode_step_mean_ms": round(1000 * mean_step, 2),
        "decode_step_p50_ms": round(1000 * _percentile(steady, 50), 2),
        "decode_step_p99_ms": round(1000 * _percentile(steady, 99), 2),
        "tok_s_user": round(1 / mean_step, 2) if steady else None,
        "tok_s_aggregate": round(batch / mean_step, 1) if steady else None,
        "e2e_s": round(prefill_time + sum(step_times), 2),
        "board_temp_before_prefill_c": temp_before_prefill,
        "board_temp_after_prefill_c": temp_after_prefill,
        "min_aiclk_after_prefill_mhz": clk_after_prefill,
        "board_temp_before_decode_c": temp_before_decode,
    }
    return metrics, outputs


def _check_outputs(tokenizer, vocab_size, isl, prompts, outputs):
    """Correctness gates; returns (list of failures, info dict)."""
    failures, info = [], {}
    batch = len(outputs)
    for b, toks in enumerate(outputs):
        bad = [t for t in toks if not (0 <= t < vocab_size)]
        if bad:
            failures.append(f"user {b}: {len(bad)} token ids outside the vocabulary (e.g. {bad[:3]})")
    # Quality gates look at the generation up to the first end-of-generation token: the sweep keeps
    # decoding past it for timing, and whatever follows is outside the model's training distribution.
    stop_ids = _stop_token_ids(tokenizer)
    generated = [_truncate_at_stop([t for t in toks if 0 <= t < vocab_size], stop_ids) for toks in outputs]
    texts = [tokenizer.decode(toks) for toks in generated]
    info["finished_users"] = sum(1 for g, toks in zip(generated, outputs) if len(g) < len(toks))
    # Every GPT-OSS reply opens the harmony analysis/final channel, so the first generated token (argmax of
    # the prefill logits) must be <|channel|>. Garbage prefill logits fail this gate even when the decoded
    # garbage is diverse enough to slip past the run-length / diversity heuristic below.
    channel_id = _channel_token_id(tokenizer)
    bad_start = [b for b, toks in enumerate(outputs) if not toks or toks[0] != channel_id]
    info["users_not_starting_with_channel"] = bad_start
    if bad_start:
        failures.append(
            f"{len(bad_start)}/{batch} users did not start with <|channel|> (prefill logits wrong): "
            + "; ".join(f"user {b}: {texts[b][:80]!r}" for b in bad_start[:4])
        )
    degenerate = [b for b, toks in enumerate(generated) if _degenerate(toks)]
    info["degenerate_users"] = degenerate
    if len(degenerate) > max(1, batch // 4):
        failures.append(f"{len(degenerate)}/{batch} users produced degenerate output (users {degenerate[:8]})")

    if isl == 128:
        hits = 0
        misses = []
        for b, text in enumerate(texts):
            keywords = QA_KEYWORDS[b % len(QA_KEYWORDS)]
            if any(_keyword_present(text.lower(), k) for k in keywords):
                hits += 1
            else:
                misses.append(b)
        info["qa_accuracy"] = round(hits / batch, 3)
        info["qa_misses"] = misses
        if hits / batch < 0.75:
            failures.append(
                f"QA keyword accuracy {hits}/{batch} below 75%; misses: "
                + "; ".join(f"user {b}: {texts[b][:120]!r}" for b in misses[:4])
            )
    else:
        # Same prompt for every user: how many users agree with user 0 over the first 16 tokens.
        head = 16
        agree = sum(1 for toks in generated if toks[:head] == generated[0][:head])
        info["users_agreeing_with_user0_first16"] = f"{agree}/{batch}"
    info["sample_output"] = texts[0][:160]
    # Keep enough for a post-mortem: user 0's whole generation (stop-truncated) plus every flagged user's.
    flagged = sorted(set([0] + degenerate + bad_start))
    info["full_outputs"] = {str(b): tokenizer.decode(outputs[b]) for b in flagged if b < batch}
    return failures, info


def _markdown_table(rows):
    cols = [
        ("batch", "B"),
        ("isl_nominal", "ISL"),
        ("isl_padded", "ISL pad"),
        ("osl", "OSL"),
        ("prefill_compile_s", "compile s"),
        ("prefill_total_s", "prefill s"),
        ("prefill_per_user_ms", "prefill/user ms"),
        ("decode_step_mean_ms", "step ms"),
        ("decode_step_p99_ms", "p99 ms"),
        ("tok_s_user", "tok/s/user"),
        ("tok_s_aggregate", "tok/s agg"),
        ("qa_accuracy", "QA acc"),
        ("status", "status"),
    ]
    out = ["| " + " | ".join(h for _, h in cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(str(r.get(k, "")) for k, _ in cols) + " |")
    return "\n".join(out)


@pytest.mark.timeout(7200)
@pytest.mark.parametrize("batch_size", BATCH_SIZES, ids=[f"batch{b}" for b in BATCH_SIZES])
@parametrize_mesh_with_fabric([(1, 8)])
def test_multi_user_regression(mesh_device, device_params, batch_size, state_dict):
    mesh_shape = tuple(mesh_device.shape)
    if mesh_shape[0] != 1 or mesh_shape[1] < 8:
        pytest.skip(f"multi-user single-row sweep targets 1x8 meshes, got {mesh_shape}")

    setup = TestFactory.setup_test(mesh_device, use_real_weights=False)
    # Per-user context budget, power of two, at most 64K (mirrors the server's context-capped concurrency).
    max_seq_len = min(MAX_CONTEXT_PER_USER, TOTAL_KV_TOKENS // batch_size)
    if os.getenv("GPT_OSS_REGRESSION_POW2_CONTEXT", "1") not in ("0", "false", "False"):
        max_seq_len = 1 << (max_seq_len.bit_length() - 1)  # power of two (default)
    else:
        max_seq_len = max(1024, (max_seq_len // 1024) * 1024)  # multiple of 1024: fits more of the KV budget
    page_params = {
        "page_block_size": BLOCK_SIZE,
        "page_max_num_blocks_per_dp": batch_size * (max_seq_len // BLOCK_SIZE),
    }
    pairs = [(i, o) for i, o in _selected_pairs() if i + o <= max_seq_len and i in PROMPT_FILES]
    skipped = [(i, o) for i, o in _selected_pairs() if (i, o) not in pairs]
    logger.info(f"batch {batch_size}: context {max_seq_len} tokens/user, pairs {pairs}, skipped {skipped}")

    # The demo's page table is an unseeded random block permutation; seed it so a layout-dependent failure
    # reproduces from run to run (the seed is recorded in the results).
    page_table_seed = int(os.getenv("GPT_OSS_REGRESSION_PAGE_TABLE_SEED", "1234"))
    torch.manual_seed(page_table_seed)
    model_args, models, page_table, tt_kv_cache, tokenizer, _processor, _cfg = prepare_gpt_oss_generator_args(
        num_devices=mesh_device.get_num_devices(),
        data_parallel=1,
        mesh_device=mesh_device,
        global_batch_size=batch_size,
        optimizations=None,
        max_seq_len=max_seq_len,
        page_params=page_params,
        paged_attention=True,
        mesh_config=setup["mesh_config"],
        state_dict=state_dict,
        users_row_sharded=False,
    )
    # Debug knob: GPT_OSS_REGRESSION_DECODE_K_CHUNK=<tokens> overrides the paged-SDPA decode K chunk size (default 128).
    k_chunk = os.getenv("GPT_OSS_REGRESSION_DECODE_K_CHUNK")
    if k_chunk:
        for m in models:
            for layer in m.layers:
                object.__setattr__(layer.self_attn.program_config, "decode_k_chunk_size", int(k_chunk))
        logger.info(f"decode_k_chunk_size overridden to {k_chunk}")
    generator = Generator(models, model_args, mesh_device, processor=None, tokenizer=tokenizer)
    assert all(getattr(m, "sampling", None) is not None for m in models), "on-device sampling expected on 1x8"
    sampling = SamplingParams(
        temperature=[0.0] * batch_size,
        top_k=[1] * batch_size,
        top_p=[1.0] * batch_size,
        enable_log_probs=[False] * batch_size,
        num_logprobs=[0] * batch_size,
    )
    vocab_size = model_args[0].vocab_size
    model_name = model_args[0].model_name

    out_dir = Path("generated/gpt_oss_multi_user_regression")
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = os.getenv("GPT_OSS_REGRESSION_TAG", "")  # e.g. "_baseline" to keep control runs in their own file
    out_file = out_dir / f"{model_name}_{mesh_shape[0]}x{mesh_shape[1]}{tag}.jsonl"
    meta = {
        "model": model_name,
        "mesh": f"{mesh_shape[0]}x{mesh_shape[1]}",
        "git": _git_rev(),
        "time": datetime.now().isoformat(),
        "page_table_seed": page_table_seed,
        "context_per_user": max_seq_len,
        "decode_k_chunk": int(k_chunk) if k_chunk else 128,
        "cooldown_c": COOLDOWN_C or None,
    }

    # Eager compile of every length first (see _precompile_prefill_lengths); the traced 128-token prefill and the
    # decode trace are then captured by the first pair with all programs already resident.
    compile_times = _precompile_prefill_lengths(
        generator, models, model_args, tt_kv_cache, page_table, tokenizer, {i for i, _ in pairs}, max_seq_len
    )
    rows, all_failures, warmed_lengths = [], [], set()
    for isl, osl in pairs:
        prompts = _prompts_for(isl, batch_size)
        logger.info(f"== batch {batch_size} ISL {isl} OSL {osl}")
        metrics, outputs = _run_pair(
            generator,
            models,
            model_args,
            tt_kv_cache,
            page_table,
            tokenizer,
            prompts,
            isl,
            osl,
            sampling,
            max_seq_len,
            warmed_lengths,
            compile_times,
        )
        failures, info = _check_outputs(tokenizer, vocab_size, isl, prompts, outputs)
        row = {**meta, **metrics, **info, "status": "FAIL" if failures else "ok"}
        rows.append(row)
        with open(out_file, "a") as f:
            f.write(json.dumps(row) + "\n")
        logger.info(
            f"   prefill {metrics['prefill_total_s']:.2f}s ({metrics['prefill_per_user_ms']:.0f} ms/user), "
            f"decode {metrics['decode_step_mean_ms']:.1f} ms/step (p99 {metrics['decode_step_p99_ms']:.1f}), "
            f"{metrics['tok_s_user']} tok/s/user, {metrics['tok_s_aggregate']} tok/s aggregate; "
            f"{'; '.join(failures) if failures else 'checks ok'} | {info.get('sample_output', '')[:100]!r}"
        )
        all_failures.extend(f"ISL {isl} OSL {osl}: {msg}" for msg in failures)

    logger.info(
        f"\nGPT-OSS multi-user regression, {model_name} on {meta['mesh']} ({meta['git']}), batch {batch_size}:\n"
        + _markdown_table(rows)
    )
    assert not all_failures, "Regression failures:\n" + "\n".join(all_failures)
