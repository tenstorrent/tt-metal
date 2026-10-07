# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4.1-Flash text demo on the 4x8 Blackhole galaxy: paged batched PREFILL -> traced closed-loop DECODE (same structure as
``models/demos/gpt_oss/demo/text_demo.py``): pytest-parametrized scenarios over (input prompts, batch_size, max_seq_len, max_generated_tokens, page_params,
sampling_params, enable_decode_trace, ...), real HF tokenizer + the checkpoint's chat template, ``preprocess_inputs_prefill``, ``Generator.prefill_forward_text``
/ ``decode_forward``. Prints per-user text, TTFT, prefill tok/s, decode ms/token and tok/s/user.

    pytest models/demos/blackhole/deepseek_v41_flash/demo/text_demo.py -k prefill_128_b16

Any batch size 1..128 is accepted: it is padded to a multiple of the 4 mesh rows (users per row = ceil(batch / 4)); padding users replay the last real prompt
and are not printed. Any ISL: the prompt is processed in chunks (``DSV41_PREFILL_ROW_TOKENS`` tokens per mesh row per pass). Env: DSV41_LAYERS (e.g. 0-3 for a
few-layer smoke run), DSV41_ENGRAM_RAM=0 (do not hold the 2 x 95 GB Engram tables in host RAM; slower decode host time).
Supported (ISL, batch) combinations and limits: see the ``Supported`` table in the module docstring of tt/dsv41_model.py / the final report.
"""

import faulthandler
import json
import os
import signal
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.sampling import SamplingParams
from models.demos.blackhole.deepseek_v41_flash.tt.common import create_tt_model, default_page_params
from models.demos.blackhole.deepseek_v41_flash.tt.generator import Generator
from models.demos.utils.llm_demo_utils import create_benchmark_data  # noqa: F401  (kept for CI perf reporting parity)
from models.perf.benchmarking_utils import BenchmarkProfiler
from models.tt_transformers.demo.simple_text_demo import load_inputs
from models.tt_transformers.tt.common import PagedAttentionConfig, preprocess_inputs_prefill

faulthandler.register(
    signal.SIGUSR1, all_threads=True
)  # hangwatch.sh sends USR1 before killing a stalled job: python stacks into the log

PROMPTS = "models/demos/blackhole/deepseek_v41_flash/demo/sample_prompts"
LONG = "models/tt_transformers/demo/sample_prompts"
GREEDY = {"temperature": 0, "top_p": 0.08}
Q128 = f"{PROMPTS}/input_data_questions_prefill_128.json"
GSM = f"{PROMPTS}/input_data_gsm8k_128.json"
STRUCT = f"{PROMPTS}/input_data_struct_16.json"
SKIP_BIG = "needs the sharded-KV / two-level top-k pieces of the capacity design doc and prefill indexer; not executed (see report)"


def _s(
    prompts,
    batch,
    max_seq_len,
    max_gen,
    id,
    warmup=True,
    instruct=False,
    stop_at_eos=False,
    skip=None,
    repeat=1,
    prefill_trace=True,
    chunk=None,
):
    """(input_prompts, batch_size, repeat_batches, max_seq_len, max_generated_tokens, page_params, sampling_params, enable_decode_trace, enable_prefill_trace, prefill_chunk, warmup_prefill, instruct, stop_at_eos)
    prefill_chunk: tokens per user per prefill chunk (multiple of 128; None = auto from DSV41_PREFILL_ROW_TOKENS); the chunk is traced once and replayed.
    """
    marks = [pytest.mark.skip(reason=skip)] if skip else []
    return pytest.param(
        prompts,
        batch,
        repeat,
        max_seq_len,
        max_gen,
        None,
        GREEDY,
        True,
        prefill_trace,
        chunk,
        warmup,
        instruct,
        stop_at_eos,
        id=id,
        marks=marks,
    )


SCENARIOS = [
    _s(Q128, 16, 512, 64, "prefill_128_b16"),
    _s(Q128, 8, 512, 64, "prefill_128_b8"),
    _s(Q128, 64, 512, 64, "prefill_128_b64"),
    _s(Q128, 1, 512, 64, "prefill_128_b1"),
    _s(Q128, 4, 512, 64, "prefill_128_b4"),
    _s(Q128, 32, 512, 64, "prefill_128_b32"),
    _s(Q128, 128, 512, 32, "prefill_128_b128"),
    _s(GSM, 16, 512, 384, "gsm8k_b16", instruct=True, stop_at_eos=True),
    _s(GSM, 64, 512, 384, "gsm8k_b64", instruct=True, stop_at_eos=True),
    *[
        _s(f"{PROMPTS}/input_data_gsm8k_o{o}.json", 64, 512, 352, f"gsm8k_b64_o{o}", instruct=True, stop_at_eos=True)
        for o in range(0, 512, 64)
    ],
    _s(["What is the capital of France?"], 16, 512, 64, "same_prompt_b16", instruct=True),
    _s(f"{LONG}/input_data_long_2k.json", 16, 4096, 64, "isl2k_b16"),
    _s(f"{LONG}/input_data_long_4k.json", 1, 8192, 64, "isl4k_b1"),
    _s(f"{LONG}/input_data_long_4k.json", 16, 8192, 64, "isl4k_b16"),
    _s(f"{LONG}/input_data_long_4k.json", 32, 8192, 64, "isl4k_b32"),
    _s(f"{LONG}/input_data_long_8k.json", 1, 16384, 64, "isl8k_b1"),
    _s(f"{LONG}/input_data_long_8k.json", 16, 16384, 64, "isl8k_b16"),
    _s(f"{LONG}/input_data_long_16k.json", 1, 32768, 64, "isl16k_b1"),
    _s(f"{LONG}/input_data_long_16k.json", 16, 32768, 64, "isl16k_b16"),
    _s(f"{LONG}/input_data_long_32k.json", 1, 65536, 64, "isl32k_b1"),
    _s(f"{LONG}/input_data_long_32k.json", 4, 65536, 64, "isl32k_b4"),
    _s(f"{LONG}/input_data_long_64k.json", 1, 70000, 64, "isl64k_b1"),
    _s(f"{LONG}/input_data_long_64k.json", 4, 70000, 64, "isl64k_b4"),
    _s(GSM, 32, 512, 384, "gsm8k_b32", instruct=True, stop_at_eos=True),
    *[
        _s(STRUCT, b, 512, 256, f"struct_b{b}", instruct=True, stop_at_eos=True) for b in (8, 16, 32)
    ],  # high-acceptance workload (JSON / CSV / code / repeated text), spec adaptive-k studies
    _s(f"{LONG}/input_data_long_2k.json", 32, 4096, 64, "isl2k_b32_ragged"),
    _s(f"{LONG}/input_data_long_2k.json", 16, 4096, 64, "isl2k_b16_ragged"),
    _s(f"{LONG}/input_data_long_2k.json", 16, 4096, 64, "isl2k_b16_ragged_u4"),
    _s(f"{LONG}/input_data_long_4k.json", 64, 8192, 64, "isl4k_b64"),
    _s(f"{LONG}/input_data_long_8k.json", 32, 16384, 64, "isl8k_b32"),
    _s(f"{LONG}/input_data_long_8k.json", 64, 16384, 64, "isl8k_b64"),
    _s(f"{LONG}/input_data_long_16k.json", 32, 32768, 64, "isl16k_b32"),
    _s(f"{LONG}/input_data_long_16k.json", 64, 32768, 64, "isl16k_b64"),
    _s(f"{LONG}/input_data_long_64k.json", 16, 70000, 64, "isl64k_b16"),
    _s(f"{LONG}/input_data_long_64k.json", 32, 70000, 64, "isl64k_b32"),
    _s(f"{LONG}/input_data_long_64k.json", 64, 70000, 64, "isl64k_b64"),
    _s(f"{LONG}/input_data_long_2k.json", 4, 4096, 64, "isl2k_b4"),
    _s(f"{LONG}/input_data_long_4k.json", 4, 8192, 64, "isl4k_b4"),
    _s(f"{LONG}/input_data_long_8k.json", 4, 16384, 64, "isl8k_b4"),
    _s(f"{LONG}/input_data_long_16k.json", 4, 32768, 64, "isl16k_b4"),
    _s(f"{LONG}/input_data_long_4k.json", 128, 8192, 32, "isl4k_b128"),
    _s(f"{LONG}/input_data_long_8k.json", 128, 16384, 32, "isl8k_b128"),
    _s(f"{LONG}/input_data_long_16k.json", 128, 18432, 32, "isl16k_b128"),
    _s(f"{LONG}/input_data_long_64k.json", 128, 70000, 64, "isl64k_b128"),
    _s(f"{LONG}/input_data_long_32k.json", 16, 40000, 64, "isl32k_b16"),
    _s(GSM, 4, 512, 384, "gsm8k_b4", instruct=True, stop_at_eos=True),
    _s(GSM, 8, 512, 384, "gsm8k_b8", instruct=True, stop_at_eos=True),
    _s(f"{LONG}/input_data_long_4k.json", 8, 8192, 64, "isl4k_b8"),
    _s(f"{LONG}/input_data_long_8k.json", 8, 16384, 64, "isl8k_b8"),
    _s(f"{LONG}/input_data_long_32k.json", 8, 40000, 64, "isl32k_b8"),
    _s(f"{LONG}/input_data_long_64k.json", 8, 70000, 64, "isl64k_b8"),
    _s(f"{PROMPTS}/input_data_long_128k_exact.json", 8, 135000, 64, "isl128k_b8"),
    _s(f"{PROMPTS}/input_data_long_256k_exact.json", 8, 270000, 64, "isl256k_b8"),
    _s(f"{PROMPTS}/input_data_long_128k_exact.json", 16, 135000, 64, "isl128k_b16"),
    _s(f"{PROMPTS}/input_data_long_256k_exact.json", 16, 270000, 64, "isl256k_b16"),
    _s(GSM, 32, 512, 384, "gsm8k_b32", instruct=True, stop_at_eos=True),
    _s(f"{LONG}/input_data_long_32k.json", 32, 40000, 64, "isl32k_b32"),
    _s(f"{PROMPTS}/input_data_long_128k_exact.json", 32, 135000, 64, "isl128k_b32"),
    _s(f"{PROMPTS}/input_data_long_256k_exact.json", 32, 270000, 64, "isl256k_b32"),
    _s(f"{LONG}/input_data_long_32k.json", 64, 40000, 64, "isl32k_b64"),
    _s(f"{PROMPTS}/input_data_long_128k_exact.json", 64, 135000, 64, "isl128k_b64"),
    _s(f"{PROMPTS}/input_data_long_256k_exact.json", 64, 270000, 64, "isl256k_b64"),
    _s(GSM, 128, 512, 384, "gsm8k_b128", instruct=True, stop_at_eos=True),
    _s(f"{LONG}/input_data_long_32k.json", 128, 40000, 64, "isl32k_b128"),
    _s(f"{PROMPTS}/input_data_long_128k_exact.json", 128, 135000, 64, "isl128k_b128"),
    _s(f"{PROMPTS}/input_data_long_256k_exact.json", 128, 270000, 64, "isl256k_b128"),
    _s(f"{LONG}/input_data_long_128k.json", 1, 135000, 64, "isl128k_b1"),
    _s(f"{LONG}/input_data_long_128k.json", 4, 135000, 64, "isl128k_b4"),
    _s(f"{LONG}/input_data_long_256k.json", 4, 270000, 64, "isl256k_b4"),
    _s(f"{LONG}/input_data_long_256k.json", 1, 270000, 64, "isl256k_b1"),
    _s(
        f"{PROMPTS}/input_data_long_1M.json",
        1,
        1050000,
        64,
        "isl1M_b1",
        skip="1M: " + SKIP_BIG + " (never executed by instruction)",
    ),
]


def decode_time_sum(profiler, n_dec, batch_idx):
    """Seconds of the steady decode steps (1..n_dec-1) of one repeat batch."""
    return sum(profiler.get_duration(f"inference_decode_time_{i}", iteration=batch_idx) for i in range(1, n_dec))


@torch.no_grad()
def _run_demo(
    mesh_device,
    input_prompts,
    batch_size,
    repeat_batches,
    max_seq_len,
    max_generated_tokens,
    page_params,
    sampling_params,
    enable_decode_trace,
    enable_prefill_trace,
    prefill_chunk,
    warmup_prefill,
    instruct,
    stop_at_eos,
    cache=None,
    build_max_seq_len=None,
):
    max_generated_tokens = int(
        os.environ.get("DSV41_MAX_GEN", max_generated_tokens)
    )  # more rounds per policy window for the spec studies
    mesh_rows = mesh_device.shape[0]
    assert 1 <= batch_size <= 128, "batch_size must be 1..128 (mHC kernels: at most 32 users per mesh row)"
    users_per_row = -(-batch_size // mesh_rows)
    padded_batch = users_per_row * mesh_rows
    a, _, b = os.environ.get("DSV41_LAYERS", "0-39").partition("-")
    layer_ids = list(range(int(a), int(b or a) + 1))
    greedy = sampling_params["temperature"] == 0
    assert greedy, "DSV4.1 decode samples greedily on the device; only temperature 0 is implemented"

    profiler = BenchmarkProfiler()
    profiler.start("run")
    if page_params is None:
        page_params = default_page_params(max_seq_len, users_per_row)
    paged_attention_config = PagedAttentionConfig(
        block_size=page_params["page_block_size"], max_num_blocks=page_params["page_max_num_blocks_per_dp"]
    )

    # early support check (cheap, before the 15+ min model build): context beyond the dense limit needs the indexer path (see tt/dsv41_model.py)
    from models.demos.blackhole.deepseek_v41_flash.tt.model_args import DSV41ModelArgs, dense_context_limit

    limit = dense_context_limit(layer_ids)
    if isinstance(input_prompts, list) and len(input_prompts) == 1:
        probe = input_prompts
    else:
        probe, _ = load_inputs(input_prompts, batch_size, instruct=instruct)
    probe_args = DSV41ModelArgs(mesh_device, padded_batch, max_seq_len, layer_ids)
    probe_len = max(len(probe_args.encode_prompt(p, instruct=instruct)) for p in probe[:batch_size])
    ctx = min(probe_len, max_seq_len - max_generated_tokens) + max_generated_tokens
    if ctx > limit:
        logger.info(
            f"context {ctx} > {limit} compressed-dense limit: decode indexer + sparse prefill are enabled for this model"
        )

    profiler.start("generator_setup")
    key = (padded_batch, tuple(layer_ids))
    reconfig_ok = os.environ.get("DSV41_SESSION_RECONFIG", "1") == "1"
    if cache is not None and key in cache and (max_seq_len <= cache[key][1].max_ctx or not reconfig_ok):
        model_args, model, generator = cache[key]
        assert max_seq_len <= model.max_ctx, "cached model was built for a shorter max_seq_len"
        model_args.max_seq_len = max_seq_len
    else:  # (no cached model, another batch size, or a longer context than the cached model was built for)
        build_len = build_max_seq_len or max_seq_len
        if build_len != max_seq_len:
            page_params = default_page_params(build_len, users_per_row)
            paged_attention_config = PagedAttentionConfig(
                block_size=page_params["page_block_size"], max_num_blocks=page_params["page_max_num_blocks_per_dp"]
            )
        other = [k for k in cache if k[1] == tuple(layer_ids)] if cache is not None and reconfig_ok else []
        if other:
            # a session scenario with ANOTHER batch size (or a longer context): keep the weights of the model that is already built, release + rebuild only the batch dependent
            # state (Model.reconfigure; a second full build in the process would run out of DRAM)
            _, model, generator = cache.pop(other[0])
            model_args = generator.reconfigure(
                padded_batch,
                build_len,
                paged_attention_config,
                layer_ids=layer_ids,
                log=lambda m: logger.info(m),
            )
            cache[key] = (model_args, model, generator)
        else:
            model_args, model, tt_kv_cache, _ = create_tt_model(
                mesh_device,
                padded_batch,
                build_len,
                paged_attention_config,
                layer_ids=layer_ids,
                log=lambda m: logger.info(m),
            )
            generator = Generator([model], [model_args], mesh_device, tokenizer=model_args.tokenizer)
            if cache is not None:
                cache[key] = (model_args, model, generator)
    tokenizer = model_args.tokenizer
    spec_k = int(
        os.environ.get("DSV41_SPEC", "0")
    )  # speculative decoding: k drafts per round (DSpark drafter), 0 = off
    if spec_k and batch_size >= 128 and os.environ.get("DSV41_SPEC_B128") != "1":
        # measured at 40 layers (GSM8K, B=128): k=1 0.69x, k=3 0.85x of plain decode (chunked 64/128-row verify rounds of 243/400 ms): spec decode is OFF by default for B >= 128
        logger.info(f"DSV41_SPEC={spec_k} ignored at batch {batch_size}: plain decode (DSV41_SPEC_B128=1 forces the spec runner)")
        spec_k = 0
    if spec_k and os.environ.get("DSV41_SPEC_EARLY") == "1" and not hasattr(generator, "spec"):
        # build the spec runner(s) BEFORE any prefill / decode trace exists: persistent tensors allocated after a captured trace can sit on that trace's scratch memory and
        # are overwritten by its replays (a later prefill then hangs the spec traces)
        generator.enable_spec(spec_k)
    profiler.end("generator_setup")

    if isinstance(input_prompts, list) and len(input_prompts) == 1:
        real_prompts = input_prompts * batch_size
    else:
        real_prompts, _ = load_inputs(input_prompts, batch_size, instruct=instruct)
    real_prompts = real_prompts[:batch_size]
    # padding users (batch not a multiple of the mesh rows) replay the last real prompt
    input_prompts_all = real_prompts + [real_prompts[-1]] * (padded_batch - batch_size)
    repeat_batch_prompts = [
        [input_prompts_all[(j + i) % len(input_prompts_all)] for j in range(len(input_prompts_all))]
        for i in range(repeat_batches)
    ]
    device_sampling_params = SamplingParams(temperature=0, top_k=1, top_p=1.0)
    num_tokens_generated_decode = []

    for batch_idx, prompts_batch in enumerate(repeat_batch_prompts):
        logger.info(f"Processing batch {batch_idx}")
        if (
            os.environ.get("DSV41_RAGGED") == "2"
        ):  # uniform control: every user gets user-4's ragged truncation (same length for all)
            q_ = prompts_batch[4]
            prompts_batch = [q_[: max(64, int(len(q_) * (0.5 + 0.5 * 4 / len(prompts_batch))))]] * len(prompts_batch)
        if (
            os.environ.get("DSV41_RAGGED") == "1"
        ):  # ragged prompt lengths: user u keeps the first (50 + 50 u / B) % of its prompt text
            prompts_batch = [
                p[: max(64, int(len(p) * (0.5 + 0.5 * u / len(prompts_batch))))] for u, p in enumerate(prompts_batch)
            ]
        profiler.start("preprocess_prefill_inputs", iteration=batch_idx)
        input_tokens_prefill, encoded_prompts, decoding_pos, prefill_lens = preprocess_inputs_prefill(
            prompts_batch,
            tokenizer,
            [model_args],
            instruct=instruct,
            max_generated_tokens=max_generated_tokens,
            max_prefill_len=max_seq_len,
        )
        input_tokens_prefill = torch.stack(input_tokens_prefill).view(padded_batch, -1)
        profiler.end("preprocess_prefill_inputs", iteration=batch_idx)
        logger.info(f"Encoded lengths: {decoding_pos}")

        prefill_kw = dict(
            prompt_lens=decoding_pos,
            sampling_params=device_sampling_params,
            enable_trace=enable_prefill_trace,
            chunk=prefill_chunk,
        )
        if (
            warmup_prefill
        ):  # compile / trace-capture run (idempotent: it writes the same state), reported separately and excluded from TTFT
            profiler.start("compile_prefill", iteration=batch_idx)
            generator.prefill_forward_text(input_tokens_prefill, **prefill_kw)
            profiler.end("compile_prefill", iteration=batch_idx)
        else:
            profiler.start("compile_prefill", iteration=batch_idx)
            profiler.end("compile_prefill", iteration=batch_idx)

        profiler.start("inference_prefill", iteration=batch_idx)
        prefilled_token, _ = generator.prefill_forward_text(input_tokens_prefill, **prefill_kw)
        profiler.end("inference_prefill", iteration=batch_idx)
        prefilled_token = prefilled_token.view(-1)
        logger.info("FIRSTTOK_IDS " + json.dumps([int(t) for t in prefilled_token[:batch_size]]))
        generator.m.log_dram("PF_END (after prefill, trace held)")
        if (
            os.environ.get("DSV41_PREFILL_ONLY") == "1"
        ):  # prefill measurement (DSV41_UNI_NODECODE: no decode possible): TTFT + first tokens, no decode
            prefill_t = profiler.get_duration("inference_prefill")
            real_tokens = sum(decoding_pos[:batch_size])
            logger.info(f"PREFILL_ONLY first tokens {prefilled_token[:batch_size].tolist()}")
            logger.info(
                f"TTFT (whole batch of {batch_size} users, ISL max {max(decoding_pos[:batch_size])}): {prefill_t * 1000:.0f} ms "
                f"-> prefill {real_tokens / prefill_t:.0f} tok/s ({prefill_t / batch_size * 1000:.0f} ms/user amortised)"
            )
            if hasattr(generator.m, "prefill_model"):
                logger.info(f"prefill timing {getattr(generator.m.prefill_model, 'timing', {})}")
            if os.environ.get(
                "DSV41_PREFILL_LOGITS"
            ):  # one more prefill (not timed) returning the logits of every user's last token
                _, lg_ = generator.prefill_forward_text(input_tokens_prefill, return_logits=True, **prefill_kw)
                lg_ = lg_[:batch_size].float()
                torch.save(lg_, os.environ["DSV41_PREFILL_LOGITS"])
                for u_ in range(min(batch_size, 2)):
                    v_, i_ = lg_[u_].topk(5)
                    logger.info(
                        f"PREFILL_ONLY user {u_} top5 ids {i_.tolist()} logits {[round(float(x), 3) for x in v_]}"
                    )
            for rt_ in [
                int(x) for x in os.environ.get("DSV41_ROWTOK_SWEEP", "").split(",") if x
            ]:  # chunk-size sweep: DSV41_PREFILL_ROW_TOKENS per entry, 2 calls each
                os.environ["DSV41_PREFILL_ROW_TOKENS"] = str(rt_)
                try:
                    for rep_ in range(2):
                        t_ = time.perf_counter()
                        generator.prefill_forward_text(input_tokens_prefill, **prefill_kw)
                        dt_ = time.perf_counter() - t_
                    tm_ = generator.m.timing
                    nrow_ = rt_ // users_per_row * users_per_row  # tokens per row per chunk
                    logger.info(
                        f"ROWTOK_SWEEP row_tokens={rt_} chunk={max(128, rt_ // users_per_row // 128 * 128)}: TTFT {dt_ * 1e3:.0f} ms, {real_tokens / dt_:.0f} tok/s, "
                        f"replay_loop {tm_.get('total_replay_loop', float('nan')):.2f} s, replay {tm_.get('replay_per_chunk', float('nan')):.2f} s, host {tm_.get('host_per_chunk', float('nan')):.2f} s"
                    )
                except Exception as e_:
                    logger.info(f"ROWTOK_SWEEP row_tokens={rt_} FAILED: {type(e_).__name__}: {str(e_)[:300]}")
                    break
            return
        pre_spec = None
        if spec_k and os.environ.get("DSV41_SPEC_DIAG") == "1":
            # DIAG: row-0 logits of the first spec round (position S, token = first) vs ONE plain decode step on the same prefill state, tail replay vs full replay seeding
            cp = torch.tensor(decoding_pos)
            generator.decode_forward(prefilled_token, cp, enable_trace=False, reload_inputs=True)
            lg_plain = generator.m.read_logits().float()[:batch_size]
            generator.m.release_trace()
            f1, _ = generator.prefill_forward_text(input_tokens_prefill, **prefill_kw)
            f1 = f1.view(-1)
            generator.enable_spec(spec_k)
            sp = generator.spec
            Bn = padded_batch

            def pcc(a_, b_):
                a_, b_ = a_ - a_.mean(), b_ - b_.mean()
                return float((a_ * b_).sum() / (a_.norm() * b_.norm()))

            for mode in ("tail", "full"):
                os.environ["DSV41_SPEC_FULL_REPLAY"] = "1" if mode == "full" else "0"
                X, base = sp.seed(input_tokens_prefill, decoding_pos, f1)
                a_, m_, d_ = sp._round(X, base)
                lg = (
                    generator.m.head.gather_logits(sp.dec.logits)[: Bn * sp.n]
                    .reshape(Bn, sp.n, -1)
                    .float()[:batch_size, 0]
                )
                pc = [pcc(lg[u], lg_plain[u]) for u in range(batch_size)]
                logger.info(
                    f"DIAG {mode} replay: row-0 logits PCC vs plain decode step per user min {min(pc):.5f} mean {sum(pc) / len(pc):.5f}; argmax equal {int((lg.argmax(-1) == lg_plain.argmax(-1)).sum())}/{batch_size}"
                )
            raise SystemExit("DIAG done")
        if spec_k and os.environ.get("DSV41_SPEC_FIRST") == "1":
            # spec pass FIRST on this very prefill state, then a fresh prefill for the plain decode (isolates 'second prefill leaves a different state' from spec bugs)
            if not hasattr(generator, "spec"):
                generator.enable_spec(spec_k)
            eos_ = tokenizer.eos_token_id if stop_at_eos else None
            act_ = torch.tensor([u < batch_size for u in range(padded_batch)])
            t_sp = time.perf_counter()
            gen_sp0, st0 = generator.spec_decode(
                input_tokens_prefill, decoding_pos, prefilled_token, max_generated_tokens, eos=eos_, active=act_
            )
            pre_spec = (gen_sp0, st0, time.perf_counter() - t_sp)
            generator.spec.release()
            prefilled_token, _ = generator.prefill_forward_text(input_tokens_prefill, **prefill_kw)
            prefilled_token = prefilled_token.view(-1)
        logger.info(f"First generated token: {tokenizer.decode(prefilled_token[0])!r}")

        all_outputs = [list(encoded_prompts[u][: prefill_lens[u]][: decoding_pos[u]]) for u in range(padded_batch)]
        for u in range(padded_batch):
            all_outputs[u].append(int(prefilled_token[u]))
        user_done = [False] * padded_batch
        plain_gaps = [[] for _ in range(padded_batch)]
        for u in range(batch_size, padded_batch):
            user_done[u] = True
        current_pos = torch.tensor(decoding_pos)
        out_tok = prefilled_token
        iteration, users_decoding = 0, True

        profiler.start("inference_decode", iteration=batch_idx)
        while users_decoding and iteration < max_generated_tokens:
            profiler.start(
                "compile_decode" if iteration == 0 else f"inference_decode_time_{iteration}", iteration=batch_idx
            )
            out_tok, _ = generator.decode_forward(
                out_tok,
                current_pos,
                enable_trace=enable_decode_trace,
                sampling_params=device_sampling_params,
                reload_inputs=iteration == 0 or not enable_decode_trace,
                reload_page_table=False,
                reload_sampling_params=False,
                reset_sampling_state=iteration == 0,
            )
            profiler.end(
                "compile_decode" if iteration == 0 else f"inference_decode_time_{iteration}", iteration=batch_idx
            )
            if (
                spec_k and os.environ.get("DSV41_SPEC_GAPS", "1") == "1"
            ):  # near-tie evidence of the plain stream (outside the timed region)
                t2 = generator.m.read_logits().topk(2, dim=-1).values
                for u in range(padded_batch):
                    plain_gaps[u].append(float(t2[u, 0] - t2[u, 1]))
            current_pos += 1
            for u in range(padded_batch):
                t = int(out_tok[u])
                if user_done[u]:
                    continue
                if t == tokenizer.eos_token_id and stop_at_eos:
                    user_done[u] = True
                    if all(user_done):
                        users_decoding = False
                else:
                    all_outputs[u].append(t)
            iteration += 1
        profiler.end("inference_decode", iteration=batch_idx)

        logger.info("Finished decoding, printing the final outputs...\n")
        logger.info("DONEFLAGS " + json.dumps([int(user_done[u]) for u in range(batch_size)]))
        for i in range(batch_size):
            text = tokenizer.decode(all_outputs[i])
            prompt_with_tags = tokenizer.decode(model_args.encode_prompt(prompts_batch[i], instruct=instruct))
            text_after = text.replace(prompt_with_tags, "", 1)
            p = prompts_batch[i]
            short = (p[:100] + "\n<long prompt not printed in full>\n" + p[-100:]) if len(p) > 200 else p
            logger.info(
                f"\n==REPEAT BATCH {batch_idx}\n==USER {i} - PROMPT ({decoding_pos[i]} tokens)\n{short}\n==USER {i} - OUTPUT\n{text_after.strip()}\n"
            )
        num_tokens_generated_decode.append(iteration)
        generator.m.release_trace()  # next repeat batch re-captures (page table / state changed)
        if spec_k:
            # ---- speculative pass on the SAME prompts: a fresh prefill (the plain decode advanced the state), drafter seeding from the prompt tail, spec loop ----
            plain_gen = [all_outputs[u][decoding_pos[u] :] for u in range(padded_batch)]
            plain_ms = 1000 * decode_time_sum(profiler, iteration, batch_idx) / max(iteration - 1, 1)
            if pre_spec is None:
                first2, _ = generator.prefill_forward_text(input_tokens_prefill, **prefill_kw)
                first2 = first2.view(-1)
            else:
                first2 = prefilled_token
            if not hasattr(
                generator, "spec"
            ):  # built AFTER the prefills (drafter + spec step state would not fit next to the prefill chunk memory): free the traces first
                generator.m.release_trace()
                if (
                    os.environ.get("DSV41_SPEC_FREE_PREFILL", "0") == "1"
                ):  # frees only ~11 MiB/bank and (at ISL > 512) invalidated the decode key slab: off by default
                    generator.m.prefill_model.teardown_dyn()
                generator.enable_spec(spec_k)
            adaptive = os.environ.get("DSV41_SPEC_ADAPT") == "1"
            policies = (
                [p_ for p_ in os.environ.get("DSV41_SPEC_POLICIES", "adapt").split(",") if p_] if adaptive else [None]
            )
            for pi, pol in enumerate(policies):
                if (
                    pi > 0
                ):  # fresh prefill (the previous spec pass advanced the state); spec traces are released first (re-captured by seed) so the prefill replay cannot clobber them
                    generator.spec.release()
                    if (
                        os.environ.get("DSV41_SPEC_NOREPREFILL") != "1"
                    ):  # (diagnostic switch: stale state, to see whether the prefill is what hangs the spec compile pass)
                        first2, _ = generator.prefill_forward_text(input_tokens_prefill, **prefill_kw)
                        first2 = first2.view(-1)
                        if (
                            os.environ.get("DSV41_SPEC_AFTER_PLAIN") == "1"
                        ):  # diagnostic: is the prefill-after-spec state poisoned? 6 eager plain decode steps vs the first plain stream
                            cp_, ot_, ok_ = torch.tensor(decoding_pos), first2.clone(), 0
                            seq_ = [[] for _ in range(padded_batch)]
                            for _s in range(6):
                                ot_, _ = generator.decode_forward(
                                    ot_,
                                    cp_,
                                    enable_trace=False,
                                    sampling_params=device_sampling_params,
                                    reload_inputs=True,
                                )
                                cp_ = cp_ + 1
                                for u_ in range(padded_batch):
                                    seq_[u_].append(int(ot_[u_]))
                            ok_ = sum(int(seq_[u_] == plain_gen[u_][1:7]) for u_ in range(batch_size))
                            logger.info(
                                f"REPREFILL_PLAIN_CHECK: {ok_}/{batch_size} users reproduce the first plain stream (6 tokens); first token ok {int((first2[:batch_size] == prefilled_token[:batch_size]).sum())}/{batch_size}; user0 {seq_[0]} vs {plain_gen[0][1:7]}"
                            )
                            raise SystemExit("REPREFILL_PLAIN_CHECK done")
                if pol is not None:
                    generator.spec.policy = pol
                eos = tokenizer.eos_token_id if stop_at_eos else None
                active = torch.tensor([u < batch_size for u in range(padded_batch)])
                if pre_spec is None:
                    t_sp = time.perf_counter()
                    gen_sp, st = generator.spec_decode(
                        input_tokens_prefill, decoding_pos, first2, max_generated_tokens, eos=eos, active=active
                    )
                    t_sp = time.perf_counter() - t_sp
                else:
                    gen_sp, st, t_sp = pre_spec
                ident, first_div, gaps_div, table, suspects = 0, [], [], [], []
                for u in range(batch_size):
                    pg = float("nan")
                    ps = [t for t in plain_gen[u]]
                    sp = [t for t in gen_sp[u] if t != eos][: len(ps)]
                    L_ = min(len(ps), len(sp))
                    dv = next((i for i in range(L_) if ps[i] != sp[i]), -1)
                    first_div.append(dv)
                    ident += int(dv == -1)
                    if dv > 0 and len(generator.spec.gaps[u]) >= dv:
                        pg = plain_gaps[u][dv - 1] if len(plain_gaps[u]) >= dv else float("nan")
                        gaps_div.append(
                            f"user {u} token {dv}: top1-top2 gap plain {pg:.3f} / spec {generator.spec.gaps[u][dv - 1]:.3f}"
                        )
                        if pg > 0.1:
                            suspects.append(f"user {u} (mesh row {u // generator.m.U}) token {dv} plain gap {pg:.3f}")
                    mh = getattr(generator.spec, "m_hist", [[]] * batch_size)[u]
                    table.append(
                        f"  user {u:3d} row {u // generator.m.U} len {decoding_pos[u]:5d}: first token {'ok' if int(first2[u]) == int(prefilled_token[u]) else 'MISMATCH'}, "
                        f"first divergence {dv:4d}, plain gap {pg if dv > 0 and len(plain_gaps[u]) >= dv else float('nan'):.3f}, "
                        f"spec gap {generator.spec.gaps[u][dv - 1] if dv > 0 and len(generator.spec.gaps[u]) >= dv else float('nan'):.3f}, "
                        f"rounds {len(mh)}, mean accepted {sum(mh) / max(len(mh), 1):.2f}, tokens {len(gen_sp[u])}"
                    )
                X0_ = getattr(generator.spec, "X0", None)
                if X0_ is not None:  # quality of the seeded first drafts vs the plain greedy continuation (drafter / seeding diagnostic)
                    hit_ = [0] * (X0_.shape[1] - 1)
                    for u in range(batch_size):
                        for j_ in range(X0_.shape[1] - 1):
                            hit_[j_] += int(len(plain_gen[u]) > 1 + j_ and int(X0_[u, 1 + j_]) == int(plain_gen[u][1 + j_]))
                    logger.info(
                        f"SPEC seed draft check: first-block draft j matches the plain continuation for {hit_} of {batch_size} users (j=1..); user0 X0 {X0_[0].tolist()} plain {plain_gen[0][:X0_.shape[1]]}"
                    )
                grp = {}
                for u in range(batch_size):
                    grp.setdefault(tuple(int(t) for t in input_tokens_prefill[u][: int(decoding_pos[u])]), []).append(u)
                bad_p = bad_s = 0
                for us in grp.values():
                    for u in us[1:]:
                        n_ = min(len(plain_gen[u]), len(plain_gen[us[0]]))
                        bad_p += int(plain_gen[u][:n_] != plain_gen[us[0]][:n_])
                        n_ = min(len(gen_sp[u]), len(gen_sp[us[0]]), 48)
                        bad_s += int(gen_sp[u][:n_] != gen_sp[us[0]][:n_])
                logger.info(
                    f"SPEC self-consistency: {len(grp)} distinct prompts among {batch_size} users; users differing from the first user of their identical-prompt group: plain {bad_p}, spec {bad_s} (first 48 tokens)"
                )
                tok_s_plain = 1000.0 / plain_ms if plain_ms == plain_ms and plain_ms > 0 else float("nan")
                logger.info(
                    f"=== SPEC k={spec_k if pol is None else pol} (batch {batch_size}): {st['rounds']} rounds, {st['accepted_per_round']:.3f} accepted drafts/round "
                    f"(CPU reference GSM8K: k=3 -> 2.30), P(m>=j) {[round(x, 3) for x in st['p_ge']]} (CPU: 0.905/0.796/0.693/0.598/0.489 conditional-free per position), "
                    f"{st['tok_per_round']:.3f} tokens/round/user, round {st['round_ms']:.1f} ms -> {st['tok_s_user']:.1f} tok/s/user vs plain decode "
                    f"{plain_ms:.1f} ms/token = {tok_s_plain:.1f} tok/s/user (same run) => {st['tok_s_user'] / tok_s_plain:.2f}x ==="
                )
                logger.info(
                    f"SPEC exactness vs the plain greedy stream of this run: {ident}/{batch_size} users identical, first divergence per user {first_div}; first token spec==plain: "
                    f"{int((first2[:batch_size] == prefilled_token[:batch_size]).sum())}/{batch_size}; spec wall incl. seeding {t_sp:.1f} s"
                )
                for q_, v_ in st.get("by_policy", {}).items() if str(pol).startswith("cycle:") else []:
                    logger.info(
                        "SPEC_RESULT_CYCLE "
                        + json.dumps(
                            {
                                "scenario": os.environ.get("DSV41_CUR_SCENARIO", ""),
                                "B": batch_size,
                                "policy": q_,
                                "rounds": v_["rounds"],
                                "accepted_per_round": round(v_["accepted_per_round"], 4),
                                "tok_per_round": round(v_["tok_per_round"], 4),
                                "round_ms": round(v_["round_ms"], 2),
                                "spec_tok_s_user": round(v_["tok_s_user"], 2),
                                "plain_tok_s_user": round(tok_s_plain, 2),
                                "speedup": round(v_["tok_s_user"] / tok_s_plain, 3),
                                "k_hist": v_["k_hist"],
                            }
                        )
                    )
                logger.info(
                    "SPEC_RESULT "
                    + json.dumps(
                        {
                            "scenario": os.environ.get("DSV41_CUR_SCENARIO", ""),
                            "B": batch_size,
                            "policy": pol if pol is not None else f"k{spec_k}",
                            "rounds": st["rounds"],
                            "accepted_per_round": round(st["accepted_per_round"], 4),
                            "tok_per_round": round(st["tok_per_round"], 4),
                            "round_ms": round(st["round_ms"], 2),
                            "spec_tok_s_user": round(st["tok_s_user"], 2),
                            "plain_ms": round(plain_ms, 2),
                            "plain_tok_s_user": round(tok_s_plain, 2),
                            "speedup": round(st["tok_s_user"] / tok_s_plain, 3),
                            "identical": ident,
                            "p_ge": [round(x, 3) for x in st["p_ge"]],
                            "k_hist": st.get("k_hist"),
                            "times": {k_: round(v_, 1) for k_, v_ in st.get("times", {}).items()},
                            "exp_tokens": round(st.get("mean_expected_tokens", float("nan")), 3),
                            "max_plain_gap_at_div": max(
                                [float(x.split("gap ")[1].split()[0]) for x in suspects] or [0.0]
                            ),
                        }
                    )
                )
                if hasattr(generator.spec, "conf_report"):
                    logger.info("SPEC_CONF " + json.dumps({"policy": pol, **generator.spec.conf_report()}))
                logger.info("SPEC divergence near-tie evidence: " + "; ".join(gaps_div))
                logger.info("SPEC per-user table:\n" + "\n".join(table))
                if os.environ.get("DSV41_RAGGED") in ("1", "2"):
                    for u in range(batch_size):
                        logger.info(
                            f"STREAMDUMP user {u} len {decoding_pos[u]} plain {plain_gen[u][:48]} spec {gen_sp[u][:48]}"
                        )
                logger.info(
                    "SPEC SUSPECTED REAL DIVERGENCES (plain top1-top2 gap > 0.1 at the first divergence): "
                    + ("none" if not suspects else "; ".join(suspects))
                )
                for i in range(min(batch_size, int(os.environ.get("DSV41_SPEC_PRINT", "2")))):
                    logger.info(f"==USER {i} - SPEC OUTPUT\n{tokenizer.decode(gen_sp[i]).strip()}\n")
            generator.spec.release()

    profiler.end("run")
    n_dec = num_tokens_generated_decode[0]
    prefill_t = profiler.get_duration("inference_prefill")
    decode_t = (
        sum(profiler.get_duration(f"inference_decode_time_{i}") for i in range(1, n_dec)) if n_dec > 1 else float("nan")
    )
    tok_s_user = (n_dec - 1) / decode_t if n_dec > 1 else float("nan")
    real_tokens = sum(decoding_pos[:batch_size])
    logger.info("=== Performance metrics ===")
    logger.info(
        f"Prefill compile run: {profiler.get_duration('compile_prefill'):.2f} s; decode compile+first step: {profiler.get_duration('compile_decode'):.2f} s"
    )
    logger.info(
        f"TTFT (whole batch of {batch_size} users, ISL max {max(decoding_pos[:batch_size])}): {prefill_t * 1000:.0f} ms "
        f"-> prefill {real_tokens / prefill_t:.0f} tok/s ({prefill_t / batch_size * 1000:.0f} ms/user amortised)"
    )
    logger.info(
        f"Decode: {1000 * decode_t / max(n_dec - 1, 1):.1f} ms/token @ {tok_s_user:.2f} tok/s/user ({tok_s_user * batch_size:.1f} tok/s throughput), batch {batch_size} (padded {padded_batch}), host breakdown {generator.m.timing}"
    )


@pytest.mark.timeout(14400)
@pytest.mark.parametrize(
    "input_prompts, batch_size, repeat_batches, max_seq_len, max_generated_tokens, page_params, sampling_params, enable_decode_trace, enable_prefill_trace, prefill_chunk, warmup_prefill, instruct, stop_at_eos",
    SCENARIOS,
)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {
                "l1_small_size": 16384,
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "trace_region_size": int(os.environ.get("DSV41_TRACE_REGION", "1600000000")),
            },
            id="ring",
        )
    ],
    indirect=True,
)
@torch.no_grad()
def test_dsv41_demo(
    mesh_device,
    device_params,
    input_prompts,
    batch_size,
    repeat_batches,
    max_seq_len,
    max_generated_tokens,
    page_params,
    sampling_params,
    enable_decode_trace,
    enable_prefill_trace,
    prefill_chunk,
    warmup_prefill,
    instruct,
    stop_at_eos,
):
    _run_demo(
        mesh_device,
        input_prompts,
        batch_size,
        repeat_batches,
        max_seq_len,
        max_generated_tokens,
        page_params,
        sampling_params,
        enable_decode_trace,
        enable_prefill_trace,
        prefill_chunk,
        warmup_prefill,
        instruct,
        stop_at_eos,
    )


def _run_demo_wrap(mesh_device, prompts, bs, rep, msl, mgt, pp, sp, dtr, ptr, pch, wu, ins, eos, cache, build_len):
    if os.environ.get("DSV41_TEST_FAIL_BUDGET") and os.environ.get("DSV41_TEST_FAIL_BUDGET") == os.environ.get(
        "DSV41_PREFILL_ROW_TOKENS"
    ):  # test hook of the failure path
        raise RuntimeError("injected failure")
    _run_demo(
        mesh_device,
        prompts,
        bs,
        rep,
        msl,
        mgt,
        pp,
        sp,
        dtr,
        ptr,
        pch,
        wu,
        ins,
        eos,
        cache=cache,
        build_max_seq_len=build_len,
    )


@pytest.mark.timeout(14400)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {
                "l1_small_size": 16384,
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "trace_region_size": int(os.environ.get("DSV41_TRACE_REGION", "1600000000")),
            },
            id="ring",
        )
    ],
    indirect=True,
)
def test_dsv41_demo_session(mesh_device, device_params):
    """Several scenarios of the same padded batch in ONE process, sharing one model build (a 40-layer build takes ~30 min):
    DSV41_SESSION=gsm8k_b16,prefill_128_b16,same_prompt_b16 pytest demo/text_demo.py -k session"""
    ids = os.environ.get("DSV41_SESSION", "prefill_128_b16,gsm8k_b16").split(",")
    byid = {s.id: s for s in SCENARIOS}
    omodes = [
        i.partition("@")[2] for i in ids
    ]  # optional prefill-optimisation mode per scenario: id@m (baseline) / @P / @R / @E (see tests/test_prefill_scen_device.py)
    ids = [i.partition("@")[0] for i in ids]
    chosen = [byid[i] for i in ids]
    pad_b = lambda bs: -(-bs // mesh_device.shape[0]) * mesh_device.shape[0]
    per_scenario_ctx = (
        os.environ.get("DSV41_SESSION_CTX_PER_SCENARIO") == "1"
    )  # size every build / reconfigure for the scenario's own max_seq_len (a longer context than the cached model's -> reconfigure)
    build_len_of = (
        {}
    )  # padded batch -> longest max_seq_len of its scenarios (a session with several batch sizes reconfigures the model between them)
    for s_ in chosen:
        build_len_of[pad_b(s_.values[1])] = max(build_len_of.get(pad_b(s_.values[1]), 0), s_.values[3])
    cache = {}
    failed = {}
    modes = [
        m for m in os.environ.get("DSV41_PF_ASYNC_LIST", "").split(",") if m
    ]  # per-scenario DSV41_PF_ASYNC (A/B in one process)
    rowtok = [
        m for m in os.environ.get("DSV41_ROWTOK_LIST", "").split(",") if m
    ]  # per-scenario DSV41_PREFILL_ROW_TOKENS (chunk-size sweep in one process)
    umoes = [m for m in os.environ.get("DSV41_PF_UMOE_LIST", "").split(",") if m]
    ab = os.environ.get(
        "DSV41_PFA_AB"
    )  # "flagsA|flagsB|...": prefill-tuning flags (tt/pf_tune.py, KEY=val joined by ",") of scenario i, cycling; "-" = baseline. Same build, same host: a paired A/B
    for si, (s, mode) in enumerate(zip(chosen, omodes)):
        prompts, bs, rep, msl, mgt, pp, sp, dtr, ptr, pch, wu, ins, eos = s.values
        if modes:
            os.environ["DSV41_PF_ASYNC"] = modes[si % len(modes)]
        if rowtok:
            os.environ["DSV41_PREFILL_ROW_TOKENS"] = rowtok[si % len(rowtok)]
        if ab:
            spec = ab.split("|")[si % len(ab.split("|"))]
            for k in [k for k in os.environ if k.startswith("DSV41_PFA_") and k != "DSV41_PFA_AB"]:
                del os.environ[k]
            os.environ["DSV41_PREFILL_OPT"] = (
                "1" if "OPT" in spec.split(",") else "0"
            )  # umbrella flag (tt/pf_tune.py): token OPT
            for kv in spec.split(","):
                if kv not in ("-", "OPT"):
                    os.environ[kv.split("=", 1)[0]] = kv.split("=", 1)[1]
            for _, (_, m, _) in cache.items():  # new capture of the chunk trace with the new flags
                if getattr(m, "prefill_model", None) is not None:
                    m.prefill_model.teardown_dyn()
            logger.info(f"=== PFA_AB scenario {si} {s.id}: flags '{spec}' ===")
        if (
            umoes
        ):  # per-scenario DSV41_PF_UMOE_LIST: 1 = unified prefill MoE (build with DSV41_PREFILL_MOE=unified), 0 = the moe_compute prefill path, same build
            for _, (_, m, _) in cache.items():
                pm_ = getattr(m, "prefill_model", None)
                if pm_ is None:
                    continue
                pm_.teardown_dyn()
                for _, pl_ in pm_.layers:
                    if not hasattr(pl_, "_umoe_saved"):
                        pl_._umoe_saved = pl_.umoe
                    pl_.umoe = pl_._umoe_saved if umoes[si % len(umoes)] == "1" else None
                from models.demos.blackhole.deepseek_v41_flash.tt import uni_policy

                uni_policy.set_in_use(
                    umoes[si % len(umoes)] == "1" and any(p_.umoe is not None for _, p_ in pm_.layers)
                )
        if mode:
            os.environ["DSV41_PF_MHC"] = "packed" if set(mode) & set("PREQ") else "0"
            os.environ["DSV41_PF_ROUTE_OWN"] = "1" if set(mode) & set("RE") else "0"
            os.environ["DSV41_PF_ENGRAM_OWN"] = "1" if set(mode) & set("EQ") else "0"
            for _, (_, m_, _) in cache.items():  # force a new prefill trace capture in the new mode
                pm_ = getattr(m_, "prefill_model", None)
                if pm_ is not None and getattr(pm_, "dyn", None) is not None:
                    pm_.teardown_dyn()
                    pm_.dyn_out = None
                    pm_.head_out = []
        logger.info(
            f"=== session scenario {s.id} (DSV41_PF_ASYNC={os.environ.get('DSV41_PF_ASYNC')}, ROW_TOKENS={os.environ.get('DSV41_PREFILL_ROW_TOKENS')}"
            + (
                f" MODE {mode}: PF_MHC={os.environ['DSV41_PF_MHC']} ROUTE_OWN={os.environ['DSV41_PF_ROUTE_OWN']} ENGRAM_OWN={os.environ['DSV41_PF_ENGRAM_OWN']}"
                if mode
                else ""
            )
            + ") ==="
        )
        os.environ["DSV41_CUR_SCENARIO"] = s.id
        os.environ["DSV41_RAGGED"] = "1" if s.id.endswith("_ragged") else "2" if s.id.endswith("_ragged_u4") else "0"
        if (
            os.environ.get("DSV41_SESSION_NOWARM") == "1"
        ):  # chunk-size calibration: no compile run (total_replay_loop excludes the capture)
            wu = False
        cont = (
            os.environ.get("DSV41_SESSION_CONTINUE") == "1"
        )  # keep going after a failing scenario (OOM at a large budget)
        budget_i = (
            int(os.environ.get("DSV41_PREFILL_ROW_TOKENS") or 0)
            if (os.environ.get("DSV41_PREFILL_ROW_TOKENS") or "0").isdigit()
            else 0
        )
        if cont and budget_i >= min(failed.get(s.id, 1 << 30), failed.get("*", 1 << 30)):
            logger.info(
                f"=== session scenario {s.id} ROW_TOKENS={budget_i} SKIPPED (budget >= failed {min(failed.get(s.id, 1 << 30), failed.get('*', 1 << 30))}) ==="
            )
            continue
        try:
            _run_demo_wrap(
                mesh_device,
                prompts,
                bs,
                rep,
                msl,
                mgt,
                pp,
                sp,
                dtr,
                ptr,
                pch,
                wu,
                ins,
                eos,
                cache,
                msl if per_scenario_ctx else build_len_of[pad_b(bs)],
            )
        except Exception as e:  # noqa: BLE001
            if not cont:
                raise
            oom = any(w in str(e) for w in ("Out of Memory", "out of memory", "DRAM", "OOM", "allocate"))
            key = (
                "*" if (oom and os.environ.get("DSV41_PREFILL_SPAD_MAX")) else s.id
            )  # sized for the max context: a DRAM failure holds for every ISL
            failed[key] = min(failed.get(key, 1 << 30), budget_i)
            import traceback

            logger.error(
                "FAILTRACE "
                + " <- ".join(
                    l.strip().replace("\n", " ")[:160]
                    for l in traceback.format_exc().splitlines()
                    if l.strip().startswith("File")
                )[-1500:]
            )
            logger.error(f"=== SCENARIO FAILED {s.id} ROW_TOKENS={budget_i}: {type(e).__name__}: {str(e)[:400]} ===")
            for _, (_, m, _) in cache.items():
                try:
                    pm_ = getattr(m, "prefill_model", None)
                    if pm_ is not None and getattr(pm_, "dyn_trace", None) is not None:
                        try:  # a failure INSIDE begin/end_trace_capture leaves the capture open: close it, or every later write fails
                            ttnn.end_trace_capture(mesh_device, pm_.dyn_trace, cq_id=0)
                        except Exception:  # noqa: BLE001
                            pass
                    m.release_trace()
                    if pm_ is not None:
                        pm_.teardown_dyn()
                except Exception as e2:  # noqa: BLE001
                    logger.error(f"cleanup after failure raised {type(e2).__name__}: {str(e2)[:200]}")
        else:
            for _, (_, m, _) in cache.items():
                m.release_trace()
