# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Minimal text-generation demo: one user, one prompt, timed.

This is a stripped-down equivalent of

    pytest models/tt_transformers/demo/simple_text_demo.py -k "performance and batch-1"

with every configuration knob resolved to a constant. Run it with:

    TT_VISIBLE_DEVICES=0 HF_MODEL=Qwen/Qwen3-8B \
    pytest models/tt_transformers/tests/test_long_context.py -s
"""

import bz2
import os
import time

import pytest
import torch
from loguru import logger
from tracy import signpost

import ttnn
from models.tt_transformers.tt.common import (
    PagedAttentionConfig,
    RopeScalingYarn,
    create_tt_model,
    preprocess_inputs_prefill,
    sample_host,
)
from models.tt_transformers.tt.generator import Generator, SamplingParams
from models.tt_transformers.tt.model_config import DecodersPrecision, ModelArgs

# ---------------------------------------------------------------------------
# One prefill/decode benchmark, run once per context length below.
# ---------------------------------------------------------------------------
BATCH_SIZE = 1  # concurrent users
MAX_GENERATED_TOKENS = 256  # hard cap on the decode loop
INSTRUCT = True  # wrap the prompt in the model's chat template
PAGE_BLOCK_SIZE = 32  # KV-cache tokens per block

# (label, max_seq_len, page_max_num_blocks)
#
# max_seq_len - MAX_GENERATED_TOKENS is the prompt budget: preprocess_inputs_prefill
# left-clips any longer prompt to exactly that (tt/common.py:283).
# page_max_num_blocks * PAGE_BLOCK_SIZE must cover prompt + generated:
#   32 * 1032 = 33024   and   32 * 2056 = 65792
#
# A length beyond the model's NATIVE window (Qwen3-8B: max_position_embeddings=40960)
# is not skipped -- `_extend_context_with_yarn` below stretches the RoPE frequencies with
# YaRN so it runs. The summary's `rope` column records which rows were extended.
CONTEXT_CONFIGS = [
    ("32k", 33024, 1032),  # 33024 - 256 = 32768 prompt tokens
    ("64k", 65792, 2056),  # 65792 - 256 = 65536 prompt tokens
]

# Each parametrization appends a dict here and fills it in as it progresses, so a run
# that dies partway still appears in the summary with the stage it reached.
_RESULTS = []


def _native_context_len():
    """The model's trained context window, read from its HF config (no weights loaded)."""
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(os.environ["HF_MODEL"])
    cfg = getattr(cfg, "text_config", cfg)  # some configs nest the text params
    return cfg.max_position_embeddings


def _extend_context_with_yarn(monkeypatch, target_len, native_len):
    """Stretch the model's context window to `target_len` using YaRN RoPE scaling.

    `max_position_embeddings` is a *training* limit, not a structural one: RoPE angles are
    computed (theta_i(pos) = pos / base^(2i/d)), not looked up, so nothing overflows past it.
    What breaks is distribution -- the slowest frequency component rotates only ~2.9 degrees
    across Qwen3-8B's entire 40960-token training range, so positions beyond it produce
    angles the model has never seen. YaRN interpolates exactly those low-frequency
    components and leaves the fast ones (which cycle many times within the trained range,
    and carry short-range detail) untouched.

    Injected by wrapping ModelArgs.__init__ because the RoPE tables are built inside
    Transformer.__init__ (model.py:72 -> RotarySetup(rope_scaling=args.rope_scaling)), while
    `create_tt_model` constructs ModelArgs and Transformer in one call. Patching here avoids
    duplicating that function's warm-weight-cache branch, which is what keeps the build at
    ~6s instead of a cold 16GB load. The cache is unaffected: its identity key covers
    (model_name, n_layers, mesh_shape, components, build_variant) and none of those read
    rope_scaling or max_context_len.

    Applies the patch only; the caller reads the *resulting* config off the built model so
    the summary reports what happened rather than what was intended.
    """
    factor = target_len / native_len
    orig_init = ModelArgs.__init__

    def _init_with_yarn(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        # factor and original_max_position_embeddings must both be non-None: they reach
        # YarnRotaryEmbedding.__init__ via rope_scaling.model_dump(exclude_none=True), so a
        # None would silently drop a required argument. rope_type is declared exclude=True
        # and is correctly omitted from that dump.
        self.rope_scaling = RopeScalingYarn(
            rope_type="yarn",
            factor=factor,
            original_max_position_embeddings=native_len,
        )
        self.max_context_len = target_len  # lifts the guard at tt/common.py:259

    monkeypatch.setattr(ModelArgs, "__init__", _init_with_yarn)
    logger.info(f"Extending context {native_len} -> {target_len} with YaRN (factor {factor:.2f})")


@pytest.fixture(scope="module", autouse=True)
def _benchmark_summary():
    """Print one table covering every parametrization, after the last one finishes."""
    _RESULTS.clear()
    yield
    if not _RESULTS:
        return

    def cell(row, key, fmt):
        v = row.get(key)
        return "-" if v is None else format(v, fmt)

    header = (
        f"{'ctx':<6}{'prompt tok':>11}{'build s':>9}{'TTFT ms':>11}"
        f"{'compile ms':>12}{'decode ms':>11}{'tok/s/u':>9}  {'rope':<12}{'mode':<7}status"
    )
    lines = ["", "=" * len(header), "BENCHMARK SUMMARY", "=" * len(header), header, "-" * len(header)]
    for row in _RESULTS:
        status = row["status"] if row["status"] == "ok" else f"FAILED at {row['stage']}"
        lines.append(
            f"{row['label']:<6}"
            f"{cell(row, 'prompt_tokens', 'd'):>11}"
            f"{cell(row, 'build_s', '.1f'):>9}"
            f"{cell(row, 'ttft_ms', '.2f'):>11}"
            f"{cell(row, 'compile_ms', '.2f'):>12}"
            f"{cell(row, 'decode_ms', '.2f'):>11}"
            f"{cell(row, 'tok_s_user', '.1f'):>9}"
            f"  {row.get('rope', '-'):<12}{row.get('mode', '-'):<7}{status}"
        )
    lines.append("=" * len(header))
    logger.info("\n".join(lines))


TEMPERATURE = 0.0  # 0 == greedy (argmax)
TOP_P = 0.08
TOP_K = 32

ENABLE_TRACE = True  # replay a recorded command stream instead of re-dispatching

# Decode steps used under --tracy_decode. 2 is the floor, not a round number: iteration 0
# is the compile/capture pass and is excluded from the steady-state average, so 1 would
# leave no samples at all. Prefill emits token 1, so this yields 3 tokens total.
TRACY_MAX_GENERATED_TOKENS = 2

# ~190k tokens of A Tale of Two Cities, the corpus the sibling tests in this
# directory use (generate_reference_outputs.py:49). preprocess_inputs_prefill
# left-clips it to each parametrization's max_seq_len - MAX_GENERATED_TOKENS, accounting
# for chat-template overhead itself.
_CORPUS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tale-of-two-cities.txt.bz2")
with bz2.open(_CORPUS, "rt", encoding="utf-8") as _f:
    PROMPT = _f.read()


@torch.no_grad()
@pytest.mark.timeout(1800)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": True, "num_command_queues": 1}],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
# Not indirect: these go straight to the test function, unlike mesh_device/device_params
# which are consumed by fixtures.
@pytest.mark.parametrize(
    "label, max_seq_len, page_max_num_blocks",
    CONTEXT_CONFIGS,
    ids=[c[0] for c in CONTEXT_CONFIGS],
)
def test_text_demo(mesh_device, reset_seeds, monkeypatch, request, label, max_seq_len, page_max_num_blocks):
    assert os.getenv("HF_MODEL"), "Set HF_MODEL, e.g. export HF_MODEL=Qwen/Qwen3-8B"
    prompt_budget = max_seq_len - MAX_GENERATED_TOKENS
    logger.info(f"[{label}] target prompt {prompt_budget} tokens, max_seq_len {max_seq_len}")

    # Registered up front so a failure still shows up in the end-of-run summary.
    # `stage` advances through the test; if we raise, it names where we stopped.
    row = {"label": label, "status": "FAILED", "stage": "setup"}
    _RESULTS.append(row)

    # --tracy_decode sets BOTH settings a profiling run needs. Half of it is worse than
    # none: trace left on dies in post-processing with "Device data mismatch"; 256 decode
    # steps left on overflows the device marker buffer and the CSV comes back incomplete.
    tracy_decode = request.config.getoption("--tracy_decode")
    enable_trace = False if tracy_decode else ENABLE_TRACE
    max_generated_tokens = TRACY_MAX_GENERATED_TOKENS if tracy_decode else MAX_GENERATED_TOKENS
    row["mode"] = "tracy" if tracy_decode else "bench"
    if tracy_decode:
        logger.warning(
            f"[{label}] --tracy_decode: trace disabled, {max_generated_tokens} decode steps. The "
            "timings below are NOT valid measurements (the decode average is a single "
            "sample at the shortest context) -- read the profiler CSV instead."
        )

    # If this context exceeds what the model was trained on, stretch it with YaRN rather
    # than skipping. NOTE: this makes long-context output *plausible*, not *validated* --
    # those positions are still outside the training range, so treat the numbers as timing
    # data. Applied only when needed, since static YaRN perturbs short-context behaviour too.
    native_ctx = _native_context_len()
    extension_requested = max_seq_len > native_ctx
    if extension_requested:
        _extend_context_with_yarn(monkeypatch, max_seq_len, native_ctx)
    else:
        logger.info(f"[{label}] within native context window ({native_ctx}); no RoPE scaling")

    # -- Step 1: describe the KV-cache memory layout -------------------------
    # Paged attention stores the KV cache as a pool of fixed-size blocks rather
    # than one contiguous buffer per user. This object just carries the two
    # numbers that define the pool; nothing is allocated yet.
    paged_attention_config = PagedAttentionConfig(
        block_size=PAGE_BLOCK_SIZE,
        max_num_blocks=page_max_num_blocks,
    )

    # -- Step 2: build the model on the device -------------------------------
    # Reads HF_MODEL, loads the weights (or the warm ttnn cache), quantizes
    # them, and lays them out across the chips in `mesh_device`. This is the
    # slow step: minutes on a cold cache, ~seconds on a warm one.
    #
    # `optimizations` is a callable because it needs the model's layer count,
    # which isn't known until the config has been read. ModelArgs calls it.
    logger.info("Building model...")
    t_build = time.perf_counter()
    model_args, model, kv_cache, _state_dict = create_tt_model(
        mesh_device,
        instruct=INSTRUCT,
        max_batch_size=BATCH_SIZE,
        optimizations=lambda args: DecodersPrecision.performance(args.n_layers, args.model_name),
        max_seq_len=max_seq_len,
        paged_attention_config=paged_attention_config,
        dtype=ttnn.bfloat8_b,
    )
    build_time = time.perf_counter() - t_build
    row["build_s"] = build_time
    row["stage"] = "tokenize"
    tokenizer = model_args.tokenizer

    # Label from the model that actually got built, not from what we asked for: a patch that
    # failed to apply, or a model shipping its own rope_scaling, would otherwise be reported
    # wrongly. `applied` is None when no scaling is in effect.
    applied = model_args.rope_scaling
    if applied is None:
        row["rope"] = "native"
    elif applied.factor is not None:
        row["rope"] = f"{applied.rope_type.value} x{applied.factor:.2f}"
    else:
        row["rope"] = applied.rope_type.value
    if extension_requested:
        # Without this, a patch that silently missed would still pass (max_context_len is
        # what gates the run) while the summary claimed YaRN and the rotations were native.
        assert applied is not None and applied.rope_type.value == "yarn", (
            f"YaRN extension was requested but the built model has rope_scaling={applied!r} "
            "-- the ModelArgs.__init__ patch did not take effect"
        )
        assert (
            model_args.max_context_len >= max_seq_len
        ), f"max_context_len {model_args.max_context_len} < requested {max_seq_len}"
    logger.info(f"Model built in {build_time:.1f}s ({model_args.n_layers} layers, {model_args.device_name})")

    # -- Step 3: map logical positions to physical blocks --------------------
    # Row b of the page table lists, in order, the physical block IDs that hold
    # user b's KV cache. The shuffle is deliberate: an identity mapping would
    # still pass even if the indirection were broken.
    permutation = torch.randperm(page_max_num_blocks)
    page_table = torch.argsort(permutation).reshape(BATCH_SIZE, page_max_num_blocks // BATCH_SIZE)

    # -- Step 4: wrap the model in the inference driver ----------------------
    # Generator owns prefill/decode orchestration and trace capture. It takes
    # lists because it supports data parallelism (one entry per submesh); we
    # have one group, so single-element lists.
    generator = Generator([model], [model_args], mesh_device, tokenizer=tokenizer)

    # -- Step 5: turn the prompt into token IDs ------------------------------
    # Applies the chat template (INSTRUCT), tokenizes, and returns:
    #   input_tokens_pt : padded token IDs, one row per user
    #   encoded_prompts : unpadded token IDs
    #   decoding_pos    : true prompt length per user -> the first decode position
    #   prefill_lens    : padded length per user
    input_tokens_pt, encoded_prompts, decoding_pos, prefill_lens = preprocess_inputs_prefill(
        [PROMPT],
        tokenizer,
        [model_args],
        INSTRUCT,
        # MAX_GENERATED_TOKENS, not max_generated_tokens: this argument sets the PROMPT
        # budget (preprocess_inputs_prefill does `max_prefill_len -= max_generated_tokens`),
        # so passing --tracy_decode's cap of 2 would leave 33022 tokens for the prompt, which
        # up to 65536 -- profiling a 64k prefill while the label still said 32k. The prompt
        # must be identical in both modes; only the decode-loop length differs.
        MAX_GENERATED_TOKENS,
        max_prefill_len=max_seq_len,
    )
    input_tokens = torch.stack(input_tokens_pt).view(BATCH_SIZE, -1)
    row["prompt_tokens"] = len(encoded_prompts[0])
    row["stage"] = "prefill"
    logger.info(f"Prompt is {decoding_pos[0]} tokens (padded to {prefill_lens[0]})")

    assert (
        max_generated_tokens + max(decoding_pos) <= max_seq_len
    ), f"prompt ({max(decoding_pos)}) + generated ({max_generated_tokens}) must fit in max_seq_len ({max_seq_len})"

    # -- Step 6: decide where sampling runs ----------------------------------
    # If the model can pick the next token on the device, we hand it the
    # sampling parameters and get token IDs back. Otherwise we get raw logits
    # and sample on the host, which costs a round trip per token.
    use_device_sampling = model._supports_on_device_sampling
    sampling_params = SamplingParams(temperature=TEMPERATURE, top_k=TOP_K, top_p=TOP_P) if use_device_sampling else None
    logger.info(f"Sampling on {'device' if use_device_sampling else 'host'}")

    # -- Step 7: prefill -----------------------------------------------------
    # One pass over the whole prompt. Fills the KV cache and produces the first
    # generated token. The first call compiles kernels and captures traces, so
    # it is run once untimed and then again for the measurement.
    prefill_kwargs = dict(
        page_table=page_table,
        kv_cache=[kv_cache],
        prompt_lens=decoding_pos,
        sampling_params=sampling_params,
        enable_trace=enable_trace,
        # Skip Generator.warmup_model_prefill's sweep over every padded length <=
        # max_prefill_chunk_size. We discard the first call anyway, so it records the
        # one trace we need; sweeping the rest just inflates runtime.
        warmup_prefill=False,
    )

    logger.info("Prefill warmup (compile + trace capture)...")
    t_warm = time.perf_counter()
    generator.prefill_forward_text(input_tokens, **prefill_kwargs)
    logger.info(f"Prefill warmup took {time.perf_counter() - t_warm:.1f}s")

    logger.info("Prefill (timed)...")
    t_prefill = time.perf_counter()
    prefill_out = generator.prefill_forward_text(input_tokens, **prefill_kwargs)
    ttft = time.perf_counter() - t_prefill

    if sampling_params is not None and isinstance(prefill_out, tuple):
        out_tok, _log_probs = prefill_out  # device already sampled
    else:
        out_tok = torch.argmax(prefill_out, dim=-1)  # host argmax over logits

    # -- Step 8: decode loop -------------------------------------------------
    # One token per iteration. current_pos starts at the true prompt length
    # (not the padded length) so KV writes and RoPE land at the right offsets.
    current_pos = torch.tensor(decoding_pos)
    generated_tokens = list(encoded_prompts[0]) + [int(out_tok[0].item())]
    decode_times = []

    row["stage"] = "decode"
    # Timeline marker for the profiler. Everything after this is decode, so a report can be
    # sliced to it with `tt-perf-report --start-signpost decode`; without it the warmup
    # prefill, the timed prefill, and the decode steps are all averaged into one pool.
    # Safe outside a profiling run -- signpost() just logs its header.
    signpost("decode")
    logger.info("Decoding...")
    for iteration in range(max_generated_tokens):
        t_step = time.perf_counter()
        logits, _log_probs = generator.decode_forward(
            out_tok,
            current_pos,
            enable_trace=enable_trace,
            page_table=page_table,
            kv_cache=[kv_cache],
            reset_batch=(iteration == 0),
            sampling_params=sampling_params,
            # Token history for the frequency/presence/repetition penalties. Inert
            # while those are at their no-op defaults, but omitting them makes a
            # later `frequency_penalty=...` silently do nothing instead of erroring.
            prompt_tokens=input_tokens,
            output_tokens=out_tok,
        )
        decode_times.append(time.perf_counter() - t_step)

        if use_device_sampling:
            out_tok = logits.unsqueeze(1)  # already token IDs
        else:
            _, out_tok = sample_host(logits, temperature=TEMPERATURE, top_p=TOP_P, on_host=True)

        current_pos += 1

        token = int(out_tok[0].item())
        if token in tokenizer.stop_tokens:
            logger.info(f"Hit EOS at iteration {iteration}")
            break
        generated_tokens.append(token)

    # -- Step 9: report ------------------------------------------------------
    # Iteration 0 carries kernel compile and trace capture, so it is excluded
    # from the steady-state average and reported separately.
    #
    # `new_tokens` is what the model actually produced: one token from prefill
    # plus one per decode step that was kept. It is NOT len(decode_times) -- the
    # EOS iteration runs but its token is deliberately dropped (see Step 8).
    new_tokens = generated_tokens[len(encoded_prompts[0]) :]
    full_text = tokenizer.decode(generated_tokens)
    prompt_text = tokenizer.decode(model_args.encode_prompt(PROMPT, instruct=INSTRUCT))
    answer = full_text.replace(prompt_text, "", 1).strip()

    steady_times = decode_times[1:]
    avg_decode = sum(steady_times) / len(steady_times) if steady_times else float("nan")

    row["ttft_ms"] = ttft * 1000
    if decode_times:
        row["compile_ms"] = decode_times[0] * 1000
    if steady_times:
        row["decode_ms"] = avg_decode * 1000
        row["tok_s_user"] = 1 / avg_decode

    logger.info(f"\n==PROMPT ({len(PROMPT)} chars, clipped)\n{PROMPT[:160]}...\n\n==OUTPUT\n{answer}\n")
    logger.info(f"=== Performance metrics [{label}] ===")
    logger.info(f"[{label}] Prompt tokens:          {len(encoded_prompts[0])}")
    logger.info(f"[{label}] Model build:            {build_time:.1f}s")
    logger.info(f"[{label}] Time to first token:    {ttft * 1000:.2f}ms")
    if decode_times:
        logger.info(f"[{label}] Decode compile (iter 0):{decode_times[0] * 1000:9.2f}ms")
    if steady_times:
        logger.info(
            f"[{label}] Decode steady state:    {avg_decode * 1000:.2f}ms "
            f"@ {1 / avg_decode:.1f} tok/s/user ({BATCH_SIZE / avg_decode:.1f} tok/s)"
        )
    logger.info(f"[{label}] Tokens generated:       {len(new_tokens)} (1 prefill + {len(decode_times)} decode steps)")

    # -- Step 10: gate the result --------------------------------------------
    # Without these the test only proves "did not crash": every metric above is
    # printed just as happily for a model emitting garbage. Mirrors the upstream
    # demo's special-token check (simple_text_demo.py:1443). `new_tokens` already
    # excludes the EOS token, which Step 8 drops rather than appends.
    assert len(new_tokens) >= 2, f"model generated only {len(new_tokens)} token(s)"
    special = sorted({t for t in new_tokens if t in tokenizer.all_special_ids})
    assert not special, f"model produced special tokens in its output: {special[:8]}"

    row["status"] = "ok"  # only after the output gate above has passed
