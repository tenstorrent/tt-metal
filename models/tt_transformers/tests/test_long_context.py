# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Minimal text-generation demo: one user, one prompt, timed.

This is a stripped-down equivalent of

    pytest models/tt_transformers/demo/simple_text_demo.py -k "performance and batch-1"

with every configuration knob resolved to a constant. Run it with:

    TT_VISIBLE_DEVICES=0 HF_MODEL=Qwen/Qwen3-8B \
    pytest models/tt_transformers/tests/test_long_context.py -s

Every context length in CONTEXT_CONFIGS is run at every batch size in BATCH_SIZES, giving
ids like `32k-b2`. Select a subset with -k:

    -k 32k        every batch at 32k          -k b1     every context at batch 1
    -k 32k-b4     one cell
"""

import bz2
import os
import sys
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
from models.tt_transformers.tt.model_config import DecodersPrecision, ModelArgs, TensorGroup

# ---------------------------------------------------------------------------
# One prefill/decode benchmark, run once per (context length, batch size) below.
# ---------------------------------------------------------------------------
# Concurrent users. 1 stays first: it is the baseline the others are read against, and the
# only row with a prior measurement.
#
# Decode pads any batch up to a single 32-row tile (model_config.py:689,
# tile_padded_batch_rows = TILE_SIZE * ceil(max_batch_size / TILE_SIZE)), so the weight
# matmuls, the norms and the chip-to-chip collectives run identical work at 1, 2 and 4 users.
# Measured at 32k, batch 1, per token: 19.96 of 27.81 ms is that fixed part; only attention
# over the KV cache (7.56 ms) and the paged cache write (0.29 ms) are charged per user. So
# the expectation is ~19.96 + 7.85*batch ms/token, i.e. 2.17x aggregate throughput at 4 users
# for 54% of the per-user rate. Confirming that trade is the point of this sweep.
BATCH_SIZES = [1, 2, 4]
MAX_GENERATED_TOKENS = 256  # hard cap on the decode loop
INSTRUCT = True  # wrap the prompt in the model's chat template
# The precision policy under measurement. Hoisted to module scope because the KV-cache size
# estimate below reads the cache dtype out of it while create_tt_model builds the model from
# it; sourcing both from one name is what stops the estimate from describing a different
# configuration than the one that runs.
OPTIMIZATIONS = DecodersPrecision.performance
# KV-cache tokens per block. Attention reads one tile row (32 positions x 128 head dims,
# 4 tiles, 4352 bytes) then consults the page table for the next block's physical address,
# so at 32 -- one tile row per block -- every single read is followed by a random jump.
# Measured in-model at 32k context, attention device time per 2 decode steps:
#   32 -> 15511 us    128 -> 15321 us    256 -> 15206/15214 us    384 -> 15324    512 -> 15307
# 256 was run twice and reproduced to 7.7 us (0.05%), while its neighbours cluster 105 us
# above it, so the dip is real. Why 256 specifically is not understood: run length alone
# predicts 512 should win, and it does not; matching the block size to sdpa_decode_k_chunk_size
# was tested at 512/512 and was the worst configuration measured. Whole model 28.195 -> 28.03
# ms/token (0.59%); the paged cache write improves 3.1% on the same mechanism.
# Trade-off: a block is the allocation unit, so a short sequence still occupies a whole
# block. Fine for this long-context benchmark, wasteful for many concurrent short chats.
# Scope: block size is a serving parameter, not a model one -- ModelArgs only accepts it
# (PagedAttentionConfig defaults to 32) and never chooses it, and under vLLM it arrives as
# cache_config.block_size. So this speeds up THIS benchmark; a deployment gets it only by
# setting its own block size to match. Worth passing to whoever owns serving config for
# long-context workloads on this hardware; it is not a change we can make on their behalf.
PAGE_BLOCK_SIZE = 256

# (label, max_seq_len, page_blocks_per_user)
#
# max_seq_len - MAX_GENERATED_TOKENS is the prompt budget: preprocess_inputs_prefill
# left-clips any longer prompt to exactly that (tt/common.py:283).
# page_blocks_per_user * PAGE_BLOCK_SIZE must cover ONE user's prompt + generated:
#   256 * 129 = 33024   and   256 * 257 = 65792
#
# The device pool is page_blocks_per_user * batch_size. PagedAttentionConfig sizes one
# SHARED pool -- attention.py:396 allocates [max_num_blocks, kv_heads, block_size, head_dim],
# with no batch dimension -- and the page table below hands each user max_num_blocks //
# batch_size of it (the same split get_decode_mask assumes at tt/common.py:1015). A pool that
# did not scale with the batch would therefore not fail; it would quietly give each user a
# fraction of the context this row claims to measure.
#
# A length beyond the model's NATIVE window (Qwen3-8B: max_position_embeddings=40960)
# is not skipped -- `_extend_context_with_yarn` below stretches the RoPE frequencies with
# YaRN so it runs. The summary's `rope` column records which rows were extended.
CONTEXT_CONFIGS = [
    ("32k", 33024, 129),  # 33024 - 256 = 32768 prompt tokens per user
    ("64k", 65792, 257),  # 65792 - 256 = 65536 prompt tokens per user
]

# Each parametrization appends a dict here and fills it in as it progresses, so a run
# that dies partway still appears in the summary with the stage it reached.
_RESULTS = []


def _hf_text_config():
    """The model's own config, read from HuggingFace (no weights loaded)."""
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(os.environ["HF_MODEL"])
    return getattr(cfg, "text_config", cfg)  # some configs nest the text params


def _native_context_len():
    """The model's trained context window."""
    return _hf_text_config().max_position_embeddings


# Bytes per element on device. The block formats carry one shared exponent byte per 16
# values on top of the mantissa, which is where the .0625 comes from: bfp8_b is 1 + 1/16,
# bfp4_b is 0.5 + 1/16. A dtype of None means "keep the torch dtype", and init_kv_cache
# builds the cache with torch.zeros -> float32; that never happens with the settings this
# test uses (ModelOptimizations always merges over a default of BFP8 for the KV cache,
# model_config.py:314) but costing it at 4 bytes keeps the estimate fail-safe rather than
# fail-open if that ever changes.
_BYTES_PER_ELEMENT = {ttnn.bfloat4_b: 0.5625, ttnn.bfloat8_b: 1.0625, ttnn.bfloat16: 2.0, None: 4.0}

# Per-chip DRAM available to the KV cache, in bytes.
#
# A Wormhole chip has 12 GiB: dram_bank_size 2147483648 x 6 channels
# (tt_metal/soc_descriptors/wormhole_b0_80_arch.yaml). Qwen3-8B's weights take ~3.2 GiB of
# that per chip on an N300 -- 4.10B parameters after the 2-way split, of which the 1.81B in
# ff1/ff3 are bfp4_b at 0.5625 B and the rest bfp8_b at 1.0625 B -- and the runtime reserves
# a further, unmeasured amount for command queues, trace buffers and program caches. 8 GiB
# is what is left, rounded down.
#
# This is DERIVED, not measured: it is a pre-flight check that stops a hopeless row from
# taking the whole pytest session down with a device OOM (which would lose the rows that did
# fit), not a guarantee that everything under it succeeds. A row close to the line can still
# fail on hardware.
KV_CACHE_BUDGET_BYTES = 8 * 1024**3


def _kv_cache_bytes(max_seq_len, batch_size, num_devices, optimizations):
    """Bytes of KV cache one chip holds for `batch_size` users at `max_seq_len` positions.

    The paged pool is replicated across the mesh and sharded over heads: attention.py:396
    allocates [max_num_blocks, n_local_kv_heads, block_size, head_dim] per layer, twice (keys
    and values), where n_local_kv_heads is the model's KV heads divided by the mesh width.
    max_num_blocks * block_size is batch_size * max_seq_len positions, so the block size
    itself drops out and only the total position count matters.
    """
    cfg = _hf_text_config()
    n_kv_heads = cfg.num_key_value_heads
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    n_local_kv_heads = max(1, n_kv_heads // num_devices)

    # Read the dtype off the same optimizations callable the model is built with, rather than
    # assuming bfp8_b: dropping the KV cache to bfp4_b costs 53% as much and changes which
    # rows fit, so a hardcoded figure would keep refusing a row that had become runnable.
    kv_dtype = optimizations(cfg.num_hidden_layers, os.environ["HF_MODEL"].strip("/").split("/")[-1]).get_tensor_dtype(
        decoder_id=0, tensor=TensorGroup.KV_CACHE
    )
    bytes_per_element = _BYTES_PER_ELEMENT.get(kv_dtype, 4.0)

    positions = batch_size * max_seq_len
    elements = cfg.num_hidden_layers * 2 * n_local_kv_heads * positions * head_dim
    return elements * bytes_per_element


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

    show_acc = any("top1" in r for r in _RESULTS)
    header = (
        f"{'ctx':<6}{'users':>6}{'prompt tok':>11}{'build s':>9}{'TTFT ms':>11}"
        f"{'compile ms':>12}{'decode ms':>11}{'tok/s/u':>9}{'tok/s':>8}"
        + (f"{'scored':>8}{'top-1 %':>9}{'top-5 %':>9}" if show_acc else "")
        + f"  {'rope':<12}{'mode':<7}status"
    )
    lines = ["", "=" * len(header), "BENCHMARK SUMMARY", "=" * len(header), header, "-" * len(header)]
    for row in _RESULTS:
        status = row["status"] if row["status"] == "ok" else f"FAILED at {row['stage']}"
        lines.append(
            f"{row['label']:<6}"
            f"{cell(row, 'batch', 'd'):>6}"
            f"{cell(row, 'prompt_tokens', 'd'):>11}"
            f"{cell(row, 'build_s', '.1f'):>9}"
            f"{cell(row, 'ttft_ms', '.2f'):>11}"
            f"{cell(row, 'compile_ms', '.2f'):>12}"
            f"{cell(row, 'decode_ms', '.2f'):>11}"
            f"{cell(row, 'tok_s_user', '.1f'):>9}"
            f"{cell(row, 'tok_s_total', '.1f'):>8}"
            + (
                f"{cell(row, 'scored', 'd'):>8}{cell(row, 'top1', '.2f'):>9}{cell(row, 'top5', '.2f'):>9}"
                if show_acc
                else ""
            )
            + f"  {row.get('rope', '-'):<12}{row.get('mode', '-'):<7}{status}"
        )
    lines.append("=" * len(header))
    logger.info("\n".join(lines))


TEMPERATURE = 0.0  # 0 == greedy (argmax)
TOP_P = 0.08
TOP_K = 32

# --accuracy scores this many predictions, taken from the END of the reference.
#
# The count and the placement are separate decisions and the placement is the important one.
# simple_text_demo splits its reference in half, which at 32k would score 16,384 predictions
# sweeping contexts 16,384 -> 32,768 and cost ~25 minutes untraced. But what needs testing is
# the depth we actually serve, and 1,024 predictions taken from the tail sit at contexts
# 31,744 -> 32,768 -- the same number the old 1024-token check scored, at 30x the context, in
# ~2 minutes rather than 25.
#
# Raise this to widen the swept range (useful for finding WHERE degradation starts rather than
# WHETHER it is present at depth); the cost is linear and untraced steps are ~92 ms at 32k.
# Clamped to half the reference so a short reference still behaves as it did before.
ACCURACY_MAX_SCORED = 1024

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


class TokenAccuracy:
    """Teacher-forced top-1/top-5 agreement against a full-precision reference.

    Ported from simple_text_demo.py's TokenAccuracy, with one change: the reference path is
    passed in rather than derived from the model name alone. That demo can only ever score at
    1024 tokens of context -- it has no rotary-scaling path, so its context is capped at the
    model's trained window -- and each context length here needs its own reference, sized to it.

    How teacher forcing works: at every position the model's prediction is recorded and then
    *discarded*, and the reference's token is fed forward instead. So the model is scored on
    N independent one-step predictions from a known-good prefix, rather than being allowed to
    wander down its own sequence where one early divergence would corrupt every later position.
    """

    def __init__(self, reference_path):
        assert os.path.exists(reference_path), (
            f"No reference at {reference_path}. Generate one sized for this context with:\n"
            f"  HF_MODEL=$HF_MODEL python models/tt_transformers/tests/generate_reference_outputs.py "
            f"--total_length <ctx> --model $HF_MODEL --output_file {reference_path}"
        )
        logger.info(f"Loading reference from {reference_path}")
        data = torch.load(reference_path)
        reference_tokens = data["reference_tokens"]
        # Everything up to `split` primes the context; the tail after it is scored. Taking the
        # tail rather than the second half is what puts the scored predictions at full depth:
        # a 32768-token reference scoring 1024 predictions covers contexts 31744 -> 32768, not
        # 16384 -> 32768. Never more than half, so a short reference behaves as it always did.
        total = reference_tokens.shape[-1]
        scored = min(ACCURACY_MAX_SCORED, total // 2)
        split = total - scored
        self.input_prompt = reference_tokens[0, :split]
        self.reference_tokens = reference_tokens[0, split:]
        self.top5_tokens = data["top5_tokens"][split - 1 :, :]
        self.maxindex = len(self.reference_tokens) - 1
        self.gt_pos = -1
        self.store_predicted_tokens = []

    def prompt_text(self, tokenizer):
        return tokenizer.decode(self.input_prompt.tolist())

    def collect_predicted_tokens(self, token):
        """Record what the model predicted; return what it must be fed next."""
        self.store_predicted_tokens.append(token)
        self.gt_pos += 1
        return self.reference_tokens[min(self.gt_pos, self.maxindex)].unsqueeze(-1).unsqueeze(-1)

    def compute_accuracy(self):
        n = min(len(self.reference_tokens), len(self.store_predicted_tokens))
        top1 = sum(self.top5_tokens[i, 0].item() == self.store_predicted_tokens[i] for i in range(n))
        top5 = sum(self.store_predicted_tokens[i] in self.top5_tokens[i, :] for i in range(n))
        return top1 / n, top5 / n, n


# --accuracy is slower per step than the performance run: teacher forcing forbids Metal trace,
# so a step costs ~92 ms rather than ~27. At the default ACCURACY_MAX_SCORED of 1024 that is
# ~2 minutes plus a long prefill, well inside the performance-mode limit -- the headroom here
# exists so raising that cap does not silently turn into a timeout that reads like a hang. The
# marker is evaluated at collection, before any fixture can see the option, so read argv.
#
# The performance-mode limit is 4x what one user needs because prefill is sequential in the
# batch (generator.py's prefill loop runs one user at a time) and this test prefills twice per
# row -- once to compile and capture the trace, once timed. So the prefill phase costs
# 2 * batch full passes, and BATCH_SIZES tops out at 4. Decode is unaffected: it is one traced
# step per token whatever the batch.
_ACCURACY_RUN = "--accuracy" in sys.argv
TEST_TIMEOUT = 14400 if _ACCURACY_RUN else 1800 * max(BATCH_SIZES)


@torch.no_grad()
@pytest.mark.timeout(TEST_TIMEOUT)
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
#
# Stacked parametrize marks are applied bottom-up and the test id is assembled in that same
# order, so the context mark stays innermost (closest to the function) to keep ids reading
# "32k-b2" rather than "b2-32k". `-k 32k` still selects every batch at that context.
@pytest.mark.parametrize("batch_size", BATCH_SIZES, ids=[f"b{b}" for b in BATCH_SIZES])
@pytest.mark.parametrize(
    "label, max_seq_len, page_blocks_per_user",
    CONTEXT_CONFIGS,
    ids=[c[0] for c in CONTEXT_CONFIGS],
)
def test_text_demo(
    mesh_device, reset_seeds, monkeypatch, request, label, max_seq_len, page_blocks_per_user, batch_size
):
    assert os.getenv("HF_MODEL"), "Set HF_MODEL, e.g. export HF_MODEL=Qwen/Qwen3-8B"
    prompt_budget = max_seq_len - MAX_GENERATED_TOKENS
    logger.info(f"[{label}-b{batch_size}] target prompt {prompt_budget} tokens, max_seq_len {max_seq_len}")

    # Refuse a row whose KV cache cannot fit before building anything. Skipping is not
    # squeamishness: a device-side out-of-memory failure here takes the whole pytest session
    # down, so letting one hopeless row run would also lose the results of every row that fit.
    num_devices = mesh_device.get_num_devices() if isinstance(mesh_device, ttnn.MeshDevice) else 1
    kv_bytes = _kv_cache_bytes(max_seq_len, batch_size, num_devices, OPTIMIZATIONS)
    if kv_bytes > KV_CACHE_BUDGET_BYTES:
        pytest.skip(
            f"[{label}-b{batch_size}] KV cache needs {kv_bytes / 1024**3:.1f} GiB per chip, over the "
            f"{KV_CACHE_BUDGET_BYTES / 1024**3:.0f} GiB budget. {batch_size} users x {max_seq_len} positions "
            f"across {num_devices} chips. Halving the cache dtype to bfp4_b would bring it to "
            f"{kv_bytes * 0.5625 / 1.0625 / 1024**3:.1f} GiB."
        )

    # Registered up front so a failure still shows up in the end-of-run summary.
    # `stage` advances through the test; if we raise, it names where we stopped.
    row = {"label": label, "batch": batch_size, "status": "FAILED", "stage": "setup"}
    _RESULTS.append(row)

    # --tracy_decode sets BOTH settings a profiling run needs. Half of it is worse than
    # none: trace left on dies in post-processing with "Device data mismatch"; 256 decode
    # steps left on overflows the device marker buffer and the CSV comes back incomplete.
    tracy_decode = request.config.getoption("--tracy_decode")
    # --accuracy turns this into a token-accuracy test: the model is teacher-forced through a
    # full-precision reference and scored on agreement, instead of being timed. Trace must be off
    # (teacher forcing rewrites the token between steps, which a replayed command stream cannot
    # see) and the prompt comes from the reference rather than the corpus, so the context the
    # model is scored on is exactly the one the reference was built from.
    accuracy = request.config.getoption("--accuracy")
    assert not (accuracy and tracy_decode), "--accuracy and --tracy_decode are mutually exclusive"
    # Teacher forcing reads and rewrites user 0's token only (see the decode loop), so at
    # batch > 1 it would score one user while feeding the reference to all of them -- and
    # report a plausible top-1 figure while doing it. Refuse rather than mislead.
    assert (
        not accuracy or batch_size == 1
    ), f'--accuracy is single-user; got batch {batch_size}. Select one batch, e.g. -k "{label}-b1 and accuracy"'

    token_acc = None
    if accuracy:
        ref_path = request.config.getoption("--accuracy_ref") or os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "reference_outputs",
            f"{os.environ['HF_MODEL'].split('/')[-1]}_{label}.refpt",
        )
        token_acc = TokenAccuracy(ref_path)

    enable_trace = False if (tracy_decode or accuracy) else ENABLE_TRACE
    if accuracy:
        max_generated_tokens = len(token_acc.reference_tokens)
    elif tracy_decode:
        max_generated_tokens = TRACY_MAX_GENERATED_TOKENS
    else:
        max_generated_tokens = MAX_GENERATED_TOKENS
    row["mode"] = "accuracy" if accuracy else ("tracy" if tracy_decode else "bench")
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
        max_num_blocks=page_blocks_per_user * batch_size,
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
        max_batch_size=batch_size,
        optimizations=lambda args: OPTIMIZATIONS(args.n_layers, args.model_name),
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
    # Divides exactly: the pool is page_blocks_per_user * batch_size by construction, so
    # every user gets page_blocks_per_user blocks and none are left over.
    permutation = torch.randperm(page_blocks_per_user * batch_size)
    page_table = torch.argsort(permutation).reshape(batch_size, page_blocks_per_user)

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
    # In accuracy mode the prompt is the reference's first half, and the chat template is off:
    # generate_reference_outputs.py encodes the corpus with instruct=False, so templating here
    # would shift every position relative to the reference and score a different sequence.
    prompt_text = token_acc.prompt_text(tokenizer) if accuracy else PROMPT
    instruct = False if accuracy else INSTRUCT
    # One entry per user. Every user gets the SAME prompt, which is the measurement we want
    # rather than a shortcut: sampling is greedy, so all users stay at an identical position
    # and produce identical tokens, which puts every one of them at full context depth on
    # every step. That is the worst case for the only part of decode that scales with the
    # batch. It also has to be done -- preprocess_inputs_prefill returns one row per prompt,
    # and reshaping a single row into `batch_size` rows SUCCEEDS, silently splitting one
    # prompt into fragments and calling them users.
    input_tokens_pt, encoded_prompts, decoding_pos, prefill_lens = preprocess_inputs_prefill(
        [prompt_text] * batch_size,
        tokenizer,
        [model_args],
        instruct,
        # MAX_GENERATED_TOKENS, not max_generated_tokens: this argument sets the PROMPT
        # budget (preprocess_inputs_prefill does `max_prefill_len -= max_generated_tokens`),
        # so passing --tracy_decode's cap of 2 would leave 33022 tokens for the prompt, which
        # up to 65536 -- profiling a 64k prefill while the label still said 32k. The prompt
        # must be identical in both modes; only the decode-loop length differs.
        MAX_GENERATED_TOKENS,
        max_prefill_len=max_seq_len,
    )
    input_tokens = torch.stack(input_tokens_pt).view(batch_size, -1)
    if accuracy:
        assert len(encoded_prompts[0]) == len(token_acc.input_prompt), (
            f"reference prompt is {len(token_acc.input_prompt)} tokens but re-encoded to "
            f"{len(encoded_prompts[0])}. The reference is handed over as text and re-tokenized, "
            "so a non-round-tripping tokenizer would shift every scored position. Refusing to "
            "report accuracy against a misaligned reference."
        )
    row["prompt_tokens"] = len(encoded_prompts[0])
    row["stage"] = "prefill"
    logger.info(f"Prompt is {decoding_pos[0]} tokens (padded to {prefill_lens[0]})")

    assert (
        max_generated_tokens + max(decoding_pos) <= max_seq_len
    ), f"prompt ({max(decoding_pos)}) + generated ({max_generated_tokens}) must fit in max_seq_len ({max_seq_len})"

    # The same budget again, against the KV cache one user actually owns rather than against
    # the declared context length. Redundant while page_blocks_per_user * PAGE_BLOCK_SIZE
    # equals max_seq_len, and no longer redundant the moment the pool is a product: an error
    # in that multiplication is exactly what this catches, and without it the failure is users
    # overwriting one another's cache blocks and returning wrong tokens rather than raising.
    # Mirrors simple_text_demo.py:1172-1178, which checks the same thing per user.
    per_user_cache_tokens = PAGE_BLOCK_SIZE * page_blocks_per_user
    assert max_generated_tokens + max(decoding_pos) <= per_user_cache_tokens, (
        f"prompt ({max(decoding_pos)}) + generated ({max_generated_tokens}) exceeds the "
        f"{per_user_cache_tokens} tokens of KV cache each of the {batch_size} user(s) owns "
        f"({page_blocks_per_user} blocks x {PAGE_BLOCK_SIZE} tokens)"
    )

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
        # Teacher forcing, before the forward pass that consumes out_tok: score what the model
        # predicted, then overwrite it with the reference's token. Without this the model walks
        # its own sequence and a single early divergence corrupts every later position, which
        # measures drift rather than per-step precision loss.
        if accuracy:
            out_tok[0] = token_acc.collect_predicted_tokens(out_tok[0].item())

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
        # In accuracy mode a stop token is just one more prediction to score -- the sequence is
        # the reference's, not the model's, so there is nothing to stop. Breaking here would
        # silently truncate the run and report agreement over a prefix.
        if token in tokenizer.stop_tokens and not accuracy:
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
    echoed_prompt = tokenizer.decode(model_args.encode_prompt(prompt_text, instruct=instruct))
    answer = full_text.replace(echoed_prompt, "", 1).strip()

    steady_times = decode_times[1:]
    avg_decode = sum(steady_times) / len(steady_times) if steady_times else float("nan")

    row["ttft_ms"] = ttft * 1000
    if decode_times:
        row["compile_ms"] = decode_times[0] * 1000
    if steady_times:
        row["decode_ms"] = avg_decode * 1000
        row["tok_s_user"] = 1 / avg_decode
        # Aggregate across users. The whole point of the batch sweep is the gap between this
        # column and the per-user one: the fixed 19.96 ms/token is paid once however many
        # users are in flight, so tok/s should rise while tok/s/u falls.
        row["tok_s_total"] = batch_size / avg_decode

    # In accuracy mode the "output" is the reference's own continuation (teacher forcing fed it
    # back every step), so printing it as the model's output would be actively misleading -- and
    # the prompt is the reference's first half, not the corpus.
    if accuracy:
        logger.info(f"\n==PROMPT (reference first half, {len(encoded_prompts[0])} tokens)\n{prompt_text[:160]}...\n")
    else:
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
            f"@ {1 / avg_decode:.1f} tok/s/user ({batch_size / avg_decode:.1f} tok/s)"
        )
    logger.info(f"[{label}] Tokens generated:       {len(new_tokens)} (1 prefill + {len(decode_times)} decode steps)")

    if accuracy:
        top1, top5, scored = token_acc.compute_accuracy()
        row["top1"], row["top5"], row["scored"] = top1 * 100, top5 * 100, scored
        logger.info(f"[{label}] === Token accuracy ===")
        logger.info(f"[{label}] Predictions scored:     {scored}")
        logger.info(f"[{label}] Top-1 agreement:        {top1 * 100:.2f}%")
        logger.info(f"[{label}] Top-5 agreement:        {top5 * 100:.2f}%")
        logger.info(
            f"[{label}] Context spanned:        {len(encoded_prompts[0])} -> "
            f"{len(encoded_prompts[0]) + scored} tokens"
        )

    # -- Step 10: gate the result --------------------------------------------
    # Without these the test only proves "did not crash": every metric above is
    # printed just as happily for a model emitting garbage. Mirrors the upstream
    # demo's special-token check (simple_text_demo.py:1443). `new_tokens` already
    # excludes the EOS token, which Step 8 drops rather than appends.
    #
    # Batch > 1 is still gated on user 0 alone, and that is sufficient rather than lazy:
    # every user was handed the same prompt and sampling is greedy, so all users decode the
    # same tokens from the same position. User 0 is the whole batch. The printed sample
    # answer and the EOS break in Step 8 read user 0 for the same reason. Give users
    # different prompts and all three of those need revisiting.
    if accuracy:
        # The generated sequence here is the reference's, not the model's, so the special-token
        # gate below would be testing the corpus. What matters instead is that every position
        # got scored -- a short run would silently report accuracy over a prefix.
        assert scored == len(token_acc.reference_tokens), (
            f"scored {scored} of {len(token_acc.reference_tokens)} reference positions; "
            "the decode loop stopped early (EOS in the reference?)"
        )
    else:
        assert len(new_tokens) >= 2, f"model generated only {len(new_tokens)} token(s)"
        special = sorted({t for t in new_tokens if t in tokenizer.all_special_ids})
        assert not special, f"model produced special tokens in its output: {special[:8]}"

    row["status"] = "ok"  # only after the output gate above has passed
