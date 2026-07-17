#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Single-P150 BoolQ evaluation: opens a [1, 1] mesh, loads Llama-3.2-1B-Instruct,
and answers the first 256 questions of the ``google/boolq`` train split in
batches of 16. Prefill + on-device decode are inlined here (adapted from
``utils.ttt_generation_worker.TttGenerationWorker``) rather than instantiated as
a class.

Stdout and stderr are redirected at the file-descriptor level *before any
heavy imports* so that C++ / metal log lines (which write to fd 1/2 via
``printf``/``fprintf``) also land in the log file. Only the log-path banner
goes to the user's terminal.

Outputs (all under ``generated/boolq_bench/<utc-timestamp>/``):
    - output.log       : everything written to fd 1/2 during the run
    - responses.csv    : one row per question (idx, question, expected,
                         reply, prediction, correct, prompt_len, ...)

Run:
    cd tt-metal
    python3 tt-train/sources/examples/grpo_remote_rollout/prompt_bench/boolq_bench.py
"""

from __future__ import annotations

# -----------------------------------------------------------------------------
# stdout / stderr redirection --- MUST happen before importing ttnn / torch.
# -----------------------------------------------------------------------------
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parents[4]  # .../tt-metal

_RUN_STAMP = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
_OUTPUT_DIR = _REPO_ROOT / "generated" / "boolq_bench" / _RUN_STAMP
_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
_LOG_PATH = _OUTPUT_DIR / "output.log"
_CSV_PATH = _OUTPUT_DIR / "responses.csv"

# Announce the log path on the user's real terminal (best effort; skip if no tty).
try:
    with open("/dev/tty", "w") as _tty:
        _tty.write(f"[boolq_bench] logging to: {_LOG_PATH}\n")
        _tty.write(f"[boolq_bench] responses csv: {_CSV_PATH}\n")
        _tty.flush()
except OSError:
    pass

# Force Python to not buffer its own stdio.
os.environ.setdefault("PYTHONUNBUFFERED", "1")

# Redirect fd 1 & fd 2 to the log file so C-level printf / metal logger output
# is captured too. Then rebind sys.stdout / sys.stderr to line-buffered text
# wrappers over the same fds.
_log_fd = os.open(str(_LOG_PATH), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
os.dup2(_log_fd, 1)
os.dup2(_log_fd, 2)
os.close(_log_fd)
sys.stdout = os.fdopen(1, "w", buffering=1, encoding="utf-8", errors="replace")
sys.stderr = os.fdopen(2, "w", buffering=1, encoding="utf-8", errors="replace")

print(f"[boolq_bench] run started at {_RUN_STAMP} (UTC)")
print(f"[boolq_bench] output dir: {_OUTPUT_DIR}")

# -----------------------------------------------------------------------------
# Regular imports.
# -----------------------------------------------------------------------------
import csv  # noqa: E402
import gc  # noqa: E402
import time  # noqa: E402
from typing import Any, FrozenSet, List, Sequence, Tuple  # noqa: E402

# Make ``utils.*`` importable when run directly.
_EXAMPLE_ROOT = str(_THIS_DIR.parent)
if _EXAMPLE_ROOT not in sys.path:
    sys.path.insert(0, _EXAMPLE_ROOT)


# -----------------------------------------------------------------------------
# Constants.
# -----------------------------------------------------------------------------
MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
MESH_SHAPE = (1, 1)  # single P150 board
BATCH_SIZE = 16  # 16 prompts per prefill+decode pass
NUM_QUESTIONS = 256  # first 256 BoolQ train questions
MAX_SEQ_LEN = 2048  # must fit longest prompt + MAX_NEW_TOKENS
MAX_NEW_TOKENS = 128  # short answer (yes/no + brief reason) --- plenty of decode budget

# Paged-KV sizing.
_PAGED_BLOCK_SIZE = 32
_MIN_NUM_BLOCKS = 1024

# Trace-region size that tt-transformers' decode trace needs.
_TRACE_REGION_SIZE = 50_000_000

# Async-read cadence for the decode loop.
_READ_EVERY = 4

# BoolQ system prompt: ask for a short answer that still includes yes/no, so we
# can see the model's reasoning without letting it ramble past MAX_NEW_TOKENS.
SYSTEM_PROMPT = (
    "Answer the following yes/no question based on the provided passage. "
    "Give a short answer (one or two sentences) that clearly includes the word "
    "'yes' or the word 'no'."
)


# -----------------------------------------------------------------------------
# Prompt preparation.
# -----------------------------------------------------------------------------
def _apply_chat(tokenizer, messages) -> List[int]:
    """Chat-templated token IDs; unwraps ``BatchEncoding`` for newer transformers."""
    result = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    ids = result["input_ids"] if hasattr(result, "keys") else result
    return [int(t) for t in ids]


def _load_boolq_prompts(tokenizer, num_questions: int) -> List[dict]:
    """Load the first ``num_questions`` BoolQ train items and return one dict
    per item with ``question``, ``passage``, ``expected`` ('yes'/'no') and
    chat-templated ``ids``."""
    from datasets import load_dataset

    print(f"[boolq_bench] loading google/boolq train[:{num_questions}]")
    ds = load_dataset("google/boolq", split="train").select(range(num_questions))
    prompts: List[dict] = []
    for i, ex in enumerate(ds):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Question: {ex['question']}? Context: {ex['passage']}",
            },
        ]
        ids = _apply_chat(tokenizer, messages)
        prompts.append(
            {
                "idx": i,
                "question": ex["question"],
                "passage": ex["passage"],
                "expected": "yes" if ex["answer"] else "no",
                "ids": ids,
            }
        )
    lens = [len(p["ids"]) for p in prompts]
    print(
        f"[boolq_bench] prompts ready: n={len(prompts)}  "
        f"len min={min(lens)} med={sorted(lens)[len(lens) // 2]} max={max(lens)}"
    )
    return prompts


# -----------------------------------------------------------------------------
# Model setup (adapted from TttGenerationWorker.__init__, DP=1).
# -----------------------------------------------------------------------------
def _build_model(mesh_device: Any) -> Tuple[Any, Any, Any, Any, Any, Any, int]:
    """Return ``(model_args, model, generator, tt_kv_cache, page_table,
    paged_attention_config, paged_cache_max_seq_len)``.

    Mirrors ``TttGenerationWorker`` for a single submesh (DP=1) with the
    caller-supplied ``BATCH_SIZE``.
    """
    import torch
    import ttnn

    from models.tt_transformers.tt.common import PagedAttentionConfig
    from models.tt_transformers.tt.generator import Generator
    from models.tt_transformers.tt.model import Transformer
    from models.tt_transformers.tt.model_config import ModelArgs

    from utils.llama_ttt_presets import bf16_attn_bfp8_mlp_optimizations

    dtype = ttnn.bfloat16

    required_blocks_per_user = (MAX_SEQ_LEN + _PAGED_BLOCK_SIZE - 1) // _PAGED_BLOCK_SIZE
    max_num_blocks = max(_MIN_NUM_BLOCKS, BATCH_SIZE * required_blocks_per_user)
    blocks_per_user = max_num_blocks // BATCH_SIZE
    max_num_blocks = blocks_per_user * BATCH_SIZE
    paged_cache_max_seq_len = _PAGED_BLOCK_SIZE * blocks_per_user
    paged_attention_config = PagedAttentionConfig(block_size=_PAGED_BLOCK_SIZE, max_num_blocks=max_num_blocks)

    page_table = torch.arange(max_num_blocks, dtype=torch.int32).repeat(1).reshape(BATCH_SIZE, blocks_per_user)

    os.environ["HF_MODEL"] = MODEL_ID  # ModelArgs reads HF_MODEL from env

    model_args = ModelArgs(
        mesh_device,
        instruct=True,
        max_batch_size=BATCH_SIZE,
        optimizations=lambda ma: bf16_attn_bfp8_mlp_optimizations(ma.n_layers, ma.model_name),
        max_seq_len=MAX_SEQ_LEN,
        cache_hf=True,
        dummy_weights=False,
    )
    model_args.lm_head_dtype = ttnn.bfloat16
    model_args.ccl_dtype = ttnn.bfloat16

    state_dict = model_args.load_state_dict()
    weight_cache_path = model_args.weight_cache_path(dtype)

    model = Transformer(
        args=model_args,
        mesh_device=mesh_device,
        dtype=dtype,
        state_dict=state_dict,
        weight_cache_path=weight_cache_path,
        paged_attention_config=paged_attention_config,
    )
    tt_kv_cache = [[layer.attention.layer_past for layer in model.layers]]

    generator = Generator(model=[model], model_args=[model_args], mesh_device=mesh_device, tokenizer=None)

    assert model.sampling is not None, (
        "On-device sampling unavailable for this vocab_size / mesh shape; the decode "
        "loop bakes sampling into the trace and requires model.sampling."
    )
    return (
        model_args,
        model,
        generator,
        tt_kv_cache,
        page_table,
        paged_attention_config,
        paged_cache_max_seq_len,
    )


def _reset_kv_cache(model: Any, generator: Any) -> None:
    import ttnn

    for layer in model.layers:
        k_cache, v_cache = layer.attention.layer_past
        ttnn.mul(k_cache, 0, output_tensor=k_cache)
        ttnn.mul(v_cache, 0, output_tensor=v_cache)
    generator.prev_page_table = None


# -----------------------------------------------------------------------------
# Batch generation (adapted from TttGenerationWorker.generate).
# -----------------------------------------------------------------------------
def _generate_batch(
    *,
    model: Any,
    generator: Any,
    tt_kv_cache: Any,
    page_table: Any,
    sampling_params: Any,
    batch_prompts_ids: List[List[int]],
    pad_token_id: int,
    stop_token_ids: FrozenSet[int],
    max_new_tokens: int,
    paged_cache_max_seq_len: int,
    max_prefill_len: int,
) -> Tuple[List[List[int]], float, float, int]:
    """Prefill + on-device decode for a batch. Returns
    ``(completions, prefill_seconds, decode_seconds, decode_steps)``.
    """
    import torch
    import ttnn

    _reset_kv_cache(model, generator)

    # Truncate from the tail if any prompt overflows the prefill budget.
    prompts = [list(p) for p in batch_prompts_ids]
    for i, p in enumerate(prompts):
        if len(p) > max_prefill_len:
            prompts[i] = p[-max_prefill_len:]

    prompt_lens = [len(p) for p in prompts]
    max_prompt_len = max(prompt_lens)
    assert max_prompt_len + max_new_tokens <= paged_cache_max_seq_len, (
        f"prompt_len {max_prompt_len} + max_new_tokens {max_new_tokens} exceeds "
        f"paged cache {paged_cache_max_seq_len}"
    )
    assert (
        len(prompts) == BATCH_SIZE
    ), f"batch must be exactly {BATCH_SIZE} (pad short batches upstream); got {len(prompts)}"

    input_tokens = torch.full((BATCH_SIZE, max_prompt_len), pad_token_id, dtype=torch.int32)
    for i, p in enumerate(prompts):
        input_tokens[i, : len(p)] = torch.tensor(p, dtype=torch.int32)

    t_prefill = time.perf_counter()
    prefill_out = generator.prefill_forward_text(
        input_tokens,
        page_table=page_table,
        kv_cache=tt_kv_cache,
        prompt_lens=prompt_lens,
        sampling_params=sampling_params,
        warmup_prefill=False,
        enable_trace=True,
    )
    prefilled_token = (prefill_out[0] if isinstance(prefill_out, tuple) else prefill_out).reshape(-1)
    prefill_s = time.perf_counter() - t_prefill

    completions: List[List[int]] = [[] for _ in range(BATCH_SIZE)]
    user_done = [False] * BATCH_SIZE

    def _collect_step(step_tokens: List[int]) -> None:
        for u in range(BATCH_SIZE):
            if user_done[u]:
                continue
            tok = step_tokens[u]
            if tok in stop_token_ids:
                user_done[u] = True
            else:
                completions[u].append(tok)

    _collect_step([int(t) for t in prefilled_token.tolist()])

    if all(user_done) or max_new_tokens <= 1:
        return completions, prefill_s, 0.0, 0

    current_pos = torch.tensor(prompt_lens, dtype=torch.int32)
    out_tok = prefilled_token.unsqueeze(1)

    buffered_reads: List[Any] = []
    read_events: Any = None

    def _drain() -> None:
        for ev in read_events:
            ttnn.event_synchronize(mesh_event=ev)
        for step_reads in buffered_reads:
            gathered = generator.process_decode_output_host(step_reads, is_tokens=True)
            tokens = gathered[0] if isinstance(gathered, tuple) else gathered
            _collect_step([int(t) for t in tokens.reshape(-1).tolist()])

    t_decode = time.perf_counter()
    steps_executed = 0
    for step in range(max_new_tokens - 1):
        decoded = generator.decode_forward(
            out_tok,
            current_pos,
            page_table=page_table,
            kv_cache=tt_kv_cache,
            enable_trace=True,
            sampling_params=sampling_params,
            reset_batch=(step == 0),
            prompt_tokens=input_tokens,
            output_tokens=out_tok,
            read_from_device=False,
        )
        step_reads, read_events = generator.read_decode_output(decoded, async_read=True)
        buffered_reads.append(step_reads)
        current_pos = current_pos + 1
        steps_executed += 1

        if (step + 1) % _READ_EVERY == 0:
            _drain()
            buffered_reads = []
            if all(user_done):
                break

    if buffered_reads:
        _drain()

    decode_s = time.perf_counter() - t_decode
    return completions, prefill_s, decode_s, steps_executed


# -----------------------------------------------------------------------------
# Answer parsing.
# -----------------------------------------------------------------------------
def _parse_yes_no(text: str) -> str:
    """Return 'yes', 'no', or 'unknown'. Uses whichever word appears FIRST in
    the reply so multi-sentence answers like ``"Yes, because ..."`` and
    ``"No, the passage says ..."`` are classified correctly."""
    import re

    match = re.search(r"\b(yes|no)\b", text.lower())
    return match.group(1) if match else "unknown"


# -----------------------------------------------------------------------------
# Main.
# -----------------------------------------------------------------------------
def main() -> int:
    import ttnn
    from transformers import AutoTokenizer

    from models.common.sampling import SamplingParams

    from utils.llama_ttt_presets import llama_stop_and_pad

    required_chips = MESH_SHAPE[0] * MESH_SHAPE[1]
    available_chips = len(ttnn.get_device_ids())
    if available_chips < required_chips:
        print(f"[boolq_bench] need >= {required_chips} chip(s); host exposes {available_chips}")
        return 1

    print(f"[boolq_bench] model={MODEL_ID}  batch_size={BATCH_SIZE}  max_new_tokens={MAX_NEW_TOKENS}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    stop_token_ids_seq, pad_token_id = llama_stop_and_pad(MODEL_ID)
    stop_token_ids: FrozenSet[int] = frozenset(int(t) for t in stop_token_ids_seq)

    prompts = _load_boolq_prompts(tokenizer, NUM_QUESTIONS)

    assert NUM_QUESTIONS % BATCH_SIZE == 0, (
        f"NUM_QUESTIONS ({NUM_QUESTIONS}) must be a multiple of BATCH_SIZE ({BATCH_SIZE}) "
        "so every batch is full (the decode trace is captured for this exact batch size)."
    )

    # ETH dispatch is only valid on N300/T3K/N300_2x2 (Wormhole). Single-P150
    # (Blackhole) must use WORKER dispatch; ``DispatchCoreConfig()`` picks the
    # cluster-appropriate type/axis automatically.
    mesh_device = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(*MESH_SHAPE),
        trace_region_size=_TRACE_REGION_SIZE,
        dispatch_core_config=ttnn.DispatchCoreConfig(),
    )

    model_args = model = generator = tt_kv_cache = page_table = None
    try:
        _t_setup = time.perf_counter()
        model_args, model, generator, tt_kv_cache, page_table, _, paged_cache_max_seq_len = _build_model(mesh_device)
        max_prefill_len = model_args.max_seq_len - MAX_NEW_TOKENS
        print(
            f"[boolq_bench] model built in {time.perf_counter() - _t_setup:.2f}s  "
            f"max_prefill_len={max_prefill_len}  paged_cache_max_seq_len={paged_cache_max_seq_len}"
        )

        sampling_params = SamplingParams(temperature=0.0, top_k=0, top_p=1.0, seed=0)

        with open(_CSV_PATH, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "idx",
                    "expected",
                    "prediction",
                    "correct",
                    "prompt_len",
                    "new_tokens",
                    "question",
                    "reply",
                ]
            )
            f.flush()

            correct_count = 0
            total_prefill_s = 0.0
            total_decode_s = 0.0
            total_new_tokens = 0

            for batch_start in range(0, NUM_QUESTIONS, BATCH_SIZE):
                batch = prompts[batch_start : batch_start + BATCH_SIZE]
                batch_ids = [p["ids"] for p in batch]

                print(
                    f"\n[boolq_bench] ===== batch {batch_start // BATCH_SIZE + 1}/"
                    f"{NUM_QUESTIONS // BATCH_SIZE}  (questions {batch_start}"
                    f"..{batch_start + BATCH_SIZE - 1}) ====="
                )
                t0 = time.perf_counter()
                completions, prefill_s, decode_s, steps = _generate_batch(
                    model=model,
                    generator=generator,
                    tt_kv_cache=tt_kv_cache,
                    page_table=page_table,
                    sampling_params=sampling_params,
                    batch_prompts_ids=batch_ids,
                    pad_token_id=pad_token_id,
                    stop_token_ids=stop_token_ids,
                    max_new_tokens=MAX_NEW_TOKENS,
                    paged_cache_max_seq_len=paged_cache_max_seq_len,
                    max_prefill_len=max_prefill_len,
                )
                total_s = time.perf_counter() - t0
                total_prefill_s += prefill_s
                total_decode_s += decode_s
                total_new_tokens += sum(len(c) for c in completions)

                for prompt_entry, completion in zip(batch, completions):
                    reply_text = tokenizer.decode(completion, skip_special_tokens=True)
                    prediction = _parse_yes_no(reply_text)
                    is_correct = prediction == prompt_entry["expected"]
                    if is_correct:
                        correct_count += 1
                    writer.writerow(
                        [
                            prompt_entry["idx"],
                            prompt_entry["expected"],
                            prediction,
                            int(is_correct),
                            len(prompt_entry["ids"]),
                            len(completion),
                            prompt_entry["question"],
                            reply_text,
                        ]
                    )
                    verdict = "OK" if is_correct else "--"
                    print(
                        f"[boolq_bench] q#{prompt_entry['idx']:>3}  "
                        f"expected={prompt_entry['expected']:>3}  "
                        f"pred={prediction:>7}  {verdict}  "
                        f"prompt_len={len(prompt_entry['ids']):>4}  "
                        f"new_tokens={len(completion):>3}"
                    )
                    print(f"    Q: {prompt_entry['question']}")
                    # Indent every line of the reply so multi-line answers stay grouped
                    # under the log entry (and are visually easy to skim).
                    for line in (reply_text.rstrip() or "<empty>").splitlines() or ["<empty>"]:
                        print(f"    A: {line}")
                f.flush()

                decode_tok_s = (steps / decode_s) if decode_s > 0 else 0.0
                print(
                    f"[boolq_bench] batch done: prefill={prefill_s:.2f}s  "
                    f"decode={decode_s:.2f}s ({steps} steps, {decode_tok_s:.1f} tok/s/batch)  "
                    f"total={total_s:.2f}s"
                )

        accuracy = correct_count / NUM_QUESTIONS
        print(
            f"\n[boolq_bench] DONE  accuracy={correct_count}/{NUM_QUESTIONS}={accuracy:.3f}  "
            f"total_prefill={total_prefill_s:.2f}s  total_decode={total_decode_s:.2f}s  "
            f"total_new_tokens={total_new_tokens}"
        )
    finally:
        model = None
        generator = None
        tt_kv_cache = None
        model_args = None
        page_table = None
        gc.collect()
        ttnn.close_mesh_device(mesh_device)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
