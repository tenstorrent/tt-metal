#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""NemotronH-30B end-to-end demo with ISL sweep and profiling.

Runs the full pipeline (chat template → prefill → traced decode) at one or
more input sequence lengths (ISL) and prints a performance table.

Usage examples
--------------
# Single prompt, default ISL (use whatever the prompt tokenises to):
    python demo/simple_text_demo.py \\
        --prompt "Explain how Mamba2 SSM layers work" \\
        --osl 120

# ISL sweep (pads/truncates a fixed seed text to hit exactly N tokens):
    python demo/simple_text_demo.py \\
        --isl 128 512 1024 2048 \\
        --osl 100

# Chat mode with a custom system prompt:
    python demo/simple_text_demo.py \\
        --prompt "What is the capital of France?" \\
        --system "You are a helpful AI assistant." \\
        --osl 60

# ISL sweep, eager decode (no trace, cpu_gate=True):
    python demo/simple_text_demo.py --isl 256 1024 --osl 50 --eager

# Quiet (no per-token output, summary table only):
    python demo/simple_text_demo.py --isl 128 512 1024 --osl 100 --quiet

Environment
-----------
    source python_env/bin/activate
    export TT_METAL_HOME=$(pwd)
    export PYTHONPATH=ttnn:tools:.
    export LD_LIBRARY_PATH=build_Release/lib:/opt/openmpi-v5.0.7-ulfm/lib:$LD_LIBRARY_PATH
"""
from __future__ import annotations

import argparse
import os
import sys
import time

# ---------------------------------------------------------------------------
# Path bootstrap (works whether invoked directly or via pytest)
# ---------------------------------------------------------------------------
_ROOT = os.environ.get("TT_METAL_HOME", os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../")))
for _p in (os.path.join(_ROOT, "ttnn"), os.path.join(_ROOT, "tools"), _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch

torch.set_flush_denormal(True)
torch.set_num_threads(1)

# ---------------------------------------------------------------------------
# Seed text for ISL sweep — long enough for any supported ISL
# ---------------------------------------------------------------------------
_SEED_TEXT = (
    "Tenstorrent Blackhole is a high-performance AI accelerator chip designed for "
    "large-scale machine learning inference and training workloads. "
    "The chip features a novel tile-based dataflow execution model where each Tensix "
    "core operates on its own dispatch queue, enabling fine-grained parallelism across "
    "thousands of independent compute elements. "
    "NemotronH-30B is a 52-layer hybrid model combining 23 Mamba2 SSM layers, "
    "23 sparse MoE transformer layers, and 6 dense-attention layers. "
    "The SSM layers maintain a persistent state tensor of shape [1, 64, 64, 128] "
    "that is updated every token without any KV cache. "
    "The MoE layers use top-6 routing over 128 experts with a shared expert path. "
    "Inference on the Blackhole QuietBox uses tensor parallelism across four chips "
    "(TP=4) connected via FABRIC_1D with linear CCL topology. "
    "The traced decode path captures a single-token forward pass as a TTNN trace, "
    "then replays it for each generated token, achieving approximately 18 tokens "
    "per second at steady state on the TP=4 configuration. "
    "Weight sharding follows column-parallel for up-projections and row-parallel for "
    "down-projections, with all-reduce CCL operations after each row-parallel matmul. "
    "Expert weights are stored in bfloat16 and sharded column-parallel across four "
    "devices, requiring 13.7 GiB per device (fits within the 32 GiB DRAM budget). "
    "The paged KV cache for the six dense-attention layers uses a block size of 32 "
    "tokens and a default maximum sequence length of 4096 tokens. "
    "Positional encoding uses RoPE with a base frequency of 10000, applied only in "
    "dense-attention layers (Mamba2 layers are position-agnostic by design). "
    "The vocabulary size is 131072 tokens with tied embeddings disabled, and the "
    "model uses bfloat16 throughout for all activations and most weights. "
    "Pre-warm of DRAM weight addresses before the first forward pass is required "
    "to avoid defective address regions on device 2 of the QuietBox configuration. "
    "The SSM scalar weights (dt_bias, A_log, D) are uploaded to L1 to avoid DRAM "
    "address corruption during the initial allocation pass. "
) * 8  # repeat to give plenty of tokens for large ISL targets


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_prompt_of_isl(tokenizer, isl: int) -> str:
    """Return a string that tokenises to exactly `isl` tokens."""
    ids = tokenizer.encode(_SEED_TEXT, add_special_tokens=True)
    if len(ids) < isl:
        # Repeat until we have enough tokens
        while len(ids) < isl:
            ids = ids + tokenizer.encode(_SEED_TEXT, add_special_tokens=False)
    ids = ids[:isl]
    return tokenizer.decode(ids)


def _fmt_ms(s: float) -> str:
    return f"{s * 1000:.0f} ms"


def _fmt_tps(tps: float) -> str:
    return f"{tps:.2f} tok/s" if tps > 0 else "—"


def _print_table(rows: list[dict]) -> None:
    if not rows:
        return
    cols = ["ISL", "OSL", "TTFT", "Prefill ms/tok", "Decode tok/s", "Decode ms/tok", "Warmup"]
    widths = [max(len(c), max(len(str(r.get(c, ""))) for r in rows)) for c in cols]
    sep = "  ".join("-" * w for w in widths)
    header = "  ".join(c.ljust(w) for c, w in zip(cols, widths))
    print(f"\n{'='*len(sep)}")
    print("NemotronH-30B Performance Summary (TP=4 QB)")
    print(f"{'='*len(sep)}")
    print(header)
    print(sep)
    for r in rows:
        print("  ".join(str(r.get(c, "—")).ljust(w) for c, w in zip(cols, widths)))
    print(f"{'='*len(sep)}\n")


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------


def _encode_prompt(
    tokenizer,
    prompt: str | None,
    system: str | None,
    isl_list_mode: bool,
    token_ids: list[int] | None = None,
) -> list[int]:
    """Return token ids for one run, applying the chat template when appropriate."""
    if token_ids is not None:
        return token_ids
    user_text = prompt or "Explain the key differences between SSM and attention mechanisms."
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user_text})
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    return tokenizer.encode(user_text, add_special_tokens=True)


def run_demo(
    isl_list: list[int] | None,
    prompt: str | None,
    system: str | None,
    osl: int,
    eager: bool,
    quiet: bool,
    max_seq_len: int,
) -> None:
    """Run the demo using the NemotronHForCausalLM Generator class.

    Uses the same prefill_forward / decode_forward call pattern as the
    tt-inference-server so the demo exercises the real serving code path.
    """
    from models.demos.nvidia_nvidia_nemotron_3_nano_30b_a3b_bf16.tt import NemotronHForCausalLM
    from models.demos.nvidia_nvidia_nemotron_3_nano_30b_a3b_bf16.tt.generate import _load_tokenizer
    from models.demos.nvidia_nvidia_nemotron_3_nano_30b_a3b_bf16.tt.tp import close_device_tp4, open_device_tp4

    print("Opening TP=4 mesh device (4× Blackhole)...", flush=True)
    mesh = open_device_tp4()
    tokenizer = _load_tokenizer()

    # initialize_vllm_model loads WeightCache internally (same as the server does).
    # hf_config is unused by this model's factory; pass None for compatibility.
    print(f"Initializing NemotronHForCausalLM (max_seq_len={max_seq_len})...", flush=True)
    gen = NemotronHForCausalLM.initialize_vllm_model(
        hf_config=None,
        mesh_device=mesh,
        max_batch_size=1,
        max_seq_len=max_seq_len,
    )
    kv_cache = gen.allocate_kv_cache(max_batch_size=1, max_seq_len=max_seq_len)

    # Warmup: compile kernels once before the ISL sweep.
    print("Warming up (compiling kernels)...", flush=True)
    gen.warmup_model_prefill(kv_cache=kv_cache, enable_trace=False)
    if not eager:
        gen.warmup_model_decode(kv_cache=kv_cache, enable_trace=True)

    # Build the list of (label, token_ids) pairs.
    runs: list[tuple[str, list[int]]] = []
    if isl_list:
        for isl in isl_list:
            seed_prompt = _make_prompt_of_isl(tokenizer, isl)
            ids = tokenizer.encode(seed_prompt, add_special_tokens=True)
            runs.append((f"ISL={len(ids)} tokens (raw)", ids))
    else:
        ids = _encode_prompt(tokenizer, prompt, system, isl_list_mode=False)
        runs.append((f"ISL={len(ids)} tokens (chat)", ids))

    table_rows: list[dict] = []

    for run_idx, (label, token_ids) in enumerate(runs):
        isl = len(token_ids)
        print(f"\n{'─'*60}", flush=True)
        print(f"Run {run_idx+1}/{len(runs)}: {label}", flush=True)
        print(f"  Max output: {osl} tokens | Mode: {'eager' if eager else 'traced'}", flush=True)

        # Reset state between requests (same as server does between sequences).
        gen.reset_state()

        tokens_pt = torch.tensor([token_ids], dtype=torch.int64)  # [1, S]
        start_pos = torch.zeros(1, dtype=torch.int64)

        # ── Prefill ────────────────────────────────────────────────────────
        # prefill_forward runs S=1 decode steps internally (no batched kernel).
        # Returns logits [1, vocab] for the last prompt position.
        t_prefill_start = time.perf_counter()
        last_logits = gen.prefill_forward(tokens_pt, current_pos=start_pos)  # [1, vocab]
        t_prefill = time.perf_counter() - t_prefill_start

        # First generated token comes from the last-prompt-position logits.
        next_tok = int(last_logits[0].argmax())
        generated: list[int] = [next_tok]
        if not quiet:
            print(f"  First token ({t_prefill*1000:.0f} ms): {repr(tokenizer.decode([next_tok]))}", flush=True)

        # ── Decode loop ────────────────────────────────────────────────────
        # Feed tokens one at a time, exactly as the inference server does.
        # current_pos at step s = isl + s (the slot the token occupies).
        t_decode_start = time.perf_counter()
        for step in range(osl - 1):
            if generated[-1] == tokenizer.eos_token_id:
                break
            tok_t = torch.tensor([[next_tok]], dtype=torch.int64)  # [1, 1]
            pos_t = torch.tensor([isl + step], dtype=torch.int64)  # [1]
            logits_t = gen.decode_forward(tok_t, current_pos=pos_t)  # [1, 1, vocab]
            next_tok = int(logits_t[0, 0].argmax())
            generated.append(next_tok)
            if not quiet and step < 3:
                print(f"  tok {step+2}: {repr(tokenizer.decode([next_tok]))}", flush=True)
        t_decode = time.perf_counter() - t_decode_start

        n_decode = len(generated) - 1  # tokens produced by decode_forward
        decode_tps = n_decode / t_decode if t_decode > 0 and n_decode > 0 else 0.0
        prefill_ptok_ms = t_prefill / max(isl - 1, 1) * 1000
        decode_ptok_ms = 1000.0 / decode_tps if decode_tps > 0 else 0.0

        text = tokenizer.decode(generated, skip_special_tokens=True)
        if not quiet:
            print(f"\nOutput (first 200 chars): {repr(text[:200])}")

        row = {
            "ISL": isl,
            "OSL": len(generated),
            "TTFT": _fmt_ms(t_prefill),
            "Prefill ms/tok": f"{prefill_ptok_ms:.0f} ms",
            "Decode tok/s": _fmt_tps(decode_tps),
            "Decode ms/tok": f"{decode_ptok_ms:.0f} ms" if decode_ptok_ms > 0 else "—",
            "Warmup": "done",
        }
        table_rows.append(row)
        print(f"  Wall time: {t_prefill + t_decode:.1f}s", flush=True)

    _print_table(table_rows)

    print("Closing device...", flush=True)
    close_device_tp4(mesh)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="NemotronH-30B end-to-end demo with optional ISL sweep",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--isl",
        type=int,
        nargs="+",
        metavar="N",
        help="Input sequence length(s) in tokens for the sweep. "
        "A seed text is padded/truncated to exactly N tokens. "
        "Cannot be combined with --prompt.",
    )
    p.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="User message for chat-mode inference. "
        "Tokenised length determines the ISL. "
        "Cannot be combined with --isl.",
    )
    p.add_argument("--system", type=str, default=None, help="Optional system prompt prepended in chat mode.")
    p.add_argument(
        "--osl", type=int, default=100, help="Max output sequence length (new tokens to generate). Default: 100."
    )
    p.add_argument(
        "--max-seq-len",
        type=int,
        default=32_768,
        help="KV-cache window (prompt + output must fit). Default: 32768. "
        "Model supports up to 262144; memory cost at 256k is ~1.57 GB. "
        "Prefill is sequential (~45 ms/tok) so very long ISL is slow.",
    )
    p.add_argument("--eager", action="store_true", help="Disable traced decode (cpu_gate=True). Slower but simpler.")
    p.add_argument("--quiet", action="store_true", help="Suppress per-token progress; print summary table only.")
    args = p.parse_args()
    if args.isl and args.prompt:
        p.error("--isl and --prompt are mutually exclusive.")
    return args


if __name__ == "__main__":
    args = _parse_args()
    run_demo(
        isl_list=args.isl,
        prompt=args.prompt,
        system=args.system,
        osl=args.osl,
        eager=args.eager,
        quiet=args.quiet,
        max_seq_len=args.max_seq_len,
    )
