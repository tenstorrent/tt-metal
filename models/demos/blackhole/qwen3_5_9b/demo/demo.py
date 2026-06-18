# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Standalone text-generation demo for the Qwen3.5 hybrid model on Blackhole P150.

Drives tt/model.py end to end — open a mesh, load a checkpoint, tokenize a prompt, greedily
generate, detokenize — for any Qwen3.5-family checkpoint (the model code is config-driven, so the
27B 3.6 multimodal checkpoint's TEXT backbone runs through the exact same path as the 9B). Defaults
target Qwen/Qwen3.6-27B on a (1,4) tensor-parallel mesh, the bring-up configuration.

Two execution modes, selectable with --mode:
  * eager  — every prefill/decode step dispatches op-by-op from host (the validated reference path).
  * trace  — prefill and the per-token decode step are each captured ONCE as a ttnn trace and then
             replayed with a single device dispatch, the path a real server uses to hide host
             dispatch latency on the (otherwise dispatch-bound) decode. See TracedRunner.
  * both   — run eager then trace and report tokens/s for each (and whether they agree token-for-token).

Run (4xP150, the default mesh):
    python models/demos/blackhole/qwen3_5_9b/demo/demo.py --mode both --max-new-tokens 64
Smoke (truncate the stack so it builds + runs fast; output is NOT coherent at <64 layers):
    python models/demos/blackhole/qwen3_5_9b/demo/demo.py --n-layers 8 --max-new-tokens 4 --mode both
"""
import argparse
import json
import time

import torch
from loguru import logger
from transformers import AutoTokenizer

import ttnn
from models.demos.blackhole.qwen3_5_9b.tt.model import Qwen35Model

# A trace bakes in op dispatch + buffer ADDRESSES, so the captured 64-layer prefill and decode graphs
# need a sizeable per-device trace region. 300 MB clears both graphs at 64 layers on the (1,4) mesh;
# if it is ever too small ttnn raises an error naming the exact size needed, which --trace-region-size
# can then set. (Eager mode allocates none of this, but opening one mesh for both modes keeps it simple.)
DEFAULT_TRACE_REGION_SIZE = 300_000_000 * 3


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen3.6-27B", help="HF hub id or local checkpoint dir (sets HF_MODEL)")
    p.add_argument("--tp", type=int, default=4, help="tensor-parallel device count; mesh is (1, tp)")
    p.add_argument("--prompt", default="Give me a short introduction to large language models.")
    p.add_argument(
        "--prompt-file",
        default=None,
        help="file of long context text; --prompt becomes the instruction appended after it",
    )
    p.add_argument(
        "--max-prompt-tokens",
        type=int,
        default=None,
        help="truncate the loaded context to this many tokens (e.g. 10000)",
    )
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument(
        "--max-seq-len", type=int, default=2048, help="KV-cache / RoPE length; must cover prompt + new tokens"
    )
    p.add_argument("--mode", choices=["eager", "trace", "both"], default="both")
    p.add_argument("--n-layers", type=int, default=None, help="truncate the decoder stack (smoke tests only)")
    p.add_argument("--raw", action="store_true", help="encode the prompt verbatim instead of the chat template")
    p.add_argument("--trace-region-size", type=int, default=DEFAULT_TRACE_REGION_SIZE)
    p.add_argument("--no-warmup", dest="warmup", action="store_false", help="skip the JIT-compile warmup pass")
    return p.parse_args()


# ── Mesh open/close (standalone, no pytest fixture) ─────────────────────────────────────
def open_mesh(tp, trace_region_size):
    """Open a (1, tp) mesh, enabling the 1D fabric the TP reduce-scatters/all-gathers ride.

    Mirrors the conftest mesh_device fixture for a script: set_fabric_config MUST precede
    open_mesh_device (the CCL the model builds binds to the fabric at mesh-open time). On a single
    device (tp=1) the fabric is left disabled — every CCL op in the model short-circuits to a no-op.
    """
    if tp > 1:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D, ttnn.FabricReliabilityMode.STRICT_INIT)
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, tp), trace_region_size=trace_region_size)
    logger.info(f"Opened mesh {tuple(mesh.shape)} ({mesh.get_num_devices()} devices), trace_region={trace_region_size}")
    return mesh


def close_mesh(mesh, tp):
    ttnn.close_mesh_device(mesh)
    if tp > 1:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


# ── Prompt / tokenizer ──────────────────────────────────────────────────────────────────
def load_prompt_text(path):
    """Read long-context text from a file: a sample_prompts JSON list/dict with a 'prompt' (or
    'context') field, else the raw file contents. Lets the demo summarize a real document."""
    raw = open(path).read()
    try:
        d = json.loads(raw)
        if isinstance(d, list) and d and isinstance(d[0], dict):
            return d[0].get("prompt") or d[0].get("context") or raw
        if isinstance(d, dict):
            return d.get("prompt") or d.get("context") or raw
    except json.JSONDecodeError:
        pass
    return raw


def encode_prompt(tokenizer, prompt, raw, prompt_file=None, max_prompt_tokens=None):
    """Prompt text -> token-id row [1, S]. Uses the chat template for the instruct checkpoint by
    default (coherent generation); --raw encodes verbatim. Falls back to verbatim if the checkpoint
    ships no chat template.

    With prompt_file, the file's text becomes the CONTEXT and `prompt` the instruction appended after
    it ("<document>\\n\\n<instruction>") — the summarize-a-document shape. max_prompt_tokens caps the
    context length (tokenize → truncate → detokenize) so a 16k doc can be cut to a clean ~10k.
    """
    if prompt_file:
        ctx = load_prompt_text(prompt_file)
        if max_prompt_tokens:
            ids = tokenizer(ctx, return_tensors="pt").input_ids[0][:max_prompt_tokens]
            ctx = tokenizer.decode(ids, skip_special_tokens=True)
        prompt = f"{ctx}\n\n{prompt}"
    if not raw and tokenizer.chat_template:
        # return_dict=True keeps the result a BatchEncoding across transformers versions (some return
        # a bare tensor, some a dict, when return_tensors is set) — pull input_ids out explicitly.
        enc = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True, return_tensors="pt", return_dict=True
        )
        return enc["input_ids"].to(torch.int32)
    return tokenizer(prompt, return_tensors="pt").input_ids.to(torch.int32)


def pad_prompt(prompt_ids, max_seq_len, max_new_tokens):
    """Right-pad the prompt to a 32-multiple (the GDN chunk kernel needs a tile-aligned seq) and
    return (padded [1, T_pad], valid_len, T_pad). Identical to Qwen35Model.generate's padding — the
    next-token logit is read at valid_len-1, which trailing pad tokens cannot influence (causal
    attention + lower-triangular GDN), and decode then continues from T_pad. See model.generate."""
    prompt = torch.as_tensor(prompt_ids, dtype=torch.int32).reshape(1, -1)
    valid_len = prompt.shape[1]
    pad = (-valid_len) % 32
    if pad:
        prompt = torch.cat([prompt, torch.zeros(1, pad, dtype=torch.int32)], dim=1)
    seq_len = prompt.shape[1]
    assert (
        seq_len + max_new_tokens <= max_seq_len
    ), f"padded prompt ({seq_len}) + new tokens ({max_new_tokens}) exceeds max_seq_len ({max_seq_len})"
    return prompt, valid_len, seq_len


# ── Eager generation (timed) ────────────────────────────────────────────────────────────
def run_eager(model, prompt_ids, max_new_tokens, eos_token_id):
    """Greedy generation via the model's public prefill/decode, timed prefill (TTFT) vs decode (tok/s).

    This is Qwen35Model.generate inlined so prefill and the decode loop can be timed separately.
    prefill() returns host logits and decode() returns the on-device argmax id; both force a device
    sync on read, so the wall-clock around them is honest end-to-end latency. decode() reads back only
    the id (not the full vocab), which is where the per-token readback cost went.

    Returns (tokens, ttft_s, prefill_tok_s, decode_tok_s). prefill_tok_s = seq_len / ttft is the prompt
    INGESTION rate over the padded length actually pushed through the model (all positions run in one
    shot), the throughput companion to TTFT (the latency).
    """
    prompt, valid_len, seq_len = pad_prompt(prompt_ids, model.args.max_seq_len, max_new_tokens)
    model.reset_state()

    t0 = time.time()
    logits = model.prefill(prompt, valid_len=valid_len)
    ttft = time.time() - t0
    prefill_tok_s = seq_len / ttft

    next_id = int(torch.argmax(logits).item())
    out = [next_id]
    cur_pos = seq_len
    t0 = time.time()
    for _ in range(max_new_tokens - 1):
        if eos_token_id is not None and next_id == eos_token_id:
            break
        # decode argmaxes on device and reads back only the id (no full-vocab readback per step).
        next_id = int(
            model.decode(torch.tensor([[next_id]], dtype=torch.int32), torch.tensor([cur_pos], dtype=torch.int32))[0]
        )
        out.append(next_id)
        cur_pos += 1
    decode_s = time.time() - t0
    tok_s = (len(out) - 1) / decode_s if len(out) > 1 else float("nan")
    return out, ttft, prefill_tok_s, tok_s


# ── main ────────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    mesh = open_mesh(args.tp, args.trace_region_size)
    try:
        logger.info(f"Loading {args.model} (TP={args.tp}) — this loads the checkpoint to host then shards to device...")
        model = Qwen35Model.from_pretrained(
            mesh, max_batch_size=1, max_seq_len=args.max_seq_len, n_layers=args.n_layers, hf_model=args.model
        )
        tokenizer = AutoTokenizer.from_pretrained(model.args.CKPT_DIR, trust_remote_code=True)
        eos = tokenizer.eos_token_id

        input_ids = encode_prompt(tokenizer, args.prompt, args.raw, args.prompt_file, args.max_prompt_tokens)
        logger.info(f"Prompt is {input_ids.shape[1]} tokens; generating {args.max_new_tokens}")

        # Warm up the kernel JIT (prefill + one decode at the timed runs' shapes) so the reported
        # eager/trace numbers reflect steady state, not first-run compilation. Without this, whichever
        # mode runs first eats all the compile cost and the comparison is meaningless.
        if args.warmup:
            logger.info("Warming up (compiling kernels)...")
            run_eager(model, input_ids, max_new_tokens=2, eos_token_id=None)

        results = {}
        if args.mode in ("eager", "both"):
            toks, ttft, prefill_tok_s, tok_s = run_eager(model, input_ids, args.max_new_tokens, eos)
            results["eager"] = toks
            logger.info(f"[eager]  prefill: TTFT={ttft:.3f}s ({prefill_tok_s:.1f} tok/s)  decode: {tok_s:.2f} tok/s")
            print(f"\n=== EAGER ===\n{tokenizer.decode(toks, skip_special_tokens=True)}\n")

        if args.mode in ("trace", "both"):
            from models.demos.blackhole.qwen3_5_9b.demo.trace_runner import TracedRunner

            toks, ttft, prefill_tok_s, tok_s, capture_s = TracedRunner(model).generate(
                input_ids, args.max_new_tokens, eos
            )
            results["trace"] = toks
            logger.info(
                f"[trace]  prefill: TTFT={ttft:.3f}s ({prefill_tok_s:.1f} tok/s)  decode: {tok_s:.2f} tok/s"
                f"  (one-time capture={capture_s:.2f}s)"
            )
            print(f"\n=== TRACE ===\n{tokenizer.decode(toks, skip_special_tokens=True)}\n")

        if args.mode == "both":
            n = min(len(results["eager"]), len(results["trace"]))
            matches = sum(a == b for a, b in zip(results["eager"][:n], results["trace"][:n]))
            logger.info(f"eager vs trace token agreement: {matches}/{n}")
    finally:
        close_mesh(mesh, args.tp)


if __name__ == "__main__":
    main()
