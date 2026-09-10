#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Compare the first N tokens (default 30) of the Llama-3.2-1B-Instruct
completion produced by the two Tenstorrent inference stacks:

  - ttml   : LlamaGRPOCompleter (C++ Llama binding driven from Python)
  - worker : TttGenerationWorker on top of models.tt_transformers.tt.Transformer

Both paths generate at temperature=1.0 with a fixed seed. Each row of the
printed table is (token_id, decoded token text, probability of the sampled
token under that path's softmax). Because the two paths sample independently
(different sampler kernels + different logits numerics + different RNG
plumbing), the two token streams will diverge quickly -- that's the point.

Both paths grab the device exclusively, so this script runs each half in its
own subprocess and joins the results at the end. You can also invoke the
halves manually:

    python3 compare_llama_first30.py --mode ttml --out /tmp/ttml.json --prompt "..."
    python3 compare_llama_first30.py --mode worker --out /tmp/worker.json --prompt "..."
    python3 compare_llama_first30.py --mode print /tmp/ttml.json /tmp/worker.json

Default driver mode (no --mode) does all three sequentially.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, List, Sequence, Tuple

DEFAULT_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
DEFAULT_PROMPT = "What is 12 * 13?"
DEFAULT_MAX_TOKENS = 30
DEFAULT_TEMPERATURE = 1.0
DEFAULT_SEED = 42

_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parent


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------
def build_chat_prompt(tokenizer: Any, prompt: str) -> str:
    """Wrap ``prompt`` in a single-turn user chat template and return the
    tokenizer's rendered string (add_generation_prompt=True). Used verbatim
    on both stacks so the KV-cache prefix is identical."""
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def decode_token(tokenizer: Any, tok_id: int) -> str:
    """Best-effort per-token decode (no special-token stripping)."""
    return tokenizer.decode([tok_id], skip_special_tokens=False)


# --------------------------------------------------------------------------
# ttml path: LlamaGRPOCompleter
# --------------------------------------------------------------------------
def run_ttml(prompt: str, max_tokens: int, temperature: float, seed: int, model_source: str) -> dict:
    import numpy as np
    import torch
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Route imports through the shared examples/grpo/utils package layout.
    sys.path.insert(0, str(_REPO_ROOT / "tt-train" / "sources" / "examples" / "grpo"))

    import ttnn  # noqa: F401
    from transformers import AutoTokenizer
    from ttml.common.config import DeviceConfig, TrainingConfig, get_model_config, load_config
    from utils.llama_completer import LlamaCompletionCtx, LlamaGRPOCompleter

    tokenizer = AutoTokenizer.from_pretrained(model_source)

    # Reuse the boolq_accuracy YAML: single p150, mesh_shape [1,1], llama3_2_1B.
    cfg_path = _REPO_ROOT / "tt-train" / "sources" / "examples" / "grpo" / "boolq" / "boolq_accuracy_example.yaml"
    raw = load_config(str(cfg_path))
    training_config = TrainingConfig(raw)
    device_config = DeviceConfig(raw)
    transformer_config = get_model_config(training_config.model_config)

    completer = LlamaGRPOCompleter(
        ctx=LlamaCompletionCtx(
            max_tokens_to_complete=max_tokens,
            temperature=temperature,
            completions_per_prompt=1,
        ),
        transformer_config=transformer_config,
        device_config=device_config,
        model_source=model_source,
    )

    templated = build_chat_prompt(tokenizer, prompt)
    prompt_ids: List[int] = tokenizer.encode(templated)

    # Sample max_tokens tokens (temperature > 0 -> full sampling path).
    completions: List[List[int]] = completer.generate([prompt_ids])
    assert len(completions) == 1
    sampled_ids: List[int] = completions[0][:max_tokens]

    # Force-decode score: run compute_nlog_probs over (prompt, sampled) and
    # convert the completion-position negative log-probs to probabilities.
    nlog_tt, mask_tt = completer.compute_nlog_probs([prompt_ids], [sampled_ids])
    nlog = ttnn.to_torch(nlog_tt.get_value(), mesh_composer=completer._dp_composer).reshape(1, -1).float().numpy()[0]
    mask = ttnn.to_torch(mask_tt.get_value(), mesh_composer=completer._dp_composer).reshape(1, -1).float().numpy()[0]
    completion_nlog = nlog[mask > 0.5][: len(sampled_ids)]
    probs = [float(np.exp(-x)) for x in completion_nlog]

    tokens_text = [decode_token(tokenizer, tid) for tid in sampled_ids]

    return {
        "path": "ttml (LlamaGRPOCompleter)",
        "prompt": prompt,
        "templated_prompt": templated,
        "model_source": model_source,
        "temperature": temperature,
        "seed": seed,
        "token_ids": [int(t) for t in sampled_ids],
        "token_texts": tokens_text,
        "probs": probs,
    }


# --------------------------------------------------------------------------
# worker path: TttGenerationWorker
# --------------------------------------------------------------------------
def run_worker(prompt: str, max_tokens: int, temperature: float, seed: int, model_source: str) -> dict:
    import numpy as np
    import torch
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    sys.path.insert(0, str(_REPO_ROOT / "tt-train" / "sources" / "examples" / "grpo_remote_rollout"))

    import ttnn
    from transformers import AutoTokenizer
    from utils.llama_ttt_presets import bf16_attn_bfp8_mlp_optimizations, llama_stop_and_pad
    from utils.ttt_generation_worker import TttGenerationWorker

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D)

    tokenizer = AutoTokenizer.from_pretrained(model_source)

    parent_mesh = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(1, 1),
        offset=ttnn.MeshCoordinate(0, 0),
    )
    try:
        stop_ids, pad_id = llama_stop_and_pad(model_source)
        worker = TttGenerationWorker(
            mesh_device=parent_mesh,
            model_source=model_source,
            max_batch_size=1,
            max_seq_len=2048,
            instruct=True,
            optimizations=bf16_attn_bfp8_mlp_optimizations,
            stop_token_ids=stop_ids,
            pad_token_id=pad_id,
            temperature=temperature,
            top_k=0,
            top_p=1.0,
            seed=seed,
            dummy_weights=False,
        )

        templated = build_chat_prompt(tokenizer, prompt)
        prompt_ids: List[int] = tokenizer(templated, add_special_tokens=False)["input_ids"]

        completions, logprobs = worker.generate_and_get_log_probs(
            [prompt_ids],
            max_new_tokens=max_tokens,
            enable_trace=True,
            stop_at_eos=False,  # don't cut short: we want a full 30-token comparison
        )
        assert len(completions) == 1 and len(logprobs) == 1
        sampled_ids = [int(t) for t in completions[0][:max_tokens]]
        probs = [float(np.exp(lp)) for lp in logprobs[0][:max_tokens]]
        tokens_text = [decode_token(tokenizer, tid) for tid in sampled_ids]

        return {
            "path": "worker (TttGenerationWorker)",
            "prompt": prompt,
            "templated_prompt": templated,
            "model_source": model_source,
            "temperature": temperature,
            "seed": seed,
            "token_ids": sampled_ids,
            "token_texts": tokens_text,
            "probs": probs,
        }
    finally:
        try:
            ttnn.close_mesh_device(parent_mesh)
        except Exception:  # noqa: BLE001
            pass


# --------------------------------------------------------------------------
# Printing
# --------------------------------------------------------------------------
def _shorten(s: str, n: int = 18) -> str:
    r = repr(s)
    return r if len(r) <= n else r[: n - 1] + "…"


def print_table(results: Sequence[dict]) -> None:
    print("=" * 90)
    for r in results:
        print(f"Path        : {r['path']}")
        print(f"Model       : {r['model_source']}")
        print(f"Temperature : {r['temperature']}    Seed: {r['seed']}")
        print(f"Prompt      : {r['prompt']!r}")
        print(f"Templated   : {r['templated_prompt']!r}")
        n = min(len(r["token_ids"]), len(r["probs"]))
        header = f"  {'idx':>3}  {'tok_id':>7}  {'prob':>8}  {'text':<20}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for i in range(n):
            tid = r["token_ids"][i]
            p = r["probs"][i]
            txt = _shorten(r["token_texts"][i], 18)
            print(f"  {i:>3d}  {tid:>7d}  {p:>8.4f}  {txt:<20}")
        # Full decoded completion (joined tokens) for context.
        full = "".join(r["token_texts"][:n])
        print(f"  joined    : {full!r}")
        print("=" * 90)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _write_json(obj: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def _read_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _child_cmd(mode: str, out: str, args: argparse.Namespace) -> List[str]:
    return [
        sys.executable,
        str(_THIS_FILE),
        "--mode",
        mode,
        "--out",
        out,
        "--prompt",
        args.prompt,
        "--max_tokens",
        str(args.max_tokens),
        "--temperature",
        str(args.temperature),
        "--seed",
        str(args.seed),
        "--model",
        args.model,
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", choices=["ttml", "worker", "print", "both"], default="both")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--max_tokens", type=int, default=DEFAULT_MAX_TOKENS)
    ap.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", default=None, help="Output JSON path (for --mode ttml|worker).")
    ap.add_argument(
        "json_files",
        nargs="*",
        help="For --mode print: pass one or more JSON files produced by ttml/worker modes.",
    )
    args = ap.parse_args()

    if args.mode == "ttml":
        out = args.out or "/tmp/compare_llama_first30_ttml.json"
        result = run_ttml(args.prompt, args.max_tokens, args.temperature, args.seed, args.model)
        _write_json(result, out)
        print(f"[ttml] wrote {out}")
        return 0

    if args.mode == "worker":
        out = args.out or "/tmp/compare_llama_first30_worker.json"
        result = run_worker(args.prompt, args.max_tokens, args.temperature, args.seed, args.model)
        _write_json(result, out)
        print(f"[worker] wrote {out}")
        return 0

    if args.mode == "print":
        if not args.json_files:
            print("--mode print requires JSON file paths as positional args", file=sys.stderr)
            return 2
        print_table([_read_json(p) for p in args.json_files])
        return 0

    # both: subprocess self twice, then print.
    with tempfile.TemporaryDirectory(prefix="compare_llama_first30_") as td:
        ttml_json = os.path.join(td, "ttml.json")
        worker_json = os.path.join(td, "worker.json")

        print("[driver] running ttml subprocess...")
        rc = subprocess.call(_child_cmd("ttml", ttml_json, args))
        if rc != 0:
            print(f"[driver] ttml subprocess exited {rc}", file=sys.stderr)
            return rc

        print("[driver] running worker subprocess...")
        rc = subprocess.call(_child_cmd("worker", worker_json, args))
        if rc != 0:
            print(f"[driver] worker subprocess exited {rc}", file=sys.stderr)
            return rc

        print_table([_read_json(ttml_json), _read_json(worker_json)])
    return 0


if __name__ == "__main__":
    sys.exit(main())
