# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end DFlash reference check for Gemma4-31B: real target + real drafter weights,
real prompt, real speculative generation via the vendored upstream ``dflash_generate``.

Target (verifier):  google/gemma-4-31B-it        (already cached locally)
Drafter:             z-lab/gemma-4-31B-it-DFlash  (downloaded on first run, small)

This is a pure-torch/HF check -- no ttnn, no Tenstorrent hardware. It exists to prove the
architecture/weights/generation loop are wired correctly against the REAL checkpoints before
any ttnn porting starts (mirrors models/demos/blackhole/qwen36/reference/mtp_e2e_check.py's
role for the Qwen3.6 MTP port on this same branch's sibling work).

No GPU is available in this environment (torch.cuda.is_available() == False here), so this
runs on CPU. A dense 31B-parameter target in bf16 is ~62GB of weights; expect prefill and
each verify-block forward pass to take real wall-clock time (minutes, not seconds) -- keep
--max-new-tokens small for a smoke check, not a throughput measurement.

REQUIRES a separate venv pinned to transformers==5.15.0 -- the shared tt-metal python_env's
transformers==5.12.1 is missing DynamicCache.activate_past_recording(), which dflash_generate
calls unconditionally. See README.md in this directory for exact setup commands and why.

Usage:
    python -m models.demos.gemma4.reference.dflash.dflash_e2e_check \
        --max-new-tokens 32 --prompt "Write a one-line Python function that reverses a string."
"""

from __future__ import annotations

import argparse
import time

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

from models.demos.gemma4.reference.dflash import dflash as dflash_module
from models.demos.gemma4.reference.dflash.dflash import DFlash2DraftModel, DFlashDraftModel, dflash_generate

TARGET_ID = "google/gemma-4-31B-it"
DRAFT_ID = "z-lab/gemma-4-31B-it-DFlash"

if not torch.cuda.is_available():
    # Upstream's _cuda_time() calls torch.cuda.synchronize() unconditionally for return_stats=True
    # timing -- crashes with no CUDA device. Patched here (not in the vendored file, to keep it a
    # verbatim copy) rather than dropping return_stats, since the acceptance-length/timing stats are
    # exactly what this check needs to report.
    dflash_module._cuda_time = time.perf_counter


def load_models(device: torch.device):
    target_kwargs = {"attn_implementation": "sdpa", "dtype": torch.bfloat16}
    try:
        target = AutoModelForCausalLM.from_pretrained(TARGET_ID, **target_kwargs)
    except ValueError:
        target = AutoModelForImageTextToText.from_pretrained(TARGET_ID, **target_kwargs)
    target = target.to(device).eval()

    draft_config = AutoConfig.from_pretrained(DRAFT_ID)
    draft_cls = DFlash2DraftModel if "DFlash2DraftModel" in (draft_config.architectures or []) else DFlashDraftModel
    draft = draft_cls.from_pretrained(DRAFT_ID, attn_implementation="sdpa", dtype=torch.bfloat16).to(device).eval()

    tokenizer = AutoTokenizer.from_pretrained(TARGET_ID)
    return target, draft, tokenizer


def stop_token_ids(target, tokenizer) -> list[int]:
    ids = target.generation_config.eos_token_id or tokenizer.eos_token_id
    return [ids] if isinstance(ids, int) else list(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="Write a one-line Python function that reverses a string.")
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f"Loading {TARGET_ID} (target) + {DRAFT_ID} (drafter) on {device} ...", flush=True)
    t0 = time.time()
    target, draft, tokenizer = load_models(device)
    print(
        f"Loaded in {time.time() - t0:.1f}s. Drafter: {draft.config.num_hidden_layers} layers, "
        f"block_size={draft.block_size}, target_layer_ids={draft.target_layer_ids}",
        flush=True,
    )

    messages = [{"role": "user", "content": args.prompt}]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer.encode(prompt_text, return_tensors="pt", add_special_tokens=False).to(device)
    print(f"Prompt ({input_ids.shape[1]} tokens): {args.prompt!r}", flush=True)

    t0 = time.time()
    result = dflash_generate(
        draft,
        target,
        input_ids,
        max_new_tokens=args.max_new_tokens,
        stop_token_ids=stop_token_ids(target, tokenizer),
        temperature=0.0,  # greedy: argmax-match acceptance, deterministic, matches this session's
        return_stats=True,  # convention for validating against a real target model
    )
    wall = time.time() - t0

    text = tokenizer.decode(result.output_ids[0, input_ids.shape[1] :], skip_special_tokens=True)
    print("=" * 70)
    print(
        f"GENERATED ({result.num_output_tokens} tokens in {wall:.1f}s, "
        f"{result.num_output_tokens / wall:.2f} tok/s overall):"
    )
    print(text)
    print("-" * 70)
    print(
        f"time_to_first_token={result.time_to_first_token:.2f}s  "
        f"time_per_output_token={result.time_per_output_token * 1000:.1f}ms"
    )
    accepts = result.acceptance_lengths
    print(
        f"block_size={draft.block_size}  iterations={len(accepts)}  "
        f"tokens/iteration: {accepts}  avg={sum(accepts) / max(1, len(accepts)):.2f}"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
