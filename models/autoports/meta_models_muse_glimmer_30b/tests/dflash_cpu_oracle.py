# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end DFlash oracle on CPU, with real target + real drafter weights.

This exists to answer, *before* any TTNN work, the two questions that decide
whether the port is worth building and what "correct" means for it:

1. **Is DFlash output-lossless?**  Speculative decoding is supposed to return
   exactly what greedy decoding would.  We run both and assert token equality.
   If they diverge here, on the reference implementation, then any divergence
   later is not a device bug and we would be chasing the wrong thing.

2. **What is the acceptance rate?**  The whole speedup is
   ``accepted_tokens_per_target_forward``.  The block is 16 wide (1 anchor + 15
   drafted), so the ceiling is 16 tokens per target forward.  The realised rate
   is what a device implementation can hope to approach, and it bounds the
   achievable t/s/u before a single kernel is written.

Run it with a log; the target forward on CPU is slow and this prints per-step.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer
from transformers.models.muse_glimmer.modeling_muse_glimmer import MuseGlimmerForConditionalGeneration
from transformers.models.muse_glimmer_assistant.modeling_muse_glimmer_assistant import MuseGlimmerAssistantModel

TARGET_MODEL_ID = "meta-models/Muse-Glimmer-30B"
DRAFT_MODEL_ID = "meta-models/Muse-Glimmer-30B-assistant"


class ForwardCounter:
    """Counts real target forwards so we can derive tokens-per-forward."""

    def __init__(self, model):
        self.count = 0
        self.query_lens: list[int] = []
        self._handle = model.register_forward_pre_hook(self._hook, with_kwargs=True)

    def _hook(self, _module, _args, kwargs):
        self.count += 1
        ids = kwargs.get("input_ids")
        if ids is not None:
            self.query_lens.append(int(ids.shape[-1]))
        return None

    def close(self):
        self._handle.remove()


def _snapshot_with_weights(model_id: str) -> Path:
    """Resolve the snapshot that actually holds weights.

    ``refs/main`` for the target repo points at a **metadata-only** revision, so
    ``from_pretrained(model_id)`` raises "does not appear to have a file named
    model.safetensors".  The target port hits the same trap and resolves it the
    same way: find the snapshot containing weights rather than trusting the
    default revision.
    """
    from huggingface_hub.constants import HF_HUB_CACHE

    repo = Path(HF_HUB_CACHE) / f"models--{model_id.replace('/', '--')}"
    for pattern in ("snapshots/*/model.safetensors.index.json", "snapshots/*/model.safetensors"):
        candidates = sorted(repo.glob(pattern))
        if candidates:
            return candidates[0].parent
    raise FileNotFoundError(f"no cached weights for {model_id} under {repo}")


def load_models(dtype=torch.bfloat16):
    target_dir = _snapshot_with_weights(TARGET_MODEL_ID)
    draft_dir = _snapshot_with_weights(DRAFT_MODEL_ID)
    print(f"target snapshot: {target_dir.name}\ndraft  snapshot: {draft_dir.name}", flush=True)

    print("loading target ...", flush=True)
    t0 = time.time()
    target = MuseGlimmerForConditionalGeneration.from_pretrained(
        str(target_dir), dtype=dtype, local_files_only=True, device_map="cpu"
    ).eval()
    print(f"  target loaded in {time.time() - t0:.1f}s", flush=True)

    print("loading drafter ...", flush=True)
    t0 = time.time()
    drafter = MuseGlimmerAssistantModel.from_pretrained(
        str(draft_dir), dtype=dtype, local_files_only=True, device_map="cpu"
    ).eval()
    print(f"  drafter loaded in {time.time() - t0:.1f}s", flush=True)
    return target, drafter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--prompt", default="Write a Python function that merges two sorted lists.")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(TARGET_MODEL_ID, local_files_only=True)
    target, drafter = load_models()

    messages = [{"role": "user", "content": args.prompt}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt")
    print(f"prompt tokens: {inputs['input_ids'].shape[-1]}", flush=True)

    # ---- Baseline: plain greedy -------------------------------------------------
    print("\n=== baseline greedy (no speculation) ===", flush=True)
    counter = ForwardCounter(target)
    t0 = time.time()
    with torch.no_grad():
        baseline = target.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False, use_cache=True)
    baseline_time = time.time() - t0
    baseline_forwards = counter.count
    counter.close()
    baseline_new = baseline[0, inputs["input_ids"].shape[-1] :]
    print(
        f"  {baseline_new.shape[0]} tokens in {baseline_time:.1f}s "
        f"({baseline_forwards} target forwards, {baseline_time / max(baseline_new.shape[0], 1):.2f}s/token)",
        flush=True,
    )

    # ---- DFlash speculative -----------------------------------------------------
    print("\n=== DFlash speculative ===", flush=True)
    counter = ForwardCounter(target)
    t0 = time.time()
    with torch.no_grad():
        spec = target.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_cache=True,
            assistant_model=drafter,
            speculation_type="dflash",
        )
    spec_time = time.time() - t0
    spec_forwards = counter.count
    query_lens = counter.query_lens
    counter.close()
    spec_new = spec[0, inputs["input_ids"].shape[-1] :]
    print(
        f"  {spec_new.shape[0]} tokens in {spec_time:.1f}s "
        f"({spec_forwards} target forwards, {spec_time / max(spec_new.shape[0], 1):.2f}s/token)",
        flush=True,
    )

    # ---- The two questions ------------------------------------------------------
    n = min(baseline_new.shape[0], spec_new.shape[0])
    identical = bool(torch.equal(baseline_new[:n], spec_new[:n]))
    # Exclude the prefill forward from the tokens-per-forward accounting.
    decode_forwards = max(spec_forwards - 1, 1)
    accept_rate = spec_new.shape[0] / decode_forwards

    print("\n" + "=" * 66)
    print(f"output-lossless (greedy token-identical over {n} tokens): {identical}")
    if not identical:
        diff = (baseline_new[:n] != spec_new[:n]).nonzero().flatten().tolist()
        print(f"  FIRST DIVERGENCE at index {diff[0]}")
        print(f"  baseline: {baseline_new[:n].tolist()}")
        print(f"  dflash  : {spec_new[:n].tolist()}")
    print(f"accepted tokens per target forward: {accept_rate:.2f}  (ceiling 16)")
    print(f"target forwards: baseline {baseline_forwards} -> dflash {spec_forwards}")
    print(f"forward reduction: {baseline_forwards / max(spec_forwards, 1):.2f}x")
    print("=" * 66)
    print("\ntext:\n" + tok.decode(spec_new, skip_special_tokens=True))

    result = {
        "prompt": args.prompt,
        "max_new_tokens": args.max_new_tokens,
        "prompt_tokens": int(inputs["input_ids"].shape[-1]),
        "output_lossless": identical,
        "tokens_compared": int(n),
        "baseline_tokens": int(baseline_new.shape[0]),
        "baseline_target_forwards": baseline_forwards,
        "baseline_seconds": baseline_time,
        "dflash_tokens": int(spec_new.shape[0]),
        "dflash_target_forwards": spec_forwards,
        "dflash_seconds": spec_time,
        "accepted_tokens_per_target_forward": accept_rate,
        "forward_reduction": baseline_forwards / max(spec_forwards, 1),
        "dflash_query_lens": query_lens,
    }
    out = Path(args.out) if args.out else Path(__file__).with_name("dflash_cpu_oracle.json")
    out.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
