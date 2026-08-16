# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Why does DFlash diverge from sequential greedy, when it provably shouldn't?

``dflash_cpu_oracle.py`` found the reference DFlash implementation producing a
different token at index 39 of 48 versus plain greedy decoding, on identical
weights and greedy sampling.  That should be impossible by construction: a token
is only accepted when it equals the target model's own argmax, so the accepted
sequence should be exactly the greedy sequence.

There are two candidate explanations and they have very different consequences:

**(A) A logic bug** in the accept/reject path — then the TTNN port must not copy
it, and the correctness gate is "token-identical to greedy".

**(B) Target-model numerics.** Baseline greedy computes logits with a
``query_len == 1`` forward; DFlash verifies with a ``query_len == 16`` forward.
In bfloat16 those are different reduction orders through the same weights, so
near-tied logits can argmax differently.  Then divergence is a property of the
*target model*, not of DFlash, the "lossless" claim is only true in exact
arithmetic, and the port's correctness gate must be distributional rather than
token-identical.

This probe distinguishes them **without involving the drafter at all**: take the
greedy sequence, re-score it with one wide teacher-forced forward, and compare
per-position argmax against what incremental decoding produced.  Any mismatch
here is (B), because there is no speculation anywhere in this experiment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer
from transformers.models.muse_glimmer.modeling_muse_glimmer import MuseGlimmerForConditionalGeneration

TARGET_MODEL_ID = "meta-models/Muse-Glimmer-30B"


def _snapshot_with_weights(model_id: str) -> Path:
    from huggingface_hub.constants import HF_HUB_CACHE

    repo = Path(HF_HUB_CACHE) / f"models--{model_id.replace('/', '--')}"
    for pattern in ("snapshots/*/model.safetensors.index.json", "snapshots/*/model.safetensors"):
        candidates = sorted(repo.glob(pattern))
        if candidates:
            return candidates[0].parent
    raise FileNotFoundError(model_id)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--prompt", default="Write a Python function that merges two sorted lists.")
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(TARGET_MODEL_ID, local_files_only=True)
    target = MuseGlimmerForConditionalGeneration.from_pretrained(
        str(_snapshot_with_weights(TARGET_MODEL_ID)), dtype=torch.bfloat16, local_files_only=True, device_map="cpu"
    ).eval()

    text = tok.apply_chat_template(
        [{"role": "user", "content": args.prompt}], tokenize=False, add_generation_prompt=True
    )
    inputs = tok(text, return_tensors="pt")
    prompt_len = inputs["input_ids"].shape[-1]

    print("incremental greedy ...", flush=True)
    with torch.no_grad():
        greedy = target.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False, use_cache=True)
    seq = greedy[0]
    new_tokens = seq[prompt_len:]
    print(f"  {new_tokens.shape[0]} tokens", flush=True)

    # Teacher-force the identical sequence in ONE wide forward. No cache, no speculation.
    print("teacher-forced re-score (single wide forward) ...", flush=True)
    with torch.no_grad():
        out = target(input_ids=seq.unsqueeze(0), use_cache=False)
    logits = out.logits[0].float()

    # Position p's logits predict token p+1.
    mismatches = []
    for i in range(new_tokens.shape[0]):
        pos = prompt_len + i - 1
        row = logits[pos]
        top2 = torch.topk(row, 2)
        predicted = int(top2.indices[0])
        actual = int(new_tokens[i])
        gap = float(top2.values[0] - top2.values[1])
        if predicted != actual:
            mismatches.append(
                {
                    "index": i,
                    "incremental_token": actual,
                    "teacher_forced_token": predicted,
                    "top2_gap": gap,
                    "incremental_token_rank": int((row > row[actual]).sum()),
                    "logit_delta": float(row[predicted] - row[actual]),
                }
            )

    print("\n" + "=" * 70)
    print(f"positions compared: {new_tokens.shape[0]}")
    print(f"argmax mismatches between incremental and teacher-forced: {len(mismatches)}")
    for m in mismatches:
        print(
            f"  idx {m['index']:3d}: incremental {m['incremental_token']:7d} vs wide-forward "
            f"{m['teacher_forced_token']:7d} | top2 gap {m['top2_gap']:.5f} | "
            f"logit delta {m['logit_delta']:.5f} | incr rank {m['incremental_token_rank']}"
        )
    print("=" * 70)
    if mismatches:
        print(
            "\nVERDICT (B): the target model's own argmax depends on forward width in bf16.\n"
            "DFlash divergence is inherited from the target, not caused by speculation.\n"
            "A token-identical gate is therefore not achievable and not the right gate."
        )
    else:
        print(
            "\nVERDICT (A): the target is width-stable here, so the DFlash divergence\n"
            "comes from the accept/reject path itself and must be treated as a bug."
        )

    Path(__file__).with_name("dflash_divergence_probe.json").write_text(
        json.dumps(
            {
                "prompt": args.prompt,
                "positions_compared": int(new_tokens.shape[0]),
                "mismatch_count": len(mismatches),
                "mismatches": mismatches,
                "verdict": "B_target_numerics" if mismatches else "A_accept_logic",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
