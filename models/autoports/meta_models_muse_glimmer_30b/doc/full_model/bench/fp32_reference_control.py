# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Is a top-100 miss the model's error, or the reference's?

The AIME24 reference is generated with HF in **bfloat16** -- the checkpoint's own
storage dtype, and what a GPU reference would use.  At a flat position, bf16
quantisation of a 202048-wide logit vector reorders the tail freely: two of the
prefill check's positions have a TT top-5 spread under 2.0 logits with a bf16
quantum of 0.0625 there.  So "the TT token is outside HF's top 100" can mean the
model is wrong, or it can mean the *reference's* ordering past rank ~5 is noise.

This control separates them.  It keeps the bf16 reference's prompt **and its
generated continuation** -- so every position is the same position -- and only
recomputes the top-k logits with HF in **float32**.  Feeding the result back
through ``run_prefill_check`` then asks: does the TT prediction land inside a
*better-ordered* reference's top 100?

Writes ``readiness_aime24_chat_fp32.refpt`` next to the bf16 one, plus
``doc/full_model/fp32_reference_control.json`` with the per-position rank of each
reference's top-1 in the other.

Usage::

    python doc/full_model/bench/fp32_reference_control.py
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import torch

ROOT = pathlib.Path(__file__).resolve().parents[3]  # models/autoports/<model>/
REPO = ROOT.parents[2]  # the tt-metal checkout
sys.path.insert(0, str(REPO))


def say(*args) -> None:
    print(*args, flush=True)


def resolve_snapshot(model_id: str) -> str:
    from huggingface_hub.constants import HF_HUB_CACHE

    repo = pathlib.Path(HF_HUB_CACHE) / f"models--{model_id.replace('/', '--')}"
    for index_path in sorted(repo.glob("snapshots/*/model.safetensors.index.json")):
        snapshot = index_path.parent
        shards = set(json.loads(index_path.read_text())["weight_map"].values())
        if all((snapshot / shard).exists() for shard in shards):
            return str(snapshot)
    raise FileNotFoundError(f"no complete snapshot of {model_id}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-model", default="meta-models/Muse-Glimmer-30B")
    parser.add_argument("--reference", default="readiness_aime24_chat.refpt")
    parser.add_argument("--output", default="readiness_aime24_chat_fp32.refpt")
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM
    from transformers.models.muse_glimmer.configuration_muse_glimmer import MuseGlimmerConfig
    from transformers.models.muse_glimmer.modeling_muse_glimmer import MuseGlimmerForConditionalGeneration

    from models.common.readiness_check.schema import Reference, ReferenceEntry, load_reference, save_reference

    AutoModelForCausalLM._model_mapping._extra_content[MuseGlimmerConfig] = MuseGlimmerForConditionalGeneration
    snapshot = resolve_snapshot(args.hf_model)

    reference = load_reference(str(ROOT / args.reference))
    entry = reference.entries[0]
    prompt_len = int(entry.tf_prompt_len)
    gen_len = int(entry.generated_tokens.shape[1])
    sequence = torch.cat([entry.prompt_tokens[0], entry.generated_tokens[0]]).unsqueeze(0)
    say(f"CTRL reference={args.reference} prompt_len={prompt_len} gen_len={gen_len} k={reference.k}")

    started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(snapshot, dtype=torch.float32, local_files_only=True).eval().to("cpu")
    say(f"CTRL fp32 model loaded in {time.perf_counter() - started:.1f}s")
    started = time.perf_counter()
    with torch.no_grad():
        logits = model(sequence).logits[0]
    say(f"CTRL fp32 forward over {sequence.shape[1]} tokens in {time.perf_counter() - started:.1f}s")
    del model

    window = logits[prompt_len - 1 : prompt_len + gen_len - 1, :].float()
    topk = torch.topk(window, k=reference.k, dim=-1).indices.to(torch.int32)

    bf16_topk = entry.topk_tokens
    rows = []
    for index in range(gen_len):
        fp32_list = topk[index].tolist()
        bf16_list = bf16_topk[index].tolist()
        rows.append(
            {
                "gen_index": index,
                "fp32_top1": int(fp32_list[0]),
                "bf16_top1": int(bf16_list[0]),
                "top1_agrees": fp32_list[0] == bf16_list[0],
                "rank_of_bf16_top1_in_fp32": fp32_list.index(bf16_list[0]) if bf16_list[0] in fp32_list else -1,
                "rank_of_fp32_top1_in_bf16": bf16_list.index(fp32_list[0]) if fp32_list[0] in bf16_list else -1,
                "jaccard_top100": len(set(fp32_list) & set(bf16_list)) / len(set(fp32_list) | set(bf16_list)),
            }
        )
    disagree = [row for row in rows if not row["top1_agrees"]]
    outside = [row for row in rows if row["rank_of_fp32_top1_in_bf16"] < 0]
    say(f"CTRL top1 disagreements between fp32 and bf16 references: {len(disagree)}/{gen_len}")
    say(f"CTRL fp32 top1 outside the bf16 top{reference.k}: {len(outside)}/{gen_len}")
    say(
        f"CTRL mean top{reference.k} Jaccard between the two references: "
        f"{sum(r['jaccard_top100'] for r in rows) / len(rows):.4f}"
    )

    save_reference(
        Reference(
            k=reference.k,
            hf_model_id=reference.hf_model_id,
            token_ids_meta=reference.token_ids_meta,
            entries=[
                ReferenceEntry(
                    prompt_text=entry.prompt_text,
                    prompt_tokens=entry.prompt_tokens,
                    generated_tokens=entry.generated_tokens,
                    topk_tokens=topk,
                    tf_prompt_len=prompt_len,
                )
            ],
        ),
        ROOT / args.output,
    )
    out = ROOT / "doc/full_model/fp32_reference_control.json"
    out.write_text(
        json.dumps(
            {
                "bf16_reference": args.reference,
                "fp32_reference": args.output,
                "k": reference.k,
                "gen_len": gen_len,
                "top1_disagreements": len(disagree),
                "fp32_top1_outside_bf16_topk": len(outside),
                "mean_topk_jaccard": sum(r["jaccard_top100"] for r in rows) / len(rows),
                "rows": rows,
            },
            indent=2,
        )
        + "\n"
    )
    say(f"CTRL wrote {ROOT / args.output} and {out}")
    say("CTRL_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
