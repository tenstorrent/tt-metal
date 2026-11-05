# SPDX-License-Identifier: Apache-2.0
"""CPU HF next-token control at the served thermodynamics language switch."""

import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from ..tt.weights import HF_MODEL, HF_REVISION
from .generate_full_reference import load_reference


def main():
    root = Path(__file__).parents[1]
    source = root / "readiness_vllm/vllm_qualitative_outputs.json"
    rows = json.loads(source.read_text())
    entry = next(row for row in rows if "英語" in row.get("sampled_completion", ""))
    completion = entry["sampled_completion"]
    offset = completion.index("英語")
    prefix_text = completion[:offset]
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL, revision=HF_REVISION, local_files_only=True)
    prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": entry["prompt"]}],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=False,
    )
    prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
    candidates = {}
    for suffix in ("英語", " English", "English", " words", " language"):
        together = tokenizer.encode(prefix_text + suffix, add_special_tokens=False)
        assert together[: len(prefix_ids)] == prefix_ids, f"Tokenizer changed prefix for {suffix!r}"
        continuation = together[len(prefix_ids) :]
        assert continuation
        candidates[suffix] = continuation
    print(
        "HF_LANGUAGE_PROBE",
        "prompt_tokens",
        len(prompt_ids),
        "completion_prefix_tokens",
        len(prefix_ids),
        "candidates",
        candidates,
        flush=True,
    )
    model = load_reference()
    ids = torch.tensor([prompt_ids + prefix_ids], dtype=torch.long)
    assert next(model.parameters()).device.type == "cpu"
    with torch.inference_mode():
        logits = (
            model(ids, attention_mask=torch.ones_like(ids), logits_to_keep=1, use_cache=False).logits[0, -1].float()
        )
    temperature, top_k, top_p = 0.7, 32, 0.9
    sorted_logits, sorted_ids = logits.sort(descending=True)
    top_values, top_ids = sorted_logits[:top_k], sorted_ids[:top_k]
    before_nucleus = torch.softmax(top_values / temperature, dim=0)
    keep = before_nucleus.cumsum(0) - before_nucleus < top_p
    final_probs = before_nucleus * keep
    final_probs /= final_probs.sum()
    full_probs = torch.softmax(logits / temperature, dim=0)

    def describe(token):
        token = int(token)
        rank = int((logits > logits[token]).sum()) + 1
        matches = (top_ids == token).nonzero().flatten()
        index = int(matches[0]) if len(matches) else None
        return dict(
            token_id=token,
            token_text=tokenizer.decode([token]),
            rank=rank,
            logit=float(logits[token]),
            temperature_probability=float(full_probs[token]),
            top32_probability=float(before_nucleus[index]) if index is not None else 0.0,
            top32_nucleus_probability=float(final_probs[index]) if index is not None else 0.0,
        )

    report = dict(
        scope="full60 pinned HF CPU next-token probability at reconstructed served prefix",
        hf_model=HF_MODEL,
        hf_revision=HF_REVISION,
        tokenizer_revision=HF_REVISION,
        source=str(source),
        prompt=entry["prompt"],
        prompt_mode="chat",
        prompt_token_count=len(prompt_ids),
        completion_prefix_token_count=len(prefix_ids),
        completion_character_offset=offset,
        prefix_tail=prefix_text[-180:],
        prompt_token_ids=prompt_ids,
        completion_prefix_token_ids=prefix_ids,
        sampling=dict(temperature=temperature, top_p=top_p, top_k=top_k),
        candidates={
            text: dict(continuation_ids=tokens, first_token=describe(tokens[0])) for text, tokens in candidates.items()
        },
        top32=[describe(token) for token in top_ids],
        limitations=[
            "Saved served artifact contains text, not raw token IDs; completion prefix is retokenized and is not proven identical to runtime tokens.",
            "One conditional HF distribution cannot establish TT logit fidelity or rule out a serving defect.",
            "The sampled suite seed and route must be established separately; no benchmark or device profiling performed.",
        ],
    )
    output = root / "readiness_vllm/qualitative_language_hf_probe.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {"output": str(output), "target": report["candidates"]["英語"], "top": report["top32"][:5]},
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
