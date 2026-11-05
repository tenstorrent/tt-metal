# SPDX-License-Identifier: Apache-2.0
"""CPU HF controls for the exact raw-completion prompts in sampling tests."""

import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, LogitsProcessor, LogitsProcessorList

from ..tt.weights import HF_MODEL, HF_REVISION
from .generate_full_reference import load_reference


class OutputFrequencyPenalty(LogitsProcessor):
    def __init__(self, prompt_length, penalty):
        self.prompt_length, self.penalty = prompt_length, penalty

    def __call__(self, input_ids, scores):
        for row in range(input_ids.shape[0]):
            counts = torch.bincount(input_ids[row, self.prompt_length :], minlength=scores.shape[-1])
            scores[row] -= self.penalty * counts
        return scores


def main():
    root = Path(__file__).parents[1]
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL, revision=HF_REVISION, local_files_only=True)
    model = load_reference()
    assert next(model.parameters()).device.type == "cpu"
    cases = [
        dict(
            name="frequency_raw_0",
            prompt="a a a a a a a a a",
            temperature=0.0,
            frequency_penalty=0.0,
            cap=15,
            chat=False,
        ),
        dict(
            name="frequency_raw_2",
            prompt="a a a a a a a a a",
            temperature=0.0,
            frequency_penalty=2.0,
            cap=15,
            chat=False,
        ),
        dict(
            name="frequency_chat_0",
            prompt="a a a a a a a a a",
            temperature=0.0,
            frequency_penalty=0.0,
            cap=15,
            chat=True,
        ),
        *[
            dict(
                name=f"temperature_raw_seed{seed}",
                prompt="Random letter: ",
                temperature=2.0,
                frequency_penalty=0.0,
                cap=10,
                chat=False,
                seed=seed,
            )
            for seed in (42, 43, 44)
        ],
    ]
    results = []
    output = root / "readiness_vllm/raw_prompt_hf_controls.json"
    for case in cases:
        tokens = (
            tokenizer.apply_chat_template(
                [{"role": "user", "content": case["prompt"]}],
                tokenize=True,
                add_generation_prompt=True,
                return_dict=False,
            )
            if case["chat"]
            else tokenizer.encode(case["prompt"], add_special_tokens=True)
        )
        inputs = torch.tensor([tokens])
        seed = case.get("seed", 42)
        torch.manual_seed(seed)
        kwargs = dict(
            max_new_tokens=case["cap"],
            do_sample=case["temperature"] > 0,
            pad_token_id=tokenizer.pad_token_id,
            attention_mask=torch.ones_like(inputs),
        )
        if case["temperature"]:
            kwargs.update(temperature=case["temperature"], top_k=32, top_p=1.0)
        if case["frequency_penalty"]:
            kwargs["logits_processor"] = LogitsProcessorList(
                [OutputFrequencyPenalty(len(tokens), case["frequency_penalty"])]
            )
        print("HF_RAW_CONTROL", case["name"], flush=True)
        with torch.inference_mode():
            generated = model.generate(inputs, **kwargs)[0, len(tokens) :].tolist()
        results.append(
            dict(
                case,
                seed=seed,
                prompt_ids=tokens,
                output_ids=generated,
                text=tokenizer.decode(generated, skip_special_tokens=True),
                text_with_specials=tokenizer.decode(generated),
                first_token_id=generated[0] if generated else None,
            )
        )
        report = dict(
            scope="full60 pinned HF CPU controls, not serving performance",
            hf_revision=HF_REVISION,
            tokenizer_revision=HF_REVISION,
            raw_add_special_tokens=True,
            default_source="vllm/entrypoints/openai/completion/protocol.py CompletionRequest.add_special_tokens=True; stock test utils omit override",
            frequency_penalty_formula="subtract penalty * generated-token count; prompt tokens excluded",
            results=results,
            limitations=[
                "Sampled HF seed42/43/44 are controls, not the unknown unseeded serving draws.",
                "Raw instruct-model prompts are compared separately from chat rendering; neither format exempts wrong serving behavior.",
            ],
        )
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        print(json.dumps(results[-1], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
