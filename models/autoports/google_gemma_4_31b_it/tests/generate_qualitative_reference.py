"""HF controls for the installed common prompt suite, correctly chat-rendered."""

import json
from pathlib import Path

import readiness_check
import torch
from transformers import AutoTokenizer

from ..tt.weights import HF_MODEL, HF_REVISION
from .generate_full_reference import load_reference


def main():
    root = Path("models/autoports/google_gemma_4_31b_it/doc/full_model")
    source = Path(readiness_check.__file__).parent / "vllm_prompts.txt"
    prompts = [p.strip() for p in source.read_text().split("\n\n") if p.strip()]
    tok = AutoTokenizer.from_pretrained(HF_MODEL, revision=HF_REVISION)
    model = load_reference()
    outputs = []
    for i, prompt in enumerate(prompts):
        messages = [{"role": "user", "content": prompt}]
        tokens = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=False)
        print("HF_QUALITATIVE", i, flush=True)
        with torch.inference_mode():
            ids = model.generate(
                torch.tensor([tokens]),
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
                attention_mask=torch.ones(1, len(tokens), dtype=torch.long),
            )
        generated = ids[0, len(tokens) :].tolist()
        outputs.append(
            {
                "id": f"shared_{i}",
                "messages": messages,
                "prompt_tokens": tokens,
                "rendered_prompt": tok.decode(tokens),
                "generated_tokens": generated,
                "completion": tok.decode(generated),
            }
        )
        (root / "qualitative_hf.json").write_text(json.dumps(outputs, indent=2) + "\n")
        print(outputs[-1]["completion"], flush=True)
    (root / "qualitative_prompt_format.json").write_text(
        json.dumps(
            dict(
                hf_model=HF_MODEL,
                hf_revision=HF_REVISION,
                tokenizer=type(tok).__name__,
                tokenizer_revision=HF_REVISION,
                chat_template_present=bool(tok.chat_template),
                prompt_mode="chat",
                rendering="apply_chat_template(add_generation_prompt=True,return_dict=False)",
                source=str(source),
                generation={"do_sample": False, "max_new_tokens": 128},
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
