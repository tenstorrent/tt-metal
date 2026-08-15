import argparse
import json
from pathlib import Path

import openai
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="Qwen/Qwen3.6-27B")
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()

    prompts = [item.strip() for item in args.prompts.read_text().split("\n\n") if item.strip()]
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    client = openai.OpenAI(base_url=args.base_url, api_key="dummy")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    outputs_path = args.output_dir / "vllm_chat_qualitative_outputs.json"
    outputs = json.loads(outputs_path.read_text()) if outputs_path.exists() else []
    completed = {item["prompt"] for item in outputs}
    metadata = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        token_ids = tokenizer.encode(rendered, add_special_tokens=False)
        if prompt in completed:
            metadata.append(
                {
                    "prompt": prompt,
                    "messages": messages,
                    "rendered_prompt": rendered,
                    "token_ids": token_ids,
                    "num_prompt_tokens": len(token_ids),
                }
            )
            continue
        greedy = client.chat.completions.create(
            model=args.model,
            messages=messages,
            max_tokens=args.max_tokens,
            temperature=0.0,
        )
        sampled = client.chat.completions.create(
            model=args.model,
            messages=messages,
            max_tokens=args.max_tokens,
            temperature=0.7,
            top_p=0.9,
            seed=20260815,
        )
        outputs.append(
            {
                "prompt": prompt,
                "greedy_completion": greedy.choices[0].message.content,
                "sampled_completion": sampled.choices[0].message.content,
                "sampled_seed": 20260815,
            }
        )
        metadata.append(
            {
                "prompt": prompt,
                "messages": messages,
                "rendered_prompt": rendered,
                "token_ids": token_ids,
                "num_prompt_tokens": len(token_ids),
            }
        )

    outputs_path.write_text(json.dumps(outputs, indent=2))
    (args.output_dir / "vllm_chat_prompt_metadata.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "snapshot": str(tokenizer.name_or_path),
                "tokenizer_class": type(tokenizer).__name__,
                "chat_template_present": bool(tokenizer.chat_template),
                "rendering_method": "tokenizer.apply_chat_template(add_generation_prompt=True); live /v1/chat/completions",
                "sampled_seed": 20260815,
                "cases": metadata,
                "controls": [
                    "doc/datatype_sweep/artifacts/selected_qualitative_shared_suite.json",
                    "doc/full_model/artifacts/full_model_qualitative_200_final.json",
                    "doc/vllm_integration/readiness chat evidence",
                ],
            },
            indent=2,
        )
    )
    print(f"CHAT_QUALITATIVE_OK cases={len(outputs)}")


if __name__ == "__main__":
    main()
