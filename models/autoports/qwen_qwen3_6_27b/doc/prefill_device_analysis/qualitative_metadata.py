# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Persist checkpoint-correct prompts for the shared serving suite."""
import json
from pathlib import Path

from transformers import AutoTokenizer

snapshot = (
    Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
)
tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
source = Path("models/common/readiness_check/vllm_prompts.txt")
prompts = [p.strip() for p in source.read_text().split("\n\n") if p.strip()]
result = {
    "model": "Qwen/Qwen3.8-27B",
    "revision": snapshot.name,
    "tokenizer_class": type(tokenizer).__name__,
    "chat_template_present": bool(tokenizer.chat_template),
    "prompt_mode": "chat",
    "endpoint": "/v1/chat/completions",
    "prompt_source": str(source),
    "generation": {"max_tokens": 256, "greedy_temperature": 0, "sampled_temperature": 0.7, "sampled_top_p": 0.9},
    "control": "doc/qwen38_checkpoint_swap/full_model_qualitative.json: same suite/checkpoint, prior TT greedy50-token control",
    "cases": [],
}
for index, prompt in enumerate(prompts):
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )
    result["cases"].append(
        {
            "id": index,
            "prompt": prompt,
            "rendered_prompt": rendered,
            "prompt_token_ids": tokenizer.encode(rendered, add_special_tokens=False),
        }
    )
(Path(__file__).parent / "artifacts/qualitative_prompt_format.json").write_text(json.dumps(result, indent=2) + "\n")
