# SPDX-License-Identifier: Apache-2.0
"""Pinned HF AIME24 reference using the unchanged readiness schema/scoring."""
import hashlib
import json
import time
from pathlib import Path

import torch
from readiness_check.generate import DEFAULT_AIME24_PROMPTS_FILE, _generate_one_entry, _load_aime24_prompt
from readiness_check.schema import Reference, save_reference
from transformers import AutoTokenizer

from ..tt.checkpoint import MODEL_ID, REVISION, SNAPSHOT
from .hf_model import Kolibri1ForCausalLM


def main():
    torch.set_num_threads(8)
    start = time.monotonic()
    root = Path(__file__).resolve().parents[1]
    out = root / "doc/full_model"
    tokenizer = AutoTokenizer.from_pretrained(SNAPSHOT, local_files_only=True)
    prompt = _load_aime24_prompt(DEFAULT_AIME24_PROMPTS_FILE, 0)
    messages = [{"role": "user", "content": prompt}]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=False)
    (out / "aime24_rendered_prompt.txt").write_text(rendered)
    metadata = dict(
        hf_model_id=MODEL_ID,
        revision=REVISION,
        snapshot=str(SNAPSHOT),
        tokenizer_class=type(tokenizer).__name__,
        tokenizer_sha256=hashlib.sha256((SNAPSHOT / "tokenizer.json").read_bytes()).hexdigest(),
        chat_template=True,
        prompt_mode="chat",
        prompt_source="aime24",
        prompt_index=0,
        prompt_source_path=str(DEFAULT_AIME24_PROMPTS_FILE),
        prompt_token_ids=ids,
        gen_len=100,
        top_k=100,
        generation_command="python -m models.autoports.aleph_alpha_kolibri_1_bf16.tests.full_reference",
        hf_architecture="model-local independent PyTorch extension of provider-validated Stage1 ReferenceDecoder",
        tokenizer_compatibility="Explicit return_dict=False: installed Transformers returns BatchEncoding by default, which the packaged generate CLI does not normalize",
    )
    (out / "reference_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("REFERENCE_PROMPT", len(ids), flush=True)
    model = Kolibri1ForCausalLM.from_pretrained(SNAPSHOT).eval()
    prompt_tensor = torch.tensor(ids, dtype=torch.long)
    generated = model.generate(prompt_tensor[None], max_new_tokens=100)[0, len(ids) :]
    (out / "aime24_hf_completion.txt").write_text(tokenizer.decode(generated.tolist(), skip_special_tokens=False))
    torch.save(dict(prompt_tokens=prompt_tensor, generated_tokens=generated), out / "hf_generation_checkpoint.pt")
    entry = _generate_one_entry(model, tokenizer, prompt_tensor, generated, 100, torch.device("cpu"))
    save_reference(
        Reference(
            k=100,
            hf_model_id=MODEL_ID,
            entries=[entry],
            token_ids_meta=dict(
                bos_id=tokenizer.bos_token_id, eos_id=tokenizer.eos_token_id, pad_id=tokenizer.pad_token_id
            ),
        ),
        root / "readiness_aime24_chat.refpt",
    )
    metadata["seconds"] = time.monotonic() - start
    (out / "reference_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("REFERENCE_DONE", metadata["seconds"], flush=True)


if __name__ == "__main__":
    main()
