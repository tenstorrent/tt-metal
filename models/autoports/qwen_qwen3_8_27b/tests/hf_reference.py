# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Pinned, memory-mapped HF text-only control; no TTNN import or device usage."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer, GenerationConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5TextRotaryEmbedding

from models.common.readiness_check.schema import Reference, ReferenceEntry, save_reference

MODEL_ID = "Qwen/Qwen3.8-27B"
REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"


def load_hf(snapshot):
    config = AutoConfig.from_pretrained(snapshot, local_files_only=True).text_config
    config._attn_implementation = "sdpa"
    with torch.device("meta"):
        model = Qwen3_5ForCausalLM(config)
    index = json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"]
    tensors = {}
    for name, shard in index.items():
        if name.startswith("model.language_model."):
            key = name.replace("model.language_model.", "model.", 1)
        elif name == "lm_head.weight":
            key = name
        else:
            continue
        with safe_open(snapshot / shard, framework="pt", device="cpu") as f:
            tensors[key] = f.get_tensor(name)
    model.load_state_dict(tensors, strict=True, assign=True)
    model.model.rotary_emb = Qwen3_5TextRotaryEmbedding(config)
    model.generation_config = GenerationConfig.from_pretrained(snapshot, local_files_only=True)
    return model.eval()


def main():
    from huggingface_hub import snapshot_download

    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--suite", action="store_true")
    p.add_argument("--gen-len", type=int, default=100)
    a = p.parse_args()
    torch.set_num_threads(8)
    snapshot = Path(snapshot_download(MODEL_ID, revision=REVISION, local_files_only=True))
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    assert tokenizer.chat_template
    model = load_hf(snapshot)
    print("HF_LOADED", flush=True)
    corpus = Path("models/demos/deepseek_v3/demo/aime_under_8k_prompts.json")
    prompts = [json.loads(corpus.read_text())[0]["prompt"]]
    if a.suite:
        corpus = Path("models/common/readiness_check/vllm_prompts.txt")
        prompts = [s.strip() for s in corpus.read_text().split("\n\n") if s.strip()]
    entries = []
    outputs = []
    for i, prompt in enumerate(prompts):
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
        ids = torch.tensor([tokenizer(rendered, add_special_tokens=False)["input_ids"]])
        cur = ids
        cache = None
        tokens = []
        topk = []
        begin = time.perf_counter()
        with torch.no_grad():
            for step in range(a.gen_len):
                out = model(input_ids=cur, past_key_values=cache, use_cache=True, logits_to_keep=1)
                cache = out.past_key_values
                scores = out.logits[0, -1].float()
                topk.append(scores.topk(100).indices.int())
                token = int(scores.argmax())
                tokens.append(token)
                cur = torch.tensor([[token]])
                if step % 10 == 0:
                    print("HF_STEP", i, step, token, round(time.perf_counter() - begin, 2), flush=True)
        entry = ReferenceEntry(rendered, ids, torch.tensor([tokens]), torch.stack(topk), ids.shape[1])
        entries.append(entry)
        outputs.append(
            dict(
                prompt_id=i,
                prompt=prompt,
                rendered=rendered,
                token_ids=ids[0].tolist(),
                output_tokens=tokens,
                text=tokenizer.decode(tokens),
                seconds=time.perf_counter() - begin,
            )
        )
        print("HF_TEXT", i, outputs[-1]["text"], flush=True)
        a.output.with_suffix(".outputs.json").write_text(json.dumps(outputs, indent=2) + "\n")
    save_reference(
        Reference(
            100,
            MODEL_ID,
            entries,
            dict(bos_id=tokenizer.bos_token_id, eos_id=model.config.eos_token_id, pad_id=tokenizer.pad_token_id),
        ),
        a.output,
    )
    metadata = dict(
        hf_model_id=MODEL_ID,
        revision=REVISION,
        snapshot=str(snapshot),
        tokenizer=type(tokenizer).__name__,
        chat_template=True,
        prompt_source=str(corpus),
        prompt_source_sha256=hashlib.sha256(corpus.read_bytes()).hexdigest(),
        generation_length=a.gen_len,
        top_k=100,
        command=" ".join(__import__("sys").argv),
        prompt_mode="chat",
        dtype="bfloat16",
        state_load="strict assign from pinned safetensors",
        reference_method="HF sequential greedy logits and top100, 100 exact tokens, no stop truncation",
    )
    a.output.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
