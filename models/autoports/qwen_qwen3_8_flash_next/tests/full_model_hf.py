"""Pinned HF text oracle using mmap weights and exact, indexed PLE lookups.

Only storage/loading changes: every forward is the installed HF implementation.
Avoid concatenating the 95GiB PLE table; look up the original shard/row instead.
"""

import argparse, json, time
from pathlib import Path
import torch
from transformers import AutoConfig, AutoTokenizer, GenerationConfig
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpForCausalLM, Qwen4ExpTextRotaryEmbedding
from models.autoports.qwen_qwen3_8_flash_next.tt.checkpoint import CheckpointState
from models.autoports.qwen_qwen3_8_flash_next.tests.record_multichip_inputs import CHECKPOINT, MappedPLE


def build_hf(layer_indices=None, checkpoint=CHECKPOINT):
    cfg = AutoConfig.from_pretrained(checkpoint, local_files_only=True).text_config
    cfg._attn_implementation = "eager"
    cfg._experts_implementation = "eager"
    with torch.device("meta"):
        model = Qwen4ExpForCausalLM(cfg)
    model.generation_config = GenerationConfig.from_pretrained(checkpoint, local_files_only=True)
    state = CheckpointState(checkpoint)
    tensors = {}
    for name in model.state_dict():
        if "ngram_embedding.weight" in name:
            continue
        key = name.replace("model.", "model.language_model.", 1) if name.startswith("model.") else name
        tensors[name] = state[key]
    missing, extra = model.load_state_dict(tensors, assign=True, strict=False)
    assert missing == ["model.layers.1.ple.ple_embedding.ngram_embedding.weight"] and not extra, (missing, extra)
    model.model.rotary_emb = Qwen4ExpTextRotaryEmbedding(cfg)
    model.model.layers[1].ple.ple_embedding.ngram_embedding = MappedPLE(
        CheckpointState(checkpoint, "model.language_model.layers.1.")
    )
    if layer_indices is not None:
        model.model.layers = torch.nn.ModuleList([model.model.layers[i] for i in layer_indices])
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-len", type=int, default=100)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    from readiness_check import generate

    root = Path("models/autoports/qwen_qwen3_8_flash_next")
    model = build_hf()
    print("HF_MODEL_LOADED", flush=True)
    # Standard runner, with checkpoint storage loader supplied locally.
    generate.AutoModelForCausalLM.from_pretrained = lambda *args, **kw: model
    original = generate.AutoTokenizer.from_pretrained
    generate.AutoTokenizer.from_pretrained = lambda *args, **kw: original(CHECKPOINT, local_files_only=True)

    def prompt_tokens(tokenizer, prompt, *, chat_template):
        if chat_template:
            result = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True
            )
            if hasattr(result, "keys"):
                result = result["input_ids"]
            return result
        return tokenizer.encode(prompt, add_special_tokens=True)

    generate._chat_or_plain_prompt_tokens = prompt_tokens
    tick = time.perf_counter()
    generate.generate_reference(
        "Qwen/Qwen3.8-Flash-Next",
        prompt_source="aime24",
        chat_template=True,
        gen_len=a.gen_len,
        top_k=100,
        output_path=root / "readiness_aime24_chat.refpt",
    )
    metadata = dict(
        model="Qwen/Qwen3.8-Flash-Next",
        revision=CHECKPOINT.name,
        checkpoint=str(CHECKPOINT),
        tokenizer="AutoTokenizer pinned local snapshot",
        prompt_source="aime24",
        chat_template=True,
        gen_len=a.gen_len,
        top_k=100,
        command="python -m models.autoports.qwen_qwen3_8_flash_next.tests.full_model_hf --gen-len 100",
        elapsed_seconds=time.perf_counter() - tick,
    )
    (root / "doc/full_model/reference_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
