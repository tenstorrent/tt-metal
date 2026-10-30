"""Record pinned-HF decoder inputs from text; CPU reference preparation only."""

import argparse
import json
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextDecoderLayer, Gemma4TextRotaryEmbedding

from models.autoports.google_gemma_4_31b_it.tests.run_decoder import weights
from models.autoports.google_gemma_4_31b_it.tt.optimized_decoder import HF_MODEL, HF_REVISION


def record(layer, length, output, chunk_reference=False, embedding_output=None, batch=1):
    torch.set_num_threads(16)
    cfg = AutoConfig.from_pretrained(HF_MODEL, revision=HF_REVISION).text_config
    cfg._attn_implementation = "eager"
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL, revision=HF_REVISION)
    text = "Explain how a paged cache preserves a conversation when its physical memory pages move. Include an example and a test of correctness. "
    topics = [
        "Write a Python test for sorting.",
        "Explain lunar eclipses.",
        "Translate hello into French.",
        "Plan a vegetable garden.",
        "Compare two prime numbers.",
        "Describe how bread rises.",
        "Trace a cache address.",
    ]
    prompts = [text] if batch == 1 else [f"Request {i}: {topics[i % len(topics)]} " + text for i in range(batch)]
    ids = torch.cat(
        [
            tokenizer(prompt * max(1, length // 8 + 1), return_tensors="pt")["input_ids"][:, :length]
            for prompt in prompts
        ]
    )
    assert tuple(ids.shape) == (batch, length)
    idx = json.loads(Path(hf_hub_download(HF_MODEL, "model.safetensors.index.json", revision=HF_REVISION)).read_text())
    key = "model.language_model.embed_tokens.weight"
    with safe_open(hf_hub_download(HF_MODEL, idx["weight_map"][key], revision=HF_REVISION), framework="pt") as f:
        embedding = f.get_tensor(key)
        x = torch.nn.functional.embedding(ids, embedding) * torch.tensor(cfg.hidden_size**0.5, dtype=embedding.dtype)
    del embedding
    if embedding_output:
        torch.save(x, embedding_output)
    rope = Gemma4TextRotaryEmbedding(cfg)
    positions = torch.arange(length).unsqueeze(0).expand(batch, -1)
    for i in range(layer):
        with torch.device("meta"):
            ref = Gemma4TextDecoderLayer(cfg, i)
        ref.load_state_dict(weights(i), assign=True)
        ref.eval()
        kind = cfg.layer_types[i]
        cos, sin = rope(x, positions, layer_type=kind)
        if chunk_reference or length > 4096:
            if kind != "sliding_attention":
                raise ValueError("Bounded reference preparation currently supports preceding sliding layers only")
            outputs = []
            for offset in range(0, length, 1024):
                end = min(length, offset + 1024)
                start = max(0, offset - cfg.sliding_window + 1)
                qpos = torch.arange(start, end)
                allowed = (qpos[None, :] <= qpos[:, None]) & (qpos[None, :] > qpos[:, None] - cfg.sliding_window)
                mask = torch.where(allowed, 0.0, float("-inf")).to(torch.bfloat16)[None, None]
                with torch.no_grad():
                    y = ref(
                        x[:, start:end],
                        position_embeddings=(cos[:, start:end], sin[:, start:end]),
                        attention_mask=mask,
                        shared_kv_states={},
                    )
                outputs.append(y[:, offset - start :].clone())
            x = torch.cat(outputs, dim=1)
            del outputs, y
        else:
            mask = torch.arange(length)[None, :] <= torch.arange(length)[:, None]
            if kind == "sliding_attention":
                mask &= torch.arange(length)[None, :] > torch.arange(length)[:, None] - cfg.sliding_window
            mask = torch.where(mask, 0.0, float("-inf")).to(torch.bfloat16)[None, None]
            with torch.no_grad():
                x = ref(x, position_embeddings=(cos, sin), attention_mask=mask, shared_kv_states={})
        print("REFERENCE_LAYER_DONE", i, length, flush=True)
        del ref
    torch.save(x, output)
    Path(str(output) + ".json").write_text(
        json.dumps(
            dict(
                model=HF_MODEL,
                revision=HF_REVISION,
                layer=layer,
                length=length,
                source="HF text embeddings and preceding reference layers",
                prompts=prompts,
                batch=batch,
                bounded_sliding_reference=chunk_reference or length > 4096,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--length", type=int, default=34)
    p.add_argument("--output", required=True)
    p.add_argument("--chunk-reference", action="store_true")
    p.add_argument("--embedding-output")
    p.add_argument("--batch", type=int, default=1)
    a = p.parse_args()
    record(a.layer, a.length, a.output, a.chunk_reference, a.embedding_output, a.batch)
