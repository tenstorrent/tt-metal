# SPDX-FileCopyrightText: © 2026 Qwen Image 2.1 contributors
# SPDX-License-Identifier: Apache-2.0

"""Capture the pinned CUDA Qwen3-VL prompt encoder without loading the image DiT.

The ground truth comes from the unmodified upstream prompt-encoding method.
Text-only encoder weights are exported in small safetensors shards for TT loading.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from diffusers import QwenImage21Pipeline
from huggingface_hub import snapshot_download
from safetensors.torch import save_file
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb

from models.experimental.qwen_image_2_1.checkpoint import MODEL_ID, MODEL_REVISION

DEFAULT_PROMPT = "the quick brown fox jumps over the lazy dog"


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--existing-embeddings", type=Path)
    args = parser.parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the independent reference")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(snapshot_download(MODEL_ID, revision=MODEL_REVISION, local_files_only=True))
    processor = AutoProcessor.from_pretrained(checkpoint / "processor", local_files_only=True)
    encoder = (
        Qwen3VLForConditionalGeneration.from_pretrained(
            checkpoint / "text_encoder",
            torch_dtype=torch.bfloat16,
            local_files_only=True,
            attn_implementation="sdpa",
        )
        .eval()
        .to("cuda")
    )
    pipe = QwenImage21Pipeline(
        scheduler=None,
        vae=None,
        text_encoder=encoder,
        processor=processor,
        transformer=None,
    )
    language_model = encoder.model.language_model
    records = []

    def save(name, tensor):
        if isinstance(tensor, (tuple, list)):
            tensor = tensor[0]
        if not isinstance(tensor, torch.Tensor):
            return
        cpu = tensor.detach().cpu().contiguous()
        dest = args.output_dir / (name + ".pt")
        dest.parent.mkdir(parents=True, exist_ok=True)
        torch.save(cpu, dest)
        records.append({"name": name, "shape": list(cpu.shape), "dtype": str(cpu.dtype)})

    def hook(name):
        return lambda _module, _args, output: save(name, output)

    handles = [language_model.embed_tokens.register_forward_hook(hook("embedding"))]
    for index, layer in enumerate(language_model.layers):
        handles.append(layer.register_forward_hook(hook(f"layer_{index:03d}/output")))
    first = language_model.layers[0]
    for name, module in first.named_modules():
        if name:
            handles.append(module.register_forward_hook(hook("layer_000/" + name)))

    def capture_rotary(_module, positional, output):
        save("position_ids", positional[1])
        save("rotary_cos", output[0])
        save("rotary_sin", output[1])

    handles.append(language_model.rotary_emb.register_forward_hook(capture_rotary))

    def capture_attention(_module, positional, kwargs):
        hidden = kwargs.get("hidden_states", positional[0] if positional else None)
        save("layer_000/attention_input", hidden)
        if isinstance(kwargs.get("attention_mask"), torch.Tensor):
            save("causal_mask", kwargs["attention_mask"])
        shape = (*hidden.shape[:-1], -1, 128)
        attn = first.self_attn
        q = attn.q_norm(attn.q_proj(hidden).view(shape)).transpose(1, 2)
        k = attn.k_norm(attn.k_proj(hidden).view(shape)).transpose(1, 2)
        v = attn.v_proj(hidden).view(shape).transpose(1, 2)
        cos, sin = kwargs["position_embeddings"]
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        save("layer_000/rotated_q", q)
        save("layer_000/rotated_k", k)
        save("layer_000/value_heads", v)

    handles.append(first.self_attn.register_forward_pre_hook(capture_attention, with_kwargs=True))
    handles.append(
        first.self_attn.o_proj.register_forward_pre_hook(
            lambda _module, positional: save("layer_000/attention_heads_merged", positional[0])
        )
    )
    handles.append(
        first.mlp.down_proj.register_forward_pre_hook(
            lambda _module, positional: save("layer_000/mlp_product", positional[0])
        )
    )
    model_inputs = processor(
        text=[pipe.prompt_template_t2i.format(args.prompt or " ")],
        padding=True,
        padding_side="left",
        return_tensors="pt",
    )
    for key, value in model_inputs.items():
        save(key, value)
    with torch.inference_mode():
        prompt_embeds, prompt_mask, image_mask = pipe._get_qwen_prompt_embeds(args.prompt, device=torch.device("cuda"))
    save("prompt_embeds", prompt_embeds)
    save("prompt_mask", prompt_mask)
    save("image_pad_mask", image_mask)
    for handle in handles:
        handle.remove()
    equality = None
    if args.existing_embeddings:
        existing = torch.load(args.existing_embeddings, weights_only=True)
        equality = torch.equal(prompt_embeds.cpu(), existing)
        a, b = prompt_embeds.cpu().double().flatten(), existing.double().flatten()
        pcc = float(torch.dot(a - a.mean(), b - b.mean()) / ((a - a.mean()).norm() * (b - b.mean()).norm()))
        print(
            f"Existing prompt embeddings: bitwise_equal={equality}, PCC={pcc}, max_abs={float((a - b).abs().max())}",
            flush=True,
        )
        if not equality:
            raise AssertionError("Independent encoder does not match the existing CUDA capture bitwise")
    args.export_dir.mkdir(parents=True, exist_ok=True)
    text_dir = args.export_dir / "text_encoder"
    text_dir.mkdir(exist_ok=True)
    shutil.copy2(checkpoint / "text_encoder" / "config.json", text_dir / "config.json")
    shutil.copytree(checkpoint / "processor", args.export_dir / "processor", dirs_exist_ok=True)
    weight_map = {}
    for group, module in [("embedding", language_model.embed_tokens)] + [
        (f"layer_{i:03d}", layer) for i, layer in enumerate(language_model.layers)
    ]:
        prefix = (
            "model.language_model.embed_tokens"
            if group == "embedding"
            else f"model.language_model.layers.{int(group[-3:])}"
        )
        state = {f"{prefix}.{key}": value.detach().cpu().contiguous() for key, value in module.state_dict().items()}
        filename = group + ".safetensors"
        save_file(state, text_dir / filename, metadata={"source_revision": MODEL_REVISION})
        weight_map.update({key: filename for key in state})
        print(f"Exported {filename}", flush=True)
    (text_dir / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}, indent=2))
    manifest = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "prompt": args.prompt,
        "drop_idx": pipe._drop_idx,
        "template": pipe.prompt_template_t2i,
        "final_rmsnorm": "bypassed, matching pinned QwenImage21Pipeline",
        "attention_backend": "sdpa",
        "existing_capture_bitwise_equal": equality,
        "records": records,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("Independent CUDA prompt encoder capture and export complete", flush=True)


if __name__ == "__main__":
    main()
