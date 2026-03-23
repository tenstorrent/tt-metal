# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Qwen3.5-27B decode demo on P300x2 (QuietBox2, 4 Blackhole chips).

Uses tensor parallelism across 4 devices:
- DeltaNet layers: weights replicated, recurrence on host (float32)
- GatedAttention layers: standard TP with sharded QKV/WO weights
- MLP: standard TP with sharded w1/w2/w3 weights

Usage:
    export HF_MODEL=/path/to/Qwen3.5-27B
    python models/demos/qwen35/demo/demo_p300x2.py [--prompt "..."] [--max_tokens N]
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch

import ttnn
from models.tt_transformers.tt.model import Transformer
from models.tt_transformers.tt.model_config import ModelArgs


def generate(model, args, mesh_device, prompt_ids, max_tokens=200):
    """Generate tokens given prompt token IDs using the standard Transformer model."""
    tokenizer = args.tokenizer
    generated = list(prompt_ids)

    from models.tt_transformers.tt.common import Mode

    total_steps = min(len(prompt_ids) + max_tokens, args.max_seq_len)
    for step in range(total_steps):
        tok = prompt_ids[step] if step < len(prompt_ids) else generated[-1]

        # Embedding on device
        tok_tensor = torch.tensor([[tok]], dtype=torch.uint32)
        x = ttnn.from_torch(
            tok_tensor,
            dtype=ttnn.uint32,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        x_embed = model.embd(x)
        x_embed = ttnn.unsqueeze_to_4D(x_embed)

        # Position and rotation
        tt_pos = ttnn.from_torch(
            torch.tensor([step], dtype=torch.int32),
            dtype=ttnn.int32,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        rot_idxs = ttnn.from_torch(
            torch.tensor([[step]], dtype=torch.int64),
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        rot_mats = model.rope_setup.get_rot_mats(rot_idxs)

        # Forward through all layers + norm + lm_head
        logits_tt = model.forward(x_embed, tt_pos, rot_mats_global=rot_mats, mode=Mode.DECODE)

        # Gather logits from all devices (column-sharded from lm_head)
        logits_parts = [ttnn.to_torch(dt).float() for dt in ttnn.get_device_tensors(logits_tt)]
        logits_cpu = torch.cat(logits_parts, dim=-1)[0, 0, 0, : args.vocab_size]
        ttnn.deallocate(logits_tt)
        next_token = logits_cpu.argmax().item()

        if step >= len(prompt_ids) - 1:
            generated.append(next_token)
            if tokenizer.decode([next_token]) in ["<|im_end|>", "<|endoftext|>"]:
                break

    return generated


def main():
    parser = argparse.ArgumentParser(description="Qwen3.5-27B decode on P300x2 (QuietBox2)")
    parser.add_argument("--prompt", type=str, default=None, help="Text prompt")
    parser.add_argument("--prompt_file", type=str, default=None, help="JSON file with prompts")
    parser.add_argument("--max_tokens", type=int, default=200, help="Max tokens to generate")
    parser.add_argument("--max_seq_len", type=int, default=256, help="Max sequence length")
    args_cli = parser.parse_args()

    # Resolve HF model
    hf_model = os.environ.get("HF_MODEL", "Qwen/Qwen3.5-27B")
    if not os.path.isdir(hf_model):
        from huggingface_hub import snapshot_download

        hf_model = snapshot_download(hf_model)
    os.environ["HF_MODEL"] = hf_model

    # Open 4-device mesh (P300x2 = 2 P300 cards = 4 Blackhole chips)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))
    args = ModelArgs(mesh_device, max_seq_len=args_cli.max_seq_len)

    print(
        f"Qwen3.5-27B P300x2: {args.n_layers} layers, vocab={args.vocab_size}, "
        f"devices={args.num_devices}, batch={args.max_batch_size}"
    )

    sd = args.load_state_dict()
    wcp = args.weight_cache_path(dtype=ttnn.bfloat8_b)

    print("Building model...")
    model = Transformer(
        args=args,
        dtype=ttnn.bfloat8_b,
        mesh_device=mesh_device,
        state_dict=sd,
        weight_cache_path=wcp,
    )
    del sd
    print("Model ready!")

    tokenizer = args.tokenizer

    # Get prompt
    if args_cli.prompt_file:
        with open(args_cli.prompt_file) as f:
            prompts = json.load(f)
        prompt = prompts[0]["prompt"]
    elif args_cli.prompt:
        prompt = f"<|im_start|>user\n{args_cli.prompt}<|im_end|>\n<|im_start|>assistant\n"
    else:
        prompt_file = Path(__file__).parent / "sample_prompts" / "demo.json"
        with open(prompt_file) as f:
            prompts = json.load(f)
        prompt = prompts[0]["prompt"]

    ids = tokenizer.encode(prompt)
    print(f"Prompt: {repr(prompt.strip())} ({len(ids)} tokens)")
    print(f"Generating up to {args_cli.max_tokens} tokens...\n")

    t0 = time.time()
    generated = generate(model, args, mesh_device, ids, args_cli.max_tokens)
    dt = time.time() - t0

    n = len(generated) - len(ids)
    output_text = tokenizer.decode(generated)
    print(f"\n\n[{n} tokens in {dt:.1f}s = {n / dt:.2f} tok/s]")
    print(f"\nFull output:\n{output_text}")

    for submesh in mesh_device.get_submeshes():
        ttnn.close_mesh_device(submesh)
    ttnn.close_mesh_device(mesh_device)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
