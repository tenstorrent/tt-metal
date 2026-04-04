# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Qwen3-Coder-Next generation demo on QuietBox (8 WH devices).

Usage:
    HF_MODEL=<path> python models/demos/qwen3_coder_next/demo/generate.py [--prompt "..."] [--max_tokens N]
"""

import argparse
import os
import time

import torch
from transformers import AutoTokenizer

import ttnn
from models.tt_transformers.tt.model import Transformer
from models.tt_transformers.tt.model_config import ModelArgs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="def fibonacci(n):", help="Input prompt")
    parser.add_argument("--max_tokens", type=int, default=50, help="Max tokens to generate")
    args = parser.parse_args()

    hf_model = os.environ.get(
        "HF_MODEL",
        "/home/ttuser/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-Next/snapshots/a7fbcb5c0e12d62a448eaa0e260346bf5dcc0feb",
    )

    print(f"Model: {hf_model}")
    print(f"Prompt: {args.prompt}")

    # Setup
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 8), l1_small_size=65536)
    rep = ttnn.ReplicateTensorToMesh(mesh)

    os.environ["HF_MODEL"] = hf_model
    tokenizer = AutoTokenizer.from_pretrained(hf_model)

    # Load model
    print("Loading model...", flush=True)
    t0 = time.time()
    model_args = ModelArgs(mesh_device=mesh, instruct=False, max_batch_size=1, optimizations=None, max_seq_len=1024)
    state_dict = model_args.load_state_dict()
    model = Transformer(
        args=model_args,
        mesh_device=mesh,
        dtype=ttnn.bfloat8_b,
        state_dict=state_dict,
        weight_cache_path=model_args.weight_cache_path(ttnn.bfloat8_b),
    )
    emb_weight = state_dict["tok_embeddings.weight"]
    print(f"Model loaded in {time.time() - t0:.1f}s ({len(model.layers)} layers)", flush=True)

    # Tokenize prompt
    input_ids = tokenizer.encode(args.prompt)
    print(f"Prompt tokens ({len(input_ids)}): {input_ids}")

    # Initialize DeltaNet states
    for layer in model.layers:
        if hasattr(layer.attention, "initialize_states"):
            layer.attention.initialize_states(batch_size=1)

    # Generation loop
    generated = list(input_ids)
    print(f"\n--- Generation ---")
    print(args.prompt, end="", flush=True)

    for step in range(len(input_ids) + args.max_tokens):
        if step < len(input_ids):
            tok = input_ids[step]
        else:
            tok = generated[-1]

        # Embed
        x_host = emb_weight[tok].reshape(1, 1, 1, -1).expand(1, 1, 32, -1).to(torch.bfloat16).contiguous()
        x_tt = ttnn.as_tensor(
            x_host,
            dtype=ttnn.bfloat16,
            device=mesh,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=rep,
        )

        # RoPE (position = step for GQA layers)
        rot_mats = model.rope_setup.get_rot_mats(torch.tensor([step]))

        # Forward
        t_fwd = time.time()
        logits = model.forward(x_tt, current_pos=torch.tensor([step]), rot_mats_global=rot_mats, batch_size=1)
        ttnn.synchronize_device(mesh)
        fwd_ms = (time.time() - t_fwd) * 1000

        # Get next token
        logits_torch = ttnn.to_torch(logits, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))[0:1]
        next_tok = logits_torch[0, 0, 0].float().argmax().item()
        ttnn.deallocate(logits)
        ttnn.deallocate(x_tt)

        if step >= len(input_ids):
            generated.append(next_tok)
            token_str = tokenizer.decode([next_tok])
            print(token_str, end="", flush=True)

            # Stop on EOS
            if next_tok == tokenizer.eos_token_id:
                break

    print(f"\n\n--- Done ({len(generated) - len(input_ids)} tokens generated) ---")
    print(f"Full output: {tokenizer.decode(generated)}")

    ttnn.close_mesh_device(mesh)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
