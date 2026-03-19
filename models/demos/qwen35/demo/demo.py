# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Qwen3.5-27B decode demo on single P100A Blackhole.

Hybrid DeltaNet + Attention architecture (48 DeltaNet + 16 full attention layers).
DeltaNet recurrence runs on host (float32) to avoid compound quantization error.
All projections, norms, MLP, and attention run on device with bfp8/bf16 weights.

Usage:
    export HF_MODEL=/path/to/Qwen3.5-27B
    python models/demos/qwen35/demo/demo.py [--prompt "..."] [--max_tokens N]
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
from tqdm import tqdm

import ttnn
from models.common.rmsnorm import RMSNorm
from models.tt_transformers.tt.ccl import TT_CCL
from models.tt_transformers.tt.common import Mode
from models.tt_transformers.tt.decoder import TransformerBlock
from models.tt_transformers.tt.distributed_norm import DistributedNorm
from models.tt_transformers.tt.gated_attention import GatedAttention
from models.tt_transformers.tt.model_config import ModelArgs
from models.tt_transformers.tt.qwen35_decoder import DeltaNetDecoderBlock
from models.tt_transformers.tt.rope import RotarySetup


def build_model(device, args, sd):
    """Build Qwen3.5-27B model with host embedding and CPU LM head."""
    from models.tt_transformers.tt.model import Transformer

    model = Transformer.__new__(Transformer)
    model.args = args
    model.vocab_size = args.vocab_size
    model.n_layers = args.n_layers
    model.mesh_device = device
    model.dtype = ttnn.bfloat8_b
    model.model_config = args.get_model_config()
    model.grid_size = args.max_grid_size
    model.decoders_optimizations = args.decoders_optimizations
    model.prefetcher = None

    model.tt_ccl = TT_CCL(device)
    wcp = args.weight_cache_path(dtype=ttnn.bfloat8_b)

    # RoPE with partial rotation (64/256 dims)
    model.rope_setup = RotarySetup(
        device=device,
        batch_size=args.max_batch_size,
        head_dim=args.head_dim,
        max_seq_len=args.max_seq_len,
        rope_theta=args.rope_theta,
        rope_scaling=args.rope_scaling,
        use_qk_fused=args.use_qk_fused,
    )
    partial = getattr(args, "partial_rotary_factor", 1.0)
    if partial < 1.0:
        rotary_dim = int(args.head_dim * partial)
        cos_h = ttnn.to_torch(model.rope_setup.cos_matrix)
        sin_h = ttnn.to_torch(model.rope_setup.sin_matrix)
        cos_h[:, :, :, rotary_dim:] = 1.0
        sin_h[:, :, :, rotary_dim:] = 0.0
        ttnn.deallocate(model.rope_setup.cos_matrix)
        ttnn.deallocate(model.rope_setup.sin_matrix)
        model.rope_setup.cos_matrix = ttnn.from_torch(
            cos_h, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        model.rope_setup.sin_matrix = ttnn.from_torch(
            sin_h, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )

    model.trans_mats_dict = model.rope_setup.get_both_trans_mats()

    # Build layers
    layers = []
    for i in tqdm(range(args.n_layers), desc="Layers"):
        if args.layer_types[i] == "linear_attention":
            layers.append(
                DeltaNetDecoderBlock(
                    args=args,
                    mesh_device=device,
                    tt_ccl=model.tt_ccl,
                    dtype=ttnn.bfloat8_b,
                    state_dict=sd,
                    layer_num=i,
                    weight_cache_path=wcp,
                )
            )
        else:
            layers.append(
                TransformerBlock(
                    args=args,
                    mesh_device=device,
                    tt_ccl=model.tt_ccl,
                    dtype=ttnn.bfloat8_b,
                    state_dict=sd,
                    layer_num=i,
                    weight_cache_path=wcp,
                    transformation_mats=model.trans_mats_dict,
                    attention_class=GatedAttention,
                )
            )
    model.layers = layers

    # Output norm
    model.norm = DistributedNorm(
        RMSNorm(
            device=device,
            dim=args.dim,
            eps=args.norm_eps,
            state_dict=sd,
            state_dict_prefix=args.get_state_dict_prefix("", None),
            weight_cache_path=wcp,
            weight_dtype=ttnn.bfloat16,
            weight_key="norm",
            add_unit_offset=args.rms_norm_add_unit_offset,
            is_distributed=args.is_distributed_norm,
            ccl_topology=args.ccl_topology(),
            tt_ccl=model.tt_ccl,
        ),
        args,
        tt_ccl=model.tt_ccl,
        TG=False,
    )

    return model


def generate(model, args, emb_weight_cpu, lm_weight_tt, prompt_ids, max_tokens=200):
    """Generate tokens given prompt token IDs."""
    device = model.mesh_device
    tokenizer = args.tokenizer
    generated = list(prompt_ids)
    B = args.tile_padded_batch_rows

    total_steps = min(len(prompt_ids) + max_tokens, args.max_seq_len)
    for step in range(total_steps):
        tok = prompt_ids[step] if step < len(prompt_ids) else generated[-1]

        # Embedding on CPU
        emb_vec = emb_weight_cpu[tok].unsqueeze(0)
        x_pad = torch.zeros(1, 1, B, args.dim)
        x_pad[0, 0, 0, :] = emb_vec
        x = ttnn.from_torch(x_pad, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        tt_pos = ttnn.from_torch(torch.tensor([step], dtype=torch.int32), dtype=ttnn.int32, device=device)
        rot_idxs = ttnn.from_torch(torch.tensor([[step]], dtype=torch.int64), device=device)
        rot_mats = model.rope_setup.get_rot_mats(rot_idxs)

        for layer in model.layers:
            x = layer(x, current_pos=tt_pos, rot_mats_global=rot_mats, mode=Mode.DECODE)

        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        x = model.norm(x, mode=Mode.DECODE, norm_config=args.get_norm_config("lm_head", Mode.DECODE, None))

        # LM head on device
        logits_tt = ttnn.linear(x, lm_weight_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        logits_cpu = ttnn.to_torch(logits_tt).float()[0, 0, 0, : args.vocab_size]
        ttnn.deallocate(logits_tt)
        next_token = logits_cpu.argmax().item()

        if step >= len(prompt_ids) - 1:
            generated.append(next_token)
            if tokenizer.decode([next_token]) in ["<|im_end|>", "<|endoftext|>"]:
                break

    return generated


def main():
    parser = argparse.ArgumentParser(description="Qwen3.5-27B decode on P100A")
    parser.add_argument("--prompt", type=str, default=None, help="Text prompt (default: sample prompt)")
    parser.add_argument("--prompt_file", type=str, default=None, help="JSON file with prompts")
    parser.add_argument("--max_tokens", type=int, default=200, help="Max tokens to generate")
    parser.add_argument("--max_seq_len", type=int, default=256, help="Max sequence length")
    args_cli = parser.parse_args()

    # Resolve HF model name to local snapshot path
    hf_model = os.environ.get("HF_MODEL", "Qwen/Qwen3.5-27B")
    if not os.path.isdir(hf_model):
        from huggingface_hub import snapshot_download

        hf_model = snapshot_download(hf_model)
    os.environ["HF_MODEL"] = hf_model

    device = ttnn.open_device(device_id=0)
    args = ModelArgs(device, max_seq_len=args_cli.max_seq_len)

    print(f"Qwen3.5-27B: {args.n_layers} layers, vocab={args.vocab_size}")
    sd = args.load_state_dict()
    emb_weight_cpu = sd[args.get_state_dict_prefix("", None) + "tok_embeddings.weight"].float()

    # LM head on device (248320 x 5120, ~1.2 GB bfp8)
    wcp = args.weight_cache_path(dtype=ttnn.bfloat8_b)
    lm_weight_tt = ttnn.as_tensor(
        sd["output.weight"].T.unsqueeze(0).unsqueeze(0).contiguous(),  # (1,1,5120,248320)
        dtype=ttnn.bfloat8_b,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        cache_file_name=wcp / "lm_head_output",
    )

    print("Building model...")
    model = build_model(device, args, sd)
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
    generated = generate(model, args, emb_weight_cpu, lm_weight_tt, ids, args_cli.max_tokens)
    dt = time.time() - t0

    n = len(generated) - len(ids)
    output_text = tokenizer.decode(generated)
    print(f"\n\n[{n} tokens in {dt:.1f}s = {n / dt:.2f} tok/s]")
    print(f"\nFull output:\n{output_text}")

    ttnn.close_device(device)


if __name__ == "__main__":
    main()
