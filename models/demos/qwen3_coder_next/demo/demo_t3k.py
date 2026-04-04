# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Qwen3-Coder-Next decode demo on QuietBox T3K (4×N300, 8 Wormhole devices).

Uses tensor parallelism across 8 devices:
- DeltaNet layers: weights sharded, recurrence via forward_cpu (host float32)
- GatedAttention layers: standard TP with sharded QKV/WO, batch-parallel KV cache
- MoE: expert-parallel (64 experts per device)

Usage:
    export HF_MODEL=Qwen/Qwen3-Coder-Next
    python models/demos/qwen3_coder_next/demo/demo_t3k.py [--prompt "..."] [--max_tokens N]
"""

import argparse
import os

import ttnn


def main():
    parser = argparse.ArgumentParser(description="Qwen3-Coder-Next T3K Demo")
    parser.add_argument("--prompt", default="def fibonacci(n):", help="Input prompt")
    parser.add_argument("--max_tokens", type=int, default=200, help="Max tokens to generate")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--max_seq_len", type=int, default=8192, help="Max sequence length")
    args = parser.parse_args()

    hf_model = os.environ.get("HF_MODEL", "Qwen/Qwen3-Coder-Next")
    print(f"Model: {hf_model}")
    print(f"Prompt: {args.prompt}")
    print(f"Max tokens: {args.max_tokens}")

    # Configure fabric for T3K ring topology
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)

    # Open 8-device mesh (QuietBox T3K)
    print("Opening mesh device (1, 8)...")
    mesh_device = ttnn.open_mesh_device(
        ttnn.MeshShape(1, 8),
        trace_region_size=184915840,
    )
    num_devices = mesh_device.get_num_devices()
    print(f"Devices: {num_devices}, Cluster: {ttnn.cluster.get_cluster_type()}")

    try:
        # Load tokenizer
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(hf_model, trust_remote_code=True)

        # Tokenize prompt
        prompt_ids = tokenizer.encode(args.prompt, add_special_tokens=True)
        print(f"Prompt tokens: {len(prompt_ids)}")

        # TODO: Load model using Qwen3CoderNextModelArgs + Transformer
        # from models.demos.qwen3_coder_next.tt.tt_model_config import Qwen3CoderNextTTConfig
        # from models.tt_transformers.tt.model import Transformer
        #
        # model_args = Qwen3CoderNextTTConfig.from_pretrained(
        #     hf_model, mesh_device, max_batch_size=args.batch_size, max_seq_len=args.max_seq_len
        # )
        # state_dict = model_args.load_state_dict()
        # model = Transformer(args=model_args, mesh_device=mesh_device, ...)
        #
        # For now, print configuration and exit
        from models.demos.qwen3_coder_next.tt.model_config import Qwen3CoderNextConfig

        config = Qwen3CoderNextConfig()
        print(f"\nModel config:")
        print(
            f"  Layers: {config.num_hidden_layers} ({config.num_deltanet_layers} DeltaNet + {config.num_gqa_layers} GQA)"
        )
        print(f"  Hidden: {config.hidden_size}, Heads: {config.num_attention_heads}")
        print(f"  Experts: {config.num_experts} (top-{config.num_experts_per_tok})")
        print(f"  Experts/device: {config.num_experts // num_devices}")
        print(f"  Batch-parallel: {args.batch_size // num_devices} users/device")

        mem = config.memory_estimate_bytes(num_devices=num_devices, weight_dtype_bytes=1)
        print(f"  Memory/device: {mem['per_device_gb']:.1f} GB")

        print("\nDemo ready. Full model generation requires completing Phase 5 (integration).")

    finally:
        print("Closing mesh device...")
        ttnn.close_mesh_device(mesh_device)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
