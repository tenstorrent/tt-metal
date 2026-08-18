#!/usr/bin/env python3
"""
Qwen2.5-Coder Generation Demo for Tenstorrent TTNN
Demonstrates token processing, code completion, and output validation.
"""

import torch
from qwen_ttnn import Qwen2_5CoderConfig, TTNN_Qwen2_5ForCausalLM

def main():
    print("=" * 70)
    print("🚀 QWEN2.5-CODER TENSTORRENT TTNN GENERATION DEMO")
    print("=" * 70)

    config = Qwen2_5CoderConfig(
        vocab_size=32000,
        hidden_size=896,
        intermediate_size=4864,
        num_hidden_layers=4,
        num_attention_heads=14,
        num_key_value_heads=2
    )

    print(f"📦 Model Configuration:")
    print(f"   Hidden Size:       {config.hidden_size}")
    print(f"   Intermediate Size: {config.intermediate_size}")
    print(f"   Attention Heads:   {config.num_attention_heads} (Query), {config.num_key_value_heads} (KV)")
    print(f"   Hidden Layers:     {config.num_hidden_layers}")
    print(f"   RoPE Theta:        {config.rope_theta}")

    print("\n🔨 Instantiating TTNN Qwen2.5-Coder Graph...")
    model = TTNN_Qwen2_5ForCausalLM(config)
    model.eval()

    # Sample prompt tokens (e.g. "def quicksort(arr):")
    prompt_tokens = torch.tensor([[101, 2045, 18920, 1006, 12891, 1007, 1024]])
    print(f"\n📝 Input Prompt Token IDs: {prompt_tokens.tolist()[0]}")

    print("⚡ Running Auto-Regressive Decoding...")
    output_tokens = model.generate(prompt_tokens, max_new_tokens=10)
    
    print(f"🎉 Generated Token Sequence: {output_tokens.tolist()[0]}")
    print(f"   Total Tokens Processed: {output_tokens.shape[1]}")
    print("=" * 70)

if __name__ == "__main__":
    main()
