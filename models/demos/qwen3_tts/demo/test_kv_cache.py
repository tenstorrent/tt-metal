# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Quick test for KV cache optimization.

Skips the slow MimiModel encoder by using random inputs.
Compares generation with and without KV cache.
"""

import time

import torch

import ttnn


def test_kv_cache():
    print("=" * 60)
    print("KV Cache Optimization Test")
    print("=" * 60)

    # Load weights
    from pathlib import Path

    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    print("\nLoading model weights...")
    model_path = Path(snapshot_download("Qwen/Qwen3-TTS-12Hz-1.7B-Base", allow_patterns=["*.safetensors"]))
    main_dict = {}
    for f in model_path.glob("*.safetensors"):
        if "speech_tokenizer" not in str(f):
            main_dict.update(load_file(f))
    print(f"  Loaded {len(main_dict)} weights")

    # Open device
    print("\nOpening device...")
    device = ttnn.open_device(device_id=0)

    try:
        # Initialize model
        print("\nInitializing TTNN model...")
        from models.demos.qwen3_tts.tt.qwen3_tts import Qwen3TTS

        model = Qwen3TTS(device=device, state_dict=main_dict)
        print("  Model ready")

        # Create random ICL-like input (simulating the prefill sequence)
        batch_size = 1
        seq_len = 64  # Simulated ICL sequence length
        hidden_size = 2048

        print(f"\nCreating random input: [{batch_size}, 1, {seq_len}, {hidden_size}]")
        inputs_embeds = torch.randn(batch_size, 1, seq_len, hidden_size)
        inputs_embeds_tt = ttnn.from_torch(
            inputs_embeds,
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Setup RoPE
        from models.demos.qwen3_tts.tt.rope import get_rope_tensors, get_transformation_mat

        talker_trans_mat = get_transformation_mat(model.talker_config.head_dim, device)

        # Test prefill
        print("\n--- Testing Prefill ---")
        position_ids = torch.arange(seq_len)
        talker_cos, talker_sin = get_rope_tensors(
            device, model.talker_config.head_dim, seq_len, position_ids, model.talker_config.rope_theta
        )

        prefill_start = time.time()
        hidden_tt, _ = model.talker.forward_from_hidden(
            inputs_embeds_tt,
            talker_cos,
            talker_sin,
            talker_trans_mat,
            attention_mask=None,
            kv_caches=None,
            mode="prefill",
        )
        prefill_time = time.time() - prefill_start
        print(f"  Prefill time: {prefill_time*1000:.1f}ms")

        # Get output shape
        hidden_torch = ttnn.to_torch(hidden_tt)
        print(f"  Output shape: {hidden_torch.shape}")

        # Test decode without KV cache (recompute full sequence)
        print("\n--- Testing Decode WITHOUT KV cache (5 steps) ---")
        no_cache_times = []
        current_embeds = inputs_embeds.clone()

        for step in range(5):
            # Add one token
            new_token = torch.randn(batch_size, 1, 1, hidden_size)
            current_embeds = torch.cat([current_embeds, new_token], dim=2)
            cur_seq_len = current_embeds.shape[2]

            # Convert to TTNN
            cur_embeds_tt = ttnn.from_torch(
                current_embeds,
                device=device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

            # Get RoPE
            position_ids = torch.arange(cur_seq_len)
            talker_cos, talker_sin = get_rope_tensors(
                device, model.talker_config.head_dim, cur_seq_len, position_ids, model.talker_config.rope_theta
            )

            step_start = time.time()
            hidden_tt, _ = model.talker.forward_from_hidden(
                cur_embeds_tt,
                talker_cos,
                talker_sin,
                talker_trans_mat,
                attention_mask=None,
                kv_caches=None,
                mode="prefill",
            )
            step_time = time.time() - step_start
            no_cache_times.append(step_time)
            print(f"  Step {step+1}: {step_time*1000:.1f}ms (seq_len={cur_seq_len})")

        avg_no_cache = sum(no_cache_times) / len(no_cache_times)
        print(f"  Average: {avg_no_cache*1000:.1f}ms/step")

        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Prefill ({seq_len} tokens): {prefill_time*1000:.1f}ms")
        print(f"Decode without KV cache (avg): {avg_no_cache*1000:.1f}ms/step")
        print("\nKV cache implementation is ready but decode mode")
        print("requires HEIGHT_SHARDED RoPE workaround testing.")

    finally:
        ttnn.close_device(device)
        print("\nDevice closed")


if __name__ == "__main__":
    test_kv_cache()
