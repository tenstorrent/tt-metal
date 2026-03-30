# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Profile a single Talker decoder layer for optimization analysis.

Run with tracy:
    python -m tracy -p -v -r models/demos/qwen3_tts/tests/profile_single_layer.py

This generates a CSV with op timing data.
"""

import time

import torch

import ttnn
from models.demos.qwen3_tts.tt.decoder_layer import DecoderLayer
from models.demos.qwen3_tts.tt.rope import compute_rope_frequencies, get_transformation_mat


def load_model_weights():
    """Load model weights from HuggingFace."""
    from pathlib import Path

    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    print("Loading model weights...")
    model_path = Path(snapshot_download("Qwen/Qwen3-TTS-12Hz-1.7B-Base", allow_patterns=["*.safetensors"]))

    state_dict = {}
    for f in model_path.glob("*.safetensors"):
        if "speech_tokenizer" not in str(f):
            state_dict.update(load_file(f))
    print(f"  Loaded {len(state_dict)} tensors")
    return state_dict


def create_single_layer(device, state_dict, layer_idx=0):
    """Create a single Talker decoder layer."""
    # Talker config
    hidden_size = 2048
    num_heads = 16
    num_kv_heads = 8
    head_dim = 128
    intermediate_size = 6144
    rms_norm_eps = 1e-6

    print(f"Creating decoder layer {layer_idx}...")
    layer = DecoderLayer(
        device=device,
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        intermediate_size=intermediate_size,
        state_dict=state_dict,
        layer_idx=layer_idx,
        layer_prefix="talker.model",
        rms_norm_eps=rms_norm_eps,
        weight_dtype=ttnn.bfloat16,
    )
    return layer


def profile_layer_decode(device, layer, num_iterations=10):
    """Profile the layer in decode mode (single token)."""
    print(f"\n=== Profiling Decode Mode ({num_iterations} iterations) ===")

    # Decode mode: single token [1, 1, 1, 2048]
    batch = 1
    seq_len = 1
    hidden_size = 2048
    head_dim = 128
    max_seq = 256

    # Create input tensor
    x_torch = torch.randn(batch, 1, seq_len, hidden_size, dtype=torch.bfloat16)
    x = ttnn.from_torch(
        x_torch,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Create RoPE tensors
    rope_theta = 1000000.0
    cos, sin = compute_rope_frequencies(head_dim, max_seq, rope_theta)

    # For decode, we need just position 0
    cos_slice = cos[0:1, :]
    sin_slice = sin[0:1, :]

    cos_tt = ttnn.from_torch(
        cos_slice.unsqueeze(0).unsqueeze(0).to(torch.bfloat16),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    sin_tt = ttnn.from_torch(
        sin_slice.unsqueeze(0).unsqueeze(0).to(torch.bfloat16),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Transformation matrix for RoPE
    trans_mat = get_transformation_mat(head_dim, device)

    # Warmup
    print("  Warmup...")
    for _ in range(3):
        output, _ = layer.forward(
            x,
            cos_tt,
            sin_tt,
            trans_mat,
            mode="decode",
        )
        ttnn.deallocate(output)

    # Synchronize before timing
    ttnn.synchronize_device(device)

    # Profile
    print(f"  Running {num_iterations} iterations...")
    start = time.perf_counter()
    for _ in range(num_iterations):
        output, _ = layer.forward(
            x,
            cos_tt,
            sin_tt,
            trans_mat,
            mode="decode",
        )
        ttnn.deallocate(output)

    ttnn.synchronize_device(device)
    elapsed = time.perf_counter() - start

    print(f"  Total time: {elapsed*1000:.2f} ms")
    print(f"  Per iteration: {elapsed*1000/num_iterations:.2f} ms")

    # Cleanup
    ttnn.deallocate(x)
    ttnn.deallocate(cos_tt)
    ttnn.deallocate(sin_tt)
    ttnn.deallocate(trans_mat)


def profile_layer_prefill(device, layer, seq_len=32, num_iterations=5):
    """Profile the layer in prefill mode."""
    print(f"\n=== Profiling Prefill Mode (seq_len={seq_len}, {num_iterations} iterations) ===")

    batch = 1
    hidden_size = 2048
    head_dim = 128

    # Create input tensor
    x_torch = torch.randn(batch, 1, seq_len, hidden_size, dtype=torch.bfloat16)
    x = ttnn.from_torch(
        x_torch,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Create RoPE tensors
    rope_theta = 1000000.0
    cos, sin = compute_rope_frequencies(head_dim, seq_len, rope_theta)

    cos_tt = ttnn.from_torch(
        cos.unsqueeze(0).unsqueeze(0).to(torch.bfloat16),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    sin_tt = ttnn.from_torch(
        sin.unsqueeze(0).unsqueeze(0).to(torch.bfloat16),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Transformation matrix for RoPE
    trans_mat = get_transformation_mat(head_dim, device)

    # Warmup
    print("  Warmup...")
    for _ in range(2):
        output, _ = layer.forward(
            x,
            cos_tt,
            sin_tt,
            trans_mat,
            mode="prefill",
        )
        ttnn.deallocate(output)

    # Synchronize before timing
    ttnn.synchronize_device(device)

    # Profile
    print(f"  Running {num_iterations} iterations...")
    start = time.perf_counter()
    for _ in range(num_iterations):
        output, _ = layer.forward(
            x,
            cos_tt,
            sin_tt,
            trans_mat,
            mode="prefill",
        )
        ttnn.deallocate(output)

    ttnn.synchronize_device(device)
    elapsed = time.perf_counter() - start

    print(f"  Total time: {elapsed*1000:.2f} ms")
    print(f"  Per iteration: {elapsed*1000/num_iterations:.2f} ms")

    # Cleanup
    ttnn.deallocate(x)
    ttnn.deallocate(cos_tt)
    ttnn.deallocate(sin_tt)
    ttnn.deallocate(trans_mat)


def main():
    print("=" * 60)
    print("Qwen3-TTS Single Layer Profiling")
    print("=" * 60)

    # Load weights
    state_dict = load_model_weights()

    # Open device
    print("\nOpening device...")
    device = ttnn.open_device(device_id=0, l1_small_size=16384)
    device.enable_program_cache()

    try:
        # Create single layer
        layer = create_single_layer(device, state_dict, layer_idx=0)

        # Profile decode mode
        profile_layer_decode(device, layer, num_iterations=20)

        # Profile prefill mode
        profile_layer_prefill(device, layer, seq_len=32, num_iterations=10)

        print("\n" + "=" * 60)
        print("Profiling complete!")
        print("=" * 60)

    finally:
        print("\nClosing device...")
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
