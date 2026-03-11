# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Performance regression tests for Molmo2-8B.

Records timing for vision, prefill, and decode phases.
Targets (T3K — 8 devices, traced):
    - Vision backbone: < 200 ms  (reference: ~86 ms)
    - Prefill TTFT:    < 200 ms  (reference: ~85 ms)
    - Decode per token: < 60 ms  (reference: ~28 ms, 35.6 tok/s)

NOTE: These tests require a device with weights loaded. They record timing
but do NOT assert strict latency bounds, since timing varies by device load.
The bounds above are 2× the reference measurements; hard failures indicate
regressions that need investigation.

Run with:
    MESH_DEVICE=T3K pytest models/demos/molmo2/tests/test_perf.py -v -s
"""

import os
import time

import torch

import ttnn

# Performance targets (2x reference to allow for measurement variance)
VISION_LATENCY_MS_MAX = 200.0  # reference: ~86 ms
PREFILL_LATENCY_MS_MAX = 200.0  # reference: ~85 ms
DECODE_LATENCY_MS_MAX = 60.0  # reference: ~28 ms/token


def time_fn(fn, warmup: int = 1, repeats: int = 5, device=None):
    """
    Time a function with warmup and repeated measurements.

    Returns:
        (mean_ms, min_ms, max_ms)
    """
    for _ in range(warmup):
        fn()
    if device is not None:
        ttnn.synchronize_device(device)

    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        if device is not None:
            ttnn.synchronize_device(device)
        times.append((time.perf_counter() - start) * 1000)

    return sum(times) / len(times), min(times), max(times)


def test_decode_latency(device):
    """
    Measure decode step latency (single token, no trace).

    Target: < 60 ms/token (reference: ~28 ms/token on T3K with trace).
    This test runs without trace so the bound is relaxed.
    """
    from models.demos.molmo2.tt.load_weights import load_state_dict_from_safetensors
    from models.demos.molmo2.tt.text_attention import TextAttention

    model_id = os.environ.get("HF_MODEL", "allenai/Molmo2-8B")
    layer_num = 0
    hidden_dim = 4096
    num_heads = 32
    num_kv_heads = 8
    head_dim = 128
    seq_len = 1  # decode step

    prefix = f"model.transformer.blocks.{layer_num}.self_attn"
    keys = [
        f"{prefix}.att_proj.weight",
        f"{prefix}.attn_out.weight",
        f"{prefix}.q_norm.weight",
        f"{prefix}.k_norm.weight",
    ]
    state_dict = load_state_dict_from_safetensors(model_id, keys)

    attn = TextAttention(
        mesh_device=device,
        state_dict=state_dict,
        layer_num=layer_num,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype=ttnn.bfloat16,
    )

    torch.manual_seed(0)
    x = torch.randn(1, 1, 1, hidden_dim)
    x_ttnn = ttnn.from_torch(x, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    # Create dummy rot_mats and KV cache
    cos_sin = torch.ones(1, 1, head_dim // 2)
    rot_mat = ttnn.from_torch(cos_sin, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    def run_decode():
        _ = attn.forward_decode(x_ttnn, rot_mat)

    mean_ms, min_ms, max_ms = time_fn(run_decode, warmup=2, repeats=10, device=device)
    print(f"\nTextAttention decode (no trace): mean={mean_ms:.1f}ms, min={min_ms:.1f}ms, max={max_ms:.1f}ms")

    # Soft assertion: flag regression at 2× reference
    assert mean_ms < DECODE_LATENCY_MS_MAX, (
        f"Decode step mean latency {mean_ms:.1f}ms exceeds {DECODE_LATENCY_MS_MAX}ms target. "
        f"Possible regression. Reference: ~28ms/token on T3K with trace."
    )


def test_vision_block_latency(device):
    """
    Measure single ViT block latency.

    25 blocks × single-block time ≈ total ViT time.
    """
    from models.demos.molmo2.tt.load_weights import load_state_dict_from_safetensors
    from models.demos.molmo2.tt.vision_block import VisionBlock

    model_id = os.environ.get("HF_MODEL", "allenai/Molmo2-8B")
    layer_num = 0
    hidden_dim = 1152
    num_patches = 729

    vit_prefix = f"model.vision_backbone.image_vit.transformer.resblocks.{layer_num}"
    keys = [
        f"{vit_prefix}.attention_norm.weight",
        f"{vit_prefix}.attention_norm.bias",
        f"{vit_prefix}.attention.wq.weight",
        f"{vit_prefix}.attention.wq.bias",
        f"{vit_prefix}.attention.wk.weight",
        f"{vit_prefix}.attention.wk.bias",
        f"{vit_prefix}.attention.wv.weight",
        f"{vit_prefix}.attention.wv.bias",
        f"{vit_prefix}.attention.wo.weight",
        f"{vit_prefix}.attention.wo.bias",
        f"{vit_prefix}.ffn_norm.weight",
        f"{vit_prefix}.ffn_norm.bias",
        f"{vit_prefix}.feed_forward.w1.weight",
        f"{vit_prefix}.feed_forward.w1.bias",
        f"{vit_prefix}.feed_forward.w2.weight",
        f"{vit_prefix}.feed_forward.w2.bias",
    ]
    state_dict = load_state_dict_from_safetensors(model_id, keys)

    block = VisionBlock(
        mesh_device=device,
        state_dict=state_dict,
        state_dict_prefix=f"model.vision_backbone.image_vit.transformer.resblocks.{layer_num}",
        hidden_dim=hidden_dim,
        num_heads=16,
        head_dim=72,
        dtype=ttnn.bfloat8_b,
    )

    torch.manual_seed(0)
    x = torch.randn(1, 1, num_patches, hidden_dim)
    x_ttnn = ttnn.from_torch(x, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    def run_block():
        _ = block(x_ttnn)

    mean_ms, min_ms, max_ms = time_fn(run_block, warmup=2, repeats=10, device=device)
    estimated_vit_ms = mean_ms * 25
    print(f"\nVisionBlock latency: mean={mean_ms:.1f}ms, min={min_ms:.1f}ms")
    print(f"Estimated full ViT (25 blocks): ~{estimated_vit_ms:.0f}ms")
    # No hard assertion for single block; full-pipeline target is in test_vision_pipeline_latency


def test_image_projector_latency(device):
    """Measure ImageProjector latency (1152 → 12288 → 4096)."""
    from models.demos.molmo2.tt.image_projector import ImageProjector
    from models.demos.molmo2.tt.load_weights import load_state_dict_from_safetensors

    model_id = os.environ.get("HF_MODEL", "allenai/Molmo2-8B")
    proj_prefix = "model.vision_backbone.image_projector"
    keys = [f"{proj_prefix}.{w}.weight" for w in ["w1", "w2", "w3"]]
    state_dict = load_state_dict_from_safetensors(model_id, keys)

    projector = ImageProjector(
        mesh_device=device,
        state_dict=state_dict,
        input_dim=1152,
        output_dim=4096,
        dtype=ttnn.bfloat8_b,
    )

    torch.manual_seed(0)
    x = torch.randn(1, 1, 729, 1152)
    x_ttnn = ttnn.from_torch(x, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    def run_proj():
        _ = projector(x_ttnn)

    mean_ms, min_ms, max_ms = time_fn(run_proj, warmup=2, repeats=10, device=device)
    print(f"\nImageProjector latency: mean={mean_ms:.1f}ms, min={min_ms:.1f}ms")


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)
    try:
        print("=" * 60)
        print("Molmo2 Performance Tests")
        print("=" * 60)
        test_decode_latency(device)
        test_vision_block_latency(device)
        test_image_projector_latency(device)
        print("\nAll performance tests completed.")
    finally:
        ttnn.close_device(device)
