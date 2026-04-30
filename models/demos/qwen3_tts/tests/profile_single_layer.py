# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Profile a single Talker decoder layer for optimization analysis.

Run with tracy:
    python -m tracy -p -v -r --op-support-count 2600 --dump-device-data-mid-run models/demos/qwen3_tts/tests/profile_single_layer.py

This generates a CSV with op timing data.
"""

import argparse
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


def profile_layer_decode(
    device,
    layer,
    num_iterations=1,
    num_warmup=2,
    with_kv_cache: bool = False,
    max_seq: int = 256,
    cur_pos: int = 100,
    num_kv_heads: int = 8,
    num_heads: int = 16,
):
    """Profile the layer in decode mode (single token).

    When `with_kv_cache=True`, runs the trace-compatible decode path used by the
    demo: a real `[1, num_kv_heads, max_seq, head_dim]` KV cache is allocated,
    `cur_pos_tensor` and a `[1, num_heads, 1, max_seq]` `decode_attn_mask` are
    passed in, and `paged_update_cache` updates K/V at `cur_pos` before SDPA
    reads the full cache. Without the flag, the older "bare layer math" path
    runs (no cache update, single-token K/V).
    """
    label = "with-KV-cache" if with_kv_cache else "no-KV-cache"
    print(f"\n=== Profiling Decode Mode [{label}] " f"(warmup={num_warmup}, measure={num_iterations}) ===")

    batch = 1
    seq_len = 1
    hidden_size = 2048
    head_dim = 128

    x_torch = torch.randn(batch, 1, seq_len, hidden_size, dtype=torch.bfloat16)
    x = ttnn.from_torch(
        x_torch,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # RoPE tensors. For with_kv_cache, use the slice at position cur_pos so the
    # rotation matches what real decode at that token position would apply.
    rope_theta = 1000000.0
    cos, sin = compute_rope_frequencies(head_dim, max_seq, rope_theta)
    pos_idx = cur_pos if with_kv_cache else 0
    cos_slice = cos[pos_idx : pos_idx + 1, :]
    sin_slice = sin[pos_idx : pos_idx + 1, :]
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
    trans_mat = get_transformation_mat(head_dim, device)

    # Real-decode extras: KV cache, cur_pos_tensor, decode_attn_mask
    k_cache = v_cache = cur_pos_tt = decode_mask_tt = None
    if with_kv_cache:
        # KV cache pre-filled with random data up to cur_pos (rest zeros).
        k_init = torch.zeros(batch, num_kv_heads, max_seq, head_dim, dtype=torch.bfloat16)
        v_init = torch.zeros(batch, num_kv_heads, max_seq, head_dim, dtype=torch.bfloat16)
        k_init[:, :, :cur_pos, :] = torch.randn(batch, num_kv_heads, cur_pos, head_dim, dtype=torch.bfloat16)
        v_init[:, :, :cur_pos, :] = torch.randn(batch, num_kv_heads, cur_pos, head_dim, dtype=torch.bfloat16)
        k_cache = ttnn.from_torch(
            k_init,
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        v_cache = ttnn.from_torch(
            v_init,
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        cur_pos_tt = ttnn.from_torch(
            torch.tensor([cur_pos], dtype=torch.int32),
            device=device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # Mask: 0 for positions <= cur_pos, -inf beyond.
        mask = torch.full((1, num_heads, 1, max_seq), float("-inf"))
        mask[..., : cur_pos + 1] = 0.0
        decode_mask_tt = ttnn.from_torch(
            mask,
            device=device,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def run_one():
        if with_kv_cache:
            output, _ = layer.forward(
                x,
                cos_tt,
                sin_tt,
                trans_mat,
                kv_cache=(k_cache, v_cache),
                mode="decode",
                cur_pos_tensor=cur_pos_tt,
                decode_attn_mask=decode_mask_tt,
            )
        else:
            output, _ = layer.forward(x, cos_tt, sin_tt, trans_mat, mode="decode")
        ttnn.deallocate(output)

    print(f"  Warmup ({num_warmup} iter)...")
    for _ in range(num_warmup):
        run_one()
    ttnn.synchronize_device(device)

    print(f"  Running {num_iterations} iterations...")
    start = time.perf_counter()
    for _ in range(num_iterations):
        run_one()
    ttnn.synchronize_device(device)
    elapsed = time.perf_counter() - start

    print(f"  Total time: {elapsed*1000:.2f} ms")
    print(f"  Per iteration: {elapsed*1000/num_iterations:.2f} ms")

    ttnn.deallocate(x)
    ttnn.deallocate(cos_tt)
    ttnn.deallocate(sin_tt)
    ttnn.deallocate(trans_mat)
    if with_kv_cache:
        ttnn.deallocate(k_cache)
        ttnn.deallocate(v_cache)
        ttnn.deallocate(cur_pos_tt)
        ttnn.deallocate(decode_mask_tt)


def profile_layer_prefill(device, layer, seq_len=32, num_iterations=1, num_warmup=2):
    """Profile the layer in prefill mode."""
    print(f"\n=== Profiling Prefill Mode (seq_len={seq_len}, warmup={num_warmup}, measure={num_iterations}) ===")

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
    print(f"  Warmup ({num_warmup} iter)...")
    for _ in range(num_warmup):
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
    parser = argparse.ArgumentParser(description="Qwen3-TTS single-layer tracy profiling")
    parser.add_argument(
        "--mode",
        choices=["prefill", "decode", "both"],
        default="both",
        help="Which mode(s) to profile. 'both' runs prefill first then decode so the "
        "first-SdpaDecode phase-split heuristic in parse_profiler_report.py works.",
    )
    parser.add_argument(
        "--prefill-seq-len",
        type=int,
        default=128,
        help="Prefill sequence length. Must differ from decode (1) so the combined "
        "tracy CSV can be split by input-shape transition.",
    )
    parser.add_argument("--warmup", type=int, default=2, help="Warmup iterations per mode")
    parser.add_argument("--measure", type=int, default=1, help="Measurement iterations per mode")
    parser.add_argument(
        "--decode-with-kv-cache",
        action="store_true",
        help="Run the trace-compatible decode path: real KV cache, cur_pos_tensor, "
        "decode_attn_mask. Surfaces paged_update_cache + full-cache SDPA shapes.",
    )
    parser.add_argument("--max-seq", type=int, default=256, help="KV cache max sequence length")
    parser.add_argument("--cur-pos", type=int, default=100, help="Current decode position (cache fill level)")
    args = parser.parse_args()

    print("=" * 60)
    print(f"Qwen3-TTS Single Layer Profiling (mode={args.mode})")
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

        # IMPORTANT: prefill is run BEFORE decode so that downstream parsers
        # (parse_profiler_report.py) can split phases by the first SdpaDecode op.
        if args.mode in ("prefill", "both"):
            profile_layer_prefill(
                device,
                layer,
                seq_len=args.prefill_seq_len,
                num_iterations=args.measure,
                num_warmup=args.warmup,
            )

        if args.mode in ("decode", "both"):
            profile_layer_decode(
                device,
                layer,
                num_iterations=args.measure,
                num_warmup=args.warmup,
                with_kv_cache=args.decode_with_kv_cache,
                max_seq=args.max_seq,
                cur_pos=args.cur_pos,
            )

        print("\n" + "=" * 60)
        print("Profiling complete!")
        print("=" * 60)

    finally:
        print("\nClosing device...")
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
