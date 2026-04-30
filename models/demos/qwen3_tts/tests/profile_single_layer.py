# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Profile a single Qwen3-TTS decoder layer for optimization analysis.

Supports both sub-blocks via --block {talker,code_predictor}:
    Talker         hidden=2048, intermediate=6144, 28 layers, runs once per frame.
    CodePredictor  hidden=1024, intermediate=3072,  5 layers, runs 14× per frame
                   (one decode trace per non-first codebook).

Run with tracy (one block per invocation):
    python -m tracy -p -v -r --op-support-count 2600 --dump-device-data-mid-run \
        models/demos/qwen3_tts/tests/profile_single_layer.py --block talker

    python -m tracy -p -v -r --op-support-count 2600 --dump-device-data-mid-run \
        models/demos/qwen3_tts/tests/profile_single_layer.py --block code_predictor \
        --prefill-seq-len 2

Generates a CSV with op timing data. parse_profiler_report.py splits prefill vs decode
by the first SdpaDecode op — that's why prefill runs *before* decode here.
"""

import argparse
import time
from dataclasses import dataclass

import torch

import ttnn
from models.demos.qwen3_tts.tt.decoder_layer import DecoderLayer
from models.demos.qwen3_tts.tt.model_config import Qwen3TTSCodePredictorConfig, Qwen3TTSTalkerConfig
from models.demos.qwen3_tts.tt.rope import compute_rope_frequencies, get_transformation_mat


@dataclass
class BlockProfile:
    name: str
    layer_prefix: str
    hidden_size: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    intermediate_size: int
    num_hidden_layers: int
    rms_norm_eps: float


def make_block_profile(block: str) -> BlockProfile:
    if block == "talker":
        cfg = Qwen3TTSTalkerConfig()
        return BlockProfile(
            name="talker",
            layer_prefix="talker.model",
            hidden_size=cfg.hidden_size,
            num_heads=cfg.num_attention_heads,
            num_kv_heads=cfg.num_key_value_heads,
            head_dim=cfg.head_dim,
            intermediate_size=cfg.intermediate_size,
            num_hidden_layers=cfg.num_hidden_layers,
            rms_norm_eps=cfg.rms_norm_eps,
        )
    if block == "code_predictor":
        cfg = Qwen3TTSCodePredictorConfig()
        return BlockProfile(
            name="code_predictor",
            layer_prefix="talker.code_predictor.model",
            hidden_size=cfg.hidden_size,
            num_heads=cfg.num_attention_heads,
            num_kv_heads=cfg.num_key_value_heads,
            head_dim=cfg.head_dim,
            intermediate_size=cfg.intermediate_size,
            num_hidden_layers=cfg.num_hidden_layers,
            rms_norm_eps=cfg.rms_norm_eps,
        )
    raise ValueError(f"Unknown block {block!r}; expected 'talker' or 'code_predictor'")


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


def create_single_layer(device, state_dict, prof: BlockProfile, layer_idx: int = 0):
    """Create a single decoder layer for the given sub-block."""
    print(f"Creating {prof.name} decoder layer {layer_idx} (prefix={prof.layer_prefix})...")
    return DecoderLayer(
        device=device,
        hidden_size=prof.hidden_size,
        num_heads=prof.num_heads,
        num_kv_heads=prof.num_kv_heads,
        head_dim=prof.head_dim,
        intermediate_size=prof.intermediate_size,
        state_dict=state_dict,
        layer_idx=layer_idx,
        layer_prefix=prof.layer_prefix,
        rms_norm_eps=prof.rms_norm_eps,
        weight_dtype=ttnn.bfloat16,
    )


def profile_layer_decode(
    device,
    layer,
    prof: BlockProfile,
    num_iterations: int = 1,
    num_warmup: int = 2,
    with_kv_cache: bool = False,
    max_seq: int = 256,
    cur_pos: int = 100,
):
    """Profile the layer in decode mode (single token).

    When ``with_kv_cache=True``, runs the trace-compatible decode path used by the
    demo: a real ``[1, num_kv_heads, max_seq, head_dim]`` KV cache is allocated,
    ``cur_pos_tensor`` and a ``[1, num_heads, 1, max_seq]`` ``decode_attn_mask`` are
    passed in, and ``paged_update_cache`` updates K/V at ``cur_pos`` before SDPA
    reads the full cache. Without the flag, the older "bare layer math" path runs.
    """
    label = "with-KV-cache" if with_kv_cache else "no-KV-cache"
    print(f"\n=== Profiling {prof.name} Decode [{label}] (warmup={num_warmup}, measure={num_iterations}) ===")

    batch = 1
    seq_len = 1

    x_torch = torch.randn(batch, 1, seq_len, prof.hidden_size, dtype=torch.bfloat16)
    x = ttnn.from_torch(
        x_torch, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    # RoPE tensors at the cur_pos slice so the rotation matches a real decode step.
    rope_theta = 1000000.0
    cos, sin = compute_rope_frequencies(prof.head_dim, max_seq, rope_theta)
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
    trans_mat = get_transformation_mat(prof.head_dim, device)

    k_cache = v_cache = cur_pos_tt = decode_mask_tt = None
    if with_kv_cache:
        k_init = torch.zeros(batch, prof.num_kv_heads, max_seq, prof.head_dim, dtype=torch.bfloat16)
        v_init = torch.zeros(batch, prof.num_kv_heads, max_seq, prof.head_dim, dtype=torch.bfloat16)
        k_init[:, :, :cur_pos, :] = torch.randn(batch, prof.num_kv_heads, cur_pos, prof.head_dim, dtype=torch.bfloat16)
        v_init[:, :, :cur_pos, :] = torch.randn(batch, prof.num_kv_heads, cur_pos, prof.head_dim, dtype=torch.bfloat16)
        k_cache = ttnn.from_torch(
            k_init, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        v_cache = ttnn.from_torch(
            v_init, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        cur_pos_tt = ttnn.from_torch(
            torch.tensor([cur_pos], dtype=torch.int32),
            device=device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        mask = torch.full((1, prof.num_heads, 1, max_seq), float("-inf"))
        mask[..., : cur_pos + 1] = 0.0
        decode_mask_tt = ttnn.from_torch(
            mask, device=device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
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


def profile_layer_prefill(
    device, layer, prof: BlockProfile, seq_len: int = 32, num_iterations: int = 1, num_warmup: int = 2
):
    """Profile the layer in prefill mode."""
    print(f"\n=== Profiling {prof.name} Prefill (seq_len={seq_len}, warmup={num_warmup}, measure={num_iterations}) ===")

    batch = 1
    x_torch = torch.randn(batch, 1, seq_len, prof.hidden_size, dtype=torch.bfloat16)
    x = ttnn.from_torch(
        x_torch, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    rope_theta = 1000000.0
    cos, sin = compute_rope_frequencies(prof.head_dim, seq_len, rope_theta)
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
    trans_mat = get_transformation_mat(prof.head_dim, device)

    print(f"  Warmup ({num_warmup} iter)...")
    for _ in range(num_warmup):
        output, _ = layer.forward(x, cos_tt, sin_tt, trans_mat, mode="prefill")
        ttnn.deallocate(output)
    ttnn.synchronize_device(device)

    print(f"  Running {num_iterations} iterations...")
    start = time.perf_counter()
    for _ in range(num_iterations):
        output, _ = layer.forward(x, cos_tt, sin_tt, trans_mat, mode="prefill")
        ttnn.deallocate(output)
    ttnn.synchronize_device(device)
    elapsed = time.perf_counter() - start

    print(f"  Total time: {elapsed*1000:.2f} ms")
    print(f"  Per iteration: {elapsed*1000/num_iterations:.2f} ms")

    ttnn.deallocate(x)
    ttnn.deallocate(cos_tt)
    ttnn.deallocate(sin_tt)
    ttnn.deallocate(trans_mat)


def main():
    parser = argparse.ArgumentParser(description="Qwen3-TTS single-layer tracy profiling")
    parser.add_argument(
        "--block",
        choices=["talker", "code_predictor"],
        default="talker",
        help="Sub-block to profile (talker: hidden=2048, 28L; code_predictor: hidden=1024, 5L).",
    )
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
        "tracy CSV can be split by input-shape transition. For code_predictor, the "
        "in-demo CP prefill uses seq_len=2 (past_hidden + code0).",
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
    print(f"Qwen3-TTS Single Layer Profiling [block={args.block} mode={args.mode}]")
    print("=" * 60)

    prof = make_block_profile(args.block)
    print(
        f"  hidden={prof.hidden_size}  intermediate={prof.intermediate_size}  "
        f"heads={prof.num_heads}/{prof.num_kv_heads}  head_dim={prof.head_dim}  "
        f"num_hidden_layers={prof.num_hidden_layers}"
    )

    state_dict = load_model_weights()

    print("\nOpening device...")
    device = ttnn.open_device(device_id=0, l1_small_size=16384)
    device.enable_program_cache()

    try:
        layer = create_single_layer(device, state_dict, prof, layer_idx=0)

        # Prefill BEFORE decode so parse_profiler_report.py can split phases.
        if args.mode in ("prefill", "both"):
            profile_layer_prefill(
                device, layer, prof, seq_len=args.prefill_seq_len, num_iterations=args.measure, num_warmup=args.warmup
            )

        if args.mode in ("decode", "both"):
            profile_layer_decode(
                device,
                layer,
                prof,
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
