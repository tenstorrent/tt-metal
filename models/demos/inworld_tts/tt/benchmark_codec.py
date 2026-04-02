"""Benchmark and PCC audit for Inworld TTS codec decoder with real xcodec2 weights.

Tests each block independently with real checkpoint weights, measures:
- PCC vs PyTorch reference
- Execution time
- Whether the block runs on TTNN or CPU

Usage:
    export TT_METAL_HOME=$(pwd) && export PYTHONPATH=$(pwd) && export ARCH_NAME=wormhole_b0
    source python_env/bin/activate
    python models/demos/inworld_tts/tt/benchmark_codec.py
"""

import os
import sys
import time

# Add train_venv for vector_quantize_pytorch
TRAIN_VENV = os.path.join(os.path.dirname(os.path.dirname(__file__)), "train_venv", "lib")
for p in sorted(
    [os.path.join(TRAIN_VENV, d, "site-packages") for d in os.listdir(TRAIN_VENV) if d.startswith("python")],
    reverse=True,
):
    if os.path.isdir(p):
        sys.path.append(p)

import torch
import torch.nn.functional as F

import ttnn
from models.demos.inworld_tts.reference import functional as ref
from models.demos.inworld_tts.tt.attention import TtAttention
from models.demos.inworld_tts.tt.codec_decoder import TtCodecDecoder
from models.demos.inworld_tts.tt.mlp import TtMLP
from models.demos.inworld_tts.tt.resnet_block import TtResnetBlock
from models.demos.inworld_tts.tt.transformer_block import TtTransformerBlock
from models.demos.inworld_tts.tt.vocos_backbone import TtVocosBackbone

CODEC_CKPT = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "training/vectorized_data_full/.cache/models--HKUSTAudio--xcodec2/"
    "snapshots/06071873ab345f44488d235dae3cb10b5901fd90/ckpt/epoch=4-step=1400000.ckpt",
)


def compute_pcc(a, b):
    return torch.corrcoef(torch.stack([a.flatten().float(), b.flatten().float()]))[0, 1].item()


def load_codec_weights():
    """Load real xcodec2 weights."""
    from vector_quantize_pytorch import ResidualFSQ

    print(f"Loading xcodec2 checkpoint...")
    ckpt = torch.load(CODEC_CKPT, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]

    # Build quantizer
    quantizer = ResidualFSQ(dim=2048, levels=[4, 4, 4, 4, 4, 4, 4, 4], num_quantizers=1)
    q_sd = {k.replace("generator.quantizer.", ""): v for k, v in sd.items() if k.startswith("generator.quantizer.")}
    quantizer.load_state_dict(q_sd, strict=False)
    quantizer.eval()

    # Extract decoder state dict
    decoder_sd = {}
    for k, v in sd.items():
        if k.startswith("generator.backbone."):
            decoder_sd[k.replace("generator.", "")] = v
        elif k.startswith("generator.head."):
            decoder_sd[k.replace("generator.", "")] = v
        elif k.startswith("fc_post_a."):
            decoder_sd[k] = v

    return quantizer, decoder_sd


def time_fn(fn, *args, warmup=1, repeat=3, **kwargs):
    """Time a function with warmup."""
    for _ in range(warmup):
        result = fn(*args, **kwargs)
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        times.append(time.perf_counter() - t0)
    return result, min(times), sum(times) / len(times)


def benchmark_fsq_dequantize(quantizer, seq_len=100):
    """Benchmark FSQ dequantize (CPU boundary)."""
    print("\n=== FSQ Dequantize (CPU) ===")
    torch.manual_seed(42)
    vq_codes = torch.randint(0, 65536, (1, 1, seq_len))

    def run():
        codes = vq_codes.transpose(1, 2)  # [1, T, 1]
        return quantizer.get_output_from_indices(codes)

    with torch.no_grad():
        result, t_min, t_avg = time_fn(run, warmup=2, repeat=5)
    print(f"  Output shape: {result.shape}")
    print(f"  Time: {t_min*1000:.2f}ms (min), {t_avg*1000:.2f}ms (avg)")
    print(f"  Device: CPU (expected)")
    return result


def benchmark_fc_post_a(decoder_sd, vq_emb, device):
    """Benchmark fc_post_a Linear(2048, 1024)."""
    print("\n=== fc_post_a Linear(2048, 1024) (TTNN) ===")
    w = decoder_sd["fc_post_a.weight"]
    b = decoder_sd["fc_post_a.bias"]

    # Reference
    ref_out = F.linear(vq_emb, w, b)

    # TTNN
    w_tt = ttnn.from_torch(
        w.T.unsqueeze(0).unsqueeze(0),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    b_tt = ttnn.from_torch(
        b.unsqueeze(0).unsqueeze(0).unsqueeze(0),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    x_tt = ttnn.from_torch(vq_emb.unsqueeze(0), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    cfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False
    )

    def run():
        return ttnn.linear(x_tt, w_tt, bias=b_tt, compute_kernel_config=cfg)

    tt_out, t_min, t_avg = time_fn(run, warmup=2, repeat=5)
    tt_out_torch = ttnn.to_torch(tt_out).squeeze(0)

    pcc = compute_pcc(ref_out, tt_out_torch)
    print(f"  PCC: {pcc:.6f}")
    print(f"  Time: {t_min*1000:.2f}ms (min), {t_avg*1000:.2f}ms (avg)")
    print(f"  Device: TTNN")
    return ref_out, pcc


def benchmark_attention(decoder_sd, device, seq_len=100):
    """Benchmark single attention block."""
    print("\n=== Attention (TTNN) - Layer 0 ===")
    torch.manual_seed(42)
    x = torch.randn(1, seq_len, 1024)

    # Reference
    prefix = "backbone.transformers.0.att."
    rope_cache = ref.build_rope_cache(16, 64)
    ref_out = ref.attention_forward(
        x, decoder_sd[prefix + "c_attn.weight"], decoder_sd[prefix + "c_proj.weight"], 16, rope_cache
    )

    # TTNN
    tt_attn = TtAttention(device=device, state_dict=decoder_sd, layer_num=0, state_dict_prefix="backbone.")
    x_tt = ttnn.from_torch(x.unsqueeze(0), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    def run():
        return tt_attn(x_tt)

    tt_out, t_min, t_avg = time_fn(run, warmup=1, repeat=3)
    tt_out_torch = ttnn.to_torch(tt_out).squeeze(0)

    pcc = compute_pcc(ref_out, tt_out_torch)
    print(f"  PCC: {pcc:.6f}")
    print(f"  Time: {t_min*1000:.2f}ms (min), {t_avg*1000:.2f}ms (avg)")
    print(f"  Device: TTNN (matmuls + SDPA on device, RoPE on host)")


def benchmark_mlp(decoder_sd, device, seq_len=100):
    """Benchmark single MLP block."""
    print("\n=== MLP (TTNN) - Layer 0 ===")
    torch.manual_seed(42)
    x = torch.randn(1, seq_len, 1024)

    prefix = "backbone.transformers.0.mlp."
    ref_out = ref.mlp_forward(x, decoder_sd[prefix + "fc1.weight"], decoder_sd[prefix + "fc2.weight"])

    tt_mlp = TtMLP(device=device, state_dict=decoder_sd, layer_num=0, state_dict_prefix="backbone.")
    x_tt = ttnn.from_torch(x.unsqueeze(0), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    def run():
        return tt_mlp(x_tt)

    tt_out, t_min, t_avg = time_fn(run, warmup=1, repeat=3)
    tt_out_torch = ttnn.to_torch(tt_out).squeeze(0)

    pcc = compute_pcc(ref_out, tt_out_torch)
    print(f"  PCC: {pcc:.6f}")
    print(f"  Time: {t_min*1000:.2f}ms (min), {t_avg*1000:.2f}ms (avg)")
    print(f"  Device: TTNN")


def benchmark_transformer_block(decoder_sd, device, seq_len=100):
    """Benchmark single transformer block."""
    print("\n=== TransformerBlock (TTNN) - Layer 0 ===")
    torch.manual_seed(42)
    x = torch.randn(1, seq_len, 1024)

    prefix = "backbone.transformers.0."
    weights = {
        "att_norm_weight": decoder_sd[prefix + "att_norm.weight"],
        "c_attn_weight": decoder_sd[prefix + "att.c_attn.weight"],
        "c_proj_weight": decoder_sd[prefix + "att.c_proj.weight"],
        "ffn_norm_weight": decoder_sd[prefix + "ffn_norm.weight"],
        "fc1_weight": decoder_sd[prefix + "mlp.fc1.weight"],
        "fc2_weight": decoder_sd[prefix + "mlp.fc2.weight"],
    }
    rope_cache = ref.build_rope_cache(16, 64)
    ref_out = ref.transformer_block_forward(x, weights, 16, rope_cache)

    tt_block = TtTransformerBlock(device=device, state_dict=decoder_sd, layer_num=0, state_dict_prefix="backbone.")
    x_tt = ttnn.from_torch(x.unsqueeze(0), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    def run():
        return tt_block(x_tt)

    tt_out, t_min, t_avg = time_fn(run, warmup=1, repeat=3)
    tt_out_torch = ttnn.to_torch(tt_out).squeeze(0)

    pcc = compute_pcc(ref_out, tt_out_torch)
    print(f"  PCC: {pcc:.6f}")
    print(f"  Time: {t_min*1000:.2f}ms (min), {t_avg*1000:.2f}ms (avg)")
    print(f"  Device: TTNN (RMSNorm + Attention + MLP)")


def benchmark_resnet_block(decoder_sd, device, seq_len=100):
    """Benchmark single ResnetBlock."""
    print("\n=== ResnetBlock (TTNN) - prior_net.0 ===")
    torch.manual_seed(42)
    x_bct = torch.randn(1, 1024, seq_len)

    # Reference
    rw = {}
    prefix = "backbone.prior_net.0."
    for k in [
        "norm1.weight",
        "norm1.bias",
        "conv1.weight",
        "conv1.bias",
        "norm2.weight",
        "norm2.bias",
        "conv2.weight",
        "conv2.bias",
    ]:
        short = k.replace(".", "_")
        rw[short.replace("_weight", "_weight").replace("_bias", "_bias")] = decoder_sd[prefix + k]
    # Fix key naming
    ref_weights = {
        "norm1_weight": decoder_sd[prefix + "norm1.weight"],
        "norm1_bias": decoder_sd[prefix + "norm1.bias"],
        "conv1_weight": decoder_sd[prefix + "conv1.weight"],
        "conv1_bias": decoder_sd[prefix + "conv1.bias"],
        "norm2_weight": decoder_sd[prefix + "norm2.weight"],
        "norm2_bias": decoder_sd[prefix + "norm2.bias"],
        "conv2_weight": decoder_sd[prefix + "conv2.weight"],
        "conv2_bias": decoder_sd[prefix + "conv2.bias"],
    }
    ref_out = ref.resnet_block_forward(x_bct, ref_weights)

    # TTNN: input [1, 1, T, C]
    x_ttnn = ttnn.from_torch(
        x_bct.permute(0, 2, 1).unsqueeze(0),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    tt_block = TtResnetBlock(device=device, state_dict=decoder_sd, block_prefix=prefix)

    def run():
        return tt_block(x_ttnn)

    tt_out, t_min, t_avg = time_fn(run, warmup=1, repeat=3)
    tt_out_torch = ttnn.to_torch(tt_out).squeeze(0).permute(0, 2, 1)

    pcc = compute_pcc(ref_out, tt_out_torch)
    print(f"  PCC: {pcc:.6f}")
    print(f"  Time: {t_min*1000:.2f}ms (min), {t_avg*1000:.2f}ms (avg)")
    print(f"  Device: TTNN (Conv1d + SiLU on device, GroupNorm on host)")


def benchmark_vocos_backbone(decoder_sd, device, seq_len=100):
    """Benchmark full VocosBackbone (12 layers)."""
    print("\n=== VocosBackbone (12 layers) (TTNN) ===")
    torch.manual_seed(42)
    x = torch.randn(1, seq_len, 1024)

    # Reference
    backbone_weights = ref.extract_backbone_weights(decoder_sd)
    ref_out = ref.vocos_backbone_forward(x, backbone_weights, 16, 64, 12)

    # TTNN
    tt_backbone = TtVocosBackbone(device=device, state_dict=decoder_sd, state_dict_prefix="backbone.")

    def run():
        return tt_backbone(x)

    tt_out, t_min, t_avg = time_fn(run, warmup=1, repeat=3)
    tt_out_torch = ttnn.to_torch(tt_out).squeeze(0)

    pcc = compute_pcc(ref_out, tt_out_torch)
    print(f"  PCC: {pcc:.6f}")
    print(f"  Time: {t_min*1000:.2f}ms (min), {t_avg*1000:.2f}ms (avg)")
    print(f"  Device: TTNN (embed Conv1d + 4 ResnetBlocks + 12 TransformerBlocks + LayerNorm)")


def benchmark_istft_head(decoder_sd, device, seq_len=100):
    """Benchmark ISTFTHead (CPU boundary)."""
    print("\n=== ISTFTHead (CPU) ===")
    torch.manual_seed(42)
    x = torch.randn(1, seq_len, 1024)

    out_w = decoder_sd["head.out.weight"]
    out_b = decoder_sd["head.out.bias"]

    def run():
        return ref.istft_head_forward(x, out_w, out_b)

    result, t_min, t_avg = time_fn(run, warmup=2, repeat=5)
    print(f"  Output shape: {result.shape}")
    print(f"  Time: {t_min*1000:.2f}ms (min), {t_avg*1000:.2f}ms (avg)")
    print(f"  Device: CPU (FFT operations)")


def benchmark_full_decoder(quantizer, decoder_sd, device, seq_len=100):
    """Benchmark full codec decoder pipeline."""
    print("\n=== Full Codec Decoder (end-to-end) ===")
    torch.manual_seed(42)
    vq_codes = torch.randint(0, 65536, (1, seq_len))

    # Reference
    backbone_weights = ref.extract_backbone_weights(decoder_sd)
    istft_weights = ref.extract_istft_weights(decoder_sd)

    def run_ref():
        return ref.codec_decoder_forward(
            vq_codes,
            quantizer,
            decoder_sd["fc_post_a.weight"],
            decoder_sd["fc_post_a.bias"],
            backbone_weights,
            istft_weights,
        )

    with torch.no_grad():
        ref_out, ref_min, ref_avg = time_fn(run_ref, warmup=1, repeat=3)

    # TTNN
    tt_decoder = TtCodecDecoder(
        device=device,
        state_dict=decoder_sd,
        quantizer=quantizer,
        backbone_prefix="backbone.",
        head_prefix="head.",
    )

    def run_ttnn():
        return tt_decoder(vq_codes)

    with torch.no_grad():
        tt_out, tt_min, tt_avg = time_fn(run_ttnn, warmup=1, repeat=3)

    pcc = compute_pcc(
        ref_out.squeeze(),
        torch.tensor(tt_out.squeeze().numpy()) if not isinstance(tt_out, torch.Tensor) else tt_out.squeeze(),
    )
    print(f"  PCC: {pcc:.6f}")
    print(f"  Reference time: {ref_min*1000:.2f}ms (min), {ref_avg*1000:.2f}ms (avg)")
    print(f"  TTNN time:      {tt_min*1000:.2f}ms (min), {tt_avg*1000:.2f}ms (avg)")
    print(f"  Speedup:        {ref_min/tt_min:.2f}x")


def print_placement_summary():
    """Print which blocks run on TTNN vs CPU."""
    print("\n" + "=" * 70)
    print("BLOCK PLACEMENT SUMMARY")
    print("=" * 70)
    print(f"{'Block':<35} {'Device':<10} {'Notes'}")
    print("-" * 70)
    print(f"{'FSQ Dequantize':<35} {'CPU':<10} {'Codebook lookup (expected)'}")
    print(f"{'fc_post_a Linear(2048,1024)':<35} {'TTNN':<10} {'ttnn.linear'}")
    print(f"{'Embed Conv1d(1024,1024,k=7)':<35} {'TTNN':<10} {'ttnn.conv1d (BLOCK_SHARDED)'}")
    print(f"{'ResnetBlock GroupNorm(32)':<35} {'CPU':<10} {'Host roundtrip (F.group_norm)'}")
    print(f"{'ResnetBlock SiLU':<35} {'TTNN':<10} {'ttnn.silu'}")
    print(f"{'ResnetBlock Conv1d(1024,1024,k=3)':<35} {'TTNN':<10} {'ttnn.conv1d (BLOCK_SHARDED)'}")
    print(f"{'ResnetBlock Residual Add':<35} {'TTNN':<10} {'ttnn.add'}")
    print(f"{'TransformerBlock RMSNorm':<35} {'TTNN':<10} {'ttnn.rms_norm'}")
    print(f"{'TransformerBlock Attention QKV':<35} {'TTNN':<10} {'ttnn.linear (fused QKV)'}")
    print(f"{'TransformerBlock Attention RoPE':<35} {'CPU':<10} {'Host (interleaved pairs, 5D reshape)'}")
    print(f"{'TransformerBlock Attention SDPA':<35} {'TTNN':<10} {'ttnn.transformer.scaled_dot_product_attention'}")
    print(f"{'TransformerBlock Attention OutProj':<35} {'TTNN':<10} {'ttnn.linear'}")
    print(f"{'TransformerBlock MLP fc1':<35} {'TTNN':<10} {'ttnn.linear'}")
    print(f"{'TransformerBlock MLP SiLU':<35} {'TTNN':<10} {'ttnn.silu'}")
    print(f"{'TransformerBlock MLP fc2':<35} {'TTNN':<10} {'ttnn.linear'}")
    print(f"{'Final LayerNorm':<35} {'TTNN':<10} {'ttnn.layer_norm'}")
    print(f"{'ISTFTHead Linear(1024,1282)':<35} {'CPU':<10} {'Before FFT (could move to TTNN)'}")
    print(f"{'ISTFTHead ISTFT (FFT)':<35} {'CPU':<10} {'FFT operations (expected)'}")
    print("-" * 70)
    ttnn_count = 13
    cpu_count = 4
    print(f"TTNN: {ttnn_count} ops  |  CPU: {cpu_count} ops (FSQ, GroupNorm, RoPE, ISTFT)")
    print()


def main():
    device = ttnn.open_device(device_id=0, l1_small_size=16384)
    try:
        quantizer, decoder_sd = load_codec_weights()

        seq_len = 100  # 100 tokens = 2 seconds of audio

        print(f"\nBenchmarking with seq_len={seq_len} ({seq_len/50:.1f}s of audio)")
        print("=" * 70)

        # Individual blocks
        with torch.no_grad():
            vq_emb = benchmark_fsq_dequantize(quantizer, seq_len)
            benchmark_fc_post_a(decoder_sd, vq_emb, device)
            benchmark_attention(decoder_sd, device, seq_len)
            benchmark_mlp(decoder_sd, device, seq_len)
            benchmark_transformer_block(decoder_sd, device, seq_len)

        # Need device reset between conv tests to avoid L1 fragmentation
        ttnn.close_device(device)
        device = ttnn.open_device(device_id=0, l1_small_size=16384)

        with torch.no_grad():
            benchmark_resnet_block(decoder_sd, device, seq_len)

        ttnn.close_device(device)
        device = ttnn.open_device(device_id=0, l1_small_size=16384)

        with torch.no_grad():
            benchmark_istft_head(decoder_sd, device, seq_len)
            benchmark_vocos_backbone(decoder_sd, device, seq_len)

        ttnn.close_device(device)
        device = ttnn.open_device(device_id=0, l1_small_size=16384)

        with torch.no_grad():
            benchmark_full_decoder(quantizer, decoder_sd, device, seq_len)

        # Summary
        print_placement_summary()

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
