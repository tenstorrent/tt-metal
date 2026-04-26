# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
PCC Comparison: TTNN vs Official qwen_tts Implementation

Uses extracted tensors from official qwen_tts to verify TTNN implementation.
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

import ttnn


def compute_pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute Pearson Correlation Coefficient between two tensors."""
    a = a.flatten().float()
    b = b.flatten().float()

    a_mean = a.mean()
    b_mean = b.mean()

    a_centered = a - a_mean
    b_centered = b - b_mean

    cov = (a_centered * b_centered).sum()
    std_a = torch.sqrt((a_centered**2).sum())
    std_b = torch.sqrt((b_centered**2).sum())

    if std_a == 0 or std_b == 0:
        return 0.0

    pcc = cov / (std_a * std_b)
    return pcc.item()


def test_rmsnorm_pcc(device, prefill_tensors, weights):
    """Test RMSNorm using extracted talker_norm input/output."""
    print("\n" + "=" * 80)
    print("TEST: RMSNorm (Talker Final Norm)")
    print("=" * 80)

    # Get reference input/output
    ref_input = prefill_tensors["talker_norm_input"]  # [1, 111, 2048]
    ref_output = prefill_tensors["talker_norm_output"]  # [1, 111, 2048]
    weight = weights["talker.norm.weight"]  # [2048]

    print(f"  Input shape: {ref_input.shape}")
    print(f"  Weight shape: {weight.shape}")

    # Pad sequence to multiple of 32 for TTNN
    batch, seq, hidden = ref_input.shape
    pad_seq = ((seq + 31) // 32) * 32
    padding_needed = pad_seq - seq

    x_padded = F.pad(ref_input, (0, 0, 0, padding_needed))  # [1, 128, 2048]
    print(f"  Padded shape: {x_padded.shape}")

    # Convert to TTNN format [1, 1, seq, hidden]
    x_4d = x_padded.unsqueeze(0)  # [1, 1, 128, 2048]
    x_tt = ttnn.from_torch(x_4d, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Weight needs [1, 1, 32, hidden] for TTNN
    weight_4d = weight.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    weight_4d = weight_4d.expand(1, 1, 32, hidden).contiguous()
    weight_tt = ttnn.from_torch(weight_4d, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Run TTNN RMSNorm
    eps = 1e-6
    ttnn_output = ttnn.rms_norm(x_tt, epsilon=eps, weight=weight_tt)
    ttnn_output = ttnn.to_torch(ttnn_output).squeeze(0)  # [1, 128, 2048]

    # Remove padding
    ttnn_output_unpadded = ttnn_output[:, :seq, :]

    # Compare
    pcc = compute_pcc(ref_output, ttnn_output_unpadded)
    print(f"  Reference output stats: mean={ref_output.mean():.4f}, std={ref_output.std():.4f}")
    print(f"  TTNN output stats: mean={ttnn_output_unpadded.mean():.4f}, std={ttnn_output_unpadded.std():.4f}")
    print(f"  PCC: {pcc:.6f}")

    status = "PASS" if pcc > 0.99 else "FAIL"
    print(f"  Status: {status}")

    return pcc


def test_embedding_pcc(device, prefill_tensors, weights):
    """Test embeddings using extracted text_embedding input/output."""
    print("\n" + "=" * 80)
    print("TEST: Text Embedding Lookup")
    print("=" * 80)

    # Get reference
    ref_input = prefill_tensors["text_embedding_input"]  # [1, 43]
    ref_output = prefill_tensors["text_embedding_output"]  # [1, 43, 2048]
    embedding_weight = weights["talker.text_embedding.weight"]  # [151936, 2048]

    print(f"  Input IDs shape: {ref_input.shape}")
    print(f"  Embedding weight shape: {embedding_weight.shape}")

    # Run PyTorch embedding lookup
    pytorch_output = F.embedding(ref_input, embedding_weight)

    # Compare PyTorch with reference
    pcc_pytorch = compute_pcc(ref_output, pytorch_output)
    print(f"  PyTorch embedding PCC vs official: {pcc_pytorch:.6f}")

    # Run TTNN embedding
    # Pad to multiple of 32
    batch, seq = ref_input.shape
    pad_seq = ((seq + 31) // 32) * 32
    padding_needed = pad_seq - seq

    input_padded = F.pad(ref_input, (0, padding_needed))  # [1, 64]
    input_tt = ttnn.from_torch(input_padded, dtype=ttnn.uint32, device=device)

    # Embedding weight on device
    weight_tt = ttnn.from_torch(
        embedding_weight.unsqueeze(0).unsqueeze(0), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )

    # TTNN embedding lookup
    ttnn_output = ttnn.embedding(input_tt, weight_tt)
    ttnn_output = ttnn.to_torch(ttnn_output)  # [1, 64, 2048]
    ttnn_output_unpadded = ttnn_output[:, :seq, :]

    pcc_ttnn = compute_pcc(ref_output, ttnn_output_unpadded)
    print(f"  TTNN embedding PCC vs official: {pcc_ttnn:.6f}")

    status = "PASS" if pcc_ttnn > 0.99 else "FAIL"
    print(f"  Status: {status}")

    return pcc_ttnn


def test_mlp_pcc(device, weights):
    """Test MLP (SwiGLU) using layer 0 weights."""
    print("\n" + "=" * 80)
    print("TEST: MLP (SwiGLU)")
    print("=" * 80)

    # Get weights
    gate_weight = weights["talker.layer_0.mlp.gate_proj.weight"]  # [6144, 2048]
    up_weight = weights["talker.layer_0.mlp.up_proj.weight"]  # [6144, 2048]
    down_weight = weights["talker.layer_0.mlp.down_proj.weight"]  # [2048, 6144]

    print(f"  Gate weight: {gate_weight.shape}")
    print(f"  Up weight: {up_weight.shape}")
    print(f"  Down weight: {down_weight.shape}")

    # Create test input (padded for tile)
    torch.manual_seed(42)
    batch, seq, hidden = 1, 32, 2048
    x = torch.randn(batch, seq, hidden)

    # Reference MLP forward
    gate = F.linear(x, gate_weight)  # [1, 32, 6144]
    up = F.linear(x, up_weight)  # [1, 32, 6144]
    intermediate = F.silu(gate) * up  # [1, 32, 6144]
    ref_output = F.linear(intermediate, down_weight)  # [1, 32, 2048]

    # TTNN MLP forward
    x_tt = ttnn.from_torch(
        x.unsqueeze(0), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device  # [1, 1, 32, 2048]
    )

    # Weights need to be transposed for ttnn.linear: [out, in] -> [in, out]
    gate_tt = ttnn.from_torch(
        gate_weight.T.unsqueeze(0).unsqueeze(0).contiguous(),  # [1, 1, 2048, 6144]
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    up_tt = ttnn.from_torch(
        up_weight.T.unsqueeze(0).unsqueeze(0).contiguous(),  # [1, 1, 2048, 6144]
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    down_tt = ttnn.from_torch(
        down_weight.T.unsqueeze(0).unsqueeze(0).contiguous(),  # [1, 1, 6144, 2048]
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    # Gate and Up projections
    gate_out = ttnn.linear(x_tt, gate_tt)
    up_out = ttnn.linear(x_tt, up_tt)

    # SwiGLU: silu(gate) * up
    gate_activated = ttnn.silu(gate_out)
    intermediate_tt = ttnn.mul(gate_activated, up_out)

    # Down projection
    ttnn_output = ttnn.linear(intermediate_tt, down_tt)
    ttnn_output = ttnn.to_torch(ttnn_output).squeeze(0)

    pcc = compute_pcc(ref_output, ttnn_output)
    print(f"  Reference output stats: mean={ref_output.mean():.4f}, std={ref_output.std():.4f}")
    print(f"  TTNN output stats: mean={ttnn_output.mean():.4f}, std={ttnn_output.std():.4f}")
    print(f"  PCC: {pcc:.6f}")

    status = "PASS" if pcc > 0.99 else "FAIL"
    print(f"  Status: {status}")

    return pcc


def test_attention_pcc(device, weights):
    """Test attention using layer 0 weights (no RoPE for simplicity)."""
    print("\n" + "=" * 80)
    print("TEST: Attention (Identity RoPE)")
    print("=" * 80)

    from models.demos.qwen3_tts.reference.functional import attention

    # Get weights
    q_proj = weights["talker.layer_0.self_attn.q_proj.weight"]
    k_proj = weights["talker.layer_0.self_attn.k_proj.weight"]
    v_proj = weights["talker.layer_0.self_attn.v_proj.weight"]
    o_proj = weights["talker.layer_0.self_attn.o_proj.weight"]
    q_norm = weights["talker.layer_0.self_attn.q_norm.weight"]
    k_norm = weights["talker.layer_0.self_attn.k_norm.weight"]

    print(f"  Q proj: {q_proj.shape}")
    print(f"  K proj: {k_proj.shape}")
    print(f"  V proj: {v_proj.shape}")

    # Config
    num_heads = 16
    num_kv_heads = 8
    head_dim = 2048 // 16
    rms_norm_eps = 1e-6

    # Create test input
    torch.manual_seed(42)
    batch, seq, hidden = 1, 32, 2048
    x = torch.randn(batch, seq, hidden)

    # Identity RoPE (cos=1, sin=0)
    cos = torch.ones(seq, head_dim)
    sin = torch.zeros(seq, head_dim)

    # Reference attention
    ref_output = attention(
        x,
        q_proj_weight=q_proj,
        k_proj_weight=k_proj,
        v_proj_weight=v_proj,
        o_proj_weight=o_proj,
        q_norm_weight=q_norm,
        k_norm_weight=k_norm,
        cos=cos,
        sin=sin,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        rms_norm_eps=rms_norm_eps,
        attention_mask=None,
        use_mrope=False,
    )

    print(f"  Reference output: {ref_output.shape}")
    print(f"  Reference stats: mean={ref_output.mean():.4f}, std={ref_output.std():.4f}")

    # Note: Full TTNN attention test would require loading TtAttention class
    # For now, just verify reference runs correctly
    print(f"  TTNN attention test requires full model loading (skipped)")

    return None


def run_all_tests(device):
    """Run all PCC comparison tests."""
    print("=" * 80)
    print("QWEN3-TTS: TTNN vs Official qwen_tts PCC Comparison")
    print("=" * 80)

    # Load extracted tensors
    tensor_dir = Path("/tmp/qwen_tts_tensors")
    if not (tensor_dir / "prefill_tensors.pt").exists():
        print("ERROR: Extracted tensors not found. Run extract_prefill.py first.")
        return

    prefill_tensors = torch.load(tensor_dir / "prefill_tensors.pt")
    weights = torch.load(tensor_dir / "model_weights.pt")

    print(f"Loaded {len(prefill_tensors)} prefill tensors")
    print(f"Loaded {len(weights)} weight tensors")

    results = {}

    # Test RMSNorm
    pcc = test_rmsnorm_pcc(device, prefill_tensors, weights)
    results["rmsnorm"] = pcc

    # Test Embedding
    pcc = test_embedding_pcc(device, prefill_tensors, weights)
    results["embedding"] = pcc

    # Test MLP
    pcc = test_mlp_pcc(device, weights)
    results["mlp"] = pcc

    # Test Attention (reference only for now)
    test_attention_pcc(device, weights)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for name, pcc in results.items():
        if pcc is not None:
            status = "PASS" if pcc > 0.99 else "FAIL"
            print(f"  {name}: PCC={pcc:.6f} [{status}]")
        else:
            print(f"  {name}: Skipped")


def main():
    parser = argparse.ArgumentParser(description="PCC Comparison: TTNN vs Official")
    parser.add_argument("--device-id", type=int, default=0, help="Device ID")
    args = parser.parse_args()

    device = ttnn.open_device(device_id=args.device_id)

    try:
        run_all_tests(device)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
