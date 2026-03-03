# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
Layer-by-Layer PCC Test with HiFi4 Precision

Tests if using HiFi4 math fidelity improves cumulative PCC.
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

import ttnn


def compute_pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute Pearson Correlation Coefficient."""
    a = a.flatten().float()
    b = b.flatten().float()
    a_centered = a - a.mean()
    b_centered = b - b.mean()
    cov = (a_centered * b_centered).sum()
    std_a = torch.sqrt((a_centered**2).sum())
    std_b = torch.sqrt((b_centered**2).sum())
    if std_a == 0 or std_b == 0:
        return 0.0
    return (cov / (std_a * std_b)).item()


class HighPrecisionDecoderLayer:
    """Decoder layer with HiFi4 precision for all operations."""

    def __init__(self, device, layer_idx, state_dict, config):
        self.device = device
        self.layer_idx = layer_idx
        self.hidden_size = config["hidden_size"]
        self.num_heads = config["num_heads"]
        self.num_kv_heads = config["num_kv_heads"]
        self.head_dim = config["head_dim"]
        self.intermediate_size = config["intermediate_size"]
        self.rms_norm_eps = config["rms_norm_eps"]

        prefix = f"talker.model.layers.{layer_idx}"

        # HiFi4 compute kernel config - highest precision
        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,  # Highest precision
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        # Load weights with float32 on host, then convert
        # Input layernorm
        input_ln_weight = state_dict[f"{prefix}.input_layernorm.weight"]
        self.input_ln_weight = self._prepare_norm_weight(input_ln_weight)

        # Post-attention layernorm
        post_ln_weight = state_dict[f"{prefix}.post_attention_layernorm.weight"]
        self.post_ln_weight = self._prepare_norm_weight(post_ln_weight)

        # Attention weights - use float32 storage if possible, otherwise bfloat16
        q_proj = state_dict[f"{prefix}.self_attn.q_proj.weight"]
        k_proj = state_dict[f"{prefix}.self_attn.k_proj.weight"]
        v_proj = state_dict[f"{prefix}.self_attn.v_proj.weight"]
        o_proj = state_dict[f"{prefix}.self_attn.o_proj.weight"]

        # Fused QKV
        qkv_weight = torch.cat([q_proj, k_proj, v_proj], dim=0)
        qkv_weight = torch.transpose(qkv_weight, -2, -1).unsqueeze(0).unsqueeze(0)
        self.wqkv = ttnn.from_torch(qkv_weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        o_proj = torch.transpose(o_proj, -2, -1).unsqueeze(0).unsqueeze(0)
        self.wo = ttnn.from_torch(o_proj, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        # QK-norm weights
        q_norm = state_dict[f"{prefix}.self_attn.q_norm.weight"]
        k_norm = state_dict[f"{prefix}.self_attn.k_norm.weight"]
        self.q_norm_weight = self._prepare_qk_norm_weight(q_norm)
        self.k_norm_weight = self._prepare_qk_norm_weight(k_norm)

        # MLP weights
        gate_proj = state_dict[f"{prefix}.mlp.gate_proj.weight"]
        up_proj = state_dict[f"{prefix}.mlp.up_proj.weight"]
        down_proj = state_dict[f"{prefix}.mlp.down_proj.weight"]

        gate_proj = torch.transpose(gate_proj, -2, -1).unsqueeze(0).unsqueeze(0)
        up_proj = torch.transpose(up_proj, -2, -1).unsqueeze(0).unsqueeze(0)
        down_proj = torch.transpose(down_proj, -2, -1).unsqueeze(0).unsqueeze(0)

        self.gate_proj = ttnn.from_torch(gate_proj, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        self.up_proj = ttnn.from_torch(up_proj, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        self.down_proj = ttnn.from_torch(down_proj, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    def _prepare_norm_weight(self, weight):
        """Prepare RMSNorm weight for TTNN."""
        # Expand to [1, 1, 32, hidden_size] for tile layout
        weight_4d = weight.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        weight_4d = weight_4d.expand(1, 1, 32, -1).contiguous()
        return ttnn.from_torch(weight_4d, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)

    def _prepare_qk_norm_weight(self, weight):
        """Prepare QK-norm weight for TTNN."""
        TILE = 32
        weight_shaped = weight.unsqueeze(0).view(1, 1, self.head_dim).reshape([1, 1, self.head_dim // TILE, TILE])
        return ttnn.from_torch(weight_shaped, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device)

    def forward(self, x, cos, sin, transformation_mat):
        """Forward pass with HiFi4 precision."""
        batch_size = x.shape[0]
        seq_len = x.shape[-2]

        # Pre-attention norm
        residual = x
        x = ttnn.rms_norm(
            x, epsilon=self.rms_norm_eps, weight=self.input_ln_weight, compute_kernel_config=self.compute_kernel_config
        )

        # QKV projection
        xqkv = ttnn.linear(x, self.wqkv, compute_kernel_config=self.compute_kernel_config)

        # Split heads
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            xqkv, num_heads=self.num_heads, num_kv_heads=self.num_kv_heads, transpose_k_heads=False
        )
        ttnn.deallocate(xqkv)

        # QK-norm with HiFi4
        q = ttnn.rms_norm(
            q, epsilon=self.rms_norm_eps, weight=self.q_norm_weight, compute_kernel_config=self.compute_kernel_config
        )
        k = ttnn.rms_norm(
            k, epsilon=self.rms_norm_eps, weight=self.k_norm_weight, compute_kernel_config=self.compute_kernel_config
        )

        # RoPE
        if q.dtype != ttnn.bfloat16:
            q = ttnn.typecast(q, dtype=ttnn.bfloat16)
        if k.dtype != ttnn.bfloat16:
            k = ttnn.typecast(k, dtype=ttnn.bfloat16)

        q = ttnn.experimental.rotary_embedding_llama(q, cos, sin, transformation_mat, is_decode_mode=False)
        k = ttnn.experimental.rotary_embedding_llama(k, cos, sin, transformation_mat, is_decode_mode=False)

        # Attention with HiFi4
        scale = self.head_dim**-0.5
        attn_output = ttnn.transformer.scaled_dot_product_attention(
            q, k, v, is_causal=True, scale=scale, compute_kernel_config=self.compute_kernel_config
        )

        ttnn.deallocate(q)
        ttnn.deallocate(k)
        ttnn.deallocate(v)

        # Concat heads
        attn_output = ttnn.experimental.nlp_concat_heads(attn_output)

        # Output projection
        attn_output = ttnn.linear(attn_output, self.wo, compute_kernel_config=self.compute_kernel_config)

        # Residual
        x = ttnn.add(residual, attn_output)
        ttnn.deallocate(attn_output)

        # Pre-MLP norm
        residual = x
        x = ttnn.rms_norm(
            x, epsilon=self.rms_norm_eps, weight=self.post_ln_weight, compute_kernel_config=self.compute_kernel_config
        )

        # MLP with HiFi4
        gate = ttnn.linear(x, self.gate_proj, compute_kernel_config=self.compute_kernel_config)
        up = ttnn.linear(x, self.up_proj, compute_kernel_config=self.compute_kernel_config)
        ttnn.deallocate(x)

        gate = ttnn.silu(gate)
        intermediate = ttnn.mul(gate, up)
        ttnn.deallocate(gate)
        ttnn.deallocate(up)

        mlp_out = ttnn.linear(intermediate, self.down_proj, compute_kernel_config=self.compute_kernel_config)
        ttnn.deallocate(intermediate)

        # Residual
        output = ttnn.add(residual, mlp_out)
        ttnn.deallocate(mlp_out)

        return output


def load_weights():
    """Load HF weights."""
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    print("Loading model weights...")
    model_path = snapshot_download("Qwen/Qwen3-TTS-12Hz-1.7B-Base", allow_patterns=["*.safetensors"])
    model_path = Path(model_path)

    state_dict = {}
    for f in model_path.glob("*.safetensors"):
        state_dict.update(load_file(f))

    print(f"  Loaded {len(state_dict)} weight tensors")
    return state_dict


def run_test(device):
    """Run cumulative layer test with HiFi4 precision."""
    print("=" * 80)
    print("Layer-by-Layer PCC Test with HiFi4 Precision")
    print("=" * 80)

    # Load hidden states
    hidden_states_path = Path("/tmp/qwen_tts_tensors/layer_hidden_states.pt")
    if not hidden_states_path.exists():
        print(f"ERROR: {hidden_states_path} not found.")
        return

    hidden_states = torch.load(hidden_states_path)
    print(f"Loaded {len(hidden_states)} hidden state tensors")

    # Load weights
    weights = load_weights()

    # Config
    config = {
        "hidden_size": 2048,
        "num_heads": 16,
        "num_kv_heads": 8,
        "head_dim": 128,
        "intermediate_size": 6144,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000.0,
    }

    # Get initial hidden state
    layer0_input = hidden_states["layer_0_input"]
    batch, seq_len, hidden = layer0_input.shape

    # Pad for tile alignment
    pad_seq = ((seq_len + 31) // 32) * 32
    padding = pad_seq - seq_len

    input_padded = F.pad(layer0_input, (0, 0, 0, padding))
    input_4d = input_padded.unsqueeze(1)

    # Convert to TTNN
    hidden_tt = ttnn.from_torch(input_4d, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Get RoPE tensors
    from models.demos.qwen3_tts.tt.rope import get_rope_tensors, get_transformation_mat

    position_ids = torch.arange(pad_seq)
    cos, sin = get_rope_tensors(device, config["head_dim"], pad_seq, position_ids, config["rope_theta"])
    trans_mat = get_transformation_mat(config["head_dim"], device)

    # Run through layers sequentially with HiFi4
    print("\nRunning layers with HiFi4 precision:")
    for i in range(28):
        # Create high-precision layer
        layer = HighPrecisionDecoderLayer(device, i, weights, config)

        # Forward pass
        hidden_tt = layer.forward(hidden_tt, cos, sin, trans_mat)

        # Check PCC at key layers
        if i in [0, 1, 5, 10, 15, 20, 27]:
            temp_torch = ttnn.to_torch(hidden_tt).squeeze(1)[:, :seq_len, :]
            official_key = f"layer_{i}_output"
            if official_key in hidden_states:
                official = hidden_states[official_key]
                pcc = compute_pcc(official, temp_torch)
                print(f"  After layer {i}: PCC={pcc:.6f}")

    # Get final output
    final_output = ttnn.to_torch(hidden_tt).squeeze(1)[:, :seq_len, :]
    official_final = hidden_states["layer_27_output"]
    final_pcc = compute_pcc(official_final, final_output)

    print(f"\nFinal PCC (after all layers): {final_pcc:.6f}")

    if final_pcc > 0.9:
        print("*** HiFi4 SIGNIFICANTLY IMPROVED PCC! ***")
    elif final_pcc > 0.7:
        print("*** HiFi4 improved PCC moderately ***")
    else:
        print("*** HiFi4 did not significantly improve PCC ***")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-id", type=int, default=0)
    args = parser.parse_args()

    device = ttnn.open_device(device_id=args.device_id)
    try:
        run_test(device)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
