"""Component-level PCC validation: TTNN vs torch reference for each module.

Finds exactly where the computation diverges.
Run: pytest models/demos/blackhole/qwen3_5_9b/tests/test_component_pcc.py -v -s --noconftest --timeout=600
"""
import glob

import pytest
import torch
import torch.nn.functional as F
from loguru import logger

import ttnn
from models.demos.blackhole.qwen3_5_9b.tt.model_config import Qwen35ModelArgs
from models.demos.blackhole.qwen3_5_9b.tt.weight_mapping import remap_qwen35_state_dict

CHECKPOINT_DIR = "/local/ttuser/atupe/Qwen9b"


def compute_pcc(a, b):
    a_flat = a.float().flatten()
    b_flat = b.float().flatten()
    a_c = a_flat - a_flat.mean()
    b_c = b_flat - b_flat.mean()
    return ((a_c * b_c).sum() / (a_c.norm() * b_c.norm() + 1e-8)).item()


@pytest.fixture(scope="module")
def device():
    dev = ttnn.open_device(device_id=0)
    yield dev
    ttnn.close_device(dev)


@pytest.fixture(scope="module")
def setup(device):
    args = Qwen35ModelArgs(mesh_device=device, checkpoint_dir=CHECKPOINT_DIR)
    from safetensors import safe_open

    raw = {}
    for path in sorted(glob.glob(f"{CHECKPOINT_DIR}/model.safetensors-*.safetensors")):
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                raw[key] = f.get_tensor(key)
    sd = remap_qwen35_state_dict(raw)
    return args, sd, raw


class TestEmbedding:
    def test_embedding_lookup(self, device, setup):
        """Check if embedding produces correct values."""
        args, sd, raw = setup
        embed_w = sd["tok_embeddings.weight"]  # [vocab_size, hidden_size]

        # Torch reference
        token_ids = torch.tensor([[760, 6511, 314, 9338, 369]])  # "The capital of France is"
        ref = F.embedding(token_ids, embed_w)  # [1, 5, 4096]

        # TTNN
        embed_ttnn = ttnn.from_torch(
            embed_w.unsqueeze(0).unsqueeze(0),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )
        ids_ttnn = ttnn.from_torch(token_ids, dtype=ttnn.uint32, device=device)
        out_ttnn = ttnn.embedding(ids_ttnn, embed_ttnn, layout=ttnn.TILE_LAYOUT)
        out_torch = ttnn.to_torch(out_ttnn)

        pcc = compute_pcc(ref.to(torch.bfloat16), out_torch)
        logger.info(f"Embedding PCC: {pcc:.6f}")
        logger.info(f"Ref shape: {ref.shape}, TTNN shape: {out_torch.shape}")
        logger.info(f"Ref[0,0,:5]: {ref[0,0,:5]}")
        logger.info(f"TTNN[0,0,:5]: {out_torch[0,0,:5]}")
        assert pcc > 0.99, f"Embedding PCC too low: {pcc}"


class TestLMHead:
    def test_lm_head_precision(self, device, setup):
        """Check if LM head produces correct logits."""
        args, sd, raw = setup
        lm_w = sd["output.weight"]  # [vocab_size, hidden_size]

        x_cpu = torch.randn(1, 1, 4096, dtype=torch.bfloat16)

        # Torch reference
        ref = F.linear(x_cpu, lm_w.to(torch.bfloat16))  # [1, 1, vocab_size]

        # TTNN with bfloat8_b (current)
        lm_w_t = lm_w.T.contiguous()  # [4096, vocab_size]
        lm_ttnn_bf8 = ttnn.from_torch(lm_w_t, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
        x_ttnn = ttnn.from_torch(x_cpu, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        out_bf8 = ttnn.to_torch(ttnn.linear(x_ttnn, lm_ttnn_bf8))

        pcc_bf8 = compute_pcc(ref, out_bf8)
        logger.info(f"LM Head PCC (bfloat8_b): {pcc_bf8:.6f}")

        # TTNN with bfloat16 (higher precision)
        lm_ttnn_bf16 = ttnn.from_torch(lm_w_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        out_bf16 = ttnn.to_torch(ttnn.linear(x_ttnn, lm_ttnn_bf16))

        pcc_bf16 = compute_pcc(ref, out_bf16)
        logger.info(f"LM Head PCC (bfloat16): {pcc_bf16:.6f}")

        # Check top-k agreement
        ref_top10 = ref.squeeze().topk(10).indices.tolist()
        bf8_top10 = out_bf8.squeeze().topk(10).indices.tolist()
        bf16_top10 = out_bf16.squeeze().topk(10).indices.tolist()
        overlap_bf8 = len(set(ref_top10) & set(bf8_top10))
        overlap_bf16 = len(set(ref_top10) & set(bf16_top10))
        logger.info(f"Top-10 overlap (bf8): {overlap_bf8}/10")
        logger.info(f"Top-10 overlap (bf16): {overlap_bf16}/10")


class TestRMSNorm:
    def test_standard_rmsnorm(self, device, setup):
        args, sd, raw = setup
        from models.demos.blackhole.qwen3_5_9b.tt.qwen35_decoder import rms_norm_ttnn

        w = sd["layers.0.input_layernorm.weight"]
        x = torch.randn(1, 4, 4096, dtype=torch.bfloat16)

        # Torch ref
        x_f = x.float()
        var = x_f.pow(2).mean(-1, keepdim=True)
        ref = (x_f * torch.rsqrt(var + 1e-6) * w.float()).to(torch.bfloat16)

        # TTNN
        x_t = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        w_t = ttnn.from_torch(w, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        out = ttnn.to_torch(rms_norm_ttnn(x_t, w_t, eps=1e-6))

        pcc = compute_pcc(ref, out)
        logger.info(f"RMSNorm PCC: {pcc:.6f}")
        assert pcc > 0.98, f"RMSNorm PCC too low: {pcc}"


class TestMLP:
    def test_mlp_pcc(self, device, setup):
        args, sd, raw = setup

        gate_w = sd["layers.0.mlp.gate_proj.weight"]
        up_w = sd["layers.0.mlp.up_proj.weight"]
        down_w = sd["layers.0.mlp.down_proj.weight"]

        x = torch.randn(1, 4, 4096, dtype=torch.bfloat16)
        ref = F.linear(
            F.silu(F.linear(x, gate_w.to(torch.bfloat16))) * F.linear(x, up_w.to(torch.bfloat16)),
            down_w.to(torch.bfloat16),
        )

        from models.demos.blackhole.qwen3_5_9b.tt.qwen35_mlp import Qwen35MLP

        mlp = Qwen35MLP(args, sd, 0, device)
        x_t = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        out = ttnn.to_torch(mlp.forward(x_t))

        pcc = compute_pcc(ref, out)
        logger.info(f"MLP PCC: {pcc:.6f}")
        logger.info(f"Ref range: [{ref.min():.4f}, {ref.max():.4f}]")
        logger.info(f"TTNN range: [{out.min():.4f}, {out.max():.4f}]")
        # MLP uses bf8b weights so lower threshold
        assert pcc > 0.90, f"MLP PCC too low: {pcc}"


class TestGatedAttention:
    def test_gated_attention_vs_torch(self, device, setup):
        """Compare TTNN gated attention against torch reference for layer 3."""
        args, sd, raw = setup
        from models.demos.blackhole.qwen3_5_9b.tt.qwen35_gated_attention import Qwen35GatedAttention
        from models.demos.blackhole.qwen3_5_9b.tt.qwen35_rope import Qwen35RoPESetup, compute_rope_freqs
        from models.experimental.gated_attention_gated_deltanet.torch_functional.gated_attention import (
            gated_attention_forward,
        )

        layer_num = 3
        B, T = 1, 4

        # Torch reference (uses HF convention: weight is [out, in])
        prefix = f"layers.{layer_num}.self_attn"
        q_w = sd[f"{prefix}.q_proj.weight"]
        k_w = sd[f"{prefix}.k_proj.weight"]
        v_w = sd[f"{prefix}.v_proj.weight"]
        o_w = sd[f"{prefix}.o_proj.weight"]
        q_norm = sd[f"{prefix}.q_norm.weight"]
        k_norm = sd[f"{prefix}.k_norm.weight"]

        x = torch.randn(B, T, 4096, dtype=torch.bfloat16)

        # RoPE for torch (cast to bfloat16 to match input dtype)
        cos_cpu, sin_cpu = compute_rope_freqs(64, 2048, theta=10_000_000)
        pos_ids = torch.arange(T)
        cos_t = cos_cpu[pos_ids].unsqueeze(0).to(torch.bfloat16)  # [1, T, 64]
        sin_t = sin_cpu[pos_ids].unsqueeze(0).to(torch.bfloat16)

        ref_out, _, _ = gated_attention_forward(
            hidden_states=x,
            q_proj_weight=q_w,
            k_proj_weight=k_w,
            v_proj_weight=v_w,
            o_proj_weight=o_w,
            q_norm_weight=q_norm,
            k_norm_weight=k_norm,
            cos=cos_t,
            sin=sin_t,
            num_attention_heads=16,
            num_key_value_heads=4,
            head_dim=256,
            norm_eps=1e-6,
        )

        # TTNN
        attn = Qwen35GatedAttention(args, sd, layer_num, device)
        rope = Qwen35RoPESetup(device, args)
        pos = torch.arange(T).unsqueeze(0)
        cos_ttnn, sin_ttnn = rope.get_rot_mats(pos)

        x_t = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        out = ttnn.to_torch(attn.forward(x_t, cos_ttnn, sin_ttnn))

        pcc = compute_pcc(ref_out, out)
        logger.info(f"Gated Attention PCC: {pcc:.6f}")
        logger.info(f"Ref range: [{ref_out.min():.4f}, {ref_out.max():.4f}]")
        logger.info(f"TTNN range: [{out.min():.4f}, {out.max():.4f}]")
        assert pcc > 0.90, f"Gated Attention PCC too low: {pcc}"


class TestDeltaNet:
    def test_deltanet_vs_torch(self, device, setup):
        """Compare TTNN deltanet against torch reference for layer 0."""
        args, sd, raw = setup
        from models.demos.blackhole.qwen3_5_9b.tt.qwen35_gated_deltanet import Qwen35GatedDeltaNet
        from models.experimental.gated_attention_gated_deltanet.torch_functional.gated_deltanet import (
            gated_deltanet_forward,
        )

        layer_num = 0
        B, T = 1, 4

        prefix = f"layers.{layer_num}.linear_attn"
        x = torch.randn(B, T, 4096, dtype=torch.bfloat16)

        # Torch reference (note: torch uses [out, in] convention, no transpose)
        # Cast all weights to float32 to avoid dtype mismatch in torch reference
        def to_f32(t):
            return t.float() if t is not None else None

        ref_out, _ = gated_deltanet_forward(
            hidden_states=x.float(),
            q_proj_weight=to_f32(sd[f"{prefix}.q_proj.weight"]),
            k_proj_weight=to_f32(sd[f"{prefix}.k_proj.weight"]),
            v_proj_weight=to_f32(sd[f"{prefix}.v_proj.weight"]),
            a_proj_weight=to_f32(sd[f"{prefix}.in_proj_a.weight"]),
            b_proj_weight=to_f32(sd[f"{prefix}.in_proj_b.weight"]),
            o_proj_weight=to_f32(sd[f"{prefix}.out_proj.weight"]),
            q_conv_weight=to_f32(sd[f"{prefix}.q_conv.weight"]),
            k_conv_weight=to_f32(sd[f"{prefix}.k_conv.weight"]),
            v_conv_weight=to_f32(sd[f"{prefix}.v_conv.weight"]),
            q_conv_bias=to_f32(sd.get(f"{prefix}.q_conv.bias")),
            k_conv_bias=to_f32(sd.get(f"{prefix}.k_conv.bias")),
            v_conv_bias=to_f32(sd.get(f"{prefix}.v_conv.bias")),
            A_log=to_f32(sd[f"{prefix}.A_log"]),
            dt_bias=to_f32(sd[f"{prefix}.dt_bias"]),
            o_norm_weight=to_f32(sd[f"{prefix}.norm.weight"]),
            g_proj_weight=to_f32(sd[f"{prefix}.in_proj_z.weight"]),
            num_heads=16,
            num_v_heads=32,
            head_k_dim=128,
            head_v_dim=128,
            conv_kernel_size=4,
            use_gate=True,
            norm_eps=1e-6,
            mode="fused_recurrent",
            recurrent_state=None,
            output_final_state=True,
        )

        # TTNN
        deltanet = Qwen35GatedDeltaNet(args, sd, layer_num, device)
        deltanet.reset_state(B)
        x_t = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        out = ttnn.to_torch(deltanet.forward(x_t, mode="recurrent"))

        pcc = compute_pcc(ref_out, out)
        logger.info(f"DeltaNet PCC: {pcc:.6f}")
        logger.info(f"Ref range: [{ref_out.min():.4f}, {ref_out.max():.4f}]")
        logger.info(f"TTNN range: [{out.min():.4f}, {out.max():.4f}]")
        assert pcc > 0.85, f"DeltaNet PCC too low: {pcc}"


class TestFullForwardOneLayer:
    def test_decoder_block_pcc(self, device, setup):
        """Test full decoder block (norm + attention + norm + MLP) for layer 0."""
        args, sd, raw = setup
        from models.demos.blackhole.qwen3_5_9b.tt.qwen35_decoder import Qwen35TransformerBlock

        layer_num = 0  # DeltaNet layer
        B, T = 1, 4
        x = torch.randn(B, T, 4096, dtype=torch.bfloat16)

        block = Qwen35TransformerBlock(args, sd, layer_num, device)
        block.attention.reset_state(B)

        x_t = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        out = ttnn.to_torch(block.forward(x_t, mode="prefill", chunk_size=64))

        logger.info(f"Decoder block 0 output:")
        logger.info(f"Shape: {out.shape}")
        logger.info(f"Range: [{out.min():.4f}, {out.max():.4f}]")
        logger.info(f"Mean: {out.float().mean():.4f}")
        logger.info(f"Std: {out.float().std():.4f}")
        logger.info(f"Has NaN: {torch.isnan(out).any()}")

        # Just check it doesn't produce garbage
        assert not torch.isnan(out).any()
        assert out.float().std() > 0.01, "Output is near-constant"
