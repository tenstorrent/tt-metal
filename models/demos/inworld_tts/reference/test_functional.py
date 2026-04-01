"""Tests for Inworld TTS reference implementation.

Verifies each standalone block against the official inworld-ai/tts modules.
Run: pytest models/demos/inworld_tts/reference/test_functional.py -v
"""

import sys

import pytest
import torch

sys.path.insert(0, "/tmp/inworld-tts")

from models.demos.inworld_tts.reference import functional as ref


def compute_pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute Pearson Correlation Coefficient between two tensors."""
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()
    return torch.corrcoef(torch.stack([a_flat, b_flat]))[0, 1].item()


# ---------------------------------------------------------------------------
# Test RMSNorm
# ---------------------------------------------------------------------------
class TestRMSNorm:
    def test_matches_official(self):
        from tts.core.codec.decoder_modules import RMSNorm

        torch.manual_seed(42)
        dim = 1024
        x = torch.randn(1, 64, dim)

        official = RMSNorm(dim)
        official_out = official(x)

        ref_out = ref.rmsnorm_forward(x, official.weight.data)

        pcc = compute_pcc(ref_out, official_out)
        assert pcc > 0.9999, f"RMSNorm PCC {pcc:.6f} < 0.9999"
        assert torch.allclose(ref_out, official_out, atol=1e-5)


# ---------------------------------------------------------------------------
# Test RoPE
# ---------------------------------------------------------------------------
class TestRoPE:
    def test_matches_torchtune(self):
        import torchtune.modules as torchtune_module

        torch.manual_seed(42)
        dim = 64
        B, H, T = 1, 16, 128

        # x shape [B, H, T, D] -- same as what Attention passes to torchtune
        x = torch.randn(B, H, T, dim)

        # Official torchtune RoPE (sees dim 1 = H = 16 as position axis)
        official_rope = torchtune_module.RotaryPositionalEmbeddings(dim=dim)
        official_out = official_rope(x)

        # Our reference -- build cache for H positions (what torchtune does)
        rope_cache = ref.build_rope_cache(H, dim)
        ref_out = ref.apply_rope(x, rope_cache)

        pcc = compute_pcc(ref_out, official_out)
        assert pcc > 0.9999, f"RoPE PCC {pcc:.6f} < 0.9999"


# ---------------------------------------------------------------------------
# Test Attention
# ---------------------------------------------------------------------------
class TestAttention:
    def test_matches_official(self):
        import torchtune.modules as torchtune_module
        from tts.core.codec.decoder_modules import Attention

        torch.manual_seed(42)
        dim = 1024
        n_heads = 16
        pos_emb_dim = 64
        seq_len = 64

        rope = torchtune_module.RotaryPositionalEmbeddings(dim=pos_emb_dim)
        official = Attention(dim=dim, n_heads=n_heads, rotary_embed=rope)
        official.eval()

        x = torch.randn(1, seq_len, dim)
        official_out = official(x)

        rope_cache = ref.build_rope_cache(n_heads, pos_emb_dim)
        ref_out = ref.attention_forward(
            x,
            official.c_attn.weight.data,
            official.c_proj.weight.data,
            n_heads,
            rope_cache,
        )

        pcc = compute_pcc(ref_out, official_out)
        assert pcc > 0.999, f"Attention PCC {pcc:.6f} < 0.999"


# ---------------------------------------------------------------------------
# Test MLP
# ---------------------------------------------------------------------------
class TestMLP:
    def test_matches_official(self):
        from tts.core.codec.decoder_modules import MLP

        torch.manual_seed(42)
        dim = 1024
        x = torch.randn(1, 64, dim)

        official = MLP(dim)
        official.eval()
        official_out = official(x)

        ref_out = ref.mlp_forward(x, official.fc1.weight.data, official.fc2.weight.data)

        pcc = compute_pcc(ref_out, official_out)
        assert pcc > 0.9999, f"MLP PCC {pcc:.6f} < 0.9999"


# ---------------------------------------------------------------------------
# Test TransformerBlock
# ---------------------------------------------------------------------------
class TestTransformerBlock:
    def test_matches_official(self):
        import torchtune.modules as torchtune_module
        from tts.core.codec.decoder_modules import TransformerBlock

        torch.manual_seed(42)
        dim = 1024
        n_heads = 16
        pos_emb_dim = 64
        seq_len = 64

        rope = torchtune_module.RotaryPositionalEmbeddings(dim=pos_emb_dim)
        official = TransformerBlock(dim=dim, n_heads=n_heads, rotary_embed=rope)
        official.eval()

        x = torch.randn(1, seq_len, dim)
        official_out = official(x)

        weights = {
            "att_norm_weight": official.att_norm.weight.data,
            "c_attn_weight": official.att.c_attn.weight.data,
            "c_proj_weight": official.att.c_proj.weight.data,
            "ffn_norm_weight": official.ffn_norm.weight.data,
            "fc1_weight": official.mlp.fc1.weight.data,
            "fc2_weight": official.mlp.fc2.weight.data,
        }

        rope_cache = ref.build_rope_cache(n_heads, pos_emb_dim)
        ref_out = ref.transformer_block_forward(x, weights, n_heads, rope_cache)

        pcc = compute_pcc(ref_out, official_out)
        assert pcc > 0.999, f"TransformerBlock PCC {pcc:.6f} < 0.999"


# ---------------------------------------------------------------------------
# Test ResnetBlock
# ---------------------------------------------------------------------------
class TestResnetBlock:
    def test_matches_official(self):
        from tts.core.codec.decoder_modules import ResnetBlock

        torch.manual_seed(42)
        x = torch.randn(1, 1024, 64)  # [B, C, T]

        official = ResnetBlock(in_channels=1024, dropout=0.1, temb_channels=0)
        official.eval()
        official_out = official(x)

        weights = {
            "norm1_weight": official.norm1.weight.data,
            "norm1_bias": official.norm1.bias.data,
            "conv1_weight": official.conv1.weight.data,
            "conv1_bias": official.conv1.bias.data,
            "norm2_weight": official.norm2.weight.data,
            "norm2_bias": official.norm2.bias.data,
            "conv2_weight": official.conv2.weight.data,
            "conv2_bias": official.conv2.bias.data,
        }

        ref_out = ref.resnet_block_forward(x, weights)

        pcc = compute_pcc(ref_out, official_out)
        assert pcc > 0.9999, f"ResnetBlock PCC {pcc:.6f} < 0.9999"


# ---------------------------------------------------------------------------
# Test VocosBackbone
# ---------------------------------------------------------------------------
class TestVocosBackbone:
    def test_matches_official(self):
        from tts.core.codec.decoder_modules import VocosBackbone

        torch.manual_seed(42)
        dim = 1024
        depth = 12
        heads = 16
        pos_emb_dim = 64
        seq_len = 64

        official = VocosBackbone(hidden_dim=dim, depth=depth, heads=heads, pos_meb_dim=pos_emb_dim)
        official.eval()

        x = torch.randn(1, seq_len, dim)

        # Official forward: input is [B, T, C], internal transpose to [B, C, T] for conv
        official_out = official(x)

        # Extract weights
        sd = official.state_dict()
        backbone_weights = ref.extract_backbone_weights(sd)

        ref_out = ref.vocos_backbone_forward(x, backbone_weights, heads, pos_emb_dim, depth)

        pcc = compute_pcc(ref_out, official_out)
        assert pcc > 0.999, f"VocosBackbone PCC {pcc:.6f} < 0.999"


# ---------------------------------------------------------------------------
# Test ISTFTHead
# ---------------------------------------------------------------------------
class TestISTFTHead:
    def test_matches_official(self):
        from tts.core.codec.decoder_modules import ISTFTHead

        torch.manual_seed(42)
        dim = 1024
        n_fft = 1280
        hop_length = 320
        seq_len = 64

        official = ISTFTHead(dim=dim, n_fft=n_fft, hop_length=hop_length, padding="same")
        official.eval()

        x = torch.randn(1, seq_len, dim)
        official_out = official(x)

        ref_out = ref.istft_head_forward(
            x,
            official.out.weight.data,
            official.out.bias.data,
            n_fft=n_fft,
            hop_length=hop_length,
        )

        pcc = compute_pcc(ref_out, official_out)
        assert pcc > 0.999, f"ISTFTHead PCC {pcc:.6f} < 0.999"


# ---------------------------------------------------------------------------
# Test Full Codec Decoder
# ---------------------------------------------------------------------------
class TestCodecDecoder:
    def test_matches_official(self):
        from tts.core.codec.decoder_modules import Generator

        torch.manual_seed(42)
        seq_len = 32

        official = Generator(hidden_dim=1024, depth=12, heads=16, pos_meb_dim=64, hop_length=320, vq_dim=2048)
        official.eval()

        # Generate random VQ codes
        vq_codes = torch.randint(0, 65536, (1, 1, seq_len))

        # Official forward: quantizer.get_output_from_indices -> backbone -> head
        with torch.no_grad():
            vq_emb = official.quantizer.get_output_from_indices(vq_codes.transpose(1, 2))

        # We can test backbone + head with known embeddings
        # Use a simulated post-fc embedding for PCC comparison
        x_input = torch.randn(1, seq_len, 1024)

        with torch.no_grad():
            official_backbone_out = official.backbone(x_input)
            official_full_out = official.head(official_backbone_out)

        sd = official.state_dict()
        backbone_weights = ref.extract_backbone_weights(sd)
        istft_weights = ref.extract_istft_weights(sd)

        ref_backbone_out = ref.vocos_backbone_forward(x_input, backbone_weights)
        ref_full_out = ref.istft_head_forward(
            ref_backbone_out,
            istft_weights["out_weight"],
            istft_weights["out_bias"],
        )

        backbone_pcc = compute_pcc(ref_backbone_out, official_backbone_out)
        full_pcc = compute_pcc(ref_full_out, official_full_out)

        assert backbone_pcc > 0.999, f"Backbone PCC {backbone_pcc:.6f} < 0.999"
        assert full_pcc > 0.999, f"Full decoder PCC {full_pcc:.6f} < 0.999"


# ---------------------------------------------------------------------------
# Test Golden Output Generation
# ---------------------------------------------------------------------------
class TestGoldenGeneration:
    def test_generate_and_save_golden(self, tmp_path):
        """Generate golden outputs with random weights for TTNN verification."""
        from tts.core.codec.decoder_modules import VocosBackbone

        torch.manual_seed(0)

        backbone = VocosBackbone(hidden_dim=1024, depth=12, heads=16, pos_meb_dim=64)
        backbone.eval()

        x = torch.randn(1, 64, 1024)

        with torch.no_grad():
            output = backbone(x)

        sd = backbone.state_dict()
        backbone_weights = ref.extract_backbone_weights(sd)
        ref_out = ref.vocos_backbone_forward(x, backbone_weights)

        pcc = compute_pcc(ref_out, output)

        golden = {
            "input": x,
            "output": output,
            "ref_output": ref_out,
            "pcc": pcc,
            "config": {"dim": 1024, "depth": 12, "heads": 16, "pos_emb_dim": 64, "seq_len": 64},
        }
        torch.save(golden, tmp_path / "vocos_backbone_golden.pt")
        assert pcc > 0.999


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
