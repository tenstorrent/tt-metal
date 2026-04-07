"""PCC verification tests for Wav2Vec2-BERT encoder TTNN ops.

Tests GLU (split + sigmoid + mul) and depthwise conv k=31 on device
against PyTorch reference implementations.

Run: pytest models/demos/inworld_tts/tt/test_encoder_ttnn.py -v -s
"""

import pytest
import torch
import torch.nn.functional as F

import ttnn
from models.demos.inworld_tts.tt.model_config import W2V_DEPTHWISE_KERNEL, W2V_DIM, get_compute_kernel_config_hifi4

L1 = ttnn.L1_MEMORY_CONFIG
DRAM = ttnn.DRAM_MEMORY_CONFIG


def compute_pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute Pearson Correlation Coefficient."""
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()
    return torch.corrcoef(torch.stack([a_flat, b_flat]))[0, 1].item()


def _bf16(t: torch.Tensor) -> torch.Tensor:
    """Quantize tensor to bfloat16 and back to float32."""
    return t.to(torch.bfloat16).to(torch.float32)


@pytest.fixture(scope="module")
def device():
    dev = ttnn.open_device(device_id=0, l1_small_size=16384)
    yield dev
    ttnn.close_device(dev)


class TestTtW2vConvModule:
    def test_glu_on_device(self, device):
        """Test GLU (split + sigmoid + mul) on device vs PyTorch reference."""
        torch.manual_seed(42)
        T = 64

        # Input: [1, 1, T, 2048] (output of pointwise_conv1)
        x = _bf16(torch.randn(1, 1, T, 2 * W2V_DIM))

        # Reference: GLU on host
        h1_ref, h2_ref = x.chunk(2, dim=-1)
        ref_out = h1_ref * torch.sigmoid(h2_ref)  # [1, 1, T, 1024]

        # TTNN: GLU on device
        x_tt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=L1)
        h1 = x_tt[:, :, :, :W2V_DIM]
        h2 = x_tt[:, :, :, W2V_DIM:]
        tt_out = ttnn.mul(h1, ttnn.sigmoid(h2), memory_config=L1)
        tt_out_torch = ttnn.to_torch(tt_out)

        pcc = compute_pcc(ref_out, tt_out_torch)
        print(f"GLU PCC: {pcc:.6f}")
        assert pcc > 0.99, f"GLU PCC {pcc:.6f} < 0.99"

    def test_depthwise_conv_k31(self, device):
        """Test depthwise conv k=31 groups=1024 on device vs PyTorch reference."""
        torch.manual_seed(42)
        T = 64

        # Input: [1, 1, T, 1024] (after GLU)
        x = _bf16(torch.randn(1, 1, T, W2V_DIM))
        # Depthwise conv weight: [1024, 1, 31]
        dw_weight = _bf16(torch.randn(W2V_DIM, 1, W2V_DEPTHWISE_KERNEL))

        # Reference: causal left-pad + depthwise conv on host
        x_bct = x.squeeze(0).transpose(1, 2)  # [1, 1024, T]
        x_padded = F.pad(x_bct, (W2V_DEPTHWISE_KERNEL - 1, 0))
        ref_out = F.conv1d(x_padded, dw_weight, groups=W2V_DIM)  # [1, 1024, T]
        ref_out = ref_out.transpose(1, 2).unsqueeze(0)  # [1, 1, T, 1024]

        # TTNN: depthwise conv on device
        x_tt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=L1)
        x_tt = ttnn.to_layout(x_tt, ttnn.ROW_MAJOR_LAYOUT)

        w_tt = ttnn.from_torch(dw_weight.to(torch.float32), dtype=ttnn.float32)

        compute_config = get_compute_kernel_config_hifi4()
        conv_config = ttnn.Conv1dConfig(
            weights_dtype=ttnn.bfloat16,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            act_block_h_override=32,
        )

        tt_out, out_len, [cached_w, cached_b] = ttnn.conv1d(
            input_tensor=x_tt,
            weight_tensor=w_tt,
            in_channels=W2V_DIM,
            out_channels=W2V_DIM,
            device=device,
            bias_tensor=None,
            kernel_size=W2V_DEPTHWISE_KERNEL,
            stride=1,
            padding=(W2V_DEPTHWISE_KERNEL - 1, 0),
            batch_size=1,
            input_length=T,
            dtype=ttnn.bfloat16,
            conv_config=conv_config,
            compute_config=compute_config,
            groups=W2V_DIM,
            return_output_dim=True,
            return_weights_and_bias=True,
        )
        tt_out = ttnn.sharded_to_interleaved(tt_out, DRAM)
        tt_out = ttnn.to_layout(tt_out, ttnn.TILE_LAYOUT)
        tt_out_torch = ttnn.to_torch(tt_out)

        pcc = compute_pcc(ref_out, tt_out_torch)
        print(f"Depthwise Conv k=31 PCC: {pcc:.6f}")
        assert pcc > 0.99, f"Depthwise Conv k=31 PCC {pcc:.6f} < 0.99"


class TestTtW2vSelfAttention:
    def test_pcc_vs_reference(self, device):
        torch.manual_seed(42)
        from models.demos.inworld_tts.reference.functional import w2v_relative_position_bias

        T, dim, n_heads, head_dim = 64, 1024, 16, 64
        x = _bf16(torch.randn(1, T, dim))

        wq = _bf16(torch.randn(dim, dim))
        bq = _bf16(torch.randn(dim))
        wk = _bf16(torch.randn(dim, dim))
        bk = _bf16(torch.randn(dim))
        wv = _bf16(torch.randn(dim, dim))
        bv = _bf16(torch.randn(dim))
        wo = _bf16(torch.randn(dim, dim))
        bo = _bf16(torch.randn(dim))
        dist_emb = _bf16(torch.randn(73, 64))

        # Reference
        q = F.linear(x, wq, bq).view(1, T, n_heads, head_dim).permute(0, 2, 1, 3)
        k = F.linear(x, wk, bk).view(1, T, n_heads, head_dim).permute(0, 2, 1, 3)
        v = F.linear(x, wv, bv).view(1, T, n_heads, head_dim).permute(0, 2, 1, 3)
        scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) / (head_dim**0.5)
        pos_bias = w2v_relative_position_bias(q.float(), dist_emb.float(), T)
        scores = scores + pos_bias
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v.float())
        out = out.permute(0, 2, 1, 3).contiguous().view(1, T, dim)
        ref_out = F.linear(out, wo, bo)

        # TTNN
        prefix = "encoder.layers.0.self_attn."
        sd = {
            prefix + "linear_q.weight": wq,
            prefix + "linear_q.bias": bq,
            prefix + "linear_k.weight": wk,
            prefix + "linear_k.bias": bk,
            prefix + "linear_v.weight": wv,
            prefix + "linear_v.bias": bv,
            prefix + "linear_out.weight": wo,
            prefix + "linear_out.bias": bo,
            prefix + "distance_embedding.weight": dist_emb,
        }
        from models.demos.inworld_tts.tt.wav2vec2_bert import TtW2vSelfAttention

        tt_attn = TtW2vSelfAttention(device, sd, prefix)
        x_tt = ttnn.from_torch(
            x.to(torch.bfloat16).unsqueeze(0),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        tt_out = ttnn.to_torch(tt_attn(x_tt)).squeeze(0)

        pcc = compute_pcc(ref_out, tt_out)
        print(f"W2v Self-Attention PCC: {pcc:.6f}")
        assert pcc > 0.99, f"W2v Self-Attention PCC {pcc:.6f} < 0.99"


class TestTtSemanticEncoder:
    def test_pcc_vs_reference(self, device):
        torch.manual_seed(42)
        T, C = 64, 1024

        prefix = "SemanticEncoder_module."
        sd = {
            prefix + "initial_conv.weight": _bf16(torch.randn(C, C, 3)),
            prefix + "initial_conv.bias": _bf16(torch.randn(C)),
            prefix + "residual_blocks.1.weight": _bf16(torch.randn(C, C, 3)),
            prefix + "residual_blocks.1.bias": _bf16(torch.randn(C)),
            prefix + "residual_blocks.3.weight": _bf16(torch.randn(C, C, 3)),
            prefix + "residual_blocks.3.bias": _bf16(torch.randn(C)),
            prefix + "final_conv.weight": _bf16(torch.randn(C, C, 3)),
            prefix + "final_conv.bias": _bf16(torch.randn(C)),
        }

        x = _bf16(torch.randn(1, C, T))

        # Reference (CPU)
        h = F.conv1d(x, sd[prefix + "initial_conv.weight"], sd[prefix + "initial_conv.bias"], padding=1)
        res = h
        h = F.relu(h)
        h = F.conv1d(h, sd[prefix + "residual_blocks.1.weight"], sd[prefix + "residual_blocks.1.bias"], padding=1)
        h = F.relu(h)
        h = F.conv1d(h, sd[prefix + "residual_blocks.3.weight"], sd[prefix + "residual_blocks.3.bias"], padding=1)
        h = res + h
        ref_out = F.conv1d(h, sd[prefix + "final_conv.weight"], sd[prefix + "final_conv.bias"], padding=1)

        # TTNN
        from models.demos.inworld_tts.tt.codec_encoder import TtSemanticEncoder

        tt_enc = TtSemanticEncoder(device, sd, prefix=prefix)
        tt_out = tt_enc(x)

        pcc = compute_pcc(ref_out, tt_out)
        print(f"SemanticEncoder PCC: {pcc:.6f}")
        assert pcc > 0.99, f"SemanticEncoder PCC {pcc:.6f} < 0.99"


ACOUSTIC_FIR_LEN = 12


def _acoustic_snake_act_sd(sd: dict, p: str, channels: int, k_fir: int = ACOUSTIC_FIR_LEN) -> None:
    """Add Activation1d (SnakeBeta + FIR) keys at prefix p (must end with '.')."""
    sd[p + "act.alpha"] = _bf16(torch.randn(channels))
    sd[p + "act.beta"] = _bf16(torch.rand(channels) + 4.1)
    sd[p + "upsample.filter"] = _bf16(torch.randn(1, 1, k_fir) / 4)
    sd[p + "downsample.lowpass.filter"] = _bf16(torch.randn(1, 1, k_fir) / 4)


def _acoustic_wn_conv_sd(sd: dict, p: str, cin: int, cout: int, kernel_size: int) -> None:
    """Add weight-norm Conv1d keys at prefix p (must end with '.')."""
    sd[p + "weight_g"] = _bf16(torch.randn(cout, 1, 1) / 10)
    sd[p + "weight_v"] = _bf16(torch.randn(cout, cin, kernel_size) / 10)
    sd[p + "bias"] = _bf16(torch.randn(cout))


def make_single_block_state_dict(block_idx: int = 1, channels=None, strides=None):
    """Create a minimal state dict with initial conv + one encoder block + final block.

    Uses block_idx=1 (cin=96, cout=192, stride=2) which is tile-aligned.
    """
    if channels is None:
        channels = [48, 96, 192, 384, 768, 1536]
    if strides is None:
        strides = [2, 2, 4, 4, 5]
    sd = {}
    k_fir = ACOUSTIC_FIR_LEN

    # Initial conv: Conv1d(1, 48, k=7)
    _acoustic_wn_conv_sd(sd, "conv_blocks.0.", 1, 48, 7)

    # All 5 encoder blocks (needed because _extract_encoder_block_weights uses these keys)
    for b in range(5):
        prefix = f"conv_blocks.{b + 1}."
        cin, cout = channels[b], channels[b + 1]
        stride = strides[b]

        for res_idx in range(3):
            ru = f"{prefix}block.{res_idx}."
            _acoustic_snake_act_sd(sd, f"{ru}block.0.", cin, k_fir)
            _acoustic_wn_conv_sd(sd, f"{ru}block.1.", cin, cin, 7)
            _acoustic_snake_act_sd(sd, f"{ru}block.2.", cin, k_fir)
            _acoustic_wn_conv_sd(sd, f"{ru}block.3.", cin, cin, 1)

        _acoustic_snake_act_sd(sd, f"{prefix}block.3.", cin, k_fir)
        _acoustic_wn_conv_sd(sd, f"{prefix}block.4.", cin, cout, stride * 2)

    # Final block
    _acoustic_snake_act_sd(sd, "conv_final_block.0.", 1536, k_fir)
    _acoustic_wn_conv_sd(sd, "conv_final_block.1.", 1536, 1024, 3)

    return sd


class TestTtEncoderBlock:
    """Test a single encoder block (3 residual units + downsample) via TtAcousticEncoder components."""

    def test_single_block_pcc(self, device):
        """Test one encoder block: block_idx=0 (cin=48, cout=96, stride=2).

        Builds the block modules directly and compares against the reference encoder_block_forward.
        """
        torch.manual_seed(42)

        from models.demos.inworld_tts.reference.functional import _extract_encoder_block_weights, encoder_block_forward
        from models.demos.inworld_tts.tt.codec_encoder import TtActivation1d, TtConv1d

        block_idx = 0
        c_in = 96
        c_out = 192
        stride = 2
        T = 64

        # Build state dict for this block
        sd = {}
        prefix = "conv_blocks.2."  # block_idx=1 in the full encoder (0-indexed as conv_blocks.2)
        k_fir = ACOUSTIC_FIR_LEN
        for res_idx in range(3):
            ru = f"{prefix}block.{res_idx}."
            _acoustic_snake_act_sd(sd, f"{ru}block.0.", c_in, k_fir)
            _acoustic_wn_conv_sd(sd, f"{ru}block.1.", c_in, c_in, 7)
            _acoustic_snake_act_sd(sd, f"{ru}block.2.", c_in, k_fir)
            _acoustic_wn_conv_sd(sd, f"{ru}block.3.", c_in, c_in, 1)
        _acoustic_snake_act_sd(sd, f"{prefix}block.3.", c_in, k_fir)
        _acoustic_wn_conv_sd(sd, f"{prefix}block.4.", c_in, c_out, stride * 2)

        bw = _extract_encoder_block_weights(sd, prefix, c_in)

        # Input
        x = _bf16(torch.randn(1, c_in, T))

        # Reference
        ref_out = encoder_block_forward(x, bw, stride)

        # TTNN: build modules
        res_acts = []
        res_convs = []
        for res_idx in range(3):
            p = f"res_{res_idx}_"
            act1 = TtActivation1d(
                bw[p + "act1_alpha"], bw[p + "act1_beta"], bw[p + "act1_up_filter"], bw[p + "act1_down_filter"], device
            )
            conv1 = TtConv1d(c_in, c_in, bw[p + "conv1_weight"], bw[p + "conv1_bias"], 7, 3, device)
            act2 = TtActivation1d(
                bw[p + "act2_alpha"], bw[p + "act2_beta"], bw[p + "act2_up_filter"], bw[p + "act2_down_filter"], device
            )
            conv2 = TtConv1d(c_in, c_in, bw[p + "conv2_weight"], bw[p + "conv2_bias"], 1, 0, device)
            res_acts.append((act1, act2))
            res_convs.append((conv1, conv2))

        final_act = TtActivation1d(bw["act_alpha"], bw["act_beta"], bw["act_up_filter"], bw["act_down_filter"], device)
        kernel_size = stride * 2
        pad_total = kernel_size - stride
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        ds_conv = TtConv1d(
            c_in,
            c_out,
            bw["downsample_weight"],
            bw["downsample_bias"],
            kernel_size,
            (pad_left, pad_right),
            device,
            stride=stride,
        )

        # Forward
        h = x
        for res_idx in range(3):
            res = h
            act1, act2 = res_acts[res_idx]
            conv1, conv2 = res_convs[res_idx]
            h = act1(h)
            h = conv1(h)
            h = act2(h)
            h = conv2(h)
            h = res + h
        h = final_act(h)
        h = ds_conv(h)

        pcc = compute_pcc(ref_out, h)
        print(f"EncoderBlock (cin={c_in}, cout={c_out}, stride={stride}) PCC: {pcc:.6f}")
        assert pcc > 0.99, f"EncoderBlock PCC {pcc:.6f} < 0.99"


# ---------------------------------------------------------------------------
# Test Full Encoder Fusion + FSQ round-trip
# ---------------------------------------------------------------------------
class TestFullEncoderFusion:
    def test_fusion_fsq_roundtrip(self, device):
        """Fusion linear + FSQ quantize: acoustic + semantic -> codes."""
        torch.manual_seed(42)
        from vector_quantize_pytorch import ResidualFSQ

        T = 64
        acoustic_out = _bf16(torch.randn(1, 1024, T))
        semantic_out = _bf16(torch.randn(1, 1024, T))

        # Reference fusion
        fused = torch.cat([acoustic_out, semantic_out], dim=1).transpose(1, 2)  # [1, T, 2048]
        fc_w = _bf16(torch.randn(2048, 2048))
        fc_b = _bf16(torch.randn(2048))
        fused_proj = F.linear(fused, fc_w, fc_b)

        quantizer = ResidualFSQ(levels=[4, 4, 4, 4, 4, 4, 4, 4], dim=2048, num_quantizers=1)
        _, ref_indices = quantizer(fused_proj)

        # TTNN fusion (fc_prior on device)
        fused_ttnn = ttnn.from_torch(
            fused.to(torch.bfloat16).unsqueeze(0),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        grid = device.compute_with_storage_grid_size()
        core_grid = ttnn.CoreGrid(y=grid.y, x=grid.x)
        fc_w_tt = ttnn.from_torch(
            fc_w.T.unsqueeze(0).unsqueeze(0),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        fc_b_tt = ttnn.from_torch(
            fc_b.unsqueeze(0).unsqueeze(0).unsqueeze(0),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=True,
        )
        projected = ttnn.linear(
            fused_ttnn,
            fc_w_tt,
            bias=fc_b_tt,
            core_grid=core_grid,
            compute_kernel_config=compute_config,
        )
        proj_torch = ttnn.to_torch(projected).float().squeeze(0)  # [1, T, 2048]

        _, tt_indices = quantizer(proj_torch)

        # Compare VQ code indices (should be identical or very close)
        match_rate = (ref_indices == tt_indices).float().mean().item()
        print(f"Fusion+FSQ code match rate: {match_rate:.4f}")
        assert match_rate > 0.95, f"Code match rate {match_rate:.4f} < 0.95"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
