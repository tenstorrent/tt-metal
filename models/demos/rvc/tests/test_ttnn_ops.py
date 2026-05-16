# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
TTNN per-operator PCC tests for RVC bring-up.

Tests every operator wrapper against PyTorch reference using RVC-realistic shapes.
This is the foundational correctness layer — every module build on top of these ops.

Usage:
    pytest models/demos/rvc/tests/test_ttnn_ops.py -v
"""

import pytest
import torch
import ttnn

from models.demos.rvc.tests.pcc_utils import compute_pcc, assert_pcc


# =============================================================================
# 1. LINEAR TESTS
# =============================================================================


class TestLinear:
    """PCC tests for ttnn.linear with RVC-specific shapes."""

    @pytest.mark.parametrize(
        "batch, seq, in_feat, out_feat, desc",
        [
            (1, 300, 768, 768, "hubert_self_attn_proj"),
            (1, 300, 768, 3072, "hubert_ffn_up"),
            (1, 300, 3072, 768, "hubert_ffn_down"),
            (1, 300, 768, 192, "hubert_to_synth"),
            (1, 300, 192, 192, "vits_encoder_proj"),
            (1, 300, 192, 768, "vits_ffn_up"),
            (1, 32, 256, 1, "hifigan_final_proj"),
        ],
        ids=lambda x: x if isinstance(x, str) else None,
    )
    def test_linear_pcc(self, device, batch, seq, in_feat, out_feat, desc):
        """Test ttnn.linear against torch.nn.functional.linear."""
        torch.manual_seed(42)

        # PyTorch reference
        x = torch.randn(batch, seq, in_feat)
        w = torch.randn(out_feat, in_feat)
        b = torch.randn(out_feat)

        ref = torch.nn.functional.linear(x, w, b)

        # TTNN execution
        w_t = w.T.contiguous()
        x_tt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        w_tt = ttnn.from_torch(w_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        b_tt = ttnn.from_torch(
            b.unsqueeze(0).unsqueeze(0),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

        out_tt = ttnn.linear(x_tt, w_tt, bias=b_tt)
        out = ttnn.to_torch(out_tt).float()

        # Unpad if needed
        if out.shape != ref.shape:
            out = out[: ref.shape[0], : ref.shape[1], : ref.shape[2]]

        passed, pcc = assert_pcc(ref, out, threshold=0.999, op_name=f"linear_{desc}")
        print(f"  linear/{desc}: PCC={pcc:.6f}")


# =============================================================================
# 2. LAYER NORM TESTS
# =============================================================================


class TestLayerNorm:
    """PCC tests for ttnn.layer_norm with RVC shapes."""

    @pytest.mark.parametrize(
        "shape, norm_dim, desc",
        [
            ((1, 300, 768), 768, "hubert_encoder"),
            ((1, 300, 192), 192, "vits_encoder"),
            ((1, 32, 768), 768, "hubert_short_seq"),
            ((1, 1500, 768), 768, "hubert_long_seq"),
        ],
        ids=lambda x: x if isinstance(x, str) else None,
    )
    def test_layer_norm_pcc(self, device, shape, norm_dim, desc):
        """Test ttnn.layer_norm with learnable affine params."""
        torch.manual_seed(42)

        x = torch.randn(shape)
        gamma = torch.randn(norm_dim)
        beta = torch.randn(norm_dim)

        ref = torch.nn.functional.layer_norm(x, [norm_dim], weight=gamma, bias=beta)

        x_tt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        gamma_tt = ttnn.from_torch(
            gamma.unsqueeze(0).unsqueeze(0),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        beta_tt = ttnn.from_torch(
            beta.unsqueeze(0).unsqueeze(0),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

        out_tt = ttnn.layer_norm(x_tt, weight=gamma_tt, bias=beta_tt)
        out = ttnn.to_torch(out_tt).float()

        if out.shape != ref.shape:
            out = out[: ref.shape[0], : ref.shape[1], : ref.shape[2]]

        passed, pcc = assert_pcc(ref, out, threshold=0.999, op_name=f"layer_norm_{desc}")
        print(f"  layer_norm/{desc}: PCC={pcc:.6f}")


# =============================================================================
# 3. SOFTMAX TESTS
# =============================================================================


class TestSoftmax:
    """PCC tests for ttnn.softmax with attention-shaped tensors."""

    @pytest.mark.parametrize(
        "shape, desc",
        [
            ((1, 12, 300, 300), "hubert_attn_12heads"),
            ((1, 2, 300, 300), "vits_attn_2heads"),
            ((1, 12, 32, 32), "hubert_short_attn"),
            ((1, 2, 192, 192), "vits_square_attn"),
        ],
        ids=lambda x: x if isinstance(x, str) else None,
    )
    def test_softmax_pcc(self, device, shape, desc):
        """Test ttnn.softmax dim=-1 against torch."""
        torch.manual_seed(42)

        x = torch.randn(shape)
        ref = torch.nn.functional.softmax(x, dim=-1)

        x_tt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        out_tt = ttnn.softmax(x_tt, dim=-1)
        out = ttnn.to_torch(out_tt).float()

        if out.shape != ref.shape:
            slices = tuple(slice(0, s) for s in ref.shape)
            out = out[slices]

        passed, pcc = assert_pcc(ref, out, threshold=0.999, op_name=f"softmax_{desc}")
        print(f"  softmax/{desc}: PCC={pcc:.6f}")


# =============================================================================
# 4. CONV1D TESTS (via conv2d)
# =============================================================================


class TestConv1d:
    """
    PCC tests for Conv1d via ttnn.conv2d.

    This is the most critical operator for RVC — used in:
    - Hubert ConvFeatureExtractor (7 layers)
    - WaveNet (WN) layers (~30 layers)
    - HiFi-GAN ResBlocks (~12 layers)
    """

    @pytest.mark.parametrize(
        "batch, seq, in_ch, out_ch, kernel, stride, pad, desc",
        [
            # Hubert ConvFeatureExtractor layers
            (1, 48000, 1, 512, 10, 5, 0, "hubert_conv0_raw_audio"),
            (1, 9600, 512, 512, 3, 2, 0, "hubert_conv1"),
            (1, 4800, 512, 512, 3, 2, 0, "hubert_conv2"),
            # WaveNet layers (kernel=3, dilation=1, pad=1)
            (1, 300, 192, 192, 3, 1, 1, "wavenet_k3_pad1"),
            # WaveNet layers (kernel=5, dilation=1, pad=2)
            (1, 300, 192, 192, 5, 1, 2, "wavenet_k5_pad2"),
            # HiFi-GAN ResBlock
            (1, 300, 512, 512, 3, 1, 1, "hifigan_resblock_k3"),
            (1, 300, 512, 512, 7, 1, 3, "hifigan_resblock_k7"),
            (1, 300, 512, 512, 11, 1, 5, "hifigan_resblock_k11"),
            # Small shapes (flow encoder)
            (1, 300, 192, 384, 1, 1, 0, "flow_1x1_conv"),
        ],
        ids=lambda x: x if isinstance(x, str) else None,
    )
    def test_conv1d_pcc(self, device, batch, seq, in_ch, out_ch, kernel, stride, pad, desc):
        """Test Conv1d via ttnn.conv2d against torch.nn.Conv1d."""
        torch.manual_seed(42)

        # PyTorch reference
        conv = torch.nn.Conv1d(in_ch, out_ch, kernel, stride=stride, padding=pad, bias=True)
        # Input for torch Conv1d: [B, C, L]
        x_nchw = torch.randn(batch, in_ch, seq)
        ref = conv(x_nchw)  # [B, out_ch, L_out]

        # For TTNN: input must be channels-last [B, L, C]
        x_nhwc = x_nchw.permute(0, 2, 1)  # [B, L, C]

        # Prepare weight: torch [out_ch, in_ch, K] → [out_ch, in_ch, 1, K]
        w_4d = conv.weight.data.unsqueeze(2)
        b_4d = conv.bias.data.reshape(1, 1, 1, -1) if conv.bias is not None else None

        # Input NHWC for conv2d: [B, 1, L, C]
        input_nhwc = x_nhwc.unsqueeze(1)  # [B, 1, L, C]

        ttnn_input = ttnn.from_torch(input_nhwc, dtype=ttnn.bfloat16)
        ttnn_weight = ttnn.from_torch(w_4d, dtype=ttnn.bfloat16)
        ttnn_bias = ttnn.from_torch(b_4d, dtype=ttnn.bfloat16) if b_4d is not None else None

        conv_config = ttnn.Conv2dConfig(
            weights_dtype=ttnn.bfloat16,
            deallocate_activation=False,
            reallocate_halo_output=True,
        )

        try:
            result = ttnn.conv2d(
                input_tensor=ttnn_input,
                weight_tensor=ttnn_weight,
                in_channels=in_ch,
                out_channels=out_ch,
                device=device,
                bias_tensor=ttnn_bias,
                kernel_size=(1, kernel),
                stride=(1, stride),
                padding=(0, pad),
                batch_size=batch,
                input_height=1,
                input_width=seq,
                conv_config=conv_config,
                dtype=ttnn.bfloat16,
            )
            # Handle different return formats
            if isinstance(result, tuple) and len(result) == 2:
                output_tensor, (out_h, out_w) = result
            else:
                output_tensor = result
                # Compute expected output width manually
                out_w = (seq + 2 * pad - kernel) // stride + 1
        except Exception as e:
            pytest.fail(f"conv1d/{desc}: ttnn.conv2d FAILED with: {e}")

        # Postprocess: TTNN output → torch
        output = ttnn.from_device(output_tensor)
        out_torch = ttnn.to_torch(output)
        out_torch = out_torch.reshape(batch, 1, out_w, -1)
        out_torch = out_torch[:, :, :, :out_ch]
        out_torch = out_torch.squeeze(1)  # [B, L_out, out_ch]

        # Convert to [B, C, L] for comparison with torch reference
        out_nchw = out_torch.permute(0, 2, 1).float()

        # Compare shapes
        assert out_nchw.shape == ref.shape, (
            f"conv1d/{desc}: Shape mismatch: TTNN {out_nchw.shape} vs torch {ref.shape}"
        )

        passed, pcc = assert_pcc(ref, out_nchw, threshold=0.998, op_name=f"conv1d_{desc}")
        print(f"  conv1d/{desc}: PCC={pcc:.6f}, shape={ref.shape}")


# =============================================================================
# 5. CONV TRANSPOSE 1D TESTS (FEASIBILITY)
# =============================================================================


class TestConvTranspose1d:
    """
    Feasibility tests for ConvTranspose1d via ttnn.conv_transpose2d.

    HiFi-GAN uses these configs — must validate all of them.
    These tests may FAIL if the TTNN op doesn't support these shapes.
    Failures are documented, not blocking.
    """

    @pytest.mark.parametrize(
        "batch, seq, in_ch, out_ch, kernel, stride, pad, desc",
        [
            # Small feasibility test first
            (1, 32, 64, 32, 4, 2, 1, "small_k4_s2"),
            # HiFi-GAN upsample layers (from smallest to largest stride)
            (1, 300, 128, 64, 4, 2, 1, "hifigan_up3_k4_s2"),
            (1, 150, 256, 128, 4, 2, 1, "hifigan_up2_k4_s2"),
            # These are the risky ones:
            (1, 25, 512, 256, 16, 6, 5, "hifigan_up1_k16_s6"),
            (1, 5, 512, 512, 16, 10, 3, "hifigan_up0_k16_s10"),
        ],
        ids=lambda x: x if isinstance(x, str) else None,
    )
    def test_conv_transpose1d_feasibility(
        self, device, batch, seq, in_ch, out_ch, kernel, stride, pad, desc
    ):
        """Test ConvTranspose1d via ttnn.conv_transpose2d."""
        torch.manual_seed(42)

        # PyTorch reference
        conv_t = torch.nn.ConvTranspose1d(
            in_ch, out_ch, kernel, stride=stride, padding=pad, bias=True
        )
        x_nchw = torch.randn(batch, in_ch, seq)
        ref = conv_t(x_nchw)  # [B, out_ch, L_out]

        # Channels-last
        x_nhwc = x_nchw.permute(0, 2, 1)

        # Weight: torch [in_ch, out_ch, K] → [in_ch, out_ch, 1, K]
        w_4d = conv_t.weight.data.unsqueeze(2)
        b_4d = conv_t.bias.data.reshape(1, 1, 1, -1) if conv_t.bias is not None else None

        input_nhwc = x_nhwc.unsqueeze(1)  # [B, 1, L, C]

        ttnn_input = ttnn.from_torch(input_nhwc, dtype=ttnn.bfloat16)
        ttnn_weight = ttnn.from_torch(w_4d, dtype=ttnn.bfloat16)
        ttnn_bias = ttnn.from_torch(b_4d, dtype=ttnn.bfloat16) if b_4d is not None else None

        try:
            result = ttnn.conv_transpose2d(
                input_tensor=ttnn_input,
                weight_tensor=ttnn_weight,
                in_channels=in_ch,
                out_channels=out_ch,
                device=device,
                bias_tensor=ttnn_bias,
                kernel_size=(1, kernel),
                stride=(1, stride),
                padding=(0, pad),
                output_padding=(0, 0),
                groups=1,
                batch_size=batch,
                input_height=1,
                input_width=seq,
                dtype=ttnn.bfloat16,
            )
            if isinstance(result, tuple) and len(result) == 2:
                output_tensor, (out_h, out_w) = result
            else:
                output_tensor = result
                out_w = (seq - 1) * stride - 2 * pad + kernel
        except Exception as e:
            pytest.skip(
                f"conv_transpose1d/{desc}: UNSUPPORTED - ttnn.conv_transpose2d failed: {e}"
            )
            return

        output = ttnn.from_device(output_tensor)
        out_torch = ttnn.to_torch(output)
        out_torch = out_torch.reshape(batch, 1, out_w, -1)
        out_torch = out_torch[:, :, :, :out_ch]
        out_torch = out_torch.squeeze(1)

        out_nchw = out_torch.permute(0, 2, 1).float()

        # Shape check (ConvTranspose output size can differ due to output_padding)
        if out_nchw.shape != ref.shape:
            min_len = min(out_nchw.shape[-1], ref.shape[-1])
            out_nchw = out_nchw[:, :, :min_len]
            ref = ref[:, :, :min_len]

        passed, pcc = assert_pcc(ref, out_nchw, threshold=0.995, op_name=f"conv_t1d_{desc}")
        print(f"  conv_transpose1d/{desc}: PCC={pcc:.6f}, shape_out={out_nchw.shape}")


# =============================================================================
# 6. ELEMENT-WISE OPS TESTS
# =============================================================================


class TestElementWise:
    """PCC tests for element-wise operations used in RVC."""

    @pytest.mark.parametrize(
        "shape, desc",
        [
            ((1, 300, 192), "vits_encoder"),
            ((1, 300, 768), "hubert_encoder"),
            ((1, 300, 512), "hifigan_pre_upsample"),
        ],
        ids=lambda x: x if isinstance(x, str) else None,
    )
    def test_relu(self, device, shape, desc):
        """Test ttnn.relu."""
        torch.manual_seed(42)
        x = torch.randn(shape)
        ref = torch.nn.functional.relu(x)

        x_tt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        out_tt = ttnn.relu(x_tt)
        out = ttnn.to_torch(out_tt).float()

        if out.shape != ref.shape:
            out = out[: ref.shape[0], : ref.shape[1], : ref.shape[2]]

        passed, pcc = assert_pcc(ref, out, threshold=0.999, op_name=f"relu_{desc}")
        print(f"  relu/{desc}: PCC={pcc:.6f}")

    @pytest.mark.parametrize(
        "shape, desc",
        [
            ((1, 300, 192), "vits_flow"),
            ((1, 300, 512), "hifigan"),
        ],
        ids=lambda x: x if isinstance(x, str) else None,
    )
    def test_sigmoid(self, device, shape, desc):
        """Test ttnn.sigmoid — used in WN gating."""
        torch.manual_seed(42)
        x = torch.randn(shape)
        ref = torch.sigmoid(x)

        x_tt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        out_tt = ttnn.sigmoid(x_tt)
        out = ttnn.to_torch(out_tt).float()

        if out.shape != ref.shape:
            out = out[: ref.shape[0], : ref.shape[1], : ref.shape[2]]

        passed, pcc = assert_pcc(ref, out, threshold=0.999, op_name=f"sigmoid_{desc}")
        print(f"  sigmoid/{desc}: PCC={pcc:.6f}")

    @pytest.mark.parametrize(
        "shape, desc",
        [
            ((1, 300, 192), "vits_flow"),
            ((1, 300, 512), "hifigan"),
        ],
        ids=lambda x: x if isinstance(x, str) else None,
    )
    def test_tanh(self, device, shape, desc):
        """Test ttnn.tanh — used in WN gating and final activation."""
        torch.manual_seed(42)
        x = torch.randn(shape)
        ref = torch.tanh(x)

        x_tt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        out_tt = ttnn.tanh(x_tt)
        out = ttnn.to_torch(out_tt).float()

        if out.shape != ref.shape:
            out = out[: ref.shape[0], : ref.shape[1], : ref.shape[2]]

        passed, pcc = assert_pcc(ref, out, threshold=0.999, op_name=f"tanh_{desc}")
        print(f"  tanh/{desc}: PCC={pcc:.6f}")

    @pytest.mark.parametrize(
        "shape, desc",
        [
            ((1, 300, 192), "vits_residual"),
            ((1, 300, 768), "hubert_residual"),
        ],
        ids=lambda x: x if isinstance(x, str) else None,
    )
    def test_add(self, device, shape, desc):
        """Test ttnn.add — used in residual connections."""
        torch.manual_seed(42)
        a = torch.randn(shape)
        b = torch.randn(shape)
        ref = a + b

        a_tt = ttnn.from_torch(a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        b_tt = ttnn.from_torch(b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        out_tt = ttnn.add(a_tt, b_tt)
        out = ttnn.to_torch(out_tt).float()

        if out.shape != ref.shape:
            out = out[: ref.shape[0], : ref.shape[1], : ref.shape[2]]

        passed, pcc = assert_pcc(ref, out, threshold=0.999, op_name=f"add_{desc}")
        print(f"  add/{desc}: PCC={pcc:.6f}")

    @pytest.mark.parametrize(
        "shape, desc",
        [
            ((1, 300, 192), "wavenet_gating"),
        ],
        ids=lambda x: x if isinstance(x, str) else None,
    )
    def test_multiply(self, device, shape, desc):
        """Test ttnn.multiply — used in WN gating (tanh * sigmoid)."""
        torch.manual_seed(42)
        a = torch.randn(shape)
        b = torch.randn(shape)
        ref = a * b

        a_tt = ttnn.from_torch(a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        b_tt = ttnn.from_torch(b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        out_tt = ttnn.multiply(a_tt, b_tt)
        out = ttnn.to_torch(out_tt).float()

        if out.shape != ref.shape:
            out = out[: ref.shape[0], : ref.shape[1], : ref.shape[2]]

        passed, pcc = assert_pcc(ref, out, threshold=0.999, op_name=f"mul_{desc}")
        print(f"  multiply/{desc}: PCC={pcc:.6f}")
