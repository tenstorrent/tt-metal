"""TTNN implementation of the Inworld TTS codec encoder.

Pipeline: AcousticEncoder + Wav2Vec2-BERT -> SemanticEncoder -> Fusion -> FSQ quantize.

TTNN accelerated:
- SemanticEncoder: Conv1d + ReLU + residual add on device
- AcousticEncoder: Conv1d on device where possible, SnakeBeta on host
- fc_prior: Linear(2048, 2048) on device
- Wav2Vec2-BERT: FFN/Linear/LayerNorm on device (via TtWav2Vec2Bert)

CPU boundaries:
- SnakeBeta activation (custom: x + 1/beta * sin^2(alpha*x))
- Anti-aliased resampling (FIR filters)
- FSQ quantize (codebook + rounding)
- Feature extraction (AutoFeatureExtractor)
"""

from typing import Dict, Optional

import torch
import torch.nn.functional as F

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.inworld_tts.reference.functional import (
    activation1d_forward,
    weight_norm_compute,
)
from models.demos.inworld_tts.tt.model_config import ENCODER_CHANNELS, ENCODER_STRIDES, get_compute_kernel_config_hifi4
from models.demos.inworld_tts.tt.wav2vec2_bert import TtWav2Vec2Bert

L1 = ttnn.L1_MEMORY_CONFIG

def ttnn_snake_beta(x, alpha, beta):
    """SnakeBeta activation: x + (1/beta) * sin^2(alpha * x).

    alpha, beta: [C] learnable parameters, broadcast over [B, C, T].
    """
    alpha = alpha.reshape([1, 1, 1, -1])
    beta = beta.reshape([1, 1, 1, -1])
    return x + (1.0 / beta) * ttnn.pow(ttnn.sin(alpha * x), 2)


class TtActivation1d(LightweightModule):
    """Anti-aliased SnakeBeta activation using CPU reference implementation.

    CPU implementation that wraps activation1d_forward for compatibility with
    TtEncoderResidualUnit and TtEncoderBlock classes.
    """

    def __init__(
        self,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        up_filter: torch.Tensor,
        down_filter: torch.Tensor,
        channels : int,
        device,
    ):
        """Initialize anti-aliased SnakeBeta activation.

        Args:
            alpha: [C] SnakeBeta alpha parameter
            beta: [C] SnakeBeta beta parameter
            up_filter: [1, 1, K] FIR upsampling filter
            down_filter: [1, 1, K] FIR lowpass/downsampling filter
            channels: number of channels (C)
            device: TTNN device (unused, for API compatibility)
        """
        super().__init__()
        self.device = device
        self.channels = channels
        
        # Store parameters as ttnn tensors
        self.alpha = ttnn.from_torch(alpha, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        self.beta = ttnn.from_torch(beta, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        
        # Prepare filters
        # For conv_transpose2d: weight shape should be [in_channels, out_channels, kernel_h, kernel_w]
        # We have [1, 1, K], need to reshape to [C, 1, 1, K] for depthwise (groups=C)
        up_filter_expanded = up_filter.squeeze(0)  # [1, K]
        up_filter_4d = up_filter_expanded.unsqueeze(0).unsqueeze(0).expand(channels, 1, 1, -1)  # [C, 1, 1, K]
        self.up_filter = ttnn.from_torch(up_filter_4d * 2.0, dtype=ttnn.bfloat16)
        
        # For conv1d: weight shape should be [out_channels, in_channels//groups, kernel_size]
        # For depthwise (groups=C), this becomes [C, 1, K]
        down_filter_expanded = down_filter.squeeze(0).expand(channels, -1, -1)  # [C, 1, K]
        self.down_filter = ttnn.from_torch(down_filter_expanded, dtype=ttnn.bfloat16)
        
        # Store kernel sizes and padding
        self.K_up = up_filter_4d.shape[3]
        self.K_down = down_filter_expanded.shape[2]
        self.kernel_size = (1, self.K_up)
        self.pad_w = (self.K_up - 2) // 2
        
        # For downsampling conv1d padding
        self.pad_left = self.K_down // 2
        self.pad_right = self.K_down // 2 - 1 if self.K_down % 2 == 0 else self.K_down // 2
        
        # Conv2d configuration
        self.conv_config = ttnn.Conv2dConfig(
            weights_dtype=ttnn.float32,
            deallocate_activation=False,
            output_layout=ttnn.TILE_LAYOUT,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            act_block_h_override=32,
            config_tensors_in_dram=True,
        )
        self.compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=True,
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply anti-aliased SnakeBeta activation using TTNN operations.

        Args:
            x: [B, C, T] input tensor (torch)
        Returns:
            [B, C, T] output tensor (torch)
        """
        B, _, T, C = x.shape

        # Convert torch tensor to ttnn tensor: [B, C, T] -> [B, 1, T, C] (NHWC with H=1)
        
        # === UPSAMPLE using TTNN ConvTranspose2D (height=1 for 1D operation) ===
        x_up_ttnn, [out_h, out_w], [self.up_filter, _] = ttnn.conv_transpose2d(
            input_tensor=x,
            weight_tensor=self.up_filter,
            in_channels=C,
            out_channels=C,
            device=self.device,
            bias_tensor=None,
            kernel_size=self.kernel_size,  # (1, K)
            stride=(1, 2),  # 2x upsampling in width dimension
            padding=(0, self.pad_w),
            output_padding=(0, 0),
            batch_size=B,
            input_height=1,
            input_width=T,
            conv_config=self.conv_config,
            compute_config=self.compute_config,
            groups=C,  # Depthwise
            mirror_kernel=False,
            return_output_dim=True,
            return_weights_and_bias=True,
            dtype=ttnn.bfloat16,
        )
        
        # Apply SnakeBeta activation using TTNN ops
        x_act = ttnn_snake_beta(x_up_ttnn, self.alpha, self.beta)

        # === DOWNSAMPLE using TTNN Conv1d with stride=2 ===
        # Conv1d expects [B, 1, W, C] input
        x_down_ttnn, out_len, [self.down_filter, _] = ttnn.conv1d(
            input_tensor=x_act,
            weight_tensor=self.down_filter,
            in_channels=C,
            out_channels=C,
            device=self.device,
            bias_tensor=None,
            kernel_size=self.K_down,
            stride=2,  # Decimate by 2
            padding=(self.pad_left, self.pad_right),
            batch_size=B,
            input_length=out_w,  # Use output width from conv_transpose2d
            dtype=ttnn.bfloat16,
            conv_config=ttnn.Conv1dConfig(
                weights_dtype=ttnn.float32,
                deallocate_activation=False,
                shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                act_block_h_override=32,
                config_tensors_in_dram=True,
            ),
            compute_config=self.compute_config,
            groups=C,  # Depthwise
            return_output_dim=True,
            return_weights_and_bias=True,
        )
        
        # Convert back to torch tensor: [B, 1, T_out, C] -> [B, C, T_out]
        
        return x_down_ttnn




class TtConv1d(LightweightModule):
    def __init__(self, in_channels : int, out_channels: int, weight: torch.Tensor, bias: Optional[torch.Tensor], kernel_size: int, padding, device, stride=1):
        super().__init__()
        self.weight = ttnn.from_torch(weight, dtype=ttnn.bfloat16)
        # Reshape bias to [1, 1, 1, out_channels] for conv1d (implemented as conv2d with height=1)
        if bias is not None:
            bias_reshaped = bias.reshape(1, 1, 1, -1)
            self.bias = ttnn.from_torch(bias_reshaped, dtype=ttnn.bfloat16)
        else:
            self.bias = None
        self.conv_config = ttnn.Conv1dConfig(
            weights_dtype=ttnn.float32,
            deallocate_activation=True,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            act_block_h_override=32,
        )
        self.stride=stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.device = device
        self.padding = padding
        self.compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=True,
        )
    def forward(self, x) :
        """Run Conv1d on input tensor.

        Args:
            x: [B, C, T] torch tensor (channels-first format)
        Returns:
            [B, C, T] torch tensor
        """
        
        output_tensor, out_length, (self.weight, self.bias) = ttnn.conv1d(
            input_tensor=x,
            weight_tensor=self.weight,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            device=self.device,
            bias_tensor=self.bias,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            batch_size=x.shape[0],
            input_length=x.shape[-2],
            dtype=ttnn.bfloat16,
            conv_config=self.conv_config,
            compute_config=self.compute_config,  # fp32_dest_acc_en=True
            groups=1,
            return_output_dim=True,
            return_weights_and_bias=True,
        )
        
        return output_tensor


class TtEncoderResidualUnit(LightweightModule):
    """TTNN implementation of Encoder ResidualUnit.
    
    Architecture: Activation1d -> Conv1d(k=7) -> Activation1d -> Conv1d(k=1) + skip
    Uses TtActivation1d and TtConv1d for CPU-based operations.
    """
    
    def __init__(self, weights: Dict[str, torch.Tensor], device):
        """Initialize encoder residual unit.
        
        Args:
            weights: Dict with keys:
                act1_alpha, act1_beta, act1_up_filter, act1_down_filter,
                conv1_weight, conv1_bias,
                act2_alpha, act2_beta, act2_up_filter, act2_down_filter,
                conv2_weight, conv2_bias
            device: TTNN device
        """
        super().__init__()
        self.device = device
        

        C_in = weights["conv1_weight"].shape[1]  # input channels
        C_out = weights["conv1_weight"].shape[0]  # output channels
        # First activation (anti-aliased SnakeBeta)
        self.act1 = TtActivation1d(
            alpha=weights["act1_alpha"],
            beta=weights["act1_beta"],
            up_filter=weights["act1_up_filter"],
            down_filter=weights["act1_down_filter"],
            channels=C_in,
            device=device,
        )
        
        # First conv: k=7, pad=3
        self.conv1 = TtConv1d(
            in_channels=C_in,
            out_channels=C_out,
            weight=weights["conv1_weight"],
            bias=weights["conv1_bias"],
            kernel_size=7,
            padding=(3, 3),
            device=device,
        )
        
        # Second activation (anti-aliased SnakeBeta)
        self.act2 = TtActivation1d(
            alpha=weights["act2_alpha"],
            beta=weights["act2_beta"],
            up_filter=weights["act2_up_filter"],
            down_filter=weights["act2_down_filter"],
            channels=C_out,
            device=device,
        )
        
        # Second conv: k=1, pad=0
        C_in2 = weights["conv2_weight"].shape[1]
        C_out2 = weights["conv2_weight"].shape[0]
        self.conv2 = TtConv1d(
            in_channels=C_in2,
            out_channels=C_out2,
            weight=weights["conv2_weight"],
            bias=weights["conv2_bias"],
            kernel_size=1,
            padding=0,
            device=device,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply residual unit: act1 -> conv1 -> act2 -> conv2 + skip.
        
        Args:
            x: [B, C, T] input tensor
        Returns:
            [B, C, T] output tensor
        """
        # First activation + conv
        h = self.act1(x)
        h = self.conv1(h)
        # Second activation + conv
        h = self.act2(h)
        h = self.conv2(h)
        
        # Residual connection
        return x + h


class TtEncoderBlock(LightweightModule):
    """TTNN implementation of Encoder Block.
    
    Architecture: 3x ResidualUnit -> Activation1d -> Downsampling Conv1d
    Uses TtEncoderResidualUnit, TtActivation1d, and TtConv1d.
    """
    
    def __init__(self, weights: Dict[str, torch.Tensor], stride: int, device):
        """Initialize encoder block.
        
        Args:
            weights: Dict with keys:
                res_{0,1,2}_* (for 3 residual units),
                act_alpha, act_beta, act_up_filter, act_down_filter,
                downsample_weight, downsample_bias
            stride: Downsampling stride (2, 4, or 5)
            device: TTNN device
        """
        super().__init__()
        self.device = device
        self.stride = stride
        
        # Create 3 residual units
        self.residual_units = []
        for i in range(3):
            prefix = f"res_{i}_"
            res_weights = {k[len(prefix):]: v for k, v in weights.items() if k.startswith(prefix)}
            self.residual_units.append(TtEncoderResidualUnit(res_weights, device))
        

        C_in = weights["downsample_weight"].shape[1]  # input channels
        C_out = weights["downsample_weight"].shape[0]  # output channels

        # Final activation before downsampling
        self.final_activation = TtActivation1d(
            alpha=weights["act_alpha"],
            beta=weights["act_beta"],
            up_filter=weights["act_up_filter"],
            down_filter=weights["act_down_filter"],
            channels=C_in,
            device=device,
        )
        
        # Downsampling conv: kernel_size = stride * 2
        kernel_size = stride * 2
        pad_total = kernel_size - stride
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        
        # Note: For downsampling, we need to handle padding externally since
        # asymmetric padding (pad_left != pad_right) requires F.pad
        self.pad_left = pad_left
        self.pad_right = pad_right
        
        self.downsample_conv = TtConv1d(
            in_channels=C_in,
            out_channels=C_out,
            weight=weights["downsample_weight"],
            bias=weights["downsample_bias"],
            kernel_size=kernel_size,
            padding=(pad_left, pad_right),
            device=device,
            stride=stride,
        )
    
        # Store downsampling parameters for manual conv computation
        # (TtConv1d doesn't support variable stride, so we'll use reference for now)
        self.downsample_weight = weights["downsample_weight"]
        self.downsample_bias = weights["downsample_bias"]
        self.kernel_size = kernel_size
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply encoder block: 3x residual -> activation -> downsample.
        
        Args:
            x: [B, C_in, T] input tensor
        Returns:
            [B, C_out, T // stride] output tensor
        """
        # Apply 3 residual units
        for res_unit in self.residual_units:
            x = res_unit(x)
        
        # Apply final activation
        x = self.final_activation(x)
        # Apply downsampling conv (with asymmetric padding)
        x  = self.downsample_conv(x)
        return x


class TtAcousticEncoder(LightweightModule):
    """AcousticEncoder -- Conv1d chains with SnakeBeta activation.

    Conv1d ops use precomputed weight-normed weights.
    SnakeBeta + anti-aliased resampling stays on CPU (custom activations).
    Runs on CPU via reference functions (Conv1d channels 48-1536 don't tile-align well).
    """

    def __init__(self, state_dict: Dict[str, torch.Tensor], device):
        super().__init__()
        self.channels = ENCODER_CHANNELS
        self.strides = ENCODER_STRIDES

        # Precompute weight-normed initial conv
        self.initial_weight = weight_norm_compute(
            state_dict["conv_blocks.0.weight_g"],
            state_dict["conv_blocks.0.weight_v"],
        )
        self.initial_bias = state_dict["conv_blocks.0.bias"]
        self.initial_conv1d = TtConv1d(
            in_channels=1,
            out_channels=48,
            weight=self.initial_weight,
            bias=self.initial_bias,
            kernel_size=7,
            padding=3,  # k=7 Conv1d uses pad=3
            device=device,
        )

        # Create 5 encoder blocks
        self.encoder_blocks = []
        for block_idx in range(5):
            prefix = f"conv_blocks.{block_idx + 1}."
            from models.demos.inworld_tts.reference.functional import _extract_encoder_block_weights

            block_weights = _extract_encoder_block_weights(state_dict, prefix, self.channels[block_idx])
            self.encoder_blocks.append(TtEncoderBlock(block_weights, self.strides[block_idx], device))

        # Precompute final block weights
        final_prefix = "conv_final_block."
        self.final_activation = TtActivation1d(
            alpha=state_dict[final_prefix + "0.act.alpha"],
            beta=state_dict[final_prefix + "0.act.beta"],
            up_filter=state_dict[final_prefix + "0.upsample.filter"],
            down_filter=state_dict[final_prefix + "0.downsample.lowpass.filter"],
            channels=1536,
            device=device,
        )
        final_weight = weight_norm_compute(
            state_dict[final_prefix + "1.weight_g"],
            state_dict[final_prefix + "1.weight_v"],
        )
        final_bias = state_dict[final_prefix + "1.bias"]
        self.final_conv1d = TtConv1d(
            in_channels=1536,
            out_channels=1024,
            weight=final_weight,
            bias=final_bias,
            kernel_size=3,
            padding=1,
            device=device,
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Forward pass using TtEncoderBlock instances.

        Args:
            waveform: [B, 1, samples] input audio
        Returns:
            [B, 1024, T] acoustic features
        """
        # Initial conv: Conv1d(1, 48, k=7, pad=3)
        x = self.initial_conv1d(waveform)
        # 5 encoder blocks with progressive downsampling
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x)

        # Final block: SnakeBeta(1536) + Conv1d(1536, 1024, k=3)
        x = self.final_activation(x)
        x = self.final_conv1d(x)

        return x


class TtSemanticEncoder(LightweightModule):
    """SemanticEncoder -- CPU-based Conv1d chain.

    Architecture: initial_conv -> N residual blocks (ReLU + Conv + ReLU + Conv) -> final_conv
    All Conv1d(1024, 1024, k=3). Stays on CPU because ttnn.conv1d compilation is slow
    for 5 separate convolutions (~10min compile), while runtime is only ~3ms.
    """

    def __init__(self, device, state_dict: Dict[str, torch.Tensor], prefix: str = "SemanticEncoder_module."):
        super().__init__()

        self.initial_conv_weight = state_dict[prefix + "initial_conv.weight"]
        self.initial_conv_bias = state_dict.get(prefix + "initial_conv.bias")

        self.res_blocks = []
        block_idx = 1
        while prefix + f"residual_blocks.{block_idx}.weight" in state_dict:
            block = {
                "conv1_weight": state_dict[prefix + f"residual_blocks.{block_idx}.weight"],
                "conv1_bias": state_dict.get(prefix + f"residual_blocks.{block_idx}.bias"),
                "conv2_weight": state_dict[prefix + f"residual_blocks.{block_idx + 2}.weight"],
                "conv2_bias": state_dict.get(prefix + f"residual_blocks.{block_idx + 2}.bias"),
            }
            self.res_blocks.append(block)
            block_idx += 4

        self.final_conv_weight = state_dict[prefix + "final_conv.weight"]
        self.final_conv_bias = state_dict.get(prefix + "final_conv.bias")

    def forward(self, semantic_features: torch.Tensor) -> torch.Tensor:
        """Forward pass on CPU.

        Args:
            semantic_features: [B, 1024, T] from Wav2Vec2-BERT
        Returns:
            [B, 1024, T]
        """
        x = F.conv1d(semantic_features, self.initial_conv_weight, self.initial_conv_bias, padding=1)

        for block in self.res_blocks:
            res = x
            h = F.relu(x)
            h = F.conv1d(h, block["conv1_weight"], block["conv1_bias"], padding=1)
            h = F.relu(h)
            h = F.conv1d(h, block["conv2_weight"], block["conv2_bias"], padding=1)
            x = res + h

        x = F.conv1d(x, self.final_conv_weight, self.final_conv_bias, padding=1)
        return x


class TtCodecEncoder(LightweightModule):
    """Full codec encoder: Wav2Vec2-BERT + AcousticEncoder + SemanticEncoder + Fusion + FSQ quantize."""

    def __init__(
        self,
        device,
        state_dict: Dict[str, torch.Tensor],
        quantizer=None,
        dtype=ttnn.bfloat16,
        acoustic_prefix: str = "CodecEnc.",
        semantic_prefix: str = "SemanticEncoder_module.",
        w2v_state_dict: Optional[Dict[str, torch.Tensor]] = None,
    ):
        super().__init__()
        self.device = device
        self.quantizer = quantizer

        # Acoustic encoder (CPU -- SnakeBeta + non-standard channel sizes)
        acoustic_sd = {k[len(acoustic_prefix) :]: v for k, v in state_dict.items() if k.startswith(acoustic_prefix)}
        self.acoustic_encoder = TtAcousticEncoder(acoustic_sd)

        # Wav2Vec2-BERT (TTNN -- FFN/Linear/LayerNorm on device)
        self.w2v_bert = TtWav2Vec2Bert(device, state_dict=w2v_state_dict, dtype=dtype)

        # Semantic encoder (TTNN -- Conv1d + ReLU + Add on device)
        self.semantic_encoder = TtSemanticEncoder(device, state_dict, prefix=semantic_prefix)

        # Feature extractor (lazy-loaded)
        self._feature_extractor = None

        # fc_prior: Linear(2048, 2048) -- TTNN
        fc_w = state_dict["fc_prior.weight"]
        fc_b = state_dict["fc_prior.bias"]
        grid = device.compute_with_storage_grid_size()
        self.core_grid = ttnn.CoreGrid(y=grid.y, x=grid.x)
        self.fc_prior_weight = ttnn.from_torch(
            fc_w.T.unsqueeze(0).unsqueeze(0),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.fc_prior_bias = ttnn.from_torch(
            fc_b.unsqueeze(0).unsqueeze(0).unsqueeze(0),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.compute_kernel_config = get_compute_kernel_config_hifi4()

    def _get_feature_extractor(self):
        """Lazy-load AutoFeatureExtractor for mel filterbank preprocessing."""
        if self._feature_extractor is None:
            from transformers import AutoFeatureExtractor

            self._feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")
        return self._feature_extractor

    def _extract_semantic_features(self, waveform: torch.Tensor, sample_rate: int = 16000) -> torch.Tensor:
        """Extract semantic features from waveform using Wav2Vec2-BERT.

        Args:
            waveform: [B, 1, samples] raw audio
        Returns:
            [B, 1024, T] semantic features (channels-first for SemanticEncoder)
        """
        fe = self._get_feature_extractor()
        audio_np = waveform.squeeze(1).numpy()
        inputs = fe(audio_np, sampling_rate=sample_rate, return_tensors="pt", padding=True)
        input_features = inputs["input_features"].to(waveform.device)

        # Run Wav2Vec2-BERT (returns [B, T, 1024])
        hidden = self.w2v_bert(input_features)

        # Transpose to channels-first [B, 1024, T] for SemanticEncoder
        return hidden.transpose(1, 2)

    def forward(
        self,
        waveform: torch.Tensor,
        semantic_features: Optional[torch.Tensor] = None,
        sample_rate: int = 16000,
    ) -> torch.Tensor:
        """Full codec encoder forward.

        Args:
            waveform: [B, 1, samples] raw audio waveform
            semantic_features: optional [B, 1024, T] pre-computed. If None, extracted via Wav2Vec2-BERT.
            sample_rate: audio sample rate (default 16000)
        Returns:
            [B, 1, T] integer VQ codes
        """
        if self.quantizer is None:
            raise ValueError("Quantizer required for FSQ quantization")

        # Step 1: Acoustic encoder (CPU)
        acoustic_out = self.acoustic_encoder(waveform)  # [B, 1024, T]

        # Step 2: Extract or use provided semantic features
        if semantic_features is None:
            semantic_features = self._extract_semantic_features(waveform, sample_rate)

        # Step 3: Semantic encoder (TTNN Conv1d + ReLU)
        semantic_out = self.semantic_encoder(semantic_features)  # [B, 1024, T]

        # Step 4: Fuse acoustic + semantic
        fused = torch.cat([acoustic_out, semantic_out], dim=1)  # [B, 2048, T]
        fused = fused.transpose(1, 2)  # [B, T, 2048]

        # Step 5: fc_prior projection (TTNN)
        fused_ttnn = ttnn.from_torch(
            fused.unsqueeze(0),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
        )
        projected = ttnn.linear(
            fused_ttnn,
            self.fc_prior_weight,
            bias=self.fc_prior_bias,
            core_grid=self.core_grid,
            memory_config=L1,
            compute_kernel_config=self.compute_kernel_config,
        )

        # Back to CPU for FSQ quantization (needs float32)
        projected_torch = ttnn.to_torch(projected).float()
        if projected_torch.dim() == 4:
            projected_torch = projected_torch.squeeze(0)

        # Step 6: FSQ quantize (CPU)
        _, indices = self.quantizer(projected_torch)
        vq_codes = indices.squeeze(-1).unsqueeze(1)  # [B, 1, T]

        return vq_codes

    def forward_acoustic_only(self, waveform: torch.Tensor) -> torch.Tensor:
        """Run only the acoustic encoder."""
        return self.acoustic_encoder(waveform)

    def forward_semantic_only(self, semantic_features: torch.Tensor) -> torch.Tensor:
        """Run only the semantic encoder."""
        return self.semantic_encoder(semantic_features)

    def forward_w2v_only(self, input_features: torch.Tensor) -> torch.Tensor:
        """Run only the Wav2Vec2-BERT model."""
        return self.w2v_bert(input_features)
