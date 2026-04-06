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
    encoder_block_forward,
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


class TtActivation1dTTNN(LightweightModule):
    """Anti-aliased SnakeBeta activation using TTNN operations.

    TTNN implementation:
    - Upsample 2x: ttnn.conv_transpose2d (with height=1 to mimic conv_transpose1d)
    - SnakeBeta activation using TTNN ops
    - Downsample 2x: ttnn.conv1d with stride=2
    """

    def __init__(
        self,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        up_filter: torch.Tensor,
        down_filter: torch.Tensor,
        device,
    ):
        """Initialize anti-aliased SnakeBeta activation with TTNN.

        Args:
            alpha: [C] SnakeBeta alpha parameter
            beta: [C] SnakeBeta beta parameter
            up_filter: [1, 1, K] FIR upsampling filter
            down_filter: [1, 1, K] FIR lowpass/downsampling filter
            device: TTNN device
        """
        super().__init__()
        self.device = device
        
        # Store parameters
        self.alpha = ttnn.from_torch(alpha, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)
        self.beta = ttnn.from_torch(beta, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)
        self.up_kernel = ttnn.from_torch(up_filter.squeeze(0), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)  # Will be expanded per call
        self.down_kernel = ttnn.from_torch(down_filter.squeeze(0), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)  # Will be expanded per call
        self.K_up = self.up_kernel.shape[-1]
        self.K_down = self.down_kernel.shape[-1]
        
        # Compute padding
        self.pad_left = self.K_up // 2
        self.pad_right = self.K_up // 2 - 1 if self.K_up % 2 == 0 else self.K_up // 2
        self.pad_w = (self.K_up - 2) // 2
        
        # Conv2d configuration (reusable)
        self.conv_config = ttnn.Conv2dConfig(
            weights_dtype=ttnn.float32,
            deallocate_activation=False,
            output_layout=ttnn.TILE_LAYOUT,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            act_block_h_override=32,
        )
        self.compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply anti-aliased SnakeBeta activation using TTNN.

        Args:
            x: [B, C, T] input tensor (torch)
        Returns:
            [B, C, T] output tensor (torch)
        """
        B, C, T = x.shape

        # Expand kernels for this batch
        up_kernel = self.up_kernel.expand(C, -1, -1)  # [C, 1, K]
        down_kernel = self.down_kernel.expand(C, -1, -1)  # [C, 1, K]

        # === UPSAMPLE using TTNN ConvTranspose2D (height=1 for 1D operation) ===
        # Reshape input for 2D conv: [B, C, T] -> [B, 1, T, C] (NHWC with H=1)
        x_nhwc = x.unsqueeze(1).permute(0, 1, 3, 2)  # [B, 1, T, C]
        x_ttnn = ttnn.from_torch(x_nhwc, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)
        
        # Prepare weight for conv_transpose2d: [C, 1, 1, K] (IOHW format)
        up_weight_2d = up_kernel.unsqueeze(2) * 2.0  # [C, 1, 1, K]
        
        # ConvTranspose2D with height=1, width=T, stride=(1,2) for 2x upsampling in width
        x_up_ttnn, [out_h, out_w], [self.up_kernel, _] = ttnn.conv_transpose2d(
            input_tensor=x_ttnn,
            weight_tensor=self.up_kernel,  # [C, 1, 1, K]
            in_channels=C,
            out_channels=C,
            device=self.device,
            bias_tensor=None,
            kernel_size=(1, self.K_up),  # height=1, width=K
            stride=(1, 2),  # no stride in height, 2x in width for upsampling
            padding=(0, self.pad_w),  # no padding in height
            output_padding=(0, 0),
            batch_size=B,
            input_height=1,  # height dimension is 1
            input_width=T,  # width dimension is time
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
        # Prepare downsampling kernel
        
        # Apply downsampling filter with stride=2 via TTNN
        x_down_ttnn, out_len, [self.down_kernel, _] = ttnn.conv1d(
            input_tensor=x_act,
            weight_tensor=self.down_kernel,
            in_channels=C,
            out_channels=C,
            device=self.device,
            bias_tensor=None,
            kernel_size=self.K_down,
            stride=2,  # Decimate by 2
            padding=(self.pad_left, self.pad_right),
            batch_size=B,
            input_length=x_act.shape[2],
            dtype=ttnn.bfloat16,
            conv_config=ttnn.Conv1dConfig(
                weights_dtype=ttnn.float32,
                deallocate_activation=False,
                shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                act_block_h_override=32,
            ),
            compute_config=self.compute_config,
            groups=C,  # Depthwise
            return_output_dim=True,
            return_weights_and_bias=True,
        )
        

        return x_down_ttnn


# Legacy function wrapper for backward compatibility
def activation1d_forward_ttnn(
    x: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    up_filter: torch.Tensor,
    down_filter: torch.Tensor,
    device,
) -> torch.Tensor:
    """Anti-aliased SnakeBeta activation using TTNN operations.
    
    Legacy function wrapper. Consider using TtActivation1dTTNN class instead.
    """
    activator = TtActivation1dTTNN(alpha, beta, up_filter, down_filter, device)
    return activator.forward(x)



class TtActivation1d(LightweightModule):
    """Anti-aliased SnakeBeta activation using TTNN ops where possible.
    
    Pipeline: Upsample 2x -> SnakeBeta -> Downsample 2x to avoid aliasing.
    SnakeBeta: x + (1/beta) * sin^2(alpha * x)
    
    Due to the custom anti-aliasing with FIR filters and complex upsampling/downsampling,
    this currently falls back to CPU but accepts/returns tensors compatible with TTNN pipeline.
    """
    
    def __init__(
        self,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        up_filter: torch.Tensor,
        down_filter: torch.Tensor,
        device=None,
    ):
        """Initialize anti-aliased SnakeBeta activation.
        
        Args:
            alpha: [C] SnakeBeta alpha parameter
            beta: [C] SnakeBeta beta parameter
            up_filter: [1, 1, K] FIR upsampling filter
            down_filter: [1, 1, K] FIR lowpass/downsampling filter
            device: Optional TTNN device (for future acceleration)
        """
        super().__init__()
        # Store as torch tensors for CPU computation
        self.alpha = alpha
        self.beta = beta
        self.up_filter = up_filter
        self.down_filter = down_filter
        self.device = device
    
    def _snake_beta_ttnn(self, x: torch.Tensor) -> torch.Tensor:
        """SnakeBeta activation: x + (1/beta) * sin^2(alpha * x).
        
        Args:
            x: [B, C, T] input tensor
        Returns:
            [B, C, T] activated tensor
        """
        # Reshape alpha and beta for broadcasting: [C] -> [1, C, 1]
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)  # [1, C, 1]
        beta = self.beta.unsqueeze(0).unsqueeze(-1)  # [1, C, 1]
        
        # SnakeBeta: x + (1/beta) * sin^2(alpha * x)
        scaled = alpha * x
        sin_val = torch.sin(scaled)
        sin_sq = sin_val * sin_val
        result = x + (1.0 / beta) * sin_sq
        
        return result
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply anti-aliased SnakeBeta activation.
        
        Handles both torch.Tensor and ttnn.Tensor inputs.
        For ttnn inputs, converts to torch, processes, and converts back.
        
        Args:
            x: [B, C, T] input tensor (channels-first) - torch.Tensor or ttnn.Tensor
        Returns:
            [B, C, T] activated tensor in same format as input
        """
        # Check if input is TTNN tensor
        is_ttnn_input = hasattr(x, 'device') and hasattr(x.device, 'arch')
        
        # Convert TTNN to torch if needed
        if is_ttnn_input:
            x_torch = ttnn.to_torch(x)
        else:
            x_torch = x
        
        # Run CPU-based anti-aliased activation using reference implementation
        # (FIR filtering with zero insertion/decimation is complex to implement in TTNN)
        result = activation1d_forward(
            x_torch,
            self.alpha,
            self.beta,
            self.up_filter,
            self.down_filter,
        )
        
        # Convert back to TTNN if input was TTNN
        if is_ttnn_input and self.device is not None:
            result = ttnn.from_torch(
                result,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
            )
        
        return result


class TtConv1d(LightweightModule):
    def __init__(self, in_channels : int, out_channels: int, weight: torch.Tensor, bias: Optional[torch.Tensor], kernel_size: int, padding, device):
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
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run Conv1d on input tensor.

        Args:
            x: [B, C, T] torch tensor (channels-first format)
        Returns:
            [B, C, T] torch tensor
        """
        # Convert torch tensor to ttnn tensor
        x_ttnn = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)
        
        output_tensor, out_length, (self.weight, self.bias) = ttnn.conv1d(
            input_tensor=x_ttnn,
            weight_tensor=self.weight,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            device=self.device,
            bias_tensor=self.bias,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.padding,
            batch_size=x.shape[0],
            input_length=x.shape[2],
            dtype=ttnn.bfloat16,
            conv_config=self.conv_config,
            compute_config=self.compute_config,  # fp32_dest_acc_en=True
            groups=1,
            return_output_dim=True,
            return_weights_and_bias=True,
        )
        
        # Convert back to torch tensor
        output_torch = ttnn.to_torch(output_tensor)
        
        # Reshape from [B, 1, T, C] (NHWC) to [B, C, T] (BCT)
        # ttnn.conv1d returns 4D output in NHWC format (batch, height=1, time, channels)
        if output_torch.dim() == 4:
            output_torch = output_torch.squeeze(1).permute(0, 2, 1)  # [B, 1, T, C] -> [B, T, C] -> [B, C, T]
        
        # Convert to float32 to match expected dtype for downstream operations
        output_torch = output_torch.float()
        
        return output_torch
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

        # Precompute all encoder block weights
        self.block_weights = []
        for block_idx in range(5):
            prefix = f"conv_blocks.{block_idx + 1}."
            from models.demos.inworld_tts.reference.functional import _extract_encoder_block_weights

            bw = _extract_encoder_block_weights(state_dict, prefix, self.channels[block_idx])
            self.block_weights.append(bw)

        # Precompute final block weights
        final_prefix = "conv_final_block."
        self.final_activation = TtActivation1d(
            alpha=state_dict[final_prefix + "0.act.alpha"],
            beta=state_dict[final_prefix + "0.act.beta"],
            up_filter=state_dict[final_prefix + "0.upsample.filter"],
            down_filter=state_dict[final_prefix + "0.downsample.lowpass.filter"],
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
        """Forward pass on CPU with precomputed weights.

        Args:
            waveform: [B, 1, samples] input audio
        Returns:
            [B, 1024, T] acoustic features
        """
        # Initial conv: Conv1d(1, 48, k=7, pad=3)
        x = self.initial_conv1d(waveform)
        return x
        # 5 encoder blocks with progressive downsampling
        for block_idx in range(5):
            x = encoder_block_forward(x, self.block_weights[block_idx], self.strides[block_idx])

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
