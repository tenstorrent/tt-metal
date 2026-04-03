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


class TtAcousticEncoder(LightweightModule):
    """AcousticEncoder -- Conv1d chains with SnakeBeta activation.

    Conv1d ops use precomputed weight-normed weights.
    SnakeBeta + anti-aliased resampling stays on CPU (custom activations).
    Runs on CPU via reference functions (Conv1d channels 48-1536 don't tile-align well).
    """

    def __init__(self, state_dict: Dict[str, torch.Tensor]):
        super().__init__()
        self.channels = ENCODER_CHANNELS
        self.strides = ENCODER_STRIDES

        # Precompute weight-normed initial conv
        self.initial_weight = weight_norm_compute(
            state_dict["conv_blocks.0.weight_g"],
            state_dict["conv_blocks.0.weight_v"],
        )
        self.initial_bias = state_dict["conv_blocks.0.bias"]

        # Precompute all encoder block weights
        self.block_weights = []
        for block_idx in range(5):
            prefix = f"conv_blocks.{block_idx + 1}."
            from models.demos.inworld_tts.reference.functional import _extract_encoder_block_weights

            bw = _extract_encoder_block_weights(state_dict, prefix, self.channels[block_idx])
            self.block_weights.append(bw)

        # Precompute final block weights
        final_prefix = "conv_final_block."
        self.final_alpha = state_dict[final_prefix + "0.act.alpha"]
        self.final_beta = state_dict[final_prefix + "0.act.beta"]
        self.final_up_filter = state_dict[final_prefix + "0.upsample.filter"]
        self.final_down_filter = state_dict[final_prefix + "0.downsample.lowpass.filter"]
        self.final_weight = weight_norm_compute(
            state_dict[final_prefix + "1.weight_g"],
            state_dict[final_prefix + "1.weight_v"],
        )
        self.final_bias = state_dict[final_prefix + "1.bias"]

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Forward pass on CPU with precomputed weights.

        Args:
            waveform: [B, 1, samples] input audio
        Returns:
            [B, 1024, T] acoustic features
        """
        # Initial conv: Conv1d(1, 48, k=7, pad=3)
        x = F.conv1d(waveform, self.initial_weight, self.initial_bias, padding=3)

        # 5 encoder blocks with progressive downsampling
        for block_idx in range(5):
            x = encoder_block_forward(x, self.block_weights[block_idx], self.strides[block_idx])

        # Final block: SnakeBeta(1536) + Conv1d(1536, 1024, k=3)
        x = activation1d_forward(
            x,
            self.final_alpha,
            self.final_beta,
            self.final_up_filter,
            self.final_down_filter,
        )
        x = F.conv1d(x, self.final_weight, self.final_bias, padding=1)

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
