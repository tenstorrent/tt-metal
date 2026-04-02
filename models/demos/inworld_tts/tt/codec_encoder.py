"""TTNN implementation of the Inworld TTS codec encoder.

Pipeline: Waveform -> AcousticEncoder -> fusion with SemanticEncoder -> FSQ quantize -> VQ codes.

The encoder runs once per inference (not in the autoregressive loop), so we use
host-based execution for SnakeBeta and anti-aliased resampling. Conv1d ops use
TTNN where beneficial, but the encoder is not performance-critical.

CPU boundaries:
- SnakeBeta activation (custom, not available in TTNN)
- Anti-aliased resampling (FIR filter ops)
- FSQ quantization (codebook lookup)
- Wav2Vec2-BERT semantic features (external model)

TTNN accelerated:
- Large Conv1d operations (when channel dims are tile-friendly)
- fc_prior Linear projection
"""

from typing import Dict, Optional

import torch
import torch.nn.functional as F

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.inworld_tts.reference.functional import (
    _extract_encoder_block_weights,
    activation1d_forward,
    encoder_block_forward,
    weight_norm_compute,
)
from models.demos.inworld_tts.tt.model_config import get_compute_kernel_config_hifi4
from models.demos.inworld_tts.tt.wav2vec2_bert import TtWav2Vec2Bert


class TtAcousticEncoder(LightweightModule):
    """AcousticEncoder (CodecEnc) -- mostly host-based with precomputed weight norms.

    Since the encoder processes long sequences but runs only once per inference,
    we keep the implementation on CPU using the reference functions with precomputed
    weight normalization. This avoids the complexity of TTNN Conv1d for variable-length
    inputs with non-standard channel sizes (48, 96, etc.).
    """

    def __init__(self, state_dict: Dict[str, torch.Tensor]):
        super().__init__()

        self.channels = [48, 96, 192, 384, 768, 1536]
        self.strides = [2, 2, 4, 4, 5]

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
    """SemanticEncoder -- CPU-based since it's simple convolutions run once.

    Architecture: initial_conv -> N residual blocks (ReLU + Conv + ReLU + Conv) -> final_conv
    All Conv1d(1024, 1024, k=3) with regular weights (no weight norm).
    """

    def __init__(self, state_dict: Dict[str, torch.Tensor], prefix: str = "semantic_encoder."):
        super().__init__()
        self.prefix = prefix

        # Extract and store weights for fast access
        self.initial_conv_weight = state_dict[prefix + "initial_conv.weight"]
        self.initial_conv_bias = state_dict.get(prefix + "initial_conv.bias")

        # Count and store residual block weights
        # xcodec2 keys: residual_blocks.{idx}.weight where idx=1,3 for conv layers
        # (0,2 are ReLU activations in the Sequential)
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
            block_idx += 4  # skip: 0=ReLU, 1=Conv, 2=ReLU, 3=Conv

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
    """Full codec encoder: Wav2Vec2-BERT + AcousticEncoder + SemanticEncoder + Fusion + FSQ quantize.

    The encoder processes audio waveforms and produces integer VQ codes.
    It runs once per inference to encode the voice prompt, so performance
    is not critical. Wav2Vec2-BERT extracts semantic features from the waveform,
    eliminating the need for external semantic feature computation.

    For pragmatic reasons, the entire encoder pipeline runs on CPU with
    precomputed weight normalization. The TTNN device is used for fc_prior
    and the Wav2Vec2-BERT conformer layers.
    """

    def __init__(
        self,
        device,
        state_dict: Dict[str, torch.Tensor],
        quantizer=None,
        w2v_state_dict: Optional[Dict[str, torch.Tensor]] = None,
        dtype=ttnn.bfloat16,
        acoustic_prefix: str = "CodecEnc.",
        semantic_prefix: str = "SemanticEncoder_module.",
    ):
        super().__init__()
        self.device = device
        self.quantizer = quantizer

        # Wav2Vec2-BERT for semantic feature extraction
        # If w2v_state_dict is None, TtWav2Vec2Bert will load from HuggingFace
        self.w2v_bert = TtWav2Vec2Bert(device, state_dict=w2v_state_dict, dtype=dtype)

        # AutoFeatureExtractor for mel preprocessing (lazy import, cached)
        self._feature_extractor = None

        # Extract acoustic encoder weights (strip prefix)
        acoustic_sd = {}
        for k, v in state_dict.items():
            if k.startswith(acoustic_prefix):
                acoustic_sd[k[len(acoustic_prefix) :]] = v
        self.acoustic_encoder = TtAcousticEncoder(acoustic_sd)

        # Semantic encoder
        self.semantic_encoder = TtSemanticEncoder(state_dict, prefix=semantic_prefix)

        # fc_prior: Linear(2048, 2048) -- use TTNN for this large matmul
        fc_w = state_dict["fc_prior.weight"]
        fc_b = state_dict["fc_prior.bias"]
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

    @property
    def feature_extractor(self):
        """Lazy-load AutoFeatureExtractor for mel filterbank preprocessing."""
        if self._feature_extractor is None:
            from transformers import AutoFeatureExtractor

            self._feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")
        return self._feature_extractor

    def _extract_semantic_features(self, waveform: torch.Tensor, sample_rate: int = 16000) -> torch.Tensor:
        """Extract Wav2Vec2-BERT hidden_states[16] from a waveform.

        Args:
            waveform: [B, 1, samples] raw audio waveform (16kHz expected)
            sample_rate: audio sample rate (default 16000)
        Returns:
            [B, 1024, T] semantic features (transposed for SemanticEncoder)
        """
        # AutoFeatureExtractor expects 1D or 2D numpy/list input
        # Convert [B, 1, samples] -> list of 1D arrays
        wav_np = waveform.squeeze(1).cpu().numpy()  # [B, samples]
        inputs = self.feature_extractor(
            wav_np.tolist() if wav_np.shape[0] > 1 else wav_np[0],
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True,
        )
        input_features = inputs["input_features"]  # [B, T, 160]

        # Run Wav2Vec2-BERT encoder (16 conformer layers)
        hidden = self.w2v_bert(input_features.to(waveform.device))  # [B, T, 1024]

        # Transpose to [B, 1024, T] for SemanticEncoder
        return hidden.transpose(1, 2)

    def forward(
        self,
        waveform: torch.Tensor,
        semantic_features: Optional[torch.Tensor] = None,
        sample_rate: int = 16000,
    ) -> torch.Tensor:
        """Full codec encoder forward.

        If semantic_features is None, extracts them from the waveform using
        the built-in Wav2Vec2-BERT model. Otherwise uses the provided features.

        Args:
            waveform: [B, 1, samples] raw audio waveform
            semantic_features: optional [B, 1024, T] from Wav2Vec2-BERT hidden_states[16], transposed.
                             If None, computed automatically from waveform.
            sample_rate: audio sample rate for feature extraction (default 16000)
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

        # Step 3: Semantic encoder (CPU)
        semantic_out = self.semantic_encoder(semantic_features)  # [B, 1024, T]

        # Step 4: Fuse acoustic + semantic
        fused = torch.cat([acoustic_out, semantic_out], dim=1)  # [B, 2048, T]
        fused = fused.transpose(1, 2)  # [B, T, 2048]

        # Step 5: fc_prior projection (TTNN)
        fused_ttnn = ttnn.from_torch(
            fused.unsqueeze(0),  # [1, B, T, 2048]
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
        )
        projected = ttnn.linear(
            fused_ttnn,
            self.fc_prior_weight,
            bias=self.fc_prior_bias,
            compute_kernel_config=self.compute_kernel_config,
        )  # [1, B, T, 2048]

        # Back to CPU for FSQ quantization (needs float32 for quantizer Linear layers)
        projected_torch = ttnn.to_torch(projected).float()
        if projected_torch.dim() == 4:
            projected_torch = projected_torch.squeeze(0)  # [B, T, 2048]

        # Step 6: FSQ quantize (CPU)
        _, indices = self.quantizer(projected_torch)  # indices: [B, T, num_quantizers]

        # Convert to [B, 1, T]
        vq_codes = indices.squeeze(-1).unsqueeze(1)  # [B, 1, T]

        return vq_codes

    def forward_acoustic_only(self, waveform: torch.Tensor) -> torch.Tensor:
        """Run only the acoustic encoder (for testing).

        Args:
            waveform: [B, 1, samples]
        Returns:
            [B, 1024, T]
        """
        return self.acoustic_encoder(waveform)

    def forward_semantic_only(self, semantic_features: torch.Tensor) -> torch.Tensor:
        """Run only the semantic encoder (for testing).

        Args:
            semantic_features: [B, 1024, T]
        Returns:
            [B, 1024, T]
        """
        return self.semantic_encoder(semantic_features)

    def forward_w2v_only(self, input_features: torch.Tensor) -> torch.Tensor:
        """Run only the Wav2Vec2-BERT encoder (for testing).

        Args:
            input_features: [B, T, 160] mel filterbank features
        Returns:
            [B, T, 1024] hidden states
        """
        return self.w2v_bert(input_features)
