# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Voxtral Codec Decoder — Phase 1 CPU implementation.

The codec decoder contains several non-standard ops:
  - Causal Conv1D with left-padding (weight_norm fused at load time)
  - ALiBi + causal + sliding-window attention with QK-norm and LayerScale
  - ConvTranspose1D for 2× upsampling at 3 stages

Phase 1 (correctness): Runs on CPU using the reference implementation.
This gives guaranteed PCC=1.0 and complete pipeline validation.
Phase 2 (optimization): Move transformer blocks to TTNN device.

The codec is 300M parameters; its runtime is small relative to the
26-layer text decoder prefill, so Phase 1 CPU is acceptable for now.

Block layout (all 8 decoder_blocks):
  Block 0: initial_conv [292→1024, k=3] — causal Conv1D
  Block 1: 2-layer attn+mlp (window=2, ALiBi+QK-norm+LayerScale)
  Block 2: ConvTranspose [1024→1024, k=4, stride=2]
  Block 3: 2-layer attn+mlp (window=4)
  Block 4: ConvTranspose [1024→1024, k=4, stride=2]
  Block 5: 2-layer attn+mlp (window=8)
  Block 6: ConvTranspose [1024→1024, k=4, stride=2]
  Block 7: 2-layer attn+mlp (window=16)
  output_proj: final causal Conv1D [1024→240, k=7] → waveform patches

Output: [B, N*1920] float32 at 24kHz (N frames × 8 upsample × 240 samples/frame)
"""

from models.common.lightweightmodule import LightweightModule
from models.demos.voxtral_tts.reference.functional import codec_decoder_forward


class TtVoxtralCodecDecoder(LightweightModule):
    """Phase 1 CPU codec decoder. Delegates to reference implementation."""

    def __init__(
        self,
        device,
        state_dict,  # codec decoder state dict (audio_tokenizer.* prefix stripped)
        weight_cache_path,
        dtype,
        configuration,
    ):
        super().__init__()
        self.device = device
        self.sd_codec = state_dict

    def forward(self, semantic_codes, acoustic_codes):
        """
        Decode discrete tokens to waveform.

        Args:
            semantic_codes: torch.Tensor [B, N], int64, values 0..8191
            acoustic_codes: torch.Tensor [B, N, 36], int64, values 0..20

        Returns:
            waveform: torch.Tensor [B, N*1920], float32 at 24kHz
        """
        waveform, _ = codec_decoder_forward(
            semantic_codes,
            acoustic_codes,
            self.sd_codec,
            capture_intermediates=False,
        )
        return waveform
