"""Inference generator with metal tracing support for Inworld TTS codec decoder.

Supports two execution modes:
1. Standard: Direct forward pass (for correctness verification)
2. Traced: Metal trace capture/execute (for production throughput)

The codec decoder is non-autoregressive (processes full sequence at once),
so tracing captures the entire VocosBackbone forward pass.
"""

import torch

import ttnn
from models.demos.inworld_tts.tt.codec_decoder import TtCodecDecoder
from models.demos.inworld_tts.tt.model_config import VOCOS_DIM


class CodecDecoderGenerator:
    """Generator with metal tracing for the codec decoder."""

    def __init__(
        self,
        device,
        state_dict,
        quantizer=None,
        dtype=ttnn.bfloat16,
        backbone_prefix="backbone.",
        head_prefix="head.",
    ):
        self.device = device
        self.quantizer = quantizer

        self.decoder = TtCodecDecoder(
            device=device,
            state_dict=state_dict,
            quantizer=quantizer,
            dtype=dtype,
            backbone_prefix=backbone_prefix,
            head_prefix=head_prefix,
        )

        self.trace_id = None
        self.trace_input = None
        self.trace_output = None

    def setup_trace(self, seq_len, dtype=ttnn.bfloat16):
        """Capture metal trace for the VocosBackbone.

        Traces the backbone forward pass (main compute) for fast replay.
        FSQ dequantize and ISTFT stay on CPU (not traced).

        Args:
            seq_len: Fixed sequence length for traced execution
            dtype: Input dtype
        """
        # Step 1: Warmup -- compile all operations
        warmup_input = torch.randn(1, seq_len, VOCOS_DIM)
        _ = self.decoder.backbone(warmup_input)

        # Step 2: Pre-allocate fixed input tensor on device
        input_spec = ttnn.TensorSpec(
            (1, 1, seq_len, VOCOS_DIM),
            ttnn.DataType.BFLOAT16,
            ttnn.TILE_LAYOUT,
            ttnn.DRAM_MEMORY_CONFIG,
        )
        self.trace_input = ttnn.allocate_tensor_on_device(input_spec, self.device)

        # Copy warmup data to fixed input
        host_input = ttnn.from_torch(
            warmup_input.unsqueeze(0),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
        ttnn.copy_host_to_device_tensor(host_input, self.trace_input)

        # Step 3: Capture trace
        self.trace_id = ttnn.begin_trace_capture(self.device, cq_id=0)
        self.trace_output = self.decoder.backbone(self.trace_input)
        ttnn.end_trace_capture(self.device, self.trace_id, cq_id=0)

        self.traced_seq_len = seq_len

    def generate(self, vq_codes):
        """Generate audio from VQ codes.

        Args:
            vq_codes: [B, T] or [B, 1, T] integer VQ codes (torch tensor)
        Returns:
            [B, 1, num_samples] audio waveform (torch tensor)
        """
        return self.decoder(vq_codes)

    def generate_traced(self, vq_codes):
        """Generate audio using traced execution for the backbone.

        Args:
            vq_codes: [B, T] or [B, 1, T] integer VQ codes (torch tensor)
                      T must match the traced sequence length
        Returns:
            [B, 1, num_samples] audio waveform (torch tensor)
        """
        if self.trace_id is None:
            raise RuntimeError("Call setup_trace() first")

        # Step 1: FSQ dequantize (CPU)
        vq_emb = self.decoder.fsq_dequantize(vq_codes)  # [B, T, 2048]

        # Step 2: fc_post_a projection (TTNN)
        vq_emb_ttnn = ttnn.from_torch(
            vq_emb.unsqueeze(0),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
        )
        projected = ttnn.linear(
            vq_emb_ttnn,
            self.decoder.fc_post_a_weight,
            bias=self.decoder.fc_post_a_bias,
        )  # [1, 1, T, 1024]

        # Step 3: Convert for trace input (need torch for embed conv + prior_net)
        # Note: The trace captures backbone.forward() which includes embed conv,
        # prior_net, transformers, post_net, and final layernorm.
        # For traced execution, we need to copy the projected data to the fixed input.
        projected_torch = ttnn.to_torch(projected).squeeze(0)  # [1, T, 1024]

        # Copy to trace input
        host_tensor = ttnn.from_torch(
            projected_torch.unsqueeze(0),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
        ttnn.copy_host_to_device_tensor(host_tensor, self.trace_input)

        # Step 4: Execute trace (fast replay)
        ttnn.execute_trace(self.device, self.trace_id, cq_id=0, blocking=True)

        # Step 5: ISTFTHead (CPU)
        hidden_torch = ttnn.to_torch(self.trace_output)
        if hidden_torch.dim() == 4:
            hidden_torch = hidden_torch.squeeze(0)

        audio = self.decoder.istft_head(hidden_torch)

        return audio

    def release_trace(self):
        """Release the captured trace."""
        if self.trace_id is not None:
            ttnn.release_trace(self.device, self.trace_id)
            self.trace_id = None
            self.trace_input = None
            self.trace_output = None
