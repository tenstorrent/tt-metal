"""Inference generators with metal tracing support for Inworld TTS codec.

Supports two execution modes:
1. Standard: Direct forward pass (for correctness verification)
2. Traced: Metal trace capture/execute (for production throughput)

Generators:
- CodecDecoderGenerator: Traces fc_post_a -> VocosBackbone -> ISTFT head linear
- CodecEncoderGenerator: Traces acoustic encoder (5-block loop + final) and
  semantic encoder (initial conv -> res blocks -> final conv)
"""

import torch

import ttnn
from models.demos.inworld_tts.tt.codec_decoder import TtCodecDecoder
from models.demos.inworld_tts.tt.model_config import FSQ_VQ_DIM


class CodecDecoderGenerator:
    """Generator with metal tracing for the codec decoder.

    Traced path: fc_post_a Linear(2048, 1024) -> VocosBackbone -> ISTFT head Linear(1024, 1282).
    Host path: FSQ dequantize (before trace) and ISTFT FFT (after trace).
    """

    def __init__(
        self,
        device,
        state_dict,
        quantizer=None,
        dtype=ttnn.bfloat16,
        backbone_prefix="backbone.",
        head_prefix="head.",
        **decoder_kwargs,
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
            **decoder_kwargs,
        )

        self.trace_id = None
        self.trace_input = None
        self.trace_output = None

    def _traced_forward(self, x_tt):
        """The full TTNN decoder path: fc_post_a -> backbone -> ISTFT head linear.

        All ops are pure device ops (no from_torch/to_torch).

        Args:
            x_tt: [1, 1, T, 2048] TTNN tensor on device (FSQ dequant output)
        Returns:
            [1, 1, T, 1282] TTNN tensor on device (before ISTFT FFT)
        """
        grid = self.decoder.core_grid
        compute_cfg = self.decoder.compute_kernel_config

        # fc_post_a: Linear(2048, 1024)
        projected = ttnn.linear(
            x_tt,
            self.decoder.fc_post_a_weight,
            bias=self.decoder.fc_post_a_bias,
            core_grid=grid,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=compute_cfg,
        )

        # VocosBackbone
        hidden = self.decoder.backbone(projected)

        # ISTFT head linear: Linear(1024, 1282)
        out = ttnn.linear(
            hidden,
            self.decoder.istft_linear_weight,
            bias=self.decoder.istft_linear_bias,
            core_grid=grid,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=compute_cfg,
        )
        return out

    def setup_trace(self, seq_len, dtype=ttnn.bfloat16):
        """Capture metal trace for the full TTNN decoder path.

        Traces: fc_post_a Linear -> VocosBackbone -> ISTFT head Linear.
        FSQ dequantize and ISTFT FFT stay on CPU (not traced).

        Args:
            seq_len: Fixed sequence length for traced execution
            dtype: Input dtype
        """
        # Step 1: Warmup -- compile all operations
        warmup_data = torch.randn(1, 1, seq_len, FSQ_VQ_DIM)
        warmup_tt = ttnn.from_torch(
            warmup_data,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
        )
        warmup_out = self._traced_forward(warmup_tt)
        # Force sync to ensure compilation is done
        _ = ttnn.to_torch(warmup_out)

        # Step 2: Pre-allocate fixed input tensor on device
        input_spec = ttnn.TensorSpec(
            (1, 1, seq_len, FSQ_VQ_DIM),
            ttnn.DataType.BFLOAT16,
            ttnn.TILE_LAYOUT,
        )
        self.trace_input = ttnn.allocate_tensor_on_device(input_spec, self.device)

        # Copy warmup data to fixed input
        host_input = ttnn.from_torch(
            warmup_data,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
        ttnn.copy_host_to_device_tensor(host_input, self.trace_input)

        # Step 3: Capture trace
        self.trace_id = ttnn.begin_trace_capture(self.device, cq_id=0)
        self.trace_output = self._traced_forward(self.trace_input)
        ttnn.end_trace_capture(self.device, self.trace_id, cq_id=0)

        self.traced_seq_len = seq_len

    def generate(self, vq_codes):
        """Generate audio from VQ codes (non-traced).

        Args:
            vq_codes: [B, T] or [B, 1, T] integer VQ codes (torch tensor)
        Returns:
            [B, 1, num_samples] audio waveform (torch tensor)
        """
        return self.decoder(vq_codes)

    def generate_traced(self, vq_codes):
        """Generate audio using traced execution.

        Traced path: fc_post_a -> VocosBackbone -> ISTFT head linear (device).
        Host path: FSQ dequantize (before) and ISTFT FFT (after).

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

        # Step 2: Copy to trace input
        host_tensor = ttnn.from_torch(
            vq_emb.unsqueeze(0),  # [1, B, T, 2048]
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
        ttnn.copy_host_to_device_tensor(host_tensor, self.trace_input)

        # Step 3: Execute trace (fast replay)
        ttnn.execute_trace(self.device, self.trace_id, cq_id=0, blocking=True)

        # Step 4: Read ISTFT head linear output and do ISTFT FFT on CPU
        x_pred = ttnn.to_torch(self.trace_output).float()
        if x_pred.dim() == 4:
            x_pred = x_pred.squeeze(0)  # [B, T, 1282]
        x_pred = x_pred.transpose(1, 2)  # [B, 1282, T]

        # ISTFT signal processing (same as TtCodecDecoder.istft_head but without the linear)
        audio = self.decoder._istft_from_pred(x_pred)
        return audio

    def release_trace(self):
        """Release the captured trace."""
        if self.trace_id is not None:
            ttnn.release_trace(self.device, self.trace_id)
            self.trace_id = None
            self.trace_input = None
            self.trace_output = None


class CodecEncoderGenerator:
    """Generator with metal tracing for the codec encoder components.

    Traces:
    - Wav2Vec2-BERT: feature projection LN + Linear + 16 conformer layers
      Input: [1, 1, T, 160] TILE_LAYOUT
      Output: [1, 1, T, 1024] TILE_LAYOUT

    - AcousticEncoder: 5-block encoder loop + final activation + final conv
      Input: [1, 1, T_initial, 48] ROW_MAJOR (after initial conv)
      Output: [1, 1, T_final, 1024] ROW_MAJOR

    - SemanticEncoder: initial conv -> res blocks -> final conv
      Input: [1, 1, T, 1024] ROW_MAJOR
      Output: [1, 1, T, 1024] ROW_MAJOR

    Trace requires fixed input shapes, so waveform length must be known at setup time.
    """

    def __init__(self, device, acoustic_encoder, semantic_encoder=None, w2v_bert=None):
        """Initialize encoder generator.

        Args:
            device: TTNN device
            acoustic_encoder: TtAcousticEncoder instance
            semantic_encoder: Optional TtSemanticEncoder instance
            w2v_bert: Optional TtWav2Vec2Bert instance
        """
        self.device = device
        self.acoustic_encoder = acoustic_encoder
        self.semantic_encoder = semantic_encoder
        self.w2v_bert = w2v_bert

        # Acoustic encoder trace state
        self.acoustic_trace_id = None
        self.acoustic_trace_input = None
        self.acoustic_trace_output = None
        self.acoustic_initial_T = None  # T after initial conv

        # Semantic encoder trace state
        self.semantic_trace_id = None
        self.semantic_trace_input = None
        self.semantic_trace_output = None
        self.semantic_T = None

        # Wav2Vec2-BERT trace state
        self.w2v_trace_id = None
        self.w2v_trace_input = None
        self.w2v_trace_output = None
        self.w2v_T = None

    def _w2v_forward_ttnn(self, x_tt):
        """The traceable portion of Wav2Vec2-BERT: feature projection + conformer layers.

        All ops are pure device ops (no from_torch/to_torch).

        Args:
            x_tt: [1, 1, T, 160] TTNN TILE_LAYOUT on device
        Returns:
            [1, 1, T, 1024] TTNN TILE_LAYOUT on device
        """
        return self.w2v_bert.forward_ttnn(x_tt)

    def _acoustic_forward_ttnn(self, x_tt, T_initial):
        """The traceable portion of AcousticEncoder: 5-block loop + final.

        All ops are pure device ops (forward_ttnn calls).

        Args:
            x_tt: [1, 1, T_initial, 48] ROW_MAJOR on device
            T_initial: time dimension after initial conv
        Returns:
            [1, 1, T_final, 1024] ROW_MAJOR on device
        """
        enc = self.acoustic_encoder
        C = 48
        T = T_initial
        for block_idx in range(5):
            for res_idx in range(3):
                res = x_tt
                act1, act2 = enc.block_activations[block_idx][res_idx]
                conv1, conv2 = enc.block_res_convs[block_idx][res_idx]
                x_tt = act1.forward_ttnn(x_tt, C, T)
                x_tt, _ = conv1.forward_ttnn(x_tt, T)
                x_tt = act2.forward_ttnn(x_tt, C, T)
                x_tt, _ = conv2.forward_ttnn(x_tt, T)
                x_tt = ttnn.add(res, x_tt)
            x_tt = enc.block_final_act[block_idx].forward_ttnn(x_tt, C, T)
            C_out = enc.channels[block_idx + 1]
            x_tt, T = enc.block_downsample[block_idx].forward_ttnn(x_tt, T)
            C = C_out

        # Final block: SnakeBeta(1536) + Conv1d(1536, 1024, k=3)
        x_tt = enc.final_activation.forward_ttnn(x_tt, C, T)
        x_tt, _ = enc.final_conv1d.forward_ttnn(x_tt, T)
        return x_tt

    def _semantic_forward_ttnn(self, x_tt, T):
        """The traceable portion of SemanticEncoder: all convs + relus + residuals.

        All ops are pure device ops.

        Args:
            x_tt: [1, 1, T, 1024] ROW_MAJOR on device
            T: time dimension
        Returns:
            [1, 1, T, 1024] ROW_MAJOR on device (from final conv, in DRAM)
        """
        enc = self.semantic_encoder

        # Initial conv
        x = enc._conv1d(x_tt, "init", enc.initial_w, enc.initial_b, T)

        # Residual blocks
        for i, (c1w, c1b, c2w, c2b) in enumerate(enc.res_blocks):
            res = x
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
            x = ttnn.relu(x)
            x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
            x = enc._conv1d(x, f"res{i}_c1", c1w, c1b, T)
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
            x = ttnn.relu(x)
            x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
            x = enc._conv1d(x, f"res{i}_c2", c2w, c2b, T)
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
            res = ttnn.to_layout(res, ttnn.TILE_LAYOUT)
            x = ttnn.add(res, x)
            x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)

        # Final conv
        x = enc._conv1d(x, "final", enc.final_w, enc.final_b, T)
        return x

    def setup_w2v_trace(self, seq_len):
        """Capture metal trace for Wav2Vec2-BERT.

        Traces: feature projection LN + Linear(160, 1024) + 16 conformer layers.
        The mel filterbank feature extraction stays on CPU.

        The first warmup call populates the distance index cache in each
        TtW2vSelfAttention layer, so the trace capture does not hit from_torch.

        Args:
            seq_len: Fixed sequence length T for mel features [1, 1, T, 160]
        """
        if self.w2v_bert is None:
            raise RuntimeError("TtWav2Vec2Bert not provided")

        self.w2v_T = seq_len

        # Step 1: Warmup -- compile all ops and populate distance index caches
        warmup_data = torch.randn(1, 1, seq_len, 160).to(torch.bfloat16)
        warmup_tt = ttnn.from_torch(
            warmup_data,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
        )
        warmup_out = self._w2v_forward_ttnn(warmup_tt)
        _ = ttnn.to_torch(warmup_out)  # sync to ensure compilation is done

        # Step 2: Allocate fixed input tensor on device
        input_spec = ttnn.TensorSpec(
            (1, 1, seq_len, 160),
            ttnn.DataType.BFLOAT16,
            ttnn.TILE_LAYOUT,
        )
        self.w2v_trace_input = ttnn.allocate_tensor_on_device(input_spec, self.device)

        host_input = ttnn.from_torch(
            warmup_data,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
        ttnn.copy_host_to_device_tensor(host_input, self.w2v_trace_input)

        # Step 3: Capture trace
        self.w2v_trace_id = ttnn.begin_trace_capture(self.device, cq_id=0)
        self.w2v_trace_output = self._w2v_forward_ttnn(self.w2v_trace_input)
        ttnn.end_trace_capture(self.device, self.w2v_trace_id, cq_id=0)

    def run_w2v_traced(self, input_features):
        """Run Wav2Vec2-BERT using traced execution.

        Args:
            input_features: [B, T, 160] torch tensor (mel filterbank features)
                            T must match the traced sequence length
        Returns:
            [B, T, 1024] torch tensor (hidden states)
        """
        if self.w2v_trace_id is None:
            raise RuntimeError("Call setup_w2v_trace() first")

        # Copy input to trace input: [B, T, 160] -> [1, B, T, 160]
        host_tensor = ttnn.from_torch(
            input_features.to(torch.bfloat16).unsqueeze(0),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
        ttnn.copy_host_to_device_tensor(host_tensor, self.w2v_trace_input)

        # Execute trace (fast replay)
        ttnn.execute_trace(self.device, self.w2v_trace_id, cq_id=0, blocking=True)

        # Read output: [1, 1, T, 1024] -> [B, T, 1024]
        out = ttnn.to_torch(self.w2v_trace_output).float()
        return out.squeeze(0)  # [B, T, 1024]

    def setup_acoustic_trace(self, n_samples):
        """Capture metal trace for AcousticEncoder.

        The initial Conv1d(1, 48, k=7) runs on host (torch interface).
        Everything after that (5 encoder blocks + final) is traced.

        Args:
            n_samples: Fixed waveform length (number of audio samples)
        """
        enc = self.acoustic_encoder

        # Run initial conv on host to determine T_initial
        warmup_waveform = torch.randn(1, 1, n_samples)
        x_initial = enc.initial_conv1d(warmup_waveform)  # [1, 48, T_initial]
        T_initial = x_initial.shape[2]
        self.acoustic_initial_T = T_initial

        # Convert to NHWC for device
        x_nhwc = x_initial.permute(0, 2, 1).unsqueeze(0).to(torch.bfloat16)  # [1, 1, T, 48]

        # Step 1: Warmup -- compile all ops
        x_tt = ttnn.from_torch(x_nhwc, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device)
        warmup_out = self._acoustic_forward_ttnn(x_tt, T_initial)
        _ = ttnn.to_torch(warmup_out)

        # Step 2: Allocate fixed input tensor
        input_spec = ttnn.TensorSpec(
            (1, 1, T_initial, 48),
            ttnn.DataType.BFLOAT16,
            ttnn.ROW_MAJOR_LAYOUT,
        )
        self.acoustic_trace_input = ttnn.allocate_tensor_on_device(input_spec, self.device)

        # Copy warmup data
        host_input = ttnn.from_torch(x_nhwc, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        ttnn.copy_host_to_device_tensor(host_input, self.acoustic_trace_input)

        # Step 3: Capture trace
        self.acoustic_trace_id = ttnn.begin_trace_capture(self.device, cq_id=0)
        self.acoustic_trace_output = self._acoustic_forward_ttnn(self.acoustic_trace_input, T_initial)
        ttnn.end_trace_capture(self.device, self.acoustic_trace_id, cq_id=0)

        self.acoustic_n_samples = n_samples

    def run_acoustic_traced(self, waveform):
        """Run acoustic encoder with traced execution.

        Args:
            waveform: [B, 1, n_samples] torch tensor (must match setup n_samples)
        Returns:
            [B, 1024, T] torch tensor
        """
        if self.acoustic_trace_id is None:
            raise RuntimeError("Call setup_acoustic_trace() first")

        # Initial conv on host
        x_initial = self.acoustic_encoder.initial_conv1d(waveform)  # [B, 48, T]

        # Copy to trace input
        x_nhwc = x_initial.permute(0, 2, 1).unsqueeze(0).to(torch.bfloat16)
        host_tensor = ttnn.from_torch(x_nhwc, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        ttnn.copy_host_to_device_tensor(host_tensor, self.acoustic_trace_input)

        # Execute trace
        ttnn.execute_trace(self.device, self.acoustic_trace_id, cq_id=0, blocking=True)

        # Read output: [1, 1, T_final, 1024] -> [B, 1024, T_final]
        out = ttnn.to_torch(self.acoustic_trace_output).float()
        return out.squeeze(0).permute(0, 2, 1)

    def setup_semantic_trace(self, T):
        """Capture metal trace for SemanticEncoder.

        Args:
            T: Fixed time dimension for semantic features
        """
        if self.semantic_encoder is None:
            raise RuntimeError("SemanticEncoder not provided")

        self.semantic_T = T

        # Step 1: Warmup
        warmup_data = torch.randn(1, 1, T, 1024).to(torch.bfloat16)
        x_tt = ttnn.from_torch(warmup_data, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device)
        warmup_out = self._semantic_forward_ttnn(x_tt, T)
        _ = ttnn.to_torch(warmup_out)

        # Step 2: Allocate fixed input
        input_spec = ttnn.TensorSpec(
            (1, 1, T, 1024),
            ttnn.DataType.BFLOAT16,
            ttnn.ROW_MAJOR_LAYOUT,
        )
        self.semantic_trace_input = ttnn.allocate_tensor_on_device(input_spec, self.device)

        host_input = ttnn.from_torch(warmup_data, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        ttnn.copy_host_to_device_tensor(host_input, self.semantic_trace_input)

        # Step 3: Capture trace
        self.semantic_trace_id = ttnn.begin_trace_capture(self.device, cq_id=0)
        self.semantic_trace_output = self._semantic_forward_ttnn(self.semantic_trace_input, T)
        ttnn.end_trace_capture(self.device, self.semantic_trace_id, cq_id=0)

    def run_semantic_traced(self, semantic_features):
        """Run semantic encoder with traced execution.

        Args:
            semantic_features: [B, 1024, T] torch tensor (must match setup T)
        Returns:
            [B, 1024, T] torch tensor
        """
        if self.semantic_trace_id is None:
            raise RuntimeError("Call setup_semantic_trace() first")

        # Convert to NHWC: [B, 1024, T] -> [1, 1, T, 1024]
        x_nhwc = semantic_features.permute(0, 2, 1).unsqueeze(0).to(torch.bfloat16)
        host_tensor = ttnn.from_torch(x_nhwc, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        ttnn.copy_host_to_device_tensor(host_tensor, self.semantic_trace_input)

        # Execute trace
        ttnn.execute_trace(self.device, self.semantic_trace_id, cq_id=0, blocking=True)

        # Read output: [1, 1, T, 1024] -> [B, 1024, T]
        out = ttnn.to_torch(self.semantic_trace_output).float()
        return out.squeeze(0).permute(0, 2, 1)

    def release_traces(self):
        """Release all captured traces."""
        if self.w2v_trace_id is not None:
            ttnn.release_trace(self.device, self.w2v_trace_id)
            self.w2v_trace_id = None
            self.w2v_trace_input = None
            self.w2v_trace_output = None
        if self.acoustic_trace_id is not None:
            ttnn.release_trace(self.device, self.acoustic_trace_id)
            self.acoustic_trace_id = None
            self.acoustic_trace_input = None
            self.acoustic_trace_output = None
        if self.semantic_trace_id is not None:
            ttnn.release_trace(self.device, self.semantic_trace_id)
            self.semantic_trace_id = None
            self.semantic_trace_input = None
            self.semantic_trace_output = None
