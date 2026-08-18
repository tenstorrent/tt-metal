# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
Speaker Encoder (ECAPA-TDNN) implementation for Qwen3-TTS.

The Speaker Encoder extracts speaker embeddings from mel spectrograms.
These embeddings are used for voice cloning in the ICL input.

Architecture:
    - TimeDelayNetBlock (128 -> 512)
    - 3x SERes2NetBlocks (512 -> 512)
    - Multi-scale Feature Aggregation (MFA)
    - Attentive Statistics Pooling
    - Final FC (3072 -> 2048)

Note: Reflect ``conv1d`` runs on host PyTorch inside ``_conv1d_same_padding`` (no TTNN
reflect pad). Activations use ``ttnn.relu`` / ``ttnn.tanh`` where possible; conv/FC
weights are cached on device. Mel filterbank + Hann window are cached on host.
"""

import os
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

import ttnn
from models.common.lightweightmodule import LightweightModule


class SpeakerEncoderConfig:
    """Configuration for Speaker Encoder."""

    def __init__(
        self,
        sample_rate: int = 24000,
        n_mels: int = 128,
        channels: int = 512,
        output_dim: int = 2048,
        num_blocks: int = 4,  # 1 initial + 3 SE-Res2Net
    ):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.channels = channels
        self.output_dim = output_dim
        self.num_blocks = num_blocks


class SpeakerEncoder(LightweightModule):
    """
    Speaker Encoder (ECAPA-TDNN) for extracting speaker embeddings.

    Takes mel spectrograms as input and outputs speaker embedding vectors
    that capture voice characteristics for voice cloning.

    Args:
        device: TTNN device
        state_dict: Model weights
        config: Speaker encoder configuration
        weight_cache_path: Optional path for weight caching
    """

    def __init__(
        self,
        device,
        state_dict: dict,
        config: Optional[SpeakerEncoderConfig] = None,
        weight_cache_path=None,
    ):
        super().__init__()
        self.device = device
        self.config = config or SpeakerEncoderConfig()

        # Extract speaker encoder weights from state dict
        self.weights = {}
        prefix = "speaker_encoder."
        for k, v in state_dict.items():
            if k.startswith(prefix):
                self.weights[k[len(prefix) :]] = v

        # Store weights as PyTorch tensors for CPU computation
        # Can be converted to TTNN when native ops are available
        self.pytorch_weights = {k: v.float() for k, v in self.weights.items()}
        self._conv1d_param_tt_cache = {}
        self._conv1d_prepared_cache = {}  # keyed by (cache_key, input_length, ...)
        self._se_current_cache_id = None  # set by _se_res2net_block per block
        self._mel_stft_key = None
        self._fc_linear_weight_tt = None
        self._fc_bias_tt = None
        self._compute_kernel_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
        )
        self._conv1d_config = ttnn.Conv1dConfig(
            weights_dtype=ttnn.bfloat16,
            shard_layout=None,
            deallocate_activation=False,
        )
        self._ttnn_conv1d = ttnn.conv1d

    def compute_mel_spectrogram(
        self,
        audio: torch.Tensor,
        n_fft: int = 1024,
        num_mels: int = 128,
        sampling_rate: int = 24000,
        hop_size: int = 256,
        win_size: int = 1024,
        fmin: int = 0,
        fmax: int = 12000,
    ) -> torch.Tensor:
        """
        Compute mel spectrogram from audio waveform.

        Args:
            audio: Audio waveform [num_samples] or [batch, num_samples]

        Returns:
            Mel spectrogram [batch, num_mels, time]
        """
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        device = audio.device
        mel_basis, hann_window = self._mel_stft_constants(n_fft, num_mels, sampling_rate, win_size, fmin, fmax)
        mel_basis = mel_basis.to(device)
        hann_window = hann_window.to(device)

        # Padding
        padding = (n_fft - hop_size) // 2
        y = F.pad(audio.unsqueeze(1), (padding, padding), mode="reflect").squeeze(1)

        # STFT
        spec = torch.stft(
            y,
            n_fft,
            hop_length=hop_size,
            win_length=win_size,
            window=hann_window,
            center=False,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        spec = torch.sqrt(torch.view_as_real(spec).pow(2).sum(-1) + 1e-9)

        # Apply mel filterbank
        mel_spec = torch.matmul(mel_basis, spec)

        # Log compression
        mel_spec = torch.log(torch.clamp(mel_spec, min=1e-5))

        return mel_spec

    def _mel_stft_constants(
        self,
        n_fft: int,
        num_mels: int,
        sampling_rate: int,
        win_size: int,
        fmin: int,
        fmax: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        key = (n_fft, num_mels, sampling_rate, win_size, fmin, fmax)
        if self._mel_stft_key != key:
            from librosa.filters import mel as librosa_mel_fn

            mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
            self._mel_basis_cpu = torch.from_numpy(mel).float()
            self._hann_window_cpu = torch.hann_window(win_size).float()
            self._mel_stft_key = key
        return self._mel_basis_cpu, self._hann_window_cpu

    def _conv1d_params_to_ttnn(self, weight: torch.Tensor, bias: torch.Tensor) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        key = (weight.data_ptr(), bias.data_ptr())
        hit = self._conv1d_param_tt_cache.get(key)
        if hit is None:
            w_tt = ttnn.from_torch(
                weight,
                device=self.device,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            b_tt = ttnn.from_torch(
                bias,
                device=self.device,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            hit = (w_tt, b_tt)
            self._conv1d_param_tt_cache[key] = hit
        return hit

    def _torch_ncl_to_ttnn_nlc(self, x_ncl: torch.Tensor) -> ttnn.Tensor:
        return ttnn.from_torch(
            x_ncl.permute(0, 2, 1).contiguous(),
            device=self.device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _ttnn_nlc_to_ttnn_ncl(self, x_nlc: ttnn.Tensor) -> ttnn.Tensor:
        """NLC ``[B, L, C]`` (or ``[B, 1, L, C]``) → NCL ``[B, C, L]`` on device; no host round-trip."""
        m = ttnn.DRAM_MEMORY_CONFIG
        sh = tuple(x_nlc.shape)
        if len(sh) == 4 and sh[1] == 1:
            x_nlc = ttnn.reshape(x_nlc, (sh[0], sh[2], sh[3]), memory_config=m)
        return ttnn.permute(x_nlc, (0, 2, 1), memory_config=m)

    def _run_ttnn_conv1d(
        self,
        x_nlc: ttnn.Tensor,
        weight: ttnn.Tensor,
        bias: ttnn.Tensor,
        input_length: int,
        *,
        padding: int = 0,
        dilation: int = 1,
        cache_key: Optional[Tuple] = None,  # accepted for backward compat; unused
    ) -> Tuple[ttnn.Tensor, int, int]:
        """Single TTNN conv1d entry point used by speaker encoder.

        This path is only valid before any other trace has been *executed* on
        the device. Calling it post-trace-exec will hang or produce inf
        outputs (the conv2d execution path is unstable in that state).
        Production calls go through the per-block traces captured in
        ``capture_se_block_traces`` instead.
        """
        mc = ttnn.DRAM_MEMORY_CONFIG
        out_channels = int(weight.shape[0])
        kernel_size = int(weight.shape[-1] if len(tuple(weight.shape)) == 3 else weight.shape[-2])
        bias_tt = bias
        if len(tuple(bias_tt.shape)) == 1:
            bias_tt = ttnn.reshape(bias_tt, (1, 1, 1, out_channels), memory_config=mc)
        y, y_len = self._ttnn_conv1d(
            input_tensor=x_nlc,
            weight_tensor=weight,
            device=self.device,
            in_channels=int(weight.shape[1]),
            out_channels=out_channels,
            batch_size=int(x_nlc.shape[0]),
            input_length=input_length,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation,
            bias_tensor=bias_tt,
            conv_config=self._conv1d_config,
            compute_config=self._compute_kernel_config,
            dtype=ttnn.bfloat16,
            memory_config=mc,
            return_output_dim=True,
        )
        return y, int(y_len), out_channels

    def _conv1d_same_padding(self, x: ttnn.Tensor, weight, bias, dilation: int = 1) -> ttnn.Tensor:
        """Same reflect pad + conv1d on host; I/O NLC ``[batch, seq, channels]``.

        ``weight``/``bias`` may be either torch.Tensor (preferred — avoids a
        device round-trip) or ttnn.Tensor on device (legacy callers). The
        ``ttnn.to_torch(weight)`` round-trip on device-backed weights returns
        garbage post-trace-exec, so when we have torch weights in hand we
        skip the round-trip entirely.
        """
        x_t = ttnn.to_torch(x, dtype=torch.float32)
        if x_t.dim() == 4:
            x_t = x_t.squeeze(1)
        x_t = x_t.permute(0, 2, 1).contiguous()
        if isinstance(weight, torch.Tensor):
            w_t = weight.float()
        else:
            w_t = ttnn.to_torch(weight, dtype=torch.float32)
        if w_t.dim() == 4:
            w_t = w_t.squeeze(-1)
        if isinstance(bias, torch.Tensor):
            b_t = bias.float().reshape(-1)
        else:
            b_t = ttnn.to_torch(bias, dtype=torch.float32).reshape(-1)

        kernel_size = w_t.shape[-1]
        effective_kernel = dilation * (kernel_size - 1) + 1
        pad_total = effective_kernel - 1
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        x_padded = F.pad(x_t, (pad_left, pad_right), mode="reflect")
        out_t = F.conv1d(x_padded, w_t, b_t, dilation=dilation)
        out_nlc = out_t.permute(0, 2, 1).contiguous()
        return ttnn.from_torch(
            out_nlc,
            device=self.device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _time_delay_net_block(
        self, x: ttnn.Tensor, conv_weight: torch.Tensor, conv_bias: torch.Tensor, dilation: int = 1
    ) -> ttnn.Tensor:
        """Time Delay Network Block in TTNN (NLC in/out)."""
        # Pass torch weights directly into _conv1d_same_padding (avoids a
        # device round-trip whose output is corrupted post-trace-exec).
        y_tt = self._conv1d_same_padding(x, conv_weight, conv_bias, dilation)
        return ttnn.relu(y_tt)

    def _res2net_block(self, x: ttnn.Tensor, prefix: str, scale: int = 8, dilation: int = 1) -> ttnn.Tensor:
        """Res2Net block with multi-scale feature extraction in TTNN (NLC)."""
        mc = ttnn.L1_MEMORY_CONFIG
        x_nlc = (
            x if len(tuple(x.shape)) == 3 else ttnn.reshape(x, (x.shape[0], x.shape[2], x.shape[3]), memory_config=mc)
        )
        batch, seq_len, channels = int(x_nlc.shape[0]), int(x_nlc.shape[1]), int(x_nlc.shape[2])
        assert channels % scale == 0, f"channels {channels} must be divisible by scale {scale}"
        part_channels = channels // scale

        parts = []
        for i in range(scale):
            c0, c1 = i * part_channels, (i + 1) * part_channels
            parts.append(ttnn.slice(x_nlc, [0, 0, c0], [batch, seq_len, c1], memory_config=mc))
        outputs = []

        output_part = None
        for i, hidden_part in enumerate(parts):
            if i == 0:
                output_part = hidden_part
            elif i == 1:
                conv_weight = self.pytorch_weights.get(f"{prefix}blocks.{i-1}.conv.weight")
                conv_bias = self.pytorch_weights.get(f"{prefix}blocks.{i-1}.conv.bias")
                if conv_weight is not None:
                    output_part = self._time_delay_net_block(hidden_part, conv_weight, conv_bias, dilation=dilation)
                else:
                    output_part = hidden_part
            else:
                conv_weight = self.pytorch_weights.get(f"{prefix}blocks.{i-1}.conv.weight")
                conv_bias = self.pytorch_weights.get(f"{prefix}blocks.{i-1}.conv.bias")
                if conv_weight is not None:
                    output_part = self._time_delay_net_block(
                        ttnn.add(hidden_part, output_part, memory_config=mc),
                        conv_weight,
                        conv_bias,
                        dilation=dilation,
                    )
                else:
                    output_part = ttnn.add(hidden_part, output_part, memory_config=mc)
            outputs.append(output_part)

        return ttnn.concat(outputs, dim=2, memory_config=mc)

    def _squeeze_excitation_block(
        self,
        x: ttnn.Tensor,
        conv1_weight: ttnn.Tensor,
        conv1_bias: ttnn.Tensor,
        conv2_weight: ttnn.Tensor,
        conv2_bias: ttnn.Tensor,
    ) -> ttnn.Tensor:
        """Squeeze-and-Excitation block in TTNN (NLC in/out)."""
        mc = ttnn.L1_MEMORY_CONFIG
        x_nlc = (
            x if len(tuple(x.shape)) == 3 else ttnn.reshape(x, (x.shape[0], x.shape[2], x.shape[3]), memory_config=mc)
        )
        batch = int(x_nlc.shape[0])
        seq_len = int(x_nlc.shape[1])

        # NOTE: ALL SE-block intermediates use DRAM (not L1). When SpeakerEncoder
        # is invoked from a server context that holds persistent buffers in L1
        # (KV caches, trace mask tensors, prepared conv weights), the L1
        # allocator can't satisfy fresh allocations from these ops and the
        # call hangs silently. DRAM is plentiful and removes the contention.
        dram = ttnn.DRAM_MEMORY_CONFIG

        # Squeeze over sequence length: [B, L, C] -> [B, 1, C]
        y = ttnn.mean(x_nlc, dim=1, keepdim=True)
        y = ttnn.to_layout(y, ttnn.TILE_LAYOUT, memory_config=dram)

        se_cache_id = getattr(self, "_se_current_cache_id", None)
        y, y_len, out1_ch = self._run_ttnn_conv1d(
            y,
            conv1_weight,
            conv1_bias,
            input_length=1,
            cache_key=("se", se_cache_id, "conv1") if se_cache_id is not None else None,
        )
        y = ttnn.reshape(y, (batch, y_len, out1_ch), memory_config=dram)
        y = ttnn.relu(y, memory_config=dram)

        y, y_len, out2_ch = self._run_ttnn_conv1d(
            y,
            conv2_weight,
            conv2_bias,
            input_length=y_len,
            cache_key=("se", se_cache_id, "conv2") if se_cache_id is not None else None,
        )
        y = ttnn.reshape(y, (batch, y_len, out2_ch), memory_config=dram)
        y = ttnn.sigmoid(y, memory_config=dram)
        if bool(int(os.environ.get("SE_VDBG", "0"))):
            sv = ttnn.to_torch(y, dtype=torch.float32).flatten()
            iv = ttnn.to_torch(x_nlc, dtype=torch.float32).flatten()
            print(
                f"[SE_VDBG]   UNtraced scale: min={sv.min().item():.4f} max={sv.max().item():.4f} mean={sv.mean().item():.4f}",
                flush=True,
            )
            print(
                f"[SE_VDBG]   UNtraced x_nlc: min={iv.min().item():.4f} max={iv.max().item():.4f} norm={iv.norm().item():.4f}",
                flush=True,
            )

        # Apply channel-wise scale via broadcasting: [B,L,C] * [B,1,C] -> [B,L,C].
        # Skips materializing y to seq-length (saves Untilize+Repeat+Tilize chain).
        return ttnn.multiply(x_nlc, y, memory_config=dram)

    def _squeeze_excitation_block_traced(self, x: ttnn.Tensor, block_idx: int) -> ttnn.Tensor:
        """SE block via pre-captured trace. ``x`` is [B, L, C]; output is [B, L, C]."""
        _dbg = bool(int(os.environ.get("SE_DBG", "0")))

        def _mk(label):
            if _dbg:
                ttnn.synchronize_device(self.device)
                print(f"[SE_DBG]     SE_TR{block_idx}: {label}", flush=True)

        info = self._se_traces[block_idx]
        mc = ttnn.DRAM_MEMORY_CONFIG
        x_nlc = (
            x if len(tuple(x.shape)) == 3 else ttnn.reshape(x, (x.shape[0], x.shape[2], x.shape[3]), memory_config=mc)
        )
        _mk("entry")
        y = ttnn.mean(x_nlc, dim=1, keepdim=True)  # [B, 1, C] TILE
        _mk("mean done")
        # input_tt is ROW_MAJOR_LAYOUT [1, 1, in_c]. Convert to match.
        y_rm = ttnn.to_layout(y, ttnn.ROW_MAJOR_LAYOUT, memory_config=mc)
        _mk("to_row_major done")
        ttnn.copy(y_rm, info["input_tt"])
        ttnn.synchronize_device(self.device)
        _mk("copy_d2d synced")
        ttnn.execute_trace(self.device, info["trace_id"], cq_id=0, blocking=True)
        _mk("execute_trace done")
        if bool(int(os.environ.get("SE_VDBG", "0"))):
            scale_v = ttnn.to_torch(info["output_tt"], dtype=torch.float32).flatten()
            inp_v = ttnn.to_torch(info["input_tt"], dtype=torch.float32).flatten()
            print(
                f"[SE_VDBG]   block{block_idx} scale: min={scale_v.min().item():.4f} max={scale_v.max().item():.4f} mean={scale_v.mean().item():.4f}",
                flush=True,
            )
            print(
                f"[SE_VDBG]   block{block_idx} traced_in: min={inp_v.min().item():.4f} max={inp_v.max().item():.4f} norm={inp_v.norm().item():.4f}",
                flush=True,
            )
        out = ttnn.multiply(x_nlc, info["output_tt"], memory_config=mc)
        _mk("multiply done")
        return out

    def _se_res2net_block(self, x: ttnn.Tensor, block_idx: int, scale: int = 8) -> ttnn.Tensor:
        """SERes2NetBlock in TTNN (NLC): TDNN1 -> Res2Net -> TDNN2 -> SE -> residual add."""
        prefix = f"blocks.{block_idx}."
        residual = x

        _dbg = bool(int(os.environ.get("SE_DBG", "0")))

        def _mk(label):
            if _dbg:
                ttnn.synchronize_device(self.device)
                print(f"[SE_DBG]   block{block_idx}: {label}", flush=True)

        # TDNN1
        tdnn1_weight = self.pytorch_weights.get(f"{prefix}tdnn1.conv.weight")
        tdnn1_bias = self.pytorch_weights.get(f"{prefix}tdnn1.conv.bias")
        if tdnn1_weight is not None:
            x = self._time_delay_net_block(x, tdnn1_weight, tdnn1_bias, dilation=1)
        _mk("after TDNN1")

        # Res2Net
        # Pass dilation per HF: blocks[1]=2, blocks[2]=3, blocks[3]=4 (matches
        # config.enc_dilations[i] in HF Qwen3TTSSpeakerEncoder). Was always 1.
        x = self._res2net_block(x, f"{prefix}res2net_block.", scale, dilation=block_idx + 1)
        _mk("after Res2Net")

        # TDNN2
        tdnn2_weight = self.pytorch_weights.get(f"{prefix}tdnn2.conv.weight")
        tdnn2_bias = self.pytorch_weights.get(f"{prefix}tdnn2.conv.bias")
        if tdnn2_weight is not None:
            x = self._time_delay_net_block(x, tdnn2_weight, tdnn2_bias, dilation=1)
        _mk("after TDNN2")

        # SE block — prepared conv weights are cached inside _run_ttnn_conv1d
        # under cache_key=("se", block_idx, "convN"). First call preps via
        # ttnn.prepare_conv_weights/bias; subsequent calls reuse and skip prep.
        se_conv1_weight = self.pytorch_weights.get(f"{prefix}se_block.conv1.weight")
        se_conv1_bias = self.pytorch_weights.get(f"{prefix}se_block.conv1.bias")
        se_conv2_weight = self.pytorch_weights.get(f"{prefix}se_block.conv2.weight")
        se_conv2_bias = self.pytorch_weights.get(f"{prefix}se_block.conv2.bias")
        if se_conv1_weight is not None:
            se_traces = getattr(self, "_se_traces", None)
            use_traces = se_traces and block_idx in se_traces and getattr(self, "_se_traces_active", False)
            if use_traces:
                # Traced path — replays pre-captured ops; safe post-trace-exec.
                x = self._squeeze_excitation_block_traced(x, block_idx)
            else:
                se_w1_tt, se_b1_tt = self._conv1d_params_to_ttnn(se_conv1_weight, se_conv1_bias)
                se_w2_tt, se_b2_tt = self._conv1d_params_to_ttnn(se_conv2_weight, se_conv2_bias)
                _mk("after SE conv weight upload")
                self._se_current_cache_id = block_idx
                try:
                    x = self._squeeze_excitation_block(x, se_w1_tt, se_b1_tt, se_w2_tt, se_b2_tt)
                finally:
                    self._se_current_cache_id = None
            _mk("after SE block")

        result = ttnn.add(x, residual, memory_config=ttnn.L1_MEMORY_CONFIG)
        _mk("after residual add")
        return result

    def _attentive_statistics_pooling(
        self,
        x: ttnn.Tensor,
        tdnn_weight: torch.Tensor,
        tdnn_bias: torch.Tensor,
        conv_weight: torch.Tensor,
        conv_bias: torch.Tensor,
        eps: float = 1e-12,
    ) -> ttnn.Tensor:
        """Attentive Statistics Pooling in TTNN, NLC input/output."""
        mc = ttnn.L1_MEMORY_CONFIG
        x_nlc = (
            x if len(tuple(x.shape)) == 3 else ttnn.reshape(x, (x.shape[0], x.shape[2], x.shape[3]), memory_config=mc)
        )
        seq_len = int(x_nlc.shape[1])

        mean = ttnn.mean(x_nlc, dim=1, keepdim=True)
        centered = ttnn.subtract(x_nlc, mean, memory_config=mc)
        std = ttnn.sqrt(ttnn.clamp(ttnn.mean(ttnn.multiply(centered, centered), dim=1, keepdim=True), min=eps))

        mean_expanded = ttnn.repeat(mean, (1, seq_len, 1))
        std_expanded = ttnn.repeat(std, (1, seq_len, 1))
        attention_input = ttnn.concat([x_nlc, mean_expanded, std_expanded], dim=2, memory_config=mc)

        # Pass torch weights directly (avoids device round-trip).
        # HF AttentiveStatisticsPooling does conv → ReLU → tanh → conv (the
        # self.tdnn is TimeDelayNetBlock = conv + ReLU). We had been doing
        # conv → tanh → conv which drifted the speaker embedding. Verified
        # against QwenLM/Qwen3-TTS reference: PCC 0.96 → 0.9999 after fix.
        a_tt = self._conv1d_same_padding(attention_input, tdnn_weight, tdnn_bias)
        a_tt = ttnn.relu(a_tt)
        a_tt = ttnn.tanh(a_tt)
        a_tt = self._conv1d_same_padding(a_tt, conv_weight, conv_bias)
        attention = ttnn.softmax(a_tt, dim=1, memory_config=mc)

        weighted_mean = ttnn.sum(ttnn.multiply(attention, x_nlc), dim=1, keepdim=True)
        centered_w = ttnn.subtract(x_nlc, weighted_mean, memory_config=mc)
        weighted_std = ttnn.sqrt(
            ttnn.clamp(
                ttnn.sum(ttnn.multiply(attention, ttnn.multiply(centered_w, centered_w)), dim=1, keepdim=True), min=eps
            )
        )
        return ttnn.concat([weighted_mean, weighted_std], dim=2, memory_config=mc)

    def _ensure_fc_linear_params(self, fc_weight: torch.Tensor, fc_bias: torch.Tensor) -> None:
        """Lazy-build ``[1, 1, in, out]`` weight + bias for ``ttnn.linear`` (matches Talker layout)."""
        if self._fc_linear_weight_tt is not None:
            return
        w = fc_weight.squeeze(-1).float()
        in_f, out_f = w.shape[1], w.shape[0]
        w_host = w.T.contiguous().reshape(1, 1, in_f, out_f)
        self._fc_linear_weight_tt = ttnn.from_torch(
            w_host,
            device=self.device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self._fc_bias_tt = ttnn.from_torch(
            fc_bias.float().reshape(1, 1, 1, -1),
            device=self.device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def forward(self, mel_spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Extract speaker embedding from mel spectrogram.

        Args:
            mel_spectrogram: Mel spectrogram [batch, n_mels, time] or [batch, time, n_mels]

        Returns:
            Speaker embedding [batch, 2048]
        """
        hidden = mel_spectrogram.float()

        # Transpose if needed: convolutions expect [batch, n_mels, time]
        # n_mels is fixed at 128, so check which dimension is 128
        if hidden.shape[2] == self.config.n_mels:
            # Input is [batch, time, n_mels], transpose to [batch, n_mels, time]
            hidden = hidden.transpose(1, 2)

        hidden_tt = self._torch_ncl_to_ttnn_nlc(hidden)
        hidden_states_list_tt = []

        _dbg = bool(int(os.environ.get("SE_DBG", "0")))
        _vdbg = bool(int(os.environ.get("SE_VDBG", "0")))

        def _mark(label):
            if _dbg:
                ttnn.synchronize_device(self.device)
                print(f"[SE_DBG] {label}", flush=True)

        def _dump(label, t):
            if _vdbg:
                ttnn.synchronize_device(self.device)
                try:
                    arr = ttnn.to_torch(t, dtype=torch.float32)
                    print(
                        f"[SE_VDBG] {label}: shape={tuple(arr.shape)} "
                        f"min={arr.min().item():.4g} max={arr.max().item():.4g} "
                        f"|x|max={arr.abs().max().item():.4g} "
                        f"nan={arr.isnan().sum().item()} inf={arr.isinf().sum().item()}",
                        flush=True,
                    )
                except Exception as e:
                    print(f"[SE_VDBG] {label}: dump failed: {e}", flush=True)

        _mark("forward start")
        _dump("input mel→ttnn", hidden_tt)

        # blocks[0]: Initial TDNN
        conv_weight = self.pytorch_weights.get("blocks.0.conv.weight")
        conv_bias = self.pytorch_weights.get("blocks.0.conv.bias")
        if conv_weight is not None:
            hidden_tt = self._time_delay_net_block(hidden_tt, conv_weight, conv_bias, dilation=1)
        hidden_states_list_tt.append(hidden_tt)
        _mark("after blocks[0] TDNN")
        _dump("after TDNN0", hidden_tt)

        # blocks[1-3]: SERes2NetBlocks
        for block_idx in range(1, 4):
            if f"blocks.{block_idx}.tdnn1.conv.weight" in self.pytorch_weights:
                hidden_tt = self._se_res2net_block(hidden_tt, block_idx, scale=8)
            hidden_states_list_tt.append(hidden_tt)
            _mark(f"after SERes2Net block {block_idx}")
            _dump(f"after SERes2Net block {block_idx}", hidden_tt)

        # MFA: concatenate blocks 1-3
        hidden_tt = ttnn.concat(hidden_states_list_tt[1:], dim=2, memory_config=ttnn.L1_MEMORY_CONFIG)
        _mark("after MFA concat")

        # MFA TDNN
        mfa_weight = self.pytorch_weights.get("mfa.conv.weight")
        mfa_bias = self.pytorch_weights.get("mfa.conv.bias")
        if mfa_weight is not None:
            hidden_tt = self._time_delay_net_block(hidden_tt, mfa_weight, mfa_bias, dilation=1)
        _mark("after MFA TDNN")

        # Attentive Statistics Pooling
        asp_tdnn_weight = self.pytorch_weights.get("asp.tdnn.conv.weight")
        asp_tdnn_bias = self.pytorch_weights.get("asp.tdnn.conv.bias")
        asp_conv_weight = self.pytorch_weights.get("asp.conv.weight")
        asp_conv_bias = self.pytorch_weights.get("asp.conv.bias")

        if asp_tdnn_weight is not None:
            hidden_tt = self._attentive_statistics_pooling(
                hidden_tt, asp_tdnn_weight, asp_tdnn_bias, asp_conv_weight, asp_conv_bias
            )
            _mark("after ASP")
            _dump("after ASP", hidden_tt)
        else:
            mean = ttnn.mean(hidden_tt, dim=1, keepdim=True)
            centered = ttnn.subtract(hidden_tt, mean, memory_config=ttnn.L1_MEMORY_CONFIG)
            std = ttnn.sqrt(
                ttnn.clamp(
                    ttnn.mean(ttnn.multiply(centered, centered), dim=1, keepdim=True),
                    min=1e-12,
                )
            )
            hidden_tt = ttnn.concat([mean, std], dim=2, memory_config=ttnn.L1_MEMORY_CONFIG)

        # Final FC: ``ttnn.linear`` (kernel 1 conv); weights prepared once
        fc_weight = self.pytorch_weights.get("fc.weight")
        fc_bias = self.pytorch_weights.get("fc.bias")
        if fc_weight is not None:
            self._ensure_fc_linear_params(fc_weight, fc_bias)
            _mark("FC: ensure_params done")
            b, tlen, ch = int(hidden_tt.shape[0]), int(hidden_tt.shape[1]), int(hidden_tt.shape[2])
            # NOTE: ttnn.reshape with explicit memory_config hangs
            # post-trace-exec on this device. Use the bare reshape (it is a
            # metadata-only op and re-uses the input's storage).
            x_tt = ttnn.reshape(hidden_tt, (b, 1, tlen, ch))
            _mark("FC: reshape done")
            x_tt = ttnn.to_layout(x_tt, ttnn.TILE_LAYOUT)
            _mark("FC: to_layout done")
            fc_trace = getattr(self, "_fc_trace", None)
            use_fc_trace = fc_trace is not None and getattr(self, "_se_traces_active", False)
            if use_fc_trace:
                # Traced path — copy x into the persistent input buffer and
                # execute the captured ttnn.linear trace.
                ttnn.copy(x_tt, fc_trace["input_tt"])
                _mark("FC: copy_d2d done")
                ttnn.execute_trace(self.device, fc_trace["trace_id"], cq_id=0, blocking=False)
                _mark("FC: execute_trace launched")
                ttnn.synchronize_device(self.device)
                _mark("FC: execute_trace synced")
                out_tt = fc_trace["output_tt"]
            else:
                out_tt = ttnn.linear(
                    x_tt,
                    self._fc_linear_weight_tt,
                    bias=self._fc_bias_tt,
                    compute_kernel_config=self._compute_kernel_config,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
            _mark("after FC linear")
            hidden = ttnn.to_torch(out_tt, dtype=torch.float32).reshape(b, -1)
            _mark("after to_torch")
        else:
            hidden = ttnn.to_torch(hidden_tt, dtype=torch.float32).reshape(int(hidden_tt.shape[0]), -1)

        return hidden

    def activate_traced_extract(self) -> None:
        """Switch ``extract_speaker_embedding`` to use the captured SE/FC
        traces. Call this AFTER any registered-voice precomputes (which
        should use the untraced path so their cached embeddings match the
        bit-exact reference values), and BEFORE the first request-time
        ECAPA call (which must use the traced path to avoid hangs/inf).
        """
        self._se_traces_active = True

    def capture_fc_trace(self) -> None:
        """Pre-capture the final FC linear into a trace.

        Same rationale as ``capture_se_block_traces``: post-trace-exec the
        non-traced ``ttnn.linear`` execution hangs on this device. Replaying a
        pre-captured trace bypasses the unstable dispatch path. FC input is a
        fixed ``[1, 1, 3072]`` tensor; output is ``[1, 1, 2048]``.
        """
        if getattr(self, "_fc_trace", None):
            return
        fc_weight = self.pytorch_weights.get("fc.weight")
        fc_bias = self.pytorch_weights.get("fc.bias")
        if fc_weight is None:
            return
        self._ensure_fc_linear_params(fc_weight, fc_bias)
        in_f = int(self._fc_linear_weight_tt.shape[2])
        out_f = int(self._fc_linear_weight_tt.shape[3])
        mc = ttnn.DRAM_MEMORY_CONFIG

        fc_input_tt = ttnn.from_torch(
            torch.zeros(1, 1, 1, in_f, dtype=torch.bfloat16),
            device=self.device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=mc,
        )

        # Untraced warmup so kernel JIT runs OUTSIDE trace capture (otherwise
        # the JIT triggers host writes that violate trace constraints).
        _wu = ttnn.linear(
            fc_input_tt,
            self._fc_linear_weight_tt,
            bias=self._fc_bias_tt,
            compute_kernel_config=self._compute_kernel_config,
            memory_config=mc,
        )
        ttnn.deallocate(_wu)
        ttnn.synchronize_device(self.device)

        trace_id = ttnn.begin_trace_capture(self.device, cq_id=0)
        try:
            output_tt = ttnn.linear(
                fc_input_tt,
                self._fc_linear_weight_tt,
                bias=self._fc_bias_tt,
                compute_kernel_config=self._compute_kernel_config,
                memory_config=mc,
            )
        finally:
            ttnn.end_trace_capture(self.device, trace_id, cq_id=0)
        ttnn.synchronize_device(self.device)

        self._fc_trace = {
            "input_tt": fc_input_tt,
            "output_tt": output_tt,
            "trace_id": trace_id,
            "in_f": in_f,
            "out_f": out_f,
        }

    def capture_se_block_traces(self) -> None:
        """Pre-capture an execute_trace for each SE block's compute path.

        Uses ``ttnn.conv1d`` with the canonical unit-test config
        (ROW_MAJOR_LAYOUT NLC input, auto-shard, HiFi4 + fp32_accum), which
        works for our [1, 1, in_c] shape without the host-fallback weight
        prep that breaks in trace capture.
        """
        if getattr(self, "_se_traces", None):
            return
        self._se_traces: dict = {}
        mc = ttnn.DRAM_MEMORY_CONFIG
        for block_idx in (1, 2, 3):
            prefix = f"blocks.{block_idx}."
            w1 = self.pytorch_weights.get(f"{prefix}se_block.conv1.weight")
            b1 = self.pytorch_weights.get(f"{prefix}se_block.conv1.bias")
            w2 = self.pytorch_weights.get(f"{prefix}se_block.conv2.weight")
            b2 = self.pytorch_weights.get(f"{prefix}se_block.conv2.bias")
            if w1 is None or w2 is None:
                continue

            in_c = int(w1.shape[1])
            bottleneck_c = int(w1.shape[0])
            out_c = int(w2.shape[0])

            # Persistent input buffer [1, 1, in_c] in ROW_MAJOR_LAYOUT NLC
            # (matches the conv1d unit test pattern in
            # tests/ttnn/unit_tests/operations/conv/test_conv1d.py).
            input_tt = ttnn.from_torch(
                torch.zeros(1, 1, in_c, dtype=torch.bfloat16),
                device=self.device,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=mc,
            )

            # Conv weights and biases as simple from_torch (no special layout).
            def _w_tt(w_torch):
                return ttnn.from_torch(w_torch.float(), dtype=ttnn.bfloat16)

            def _b_tt(b_torch):
                return ttnn.from_torch(b_torch.float().reshape(1, 1, 1, -1), dtype=ttnn.bfloat16)

            w1_tt = _w_tt(w1)
            b1_tt = _b_tt(b1)
            w2_tt = _w_tt(w2)
            b2_tt = _b_tt(b2)

            # Use HiFi4 + fp32_accum to match the precision of the host
            # fp32 conv path that the model was trained against.
            se_conv_config = ttnn.Conv1dConfig(
                weights_dtype=ttnn.bfloat16,
                shard_layout=None,
                deallocate_activation=False,
            )
            se_compute_config = ttnn.init_device_compute_kernel_config(
                self.device.arch(),
                math_fidelity=ttnn.MathFidelity.HiFi4,
                fp32_dest_acc_en=True,
            )

            # Pre-warm conv1d OUTSIDE trace to extract prepared weights.
            # Subsequent calls inside trace use already-prepared weights so
            # no host→device writes happen during capture.
            [_wu_y, _wu_len, [w1_prep, b1_prep]] = ttnn.conv1d(
                input_tensor=input_tt,
                weight_tensor=w1_tt,
                bias_tensor=b1_tt,
                in_channels=in_c,
                out_channels=bottleneck_c,
                device=self.device,
                kernel_size=1,
                stride=1,
                padding=0,
                batch_size=1,
                input_length=1,
                conv_config=se_conv_config,
                compute_config=se_compute_config,
                return_output_dim=True,
                return_weights_and_bias=True,
            )
            ttnn.deallocate(_wu_y)
            # Need an intermediate ROW_MAJOR buffer with conv1's output shape
            # to warm conv2 with prepared weights too.
            wu_mid = ttnn.from_torch(
                torch.zeros(1, 1, bottleneck_c, dtype=torch.bfloat16),
                device=self.device,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=mc,
            )
            [_wu_y2, _wu_len2, [w2_prep, b2_prep]] = ttnn.conv1d(
                input_tensor=wu_mid,
                weight_tensor=w2_tt,
                bias_tensor=b2_tt,
                in_channels=bottleneck_c,
                out_channels=out_c,
                device=self.device,
                kernel_size=1,
                stride=1,
                padding=0,
                batch_size=1,
                input_length=1,
                conv_config=se_conv_config,
                compute_config=se_compute_config,
                return_output_dim=True,
                return_weights_and_bias=True,
            )
            ttnn.deallocate(_wu_y2)
            ttnn.deallocate(wu_mid)
            ttnn.synchronize_device(self.device)

            def _se_compute(x):
                # x is [1, 1, in_c] NLC ROW_MAJOR (matches input_tt).
                # conv1d returns [1, 1, 1, out_c]; reshape to NLC for next conv.
                [y, _y_len] = ttnn.conv1d(
                    input_tensor=x,
                    weight_tensor=w1_prep,
                    bias_tensor=b1_prep,
                    in_channels=in_c,
                    out_channels=bottleneck_c,
                    device=self.device,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    batch_size=1,
                    input_length=1,
                    conv_config=se_conv_config,
                    compute_config=se_compute_config,
                    return_output_dim=True,
                    return_weights_and_bias=False,
                )
                y = ttnn.reshape(y, (1, 1, bottleneck_c))
                y = ttnn.relu(y)
                [y, _y_len] = ttnn.conv1d(
                    input_tensor=y,
                    weight_tensor=w2_prep,
                    bias_tensor=b2_prep,
                    in_channels=bottleneck_c,
                    out_channels=out_c,
                    device=self.device,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    batch_size=1,
                    input_length=1,
                    conv_config=se_conv_config,
                    compute_config=se_compute_config,
                    return_output_dim=True,
                    return_weights_and_bias=False,
                )
                y = ttnn.reshape(y, (1, 1, out_c))
                return ttnn.sigmoid(y)

            # Untraced warmup so kernels are JIT-compiled before capture.
            _wu = _se_compute(input_tt)
            ttnn.deallocate(_wu)
            ttnn.synchronize_device(self.device)

            # Capture trace.
            trace_id = ttnn.begin_trace_capture(self.device, cq_id=0)
            try:
                output_tt = _se_compute(input_tt)
            finally:
                ttnn.end_trace_capture(self.device, trace_id, cq_id=0)
            ttnn.synchronize_device(self.device)

            self._se_traces[block_idx] = {
                "input_tt": input_tt,
                "output_tt": output_tt,  # writeable by execute_trace
                "trace_id": trace_id,
                "in_c": in_c,
                "out_c": out_c,
            }

    def forward_from_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Extract speaker embedding directly from audio waveform.

        Args:
            audio: Audio waveform [num_samples] or [batch, num_samples]

        Returns:
            Speaker embedding [batch, 2048]
        """
        mel = self.compute_mel_spectrogram(audio)
        return self.forward(mel)

    def to_ttnn(self, speaker_embedding: torch.Tensor) -> ttnn.Tensor:
        """
        Convert speaker embedding to TTNN tensor.

        Args:
            speaker_embedding: PyTorch speaker embedding [batch, 2048]

        Returns:
            TTNN tensor [batch, 1, 1, 2048]
        """
        # Reshape for TTNN: [batch, 2048] -> [batch, 1, 1, 2048]
        embedding = speaker_embedding.unsqueeze(1).unsqueeze(1)
        return ttnn.from_torch(
            embedding,
            device=self.device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
