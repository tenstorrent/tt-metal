# Copyright (c) 2025
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch
from scipy.signal.windows import hann

import ttnn


class TtSnake:
    def __init__(self, alpha: torch.Tensor, device):
        """
        TTNN implementation of Snake activation.
        Snake(x) = x + (1/alpha + eps) * sin(alpha * x)^2
        alpha shape should match the channel dimension of x.
        """
        self.device = device
        # Prepare alpha: [1, 1, 1, C] for broadcasting with [N, 1, L, C] input
        alpha_reshaped = alpha.view(1, 1, 1, -1)
        self.alpha_tensor = ttnn.from_torch(alpha_reshaped, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        inv_alpha = 1.0 / (alpha_reshaped + 1e-9)
        self.inv_alpha_tensor = ttnn.from_torch(inv_alpha, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        # x is [B, 1, L, C]
        ax = ttnn.mul(x, self.alpha_tensor)
        sin_ax = ttnn.sin(ax)
        sin_ax2 = ttnn.square(sin_ax)
        term = ttnn.mul(self.inv_alpha_tensor, sin_ax2)
        return ttnn.add(x, term)


class TtResBlock:
    def __init__(self, device, state_dict, prefix="", kernel_size=3, dilations=[1, 3, 5]):
        self.device = device
        self.convs1_weights = []
        self.convs1_biases = []
        self.convs2_weights = []
        self.convs2_biases = []
        self.activations1 = []
        self.activations2 = []

        self.num_layers = len(dilations)
        self.kernel_size = kernel_size
        self.dilations1 = dilations
        self.dilations2 = [1] * len(dilations)

        w0 = state_dict[f"{prefix}convs1.0.weight"]
        if w0 is None:
            w0 = state_dict[f"{prefix}convs1.0.weight.data"]
        self.out_channels, self.in_channels, self.ks = w0.shape

        self.conv_config = ttnn.Conv2dConfig(
            weights_dtype=ttnn.bfloat16,
            config_tensors_in_dram=True,
        )

        for i in range(self.num_layers):
            # Activations
            alpha1 = state_dict[f"{prefix}activations1.{i}.alpha"]
            alpha2 = state_dict[f"{prefix}activations2.{i}.alpha"]
            self.activations1.append(TtSnake(alpha1, device))
            self.activations2.append(TtSnake(alpha2, device))

            # Convs 1
            w1 = state_dict[f"{prefix}convs1.{i}.weight"].unsqueeze(2)  # [C_out, C_in, 1, K]
            b1 = state_dict[f"{prefix}convs1.{i}.bias"]
            self.convs1_weights.append(ttnn.from_torch(w1, dtype=ttnn.bfloat16))
            self.convs1_biases.append(ttnn.from_torch(b1.view(1, 1, 1, -1), dtype=ttnn.bfloat16))

            # Convs 2
            w2 = state_dict[f"{prefix}convs2.{i}.weight"].unsqueeze(2)  # [C_out, C_in, 1, K]
            b2 = state_dict[f"{prefix}convs2.{i}.bias"]
            self.convs2_weights.append(ttnn.from_torch(w2, dtype=ttnn.bfloat16))
            self.convs2_biases.append(ttnn.from_torch(b2.view(1, 1, 1, -1), dtype=ttnn.bfloat16))

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        # x expected in [B, 1, L, C]
        batch_size = x.shape[0]
        length = x.shape[2]

        for i in range(self.num_layers):
            xt = self.activations1[i](x)

            # Conv1
            padding1 = (self.kernel_size - 1) * self.dilations1[i] // 2
            xt, _, _ = ttnn.conv1d(
                input_tensor=xt,
                weight_tensor=self.convs1_weights[i],
                device=self.device,
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                batch_size=batch_size,
                input_length=length,
                kernel_size=self.kernel_size,
                stride=1,
                padding=padding1,
                dilation=self.dilations1[i],
                groups=1,
                bias_tensor=self.convs1_biases[i],
                dtype=ttnn.bfloat16,
                conv_config=self.conv_config,
                return_output_dim=True,
                return_weights_and_bias=True,
            )

            xt = self.activations2[i](xt)

            # Conv2
            padding2 = (self.kernel_size - 1) * self.dilations2[i] // 2
            xt, _, _ = ttnn.conv1d(
                input_tensor=xt,
                weight_tensor=self.convs2_weights[i],
                device=self.device,
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                batch_size=batch_size,
                input_length=length,
                kernel_size=self.kernel_size,
                stride=1,
                padding=padding2,
                dilation=self.dilations2[i],
                groups=1,
                bias_tensor=self.convs2_biases[i],
                dtype=ttnn.bfloat16,
                conv_config=self.conv_config,
                return_output_dim=True,
                return_weights_and_bias=True,
            )

            x = ttnn.add(xt, x)
        return x


class TtCausalResBlock:
    """
    TTNN ResBlock using CausalConv1d (causal_type='left') for all internal convolutions.

    Used in CausalHiFTGenerator for both source_resblocks and main resblocks.
    """

    def __init__(self, device, state_dict, prefix="", kernel_size=3, dilations=[1, 3, 5]):
        self.device = device
        self.num_layers = len(dilations)
        self.convs1 = []
        self.convs2 = []
        self.activations1 = []
        self.activations2 = []

        for i in range(self.num_layers):
            # Activations
            alpha1 = state_dict[f"{prefix}activations1.{i}.alpha"]
            alpha2 = state_dict[f"{prefix}activations2.{i}.alpha"]
            self.activations1.append(TtSnake(alpha1, device))
            self.activations2.append(TtSnake(alpha2, device))

            w1 = state_dict[f"{prefix}convs1.{i}.weight"]
            b1 = state_dict[f"{prefix}convs1.{i}.bias"]
            out_ch, in_ch, ks = w1.shape
            self.convs1.append(
                TtCausalConv1d(
                    device,
                    w1,
                    b1,
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=ks,
                    dilation=dilations[i],
                    causal_type="left",
                )
            )

            w2 = state_dict[f"{prefix}convs2.{i}.weight"]
            b2 = state_dict[f"{prefix}convs2.{i}.bias"]
            self.convs2.append(
                TtCausalConv1d(
                    device,
                    w2,
                    b2,
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=ks,
                    dilation=1,
                    causal_type="left",
                )
            )

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        for i in range(self.num_layers):
            xt = self.activations1[i](x)
            xt = self.convs1[i](xt)
            xt = self.activations2[i](xt)
            xt = self.convs2[i](xt)
            x = ttnn.add(xt, x)
        return x


class TtConv1d:
    """Wrapper for a single TTNN conv1d layer with pre-loaded weights."""

    def __init__(self, device, weight, bias, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        self.device = device
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.conv_config = ttnn.Conv2dConfig(
            weights_dtype=ttnn.bfloat16,
            config_tensors_in_dram=True,
        )

        # weight: [out_C, in_C, K] -> [out_C, in_C, 1, K]
        self.weight_tt = ttnn.from_torch(weight.unsqueeze(2), dtype=ttnn.bfloat16)
        self.bias_tt = ttnn.from_torch(bias.view(1, 1, 1, -1), dtype=ttnn.bfloat16)

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        # x: [B, 1, L, C]
        batch_size = x.shape[0]
        length = x.shape[2]

        y, _, _ = ttnn.conv1d(
            input_tensor=x,
            weight_tensor=self.weight_tt,
            device=self.device,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            batch_size=batch_size,
            input_length=length,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=1,
            bias_tensor=self.bias_tt,
            dtype=ttnn.bfloat16,
            conv_config=self.conv_config,
            return_output_dim=True,
            return_weights_and_bias=True,
        )
        return y


class TtHiFTGenerator:
    """
    TTNN implementation of the HiFTGenerator vocoder.

    Maps the PyTorch HiFTGenerator architecture to Tenstorrent hardware.
    The STFT/ISTFT operations and f0 prediction remain on host CPU
    since they involve complex FFT operations not suited for accelerator mapping.

    Architecture:
        conv_pre -> [upsample + source_fusion + resblocks] x num_upsamples -> conv_post -> ISTFT

    Default config (CosyVoice-300M):
        - in_channels: 80 (mel spectrogram bins)
        - base_channels: 512
        - upsample_rates: [8, 8]
        - upsample_kernel_sizes: [16, 16]
        - resblock_kernel_sizes: [3, 7, 11]
        - resblock_dilation_sizes: [[1,3,5], [1,3,5], [1,3,5]]
        - source_resblock_kernel_sizes: [7, 11]
        - source_resblock_dilation_sizes: [[1,3,5], [1,3,5]]
        - istft n_fft: 16, hop_len: 4
    """

    def __init__(
        self,
        device,
        state_dict,
        prefix="",
        in_channels=80,
        base_channels=512,
        upsample_rates=None,
        upsample_kernel_sizes=None,
        resblock_kernel_sizes=None,
        resblock_dilation_sizes=None,
        source_resblock_kernel_sizes=None,
        source_resblock_dilation_sizes=None,
        istft_params=None,
        lrelu_slope=0.1,
    ):
        self.device = device
        self.lrelu_slope = lrelu_slope

        # Defaults matching CosyVoice-300M
        if upsample_rates is None:
            upsample_rates = [8, 8]
        if upsample_kernel_sizes is None:
            upsample_kernel_sizes = [16, 16]
        if resblock_kernel_sizes is None:
            resblock_kernel_sizes = [3, 7, 11]
        if resblock_dilation_sizes is None:
            resblock_dilation_sizes = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
        if source_resblock_kernel_sizes is None:
            source_resblock_kernel_sizes = [7, 11]
        if source_resblock_dilation_sizes is None:
            source_resblock_dilation_sizes = [[1, 3, 5], [1, 3, 5]]
        if istft_params is None:
            istft_params = {"n_fft": 16, "hop_len": 4}

        self.upsample_rates = upsample_rates
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.num_upsamples = len(upsample_rates)
        self.num_kernels = len(resblock_kernel_sizes)
        self.istft_params = istft_params

        self.conv_config = ttnn.Conv2dConfig(
            weights_dtype=ttnn.bfloat16,
            config_tensors_in_dram=True,
        )

        # ---- conv_pre: Conv1d(in_channels, base_channels, 7, 1, padding=3) ----
        self.conv_pre = TtConv1d(
            device,
            weight=state_dict[f"{prefix}conv_pre.weight"],
            bias=state_dict[f"{prefix}conv_pre.bias"],
            in_channels=in_channels,
            out_channels=base_channels,
            kernel_size=7,
            padding=3,
        )

        # ---- Upsampling layers (ConvTranspose1d) ----
        self.ups_weights = []
        self.ups_biases = []
        self.ups_in_channels = []
        self.ups_out_channels = []

        for i in range(self.num_upsamples):
            in_ch = base_channels // (2**i)
            out_ch = base_channels // (2 ** (i + 1))
            w = state_dict[f"{prefix}ups.{i}.weight"].unsqueeze(2)  # [in_C, out_C, 1, k]
            b = state_dict[f"{prefix}ups.{i}.bias"].view(1, 1, 1, -1)
            self.ups_weights.append(ttnn.from_torch(w, dtype=ttnn.bfloat16))
            self.ups_biases.append(ttnn.from_torch(b, dtype=ttnn.bfloat16))
            self.ups_in_channels.append(in_ch)
            self.ups_out_channels.append(out_ch)

        # ---- Source downsampling layers (Conv1d) ----
        import numpy as np

        downsample_rates = [1] + upsample_rates[::-1][:-1]
        downsample_cum_rates = np.cumprod(downsample_rates)

        self.source_downs = []
        self.source_resblocks = []
        source_in_channels = istft_params["n_fft"] + 2

        for i, (u, k, d) in enumerate(
            zip(downsample_cum_rates[::-1], source_resblock_kernel_sizes, source_resblock_dilation_sizes)
        ):
            u = int(u)
            out_ch = base_channels // (2 ** (i + 1))

            if u == 1:
                sd = TtConv1d(
                    device,
                    weight=state_dict[f"{prefix}source_downs.{i}.weight"],
                    bias=state_dict[f"{prefix}source_downs.{i}.bias"],
                    in_channels=source_in_channels,
                    out_channels=out_ch,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            else:
                sd = TtConv1d(
                    device,
                    weight=state_dict[f"{prefix}source_downs.{i}.weight"],
                    bias=state_dict[f"{prefix}source_downs.{i}.bias"],
                    in_channels=source_in_channels,
                    out_channels=out_ch,
                    kernel_size=u * 2,
                    stride=u,
                    padding=u // 2,
                )
            self.source_downs.append(sd)

            # Source ResBlock
            self.source_resblocks.append(
                TtResBlock(device, state_dict, prefix=f"{prefix}source_resblocks.{i}.", kernel_size=k, dilations=d)
            )

        # ---- Main ResBlocks ----
        self.resblocks = []
        for i in range(self.num_upsamples):
            ch = base_channels // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                idx = i * self.num_kernels + j
                self.resblocks.append(
                    TtResBlock(device, state_dict, prefix=f"{prefix}resblocks.{idx}.", kernel_size=k, dilations=d)
                )

        # ---- conv_post: Conv1d(ch, n_fft + 2, 7, 1, padding=3) ----
        last_ch = base_channels // (2**self.num_upsamples)
        post_out = istft_params["n_fft"] + 2
        self.conv_post = TtConv1d(
            device,
            weight=state_dict[f"{prefix}conv_post.weight"],
            bias=state_dict[f"{prefix}conv_post.bias"],
            in_channels=last_ch,
            out_channels=post_out,
            kernel_size=7,
            padding=3,
        )

    def _apply_upsample(self, x: ttnn.Tensor, i: int) -> ttnn.Tensor:
        """Apply leaky relu + transposed convolution for upsampling stage i."""
        batch_size = x.shape[0]
        length = x.shape[2]

        u = self.upsample_rates[i]
        k = self.upsample_kernel_sizes[i]
        padding = (k - u) // 2

        x = ttnn.leaky_relu(x, negative_slope=self.lrelu_slope)

        x = ttnn.conv_transpose2d(
            input_tensor=x,
            weight_tensor=self.ups_weights[i],
            device=self.device,
            in_channels=self.ups_in_channels[i],
            out_channels=self.ups_out_channels[i],
            batch_size=batch_size,
            input_height=1,
            input_width=length,
            kernel_size=(1, k),
            stride=(1, u),
            padding=(0, padding),
            output_padding=(0, 0),
            groups=1,
            bias_tensor=self.ups_biases[i],
            conv_config=self.conv_config,
        )

        return x

    def _apply_reflection_pad(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """
        Apply ReflectionPad1d((1, 0)) - pad 1 element on the left using reflection.
        x: [B, 1, L, C] -> pads along the L dimension.
        For reflection pad (1, 0), the first element is reflected: new_x[0] = x[1], then x[0], x[1], ...
        """
        # Move to interleaved DRAM if sharded (slice doesn't support BLOCK_SHARDED)
        dram_interleaved = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
        x = ttnn.to_memory_config(x, dram_interleaved)

        # Extract the second time-step and prepend it
        x_pad = ttnn.slice(x, [0, 0, 1, 0], [x.shape[0], 1, 2, x.shape[3]])
        x = ttnn.concat([x_pad, x], dim=2)
        return x

    def decode(self, x: ttnn.Tensor, s_stft: ttnn.Tensor) -> ttnn.Tensor:
        """
        Main decode forward pass.

        Args:
            x: Input mel features on device, shape [B, 1, L, C] where C=80
            s_stft: Source STFT features on device, shape [B, 1, L_src, C_src] where C_src=n_fft+2

        Returns:
            Output tensor [B, 1, L_out, C_out] where C_out = n_fft + 2
        """
        # conv_pre
        x = self.conv_pre(x)

        for i in range(self.num_upsamples):
            # Upsample (leaky_relu + conv_transpose)
            x = self._apply_upsample(x, i)

            # Reflection pad on last upsample
            if i == self.num_upsamples - 1:
                x = self._apply_reflection_pad(x)

            # Source fusion
            si = self.source_downs[i](s_stft)
            si = self.source_resblocks[i](si)
            x = ttnn.add(x, si)

            # Multi-kernel ResBlock fusion
            xs = None
            for j in range(self.num_kernels):
                rb_out = self.resblocks[i * self.num_kernels + j](x)
                if xs is None:
                    xs = rb_out
                else:
                    xs = ttnn.add(xs, rb_out)

            # Average over kernels
            inv_num_kernels = 1.0 / self.num_kernels
            x = ttnn.mul(xs, inv_num_kernels)

        # Final activation + conv_post
        x = ttnn.leaky_relu(x, negative_slope=0.01)
        x = self.conv_post(x)

        return x

    @staticmethod
    def _stft(x: torch.Tensor, n_fft: int, hop_len: int, stft_window: torch.Tensor):
        """
        Compute STFT on host CPU.

        Args:
            x: Audio waveform [B, T] (squeezed from [B, 1, T])
            n_fft: FFT size
            hop_len: Hop length
            stft_window: Hann window tensor

        Returns:
            Tuple of (real, imag) spectrograms, each [B, n_fft//2+1, num_frames]
        """
        spec = torch.stft(
            x,
            n_fft,
            hop_len,
            n_fft,
            window=stft_window.to(x.device),
            return_complex=True,
        )
        spec = torch.view_as_real(spec)  # [B, F, TT, 2]
        return spec[..., 0], spec[..., 1]

    @staticmethod
    def _istft(magnitude: torch.Tensor, phase: torch.Tensor, n_fft: int, hop_len: int, stft_window: torch.Tensor):
        """
        Compute inverse STFT on host CPU.

        Args:
            magnitude: Magnitude spectrogram [B, n_fft//2+1, T]
            phase: Phase spectrogram [B, n_fft//2+1, T]
            n_fft: FFT size
            hop_len: Hop length
            stft_window: Hann window tensor

        Returns:
            Audio waveform [B, T_out]
        """
        magnitude = torch.clip(magnitude, max=1e2)
        real = magnitude * torch.cos(phase)
        img = magnitude * torch.sin(phase)
        inverse_transform = torch.istft(
            torch.complex(real, img),
            n_fft,
            hop_len,
            n_fft,
            window=stft_window.to(magnitude.device),
        )
        return inverse_transform

    def decode_full(
        self,
        speech_feat: torch.Tensor,
        source_audio: torch.Tensor,
        audio_limit: float = 0.99,
    ) -> torch.Tensor:
        """
        Full vocoder decode pipeline:
            1. Compute STFT of source audio on host CPU
            2. Transfer mel features and STFT to device
            3. Run neural network decode on device
            4. Transfer output back to host
            5. Apply ISTFT to reconstruct audio waveform

        Args:
            speech_feat: Mel spectrogram features [B, C, L] (C=80, on CPU)
            source_audio: Source audio waveform [B, 1, T] (on CPU)
            audio_limit: Clamp threshold for output audio

        Returns:
            Generated audio waveform [B, T_out] (on CPU)
        """
        n_fft = self.istft_params["n_fft"]
        hop_len = self.istft_params["hop_len"]
        stft_window = torch.from_numpy(hann(n_fft, sym=True).astype(np.float32))

        # Step 1: STFT on host CPU
        s_stft_real, s_stft_imag = self._stft(source_audio.squeeze(1), n_fft, hop_len, stft_window)
        s_stft = torch.cat([s_stft_real, s_stft_imag], dim=1)  # [B, n_fft+2, T_stft]

        # Step 2: Convert to TTNN layout [B, 1, L, C] and transfer to device
        x_tt = speech_feat.transpose(1, 2).unsqueeze(1)  # [B, 1, L, 80]
        x_tt = ttnn.from_torch(x_tt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)

        s_stft_tt = s_stft.transpose(1, 2).unsqueeze(1)  # [B, 1, T_stft, n_fft+2]
        s_stft_tt = ttnn.from_torch(s_stft_tt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)

        # Step 3: Neural network decode on device
        y_tt = self.decode(x_tt, s_stft_tt)

        # Step 4: Transfer back to host CPU
        y_cpu = ttnn.to_torch(y_tt)  # [B, 1, L_out, n_fft+2]
        y_cpu = y_cpu.squeeze(1).transpose(1, 2).float()  # [B, n_fft+2, L_out]

        # Step 5: ISTFT on host CPU
        n_half = n_fft // 2 + 1
        magnitude = torch.exp(y_cpu[:, :n_half, :])
        phase = torch.sin(y_cpu[:, n_half:, :])  # sin is redundancy per reference

        audio = self._istft(magnitude, phase, n_fft, hop_len, stft_window)
        audio = torch.clamp(audio, -audio_limit, audio_limit)

        return audio


class TtCausalConv1d:
    """
    TTNN implementation of CausalConv1d.

    Supports both causal_type='left' (pad zeros on left) and 'right' (pad zeros on right).
    Uses symmetric padding in ttnn.conv1d then trims the output to match causal behavior.

    For non-streaming inference, the cache is always zeros.
    """

    def __init__(
        self,
        device,
        weight,
        bias,
        in_channels,
        out_channels,
        kernel_size,
        dilation=1,
        groups=1,
        causal_type="left",
    ):
        self.device = device
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.groups = groups
        self.causal_type = causal_type

        # Match reference: int((kernel_size * dilation - dilation) / 2) * 2 + (kernel_size + 1) % 2
        self.causal_padding = int((kernel_size * dilation - dilation) / 2) * 2 + (kernel_size + 1) % 2

        self.conv_config = ttnn.Conv2dConfig(
            weights_dtype=ttnn.bfloat16,
            config_tensors_in_dram=True,
        )

        # weight: [out_C, in_C/groups, K] -> [out_C, in_C/groups, 1, K]
        self.weight_tt = ttnn.from_torch(weight.unsqueeze(2), dtype=ttnn.bfloat16)
        self.bias_tt = ttnn.from_torch(bias.view(1, 1, 1, -1), dtype=ttnn.bfloat16)

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        # x: [B, 1, L, C]
        batch_size = x.shape[0]
        length = x.shape[2]

        # Symmetric padding produces output length = L + 2*cp - (K-1)*D
        # Causal (left or right) produces output length = L + cp - (K-1)*D = L
        # Extra output from symmetric = cp elements to trim
        padding = self.causal_padding

        y, _, _ = ttnn.conv1d(
            input_tensor=x,
            weight_tensor=self.weight_tt,
            device=self.device,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            batch_size=batch_size,
            input_length=length,
            kernel_size=self.kernel_size,
            stride=1,
            padding=padding,
            dilation=self.dilation,
            groups=self.groups,
            bias_tensor=self.bias_tt,
            dtype=ttnn.bfloat16,
            conv_config=self.conv_config,
            return_output_dim=True,
            return_weights_and_bias=True,
        )

        # Trim to original length
        out_length = y.shape[2]
        if out_length > length:
            dram_interleaved = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
            y = ttnn.to_memory_config(y, dram_interleaved)
            if self.causal_type == "left":
                # Left causal: zeros on left, trim extra from right
                y = ttnn.slice(y, [0, 0, 0, 0], [batch_size, 1, length, self.out_channels])
            else:
                # Right causal: zeros on right, trim extra from left
                extra = out_length - length
                y = ttnn.slice(y, [0, 0, extra, 0], [batch_size, 1, out_length, self.out_channels])

        return y


class TtCausalConv1dUpsample:
    """
    TTNN implementation of CausalConv1dUpsample.

    Architecture: nn.Upsample(scale_factor=stride, mode='nearest') + Conv1d(kernel_size, stride=1, padding=0)
    With left causal padding = kernel_size - 1.
    """

    def __init__(self, device, weight, bias, in_channels, out_channels, kernel_size, stride):
        self.device = device
        self.stride = stride
        self.causal_padding = kernel_size - 1

        # The inner conv is a standard CausalConv1d (no dilation, groups=1)
        self.conv = TtCausalConv1d(
            device,
            weight,
            bias,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=1,
            groups=1,
        )

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        # x: [B, 1, L, C]
        # Step 1: Nearest-neighbor upsample along the L dimension by self.stride
        # ttnn doesn't have a direct 1D upsample, so we use repeat_interleave or reshape trick
        # Move to host, upsample, move back
        x_cpu = ttnn.to_torch(x)  # [B, 1, L, C]
        x_cpu = x_cpu.squeeze(1).transpose(1, 2)  # [B, C, L]
        x_up = torch.nn.functional.interpolate(x_cpu, scale_factor=self.stride, mode="nearest")  # [B, C, L*stride]
        x_up = x_up.transpose(1, 2).unsqueeze(1)  # [B, 1, L*stride, C]
        x = ttnn.from_torch(x_up, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)

        # Step 2: Causal Conv1d
        x = self.conv(x)

        return x


class TtCausalHiFTGenerator:
    """
    TTNN implementation of the CausalHiFTGenerator vocoder.

    This is the actual deployed model for CosyVoice-300M.

    Key differences from TtHiFTGenerator:
        - Uses CausalConv1d (left-padded) instead of symmetric Conv1d
        - Uses CausalConv1dUpsample (Upsample + Conv1d) instead of ConvTranspose1d
        - 3 upsample stages [8, 5, 3] instead of 2
        - conv_pre kernel_size = conv_pre_look_right + 1 (right-causal)

    Default config (CosyVoice3-0.5B):
        - in_channels: 80
        - base_channels: 512
        - upsample_rates: [8, 5, 3]
        - upsample_kernel_sizes: [16, 11, 7]
        - resblock_kernel_sizes: [3, 7, 11]
        - source_resblock_kernel_sizes: [7, 7, 11]
    """

    def __init__(
        self,
        device,
        state_dict,
        prefix="",
        in_channels=80,
        base_channels=512,
        upsample_rates=None,
        upsample_kernel_sizes=None,
        resblock_kernel_sizes=None,
        resblock_dilation_sizes=None,
        source_resblock_kernel_sizes=None,
        source_resblock_dilation_sizes=None,
        istft_params=None,
        lrelu_slope=0.1,
        conv_pre_look_right=4,
    ):
        self.device = device
        self.lrelu_slope = lrelu_slope

        # Defaults matching CosyVoice3-0.5B
        if upsample_rates is None:
            upsample_rates = [8, 5, 3]
        if upsample_kernel_sizes is None:
            upsample_kernel_sizes = [16, 11, 7]
        if resblock_kernel_sizes is None:
            resblock_kernel_sizes = [3, 7, 11]
        if resblock_dilation_sizes is None:
            resblock_dilation_sizes = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
        if source_resblock_kernel_sizes is None:
            source_resblock_kernel_sizes = [7, 7, 11]
        if source_resblock_dilation_sizes is None:
            source_resblock_dilation_sizes = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
        if istft_params is None:
            istft_params = {"n_fft": 16, "hop_len": 4}

        self.upsample_rates = upsample_rates
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.num_upsamples = len(upsample_rates)
        self.num_kernels = len(resblock_kernel_sizes)
        self.istft_params = istft_params

        # ---- conv_pre: CausalConv1d(in_channels, base_channels, look_right+1, causal_type='right') ----
        conv_pre_ks = conv_pre_look_right + 1
        self.conv_pre = TtCausalConv1d(
            device,
            weight=state_dict[f"{prefix}conv_pre.weight"],
            bias=state_dict[f"{prefix}conv_pre.bias"],
            in_channels=in_channels,
            out_channels=base_channels,
            kernel_size=conv_pre_ks,
            causal_type="right",
        )

        # ---- Upsampling layers (CausalConv1dUpsample = Upsample + Conv1d) ----
        self.ups = []
        for i in range(self.num_upsamples):
            in_ch = base_channels // (2**i)
            out_ch = base_channels // (2 ** (i + 1))
            self.ups.append(
                TtCausalConv1dUpsample(
                    device,
                    weight=state_dict[f"{prefix}ups.{i}.weight"],
                    bias=state_dict[f"{prefix}ups.{i}.bias"],
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=upsample_kernel_sizes[i],
                    stride=upsample_rates[i],
                )
            )

        # ---- Source downsampling layers (CausalConv1d / CausalConv1dDownSample) ----
        downsample_rates = [1] + upsample_rates[::-1][:-1]
        downsample_cum_rates = np.cumprod(downsample_rates)

        self.source_downs = []
        self.source_resblocks = []
        source_in_channels = istft_params["n_fft"] + 2

        for i, (u, k, d) in enumerate(
            zip(downsample_cum_rates[::-1], source_resblock_kernel_sizes, source_resblock_dilation_sizes)
        ):
            u = int(u)
            out_ch = base_channels // (2 ** (i + 1))

            if u == 1:
                sd = TtCausalConv1d(
                    device,
                    weight=state_dict[f"{prefix}source_downs.{i}.weight"],
                    bias=state_dict[f"{prefix}source_downs.{i}.bias"],
                    in_channels=source_in_channels,
                    out_channels=out_ch,
                    kernel_size=1,
                )
            else:
                # CausalConv1dDownSample: Conv1d with stride, causal_padding = stride - 1
                # For TTNN, we use TtConv1d with stride and appropriate padding
                sd = TtConv1d(
                    device,
                    weight=state_dict[f"{prefix}source_downs.{i}.weight"],
                    bias=state_dict[f"{prefix}source_downs.{i}.bias"],
                    in_channels=source_in_channels,
                    out_channels=out_ch,
                    kernel_size=u * 2,
                    stride=u,
                    padding=u - 1,  # causal_padding for CausalConv1dDownSample
                )
            self.source_downs.append(sd)

            self.source_resblocks.append(
                TtCausalResBlock(
                    device, state_dict, prefix=f"{prefix}source_resblocks.{i}.", kernel_size=k, dilations=d
                )
            )

        # ---- Main ResBlocks ----
        self.resblocks = []
        for i in range(self.num_upsamples):
            ch = base_channels // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                idx = i * self.num_kernels + j
                self.resblocks.append(
                    TtCausalResBlock(device, state_dict, prefix=f"{prefix}resblocks.{idx}.", kernel_size=k, dilations=d)
                )

        # ---- conv_post: CausalConv1d(ch, n_fft + 2, 7, causal_type='left') ----
        last_ch = base_channels // (2**self.num_upsamples)
        post_out = istft_params["n_fft"] + 2
        self.conv_post = TtCausalConv1d(
            device,
            weight=state_dict[f"{prefix}conv_post.weight"],
            bias=state_dict[f"{prefix}conv_post.bias"],
            in_channels=last_ch,
            out_channels=post_out,
            kernel_size=7,
        )

    def decode(self, x: ttnn.Tensor, s_stft: ttnn.Tensor) -> ttnn.Tensor:
        """
        Main decode forward pass for causal generator.

        Args:
            x: Input mel features on device, shape [B, 1, L, C] where C=80
            s_stft: Source STFT features on device, shape [B, 1, L_src, C_src]

        Returns:
            Output tensor [B, 1, L_out, C_out] where C_out = n_fft + 2
        """
        x = self.conv_pre(x)

        for i in range(self.num_upsamples):
            x = ttnn.leaky_relu(x, negative_slope=self.lrelu_slope)
            x = self.ups[i](x)

            if i == self.num_upsamples - 1:
                # Reflection pad on last upsample
                dram_interleaved = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
                x = ttnn.to_memory_config(x, dram_interleaved)
                x_pad = ttnn.slice(x, [0, 0, 1, 0], [x.shape[0], 1, 2, x.shape[3]])
                x = ttnn.concat([x_pad, x], dim=2)

            # Source fusion
            si = self.source_downs[i](s_stft)
            si = self.source_resblocks[i](si)
            x = ttnn.add(x, si)

            # Multi-kernel ResBlock fusion
            xs = None
            for j in range(self.num_kernels):
                rb_out = self.resblocks[i * self.num_kernels + j](x)
                if xs is None:
                    xs = rb_out
                else:
                    xs = ttnn.add(xs, rb_out)

            inv_num_kernels = 1.0 / self.num_kernels
            x = ttnn.mul(xs, inv_num_kernels)

        x = ttnn.leaky_relu(x, negative_slope=0.01)
        x = self.conv_post(x)

        return x

    def decode_full(
        self,
        speech_feat: torch.Tensor,
        source_audio: torch.Tensor,
        audio_limit: float = 0.99,
    ) -> torch.Tensor:
        """Full vocoder decode pipeline for causal generator."""
        n_fft = self.istft_params["n_fft"]
        hop_len = self.istft_params["hop_len"]
        stft_window = torch.from_numpy(hann(n_fft, sym=True).astype(np.float32))

        s_stft_real, s_stft_imag = TtHiFTGenerator._stft(source_audio.squeeze(1), n_fft, hop_len, stft_window)
        s_stft = torch.cat([s_stft_real, s_stft_imag], dim=1)

        x_tt = speech_feat.transpose(1, 2).unsqueeze(1)
        x_tt = ttnn.from_torch(x_tt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)

        s_stft_tt = s_stft.transpose(1, 2).unsqueeze(1)
        s_stft_tt = ttnn.from_torch(s_stft_tt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)

        y_tt = self.decode(x_tt, s_stft_tt)

        y_cpu = ttnn.to_torch(y_tt)
        y_cpu = y_cpu.squeeze(1).transpose(1, 2).float()

        n_half = n_fft // 2 + 1
        magnitude = torch.exp(y_cpu[:, :n_half, :])
        phase = torch.sin(y_cpu[:, n_half:, :])

        audio = TtHiFTGenerator._istft(magnitude, phase, n_fft, hop_len, stft_window)
        audio = torch.clamp(audio, -audio_limit, audio_limit)

        return audio
