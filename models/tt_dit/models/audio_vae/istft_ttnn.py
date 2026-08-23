import math

import torch

import ttnn
from models.tt_dit.layers.module import Module, Parameter


class ISTFT(Module):
    def __init__(
        self,
        *,
        n_fft: int,
        hop_length: int,
        mesh_device: ttnn.MeshDevice,
        dtype: ttnn.DataType = ttnn.bfloat16,
    ) -> None:
        super().__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_freqs = n_fft // 2 + 1
        self.mesh_device = mesh_device
        self.dtype = dtype

        self.cos_basis = Parameter(
            total_shape=[1, self.n_freqs, self.n_fft],
            device=mesh_device,
            dtype=dtype,
            pad_value=0.0,
        )

        self.sin_basis = Parameter(
            total_shape=[1, self.n_freqs, self.n_fft],
            device=mesh_device,
            dtype=dtype,
            pad_value=0.0,
        )

        self.window = Parameter(
            total_shape=[1, 1, self.n_fft],
            device=mesh_device,
            dtype=dtype,
            pad_value=0.0,
        )

        # conv_transpose2d accepts host weights and prepares them internally.
        self.ola_weight = Parameter(
            total_shape=[self.n_fft, 1, 1, self.n_fft],
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=dtype,
            on_host=True,
        )

        self._window_torch = torch.hann_window(self.n_fft)

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        k = torch.arange(self.n_freqs).float()
        n = torch.arange(self.n_fft).float()

        angle = 2 * math.pi * k[:, None] * n[None, :] / self.n_fft

        weights = torch.ones(self.n_freqs)
        if self.n_freqs > 1:
            weights[1:] = 2.0
            if self.n_fft % 2 == 0:
                weights[-1] = 1.0

        state["cos_basis"] = (torch.cos(angle) / self.n_fft * weights[:, None]).unsqueeze(0)

        state["sin_basis"] = (torch.sin(angle) / self.n_fft * weights[:, None]).unsqueeze(0)

        state["window"] = self._window_torch.reshape(1, 1, self.n_fft)

        ola_weight = torch.zeros(self.n_fft, 1, 1, self.n_fft)
        for idx in range(self.n_fft):
            ola_weight[idx, 0, 0, idx] = 1.0

        state["ola_weight"] = ola_weight

    def _compute_config(self):
        return ttnn.init_device_compute_kernel_config(
            self.mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )

    def inverse_frames(
        self,
        real: ttnn.Tensor,
        imag: ttnn.Tensor,
    ) -> ttnn.Tensor:
        compute_config = self._compute_config()

        real_time = ttnn.matmul(
            real,
            self.cos_basis.data,
            compute_kernel_config=compute_config,
        )

        imag_time = ttnn.matmul(
            imag,
            self.sin_basis.data,
            compute_kernel_config=compute_config,
        )

        frames = ttnn.subtract(real_time, imag_time)

        ttnn.deallocate(real_time)
        ttnn.deallocate(imag_time)

        return frames

    def forward(
        self,
        real: ttnn.Tensor,
        imag: ttnn.Tensor,
    ) -> ttnn.Tensor:
        batch = int(real.shape[0])
        num_frames = int(real.shape[-2])

        frames = self.inverse_frames(real, imag)

        windowed = ttnn.multiply(
            frames,
            self.window.data,
        )

        ola_input = ttnn.reshape(
            windowed,
            [batch, 1, num_frames, self.n_fft],
        )

        conv_config = ttnn.Conv2dConfig(
            weights_dtype=self.dtype,
            shard_layout=None,
            deallocate_activation=False,
            enable_act_double_buffer=False,
            output_layout=ttnn.TILE_LAYOUT,
            config_tensors_in_dram=False,
        )

        result = ttnn.conv_transpose2d(
            input_tensor=ola_input,
            weight_tensor=self.ola_weight.data,
            bias_tensor=None,
            in_channels=self.n_fft,
            out_channels=1,
            batch_size=batch,
            input_height=1,
            input_width=num_frames,
            kernel_size=(1, self.n_fft),
            stride=(1, self.hop_length),
            padding=(0, 0),
            output_padding=(0, 0),
            dilation=(1, 1),
            groups=1,
            device=self.mesh_device,
            conv_config=conv_config,
            compute_config=self._compute_config(),
            mirror_kernel=True,
            return_output_dim=True,
            dtype=self.dtype,
        )

        ola, dims = result[0], result[1]
        output_length = int(dims[1])

        ola = ttnn.reshape(
            ola,
            [batch, 1, output_length, 1],
        )

        # torch.istft(center=True)
        crop = self.n_fft // 2

        ola = ttnn.slice(
            ola,
            [0, 0, crop, 0],
            [batch, 1, output_length - crop, 1],
        )

        # Window-square envelope normalization.
        envelope = torch.zeros(output_length)

        for frame_idx in range(num_frames):
            start = frame_idx * self.hop_length
            envelope[start : start + self.n_fft] += self._window_torch.square()

        envelope = envelope[crop : output_length - crop]
        envelope = envelope.clamp_min(1e-11)

        envelope_tt = ttnn.from_torch(
            envelope.reshape(1, 1, -1, 1),
            device=self.mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=self.dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        return ttnn.divide(ola, envelope_tt)
