from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

import ttnn


@dataclass(frozen=True)
class PatchifyMetadata:
    """Metadata needed to undo patchify padding/packing."""

    original_seq_len: int
    pad_length: int
    patch_size: int


def _patchify_pad_meta(seq_len: int, patch_size: int) -> PatchifyMetadata:
    if patch_size <= 0:
        raise ValueError(f"patch_size must be > 0, got {patch_size}")
    remainder = seq_len % patch_size
    pad_length = 0 if remainder == 0 else (patch_size - remainder)
    return PatchifyMetadata(original_seq_len=seq_len, pad_length=pad_length, patch_size=patch_size)


def _pad_seq_len_to_patch_size(hidden_states: ttnn.Tensor, patch_size: int, *, value: float = 0.0) -> ttnn.Tensor:
    """
    Pads `hidden_states` along the sequence dimension so its length is divisible by `patch_size`.

    Expected shape: [B, T, C] (row-major).
    """
    if len(hidden_states.shape) != 3:
        raise ValueError(f"Expected hidden_states rank-3 [B, T, C], got shape={hidden_states.shape}")
    if not ttnn.is_tensor_storage_on_device(hidden_states):
        raise AssertionError("Expected hidden_states to be device-resident (TTNN tensor on device).")
    if patch_size <= 0:
        raise ValueError(f"patch_size must be > 0, got {patch_size}")

    meta = _patchify_pad_meta(int(hidden_states.shape[1]), patch_size)
    if meta.pad_length == 0:
        return hidden_states

    # ttnn.pad operates on rank-4 tensors with explicit (N, C, H, W) padding tuples.
    # Treat sequence length as H and channels as W: [B, 1, T, C]
    hs4 = ttnn.unsqueeze(hidden_states, 1)
    hs4 = ttnn.pad(hs4, padding=((0, 0), (0, 0), (0, meta.pad_length), (0, 0)), value=value)
    return ttnn.squeeze(hs4, 1)


def _maybe_get_state_dict_key(state_dict: dict, candidates: Tuple[str, ...]) -> str:
    for k in candidates:
        if k in state_dict:
            return k
    raise KeyError(f"None of the candidate keys were found in state_dict: {candidates}")


def _to_numpy_host_array(x):
    """
    Convert `x` into a CPU NumPy array for host-side preprocessing.

    This module's weights may come from:
    - NumPy arrays (already fine)
    - Torch tensors (including bfloat16, which cannot be implicitly converted to NumPy)
    """

    # Torch tensors: detach and convert to float32 on CPU before NumPy ops.
    # Import torch lazily so this file can be imported even in minimal environments.
    try:
        import torch  # type: ignore

        if isinstance(x, torch.Tensor):
            return x.detach().to(dtype=torch.float32, device="cpu").numpy()
    except Exception:
        # If torch isn't installed or something unexpected happens, fall through to numpy conversion attempt.
        pass

    # NumPy arrays / array-likes
    return np.asarray(x)


class TtAceStepPatchEmbed1D:
    """
    TTNN equivalent of HF's `proj_in` patch embedding in `AceStepDiTModel`.

    HF reference (conceptually):
      x: [B, T, C]
      if T % patch_size != 0: pad T
      x = x.transpose(1,2)              # [B, C, T]
      x = Conv1d(C -> inner_dim, k=p, s=p)(x)  # [B, inner_dim, T/p]
      x = x.transpose(1,2)              # [B, T/p, inner_dim]

    TTNN conv1d expects input shaped [B, T, C], so we can call it directly without transposes.
    """

    def __init__(
        self,
        *,
        config,
        state_dict: dict,
        base_address: str,
        device: ttnn.Device,
        expected_input_length: int | None = None,
        activation_dtype: ttnn.DataType | None = None,
        weights_dtype: ttnn.DataType | None = None,
        math_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.HiFi2,
        math_approx_mode: bool = False,
    ) -> None:
        self.config = config
        self.device = device
        if activation_dtype is None:
            activation_dtype = getattr(ttnn, "bfloat16", None)
        if weights_dtype is None:
            weights_dtype = getattr(ttnn, "bfloat16", None)
        if activation_dtype is None or weights_dtype is None:
            raise RuntimeError("TTNN build missing bfloat16 dtype; pass activation_dtype/weights_dtype explicitly.")

        self.patch_size = int(getattr(config, "patch_size"))
        self.in_channels = int(getattr(config, "in_channels"))
        self.out_channels = int(getattr(config, "hidden_size"))
        self.expected_input_length = int(expected_input_length) if expected_input_length is not None else None

        weight_key = _maybe_get_state_dict_key(
            state_dict,
            (
                f"{base_address}.1.weight",  # nn.Sequential(Lambda, Conv1d, Lambda)
                f"{base_address}.weight",
            ),
        )
        bias_key = _maybe_get_state_dict_key(
            state_dict,
            (
                f"{base_address}.1.bias",
                f"{base_address}.bias",
            ),
        )

        weight_host = state_dict[weight_key]  # [out_channels, in_channels, patch_size]
        bias_host = state_dict[bias_key]  # [out_channels]

        if weight_host.shape[0] != self.out_channels:
            raise ValueError(
                f"Unexpected proj_in out_channels: got {weight_host.shape[0]}, expected {self.out_channels}"
            )
        if weight_host.shape[1] != self.in_channels:
            raise ValueError(f"Unexpected proj_in in_channels: got {weight_host.shape[1]}, expected {self.in_channels}")
        if int(weight_host.shape[2]) != self.patch_size:
            raise ValueError(f"Unexpected proj_in kernel_size: got {weight_host.shape[2]}, expected {self.patch_size}")

        # We must avoid conv2d's internal "pull back to host" fallback. That means weights/bias must be
        # prepared (host-side preprocessing) *before* the first conv invocation, then moved to device once.
        #
        # `prepare_conv_weights/prepare_conv_bias` expect HOST tensors and return prepared tensors on DEVICE.
        weight_host_tt = ttnn.as_tensor(weight_host, dtype=weights_dtype, layout=ttnn.ROW_MAJOR_LAYOUT)
        # Bias for conv1d/conv2d path is expected in NHWC-like rank-4 form [1,1,1,C]
        bias_host_tt = ttnn.as_tensor(bias_host.reshape(1, 1, 1, -1), dtype=weights_dtype, layout=ttnn.ROW_MAJOR_LAYOUT)
        self._weight_host_tt = weight_host_tt
        self._bias_host_tt = bias_host_tt
        self._packed_for: tuple[int, int] | None = None  # (batch_size, padded_input_length)

        self.conv_config = ttnn.Conv1dConfig(
            weights_dtype=weights_dtype, shard_layout=None, deallocate_activation=False
        )
        self.compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=math_fidelity,
            math_approx_mode=math_approx_mode,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        self.activation_dtype = activation_dtype
        # Pre-pack conv weights/bias at init when expected_input_length is known.
        # This keeps the entire forward pass device-pure (no TTNN->host mid-run transfers).
        #
        # When expected_input_length is not provided (e.g. unit tests), we lazily pack on first forward
        # based on runtime input shape and cache the packed weights.
        self.weight = None
        self.bias = None
        if self.expected_input_length is not None:
            padded_len = int(((self.expected_input_length + self.patch_size - 1) // self.patch_size) * self.patch_size)
            self._ensure_packed(batch_size=1, padded_input_length=padded_len)

    def _ensure_packed(self, *, batch_size: int, padded_input_length: int) -> None:
        if self._packed_for == (batch_size, padded_input_length) and self.weight is not None and self.bias is not None:
            return

        self.weight = ttnn.prepare_conv_weights(
            weight_tensor=self._weight_host_tt,
            input_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            input_layout=ttnn.ROW_MAJOR_LAYOUT,
            weights_format="OIHW",
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            batch_size=batch_size,
            input_height=1,
            input_width=padded_input_length,
            kernel_size=(1, self.patch_size),
            stride=(1, self.patch_size),
            padding=(0, 0),
            dilation=(1, 1),
            has_bias=True,
            groups=1,
            device=self.device,
            input_dtype=self.activation_dtype,
            conv_config=self.conv_config,
            compute_config=self.compute_config,
        )
        self.bias = ttnn.prepare_conv_bias(
            bias_tensor=self._bias_host_tt,
            input_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            input_layout=ttnn.ROW_MAJOR_LAYOUT,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            batch_size=batch_size,
            input_height=1,
            input_width=padded_input_length,
            kernel_size=(1, self.patch_size),
            stride=(1, self.patch_size),
            padding=(0, 0),
            dilation=(1, 1),
            device=self.device,
            input_dtype=self.activation_dtype,
            groups=1,
            conv_config=self.conv_config,
            compute_config=self.compute_config,
        )
        self._packed_for = (batch_size, padded_input_length)

    def forward(self, hidden_states: ttnn.Tensor) -> Tuple[ttnn.Tensor, PatchifyMetadata]:
        if len(hidden_states.shape) != 3:
            raise ValueError(f"Expected hidden_states shape [B, T, C], got {hidden_states.shape}")
        if not ttnn.is_tensor_storage_on_device(hidden_states):
            raise AssertionError("Expected hidden_states to be device-resident (TTNN tensor on device).")
        if int(hidden_states.shape[-1]) != self.in_channels:
            raise ValueError(f"Expected in_channels={self.in_channels}, got C={hidden_states.shape[-1]}")

        meta = _patchify_pad_meta(int(hidden_states.shape[1]), self.patch_size)
        hidden_states = _pad_seq_len_to_patch_size(hidden_states, self.patch_size, value=0.0)

        batch_size = int(hidden_states.shape[0])
        input_length = int(hidden_states.shape[1])

        self._ensure_packed(batch_size=batch_size, padded_input_length=input_length)

        weight_tensor = self.weight
        bias_tensor = self.bias
        if weight_tensor is None or bias_tensor is None:
            raise RuntimeError("Internal error: conv weights/bias were not packed")
        if not ttnn.is_tensor_storage_on_device(weight_tensor):
            raise AssertionError("Expected proj_in weight to be device-resident.")
        if not ttnn.is_tensor_storage_on_device(bias_tensor):
            raise AssertionError("Expected proj_in bias to be device-resident.")

        conv_ret = ttnn.conv1d(
            input_tensor=hidden_states,
            weight_tensor=weight_tensor,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            device=self.device,
            bias_tensor=bias_tensor,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding=0,
            batch_size=batch_size,
            input_length=input_length,
            conv_config=self.conv_config,
            compute_config=self.compute_config,
            groups=1,
            return_output_dim=True,
            return_weights_and_bias=False,
            dtype=self.activation_dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        if len(conv_ret) != 2:
            raise RuntimeError(
                "Unexpected `ttnn.conv1d` return arity. "
                f"Expected 2 when return_output_dim=True and return_weights_and_bias=False, got {len(conv_ret)}."
            )

        out, out_length = conv_ret

        # conv1d returns rank-4 [1, B, out_length, out_channels]
        out = ttnn.squeeze(out, 0)
        out = ttnn.reshape(out, (batch_size, out_length, out.shape[-1]))
        return out, meta

    def __call__(self, hidden_states: ttnn.Tensor) -> Tuple[ttnn.Tensor, PatchifyMetadata]:
        return self.forward(hidden_states)


class TtAceStepDePatchify1D:
    """
    TTNN equivalent of HF's `proj_out` de-patchify in `AceStepDiTModel`.

    HF reference (conceptually):
      x: [B, T_p, inner_dim]
      x = x.transpose(1,2)                       # [B, inner_dim, T_p]
      x = ConvTranspose1d(inner_dim -> out_ch, k=p, s=p)(x)  # [B, out_ch, T_p*p]
      x = x.transpose(1,2)                       # [B, T, out_ch]

    Because kernel_size == stride == patch_size and padding == 0, this ConvTranspose1d is non-overlapping and can be
    implemented as a per-token linear projection that expands each token into `patch_size` frames.
    """

    def __init__(
        self,
        *,
        config,
        state_dict: dict,
        base_address: str,
        device: ttnn.Device,
        activation_dtype: ttnn.DataType | None = None,
        weights_dtype: ttnn.DataType | None = None,
    ) -> None:
        self.config = config
        self.device = device
        if activation_dtype is None:
            activation_dtype = getattr(ttnn, "bfloat16", None)
        if weights_dtype is None:
            weights_dtype = getattr(ttnn, "bfloat16", None)
        if activation_dtype is None or weights_dtype is None:
            raise RuntimeError("TTNN build missing bfloat16 dtype; pass activation_dtype/weights_dtype explicitly.")

        self.patch_size = int(getattr(config, "patch_size"))
        self.in_channels = int(getattr(config, "hidden_size"))
        self.out_channels = int(getattr(config, "audio_acoustic_hidden_dim"))

        weight_key = _maybe_get_state_dict_key(
            state_dict,
            (
                f"{base_address}.1.weight",  # nn.Sequential(Lambda, ConvTranspose1d, Lambda)
                f"{base_address}.weight",
            ),
        )
        bias_key = _maybe_get_state_dict_key(
            state_dict,
            (
                f"{base_address}.1.bias",
                f"{base_address}.bias",
            ),
        )

        weight_host = state_dict[weight_key]  # ConvTranspose1d weight: [in_channels, out_channels, patch_size]
        bias_host = state_dict[bias_key]  # [out_channels]

        if int(weight_host.shape[0]) != self.in_channels:
            raise ValueError(
                f"Unexpected proj_out in_channels: got {weight_host.shape[0]}, expected {self.in_channels}"
            )
        if int(weight_host.shape[1]) != self.out_channels:
            raise ValueError(
                f"Unexpected proj_out out_channels: got {weight_host.shape[1]}, expected {self.out_channels}"
            )
        if int(weight_host.shape[2]) != self.patch_size:
            raise ValueError(f"Unexpected proj_out kernel_size: got {weight_host.shape[2]}, expected {self.patch_size}")

        # Build a linear weight matrix W2d of shape [out_features, in_features] where out_features = out_channels*patch.
        #
        # HF uses `ConvTranspose1d(in=inner_dim, out=out_ch, kernel_size=p, stride=p)` on a tensor shaped
        #   x: [B, inner_dim, T_p]
        # producing:
        #   y: [B, out_ch, T_p * p]
        #
        # For stride==kernel==p and padding==0, this is a non-overlapping upsample: each input time step contributes
        # independently to `p` consecutive output samples for *each* output channel.
        #
        # If we flatten each input token to length `inner_dim` and want an output token of length `out_ch * p`,
        # the correct stacked weight rows are:
        #   row (out_ch * p + i) == convtranspose weight[:, out_ch, i]  (shape [in])
        # i.e. stack `p` slices of shape [in, out] into [in, out*p] then transpose to [out*p, in].
        # `torch.nn.ConvTranspose1d` weight layout is [in_channels, out_channels, kernel_size].
        # For each kernel tap i, the slice `torch_weight[:, :, i]` is [in, out] and maps to a contiguous
        # block of `out` outputs in the upsampled time axis.
        weight_host_np = _to_numpy_host_array(weight_host)
        bias_host_np = _to_numpy_host_array(bias_host)

        w_blocks = []
        for i in range(self.patch_size):
            w_blocks.append(weight_host_np[:, :, i])  # [in, out]

        w_io_times_p = np.concatenate(w_blocks, axis=1)  # [in, out*p]
        w2d = np.ascontiguousarray(np.swapaxes(w_io_times_p, 0, 1))  # [out*p, in]

        # Host -> device transfer happens once here (allowed). Keep weights device-resident for all forwards.
        self.weight = ttnn.as_tensor(
            w2d,
            dtype=weights_dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.bias = ttnn.as_tensor(
            bias_host_np.reshape(1, 1, 1, -1),
            dtype=activation_dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.activation_dtype = activation_dtype

    def forward(self, hidden_states: ttnn.Tensor, meta: PatchifyMetadata) -> ttnn.Tensor:
        if meta.patch_size != self.patch_size:
            raise ValueError(
                f"Patch size mismatch: meta.patch_size={meta.patch_size} vs module.patch_size={self.patch_size}"
            )
        if len(hidden_states.shape) != 3:
            raise ValueError(f"Expected hidden_states shape [B, T_p, inner_dim], got {hidden_states.shape}")
        if not ttnn.is_tensor_storage_on_device(hidden_states):
            raise AssertionError("Expected hidden_states to be device-resident (TTNN tensor on device).")
        if not ttnn.is_tensor_storage_on_device(self.weight):
            raise AssertionError("Expected proj_out weight to be device-resident.")
        if not ttnn.is_tensor_storage_on_device(self.bias):
            raise AssertionError("Expected proj_out bias to be device-resident.")
        if int(hidden_states.shape[-1]) != self.in_channels:
            raise ValueError(f"Expected inner_dim={self.in_channels}, got {hidden_states.shape[-1]}")

        b = int(hidden_states.shape[0])
        t_p = int(hidden_states.shape[1])

        # Matmul path requires TILE; reshapes are safest in ROW_MAJOR (tile padding can corrupt naive reshapes).
        m = b * t_p
        n = self.out_channels * self.patch_size

        x2d = ttnn.reshape(hidden_states, (m, self.in_channels))
        x2d = ttnn.to_layout(x2d, ttnn.TILE_LAYOUT)

        w_tile = ttnn.to_layout(self.weight, ttnn.TILE_LAYOUT)
        y2d = ttnn.linear(
            x2d,
            w_tile,
            bias=None,
            transpose_b=True,  # y = x @ W^T
            dtype=self.activation_dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        y2d_rm = ttnn.to_layout(y2d, ttnn.ROW_MAJOR_LAYOUT)

        # `ttnn.linear` outputs may be padded to tile alignment on the last dim; reshape using the true M/N.
        if y2d_rm.shape != (m, n):
            if int(y2d_rm.shape[0]) != m:
                raise RuntimeError(f"Unexpected linear output rows: got {y2d_rm.shape[0]}, expected {m}")
            y2d_rm = y2d_rm[:, :n]

        y = ttnn.reshape(y2d_rm, (b, t_p, self.patch_size, self.out_channels))
        y = ttnn.reshape(y, (b, t_p * self.patch_size, self.out_channels))

        # Add bias per output channel (broadcast across batch and time)
        y4 = ttnn.unsqueeze(y, 1)  # [B, 1, T, C]
        y4 = y4 + self.bias
        y = ttnn.squeeze(y4, 1)

        # Trim any pad added during patchify
        if meta.pad_length:
            y = y[:, : meta.original_seq_len, :]
        return y

    def __call__(self, hidden_states: ttnn.Tensor, meta: PatchifyMetadata) -> ttnn.Tensor:
        return self.forward(hidden_states, meta)
