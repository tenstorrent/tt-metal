from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

import ttnn

from .math_perf_env import (
    ace_step_binary_kwargs,
    ace_step_dense_linear_program_config,
    ace_step_dit_weight_dtype,
    ace_step_dit_weight_layout,
    ace_step_dit_weight_memory_config,
    ace_step_ensure_l1_activation,
    ace_step_ensure_tile_layout,
    ace_step_init_dit_linear_compute_kernel_config,
    ace_step_linear_l1_memory_config,
    ace_step_pad_activation_kwargs,
    ace_step_reshape_kwargs,
)


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

    Operates on TILE + L1 tensors to avoid DRAM ``Tilize`` before ``proj_in``.
    """
    if len(hidden_states.shape) != 3:
        raise ValueError(f"Expected hidden_states rank-3 [B, T, C], got shape={hidden_states.shape}")
    if not ttnn.is_tensor_storage_on_device(hidden_states):
        raise AssertionError("Expected hidden_states to be device-resident (TTNN tensor on device).")
    if patch_size <= 0:
        raise ValueError(f"patch_size must be > 0, got {patch_size}")

    meta = _patchify_pad_meta(int(hidden_states.shape[1]), patch_size)
    if meta.pad_length == 0:
        return ace_step_ensure_l1_activation(ttnn, ace_step_ensure_tile_layout(ttnn, hidden_states))

    _l1_mc = ace_step_linear_l1_memory_config(ttnn)
    x = ace_step_ensure_tile_layout(ttnn, hidden_states)
    if _l1_mc is not None:
        x = ace_step_ensure_l1_activation(ttnn, x, _l1_mc)
    hs4 = ttnn.unsqueeze(x, 1)
    hs4 = ttnn.pad(
        hs4,
        padding=((0, 0), (0, 0), (0, meta.pad_length), (0, 0)),
        value=value,
        **ace_step_pad_activation_kwargs(ttnn, _l1_mc),
    )
    out = ttnn.squeeze(hs4, 1)
    if _l1_mc is not None:
        out = ace_step_ensure_l1_activation(ttnn, out, _l1_mc)
    return out


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

    try:
        import torch  # type: ignore

        if isinstance(x, torch.Tensor):
            return x.detach().to(dtype=torch.float32, device="cpu").numpy()
    except Exception:
        pass

    return np.asarray(x)


def _proj_in_conv_weight_to_linear(
    weight_host_np: np.ndarray, *, out_channels: int, in_channels: int, patch_size: int
) -> np.ndarray:
    """Map ``Conv1d`` weight ``[O, I, P]`` to linear ``[O, I*P]`` for non-overlapping patch matmul."""
    if tuple(weight_host_np.shape) != (out_channels, in_channels, patch_size):
        raise ValueError(
            f"proj_in weight shape mismatch: got {tuple(weight_host_np.shape)}, "
            f"expected ({out_channels}, {in_channels}, {patch_size})"
        )
    # Feature order is patch_pos * in_channels + channel (see ``reshape`` in ``forward``).
    w_p_ic = np.transpose(weight_host_np, (0, 2, 1))  # [O, P, I]
    return np.ascontiguousarray(w_p_ic.reshape(out_channels, patch_size * in_channels))


class TtAceStepPatchEmbed1D:
    """
    TTNN equivalent of HF's `proj_in` patch embedding in `AceStepDiTModel`.

    Uses a fused **linear** on TILE/L1 activations instead of ``ttnn.conv1d`` so Tracy does not
    pay ``Tilize`` / ``TilizeWithValPadding`` / ``CopyDevice`` / im2col DRAM matmul on every forward
    (see ``perf_dit_4`` stacked report). Weights stay in DRAM; activations are L1-interleaved.

    HF reference (conceptually):
      x: [B, T, C] → pad T → group patches → linear(C*p → H) → [B, T/p, H]
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
        weights_dtype = ace_step_dit_weight_dtype(ttnn, weights_dtype)

        self.patch_size = int(getattr(config, "patch_size"))
        self.in_channels = int(getattr(config, "in_channels"))
        self.out_channels = int(getattr(config, "hidden_size"))
        self.expected_input_length = int(expected_input_length) if expected_input_length is not None else None
        self.in_features = self.in_channels * self.patch_size

        weight_key = _maybe_get_state_dict_key(
            state_dict,
            (
                f"{base_address}.1.weight",
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

        weight_host = _to_numpy_host_array(state_dict[weight_key])
        bias_host = _to_numpy_host_array(state_dict[bias_key])

        if weight_host.shape[0] != self.out_channels:
            raise ValueError(
                f"Unexpected proj_in out_channels: got {weight_host.shape[0]}, expected {self.out_channels}"
            )
        if weight_host.shape[1] != self.in_channels:
            raise ValueError(f"Unexpected proj_in in_channels: got {weight_host.shape[1]}, expected {self.in_channels}")
        if int(weight_host.shape[2]) != self.patch_size:
            raise ValueError(f"Unexpected proj_in kernel_size: got {weight_host.shape[2]}, expected {self.patch_size}")

        w2d = _proj_in_conv_weight_to_linear(
            weight_host,
            out_channels=self.out_channels,
            in_channels=self.in_channels,
            patch_size=self.patch_size,
        )
        _w_layout = ace_step_dit_weight_layout(ttnn, weights_dtype, default_layout=ttnn.TILE_LAYOUT)
        _w_mc = ace_step_dit_weight_memory_config(ttnn) or ttnn.DRAM_MEMORY_CONFIG
        self.weight = ttnn.as_tensor(
            w2d,
            dtype=weights_dtype,
            layout=_w_layout,
            device=self.device,
            memory_config=_w_mc,
        )
        self.bias = ttnn.as_tensor(
            bias_host.reshape(1, 1, 1, -1),
            dtype=activation_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=_w_mc,
        )
        self.activation_dtype = activation_dtype
        self._ck = ace_step_init_dit_linear_compute_kernel_config(device) or ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=math_fidelity,
            math_approx_mode=math_approx_mode,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        self._l1_mc = ace_step_linear_l1_memory_config(ttnn)
        self._pc_cache: dict = {}

    def forward(self, hidden_states: ttnn.Tensor) -> Tuple[ttnn.Tensor, PatchifyMetadata]:
        if len(hidden_states.shape) != 3:
            raise ValueError(f"Expected hidden_states shape [B, T, C], got {hidden_states.shape}")
        if not ttnn.is_tensor_storage_on_device(hidden_states):
            raise AssertionError("Expected hidden_states to be device-resident (TTNN tensor on device).")
        if int(hidden_states.shape[-1]) != self.in_channels:
            raise ValueError(f"Expected in_channels={self.in_channels}, got C={hidden_states.shape[-1]}")

        meta = _patchify_pad_meta(int(hidden_states.shape[1]), self.patch_size)
        hidden_states = _pad_seq_len_to_patch_size(hidden_states, self.patch_size, value=0.0)

        b = int(hidden_states.shape[0])
        t = int(hidden_states.shape[1])
        if t % self.patch_size != 0:
            raise ValueError(f"Padded sequence length {t} not divisible by patch_size={self.patch_size}")
        t_p = t // self.patch_size

        _sr = ace_step_reshape_kwargs(ttnn)
        _l1_mc = self._l1_mc
        x = ace_step_ensure_tile_layout(ttnn, hidden_states)
        if _l1_mc is not None:
            x = ace_step_ensure_l1_activation(ttnn, x, _l1_mc)

        # [B, T, C] → [B, T_p, P, C] → [B*T_p, P*C]
        x = ttnn.reshape(x, (b, t_p, self.patch_size, self.in_channels), **_sr)
        x = ttnn.reshape(x, (b * t_p, self.in_features), **_sr)
        if _l1_mc is not None:
            x = ace_step_ensure_l1_activation(ttnn, x, _l1_mc)

        m = b * t_p
        _pc = self._pc_cache.get((m, self.in_features, self.out_channels))
        if _pc is None:
            _pc = ace_step_dense_linear_program_config(
                self.device,
                seq_len=m,
                in_dim=self.in_features,
                out_dim=self.out_channels,
            )
            if _pc is not None:
                self._pc_cache[(m, self.in_features, self.out_channels)] = _pc

        _lin_mc = _l1_mc or ttnn.DRAM_MEMORY_CONFIG
        _lin_kw: dict = {"dtype": self.activation_dtype, "memory_config": _lin_mc}
        if self._ck is not None:
            _lin_kw["compute_kernel_config"] = self._ck
        if _pc is not None:
            _lin_kw["program_config"] = _pc

        w_tile = ace_step_ensure_tile_layout(ttnn, self.weight)
        y2d = ttnn.linear(x, w_tile, bias=None, transpose_b=True, **_lin_kw)
        if int(y2d.shape[-1]) > self.out_channels:
            y2d = y2d[:, : self.out_channels]
        if _l1_mc is not None:
            y2d = ace_step_ensure_l1_activation(ttnn, y2d, _l1_mc)

        out = ttnn.reshape(y2d, (b, t_p, self.out_channels), **_sr)
        out4 = ttnn.unsqueeze(out, 1)
        _bkw = ace_step_binary_kwargs(ttnn, _l1_mc)
        out4 = ttnn.add(out4, self.bias, **_bkw)
        if _l1_mc is not None:
            out4 = ace_step_ensure_l1_activation(ttnn, out4, _l1_mc)
        out = ttnn.squeeze(out4, 1)
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
        weights_dtype = ace_step_dit_weight_dtype(ttnn, weights_dtype)

        self.patch_size = int(getattr(config, "patch_size"))
        self.in_channels = int(getattr(config, "hidden_size"))
        self.out_channels = int(getattr(config, "audio_acoustic_hidden_dim"))

        weight_key = _maybe_get_state_dict_key(
            state_dict,
            (
                f"{base_address}.1.weight",
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

        weight_host = state_dict[weight_key]
        bias_host = state_dict[bias_key]

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

        weight_host_np = _to_numpy_host_array(weight_host)
        bias_host_np = _to_numpy_host_array(bias_host)

        w_blocks = []
        for i in range(self.patch_size):
            w_blocks.append(weight_host_np[:, :, i])

        w_io_times_p = np.concatenate(w_blocks, axis=1)
        w2d = np.ascontiguousarray(np.swapaxes(w_io_times_p, 0, 1))

        _w_layout = ace_step_dit_weight_layout(ttnn, weights_dtype, default_layout=ttnn.TILE_LAYOUT)
        _w_mc = ace_step_dit_weight_memory_config(ttnn) or ttnn.DRAM_MEMORY_CONFIG
        self.weight = ttnn.as_tensor(
            w2d,
            dtype=weights_dtype,
            layout=_w_layout,
            device=self.device,
            memory_config=_w_mc,
        )
        self.bias = ttnn.as_tensor(
            bias_host_np.reshape(1, 1, 1, -1),
            dtype=activation_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=_w_mc,
        )
        self.activation_dtype = activation_dtype
        self._ck = ace_step_init_dit_linear_compute_kernel_config(device)
        self._pc_cache: dict = {}
        self._l1_mc = ace_step_linear_l1_memory_config(ttnn)

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

        m = b * t_p
        n = self.out_channels * self.patch_size

        _sr = ace_step_reshape_kwargs(ttnn)
        _l1_mc = self._l1_mc
        x2d = ace_step_ensure_tile_layout(ttnn, ttnn.reshape(hidden_states, (m, self.in_channels), **_sr))
        if _l1_mc is not None:
            x2d = ace_step_ensure_l1_activation(ttnn, x2d, _l1_mc)

        w_tile = ace_step_ensure_tile_layout(ttnn, self.weight)
        _pc = self._pc_cache.get((m, self.in_channels, n))
        if _pc is None:
            _pc = ace_step_dense_linear_program_config(self.device, seq_len=m, in_dim=self.in_channels, out_dim=n)
            if _pc is not None:
                self._pc_cache[(m, self.in_channels, n)] = _pc
        _lin_mc = _l1_mc or ttnn.DRAM_MEMORY_CONFIG
        _lin_kw: dict = {"dtype": self.activation_dtype, "memory_config": _lin_mc}
        if self._ck is not None:
            _lin_kw["compute_kernel_config"] = self._ck
        if _pc is not None:
            _lin_kw["program_config"] = _pc
        y2d = ttnn.linear(x2d, w_tile, bias=None, transpose_b=True, **_lin_kw)
        if int(y2d.shape[-1]) > n:
            y2d = y2d[:, :n]
        if _l1_mc is not None:
            y2d = ace_step_ensure_l1_activation(ttnn, y2d, _l1_mc)

        y = ttnn.reshape(y2d, (b, t_p * self.patch_size, self.out_channels), **_sr)
        if _l1_mc is not None:
            y = ace_step_ensure_l1_activation(ttnn, y, _l1_mc)

        y4 = ttnn.unsqueeze(y, 1)
        _bkw = ace_step_binary_kwargs(ttnn, _l1_mc)
        y4 = ttnn.add(y4, self.bias, **_bkw)
        if _l1_mc is not None:
            y4 = ace_step_ensure_l1_activation(ttnn, y4, _l1_mc)
        y = ttnn.squeeze(y4, 1)

        if meta.pad_length:
            y = y[:, : meta.original_seq_len, :]
        return y

    def __call__(self, hidden_states: ttnn.Tensor, meta: PatchifyMetadata) -> ttnn.Tensor:
        return self.forward(hidden_states, meta)
