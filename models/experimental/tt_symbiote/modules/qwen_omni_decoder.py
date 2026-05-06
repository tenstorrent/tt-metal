# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import math
import os

import torch
import torch.nn.functional as F

import ttnn
from models.experimental.tt_symbiote.core.module import TTNNModule
from models.experimental.tt_symbiote.core.utils import torch_dtype_to_ttnn_dtype, tree_map
from models.experimental.tt_symbiote.models.qwen_omni.distributed_config import (
    qwen_omni_replicated_concat_dim0_tensor_config,
)


def _mesh_host_stitch_device_shards(tt_tensor: ttnn.Tensor, mesh_device) -> torch.Tensor | None:
    """Concat host tensors when each mesh device holds a slice of dim ``d`` (e.g. conv width 48×8=384)."""
    if mesh_device is None or not hasattr(mesh_device, "get_num_devices"):
        return None
    nd = int(mesh_device.get_num_devices())
    if nd <= 1:
        return None
    shards = ttnn.get_device_tensors(tt_tensor)
    if len(shards) != nd:
        return None
    local = tuple(int(x) for x in shards[0].shape)
    for t in (tt_tensor.shape, getattr(tt_tensor, "padded_shape", None)):
        if t is None:
            continue
        logical = tuple(int(x) for x in t)
        if len(logical) != len(local):
            continue
        for d in range(len(logical)):
            if local[d] != logical[d] and local[d] * nd == logical[d]:
                parts = [ttnn.to_torch(s).contiguous() for s in shards]
                return torch.cat(parts, dim=d)
    return None


def _code2wav_bct_replicated_mesh_config(mesh_device):
    """code2wav ``[B, C, T]`` activations are replicated per mesh device (see ``TTNNQwenOmniConv2dNHWC``)."""
    return qwen_omni_replicated_concat_dim0_tensor_config(mesh_device)


def _ttnn_mesh_to_torch_one_replica(tt_tensor: ttnn.Tensor, mesh_device) -> torch.Tensor:
    """Host ``torch`` tensor matching one logical replica (avoids bad concat/slice on rank-3 BCT)."""
    if mesh_device is None or mesh_device.get_num_devices() <= 1:
        return ttnn.to_torch(tt_tensor).contiguous()
    stitched = _mesh_host_stitch_device_shards(tt_tensor, mesh_device)
    if stitched is not None:
        return stitched.contiguous()
    shards = ttnn.get_device_tensors(tt_tensor)
    if shards:
        return ttnn.to_torch(shards[0]).contiguous()
    # Replicated mesh: plain to_torch may stack replicas on the wrong axis; match TorchTTNNTensor.to_torch (concat batch, then [:lead]).
    composer = ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    result = ttnn.to_torch(tt_tensor, mesh_composer=composer)
    lead = int(tt_tensor.shape[0])
    if result.dim() >= 1 and int(result.shape[0]) > lead:
        result = result[:lead].contiguous()
    return result.contiguous()


def _ensure_code2wav_bct_full_t(out, mesh_device, expected_t: int):
    """Stitch width-sharded BCT conv shards on time and re-upload with ReplicateTensorToMesh; no-op if each device already has full T."""
    if mesh_device is None or not hasattr(mesh_device, "get_num_devices"):
        return out
    nd = int(mesh_device.get_num_devices())
    if nd <= 1:
        return out

    from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

    tt = None
    if isinstance(out, TorchTTNNTensor):
        tt = out.ttnn_tensor if out.ttnn_tensor is not None else None
    elif isinstance(out, ttnn.Tensor):
        tt = out
    if tt is None:
        return out

    shards = ttnn.get_device_tensors(tt)
    if len(shards) != nd:
        return out
    shard_t = int(shards[0].shape[-1])
    if shard_t == expected_t:
        return out
    if shard_t * nd != expected_t:
        return out

    parts = [ttnn.to_torch(s).contiguous() for s in shards]
    full = torch.cat(parts, dim=-1)
    return ttnn.from_torch(
        full,
        dtype=torch_dtype_to_ttnn_dtype(full.dtype),
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _upload_bct_replicated(x_t: torch.Tensor, mesh_device):
    """Upload a host ``[B, C, T]`` torch tensor to TTNN with ``ReplicateTensorToMesh``."""
    mesh_mapper = None
    if mesh_device is not None and hasattr(mesh_device, "get_num_devices") and mesh_device.get_num_devices() > 1:
        mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device)
    return ttnn.from_torch(
        x_t.contiguous(),
        dtype=torch_dtype_to_ttnn_dtype(x_t.dtype),
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=mesh_mapper,
    )


def _materialize_code2wav_bct_from_ttnn(tt_tensor: ttnn.Tensor, mesh_device) -> torch.Tensor:
    """One logical [B,C,T] on host; prefer one replica readback (dim=0 compose can truncate T on mismatched mesh metadata)."""
    return _ttnn_mesh_to_torch_one_replica(tt_tensor, mesh_device)


def _materialize_code2wav_chain_output(x, mesh_device) -> torch.Tensor:
    """Convert symbiote / TTNN activations after conv to plain ``torch`` for ``+ residual``."""
    from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

    if isinstance(x, TorchTTNNTensor):
        if x.ttnn_tensor is not None:
            return _materialize_code2wav_bct_from_ttnn(x.ttnn_tensor, mesh_device)
        return x.elem.contiguous()
    if isinstance(x, ttnn.Tensor):
        return _materialize_code2wav_bct_from_ttnn(x, mesh_device)
    return x.contiguous()


class TTNNQwen3OmniMoeCausalConvNet(TTNNModule):
    """HF ``Qwen3OmniMoeCausalConvNet``: causal padding on host ``torch``, convolution via :class:`TTNNConv1d`."""

    def __init__(self):
        super().__init__()
        self.conv = None
        self.stride = None
        self.kernel_size = None
        self.dilation = None
        self.padding = None

    @classmethod
    def from_torch(cls, m, *args, **kwargs):
        from models.experimental.tt_symbiote.models.qwen_omni.qwen_omni_modules import TTNNConv1d

        new = cls()
        new._fallback_torch_layer = m
        new.stride = m.stride
        new.kernel_size = m.kernel_size
        new.dilation = m.dilation
        new.padding = m.padding
        new.conv = TTNNConv1d.from_torch(m.conv)
        return new

    @staticmethod
    def _causal_conv_host_time_threshold() -> int:
        """Above this padded T, run HF nn.Conv1d on host (TTNN conv OOMs on very long code2wav). Env: TT_SYMBIOTE_CODE2WAV_CAUSAL_CONV_HOST_T (default 8192)."""
        raw = os.environ.get("TT_SYMBIOTE_CODE2WAV_CAUSAL_CONV_HOST_T", "8192")
        try:
            v = int(raw)
        except ValueError:
            v = 8192
        return max(512, v)

    def _get_extra_padding_for_conv1d(self, hidden_state: torch.Tensor) -> int:
        length = hidden_state.shape[-1]
        n_frames = (length - self.kernel_size + self.padding) / self.stride + 1
        ideal_length = (math.ceil(n_frames) - 1) * self.stride + (self.kernel_size - self.padding)
        return ideal_length - length

    def set_output_tensors_config_impl(self, output_tensors):
        if self.conv is not None:
            return self.conv.set_output_tensors_config_impl(output_tensors)
        return super().set_output_tensors_config_impl(output_tensors)

    def forward(self, hidden_state):
        x_t = _materialize_code2wav_chain_output(hidden_state, self.device)
        extra_padding = self._get_extra_padding_for_conv1d(x_t)
        t_padded = int(x_t.shape[-1]) + self.padding + int(extra_padding)
        expected_t = (t_padded - self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1
        x_t = F.pad(x_t, (self.padding, extra_padding), mode="constant", value=0)
        t_pad = int(x_t.shape[-1])
        # Long streams: TTNN conv can OOM (e.g. T~5e5); host path uses same padded tensor and weights as _fallback_torch_layer.conv.
        if t_pad > self._causal_conv_host_time_threshold():
            hf = self._fallback_torch_layer
            with torch.no_grad():
                out_t = hf.conv(x_t).contiguous()
            out = _upload_bct_replicated(out_t, self.device)
            return _ensure_code2wav_bct_full_t(out, self.device, expected_t)
        out = self.conv(x_t)
        return _ensure_code2wav_bct_full_t(out, self.device, expected_t)


class TTNNQwen3OmniMoeCausalTransConvNet(TTNNModule):
    """HF ``Qwen3OmniMoeCausalTransConvNet``: TTNN transpose conv, then host crop to match HF time trimming."""

    def __init__(self):
        super().__init__()
        self.conv = None
        self.left_pad = None
        self.right_pad = None

    @classmethod
    def from_torch(cls, m, *args, **kwargs):
        from models.experimental.tt_symbiote.models.qwen_omni.qwen_omni_modules import TTNNConvTranspose1d

        new = cls()
        new._fallback_torch_layer = m
        new.left_pad = int(m.left_pad)
        new.right_pad = int(m.right_pad)
        new.conv = TTNNConvTranspose1d.from_torch(m.conv)
        return new

    @staticmethod
    def _causal_trans_conv_host_time_threshold():
        """If T exceeds threshold, run HF transposed conv on host (DRAM/clicks vs TTNN). Env TT_SYMBIOTE_CODE2WAV_TRANS_CONV_HOST_T (default 1; 0 = TTNN-only)."""
        raw = os.environ.get("TT_SYMBIOTE_CODE2WAV_TRANS_CONV_HOST_T", "1")
        try:
            v = int(raw)
        except ValueError:
            v = 1
        if v <= 0:
            return None
        return max(1, v)

    def set_output_tensors_config_impl(self, output_tensors):
        if self.conv is not None:
            return self.conv.set_output_tensors_config_impl(output_tensors)
        return super().set_output_tensors_config_impl(output_tensors)

    def forward(self, hidden_state):
        x_t = _materialize_code2wav_chain_output(hidden_state, self.device)
        t_in = int(x_t.shape[-1])
        th = self._causal_trans_conv_host_time_threshold()
        dev = self.device
        mesh_mapper = None
        if dev is not None and hasattr(dev, "get_num_devices") and dev.get_num_devices() > 1:
            mesh_mapper = ttnn.ReplicateTensorToMesh(dev)

        if th is not None and t_in > th:
            hf = self._fallback_torch_layer
            with torch.no_grad():
                y_t = hf(x_t).contiguous()
            return ttnn.from_torch(
                y_t,
                dtype=torch_dtype_to_ttnn_dtype(y_t.dtype),
                layout=ttnn.TILE_LAYOUT,
                device=dev,
                mesh_mapper=mesh_mapper,
            )

        out = self.conv(x_t)
        if isinstance(out, ttnn.Tensor):
            y_t = _materialize_code2wav_bct_from_ttnn(out, self.device)
        else:
            y_t = out
        t_out = int(y_t.shape[-1])
        end = t_out - self.right_pad
        y_t = y_t[..., self.left_pad : end].contiguous()
        return ttnn.from_torch(
            y_t,
            dtype=torch_dtype_to_ttnn_dtype(y_t.dtype),
            layout=ttnn.TILE_LAYOUT,
            device=dev,
            mesh_mapper=mesh_mapper,
        )


class TTNNQwen3OmniMoeConvNeXtBlock(TTNNModule):
    """ConvNeXtBlock: TTNN depthwise (TTNNQwen3OmniMoeCausalConvNet) + host LayerNorm/pwconv/GELU/residual so branch and shortcut match T."""

    def __init__(self):
        super().__init__()
        self.dwconv = None

    @classmethod
    def from_torch(cls, m, *args, **kwargs):
        new = cls()
        new._fallback_torch_layer = m
        new.dwconv = TTNNQwen3OmniMoeCausalConvNet.from_torch(m.dwconv)
        return new

    def set_output_tensors_config_impl(self, output_tensors):
        cfg = _code2wav_bct_replicated_mesh_config(self.device)
        if cfg is None:
            return super().set_output_tensors_config_impl(output_tensors)

        from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

        def apply(e):
            if isinstance(e, TorchTTNNTensor):
                e.set_distributed_tensor_config(cfg)
            return e

        return tree_map(apply, output_tensors)

    def forward(self, hidden_states):
        dev = self.device
        residual = _materialize_code2wav_chain_output(hidden_states, dev)

        hidden_states = self.dwconv(hidden_states)
        x = _materialize_code2wav_chain_output(hidden_states, dev)

        hf = self._fallback_torch_layer
        x = x.permute(0, 2, 1)
        x = hf.norm(x)
        x = hf.pwconv1(x)
        x = hf.act(x)
        x = hf.pwconv2(x)
        x = hf.gamma * x
        x = x.permute(0, 2, 1)

        result = x + residual
        return _upload_bct_replicated(result, dev)


class TTNNQwen3OmniMoeCode2WavDecoderResidualUnit(TTNNModule):
    """code2wav residual unit: TTNN SnakeBeta + CausalConvNet; host add (conv stitches T for branch vs shortcut)."""

    def __init__(self):
        super().__init__()
        self.act1 = None
        self.conv1 = None
        self.act2 = None
        self.conv2 = None

    @classmethod
    def from_torch(cls, m, *args, **kwargs):
        from models.experimental.tt_symbiote.models.qwen_omni.qwen_omni_modules import TTNNSnakeBeta

        new = cls()
        new._fallback_torch_layer = m
        new.act1 = TTNNSnakeBeta.from_torch(m.act1)
        new.conv1 = TTNNQwen3OmniMoeCausalConvNet.from_torch(m.conv1)
        new.act2 = TTNNSnakeBeta.from_torch(m.act2)
        new.conv2 = TTNNQwen3OmniMoeCausalConvNet.from_torch(m.conv2)
        return new

    def set_output_tensors_config_impl(self, output_tensors):
        cfg = _code2wav_bct_replicated_mesh_config(self.device)
        if cfg is None:
            return super().set_output_tensors_config_impl(output_tensors)

        from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

        def apply(e):
            if isinstance(e, TorchTTNNTensor):
                e.set_distributed_tensor_config(cfg)
            return e

        return tree_map(apply, output_tensors)

    def forward(self, hidden_state):
        dev = self.device
        if self.act1 is not None:
            self.act1._bypass_tensor_wrapping = False
        if self.act2 is not None:
            self.act2._bypass_tensor_wrapping = False

        residual = _materialize_code2wav_chain_output(hidden_state, dev)

        hidden_state = self.act1(hidden_state)
        hidden_state = self.conv1(hidden_state)
        hidden_state = self.act2(hidden_state)
        hidden_state = self.conv2(hidden_state)

        branch = _materialize_code2wav_chain_output(hidden_state, dev)
        result = branch + residual
        return _upload_bct_replicated(result, dev)
