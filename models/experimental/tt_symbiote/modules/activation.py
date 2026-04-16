# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Activation function implementations for TTNN."""

import math
import os

import torch
import torch.nn.functional as F

import ttnn
from models.experimental.tt_symbiote.core.module import TTNNModule
from models.experimental.tt_symbiote.core.run_config import DistributedTensorConfig
from models.experimental.tt_symbiote.core.utils import torch_dtype_to_ttnn_dtype, tree_map


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
    if mesh_device is None or mesh_device.get_num_devices() <= 1:
        return None
    return DistributedTensorConfig(
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0),
        replicate_compose_slice_dim0_to_leading=True,
    )


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
    # Replicated mesh layouts often yield no per-device shards here; plain ``to_torch`` can then
    # compose 8 replicas along the wrong axis (e.g. T×8 → 5464 vs 683). Match
    # ``TorchTTNNTensor.to_torch``: concat on batch dim then keep leading batch only.
    composer = ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    result = ttnn.to_torch(tt_tensor, mesh_composer=composer)
    lead = int(tt_tensor.shape[0])
    if result.dim() >= 1 and int(result.shape[0]) > lead:
        result = result[:lead].contiguous()
    return result.contiguous()


class TTNNSilu(TTNNModule):
    """TTNN-accelerated SiLU activation function."""

    def __init__(self):
        super().__init__()
        self._fallback_torch_layer = torch.nn.SiLU()

    def forward(self, input_tensor: ttnn.Tensor) -> ttnn.Tensor:
        """Forward pass through SiLU activation."""
        if input_tensor.layout != ttnn.TILE_LAYOUT:
            input_tensor = ttnn.to_layout(input_tensor, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        tt_output = ttnn.silu(input_tensor, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return tt_output


class TTNNReLU(TTNNModule):
    """TTNN-accelerated ReLU activation function."""

    def __init__(self):
        super().__init__()
        self._fallback_torch_layer = torch.nn.ReLU()

    def forward(self, input_tensor: ttnn.Tensor) -> ttnn.Tensor:
        """Forward pass through ReLU activation."""
        if input_tensor.layout != ttnn.TILE_LAYOUT:
            input_tensor = ttnn.to_layout(input_tensor, ttnn.TILE_LAYOUT, memory_config=input_tensor.memory_config())
        tt_output = ttnn.relu(input_tensor, memory_config=input_tensor.memory_config())
        return tt_output


class TTNNGelu(TTNNModule):
    """TTNN-accelerated GELU activation function."""

    def __init__(self):
        super().__init__()
        self._fallback_torch_layer = torch.nn.GELU()

    def forward(self, input_tensor: ttnn.Tensor) -> ttnn.Tensor:
        """Forward pass through GELU activation."""
        if input_tensor.layout != ttnn.TILE_LAYOUT:
            input_tensor = ttnn.to_layout(input_tensor, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        tt_output = ttnn.gelu(input_tensor, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return tt_output


class TTNNSnakeBeta(TTNNModule):
    """TTNN SnakeBeta (HF ``SnakeBeta`` in Qwen3-Omni code2wav decoder): x + 1/b * sin^2(x*a)."""

    def __init__(self, in_features: int):
        super().__init__()
        self.in_features = in_features
        self.no_div_by_zero = 0.000000001
        self.alpha = None
        self.beta = None

    @classmethod
    def from_torch(cls, torch_layer, *args, **kwargs):
        in_features = int(getattr(torch_layer, "in_features", torch_layer.alpha.shape[0]))
        new_layer = cls(in_features)
        new_layer._fallback_torch_layer = torch_layer
        return new_layer

    def move_weights_to_device_impl(self):
        super().move_weights_to_device_impl()
        tl = self.torch_layer
        if tl is None:
            return
        # Keep alpha/beta in float32: sin(x * exp(alpha)) is extremely sensitive to
        # argument precision — bfloat16 can't resolve the correct phase for |arg| > ~10,
        # producing audio-destroying noise.
        w_alpha = tl.alpha.detach().float().contiguous()
        w_beta = tl.beta.detach().float().contiguous()
        mesh_mapper = None
        if self.device is not None and hasattr(self.device, "get_num_devices") and self.device.get_num_devices() > 1:
            mesh_mapper = ttnn.ReplicateTensorToMesh(self.device)
        self.alpha = ttnn.from_torch(
            w_alpha,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            mesh_mapper=mesh_mapper,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.beta = ttnn.from_torch(
            w_beta,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            mesh_mapper=mesh_mapper,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def set_output_tensors_config_impl(self, output_tensors):
        """code2wav ``[B, C, T]``: avoid grid ``ConcatMesh2d`` on the last dim (breaks ``+ residual``).

        Do **not** use ``MeshComposerConfig([0, len(shape)])`` for rank-3 tensors: TTNN treats entries as
        dimension indices, so ``[0, 3]`` is invalid for 3D (dims are 0..2). Same pattern as
        ``TTNNQwenLayerNorm`` gather path: ``ConcatMeshToTensor(dim=0)`` + keep one batch replica.
        """
        if self.device_state is None or self.device is None or self.device.get_num_devices() <= 1:
            return super().set_output_tensors_config_impl(output_tensors)

        from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

        def _materialize_one_replica(t):
            if not isinstance(t, TorchTTNNTensor) or t.ttnn_tensor is None:
                return t
            tt = t.ttnn_tensor
            pt = _ttnn_mesh_to_torch_one_replica(tt, self.device)
            t.elem = pt.contiguous()
            t.ttnn_tensor = None
            if getattr(t, "_distributed_tensor_config", None) is not None:
                t._distributed_tensor_config = None
            return t

        return tree_map(_materialize_one_replica, output_tensors)

    @staticmethod
    def _snake_beta_chunk_t() -> int:
        """Max time length per SnakeBeta kernel to cap FP32 activation DRAM (code2wav BCT can be 100k+)."""
        raw = os.environ.get("TT_SYMBIOTE_SNAKEBETA_CHUNK_T", "4096")
        try:
            v = int(raw)
        except ValueError:
            v = 4096
        return max(512, v)

    def _forward_fp32_core(
        self,
        input_fp32: ttnn.Tensor,
        alpha_exp: ttnn.Tensor,
        reciprocal_beta: ttnn.Tensor,
    ) -> ttnn.Tensor:
        """``x + 1/b * sin^2(x*exp(alpha))`` in FP32; frees intermediate tensors before return."""
        x_times_alpha = ttnn.multiply(input_fp32, alpha_exp)
        sin_result = ttnn.sin(x_times_alpha)
        sin_squared = ttnn.pow(sin_result, 2.0)
        scaled_sin = ttnn.multiply(reciprocal_beta, sin_squared)
        result = ttnn.add(input_fp32, scaled_sin, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        for t in (x_times_alpha, sin_result, sin_squared, scaled_sin):
            try:
                ttnn.deallocate(t)
            except Exception:
                pass
        return result

    def forward(self, input_tensor: ttnn.Tensor) -> ttnn.Tensor:
        if input_tensor.layout != ttnn.TILE_LAYOUT:
            input_tensor = ttnn.to_layout(input_tensor, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        out_dtype = input_tensor.dtype
        shape = tuple(int(s) for s in input_tensor.shape)
        if len(shape) != 3:
            raise ValueError(f"TTNNSnakeBeta expects rank-3 [B,C,T], got shape {shape}")
        b, c, t_len = shape

        chunk_t = self._snake_beta_chunk_t()
        # Broadcast helpers (small): shared across time chunks.
        alpha_expanded = ttnn.unsqueeze(self.alpha, 0)
        alpha_expanded = ttnn.unsqueeze(alpha_expanded, -1)
        beta_expanded = ttnn.unsqueeze(self.beta, 0)
        beta_expanded = ttnn.unsqueeze(beta_expanded, -1)
        alpha_exp = ttnn.exp(alpha_expanded)
        try:
            ttnn.deallocate(alpha_expanded)
        except Exception:
            pass
        beta_exp = ttnn.exp(beta_expanded)
        beta_plus_eps = ttnn.add(beta_exp, self.no_div_by_zero)
        reciprocal_beta = ttnn.reciprocal(beta_plus_eps)
        try:
            ttnn.deallocate(beta_expanded)
            ttnn.deallocate(beta_exp)
            ttnn.deallocate(beta_plus_eps)
        except Exception:
            pass

        if t_len <= chunk_t:
            if out_dtype != ttnn.float32:
                input_fp32 = ttnn.typecast(input_tensor, ttnn.float32)
            else:
                input_fp32 = input_tensor
            result = self._forward_fp32_core(input_fp32, alpha_exp, reciprocal_beta)
            if out_dtype != ttnn.float32:
                if input_fp32 is not input_tensor:
                    try:
                        ttnn.deallocate(input_fp32)
                    except Exception:
                        pass
                # Do not deallocate ``result`` before/after typecast: some TTNN paths may alias buffers;
                # freeing FP32 activations here corrupted long code2wav outputs (truncated / bad audio).
                result = ttnn.typecast(result, out_dtype)
            try:
                ttnn.deallocate(alpha_exp)
                ttnn.deallocate(reciprocal_beta)
            except Exception:
                pass
            return result

        # Long sequences: chunk along T, then stitch. Device-side ``ttnn.concat`` over many chunks keeps
        # every chunk tensor alive until concat completes and allocates a full-length output — peak DRAM
        # can OOM on long code2wav streams (e.g. T~5e5). Stitch on host with ``torch.cat`` (same dtypes as
        # the per-chunk TTNN outputs), deallocate each chunk TTNN tensor, then one ``from_torch`` upload —
        # numerically identical to a single device concat, lower peak device memory.
        out_chunks = []
        for t0 in range(0, t_len, chunk_t):
            t1 = min(t0 + chunk_t, t_len)
            sl = ttnn.slice(input_tensor, (0, 0, t0), (b, c, t1))
            if out_dtype != ttnn.float32:
                sl_fp32 = ttnn.typecast(sl, ttnn.float32)
            else:
                sl_fp32 = sl
            res_fp32 = self._forward_fp32_core(sl_fp32, alpha_exp, reciprocal_beta)
            if out_dtype != ttnn.float32:
                out_chunks.append(ttnn.typecast(res_fp32, out_dtype))
            else:
                out_chunks.append(res_fp32)

        try:
            ttnn.deallocate(input_tensor)
        except Exception:
            pass

        torch_parts = []
        mesh_dev = self.device
        for ch in out_chunks:
            torch_parts.append(_ttnn_mesh_to_torch_one_replica(ch, mesh_dev))
            try:
                ttnn.deallocate(ch)
            except Exception:
                pass
        merged_torch = torch.cat(torch_parts, dim=2)
        try:
            ttnn.deallocate(alpha_exp)
            ttnn.deallocate(reciprocal_beta)
        except Exception:
            pass
        return _upload_bct_replicated(merged_torch, mesh_dev)


def _try_dealloc_ttnn(x):
    """Best-effort deallocate a TTNN tensor (or a TorchTTNNTensor wrapping one) to free device DRAM."""
    from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

    tt = None
    if isinstance(x, ttnn.Tensor):
        tt = x
    elif isinstance(x, TorchTTNNTensor):
        tt = getattr(x, "ttnn_tensor", None)
    if tt is not None:
        try:
            ttnn.deallocate(tt)
        except Exception:
            pass


def _ensure_code2wav_bct_full_t(out, mesh_device, expected_t: int):
    """Stitch width-sharded BCT conv output and re-upload as replicated.

    TTNN conv2d on a mesh can distribute spatial width across devices even for replicated
    input, producing ``[B, C, T_local]`` where ``T_local * nd == T_full``.  When
    ``shard_t == expected_t`` (replicated, each device already has full T) this is a no-op.
    Otherwise reads all device shards, cats on time, and re-uploads with
    ``ReplicateTensorToMesh``.
    """
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
    """One logical ``[B, C, T]`` on host.

    ``ConcatMeshToTensor(dim=0)`` + ``[:batch]`` can truncate the **time** dimension when mesh
    layout/shape metadata does not match a pure batch stack (symptom: shortened / corrupted wav).
    Reading ``get_device_tensors(...)[0]`` matches other models' replicated-activation readback.
    """
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
        from models.experimental.tt_symbiote.modules.conv import TTNNConv1d

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
        """If padded time length exceeds this, run HF ``nn.Conv1d`` on host (same math as HF forward).

        Very long code2wav streams (hundreds of k samples after upsampling) make a full-width TTNN
        conv2d/reshape exceed device DRAM; host ``conv1d`` matches ``Qwen3OmniMoeCausalConvNet`` exactly.
        Override with env ``TT_SYMBIOTE_CODE2WAV_CAUSAL_CONV_HOST_T`` (default ``8192``).
        """
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
        _try_dealloc_ttnn(hidden_state)
        extra_padding = self._get_extra_padding_for_conv1d(x_t)
        t_padded = int(x_t.shape[-1]) + self.padding + int(extra_padding)
        expected_t = (t_padded - self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1
        x_t = F.pad(x_t, (self.padding, extra_padding), mode="constant", value=0)
        t_pad = int(x_t.shape[-1])
        # Long streams: TTNN conv uploads full [B,C,T] and can OOM in conv2d/DRAM slice (e.g. T~5e5).
        # HF reference uses this same padded tensor → same nn.Conv1d weights as ``_fallback_torch_layer.conv``.
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
        from models.experimental.tt_symbiote.modules.conv import TTNNConvTranspose1d

        new = cls()
        new._fallback_torch_layer = m
        new.left_pad = int(m.left_pad)
        new.right_pad = int(m.right_pad)
        new.conv = TTNNConvTranspose1d.from_torch(m.conv)
        return new

    def set_output_tensors_config_impl(self, output_tensors):
        if self.conv is not None:
            return self.conv.set_output_tensors_config_impl(output_tensors)
        return super().set_output_tensors_config_impl(output_tensors)

    def forward(self, hidden_state):
        x_t = _materialize_code2wav_chain_output(hidden_state, self.device)
        _try_dealloc_ttnn(hidden_state)
        out = self.conv(x_t)
        if isinstance(out, ttnn.Tensor):
            y_t = _materialize_code2wav_bct_from_ttnn(out, self.device)
            try:
                ttnn.deallocate(out)
            except Exception:
                pass
        else:
            y_t = out
        t_out = int(y_t.shape[-1])
        end = t_out - self.right_pad
        y_t = y_t[..., self.left_pad : end].contiguous()
        dev = self.device
        mesh_mapper = None
        if dev is not None and hasattr(dev, "get_num_devices") and dev.get_num_devices() > 1:
            mesh_mapper = ttnn.ReplicateTensorToMesh(dev)
        return ttnn.from_torch(
            y_t,
            dtype=torch_dtype_to_ttnn_dtype(y_t.dtype),
            layout=ttnn.TILE_LAYOUT,
            device=dev,
            mesh_mapper=mesh_mapper,
        )


class TTNNQwen3OmniMoeConvNeXtBlock(TTNNModule):
    """Qwen3-Omni ConvNeXtBlock: TTNN depthwise conv, host norm/linear/residual.

    ``dwconv`` (depthwise conv) runs on TTNN via :class:`TTNNQwen3OmniMoeCausalConvNet` which
    stitches width-sharded output back to full ``T``.  LayerNorm, pointwise linears, GELU, and
    the residual add execute on host torch (cheap element-wise / small matmul) so that the
    shortcut and branch always match in time dimension.
    """

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

        prev = hidden_states
        hidden_states = self.dwconv(hidden_states)
        _try_dealloc_ttnn(prev)
        x = _materialize_code2wav_chain_output(hidden_states, dev)
        _try_dealloc_ttnn(hidden_states)

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
    """code2wav residual block: TTNN SnakeBeta + CausalConvNet, host-side residual add.

    :class:`TTNNQwen3OmniMoeCausalConvNet` stitches width-sharded conv output to full ``T``
    so the branch and shortcut match when added on host.
    """

    def __init__(self):
        super().__init__()
        self.act1 = None
        self.conv1 = None
        self.act2 = None
        self.conv2 = None

    @classmethod
    def from_torch(cls, m, *args, **kwargs):
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

        prev = hidden_state
        hidden_state = self.act1(hidden_state)
        _try_dealloc_ttnn(prev)

        prev = hidden_state
        hidden_state = self.conv1(hidden_state)
        _try_dealloc_ttnn(prev)

        prev = hidden_state
        hidden_state = self.act2(hidden_state)
        _try_dealloc_ttnn(prev)

        prev = hidden_state
        hidden_state = self.conv2(hidden_state)
        _try_dealloc_ttnn(prev)

        branch = _materialize_code2wav_chain_output(hidden_state, dev)
        _try_dealloc_ttnn(hidden_state)
        result = branch + residual
        return _upload_bct_replicated(result, dev)
