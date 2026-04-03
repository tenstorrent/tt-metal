# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Activation function implementations for TTNN."""

import torch

import ttnn
from models.experimental.tt_symbiote.core.module import TTNNModule
from models.experimental.tt_symbiote.core.utils import tree_map


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
        w_alpha = tl.alpha.detach().to(torch.bfloat16).contiguous()
        w_beta = tl.beta.detach().to(torch.bfloat16).contiguous()
        mesh_mapper = None
        if self.device is not None and hasattr(self.device, "get_num_devices") and self.device.get_num_devices() > 1:
            mesh_mapper = ttnn.ReplicateTensorToMesh(self.device)
        self.alpha = ttnn.from_torch(
            w_alpha,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            mesh_mapper=mesh_mapper,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.beta = ttnn.from_torch(
            w_beta,
            dtype=ttnn.bfloat16,
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
            b = int(tt.shape[0])
            pt = ttnn.to_torch(tt, mesh_composer=ttnn.ConcatMeshToTensor(self.device, dim=0))
            if pt.shape[0] > b:
                pt = pt[:b]
            t.elem = pt.contiguous()
            t.ttnn_tensor = None
            if getattr(t, "_distributed_tensor_config", None) is not None:
                t._distributed_tensor_config = None
            return t

        return tree_map(_materialize_one_replica, output_tensors)

    def forward(self, input_tensor: ttnn.Tensor) -> ttnn.Tensor:
        if input_tensor.layout != ttnn.TILE_LAYOUT:
            input_tensor = ttnn.to_layout(input_tensor, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        alpha_expanded = ttnn.unsqueeze(self.alpha, 0)
        alpha_expanded = ttnn.unsqueeze(alpha_expanded, -1)
        beta_expanded = ttnn.unsqueeze(self.beta, 0)
        beta_expanded = ttnn.unsqueeze(beta_expanded, -1)

        alpha_exp = ttnn.exp(alpha_expanded)
        beta_exp = ttnn.exp(beta_expanded)

        x_times_alpha = ttnn.multiply(input_tensor, alpha_exp)
        sin_result = ttnn.sin(x_times_alpha)
        sin_squared = ttnn.pow(sin_result, 2.0)
        beta_plus_eps = ttnn.add(beta_exp, self.no_div_by_zero)
        reciprocal_beta = ttnn.reciprocal(beta_plus_eps)
        scaled_sin = ttnn.multiply(reciprocal_beta, sin_squared)
        return ttnn.add(input_tensor, scaled_sin, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def _materialize_code2wav_bct_from_ttnn(tt_tensor: ttnn.Tensor, mesh_device) -> torch.Tensor:
    """One logical ``[B, C, T]`` batch on host (replicated mesh → concat batch, keep first replica)."""
    if mesh_device is None or mesh_device.get_num_devices() <= 1:
        return ttnn.to_torch(tt_tensor).contiguous()
    b = int(tt_tensor.shape[0])
    pt = ttnn.to_torch(
        tt_tensor,
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0),
        device=mesh_device,
    )
    if pt.shape[0] > b:
        pt = pt[:b]
    return pt.contiguous()


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


class TTNNQwen3OmniMoeCode2WavDecoderResidualUnit(TTNNModule):
    """code2wav residual block: materialize shortcut and branch output so ``+`` sees matching ``[B,C,T]`` on mesh."""

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
        new.conv1 = m.conv1
        new.act2 = TTNNSnakeBeta.from_torch(m.act2)
        new.conv2 = m.conv2
        return new

    def forward(self, hidden_state):
        dev = self.device
        # TTNN children of a TTNN parent default to bypass=True, so ``module_run`` skips
        # ``post_process_ttnn_module_output``. SnakeBeta then returns a raw ``ttnn.Tensor``,
        # which HF ``Qwen3OmniMoeCausalConvNet`` passes to ``F.pad`` → TypeError (torch pad
        # expects ``torch.Tensor``). Disable bypass so act outputs are wrapped / materialized
        # like top-level code2wav SnakeBeta blocks.
        if self.act1 is not None:
            self.act1._bypass_tensor_wrapping = False
        if self.act2 is not None:
            self.act2._bypass_tensor_wrapping = False
        residual = _materialize_code2wav_bct_from_ttnn(hidden_state, dev)
        hidden_state = self.act1(hidden_state)
        hidden_state = self.conv1(hidden_state)
        hidden_state = self.act2(hidden_state)
        hidden_state = self.conv2(hidden_state)
        hidden_state = _materialize_code2wav_chain_output(hidden_state, dev)
        return hidden_state + residual
