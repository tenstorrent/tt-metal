# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from ttnn.model_preprocessing import preprocess_linear_bias, preprocess_linear_weight
from models.experimental.tt_symbiote.core.module import (
    TTNNModule,
    run_on_devices,
    SHARDED_COLLECTIVE_LINEAR_DEVICE_ARCHS,
)
from models.experimental.tt_symbiote.modules.linear import (
    TTNNLinearLLamaIColShardedWAllReducedFusedGateUp,
    TTNNLinearLLamaIColShardedWRowSharded,
    _dp_matmul_program_config,
    _tp_requires_ccl,
    _tp_mesh_mapper,
    _linear_mesh_num_devices,
    _ccl_num_links,
)


class TTNNDotsOCRFusedGateUpRowSharded(TTNNLinearLLamaIColShardedWAllReducedFusedGateUp):
    def set_weight_dtype(self, dtype):
        self._weight_dtype = dtype
        return self

    def move_weights_to_device_impl(self):
        if not _tp_requires_ccl(self.device):
            weight = torch.cat([self._gate_weight_torch, self._up_weight_torch], dim=0)
            self.tt_weight_host = preprocess_linear_weight(
                weight,
                dtype=ttnn.bfloat4_b,
                layout=ttnn.TILE_LAYOUT,
                weights_mesh_mapper=_tp_mesh_mapper(self.device, self.weight_dim),
            )
            self.tt_weight = ttnn.to_device(self.tt_weight_host, self.device)

            if self._gate_bias_torch is not None or self._up_bias_torch is not None:
                intermediate = self._gate_weight_torch.shape[0]
                zeros_dtype = self._gate_weight_torch.dtype
                gate_bias = (
                    self._gate_bias_torch
                    if self._gate_bias_torch is not None
                    else torch.zeros(intermediate, dtype=zeros_dtype)
                )
                up_bias = (
                    self._up_bias_torch
                    if self._up_bias_torch is not None
                    else torch.zeros(intermediate, dtype=zeros_dtype)
                )
                bias = torch.cat([gate_bias, up_bias], dim=0)
                self.tt_bias_host = preprocess_linear_bias(
                    bias,
                    dtype=ttnn.bfloat8_b,
                    layout=ttnn.TILE_LAYOUT,
                    weights_mesh_mapper=_tp_mesh_mapper(self.device, self.input_dim),
                )
                self.tt_bias = ttnn.to_device(self.tt_bias_host, self.device)
            else:
                self.tt_bias = None

            self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.LoFi,
                math_approx_mode=False,
                fp32_dest_acc_en=False,
                packer_l1_acc=True,
            )
            return

        num_devices = self.device.get_num_devices() if hasattr(self.device, "get_num_devices") else 1
        intermediate = self._gate_weight_torch.shape[0]
        shard = intermediate // num_devices
        weight_chunks = []
        bias_chunks = []
        for i in range(num_devices):
            start = i * shard
            end = (i + 1) * shard
            weight_chunks.extend([self._gate_weight_torch[start:end], self._up_weight_torch[start:end]])
            if self._gate_bias_torch is not None or self._up_bias_torch is not None:
                dtype = self._gate_weight_torch.dtype
                gate_bias = (
                    self._gate_bias_torch[start:end]
                    if self._gate_bias_torch is not None
                    else torch.zeros(shard, dtype=dtype)
                )
                up_bias = (
                    self._up_bias_torch[start:end]
                    if self._up_bias_torch is not None
                    else torch.zeros(shard, dtype=dtype)
                )
                bias_chunks.extend([gate_bias, up_bias])

        weight = torch.cat(weight_chunks, dim=0)
        self.tt_weight_host = preprocess_linear_weight(
            weight,
            dtype=getattr(self, "_weight_dtype", ttnn.bfloat4_b),
            layout=ttnn.TILE_LAYOUT,
            weights_mesh_mapper=_tp_mesh_mapper(self.device, self.weight_dim),
        )
        self.tt_weight = ttnn.to_device(self.tt_weight_host, self.device)

        if bias_chunks:
            bias = torch.cat(bias_chunks, dim=0)
            self.tt_bias_host = preprocess_linear_bias(
                bias,
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                weights_mesh_mapper=_tp_mesh_mapper(self.device, self.input_dim),
            )
            self.tt_bias = ttnn.to_device(self.tt_bias_host, self.device)
        else:
            self.tt_bias = None

        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )

    @run_on_devices(*SHARDED_COLLECTIVE_LINEAR_DEVICE_ARCHS)
    def forward(self, input_tensor: ttnn.Tensor, output_memory_config=None) -> ttnn.Tensor:
        if input_tensor.layout != ttnn.TILE_LAYOUT:
            input_tensor = ttnn.to_layout(input_tensor, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        input_tensor_shape = list(input_tensor.shape)
        input_shape = list(input_tensor_shape)
        if len(input_shape) == 2:
            input_shape.insert(0, 1)
        if len(input_shape) == 3:
            input_shape.insert(1, 1)
        input_tensor = ttnn.reshape(input_tensor, input_shape)
        needs_ccl = _linear_mesh_num_devices(self.device) > 1 and _tp_requires_ccl(self.device)
        matmul_mc = ttnn.DRAM_MEMORY_CONFIG if needs_ccl else (output_memory_config or ttnn.DRAM_MEMORY_CONFIG)
        # Fuse bias into the matmul kernel on single-device (no CCL would scale
        # the bias by num_devices). Saves one BinaryNg per layer when bias is set.
        fused_bias = None if needs_ccl else self.tt_bias
        tt_output = ttnn.linear(
            input_tensor,
            self.tt_weight,
            bias=fused_bias,
            dtype=ttnn.bfloat8_b,
            memory_config=matmul_mc,
            compute_kernel_config=self.compute_kernel_config,
            program_config=_dp_matmul_program_config(self.device, input_shape, self.tt_weight.shape),
        )
        if needs_ccl:
            tt_output = ttnn.reduce_scatter(
                tt_output,
                dim=3,
                num_links=_ccl_num_links(self.device),
                cluster_axis=1,
                memory_config=output_memory_config or ttnn.DRAM_MEMORY_CONFIG,
                topology=ttnn.Topology.Linear,
            )
            if self.tt_bias is not None:
                tt_output += self.tt_bias
        return ttnn.reshape(tt_output, input_tensor_shape[:-1] + [-1])


class TTNNDotsOCRRowShardedNoAllGather(TTNNLinearLLamaIColShardedWRowSharded):
    def set_weight_dtype(self, dtype):
        self._weight_dtype = dtype
        return self

    def move_weights_to_device_impl(self):
        weight_dtype = getattr(self, "_weight_dtype", ttnn.bfloat4_b)
        if isinstance(self.tt_weight_host, torch.Tensor):
            self.tt_weight_host = preprocess_linear_weight(
                self.tt_weight_host,
                dtype=weight_dtype,
                layout=ttnn.TILE_LAYOUT,
                weights_mesh_mapper=_tp_mesh_mapper(self.device, self.weight_dim),
            )
        if isinstance(self.tt_bias_host, torch.Tensor):
            self.tt_bias_host = preprocess_linear_bias(
                self.tt_bias_host,
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                weights_mesh_mapper=_tp_mesh_mapper(self.device, self.input_dim),
            )
        self.tt_weight = ttnn.to_device(self.tt_weight_host, self.device)
        self.tt_bias = ttnn.to_device(self.tt_bias_host, self.device) if self.tt_bias_host is not None else None
        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )

    @run_on_devices(*SHARDED_COLLECTIVE_LINEAR_DEVICE_ARCHS)
    def forward(self, input_tensor: ttnn.Tensor, output_memory_config=None) -> ttnn.Tensor:
        if input_tensor.layout != ttnn.TILE_LAYOUT:
            input_tensor = ttnn.to_layout(input_tensor, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        input_tensor_shape = list(input_tensor.shape)
        input_shape = list(input_tensor_shape)
        while len(input_shape) < 4:
            input_shape.insert(1, 1)
        input_tensor = ttnn.reshape(input_tensor, input_shape)
        needs_ccl = _linear_mesh_num_devices(self.device) > 1 and _tp_requires_ccl(self.device)
        fused_bias = None if needs_ccl else self.tt_bias
        matmul_mc = ttnn.DRAM_MEMORY_CONFIG if needs_ccl else (output_memory_config or ttnn.DRAM_MEMORY_CONFIG)
        tt_output = ttnn.linear(
            input_tensor,
            self.tt_weight,
            bias=fused_bias,
            dtype=ttnn.bfloat8_b,
            memory_config=matmul_mc,
            compute_kernel_config=self.compute_kernel_config,
            program_config=_dp_matmul_program_config(self.device, input_shape, self.tt_weight.shape),
        )
        if needs_ccl:
            tt_output = ttnn.reduce_scatter(
                tt_output,
                dim=3,
                num_links=_ccl_num_links(self.device),
                cluster_axis=1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                topology=ttnn.Topology.Linear,
            )
            if self.tt_bias is not None:
                tt_output += self.tt_bias
        return ttnn.reshape(tt_output, input_tensor_shape[:-1] + [-1])


class TTNNDotsOCRMLP(TTNNModule):
    """SwiGLU MLP with fused gate+up projection.

    Original (two separate matmuls + two CCL all-reduces):
        gate = gate_proj(x)              # matmul + reduce_scatter + all_gather
        up   = up_proj(x)                # matmul + reduce_scatter + all_gather
        out  = down_proj(silu(gate) * up)

    Fused (one matmul + one CCL all-reduce, mirrors gemma4_mlp.py):
        gate_up = fused_gate_up_proj(x)  # matmul (2x out) + reduce_scatter + all_gather
        gate    = gate_up[..., :I]
        up      = gate_up[..., I:]
        out     = down_proj(silu(gate) * up)

    Fusing halves the number of CCL launches in MLP. Each all-reduce on N300 has
    ~120 us of latency overhead independent of payload size, so combining the
    two avoids that overhead. The matmul itself becomes 2x larger but the FLOPs
    are identical to the two unfused matmuls; bandwidth-bound decode therefore
    breaks roughly even on raw matmul time and saves the CCL latency outright.
    """

    @classmethod
    def from_torch(cls, torch_mlp):
        tt_module = cls()
        tt_module._fallback_torch_layer = torch_mlp

        # Fuse gate_proj + up_proj into a single column-sharded all-reduced linear.
        tt_module.intermediate_size = torch_mlp.gate_proj.out_features

        tt_module.fused_gate_up_proj = TTNNDotsOCRFusedGateUpRowSharded.from_two_torch(
            torch_mlp.gate_proj, torch_mlp.up_proj
        )
        # Keep individual proj refs for compatibility with anything that
        # introspects the module; they are no longer used in forward.
        tt_module.gate_proj = None
        tt_module.up_proj = None

        tt_module.down_proj = TTNNDotsOCRRowShardedNoAllGather.from_torch(torch_mlp.down_proj)
        return tt_module

    def set_weight_dtype(self, dtype):
        self.fused_gate_up_proj.set_weight_dtype(dtype)
        self.down_proj.set_weight_dtype(dtype)
        return self

    def forward(self, hidden_states: ttnn.Tensor) -> ttnn.Tensor:
        if hidden_states.layout != ttnn.TILE_LAYOUT:
            hidden_states = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)

        batch_size = hidden_states.shape[0]
        seq_len = hidden_states.shape[1]
        is_decode = int(seq_len) == 1
        activation_mc = ttnn.L1_MEMORY_CONFIG if is_decode else ttnn.DRAM_MEMORY_CONFIG
        if is_decode and hidden_states.memory_config().buffer_type != ttnn.BufferType.L1:
            hidden_states = ttnn.to_memory_config(hidden_states, activation_mc)

        gate_up = self.fused_gate_up_proj(hidden_states, output_memory_config=activation_mc if is_decode else None)

        if is_decode and gate_up.memory_config().buffer_type != ttnn.BufferType.L1:
            gate_up = ttnn.to_memory_config(gate_up, activation_mc)

        # Slice into gate / up halves along the last dim. ttnn.slice on a TILE
        # tensor is a metadata + small copy op, far cheaper than a second CCL.
        I = int(gate_up.shape[-1]) // 2
        gate = ttnn.slice(gate_up, [0, 0, 0], [batch_size, seq_len, I], memory_config=activation_mc)
        up = ttnn.slice(gate_up, [0, 0, I], [batch_size, seq_len, 2 * I], memory_config=activation_mc)
        ttnn.deallocate(gate_up)

        # ``fast_and_approximate_mode=True`` routes the fused SILU through
        # the polynomial exp/sigmoid path. SILU dominates this op (the
        # elementwise multiply is cheap; the per-tile exp+sigmoid is the
        # bottleneck), so the approx path measurably cuts the BinaryNg time
        # in both decode and prefill MLP per layer.
        gate_up_mul = ttnn.mul(
            gate,
            up,
            input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
            fast_and_approximate_mode=True,
            dtype=ttnn.bfloat8_b,
            memory_config=activation_mc,
        )
        ttnn.deallocate(gate)
        ttnn.deallocate(up)

        down_output_mc = activation_mc if is_decode else None
        output = self.down_proj(gate_up_mul, output_memory_config=down_output_mc)
        ttnn.deallocate(gate_up_mul)

        return output
