# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import math

import torch
import ttnn
from ttnn.model_preprocessing import preprocess_linear_bias
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
    _dram_sharded_mem_config_2d,
)


_GATE_UP_GRID = (8, 7)
_GATE_UP_INPUT_SHARD_CORES = 16
_GATE_UP_IN0_BLOCK_W = 3
_GATE_UP_PER_CORE_M = 1
_GATE_UP_PER_CORE_N = 10
_GATE_UP_OUT_SUBBLOCK_W = 5

_DOWN_PROJ_NUM_CORES = 8
_DOWN_PROJ_IN0_BLOCK_W = 7
_DOWN_PROJ_PER_CORE_M = 1
_DOWN_PROJ_PER_CORE_N = 6


def _gate_up_input_memory_config(k: int):
    input_grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(_GATE_UP_INPUT_SHARD_CORES // 2 - 1, 1))]
    )
    return ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        buffer_type=ttnn.BufferType.L1,
        shard_spec=ttnn.ShardSpec(
            input_grid,
            [ttnn.TILE_SIZE, k // _GATE_UP_INPUT_SHARD_CORES],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )


def _gate_up_output_memory_config():
    output_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))])
    return ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        buffer_type=ttnn.BufferType.L1,
        shard_spec=ttnn.ShardSpec(
            output_grid,
            [ttnn.TILE_SIZE, ttnn.TILE_SIZE * _GATE_UP_PER_CORE_N],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )


def _width_sharded_l1_memory_config(num_cores: int, shard_shape):
    shard_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores - 1, 0))])
    return ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        buffer_type=ttnn.BufferType.L1,
        shard_spec=ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR),
    )


def _down_proj_input_memory_config(k: int):
    return _width_sharded_l1_memory_config(
        _DOWN_PROJ_NUM_CORES,
        [ttnn.TILE_SIZE, k // _DOWN_PROJ_NUM_CORES],
    )


def _down_proj_output_memory_config():
    return _width_sharded_l1_memory_config(
        _DOWN_PROJ_NUM_CORES,
        [ttnn.TILE_SIZE, ttnn.TILE_SIZE * _DOWN_PROJ_PER_CORE_N],
    )


def _down_proj_dram_sharded_program_config(input_shape, weight_shape):
    if int(input_shape[-2]) > ttnn.TILE_SIZE:
        return None
    if int(input_shape[-1]) != 8960 or int(weight_shape[-2]) != 8960 or int(weight_shape[-1]) != 1536:
        return None

    return ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=_DOWN_PROJ_IN0_BLOCK_W,
        per_core_M=_DOWN_PROJ_PER_CORE_M,
        per_core_N=_DOWN_PROJ_PER_CORE_N,
        fused_activation=None,
    )


def _gate_up_matmul_program_config(input_shape, weight_shape):
    if int(input_shape[-2]) > ttnn.TILE_SIZE:
        return None
    if int(input_shape[-1]) != 1536 or int(weight_shape[-2]) != 1536 or int(weight_shape[-1]) != 17920:
        return None

    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=_GATE_UP_GRID,
        in0_block_w=_GATE_UP_IN0_BLOCK_W,
        out_subblock_h=1,
        out_subblock_w=_GATE_UP_OUT_SUBBLOCK_W,
        out_block_h=_GATE_UP_PER_CORE_M,
        out_block_w=_GATE_UP_PER_CORE_N,
        per_core_M=_GATE_UP_PER_CORE_M,
        per_core_N=_GATE_UP_PER_CORE_N,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=False,
    )


class TTNNDotsOCRFusedGateUpRowSharded(TTNNLinearLLamaIColShardedWAllReducedFusedGateUp):
    def set_weight_dtype(self, dtype):
        self._weight_dtype = dtype
        return self

    def move_weights_to_device_impl(self):
        weight_dtype = getattr(self, "_weight_dtype", ttnn.bfloat4_b)
        weight_mapper = _tp_mesh_mapper(self.device, self.weight_dim)
        needs_tp_ccl = _tp_requires_ccl(self.device)
        num_devices = self.device.get_num_devices() if hasattr(self.device, "get_num_devices") else 1
        intermediate = self._gate_weight_torch.shape[0]
        shard = intermediate // num_devices if needs_tp_ccl else intermediate
        weight_chunks = []
        bias_chunks = []
        for i in range(num_devices if needs_tp_ccl else 1):
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
        weight_t = weight.T.contiguous()
        mesh_shape = list(self.device.shape) if hasattr(self.device, "shape") else [1, 1]
        num_tp = int(mesh_shape[-1]) if mesh_shape else 1
        k_per_device = math.ceil(int(weight_t.shape[-2]) / num_tp) if needs_tp_ccl else int(weight_t.shape[-2])
        self.tt_weight = ttnn.as_tensor(
            weight_t,
            device=self.device,
            dtype=weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=weight_mapper,
            memory_config=_dram_sharded_mem_config_2d(self.device, k=k_per_device, n=int(weight_t.shape[-1])),
        )
        self.tt_weight_prefill = ttnn.as_tensor(
            weight_t,
            device=self.device,
            dtype=weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=weight_mapper,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

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
        program_config = _gate_up_matmul_program_config(input_shape, self.tt_weight.shape)
        uses_sharded_matmul = program_config is not None
        if uses_sharded_matmul:
            input_tensor = ttnn.to_memory_config(input_tensor, _gate_up_input_memory_config(int(input_shape[-1])))
            matmul_mc = _gate_up_output_memory_config()
            weight = self.tt_weight
        else:
            program_config = _dp_matmul_program_config(self.device, input_shape, self.tt_weight.shape)
            matmul_mc = ttnn.DRAM_MEMORY_CONFIG if needs_ccl else (output_memory_config or ttnn.DRAM_MEMORY_CONFIG)
            weight = self.tt_weight_prefill
            if input_tensor.is_sharded():
                input_tensor = ttnn.sharded_to_interleaved(input_tensor, ttnn.DRAM_MEMORY_CONFIG)
        # Fuse bias into the matmul kernel on single-device (no CCL would scale
        # the bias by num_devices). Saves one BinaryNg per layer when bias is set.
        fused_bias = None if (needs_ccl or uses_sharded_matmul) else self.tt_bias
        tt_output = ttnn.linear(
            input_tensor,
            weight,
            bias=fused_bias,
            dtype=ttnn.bfloat8_b,
            memory_config=matmul_mc,
            compute_kernel_config=self.compute_kernel_config,
            program_config=program_config,
        )
        if uses_sharded_matmul and not needs_ccl and self.tt_bias is not None:
            tt_output = ttnn.sharded_to_interleaved(tt_output, ttnn.DRAM_MEMORY_CONFIG)
            tt_output += self.tt_bias
        if needs_ccl:
            if uses_sharded_matmul:
                tt_output = ttnn.sharded_to_interleaved(tt_output, ttnn.DRAM_MEMORY_CONFIG)
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
            weight = self.tt_weight_host.T.contiguous()
            mesh_shape = list(self.device.shape) if hasattr(self.device, "shape") else [1, 1]
            num_tp = int(mesh_shape[-1]) if mesh_shape else 1
            needs_tp_ccl = _tp_requires_ccl(self.device)
            k_per_device = math.ceil(int(weight.shape[-2]) / num_tp) if needs_tp_ccl else int(weight.shape[-2])
            self.tt_weight = ttnn.as_tensor(
                weight,
                device=self.device,
                dtype=weight_dtype,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=_tp_mesh_mapper(self.device, self.weight_dim),
                memory_config=_dram_sharded_mem_config_2d(self.device, k=k_per_device, n=int(weight.shape[-1])),
            )
        else:
            self.tt_weight = ttnn.to_device(self.tt_weight_host, self.device)
        if isinstance(self.tt_bias_host, torch.Tensor):
            self.tt_bias_host = preprocess_linear_bias(
                self.tt_bias_host,
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                weights_mesh_mapper=_tp_mesh_mapper(self.device, self.input_dim),
            )
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
        program_config = _down_proj_dram_sharded_program_config(input_shape, self.tt_weight.shape)
        uses_sharded_matmul = program_config is not None
        if uses_sharded_matmul:
            input_tensor = ttnn.to_memory_config(input_tensor, _down_proj_input_memory_config(int(input_shape[-1])))
            matmul_mc = _down_proj_output_memory_config()
        else:
            program_config = _dp_matmul_program_config(self.device, input_shape, self.tt_weight.shape)
            matmul_mc = ttnn.DRAM_MEMORY_CONFIG if needs_ccl else (output_memory_config or ttnn.DRAM_MEMORY_CONFIG)
        fused_bias = None if (needs_ccl or uses_sharded_matmul) else self.tt_bias
        tt_output = ttnn.linear(
            input_tensor,
            self.tt_weight,
            bias=fused_bias,
            dtype=ttnn.bfloat8_b,
            memory_config=matmul_mc,
            compute_kernel_config=self.compute_kernel_config,
            program_config=program_config,
        )
        if uses_sharded_matmul and not needs_ccl and self.tt_bias is not None:
            tt_output = ttnn.sharded_to_interleaved(tt_output, ttnn.DRAM_MEMORY_CONFIG)
            tt_output += self.tt_bias
        if needs_ccl:
            if uses_sharded_matmul:
                tt_output = ttnn.sharded_to_interleaved(tt_output, ttnn.DRAM_MEMORY_CONFIG)
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
