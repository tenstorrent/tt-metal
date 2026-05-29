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
    _decode_down_proj_dram_sharded_program_config,
    _decode_down_proj_input_memory_config,
    _decode_gate_up_dram_sharded_program_config,
    _dp_matmul_program_config,
    _dram_sharded_mem_config_2d,
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
            return super().move_weights_to_device_impl()

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

        # Single-device DRAM-sharded fast path for decode shapes only (prefill
        # falls back to the existing 2D-mcast path with DRAM_INTERLEAVED weight,
        # because the prefill kernel cannot consume DRAM_WIDTH_SHARDED operand B).
        # Input must be L1 width-sharded 16c 8x2 (matches the sharded RMSNorm
        # output of post_attention_layernorm — no reshard needed). Output lands
        # L1 width-sharded on the same 16c 8x2 grid, BFP8.
        dram_shard_cfg = getattr(self, "_gate_up_dram_input_shard_cfg", None)
        dram_pc = (
            _decode_gate_up_dram_sharded_program_config(input_shape, self.tt_weight.shape)
            if (not needs_ccl and dram_shard_cfg is not None)
            else None
        )
        if dram_pc is not None and dram_shard_cfg is not None:
            dram_weight, dram_bias = self._get_gate_up_dram_sharded_weight()
            input_tensor = ttnn.to_memory_config(input_tensor, dram_shard_cfg)
            tt_output = ttnn.linear(
                input_tensor,
                dram_weight,
                bias=dram_bias,
                dtype=ttnn.bfloat8_b,
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                compute_kernel_config=self._gate_up_decode_compute_kernel_config,
                program_config=dram_pc,
            )
            return ttnn.reshape(tt_output, input_tensor_shape[:-1] + [-1])

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

    def _down_proj_use_dram_sharded(self) -> bool:
        # Pattern-match dots.ocr down_proj (K=8960, N=1536) on single-device.
        if _tp_requires_ccl(self.device):
            return False
        return int(self.in_features) == 8960 and int(self.out_features) == 1536

    def move_weights_to_device_impl(self):
        weight_dtype = getattr(self, "_weight_dtype", ttnn.bfloat4_b)
        use_dram_sharded = self._down_proj_use_dram_sharded()
        # Stash raw torch weight/bias BEFORE preprocess overwrites them, so we
        # can construct the second DRAM_WIDTH_SHARDED copy via ``as_tensor``
        # (the o_proj-style path that doesn't trigger a 12-core reshard kernel).
        raw_weight_torch = (
            self.tt_weight_host.clone() if use_dram_sharded and isinstance(self.tt_weight_host, torch.Tensor) else None
        )
        raw_bias_torch = (
            self.tt_bias_host.clone() if use_dram_sharded and isinstance(self.tt_bias_host, torch.Tensor) else None
        )
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
        # Second weight (DRAM_WIDTH_SHARDED) for the decode DRAM-sharded matmul.
        # Allocated via ``as_tensor`` so no reshard kernel is launched. Prefill
        # uses ``self.tt_weight`` (DRAM_INTERLEAVED). Memory cost: ~7 MB / layer
        # for down_proj (8960x1536 BFP4; N=1536 is exactly 12*32*4 so no padding).
        self._down_proj_dram_input_shard_cfg = (
            _decode_down_proj_input_memory_config(self.in_features) if use_dram_sharded else None
        )
        self._down_proj_dram_weight = None
        self._down_proj_dram_bias = None
        if use_dram_sharded and raw_weight_torch is not None:
            weight_t = raw_weight_torch.T.contiguous()
            self._down_proj_dram_weight = ttnn.as_tensor(
                weight_t,
                device=self.device,
                dtype=weight_dtype,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=_tp_mesh_mapper(self.device, self.weight_dim),
                memory_config=_dram_sharded_mem_config_2d(
                    self.device, k=int(weight_t.shape[-2]), n=int(weight_t.shape[-1])
                ),
            )
            if raw_bias_torch is not None:
                bias_2d = raw_bias_torch.reshape((1, -1))
                self._down_proj_dram_bias = ttnn.as_tensor(
                    bias_2d,
                    device=self.device,
                    dtype=ttnn.bfloat8_b,
                    layout=ttnn.TILE_LAYOUT,
                    mesh_mapper=_tp_mesh_mapper(self.device, self.input_dim),
                    memory_config=_dram_sharded_mem_config_2d(self.device, k=ttnn.TILE_SIZE, n=int(bias_2d.shape[-1])),
                )

    def _get_down_proj_dram_sharded_weight(self):
        return self._down_proj_dram_weight, self._down_proj_dram_bias

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

        # Decode fast path: DRAM-sharded 8c (8x1 grid) matmul with
        # ``DRAM_WIDTH_SHARDED`` weight + L1 width-sharded I/O (44us
        # standalone). Caller is expected to hand us an 8c L1 width-sharded
        # input (MLP does one I2S on the silu_mul output before calling
        # ``down_proj``); if it didn't, ``to_memory_config`` does the I2S
        # here as a defensive fallback.
        dram_shard_cfg = getattr(self, "_down_proj_dram_input_shard_cfg", None)
        dram_pc = (
            _decode_down_proj_dram_sharded_program_config(input_shape, self.tt_weight.shape)
            if (not needs_ccl and dram_shard_cfg is not None)
            else None
        )
        if dram_pc is not None and dram_shard_cfg is not None:
            dram_weight, dram_bias = self._get_down_proj_dram_sharded_weight()
            input_tensor = ttnn.to_memory_config(input_tensor, dram_shard_cfg)
            tt_output = ttnn.linear(
                input_tensor,
                dram_weight,
                bias=dram_bias if dram_bias is not None else fused_bias,
                dtype=ttnn.bfloat8_b,
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                compute_kernel_config=self.compute_kernel_config,
                program_config=dram_pc,
            )
            return ttnn.reshape(tt_output, input_tensor_shape[:-1] + [-1])

        matmul_mc = ttnn.DRAM_MEMORY_CONFIG if needs_ccl else (output_memory_config or ttnn.DRAM_MEMORY_CONFIG)

        # Prefill down_proj (K=8960, N=1536): ttnn.matmul's auto-heuristic picks a degenerate
        # config for this huge-K / small-N shape (~16 ms, ~5% compute, ~0.2% BW). minimal_matmul
        # is the production prefill path (cf. models/tt_transformers/tt/mlp.py:276 and
        # model_config.py:1318-1323). Single-device only; the CCL (TP) path keeps ttnn.linear +
        # reduce_scatter, and decode uses the DRAM-sharded path above.
        if not needs_ccl and int(input_shape[-2]) > 1:
            ff2_mm_config = ttnn.MinimalMatmulConfig(
                M_block_size=8,
                K_block_size=8,
                N_block_size=8,
                compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
            )
            tt_output = ttnn.experimental.minimal_matmul(
                input_tensor,
                self.tt_weight,
                bias_tensor=fused_bias,
                config=ff2_mm_config,
                memory_config=matmul_mc,
                dtype=ttnn.bfloat8_b,
                compute_kernel_config=self.compute_kernel_config,
            )
            return ttnn.reshape(tt_output, input_tensor_shape[:-1] + [-1])

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

        seq_len = hidden_states.shape[1]
        is_decode = int(seq_len) == 1
        activation_mc = ttnn.L1_MEMORY_CONFIG if is_decode else ttnn.DRAM_MEMORY_CONFIG

        gate_up = self.fused_gate_up_proj(hidden_states, output_memory_config=activation_mc if is_decode else None)

        # In decode the fused gate-up matmul now lands in L1_WIDTH_SHARDED
        # (16c 8x2 BFP8). Chunk on a sharded TILE tensor falls into the
        # ``rm_only`` path of ``ttnn.slice`` (untilize+slice+tilize per half,
        # ~40us). Single brief ``sharded_to_interleaved`` so the chunk stays
        # on its cheap tile-aware path (~5us combined).
        gate_up_was_sharded = gate_up.memory_config().is_sharded()
        if gate_up_was_sharded:
            gate_up = ttnn.sharded_to_interleaved(gate_up, ttnn.L1_MEMORY_CONFIG)
        elif is_decode and gate_up.memory_config().buffer_type != ttnn.BufferType.L1:
            gate_up = ttnn.to_memory_config(gate_up, activation_mc)

        # Split the fused gate/up activation in half along the last dim.
        # ``ttnn.chunk`` calls ``ttnn.slice`` per half; for L1_INTERLEAVED
        # TILE input with tile-aligned begins this stays on the tile-aware
        # path (no rm_only untilize/tilize).
        gate, up = ttnn.chunk(gate_up, 2, dim=-1)
        ttnn.deallocate(gate_up)

        # silu_mul stays on L1 interleaved (cheap multiply, no presharding of
        # gate/up). We then I2S the silu_mul OUTPUT once, into the 8c L1
        # width-sharded layout that the DRAM-sharded down_proj expects. That
        # is a single I2S per layer (vs the two pre-silu_mul I2S reshards
        # the "preshard gate+up" variant needed) and lets down_proj engage
        # its faster DRAM-sharded 8c kernel (~44us, vs the mcast1d 8x3
        # interleaved-I/O variant's ~52us).
        down_dram_shard_cfg = getattr(self.down_proj, "_down_proj_dram_input_shard_cfg", None) if is_decode else None

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

        if down_dram_shard_cfg is not None:
            gate_up_mul = ttnn.to_memory_config(gate_up_mul, down_dram_shard_cfg)

        down_output_mc = activation_mc if (is_decode and down_dram_shard_cfg is None) else None
        output = self.down_proj(gate_up_mul, output_memory_config=down_output_mc)
        ttnn.deallocate(gate_up_mul)

        return output
