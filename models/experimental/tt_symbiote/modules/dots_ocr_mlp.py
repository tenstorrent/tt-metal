# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import math

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
    TTNNLinearLLamaIReplicatedWColSharded,
    TTNNLinearLLamaIColShardedWAllReduced,
    _decode_down_proj_dram_sharded_program_config,
    _decode_down_proj_input_memory_config,
    _decode_gate_up_dram_sharded_program_config,
    _decode_gate_up_col_dram_sharded_program_config,
    _decode_linear_output_memory_config,
    _l1_width_sharded_mem_config,
    _dp_matmul_program_config,
    _dram_sharded_mem_config_2d,
    _tp_requires_ccl,
    _tp_mesh_mapper,
    _tp_num_shards,
    _linear_mesh_num_devices,
    _ccl_num_links,
    _ccl_worker_kwargs,
)

# Decode column-parallel fused gate_up (M=32, K=1536, N=2*intermediate/TP), DRAM-
# width-sharded, weight-bandwidth-bound. The bandwidth-optimal compute-grid size
# depends on the WEIGHT DTYPE -- a heavier (BFP8) weight needs more readers to hit
# high DRAM utilization. Both grids divide k_tiles=48 so the per-core K-shard stays
# tile-aligned (else _decode_gate_up_col_dram_sharded_program_config returns None and
# the path silently falls back to a ~37 us generic mcast). Device-profiler sweep
# 2026-06-15 (prof_gate_up_col_decode.py, 32x1536x4480):
#   BFP4 weight: nc=12 ~21 us (one core/DRAM bank) < nc=16 23.5 < nc=8 25 < nc=24 30
#   BFP8 weight: nc=16 ~32 us (81% util) < nc=24 32.5 < nc=12 37.6 (69%) < nc=8 39.8
# 21 us is unreachable for BFP8 (7.3 MB weight / 288 GB/s peak => ~25 us floor); it
# is the BFP4 number. So bf4 layers use 12c, bf8 layers use 16c -- each its optimum.
_GATE_UP_COL_DRAM_SHARDED_GRID_BF4 = (6, 2)  # 12 cores
_GATE_UP_COL_DRAM_SHARDED_GRID_BF8 = (8, 2)  # 16 cores


def _gate_up_col_dram_grid(weight_dtype):
    """Bandwidth-optimal compute grid for the decode gate_up, by weight dtype."""
    return _GATE_UP_COL_DRAM_SHARDED_GRID_BF8 if weight_dtype == ttnn.bfloat8_b else _GATE_UP_COL_DRAM_SHARDED_GRID_BF4


class TTNNDotsOCRFusedGateUpRowSharded(TTNNLinearLLamaIColShardedWAllReducedFusedGateUp):
    def set_weight_dtype(self, dtype):
        self._weight_dtype = dtype
        return self

    def move_weights_to_device_impl(self):
        if not _tp_requires_ccl(self.device):
            return super().move_weights_to_device_impl()

        num_devices = _tp_num_shards(self.device)
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

        matmul_mc = (
            _decode_linear_output_memory_config(self.device, input_shape)
            if needs_ccl
            else (output_memory_config or ttnn.DRAM_MEMORY_CONFIG)
        )
        # Fuse bias into the matmul kernel on single-device (no CCL would scale
        # the bias by num_devices). Saves one BinaryNg per layer when bias is set.
        fused_bias = None if needs_ccl else self.tt_bias

        prefill_compute_config = (
            self._gate_up_decode_compute_kernel_config if not needs_ccl else self.compute_kernel_config
        )
        tt_output = ttnn.linear(
            input_tensor,
            self.tt_weight,
            bias=fused_bias,
            dtype=ttnn.bfloat8_b,
            memory_config=matmul_mc,
            compute_kernel_config=prefill_compute_config,
            program_config=_dp_matmul_program_config(self.device, input_shape, self.tt_weight.shape),
        )
        if needs_ccl:
            tt_output = ttnn.reduce_scatter(
                tt_output,
                dim=3,
                num_links=_ccl_num_links(self.device),
                cluster_axis=1,
                memory_config=output_memory_config or _decode_linear_output_memory_config(self.device, input_shape),
                topology=ttnn.Topology.Linear,
                **_ccl_worker_kwargs("reduce_scatter"),
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

        matmul_mc = (
            _decode_linear_output_memory_config(self.device, input_shape)
            if needs_ccl
            else (output_memory_config or ttnn.DRAM_MEMORY_CONFIG)
        )
        output_mc = output_memory_config or _decode_linear_output_memory_config(self.device, input_shape)

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
                memory_config=output_mc,
                topology=ttnn.Topology.Linear,
                **_ccl_worker_kwargs("reduce_scatter"),
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

        if is_decode and output.memory_config().buffer_type != ttnn.BufferType.L1:
            output = ttnn.to_memory_config(output, ttnn.L1_MEMORY_CONFIG)
        return output


class TTNNDotsOCRFusedGateUpColParallel(TTNNLinearLLamaIReplicatedWColSharded):
    """Fused gate+up for the column-parallel (Megatron) text SwiGLU.

    Single column-sharded matmul producing ``[*, 2*intermediate]`` (N-sharded
    across the TP axis), replacing the two separate gate/up matmuls in
    ``TTNNDotsOCRMLPColParallel``. The two halves are interleaved per TP device
    on the host so that the on-device N-shard each chip holds is exactly
    ``[gate_i | up_i]`` for its slice ``i`` of the intermediate dim. A local
    ``ttnn.chunk`` on the last dim then recovers the matching gate/up shards
    (mirrors the row-parallel ``TTNNDotsOCRFusedGateUpRowSharded`` interleave so
    the silu*mul shard lines up with the down_proj K-shard, no reshard).

    Replicated activation in, N-sharded out, NO collective (column-parallel).
    """

    @classmethod
    def from_two_torch(cls, gate_linear: torch.nn.Linear, up_linear: torch.nn.Linear):
        new_linear = cls(in_features=gate_linear.in_features, out_features=gate_linear.out_features * 2)
        new_linear._fallback_torch_layer = gate_linear
        new_linear._gate_weight_torch = gate_linear.weight
        new_linear._up_weight_torch = up_linear.weight
        new_linear._gate_bias_torch = gate_linear.bias if gate_linear.bias is not None else None
        new_linear._up_bias_torch = up_linear.bias if up_linear.bias is not None else None
        new_linear.weight = None
        new_linear.bias = None
        return new_linear

    def preprocess_weights_impl(self):
        self.tt_weight_host = None
        self.tt_bias_host = None

    def move_weights_to_device_impl(self):
        # Interleave gate/up by TP shard BEFORE the column-shard mapper runs, so
        # device ``i`` ends up with ``[gate_i | up_i]`` (and not gate-then-up,
        # which would split a single half across devices). num_tp is the TP
        # (last) mesh axis size; single-device collapses to a plain fuse.
        mesh_shape = list(self.device.shape) if hasattr(self.device, "shape") else [1, 1]
        num_tp = int(mesh_shape[-1]) if mesh_shape else 1
        intermediate = int(self._gate_weight_torch.shape[0])
        shard = intermediate // num_tp
        has_bias = self._gate_bias_torch is not None or self._up_bias_torch is not None
        weight_chunks = []
        bias_chunks = []
        for i in range(num_tp):
            start = i * shard
            end = (i + 1) * shard
            weight_chunks.extend([self._gate_weight_torch[start:end], self._up_weight_torch[start:end]])
            if has_bias:
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

        # Hand the fused [2*intermediate, hidden] weight to the base column-shard
        # loader, which transposes to [hidden, 2*intermediate] and shards the
        # last (N) dim across the TP axis -- yielding the per-device interleave.
        fused_weight_torch = torch.cat(weight_chunks, dim=0)
        fused_bias_torch = torch.cat(bias_chunks, dim=0) if bias_chunks else None
        self.tt_weight_host = fused_weight_torch
        self.tt_bias_host = fused_bias_torch
        super().move_weights_to_device_impl()

        self._build_col_gate_up_dram_sharded(fused_weight_torch, fused_bias_torch, num_tp)

    def _build_col_gate_up_dram_sharded(self, fused_weight_torch, fused_bias_torch, num_tp):
        self._gate_up_col_dram_pc = None
        self._gate_up_col_dram_weight = None
        self._gate_up_col_dram_bias = None
        self._gate_up_col_dram_input_cfg = None
        self._gate_up_col_half = None

        weight_dtype = getattr(self, "_weight_dtype", ttnn.bfloat4_b)
        # Compute grid sized to the weight dtype (heavier BFP8 weight -> more readers).
        col_grid = _gate_up_col_dram_grid(weight_dtype)
        col_num_cores = col_grid[0] * col_grid[1]
        # [2*intermediate, hidden] -> [hidden, 2*intermediate]; the mapper shards
        # the last (N) dim across the TP axis, so each chip holds [hidden, n_per_dev].
        weight_t = fused_weight_torch.T.contiguous()
        k = int(weight_t.shape[-2])
        n_global = int(weight_t.shape[-1])
        n_per_dev = math.ceil(n_global / num_tp)

        program_config = _decode_gate_up_col_dram_sharded_program_config(
            k_per_dev=k, n_per_dev=n_per_dev, num_cores=col_num_cores
        )
        if program_config is None:
            # k not tile-divisible across the compute grid; keep the interleaved path.
            return

        # weight_dim == -1 for this class: both the [hidden, 2*inter] weight and the
        # [1, 2*inter] bias are sharded on their last (N) dim across the TP axis.
        weight_mapper = _tp_mesh_mapper(self.device, self.weight_dim)
        bias_mapper = weight_mapper
        self._gate_up_col_dram_weight = ttnn.as_tensor(
            weight_t,
            device=self.device,
            dtype=weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=weight_mapper,
            memory_config=_dram_sharded_mem_config_2d(self.device, k=k, n=n_per_dev),
        )
        if fused_bias_torch is not None:
            bias_2d = fused_bias_torch.reshape((1, -1))
            bias_n_per_dev = math.ceil(int(bias_2d.shape[-1]) / num_tp)
            self._gate_up_col_dram_bias = ttnn.as_tensor(
                bias_2d,
                device=self.device,
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=bias_mapper,
                memory_config=_dram_sharded_mem_config_2d(self.device, k=ttnn.TILE_SIZE, n=bias_n_per_dev),
            )
        self._gate_up_col_dram_pc = program_config
        # Input is K-width-sharded across the SAME grid as the matmul's num_cores
        # (12 = 6x2), not the row-parallel 16-core (8x2) grid -- the DRAM-sharded
        # kernel reads each core's K-shard, so the input shard grid must match.
        self._gate_up_col_dram_input_cfg = _l1_width_sharded_mem_config(k=k, grid=col_grid)
        # Per-device logical width of each of gate_i / up_i (= intermediate/num_tp).
        # The DRAM-sharded matmul pads N up to num_cores*per_core_N tiles; the MLP
        # slices [0:half] and [half:2*half] to recover gate/up and drop the padding.
        self._gate_up_col_half = n_per_dev // 2
        self._gate_up_col_decode_ckc = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )

    @run_on_devices(*SHARDED_COLLECTIVE_LINEAR_DEVICE_ARCHS)
    def forward(self, input_tensor: ttnn.Tensor) -> ttnn.Tensor:
        if input_tensor.layout != ttnn.TILE_LAYOUT:
            input_tensor = ttnn.to_layout(input_tensor, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        input_tensor_shape = list(input_tensor.shape)
        input_shape = list(input_tensor_shape)
        while len(input_shape) < 4:
            input_shape.insert(1, 1)

        program_config = getattr(self, "_gate_up_col_dram_pc", None)
        is_decode = int(input_shape[-2]) <= ttnn.TILE_SIZE
        if is_decode and program_config is not None:
            input_4d = ttnn.reshape(input_tensor, input_shape)
            # Downcast the activation to BFP8 *inside* the reshard that already runs
            # (the dtype= arg), so there is no separate typecast op. The DRAM-sharded
            # matmul is weight-bound, so a standalone typecast would be pure decode
            # overhead; folding it into the existing to_memory_config is free. BFP8
            # activation is accuracy-neutral here (output text stays byte-identical).
            input_4d = ttnn.to_memory_config(input_4d, self._gate_up_col_dram_input_cfg, dtype=ttnn.bfloat8_b)
            tt_output = ttnn.linear(
                input_4d,
                self._gate_up_col_dram_weight,
                bias=self._gate_up_col_dram_bias,
                dtype=ttnn.bfloat8_b,
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                compute_kernel_config=self._gate_up_col_decode_ckc,
                program_config=program_config,
            )
            return ttnn.reshape(tt_output, input_tensor_shape[:-1] + [-1])

        if is_decode:
            input_4d = ttnn.reshape(input_tensor, input_shape)
            tt_output = ttnn.linear(
                input_4d,
                self.tt_weight,
                bias=self.tt_bias,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                compute_kernel_config=self.compute_kernel_config,
                program_config=_dp_matmul_program_config(self.device, input_shape, self.tt_weight.shape),
            )
            return ttnn.reshape(tt_output, input_tensor_shape[:-1] + [-1])

        # Prefill / fallback: base column-sharded interleaved matmul (no padding).
        return super().forward(input_tensor)


class TTNNDotsOCRMLPColParallel(TTNNModule):
    """Column-parallel (Megatron) text SwiGLU: REPLICATED in -> REPLICATED out.

    A/B alternative to the default row-parallel ``TTNNDotsOCRMLP`` (kept side by
    side for perf comparison). gate/up are column-parallel (weight N-sharded,
    activation replicated, NO collective); down is row-parallel and all-reduces
    (reduce_scatter + all_gather) so the output returns replicated. The gate/up
    column shards (intermediate/num_tp) line up exactly with the down K-shard, so
    no reshard sits between the silu*mul and down.

    Trade vs the row-parallel default: no wide reduce_scatter on the
    ``[*, 2*intermediate]`` gate_up partial, but the activation is replicated
    in/out (no hidden-dim sharding savings) and the down all-reduce is
    reduce_scatter + all_gather on the ``[*, hidden]`` output (2 collectives)
    rather than the row-parallel path's 2 reduce_scatters.

    NOTE: ``down_proj`` (the QKV-style all-reduced class) keeps its weight in
    ``bfloat8_b``; the row-parallel default down runs ``bfloat4_b``. Account for
    that when comparing absolute down-matmul time.
    """

    @classmethod
    def from_torch(cls, torch_mlp):
        tt_module = cls()
        tt_module._fallback_torch_layer = torch_mlp
        tt_module.intermediate_size = torch_mlp.gate_proj.out_features
        tt_module.gate_proj = TTNNLinearLLamaIReplicatedWColSharded.from_torch(torch_mlp.gate_proj)
        tt_module.up_proj = TTNNLinearLLamaIReplicatedWColSharded.from_torch(torch_mlp.up_proj)
        tt_module.down_proj = TTNNLinearLLamaIColShardedWAllReduced.from_torch(torch_mlp.down_proj)
        return tt_module

    def set_weight_dtype(self, dtype):
        self.gate_proj.set_weight_dtype(dtype)
        self.up_proj.set_weight_dtype(dtype)
        # down is the QKV-style all-reduced class; it hardcodes bfloat8_b and has
        # no set_weight_dtype, so only forward the dtype when supported.
        if hasattr(self.down_proj, "set_weight_dtype"):
            self.down_proj.set_weight_dtype(dtype)
        return self

    def forward(self, hidden_states: ttnn.Tensor) -> ttnn.Tensor:
        if hidden_states.layout != ttnn.TILE_LAYOUT:
            hidden_states = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)

        # Column-parallel gate/up: replicated activation in, N-sharded out, no CCL.
        gate = self.gate_proj(hidden_states)
        up = self.up_proj(hidden_states)
        gate_up_mul = ttnn.mul(
            gate,
            up,
            input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
            fast_and_approximate_mode=True,
            dtype=ttnn.bfloat8_b,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(gate)
        ttnn.deallocate(up)

        # Row-parallel down: K-sharded in -> reduce_scatter + all_gather -> replicated.
        return self.down_proj(gate_up_mul)


class TTNNDotsOCRMLPColParallelFusedGateUp(TTNNModule):
    """Column-parallel text SwiGLU with FUSED gate+up.

    Fused-gate-up counterpart of ``TTNNDotsOCRMLPColParallel``. Instead of two
    separate column-parallel matmuls (gate and up), a single column-sharded
    matmul emits ``[*, 2*intermediate]`` (N-sharded), which is then split locally
    on each device into its matching ``[gate_i | up_i]`` shard via ``ttnn.chunk``.

    Two output contracts (the down all-reduce is the only CCL in this MLP, and a
    TP=4 sweep showed it costs ~55 us = ~1/3 of the decode forward):

    * ``replicated_output=True`` (default): down is row-parallel + all-reduce
      (reduce_scatter + all_gather) -> REPLICATED full-hidden out. Safe drop-in
      contract (replicated in -> replicated out).
    * ``replicated_output=False``: down does reduce_scatter ONLY (no all_gather,
      via ``TTNNDotsOCRRowShardedNoAllGather``) -> hidden-SHARDED out. This is the
      only CCL win available on the dots_ocr TP=4 stack (Ring topology is
      unroutable on the (1,4) line and num_links>1 deadlocks): it drops the
      all_gather, measured at -14.8% decode MLP wall-time (164 -> 140 us/token,
      CCL 55 -> 31 us) at identical PCC. The caller MUST then consume a
      hidden-sharded residual (same contract the row-parallel default produces).
    """

    @classmethod
    def from_torch(cls, torch_mlp, replicated_output: bool = True):
        tt_module = cls()
        tt_module._fallback_torch_layer = torch_mlp
        tt_module.intermediate_size = torch_mlp.gate_proj.out_features
        tt_module.replicated_output = replicated_output
        tt_module.fused_gate_up_proj = TTNNDotsOCRFusedGateUpColParallel.from_two_torch(
            torch_mlp.gate_proj, torch_mlp.up_proj
        )
        # Kept for introspection parity with the unfused col variant; unused in forward.
        tt_module.gate_proj = None
        tt_module.up_proj = None
        if replicated_output:
            tt_module.down_proj = TTNNLinearLLamaIColShardedWAllReduced.from_torch(torch_mlp.down_proj)
        else:
            # reduce_scatter-only down -> hidden-sharded output (drops the all_gather).
            tt_module.down_proj = TTNNDotsOCRRowShardedNoAllGather.from_torch(torch_mlp.down_proj)
        return tt_module

    def set_weight_dtype(self, dtype):
        self.fused_gate_up_proj.set_weight_dtype(dtype)
        # The all-reduced down (replicated path) hardcodes bfloat8_b and has no
        # set_weight_dtype; the reduce_scatter-only down does. Forward only when supported.
        if hasattr(self.down_proj, "set_weight_dtype"):
            self.down_proj.set_weight_dtype(dtype)
        return self

    def forward(self, hidden_states: ttnn.Tensor) -> ttnn.Tensor:
        if hidden_states.layout != ttnn.TILE_LAYOUT:
            hidden_states = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)

        # Fused column-parallel gate/up: replicated activation in, 2N-sharded out, no CCL.
        gate_up = self.fused_gate_up_proj(hidden_states)

        # The decode DRAM-sharded gate_up lands L1 width-sharded with N padded up to
        # 16*per_core_N tiles. ``ttnn.chunk`` would split at the PADDED midpoint and
        # corrupt the gate/up boundary, so sharded_to_interleaved first, then slice
        # the two halves by their logical per-device width (dropping the padding).
        half = getattr(self.fused_gate_up_proj, "_gate_up_col_half", None)
        if gate_up.memory_config().is_sharded():
            gate_up = ttnn.sharded_to_interleaved(gate_up, ttnn.L1_MEMORY_CONFIG)
        if half is not None:
            # Per-device the shard is [gate_i | up_i], each ``half`` wide; the matmul
            # may have padded a few zero tiles past 2*half, which these slices drop.
            shape = list(gate_up.shape)
            rank = len(shape)
            gate = ttnn.slice(gate_up, [0] * rank, shape[:-1] + [half])
            up = ttnn.slice(gate_up, [0] * (rank - 1) + [half], shape[:-1] + [2 * half])
        else:
            # Interleaved fallback (prefill / non-DRAM-sharded): exact 2*half width.
            gate, up = ttnn.chunk(gate_up, 2, dim=-1)
        ttnn.deallocate(gate_up)

        # Decode keeps the silu*mul result (down_proj's in0) L1-resident: it is a
        # tiny [32, intermediate/TP] tensor (~70 KB BFP8), and feeding the down
        # 1D-mcast matmul from L1 avoids the DRAM write + DRAM read the profiler
        # flagged ("place input 0 in L1"). Prefill stays DRAM (large activation).
        # Mirrors the row-parallel TTNNDotsOCRMLP decode path.
        is_decode = int(hidden_states.shape[1]) == 1
        mul_mc = ttnn.L1_MEMORY_CONFIG if is_decode else ttnn.DRAM_MEMORY_CONFIG
        gate_up_mul = ttnn.mul(
            gate,
            up,
            input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
            fast_and_approximate_mode=True,
            dtype=ttnn.bfloat8_b,
            memory_config=mul_mc,
        )
        ttnn.deallocate(gate)
        ttnn.deallocate(up)

        # Down (K-sharded in): all-reduce -> replicated, or reduce_scatter-only ->
        # hidden-sharded, depending on ``replicated_output``.
        output = self.down_proj(gate_up_mul)
        ttnn.deallocate(gate_up_mul)
        if int(hidden_states.shape[1]) == 1 and output.memory_config().buffer_type != ttnn.BufferType.L1:
            output = ttnn.to_memory_config(output, ttnn.L1_MEMORY_CONFIG)
        return output
