# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Linear layer implementations for TTNN."""

import math

from torch import nn
import torch
from ttnn.model_preprocessing import preprocess_linear_bias, preprocess_linear_weight
import ttnn
from models.experimental.tt_symbiote.core.module import (
    TTNNModule,
    deallocate_weights_after,
    run_on_devices,
    SHARDED_COLLECTIVE_LINEAR_DEVICE_ARCHS,
)
from models.experimental.tt_symbiote.core.run_config import trace_disabled, trace_enabled


def _tp_mesh_mapper(device, dim):
    if (
        hasattr(device, "get_num_devices")
        and device.get_num_devices() > 1
        and hasattr(device, "shape")
        and list(device.shape)[-1] == 1
    ):
        return ttnn.ShardTensor2dMesh(device, dims=(None, dim), mesh_shape=list(device.shape))
    return ttnn.shard_tensor_to_mesh_mapper(device, dim=dim)


def _tp_requires_ccl(device):
    return not (hasattr(device, "shape") and list(device.shape)[-1] == 1)


def _ccl_num_links(device) -> int:
    """Number of ethernet links to use for reduce_scatter / all_gather.

    Pinned to 1 for now: the dots_ocr decode trace re-uses the same CCL
    semaphores across reduce_scatter and all_gather inside a single layer,
    and the multi-link path on this CCL stack reorders completions which
    triggers ``Event Order Issue`` (expected event N but got M). The
    tt_transformers / gemma3 paths that use ``num_links=2`` go through the
    newer global-semaphore CCL APIs (``tt_ccl.line_*``), which dots_ocr does
    not. Until we migrate to that path, stay on 1 link to keep correctness.
    """
    return 1


def _largest_divisor_at_most(value: int, limit: int) -> int:
    for candidate in range(min(value, limit), 0, -1):
        if value % candidate == 0:
            return candidate
    return 1


def _out_subblock_w(per_core_n: int, out_subblock_h: int = 1) -> int:
    """Pick out_subblock_w that divides per_core_n and fits DST budget.

    Capped at 4 because some callers wrap the matmul with
    ``fp32_dest_acc_en=True`` compute configs (DST budget = 4 tiles).
    Pushing to 8 trips the
    ``out_subblock_w * out_subblock_h <= available_reg_count`` assertion
    in matmul_device_operation.cpp:1010 for those callers.
    """
    for candidate in range(4 // out_subblock_h, 0, -1):
        if per_core_n % candidate == 0:
            return candidate
    return 1


def _dp_prefill_matmul_program_config(device, input_shape, weight_shape):
    seq_len = int(input_shape[2])
    if seq_len <= 32:
        return None

    grid = device.compute_with_storage_grid_size()
    grid_x, grid_y = int(grid.x), int(grid.y)
    m_dim = int(input_shape[0]) * int(input_shape[1]) * seq_len
    k_dim = int(weight_shape[-2])
    n_dim = int(weight_shape[-1])

    tile = 32
    k_tiles = math.ceil(k_dim / tile)
    per_core_m = math.ceil(m_dim / (tile * grid_y))
    per_core_n = math.ceil(n_dim / (tile * grid_x))

    if per_core_n > 24:
        return None

    if k_tiles % grid_y == 0:
        k_tiles_per_grid_row = k_tiles // grid_y
        in0_block_w = _largest_divisor_at_most(k_tiles_per_grid_row, 8)
    else:
        in0_block_w = 2 if k_tiles % 2 == 0 else 1

    out_subblock_h = 1
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(grid_x, grid_y),
        in0_block_w=in0_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=_out_subblock_w(per_core_n, out_subblock_h),
        per_core_M=per_core_m,
        per_core_N=per_core_n,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=False,
    )


def _dp_decode_matmul_program_config(device, input_shape, weight_shape):
    """Decode-mode (M==1 tile) matmul program config.

    Decode shape M=32 (one tile after rounding to TILE_SIZE) is bandwidth-bound
    on weight reads. The default ttnn auto-config selects ``in0_block_w=1`` for
    these matmuls (the perf-trace summary explicitly flags this as suboptimal:
    "in0_block_w=1 is small, try in0_block_w=2 or above"), so we set an
    explicit ``MatmulMultiCoreReuseMultiCast1DProgramConfig`` with
    ``per_core_M=1`` (only one M tile in decode), ``mcast_in0=True`` to share
    the small input across all cores, and the largest divisor of K_tiles up to
    4 for ``in0_block_w``. Same shape pattern as
    ``qwen_moe._make_sparse_matmul_program_config``, which is the proven
    decode-mode config for DRAM-interleaved input + weights in tt-symbiote.

    Multi-device CCL safe: the matmul writes to ``DRAM_INTERLEAVED`` regardless
    of the program_config (program_config tunes matmul kernel internals only),
    so reduce_scatter / all_gather see the same input layout either way.
    """
    seq_len = int(input_shape[-2])
    if seq_len > 32:
        return None  # not a decode shape

    grid = device.compute_with_storage_grid_size()
    grid_x, grid_y = int(grid.x), int(grid.y)

    k_dim = int(weight_shape[-2])
    n_dim = int(weight_shape[-1])

    tile = ttnn.TILE_SIZE
    k_tiles = math.ceil(k_dim / tile)
    n_tiles = math.ceil(n_dim / tile)

    num_cores = max(1, grid_x * grid_y)
    per_core_n = max(1, math.ceil(n_tiles / num_cores))

    # For very large N (e.g. lm_head N=151936 → per_core_n≈75) the per-core
    # L1 footprint of 1D-mcast becomes tight (output tiles + weight cache +
    # circular buffers) and the default ttnn config is typically as fast or
    # faster. Bail out and let the auto-config handle it. Tracy dispatch
    # ablation: lifting this to per_core_n>96 to cover lm_head regressed
    # decode wall-clock by +22% in dispatch mode (9.82 s -> 12.06 s for
    # 180 tokens) -- the per-core 75-tile output buffer thrashes L1 enough
    # to lose vs the auto-config.
    if per_core_n > 32:
        return None

    # Cap in0_block_w so the per-core L1 weight cache stays within ~256 KB.
    # Each weight tile is ~1 KB for BFP8 / ~2 KB for BF16; using BFP8 lower
    # bound here keeps us safe even on huge-N matmuls.
    weight_tile_bytes = 1024
    max_l1_weight_bytes = 256 * 1024
    max_in0_block_w_for_l1 = max(1, max_l1_weight_bytes // (per_core_n * weight_tile_bytes))
    in0_block_w_cap = min(4, max_in0_block_w_for_l1)
    in0_block_w = _largest_divisor_at_most(k_tiles, in0_block_w_cap)

    out_subblock_h = 1
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(grid_x, grid_y),
        in0_block_w=in0_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=_out_subblock_w(per_core_n, out_subblock_h),
        per_core_M=1,
        per_core_N=per_core_n,
        fuse_batch=False,
        fused_activation=None,
        mcast_in0=True,
    )


def _dp_matmul_program_config(device, input_shape, weight_shape):
    """Pick decode vs prefill matmul program config based on input seq length.

    Decode-shape matmuls (M<=32) get an explicit 1D-mcast config tuned for
    ``per_core_M=1``; prefill-shape matmuls reuse the existing 2D program
    config helper. Both work for single-device and multi-device (CCL) paths
    — the matmul kernel writes to DRAM regardless of program_config, so
    downstream reduce_scatter / all_gather see the same tensor layout.
    """
    seq_len = int(input_shape[-2])
    if seq_len <= 32:
        return _dp_decode_matmul_program_config(device, input_shape, weight_shape)
    return _dp_prefill_matmul_program_config(device, input_shape, weight_shape)


def _decode_linear_output_memory_config(device, input_shape):
    if _tp_requires_ccl(device):
        return ttnn.DRAM_MEMORY_CONFIG
    return ttnn.L1_MEMORY_CONFIG if int(input_shape[-2]) <= 32 else ttnn.DRAM_MEMORY_CONFIG


def _linear_mesh_num_devices(device) -> int:
    """Rank count on the active mesh. Single-device meshes cannot use fabric CCLs."""
    if device is None or not hasattr(device, "get_num_devices"):
        return 1
    return int(device.get_num_devices())


@trace_enabled
class TTNNLinear(TTNNModule):
    """TTNN-accelerated linear layer."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

    @classmethod
    def from_parameters(cls, weight, bias=None):
        """Create TTNNLinear from a weight parameter."""
        new_linear = cls(
            in_features=weight.shape[1],
            out_features=weight.shape[0],
        )
        new_linear.weight = weight
        new_linear.bias = bias
        new_linear.preprocess_weights()
        del new_linear.weight
        del new_linear.bias
        return new_linear

    @classmethod
    def from_torch(cls, linear: nn.Linear):
        """Create TTNNLinear from PyTorch Linear layer."""
        new_linear = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
        )
        new_linear._fallback_torch_layer = linear
        new_linear.weight = linear.weight
        new_linear.bias = linear.bias
        return new_linear

    @property
    def _parameters(self):
        return self.torch_layer._parameters

    def preprocess_weights_impl(self):
        """Preprocess linear weights for TTNN."""
        self.tt_weight_host = preprocess_linear_weight(self.weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        self.tt_bias_host = None
        if self.bias is not None:
            self.tt_bias_host = preprocess_linear_bias(self.bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    def move_weights_to_device_impl(self):
        """Move weights to TTNN device."""
        self.tt_weight = ttnn.to_device(self.tt_weight_host, self.device)
        self.tt_bias = ttnn.to_device(self.tt_bias_host, self.device) if self.tt_bias_host is not None else None

    def deallocate_weights_impl(self):
        """Deallocate weights from device."""
        ttnn.deallocate(self.tt_weight)
        if self.tt_bias is not None:
            ttnn.deallocate(self.tt_bias)
        super().deallocate_weights_impl()

    def forward(self, input_tensor: ttnn.Tensor) -> ttnn.Tensor:
        """Forward pass through linear layer."""
        if input_tensor.layout != ttnn.TILE_LAYOUT:
            input_tensor = ttnn.to_layout(input_tensor, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        input_tensor_shape = list(input_tensor.shape)
        input_shape = list(input_tensor_shape)
        while len(input_shape) < 4:
            input_shape.insert(1, 1)  # Add batch dimensions if needed
        input_tensor = ttnn.reshape(input_tensor, input_shape)
        tt_output = ttnn.linear(input_tensor, self.tt_weight, bias=self.tt_bias, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        tt_output = ttnn.reshape(tt_output, input_tensor_shape[:-1] + [self.out_features])
        return tt_output


class TTNNLinearInputShardedWeightSharded(TTNNLinear):
    """TTNN-accelerated linear layer."""

    def __init__(self, in_features, out_features, input_dim, weight_dim) -> None:
        super().__init__(in_features, out_features)
        self.input_dim = input_dim
        self.weight_dim = weight_dim
        assert (
            self.input_dim == -1
        ), f"Only input sharding on second to last dimension is supported, got {self.input_dim}."
        assert self.weight_dim == -2, f"Only weight sharding on last dimension is supported, got {self.weight_dim}."

    def preprocess_weights_impl(self):
        self.tt_bias_host = self.bias
        self.tt_weight_host = self.weight

    def move_weights_to_device_impl(self):
        if isinstance(self.tt_weight_host, torch.Tensor):
            self.tt_weight_host = preprocess_linear_weight(
                self.tt_weight_host,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                weights_mesh_mapper=_tp_mesh_mapper(self.device, self.weight_dim),
            )
        if isinstance(self.tt_bias_host, torch.Tensor):
            self.tt_bias_host = preprocess_linear_bias(
                self.tt_bias_host,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                weights_mesh_mapper=_tp_mesh_mapper(self.device, self.input_dim),
            )
        self.tt_weight = ttnn.to_device(self.tt_weight_host, self.device)
        self.tt_bias = ttnn.to_device(self.tt_bias_host, self.device) if self.tt_bias_host is not None else None
        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )


class TTNNLinearIColShardedWRowSharded(TTNNLinearInputShardedWeightSharded):
    """TTNN-accelerated linear layer with input and weight sharded on last dimension."""

    def __init__(self, in_features, out_features) -> None:
        super().__init__(in_features, out_features, input_dim=-1, weight_dim=-2)

    @run_on_devices(*SHARDED_COLLECTIVE_LINEAR_DEVICE_ARCHS)
    def forward(self, input_tensor: ttnn.Tensor) -> ttnn.Tensor:
        """Forward pass through linear layer.

        On single-device (no CCL), bias is folded directly into the matmul
        kernel via ``ttnn.linear(bias=...)`` to eliminate the post-matmul
        BinaryNg add. With CCL we must keep bias post-reduce_scatter so the
        bias is not summed across devices.
        """
        if input_tensor.layout != ttnn.TILE_LAYOUT:
            input_tensor = ttnn.to_layout(input_tensor, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        input_tensor_shape = list(input_tensor.shape)
        input_shape = list(input_tensor_shape)
        while len(input_shape) < 4:
            input_shape.insert(1, 1)  # Add batch dimensions if needed
        input_tensor = ttnn.reshape(input_tensor, input_shape)
        needs_ccl = _linear_mesh_num_devices(self.device) > 1 and _tp_requires_ccl(self.device)
        fused_bias = None if needs_ccl else self.tt_bias
        tt_output = ttnn.linear(
            input_tensor,
            self.tt_weight,
            bias=fused_bias,
            memory_config=_decode_linear_output_memory_config(self.device, input_shape),
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
        tt_output = ttnn.reshape(tt_output, input_tensor_shape[:-1] + [-1])
        return tt_output


class TTNNLinearIColShardedWAllReduced(TTNNLinearIColShardedWRowSharded):
    @run_on_devices(*SHARDED_COLLECTIVE_LINEAR_DEVICE_ARCHS)
    def forward(self, input_tensor: ttnn.Tensor) -> ttnn.Tensor:
        """Forward pass: matmul + all_reduce.

        The input is column-sharded across devices. After matmul each device
        holds a partial sum.  all_reduce sums the partials so every device
        gets the full output (replicated).

        On single-device (no CCL), bias is fused into the matmul kernel via
        ``ttnn.linear(bias=...)`` to remove a separate device op. With CCL,
        the bias must be added AFTER the all-reduce so it is not summed
        ``num_devices`` times.
        """

        if input_tensor.layout != ttnn.TILE_LAYOUT:
            input_tensor = ttnn.to_layout(input_tensor, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        input_tensor_shape = list(input_tensor.shape)
        input_shape = list(input_tensor_shape)
        if len(input_shape) == 2:
            input_shape.insert(0, 1)  # Add batch dimension if missing
        if len(input_shape) == 3:
            input_shape.insert(1, 1)  # Add batch dimensions if needed
        input_tensor = ttnn.reshape(input_tensor, input_shape)

        needs_ccl = _linear_mesh_num_devices(self.device) > 1 and _tp_requires_ccl(self.device)
        # ``_bias_fused_into_matmul`` is set in move_weights_to_device_impl when
        # bias is prepared as replicated/divided so it can be fused into the
        # matmul (post-RS, the divided contributions sum back to the full
        # bias on each N-shard). For single-device, bias is already fused.
        bias_fused = bool(getattr(self, "_bias_fused_into_matmul", False))
        fused_bias = self.tt_bias if (not needs_ccl) or bias_fused else None
        tt_output = ttnn.linear(
            input_tensor,
            self.tt_weight,
            bias=fused_bias,
            memory_config=_decode_linear_output_memory_config(self.device, input_shape),
            compute_kernel_config=self.compute_kernel_config,
            program_config=_dp_matmul_program_config(self.device, input_shape, self.tt_weight.shape),
        )
        # Decompose all_reduce into reduce_scatter + all_gather for trace compatibility.
        # ttnn.all_reduce internally allocates an intermediate buffer dynamically, which
        # is incompatible with TTNN trace capture (requires stable buffer addresses).
        if needs_ccl:
            num_links = _ccl_num_links(self.device)
            tt_output = ttnn.reduce_scatter(
                tt_output,
                dim=3,
                num_links=num_links,
                cluster_axis=1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                topology=ttnn.Topology.Linear,
            )
            # Fallback: if bias was not fused (e.g. legacy sharded mapper
            # path), add it here while the tensor is still N-sharded.
            if self.tt_bias is not None and not bias_fused:
                tt_output += self.tt_bias
            tt_output = ttnn.all_gather(
                tt_output,
                dim=3,
                num_links=num_links,
                cluster_axis=1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                topology=ttnn.Topology.Linear,
            )

        tt_output = ttnn.reshape(tt_output, input_tensor_shape[:-1] + [-1])
        return tt_output


@trace_disabled
class TTNNLinearLLama(TTNNLinear):
    """TTNN Linear layer optimized for LLaMA models using bfloat8."""

    def preprocess_weights_impl(self):
        """Preprocess linear weights with bfloat8 precision."""
        self.tt_weight_host = preprocess_linear_weight(self.weight, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
        self.tt_bias_host = None
        if self.bias is not None:
            self.tt_bias_host = preprocess_linear_bias(self.bias, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)

    @deallocate_weights_after
    def forward(self, input_tensor: ttnn.Tensor) -> ttnn.Tensor:
        """Forward pass with automatic weight deallocation."""
        return super().forward(input_tensor)


@trace_disabled
class TTNNLinearLLamaIColShardedWRowSharded(TTNNLinearIColShardedWRowSharded):
    """TTNN Linear layer optimized for LLaMA models using bfloat8."""

    def move_weights_to_device_impl(self):
        if isinstance(self.tt_weight_host, torch.Tensor):
            self.tt_weight_host = preprocess_linear_weight(
                self.tt_weight_host,
                dtype=ttnn.bfloat8_b,
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

    @deallocate_weights_after
    @run_on_devices(*SHARDED_COLLECTIVE_LINEAR_DEVICE_ARCHS)
    def forward(self, input_tensor: ttnn.Tensor) -> ttnn.Tensor:
        """Forward pass with automatic weight deallocation."""
        return super().forward(input_tensor)


class TTNNLinearInputReplicatedWeightSharded(TTNNLinear):
    """TTNN-accelerated linear layer."""

    def __init__(self, in_features, out_features, weight_dim) -> None:
        super().__init__(in_features, out_features)
        self.weight_dim = weight_dim
        assert self.weight_dim == -1, f"Only weight sharding on last dimension is supported, got {self.weight_dim}."

    def preprocess_weights_impl(self):
        self.tt_bias_host = self.bias
        self.tt_weight_host = self.weight

    def move_weights_to_device_impl(self):
        if isinstance(self.tt_weight_host, torch.Tensor):
            self.tt_weight_host = preprocess_linear_weight(
                self.tt_weight_host,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                weights_mesh_mapper=_tp_mesh_mapper(self.device, self.weight_dim),
            )
        if isinstance(self.tt_bias_host, torch.Tensor):
            self.tt_bias_host = preprocess_linear_bias(
                self.tt_bias_host,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                weights_mesh_mapper=_tp_mesh_mapper(self.device, self.weight_dim),
            )
        self.tt_weight = ttnn.to_device(self.tt_weight_host, self.device)
        self.tt_bias = ttnn.to_device(self.tt_bias_host, self.device) if self.tt_bias_host is not None else None
        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )


class TTNNLinearIReplicatedWColSharded(TTNNLinearInputReplicatedWeightSharded):
    """TTNN-accelerated linear layer with input and weight sharded on last dimension."""

    def __init__(self, in_features, out_features) -> None:
        super().__init__(in_features, out_features, weight_dim=-1)

    @run_on_devices(*SHARDED_COLLECTIVE_LINEAR_DEVICE_ARCHS)
    def forward(self, input_tensor: ttnn.Tensor) -> ttnn.Tensor:
        """Forward pass through linear layer.

        Bias is fused into the matmul via ``ttnn.linear(bias=...)``. This is safe
        for IReplicatedWColSharded because there is no CCL on the matmul output
        (each device produces its own column slice independently), so adding the
        bias inside the matmul kernel is mathematically identical to a separate
        post-matmul add but saves one device op per call. (For the column-
        sharded all-reduced variants the bias *must* stay post-CCL — fusing
        would cause the bias to be summed num_devices times by the all-reduce.)

        Both ``MatmulMultiCoreReuseMultiCast1DProgramConfig`` (decode) and
        ``MatmulMultiCoreReuseMultiCastProgramConfig`` (prefill) accept a fused
        ``bias`` tensor in the ttnn matmul kernel, so we always pass the bias.
        """
        if input_tensor.layout != ttnn.TILE_LAYOUT:
            input_tensor = ttnn.to_layout(input_tensor, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        input_tensor_shape = list(input_tensor.shape)
        input_shape = list(input_tensor_shape)
        while len(input_shape) < 4:
            input_shape.insert(1, 1)  # Add batch dimensions if needed
        input_tensor = ttnn.reshape(input_tensor, input_shape)
        program_config = _dp_matmul_program_config(self.device, input_shape, self.tt_weight.shape)
        tt_output = ttnn.linear(
            input_tensor,
            self.tt_weight,
            bias=self.tt_bias,
            memory_config=_decode_linear_output_memory_config(self.device, input_shape),
            compute_kernel_config=self.compute_kernel_config,
            program_config=program_config,
        )
        tt_output = ttnn.reshape(tt_output, input_tensor_shape[:-1] + [-1])
        return tt_output


class TTNNLinearLLamaIColShardedWAllReduced(TTNNLinearIColShardedWAllReduced):
    """Column-sharded linear with matmul + all-gather; weights in bfloat8_b (e.g. dots.ocr QKV).

    Step-1 quality baseline: keep **BF16 activations × BFP8 weights** for fused QKV
    while other dots matmuls are moved to BFP4 weights incrementally.

    Compute kernel is tuned for bfloat8_b weights: HiFi2 matches the vision-tower
    setting that runs at ~39% of peak FLOPs (perf.txt) and avoids the 2x cost of
    HiFi4 phases with no precision benefit (input is already capped by BFP8).
    ``fp32_dest_acc_en=False`` doubles the dst register size (4 -> 8 tiles), roughly
    halving the number of matmul passes for these decode-bound projections —
    matches the working pattern in qwen_attention.py / linear_intelligent.py.

    Bias-fusion (multi-device): we replicate the bias on every device but
    pre-divide by ``num_devices`` so it can be fused directly into the matmul
    kernel. After ``reduce_scatter`` sums the per-device partials, the
    contribution from each device's ``bias/k`` adds up to the full bias on
    the relevant N-shard, exactly matching the original behaviour. This
    eliminates one ``ttnn.add`` per layer in TP decode (~28 ops/token).
    """

    def move_weights_to_device_impl(self):
        if isinstance(self.tt_weight_host, torch.Tensor):
            self.tt_weight_host = preprocess_linear_weight(
                self.tt_weight_host,
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                weights_mesh_mapper=_tp_mesh_mapper(self.device, self.weight_dim),
            )
        if isinstance(self.tt_bias_host, torch.Tensor):
            num_devices = _linear_mesh_num_devices(self.device)
            multi_device_ccl = num_devices > 1 and _tp_requires_ccl(self.device)
            if multi_device_ccl:
                bias_torch = self.tt_bias_host / float(num_devices)
                bias_mapper = ttnn.replicate_tensor_to_mesh_mapper(self.device)
                self._bias_fused_into_matmul = True
            else:
                bias_torch = self.tt_bias_host
                bias_mapper = _tp_mesh_mapper(self.device, self.input_dim)
                self._bias_fused_into_matmul = False
            self.tt_bias_host = preprocess_linear_bias(
                bias_torch,
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                weights_mesh_mapper=bias_mapper,
            )
        else:
            self._bias_fused_into_matmul = False
        self.tt_weight = ttnn.to_device(self.tt_weight_host, self.device)
        self.tt_bias = ttnn.to_device(self.tt_bias_host, self.device) if self.tt_bias_host is not None else None
        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )


class TTNNLinearLLamaIColShardedWAllReducedFusedGateUp(TTNNLinearLLamaIColShardedWAllReduced):
    @classmethod
    def from_two_torch(cls, gate_linear: nn.Linear, up_linear: nn.Linear):
        in_features = gate_linear.in_features
        intermediate = gate_linear.out_features
        new_linear = cls(in_features=in_features, out_features=intermediate * 2)
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
        weight_mapper = _tp_mesh_mapper(self.device, self.weight_dim)
        gate_w_host = preprocess_linear_weight(
            self._gate_weight_torch,
            dtype=ttnn.bfloat4_b,
            layout=ttnn.TILE_LAYOUT,
            weights_mesh_mapper=weight_mapper,
        )
        up_w_host = preprocess_linear_weight(
            self._up_weight_torch,
            dtype=ttnn.bfloat4_b,
            layout=ttnn.TILE_LAYOUT,
            weights_mesh_mapper=weight_mapper,
        )
        gate_w = ttnn.to_device(gate_w_host, self.device)
        up_w = ttnn.to_device(up_w_host, self.device)
        self.tt_weight = ttnn.concat([gate_w, up_w], dim=-1)
        ttnn.deallocate(gate_w)
        ttnn.deallocate(up_w)

        has_bias = self._gate_bias_torch is not None or self._up_bias_torch is not None
        if has_bias:
            bias_mapper = _tp_mesh_mapper(self.device, self.input_dim)
            intermediate = self._gate_weight_torch.shape[0]
            zeros_dtype = self._gate_weight_torch.dtype
            g = (
                self._gate_bias_torch
                if self._gate_bias_torch is not None
                else torch.zeros(intermediate, dtype=zeros_dtype)
            )
            u = self._up_bias_torch if self._up_bias_torch is not None else torch.zeros(intermediate, dtype=zeros_dtype)
            gate_b_host = preprocess_linear_bias(
                g, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, weights_mesh_mapper=bias_mapper
            )
            up_b_host = preprocess_linear_bias(
                u, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, weights_mesh_mapper=bias_mapper
            )
            gate_b = ttnn.to_device(gate_b_host, self.device)
            up_b = ttnn.to_device(up_b_host, self.device)
            self.tt_bias = ttnn.concat([gate_b, up_b], dim=-1)
            ttnn.deallocate(gate_b)
            ttnn.deallocate(up_b)
        else:
            self.tt_bias = None

        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )


class TTNNLinearLLamaIReplicatedWColSharded(TTNNLinearIReplicatedWColSharded):
    """Weight column-sharded linear with configurable low-precision weights.

    See TTNNLinearLLamaIColShardedWAllReduced for the compute-kernel rationale —
    HiFi2 + fp32_dest_acc_en=False matches the BF16×BFP4/BFP8 matmul schedule on Wormhole.
    """

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
                weights_mesh_mapper=_tp_mesh_mapper(self.device, self.weight_dim),
            )
        self.tt_weight = ttnn.to_device(self.tt_weight_host, self.device)
        self.tt_bias = ttnn.to_device(self.tt_bias_host, self.device) if self.tt_bias_host is not None else None
        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )

    @run_on_devices(*SHARDED_COLLECTIVE_LINEAR_DEVICE_ARCHS)
    def forward(self, input_tensor: ttnn.Tensor) -> ttnn.Tensor:
        """bf16 output for residual stream; weights are ``bfloat4_b``."""
        if input_tensor.layout != ttnn.TILE_LAYOUT:
            input_tensor = ttnn.to_layout(input_tensor, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        input_tensor_shape = list(input_tensor.shape)
        input_shape = list(input_tensor_shape)
        while len(input_shape) < 4:
            input_shape.insert(1, 1)
        input_tensor = ttnn.reshape(input_tensor, input_shape)
        program_config = _dp_matmul_program_config(self.device, input_shape, self.tt_weight.shape)
        tt_output = ttnn.linear(
            input_tensor,
            self.tt_weight,
            bias=self.tt_bias,
            dtype=ttnn.bfloat16,
            memory_config=_decode_linear_output_memory_config(self.device, input_shape),
            compute_kernel_config=self.compute_kernel_config,
            program_config=program_config,
        )
        tt_output = ttnn.reshape(tt_output, input_tensor_shape[:-1] + [-1])
        return tt_output


@trace_disabled
class TTNNLinearLLamaBFloat16(TTNNLinear):
    """TTNN Linear layer optimized for LLaMA models using bfloat16."""

    @deallocate_weights_after
    def forward(self, input_tensor: ttnn.Tensor) -> ttnn.Tensor:
        """Forward pass with automatic weight deallocation."""
        return super().forward(input_tensor)


class PytorchLinearActivation(nn.Module):
    def __init__(self, dense, act_fn) -> None:
        super().__init__()
        self.dense = dense
        self.intermediate_act_fn = act_fn

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)

        return hidden_states


class TTNNLinearActivation(TTNNModule):
    """Linear layer with activation using TTNN."""

    @classmethod
    def from_parameters(cls, weight, linear_class, ttnn_act_fn, nn_act_fn, bias=None):
        new_linear = cls()
        new_linear.dense = linear_class.from_parameters(weight=weight, bias=bias)
        new_linear.activation = ttnn_act_fn
        return new_linear

    @classmethod
    def from_torch(cls, linear: nn.Linear, linear_class, ttnn_act_fn, nn_act_fn):
        new_linear = cls()
        new_linear._fallback_torch_layer = PytorchLinearActivation(dense=linear, act_fn=nn_act_fn)
        new_linear.dense = linear_class.from_torch(linear)
        new_linear.activation = ttnn_act_fn
        return new_linear

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states


class TTNNLinearGelu:
    """Linear layer with GELU activation using TTNN."""

    @classmethod
    def from_parameters(cls, weight, bias=None, linear_class=TTNNLinear):
        new_linear = TTNNLinearActivation.from_parameters(weight, linear_class, ttnn.gelu, nn.GELU(), bias)
        return new_linear

    @classmethod
    def from_torch(cls, linear: nn.Linear, linear_class=TTNNLinear):
        new_linear = TTNNLinearActivation.from_torch(linear, linear_class, ttnn.gelu, nn.GELU())
        return new_linear


class TTNNLinearSilu:
    """SiLU activated Linear module with TTNN acceleration."""

    @classmethod
    def from_parameters(cls, weight, bias=None, linear_class=TTNNLinear):
        new_linear = TTNNLinearActivation.from_parameters(weight, linear_class, ttnn.silu, nn.SiLU(), bias)
        return new_linear

    @classmethod
    def from_torch(cls, linear: nn.Linear, linear_class=TTNNLinear):
        new_linear = TTNNLinearActivation.from_torch(linear, linear_class, ttnn.silu, nn.SiLU())
        return new_linear


class TTNNViTIntermediate(TTNNLinearGelu):
    """ViT Intermediate module with TTNN acceleration."""

    @classmethod
    def from_torch(cls, torch_vit_intermediate: "ViTIntermediate"):
        assert (
            torch_vit_intermediate.intermediate_act_fn.__class__.__name__ == "GELUActivation"
        ), "Only GELU activation is supported."
        new_intermediate = cls()
        new_intermediate._fallback_torch_layer = torch_vit_intermediate
        new_intermediate.dense = TTNNLinear.from_torch(torch_vit_intermediate.dense)
        return new_intermediate


# =============================================================================
# DRAM-width-sharded LM head with N-chunk splitting.
# =============================================================================
#
# LM head chunk weights: ``bfloat4_b`` (BF16 activations from the decoder).
# Mirrors the pattern from ``models/tt_transformers/tt/lm_head.py``: weights
# are stored with ``TensorMemoryLayout.WIDTH_SHARDED`` over the 12 DRAM banks
# (vs DRAM-interleaved which reads serially from one bank), and the matmul
# uses ``MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig`` so the
# kernel can stream weight tiles from all 12 banks in parallel.
#
# Why N-parallel + chunking and not just K-parallel:
#  - dots.ocr currently runs lm_head as ``TTNNLinearLLamaIColShardedWAllReduced``
#    (K-parallel + all_reduce). With K=hidden=1536 split 2-way → K_per_dev=768
#    (24 tiles), the GCD with N=151936/2=75968 (2374 tiles) is 2 → at most
#    2 compute cores can divide both dims cleanly, far too few for the
#    DRAM-sharded matmul to be efficient.
#  - Switching to N-parallel makes K=1536 (48 tiles) replicated on every
#    device. With N split into chunks of ~12K cols each (≈384 tiles), the
#    8 compute cores divide both 48 K-tiles and 384 N-tiles cleanly
#    (in0_block_w=6, per_core_N=48). Output buffer per core stays small
#    (≈48 KB at bf8) so trace L1 plan fits. There is one extra all_gather
#    on the K-replicate (input went from K-sharded to replicated), but the
#    matmul-side win dominates.
#
# Failure modes to watch for during validation:
#  - L1 OOM in trace if the per-chunk output buffer plus the existing
#    decode-stage buffers exceed per-core L1. Lower MAX_COLUMNS_PER_CHUNK
#    to halve per-core output if hit.
#  - ``dram_grid`` on the device must have x=12 (Wormhole). On Blackhole
#    (8 banks) the chunk math will need a different MAX_COLUMNS_PER_CHUNK.
#  - The all_gather on the input from K-sharded → replicated must succeed
#    on the same fabric topology used elsewhere (FABRIC_1D_RING, num_links=1).
# =============================================================================


def _dram_sharded_mem_config_2d(device, k: int, n: int):
    """Per-device DRAM-WIDTH-SHARDED memory config for a [k, n] weight slice.

    Splits the n columns over the device's DRAM-bank cores (12 on
    Wormhole) and pads up to ``tile_size * dram_cores``. Mirrors
    ``tt_transformers/tt/model_config.create_dram_sharded_mem_config``.
    """
    tile = ttnn.TILE_SIZE
    dram_grid = device.dram_grid_size()
    dram_cores = int(dram_grid.x)
    padded_n = math.ceil(n / (tile * dram_cores)) * (tile * dram_cores)
    shard_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_cores - 1, 0))])
    shard_spec = ttnn.ShardSpec(shard_grid, (k, padded_n // dram_cores), ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, shard_spec)


def _dram_matmul_program_config_for(k: int, n: int, num_cores: int = 8):
    """Build a DRAM-sharded matmul program config for an [M=32, K, N] decode matmul."""
    tile = ttnn.TILE_SIZE
    in0_block_w = _largest_divisor_at_most(k // tile // num_cores, 8) if (k // tile) % num_cores == 0 else 1
    per_core_m = 1
    per_core_n = math.ceil(n / (tile * num_cores))
    return ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=max(1, in0_block_w),
        per_core_M=per_core_m,
        per_core_N=per_core_n,
        fused_activation=None,
    )


class TTNNDotsOCRDRAMShardedLMHead(TTNNModule):
    """LM head with DRAM-width-sharded weight chunks.

    Mesh layout: dim 0 = DP axis (replicate weight), dim -1 = TP axis (shard
    weight along N). Vocab is padded to a multiple of ``chunk_align * num_tp``
    so the per-TP-device slice stays cleanly chunkable.

    Forward path (per chip):
        1. (TP only) all_gather input on dim=-1 (K-sharded → replicated).
        2. interleaved_to_sharded(input) once -- L1 width-sharded across 8 cores.
        3. For each weight chunk: ttnn.matmul with
           MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig, output is
           L1 width-sharded; sharded_to_interleaved(output) for concat.
        4. concat across chunks → ``[B, M, size_per_device]`` per chip.
        5. (TP only) all_gather across mesh on N -- one CCL.
        6. Slice off vocab padding, return ``[B, M, vocab]``.

    Previous correctness bug (now fixed): weight loader used
    ``shard_tensor_to_mesh_mapper(dim=-1)`` which always sharded across the
    first mesh axis, so on T3K mesh (8,1) (DP=8, TP=1) every chip got 1/8 of
    the vocab. ``_tp_requires_ccl`` is False for that mesh, so no all_gather
    ran and argmax saw a tiny garbage slice. Fix: use ``_tp_mesh_mapper`` so
    the DP axis is **replicated** and only the TP axis is sharded.
    """

    MAX_COLUMNS_PER_CHUNK = 12288  # see top-of-block notes

    # Decode LM head is bandwidth-bound: each step does sharded_to_interleaved
    # (width-sharded RMSNorm -> interleaved), to_memory_config (interleaved ->
    # L1 width-sharded for matmul), per-chunk DRAM-sharded matmul, then
    # sharded_to_interleaved again. Future work: (1) emit RMSNorm in interleaved
    # TILE BF16 when M==1 to skip the first de-shard; (2) A/B L1 vs DRAM for
    # sharded_to_interleaved staging on tiny tensors (watch L1 trace budgets).

    def __init__(self) -> None:
        super().__init__()
        self.in_features = 0
        self.out_features = 0
        self._weight_torch = None
        self._bias_torch = None
        self.tt_weight_chunks = []
        self.tt_bias_chunks = []
        self._chunk_program_configs = []
        self._chunk_n_cols_per_device = []
        self._padded_vocab = 0
        self._size_per_device = 0
        self._input_shard_cfg = None
        # Note: attr name is `compute_kernel_config` (no underscore prefix) so
        # the pipeline's post-weight-load override in
        # ``TTNNDotsOCRPipeline._set_device_and_preprocess`` (which assigns
        # ``self.lm_head.compute_kernel_config = ...``) actually replaces the
        # config used during forward.
        self.compute_kernel_config = None

    @classmethod
    def from_torch(cls, linear: nn.Linear) -> "TTNNDotsOCRDRAMShardedLMHead":
        new = cls()
        new.in_features = int(linear.in_features)
        new.out_features = int(linear.out_features)
        new._fallback_torch_layer = linear
        new._weight_torch = linear.weight  # [out, in] = [vocab, hidden]
        new._bias_torch = linear.bias
        return new

    def preprocess_weights_impl(self):
        # Defer all heavy work to move_weights_to_device_impl where we know
        # the mesh shape and the DRAM bank count.
        return

    def move_weights_to_device_impl(self):
        device = self.device
        tile = ttnn.TILE_SIZE

        # Mesh layout: dim 0 is the DP axis (replicate weight across), dim -1
        # is the TP axis (shard weight across N). ``T3K + DOTS_OCR_PARALLELISM=DP``
        # gives mesh_shape=(8, 1) → num_tp=1, num_dp=8: vocab is **replicated**
        # on every chip. If we instead used ``shard_tensor_to_mesh_mapper(dim=-1)``
        # (which shards along the FIRST mesh axis), each chip would only see
        # 1/8 of the vocab logits, ``_tp_requires_ccl`` would be False so the
        # final all_gather would be skipped, and argmax would silently pick a
        # garbage token from the local 1/8 slice. That was the corruption
        # observed previously when this class was wired in.
        mesh_shape = list(device.shape) if hasattr(device, "shape") else [1, 1]
        num_tp = int(mesh_shape[-1]) if mesh_shape else 1
        self._num_tp = num_tp

        # Alignment constraints for the DRAM-sharded matmul kernel:
        #   * weight is WIDTH_SHARDED across ``dram_cores`` DRAM banks per
        #     device → ``chunk_n`` must be a multiple of
        #     ``tile * dram_cores`` (=384 on Wormhole).
        #   * output is WIDTH_SHARDED across ``num_compute_cores`` compute
        #     cores → ``chunk_n`` must also be a multiple of
        #     ``tile * num_compute_cores`` (=256 on the 1×8 compute grid).
        # If only the dram-side alignment is satisfied, the kernel
        # auto-rounds ``per_core_N = ceil(N_tiles / num_compute_cores)`` and
        # writes padding tiles in the last compute core's output. After
        # ``sharded_to_interleaved → concat`` those padding tiles slot
        # between vocab segments and silently shift every logit past the
        # first chunk, which corrupts ``argmax``. So align to the LCM.
        dram_cores = int(device.dram_grid_size().x) if hasattr(device, "dram_grid_size") else 12
        num_compute_cores = 8  # matches _dram_matmul_program_config_for(num_cores=8)
        chunk_align = (tile * dram_cores * tile * num_compute_cores) // math.gcd(
            tile * dram_cores, tile * num_compute_cores
        )  # lcm

        # Pad vocab so size_per_device is itself ``chunk_align``-aligned.
        # That guarantees the residual last chunk is also a multiple of
        # ``chunk_align`` (since (size_per_device - k*base_chunk) stays
        # in the lattice if both terms do).
        per_tp_align = chunk_align * num_tp
        padded_vocab = math.ceil(self.out_features / per_tp_align) * per_tp_align
        self._padded_vocab = padded_vocab
        size_per_device = padded_vocab // num_tp
        self._size_per_device = size_per_device

        weight_torch = self._weight_torch  # [out, in]
        if weight_torch.dim() != 2:
            raise ValueError(f"Expected weight [out,in], got {tuple(weight_torch.shape)}")
        # Pad output dim to padded_vocab with zeros.
        if int(weight_torch.shape[0]) < padded_vocab:
            pad_rows = padded_vocab - int(weight_torch.shape[0])
            weight_torch = torch.cat(
                [weight_torch, torch.zeros(pad_rows, weight_torch.shape[1], dtype=weight_torch.dtype)],
                dim=0,
            )
        # Transpose to [in, out] = [hidden, padded_vocab] for matmul.
        weight_t = weight_torch.transpose(-2, -1).contiguous()

        # Decide how many chunks per device. Chunks let the per-core L1
        # output buffer stay small in the DRAM-sharded matmul kernel.
        num_chunks = max(1, math.ceil(size_per_device / self.MAX_COLUMNS_PER_CHUNK))
        base_chunk = math.ceil(size_per_device / num_chunks)
        base_chunk = math.ceil(base_chunk / chunk_align) * chunk_align
        chunk_sizes = []
        remaining = size_per_device
        for _ in range(num_chunks - 1):
            take = min(base_chunk, remaining)
            chunk_sizes.append(take)
            remaining -= take
        chunk_sizes.append(remaining)
        # Final assertion: every chunk must divide cleanly across both
        # 12 DRAM banks (weight side) and 8 compute cores (output side).
        for cn in chunk_sizes:
            assert (
                cn % chunk_align == 0
            ), f"chunk_n={cn} not aligned to lcm(tile*dram_cores, tile*compute_cores)={chunk_align}"
        self._chunk_n_cols_per_device = chunk_sizes

        # Use ``_tp_mesh_mapper`` so weight is replicated across the DP axis
        # and sharded across the TP axis. The "combined" tensor we build
        # per chunk has width ``num_tp * chunk_n`` (one slice per TP-axis
        # device) — in DP-only mode (num_tp=1) that is just the per-chip
        # weight, which is then auto-replicated across the DP axis.
        is_multi_device = hasattr(device, "get_num_devices") and int(device.get_num_devices()) > 1
        weight_mapper = _tp_mesh_mapper(device, dim=-1) if is_multi_device else None

        self.tt_weight_chunks = []
        for chunk_idx, chunk_n in enumerate(chunk_sizes):
            tp_splits = []
            for tp_idx in range(num_tp):
                start = tp_idx * size_per_device + sum(chunk_sizes[:chunk_idx])
                end = start + chunk_n
                tp_splits.append(weight_t[:, start:end])
            combined = torch.cat(tp_splits, dim=-1)  # [hidden, num_tp*chunk_n]

            mem_cfg = _dram_sharded_mem_config_2d(device, k=int(combined.shape[0]), n=chunk_n)
            tt_chunk = ttnn.as_tensor(
                combined,
                device=device,
                mesh_mapper=weight_mapper,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat4_b,
                memory_config=mem_cfg,
            )
            self.tt_weight_chunks.append(tt_chunk)

        # Bias is rarely set on lm_head; keep simple. If present, split
        # along N like the weight (no DRAM sharding needed for the bias --
        # it's small).
        self.tt_bias_chunks = []
        if self._bias_torch is not None:
            bias_torch = self._bias_torch
            if int(bias_torch.shape[0]) < padded_vocab:
                bias_torch = torch.cat(
                    [bias_torch, torch.zeros(padded_vocab - int(bias_torch.shape[0]), dtype=bias_torch.dtype)],
                    dim=0,
                )
            for chunk_idx, chunk_n in enumerate(chunk_sizes):
                tp_splits = []
                for tp_idx in range(num_tp):
                    start = tp_idx * size_per_device + sum(chunk_sizes[:chunk_idx])
                    end = start + chunk_n
                    tp_splits.append(bias_torch[start:end])
                combined_b = torch.cat(tp_splits, dim=-1).unsqueeze(0)  # [1, num_tp*chunk_n]
                tt_b = ttnn.as_tensor(
                    combined_b,
                    device=device,
                    mesh_mapper=weight_mapper,
                    layout=ttnn.TILE_LAYOUT,
                    dtype=ttnn.bfloat8_b,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                self.tt_bias_chunks.append(tt_b)

        # Build per-chunk program configs once.
        self._chunk_program_configs = [
            _dram_matmul_program_config_for(k=self.in_features, n=chunk_n, num_cores=8) for chunk_n in chunk_sizes
        ]

        # Cached input width-sharded mem config (input is M=32, K=hidden).
        # We shard the K dim across 8 cores so each core gets K/8 columns.
        # NOTE: ``create_sharded_memory_config(shape=...)`` without
        # ``use_height_and_width_as_shard_shape`` interprets ``shape`` as the
        # FULL tensor shape and divides the last dim by ``core_grid.x``. So
        # we pass ``(tile, in_features)`` — the per-core shard width then
        # comes out tile-aligned: ``in_features / 8`` tiles wide.
        in_grid = ttnn.CoreGrid(y=1, x=8)
        self._input_shard_cfg = ttnn.create_sharded_memory_config(
            shape=(tile, self.in_features),
            core_grid=in_grid,
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        )

        # Compute kernel: LoFi + BFP4 weights + packer L1 acc + FP32 dest accum
        # (bandwidth-bound LM head). Pipeline may override after weight load.
        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def deallocate_weights_impl(self):
        for w in self.tt_weight_chunks:
            ttnn.deallocate(w)
        for b in self.tt_bias_chunks:
            ttnn.deallocate(b)
        self.tt_weight_chunks = []
        self.tt_bias_chunks = []
        super().deallocate_weights_impl()

    def forward(self, input_tensor: ttnn.Tensor) -> ttnn.Tensor:
        device = self.device
        num_devices = device.get_num_devices() if hasattr(device, "get_num_devices") else 1

        x = input_tensor
        # The upstream final RMSNorm hands us a width-sharded tensor whose
        # shard shape is not necessarily tile-aligned (e.g. (32, 24) for
        # K=768 across 32 cores). The DRAM-sharded matmul kernel needs an
        # input that is L1 WIDTH-sharded with a tile-aligned shard shape, so
        # de-shard to interleaved DRAM first; the cost is one
        # ``sharded_to_interleaved`` per token call (small for [1,32,K]).
        if x.memory_config().is_sharded():
            x = ttnn.sharded_to_interleaved(x, ttnn.DRAM_MEMORY_CONFIG)
        if x.layout != ttnn.TILE_LAYOUT:
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if x.dtype != ttnn.bfloat16:
            x = ttnn.typecast(x, ttnn.bfloat16, memory_config=x.memory_config())
        # Force shape to 4D [B, 1, M, K] for the matmul kernel.
        in_shape = list(x.shape)
        if len(in_shape) == 2:
            in_shape = [1, 1] + in_shape
        elif len(in_shape) == 3:
            in_shape = [in_shape[0], 1, in_shape[1], in_shape[2]]
        if list(x.shape) != in_shape:
            x = ttnn.reshape(x, in_shape)

        # Step 1: K-sharded → replicated (only if multi-device). Each device's
        # input has K_per_dev columns; the lm_head weight is N-parallel which
        # needs the full K, so all_gather along the last dim.
        needs_ccl = num_devices > 1 and _tp_requires_ccl(device)
        if needs_ccl and int(x.shape[-1]) < self.in_features:
            x = ttnn.all_gather(
                x,
                dim=3,
                num_links=_ccl_num_links(device),
                cluster_axis=1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                topology=ttnn.Topology.Linear,
            )

        # Step 2: shard input to L1 width-sharded across 8 cores once.
        x_sharded = ttnn.to_memory_config(x, self._input_shard_cfg)

        # Step 3: per-chunk DRAM-sharded matmul, then sharded → interleaved.
        # Single-chunk decode (common when N fits one DRAM-sharded slice) avoids
        # a Python list + concat/dealloc round-trip between matmul and gather.
        num_chunks = len(self.tt_weight_chunks)
        if num_chunks == 1:
            w_chunk = self.tt_weight_chunks[0]
            pc = self._chunk_program_configs[0]
            bias_chunk = self.tt_bias_chunks[0] if self.tt_bias_chunks else None
            full = ttnn.linear(
                x_sharded,
                w_chunk,
                bias=bias_chunk,
                program_config=pc,
                compute_kernel_config=self.compute_kernel_config,
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                dtype=ttnn.bfloat8_b,
            )
            full = ttnn.sharded_to_interleaved(full, ttnn.DRAM_MEMORY_CONFIG)
        else:
            chunk_outs = []
            for i, (w_chunk, pc) in enumerate(zip(self.tt_weight_chunks, self._chunk_program_configs)):
                bias_chunk = self.tt_bias_chunks[i] if self.tt_bias_chunks else None
                out_chunk = ttnn.linear(
                    x_sharded,
                    w_chunk,
                    bias=bias_chunk,
                    program_config=pc,
                    compute_kernel_config=self.compute_kernel_config,
                    memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                    dtype=ttnn.bfloat8_b,
                )
                out_chunk = ttnn.sharded_to_interleaved(out_chunk, ttnn.DRAM_MEMORY_CONFIG)
                chunk_outs.append(out_chunk)
            full = ttnn.concat(chunk_outs, dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            for c in chunk_outs:
                ttnn.deallocate(c)
        ttnn.deallocate(x_sharded)

        # Step 4: all_gather across mesh on N to assemble full vocab on every
        # device for argmax.
        if needs_ccl:
            full = ttnn.all_gather(
                full,
                dim=3,
                num_links=_ccl_num_links(device),
                cluster_axis=1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                topology=ttnn.Topology.Linear,
            )

        # If we padded the vocab during weight loading, slice back to logical vocab.
        if self._padded_vocab > self.out_features:
            full = ttnn.slice(
                full,
                [0, 0, 0, 0],
                [int(full.shape[0]), 1, int(full.shape[-2]), self.out_features],
            )

        # Restore original logical input rank for downstream argmax.
        if len(input_tensor.shape) == 3:
            full = ttnn.reshape(full, [int(full.shape[0]), int(full.shape[-2]), int(full.shape[-1])])
        elif len(input_tensor.shape) == 2:
            full = ttnn.reshape(full, [int(full.shape[-2]), int(full.shape[-1])])
        return full
