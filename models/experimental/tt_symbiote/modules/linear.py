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
    # faster. Bail out and let the auto-config handle it.
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
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
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
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
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
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
            program_config=program_config,
        )
        tt_output = ttnn.reshape(tt_output, input_tensor_shape[:-1] + [-1])
        return tt_output


class TTNNLinearLLamaIColShardedWAllReduced(TTNNLinearIColShardedWAllReduced):
    """Column-sharded linear with matmul + all-gather; weights in bfloat8_b (e.g. dots.ocr QKV / gate / up).

    Compute kernel is tuned for bfloat8_b weights: HiFi2 matches the vision-tower
    setting that runs at ~39% of peak FLOPs (perf.txt) and avoids the 2x cost of
    HiFi4 phases with no precision benefit (input is already capped by BFP8).
    fp32_dest_acc_en=False doubles the dst register size (4 -> 8 tiles), roughly
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
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            weights_mesh_mapper=weight_mapper,
        )
        up_w_host = preprocess_linear_weight(
            self._up_weight_torch,
            dtype=ttnn.bfloat8_b,
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
    """Weight column-sharded linear with bfloat8_b weights (e.g. dots.ocr o_proj / down_proj).

    See TTNNLinearLLamaIColShardedWAllReduced for the compute-kernel rationale —
    HiFi2 + fp32_dest_acc_en=False is the proven setting for bfloat8_b weights.
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
