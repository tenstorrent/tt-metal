# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from typing import ClassVar

import torch

import ttnn

from .module import Module, Parameter


class RMSNorm(Module):
    def __init__(
        self,
        embedding_dim,
        norm_eps=1e-5,
        norm_elementwise_affine=True,
        bias=True,
        mesh_device=None,
        dtype=ttnn.bfloat16,
        fused_activation=None,
    ):
        super().__init__()

        # https://github.com/tenstorrent/tt-metal/issues/31216
        assert embedding_dim % 32 == 0, "embedding_dim must be divisible by tile size"

        self.embedding_dim = embedding_dim
        self.norm_eps = norm_eps
        self.norm_elementwise_affine = norm_elementwise_affine
        self.mesh_device = mesh_device
        self.use_bias = norm_elementwise_affine and bias
        self.fused_activation = fused_activation

        if norm_elementwise_affine:
            self.weight = Parameter(total_shape=[1, embedding_dim], device=mesh_device, dtype=dtype)
            self.bias = Parameter(total_shape=[1, embedding_dim], device=mesh_device, dtype=dtype) if bias else None
        else:
            self.weight = None
            self.bias = None

    def forward(
        self,
        x: ttnn.Tensor,
        *,
        compute_kernel_config=None,
        program_config: ttnn.LayerNormDefaultProgramConfig | ttnn.LayerNormShardedMultiCoreProgramConfig | None = None,
    ) -> ttnn.Tensor:
        return ttnn.experimental.dit_rms_norm_unary_fused(
            x,
            weight=self.weight.data if self.weight is not None else None,
            bias=self.bias.data if self.bias is not None else None,
            epsilon=self.norm_eps,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
            activation=self.fused_activation,
        )

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        if "weight" in state:
            state["weight"] = state["weight"].unsqueeze(0)

        if "bias" in state:
            state["bias"] = state["bias"].unsqueeze(0)


class LayerNorm(Module):
    def __init__(
        self,
        embedding_dim,
        norm_eps=1e-5,
        norm_elementwise_affine=True,
        bias=True,
        mesh_device=None,
        use_row_major_workaround=False,  # Issue #20789
    ):
        super().__init__()

        assert embedding_dim % 32 == 0, "embedding_dim must be divisible by tile size"

        self.embedding_dim = embedding_dim
        self.norm_eps = norm_eps
        self.norm_elementwise_affine = norm_elementwise_affine
        self.mesh_device = mesh_device
        self.use_bias = norm_elementwise_affine and bias
        self.use_row_major_workaround = use_row_major_workaround

        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            self.mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

        shape = [embedding_dim // 32, 32] if use_row_major_workaround else [1, embedding_dim]
        layout = ttnn.ROW_MAJOR_LAYOUT if self.use_row_major_workaround else ttnn.TILE_LAYOUT

        self.weight = (
            Parameter(total_shape=shape, layout=layout, device=mesh_device)
            if norm_elementwise_affine or self.use_row_major_workaround
            else None
        )
        self.bias = Parameter(total_shape=shape, layout=layout, device=mesh_device) if self.use_bias else None

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        weight = state.pop("weight", None)
        bias = state.pop("bias", None)

        # When using the row-major workaround, ensure that dummy weight/bias are created
        if self.use_row_major_workaround:
            assert self.norm_elementwise_affine == (weight is not None)
            assert self.use_bias == (bias is not None)

            if weight is None:
                weight = torch.ones(self.embedding_dim)
            if self.use_bias and bias is None:
                bias = torch.zeros(self.embedding_dim)

        if weight is not None:
            state["weight"] = weight.reshape(-1, 32) if self.use_row_major_workaround else weight.unsqueeze(0)

        if bias is not None:
            state["bias"] = bias.reshape(-1, 32) if self.use_row_major_workaround else bias.unsqueeze(0)

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        return ttnn.layer_norm(
            x,
            weight=self.weight.data if self.weight is not None else None,
            bias=self.bias.data if self.bias is not None else None,
            epsilon=self.norm_eps,
            compute_kernel_config=self.compute_kernel_config,
        )


class DistributedRMSNorm(Module):
    """
    Implements RMSNorm on an activation sharded on the reduction dimension.
    """

    def __init__(
        self,
        embedding_dim,
        norm_eps=1e-5,
        norm_elementwise_affine=True,
        bias=False,
        mesh_axis=0,
        mesh_device=None,
        ccl_manager=None,
    ):
        super().__init__()

        assert not bias, "bias is not supported for DistributedRMSNorm"
        self.embedding_dim = embedding_dim
        self.norm_eps = norm_eps
        self.norm_elementwise_affine = norm_elementwise_affine
        self.mesh_axis = mesh_axis
        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.mesh_width = tuple(mesh_device.shape)[mesh_axis]
        self.TILE_SIZE = 32

        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            self.mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

        n = self.TILE_SIZE * self.mesh_width

        # https://github.com/tenstorrent/tt-metal/issues/31216
        assert embedding_dim % n == 0, "embedding_dim must be divisible by tile size times mesh width"

        self.weight = (
            Parameter(
                total_shape=[1, embedding_dim],
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                mesh_axes=[None, mesh_axis],
            )
            if norm_elementwise_affine
            else None
        )

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        if "weight" in state:
            state["weight"] = state["weight"].reshape(1, self.embedding_dim)

    def forward(
        self,
        x: ttnn.Tensor,
        num_heads_per_device=1,
        compute_kernel_config=None,
        rope_cos=None,
        rope_sin=None,
        trans_mat=None,
        dtype=None,
        per_head_norm: bool = False,
    ) -> ttnn.Tensor:
        """
        Args:
            per_head_norm: if True, normalize each head independently with divisor `head_dim`
                instead of the global `embedding_dim`. Requires num_heads_per_device > 1.
                The pre-AG kernel emits per-head stat tiles (num_heads_per_device tiles per row);
                the AG step is SKIPPED because each device's per-head stats are already local
                (heads are not split across TP). The post-AG kernel then applies per-head RMS.

                When False (default), behavior is unchanged: global RMS across the full
                `embedding_dim` per row (same as before — used by WAN, Flux2 double block, etc.).
        """
        expected_dim = self.embedding_dim // self.mesh_width
        if x.shape[-1] != expected_dim:
            msg = (
                f"last dimension of input tensor with shape {tuple(x.shape)} should match "
                f"embedding_dim / mesh_width = {expected_dim}"
            )
            raise ValueError(msg)

        # Fused distributed RMSNorm device op (PRE sum-of-squares + fabric ring AG + POST
        # normalize, with optional fused RoPE / per-head norm).
        return ttnn.experimental.dit_fused_distributed_rmsnorm(
            x,
            self.mesh_axis,
            self.mesh_device,
            self.ccl_manager.get_ag_ping_pong_semaphore(self.mesh_axis),
            topology=self.ccl_manager.topology,
            persistent_output_buffer=self.ccl_manager.get_fused_norm_stats_buffer(
                # Key includes everything that changes the stats-buffer geometry:
                # shape, heads-per-device, RoPE presence, and weight presence (weight is
                # forwarded to create_stats_buffer and affects its sizing). Guards against
                # a shared-cache collision between two same-shape modules differing only
                # in affine geometry.
                ("rms", tuple(x.shape), num_heads_per_device, rope_cos is not None, self.weight is not None),
                lambda: ttnn.experimental.dit_fused_distributed_rmsnorm_create_stats_buffer(
                    x,
                    self.mesh_axis,
                    self.mesh_device,
                    num_heads_per_device=num_heads_per_device,
                    num_links=self.ccl_manager.num_links,
                    weight=self.weight.data if self.weight is not None else None,
                    transformation_mat=trans_mat,
                    rope_cos=rope_cos,
                    rope_sin=rope_sin,
                ),
            ),
            epsilon=self.norm_eps,
            num_heads_per_device=num_heads_per_device,
            weight=self.weight.data if self.weight is not None else None,
            compute_kernel_config=compute_kernel_config or self.compute_kernel_config,
            num_preferred_links=self.ccl_manager.num_links,  # must match create_stats_buffer above
            transformation_mat=trans_mat,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
            dtype=dtype,
            per_head_norm=per_head_norm,
        )


class DistributedLayerNorm(Module):
    """
    Implements LayerNorm on an activation sharded on the reduction dimension.
    """

    # The fused-op reciprocal LUT depends only on (device, width_per_device), so it is shared
    # across DistributedLayerNorm instances of the same shape instead of each allocating its own.
    _fused_ln_recip_cache: ClassVar[dict[tuple[int, int], "ttnn.Tensor"]] = {}

    def __init__(
        self,
        embedding_dim,
        norm_eps=1e-5,
        norm_elementwise_affine=True,
        bias=True,
        mesh_axis=0,
        mesh_device=None,
        ccl_manager=None,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.norm_eps = norm_eps
        self.norm_elementwise_affine = norm_elementwise_affine
        self.use_bias = norm_elementwise_affine and bias
        self.mesh_axis = mesh_axis
        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.mesh_width = tuple(mesh_device.shape)[mesh_axis]
        self.TILE_SIZE = 32

        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            self.mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

        n = self.TILE_SIZE * self.mesh_width
        assert embedding_dim % n == 0, "embedding_dim must be divisible by tile size times mesh width"

        # Static affine weight/bias are TILE [1, embedding_dim] sharded on the reduction axis —
        # the broadcast layout the fused dit_fused_distributed_rmsnorm op consumes (per-device
        # [1, H/mesh_width]). adaLN passes dynamic weight/bias at forward instead.
        self.weight = (
            Parameter(
                total_shape=[1, embedding_dim], layout=ttnn.TILE_LAYOUT, mesh_axes=[None, mesh_axis], device=mesh_device
            )
            if self.norm_elementwise_affine
            else None
        )
        self.bias = (
            Parameter(
                total_shape=[1, embedding_dim], layout=ttnn.TILE_LAYOUT, mesh_axes=[None, mesh_axis], device=mesh_device
            )
            if self.use_bias
            else None
        )

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        weight = state.pop("weight", None)
        bias = state.pop("bias", None)
        assert (weight is not None) == self.norm_elementwise_affine
        assert (bias is not None) == self.use_bias

        # TILE [1, embedding_dim] sharded on the reduction axis (matches DistributedRMSNorm).
        if self.norm_elementwise_affine:
            state["weight"] = weight.reshape(1, self.embedding_dim)

        if self.use_bias:
            state["bias"] = bias.reshape(1, self.embedding_dim)

    def _ensure_fused_ln_recip(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """Lazy-allocate the row-major fp32 reciprocal LUT the fused op consumes.

        The fused op's reader NoC-reads a ROW_MAJOR [1,1,1,width_per_device] DRAM tensor
        holding [1/1..1/width] (replicated per device) — unlike the composite Welford op,
        which used a HEIGHT_SHARDED L1 layout. Cached per (device, width).
        """
        width = self.embedding_dim // self.mesh_width
        key = (self.mesh_device.id(), width)
        cached = DistributedLayerNorm._fused_ln_recip_cache.get(key)
        if cached is not None:
            return cached
        recip = torch.tensor([1.0 / (i + 1) for i in range(width)], dtype=torch.float32).reshape(1, 1, 1, width)
        tensor = ttnn.from_torch(
            recip,
            dtype=ttnn.float32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        DistributedLayerNorm._fused_ln_recip_cache[key] = tensor
        return tensor

    def forward(
        self, x: ttnn.Tensor, dynamic_weight=None, dynamic_bias=None, compute_kernel_config=None, dtype=None
    ) -> ttnn.Tensor:
        assert (dynamic_weight is None) == (
            dynamic_bias is None
        ), "dynamic_weight and dynamic_bias must be either both provided or both None"
        if dynamic_weight is not None:
            assert (
                not self.norm_elementwise_affine
            ), "Module must not have weight and bias parameters when dynamic_weight and dynamic_bias are provided"
            weight = dynamic_weight
            bias = dynamic_bias
        else:
            weight = self.weight.data if self.weight is not None else None
            bias = self.bias.data if self.bias is not None else None

        # Fused Welford LayerNorm device op. weight/bias (static or adaLN, bf16 or fp32) are
        # consumed natively in-op — fp32 affine keeps the modulation precision adaLN needs.
        return ttnn.experimental.dit_fused_distributed_layernorm(
            x,
            self.mesh_axis,
            self.mesh_device,
            self.ccl_manager.get_ag_ping_pong_semaphore(self.mesh_axis),
            topology=self.ccl_manager.topology,
            persistent_output_buffer=self.ccl_manager.get_fused_norm_stats_buffer(
                ("ln", tuple(x.shape)),
                lambda: ttnn.experimental.dit_fused_distributed_layernorm_create_stats_buffer(
                    x,
                    self.mesh_axis,
                    self.mesh_device,
                    num_links=self.ccl_manager.num_links,
                ),
            ),
            epsilon=self.norm_eps,
            weight=weight,
            bias=bias,
            compute_kernel_config=compute_kernel_config or self.compute_kernel_config,
            num_preferred_links=self.ccl_manager.num_links,  # must match create_stats_buffer above
            dtype=dtype,
            reciprocals=self._ensure_fused_ln_recip(x),
        )


"""
Groupnorm that supports data parallel computation.
The number of channels and groups will be updated to match the distribution of the data across the mesh.
Set mesh_axis to None to disable data parallelism.
"""


class GroupNorm(Module):
    default_num_out_blocks = {
        # (Batch, Height, Width, Channels): num_out_blocks
    }  # used to override the num_out_blocks computed based on the input shape.

    def __init__(
        self,
        num_channels: int,
        num_groups: int,
        *,
        eps: float = 1e-5,
        mesh_device: ttnn.MeshDevice,
        mesh_axis: int | None = None,
        core_grid: ttnn.CoreGrid | None = None,
    ) -> None:
        """
        Args:
            num_channels: Number of channels in the input tensor.
            num_groups: Number of groups.
            eps: Epsilon value for numerical stability.
            mesh_device: The device to use.
            mesh_axis: The mesh axis to use for sharding.
            core_grid: The core grid to use.
            num_out_blocks: The number of output blocks to use.
        """
        super().__init__()

        self.eps = eps
        self.mesh_device = mesh_device
        self.mesh_axis = mesh_axis
        self.num_devices = tuple(mesh_device.shape)[mesh_axis] if mesh_axis is not None else 1
        self.core_grid = core_grid or ttnn.CoreGrid(x=8, y=8)  # mesh_device.core_grid # Issue on 6U 8x9 grid

        assert num_channels % num_groups == 0, "num_channels must be divisible by num_groups"
        assert num_groups % self.num_devices == 0, "num_groups must be divisible by num_devices"

        num_local_channels = num_channels // self.num_devices
        num_padded_channels = math.ceil(num_local_channels / 32) * 32

        assert num_padded_channels % num_local_channels == 0, "padded channels must be divisible by channels"

        num_padded_groups = num_groups // self.num_devices * num_padded_channels // num_local_channels

        self.num_virtual_cols = ttnn.operations.normalization.dram_group_norm_virtual_columns(
            mesh_device.core_grid, num_padded_channels, num_padded_groups
        )

        weight_shape = [
            self.num_devices,
            1,
            math.ceil(num_padded_channels // self.num_virtual_cols / 32) * self.num_virtual_cols,
            32,
        ]
        block_wt = ttnn.operations.normalization.find_max_tile_span(
            num_padded_channels, num_padded_channels // num_padded_groups, 32
        )
        mask_shape = [1, num_padded_groups, 32, 32 * block_wt]

        self.weight = Parameter(
            total_shape=weight_shape,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_axes=[mesh_axis, None, None, None],
            device=self.mesh_device,
        )
        self.bias = Parameter(
            total_shape=weight_shape,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_axes=[mesh_axis, None, None, None],
            device=self.mesh_device,
        )
        self.mask = Parameter(total_shape=mask_shape, device=self.mesh_device)

        self.num_local_channels = num_local_channels
        self.num_padded_channels = num_padded_channels
        self.num_padded_groups = num_padded_groups

    @classmethod
    def from_torch(
        cls,
        torch_ref: torch.nn.GroupNorm,
        *,
        mesh_device: ttnn.MeshDevice,
        mesh_axis: int | None = None,
        core_grid: ttnn.CoreGrid | None = None,
    ) -> GroupNorm:
        module = cls(
            num_channels=torch_ref.num_channels,
            num_groups=torch_ref.num_groups,
            eps=torch_ref.eps,
            mesh_device=mesh_device,
            mesh_axis=mesh_axis,
            core_grid=core_grid,
        )

        module.load_torch_state_dict(torch_ref.state_dict())
        return module

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        if "weight" in state:
            state["weight"] = self._prepare_param(state["weight"])
        if "bias" in state:
            state["bias"] = self._prepare_param(state["bias"])

        input_mask = ttnn.create_group_norm_input_mask(
            self.num_padded_channels, self.num_padded_groups, self.num_virtual_cols
        )
        state["mask"] = ttnn.to_torch(input_mask)

    def _prepare_param(self, param: torch.Tensor) -> torch.Tensor:
        expected_shape = (self.num_local_channels * self.num_devices,)
        assert param.shape == expected_shape, f"expected shape {expected_shape}, got {param.shape}"

        padding = self.num_padded_channels - self.num_local_channels
        params = [torch.nn.functional.pad(t, (0, padding)) for t in param.chunk(self.num_devices)]

        torch_sharded_lst = [
            ttnn.create_group_norm_weight_bias_rm(t, self.num_padded_channels, self.num_virtual_cols) for t in params
        ]
        return torch.cat(torch_sharded_lst, dim=0)

    def forward(self, x: ttnn.Tensor, num_out_blocks=-1, compute_kernel_config=None) -> ttnn.Tensor:
        batch_size, height, width, channels = x.shape
        x = x.reshape([batch_size, 1, width * height, channels])
        kwargs = dict(
            weight=self.weight.data,
            bias=self.bias.data,
            input_mask=self.mask.data,
            num_groups=self.num_padded_groups,
            epsilon=self.eps,
            core_grid=self.core_grid,
            inplace=False,
            num_out_blocks=num_out_blocks,
            output_layout=ttnn.TILE_LAYOUT,
        )
        if compute_kernel_config is not None:
            kwargs["compute_kernel_config"] = compute_kernel_config
        x = ttnn.group_norm(x, **kwargs)
        x = x.reshape([batch_size, height, width, channels])

        return x


class DistributedGroupNorm(Module):
    """GroupNorm with optional spatial stats all-gather on ``cluster_axis``.

    ``cluster_axis=None``: local ``GroupNorm`` (including channel ``mesh_axis`` TP).
    ``cluster_axis=int``: fused ``dit_fused_distributed_groupnorm``
    (PRE → fabric AG on that axis → POST). Uses plain ``[1,1,1,C]`` γ/β
    (not the DRAM-packed GroupNorm layout). Optional ``mesh_axis`` still shards
    channels (whole groups per device); orthogonal to ``cluster_axis``.
    """

    def __init__(
        self,
        num_channels: int,
        num_groups: int,
        *,
        eps: float = 1e-5,
        mesh_device: ttnn.MeshDevice,
        cluster_axis: int | None = None,
        mesh_axis: int | None = None,
        ccl_manager=None,
        core_grid: ttnn.CoreGrid | None = None,
    ) -> None:
        super().__init__()

        assert num_channels % num_groups == 0, "num_channels must be divisible by num_groups"
        assert num_channels % 32 == 0, "num_channels must be divisible by tile size (fused v1)"

        self.num_channels = num_channels
        self.num_groups = num_groups
        self.eps = eps
        self.mesh_device = mesh_device
        self.cluster_axis = cluster_axis
        self.mesh_axis = mesh_axis
        self.ccl_manager = ccl_manager
        self.core_grid = core_grid or ttnn.CoreGrid(x=8, y=8)

        self.channel_devices = tuple(mesh_device.shape)[mesh_axis] if mesh_axis is not None else 1
        self.cluster_size = tuple(mesh_device.shape)[cluster_axis] if cluster_axis is not None else 1
        assert num_groups % self.channel_devices == 0, "num_groups must be divisible by channel mesh width"

        if cluster_axis is not None and self.cluster_size > 1:
            assert ccl_manager is not None, "ccl_manager is required when cluster_axis width > 1"

        self.num_local_channels = num_channels // self.channel_devices
        self.num_local_groups = num_groups // self.channel_devices

        if cluster_axis is None:
            # Existing GroupNorm path (packed γ/β/mask, channel TP).
            self.inner = GroupNorm(
                num_channels=num_channels,
                num_groups=num_groups,
                eps=eps,
                mesh_device=mesh_device,
                mesh_axis=mesh_axis,
                core_grid=self.core_grid,
            )
            self.weight = None
            self.bias = None
        else:
            # Fused path: reuses the welford GroupNorm kernels, so γ/β/mask are prepped exactly
            # like ttnn.group_norm (DRAM-packed RM γ/β + input_mask), but with a single worker
            # core per device (num_virtual_cols == 1).
            self.inner = None
            num_padded_channels = math.ceil(self.num_local_channels / 32) * 32
            assert num_padded_channels % self.num_local_channels == 0, "padded channels must divide channels"
            num_padded_groups = self.num_local_groups * (num_padded_channels // self.num_local_channels)
            self.num_virtual_cols = 1
            self.num_padded_channels = num_padded_channels
            self.num_padded_groups = num_padded_groups

            weight_shape = [
                self.channel_devices,
                1,
                math.ceil(num_padded_channels // self.num_virtual_cols / 32) * self.num_virtual_cols,
                32,
            ]
            block_wt = ttnn.operations.normalization.find_max_tile_span(
                num_padded_channels, num_padded_channels // num_padded_groups, 32
            )
            mask_shape = [1, num_padded_groups, 32, 32 * block_wt]

            self.weight = Parameter(
                total_shape=weight_shape,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_axes=[mesh_axis, None, None, None],
                device=mesh_device,
            )
            self.bias = Parameter(
                total_shape=weight_shape,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_axes=[mesh_axis, None, None, None],
                device=mesh_device,
            )
            self.mask = Parameter(total_shape=mask_shape, device=mesh_device)

    @classmethod
    def from_torch(
        cls,
        torch_ref: torch.nn.GroupNorm,
        *,
        mesh_device: ttnn.MeshDevice,
        cluster_axis: int | None = None,
        mesh_axis: int | None = None,
        ccl_manager=None,
        core_grid: ttnn.CoreGrid | None = None,
    ) -> DistributedGroupNorm:
        module = cls(
            num_channels=torch_ref.num_channels,
            num_groups=torch_ref.num_groups,
            eps=torch_ref.eps,
            mesh_device=mesh_device,
            cluster_axis=cluster_axis,
            mesh_axis=mesh_axis,
            ccl_manager=ccl_manager,
            core_grid=core_grid,
        )
        module.load_torch_state_dict(torch_ref.state_dict())
        return module

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        if self.inner is not None:
            renamed = {f"inner.{k}": v for k, v in list(state.items())}
            state.clear()
            state.update(renamed)
            return
        if "weight" in state:
            state["weight"] = self._prepare_param(state["weight"])
        if "bias" in state:
            state["bias"] = self._prepare_param(state["bias"])
        input_mask = ttnn.create_group_norm_input_mask(
            self.num_padded_channels, self.num_padded_groups, self.num_virtual_cols
        )
        state["mask"] = ttnn.to_torch(input_mask)

    def _prepare_param(self, param: torch.Tensor) -> torch.Tensor:
        expected_shape = (self.num_local_channels * self.channel_devices,)
        assert param.shape == expected_shape, f"expected shape {expected_shape}, got {param.shape}"
        padding = self.num_padded_channels - self.num_local_channels
        params = [torch.nn.functional.pad(t, (0, padding)) for t in param.chunk(self.channel_devices)]
        torch_sharded_lst = [
            ttnn.create_group_norm_weight_bias_rm(t, self.num_padded_channels, self.num_virtual_cols) for t in params
        ]
        return torch.cat(torch_sharded_lst, dim=0)

    def forward(self, x: ttnn.Tensor, num_out_blocks=-1, compute_kernel_config=None) -> ttnn.Tensor:
        if self.inner is not None:
            return self.inner.forward(x, num_out_blocks=num_out_blocks, compute_kernel_config=compute_kernel_config)

        batch_size, height, width, channels = x.shape
        assert (
            channels == self.num_local_channels
        ), f"last dim {channels} != num_local_channels {self.num_local_channels}"
        x4 = x.reshape([batch_size, 1, width * height, channels])

        semaphores = self.ccl_manager.get_ag_ping_pong_semaphore(self.cluster_axis) if self.cluster_size > 1 else []
        persistent = None
        if self.cluster_size > 1:
            persistent = self.ccl_manager.get_fused_norm_stats_buffer(
                ("gn", tuple(x4.shape), self.num_local_groups, self.cluster_axis),
                lambda: ttnn.experimental.dit_fused_distributed_groupnorm_create_stats_buffer(
                    x4,
                    self.num_local_groups,
                    self.cluster_axis,
                    self.mesh_device,
                    num_links=self.ccl_manager.num_links,
                ),
            )

        # Mirror GroupNorm.forward kwargs; cluster_axis / mesh / semaphore / topology
        # are the only distributed additions.
        out = ttnn.experimental.dit_fused_distributed_groupnorm(
            x4,
            num_groups=self.num_padded_groups,
            epsilon=self.eps,
            cluster_axis=self.cluster_axis,
            mesh_device=self.mesh_device,
            multi_device_global_semaphore=semaphores,
            topology=self.ccl_manager.topology if self.ccl_manager is not None else ttnn.Topology.Linear,
            input_mask=self.mask.data,
            weight=self.weight.data,
            bias=self.bias.data,
            persistent_output_buffer=persistent,
            num_preferred_links=self.ccl_manager.num_links if self.ccl_manager is not None else None,
            compute_kernel_config=compute_kernel_config,
        )
        return out.reshape([batch_size, height, width, channels])


class GroupNorm3D(Module):
    """``torch.nn.GroupNorm(num_groups, num_channels)`` on a 5D BTHWC tensor, dims=3
    semantics (statistics pool over ``channels-in-group x T x H x W`` per batch).

    Routes through the DRAM-interleaved ``ttnn.group_norm``. The grid is pinned at
    construction from ``input_nhw``/``num_batches`` via
    ``determine_expected_group_norm_dram_grid_size`` (uniform multicast groups; avoids
    the mcast deadlock at small spatial sizes), so gamma/beta/mask are static
    ``Parameter``s and round-trip through ``Module.save``/``load``.
    """

    def __init__(
        self,
        num_channels: int,
        num_groups: int,
        *,
        input_nhw: int,
        num_batches: int = 1,
        eps: float = 1e-5,
        mesh_device: ttnn.MeshDevice,
        dtype: ttnn.DataType = ttnn.bfloat16,
        use_welford: bool = True,
    ) -> None:
        super().__init__()
        assert num_channels % 32 == 0, "num_channels must be divisible by tile size"
        assert num_channels % num_groups == 0, "num_channels must be divisible by num_groups"

        self.num_channels = num_channels
        self.num_groups = num_groups
        self.eps = eps
        self.mesh_device = mesh_device
        self.dtype = dtype
        # Welford avoids the (E[x^2]-E[x]^2) precision loss when groups have nonzero mean.
        self.use_welford = use_welford
        self.input_nhw = input_nhw
        self.num_batches = num_batches

        self.core_grid = ttnn.determine_expected_group_norm_dram_grid_size(
            device=mesh_device,
            num_channels=num_channels,
            num_groups=num_groups,
            input_nhw=input_nhw,
            num_batches=num_batches,
        )
        self.num_virtual_cols = ttnn.operations.normalization.dram_group_norm_virtual_columns(
            self.core_grid, num_channels, num_groups
        )

        weight_shape = [1, 1, math.ceil(num_channels // self.num_virtual_cols / 32) * self.num_virtual_cols, 32]
        block_wt = ttnn.operations.normalization.find_max_tile_span(num_channels, num_channels // num_groups, 32)
        mask_shape = [1, num_groups, 32, 32 * block_wt]

        self.weight = Parameter(total_shape=weight_shape, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=dtype, device=mesh_device)
        self.bias = Parameter(total_shape=weight_shape, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=dtype, device=mesh_device)
        self.mask = Parameter(total_shape=mask_shape, dtype=dtype, device=mesh_device)

    @classmethod
    def from_torch(
        cls,
        torch_ref: torch.nn.GroupNorm,
        *,
        input_nhw: int,
        num_batches: int = 1,
        mesh_device: ttnn.MeshDevice,
        dtype: ttnn.DataType = ttnn.bfloat16,
    ) -> GroupNorm3D:
        module = cls(
            num_channels=torch_ref.num_channels,
            num_groups=torch_ref.num_groups,
            input_nhw=input_nhw,
            num_batches=num_batches,
            eps=torch_ref.eps,
            mesh_device=mesh_device,
            dtype=dtype,
        )
        module.load_torch_state_dict(torch_ref.state_dict())
        return module

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        if "weight" in state:
            state["weight"] = ttnn.create_group_norm_weight_bias_rm(
                state["weight"], self.num_channels, self.num_virtual_cols
            )
        if "bias" in state:
            state["bias"] = ttnn.create_group_norm_weight_bias_rm(
                state["bias"], self.num_channels, self.num_virtual_cols
            )
        mask = ttnn.create_group_norm_input_mask(self.num_channels, self.num_groups, self.num_virtual_cols)
        state["mask"] = ttnn.to_torch(mask)

    def forward(self, x_BTHWC: ttnn.Tensor) -> ttnn.Tensor:
        B, T, H, W, C = x_BTHWC.shape
        # dims=3: pool over (channels-in-group, T, H, W) — frames share one group statistic.
        THW = T * H * W
        assert B == self.num_batches and B * THW == self.input_nhw, (
            f"GroupNorm3D built for input_nhw={self.input_nhw}, num_batches={self.num_batches}; "
            f"got B={B}, T*H*W={THW}"
        )

        if x_BTHWC.layout != ttnn.ROW_MAJOR_LAYOUT:
            x_BTHWC = ttnn.to_layout(x_BTHWC, ttnn.ROW_MAJOR_LAYOUT)
        x = ttnn.tilize_with_zero_padding(ttnn.reshape(x_BTHWC, (B, 1, THW, C)), use_multicore=True)

        out = ttnn.group_norm(
            x,
            num_groups=self.num_groups,
            # -1 = built-in chunk heuristic. Default 1 (with pinned core_grid) overflows L1
            # at large gathered spatial.
            num_out_blocks=-1,
            input_mask=self.mask.data,
            weight=self.weight.data,
            bias=self.bias.data,
            epsilon=self.eps,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_layout=ttnn.TILE_LAYOUT,
            core_grid=self.core_grid,
            inplace=False,
            use_welford=self.use_welford,
        )

        out = ttnn.to_layout(out, ttnn.ROW_MAJOR_LAYOUT)
        return ttnn.reshape(out, (B, T, H, W, C))
