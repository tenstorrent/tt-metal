# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from typing import ClassVar

import torch
from loguru import logger

import ttnn

from .linear import _apply_activation_fn
from .module import Module, Parameter

# Activations the fused distributed GroupNorm op can absorb into its output stage.
# Anything else is applied as a separate op, exactly like Linear's unfused activations.
_FUSABLE_GROUP_NORM_ACTIVATIONS = frozenset({"silu"})


def plain_group_norm_virtual_cols(grid_x: int, groups: int, channels: int) -> int:
    """``compute_num_virtual_cols`` of ``ttnn.group_norm`` (``groupnorm_grid_utils.cpp``): the largest
    ``nvc <= min(grid_x, groups)`` with ``(channels / nvc) % 32 == 0`` and ``groups % nvc == 0``; 0 when none."""
    nvc = min(grid_x, groups)
    while nvc > 0 and ((channels // nvc) % 32 != 0 or groups % nvc != 0):
        nvc -= 1
    return nvc


def plain_group_norm_grid(
    rows: int, channels: int, groups: int, *, grid_x: int = 8, grid_y: int = 8, batch: int = 1
) -> tuple[int, int]:
    """The ``(x, y)`` core grid the plain (DRAM, interleaved) ``ttnn.group_norm`` accepts for ``rows`` (= H * W) rows
    of ``channels`` channels in ``groups`` groups: ``grid_x x grid_y`` itself when its rule holds, else the largest
    ``grid_x x y`` below it. The rule (``groupnorm.cpp`` ``validate_dram_grid``, ``groupnorm_grid_utils.cpp``
    ``find_expected_dram_grid``): ``nvc = compute_num_virtual_cols(grid_x, groups, channels)``, ``num_virtual_rows =
    (grid_x // nvc) * y``; valid iff ``Ht >= num_virtual_rows``, ``Ht % num_virtual_rows == 0`` and (``num_virtual_rows
    < batch`` or ``num_virtual_rows % batch == 0``), ``Ht = ceil(rows / 32)``. Only ``y`` is lowered: the x extent fixes
    ``nvc`` and with it the packed gamma / beta / mask layout a ``GroupNorm`` module prepared at construction. Raises
    ``ValueError`` when no ``y`` serves the shape (then the row count is not tile-aligned or too small)."""
    ht = -(-rows // 32)
    nvc = plain_group_norm_virtual_cols(grid_x, groups, channels)
    if nvc == 0 or grid_x // nvc == 0:
        raise ValueError(f"group_norm: no virtual-column split of {channels} channels / {groups} groups on x={grid_x}")
    rows_per_y = grid_x // nvc
    for y in range(grid_y, 0, -1):
        nvr = rows_per_y * y
        if ht >= nvr and ht % nvr == 0 and (nvr < batch or nvr % batch == 0):
            return grid_x, y
    raise ValueError(
        f"group_norm: no core grid with x={grid_x} serves Ht={ht} tiles ({rows} rows) of {channels} channels / "
        f"{groups} groups (num_virtual_cols {nvc}, rows per y {rows_per_y})"
    )


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
        dynamic_weight=None,
        dynamic_bias=None,
        per_head_norm=False,
        dynamic_weight_includes_static=False,
        dynamic_tile_row_map=None,
    ) -> ttnn.Tensor:
        # per_head_norm selects the normalization semantics when the activation is
        # head-split (num_heads_per_device > 1):
        #   True  -> RMSNorm INDEPENDENTLY over each head's head_dim (per-head QK-norm, e.g.
        #            Ideogram4). No cross-device all-gather (each head is device-local).
        #   False -> one RMSNorm over the FULL per-device row, then reshape to heads
        #            (WAN2.2 / LTX "norm before splitting heads").
        # The two are NOT equivalent. Default is whole-row (False); per-head models must opt
        # in with per_head_norm=True at the call site (e.g. the Ideogram4 QK-norm).
        expected_dim = self.embedding_dim // self.mesh_width
        if x.shape[-1] != expected_dim:
            msg = (
                f"last dimension of input tensor with shape {tuple(x.shape)} should match "
                f"embedding_dim / mesh_width = {expected_dim}"
            )
            raise ValueError(msg)

        # Effective affine weight: the static per-channel weight, optionally modulated by a
        # per-sample dynamic weight (e.g. an adaLN (1 + scale) factor), unless the caller already
        # folded the static weight into it (`dynamic_weight_includes_static`). Folding it in here
        # applies the modulation inside the fused op (fp32 internals) and removes the separate
        # elementwise scale op the caller would otherwise need. RMSNorm has no bias term.
        weight = self.weight.data if self.weight is not None else None
        if dynamic_weight is not None:
            if weight is None or dynamic_weight_includes_static:
                weight = dynamic_weight
            else:
                weight = ttnn.multiply(weight, dynamic_weight)
        weight_key = tuple(weight.shape) if weight is not None else None

        # dynamic_bias is the additive half of an adaLN modulation (the `shift`), folded into the same
        # fused op as the scale so the caller needs no separate elementwise add. The device op accepts
        # a per-token bias of shape [.., N, H] alongside a per-token weight; it requires a weight
        # whenever a bias is given. Unlike the weight it does not reach create_stats_buffer, because
        # only the weight can change the stats scratch geometry -- hence it is absent from the cache key.
        if dynamic_bias is not None and weight is None:
            msg = "dynamic_bias requires a weight: pass dynamic_weight or build the norm with affine=True"
            raise ValueError(msg)

        extra = {} if dynamic_tile_row_map is None else {"affine_tile_row_map": dynamic_tile_row_map}

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
                # per_head_norm changes the stats geometry (per-head reduces locally -> no
                # all-gather scratch), so it MUST be part of the key and forwarded to
                # create_stats_buffer (which returns None for the per-head/local path).
                ("rms", tuple(x.shape), num_heads_per_device, per_head_norm, rope_cos is not None, weight_key),
                lambda: ttnn.experimental.dit_fused_distributed_rmsnorm_create_stats_buffer(
                    x,
                    self.mesh_axis,
                    self.mesh_device,
                    num_heads_per_device=num_heads_per_device,
                    per_head_norm=per_head_norm,
                    num_links=self.ccl_manager.num_links,
                    weight=weight,
                    transformation_mat=trans_mat,
                    rope_cos=rope_cos,
                    rope_sin=rope_sin,
                ),
            ),
            epsilon=self.norm_eps,
            num_heads_per_device=num_heads_per_device,
            per_head_norm=per_head_norm,
            weight=weight,
            bias=dynamic_bias,
            compute_kernel_config=compute_kernel_config or self.compute_kernel_config,
            num_preferred_links=self.ccl_manager.num_links,  # must match create_stats_buffer above
            transformation_mat=trans_mat,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
            dtype=dtype,
            **extra,
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
        holding [1/1..1/width] (replicated per device). Cached per (device, width).
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
    }  # overrides the num_out_blocks computed from the input shape.

    def __init__(
        self,
        num_channels: int,
        num_groups: int,
        *,
        eps: float = 1e-5,
        mesh_device: ttnn.MeshDevice,
        mesh_axis: int | None = None,
        core_grid: ttnn.CoreGrid | None = None,
        activation_fn: str | None = None,
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
            activation_fn: Applied after the norm. ``ttnn.group_norm`` cannot fuse one, so it
                always runs as a separate op here.
        """
        super().__init__()

        self.activation_fn = activation_fn
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
        self._gamma_torch = None  # plain affine for the exact unaligned-row path (set by _prepare_torch_state)
        self._beta_torch = None
        self._unaligned_cache = None

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
        # The plain per-channel affine, kept for the exact unaligned-row path (``_forward_unaligned``).
        self._gamma_torch = state["weight"].detach().float().clone() if "weight" in state else None
        self._beta_torch = state["bias"].detach().float().clone() if "bias" in state else None
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

    def _core_grid_for(self, rows: int, batch_size: int) -> ttnn.CoreGrid:
        """The core grid for ``rows`` (= H * W) input rows: ``self.core_grid`` when ``ttnn.group_norm``'s DRAM grid
        rule accepts it, else the largest valid sub-grid with the SAME x (so ``num_virtual_cols`` and the packed
        gamma / beta / mask layout are unchanged). Additive: every shape the fixed grid accepted keeps it (bit-identical
        to before); a shape it rejected (e.g. ``Ht`` 490 / 1960 / 504 tiles of 128 channels, which needed 16 virtual
        rows) used to ``TT_THROW`` in ``groupnorm.cpp`` ``validate_dram_grid`` and now runs on the sub-grid the same
        rule suggests. Cached per ``(rows, batch)``; logged once per shape."""
        if not hasattr(self, "_grid_cache"):
            self._grid_cache: dict[tuple[int, int], ttnn.CoreGrid] = {}
        key = (int(rows), int(batch_size))
        grid = self._grid_cache.get(key)
        if grid is None:
            gx, gy = plain_group_norm_grid(
                rows,
                self.num_padded_channels,
                self.num_padded_groups,
                grid_x=self.core_grid.x,
                grid_y=self.core_grid.y,
                batch=batch_size,
            )
            grid = self.core_grid if (gx, gy) == (self.core_grid.x, self.core_grid.y) else ttnn.CoreGrid(x=gx, y=gy)
            if grid is not self.core_grid:
                logger.info(
                    f"GroupNorm: rows {rows} (Ht {math.ceil(rows / 32)}) x {self.num_padded_channels} channels / "
                    f"{self.num_padded_groups} groups: core grid {self.core_grid.x}x{self.core_grid.y} is invalid for "
                    f"ttnn.group_norm's DRAM grid rule; using {gx}x{gy}"
                )
            self._grid_cache[key] = grid
        return grid

    # --- exact path for row counts that are not a multiple of the tile height -------------------------------------------
    # ``ttnn.group_norm`` reduces whole 32-row tiles, so an H * W that is not a multiple of 32 would fold the padding
    # rows into the statistics; the Klein admission policy refused such shapes (BFL's own reference sizing produces
    # them for most real photos: 4032x3024 -> 2352x1760, 1000x1000 -> 1008x1008). This path computes the statistics
    # explicitly in fp32 over the REAL rows (zero padding contributes nothing to the sums and the divisor is the true
    # count), normalises in fp32 and rounds to bf16 once: bf16 operands, fp32 accumulation, like the fused op.

    def _unaligned_consts(self):
        """Per-device fp32 affine [1, 1, 1, C_local], the group indicator [C_local, G_local] and its transpose."""
        if getattr(self, "_unaligned_cache", None) is not None:
            return self._unaligned_cache
        if self._gamma_torch is None or self._beta_torch is None:
            # Weights came from the tensor cache (no torch state passed through _prepare_torch_state): recover the
            # plain affine from the packed row-major device parameters (the inverse of create_group_norm_weight_bias_rm:
            # per virtual column, C_padded / nvc channels laid out in 32-wide rows).
            self._gamma_torch = self._unpack_affine(self.weight.data)
            self._beta_torch = self._unpack_affine(self.bias.data)
        assert (
            self.num_padded_channels == self.num_local_channels
        ), f"unaligned path needs tile-aligned local channels, got {self.num_local_channels}"
        d, c = self.num_devices, self.num_local_channels
        g_local = self.num_padded_groups
        cg = c // g_local
        mesh = self.mesh_device
        shape = tuple(int(v) for v in mesh.shape)
        if d > 1:
            dims = (0, None) if self.mesh_axis == 0 else (None, 0)
            mapper = ttnn.ShardTensor2dMesh(mesh, mesh_shape=shape, dims=dims)
        else:
            mapper = ttnn.ReplicateTensorToMesh(mesh)
        rep = ttnn.ReplicateTensorToMesh(mesh)

        def up(t, mp):
            return ttnn.from_torch(
                t.float(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=mp,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        gamma = up(self._gamma_torch.reshape(d, 1, 1, c), mapper)
        beta = up(self._beta_torch.reshape(d, 1, 1, c), mapper)
        ind = torch.zeros(c, g_local)
        for ch in range(c):
            ind[ch, ch // cg] = 1.0
        m = up(ind.reshape(1, 1, c, g_local), rep)
        mt = up(ind.T.contiguous().reshape(1, 1, g_local, c), rep)
        cfg = ttnn.init_device_compute_kernel_config(
            mesh.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False
        )
        self._unaligned_cache = (gamma, beta, m, mt, cg, cfg)
        return self._unaligned_cache

    def _unpack_affine(self, packed: ttnn.Tensor) -> torch.Tensor:
        """``[num_devices * C_local]`` plain affine (device order along ``mesh_axis``) from the packed parameter."""
        mesh = self.mesh_device
        shape = tuple(int(v) for v in mesh.shape)
        full = ttnn.to_torch(packed, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)).float()  # [n_all, 1, rows, 32]
        full = full.reshape(-1, full.shape[-2], full.shape[-1])
        if self.num_devices > 1:
            # shards along mesh_axis are replicated along the other axis: take one replica per shard
            stride = shape[1] if self.mesh_axis == 0 else 1
            picks = [full[i * stride] for i in range(self.num_devices)]
        else:
            picks = [full[0]]
        nvc = int(self.num_virtual_cols)
        cpc = self.num_padded_channels // nvc
        out = []
        for t in picks:
            rows = t.shape[0]
            rpc = rows // nvc
            chans = torch.cat([t[j * rpc : (j + 1) * rpc].reshape(-1)[:cpc] for j in range(nvc)])
            out.append(chans[: self.num_local_channels])
        return torch.cat(out)

    def _forward_unaligned(self, x: ttnn.Tensor, batch_size: int, height: int, width: int, channels: int) -> ttnn.Tensor:
        gamma, beta, m, mt, cg, cfg = self._unaligned_consts()
        n = height * width
        n32 = math.ceil(n / 32) * 32
        xf = ttnn.typecast(x, ttnn.float32)
        if n32 != n:
            xf = ttnn.pad(xf, [(0, 0), (0, 0), (0, n32 - n), (0, 0)], 0.0)
        cnt = float(n * cg)
        s1 = ttnn.sum(xf, dim=2, keepdim=True)  # [B, 1, 1, C]: zero rows add nothing
        mean_g = ttnn.multiply(ttnn.matmul(s1, m, compute_kernel_config=cfg), 1.0 / cnt)  # [B, 1, 1, G]
        ttnn.deallocate(s1)
        mean_c = ttnn.matmul(mean_g, mt, compute_kernel_config=cfg)  # [B, 1, 1, C]
        ttnn.deallocate(mean_g)
        xc = ttnn.subtract(xf, mean_c)  # the padding rows become -mean: masked out of the variance below
        ttnn.deallocate(xf)
        ttnn.deallocate(mean_c)
        if n32 != n:
            keep = torch.zeros(1, 1, n32, 1)
            keep[:, :, :n, :] = 1.0
            keep_t = ttnn.from_torch(
                keep, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=self.mesh_device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
            xc = ttnn.multiply(xc, keep_t)
            ttnn.deallocate(keep_t)
        s2 = ttnn.sum(ttnn.multiply(xc, xc), dim=2, keepdim=True)
        var_g = ttnn.multiply(ttnn.matmul(s2, m, compute_kernel_config=cfg), 1.0 / cnt)
        ttnn.deallocate(s2)
        rstd_g = ttnn.rsqrt(ttnn.add(var_g, self.eps))
        ttnn.deallocate(var_g)
        rstd_c = ttnn.matmul(rstd_g, mt, compute_kernel_config=cfg)
        ttnn.deallocate(rstd_g)
        y = ttnn.multiply(xc, rstd_c)
        ttnn.deallocate(xc)
        ttnn.deallocate(rstd_c)
        y = ttnn.add(ttnn.multiply(y, gamma), beta)
        if n32 != n:
            y = ttnn.slice(y, [0, 0, 0, 0], [batch_size, 1, n, channels])
        y = ttnn.typecast(y, ttnn.bfloat16)
        y = y.reshape([batch_size, height, width, channels])
        return _apply_activation_fn(y, self.activation_fn)

    def forward(self, x: ttnn.Tensor, num_out_blocks=-1, compute_kernel_config=None) -> ttnn.Tensor:
        batch_size, height, width, channels = x.shape
        x = x.reshape([batch_size, 1, width * height, channels])
        if (width * height) % 32 != 0:
            return self._forward_unaligned(x, batch_size, height, width, channels)
        try:
            core_grid = self._core_grid_for(width * height, batch_size)
        except ValueError as err:
            # A tile-aligned row count the fused op's DRAM grid rule cannot serve (e.g. 256,960 rows = 8030 tiles of 32
            # channels / 8 groups, the /2 level of an 880x1168 reference): the exact path handles any row count.
            logger.info(f"GroupNorm: {err}; using the exact unaligned path")
            return self._forward_unaligned(x, batch_size, height, width, channels)
        kwargs = dict(
            weight=self.weight.data,
            bias=self.bias.data,
            input_mask=self.mask.data,
            num_groups=self.num_padded_groups,
            epsilon=self.eps,
            core_grid=core_grid,
            inplace=False,
            num_out_blocks=num_out_blocks,
            output_layout=ttnn.TILE_LAYOUT,
        )
        if compute_kernel_config is not None:
            kwargs["compute_kernel_config"] = compute_kernel_config
        x = ttnn.group_norm(x, **kwargs)
        x = x.reshape([batch_size, height, width, channels])

        return _apply_activation_fn(x, self.activation_fn)


class DistributedGroupNorm(Module):
    """GroupNorm with optional spatial stats all-gather on ``cluster_axis``.

    ``cluster_axis=None``: local ``GroupNorm`` (including channel ``mesh_axis`` TP).
    ``cluster_axis=int``: fused ``dit_fused_distributed_groupnorm``
    (PRE → fabric AG on that axis → POST). γ/β and the input mask use the same
    DRAM-packed row-major GroupNorm layout as ``ttnn.group_norm``, with
    ``num_virtual_cols == 1``. Optional ``mesh_axis`` still shards channels
    (whole groups per device); orthogonal to ``cluster_axis``. Per-device
    channel count must be divisible by 32.

    ``activation_fn`` (e.g. ``"silu"``) is applied after the norm. On the distributed path it
    is folded into the op's output stage, removing the separate activation op's full-tensor
    DRAM read/write; otherwise it runs as a separate op. Either way the caller gets
    ``activation(group_norm(x))`` and does not need to know which happened.
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
        activation_fn: str | None = None,
    ) -> None:
        super().__init__()

        assert num_channels % num_groups == 0, "num_channels must be divisible by num_groups"
        assert num_channels % 32 == 0, "num_channels must be divisible by tile size (fused v1)"

        # Consume the activation into the op when the fused path can absorb it; whatever is
        # left in self.activation_fn is applied separately by forward().
        self.activation_fn = activation_fn
        self.fused_activation = None
        if cluster_axis is not None and activation_fn in _FUSABLE_GROUP_NORM_ACTIVATIONS:
            self.fused_activation = activation_fn
            self.activation_fn = None

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

        if cluster_axis is not None:
            assert (
                self.num_local_channels % 32 == 0
            ), f"fused path requires per-device C % 32 == 0, got {self.num_local_channels}"
            assert core_grid is None, (
                "core_grid is not supported on the fused path (cluster_axis is not None): "
                "dit_fused_distributed_groupnorm always uses the full compute grid"
            )

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
        activation_fn: str | None = None,
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
            activation_fn=activation_fn,
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
            out = self.inner.forward(x, num_out_blocks=num_out_blocks, compute_kernel_config=compute_kernel_config)
            return _apply_activation_fn(out, self.activation_fn)

        assert num_out_blocks == -1, (
            "num_out_blocks is not supported on the fused path (cluster_axis is not None): "
            "dit_fused_distributed_groupnorm derives its own block split"
        )

        batch_size, height, width, channels = x.shape
        assert (
            channels == self.num_local_channels
        ), f"last dim {channels} != num_local_channels {self.num_local_channels}"
        x4 = x.reshape([batch_size, 1, width * height, channels])

        semaphores = self.ccl_manager.get_ag_ping_pong_semaphore(self.cluster_axis) if self.cluster_size > 1 else []
        persistent = None
        if self.cluster_size > 1:
            # The fused activation does not change the stats geometry (it is applied after the
            # AG, on the output tile), so it is deliberately NOT part of the cache key.
            persistent = self.ccl_manager.get_fused_norm_stats_buffer(
                ("gn", tuple(x4.shape), self.num_padded_groups, self.cluster_axis),
                lambda: ttnn.experimental.dit_fused_distributed_groupnorm_create_stats_buffer(
                    x4,
                    self.num_padded_groups,
                    self.cluster_axis,
                    self.mesh_device,
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
            compute_kernel_config=compute_kernel_config,
            activation=self.fused_activation,
        )
        out = out.reshape([batch_size, height, width, channels])
        # No-op when the activation was consumed into the op above.
        return _apply_activation_fn(out, self.activation_fn)


class GroupNorm3D(Module):
    """``torch.nn.GroupNorm(num_groups, num_channels)`` on a 5D BTHWC tensor, dims=3
    semantics (statistics pool over ``channels-in-group x T x H x W`` per batch).

    Routes through the DRAM-interleaved ``ttnn.group_norm``. The grid is pinned at
    construction from ``input_nhw``/``num_batches`` via
    ``determine_expected_group_norm_dram_grid_size`` (uniform multicast groups; avoids
    the mcast deadlock at small spatial sizes), so gamma/beta are static
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

        self.weight = Parameter(total_shape=weight_shape, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=dtype, device=mesh_device)
        self.bias = Parameter(total_shape=weight_shape, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=dtype, device=mesh_device)

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
