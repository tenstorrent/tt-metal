# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import math

import torch
import ttnn

from models.common.lightweightmodule import LightweightModule


def _interleaved(x: ttnn.Tensor, shape, *, row_major: bool) -> ttnn.Tensor:
    """Gather to DRAM interleaved and reshape, optionally RM."""
    x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
    if row_major:
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
    return ttnn.reshape(x, shape)


_SHARD_L1_BUDGET_BYTES = 48 * 1024


def _shard_height(device, nhw: int) -> int:
    """Compute height-shard row count rounded to 32."""
    grid = device.compute_with_storage_grid_size()
    ncores = int(grid.x) * int(grid.y)
    return math.ceil(math.ceil(nhw / ncores) / 32) * 32


def sharded_chain_fits_l1(device, length: int, channels: int, dtype_bytes: int = 4) -> bool:
    """Return whether a height-sharded chain fits L1 budget."""
    return _shard_height(device, length) * channels * dtype_bytes <= _SHARD_L1_BUDGET_BYTES


def height_shard_l1(device, x: ttnn.Tensor, channels: int) -> ttnn.Tensor:
    """Reshard a tensor to height-sharded L1."""
    mem = ttnn.create_sharded_memory_config(
        shape=(_shard_height(device, x.shape[-2]), channels),
        core_grid=ttnn.CoreGrid(
            y=int(device.compute_with_storage_grid_size().y), x=int(device.compute_with_storage_grid_size().x)
        ),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    return ttnn.to_memory_config(x, mem)


def block_shard_grid(device, length: int, channels: int):
    """Plan block-shard grid for length and channels."""
    grid = device.compute_with_storage_grid_size()
    if channels % 32:
        return None
    gx, gy = channels // 32, int(grid.y)
    if gx < 2 or gx > int(grid.x):
        return None
    rows_per_core = math.ceil(math.ceil(length / gy) / 32) * 32
    if rows_per_core * gy < length:
        return None
    return gx, gy, rows_per_core


def block_chain_fits_l1(device, length: int, channels: int, dtype_bytes: int = 4) -> bool:
    """Return whether a block-sharded chain fits L1 budget."""
    plan = block_shard_grid(device, length, channels)
    if plan is None:
        return False
    _, _, rows_per_core = plan
    return rows_per_core * 32 * dtype_bytes <= _SHARD_L1_BUDGET_BYTES


def block_shard_l1(device, x: ttnn.Tensor, channels: int) -> ttnn.Tensor:
    """Reshard a tensor to block-sharded L1."""
    gx, gy, rows_per_core = block_shard_grid(device, x.shape[-2], channels)
    mem = ttnn.create_sharded_memory_config(
        shape=(rows_per_core, channels // gx),
        core_grid=ttnn.CoreGrid(y=gy, x=gx),
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    return ttnn.to_memory_config(x, mem)


def _subpixel_weight(weight: torch.Tensor, bias: torch.Tensor | None, stride: int):
    """Convert transpose-conv weights to sub-pixel conv form."""
    in_ch, out_ch, k = weight.shape
    pad_t = (k - stride) // 2
    phases = []
    for phi in range(stride):
        j0 = (phi + pad_t) % stride
        idxs = list(range(j0, k, stride))
        d = (phi + pad_t - j0) // stride
        w = torch.flip(weight[:, :, idxs], dims=[-1]).permute(1, 0, 2).contiguous()
        phases.append((w, w.shape[-1] - 1 - d, d))
    pad_l = max(p[1] for p in phases)
    pad_r = max(p[2] for p in phases)
    assert pad_l == pad_r, f"expected symmetric common padding, got {pad_l} vs {pad_r}"
    ic = pad_l + pad_r + 1
    weight_sp = torch.zeros(stride * out_ch, in_ch, ic)
    for phi, (w, p_l, _) in enumerate(phases):
        off = pad_l - p_l
        weight_sp[phi * out_ch : (phi + 1) * out_ch, :, off : off + w.shape[-1]] = w
    bias_sp = bias.repeat(stride) if bias is not None else None
    return weight_sp, bias_sp, pad_l


class TtConv1d(LightweightModule):
    def __init__(
        self,
        device,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        *,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        activation: ttnn.UnaryWithParam | None = None,
        weights_dtype: ttnn.DataType = ttnn.bfloat16,
        activations_dtype: ttnn.DataType = ttnn.float32,
        math_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en: bool = True,
        packer_l1_acc: bool = True,
        act_double_buffer: bool | None = None,
        weight_scale: float = 1.0,
        conv_config_overrides: dict | None = None,
    ):
        """Prepare a 1D convolution with optional folded bias."""
        super().__init__()
        assert weight.dim() == 3, f"expected Conv1d weight [out, in/groups, k], got {tuple(weight.shape)}"
        out_channels, in_per_group, kernel_size = weight.shape
        if weight_scale != 1.0:
            weight = weight * weight_scale

        self.device = device
        self.in_channels = in_per_group * groups
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.activations_dtype = activations_dtype

        self.tt_weight = ttnn.from_torch(weight.float(), weights_dtype)
        self.tt_bias = None
        self._raw_bias_fp32 = None
        if bias is not None:
            self.tt_bias = ttnn.from_torch(bias.reshape(1, 1, 1, -1).float(), weights_dtype)
            self._raw_bias_fp32 = ttnn.from_torch(
                bias.reshape(1, 1, 1, -1).float(), ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device
            )
        self._host_weight, self._host_bias = self.tt_weight, self.tt_bias
        self._prepared_for = None
        # Never evict: captured traces keep reads of these prepared biases.
        self._folded_bias = {}

        self.conv_config = ttnn.Conv1dConfig(
            weights_dtype=weights_dtype,
            deallocate_activation=False,
            activation=activation,
            **({"enable_act_double_buffer": act_double_buffer} if act_double_buffer is not None else {}),
        )
        if conv_config_overrides:
            for _k, _v in conv_config_overrides.items():
                setattr(self.conv_config, _k, _v)
        self.compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=math_fidelity,
            fp32_dest_acc_en=fp32_dest_acc_en,
            packer_l1_acc=packer_l1_acc,
        )

    def forward(self, x: ttnn.Tensor, cond_bias: ttnn.Tensor | None = None, keep_sharded: bool = False) -> ttnn.Tensor:
        """Run conv1d, folding speaker cond bias when provided."""
        batch_size, input_length, _ = x.shape
        # Prepared weights are shape-specific; ttnn cannot detect a stale cache.
        key = (batch_size, input_length, x.dtype, x.layout, x.memory_config())
        if key != self._prepared_for:
            self.tt_weight, self.tt_bias = self._host_weight, self._host_bias
        # The fold calls from_device, which is fatal inside a trace — warm up before capturing so
        # the fold lands in _folded_bias and the captured run only reads it.
        fold = cond_bias is not None
        bias_tensor = self.tt_bias
        fold_key = (id(cond_bias), key) if fold else None
        cached = self._folded_bias.get(fold_key) if fold else None
        if cached is not None and cached[0] is cond_bias:
            bias_tensor = cached[1]
        elif fold:
            combined = ttnn.to_layout(ttnn.add(self._raw_bias_fp32, cond_bias), ttnn.ROW_MAJOR_LAYOUT)
            bias_tensor = ttnn.from_device(combined)
            ttnn.deallocate(combined)
        out, out_length, [weight, bias] = ttnn.conv1d(
            input_tensor=x,
            weight_tensor=self.tt_weight,
            bias_tensor=bias_tensor,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            device=self.device,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            batch_size=batch_size,
            input_length=input_length,
            dtype=self.activations_dtype,
            conv_config=self.conv_config,
            compute_config=self.compute_config,
            return_output_dim=True,
            return_weights_and_bias=True,
        )
        self.tt_weight = weight
        if fold:
            self._folded_bias[fold_key] = (cond_bias, bias)
        else:
            self.tt_bias = bias
        self._prepared_for = key
        if keep_sharded:
            return ttnn.reshape(out, [batch_size, out_length, self.out_channels])
        return _interleaved(out, [batch_size, out_length, self.out_channels], row_major=False)


class TtConvTranspose1d(LightweightModule):
    def __init__(
        self,
        device,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        *,
        stride: int,
        **conv_kwargs,
    ):
        """Build a sub-pixel transpose-conv via inner TtConv1d."""
        super().__init__()
        assert weight.dim() == 3, f"expected ConvTranspose1d weight [in, out, k], got {tuple(weight.shape)}"
        in_channels, out_channels, kernel_size = weight.shape
        assert (kernel_size - stride) % 2 == 0, f"need (k - stride) even, got k={kernel_size}, stride={stride}"

        self.stride = stride
        self.out_channels = out_channels

        weight_sp, bias_sp, padding = _subpixel_weight(weight, bias, stride)
        self.conv = TtConv1d(device, weight_sp, bias_sp, stride=1, padding=padding, **conv_kwargs)
        self._inner_cond_cache = {}

    def _inner_cond(self, cond_bias: ttnn.Tensor) -> ttnn.Tensor:
        """Tile and cache stride-repeated conditioning bias."""
        hit = self._inner_cond_cache.get(id(cond_bias))
        if hit is not None and hit[0] is cond_bias:
            return hit[1]
        tiled = ttnn.concat([cond_bias] * self.stride, dim=-1)
        self._inner_cond_cache[id(cond_bias)] = (cond_bias, tiled)
        return tiled

    def release_cond_cache(self):
        """Free cached conditioning bias tensors."""
        for _, tiled in self._inner_cond_cache.values():
            if tiled.is_allocated():
                ttnn.deallocate(tiled)
        self._inner_cond_cache.clear()
        self.conv._folded_bias.clear()

    def forward(self, x: ttnn.Tensor, cond_bias: ttnn.Tensor | None = None) -> ttnn.Tensor:
        """Run transpose-conv via sub-pixel conv and reshape."""
        batch_size, input_length, _ = x.shape
        inner_cond = self._inner_cond(cond_bias) if cond_bias is not None else None
        z = self.conv(x, cond_bias=inner_cond, keep_sharded=self.stride <= 2)

        shape = [batch_size, input_length * self.stride, self.out_channels]
        if self.stride <= 2:
            return ttnn.reshape(z, shape)
        z = ttnn.to_layout(z, ttnn.ROW_MAJOR_LAYOUT)
        z = ttnn.reshape(z, shape)
        return ttnn.to_layout(z, ttnn.TILE_LAYOUT)


class TtConv2d(LightweightModule):
    def __init__(
        self,
        device,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        *,
        stride: int = 1,
        padding: int = 1,
        activation: ttnn.UnaryWithParam | None = None,
        weights_dtype: ttnn.DataType = ttnn.bfloat16,
        activations_dtype: ttnn.DataType = ttnn.float32,
        math_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en: bool = True,
        packer_l1_acc: bool = True,
    ):
        """Prepare a 2D convolution with double-buffer defaults."""
        super().__init__()
        assert weight.dim() == 4, f"expected Conv2d weight [out, in, kh, kw], got {tuple(weight.shape)}"
        out_channels, in_channels, kh, kw = weight.shape

        self.device = device
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kh, kw)
        self.stride = (stride, stride)
        self.padding = (padding, padding)
        self.activations_dtype = activations_dtype

        self.tt_weight = ttnn.from_torch(weight.float(), weights_dtype)
        self.tt_bias = None
        if bias is not None:
            self.tt_bias = ttnn.from_torch(bias.reshape(1, 1, 1, -1).float(), weights_dtype)
        self._host_weight, self._host_bias = self.tt_weight, self.tt_bias
        self._prepared_for = None

        self.conv_config = ttnn.Conv2dConfig(
            weights_dtype=weights_dtype,
            deallocate_activation=False,
            activation=activation,
            output_layout=ttnn.TILE_LAYOUT,
            config_tensors_in_dram=True,  # L1_SMALL OOM with many convs (speaker encoder)
            # full_inner_dim=True silently wrong for mel_len<128; do not enable.
            enable_act_double_buffer=True,
            enable_weights_double_buffer=True,
        )
        self.compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=math_fidelity,
            fp32_dest_acc_en=fp32_dest_acc_en,
            packer_l1_acc=packer_l1_acc,
        )

    def set_double_buffer(self, enabled: bool) -> None:
        """Enable or disable activation and weight double buffering."""
        self.conv_config.enable_act_double_buffer = enabled
        self.conv_config.enable_weights_double_buffer = enabled

    def forward(
        self,
        x: ttnn.Tensor,
        input_height: int,
        input_width: int,
        memory_config: ttnn.MemoryConfig | None = None,
    ) -> tuple[ttnn.Tensor, int, int]:
        # Prepared weights are shape-specific; ttnn cannot detect a stale cache. The double-buffer
        # flags belong in the key too: strided layers map several input lengths onto the same
        # height, so a toggle can land on an otherwise identical key.
        """Run conv2d for the given input height and width."""
        key = (
            x.shape[0],
            input_height,
            input_width,
            x.dtype,
            x.layout,
            x.memory_config(),
            memory_config,
            self.conv_config.enable_act_double_buffer,
            self.conv_config.enable_weights_double_buffer,
        )
        if key != self._prepared_for:
            self.tt_weight, self.tt_bias = self._host_weight, self._host_bias
        out, (out_h, out_w), [weight, bias] = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=self.tt_weight,
            bias_tensor=self.tt_bias,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            device=self.device,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            batch_size=x.shape[0],
            input_height=input_height,
            input_width=input_width,
            dtype=self.activations_dtype,
            conv_config=self.conv_config,
            compute_config=self.compute_config,
            memory_config=memory_config,
            return_output_dim=True,
            return_weights_and_bias=True,
        )
        self.tt_weight = weight
        self.tt_bias = bias
        self._prepared_for = key
        return out, out_h, out_w
