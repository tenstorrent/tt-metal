# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI U.S. Corp., for the TTNN port. Reference code © Meta Platforms, Inc. (Apache-2.0).
# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass

import torch

import ttnn
from models.tt_cnn.tt.builder import Conv2dConfiguration, HeightShardedStrategyConfiguration, TtConv2d

HIERA_STATE_DTYPE = ttnn.bfloat16
HIERA_MLP_HIDDEN_DTYPE = ttnn.bfloat8_b
HIERA_QKV_DTYPE = ttnn.bfloat8_b
HIERA_RESIDUAL_BRANCH_DTYPE = ttnn.bfloat16
HIERA_MATH_FIDELITY = ttnn.MathFidelity.HiFi2


@dataclass(frozen=True)
class HieraBlockSpec:
    """Topology and precision policy for one production Hiera block."""

    dim: int
    dim_out: int
    num_heads: int
    window_size: int
    q_stride: tuple[int, int] | None = None
    weight_dtype: ttnn.DataType = ttnn.bfloat8_b
    attention_weight_dtype: ttnn.DataType | None = None
    qkv_dtype: ttnn.DataType = HIERA_QKV_DTYPE
    mlp_hidden_dtype: ttnn.DataType = HIERA_MLP_HIDDEN_DTYPE
    math_fidelity: ttnn.MathFidelity = HIERA_MATH_FIDELITY
    sdpa_math_fidelity: ttnn.MathFidelity = HIERA_MATH_FIDELITY
    explicit_norm1: bool = False


HIERA_TINY_STAGE_ENDS = (0, 2, 9, 11)
HIERA_TINY_BLOCK_PLAN = (
    HieraBlockSpec(96, 96, 1, 8),
    HieraBlockSpec(96, 192, 2, 8, (2, 2)),
    HieraBlockSpec(192, 192, 2, 4),
    HieraBlockSpec(192, 384, 4, 4, (2, 2), math_fidelity=ttnn.MathFidelity.HiFi4),
    HieraBlockSpec(384, 384, 4, 14),
    HieraBlockSpec(384, 384, 4, 0),
    HieraBlockSpec(384, 384, 4, 14),
    HieraBlockSpec(384, 384, 4, 0),
    HieraBlockSpec(384, 384, 4, 14),
    HieraBlockSpec(384, 384, 4, 0, attention_weight_dtype=ttnn.bfloat16, qkv_dtype=ttnn.bfloat16),
    HieraBlockSpec(
        384, 768, 8, 14, (2, 2), qkv_dtype=ttnn.bfloat16, mlp_hidden_dtype=ttnn.bfloat16, explicit_norm1=True
    ),
    HieraBlockSpec(
        768,
        768,
        8,
        7,
        weight_dtype=ttnn.bfloat16,
        qkv_dtype=ttnn.bfloat16,
        mlp_hidden_dtype=ttnn.bfloat16,
        explicit_norm1=True,
    ),
)
HIERA_TINY_Q_POOL_BLOCKS = tuple(index for index, spec in enumerate(HIERA_TINY_BLOCK_PLAN) if spec.q_stride)
HIERA_TINY_SDPA_CHUNKS = (
    (64, 64),
    (32, 64),
    (32, 32),
    (32, 64),
    (256, 64),
    (128, 256),
    (256, 64),
    (128, 256),
    (256, 64),
    (128, 256),
    (64, 64),
    (64, 32),
)


def _transition_projection_program_config():
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(8, 8),
        in0_block_w=1,
        out_subblock_h=4,
        out_subblock_w=2,
        per_core_M=32,
        per_core_N=6,
        fuse_batch=False,
        fused_activation=None,
        mcast_in0=False,
        untilize_out=True,
    )


def _divisors_leq(n, cap):
    return [d for d in range(min(n, cap), 0, -1) if n % d == 0]


def _height_shard_cfg(M, K, N, grid_max=64, fused_activation=None):
    Mt, Kt, Nt = M // 32, K // 32, N // 32
    ncores = grid_max
    while Mt % ncores and ncores > 1:
        ncores -= 1
    per_core_M = Mt // ncores
    per_core_N = Nt
    gy = 8
    while gy > 1 and ncores % gy:
        gy -= 1
    gx = ncores // gy
    # Constraint: out_subblock_w == per_core_N OR out_subblock_h == 1.
    if per_core_N <= 8:
        ow = per_core_N
        oh = max(_divisors_leq(per_core_M, 8 // ow) or [1])
    else:
        ow = max(_divisors_leq(per_core_N, 8) or [1])
        oh = 1  # must be 1 when ow != per_core_N
    # chunk K so the mcast weight block (in0_block_w * per_core_N tiles) fits L1
    budget = max(1, 128 // max(1, per_core_N))
    in0_bw = max([d for d in _divisors_leq(Kt, Kt) if d <= budget] or [1])
    cfg = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(gx, gy),
        in0_block_w=in0_bw,
        out_subblock_h=oh,
        out_subblock_w=ow,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        fuse_batch=True,
        fused_activation=fused_activation,
        mcast_in0=False,
    )
    return cfg, (gx, gy)


def _hs_mem(M, N, grid):
    return ttnn.create_sharded_memory_config(
        (1, 1, M, N),
        core_grid=ttnn.CoreGrid(x=grid[0], y=grid[1]),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )


def _height_sharded_memory_config(tokens, channels, grid_max):
    return _hs_mem(tokens, channels, _height_shard_cfg(tokens, channels, channels, grid_max=grid_max)[1])


def _sharded_linear(
    x,
    w,
    b,
    device,
    ck,
    out_rowmajor=False,
    out_sharded=False,
    flatten_output=False,
    dtype=ttnn.bfloat16,
    fused_activation=None,
):
    batch, height, width, channels = x.shape
    tokens = batch * height * width
    output_channels = w.shape[-1]
    compute_grid = device.compute_with_storage_grid_size()
    program_config, grid = _height_shard_cfg(
        tokens,
        channels,
        output_channels,
        grid_max=compute_grid.x * compute_grid.y,
        fused_activation=fused_activation,
    )
    input_is_sharded = x.memory_config().is_sharded()
    direct_layout_shard = not input_is_sharded and x.layout == ttnn.ROW_MAJOR_LAYOUT
    if input_is_sharded:
        linear_input = ttnn.reshape(x, (1, tokens, channels))
    elif direct_layout_shard:
        linear_input = ttnn.reshape(
            ttnn.to_layout(x, ttnn.TILE_LAYOUT, memory_config=_hs_mem(tokens, channels, grid)),
            (1, tokens, channels),
        )
    else:
        linear_input = ttnn.reshape(_to_tile(x, ttnn.DRAM_MEMORY_CONFIG), (1, tokens, channels))
    if not linear_input.memory_config().is_sharded():
        sharded_input = ttnn.to_memory_config(linear_input, _hs_mem(tokens, channels, grid))
        owns_input = True
    else:
        sharded_input = linear_input
        owns_input = direct_layout_shard or not input_is_sharded
    output = ttnn.linear(
        sharded_input,
        w,
        bias=b,
        program_config=program_config,
        memory_config=_hs_mem(tokens, output_channels, grid),
        compute_kernel_config=ck,
        dtype=dtype,
    )
    if owns_input:
        ttnn.deallocate(sharded_input)
    if out_sharded:
        if out_rowmajor:
            output = _from_tile_rm(output, _hs_mem(tokens, output_channels, grid))
        output_shape = (
            (1, 1, tokens, output_channels)
            if flatten_output
            else (
                batch,
                height,
                width,
                output_channels,
            )
        )
        return ttnn.reshape(output, output_shape)
    output = ttnn.to_memory_config(output, ttnn.DRAM_MEMORY_CONFIG)
    if out_rowmajor:
        output = _from_tile_rm(output, ttnn.DRAM_MEMORY_CONFIG)
    return ttnn.reshape(output, (batch, height, width, output_channels))


def _to_tile(x, memory_config=ttnn.L1_MEMORY_CONFIG):
    # Skip the conversion op when already tiled: every ttnn op has a fixed
    # device-execution floor (~0.2ms), so a redundant no-op layout change still
    # costs real time. The trunk is bound by op count, not FLOPs, so guarding
    # these removes ~40 wasted ops across the 12 blocks.
    if x.layout == ttnn.TILE_LAYOUT:
        return x
    return ttnn.to_layout(x, ttnn.TILE_LAYOUT, memory_config=memory_config)


def _from_tile_rm(x, memory_config=ttnn.L1_MEMORY_CONFIG):
    if x.layout == ttnn.ROW_MAJOR_LAYOUT:
        return x
    return ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT, memory_config=memory_config)


def _gather_tokens(x, indices, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    x = _to_tile(x, ttnn.DRAM_MEMORY_CONFIG)
    if x.memory_config().is_sharded():
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
    table = ttnn.reshape(x, (int(x.shape[2]), int(x.shape[3])))
    gathered = ttnn.embedding(
        indices,
        table,
        layout=ttnn.TILE_LAYOUT,
        memory_config=memory_config,
    )
    return ttnn.reshape(gathered, (1, 1, int(indices.shape[-1]), int(x.shape[3])))


class TtMultiScaleAttention:
    def __init__(self, parameters, device, spec, compute_kernel_config, block_index):
        self.device = device
        self.p = parameters
        self.block_index = block_index
        self.dim = spec.dim
        self.dim_out = spec.dim_out
        self.num_heads = spec.num_heads
        self.head_dim = spec.dim_out // spec.num_heads
        self.q_stride = spec.q_stride
        self.qkv_dtype = spec.qkv_dtype
        self.compute_kernel_config = compute_kernel_config
        self.sdpa_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=spec.sdpa_math_fidelity,
            math_approx_mode=True,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )
        q_chunk_size, k_chunk_size = HIERA_TINY_SDPA_CHUNKS[block_index]
        self.sdpa_program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
            q_chunk_size=q_chunk_size,
            k_chunk_size=k_chunk_size,
            exp_approx_mode=False,
            max_cores_per_head_batch=16,
        )
        self.local_window_group_size = 1
        self.local_window_mask = None

    def _qkv(self, x, keep_tiled: bool):
        if self.block_index not in (10, 11):
            return _sharded_linear(
                x,
                self.p.qkv.weight,
                self.p.qkv.bias,
                self.device,
                self.compute_kernel_config,
                out_rowmajor=not keep_tiled,
                dtype=self.qkv_dtype,
            )
        input_shape = tuple(int(dim) for dim in x.shape)
        if self.block_index == 11:
            x = ttnn.reshape(x, (1, 1, input_shape[0] * input_shape[1] * input_shape[2], input_shape[3]))
        x = _to_tile(x, ttnn.DRAM_MEMORY_CONFIG)
        qkv = ttnn.linear(
            x,
            self.p.qkv.weight,
            bias=self.p.qkv.bias,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=self.qkv_dtype,
        )
        ttnn.deallocate(x)
        result = qkv if keep_tiled else _from_tile_rm(qkv, ttnn.DRAM_MEMORY_CONFIG)
        if self.block_index == 11:
            result = ttnn.reshape(result, (*input_shape[:-1], 3 * self.dim_out))
        return result

    def _merge_heads(self, out, B: int, H: int, W: int):
        if self.num_heads > 1:
            out = ttnn.transformer.concatenate_heads(out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.reshape(out, (B, H, W, self.dim_out))

    def _pool_query(self, q, *, batch_size: int, height: int, width: int):
        output_height = height // self.q_stride[0]
        output_width = width // self.q_stride[1]
        q = _from_tile_rm(q, ttnn.DRAM_MEMORY_CONFIG)
        q = ttnn.reshape(q, (batch_size * self.num_heads, height, width, self.head_dim))
        q = ttnn.max_pool2d(
            input_tensor=q,
            channels=self.head_dim,
            batch_size=batch_size * self.num_heads,
            input_h=height,
            input_w=width,
            kernel_size=self.q_stride,
            stride=self.q_stride,
            padding=(0, 0),
            dilation=(1, 1),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            deallocate_input=True,
            dtype=ttnn.bfloat16,
            output_layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        return q, output_height, output_width

    def _sdpa(self, q, k, v, *, attention_mask=None, memory_config=None):
        out = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            is_causal=False,
            scale=1.0 / (self.head_dim**0.5),
            memory_config=memory_config,
            compute_kernel_config=self.sdpa_compute_kernel_config,
            program_config=self.sdpa_program_config,
        )
        for tensor in (q, k, v):
            ttnn.deallocate(tensor)
        return out

    def _attend_local_sequence(
        self,
        qkv,
        *,
        spatial_height,
        spatial_width,
        window_size,
        q_stride,
    ):
        window_batch = (spatial_height // window_size) * (spatial_width // window_size)
        window_area = window_size * window_size
        attention_mask = self.local_window_mask
        group_size = self.local_window_group_size if attention_mask is not None else 1
        attention_batch = window_batch // group_size
        attention_area = window_area * group_size
        qkv = ttnn.reshape(qkv, (attention_batch, attention_area, 3 * self.dim_out))
        qkv = _to_tile(qkv, ttnn.DRAM_MEMORY_CONFIG)
        q, k, v = ttnn.transformer.split_query_key_value_and_split_heads(
            qkv,
            num_heads=self.num_heads,
            transpose_key=False,
            memory_config=ttnn.L1_MEMORY_CONFIG if q_stride == 1 else ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(qkv)
        output_window = window_size
        if q_stride != 1:
            q, output_height, output_width = self._pool_query(
                q,
                batch_size=window_batch,
                height=window_size,
                width=window_size,
            )
            output_window = output_height
            pooled_area = output_height * output_width
            q = ttnn.reshape(
                q,
                (
                    attention_batch,
                    self.num_heads,
                    pooled_area * group_size,
                    self.head_dim,
                ),
            )
            q = _to_tile(q, ttnn.DRAM_MEMORY_CONFIG)
        out = self._sdpa(q, k, v, attention_mask=attention_mask)
        output_area = output_window * output_window
        if group_size > 1:
            context = ttnn.transformer.concatenate_heads(out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(out)
        else:
            context = self._merge_heads(out, window_batch, 1, output_area)
        return ttnn.reshape(context, (1, 1, window_batch * output_area, self.dim_out))

    def _tile_padded_qkv(self, x, input_pad, window_indices, key_tokens, memory_config):
        _, height, width, channels = (int(dim) for dim in x.shape)
        flat = ttnn.reshape(
            (ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG) if x.memory_config().is_sharded() else x),
            (1, 1, height * width, channels),
        )
        padded = ttnn.concat([flat, input_pad], dim=2)
        windowed = _gather_tokens(padded, window_indices.w14_padded224_from_raster64)
        ttnn.deallocate(padded)
        full_qkv = self._qkv(windowed, keep_tiled=True)
        ttnn.deallocate(windowed)
        window_batch = 25
        qkv = ttnn.reshape(full_qkv, (window_batch, 224, 3 * self.dim_out))
        if key_tokens != 224:
            qkv = ttnn.slice(
                qkv,
                [0, 0, 0],
                [window_batch, key_tokens, 3 * self.dim_out],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            ttnn.deallocate(full_qkv)
        q, k, v = ttnn.transformer.split_query_key_value_and_split_heads(
            qkv,
            num_heads=self.num_heads,
            transpose_key=False,
            memory_config=memory_config,
        )
        ttnn.deallocate(qkv)
        return q, k, v

    def _attend_tile_padded_windows(self, x, input_pad, window_indices):
        window_batch = 25
        window_tokens = 224
        q, k, v = self._tile_padded_qkv(
            x,
            input_pad,
            window_indices,
            key_tokens=window_tokens,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        out = self._sdpa(
            q,
            k,
            v,
            attention_mask=window_indices.w14_padded224_mask,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        merged = ttnn.transformer.concatenate_heads(out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(out)
        merged = ttnn.reshape(merged, (1, 1, window_batch * window_tokens, self.dim_out))
        raster = _gather_tokens(merged, window_indices.raster64_from_w14_padded224)
        ttnn.deallocate(merged)
        return ttnn.reshape(raster, (1, 64, 64, self.dim_out))

    def _attend_tile_padded_transition_windows(self, x, input_pad, window_indices):
        window_batch = 25
        q, k, v = self._tile_padded_qkv(
            x,
            input_pad,
            window_indices,
            key_tokens=196,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        q, _, _ = self._pool_query(q, batch_size=window_batch, height=14, width=14)
        q = ttnn.reshape(q, (window_batch, self.num_heads, 49, self.head_dim))
        q = _to_tile(q, ttnn.DRAM_MEMORY_CONFIG)
        out = self._sdpa(q, k, v)
        merged = ttnn.transformer.concatenate_heads(out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(out)
        merged = ttnn.reshape(merged, (1, 1, window_batch * 49, self.dim_out))
        raster = _gather_tokens(merged, window_indices.raster32_from_w7)
        ttnn.deallocate(merged)
        return ttnn.reshape(raster, (1, 32, 32, self.dim_out))

    def attend(self, qkv, *, merge_heads=True):
        B, H, W, _ = qkv.shape
        qkv_seq = ttnn.reshape(qkv, (B, H * W, 3 * self.dim_out))
        qkv_seq = _to_tile(qkv_seq, ttnn.DRAM_MEMORY_CONFIG)
        q, k, v = ttnn.transformer.split_query_key_value_and_split_heads(
            qkv_seq,
            num_heads=self.num_heads,
            transpose_key=False,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.deallocate(qkv_seq)
        out = self._sdpa(q, k, v)
        if not merge_heads:
            return out, (H, W)
        return self._merge_heads(out, B, H, W), (H, W)


class TtMultiScaleBlock:
    def __init__(
        self,
        parameters,
        device,
        spec: HieraBlockSpec,
        in_hw: tuple[int, int],
        block_index: int = -1,
        window_indices=None,
    ):
        self.device = device
        self.p = parameters
        compute_grid = device.compute_with_storage_grid_size()
        self.max_cores = compute_grid.x * compute_grid.y
        dim = spec.dim
        dim_out = spec.dim_out
        window_size = spec.window_size
        q_stride = spec.q_stride
        self.dim = dim
        self.dim_out = dim_out
        self.window_size = window_size
        self.q_stride = q_stride
        self.block_index = block_index
        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=spec.math_fidelity,
            math_approx_mode=True,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )
        self.mlp_hidden_dtype = spec.mlp_hidden_dtype
        self.window_indices = window_indices
        input_tokens = in_hw[0] * in_hw[1]
        output_hw = in_hw if q_stride is None else tuple(size // stride for size, stride in zip(in_hw, q_stride))
        self.input_hs_cfg = _height_sharded_memory_config(input_tokens, dim, self.max_cores)
        self.output_hs_cfg = _height_sharded_memory_config(output_hw[0] * output_hw[1], dim_out, self.max_cores)
        self.attn = TtMultiScaleAttention(
            parameters.attn,
            device,
            spec,
            self.compute_kernel_config,
            block_index,
        )
        group_config = {
            (4, None): (2, 4),
            (8, (2, 2)): (2, 4),
            (4, (2, 2)): (8, 2),
        }.get((window_size, q_stride))
        if group_config is not None:
            group_size, output_window = group_config
            query_area = output_window * output_window
            key_area = window_size * window_size
            grouped_mask = torch.full(
                (group_size, query_area, group_size, key_area),
                torch.finfo(torch.bfloat16).min,
                dtype=torch.bfloat16,
            )
            groups = torch.arange(group_size)
            grouped_mask[groups, :, groups, :] = 0
            grouped_mask = grouped_mask.reshape(1, 1, query_area * group_size, key_area * group_size)
            self.attn.local_window_group_size = group_size
            self.attn.local_window_mask = ttnn.from_torch(
                grouped_mask,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

    def _layernorm_sharded(self, x, w, b, hs_cfg, *, force_dram=False, interleaved_input=False):
        materialized = None
        try:
            if force_dram and x.memory_config().is_sharded():
                materialized = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
                x = materialized
            if force_dram or interleaved_input:
                return ttnn.layer_norm(x, weight=w, bias=b, memory_config=hs_cfg, epsilon=1e-6)
            return ttnn.moreh_layer_norm(x, 1, 1e-6, w, b)[0]
        finally:
            if materialized is not None:
                ttnn.deallocate(materialized)

    def _add_attention_and_mlp_residuals(self, shortcut, attention):
        hs_cfg = self.output_hs_cfg
        context = _sharded_linear(
            attention,
            self.attn.p.proj.weight,
            self.attn.p.proj.bias,
            self.device,
            self.compute_kernel_config,
            out_sharded=True,
            dtype=HIERA_RESIDUAL_BRANCH_DTYPE,
        )
        residual = shortcut if shortcut.memory_config().is_sharded() else ttnn.to_memory_config(shortcut, hs_cfg)
        x = ttnn.add(residual, context, memory_config=hs_cfg, dtype=HIERA_STATE_DTYPE)
        for tensor in (residual, context):
            ttnn.deallocate(tensor)
        normalized = self._layernorm_sharded(
            x,
            self.p.norm2.weight,
            self.p.norm2.bias,
            hs_cfg,
            force_dram=self.block_index in HIERA_TINY_STAGE_ENDS,
        )
        mlp_out = self._mlp(normalized)
        return ttnn.add(x, mlp_out, memory_config=hs_cfg, dtype=HIERA_STATE_DTYPE)

    def _call_prewindowed(
        self,
        x,
        *,
        window_batch,
        input_window_size,
    ):
        shortcut = x
        x = self._layernorm_sharded(
            x,
            self.p.norm1.weight,
            self.p.norm1.bias,
            self.input_hs_cfg,
            force_dram=self.block_index in HIERA_TINY_Q_POOL_BLOCKS,
            interleaved_input=self.block_index == 0,
        )
        output_window_size = self.window_size
        if self.dim != self.dim_out:
            shortcut = self._project_and_pool_transition_shortcut(
                x,
                window_batch,
                input_window_size,
                input_window_size,
            )
            output_window_size //= self.q_stride[0]
            output_tokens = window_batch * output_window_size * output_window_size
            shortcut = ttnn.reshape(shortcut, (1, 1, output_tokens, self.dim_out))
        q_stride = self.q_stride[0] if self.q_stride is not None else 1
        # Keep the stock BFP8 projection tiled through the head split. This
        # avoids a redundant whole-QKV untilize/re-tilize round trip.
        qkv = self.attn._qkv(x, keep_tiled=True)
        x = self.attn._attend_local_sequence(
            qkv,
            spatial_height=window_batch * input_window_size,
            spatial_width=input_window_size,
            window_size=input_window_size,
            q_stride=q_stride,
        )
        output_tokens = window_batch * output_window_size * output_window_size
        x = ttnn.reshape(x, (1, 1, output_tokens, self.dim_out))
        x = self._add_attention_and_mlp_residuals(shortcut, x)
        return x, output_window_size

    def _mlp(self, x):
        batch, height, width, channels = x.shape
        tokens = batch * height * width
        hidden = self.p.mlp.layers["0"].weight.shape[-1]
        first_program, grid = _height_shard_cfg(
            tokens,
            channels,
            hidden,
            grid_max=self.max_cores,
            fused_activation=ttnn.UnaryOpType.GELU,
        )
        second_program, _ = _height_shard_cfg(tokens, hidden, channels, grid_max=self.max_cores)
        linear_input = ttnn.reshape(_to_tile(x, ttnn.DRAM_MEMORY_CONFIG), (1, tokens, channels))
        sharded_input = (
            linear_input
            if linear_input.memory_config().is_sharded()
            else ttnn.to_memory_config(linear_input, _hs_mem(tokens, channels, grid))
        )
        hidden_state = ttnn.linear(
            sharded_input,
            self.p.mlp.layers["0"].weight,
            bias=self.p.mlp.layers["0"].bias,
            program_config=first_program,
            memory_config=_hs_mem(tokens, hidden, grid),
            compute_kernel_config=self.compute_kernel_config,
            dtype=self.mlp_hidden_dtype,
        )
        output = ttnn.linear(
            hidden_state,
            self.p.mlp.layers["1"].weight,
            bias=self.p.mlp.layers["1"].bias,
            program_config=second_program,
            memory_config=_hs_mem(tokens, channels, grid),
            compute_kernel_config=self.compute_kernel_config,
            dtype=HIERA_RESIDUAL_BRANCH_DTYPE,
        )
        ttnn.deallocate(hidden_state)
        ttnn.deallocate(sharded_input)
        return ttnn.reshape(output, (batch, height, width, channels))

    def _project_transition_shortcut(self, x):
        if self.block_index != 1:
            return _sharded_linear(
                x,
                self.p.proj.weight,
                self.p.proj.bias,
                self.device,
                self.compute_kernel_config,
                out_sharded=True,
                flatten_output=True,
                dtype=ttnn.bfloat16,
            )
        fallback_x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG) if x.memory_config().is_sharded() else x
        linear_input = _to_tile(fallback_x, ttnn.DRAM_MEMORY_CONFIG)
        projected = ttnn.linear(
            linear_input,
            self.p.proj.weight,
            bias=self.p.proj.bias,
            program_config=_transition_projection_program_config(),
            compute_kernel_config=self.compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
        )
        row_major = _from_tile_rm(projected, ttnn.DRAM_MEMORY_CONFIG)
        if row_major is not projected:
            ttnn.deallocate(projected)
        if linear_input is not fallback_x and isinstance(linear_input, ttnn.Tensor):
            ttnn.deallocate(linear_input)
        if fallback_x is not x and isinstance(fallback_x, ttnn.Tensor):
            ttnn.deallocate(fallback_x)
        return row_major

    def _project_and_pool_transition_shortcut(
        self,
        x,
        batch_size,
        height,
        width,
    ):
        projected = self._project_transition_shortcut(x)
        if not projected.memory_config().is_sharded():
            projected = ttnn.reshape(projected, (batch_size, height, width, self.dim_out))
        shortcut = ttnn.max_pool2d(
            input_tensor=projected,
            channels=self.dim_out,
            batch_size=batch_size,
            input_h=height,
            input_w=width,
            kernel_size=self.q_stride,
            stride=self.q_stride,
            padding=(0, 0),
            dilation=(1, 1),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            deallocate_input=True,
            dtype=ttnn.bfloat16,
            output_layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        return ttnn.reshape(
            shortcut,
            (batch_size, height // self.q_stride[0], width // self.q_stride[1], self.dim_out),
        )

    def __call__(self, x):
        batch, height, width, _ = x.shape
        shortcut = x
        x = self._layernorm_sharded(
            x,
            self.p.norm1.weight,
            self.p.norm1.bias,
            self.input_hs_cfg,
            force_dram=self.block_index in HIERA_TINY_Q_POOL_BLOCKS,
            interleaved_input=self.block_index == 4,
        )
        if self.block_index == 10:
            shortcut = self._project_and_pool_transition_shortcut(x, batch, height, width)
            x = self.attn._attend_tile_padded_transition_windows(x, self.p.input_pad, self.window_indices)
        elif self.block_index in (4, 6, 8):
            x = self.attn._attend_tile_padded_windows(x, self.p.input_pad, self.window_indices)
        elif self.block_index == 11:
            windowed = _from_tile_rm(x, ttnn.DRAM_MEMORY_CONFIG)
            windowed = ttnn.pad(windowed, [1, 35, 35, 768], [0, 0, 0, 0], 0)
            windowed = ttnn.reshape(windowed, (1, 5, 7, 5, 7, 768))
            windowed = ttnn.permute(windowed, (0, 1, 3, 2, 4, 5))
            windowed = ttnn.reshape(windowed, (25, 7, 7, 768))
            qkv = self.attn._qkv(windowed, keep_tiled=False)
            ttnn.deallocate(windowed)
            x = self.attn.attend(qkv, merge_heads=False)[0]
            merged = ttnn.transformer.concatenate_heads(x, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(x)
            merged = ttnn.reshape(merged, (1, 1, 25 * 49, self.dim_out))
            raster = _gather_tokens(merged, self.window_indices.raster32_from_w7)
            ttnn.deallocate(merged)
            x = ttnn.reshape(raster, (1, 32, 32, self.dim_out))
        else:
            qkv = self.attn._qkv(x, keep_tiled=True)
            x = self.attn.attend(qkv)[0]
        return self._add_attention_and_mlp_residuals(shortcut, x)


class TtHiera:
    def __init__(
        self,
        parameters,
        device,
    ):
        patch_size = 256
        folded_size = patch_size + 1
        pe = parameters.patch_embed.proj
        patch_cfg = Conv2dConfiguration(
            input_height=folded_size,
            input_width=folded_size,
            in_channels=48,
            out_channels=96,
            batch_size=1,
            kernel_size=(2, 2),
            stride=(1, 1),
            padding=(0, 0),
            dilation=(1, 1),
            groups=1,
            weight=pe.weight,
            bias=pe.bias,
            slice_strategy=None,
            sharding_strategy=HeightShardedStrategyConfiguration(act_block_h_override=16 * 32),
            deallocate_activation=False,
        )
        self.patch_embed = TtConv2d(patch_cfg, device)
        self._pos_tokens = parameters.pos_tokens
        self.window_indices = parameters.window_indices
        block_hws = ((256, 256),) * 2 + ((128, 128),) * 2 + ((64, 64),) * 7 + ((32, 32),)
        self.blocks = [
            TtMultiScaleBlock(parameters.blocks[index], device, spec, block_hws[index], index, self.window_indices)
            for index, spec in enumerate(HIERA_TINY_BLOCK_PLAN)
        ]

    def _rasterize_early_windows(self, tokens, logical_hw, window_size):
        height, width = logical_hw
        indices = getattr(self.window_indices, f"raster{height}_from_w{window_size}")
        raster = _gather_tokens(tokens, indices)
        return ttnn.reshape(raster, (1, height, width, int(tokens.shape[-1])))

    def forward(self, x: ttnn.Tensor) -> list[ttnn.Tensor]:
        x = self.patch_embed(x)
        x = ttnn.reshape(x, (1, 1, 65536, 96))
        x = ttnn.add(x, self._pos_tokens, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        x = ttnn.reshape(x, (1, 256, 256, 96))
        outputs = []
        raster_tokens = ttnn.reshape(x, (1, 1, 65536, 96))
        tokens = _gather_tokens(raster_tokens, self.window_indices.w8_from_raster256)
        tokens, _ = self.blocks[0]._call_prewindowed(tokens, window_batch=1024, input_window_size=8)
        outputs.append(self._rasterize_early_windows(tokens, (256, 256), 8))
        tokens, _ = self.blocks[1]._call_prewindowed(tokens, window_batch=1024, input_window_size=8)
        tokens, _ = self.blocks[2]._call_prewindowed(tokens, window_batch=1024, input_window_size=4)
        outputs.append(self._rasterize_early_windows(tokens, (128, 128), 4))
        tokens, _ = self.blocks[3]._call_prewindowed(tokens, window_batch=1024, input_window_size=4)
        x = self._rasterize_early_windows(tokens, (64, 64), 2)
        del raster_tokens, tokens
        for block_index, block in enumerate(self.blocks[4:], start=4):
            x = block(x)
            if block_index == 9:
                outputs.append(ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG))
            elif block_index == 11:
                outputs.append(x)
        return outputs
