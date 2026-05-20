# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Centralized compute, memory, and program configurations for BGE-M3 on TTNN.

Usage:
    opts = Optimizations.build(mesh_device, max_batch_size=32, max_seq_len=512, dtype=ttnn.bfloat8_b)
    model = BgeM3Model(args, mesh_device, dtype, state_dict, optimizations=opts)
"""

from __future__ import annotations

from dataclasses import dataclass

from ttnn.device import is_blackhole as ttnn_is_blackhole

import ttnn

# ══════════════════════════════════════════════════════════════════════════════
# Dataclasses
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class MLPOptimizations:
    """Pre-resolved configs for BgeM3MLP."""

    wi_compute_kernel_cfg: object
    wo_compute_kernel_cfg: object
    wi_memcfg: ttnn.MemoryConfig
    wo_memcfg: ttnn.MemoryConfig
    activation_memcfg: ttnn.MemoryConfig
    core_grid: ttnn.CoreGrid
    wi_prg_config: object | None = None
    wo_prg_config: object | None = None
    wi_minimal_config: object | None = None


@dataclass
class AttentionOptimizations:
    """Pre-resolved configs for BgeM3Attention.

    SDPA program config is NOT included — chunk sizes depend on runtime seq_len.
    """

    qkv_compute_kernel_cfg: object
    output_compute_kernel_cfg: object
    score_compute_kernel_cfg: object
    qkv_memcfg: ttnn.MemoryConfig
    create_heads_memcfg: ttnn.MemoryConfig
    score_memcfg: ttnn.MemoryConfig
    output_memcfg: ttnn.MemoryConfig
    core_grid: ttnn.CoreGrid | None = None
    qkv_prg_config: object | None = None
    output_prg_config: object | None = None


@dataclass
class NormOptimizations:
    """Pre-resolved configs for LayerNorm1D."""

    compute_kernel_config: object
    output_memcfg: ttnn.MemoryConfig
    program_config: object | None = None
    sharded_memcfg: ttnn.MemoryConfig | None = None


# ══════════════════════════════════════════════════════════════════════════════
# Optimizations class
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class Optimizations:
    """
    Central optimization config for BGE-M3. Constructed once, passed to BgeM3Model.
    """

    mesh_device: ttnn.MeshDevice
    max_batch_size: int
    max_seq_len: int
    dtype: ttnn.DataType

    mlp: MLPOptimizations | None = None
    attention: AttentionOptimizations | None = None
    norm: NormOptimizations | None = None

    @classmethod
    def build(
        cls,
        mesh_device: ttnn.MeshDevice,
        max_batch_size: int = 1,
        max_seq_len: int = 512,
        dtype: ttnn.DataType = ttnn.bfloat16,
        *,
        hidden_size: int = 1024,
        intermediate_size: int = 4096,
    ) -> "Optimizations":
        """Build fully-resolved optimizations for the given shape and device."""
        max_batch = max(1, max_batch_size)
        core_grid = _matmul_core_grid(mesh_device, max_seq_len, max_batch)
        act_mem = _linear_activation_memory_config(max_seq_len, max_batch)

        # ── MLP ──────────────────────────────────────────────────────────

        wi_minimal = _mlp_wi_minimal_matmul_config(
            mesh_device,
            max_seq_len,
            max_batch,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )
        wi_prg = (
            None
            if wi_minimal is not None
            else _mlp_wi_program_config(
                mesh_device,
                max_seq_len,
                max_batch,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
            )
        )

        # B1/S512 tuned matmul configs for AttnOut and MLPwo (Sweep 4.2).
        tuned_b1 = max_seq_len == 512 and max_batch == 1
        wo_prg_tuned = (
            _tuned_mlp_wo_program_config(
                mesh_device,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
            )
            if tuned_b1
            else None
        )

        mlp_opts = MLPOptimizations(
            wi_compute_kernel_cfg=mlp_wi_compute_kernel_config(mesh_device, max_seq_len, max_batch, dtype=dtype),
            wo_compute_kernel_cfg=mlp_wo_compute_kernel_config(mesh_device, max_seq_len, max_batch, dtype=dtype),
            wi_memcfg=_mlp_wi_output_memory_config(max_seq_len, max_batch, mesh_device),
            wo_memcfg=_mlp_wo_output_memory_config(max_seq_len, max_batch, mesh_device),
            activation_memcfg=act_mem,
            core_grid=core_grid,
            wi_prg_config=wi_prg,
            wo_prg_config=wo_prg_tuned,
            wi_minimal_config=wi_minimal,
        )

        # ── Attention ────────────────────────────────────────────────────

        qkv_out_dim = 3 * hidden_size
        attn_opts = AttentionOptimizations(
            qkv_compute_kernel_cfg=attention_qkv_compute_kernel_config(
                mesh_device, max_seq_len, max_batch, dtype=dtype
            ),
            output_compute_kernel_cfg=attention_output_compute_kernel_config(
                mesh_device, max_seq_len, max_batch, dtype=dtype
            ),
            score_compute_kernel_cfg=sdpa_compute_kernel_config(mesh_device, max_seq_len, max_batch, dtype=dtype),
            qkv_memcfg=act_mem,
            create_heads_memcfg=_create_heads_output_memory_config(max_seq_len, max_batch, mesh_device),
            score_memcfg=act_mem,
            output_memcfg=_attention_output_memory_config(max_seq_len, max_batch, mesh_device),
            core_grid=core_grid,
            qkv_prg_config=_qkv_program_config(max_seq_len, max_batch, hidden_size, qkv_out_dim, mesh_device),
            output_prg_config=(
                _tuned_attention_output_program_config(mesh_device, hidden_size=hidden_size)
                if tuned_b1
                else _attention_output_program_config(max_seq_len, max_batch, hidden_size, mesh_device)
            ),
        )

        # ── Norm ─────────────────────────────────────────────────────────

        norm_prg, norm_sharded_mem = _layernorm_sharded_config(max_seq_len, max_batch)
        norm_opts = NormOptimizations(
            compute_kernel_config=layernorm_compute_kernel_config(mesh_device, max_seq_len, max_batch),
            output_memcfg=_linear_activation_memory_config(max_seq_len, max_batch),
            program_config=norm_prg,
            sharded_memcfg=norm_sharded_mem,
        )

        return cls(
            mesh_device=mesh_device,
            max_batch_size=max_batch,
            max_seq_len=max_seq_len,
            dtype=dtype,
            mlp=mlp_opts,
            attention=attn_opts,
            norm=norm_opts,
        )


# ══════════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════════

BGE_M3_L1_LINEAR_MAX_SEQ_LEN = 512


# ══════════════════════════════════════════════════════════════════════════════
# Device helpers
# ══════════════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════════════
# Memory config helpers
# ══════════════════════════════════════════════════════════════════════════════


def weight_dram_memory_config():
    return ttnn.DRAM_MEMORY_CONFIG


def _linear_activation_memory_config(max_seq_len, max_batch_size=None):
    s = 0 if max_seq_len is None else max_seq_len
    if s > BGE_M3_L1_LINEAR_MAX_SEQ_LEN:
        return ttnn.DRAM_MEMORY_CONFIG
    b = 1 if max_batch_size is None else max(1, max_batch_size)
    if b * s > BGE_M3_L1_LINEAR_MAX_SEQ_LEN:
        return ttnn.DRAM_MEMORY_CONFIG
    return ttnn.L1_MEMORY_CONFIG


def _mlp_wi_output_memory_config(max_seq_len, max_batch_size, mesh_device):
    max_batch = 1 if max_batch_size is None else max(1, max_batch_size)
    if max_seq_len == 512 and max_batch == 32 and mesh_device is not None and ttnn_is_blackhole(mesh_device):
        return ttnn.L1_MEMORY_CONFIG
    return _linear_activation_memory_config(max_seq_len, max_batch_size)


def _mlp_wo_output_memory_config(max_seq_len, max_batch_size, mesh_device):
    max_batch = 1 if max_batch_size is None else max(1, max_batch_size)
    if max_seq_len == 512 and max_batch == 32 and mesh_device is not None and ttnn_is_blackhole(mesh_device):
        return ttnn.L1_MEMORY_CONFIG
    return _linear_activation_memory_config(max_seq_len, max_batch_size)


def _attention_output_memory_config(max_seq_len, max_batch_size, mesh_device):
    max_batch = 1 if max_batch_size is None else max(1, max_batch_size)
    if max_seq_len == 512 and max_batch == 32 and mesh_device is not None and ttnn_is_blackhole(mesh_device):
        return ttnn.L1_MEMORY_CONFIG
    return _linear_activation_memory_config(max_seq_len, max_batch)


def _create_heads_output_memory_config(max_seq_len, max_batch_size, mesh_device):
    max_batch = 1 if max_batch_size is None else max(1, max_batch_size)
    if max_seq_len == 512 and max_batch == 32 and mesh_device is not None and ttnn_is_blackhole(mesh_device):
        return ttnn.L1_MEMORY_CONFIG
    return _linear_activation_memory_config(max_seq_len, max_batch)


# ══════════════════════════════════════════════════════════════════════════════
# Core grid
# ══════════════════════════════════════════════════════════════════════════════


def _matmul_core_grid(mesh_device, sequence_length=None, batch_size=None):
    if mesh_device is None:
        gx, gy = 8, 8
    else:
        try:
            g = mesh_device.compute_with_storage_grid_size()
            gx, gy = int(g.x), int(g.y)
        except Exception:
            gx, gy = 8, 8
    return ttnn.CoreGrid(y=gy, x=gx)


# ══════════════════════════════════════════════════════════════════════════════
# Compute kernel builders (single source of fidelity policy)
# ══════════════════════════════════════════════════════════════════════════════


def _make_compute_kernel(mesh_device, fidelity, max_seq_len=None, max_batch_size=None, fp32_dest_acc_en=None):
    packer_l1_acc = True
    # B1/S512: fp32_dest_acc_en=False lifts the matmul subblock cap (h*w <= 8
    # instead of 4) and speeds up LN reduction passes (~1.68 µs/call).
    # All other shapes: conservative fp32_dest_acc_en=True (unless caller overrides).
    max_batch = 1 if max_batch_size is None else max(1, max_batch_size)
    if fp32_dest_acc_en is None:
        fp32_dest_acc_en = not (max_seq_len == 512 and max_batch == 1)
    return ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=fp32_dest_acc_en,
        packer_l1_acc=packer_l1_acc,
    )


def _default_fidelity(mesh_device):
    return ttnn.MathFidelity.HiFi4


def matmul_compute_kernel_config(mesh_device, max_seq_len=None, max_batch_size=None):
    return _make_compute_kernel(mesh_device, _default_fidelity(mesh_device), max_seq_len, max_batch_size)


def mlp_wi_compute_kernel_config(mesh_device, max_seq_len=None, max_batch_size=None, dtype=None):
    max_batch = 1 if max_batch_size is None else max(1, max_batch_size)
    if max_seq_len == 512 and max_batch in (1, 32):
        fid = ttnn.MathFidelity.LoFi if dtype == ttnn.bfloat8_b else ttnn.MathFidelity.HiFi2
        # fp32_dest_acc_en=False lifts subblock h*w cap to 8.
        return _make_compute_kernel(mesh_device, fid, max_seq_len, max_batch, fp32_dest_acc_en=False)
    return matmul_compute_kernel_config(mesh_device, max_seq_len, max_batch)


def mlp_wo_compute_kernel_config(mesh_device, max_seq_len=None, max_batch_size=None, dtype=None):
    max_batch = 1 if max_batch_size is None else max(1, max_batch_size)
    if max_seq_len == 512 and max_batch in (1, 32):
        return _make_compute_kernel(mesh_device, ttnn.MathFidelity.LoFi, max_seq_len, max_batch)
    return matmul_compute_kernel_config(mesh_device, max_seq_len, max_batch)


def attention_qkv_compute_kernel_config(mesh_device, max_seq_len=None, max_batch_size=None, dtype=None):
    max_batch = 1 if max_batch_size is None else max(1, max_batch_size)
    if max_seq_len == 512 and max_batch in (1, 32):
        fid = ttnn.MathFidelity.LoFi if dtype == ttnn.bfloat8_b else ttnn.MathFidelity.HiFi2
        return _make_compute_kernel(mesh_device, fid, max_seq_len, max_batch, fp32_dest_acc_en=False)
    return matmul_compute_kernel_config(mesh_device, max_seq_len, max_batch)


def attention_output_compute_kernel_config(mesh_device, max_seq_len=None, max_batch_size=None, dtype=None):
    max_batch = 1 if max_batch_size is None else max(1, max_batch_size)
    if max_seq_len == 512 and max_batch in (1, 32):
        fid = ttnn.MathFidelity.LoFi if dtype == ttnn.bfloat8_b else ttnn.MathFidelity.HiFi2
        return _make_compute_kernel(mesh_device, fid, max_seq_len, max_batch, fp32_dest_acc_en=False)
    return matmul_compute_kernel_config(mesh_device, max_seq_len, max_batch)


def sdpa_compute_kernel_config(mesh_device, max_seq_len=None, max_batch_size=None, dtype=None):
    max_batch = 1 if max_batch_size is None else max(1, max_batch_size)
    if max_seq_len == 512 and max_batch == 1:
        fid = ttnn.MathFidelity.LoFi if dtype == ttnn.bfloat8_b else ttnn.MathFidelity.HiFi2
        return _make_compute_kernel(mesh_device, fid, max_seq_len, max_batch)
    if max_seq_len == 512 and max_batch == 32:
        # HiFi2 (vs HiFi4) speeds up SDPA without dropping PCC below 0.94.
        fid = ttnn.MathFidelity.HiFi2 if dtype == ttnn.bfloat8_b else ttnn.MathFidelity.HiFi4
        return _make_compute_kernel(mesh_device, fid, max_seq_len, max_batch)
    return _make_compute_kernel(mesh_device, ttnn.MathFidelity.HiFi4, max_seq_len, max_batch_size)


def layernorm_compute_kernel_config(mesh_device, max_seq_len=None, max_batch_size=None):
    # B1/S512: HiFi2 (LoFi regressed test_model PCC 1.0 -> 0.906).
    # bf8b precision at B1/S512 is protected by the bf8b->bf16 fused reshard
    # in norm.py (interleaved_to_sharded with output_dtype=bf16).
    # B32/S512: HiFi2 too (test_model passes; speeds up LN reduction).
    # All other shapes: HiFi4 (conservative default).
    max_batch = 1 if max_batch_size is None else max(1, max_batch_size)
    if max_seq_len == 512 and max_batch in (1, 32):
        return _make_compute_kernel(mesh_device, ttnn.MathFidelity.HiFi2, max_seq_len, max_batch_size)
    return _make_compute_kernel(mesh_device, ttnn.MathFidelity.HiFi4, max_seq_len, max_batch_size)


# ══════════════════════════════════════════════════════════════════════════════
# MLP program config helpers
# ══════════════════════════════════════════════════════════════════════════════


def _mlp_wi_program_config(mesh_device, max_seq_len, max_batch_size, *, hidden_size, intermediate_size):
    max_batch = 1 if max_batch_size is None else max(1, max_batch_size)
    if max_seq_len != 512:
        return None
    if max_batch == 32:
        return _b32s512_sequence_program_config(
            mesh_device,
            input_size=hidden_size,
            output_size=intermediate_size,
            in0_block_w=8,
            out_subblock_h=1,
            out_subblock_w=3,
            fused_activation=(ttnn.UnaryOpType.GELU, True),
        )
    if max_batch == 1:
        return _b1s512_mlp_wi_program_config(mesh_device, hidden_size=hidden_size, intermediate_size=intermediate_size)
    return None


def _b1s512_mlp_wi_program_config(mesh_device, *, hidden_size, intermediate_size):
    core_grid = _matmul_core_grid(mesh_device, 512, 1)
    hidden_tiles = hidden_size // 32
    m_tiles = 512 // 32
    intermediate_tiles = intermediate_size // 32
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(core_grid.x, core_grid.y),
        in0_block_w=min(4, hidden_tiles),
        out_subblock_h=1,
        out_subblock_w=2,
        per_core_M=(m_tiles + core_grid.y - 1) // core_grid.y,
        per_core_N=(intermediate_tiles + core_grid.x - 1) // core_grid.x,
        transpose_mcast=False,
        fused_activation=(ttnn.UnaryOpType.GELU, True),
    )


def _b32s512_sequence_program_config(
    mesh_device, *, input_size, output_size, in0_block_w, out_subblock_h, out_subblock_w, fused_activation
):
    grid_x, grid_y = 11, 10
    device_grid = mesh_device.compute_with_storage_grid_size()
    if device_grid.x < grid_x or device_grid.y < grid_y:
        return None
    tile_size = 32
    input_tiles = input_size // tile_size
    output_tiles = output_size // tile_size
    seq_tiles = 512 // tile_size
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(grid_x, grid_y),
        in0_block_w=min(in0_block_w, input_tiles),
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=(seq_tiles + grid_y - 1) // grid_y,
        per_core_N=(output_tiles + grid_x - 1) // grid_x,
        transpose_mcast=False,
        fused_activation=fused_activation,
        fuse_batch=False,
    )


def _mlp_wi_minimal_matmul_config(mesh_device, max_seq_len, max_batch_size, *, hidden_size, intermediate_size):
    max_batch = 1 if max_batch_size is None else max(1, max_batch_size)
    if max_seq_len != 512 or max_batch != 32:
        return None
    if hidden_size != 1024 or intermediate_size != 4096:
        return None
    if mesh_device is None or not ttnn_is_blackhole(mesh_device):
        return None
    grid_x, grid_y = 11, 10
    device_grid = mesh_device.compute_with_storage_grid_size()
    if device_grid.x < grid_x or device_grid.y < grid_y:
        return None
    return ttnn.MinimalMatmulConfig(
        M_block_size=8,
        K_block_size=8,
        N_block_size=8,
        subblock_h=8,
        subblock_w=1,
        compute_with_storage_grid_size=ttnn.CoreCoord(grid_x, grid_y),
    )


# ══════════════════════════════════════════════════════════════════════════════
# Attention program config helpers
# ══════════════════════════════════════════════════════════════════════════════


def _qkv_program_config(max_seq_len, max_batch_size, hidden_size, qkv_out_dim, mesh_device):
    max_batch = 1 if max_batch_size is None else max(1, max_batch_size)
    if (
        max_seq_len != 512
        or max_batch != 32
        or hidden_size != 1024
        or qkv_out_dim != 3072
        or mesh_device is None
        or not ttnn_is_blackhole(mesh_device)
    ):
        return None
    try:
        device_grid = mesh_device.compute_with_storage_grid_size()
        if device_grid.x < 11 or device_grid.y < 10:
            return None
    except Exception:
        return None
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(11, 10),
        in0_block_w=8,
        out_subblock_h=1,
        out_subblock_w=3,
        out_block_h=13,
        out_block_w=9,
        per_core_M=52,
        per_core_N=9,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=True,
    )


def _attention_output_program_config(max_seq_len, max_batch_size, hidden_size, mesh_device):
    max_batch = 1 if max_batch_size is None else max(1, max_batch_size)
    if (
        max_seq_len != 512
        or max_batch != 32
        or hidden_size != 1024
        or mesh_device is None
        or not ttnn_is_blackhole(mesh_device)
    ):
        return None
    try:
        device_grid = mesh_device.compute_with_storage_grid_size()
        if device_grid.x < 11 or device_grid.y < 10:
            return None
    except Exception:
        return None
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(11, 10),
        in0_block_w=2,
        out_subblock_h=1,
        out_subblock_w=4,
        per_core_M=5,
        per_core_N=32,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=False,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Norm helpers
# ══════════════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════════════
# Tuned matmul program configs (B1/S512 only)
# Sweep 4.2: AttnOut and MLPwo benefit from explicit 11x10 ibw=8 sub=2x1.
# ══════════════════════════════════════════════════════════════════════════════


def _tuned_mm2d_program_config(
    mesh_device,
    *,
    grid_x,
    grid_y,
    M,
    K,
    N,
    in0_block_w,
    out_subblock_h,
    out_subblock_w,
    fused_activation,
):
    if mesh_device is None:
        return None
    try:
        g = mesh_device.compute_with_storage_grid_size()
        dev_gx, dev_gy = int(g.x), int(g.y)
    except Exception:
        return None
    if dev_gx < grid_x or dev_gy < grid_y:
        return None
    M_tiles = M // 32
    N_tiles = N // 32
    per_core_M = (M_tiles + grid_y - 1) // grid_y
    per_core_N = (N_tiles + grid_x - 1) // grid_x
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(grid_x, grid_y),
        in0_block_w=in0_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        transpose_mcast=False,
        fused_activation=fused_activation,
    )


def _tuned_attention_output_program_config(mesh_device, *, hidden_size):
    return _tuned_mm2d_program_config(
        mesh_device,
        grid_x=11,
        grid_y=10,
        M=512,
        K=hidden_size,
        N=hidden_size,
        in0_block_w=8,
        out_subblock_h=2,
        out_subblock_w=1,
        fused_activation=None,
    )


def _tuned_mlp_wo_program_config(mesh_device, *, hidden_size, intermediate_size):
    return _tuned_mm2d_program_config(
        mesh_device,
        grid_x=11,
        grid_y=10,
        M=512,
        K=intermediate_size,
        N=hidden_size,
        in0_block_w=8,
        out_subblock_h=2,
        out_subblock_w=1,
        fused_activation=None,
    )


def _layernorm_sharded_config(max_seq_len, max_batch_size):
    """Return (program_config, sharded_memcfg) for B1/S512, else (None, None)."""
    max_batch = 1 if max_batch_size is None else max(1, max_batch_size)
    if max_seq_len != 512 or max_batch != 1:
        return None, None

    TILE_SIZE = 32
    core_grid = ttnn.CoreGrid(y=8, x=8)
    core_range = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(core_grid.x - 1, core_grid.y - 1))}
    )
    shard_height = 512 // core_grid.y
    shard_width = 1024 // core_grid.x
    sharded_mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(core_range, [shard_height, shard_width], ttnn.ShardOrientation.ROW_MAJOR),
    )
    program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=(core_grid.x, core_grid.y),
        subblock_w=shard_width // TILE_SIZE,
        block_h=shard_height // TILE_SIZE,
        block_w=shard_width // TILE_SIZE,
        inplace=False,
        use_welford=False,
        legacy_reduction=False,
        legacy_rsqrt=False,
    )
    return program_config, sharded_mem
