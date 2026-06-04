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
    wo_minimal_config: object | None = None


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
        tuned_b16 = max_seq_len == 512 and max_batch == 16
        if tuned_b1:
            wo_prg_tuned = _tuned_mlp_wo_program_config(
                mesh_device, hidden_size=hidden_size, intermediate_size=intermediate_size
            )
        elif tuned_b16:
            wo_prg_tuned = _b16_tuned_mlp_wo_program_config(
                mesh_device, hidden_size=hidden_size, intermediate_size=intermediate_size
            )
        else:
            wo_prg_tuned = None
        wo_minimal = _mlp_wo_minimal_matmul_config(
            mesh_device,
            max_seq_len,
            max_batch,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )

        mlp_opts = MLPOptimizations(
            wi_compute_kernel_cfg=mlp_wi_compute_kernel_config(mesh_device, max_seq_len, max_batch, dtype=dtype),
            wo_compute_kernel_cfg=mlp_wo_compute_kernel_config(mesh_device, max_seq_len, max_batch, dtype=dtype),
            wi_memcfg=_mlp_wi_output_memory_config(max_seq_len, max_batch, mesh_device),
            wo_memcfg=_mlp_wo_output_memory_config(max_seq_len, max_batch, mesh_device),
            activation_memcfg=act_mem,
            core_grid=core_grid,
            wi_prg_config=wi_prg,
            wo_prg_config=None if wo_minimal is not None else wo_prg_tuned,
            wi_minimal_config=wi_minimal,
            wo_minimal_config=wo_minimal,
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
                else _b16_tuned_attention_output_program_config(mesh_device, hidden_size=hidden_size)
                if tuned_b16
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


# B8 (b*s=4096) DRAM activations. NOTE: tried extending the B32 L1 output
# overrides to B8 (mlp_wi/wo, attn_output, create_heads) — ALL variants, even
# create_heads alone, fail with "static CBs clash with L1 buffers" on the 11x10
# grid: the SDPA/matmul static circular buffers leave no L1 headroom. Dead end
# unless the SDPA grid is shrunk first to free L1. Kept at default (DRAM).
def _mlp_wi_output_memory_config(max_seq_len, max_batch_size, mesh_device):
    max_batch = 1 if max_batch_size is None else max(1, max_batch_size)
    if max_seq_len == 512 and max_batch in (8, 32) and mesh_device is not None and ttnn_is_blackhole(mesh_device):
        return ttnn.L1_MEMORY_CONFIG
    return _linear_activation_memory_config(max_seq_len, max_batch_size)


def _mlp_wo_output_memory_config(max_seq_len, max_batch_size, mesh_device):
    max_batch = 1 if max_batch_size is None else max(1, max_batch_size)
    if max_seq_len == 512 and max_batch in (8, 32) and mesh_device is not None and ttnn_is_blackhole(mesh_device):
        return ttnn.L1_MEMORY_CONFIG
    return _linear_activation_memory_config(max_seq_len, max_batch_size)


def _attention_output_memory_config(max_seq_len, max_batch_size, mesh_device):
    max_batch = 1 if max_batch_size is None else max(1, max_batch_size)
    if max_seq_len == 512 and max_batch in (8, 32) and mesh_device is not None and ttnn_is_blackhole(mesh_device):
        return ttnn.L1_MEMORY_CONFIG
    return _linear_activation_memory_config(max_seq_len, max_batch)


def _create_heads_output_memory_config(max_seq_len, max_batch_size, mesh_device):
    max_batch = 1 if max_batch_size is None else max(1, max_batch_size)
    # B8: create_heads output to L1 clashes with SDPA static CBs (program 15,
    # 11x10 grid) - reverted. Stays at default (DRAM) for B8.
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
    # B8 shares the B1/B32 fidelity policy (LoFi for bf8, HiFi2 otherwise) on the
    # S512 shapes; fp32_dest_acc_en=False lifts subblock h*w cap to 8.
    if max_seq_len == 512 and max_batch in (1, 8, 16, 32):
        fid = ttnn.MathFidelity.LoFi if dtype == ttnn.bfloat8_b else ttnn.MathFidelity.HiFi2
        return _make_compute_kernel(mesh_device, fid, max_seq_len, max_batch, fp32_dest_acc_en=False)
    return matmul_compute_kernel_config(mesh_device, max_seq_len, max_batch)


def mlp_wo_compute_kernel_config(mesh_device, max_seq_len=None, max_batch_size=None, dtype=None):
    max_batch = 1 if max_batch_size is None else max(1, max_batch_size)
    if max_seq_len == 512 and max_batch == 1:
        return _make_compute_kernel(mesh_device, ttnn.MathFidelity.LoFi, max_seq_len, max_batch)
    if max_seq_len == 512 and max_batch in (8, 16, 32):
        # fp32_dest_acc_en=False for MinimalMatmul subblock 8x1 (h*w=8 > 4 cap).
        return _make_compute_kernel(mesh_device, ttnn.MathFidelity.LoFi, max_seq_len, max_batch, fp32_dest_acc_en=False)
    return matmul_compute_kernel_config(mesh_device, max_seq_len, max_batch)


def attention_qkv_compute_kernel_config(mesh_device, max_seq_len=None, max_batch_size=None, dtype=None):
    max_batch = 1 if max_batch_size is None else max(1, max_batch_size)
    if max_seq_len == 512 and max_batch in (1, 8, 16, 32):
        fid = ttnn.MathFidelity.LoFi if dtype == ttnn.bfloat8_b else ttnn.MathFidelity.HiFi2
        return _make_compute_kernel(mesh_device, fid, max_seq_len, max_batch, fp32_dest_acc_en=False)
    return matmul_compute_kernel_config(mesh_device, max_seq_len, max_batch)


def attention_output_compute_kernel_config(mesh_device, max_seq_len=None, max_batch_size=None, dtype=None):
    max_batch = 1 if max_batch_size is None else max(1, max_batch_size)
    if max_seq_len == 512 and max_batch in (1, 8, 16, 32):
        fid = ttnn.MathFidelity.LoFi if dtype == ttnn.bfloat8_b else ttnn.MathFidelity.HiFi2
        return _make_compute_kernel(mesh_device, fid, max_seq_len, max_batch, fp32_dest_acc_en=False)
    return matmul_compute_kernel_config(mesh_device, max_seq_len, max_batch)


def sdpa_compute_kernel_config(mesh_device, max_seq_len=None, max_batch_size=None, dtype=None):
    max_batch = 1 if max_batch_size is None else max(1, max_batch_size)
    if max_seq_len == 512 and max_batch == 1:
        fid = ttnn.MathFidelity.LoFi if dtype == ttnn.bfloat8_b else ttnn.MathFidelity.HiFi2
        return _make_compute_kernel(mesh_device, fid, max_seq_len, max_batch)
    # NOTE: B16 SDPA LoFi gives no speedup (bandwidth-bound, not compute-bound)
    # and drops PCC to 0.9357 (< 0.94 gate). Kept HiFi2.
    if max_seq_len == 512 and max_batch in (8, 16, 32):
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
    # NOTE: B16 LN with fp32_dest_acc_en=False saves ~0.5ms but drops PCC to
    # 0.9359 (below 0.94 gate) — reverted. B16 keeps fp32_dest_acc_en=True.
    if max_seq_len == 512 and max_batch in (1, 8, 16, 32):
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
    if max_batch == 16:
        # NOTE: B32's _b32s512_sequence_program_config (per_core_M=seq_tiles/grid_y,
        # fuse_batch=False) gives 56.3ms for B16 — it's designed for B32's batched
        # layout, not B16's fused M=8192. The 2D fused config below is correct.
        # Sweep winner for M=8192 K=1024 N=4096+GELU on the 11x10 grid:
        # ibw=4 sub=2x4 = 277.7 µs vs 810 µs default ttnn.linear routing (2.9x).
        return _b16s512_mlp_wi_program_config(mesh_device, hidden_size=hidden_size, intermediate_size=intermediate_size)
    if max_batch == 8:
        # Sweep winner for M=4096 K=1024 N=4096+GELU on the 11x10 grid:
        # ibw=8 sub=1x4 = 101.9 µs vs 400 µs default ttnn.linear routing (3.9x).
        return _b8s512_mlp_wi_program_config(mesh_device, hidden_size=hidden_size, intermediate_size=intermediate_size)
    if max_batch == 1:
        return _b1s512_mlp_wi_program_config(mesh_device, hidden_size=hidden_size, intermediate_size=intermediate_size)
    return None


def _b16s512_mlp_wi_program_config(mesh_device, *, hidden_size, intermediate_size):
    grid_x, grid_y = 11, 10
    if mesh_device is None or not ttnn_is_blackhole(mesh_device):
        return None
    try:
        g = mesh_device.compute_with_storage_grid_size()
        if int(g.x) < grid_x or int(g.y) < grid_y:
            return None
    except Exception:
        return None
    m_tiles = (16 * 512) // 32  # 256
    hidden_tiles = hidden_size // 32
    intermediate_tiles = intermediate_size // 32
    # NOTE: 1D mcast_in1 (per_core_N=full 128 tiles) overflows L1 (2.6MB > 1.57MB)
    # for N=4096. All in0_block_w=8 variants also overflow L1 regardless of
    # subblock size (8 K-tiles x per_core_N too big). ibw=4 sub=2x4 is the best
    # feasible 2D mcast config for B16 MLP-wi.
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(grid_x, grid_y),
        in0_block_w=min(4, hidden_tiles),
        out_subblock_h=2,
        out_subblock_w=4,
        per_core_M=(m_tiles + grid_y - 1) // grid_y,
        per_core_N=(intermediate_tiles + grid_x - 1) // grid_x,
        transpose_mcast=False,
        fused_activation=(ttnn.UnaryOpType.GELU, True),
    )


def _b8s512_mlp_wi_program_config(mesh_device, *, hidden_size, intermediate_size):
    grid_x, grid_y = 11, 10
    if mesh_device is None or not ttnn_is_blackhole(mesh_device):
        return None
    try:
        g = mesh_device.compute_with_storage_grid_size()
        if int(g.x) < grid_x or int(g.y) < grid_y:
            return None
    except Exception:
        return None
    m_tiles = (8 * 512) // 32  # 128
    hidden_tiles = hidden_size // 32
    intermediate_tiles = intermediate_size // 32
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(grid_x, grid_y),
        in0_block_w=min(8, hidden_tiles),
        out_subblock_h=1,
        out_subblock_w=4,
        per_core_M=(m_tiles + grid_y - 1) // grid_y,
        per_core_N=(intermediate_tiles + grid_x - 1) // grid_x,
        transpose_mcast=False,
        fused_activation=(ttnn.UnaryOpType.GELU, True),
    )


def _b1s512_mlp_wi_program_config(mesh_device, *, hidden_size, intermediate_size):
    # MLPwi (FF1, 512x1024x4096 +GELU) is the largest B1/S512 matmul. A 120-core
    # sweep (sweep_mlp_120cores.py) showed 12x10 beats 11x10 by ~6.9% for this op
    # only (25.2us vs 27.1us standalone): widening to 12 wide makes per_core_N=11
    # and fills 96 cores instead of 88. Since per_core_N=11 is not divisible by 2,
    # the winning subblock is 2x1 (not 1x2). MLPwo/AttnOut showed no gain (N=1024
    # gives per_core_N=3 at both 11 and 12 wide), so they stay at 11x10.
    #
    # Guard: this tuned config targets Blackhole's wide grid (>=11 columns x 10
    # rows). On narrower devices (e.g. Wormhole N150/N300 with an 8x8 grid) the
    # 11/12-wide grid coordinates don't exist, so return None and let ttnn pick a
    # portable default program config. Mirrors the B8/B16 guards below.
    grid_x_req, grid_y_req = 11, 10
    if mesh_device is None or not ttnn_is_blackhole(mesh_device):
        return None
    try:
        g = mesh_device.compute_with_storage_grid_size()
        if int(g.x) < grid_x_req or int(g.y) < grid_y_req:
            return None
        dev_gx = int(g.x)
    except Exception:
        return None
    hidden_tiles = hidden_size // 32
    m_tiles = 512 // 32
    intermediate_tiles = intermediate_size // 32
    # Only use 12 wide when the device actually exposes >=12 columns (Galaxy
    # Blackhole). On an 11-wide device fall back to the original 11x10 / sub 1x2.
    if dev_gx >= 12:
        grid_x, grid_y = 12, 10
        out_subblock_h, out_subblock_w = 2, 1
    else:
        grid_x, grid_y = 11, 10
        out_subblock_h, out_subblock_w = 1, 2
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(grid_x, grid_y),
        in0_block_w=min(4, hidden_tiles),
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=(m_tiles + grid_y - 1) // grid_y,
        per_core_N=(intermediate_tiles + grid_x - 1) // grid_x,
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


def _mlp_wo_minimal_matmul_config(mesh_device, max_seq_len, max_batch_size, *, hidden_size, intermediate_size):
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


# NOTE: B16 QKV (M=8192 K=1024 N=3072) explicit 2D mcast configs were tried
# (ibw8 sub2x1 overflows L1; ibw4 sub2x1 runs 51.22ms, worse than the 50.88ms
# default routing). The default ttnn.linear path is best for B16 QKV — not
# overridden.
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
    # Manual tune: ibw=4 + sub=1x8 (h*w=8, only allowed with fp32_dest=False)
    # saves ~150 us wall vs sub=1x4 (-7us/call x 24).
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(11, 10),
        in0_block_w=4,
        out_subblock_h=1,
        out_subblock_w=8,
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


# B16/S512 (M=8192) tuned matmul configs from the b16_matmuls sweep:
#   AttnOut: g11x10 ibw8 sub2x1 = 89.1 us vs 100.7 us default (1.13x)
#   MLPwo:   g11x10 ibw8 sub2x1 = 222.8 us vs 343 us default (1.54x)
# QKV default (218.9 us) was optimal so it is not overridden.
def _b16_tuned_attention_output_program_config(mesh_device, *, hidden_size):
    return _tuned_mm2d_program_config(
        mesh_device,
        grid_x=11,
        grid_y=10,
        M=16 * 512,
        K=hidden_size,
        N=hidden_size,
        in0_block_w=8,
        out_subblock_h=2,
        out_subblock_w=1,
        fused_activation=None,
    )


def _b16_tuned_mlp_wo_program_config(mesh_device, *, hidden_size, intermediate_size):
    return _tuned_mm2d_program_config(
        mesh_device,
        grid_x=11,
        grid_y=10,
        M=16 * 512,
        K=intermediate_size,
        N=hidden_size,
        in0_block_w=8,
        out_subblock_h=2,
        out_subblock_w=1,
        fused_activation=None,
    )


def _layernorm_sharded_config(max_seq_len, max_batch_size):
    """Return (program_config, sharded_memcfg) for B1/S512, else (None, None).

    NOTE: a B8 sharded LN (shard_height=512=16 tiles/core, 8x B1) overflows L1
    against the matmul L1-output buffers now in place (CB clash, program 19).
    Kept B1-only.
    """
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
