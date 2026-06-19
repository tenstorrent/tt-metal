# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Matmul program / memory configs for Pixtral vision linear layers (seq M=128 tiles).

Tuned from ``tests/test_matmul_config.py`` for the five shapes seen in vision-tower
Tracy reports on Blackhole P150 (8×4 compute grid).
"""

from __future__ import annotations

from dataclasses import dataclass

import ttnn

TILE = 32

# Shapes from vision-tower Tracy (M=128 token tiles, K×N per matmul).
VISION_MM_SEQ_LEN = 128
QKV_SHAPE = (VISION_MM_SEQ_LEN, 1024, 3072)
O_PROJ_SHAPE = (VISION_MM_SEQ_LEN, 1024, 1024)
FFN_UP_SHAPE = (VISION_MM_SEQ_LEN, 1024, 4096)
FFN_DOWN_SHAPE = (VISION_MM_SEQ_LEN, 4096, 1024)


@dataclass(frozen=True)
class MatmulShape:
    M: int
    K: int
    N: int


@dataclass(frozen=True)
class VisionMatmulPreset:
    shape: MatmulShape
    program_config: object
    in0_memory_config: ttnn.MemoryConfig
    in1_memory_config: ttnn.MemoryConfig
    out_memory_config: ttnn.MemoryConfig


def create_memory_configs(
    shape: MatmulShape, cfg: dict
) -> tuple[ttnn.MemoryConfig, ttnn.MemoryConfig, ttnn.MemoryConfig]:
    layout = cfg["layout"]

    if layout.startswith("ws"):
        in0_mem = ttnn.create_sharded_memory_config(
            (1, 1, shape.M, shape.K),
            core_grid=ttnn.CoreGrid(y=cfg["grid"][1], x=cfg["grid"][0]),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        )
    else:
        in0_mem = ttnn.L1_MEMORY_CONFIG

    if layout.endswith("ws"):
        out_mem = ttnn.MemoryConfig(
            memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            buffer_type=ttnn.BufferType.L1,
        )
    else:
        out_mem = ttnn.L1_MEMORY_CONFIG

    in1_mem = ttnn.DRAM_MEMORY_CONFIG
    return in0_mem, in1_mem, out_mem


def _pick_out_subblocks_2d(per_core_m: int, per_core_n: int) -> tuple[int, int]:
    """Pick subblock sizes for 2D multicast matmul."""
    out_subblock_w = 1
    for cand in (4, 2, 1):
        if per_core_n % cand == 0:
            out_subblock_w = cand
            break
    out_subblock_h = 1
    for cand in (4, 2, 1):
        if per_core_m % cand == 0 and cand * out_subblock_w <= 8:
            out_subblock_h = cand
            break
    return out_subblock_h, out_subblock_w


def _pick_out_subblocks_1d(per_core_n: int) -> tuple[int, int]:
    """1D mcast + width-sharded output: out_subblock_w == per_core_N or out_subblock_h == 1."""
    return 1, per_core_n


def create_program_config(cfg: dict, fused_activation=None):
    """Build the matmul program config. ``fused_activation`` bakes a unary epilogue
    (e.g. ``ttnn.UnaryOpType.SILU``) into the matmul kernel — necessary because the
    explicit ``activation`` kwarg on ``ttnn.linear`` is ignored when ``program_config``
    is set (the PC's ``fused_activation`` field takes precedence)."""
    if cfg["family"] == "1D":
        out_subblock_h, out_subblock_w = _pick_out_subblocks_1d(cfg["per_core_N"])
    else:
        out_subblock_h, out_subblock_w = _pick_out_subblocks_2d(cfg["per_core_M"], cfg["per_core_N"])
    if cfg["family"] == "1D":
        return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(cfg["grid"][0], cfg["grid"][1]),
            in0_block_w=cfg["in0_block_w"],
            per_core_M=cfg["per_core_M"],
            per_core_N=cfg["per_core_N"],
            out_subblock_h=out_subblock_h,
            out_subblock_w=out_subblock_w,
            fuse_batch=True,
            fused_activation=fused_activation,
            mcast_in0=True,
        )

    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(cfg["grid"][0], cfg["grid"][1]),
        in0_block_w=cfg["in0_block_w"],
        per_core_M=cfg["per_core_M"],
        per_core_N=cfg["per_core_N"],
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        transpose_mcast=False,
        fused_activation=fused_activation,
    )


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _pick_in0_block_w(
    mt: int,
    kt_total: int,
    *,
    kt_per_shard: int | None = None,
    max_block_w: int = 16,
    is_ws_in0: bool = False,
) -> int:
    """Largest in0_block_w that (a) divides Kt (and per-shard Kt for ws-in0)
    and (b) keeps the in0 CB L1 footprint inside our budget.

    The matmul kernel double-buffers each in0 CB, and the ws-in0 mcast path
    additionally allocates a separate receiver CB — so total in0 CB L1 cost
    per core is:
        l1-in0 (single mcast CB, dbuf):    2 × per_core_M × in0_block_w tiles
        ws-in0 (sender + receiver, dbuf):  4 × per_core_M × in0_block_w tiles

    We aim to keep the in0 CB footprint ≤ ~250 KB per core (leaving headroom
    for in1/out/intermediate CBs + sharded buffers). On P150x8 with hidden M
    ≈ 1152 (mt=36) this lands at:
        l1-in0 → in0_block_w = 2 (CB ≈ 144 KB)
        ws-in0 → in0_block_w = 1 (CB ≈ 144 KB)
    """
    in0_cb_tile_budget = 250  # ≈ 250 KB at bf16
    multiplier = 4 if is_ws_in0 else 2
    candidates = (16, 8, 4, 2, 1)
    for cand in candidates:
        if cand > max_block_w:
            continue
        if kt_total % cand != 0:
            continue
        if kt_per_shard is not None and kt_per_shard % cand != 0:
            continue
        if multiplier * mt * cand > in0_cb_tile_budget:
            continue
        return cand
    return 1


def _pick_per_core_n(nt: int, max_cells: int) -> int:
    """Smallest per_core_N that (a) keeps ceil(nt/per_core_N) ≤ max_cells, and
    (b) divides nt evenly, so the width-sharded output has uniform shard widths
    across all cores. Downstream ops (gate*up multiply, ffn_down's ws-in0
    reshard) expect uniform ws shards.
    """
    min_per_core_n = max(1, _ceil_div(nt, max_cells))
    for cand in range(min_per_core_n, nt + 1):
        if nt % cand == 0:
            return cand
    return min_per_core_n


def build_ffn_down_preset(mesh_device: ttnn.MeshDevice, m: int = VISION_MM_SEQ_LEN) -> VisionMatmulPreset:
    """Sweep-tuned FFN down preset: 1D ws/dram/ws on 8×2 grid, in0_block_w=8 → 33 TFLOPs.

    Tuned at M=128. ``per_core_M`` and per_core_N are scaled with the actual
    sequence length so the matmul kernel's core count never exceeds the grid.
    """
    K, N = FFN_DOWN_SHAPE[1], FFN_DOWN_SHAPE[2]
    mt = max(1, _ceil_div(m, TILE))
    nt = N // TILE
    kt = K // TILE
    dev = mesh_device.compute_with_storage_grid_size()
    gx, gy = min(dev.x, 8), min(dev.y, 2)
    cfg = {
        "family": "1D",
        "layout": "ws/dram/ws",
        "grid": (gx, gy),
        # in0 is ws-sharded → block_w must divide kt_per_shard. ws mcast keeps
        # both a sender and receiver CB, so the in0 CB budget is tighter here.
        "in0_block_w": _pick_in0_block_w(
            mt,
            kt,
            kt_per_shard=max(1, kt // (gx * gy)),
            max_block_w=8,
            is_ws_in0=True,
        ),
        "per_core_M": mt,
        "per_core_N": _pick_per_core_n(nt, gx * gy),
    }
    shape = MatmulShape(mt * TILE, K, N)
    in0_mem, in1_mem, out_mem = create_memory_configs(shape, cfg)
    return VisionMatmulPreset(
        shape=shape,
        program_config=create_program_config(cfg),
        in0_memory_config=in0_mem,
        in1_memory_config=in1_mem,
        out_memory_config=out_mem,
    )


def build_ffn_up_preset(
    mesh_device: ttnn.MeshDevice,
    m: int = VISION_MM_SEQ_LEN,
    activation=None,
) -> VisionMatmulPreset:
    """Sweep-tuned FFN gate/up preset: 1D l1/dram/ws on 8×8 grid, in0_block_w=4.

    Tuned at M=128. ``per_core_M`` and per_core_N are scaled with the actual
    sequence length so the matmul kernel's core count never exceeds the grid.
    Uses ceil-division on per_core_N to stay safe when the device grid is
    smaller than the 8×8 design point (e.g. P150 with compute grid 8×6).

    ``activation``: optional ``ttnn.UnaryOpType`` (e.g. ``ttnn.UnaryOpType.SILU``)
    baked into the matmul kernel as a fused epilogue — saves a separate Unary op
    after the matmul. Used for the gate-proj path of SwiGLU FFNs.
    """
    K, N = FFN_UP_SHAPE[1], FFN_UP_SHAPE[2]
    mt = max(1, _ceil_div(m, TILE))
    nt = N // TILE
    kt = K // TILE
    dev = mesh_device.compute_with_storage_grid_size()
    gx, gy = min(dev.x, 8), min(dev.y, 8)
    cfg = {
        "family": "1D",
        "layout": "l1/dram/ws",
        "grid": (gx, gy),
        "in0_block_w": _pick_in0_block_w(mt, kt, max_block_w=4),
        "per_core_M": mt,
        "per_core_N": _pick_per_core_n(nt, gx * gy),
    }
    shape = MatmulShape(mt * TILE, K, N)
    in0_mem, in1_mem, out_mem = create_memory_configs(shape, cfg)
    return VisionMatmulPreset(
        shape=shape,
        program_config=create_program_config(cfg, fused_activation=activation),
        in0_memory_config=in0_mem,
        in1_memory_config=in1_mem,
        out_memory_config=out_mem,
    )


def build_o_proj_preset(mesh_device: ttnn.MeshDevice, m: int = VISION_MM_SEQ_LEN) -> VisionMatmulPreset:
    """O-projection preset: 1D l1/dram/ws on 8×4 grid (32 cores).

    Tuned at M=128 with in0_block_w=16. NB: tried 8×2 (per_core_N=2, fixes
    tracy's 1×1 subblock advice and lifts FLOP utilization from 26%→46%) but it
    regressed wall-clock by ~1 μs — at K=1024 the per-core kernel finishes
    faster than the parallelism loss costs. The lower-FLOP-but-higher-core-count
    config wins here. Compare to ffn_down (same per_core layout, K=4096) where
    16 cores wins because K is 4× longer.
    """
    K, N = O_PROJ_SHAPE[1], O_PROJ_SHAPE[2]
    mt = max(1, _ceil_div(m, TILE))
    nt = N // TILE
    kt = K // TILE
    dev = mesh_device.compute_with_storage_grid_size()
    gx, gy = min(dev.x, 8), min(dev.y, 4)
    cfg = {
        "family": "1D",
        "layout": "l1/dram/ws",
        "grid": (gx, gy),
        "in0_block_w": _pick_in0_block_w(mt, kt, max_block_w=16),
        "per_core_M": mt,
        "per_core_N": _pick_per_core_n(nt, gx * gy),
    }
    shape = MatmulShape(mt * TILE, K, N)
    in0_mem, in1_mem, out_mem = create_memory_configs(shape, cfg)
    return VisionMatmulPreset(
        shape=shape,
        program_config=create_program_config(cfg),
        in0_memory_config=in0_mem,
        in1_memory_config=in1_mem,
        out_memory_config=out_mem,
    )


def build_qkv_preset(mesh_device: ttnn.MeshDevice, m: int = VISION_MM_SEQ_LEN) -> VisionMatmulPreset:
    """Sweep-tuned QKV preset (test_vision_matmul_sweep.py on Blackhole P150):
    1D l1/dram/ws on 8×6 grid with in0_block_w=4 → 16 µs, 49 TFLOPs at M=128.

    ``per_core_M`` and per_core_N are scaled with the actual sequence length
    so the matmul kernel's core count never exceeds the grid (was hardcoded
    to mt=4, which crashes when num_patches > 128).
    """
    K, N = QKV_SHAPE[1], QKV_SHAPE[2]
    mt = max(1, _ceil_div(m, TILE))
    nt = N // TILE
    kt = K // TILE
    dev = mesh_device.compute_with_storage_grid_size()
    gx, gy = min(dev.x, 8), min(dev.y, 6)
    cfg = {
        "family": "1D",
        "layout": "l1/dram/ws",
        "grid": (gx, gy),
        "in0_block_w": _pick_in0_block_w(mt, kt, max_block_w=4),
        "per_core_M": mt,
        "per_core_N": _pick_per_core_n(nt, gx * gy),
    }
    shape = MatmulShape(mt * TILE, K, N)
    in0_mem, in1_mem, out_mem = create_memory_configs(shape, cfg)
    return VisionMatmulPreset(
        shape=shape,
        program_config=create_program_config(cfg),
        in0_memory_config=in0_mem,
        in1_memory_config=in1_mem,
        out_memory_config=out_mem,
    )


class VisionMatmulConfigs:
    """Pre-built presets for vision attention + MLP matmuls at M=128."""

    def __init__(self, mesh_device: ttnn.MeshDevice):
        self.qkv = build_qkv_preset(mesh_device)
        self.o_proj = build_o_proj_preset(mesh_device)
        self.ffn_gate_up = build_ffn_up_preset(mesh_device)
        self.ffn_down = build_ffn_down_preset(mesh_device)


def _needs_in0_prepare(preset: VisionMatmulPreset) -> bool:
    return preset.in0_memory_config != ttnn.DRAM_MEMORY_CONFIG


def vision_linear(
    x: ttnn.Tensor,
    weight: ttnn.Tensor,
    preset: VisionMatmulPreset,
    *,
    compute_kernel_config,
    dtype=ttnn.bfloat16,
    activation=None,
    keep_sharded: bool = False,
    output_memory_config: ttnn.MemoryConfig | None = None,
    deallocate_input: bool = False,
) -> ttnn.Tensor:
    """Run ``ttnn.linear`` with a vision matmul preset (program + memory configs).

    By default the post-matmul interleaved output is DRAM (most downstream vision
    ops expect that). Pass ``output_memory_config=ttnn.L1_MEMORY_CONFIG`` to keep
    it in L1 instead — useful when the next op also lives in L1 (avoids DRAM
    round-trips). ``keep_sharded=True`` returns the raw sharded output untouched.

    ``deallocate_input=True``: when the preset's in0 layout differs from x's
    layout we have to reshard. Setting this frees ``x`` immediately after the
    reshard, before ``ttnn.linear`` allocates its CBs — required when the old
    and new ws shards both land on the same cores (e.g. ffn_down resharding
    from the ffn_up output grid).
    """
    in0 = x
    if _needs_in0_prepare(preset):
        in0 = ttnn.to_memory_config(x, preset.in0_memory_config)
        if deallocate_input and in0 is not x:
            ttnn.deallocate(x)

    out = ttnn.linear(
        in0,
        weight,
        activation=activation,
        compute_kernel_config=compute_kernel_config,
        dtype=dtype,
        program_config=preset.program_config,
        memory_config=preset.out_memory_config,
    )

    if keep_sharded:
        return out

    target = output_memory_config if output_memory_config is not None else ttnn.DRAM_MEMORY_CONFIG
    if preset.out_memory_config.memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED:
        out = ttnn.sharded_to_interleaved(out, target)
    elif preset.out_memory_config != target:
        out = ttnn.to_memory_config(out, target)
    return out
