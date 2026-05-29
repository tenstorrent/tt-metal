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


def _pick_in0_block_w(k: int, grid: tuple[int, int], width_sharded_in0: bool) -> int:
    """Largest block_w dividing K tiles and (when width-sharded) per-shard K tiles."""
    k_tiles = k // TILE
    divisors = (8, 4, 2, 1)
    if width_sharded_in0:
        k_tiles_per_shard = max(1, k_tiles // (grid[0] * grid[1]))
        for cand in divisors:
            if k_tiles % cand == 0 and k_tiles_per_shard % cand == 0:
                return cand
        return 1
    for cand in divisors:
        if k_tiles % cand == 0:
            return cand
    return 1


def choose_strategy(shape: MatmulShape) -> dict:
    mt = shape.M // TILE
    nt = shape.N // TILE

    if mt <= 4:
        if shape.K >= 4096:
            return {
                "family": "1D",
                "layout": "ws/dram/ws",
                "grid": (8, 4),
                "in0_block_w": 8,  # refined in build_vision_matmul_preset for shard grid
                "per_core_M": mt,
                "per_core_N": max(1, nt // 32),
            }
        return {
            "family": "1D",
            "layout": "l1/dram/ws",
            "grid": (8, 4),
            "in0_block_w": 4,
            "per_core_M": mt,
            "per_core_N": max(1, nt // 32),
        }

    return {
        "family": "2D",
        "layout": "bs/dram/bs",
        "grid": (8, 4),
        "in0_block_w": 8,
        "per_core_M": mt // 4,
        "per_core_N": nt // 8,
    }


def create_memory_configs(
    shape: MatmulShape, cfg: dict
) -> tuple[ttnn.MemoryConfig, ttnn.MemoryConfig, ttnn.MemoryConfig]:
    """Build in0/in1/out memory configs from the layout string 'in0/in1/out'.
    Codes: 'ws' = L1 WIDTH_SHARDED, 'l1' = L1 interleaved, 'dram' = DRAM interleaved."""
    in0_code, in1_code, out_code = cfg["layout"].split("/")

    if in0_code == "ws":
        in0_mem = ttnn.create_sharded_memory_config(
            (1, 1, shape.M, shape.K),
            core_grid=ttnn.CoreGrid(y=cfg["grid"][1], x=cfg["grid"][0]),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        )
    elif in0_code == "dram":
        in0_mem = ttnn.DRAM_MEMORY_CONFIG
    else:  # "l1"
        in0_mem = ttnn.L1_MEMORY_CONFIG

    if out_code == "ws":
        out_mem = ttnn.MemoryConfig(
            memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            buffer_type=ttnn.BufferType.L1,
        )
    elif out_code == "dram":
        out_mem = ttnn.DRAM_MEMORY_CONFIG
    else:  # "l1"
        out_mem = ttnn.L1_MEMORY_CONFIG

    # in1 (weight) is always DRAM interleaved.
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
    """1D mcast + width-sharded output: out_subblock_h==1 always; out_subblock_w
    must divide per_core_N and fit DST register cap (≤8 tiles when fp32_dest_acc_en=False)."""
    for w in (8, 4, 2, 1):
        if per_core_n % w == 0:
            return 1, w
    return 1, 1


def create_program_config(cfg: dict):
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
            fused_activation=None,
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
        fused_activation=None,
    )


def _resolve_grid(mesh_device: ttnn.MeshDevice, cfg: dict) -> tuple[int, int]:
    dev = mesh_device.compute_with_storage_grid_size()
    return (min(dev.x, cfg["grid"][0]), min(dev.y, cfg["grid"][1]))


def build_vision_matmul_preset(mesh_device: ttnn.MeshDevice, m: int, k: int, n: int) -> VisionMatmulPreset:
    shape = MatmulShape(m, k, n)
    cfg = choose_strategy(shape)
    gx, gy = _resolve_grid(mesh_device, cfg)
    grid = (gx, gy)
    cfg = {**cfg, "grid": grid}
    if cfg["layout"].startswith("ws"):
        cfg["in0_block_w"] = _pick_in0_block_w(k, grid, width_sharded_in0=True)
    in0_mem, in1_mem, out_mem = create_memory_configs(shape, cfg)
    return VisionMatmulPreset(
        shape=shape,
        program_config=create_program_config(cfg),
        in0_memory_config=in0_mem,
        in1_memory_config=in1_mem,
        out_memory_config=out_mem,
    )


def build_ffn_down_preset(mesh_device: ttnn.MeshDevice) -> VisionMatmulPreset:
    """Sweep-tuned FFN down preset: 1D ws/dram/ws on 8×2 grid, in0_block_w=8 → 33 TFLOPs.

    Counterintuitive choice: halving cores (32 → 16) doubles per_core_N (1 → 2)
    and lets in0_block_w grow from 4 to 8 (Kt_per_shard goes 128/32=4 → 128/16=8),
    halving the inner-K loop. The default 32-core grid leaves cores
    under-utilized at Mt=4."""
    shape = MatmulShape(*FFN_DOWN_SHAPE)
    dev = mesh_device.compute_with_storage_grid_size()
    gx, gy = min(dev.x, 8), min(dev.y, 2)
    cfg = {
        "family": "1D",
        "layout": "ws/dram/ws",
        "grid": (gx, gy),
        "in0_block_w": 8,
        "per_core_M": shape.M // TILE,  # 4
        "per_core_N": max(1, (shape.N // TILE) // (gx * gy)),  # 32 / 16 = 2
    }
    in0_mem, in1_mem, out_mem = create_memory_configs(shape, cfg)
    return VisionMatmulPreset(
        shape=shape,
        program_config=create_program_config(cfg),
        in0_memory_config=in0_mem,
        in1_memory_config=in1_mem,
        out_memory_config=out_mem,
    )


def build_ffn_up_preset(mesh_device: ttnn.MeshDevice) -> VisionMatmulPreset:
    """Sweep-tuned FFN gate/up preset: 1D l1/dram/ws on 8×8 grid, in0_block_w=4.

    Sweep winner was ``dram/dram/ws 8×8 w4`` at 50 TFLOPs. We use ``l1`` in0
    here to match the actual upstream layout (ffn_norm rms_norm outputs L1)
    and avoid an L1→DRAM convert per call. Same grid, same w; should land
    very close to the dram-in0 number."""
    shape = MatmulShape(*FFN_UP_SHAPE)
    dev = mesh_device.compute_with_storage_grid_size()
    gx, gy = min(dev.x, 8), min(dev.y, 8)
    cfg = {
        "family": "1D",
        "layout": "l1/dram/ws",
        "grid": (gx, gy),
        "in0_block_w": 4,
        "per_core_M": shape.M // TILE,  # 4
        "per_core_N": max(1, (shape.N // TILE) // (gx * gy)),  # 128 / 64 = 2
    }
    in0_mem, in1_mem, out_mem = create_memory_configs(shape, cfg)
    return VisionMatmulPreset(
        shape=shape,
        program_config=create_program_config(cfg),
        in0_memory_config=in0_mem,
        in1_memory_config=in1_mem,
        out_memory_config=out_mem,
    )


def build_o_proj_preset(mesh_device: ttnn.MeshDevice) -> VisionMatmulPreset:
    """Sweep-tuned O-projection preset: 1D l1/dram/ws on 8×4 grid with
    in0_block_w=16 → 11 µs, 24 TFLOPs.

    Same grid as the heuristic but ``in0_block_w=16`` (vs default 4) cuts the
    inner-K loop from 8 iters to 2. ``_pick_in0_block_w`` doesn't see 16
    because its divisor pool is ``(8, 4, 2, 1)``."""
    shape = MatmulShape(*O_PROJ_SHAPE)
    dev = mesh_device.compute_with_storage_grid_size()
    gx, gy = min(dev.x, 8), min(dev.y, 4)
    cfg = {
        "family": "1D",
        "layout": "l1/dram/ws",
        "grid": (gx, gy),
        "in0_block_w": 16,
        "per_core_M": shape.M // TILE,  # 4
        "per_core_N": max(1, (shape.N // TILE) // (gx * gy)),  # 32 / 32 = 1
    }
    in0_mem, in1_mem, out_mem = create_memory_configs(shape, cfg)
    return VisionMatmulPreset(
        shape=shape,
        program_config=create_program_config(cfg),
        in0_memory_config=in0_mem,
        in1_memory_config=in1_mem,
        out_memory_config=out_mem,
    )


def build_qkv_preset(mesh_device: ttnn.MeshDevice) -> VisionMatmulPreset:
    """Sweep-tuned QKV preset (test_vision_matmul_sweep.py on Blackhole P150):
    1D l1/dram/ws on 8×6 grid with in0_block_w=4 → 16 µs, 49 TFLOPs.

    Bypasses ``choose_strategy`` because its heuristic clamps grid_y to 4 and
    misses the 8×6 win — Mt=4 with N split as per_core_N=2 across 48 cores."""
    shape = MatmulShape(*QKV_SHAPE)
    dev = mesh_device.compute_with_storage_grid_size()
    gx, gy = min(dev.x, 8), min(dev.y, 6)
    cfg = {
        "family": "1D",
        "layout": "l1/dram/ws",
        "grid": (gx, gy),
        "in0_block_w": 4,
        "per_core_M": shape.M // TILE,  # 4
        "per_core_N": max(1, (shape.N // TILE) // (gx * gy)),  # 96 / 48 = 2
    }
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
    in0_bf8: bool = False,
) -> ttnn.Tensor:
    """Run ``ttnn.linear`` with a vision matmul preset (program + memory configs).

    By default the post-matmul interleaved output is DRAM (most downstream vision
    ops expect that). Pass ``output_memory_config=ttnn.L1_MEMORY_CONFIG`` to keep
    it in L1 instead — useful when the next op also lives in L1 (avoids DRAM
    round-trips). ``keep_sharded=True`` returns the raw sharded output untouched.
    ``in0_bf8=True`` typecasts the activation to bfloat8_b before the matmul.
    """
    in0 = x
    if _needs_in0_prepare(preset):
        in0 = ttnn.to_memory_config(x, preset.in0_memory_config)
    if in0_bf8 and in0.dtype != ttnn.bfloat8_b:
        in0 = ttnn.typecast(in0, ttnn.bfloat8_b)

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
