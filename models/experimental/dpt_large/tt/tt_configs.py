# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
TTNN-specific configuration helpers.

This module intentionally keeps the interfaces minimal so we can iterate on
sharding/tiling strategies without touching the higher-level pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import math

try:
    import ttnn  # type: ignore
except Exception:  # pragma: no cover
    ttnn = None  # type: ignore

from .config import DPTLargeConfig, DEFAULT_CONFIG


@dataclass
class TTLayerConfig:
    grid: Tuple[int, int]
    dtype: str = "bfloat16"
    shard_tokens: bool = True
    shard_heads: bool = True
    use_fused_ops: bool = True
    math_fidelity: str = "hi-fi2"
    activation_fused: bool = True
    attn_fused_qkv: bool = True
    l1_resident: bool = True
    layout: str = "TILE"
    input_mem: str = "L1"
    output_mem: str = "L1"
    # Perf-specific knobs (populated by `vit_block_config_perf`)
    use_block_sharded: bool = False
    sdpa_grid: Optional[Tuple[int, int]] = None
    qkv_memcfg: Optional[ttnn.MemoryConfig] = None
    proj_memcfg: Optional[ttnn.MemoryConfig] = None
    mlp_memcfg: Optional[ttnn.MemoryConfig] = None
    split_heads_memcfg: Optional[ttnn.MemoryConfig] = None
    qkv_program_config: Optional[object] = None
    proj_program_config: Optional[object] = None
    ff1_program_config: Optional[object] = None
    ff2_program_config: Optional[object] = None
    qk_program_config: Optional[object] = None
    softmax_program_config: Optional[object] = None
    av_program_config: Optional[object] = None
    use_default_attention_programs: bool = False
    ln_program_config: Optional[object] = None
    ln_compute_config: Optional[object] = None

    def memcfg(self):
        """Preferred activation memory config.

        Fast path uses block-sharded L1 to mirror the ViT demo; strict path
        defaults to plain L1 to keep parity and simplicity.
        """
        if ttnn is None:
            return None
        try:
            if self.use_block_sharded:
                return ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG
            return ttnn.L1_MEMORY_CONFIG
        except Exception:
            return None

    def matmul_opts(self, seq_len: int | None = None):
        # Hook to pass sequence length to downstream ops if they care.
        return {"seq_len": seq_len} if seq_len is not None else {}

    def sdpa_program_config(self, seq_len: int):
        """SDPA program config aligned with ViT demo chunking.

        Uses the configured SDPA grid when available; falls back to the
        general L1 chunking scheme if SDPA is unsupported in the runtime.
        """
        if ttnn is None:
            return None
        try:
            SDPAProgramConfig = ttnn.SDPAProgramConfig  # exposed in ttnn.__init__
        except Exception:
            return None

        # Choose chunk sizes that are multiples of 32 (tile size)
        base = min(seq_len, 256)
        base = max(32, (base // 32) * 32)
        q_chunk = base
        # When seq_len is tile-aligned (perf path padding), prefer a power-of-two
        # 32-aligned divisor so masked SDPA accepts q_seq_len/chunk and runtime
        # stats granularity constraints remain valid.
        if seq_len > 0 and (seq_len % 32) == 0:
            max_pow2 = 1 << (base.bit_length() - 1)
            for candidate in (max_pow2, max_pow2 // 2, max_pow2 // 4, max_pow2 // 8):
                if candidate < 32:
                    continue
                if seq_len % candidate == 0:
                    q_chunk = candidate
                    break
        k_chunk = q_chunk
        grid = self.sdpa_grid if self.sdpa_grid is not None else self.grid
        try:
            return SDPAProgramConfig(
                compute_with_storage_grid_size=grid,
                q_chunk_size=q_chunk,
                k_chunk_size=k_chunk,
                exp_approx_mode=False,
            )
        except Exception:
            return None


def vit_block_config(config: DPTLargeConfig = DEFAULT_CONFIG) -> TTLayerConfig:
    if config.device.endswith("n300"):
        # Single-card N300 often exposes a harvested 8x7 worker grid.
        grid = (8, 7)
        math = "hi-fi2"
    elif config.device.endswith("n150"):
        grid = (6, 6)
        math = "hi-fi3"
    elif config.device.endswith("blackhole"):
        grid = (8, 10)
        math = "hi-fi2"
    else:
        grid = (6, 6)
        math = "hi-fi3"
    return TTLayerConfig(grid=grid, math_fidelity=math, attn_fused_qkv=True, activation_fused=True)


def cnn_block_config(config: DPTLargeConfig = DEFAULT_CONFIG) -> TTLayerConfig:
    if config.device.endswith("n300"):
        grid = (6, 6)
    elif config.device.endswith("blackhole"):
        grid = (6, 8)
    else:
        grid = (4, 4)
    return TTLayerConfig(
        grid=grid,
        shard_tokens=False,
        shard_heads=False,
        l1_resident=True,
        use_fused_ops=False,
        activation_fused=False,
        math_fidelity="hi-fi3",
    )


def _build_perf_program_configs(config: DPTLargeConfig, core_grid: Tuple[int, int]):
    """Compute ViT demo-style program configs for the current DPT shape.

    Mirrors `models/demos/wormhole/vit/tt/ttnn_optimized_sharded_vit_wh.py` but
    adapts tile counts to DPT-Large geometry (padded seq len + head count).
    """
    try:
        TILE = 32
        patch_count = config.image_size // config.patch_size
        seq_len = patch_count * patch_count + 1  # include CLS
        seq_len_padded = math.ceil(seq_len / TILE) * TILE
        seqL_t = seq_len_padded // TILE
        dim_t = config.hidden_size // TILE

        core_grid_x, _ = core_grid
        dim_t__x = max(1, dim_t // core_grid_x)
        head_num = config.num_attention_heads
        head_size_t = max(1, (config.hidden_size // head_num) // TILE)
        head_seqL_t__x = max(1, (head_num * seqL_t + core_grid_x - 1) // core_grid_x)

        pc = {
            "layernorm_before_program_config": ttnn.LayerNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=core_grid,
                subblock_w=dim_t__x,
                block_h=seqL_t,
                block_w=dim_t__x,
                inplace=False,
            ),
            "query_key_value_matmul_program_config": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=core_grid,
                in0_block_w=dim_t__x,
                out_subblock_h=1,
                out_subblock_w=dim_t__x,
                per_core_M=seqL_t,
                per_core_N=3 * dim_t__x,
                transpose_mcast=False,
                fused_activation=None,
            ),
            "query_by_key_matmul_program_config": ttnn.MatmulMultiCoreReuseProgramConfig(
                compute_with_storage_grid_size=core_grid,
                in0_block_w=head_size_t,
                out_subblock_h=1,
                out_subblock_w=seqL_t,
                per_core_M=head_seqL_t__x,
                per_core_N=seqL_t,
            ),
            "softmax_program_config": ttnn.SoftmaxShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=core_grid,
                subblock_w=seqL_t,
                block_h=head_seqL_t__x,
                block_w=seqL_t,
            ),
            "attention_probabilities_by_value_matmul_program_config": ttnn.MatmulMultiCoreReuseProgramConfig(
                compute_with_storage_grid_size=core_grid,
                in0_block_w=seqL_t,
                out_subblock_h=1,
                out_subblock_w=head_size_t,
                per_core_M=head_seqL_t__x,
                per_core_N=head_size_t,
            ),
            "self_output_matmul_program_config": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=core_grid,
                in0_block_w=dim_t__x,
                out_subblock_h=1,
                out_subblock_w=dim_t__x,
                per_core_M=seqL_t,
                per_core_N=dim_t__x,
                transpose_mcast=False,
                fused_activation=None,
            ),
            "layernorm_after_output_program_config": ttnn.LayerNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=core_grid,
                subblock_w=dim_t__x,
                block_h=seqL_t,
                block_w=dim_t__x,
                inplace=False,
            ),
            "ff1_matmul_program_config": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=core_grid,
                in0_block_w=dim_t__x,
                out_subblock_h=1,
                out_subblock_w=(dim_t__x * 4) // 2,
                per_core_M=seqL_t,
                per_core_N=dim_t__x * 4,
                transpose_mcast=False,
                fused_activation=(ttnn.UnaryOpType.GELU, True),
            ),
            "ff2_matmul_program_config": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=core_grid,
                in0_block_w=dim_t__x * 4,
                out_subblock_h=1,
                out_subblock_w=dim_t__x,
                per_core_M=seqL_t,
                per_core_N=dim_t__x,
                transpose_mcast=False,
                fused_activation=None,
            ),
            "ln_compute_config": ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi2,
                math_approx_mode=True,
                fp32_dest_acc_en=False,
                packer_l1_acc=True,
            ),
            "_seqL_t": seqL_t,
            "_dim_t__x": dim_t__x,
            "_head_seqL_t__x": head_seqL_t__x,
        }
    except Exception:
        pc = {}
    return pc


def vit_block_config_perf(config: DPTLargeConfig = DEFAULT_CONFIG) -> TTLayerConfig:
    # Aggressive encoder settings for Wormhole N300 perf mode
    if config.device.endswith("n300"):
        # Pick an encoder grid that keeps the fused (B*seq) shard height tile-aligned.
        # For DPT-Large 384: seq_len_padded=640. On an 8x8 device grid, block-sharding
        # would yield shard height 80 which is not tile-aligned (32). Use 8x5 -> 128.
        grid_x = 8
        try:
            TILE = 32
            patch_count = config.image_size // config.patch_size
            seq_len = patch_count * patch_count + 1
            seq_len_padded = math.ceil(seq_len / 64) * 64
            candidates = [y for y in range(8, 0, -1) if (seq_len_padded % y) == 0 and ((seq_len_padded // y) % TILE) == 0]
            grid_y = candidates[0] if candidates else 8
        except Exception:
            grid_y = 8
        grid = (grid_x, grid_y)
        math = "hi-fi2"
    elif config.device.endswith("blackhole"):
        grid = (8, 10)
        math = "hi-fi2"
    else:
        grid = (6, 6)
        math = "hi-fi3"

    prog_cfgs = _build_perf_program_configs(config, grid)
    head_seq_tiles = prog_cfgs.get("_head_seqL_t__x")
    # Disable custom attention configs if forced or if sharding would exceed grid width.
    disable_attn_pc = getattr(config, "tt_force_default_attention_programs", False)
    if head_seq_tiles is not None and head_seq_tiles > grid[0]:
        disable_attn_pc = True

    qk_pc = None if disable_attn_pc else prog_cfgs.get("query_by_key_matmul_program_config")
    softmax_pc = None if disable_attn_pc else prog_cfgs.get("softmax_program_config")
    av_pc = None if disable_attn_pc else prog_cfgs.get("attention_probabilities_by_value_matmul_program_config")
    split_mem = getattr(ttnn, "L1_HEIGHT_SHARDED_MEMORY_CONFIG", None)
    if disable_attn_pc:
        split_mem = getattr(ttnn, "DRAM_MEMORY_CONFIG", None)

    return TTLayerConfig(
        grid=grid,
        math_fidelity=math,
        shard_tokens=True,
        shard_heads=True,
        use_fused_ops=True,
        activation_fused=True,
        l1_resident=True,
        use_block_sharded=False,
        sdpa_grid=grid,
        # Keep encoder activations block-sharded in perf mode so sharded program
        # configs (LN/QKV/FFN) are applicable.
        qkv_memcfg=getattr(ttnn, "L1_BLOCK_SHARDED_MEMORY_CONFIG", None),
        proj_memcfg=getattr(ttnn, "L1_BLOCK_SHARDED_MEMORY_CONFIG", None),
        mlp_memcfg=getattr(ttnn, "L1_BLOCK_SHARDED_MEMORY_CONFIG", None),
        split_heads_memcfg=split_mem,
        qkv_program_config=prog_cfgs.get("query_key_value_matmul_program_config"),
        qk_program_config=qk_pc,
        softmax_program_config=softmax_pc,
        av_program_config=av_pc,
        proj_program_config=prog_cfgs.get("self_output_matmul_program_config"),
        ff1_program_config=prog_cfgs.get("ff1_matmul_program_config"),
        ff2_program_config=prog_cfgs.get("ff2_matmul_program_config"),
        ln_program_config=prog_cfgs.get("layernorm_before_program_config"),
        ln_compute_config=prog_cfgs.get("ln_compute_config"),
        use_default_attention_programs=disable_attn_pc,
    )


def describe_configs(config: DPTLargeConfig = DEFAULT_CONFIG) -> Dict[str, TTLayerConfig]:
    return {
        "vit_block": (
            vit_block_config_perf(config) if getattr(config, "tt_perf_encoder", False) else vit_block_config(config)
        ),
        "cnn_block": cnn_block_config(config),
    }
