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
    # Attention can benefit from a different core grid than the encoder-wide
    # block sharding. For DPT-Large, we want attention height-sharding to align
    # with head sharding (num_heads cores) so per-core M is a multiple of seq tiles.
    attn_grid: Optional[Tuple[int, int]] = None
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
    # Stage-2 hybrid attention: SDPA currently rejects sharded operands. We keep
    # tokens sharded across blocks, but explicitly interleave around the SDPA
    # "island" using this memory config.
    attn_island_memcfg: Optional[ttnn.MemoryConfig] = None
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
    # Optional override core grid for the MLP path. Useful when tokens are
    # interleaved but we want to reshard only for the MLP matmuls.
    mlp_core_grid: Optional[Tuple[int, int]] = None

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
        # Hook to pass sequence length and layout knobs to downstream ops.
        opts: dict[str, object] = {}
        if seq_len is not None:
            opts["seq_len"] = seq_len
        if self.attn_island_memcfg is not None:
            opts["attn_island_memcfg"] = self.attn_island_memcfg
        return opts

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

    Note: The DPT ViT backbone pads the token sequence length in perf mode to a
    multiple of 64 (not 32) to make attention-mask chunking pick better divisors.
    Program configs that use `per_core_M` must match that padded length, or TTNN
    will hard-fail at runtime during trace capture.
    """
    try:
        TILE = 32
        patch_count = config.image_size // config.patch_size
        seq_len = patch_count * patch_count + 1  # include CLS
        pad_multiple = 64 if getattr(config, "tt_perf_encoder", False) else TILE
        seq_len_padded = math.ceil(seq_len / pad_multiple) * pad_multiple
        seqL_t = seq_len_padded // TILE
        dim_t = config.hidden_size // TILE

        core_grid_x, core_grid_y = core_grid
        num_cores = max(1, int(core_grid_x) * int(core_grid_y))
        dim_t__x = max(1, dim_t // core_grid_x)
        head_num = config.num_attention_heads
        head_size_t = max(1, (config.hidden_size // head_num) // TILE)
        # For attention we height-shard the flattened [B*H*seqL_t, ...] tiles
        # across the full core grid (x*y), not just across x.
        head_seqL_t__x = max(1, (head_num * seqL_t + num_cores - 1) // num_cores)
        # LayerNorm runs on block-sharded tokens across the same grid. When seqL
        # tiles divide cleanly along grid_y, each core sees seqL_t/grid_y tiles.
        ln_block_h = seqL_t
        if core_grid_y > 1 and (seqL_t % core_grid_y) == 0:
            ln_block_h = seqL_t // core_grid_y

        pc = {
            "layernorm_before_program_config": ttnn.LayerNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=core_grid,
                subblock_w=dim_t__x,
                block_h=ln_block_h,
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
                block_h=ln_block_h,
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
        # Use a 2D grid for better attention-score sharding and L1 fit at
        # 384x384 (seq padded to 640 tokens). Attention itself will explicitly
        # keep split-heads height-sharded; QKV split uses an interleaved input
        # tensor to avoid runtime constraints on sharded create_qkv_heads.
        grid = (8, 4)
        # Height-shard attention across 16 cores (one per head) on N300.
        attn_grid = (8, 2)
        math = "hi-fi2"
    elif config.device.endswith("blackhole"):
        grid = (8, 10)
        attn_grid = grid
        math = "hi-fi2"
    else:
        grid = (6, 6)
        attn_grid = grid
        math = "hi-fi3"

    prog_cfgs = _build_perf_program_configs(config, grid)
    attn_prog_cfgs = _build_perf_program_configs(config, attn_grid)
    # MLP dominates ViT-Large runtime. In practice, attention SDPA op constraints
    # vary across runtimes; keep attention operands interleaved for stability,
    # but compute separate MLP program configs for a simple 1-row core grid so
    # we can optionally reshard only for FC1/FC2.
    mlp_grid = None
    mlp_prog_cfgs = {}
    if config.device.endswith("n300"):
        # Keep MLP on the same grid as the encoder for now.
        mlp_grid = (8, 4)
        mlp_prog_cfgs = _build_perf_program_configs(config, mlp_grid)
        # `_build_perf_program_configs` defaults `per_core_M` to the full padded
        # sequence tile count. For block-sharded MLP activations across (x,y),
        # each core sees only seqL_t / grid_y tiles along M.
        try:
            seqL_t = int(mlp_prog_cfgs.get("_seqL_t"))
            dim_t__x = int(mlp_prog_cfgs.get("_dim_t__x"))
            grid_x, grid_y = int(mlp_grid[0]), int(mlp_grid[1])
            if grid_y > 1 and seqL_t % grid_y == 0:
                per_core_M = seqL_t // grid_y
                mlp_prog_cfgs["ff1_matmul_program_config"] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=mlp_grid,
                    in0_block_w=dim_t__x,
                    out_subblock_h=1,
                    out_subblock_w=dim_t__x,
                    per_core_M=per_core_M,
                    per_core_N=dim_t__x * 4,
                    transpose_mcast=False,
                    fused_activation=(ttnn.UnaryOpType.GELU, True),
                )
                mlp_prog_cfgs["ff2_matmul_program_config"] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=mlp_grid,
                    in0_block_w=dim_t__x * 4,
                    out_subblock_h=1,
                    out_subblock_w=dim_t__x,
                    per_core_M=per_core_M,
                    per_core_N=dim_t__x,
                    transpose_mcast=False,
                    fused_activation=None,
                )
        except Exception:
            # Keep the default configs if the runtime doesn't expose these types/attrs.
            pass
    # Disable custom attention configs only when explicitly forced. Attention
    # ops height-shard across x*y cores, so grid_x-only heuristics are incorrect
    # for DPT-Large (seq=640 padded).
    disable_attn_pc = getattr(config, "tt_force_default_attention_programs", False)

    qk_pc = None if disable_attn_pc else attn_prog_cfgs.get("query_by_key_matmul_program_config")
    softmax_pc = None if disable_attn_pc else attn_prog_cfgs.get("softmax_program_config")
    av_pc = None if disable_attn_pc else attn_prog_cfgs.get("attention_probabilities_by_value_matmul_program_config")
    split_mem = getattr(ttnn, "L1_HEIGHT_SHARDED_MEMORY_CONFIG", None)
    if disable_attn_pc:
        split_mem = getattr(ttnn, "DRAM_MEMORY_CONFIG", None)

    qkv_pc = prog_cfgs.get("query_key_value_matmul_program_config")
    proj_pc = prog_cfgs.get("self_output_matmul_program_config")
    if config.device.endswith("n300"):
        # Tokens are block-sharded across a 2D core grid (grid_y > 1). Adjust
        # per-core M tile counts for QKV and projection matmuls so program
        # configs match the shard spec.
        try:
            seqL_t = int(prog_cfgs.get("_seqL_t"))
            dim_t__x = int(prog_cfgs.get("_dim_t__x"))
            grid_x, grid_y = int(grid[0]), int(grid[1])
            if grid_y > 1 and seqL_t % grid_y == 0:
                per_core_M = seqL_t // grid_y
                qkv_pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=grid,
                    in0_block_w=dim_t__x,
                    out_subblock_h=1,
                    out_subblock_w=dim_t__x,
                    per_core_M=per_core_M,
                    per_core_N=3 * dim_t__x,
                    transpose_mcast=False,
                    fused_activation=None,
                )
                proj_pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=grid,
                    in0_block_w=dim_t__x,
                    out_subblock_h=1,
                    out_subblock_w=dim_t__x,
                    per_core_M=per_core_M,
                    per_core_N=dim_t__x,
                    transpose_mcast=False,
                    fused_activation=None,
                )
        except Exception:
            # If any of these program config types/attrs are missing in the runtime,
            # keep the default (may be None).
            pass

    return TTLayerConfig(
        grid=grid,
        attn_grid=attn_grid,
        math_fidelity=math,
        shard_tokens=True,
        shard_heads=True,
        use_fused_ops=True,
        activation_fused=True,
        l1_resident=True,
        use_block_sharded=False,
        sdpa_grid=grid,
        qkv_memcfg=getattr(ttnn, "L1_MEMORY_CONFIG", None),
        proj_memcfg=getattr(ttnn, "L1_MEMORY_CONFIG", None),
        # When MLP resharding is enabled in tt_modules.py, we run FC1/FC2 with
        # block-sharded activations for better matmul utilization (N300).
        mlp_memcfg=(
            getattr(ttnn, "L1_BLOCK_SHARDED_MEMORY_CONFIG", None)
            if mlp_grid is not None
            else getattr(ttnn, "L1_MEMORY_CONFIG", None)
        ),
        split_heads_memcfg=split_mem,
        attn_island_memcfg=getattr(ttnn, "DRAM_MEMORY_CONFIG", None),
        qkv_program_config=qkv_pc,
        qk_program_config=qk_pc,
        softmax_program_config=softmax_pc,
        av_program_config=av_pc,
        proj_program_config=proj_pc,
        ff1_program_config=(mlp_prog_cfgs.get("ff1_matmul_program_config") or prog_cfgs.get("ff1_matmul_program_config")),
        ff2_program_config=(mlp_prog_cfgs.get("ff2_matmul_program_config") or prog_cfgs.get("ff2_matmul_program_config")),
        ln_program_config=prog_cfgs.get("layernorm_before_program_config"),
        ln_compute_config=prog_cfgs.get("ln_compute_config"),
        # Stage-2/3 want explicit sharded attention (QK matmul + fused softmax + AV matmul).
        # Keep a switch so we can force defaults if a runtime regresses.
        use_default_attention_programs=bool(disable_attn_pc),
        mlp_core_grid=mlp_grid,
    )


def describe_configs(config: DPTLargeConfig = DEFAULT_CONFIG) -> Dict[str, TTLayerConfig]:
    return {
        "vit_block": (
            vit_block_config_perf(config) if getattr(config, "tt_perf_encoder", False) else vit_block_config(config)
        ),
        "cnn_block": cnn_block_config(config),
    }
