# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC

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
    # Optional: override FC1 output memory config to relieve L1 pressure during trace capture.
    ff1_out_memcfg: Optional[ttnn.MemoryConfig] = None
    # Optional: override FC2 output memory config to relieve L1 pressure during trace capture.
    ff2_out_memcfg: Optional[ttnn.MemoryConfig] = None
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
    # Optional override core grid for the MLP path. Useful when tokens are
    # interleaved but we want to reshard only for the MLP matmuls.
    mlp_core_grid: Optional[Tuple[int, int]] = None
    # Optional override core grid for sharded LayerNorm. Useful when the encoder-wide
    # token sharding grid is too small and layer_norm kernels hit static-CB/L1 clashes.
    ln_core_grid: Optional[Tuple[int, int]] = None

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
        batch = max(1, int(getattr(config, "tt_per_chip_batch_size", 1) or 1))
        seqL_total_t = int(seqL_t) * int(batch)

        core_grid_x, core_grid_y = core_grid
        dim_t__x = max(1, dim_t // core_grid_x)
        head_num = config.num_attention_heads
        head_size_t = max(1, (config.hidden_size // head_num) // TILE)
        # vit.md sharded split-heads flattens (B*H*seqL) across the full core grid.
        head_seqL_t__x = max(1, (head_num * seqL_total_t) // max(1, int(core_grid_x) * int(core_grid_y)))
        # Matmul+softmax subblock sizing: for larger seq (e.g., 640 tokens -> 20 tiles),
        # keeping out_subblock_w/subblock_w <= 8 avoids register pressure limits.
        qk_out_subblock_w = seqL_t if seqL_t <= 8 else 1
        softmax_subblock_w = seqL_t if seqL_t <= 8 else 1
        # MLP matmuls can exceed static CB limits on small grids; keep subblocks reasonable.
        # For correctness, some kernels require out_subblock_w == per_core_N when out_subblock_h != 1.
        ff1_out_subblock_w = min((dim_t__x * 4) // 2, 8)
        ff1_fused_activation = (ttnn.UnaryOpType.GELU, True)
        # On N300, fused activation can increase static CB pressure enough to clash with L1.
        # Run GELU as a separate op in perf mode to keep the sharded path stable.
        try:
            if config.device.endswith("n300"):
                ff1_fused_activation = None
        except Exception:
            pass
        # For 2D block sharding, each tensix row sees (B*seqL_t)/core_grid_y tiles in height.
        seqL_t__y = seqL_total_t
        if core_grid_y > 1 and (seqL_total_t % core_grid_y) == 0:
            seqL_t__y = seqL_total_t // core_grid_y
        # LayerNorm runs on block-sharded tokens across the same grid. When seqL
        # tiles divide cleanly along grid_y, each core sees (B*seqL_t)/grid_y tiles.
        ln_block_h = seqL_total_t
        if core_grid_y > 1 and (seqL_total_t % core_grid_y) == 0:
            ln_block_h = seqL_total_t // core_grid_y
        # LayerNorm sharded kernels can clash with L1 buffers on small grids. Keep subblock_w small.
        #
        # Note: do not enable in-place LayerNorm here. The Transformer block uses the pre-LN tensor
        # for residual adds, so in-place LN would either be incorrect or force move-op semantics that
        # require the input tensor to have no other live consumers.
        ln_subblock_w = 1
        ln_inplace = False
        ln_legacy_reduction = True
        ln_legacy_rsqrt = True
        # When batch is sharded across grid_y (vit.md pattern, batch == grid_y),
        # LayerNorm sees a larger per-core height (seqL_t) than the batch=1 bring-up
        # mode where we can split seq across grid_y. On N300 this can trigger static
        # CB vs L1 clashes under trace capture for DPT-Large shapes. Prefer newer
        # reduction kernels when available to reduce static CB pressure.
        try:
            if int(batch) > 1 and int(core_grid_y) == int(batch):
                ln_legacy_reduction = False
                ln_legacy_rsqrt = False
        except Exception:
            pass

        qkv_out_block_w = 3 * dim_t__x
        qkv_out_subblock_w = qkv_out_block_w
        if qkv_out_subblock_w > 8:
            for candidate in (8, 7, 6, 5, 4, 3, 2, 1):
                if (qkv_out_block_w % candidate) == 0 and candidate <= 8:
                    qkv_out_subblock_w = candidate
                    break
        pc = {
            "layernorm_before_program_config": ttnn.LayerNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=core_grid,
                subblock_w=ln_subblock_w,
                block_h=ln_block_h,
                block_w=dim_t__x,
                inplace=ln_inplace,
                legacy_reduction=ln_legacy_reduction,
                legacy_rsqrt=ln_legacy_rsqrt,
            ),
            "query_key_value_matmul_program_config": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=core_grid,
                in0_block_w=dim_t__x,
                out_subblock_h=1,
                # DPT-Large on 8x1 grids can exceed HW register constraints if we emit a full
                # per-core output width subblock (e.g., 3*dim_t__x == 12). Cap the subblock width
                # to keep (out_subblock_w*out_subblock_h) <= 8.
                out_subblock_w=qkv_out_subblock_w,
                per_core_M=seqL_t__y,
                per_core_N=qkv_out_block_w,
                transpose_mcast=False,
                fused_activation=None,
            ),
            "query_by_key_matmul_program_config": ttnn.MatmulMultiCoreReuseProgramConfig(
                compute_with_storage_grid_size=core_grid,
                in0_block_w=head_size_t,
                out_subblock_h=1,
                out_subblock_w=qk_out_subblock_w,
                per_core_M=head_seqL_t__x,
                per_core_N=seqL_t,
            ),
            "softmax_program_config": ttnn.SoftmaxShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=core_grid,
                subblock_w=softmax_subblock_w,
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
                # DPT has a larger padded seq length (e.g., 640 -> 20 tiles). Keep subblocks small
                # so (out_subblock_w*out_subblock_h) stays within the HW register budget.
                out_subblock_h=1,
                out_subblock_w=dim_t__x,
                per_core_M=seqL_t__y,
                per_core_N=dim_t__x,
                transpose_mcast=False,
                fused_activation=None,
            ),
            "layernorm_after_output_program_config": ttnn.LayerNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=core_grid,
                subblock_w=ln_subblock_w,
                block_h=ln_block_h,
                block_w=dim_t__x,
                inplace=ln_inplace,
                legacy_reduction=ln_legacy_reduction,
                legacy_rsqrt=ln_legacy_rsqrt,
            ),
            "ff1_matmul_program_config": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=core_grid,
                in0_block_w=dim_t__x,
                out_subblock_h=1,
                out_subblock_w=ff1_out_subblock_w,
                per_core_M=seqL_t__y,
                per_core_N=dim_t__x * 4,
                transpose_mcast=False,
                fused_activation=ff1_fused_activation,
            ),
            "ff2_matmul_program_config": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=core_grid,
                in0_block_w=dim_t__x * 4,
                # Keep subblocks small to satisfy HW register constraints for larger seq lengths.
                out_subblock_h=1,
                out_subblock_w=dim_t__x,
                per_core_M=seqL_t__y,
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
            "_seqL_total_t": seqL_total_t,
            "_dim_t__x": dim_t__x,
            "_head_seqL_t__x": head_seqL_t__x,
        }
    except Exception:
        pc = {}
    return pc


def vit_block_config_perf(config: DPTLargeConfig = DEFAULT_CONFIG) -> TTLayerConfig:
    # Aggressive encoder settings for Wormhole N300 perf mode
    if config.device.endswith("n300"):
        force_shard_tokens = bool(getattr(config, "tt_force_shard_tokens", False))
        per_chip_batch = int(getattr(config, "tt_per_chip_batch_size", 1) or 1)
        # Stage-2 baseline keeps tokens interleaved for strict trace stability, so the
        # encoder-wide sharding grid is effectively unused. When Stage-3 explicitly opts
        # into sharded tokens, use a taller 2D grid to reduce per-core token height and
        # avoid sharded LayerNorm static-CB/L1 clashes during trace capture on N300.
        #
        # For dp-batched Stage-3 runs, scale the encoder grid height to keep per-core token
        # shards small enough to avoid static CB/L1 pressure and unlock more cores per
        # inference.
        if force_shard_tokens:
            grid = (8, 4) if per_chip_batch >= 4 else (8, 2)
        else:
            grid = (8, 1)
        # Attention score buffers are large for DPT-Large (H=16, seq padded to 640). When
        # per-chip batch > 1, `B*H` increases (e.g., 32) and we want more cores to reduce
        # per-core head/sequence work. However, some TTNN matmul variants require the score
        # width (in tiles) to divide `grid_x` cleanly when producing height-sharded outputs.
        # For dp-batched runs, use a taller attention grid to reduce per-core (B*H*seq)
        # height and static CB pressure. With an explicit QK matmul program_config (set
        # below for per-chip batch > 1), 8x4 avoids width floor-to-multiple issues that
        # can occur with runtime-chosen kernels on full-device grids.
        # For dp-batched runs (per-chip batch > 1), prefer a taller attention grid to
        # reduce per-core score shards (DPT-Large: H=16, seq padded to 640) and static
        # CB pressure under trace. For batch==1, 8x4 can violate matmul per-core
        # constraints depending on runtime kernel selection, so keep 8x2.
        #
        # For per-chip batch >= 4, the attention score/probability tensors can exceed
        # practical L1 budgets on 8x4 grids (static-CB vs L1-buffer clash). Use a
        # taller grid to keep per-core score shards smaller.
        if force_shard_tokens and per_chip_batch >= 4:
            attn_grid = (8, 8)
        else:
            attn_grid = (8, 4) if (force_shard_tokens and per_chip_batch > 1) else (8, 2)
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
    attn_prog_cfgs = prog_cfgs if attn_grid == grid else _build_perf_program_configs(config, attn_grid)
    # Stage 2: keep MLP on the encoder grid (vit.md pattern) to avoid extra reshard
    # traffic and custom grid heuristics in the traced path.
    mlp_grid = None
    mlp_prog_cfgs: dict = {}
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
            seqL_total_t = int(prog_cfgs.get("_seqL_total_t", seqL_t))
            dim_t__x = int(prog_cfgs.get("_dim_t__x"))
            grid_x, grid_y = int(grid[0]), int(grid[1])
            if grid_y > 1 and seqL_total_t % grid_y == 0:
                per_core_M = seqL_total_t // grid_y
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

    qkv_memcfg = getattr(ttnn, "L1_BLOCK_SHARDED_MEMORY_CONFIG", None)
    proj_memcfg = getattr(ttnn, "L1_BLOCK_SHARDED_MEMORY_CONFIG", None)
    qkv_program_config = qkv_pc
    proj_program_config = proj_pc
    shard_tokens = True
    ff1_out_memcfg = None
    ff2_out_memcfg = None
    if config.device.endswith("n300"):
        if not force_shard_tokens:
            # Stage-2 strict baseline: keep the traced path stable by routing some large
            # transformer intermediates through DRAM. This avoids static circular-buffer
            # vs L1-buffer clashes during trace capture for DPT token shapes on N300.
            qkv_memcfg = getattr(ttnn, "DRAM_MEMORY_CONFIG", None)
            proj_memcfg = getattr(ttnn, "DRAM_MEMORY_CONFIG", None)
            qkv_program_config = None
            proj_program_config = None
            # Similarly, route MLP activations through DRAM to avoid static-CB/L1 clashes
            # during trace capture on small grids.
            ff1_out_memcfg = getattr(ttnn, "DRAM_MEMORY_CONFIG", None)
            ff2_out_memcfg = getattr(ttnn, "DRAM_MEMORY_CONFIG", None)
        else:
            # Stage-3: when tokens are explicitly block-sharded across a taller 2D grid
            # (e.g., 8x2), per-core activation footprints shrink enough to keep QKV/proj/MLP
            # activations in L1 and use vit.md-style sharded program configs.
            #
            # NOTE: TTNN sharded split-heads currently requires batch size == grid_y.
            # When per-chip batch matches grid_y (e.g., dp=2, batch_size=4 -> per-chip batch=2),
            # allow the fully sharded split-heads fast-path by keeping QKV block-sharded in L1.
            grid_y = int(grid[1])
            # DPT-Large uses 16 attention heads, but N300's practical core-grid x-dimension is 8.
            # TTNN's sharded split-heads fast-path requires additional head/grid constraints beyond
            # batch==grid_y and can TT_FATAL for (B>1, H=16) on 8x2 grids. Keep QKV interleaved for
            # per-chip batch>1 and reshard Q/K/V explicitly for attention.
            if per_chip_batch > 1 or per_chip_batch != grid_y:
                # Keep QKV output interleaved (DRAM) so split-heads can run without the batch==grid_y constraint,
                # then reshard Q/K/V for explicit attention.
                qkv_memcfg = getattr(ttnn, "DRAM_MEMORY_CONFIG", None)
                qkv_program_config = None
            # Projection + MLP do not share the split-heads constraint; keep them sharded in L1.
            proj_memcfg = getattr(ttnn, "L1_BLOCK_SHARDED_MEMORY_CONFIG", proj_memcfg)
            proj_program_config = proj_pc
            # MLP L1-resident activations can still hit static circular-buffer vs L1 clashes under
            # trace capture on N300; keep the Stage-2 pressure-relief routing for stability.
            ff1_out_memcfg = getattr(ttnn, "DRAM_MEMORY_CONFIG", None)
            ff2_out_memcfg = getattr(ttnn, "DRAM_MEMORY_CONFIG", None)
        # Stage-2 baseline uses interleaved tokens and SDPA, so keep the default
        # attention program configs for robustness. When Stage-3 explicitly opts
        # into sharded tokens, the default attention_softmax_ kernel can over-allocate
        # static CBs under trace; use the sharded softmax program_config from vit.md.
        if not force_shard_tokens:
            qk_pc = None
            softmax_pc = None
            av_pc = None
        else:
            # Keep QK/AV matmuls on the runtime-chosen kernels (more stable on N300),
            # but force a sharded softmax kernel aligned with the configured attention grid.
            # dp-batched Stage-3 (per-chip batch > 1) requires a program_config that matches
            # the height-sharded (B*H*seq) partitioning of the attention scores; leaving QK
            # entirely runtime-chosen can pick a full-grid matmul variant that floors score
            # width (e.g., 640 -> 512) and hard-fails when we request height-sharded outputs.
            qk_pc = None
            av_pc = None
            if (not disable_attn_pc) and softmax_pc is None:
                softmax_pc = attn_prog_cfgs.get("softmax_program_config")
            try:
                if not disable_attn_pc:
                    qk_pc = attn_prog_cfgs.get("query_by_key_matmul_program_config")
            except Exception:
                pass
        # N300: default to interleaved tokens for Stage-2 stability. Stage-3 can opt-in to
        # sharded tokens via `tt_force_shard_tokens`.
        shard_tokens = bool(getattr(config, "tt_force_shard_tokens", False))
    ln_pc = prog_cfgs.get("layernorm_before_program_config")

    return TTLayerConfig(
        grid=grid,
        attn_grid=attn_grid,
        math_fidelity=math,
        shard_tokens=shard_tokens,
        shard_heads=True,
        use_fused_ops=True,
        activation_fused=True,
        l1_resident=True,
        use_block_sharded=False,
        # SDPA (interleaved attention) benefits from a larger grid than the encoder-wide
        # block-sharding constraints (e.g., N300 uses grid_y=1 for sharded split-heads but
        # SDPA can parallelize across heads and sequence tiles).
        sdpa_grid=attn_grid or grid,
        # Match vit.md: QKV/proj/MLP operate on block-sharded activations in L1.
        qkv_memcfg=qkv_memcfg,
        proj_memcfg=proj_memcfg,
        mlp_memcfg=getattr(ttnn, "L1_BLOCK_SHARDED_MEMORY_CONFIG", None),
        ff1_out_memcfg=ff1_out_memcfg,
        ff2_out_memcfg=ff2_out_memcfg,
        split_heads_memcfg=split_mem,
        qkv_program_config=qkv_program_config,
        qk_program_config=qk_pc,
        softmax_program_config=softmax_pc,
        av_program_config=av_pc,
        proj_program_config=proj_program_config,
        ff1_program_config=(mlp_prog_cfgs.get("ff1_matmul_program_config") or prog_cfgs.get("ff1_matmul_program_config")),
        ff2_program_config=(mlp_prog_cfgs.get("ff2_matmul_program_config") or prog_cfgs.get("ff2_matmul_program_config")),
        ln_program_config=ln_pc,
        # Keep compute kernel config optional; follow vit.md defaults unless explicitly needed.
        ln_compute_config=None,
        # Stage-2/3 want explicit sharded attention (QK matmul + fused softmax + AV matmul).
        # Keep a switch so we can force defaults if a runtime regresses.
        use_default_attention_programs=bool(disable_attn_pc),
        mlp_core_grid=mlp_grid,
        ln_core_grid=None,
    )


def describe_configs(config: DPTLargeConfig = DEFAULT_CONFIG) -> Dict[str, TTLayerConfig]:
    return {
        "vit_block": (
            vit_block_config_perf(config) if getattr(config, "tt_perf_encoder", False) else vit_block_config(config)
        ),
        "cnn_block": cnn_block_config(config),
    }
