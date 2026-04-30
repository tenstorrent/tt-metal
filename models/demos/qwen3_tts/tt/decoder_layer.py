# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
Decoder layer implementation for Qwen3-TTS.

Supports both prefill mode (full sequence) and decode mode (single token with KV cache).
"""

import os
from typing import Optional, Tuple

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.qwen3_tts.tt.attention import Attention
from models.demos.qwen3_tts.tt.mlp import MLP
from models.demos.qwen3_tts.tt.rmsnorm import RMSNorm

_DECODE_SHARDED = os.environ.get("QWEN3_TTS_DECODE_SHARDED", "0") == "1"


def _build_sharded_rmsnorm_configs(device, dim: int, num_cores: int):
    """Build (input_memcfg, program_config) for a width-sharded multi-core RMSNorm
    on a [1,1,1,dim] tensor.  num_cores must divide dim/TILE."""
    TILE = 32
    assert (dim // TILE) % num_cores == 0, f"dim_tiles={dim // TILE} must be divisible by num_cores={num_cores}"
    block_w = (dim // num_cores) // TILE
    subblock_w = 4
    while subblock_w > 1 and block_w % subblock_w != 0:
        subblock_w -= 1
    compute_grid = device.compute_with_storage_grid_size()
    # Sharded layernorm requires a rectangular core grid (cpp:173).
    # Pick (cols, rows) such that cols*rows == num_cores, both ≤ compute_grid.
    cols = min(compute_grid.x, num_cores)
    while num_cores % cols != 0:
        cols -= 1
    rows = num_cores // cols
    assert (
        rows <= compute_grid.y
    ), f"Cannot fit {num_cores} cores rectangularly in {compute_grid.x}x{compute_grid.y} grid"
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(cols - 1, rows - 1))})
    in_memcfg = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, (TILE, dim // num_cores), ttnn.ShardOrientation.ROW_MAJOR),
    )
    program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=(compute_grid.x, compute_grid.y),
        subblock_w=subblock_w,
        block_h=1,
        block_w=block_w,
        inplace=False,
    )
    return in_memcfg, program_config


class DecoderLayer(LightweightModule):
    """
    Qwen3-TTS decoder layer.

    Architecture:
        x = x + attention(norm(x))
        x = x + mlp(norm(x))

    This is a simplified implementation for single device (N150/N300).
    """

    def __init__(
        self,
        device,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        intermediate_size: int,
        state_dict: dict,
        layer_idx: int,
        layer_prefix: str,
        rms_norm_eps: float = 1e-6,
        weight_dtype=ttnn.bfloat16,
        weight_cache_path=None,
    ):
        super().__init__()
        self.device = device
        self.layer_idx = layer_idx

        full_prefix = f"{layer_prefix}.layers.{layer_idx}"

        # Input layernorm (pre-attention)
        self.input_layernorm = RMSNorm(
            device=device,
            dim=hidden_size,
            state_dict=state_dict,
            weight_key=f"{full_prefix}.input_layernorm.weight",
            eps=rms_norm_eps,
            weight_dtype=ttnn.bfloat16,
            weight_cache_path=weight_cache_path,
        )

        # Self-attention
        self.attention = Attention(
            device=device,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            state_dict=state_dict,
            layer_prefix=full_prefix,
            rms_norm_eps=rms_norm_eps,
            weight_dtype=weight_dtype,
            weight_cache_path=weight_cache_path,
        )

        # Post-attention layernorm (pre-MLP)
        self.post_attention_layernorm = RMSNorm(
            device=device,
            dim=hidden_size,
            state_dict=state_dict,
            weight_key=f"{full_prefix}.post_attention_layernorm.weight",
            eps=rms_norm_eps,
            weight_dtype=ttnn.bfloat16,
            weight_cache_path=weight_cache_path,
        )

        # MLP
        self.mlp = MLP(
            device=device,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            state_dict=state_dict,
            layer_prefix=full_prefix,
            weight_dtype=weight_dtype,
            weight_cache_path=weight_cache_path,
        )

        # Optional sharded RMSNorm configs for decode-mode (env-gated).
        # Pick the largest num_cores ≤ 64 that divides dim/TILE, so the output shard
        # layout can match the QKV / MLP matmul in0 grid where possible.
        # Talker (hidden=2048): 64 cores (1 tile/core).
        # CodePredictor (hidden=1024): 32 cores.
        self._decode_ln_in_memcfg = None
        self._decode_ln_progcfg = None
        if _DECODE_SHARDED:
            dim_tiles = hidden_size // 32
            ln_num_cores = next(c for c in (64, 32, 16, 8, 4, 2, 1) if dim_tiles % c == 0)
            self._decode_ln_in_memcfg, self._decode_ln_progcfg = _build_sharded_rmsnorm_configs(
                device, hidden_size, ln_num_cores
            )

    def forward(
        self,
        x: ttnn.Tensor,
        cos: ttnn.Tensor,
        sin: ttnn.Tensor,
        transformation_mat: ttnn.Tensor,
        attention_mask: ttnn.Tensor = None,
        kv_cache: Optional[Tuple[ttnn.Tensor, ttnn.Tensor]] = None,
        start_pos: int = 0,
        mode: str = "prefill",
        cur_pos_tensor: Optional[ttnn.Tensor] = None,
        decode_attn_mask: Optional[ttnn.Tensor] = None,
        cp_prefill_mask: Optional[ttnn.Tensor] = None,
        prefill_attn_mask: Optional[ttnn.Tensor] = None,
    ) -> Tuple[ttnn.Tensor, Optional[Tuple[ttnn.Tensor, ttnn.Tensor]]]:
        """
        Apply decoder layer.

        Supports both prefill (full sequence) and decode (single token) modes.

        Args:
            x: Input tensor of shape [batch, 1, seq_len, hidden_size]
            cos: Cosine frequencies for RoPE
            sin: Sine frequencies for RoPE
            transformation_mat: Transformation matrix for RoPE
            attention_mask: Optional attention mask
            kv_cache: Optional tuple of (k_cache, v_cache) for this layer
            start_pos: Starting position in sequence (for KV cache, non-trace path)
            mode: "prefill" for full sequence or "decode" for single token
            cur_pos_tensor: Optional int32 device tensor [1] for trace-compatible decode
            decode_attn_mask: Optional float32 device tensor [1,1,1,max_seq] for decode
            cp_prefill_mask: Optional float32 device tensor [1,1,seq,max_seq] for
                trace-compatible CP prefill (writes cache at constant positions 0,1)
            prefill_attn_mask: Optional float32 device tensor [1,heads,padded_seq,max_seq]
                for trace-compatible Talker prefill (writes full K/V at position 0)

        Returns:
            Tuple of (output, updated_kv_cache) where:
            - output: tensor of shape [batch, 1, seq_len, hidden_size]
            - updated_kv_cache: tuple of (k_cache, v_cache) or None
        """
        # Pre-norm attention
        residual = x
        residual_sharded = None
        if _DECODE_SHARDED and mode == "decode" and self._decode_ln_in_memcfg is not None:
            # Convert x to sharded once and reuse it as residual_sharded later
            # (RMSNorm with inplace=False writes to a fresh output buffer; the input
            # buffer is preserved). Saves one DRAM→L1 reshard per layer.
            residual_sharded = ttnn.to_memory_config(x, self._decode_ln_in_memcfg)
            x = self.input_layernorm(
                residual_sharded,
                program_config=self._decode_ln_progcfg,
                memory_config=self._decode_ln_in_memcfg,
            )
            # do NOT deallocate residual_sharded — used for first residual add below
        else:
            x = self.input_layernorm(x)
        x, updated_kv_cache = self.attention(
            x,
            cos,
            sin,
            transformation_mat,
            attention_mask,
            kv_cache=kv_cache,
            start_pos=start_pos,
            mode=mode,
            cur_pos_tensor=cur_pos_tensor,
            decode_attn_mask=decode_attn_mask,
            cp_prefill_mask=cp_prefill_mask,
            prefill_attn_mask=prefill_attn_mask,
        )
        if _DECODE_SHARDED and mode == "decode" and self._decode_ln_in_memcfg is not None and x.is_sharded():
            # Sharded residual chain: wo (and later mlp.down) returned width-sharded.
            # `residual_sharded` was prepared earlier (shared with input_layernorm input).
            x = ttnn.add(residual_sharded, x, memory_config=self._decode_ln_in_memcfg)
            ttnn.deallocate(residual_sharded)
            residual = x  # sharded
            # post_attn layernorm consumes sharded x directly.
            x = self.post_attention_layernorm(
                x,
                program_config=self._decode_ln_progcfg,
                memory_config=self._decode_ln_in_memcfg,
            )
            x = self.mlp(x, mode=mode)
            # Second residual add: caller wants DRAM output. Both inputs are sharded;
            # ask the add to write to DRAM_INTERLEAVED so the S→I happens for free as
            # part of the binary op output staging.
            x_out = ttnn.add(residual, x, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(residual)
            return x_out, updated_kv_cache

        # First residual add: result feeds the post_attention_layernorm + MLP. Force L1
        # so we don't pay a DRAM write+read for the intermediate. Without this override,
        # ttnn.add inherits DRAM from `residual` (which is the DRAM-backed layer input).
        x = ttnn.add(residual, x, memory_config=ttnn.L1_MEMORY_CONFIG)

        # Pre-norm MLP
        residual = x  # now in L1
        if _DECODE_SHARDED and mode == "decode" and self._decode_ln_in_memcfg is not None:
            x_sharded = ttnn.to_memory_config(x, self._decode_ln_in_memcfg)
            # Same: layernorm output sharded layout matches MLP gate/up matmul in0.
            x = self.post_attention_layernorm(
                x_sharded,
                program_config=self._decode_ln_progcfg,
                memory_config=self._decode_ln_in_memcfg,
            )
            ttnn.deallocate(x_sharded)
        else:
            x = self.post_attention_layernorm(x)
        x = self.mlp(x, mode=mode)
        # Second residual add: this is the layer's *output*, returned to the caller.
        # Caller (Talker.forward) expects the per-layer output in DRAM_INTERLEAVED so
        # the next layer's residual chain has stable addresses across trace iterations.
        x = ttnn.add(residual, x, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        return x, updated_kv_cache
