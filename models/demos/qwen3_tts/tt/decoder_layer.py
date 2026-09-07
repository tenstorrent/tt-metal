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
from models.demos.qwen3_tts.tt.model_config import N150_DRAM_PREFILL_SEQS, PREFILL_SEQS, SHORT_SEQ_LIMIT
from models.demos.qwen3_tts.tt.rmsnorm import RMSNorm


def _build_sharded_rmsnorm_configs(device, dim: int, num_cores: int, m: int = 32):
    """Build (input_memcfg, program_config) for a width-sharded multi-core RMSNorm
    on a [1,1,m,dim] tensor.  m and dim/TILE must both divide cleanly.

    m=32 → decode (single-token, padded to 1 tile)
    m>32 → prefill bucket (same width-shard grid as the consumer matmul)
    """
    TILE = 32
    assert m % TILE == 0, f"m={m} must be a multiple of TILE={TILE}"
    assert (dim // TILE) % num_cores == 0, f"dim_tiles={dim // TILE} must be divisible by num_cores={num_cores}"
    block_w = (dim // num_cores) // TILE
    block_h = m // TILE
    subblock_w = 4
    while subblock_w > 1 and block_w % subblock_w != 0:
        subblock_w -= 1
    compute_grid = device.compute_with_storage_grid_size()
    # Sharded layernorm requires a rectangular core grid.
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
        ttnn.ShardSpec(grid, (m, dim // num_cores), ttnn.ShardOrientation.ROW_MAJOR),
    )
    program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=(compute_grid.x, compute_grid.y),
        subblock_w=subblock_w,
        block_h=block_h,
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
        from models.demos.qwen3_tts.tt.mesh_utils import is_n150

        self._n150 = is_n150(device)

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

        # Width-sharded RMSNorm: decode (m=32) and each prefill bucket.
        # Talker (hidden=2048): 64 cores (1 tile/core). CodePredictor: 32 cores.
        # MLP gate/up consume this layout in place. Prefill M>32 S2I's before 1D QKV in attention;
        # N150 buckets 64/128 keep sharded through LN for the MLP handoff; attention S2I's before 1D QKV.
        dim_tiles = hidden_size // 32
        ln_num_cores = next(c for c in (64, 32, 16, 8, 4, 2, 1) if dim_tiles % c == 0)
        self._decode_ln_in_memcfg, self._decode_ln_progcfg = _build_sharded_rmsnorm_configs(
            device, hidden_size, ln_num_cores, m=32
        )

        # Decode: give each norm the shard grid its CONSUMER matmul wants, so the
        # reshard between them disappears.
        #
        # The widest grid that divides hidden is 64 cores, but the DRAM-sharded decode
        # matmuls pick their grid from find_grid_k_n (it has to divide both K and N
        # tiles), which lands on 8 for QKV, 32 for gate/up and 24 for down. That
        # mismatch cost one reshard per matmul: 3 ops and ~4.8 us per layer, 84 ops and
        # ~134 us across 28 layers.
        #
        # Moving the norm is free because a decode norm is one tile tall, so it is
        # overhead-bound rather than parallelism-bound. Measured on N300, [1,1,32,2048]
        # width-sharded rms_norm: 9.66 us on 64 cores, 8.95 on 32, 9.23 on 16, 9.40 on
        # 8 — and the residual add that feeds it is flat too (5.92 / 6.01 / 5.52 / 6.01).
        # So the norm on the consumer's narrower grid is as fast or faster AND the
        # reshard goes away.
        #
        # NB this changes the norm's reduction grid, so it is NOT bit-exact: the
        # width-wise sum is split across a different number of cores.
        # QWEN3_TTS_LN_CONSUMER_GRID=0 restores the single 64-core norm grid + reshards.
        self._decode_ln_attn = None
        self._decode_ln_mlp = None
        _ln_consumer_grid = os.environ.get("QWEN3_TTS_LN_CONSUMER_GRID", "1") != "0"
        for attr, src, keep in (
            ("_decode_ln_attn", getattr(self.attention, "_decode_wqkv_in0_memcfg", None), None),
            ("_decode_ln_mlp", getattr(self.mlp, "_decode_gate_up_in0_memcfg", None), None),
        ):
            cores = None
            if src is not None:
                try:
                    cores = src.shard_spec.grid.num_cores()
                except Exception:
                    cores = None
            if _ln_consumer_grid and cores and cores != ln_num_cores and dim_tiles % cores == 0:
                try:
                    setattr(self, attr, _build_sharded_rmsnorm_configs(device, hidden_size, cores, m=32))
                except Exception:
                    setattr(self, attr, None)
        self._prefill_ln_configs = {
            m: _build_sharded_rmsnorm_configs(device, hidden_size, ln_num_cores, m=m) for m in PREFILL_SEQS
        }

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
        seq_len_at_entry = x.shape[-2]
        decode_path = mode == "decode"
        prefill_path = mode == "prefill" and seq_len_at_entry in self._prefill_ln_configs
        if prefill_path:
            ln_in_memcfg, ln_progcfg = self._prefill_ln_configs[seq_len_at_entry]
        else:
            # input_layernorm feeds attention's QKV; emit that matmul's shard grid.
            if decode_path and self._decode_ln_attn is not None:
                ln_in_memcfg, ln_progcfg = self._decode_ln_attn
            else:
                ln_in_memcfg = self._decode_ln_in_memcfg
                ln_progcfg = self._decode_ln_progcfg

        # Pre-norm attention
        residual = x
        residual_sharded = None
        _own_residual_sharded = False
        if decode_path:
            # Decode: keep the chain sharded. Skip I2S when the previous layer's
            # residual add already wrote this shard spec.
            if x.memory_config() == ln_in_memcfg:
                residual_sharded = x
                _own_residual_sharded = False
            else:
                residual_sharded = ttnn.to_memory_config(x, ln_in_memcfg)
                _own_residual_sharded = True
            x = self.input_layernorm(
                residual_sharded,
                program_config=ln_progcfg,
                memory_config=ln_in_memcfg,
            )
        elif prefill_path:
            # Seq<=32 stays sharded for DRAM-sharded QKV. N150 buckets 64/128 stay sharded
            # for the MLP LN handoff; attention S2I's to L1 before 1D QKV.
            if x.memory_config() == ln_in_memcfg:
                x_sharded = x
                _own_x_sharded = False
            else:
                x_sharded = ttnn.to_memory_config(x, ln_in_memcfg)
                _own_x_sharded = True
            x = self.input_layernorm(
                x_sharded,
                program_config=ln_progcfg,
                memory_config=ln_in_memcfg,
            )
            if _own_x_sharded:
                ttnn.deallocate(x_sharded)
            _keep_sharded_qkv = seq_len_at_entry <= SHORT_SEQ_LIMIT or (
                self._n150 and seq_len_at_entry in N150_DRAM_PREFILL_SEQS
            )
            if seq_len_at_entry > SHORT_SEQ_LIMIT and not _keep_sharded_qkv:
                x_il = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
                ttnn.deallocate(x)
                x = x_il
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
        if decode_path and x.is_sharded():
            # Sharded residual chain: wo (and later mlp.down) returned width-sharded.
            # `residual_sharded` was prepared earlier (shared with input_layernorm input).
            # post_attention_layernorm feeds the MLP's gate/up, so the residual add
            # writes THAT grid and the norm keeps it — no reshard into gate/up.
            _mlp_memcfg, _mlp_progcfg = self._decode_ln_mlp or (
                self._decode_ln_in_memcfg,
                self._decode_ln_progcfg,
            )
            x = ttnn.add(residual_sharded, x, memory_config=_mlp_memcfg)
            ttnn.deallocate(residual_sharded)
            residual = x  # sharded
            # post_attn layernorm consumes sharded x directly.
            x = self.post_attention_layernorm(
                x,
                program_config=_mlp_progcfg,
                memory_config=_mlp_memcfg,
            )
            x = self.mlp(x, mode=mode)
            # Write the next layer's input-LN shard spec so it skips the I2S/reshard.
            x_out = ttnn.add(residual, x, memory_config=ln_in_memcfg)
            ttnn.deallocate(residual)
            return x_out, updated_kv_cache

        # First residual add: result feeds the post_attention_layernorm + MLP.
        if residual_sharded is not None and _own_residual_sharded:
            ttnn.deallocate(residual_sharded)
        if prefill_path:
            # Write the sum in the LN shard spec so post-attn LN (and the next
            # layer's input LN) skip I2S. Decode already does this; BinaryNg
            # output layout is independent of the addends.
            attn_out = x
            x = ttnn.add(residual, attn_out, memory_config=ln_in_memcfg)
            ttnn.deallocate(attn_out)
            residual = x
            x = self.post_attention_layernorm(
                x,
                program_config=ln_progcfg,
                memory_config=ln_in_memcfg,
            )
            x = self.mlp(x, mode=mode)
            mlp_out = x
            x = ttnn.add(residual, mlp_out, memory_config=ln_in_memcfg)
            ttnn.deallocate(mlp_out)
            ttnn.deallocate(residual)
            return x, updated_kv_cache

        # Non-bucket prefill: force L1 so we don't inherit DRAM from residual.
        x = ttnn.add(residual, x, memory_config=ttnn.L1_MEMORY_CONFIG)

        # Pre-norm MLP
        residual = x  # now in L1
        if decode_path:
            # This norm feeds the MLP's gate/up, whose DRAM-sharded grid differs from
            # the one QKV wants, so shard straight into ITS grid and let the norm keep
            # it — otherwise a reshard sits between the norm and gate/up.
            _mlp_memcfg, _mlp_progcfg = self._decode_ln_mlp or (ln_in_memcfg, ln_progcfg)
            x_sharded = ttnn.to_memory_config(x, _mlp_memcfg)
            x = self.post_attention_layernorm(
                x_sharded,
                program_config=_mlp_progcfg,
                memory_config=_mlp_memcfg,
            )
            ttnn.deallocate(x_sharded)
        else:
            x = self.post_attention_layernorm(x)
        x = self.mlp(x, mode=mode)
        # Second residual add: this is the layer's *output*, returned to the caller.
        # Keep in L1 for all modes — the decode path returns ~4 KB/token (Talker) or
        # ~2 KB/token (CP), and the next layer reads it immediately, so DRAM round-trips
        # waste bandwidth. tt-metal trace guarantees L1 address stability for deterministic
        # op sequences (allocation pattern is identical across steps), so L1 is trace-safe.
        x = ttnn.add(residual, x, memory_config=ttnn.L1_MEMORY_CONFIG)

        return x, updated_kv_cache
