# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import functools
import os
from typing import NamedTuple

import ttnn

from .weights import AttentionWeights


def apply_qkv_projection(hidden_states, weights: AttentionWeights):
    """
    Apply QKV projection.

    MiniMax-M3 has no biases on q/k/v projections (bias=False in HF), so this is
    a plain matmul.

    Args:
        hidden_states: Input tensor [batch, seq_len, hidden_size]
        weights: Attention weights container

    Returns:
        Fused QKV tensor [batch, seq_len, total_qkv_dim]
    """
    xqkv_fused = ttnn.linear(hidden_states, weights.wqkv, dtype=ttnn.bfloat16)
    return xqkv_fused


def split_qkv_heads_prefill(xqkv_fused, num_heads: int, num_kv_heads: int):
    """
    Split QKV into separate head tensors for prefill mode.

    Args:
        xqkv_fused: Fused QKV tensor
        num_heads: Number of Q heads
        num_kv_heads: Number of K/V heads

    Returns:
        Tuple (Q, K, V) with shapes [1, num_heads, seq_len, head_dim]
    """
    return ttnn.experimental.nlp_create_qkv_heads(
        xqkv_fused,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        transpose_k_heads=False,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def apply_rope(tensor, rope_mats, transformation_mat, is_decode_mode: bool, kv_actual_global=None, cluster_axis=None):
    """
    Apply rotary position embedding (RoPE), supporting PARTIAL rotary.

    MiniMax-M3 rotates only the first ``rotary_dim`` (64) dims of each 128-wide
    head and passes the rest through unchanged. We detect the rotary width from
    the cos matrix (``rope_mats[0]`` is built at rotary_dim width in
    model.create_rope_setup). When rotary_dim == head_dim this is a plain full
    rotation (the rotary_embedding_llama op always rotates its full last dim, so
    we slice the head into rotate / pass-through parts ourselves).

    Two inner ops (the partial-rotary slice/concat wrapper is identical for both):
      * default (``kv_actual_global`` is None): ``rotary_embedding_llama`` with a per-chunk cos/sin
        already sliced to this chunk's positions (decode + the non-cache prefill paths).
      * indexed (``kv_actual_global`` set): ``rotary_embedding_indexed`` — ``rope_mats`` carry the
        WHOLE-cache, block-cyclic-reordered, SP-sharded cos/sin (built once), and the op derives this
        chunk's per-chip start row on-device from ``kv_actual_global`` + the device's ``cluster_axis``
        coordinate (same block-cyclic math as the KV-cache writer). No per-chunk host reshard.

    Args:
        tensor: Input tensor (Q or K), shape [..., head_dim]
        rope_mats: Tuple of (cos, sin) matrices, last dim = rotary_dim
        transformation_mat: Transformation matrix for the mode
        is_decode_mode: Whether in decode mode
        kv_actual_global: prior valid global KV length (tile-aligned). Set -> indexed on-device RoPE.
        cluster_axis: SP mesh axis the whole-cache cos/sin are sharded along (required when indexed).

    Returns:
        Tensor with RoPE applied
    """
    rotary_dim = rope_mats[0].shape[-1]
    head_dim = tensor.shape[-1]

    def _rotate(t):
        if kv_actual_global is not None:
            return ttnn.experimental.deepseek_prefill.rotary_embedding_indexed(
                t,
                rope_mats[0],
                rope_mats[1],
                transformation_mat,
                kv_actual_global=kv_actual_global,
                cluster_axis=cluster_axis,
            )
        return ttnn.experimental.rotary_embedding_llama(
            t, rope_mats[0], rope_mats[1], transformation_mat, is_decode_mode=is_decode_mode
        )

    if rotary_dim >= head_dim:
        return _rotate(tensor)

    # Partial rotary: split [..., :rotary_dim] (rotate) and [..., rotary_dim:] (pass through).
    shape = tensor.shape
    t_rot = ttnn.slice(tensor, [0, 0, 0, 0], [shape[0], shape[1], shape[2], rotary_dim])
    t_pass = ttnn.slice(tensor, [0, 0, 0, rotary_dim], [shape[0], shape[1], shape[2], head_dim])
    t_rot = _rotate(t_rot)
    out = ttnn.concat([t_rot, t_pass], dim=-1)
    t_rot.deallocate(True)
    t_pass.deallocate(True)
    return out


def apply_qk_norm_per_head(tensor, norm_weight, eps: float):
    """Apply MiniMax-M3 per-head QK-norm: RMSNorm over the head_dim (last) dimension with a
    [head_dim] gain, broadcast across all heads.

    ``tensor`` is a head-split Q or K of shape [1, n_heads, S, head_dim] (so the norm runs
    independently per (head, token) vector — this is M3's ``qk_norm_type=per_head``). The gemma
    ``(1 + w)`` fold is baked into ``norm_weight`` at load (weights.py), so this is a plain
    ttnn.rms_norm. head_dim is NOT TP-sharded, so the norm is local to each device — no cross-TP
    reduction. Applied BEFORE RoPE, on Q and K only (matches transformers minimax_m3_vl).
    """
    return ttnn.rms_norm(tensor, weight=norm_weight, epsilon=eps)


def concat_heads(tensor):
    """
    Concatenate attention heads back to hidden dimension (prefill).

    Args:
        tensor: Attention output tensor with separate heads

    Returns:
        Tensor with concatenated heads [batch, seq_len, hidden_size]
    """
    return ttnn.experimental.nlp_concat_heads(tensor, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def apply_output_projection(tensor, weights: AttentionWeights, activation_dtype):
    """
    Apply output projection.

    MiniMax-M3 has no bias on o_proj (bias=False in HF), so this is a plain matmul.

    Args:
        tensor: Attention output tensor
        weights: Attention weights container
        activation_dtype: Target dtype for output

    Returns:
        Output tensor after projection
    """
    tensor = ttnn.typecast(tensor, ttnn.bfloat8_b)
    out = ttnn.matmul(tensor, weights.o_proj, dtype=activation_dtype)
    tensor.deallocate(True)
    return out


class FusedMMRSConfig(NamedTuple):
    """Blocking for the fused attention o_proj matmul + reduce-scatter.

    The matmul runs on ``grid`` = (x, y) cores; the reduce-scatter workers take the rows from y to the
    bottom of the device grid. ``num_workers`` = RS workers per link (None: as many as that RS zone holds,
    zone // (2 * num_links) - 1). ``mm_window_blocks`` = None hands the matmul output to the RS through
    DRAM; N keeps a rolling window of N M-blocks per core in L1 instead (needs >= 2 blocks per core).
    """

    grid: tuple
    m_block: int
    k_block: int
    n_block: int
    subblock_h: int
    subblock_w: int
    chunk_width: int
    num_workers: int | None = None
    mm_window_blocks: int | None = None


# Per-M_tiles tuned configs for the fused attention o_proj matmul + reduce-scatter (Ring only).
# S=1024 (M_tiles=32): min=256us on Tracy (2026-04-23 sweep); see commit for methodology.
# M_tiles 64/80/128: swept 2026-10-12 on the Blackhole galaxy, (2,4) stage, Ring, 2 links, 12x10 core grid
# (models/demos/minimax_m3/tests/perf/bench_mmrs_oproj.py, traced device time, bf8_b out). Times are fused vs the
# unfused bf16 ttnn.matmul + reduce_scatter_minimal_async on the same shape. With the bf16 output the model uses,
# 80 tiles is 431 us; end to end (55k prompt, 5k chunks, eager) every (2,4) stage is 1.6-2.0% faster.
# 20 tiles (chunk 5120 on the whole 8x4 galaxy, SP=8) is deliberately absent: the op is faster there in isolation
# (140 vs 238 us) but the 60-layer prefill got 14% slower (6.18 vs 5.41 s), cause not yet found. One lead: the
# fused op costs ~3.5x the eager host dispatch time of the matmul + reduce-scatter it replaces (1.2 vs 0.34 ms on
# 32 chips), and the smaller per-chip work hides less of it.
_FUSED_MM_RS_CONFIGS = {
    32: FusedMMRSConfig((8, 4), 4, 4, 6, 2, 2, 2, num_workers=3),  # S=1024
    64: FusedMMRSConfig((12, 8), 5, 4, 8, 1, 4, 1, num_workers=5, mm_window_blocks=2),  # 281.5 us vs 591.6 us
    # 80 tiles = production chunk 5120 on a (2,4) stage.
    80: FusedMMRSConfig((12, 8), 5, 4, 8, 1, 4, 1, num_workers=5, mm_window_blocks=2),  # 311.0 us vs 685.7 us
    128: FusedMMRSConfig((12, 7), 5, 4, 16, 1, 4, 1, num_workers=8, mm_window_blocks=2),  # 477.3 us vs 1133.9 us
}


def is_shape_fused_mm_rs_supported(tensor) -> bool:
    # M3_FUSED_MM_RS=0 forces the unfused o_proj + reduce-scatter (A/B against the fused op).
    if os.environ.get("M3_FUSED_MM_RS", "1").strip().lower() in ("0", "false", "no", "off"):
        return False
    m_tiles = (tensor.shape[-2] + 31) // 32
    return m_tiles in _FUSED_MM_RS_CONFIGS


@functools.lru_cache(maxsize=None)
def _fused_mm_rs_kernel_configs(cfg: FusedMMRSConfig, arch):
    """(MinimalMatmulConfig, compute kernel config) for a table entry, built once: the fused op already
    costs far more host time per call than the matmul + reduce-scatter it replaces."""
    mm_config = ttnn.MinimalMatmulConfig(
        M_block_size=cfg.m_block,
        K_block_size=cfg.k_block,
        N_block_size=cfg.n_block,
        subblock_h=cfg.subblock_h,
        subblock_w=cfg.subblock_w,
        compute_with_storage_grid_size=ttnn.CoreCoord(*cfg.grid),
    )
    compute_config = ttnn.init_device_compute_kernel_config(
        arch,
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )
    return mm_config, compute_config


def apply_output_projection_fused_rs(
    tensor, weights: AttentionWeights, mesh_config, ccl_manager, activation_dtype=ttnn.bfloat16
):
    """Attention output projection + TP reduce-scatter fused into one device op.

    Replaces the sequential `apply_output_projection` + first half of
    `mesh_config.allreduce` (the reduce-scatter). The trailing all-gather and
    padding-trim stay as separate ops in `apply_allgather_and_slice`.

    Internally uses `ttnn.experimental.minimal_matmul_strided_reduce_scatter_async`,
    which overlaps MM compute on one half of the core grid with the RS ring
    traffic on the other half — MM signals RS via on-chip semaphore per output
    block so RS starts forwarding as soon as the first block lands in DRAM.

    Dtypes:
      - activation cast to bf8_b before MM (DRAM-bandwidth saver)
      - weight is bf8_b (as loaded, row-parallel sharded on K across TP)
      - MM math: LoFi (same as `ttnn.matmul` default, via compute_kernel_config)
      - output is ``activation_dtype`` (bf16), like the unfused o_proj: under the sharded residual it IS the
        residual stream, and downstream ops (e.g. the MoE router's grouped top-k) reject bf8_b.
        M3_FUSED_MM_RS_OUT=bf8_typecast instead reduce-scatters bf8_b (half the link traffic) and casts after.

    Returns the reduce-scattered tensor of shape
    [1, 1, S, padded_hidden_total / TP] in ``activation_dtype``.

    Only valid for prefill (TP > 1); the decoder path uses a different pattern.
    """
    TILE = 32
    K = tensor.shape[-1]
    N = weights.o_proj.shape[-1]
    assert K % TILE == 0 and N % TILE == 0, f"K={K}, N={N} must be tile-aligned"

    M_tiles = (tensor.shape[-2] + TILE - 1) // TILE
    if M_tiles not in _FUSED_MM_RS_CONFIGS:
        raise ValueError(
            f"No tuned fused o_proj MM+RS config for M_tiles={M_tiles}; "
            f"tuned shapes: {sorted(_FUSED_MM_RS_CONFIGS)}. "
            f"Use apply_output_projection + apply_allreduce for untuned shapes."
        )
    cfg = _FUSED_MM_RS_CONFIGS[M_tiles]
    mm_core_grid = ttnn.CoreCoord(*cfg.grid)

    tensor = ttnn.typecast(tensor, ttnn.bfloat8_b)

    mm_config, compute_config = _fused_mm_rs_kernel_configs(cfg, ccl_manager.mesh_device.arch())

    num_workers = cfg.num_workers
    if num_workers is None:
        device_grid = ccl_manager.compute_grid_size
        num_workers = ((device_grid.y - mm_core_grid.y) * device_grid.x) // (2 * ccl_manager.num_links) - 1

    # Windowed L1 hand-off: the MM output stays in L1 (a rolling window of M blocks per core) and the RS
    # reads it from there instead of a DRAM round trip. Needs both caller-owned counter arrays.
    l1_handoff = cfg.mm_window_blocks is not None
    extra_kwargs = (
        dict(mm_window_blocks=cfg.mm_window_blocks, mm_credit_counters=ccl_manager.get_mm_credit_counters_buffer())
        if l1_handoff
        else {}
    )

    bf8_rs = os.environ.get("M3_FUSED_MM_RS_OUT", "") == "bf8_typecast"
    if not bf8_rs:
        extra_kwargs["dtype"] = activation_dtype

    mm_out, rs_out = ttnn.experimental.minimal_matmul_strided_reduce_scatter_async(
        tensor,
        weights.o_proj,
        3,  # scatter on last (hidden) dim
        ccl_manager.get_rs_ping_pong_semaphore(),
        ttnn.CoreCoord(0, mm_core_grid.y),  # RS cores start just below the MM rows
        compute_kernel_config=compute_config,
        num_links=ccl_manager.num_links,
        memory_config_mm=ttnn.L1_MEMORY_CONFIG if l1_handoff else ttnn.DRAM_MEMORY_CONFIG,
        rs_intermediate_mem_config=ttnn.DRAM_MEMORY_CONFIG,
        rs_output_mem_config=ttnn.DRAM_MEMORY_CONFIG,
        topology=ccl_manager.topology,
        cluster_axis=mesh_config.tp_axis,
        bias=None,  # MiniMax-M3 has no o_proj bias
        config=mm_config,
        barrier_semaphore=ccl_manager.get_barrier_semaphore(),
        chunk_width_in_mm_blocks=cfg.chunk_width,
        num_workers_per_link=num_workers,
        mm_progress_counters=ccl_manager.get_mm_progress_counters_buffer(),
        **extra_kwargs,
    )[:2]

    tensor.deallocate(True)
    mm_out.deallocate(True)
    if bf8_rs and rs_out.dtype != activation_dtype:
        cast = ttnn.typecast(rs_out, activation_dtype)
        rs_out.deallocate(True)
        rs_out = cast
    return rs_out


def apply_allgather_and_slice(rs_out, mesh_config, ccl_manager, hidden_size: int):
    """Complete the attention-output chain after `apply_output_projection_fused_rs`.

    Runs the all-gather (TP) on the scattered output, then drops the padding
    columns that `weights.py` adds for tile alignment of the CCL ops.
    """
    gathered = mesh_config.allgather(rs_out, ccl_manager, axis=mesh_config.tp_axis)
    rs_out.deallocate(True)

    local_hidden = hidden_size // mesh_config.tp
    padded_local_hidden = ((local_hidden + 31) // 32) * 32
    if padded_local_hidden != local_hidden:
        shape = gathered.shape
        sliced = ttnn.slice(
            gathered,
            starts=[0, 0, 0, 0],
            ends=[shape[0], shape[1], shape[2], hidden_size],
            steps=[1, 1, 1, 1],
        )
        gathered.deallocate(True)
        return sliced
    return gathered


def assert_sharded_residual_unpadded(mesh_config, hidden_size: int) -> None:
    """A sharded residual requires ``hidden_size / tp`` to be tile-aligned.

    The o_proj output-dim padding (weights.py, for tile-aligned CCL) appends its columns at the END of
    the padded full width, so after a reduce-scatter they land inside the LAST TP column's slice only —
    the residual would then carry a per-column-inconsistent width that nothing downstream can trim.
    M3 is 6144/4 = 1536 (already tile-aligned) so no padding exists; fail loudly rather than silently
    corrupt a future config that does need it.
    """
    local_hidden = hidden_size // mesh_config.tp
    padded_local_hidden = ((local_hidden + 31) // 32) * 32
    assert padded_local_hidden == local_hidden, (
        f"sharded residual needs a tile-aligned per-device hidden width, but hidden_size={hidden_size} "
        f"/ tp={mesh_config.tp} = {local_hidden} pads to {padded_local_hidden}. The o_proj padding "
        f"columns would end up inside one TP column's residual shard."
    )


def apply_reduce_scatter(tensor, mesh_config, ccl_manager, hidden_size: int):
    """TP reduce-scatter only — attention's closing collective under a SHARDED residual stream.

    The replicated-residual path needs an all-reduce (``apply_allreduce`` = RS + AG) because the
    residual is full emb on every TP column. With an ``emb/tp``-sharded residual the trailing
    all-gather is exactly what the layout removes: o_proj is row-parallel, so the reduce-scatter both
    completes the partial sums AND lands the result in the residual's own layout.

    Returns [1, 1, S, hidden_size / tp].
    """
    if mesh_config.tp <= 1:
        return tensor
    assert_sharded_residual_unpadded(mesh_config, hidden_size)
    scattered = mesh_config.reduce_scatter(tensor, ccl_manager, dim=3, axis=mesh_config.tp_axis)
    tensor.deallocate(True)
    return scattered


def apply_allreduce(tensor, mesh_config, ccl_manager, hidden_size: int):
    """
    Apply tensor parallel allreduce if needed.

    Args:
        tensor: Input tensor
        mesh_config: Mesh configuration
        ccl_manager: Communication manager
        batch_size: Batch size for final reshape
        seq_len: Sequence length for final reshape
        hidden_size: Hidden size for final reshape

    Returns:
        Tensor after allreduce (if TP > 1) or original tensor
    """
    if mesh_config.tp > 1:
        tensor_allreduced = mesh_config.allreduce(tensor, ccl_manager, pad_size=0, axis=mesh_config.tp_axis)
        # ``mesh_config.allreduce`` now frees its input internally between
        # reduce_scatter and all_gather to keep peak DRAM within bounds for
        # long-context prefill; don't deallocate again here.
        tensor = tensor_allreduced

        # Remove padding added in weights.py for tile-aligned CCL operations.
        # If local_hidden was padded (e.g., 360 -> 384), we need to slice back to original hidden_size.
        local_hidden = hidden_size // mesh_config.tp
        padded_local_hidden = ((local_hidden + 31) // 32) * 32
        if padded_local_hidden != local_hidden:
            # Slice from padded_hidden back to hidden_size on the last dimension.
            # Works for both decode [1, 1, batch, padded_hidden] and prefill [1, batch, seq_len, padded_hidden].
            shape = tensor.shape
            tensor_sliced = ttnn.slice(
                tensor,
                starts=[0, 0, 0, 0],
                ends=[shape[0], shape[1], shape[2], hidden_size],
                steps=[1, 1, 1, 1],
            )
            tensor.deallocate(True)
            tensor = tensor_sliced
    return tensor
