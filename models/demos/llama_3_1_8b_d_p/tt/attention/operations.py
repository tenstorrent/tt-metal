# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Attention primitives: RoPE application and the collective helpers.

Borrowed from `minimax_m3/tt/attention/operations.py` at the same head_dim and mesh.
"""

import ttnn


def apply_rope(tensor, rope_mats, transformation_mat, is_decode_mode: bool, kv_actual_global=None, cluster_axis=None):
    """Apply rotary position embedding to Q or K.

    Llama rotates the FULL head (rotary_dim == head_dim == 128), so the partial-rotary slice/concat
    branch the donor needs for MiniMax-M3's 64-of-128 rotation is absent here — this is the plain
    full rotation, and `rotary_embedding_llama` already rotates its whole last dim.

    Two inner ops:

    * default (`kv_actual_global` is None): `rotary_embedding_llama`, with cos/sin already sliced to
      this chunk's positions. The one-shot prefill path.
    * indexed (`kv_actual_global` set): `rotary_embedding_indexed` — `rope_mats` carry the
      WHOLE-cache, block-cyclic-reordered, SP-sharded cos/sin (built once), and the op derives this
      chunk's per-chip start row on-device from `kv_actual_global` plus the device's `cluster_axis`
      coordinate, using the same block-cyclic arithmetic as the KV-cache writer. No per-chunk host
      reshard. The chunked path.

    The cos/sin tables must be built in the HF **half-split** convention that `reference/model.py`
    uses, not the Meta interleaved one — both are self-consistent, so a mismatch shows up only as a
    PCC drop.
    """
    rotary_dim = rope_mats[0].shape[-1]
    head_dim = tensor.shape[-1]
    assert rotary_dim == head_dim, (
        f"Llama rotates the full head: rope table width {rotary_dim} must equal head_dim {head_dim}. "
        "A narrower table means the partial-rotary path was carried over from the donor by mistake."
    )
    if kv_actual_global is not None:
        assert cluster_axis is not None, "indexed rope needs the SP cluster_axis the tables are sharded on"
        return ttnn.experimental.deepseek_prefill.rotary_embedding_indexed(
            tensor,
            rope_mats[0],
            rope_mats[1],
            transformation_mat,
            kv_actual_global=kv_actual_global,
            cluster_axis=cluster_axis,
        )
    return ttnn.experimental.rotary_embedding_llama(
        tensor, rope_mats[0], rope_mats[1], transformation_mat, is_decode_mode=is_decode_mode
    )


def assert_sharded_residual_unpadded(mesh_config, hidden_size):
    """A sharded residual needs `hidden/tp` tile-aligned.

    Otherwise the output-dim padding of a column-parallel projection lands INSIDE one TP column's
    residual slice after the reduce-scatter, and it is never masked off. Llama: 4096 / 4 = 1024,
    which is 32 tiles exactly — but the guard stays, because a different TP would break it silently.
    """
    shard = hidden_size // mesh_config.tp
    assert shard % ttnn.TILE_SIZE == 0, (
        f"hidden/tp = {hidden_size}/{mesh_config.tp} = {shard} is not tile-aligned; a sharded "
        "residual would carry projection padding inside a TP column's slice"
    )


def split_qkv_heads(projected, *, num_local_heads, head_dim):
    """`[1, 1, tokens, num_local_heads*head_dim]` -> `[1, num_local_heads, tokens, head_dim]`.

    q/k/v are separate projections in this checkpoint, so each is split on its own — there is no
    fused-QKV column order to get right. The reshape splits the LAST dim head-major, which is how a
    column-parallel projection lays its heads out, and the transpose then moves heads ahead of
    tokens for the attention op.
    """
    tokens = projected.shape[-2]
    width = projected.shape[-1]
    assert width == num_local_heads * head_dim, (
        f"projection width {width} != num_local_heads {num_local_heads} * head_dim {head_dim}; "
        "the per-chip head count is wrong (this model holds 2 KV heads per chip at TP=4, not 1)"
    )
    reshaped = ttnn.reshape(projected, (1, tokens, num_local_heads, head_dim))
    out = ttnn.transpose(reshaped, 1, 2)  # -> [1, num_local_heads, tokens, head_dim]
    ttnn.deallocate(reshaped)
    return out


def apply_reduce_scatter(tensor, mesh_config, ccl_manager, hidden_size):
    """The output projection's closing TP collective.

    `o_proj` is row-parallel, so every TP chip holds a PARTIAL SUM over its contraction shard and a
    collective is required either way. Which one depends on the residual scheme: reduce-scatter
    under a sharded residual (out `emb/tp`, which the caller adds straight into its residual),
    all-reduce under the replicated one (out full emb).
    """
    from ..residual import use_sharded_residual

    if mesh_config.tp <= 1:
        return tensor
    if use_sharded_residual():
        assert_sharded_residual_unpadded(mesh_config, hidden_size)
        scattered = mesh_config.reduce_scatter(tensor, ccl_manager, dim=3, axis=mesh_config.tp_axis)
        tensor.deallocate(True)
        return scattered
    return mesh_config.allreduce(tensor, ccl_manager, axis=mesh_config.tp_axis)
