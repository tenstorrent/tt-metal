# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The attention block's prefill dataflow. Ported from ``gpt_oss_d_p/tt/attention/prefill.py``.

    hidden -> wqkv -> split heads -> RoPE(q), RoPE(k) -> [write k,v to cache] -> SDPA
           -> concat heads -> o_proj -> all-reduce

The one branch in the function is which SDPA core to call:

* **cache-backed ring** (``dense_sp.dense_sp_attention``) when there is a KV cache and either
  ``cached_len > 0`` (a later chunk) or the cache's capacity exceeds this chunk's global length.
  This is the production path: chunked prefill always takes it from chunk 1 onward, and chunk 0
  takes it too whenever the cache was allocated for the full 262144-token capacity.
* **gathered one-shot** (``dense_sp.gathered_sp_attention``) when the sequence is SP-sharded but
  there is no cache to read, or a cache sized exactly to the sequence. This is what the one-shot
  acceptance mode exercises.
* **plain causal SDPA**, inline below, when ``config.sequence_parallel`` is False. Then the whole
  sequence lives on every mesh row and no SP collective is involved at all. Not a production
  path — it is the diagnostic that separates "the attention math is wrong" from "the SP
  collectives are wrong", which is why the block-level test runs both layouts.

The first two branches see the same Q/K/V, so ``test_attention_chunked_vs_ref.py`` comparing a
two-chunk run against a one-shot run also compares the two cores against each other.

Cache writes go through ``kv_cache.write_kv_chunk`` — the production seam, not a test-only
shortcut — so the tests that PCC the cache contents are checking the code the acceptance run uses.
``dense_sp_attention`` can write the chunk itself, but it is always called here with
``write_chunk=False`` so there is exactly one writer in the package.
"""

import ttnn

from .dense_sp import dense_sp_attention, gathered_sp_attention
from .kv_cache import write_kv_chunk
from .operations import (
    apply_allreduce,
    apply_output_projection,
    apply_qkv_projection,
    apply_rope,
    concat_heads,
    split_qkv_heads_prefill,
)


def attention_forward(
    hidden_states,
    rope_mats,
    *,
    weights,
    kv_cache=None,
    config,
    mesh_config,
    mesh_device,
    program_config,
    transformation_mat,
    ccl_manager,
    user_id: int = 0,
    layer_idx: int = 0,
    cached_len: int = 0,
):
    """One attention block over one prefill chunk.

    Args:
        hidden_states: ``[1, 1, tokens_local, hidden_size]``, post-norm, sequence SP-sharded and
            hidden-replicated over TP.
        rope_mats: ``[cos, sin]`` from ``tt/rope.py``, built for absolute positions
            ``[cached_len, cached_len + tokens_global)`` and sharded to match ``hidden_states``.
        weights: :class:`~.weights.AttentionWeights`.
        kv_cache: a :class:`~.kv_cache.MistralKVCache`, or None for a cacheless one-shot run.
        config: :class:`~.config.AttentionConfig`.
        mesh_config: :class:`~...tt.config.MeshConfig`.
        program_config: :class:`~.config.ProgramConfig`.
        transformation_mat: from ``rope.build_transformation_mat``.
        ccl_manager: :class:`~...tt.ccl.CCLManager`.
        user_id, layer_idx: select the cache slot (``user_id * num_layers + layer_idx``).
        cached_len: tokens already in the cache for this user/layer — the chunk's absolute start.
            Must be a multiple of the chunk size.

    Returns:
        ``[1, 1, tokens_local, hidden_size]``, all-reduced over TP.
    """
    tp = mesh_config.tp
    n_heads_local = config.num_heads // tp
    n_kv_local = config.num_kv_heads // tp
    tokens_local = hidden_states.shape[-2]
    sp = mesh_config.sp if config.sequence_parallel else 1
    tokens_global = tokens_local * sp
    logical_n = cached_len + tokens_global

    # --- projection, heads, rope -------------------------------------------------------------
    xqkv = apply_qkv_projection(hidden_states, weights)
    q, k, v = split_qkv_heads_prefill(xqkv, n_heads_local, n_kv_local)
    xqkv.deallocate(True)

    q_rot = apply_rope(q, rope_mats, transformation_mat)
    q.deallocate(True)
    k_rot = apply_rope(k, rope_mats, transformation_mat)
    k.deallocate(True)

    # --- cache write (always, so P1 can PCC per-layer K/V against the golden trace) ----------
    if kv_cache is not None:
        # update_padded_kv_cache requires the chunk's dtype to match the cache's (bfloat8_b).
        k_bf8 = ttnn.typecast(k_rot, ttnn.bfloat8_b)
        v_bf8 = ttnn.typecast(v, ttnn.bfloat8_b)
        write_kv_chunk(
            kv_cache,
            k_bf8,
            v_bf8,
            slot_idx=user_id,
            layer_idx=layer_idx,
            kv_actual=cached_len,
            sp_axis=mesh_config.sp_axis,
        )
        k_bf8.deallocate(True)
        v_bf8.deallocate(True)

    # --- SDPA ---------------------------------------------------------------------------------
    if not config.sequence_parallel:
        # Diagnostic layout: the whole sequence is on every chip, heads still TP-sharded.
        assert cached_len == 0, "the non-SP diagnostic path has no cache read; it is one-shot only"
        attn = ttnn.transformer.scaled_dot_product_attention(
            q_rot,
            k_rot,
            v,
            is_causal=True,
            scale=config.scaling,
            program_config=program_config.get_prefill_sdpa_config(mesh_device, tokens_global),
            compute_kernel_config=program_config.get_compute_kernel_config(),
        )
    elif kv_cache is not None and (cached_len > 0 or kv_cache.max_seq_len > tokens_global):
        _k_cache, _v_cache, batch_idx, capacity = kv_cache.layer_view(user_id, layer_idx)
        attn = dense_sp_attention(
            q_rot,
            _k_cache,
            _v_cache,
            None,
            None,
            kv_actual=cached_len,
            logical_n=logical_n,
            n_kv=config.num_kv_heads,
            cache_global=capacity,
            head_dim=config.head_dim,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            program_config=program_config.get_ring_sdpa_config(mesh_device),
            compute_kernel_config=program_config.get_ring_compute_kernel_config(mesh_device),
            scale=config.scaling,
            cluster_axis=mesh_config.sp_axis,
            # layer_view already folded the layer into the slot; keep one convention.
            slot_idx=batch_idx,
            layer_idx=0,
            num_layers=1,
            write_chunk=False,
        )
    else:
        attn = gathered_sp_attention(
            q_rot,
            k_rot,
            v,
            mesh_config=mesh_config,
            ccl_manager=ccl_manager,
            program_config=program_config.get_ring_sdpa_config(mesh_device),
            compute_kernel_config=program_config.get_ring_compute_kernel_config(mesh_device),
            mesh_device=mesh_device,
            scale=config.scaling,
            seq_len_global=tokens_global,
        )

    q_rot.deallocate(True)
    k_rot.deallocate(True)
    v.deallocate(True)

    # --- output projection ---------------------------------------------------------------------
    concat = concat_heads(attn)
    attn.deallocate(True)
    out = apply_output_projection(concat, weights)
    concat.deallocate(True)
    return apply_allreduce(out, mesh_config, ccl_manager)
