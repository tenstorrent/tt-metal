# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Llama-3.1-8B prefill attention forward: GQA, full RoPE, causal SDPA, row-parallel o_proj.

**HF anchor:** `transformers.models.llama.modeling_llama.LlamaAttention.forward`.
**Template:** `models/demos/gpt_oss_d_p/tt/attention/prefill.py:51`, with the sliding-window and
attention-sink arguments removed from the SDPA call (both gpt-oss-only) and the fused
matmul+reduce-scatter branch at `:292-300` deleted (Ring-only and gated off on Blackhole).

Pipeline: qkv proj -> GQA head split -> RoPE on **Q and K only** -> optional KV-cache write ->
causal SDPA -> concat heads -> o_proj -> TP collective.

**GQA is native to the SDPA op — there is no on-chip KV repeat.**
`ttnn.transformer.scaled_dot_product_attention` handles the group itself; at TP=8 it is fed 4 local
Q heads and 1 local KV head, and
`ttnn/cpp/ttnn/operations/transformer/sdpa/device/sdpa_device_operation.cpp:98` asserts
`nqh >= nkv && nqh % nkv == 0`, i.e. `4 >= 1 && 4 % 1 == 0`. A torch reference must therefore
`repeat_interleave` the KV heads while the device does not, which is the asymmetry `G-ATTN`'s
reference exists to get right.

**Two loud refusals**, both mirroring the template rather than silently running the wrong core:

* `seq_len <= 1` — this is the prefill path; decode is an explicit non-goal.
* `cached_len > 0` on the dense path — the current chunk would attend a prefix that plain
  `is_causal` SDPA cannot see (its mask assumes Q row 0 aligns with K row 0, so it is off by
  `cached_len` and **silently wrong**). `models/demos/gpt_oss_d_p/tt/attention/prefill.py:257-270`
  raises for exactly this reason; P8's ring path (`dense_sp.py`) is what makes it legal.

**P8 added the attention-core selection**, and it is a three-way choice rather than a switch, so
`select_attention_core` names it in one place and both this function and `G-CHUNK-ATTN` /
`G-MESH-KV` read the name from there (`DEC-075`). Appendix B's last row — "everything passes but
the numbers look too good | you measured the SP bootstrap because `max_seq_len == chunk_size`" — is
a mis-selection between two of these three, so which one ran is **logged on every call** and
asserted by the gates rather than inferred:

| core | when | what runs |
|---|---|---|
| `dense` | `sp == 1` or `sequence_parallel=False`, and `cached_len == 0` | plain causal SDPA on one chip's whole sequence — every P5-P7 gate |
| `sp_bootstrap` | SP > 1, `cached_len == 0` **and** `max_seq_len == chunk_global` | all-gather Q/K/V on the SP axis, dense SDPA, reduce-scatter (`dense_sp.py::sp_bootstrap_attention`) |
| `sp_ring` | SP > 1 and (`cached_len > 0` **or** `max_seq_len > chunk_global`) | ring-joint SDPA reading the prefix out of the block-cyclic cache — **delta 3** |
"""

from loguru import logger

import ttnn

from .config import AttentionConfig, ProgramConfig
from .dense_sp import dense_sp_attention, sp_bootstrap_attention, sp_ring_compute_kernel_config, sp_ring_program_config
from .kv_cache import LlamaKVCache, write_kv_chunk
from .operations import (
    apply_allreduce,
    apply_output_projection,
    apply_qkv_projection,
    apply_rope,
    concat_heads,
    split_qkv_heads_prefill,
)
from .weights import AttentionWeights

# Above this per-user sequence length the activations are carried in bf8_b rather than bf16, as
# `models/demos/gpt_oss_d_p/tt/attention/prefill.py:106-109` does. Named rather than inlined so the
# threshold is visible; no P5 gate reaches it (the longest is 2048).
_BF8_ACTIVATION_SEQ_LEN = 32 * 1024


def run_sdpa(tt_q, tt_k, tt_v, config: AttentionConfig, program_config: ProgramConfig, mesh_device, seq_len):
    """Single-chip GQA causal SDPA. `scale` is passed explicitly (recipe P5.5).

    `sliding_window_size=` and `attention_sink=` are **not** passed: they are gpt-oss-only
    arguments and Llama has neither (`bringup_log/00_MODEL_CARD.md` §3). The kernel's default scale
    already equals `head_dim**-0.5`, but passing `config.scaling` explicitly means the value the
    module reports and the value the kernel uses cannot diverge.
    """
    return ttnn.transformer.scaled_dot_product_attention(
        tt_q,
        tt_k,
        tt_v,
        is_causal=True,
        scale=config.scaling,
        program_config=program_config.get_prefill_sdpa_config(mesh_device, seq_len),
        compute_kernel_config=program_config.get_compute_kernel_config(mesh_device),
    )


def select_attention_core(config: AttentionConfig, mesh_config, kv_cache, *, seq_len, cached_len) -> str:
    """`"dense"` | `"sp_bootstrap"` | `"sp_ring"` — the ONE place the core is chosen (`DEC-075`).

    A function rather than an `if` chain inside `attention_forward`, because the gates have to
    assert *which* core ran and Appendix B's final row is a mis-selection between two of the three.
    `seq_len` is the **per-device** sequence length, so `chunk_global = seq_len * sp`.

    The `max_seq_len > chunk_global` condition is not a heuristic: the ring op enters chunked mode
    only when Q's per-device length is strictly less than K's
    (`ttnn.transformer.ring_joint_scaled_dot_product_attention` docstring), and a request whose one
    chunk fills the whole cache makes them equal. Mirrors
    `models/demos/gpt_oss_d_p/tt/attention/prefill.py:191`.
    """
    if not (config.sequence_parallel and mesh_config.sp > 1):
        return "dense"
    if cached_len > 0 or (kv_cache is not None and kv_cache.max_seq_len > seq_len * mesh_config.sp):
        return "sp_ring"
    return "sp_bootstrap"


def attention_forward(
    hidden_states,
    rope_mats,
    *,
    weights: AttentionWeights,
    kv_cache,
    config: AttentionConfig,
    mesh_config,
    mesh_device,
    program_config: ProgramConfig,
    transformation_mat,
    ccl_manager,
    user_id=0,
    batch_size=1,
    layer_idx=0,
    cached_len=0,
    indexed_rope=False,
):
    """Prefill attention forward. `[1, 1, B*S, 4096]` in, `[1, 1, B*S, 4096]` out.

    Args:
        hidden_states: post-norm input, `[1, 1, B*S, hidden]` bf16 TILE.
        rope_mats: `[cos, sin]`. Per-chunk contiguous tables from
            `tt/rope.py::build_prefill_rope`, or — when `indexed_rope` — the whole-cache
            block-cyclic SP-sharded tables from `build_indexed_rope`.
        weights: `AttentionWeights` (Q/K already Meta-swizzled by the loader).
        kv_cache: `LlamaKVCache` or `None`. `None` is the unit-test / no-cache path.
        config: `AttentionConfig`.
        mesh_config: the model's `MeshConfig` — supplies TP, the axis, and `shard_size`.
        mesh_device: the open mesh.
        program_config: `ProgramConfig` (pinned SDPA grid + the compute-kernel config).
        transformation_mat: the `[1,1,32,32]` Meta RoPE transformation matrix, or `None` to skip
            RoPE entirely (used by `G-ATTN`'s "only Q and K are rotated" invariant check).
        ccl_manager: the model's `CCLManager`; only read when `tp > 1`.
        user_id: KV-cache slot for this user's write.
        batch_size: users packed on the sequence dim.
        layer_idx: this layer's index, for the per-layer cache write.
        cached_len: valid prefix already in the cache before this chunk. Non-zero is refused on
            this dense path — see the module docstring.
        indexed_rope: use the on-device indexed RoPE (P7).
    """
    total_seq_len = hidden_states.shape[-2]
    seq_len = total_seq_len // batch_size
    activation_dtype = ttnn.bfloat8_b if seq_len > _BF8_ACTIVATION_SEQ_LEN else ttnn.bfloat16
    compute_kernel_config = program_config.get_compute_kernel_config(mesh_device)

    if seq_len <= 1:
        raise ValueError(f"Prefill requires seq_len > 1, got {seq_len}. Decode is out of scope for this iteration.")

    core = select_attention_core(config, mesh_config, kv_cache, seq_len=seq_len, cached_len=cached_len)
    if core == "dense" and cached_len > 0:
        # Fail loud rather than run a mask that is off by `cached_len`. See the module docstring.
        raise NotImplementedError(
            f"cached_len={cached_len} needs a chunk-position-aware SDPA: plain is_causal SDPA "
            f"assumes Q row 0 aligns with K row 0, so a cache-backed chunk would be silently "
            f"wrong. The ring-joint path over the block-cyclic cache (tt/attention/dense_sp.py) "
            f"needs sequence_parallel=True on a mesh with sp > 1, and is gated by G-CHUNK-ATTN."
        )
    if core != "dense" and kv_cache is None:
        raise ValueError(
            f"the {core} attention core reads the prefix out of the KV cache, so it needs one; "
            f"kv_cache=None is the unit-test path and only the dense core supports it"
        )

    q, k, v = apply_qkv_projection(hidden_states, weights, compute_kernel_config)

    # `[1, 1, B*S, ...]` -> `[B, 1, S, ...]` so the head split and RoPE see one user's sequence.
    if batch_size > 1:
        q = ttnn.reshape(q, [batch_size, 1, seq_len, -1])
        k = ttnn.reshape(k, [batch_size, 1, seq_len, -1])
        v = ttnn.reshape(v, [batch_size, 1, seq_len, -1])

    num_local_heads = mesh_config.shard_size(config.num_heads)
    num_local_kv_heads = mesh_config.shard_size(config.num_kv_heads)
    tt_q, tt_k, tt_v = split_qkv_heads_prefill(q, k, v, num_local_heads, num_local_kv_heads)
    q.deallocate(True)
    k.deallocate(True)
    v.deallocate(True)

    # RoPE on Q and K. **V is never rotated** — asserted as an invariant by `G-ATTN`.
    if transformation_mat is not None:
        rope_kv_actual = cached_len if indexed_rope else None
        rope_cluster_axis = mesh_config.sp_axis if indexed_rope else None
        if batch_size > 1 and not indexed_rope:
            # The contiguous tables cover one user's positions; the indexed tables carry the whole
            # cache and are never sliced here.
            rope_mats = [rope_mats[0][:, :, :seq_len, :], rope_mats[1][:, :, :seq_len, :]]

        def _rotate(tensor):
            rotated = apply_rope(
                tensor,
                rope_mats,
                transformation_mat,
                kv_actual_global=rope_kv_actual,
                cluster_axis=rope_cluster_axis,
            )
            tensor.deallocate(True)
            return rotated

        tt_q, tt_k = _rotate(tt_q), _rotate(tt_k)

    # Per-layer KV write: **post-RoPE K and raw V** (every template stores it that way; `DEC-021`).
    # `tt_k` / `tt_v` stay live in bf16 for the SDPA below — `write_kv_chunk` casts its own copy.
    if kv_cache is not None:
        assert isinstance(kv_cache, LlamaKVCache), f"kv_cache must be a LlamaKVCache, got {type(kv_cache).__name__}"
        write_kv_chunk(
            kv_cache,
            tt_k,
            tt_v,
            slot_idx=user_id,
            layer_idx=layer_idx,
            kv_actual=cached_len,
            sp_axis=mesh_config.sp_axis,
        )

    # --- the attention core ------------------------------------------------------------------
    logger.debug(
        f"[attention L{layer_idx}] core={core} seq_local={seq_len} sp={mesh_config.sp} "
        f"cached_len={cached_len} cache_global={kv_cache.max_seq_len if kv_cache else None}"
    )
    if core == "dense":
        tt_sdpa_out = run_sdpa(tt_q, tt_k, tt_v, config, program_config, mesh_device, seq_len)
    elif core == "sp_bootstrap":
        tt_sdpa_out = sp_bootstrap_attention(
            tt_q,
            tt_k,
            tt_v,
            mesh_config=mesh_config,
            ccl_manager=ccl_manager,
            run_sdpa=lambda q, k, v, full: run_sdpa(q, k, v, config, program_config, mesh_device, full),
            sp_axis=mesh_config.sp_axis,
        )
    else:
        # Delta 3: this chunk's local Q attends the whole prefix `[0, cached_len + chunk_global)`,
        # read back out of the block-cyclic cache. `write_chunk=False` — the per-layer seam above
        # already wrote this chunk's K/V, so letting the op write again would double-write.
        tt_sdpa_out = dense_sp_attention(
            tt_q,
            kv_cache.k,
            kv_cache.v,
            tt_k,
            tt_v,
            kv_actual=cached_len,
            logical_n=cached_len + seq_len * mesh_config.sp,
            n_kv=config.num_kv_heads,
            cache_global=kv_cache.max_seq_len,
            head_dim=config.head_dim,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            program_config=sp_ring_program_config(
                mesh_device, ccl_core_grid_offset=ccl_manager.ring_attention_ccl_core_grid_offset
            ),
            compute_kernel_config=sp_ring_compute_kernel_config(mesh_device),
            scale=config.scaling,
            cluster_axis=mesh_config.sp_axis,
            slot_idx=user_id,
            layer_idx=layer_idx,
            num_layers=kv_cache.num_layers,
            write_chunk=False,
        )
    tt_q.deallocate(True)
    tt_k.deallocate(True)
    tt_v.deallocate(True)

    tt_concat = concat_heads(tt_sdpa_out)
    tt_sdpa_out.deallocate(True)

    if batch_size > 1:
        tt_concat = ttnn.reshape(tt_concat, [1, 1, total_seq_len, -1])

    tt_out = apply_output_projection(tt_concat, weights, activation_dtype, compute_kernel_config)
    tt_concat.deallocate(True)
    return apply_allreduce(tt_out, mesh_config, ccl_manager)
