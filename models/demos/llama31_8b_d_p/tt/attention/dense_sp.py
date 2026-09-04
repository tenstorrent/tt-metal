# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Sequence-parallel ring-SDPA attention over the block-cyclic KV cache. **P8.**

**HF anchor:** `transformers.models.llama.modeling_llama.LlamaAttention.forward`'s attention core,
distributed: chunk *k*'s local queries attend the **whole** accumulated prefix `[0, logical_n)`,
which at SP > 1 lives spread across the SP axis and in the cache rather than in the live tensors.
This is **delta 3** (`BRINGUP_RECIPE.md:1650-1654`), the one the dense path refuses.

**Template:** `models/demos/gpt_oss_d_p/tt/attention/dense_sp.py:41` `dense_sp_attention`, itself
ported from `models/demos/minimax_m3/tt/attention/dense_sp.py`. Llama needs it **simpler** than
either: no attention sinks and no sliding window (`bringup_log/00_MODEL_CARD.md` §3), so
`attention_sink=` and `sliding_window_size=` drop out of the
`ring_joint_scaled_dot_product_attention` call and the compact-halo gather-length helper
(`models/demos/gpt_oss_d_p/tt/attention/dense_sp.py:30-38`) collapses to "gather the whole
per-device shard" — `_gather_seq_len` is **not** ported, because with `sliding_window_size=None` its
only branch returns `full_seq` (`:32`).

Four P8 facts, three of them recorded in P5.5's seam and all four now measured:

1. **`fp32_dest_acc_en=False` is mandatory for this op**, structurally rather than by preference:
   `use_streaming_compute = !fp32_dest_acc_en`
   (`ttnn/cpp/ttnn/operations/transformer/sdpa/device/ring_joint_sdpa_program_factory.cpp:1304`) and
   `kv_actual_isl` — which every chunked call passes — requires that streaming path (`:1306`, inside the `TT_FATAL` at `:1305-1308`),
   so `True` is refused with a `TT_FATAL`. This is the single exception to the package default of
   `True` (`DEC-030`), and `G-SP-RING` records both sides of the A/B rather than asserting it away.
2. **This op gets its OWN program config**, never a mutation of
   `attention/config.py::ProgramConfig`: it must carve the CCL column out of the compute grid, so
   its grid is `(grid.x - 1, grid.y)` = (11, 10) on this box, where the dense config's stays a
   pinned 8x8. `ring_joint_sdpa_device_operation.cpp:421` asserts
   `ccl_core_grid_offset.x >= program_config.compute_with_storage_grid_size.x`, and `tt/ccl.py`
   pins that offset at `grid.x - 1`, so `11 >= 11` holds exactly.
3. **`kv_cache_batch_idx` must be `slot_idx * num_layers + layer_idx`**, not `slot_idx`
   (`models/demos/gpt_oss_d_p/tt/attention/dense_sp.py:138-141`). Passing the slot alone makes every
   layer read layer 0's cache: layer 0 is then correct by coincidence and layers 1+ read stale K/V.
   `G-CHUNK-ATTN` carries that as its negative control, because it is the exact shape of bug a
   layer-0-only check cannot see.
4. **The op requires Q strictly shorter than the cache**, which is what makes a *full-length*
   one-shot request unable to use it: chunked-prefill mode "is entered implicitly when
   `input_tensor_q`'s per-device seq length is **less than** `input_tensor_k`'s"
   (`ttnn.transformer.ring_joint_scaled_dot_product_attention` docstring). At
   `max_seq_len == chunk_global` the two are equal, so `sp_bootstrap_attention` below is the exact
   replicated fallback for that one shape (`DEC-072`) — and Appendix B's "everything passes but the
   numbers look too good" is precisely the case of measuring that fallback while believing you
   measured the ring.
"""

import ttnn


def sp_ring_program_config(
    mesh_device, *, ccl_core_grid_offset=None, q_chunk_size=128, k_chunk_size=128
) -> ttnn.SDPAProgramConfig:
    """The ring op's OWN program config: the compute grid **minus the CCL column**.

    Not a mutation of `attention/config.py::ProgramConfig` (which is pinned at 8x8 and must stay
    there — see that file). The CCL workers live in compute column `grid.x - 1` (`tt/ccl.py`), and
    `ring_joint_sdpa_device_operation.cpp:421` requires
    `ccl_core_grid_offset.x >= sdpa_grid.x`, so the widest legal SDPA grid is exactly `grid.x - 1`.

    **Pass `ccl_core_grid_offset`** — `CCLManager.ring_attention_ccl_core_grid_offset` — and the
    assertion below becomes a real check that this file's derivation of the SDPA grid and
    `tt/ccl.py`'s derivation of the CCL offset still agree. Omitting it re-derives the offset here,
    which makes the assertion **tautological**: it recomputes `grid.x - 1` on both sides and can
    never fail, and recipe §1.4's "a control that cannot fail is worse than no control, because it
    is recorded as evidence" applies to an internal assertion just as much (`DEC-093`).

    `q_chunk_size` / `k_chunk_size` are the template's 128
    (`models/demos/gpt_oss_d_p/tt/attention/prefill.py:196-197`) rather than the dense path's
    seq-len-dependent 32/256 pair: the ring op's Q slab is one chunk, not the whole sequence, and
    recipe §2.3 measured the fused kernel's PCC moving under 4% across {32,128,256} anyway
    (`DEC-073`).
    """
    grid = mesh_device.compute_with_storage_grid_size()
    sdpa_grid_x = grid.x - 1
    ccl_offset_x = grid.x - 1 if ccl_core_grid_offset is None else ccl_core_grid_offset[0]
    assert sdpa_grid_x <= ccl_offset_x, (
        f"the ring SDPA grid x={sdpa_grid_x} must not exceed the CCL core offset x={ccl_offset_x} "
        f"(compute grid {grid.x}x{grid.y}); ring_joint_sdpa_device_operation.cpp:421 asserts it. "
        f"tt/ccl.py puts the ring-attention CCL workers in compute column grid.x - 1; if that "
        f"moved, this grid has to move with it."
    )
    return ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(sdpa_grid_x, grid.y),
        q_chunk_size=q_chunk_size,
        k_chunk_size=k_chunk_size,
        exp_approx_mode=False,
    )


def sp_ring_compute_kernel_config(mesh_device, *, fp32_dest_acc_en: bool = False):
    """The ring op's compute-kernel config. **`fp32_dest_acc_en=False`, structurally.**

    The one op in this model where `False` is mandatory rather than a preference (fact 1 in the
    module docstring). The flag is exposed only so `G-SP-RING` can record the `TT_FATAL` that
    `True` produces, instead of asserting the exception away.
    """
    return ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=fp32_dest_acc_en,
        packer_l1_acc=False,
    )


def dense_sp_attention(
    q,
    cache_k,
    cache_v,
    new_k,
    new_v,
    *,
    kv_actual,
    logical_n,
    n_kv,
    cache_global,
    head_dim,
    mesh_device,
    ccl_manager,
    program_config,
    compute_kernel_config,
    scale,
    cluster_axis,
    slot_idx=0,
    layer_idx=0,
    num_layers=1,
    write_chunk=False,
):
    """Cache-backed ring-joint SDPA over the accumulated prefix `[0:logical_n)`. **Delta 3.**

    `q` `[1, n_q_local, chunk_local, head_dim]` — this device's slice of the chunk, TP-sharded on
    heads and SP-sharded on the sequence (the P5.5 seam's docstring said `chunk_global`; the op
    takes the **per-device** length, `[b x nh x N/num_devices x dh]`, and `logical_n` is what
    carries the global figure — corrected here, `DEC-074`);
    `cache_k` / `cache_v` the block-cyclic SP caches (`LlamaKVCache.k` / `.v`, `bfloat8_b`);
    `new_k` / `new_v` this chunk's K/V, written by the op when `write_chunk` — the per-layer seam in
    `attention/prefill.py` writes them instead, so the default is `False`;
    `kv_actual` the valid prefix already cached **before** this chunk, `logical_n` the total prefix
    Q attends causally (`kv_actual + chunk_global`).
    Returns `[1, n_q_local, chunk_local, head_dim]`.

    There is **no on-chip KV repeat** here either: the op handles the GQA group, exactly as the
    dense SDPA does (`tt/attention/prefill.py`).
    """
    assert cache_k.dtype == ttnn.bfloat8_b and cache_v.dtype == ttnn.bfloat8_b, (
        f"the ring cache-read path requires a bfloat8_b KV cache; got k={cache_k.dtype}, "
        f"v={cache_v.dtype}. The persistent ring-gather buffers are bf8_b and the op requires the "
        f"gathered dtype to match the cache (DEC-021 fixes the cache dtype at bfloat8_b)."
    )
    assert logical_n > kv_actual, f"logical_n ({logical_n}) must exceed kv_actual ({kv_actual}); this chunk is empty"

    if write_chunk:
        for cache, tensor in ((cache_k, new_k), (cache_v, new_v)):
            ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
                cache,
                tensor,
                slot_idx=slot_idx,
                layer_idx=layer_idx,
                num_layers=num_layers,
                kv_actual_global=kv_actual,
                cluster_axis=cluster_axis,
            )

    out, _, _ = ttnn.transformer.ring_joint_scaled_dot_product_attention(
        q,
        cache_k,
        cache_v,
        None,  # joint_tensor_q — Llama has no joint sequence
        None,  # joint_tensor_k
        None,  # joint_tensor_v
        # Persistent ring-gather scratch, allocated once per (key, shape, dtype) and reused for
        # every layer and every chunk (`tt/ccl.py::get_ring_gather_buffer`). Llama gathers the
        # WHOLE per-device shard — there is no sliding window, so no compact halo.
        persistent_output_buffer_k=ccl_manager.get_ring_gather_buffer(
            "dense_k", n_kv, cache_global, head_dim, ttnn.bfloat8_b
        ),
        persistent_output_buffer_v=ccl_manager.get_ring_gather_buffer(
            "dense_v", n_kv, cache_global, head_dim, ttnn.bfloat8_b
        ),
        joint_strategy="rear",
        logical_n=logical_n,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
        dim=2,
        multi_device_global_semaphore=ccl_manager.ring_attention_ccl_semaphore_handles,
        num_links=ccl_manager.num_links,
        cluster_axis=cluster_axis,
        mesh_device=mesh_device,
        topology=ccl_manager.topology,
        ccl_core_grid_offset=ccl_manager.ring_attention_ccl_core_grid_offset,
        use_column_major_ccl=True,
        is_causal=True,
        scale=scale,
        is_balanced=False,
        # Fold the layer into the cache batch index, matching `update_padded_kv_cache`'s own write
        # (`batch_idx = slot * num_layers + layer`). Passing the slot alone makes every layer read
        # layer 0's cache: L0 correct by coincidence, L1+ on stale K/V. `G-CHUNK-ATTN`'s control.
        kv_cache_batch_idx=slot_idx * num_layers + layer_idx,
        kv_actual_isl=kv_actual,
    )
    return out


def sp_bootstrap_attention(
    q,
    k,
    v,
    *,
    mesh_config,
    ccl_manager,
    run_sdpa,
    sp_axis,
):
    """The **exact** SP path for a request whose single chunk is the whole cache. Not the ring.

    Needed because the ring op enters chunked mode only when Q's per-device length is *less than*
    K's (module docstring fact 4), which a one-shot `max_seq_len == chunk_global` request does not
    satisfy. All-gather Q, K and V along the SP axis, run the ordinary causal SDPA on the full
    sequence — every SP device then computes the *same* full output — and reduce-scatter it back,
    dividing by `sp` because reduce-scatter **sums** `sp` identical copies.
    Template: `models/demos/gpt_oss_d_p/tt/attention/prefill.py:233-256`.

    `run_sdpa(q, k, v, seq_len)` is the caller's dense SDPA closure, so this function holds no
    program config of its own and the one-shot arm provably runs the same kernel every P5-P7 gate
    scored (`DEC-072`).
    """
    sp = mesh_config.sp
    seq_local = q.shape[-2]
    full_seq_len = seq_local * sp

    q_full = mesh_config.allgather(q, ccl_manager, axis=sp_axis, dim=2)
    k_full = mesh_config.allgather(k, ccl_manager, axis=sp_axis, dim=2)
    v_full = mesh_config.allgather(v, ccl_manager, axis=sp_axis, dim=2)
    out_full = run_sdpa(q_full, k_full, v_full, full_seq_len)
    for t in (q_full, k_full, v_full):
        t.deallocate(True)

    scattered = ttnn.experimental.reduce_scatter_minimal_async(
        out_full,
        dim=2,
        multi_device_global_semaphore=ccl_manager.get_rs_ping_pong_semaphore(),
        num_links=ccl_manager.num_links,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        topology=ccl_manager.topology,
        cluster_axis=sp_axis,
        barrier_semaphore=ccl_manager.get_barrier_semaphore(),
    )
    out_full.deallocate(True)
    # Every SP device computed the identical full output, so the reduce-scatter sum is `sp` times
    # the answer. This is a correction for the collective, not a fudge factor.
    out = ttnn.multiply(scattered, 1.0 / sp)
    scattered.deallocate(True)
    return out
