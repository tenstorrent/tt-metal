# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""SP ring-joint SDPA over the block-cyclic SP KV cache — the P8 port (was a P5 stub).

``ttnn.transformer.ring_joint_scaled_dot_product_attention`` reads the accumulated prefix across the
SP axis *inside* the op: the ring all-gathers each chip's KV shard into a persistent scratch buffer
while an online softmax folds the partial results, so there is **no explicit AllGather** in this file
and the current chunk's queries attend the whole prefix ``[0, logical_n)``.

Ported from ``models/demos/gpt_oss_d_p/tt/attention/dense_sp.py:41`` (op call ``:106``). Three
gpt-oss-only things are dropped, because Llama has neither sinks nor a sliding window
(``00_MODEL_CARD.md`` §3):

* ``attention_sink=`` (template ``:144``) and ``sliding_window_size=`` (``:145``);
* ``_gather_seq_len`` (``:30``) — for full-causal attention it collapses to ``return full_seq``
  (``:36``), so the persistent buffer key is simply ``cache_global``;
* with them goes the whole "compact halo" branch of the device op
  (``ring_joint_sdpa_device_operation.cpp:501`` only runs under ``has_sliding_window()``), which
  is why Llama may use a q/k chunk pair the sliding path would reject.

**Three inherited constraints, none of them re-litigated here:**

* the cache must be ``bfloat8_b`` (``DEC-017``), asserted below as the template does at ``:77-81``;
* ``fp32_dest_acc_en`` **must** be ``False``, so this path builds its **own** compute-kernel config
  via :meth:`ProgramConfig.get_ring_compute_kernel_config` and never reuses
  ``get_compute_kernel_config`` (whose default is ``True``, ``DEC-031``). Measured, the requirement
  is sharper than the template's comment: ``use_streaming_compute = !fp32_dest_acc_en``
  (``ring_joint_sdpa_program_factory.cpp:1304``) and ``kv_actual_isl`` — which every chunk of this
  package's prefill passes — *requires* the streaming path (``:1306``), so for chunked prefill the
  two flags are mutually exclusive **by construction** and ``True`` is refused with a ``TT_FATAL``.
  ``DEC-084`` records what it costs (ring op alone **7.98x** off its noise floor; **1.45x** the
  end-to-end KV error of the fp32-accumulating bootstrap);
* the SDPA program grid stays **8x8** (``DEC-012`` / Appendix F.8):
  ``ring_joint_sdpa_device_operation.cpp:421`` asserts ``ccl_core_grid_offset.x >= sdpa_grid.x`` with
  the offset pinned at ``grid.x - 1 = 11`` on this Blackhole, so a device-derived grid would give
  ``11 >= 12`` and fail. ``ProgramConfig.assert_sdpa_grid_fits`` enforces it at construction.

**The shape rule that selects this path** (``DEC-021``, and the reason
``tt/attention/prefill.py`` keeps a bootstrap): the op's chunked-prefill mode needs
``N_local_q < N_local_kv`` — Q shorter than the per-chip cache shard
(``ring_joint_sdpa_device_operation.cpp:580``), and passing ``kv_actual_isl`` *requires* that
mode (``:617``). So a one-shot request whose cache is exactly one chunk long
(``max_seq_len == chunk``) cannot use this path at all, including for chunk 0. The caller decides;
:func:`dense_sp_attention` asserts it rather than letting a `TT_FATAL` explain it.
"""

from __future__ import annotations

import ttnn


def dense_sp_attention(
    tt_q,
    cache_k,
    cache_v,
    tt_k_chunk,
    tt_v_chunk,
    *,
    kv_actual,
    logical_n,
    n_kv,
    cache_global,
    head_dim,
    mesh_device,
    ccl_manager,
    program_config,
    scale,
    cluster_axis,
    compute_kernel_config=None,
    slot_idx=0,
    layer_idx=0,
    num_layers=1,
    write_chunk=True,
):
    """Cache-read ring-joint SDPA over the accumulated prefix ``[0, logical_n)``.

    Args:
        tt_q: ``[1, n_q_local, chunk_local, head_dim]`` bf16 — this chunk's queries, SP-sharded on
            the ``cluster_axis`` rows and TP-sharded on the head dim.
        cache_k, cache_v: the block-cyclic SP KV caches (``LlamaKVCache.k`` / ``.v``), ``bfloat8_b``.
        tt_k_chunk, tt_v_chunk: this chunk's K/V. **Ignored when ``write_chunk=False``** — the
            per-layer seam in ``tt/attention/prefill.py`` has already written them via
            ``write_kv_chunk``, and writing twice at the same offset is wasted DRAM traffic.
        kv_actual: valid prefix length already in the cache **before** this chunk. Drives the
            on-device KV-pad rotation (``kv_actual_isl``), so it must be tile-aligned.
        logical_n: total valid prefix after this chunk, i.e. ``kv_actual + chunk_global``. Q attends
            causally over ``[0, logical_n)``.
        n_kv: **global** KV head count (8). The persistent buffer is sharded ``dims=[None, 1]``, so
            each chip gets ``n_kv / tp`` = 1 head — matching the per-chip cache slot.
        cache_global: the cache's global sequence capacity (``LlamaKVCache.max_seq_len``); the
            persistent ring-gather buffer's seq extent.
        head_dim: 128.
        mesh_device: the ttnn mesh device.
        ccl_manager: ``CCLManager`` — supplies the ring semaphores, the ring CCL core-grid offset,
            ``num_links``, the topology and the persistent gather buffers.
        program_config: the package :class:`~.config.ProgramConfig`; this function derives the ring
            SDPA program config and the ring compute-kernel config from it.
        scale: the QK scale (``config.scaling``), passed explicitly.
        cluster_axis: the mesh axis the sequence is sharded over (``mesh_config.sp_axis``).
        compute_kernel_config: override for the ring compute-kernel config. ``None`` builds it from
            ``program_config`` with ``fp32_dest_acc_en=False``, which is what the op requires.
        slot_idx, layer_idx, num_layers: locate this (user, layer) slot in the packed cache.
        write_chunk: write ``tt_k_chunk`` / ``tt_v_chunk`` into the cache before reading it back.

    Returns:
        ``[1, n_q_local, chunk_local, head_dim]`` bf16 — this chunk's attention output, still
        SP-sharded on the ``cluster_axis`` rows.
    """
    assert cache_k.dtype == ttnn.bfloat8_b and cache_v.dtype == ttnn.bfloat8_b, (
        f"the SP ring cache-read requires a bfloat8_b KV cache; got k={cache_k.dtype}, v={cache_v.dtype}. "
        f"KV_CACHE_DTYPE=bf16 cannot ship (DEC-017): the ring path's gather buffers and the op's "
        f"dtype checks are bf8_b."
    )
    q_local = tt_q.shape[-2]
    kv_local = cache_k.shape[-2]
    # The op's own preconditions, hoisted so they read as sentences instead of TT_FATALs. All three
    # are value checks the caller controls (ring_joint_sdpa_device_operation.cpp:580, :617, :633).
    assert q_local < kv_local, (
        f"the ring cache-read path needs Q shorter than the per-chip KV shard (chunked-prefill "
        f"shape): got q_local={q_local}, kv_local={kv_local}. Equal lengths are the one-shot case — "
        f"size the cache above one chunk, or take the bootstrap in tt/attention/prefill.py (DEC-021)."
    )
    assert kv_local % q_local == 0, (
        f"KV-pad rotation needs the per-chip cache shard to be a whole number of Q slabs: "
        f"kv_local={kv_local}, q_local={q_local} (max_seq_len must be a multiple of chunk_size)"
    )
    assert kv_actual % ttnn.TILE_SIZE == 0 and logical_n % ttnn.TILE_SIZE == 0, (
        f"KV-pad rotation requires tile-aligned lengths: kv_actual={kv_actual}, logical_n={logical_n}, "
        f"TILE_SIZE={ttnn.TILE_SIZE}"
    )
    assert kv_actual < logical_n <= cache_global, (
        f"[kv_actual={kv_actual}, logical_n={logical_n}] must be a non-empty prefix inside the cache "
        f"capacity {cache_global}"
    )

    if write_chunk:
        # The package's own caller always passes write_chunk=False — `tt/attention/prefill.py` writes
        # through the per-layer `write_kv_chunk` seam before it gets here, so that the single-card and
        # SP paths share one write. This branch exists for a caller that has not written yet (the
        # template's default), and it must repeat `kv_cache._write_one`'s **dtype cast**: the op
        # requires `input.dtype == cache.dtype`, and the chunk arriving here is bf16 while the cache
        # is bf8_b. Dropping the cast would make this branch a TT_FATAL the moment anyone used it.
        for cache, chunk in ((cache_k, tt_k_chunk), (cache_v, tt_v_chunk)):
            assert chunk is not None, "write_chunk=True needs tt_k_chunk / tt_v_chunk"
            src = chunk if chunk.dtype == cache.dtype else ttnn.typecast(chunk, cache.dtype)
            ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
                cache,
                src,
                slot_idx=slot_idx,
                layer_idx=layer_idx,
                num_layers=num_layers,
                kv_actual_global=kv_actual,
                cluster_axis=cluster_axis,
            )
            if src is not chunk:
                src.deallocate(True)

    if compute_kernel_config is None:
        compute_kernel_config = program_config.get_ring_compute_kernel_config(mesh_device)

    out, _joint, _stats = ttnn.transformer.ring_joint_scaled_dot_product_attention(
        tt_q,
        cache_k,
        cache_v,
        None,
        None,
        None,
        # Persistent ring-gather scratch, allocated once per (key, shape, dtype) and reused across
        # every layer and chunk. Llama gathers the whole per-device shard (no sliding halo), so the
        # extent is `cache_global`. The dtype MUST match the bf8_b cache: the all-gather validator
        # compares page sizes between the input and the persistent buffer
        # (ring_joint_sdpa_device_operation.cpp:176).
        persistent_output_buffer_k=ccl_manager.get_ring_gather_buffer(
            "dense_k", n_kv, cache_global, head_dim, ttnn.bfloat8_b
        ),
        persistent_output_buffer_v=ccl_manager.get_ring_gather_buffer(
            "dense_v", n_kv, cache_global, head_dim, ttnn.bfloat8_b
        ),
        joint_strategy="rear",
        logical_n=logical_n,
        program_config=program_config.get_ring_sdpa_config(mesh_device),
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
        # False = the plain causal work split. `is_balanced` (zigzag) is causal-only AND incompatible
        # with kv_actual_isl (ring_joint_sdpa_device_operation.cpp:627), which every chunk passes.
        is_balanced=False,
        # Fold the layer into the cache batch index, matching update_padded_kv_cache's write
        # (batch_idx = slot*num_layers + layer). Passing slot alone makes every layer read layer 0's
        # cache: layer 0 right by coincidence, layers 1+ reading stale KV.
        kv_cache_batch_idx=slot_idx * num_layers + layer_idx,
        kv_actual_isl=kv_actual,
    )
    return out
