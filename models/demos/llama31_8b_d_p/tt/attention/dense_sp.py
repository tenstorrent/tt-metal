# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Sequence-parallel ring-SDPA attention over the block-cyclic KV cache. **Filled in P8.**

**Template:** `models/demos/gpt_oss_d_p/tt/attention/dense_sp.py:41` `dense_sp_attention`, itself
ported from `models/demos/minimax_m3/tt/attention/dense_sp.py`. Llama needs it **simpler** than
either: no attention sinks and no sliding window, so `attention_sink=` and `sliding_window_size=`
drop out of the `ring_joint_scaled_dot_product_attention` call and the compact-halo gather-length
helper (`models/demos/gpt_oss_d_p/tt/attention/dense_sp.py:30-38`) collapses to "gather the whole
per-device shard".

This file exists at P5.5 as a **named seam with a signature**, not as an empty placeholder: the
interface below is pinned in `bringup_log/03_OUTLINE.md` §2.7 so `attention/prefill.py`'s dispatch
and `tt/ccl.py`'s ring-attention semaphores and core-grid offset already line up with it.

Three P8 facts recorded here because they are decided and would otherwise be rediscovered:

1. **`fp32_dest_acc_en=False` is mandatory for this op**, structurally rather than by preference:
   `use_streaming_compute = !fp32_dest_acc_en`
   (`ttnn/cpp/ttnn/operations/transformer/sdpa/device/ring_joint_sdpa_program_factory.cpp:1304`) and
   the chunked path needs the streaming compute (`:1306`), so `True` is refused with a `TT_FATAL`.
   It is the single exception to the package default (`DEC-030`), and `G-SP-RING` records both
   sides of the A/B rather than asserting the exception away.
2. **This op gets its OWN program config**, never a mutation of
   `attention/config.py::ProgramConfig`: it must carve the CCL column out of the compute grid
   (`ttnn.CoreCoord(grid.x - 1, grid.y)`), whereas the dense config's grid stays a pinned 8x8.
3. **`kv_cache_batch_idx` must be `slot_idx * num_layers + layer_idx`**, not `slot_idx`
   (`models/demos/gpt_oss_d_p/tt/attention/dense_sp.py:138-141`). Passing the slot alone makes every
   layer read layer 0's cache: layer 0 is then correct by coincidence and layers 1+ read stale K/V.
"""


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
    """Cache-backed ring-joint SDPA over the accumulated prefix `[0:logical_n]`. **P8.**

    `q` `[1, n_q_local, chunk_global, head_dim]` block-cyclic over the chunk, SP x TP sharded;
    `cache_k` / `cache_v` the block-cyclic SP caches (`LlamaKVCache.k` / `.v`, bf8_b);
    `new_k` / `new_v` this chunk's K/V, written by the op when `write_chunk` (the per-layer seam in
    `attention/prefill.py` writes it instead, so the default is `False`);
    `kv_actual` the valid prefix already cached, `logical_n` the total prefix Q attends causally.
    Returns `[1, n_q_local, chunk_local, head_dim]`.
    """
    raise NotImplementedError(
        "dense_sp_attention is P8's: it needs the (4,8) mesh, the ring fabric, and the "
        "ring-attention semaphores from CCLManager. Gates G-SP-RING and G-CHUNK-ATTN own it. "
        "Template: models/demos/gpt_oss_d_p/tt/attention/dense_sp.py:41."
    )
