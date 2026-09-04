# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The KV pad-zero and the migration ack that rides on it.

Lifted verbatim out of `TtPrefillBlock.forward` so a model whose blocks are not `TtPrefillBlock` can
still hand off to migration correctly. Kimi-K3 is that model: only 24 of its 93 layers write a KV
slab, so its block is its own class, and the choice was between duplicating this or moving it. It is
subtle enough — two ack transports with different flush requirements, a layout guard, a metadata
form for the traced path and a scalar form for the eager one — that a second copy would drift.

Everything it used to read off `self` is a plain scalar (`layer_num`, `sp_factor`, `sp_axis`, the
global layer index) plus the mesh device and the trace controller, so the move needs no state.
`TtPrefillBlock` keeps its behaviour exactly: it now calls this with what it used to read inline.
"""

from __future__ import annotations

from typing import Optional

import ttnn


def zero_pad_and_ack(
    *,
    kvpe_cache,
    mesh_device,
    cache_layer_idx: int,
    cache_user_id: int,
    layer_num: int,
    sp_factor: int,
    sp_axis: int,
    global_layer_idx: int,
    seq_len_local: int,
    actual_end: Optional[int],
    metadata: Optional[tuple],
    d2h_service,
    metadata_msg,
    on_layer_complete,
    trace_controller,
) -> None:
    """Zero this chunk's pad window in the KV cache and tell migration the layer is done.

    A no-op unless one of the two ack transports is wired, which is what makes it safe to call
    unconditionally from a block that may or may not be running under a migration engine.

    Only call this on a layer that actually wrote KV. A Kimi-K3 KDA layer writes none, so its block
    skips it — there is no pad window to clean and no slot to ack.
    """
    # Chunked-prefill migration handoff. MLA's update_padded_kv_cache wrote this chunk as full
    # 32-row tiles, leaving stale data between the last real token (actual_end) and the next
    # 128-boundary; zero that pad window so the decode side reads clean zeros.
    #
    # Two ack transports share that zero, and exactly one is wired per run:
    #   d2h_service (single host)   — the ack is a DEVICE op enqueued on the same CQ right after the
    #       zero, so the record only reaches the host after the zero has executed; the ack (driven by
    #       record arrival) implies zero-complete with NO host sync.
    #   on_layer_complete (pipeline) — a HOST callback, so the zero must be flushed first: the
    #       migration worker reads the cache over NoC out-of-band from the ttnn command queue and
    #       without the flush could copy pre-zero data. layer_idx is GLOBAL (the scheduler orders acks
    #       across pipeline ranks).
    # cache_layer_idx is the LOCAL per-rank cache slot in both.
    if d2h_service is not None or on_layer_complete is not None:
        assert actual_end is not None or metadata is not None, "actual_end or metadata required for zero_pad"
        assert d2h_service is None or metadata_msg is not None, "metadata_msg required when d2h_service is set"
        # zero_padded_kv_cache is a DENSE (TILE) kvpe-cache op. A DSA-sparse model's kvpe cache is
        # bf16/fp8 ROW_MAJOR (sparse_sdpa reads it natively) and the op asserts TILE, so skip it for
        # sparse.
        cache_tensor = kvpe_cache.storage
        if cache_tensor.layout == ttnn.TILE_LAYOUT:
            if metadata is not None:
                # Per-element-tensor (trace-safe) path: slot_idx (metadata[0]) + valid_global=actual_end
                # (metadata[2]), each its own 1-element uint32 tensor read on-device.
                ttnn.experimental.deepseek_prefill.zero_padded_kv_cache(
                    cache_tensor,
                    metadata[0],  # slot_idx tensor
                    metadata[2],  # valid_global (= actual_end) tensor
                    cache_layer_idx,
                    layer_num,
                    seq_len_local * sp_factor,
                    sp_axis,
                )
            else:
                ttnn.experimental.deepseek_prefill.zero_padded_kv_cache(
                    cache_tensor,
                    cache_user_id,
                    cache_layer_idx,
                    layer_num,
                    actual_end,
                    seq_len_local * sp_factor,
                    sp_axis,
                )
        if d2h_service is not None:
            # Device-op ack, enqueued on the same CQ right after the zero: the record cannot reach the
            # host before the zero has executed, so the ack implies zero-complete with no host sync —
            # unlike the host-callback path below, which needs an explicit flush.
            # Capture-safe only because metadata_msg is at a fixed address: the op registers it as a
            # buffer binding, which trace replay does not re-patch. A traced caller must therefore
            # pass a persistent record (TtPrefillRuntime._trace_metadata_msg), never the per-chunk
            # socket tensor, whose address moves every chunk.
            ttnn.experimental.deepseek_prefill.outbound_socket_service_sync(d2h_service, metadata=metadata_msg)
        else:
            # Trace path: route the ack through the controller. At capture it splits the trace here (a host
            # shm bump cannot live inside a trace); at replay the controller fires the ack between the two
            # segments, after the first segment's writes flush (execute_trace blocking). Non-trace:
            # synchronize then call directly. The controller takes precedence iff it carries an ack callback
            # (runner trace path); the test path sets a controller WITHOUT an ack callback, so has_layer_ack()
            # is False (and on_layer_complete is None there, so neither fires).
            tc = trace_controller
            if tc is not None and tc.has_layer_ack():
                tc.layer_ack(global_layer_idx)
            else:
                ttnn.synchronize_device(mesh_device)
                on_layer_complete(global_layer_idx)
