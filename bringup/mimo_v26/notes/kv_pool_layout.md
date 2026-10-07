# KV pool layout (decode agent proposal, 2026-10-07) - for the prefill agent

Code: `models/demos/blackhole/mimo_v26/tt/kv_pool.py` (`KVPoolLayout`, host only). Same contract as the attention probe/module that already exists:

- Per layer, `MiMoAttention.k_cache/v_cache` = bf16 tile `[n_blocks,1,64,192]` replicated on all devices (device (r,c) writes kv head c (SWA) or c//2 (GA)); V zero padded to 192, V*0.707 folded into the weights.
- Slots: 32 per mesh row, same layout on every row, rows hold different users. GA: block `slot*nlog + i`. SWA ring: block `slot*ring_blocks + (i % ring_blocks)`, decode `cache_position_modulo = ring_blocks*64`, prefill chunked SDPA/`paged_fill_cache` use the logical row (aliasing). `ring_blocks >= chunk/64 + 2` (default chunk/64 + 8).
- Page tables int32 row-major, padded to a multiple of 8 entries.
- Prefill writes via `MiMoAttention.prefill_set(io, chunk_start, slot)` + `prefill_chunk`; decode reads/writes positions per row and per slot (`decode_pos_io/decode_pos_set` in attention.py, one shared tensor pair for all layers).
- A user's decode start position = its real prompt length (padding written by a padded prefill sits at positions >= len and is masked by causality / overwritten).
- Dead/finished slots: position 0, any token, MoE routing spread (tt/decode.py).

If prefill needs something different (more slots, non-contiguous allocation, a shared global pool), reply in LOG (track `asm`) or edit this note; decode only depends on `KVPoolLayout.page_table/logical_row`.
