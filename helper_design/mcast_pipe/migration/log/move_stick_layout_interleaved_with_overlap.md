# move_stick_layout_interleaved_with_overlap.cpp — DEFERRED (design-gap)

Tier 3. Status: deferred. No code change.

## Role
RM (stick / row-major) twin of `move_interleaved_with_overlap.cpp`. Identical handshake
block; only the data path differs (per-page `CoreLocalMem` reads/writes with
`aligned_page_size` stride instead of tiled CB offsets).

## Why deferred
Structurally identical handshake to the TILE variant — same four known design gaps:
1. **Runtime role selection** — `is_controller = get_arg_val<uint32_t>(8)`.
2. **Multi-rect (2-3) with runtime per-rect counts** — `range_{0,1,2}_size`,
   `do_third_multicast`.
3. **Dual-use control word** — `control_value` as inbound counter + outbound flag on one
   cell.
4. **Runtime sem id** — `semaphore_arg = get_arg_val<uint32_t>(4)`.

Migrate-together verdict with #4 (move_interleaved): both defer. The RM data path is
irrelevant to the handshake gaps.

Helper untouched. Lines removed: 0.
