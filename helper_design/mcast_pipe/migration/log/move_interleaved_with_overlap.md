# move_interleaved_with_overlap.cpp — DEFERRED (design-gap)

Tier 3. Status: deferred. No code change.

## Role
Single-shot go-flag mcast (TILE layout). One controller core fans a go-flag to 2-3
receiver rectangles; receivers ack first, then wait the flag, then write their CB to dst.

## Why deferred (multiple known helper design gaps)
1. **Runtime role selection.** `is_controller = get_arg_val<uint32_t>(8) == 1` picks
   sender-vs-receiver per core under ONE compiled binary. v7 `SenderPipe`/`ReceiverPipe`
   are distinct compile-time-chosen types. Same gap as sdpa_decode read_k.
2. **Multi-rectangle (2-3 rects) with runtime per-rect counts.** `range_{0,1,2}_size` and
   `do_third_multicast` are all runtime args; the controller loops `set_multicast` over up
   to three rects. v7 takes a single `McastRect` + compile-time
   `NUM_ACTIVE_RECEIVER_CORES`. Same gap as gn_v2 / welford / moe_compute.
3. **Dual-use L1 word (counter & flag).** `control_value` (runtime) is the receiver count:
   receivers `up()` the cell (counter), the controller `wait(control_value)` then
   broadcasts the SAME cell as a go-flag via `set_multicast`. One word multiplexed as
   inbound counter and outbound flag — the helper keeps these as two separate
   `data_ready` / `consumer_ready` cells.
4. **Runtime sem id.** `semaphore_arg = get_arg_val<uint32_t>(4)`; v7 sem ids are
   compile-time template params.

Any one of #1-#4 alone blocks migration; all four are present. Overlap path also depends
on runtime allocator addresses (L1->L1 in-place) so it isn't even deterministically
reachable, but the design gaps are the binding blocker regardless.

Helper untouched. Lines removed: 0.
