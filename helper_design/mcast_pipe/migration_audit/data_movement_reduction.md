DERIVED FROM: the 2026-08-03 data_movement/sort and reduction snapshot, mcast_pipe API v9, design/hazards_catalog.md, and archive/reconciliation/reconcile_2026-08-03.md

# Migration audit — data_movement + reduction

Scope: `data_movement/move`, `data_movement/sort`, `reduction/argmax`, `reduction/topk` dataflow kernels.
Verdict legend: **clean** = drops onto Pipe with no caller-side residue; **refactor(cost)** = migratable, residue/rework noted; **defer-raw(why)** = leave as-is.

## Per-kernel

| Kernel | mcast? | iterated block? | verdict | notes |
|---|---|---|---|---|
| move/`move_interleaved_with_overlap.cpp` | sem-flag mcast (3 rects) | NO (single shot) | **refactor(med)** | SENDER+RECEIVER barrier handshake. Multi-rectangle dest set + dual-use L1 word (counter & flag). Legacy API. |
| move/`move_stick_layout_interleaved_with_overlap.cpp` | sem-flag mcast (3 rects) | NO (single shot) | **refactor(med)** | Structural twin of the above; same block, RM data movement. Migrate together. |
| move/`reader_unary_local_l1_copy_backwards.cpp` | none | — | **defer-raw** | No mcast, no handshake. Not in scope. |
| sort/`coordinator_single_row_multi_core.cpp` | helper Counter signal (1 rect) | YES (Ht × substages) | **migrated v9** | Coordinator SENDER. `Mcast2D` + Counter `send_signal()` owns only the phase broadcast; row-ready and per-substage-done exact-count waits/reset remain two op-owned semaphores. Code `7337302b564`. |
| sort/`reader_single_row_multi_core.cpp` | helper Counter signal receiver | YES | **migrated v9** | Worker RECEIVER. Counter `receive_signal()` replaces the inverted level flag; row-ready `up` remains op-owned. Code `7337302b564`. |
| sort/`writer_single_row_multi_core.cpp` | none | YES | **defer, helper-neutral companion** | Confirmation-emit only (atomic inc) on the distinct `done` semaphore. It received coupled dead-ABI cleanup but has no Pipe face and is not counted as migrated. |
| sort/`cross_core_data_exchange_common.hpp` | none (peer-to-peer unicast) | YES | **defer-raw** | All-to-all peer exchange via unicast inc/wait; NO multicast. Incidental, not the block. |
| sort/`reader_cross_core_data_exchange.cpp` | none | — | **defer-raw** | Uses the peer-exchange helper above; no mcast. |
| sort/`writer_cross_core_data_exchange.cpp` | none | — | **defer-raw** | Plain write barriers; no mcast. |
| sort/`reader_single_row_single_core.cpp`, `writer_single_row_single_core.cpp` | none | — | **defer-raw** | Single-core; no handshake. |
| argmax/`reader_argmax_interleaved_multicore.cpp` | sem-flag mcast (2 rects, both modes) | YES (`k` loop) | **refactor(high)** | Reference SENDER. Two rectangles w/ INCLUDE_SRC + EXCLUDE_SRC; monotone start counter + reset done counter; data fan-in is unicast. Richest fork coverage. |
| argmax/`reader_argmax_interleaved.cpp`, `reader_argmax_tile_layout.cpp`, `argmax_*.hpp` | none | — | **defer-raw** | Single-core / no mcast. |
| topk/`reader_final_topk.cpp` | sem-flag mcast (1 rect, EXCLUDE_SRC) | YES (Ht loop) | **clean** | Cleanest RECEIVER: reset-inbound + set-ready + mcast(EXCLUDE_SRC) + barrier + wait(counter), wrapped by CB reserve/push. |
| topk/`writer_local_topk.cpp` | none (consumes mcast) | YES (Ht loop) | **refactor(low)** | SENDER companion: wait(invite) + unicast data scatter + up(counter) + reset. |
| topk/`reader_final_topk` peers: `reader_create_index_*.cpp`, `reader_create_index_tensor.cpp`, `writer_binary_interleaved.cpp`, `writer_final_topk.cpp`, `topk_dataflow_common.hpp` | none | — | **defer-raw** | No mcast / plain barriers. |

## Counts
- Kernels scanned: 23 (move 3, sort 9, argmax 5, topk 6).
- Contain a TRUE iterated mcast-block (or its tightly-paired half): **6** — sort coordinator+reader(+writer), argmax multicore, topk reader_final+writer_local.
- Single-shot mcast-block (handshake, not iterated): **2** — move interleaved + stick.
- Incidental / no-mcast (defer-raw): **15**.
- Kernels that actually *emit* a multicast: **5** — move interleaved, move stick, sort coordinator,
  argmax multicore, and topk reader_final.

## Headline blockers
1. **No data multicast anywhere in this group.** Every multicast is a 4-byte semaphore flag/value. Data fan-in/out is plain interleaved read/write (move), peer unicast (sort exchange), or unicast scatter to a coordinator (argmax, topk). API v9 has the required control-only surface (`send_signal` / `receive_signal`); the remaining question is which call sites fit its Flag or Counter staging without forcing unrelated return counters into the Pipe.
2. **Multi-rectangle destination sets.** move (2-3 rects) and argmax (2 rects, *different loopback modes per rect*) need a dest *set*, not a single rectangle. Helper must accept a list where each entry carries its own INCLUDE/EXCLUDE_SRC, or the call site keeps a loop.
3. **Cross-kernel reset ownership.** TopK still has a receiver-init ordering blocker. Sort no longer needs shared level-flag reset ownership if its phase channel is reformulated as API-v9 Counter staging: `inc_multicast` + `wait_min`, with no reset. Its two return counters remain explicitly op-owned and reset only by the coordinator.
4. **Mixed F1 within a single kernel.** argmax, topk-writer, sort-writer all use `async_write_barrier` for data and `async_atomic_barrier` for the atomic inc. The helper cannot pick one flush globally; flush kind must follow the last op (write vs atomic).
5. **Mixed F2 within a single kernel.** argmax uses a monotone (no-reset) `start` counter AND a reset `done` counter simultaneously. F2 is per-slot, not per-pipe.
6. **API era split narrowed.** Move remains on the legacy free-function API. Sort was ported upstream to `Noc` / `Semaphore<>`, so its old object-API prerequisite is gone; its remaining cost is the host/device wire rewrite and Counter-staging validation.

## Focused sort re-audit after upstream protocol split (2026-08-03)

- **Required behavior:** one coordinator broadcasts an ordered phase event to every worker; readers report
  per-row readiness, and writers report per-pair completion. Readiness and completion have different
  expected counts and are independent channels.
- **Current implementation:** one inverted-polarity level flag carries the phase event; two separate
  exact-match counters (`ready`, `done`) carry the return legs. The split is semantically significant:
  folding them together can overshoot an exact wait and deadlock at `Ht >= 2`.
- **API-v9 formulation:** host `Mcast2D(all_core_set, coordinator, handshake=false,
  data_ready=Counter, adopted sem id 0)` emits the phase channel. Kernel `McastArgs` constructs a
  Counter `SenderPipe`/`ReceiverPipe`; coordinator calls `send_signal()`, reader calls
  `receive_signal()`. The raw ready/done waits and `up`s stay outside the helper.
- **Why this is not a new helper feature:** runtime rectangle/fan-out, adopted sem IDs, a no-handshake
  control-only channel, and Counter staging already exist. The changing ready/done counts describe the
  surrounding sort protocol, not the multicast channel itself.
- **Coverage closed:** Step G added four control-only Counter `send_signal`/`receive_signal` cases
  (1×2/1×8, 2/32 back-to-back signals); the complete helper suite passes 72/72.
- **Migration unit:** coordinator + reader + writer + `sort_program_factory.cpp`. The writer is a
  helper-neutral protocol companion. Apply must decide its ledger disposition explicitly rather than
  marking a file that never references the helper as migrated.
- **Mapped validation complete:** exact fresh-cache `test_sort_long_tensor` under `--dev` confirmed
  all three JIT artifacts; `test_sort_multi_row_multi_core_no_deadlock` passed both `Ht=2`
  descending values; the complete `test_sort_long_tensor` inventory passed 7/7.
- **Recall sweep:** the whole sort kernel directory contains no additional multicast emitter; only the
  coordinator calls `set_multicast`. Reader and writer remain in the inventory solely as tightly-paired
  protocol halves. No new spelling or inventory path was found.
