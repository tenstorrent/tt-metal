# Migration audit — data_movement + reduction

Scope: `data_movement/move`, `data_movement/sort`, `reduction/argmax`, `reduction/topk` dataflow kernels.
Verdict legend: **clean** = drops onto Pipe with no caller-side residue; **refactor(cost)** = migratable, residue/rework noted; **defer-raw(why)** = leave as-is.

## Per-kernel

| Kernel | mcast? | iterated block? | verdict | notes |
|---|---|---|---|---|
| move/`move_interleaved_with_overlap.cpp` | sem-flag mcast (3 rects) | NO (single shot) | **refactor(med)** | SENDER+RECEIVER barrier handshake. Multi-rectangle dest set + dual-use L1 word (counter & flag). Legacy API. |
| move/`move_stick_layout_interleaved_with_overlap.cpp` | sem-flag mcast (3 rects) | NO (single shot) | **refactor(med)** | Structural twin of the above; same block, RM data movement. Migrate together. |
| move/`reader_unary_local_l1_copy_backwards.cpp` | none | — | **defer-raw** | No mcast, no handshake. Not in scope. |
| sort/`coordinator_single_row_multi_core.cpp` | sem-flag mcast (1 rect) | YES (Ht × substages) | **refactor(med)** | Coordinator SENDER. Two block phases (start + per-substage). Op-specific counts (`Wt/2`). Cross-kernel reset ownership. |
| sort/`reader_single_row_multi_core.cpp` | none (consumes mcast) | YES | **refactor(med)** | Worker RECEIVER (inverted flag, atomic-barrier inc). Pairs with coordinator. |
| sort/`writer_single_row_multi_core.cpp` | none | YES | **refactor(low)** | Confirmation-emit only (atomic inc). Return leg of the coordinator counter. |
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
- Kernels that actually *emit* a multicast: **4** (move interleaved, move stick, sort coordinator, argmax multicore, topk reader_final = 5 emitters; move counts as 2 files). Emitter files: 5.

## Headline blockers
1. **No data multicast anywhere in this group.** Every multicast is a 4-byte semaphore flag/value. Data fan-in/out is plain interleaved read/write (move), peer unicast (sort exchange), or unicast scatter to a coordinator (argmax, topk). A Pipe that bundles *data + flag* mcast has ZERO call sites here; the demanded primitive is **flag-only multicast + counter/flag handshake**.
2. **Multi-rectangle destination sets.** move (2-3 rects) and argmax (2 rects, *different loopback modes per rect*) need a dest *set*, not a single rectangle. Helper must accept a list where each entry carries its own INCLUDE/EXCLUDE_SRC, or the call site keeps a loop.
3. **Cross-kernel reset ownership.** In sort and topk the go/invite flag is reset on the *worker* side while the coordinator emits it; the inbound counter is reset on the coordinator side. A two-sided Pipe (`send`/`receive`) split across separate kernels must pin who resets which slot, or it will double-reset / never-reset.
4. **Mixed F1 within a single kernel.** argmax, topk-writer, sort-writer all use `async_write_barrier` for data and `async_atomic_barrier` for the atomic inc. The helper cannot pick one flush globally; flush kind must follow the last op (write vs atomic).
5. **Mixed F2 within a single kernel.** argmax uses a monotone (no-reset) `start` counter AND a reset `done` counter simultaneously. F2 is per-slot, not per-pipe.
6. **API era split.** move + sort are legacy free-function (`noc_semaphore_set_multicast`, `get_noc_multicast_addr`); argmax + topk are modern (`Semaphore<>::set_multicast<Mode>`, `Noc`). Migrating legacy kernels means also porting them to the modern Noc/Semaphore types first, raising move/sort cost to *med*.
