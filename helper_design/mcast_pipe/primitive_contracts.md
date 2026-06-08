# Step A — Primitive Contracts (`mcast_pipe`)

PRIMITIVES: Noc::async_write_multicast, Noc::async_writes_flushed, Noc::async_write_barrier,
Semaphore::set, Semaphore::set_multicast, Semaphore::wait, Semaphore::wait_min, Semaphore::up,
Semaphore::inc_multicast, MulticastEndpoint, enums{McastMode,VcSelection,BarrierMode,ResponseMode}
| recognition: noc_async_write_multicast(+_loopback_src,+_one_packet),
noc_semaphore_set_multicast(+_loopback_src), noc_semaphore_set, noc_semaphore_wait,
noc_semaphore_wait_min, noc_semaphore_inc, noc_semaphore_inc_multicast,
noc_async_writes_flushed, noc_async_write_barrier, get_noc_multicast_addr

All contracts read from source this run (no prior-run reuse). Two layers:
- **Substrate** (what the helper + bake-off kernels are BUILT FROM): object API —
  `Noc` (`noc.h`), `Semaphore<>` (`noc_semaphore.h`), `MulticastEndpoint` (`endpoints.h`).
- **Recognition family** (what Step D spots in existing kernels, however spelled): the legacy
  free functions in `dataflow_api.h` that the object methods delegate to.

Lifecycle vocabulary (dataflow): **enqueued** (descriptor consumed by NoC engine) →
**SENT** (bytes left this core's NoC engine) → **ARRIVED** (landed at receiver L1) →
**ACKed** (receiver acknowledged, visible to sender). A "wait" primitive is pinned to exactly
one of these.

---

## SUBSTRATE — `Noc` (tt_metal/hw/inc/api/dataflow/noc.h)

### A1. `Noc::async_write_multicast<McastMode, TxnIdMode, ResponseMode, max_page_size>(src, dst, size_bytes, num_dsts, src_args, dst_args, linked=false, trid)`
`noc.h:332-361`. Delegates to raw `noc_async_write_multicast` (EXCLUDE_SRC) or
`noc_async_write_multicast_loopback_src` (INCLUDE_SRC).
- **Does:** enqueues a multicast write of `size_bytes` from local L1 `src` to the
  `MulticastEndpoint` rectangle `dst`. Returns after **enqueued** — NOT sent, NOT arrived.
- **Lifecycle stage reached:** enqueued only. Pair with a flush (→SENT) or barrier (→ACKed).
- **Hard constraints baked into the object method (load-bearing for the bake-off):**
  - `static_assert(txn_id_mode == DISABLED)` — **mcast cannot use transaction ids.**
  - `static_assert(response_mode == NON_POSTED)` — **mcast cannot be POSTED.**
  - ⇒ For mcast, the *only* available post-send waits are `BarrierMode::FULL` flush/barrier.
    The flush-vs-barrier fork is therefore **FULL-flush vs FULL-barrier**, nothing finer.
- **`McastMode` (raw EXCLUDE_SRC vs `_loopback_src`):**
  - `EXCLUDE_SRC` (default): data **NOT** written to self even if self ∈ rect; `num_dsts`
    excludes self (8×8 incl self ⇒ 63). Rect must contain ≥1 *other* core.
  - `INCLUDE_SRC` (loopback): data **also** written to self; `num_dsts` counts self (⇒ 64).
    `num_dsts == 1` (self-only) is **unspecified — may hang** (raw docstring). Guard degenerate rects.
- **VC:** raw forces `NOC_MULTICAST_WRITE_VC` (=4) and `multicast_path_reserve=true`, always.
  Object method exposes no VC override on the mcast path → all mcast data rides VC 4.
- **cmd buf:** `write_cmd_buf` (0).
- `linked` = hold the static VC reservation across back-to-back packets (perf, not correctness).

### A2. `Noc::async_writes_flushed<ResponseMode, BarrierMode>(trid)`
`noc.h:560-574` → raw `noc_async_writes_flushed` (`dataflow_api.h:441`).
- **Waits for:** **SENT** — polls `ncrisc_noc_nonposted_writes_sent`: all outstanding nonposted
  writes have *departed this core's NoC engine*. Does **not** wait for arrival/ACK.
- **Cost:** local L1/register poll (cheap). `invalidate_l1_cache()` on return.
- **Mitigates:** source-L1-clobber (safe to overwrite the src buffer once SENT).
- **Does NOT guarantee:** receivers have the data.
- For mcast (NON_POSTED, no trid) this is the `FULL` variant: `noc_async_writes_flushed(noc_id)`.

### A3. `Noc::async_write_barrier<BarrierMode>(trid)`
`noc.h:537-545` → raw `noc_async_write_barrier` (`dataflow_api.h:413`).
- **Waits for:** **ACKed** — polls `ncrisc_noc_nonposted_writes_flushed`: all outstanding
  nonposted writes have *completed* (receivers acknowledged). NoC round-trip.
- **Cost:** strictly ≥ flush (waits for the far end).
- **Mitigates:** source-clobber AND gives a "receivers actually got it" guarantee.
- `BarrierMode::TXN_ID` exists but is unusable for mcast (A1 forbids trid). ⇒ mcast uses `FULL`.

---

## SUBSTRATE — `Semaphore<>` (tt_metal/hw/inc/api/dataflow/noc_semaphore.h)

### A4. `Semaphore::set(value)`
`noc_semaphore.h:109` → raw `noc_semaphore_set` (`dataflow_api.h:547`).
- **Does:** a single local L1 store `(*sem_addr) = value`. **No NoC traffic, NOT a sync op.**
- It "synchronizes" only because some other core later waits on that cell.
- **Dual role for mcast:** also *stages the 4-byte payload* that `set_multicast` will broadcast
  (see A5) — the local sem cell IS the mcast source.

### A5. `Semaphore::set_multicast<McastMode>(noc, x0, y0, x1, y1, num_dests, linked=false)`
`noc_semaphore.h:126-148` → raw `noc_semaphore_set_multicast` / `_loopback_src`.
- **Does:** multicasts the **local sem cell's current 4-byte value** (the value last written by
  `set()`) to the rectangle, setting each receiver's sem cell. Returns after **enqueued**.
- **GOTCHA (the single biggest one):** the source is the *local L1 cell*, not an argument value.
  Must `set(v)` first; must not overwrite the cell before the mcast is **SENT** (A2) or the
  in-flight payload corrupts.
- **cmd buf:** `write_reg_cmd_buf` (2) — different buf from data mcast (0) ⇒ parallel issue.
- **VC:** `NOC_MULTICAST_WRITE_VC` (=4), same as data mcast ⇒ **same-VC FIFO ordering**: a
  `async_write_multicast(...)` then `set_multicast(...)` arrive in that order at every receiver
  (data lands before the flag). This is what lets a flag-set replace a barrier.
- **McastMode:** EXCLUDE_SRC excludes self (num_dests=63 for 8×8 incl self); INCLUDE_SRC
  includes self (=64). Loopback `num_dests==1` may hang — same degeneracy as A1.
- NON_POSTED (no posted variant exposed).

### A6. `Semaphore::wait(value)`
`noc_semaphore.h:91` → raw `noc_semaphore_wait` (`dataflow_api.h:495`).
- **Waits for:** local cell `== value` **exact match** (`while ((*addr) != val)`).
- **GOTCHA:** if the cell already equals `value` from a prior round, returns immediately. With a
  level flag (VALID/INVALID) the receiver MUST reset to the off-value before the next round, or
  the protocol must use a monotone counter (→ `wait_min`). **This is the flag-vs-counter fork.**
- `invalidate_l1_cache()` each spin.

### A7. `Semaphore::wait_min(value)`
`noc_semaphore.h:100` → raw `noc_semaphore_wait_min` (`dataflow_api.h:521`).
- **Waits for:** local cell `>= value` (`while ((*addr) < val)`).
- **Counter style:** never needs a reset — each round waits for a strictly larger threshold, so
  a monotone counter that only ever increments works across iterations without a reset write.

### A8. `Semaphore::up(value)` (local) / `Semaphore::up(noc, x, y, value, vc=NOC_UNICAST_WRITE_VC)` (remote)
`noc_semaphore.h:49 / 63` → local `(*addr)+=value` / raw `noc_semaphore_inc` (`dataflow_api.h:585`).
- **Local `up`:** non-atomic `+=` to own cell. **Remote `up`:** atomic increment of a remote
  core's cell over NoC (`write_at_cmd_buf`, **VC 1** unicast by default), NON_POSTED (gets ACK)
  unless `posted` template set on the raw call.
- This is the **receiver→sender "I'm ready / I consumed it" signal** (R→S handshake half).

### A9. `Semaphore::inc_multicast(noc, x0, y0, x1, y1, value, num_dests)`
`noc_semaphore.h:162-173` → raw `noc_semaphore_inc_multicast` (`dataflow_api.h:629`).
- **Does:** atomic increment (32-bit wrap) of the sem cell on every core in the rectangle.
  `write_at_cmd_buf`, **VC 4**, `multicast_path_reserve=true`, NON_POSTED. Sender **cannot** be
  in dests; `num_dests` excludes self.
- Alternative S→R "advance the counter on all receivers" primitive (vs `set_multicast` of a flag).

---

## SUBSTRATE — addressing

### A10. `MulticastEndpoint` + `dst_args_mcast_t` (endpoints.h:23, 77-92)
- `MulticastEndpoint{}` is stateless; the rect + dest address travel in
  `dst_args_mcast_type {noc_x_start, noc_y_start, noc_x_end, noc_y_end, addr}`.
- `dst_addr_mcast` calls `::get_noc_multicast_addr(x0,y0,x1,y1,addr,noc)` — the same encoding the
  open-coded census tell uses. **This is how the `Pipe` carries "ANY rectangle + ANY dst addr".**
- Source side: `src` is a local-L1 object whose `src_args.addr` is "ANY source L1 address".

### A11. Enums (`noc.h:43-49`)
- `McastMode {INCLUDE_SRC, EXCLUDE_SRC}` — loopback selector (A1/A5). **The loopback fork.**
- `VcSelection {DEFAULT, CUSTOM}` — VC override; **not exposed on the mcast path** → mcast is
  always VC 4. (Relevant only to unicast read/write set-state.)
- `BarrierMode {TXN_ID, FULL}` — barrier granularity; **mcast is FULL-only** (A1 forbids trid).
- `ResponseMode {NON_POSTED, POSTED}` — **mcast is NON_POSTED-only** (A1 static_assert).

---

## Consequences for the bake-off backlog (preview — Step B owns the catalog)

The enum constraints **prune the fork space before we touch the device**:
| Candidate fork | Survives? | Why |
|---|---|---|
| flush (SENT) vs barrier (ACKed) | **YES** — bake off | A2 vs A3, both legal for mcast |
| flag-sem (`set`+`wait`, exact) vs counter-sem (`inc_multicast`/`up`+`wait_min`) | **YES** — bake off | A6 vs A7 reset semantics |
| EXCLUDE_SRC vs INCLUDE_SRC loopback (sender ∈ rect) | **YES** — bake off | A1/A5; degenerate `num_dests==1` is a coverage trap |
| `set_multicast` flag vs `inc_multicast` counter for S→R | **YES** — bake off | A5 vs A9 |
| POSTED vs NON_POSTED mcast | **NO** | A1 static_assert — POSTED illegal for mcast |
| `BarrierMode::TXN_ID` for mcast | **NO** | A1 forbids trid on mcast |
| custom VC for mcast | **NO** | not exposed; mcast hard-wired VC 4 |
| `linked` on/off | maybe (perf knob) | correctness-neutral; defer to perf pass if it matters |
