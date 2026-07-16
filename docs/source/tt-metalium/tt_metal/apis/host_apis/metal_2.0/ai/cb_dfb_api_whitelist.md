# CB → DFB API whitelist

> **Status:** Living document (2026-07-16). Mapping of legacy Circular Buffer (CB) device APIs to DataflowBuffer (DFB) variants for Metal 2.0 / WH-BH mechanical ports.
>
> **Consumers:** `[cb_dfb_kernel_audit_helper.md](cb_dfb_kernel_audit_helper.md)` (kernel audit Classes 1–5); porting guides may cite this list without duplicating it.
>
> **Status values:** `exists` — in-tree and usable now · `in flight` — API designed / PR open or about to land · not yet in main tree for auditors to rely on.

---

## How porting should use this list

1. **Class 1 (canonical FIFO):** use section **A** (+ **B** / **C** / public `get_*_ptr` peeks when needed). For DM transfers prefer Device 2.0 `**Noc` APIs from `noc.h`** (DFB endpoints + `offset_bytes`) over leftover `get_*_ptr` + `dataflow_api.h` `noc_async_*` (see [Access control](#access-control-get_ptr-vs-evil_set)).
2. **Classes 2–5 (cursor surgery):** keep kernel semantics; map `LocalCBInterface` pointer **mutations** to section **D** (`evil_set_*` on the `DataflowBuffer` object). Peeks of the current cursor use public `get_*_ptr` — there are **no** `evil_get_*` APIs. Do **not** invent scratchpad / LTA mid-port on WH/BH unless LLK already allows it.
3. **Never** keep `get_local_cb_interface(...).fifo_*_ptr` writes after CB→DFB uplift (section **E**).
4. Do **not** add per-pattern DFB helpers (e.g. no `evil_push_back_hold_write_ptr`) — compose canonical FIFO + `evil_set_*`.

---

## Access control: `get_*_ptr` vs `evil_set_*`

**Differentiator:** peeking the **current** FIFO cursor (where entry data lives) is public `get_*_ptr`. Mutating FIFO cursor state (rewind / jump / hold-wr) is `evil_set_*`. There is **no** `evil_get_read_ptr` / `evil_get_write_ptr`.


| API                                            | Visibility | Who may use it                                                                                                                                          |
| ---------------------------------------------- | ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `get_read_ptr()` / `get_write_ptr()`           | **public** | Any kernel that needs the **current** FIFO cursor address — local entry L1 access, or interop with helpers that still take a raw L1 address             |
| `evil_set_read_ptr()` / `evil_set_write_ptr()` | **public** | WH/BH only — kernels (Classes 2–5) that **mutate** cursors (save/restore, rewind, jump). Never `LocalCBInterface` field writes. Not available on Quasar |


**Kernel litmus (Device 2.0):**


| Kernel intent                                           | Correct API                                                                                           | Wrong                                                                      |
| ------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| Class 1 tile/stick transfer after `reserve` / `wait`    | `Noc::async_read` / `async_write` from `**noc.h`**, with DFB endpoints + `offset_bytes` when possible | Prefer not: leftover `get_*_ptr` + `noc_async_*` from `**dataflow_api.h`** |
| Peek current entry for local L1 access / helper interop | public `get_read_ptr` / `get_write_ptr`                                                               | inventing `evil_get_*`, or using `evil_set_*` for a peek                   |
| Classes 2–5 cursor surgery (mutate FIFO `fifo_*_ptr`)   | `evil_set_*` (snapshot first with public `get_*_ptr` if needed)                                       | `LocalCBInterface` field R/W; treating surgery as plain `get_*_ptr` only   |


```cpp
// ✅ Class 1 — Device 2.0 (prefer noc.h; no leftover peek)
dfb.reserve_back(1);
noc.async_write(src, dfb, size, {}, {.offset_bytes = k * tile_size});
dfb.push_back(1);

// ✅ Peek current cursor for local entry data (not surgery)
uint32_t addr = dfb.get_write_ptr();
auto* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr);
// ... fill / decode entry at current FIFO slot ...

// ✅ Cursor surgery (WH/BH) — mutate FIFO state
uint32_t saved = dfb.get_read_ptr();   // public peek to snapshot
// ...
dfb.evil_set_read_ptr(saved);          // evil: restore / rewind

// ❌ Not acceptable Class 1 style on a Metal 2.0 / Device 2.0 port
dfb.reserve_back(1);
uint32_t addr = dfb.get_write_ptr();
noc_async_write(/*...*/, addr + offset, /*...*/);  // dataflow_api.h style
dfb.push_back(1);
```

Kernels do **not** need `evil_get_`* to snapshot a cursor before surgery — use public `get_*_ptr`, then `evil_set_*` to put the cursor back (or to a computed address).

---

## A. Canonical FIFO (Class 1 — blessed)


| CB                                                | DFB                      | Status |
| ------------------------------------------------- | ------------------------ | ------ |
| `reserve_back`                                    | `reserve_back`           | exists |
| `push_back`                                       | `push_back`              | exists |
| `wait_front`                                      | `wait_front`             | exists |
| `pop_front`                                       | `pop_front`              | exists |
| `finish` (if used)                                | `finish`                 | exists |
| Tile metadata (`get_tile_size`, formats, dims, …) | same on `DataflowBuffer` | exists |


---

## B. Size / layout queries (no pointer surgery)


| CB / field                         | DFB                                                      | Status |
| ---------------------------------- | -------------------------------------------------------- | ------ |
| `fifo_page_size` / entry size      | `get_entry_size()` (+ aliases)                           | exists |
| `fifo_num_pages` / capacity        | `get_total_num_entries()` / local-TC / ring-span getters | exists |
| Total / local / span size in bytes | `get_total_size_bytes()`, local/span size APIs           | exists |


**Note:** TRISC size getters return sizes in **bytes**, not 16B units (see #49652 notes).

---

## C. Public cursor peeks + scalar tile reads

### Cursor peeks (blessed — not evil)


| CB                | DFB               | Status |
| ----------------- | ----------------- | ------ |
| `get_write_ptr()` | `get_write_ptr()` | exists |
| `get_read_ptr()`  | `get_read_ptr()`  | exists |


Public on **1xx and Quasar**. Observe the current FIFO cursor only. Prefer `noc.h` + DFB for Class 1 transfers so leftover peeks are not required; keep `get_*_ptr` when the kernel truly needs the entry address (local L1 access, or helpers that still take raw L1).

There are **no** `evil_get_write_ptr` / `evil_get_read_ptr` APIs.

### Scalar tile reads


| CB                                        | DFB                            | Status                                |
| ----------------------------------------- | ------------------------------ | ------------------------------------- |
| `read_tile_value` / `read_tile_value_u16` | templated `read_tile_value<T>` | exists for WH/BH; Quasar to follow up |
| `get_tile_address`                        | `get_tile_address`             | exists for WH/BH; Quasar follow-up    |


Treat DFB `read_tile_value` / `get_tile_address` as **QUASAR-BLOCKED**.

---

## D. Reach-around pointer APIs (Classes 2–5 — `evil_set_`* only)

**Naming:** `evil_` prefix on **DataflowBuffer member setters** that **mutate** FIFO cursors outside canonical `reserve`/`push`/`wait`/`pop`. Greppable and intentionally ugly — not a stable Quasar end-state.

**Not in this section:** peeks. Snapshotting a cursor before surgery uses public `get_*_ptr` (section **C**).


| Legacy CB / pattern                                                            | Intended `DataflowBuffer` member                             | Composition / note                                                                                      | Class | Status                                                                                                                                 |
| ------------------------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------- | ----- | -------------------------------------------------------------------------------------------------------------------------------------- |
| `get_local_cb_interface(...).fifo_wr_ptr` / `fifo_rd_ptr` assign / `+=` / wrap | `**evil_set_write_ptr` / `evil_set_read_ptr`**               | Kernel keeps arithmetic; setter takes absolute cursor — **never** leave `LocalCBInterface` field writes | 2–5   | **In-flight [#49971]([https://github.com/tenstorrent/tt-metal/pull/49971](https://github.com/tenstorrent/tt-metal/pull/49971))** (1xx) |
| Snapshot cursor then restore / rewind                                          | `get_*_ptr` then `evil_set_`*                                | Peek is public; only the mutate is evil                                                                 | 2–5   | **exists** (1xx)                                                                                                                       |
| SDPA `cb_push_back_hold_wr_ptr` (push credits, rewind wr_ptr)                  | **not a new DFB API**                                        | `dfb.push_back(n)` + `dfb.evil_set_write_ptr(rewound)` (wrap in kernel)                                 | 4     | compose                                                                                                                                |
| Bilinear / manual wr advance without credits                                   | `evil_set_write_ptr`                                         | Credits intentionally skipped                                                                           | 4     | **exists** (1xx)                                                                                                                       |
| Class 5 packer L1 acc save/restore                                             | `get_*_ptr` then `evil_set_`*                                | Same DFB object for peek and set                                                                        | 5     | **exists** (1xx)                                                                                                                       |
| Gathered-matmul ring `local_cb.fifo_rd_ptr = …` helpers                        | `evil_set_read_ptr` (+ `get_read_ptr` as needed)             | Replace ref-to-interface helpers with DFB members                                                       | 4–5   | **exists** (1xx)                                                                                                                       |
| Bare `get_*_ptr` + `dataflow_api.h` `noc_async_`* for Class 1 transfers        | **Rewrite** to `Noc::async_`* from `**noc.h`** when possible | Prefer DFB endpoint + `offset_bytes`; leftover peeks are cleanup debt, not evil                         | 1     | rewrite debt                                                                                                                           |


```cpp
// Example: SDPA hold-wr uplift (no dedicated hold API)
DataflowBuffer dfb(dfb::qkt_im);
dfb.push_back(n);
dfb.evil_set_write_ptr(/* saved or rewound base */);  // wrap logic stays in the kernel
```

**Policy:**


| Arch                   | `evil_set_`*                                                                                                                       |
| ---------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| WH / BH (1xx) v1 port  | **Allowed** — mechanical CB→DFB, no semantic change. Audit: **Portable (workaround)** + Notes **Quasar redesign required**.        |
| Quasar (2xx) end-state | **Forbidden** as the lasting fix — redesign to scratchpad + semaphores / LTA / compute self-loop / strided DFB (see audit helper). |


---

## E. Hard rejects (never whitelist)


| Pattern                                                             | Note                                                                 |
| ------------------------------------------------------------------- | -------------------------------------------------------------------- |
| Direct `get_local_cb_interface(...).fifo_*_ptr` after CB→DFB uplift | GATE — must use DFB `evil_set_`* for mutates; `get_*_ptr` for peeks  |
| Other `get_local_cb_interface(...).field` (e.g. `fifo_page_size`)   | GATE — use section **B** getters                                     |
| Kernel `get_*_ptr` + `dataflow_api.h` `noc_async_`* (Class 1)       | Prefer-fix — use `Noc` from `noc.h` with DFB endpoints when possible |
| Inventing / using `evil_get_`*                                      | Does not exist — use public `get_*_ptr`                              |
| Using `evil_set_*` when only a peek is needed                       | Wrong — public `get_*_ptr`                                           |
| `get_cb_tiles_acked_ptr` / `get_cb_tiles_received_ptr`              | Silent-wrong on Quasar                                               |
| `get_pointer_to_cb_data`                                            | Migrate to `LocalTensorAccessor` (not DFB evil)                      |
