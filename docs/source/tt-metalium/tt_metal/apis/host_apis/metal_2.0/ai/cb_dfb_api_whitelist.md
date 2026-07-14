# CB → DFB API whitelist

> **Status:** Living document (2026-07-14). Mapping of legacy Circular Buffer (CB) device APIs to DataflowBuffer (DFB) variants for Metal 2.0 / WH-BH mechanical ports.
>
> **Consumers:** `[cb_dfb_kernel_audit_helper.md](cb_dfb_kernel_audit_helper.md)` (kernel audit Classes 1–5); porting guides may cite this list without duplicating it.
>
> **Status values:** `exists` — in-tree and usable now · `in flight` — API designed / PR open or about to land · not yet in main tree for auditors to rely on.

---

## How porting should use this list

1. **Class 1 (canonical FIFO):** use section **A** (+ **B** / **C** getters) only. For DM transfers use Device 2.0 `**Noc` APIs from `noc.h`** (traits + `offset_bytes`) — **not** kernel `get_*_ptr` + `dataflow_api.h` `noc_async_*` (see [Access control](#access-control-get_ptr-vs-evil)).
2. **Classes 2–5 (pointer / credit reach-arounds):** keep kernel semantics; map `LocalCBInterface` pointer field access to section **D** (`evil_get_*` / `evil_set_*` on the `DataflowBuffer` object). Do **not** invent scratchpad / LTA mid-port on WH/BH unless LLK already allows it.
3. **Never** keep `get_local_cb_interface(...).fifo_*_ptr` writes after CB→DFB uplift (section **E**).
4. Do **not** add per-pattern DFB helpers (e.g. no `evil_push_back_hold_write_ptr`) — compose canonical FIFO + evil setters.

---

## Access control: `get_*_ptr` vs `evil_*`

**API end-state** (getters not privatized in tree yet — design target for the DFB/NOC follow-up):


| API                                            | Visibility    | Who may use it                                                                                                                                  |
| ---------------------------------------------- | ------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| `get_read_ptr()` / `get_write_ptr()`           | `**private`** | `friend` plumbing only: `noc_traits_t<DataflowBuffer>`, and any `Noc` helpers in `noc.h` that resolve L1 addrs for `async_read` / `async_write` |
| `evil_get_read_ptr()` / `evil_get_write_ptr()` | `**public**`  | Kernels (Classes 2–5) that snapshot FIFO cursors for save/restore, scatter base, rewind                                                         |
| `evil_set_read_ptr()` / `evil_set_write_ptr()` | `**public**`  | Same — mutate cursors; never `LocalCBInterface` field writes                                                                                    |


**Kernel litmus (Device 2.0):**


| Kernel intent                                  | Correct API                                                                                                                      | Wrong                                                        |
| ---------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------ |
| Class 1 tile transfer after `reserve` / `wait` | `Noc::async_read` / `async_write` (etc.) from `**noc.h`**, with DFB endpoints + `offset_bytes` — traits call private `get_*_ptr` | Kernel `get_*_ptr` + `noc_async_*` from `**dataflow_api.h**` |
| Classes 2–5 cursor surgery                     | `evil_get_*` / `evil_set_*`                                                                                                      | `get_*_ptr`, or `LocalCBInterface` field R/W                 |


```cpp
// ✅ Class 1 — Device 2.0 (kernel never touches the FIFO byte addr)
dfb.reserve_back(1);
noc.async_write(src, dfb, {.offset_bytes = k * tile_size});
dfb.push_back(1);

// ❌ Not acceptable on a Metal 2.0 / Device 2.0 port
dfb.reserve_back(1);
uint32_t addr = dfb.get_write_ptr();
noc_async_write(/*...*/, addr + offset, /*...*/);  // dataflow_api.h style
dfb.push_back(1);
```

Kernels do **not** “get a ptr then pass it into traits.” Traits/`Noc` obtain the address internally. If a kernel needs the cursor for anything other than a framework transfer, that is `**evil_*`** (and Quasar redesign debt).

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


| CB / field                         | DFB                                                      | Status                                                                   |
| ---------------------------------- | -------------------------------------------------------- | ------------------------------------------------------------------------ |
| `fifo_page_size` / entry size      | `get_entry_size()` (+ aliases)                           | in flight ([#49652](https://github.com/tenstorrent/tt-metal/pull/49652)) |
| `fifo_num_pages` / capacity        | `get_total_num_entries()` / local-TC / ring-span getters | in flight (#49652)                                                       |
| Total / local / span size in bytes | `get_total_size_bytes()`, local/span size APIs           | in flight (#49652)                                                       |


**Note:** TRISC size getters return sizes in **bytes**, not 16B units (see #49652 notes).

---

## C. Scalar tile reads


| CB                                        | DFB                            | Status                                                                                                   |
| ----------------------------------------- | ------------------------------ | -------------------------------------------------------------------------------------------------------- |
| `read_tile_value` / `read_tile_value_u16` | templated `read_tile_value<T>` | in flight ([#49617](https://github.com/tenstorrent/tt-metal/pull/49617)) — WH/BH; Quasar follow-up after |
| `get_tile_address`                        | `get_tile_address`             | in flight (#49617) on DFB for WH/BH; Quasar follow-up                                                    |


Until #49617 merges: treat DFB `read_tile_value` / `get_tile_address` as **QUASAR-BLOCKED** / WH-BH blocked on pure DFB ports that need those APIs (see audit helper Runtime fixes).

---

## D. Reach-around pointer APIs (Classes 2–5 — evil getters/setters only)

**Naming:** `evil_` prefix on **DataflowBuffer member methods**. Greppable and intentionally ugly — not a stable Quasar end-state.


| Legacy CB / pattern                                                            | Intended `DataflowBuffer` member                                                                                  | Composition / note                                                                                      | Class | Status                              |
| ------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- | ----- | ----------------------------------- |
| `get_write_ptr()` / `get_read_ptr()` used to jump, save/restore, or scatter    | `evil_get_write_ptr` / `evil_get_read_ptr`                                                                        | —                                                                                                       | 2–5   | **in flight**                       |
| `get_local_cb_interface(...).fifo_wr_ptr` / `fifo_rd_ptr` assign / `+=` / wrap | `**evil_set_write_ptr` / `evil_set_read_ptr`**                                                                    | Kernel keeps arithmetic; setter takes absolute cursor — **never** leave `LocalCBInterface` field writes | 2–5   | **in flight**                       |
| SDPA `cb_push_back_hold_wr_ptr` (push credits, rewind wr_ptr)                  | **not a new DFB API**                                                                                             | `dfb.push_back(n)` + `dfb.evil_set_write_ptr(rewound)` (wrap in kernel)                                 | 4     | compose using **in flight** setters |
| Bilinear / manual wr advance without credits                                   | `evil_set_write_ptr`                                                                                              | Credits intentionally skipped                                                                           | 4     | **in flight**                       |
| Class 5 packer L1 acc save/restore                                             | `evil_get_`* then `evil_set_*`                                                                                    | Same DFB object for get and set                                                                         | 5     | **in flight**                       |
| Gathered-matmul ring `local_cb.fifo_rd_ptr = …` helpers                        | `evil_set_read_ptr` (+ get as needed)                                                                             | Replace ref-to-interface helpers with DFB members                                                       | 4–5   | **in flight**                       |
| Bare `get_*_ptr` + `dataflow_api.h` `noc_async_*` for Class 1 transfers        | **Rewrite** to `Noc::async_*` from `**noc.h`** (private/`friend` `get_*_ptr` under the hood) — **not** OK to keep | Must-fix on Device 2.0 ports                                                                            | 1     | rewrite debt                        |


```cpp
// Example: SDPA hold-wr uplift (no dedicated hold API)
DataflowBuffer dfb(dfb::qkt_im);
dfb.push_back(n);
dfb.evil_set_write_ptr(/* saved or rewound base */);  // wrap logic stays in the kernel
```

**Policy:**


| Arch                   | Evil get/set                                                                                                                       |
| ---------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| WH / BH (1xx) v1 port  | **Allowed** — mechanical CB→DFB, no semantic change. Audit: **Portable (workaround)** + Notes **Quasar redesign required**.        |
| Quasar (2xx) end-state | **Forbidden** as the lasting fix — redesign to scratchpad + semaphores / LTA / compute self-loop / strided DFB (see audit helper). |


---

## E. Hard rejects (never whitelist)


| Pattern                                                             | Note                                            |
| ------------------------------------------------------------------- | ----------------------------------------------- |
| Direct `get_local_cb_interface(...).fifo_*_ptr` after CB→DFB uplift | GATE — must use DFB `evil_get_*` / `evil_set_*` |
| Other `get_local_cb_interface(...).field` (e.g. `fifo_page_size`)   | GATE — use section **B** getters                |
| Kernel `get_*_ptr` + `dataflow_api.h` `noc_async_*` (Class 1)       | Must-fix — use `Noc` from `noc.h` instead       |
| Kernel `get_*_ptr` for cursor save/jump/rewind                      | Must-fix — use `evil_get_*` / `evil_set_*`      |
| `get_cb_tiles_acked_ptr` / `get_cb_tiles_received_ptr`              | Silent-wrong on Quasar                          |
| `get_pointer_to_cb_data`                                            | Migrate to `LocalTensorAccessor` (not DFB evil) |
