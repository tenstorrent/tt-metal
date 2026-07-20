# CB → DFB API whitelist

> **Status:** Living document (2026-07-17). Mapping of legacy Circular Buffer (CB) device APIs to DataflowBuffer (DFB) variants for Metal 2.0 / WH–BH mechanical ports.

---

## How to use this list

1. Prefer canonical FIFO (**A**) and public peeks (**C**) for normal producer/consumer flow.
2. For DM transfers, prefer Device 2.0 `**Noc` APIs from `noc.h`** (DFB endpoints + `offset_bytes`) over leftover `get_*_ptr` + `dataflow_api.h` `noc_async_`* (see [Access control](#access-control-get_ptr-vs-evil_set)).
3. Map `LocalCBInterface` pointer **mutations** to `**evil_set_`*** (**D**). Peeks use public `get_*_ptr`.
4. Do not attempt to work around any missing DFB APIs. Please prominently note the issue in the porting report.

---

## Access control: `get_*_ptr` vs `evil_set_`*

**Differentiator:** peeking the **current** FIFO cursor (where entry data lives) is public `get_*_ptr`. Mutating FIFO cursor state (rewind / jump / hold-wr) is `evil_set_`*.


| API                                            | Visibility | Who may use it                                                                                                                              |
| ---------------------------------------------- | ---------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| `get_read_ptr()` / `get_write_ptr()`           | **public** | Any kernel that needs the **current** FIFO cursor address — local entry L1 access, or interop with helpers that still take a raw L1 address |
| `evil_set_read_ptr()` / `evil_set_write_ptr()` | **public** | WH/BH only — kernels that **mutate** cursors (save/restore, rewind, jump). Never `LocalCBInterface` field writes. Not available on Quasar   |



| Kernel intent                                           | Correct API                                                                                           | Wrong                                                                      |
| ------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| Tile/stick transfer after `reserve` / `wait`            | `Noc::async_read` / `async_write` from `**noc.h`**, with DFB endpoints + `offset_bytes` when possible | Prefer not: leftover `get_*_ptr` + `noc_async_`* from `**dataflow_api.h`** |
| Peek current entry for local L1 access / helper interop | public `get_read_ptr` / `get_write_ptr`                                                               | inventing `evil_get_`*, or using `evil_set_`* for a peek                   |
| Cursor surgery (mutate FIFO `fifo_*_ptr`)               | `evil_set_*` (snapshot first with public `get_*_ptr` if needed)                                       | `LocalCBInterface` field R/W                                               |


```cpp
// ✅ Prefer noc.h for transfers (no leftover peek)
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

// ❌ Avoid on a Metal 2.0 / Device 2.0 port
dfb.reserve_back(1);
uint32_t addr = dfb.get_write_ptr();
noc_async_write(/*...*/, addr + offset, /*...*/);  // dataflow_api.h style
dfb.push_back(1);
```

Kernels do **not** need `evil_get_`* to snapshot a cursor before surgery — use public `get_*_ptr`, then `evil_set_`* to put the cursor back (or to a computed address).

---

## A. Canonical FIFO


| CB                 | DFB            |
| ------------------ | -------------- |
| `reserve_back`     | `reserve_back` |
| `push_back`        | `push_back`    |
| `wait_front`       | `wait_front`   |
| `pop_front`        | `pop_front`    |
| `finish` (if used) | `finish`       |


### Tile / format metadata (JIT descriptors)

Legacy CB code indexes compile-time arrays from `chlkc_descriptors.h` (e.g. `pack_tile_size[cb_id]`, `unpack_src_format[cb_id]`) or free helpers like `get_tile_size(cb_id)`. On DFB, those become **member getters** on the buffer object (available when `DFB_DESCRIPTORS_DEFINED` — i.e. `chlkc_descriptors.h` is present). PACK TRISC uses `pack_`* arrays; UNPACK/MATH/DM use `unpack_`* (see `dataflow_buffer.h`).


| CB (array / free helper)                                 | DFB member getter                                                 |
| -------------------------------------------------------- | ----------------------------------------------------------------- |
| `*_tile_size[id]` / `get_tile_size(id)`                  | `get_tile_size()`                                                 |
| `*_tile_r_dim[id]` / `*_tile_c_dim[id]`                  | `get_tile_r_dim()` / `get_tile_c_dim()`                           |
| `r_dim * c_dim`                                          | `get_tile_hw()`                                                   |
| `*_tile_num_faces[id]`                                   | `get_tile_num_faces()`                                            |
| `*_tile_face_r_dim[id]`                                  | `get_face_r_dim()`                                                |
| `*_partial_face[id]` / `*_narrow_tile[id]`               | `get_partial_face()` / `get_narrow_tile()`                        |
| `*_num_faces_r_dim[id]` / `*_num_faces_c_dim[id]`        | `get_num_faces_r_dim()` / `get_num_faces_c_dim()`                 |
| `pack_dst_format` / `unpack_src_format` (role-dependent) | `get_dataformat()`                                                |
| `unpack_dst_format`                                      | `get_unpack_dst_format()` (non-PACK)                              |
| `pack_src_format` / `pack_dst_format`                    | `get_pack_src_format()` / `get_pack_dst_format()` (where emitted) |


---

## B. Size / layout queries


| CB / field                                               | DFB                                                      |
| -------------------------------------------------------- | -------------------------------------------------------- |
| `fifo_page_size` / entry size                            | `get_entry_size()` (+ aliases)                           |
| `fifo_num_pages` / capacity                              | `get_total_num_entries()` / local-TC / ring-span getters |
| Total / local / span size in bytes                       | `get_total_size_bytes()`, local/span size APIs           |
| `get_local_cb_interface(...).fifo_page_size` (and peers) | same section **B** getters                               |


**Note:** TRISC size getters return sizes in **bytes**, not 16B units (see #49652 notes).

---

## C. Public cursor peeks + scalar tile reads

### Cursor peeks (not evil)


| CB                | DFB               |
| ----------------- | ----------------- |
| `get_write_ptr()` | `get_write_ptr()` |
| `get_read_ptr()`  | `get_read_ptr()`  |


Prefer `noc.h` + DFB for transfers so leftover peeks are not required; keep `get_*_ptr` when the kernel truly needs the entry address (local L1 access, or helpers that still take raw L1).

### Scalar tile reads


| CB                                        | DFB                            |
| ----------------------------------------- | ------------------------------ |
| `read_tile_value` / `read_tile_value_u16` | templated `read_tile_value<T>` |
| `get_tile_address`                        | `get_tile_address`             |
| `get_pointer_to_cb_data`                  | `get_tile_address`             |


---

## D. Cursor surgery (`evil_set_`* only)

**Naming:** `evil_` prefix on **DataflowBuffer member setters** that **mutate** FIFO cursors outside canonical `reserve`/`push`/`wait`/`pop`. Greppable and intentionally ugly — not a stable end-state.


| Legacy CB / pattern                                                            | Intended `DataflowBuffer` member                             | Composition / note                                                                                                                                |
| ------------------------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| `get_local_cb_interface(...).fifo_wr_ptr` / `fifo_rd_ptr` assign / `+=` / wrap | `**evil_set_write_ptr` / `evil_set_read_ptr`**               | Kernel keeps arithmetic; setter takes absolute cursor — **never** leave `LocalCBInterface` field writes                                           |
| Snapshot cursor then restore / rewind                                          | `get_*_ptr` then `evil_set_`*                                | Peek is public; only the mutate is evil                                                                                                           |
| Bilinear / manual wr advance without credits                                   | `evil_set_write_ptr`                                         | Credits intentionally skipped                                                                                                                     |
| Packer L1 acc save/restore                                                     | `get_*_ptr` then `evil_set_`*                                | Same DFB object for peek and set                                                                                                                  |
| Gathered-matmul ring `local_cb.fifo_rd_ptr = …` helpers                        | `evil_set_read_ptr` (+ `get_read_ptr` as needed)             | Replace ref-to-interface helpers with DFB members                                                                                                 |
| Bare `get_*_ptr` + `dataflow_api.h` `noc_async_`* for transfers                | **Rewrite** to `Noc::async_`* from `**noc.h`** when possible | Prefer DFB endpoint + `offset_bytes`; leftover peeks are cleanup debt, not evil                                                                   |
| `get_cb_tiles_acked_ptr` / `get_cb_tiles_received_ptr`                         | **no DFB equivalent**                                        | A kernel that uses this API *cannot be ported today*. Do not attempt to work around this. Please prominently note the issue in the porting report |
