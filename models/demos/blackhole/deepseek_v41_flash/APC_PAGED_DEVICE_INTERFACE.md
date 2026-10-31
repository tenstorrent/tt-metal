# INTERFACE_DEVICE.md: device-side stores of the block-paged APC (key store, halo store)

Owner: device agent (branch `ssinghal/dsv4p1-apc-paged-dev`, worktree `/mnt/tt-data/ssinghal/wt/apc_paged_dev`, base `ssinghal/dsv4p1-apc-paged` 7fd4d24b445).
Code: `models/demos/blackhole/deepseek_v41_flash/tt/apc_paged_store.py`, kernels `tt/paged_kernels/{key_save,key_load,halo_copy}.cpp` (JIT data-movement, no ttnn C++ rebuild, no CCL).
Tests: `tests/test_apc_paged_store_cpu.py` (CPU), `tests/test_apc_paged_store_device.py` (device, (4,8) mesh). Status: see the changelog at the end.
Switch: `DSV41_APC_PAGED=1` (default 0) -> `apc_paged_store.enabled()`. Sizes per mesh row: `DSV41_APC_PAGED_HALO_A` (128), `DSV41_APC_PAGED_HALO_B` (48), `DSV41_APC_PAGED_KEYBLOCKS` (2048).
Everything below is per mesh ROW (rows = 4 at B=8): each row's 8 chips hold their own copy; a slot id means the same slot on the 8 columns of a row and is chosen per row. `rows * U` = number of prefill slots (`U` = prefill slots per row = `Up`, 2 at B=8).

## 1. Slot id tables (the only runtime input of the kernels)

Per-replay slot ids travel in small tensors, refreshed with `h2d` (recordable in the prefetch thread like `kofs`; the kernels read them at execution time, so they work inside a captured trace).

| table | shape (global) / per device | dtype, layout | content |
|---|---|---|---|
| `KeyStore.sid` | `[rows*U, 32]` / `[U, 32]` | uint32 RM, sharded over mesh rows | element 0 of row `r*U+u` = key store slot to write for slot u of mesh row r |
| `KeyStore.ltab` | `[rows*U, nbmax]` / `[U, nbmax]` (`nbmax` = key blocks, default 2048) | uint32 RM | row `r*U+u` = key store slots of chunk blocks 0,1,2,.. of the cached prefix of that slot; the list ends at the first SKIP |
| `HaloStore.save_tab[part]`, `restore_tab[part]` | `[rows*U, 32]` / `[U, 32]` | uint32 RM | element 0 = halo slot of that part to write / to read |

`SKIP = 0xFFFFFFFF` everywhere; host-side `None` or any negative id (the BlockIndex `-1`) become SKIP. Row index = `mesh_row * U + slot` (flat list of `rows*U` entries). A SKIP entry leaves the user's data and the stores untouched. Ids >= capacity are also skipped by the kernels (no write out of the store).
Helpers: `set_table(md, dev, vals, width)`, `KeyStore.set_save(vals)`, `KeyStore.set_load(vals)`, `HaloStore.set_save(part, vals)`, `HaloStore.set_restore(part, vals)`; `vals` are plain Python lists of ints/None (lists of ints for `ltab`). They use `h2d` (call from the model's prefetch thread inside `h2d.recording()` or directly on the main thread).

Slot id spaces are separate: key slots `0..K-1`, halo A `0..KA-1`, halo B `0..KB-1`; the ids are exactly the ones of `BlockIndex` (`cp.key_slot`, `cp.halo_a`, `cp.halo_b`, `plan.key_slots`, `plan.halo_a`, `plan.halo_b`).

## 2. Key store (`KeyStore`)

- Constructor `KeyStore(md, U, owners: {name: ratio}, chunk=512, K=None, nbmax=None)`. One tensor per index-key owner `store[name]` = per device `[K, 1, nrb*32, 128]` bf16 TILE zeros (ratio 2: `nrb`=1 -> 8 KiB per block; ratio 1: `nrb`=2 -> 16 KiB; the 4 owners = **40 KiB per chip per 512-token block**, K=2048 -> 80 MiB/chip = 10 MiB/bank). Slot k of a store = tiles `k*NP .. (k+1)*NP-1` (NP = nrb*4, tile order: row block, then the 4 tile columns of 32 features).
- Content of a block = the chunk's 512/ratio entries of this chip's mesh COLUMN (column-sharded exactly as the single-trace slab: global slot p on column p % 8), i.e. the scratch rows `[0, Cc8)` of the slab after `key_append` at offset 0. No cross-column traffic.
- `ks.save(slabs)`: `slabs = {name: key slab tensor}` (the persistent `PrefillSparse.keys` of the owner layers, `[U,1,T,128]` bf16 TILE). Copies scratch rows `[0, Cc8)` of slot u to store slot `sid[u]`. Run AFTER the layer's `add_keys_single` scratch write of the chunk, i.e. in the trace right after the owner layer's `key_append(kzero)` (or at the end of the chunk trace: the scratch rows are not rewritten before the next replay, but the OTHER layers' ops do not touch them; the owner layers are only 2, 8, 14, 20 so the end of the trace is fine). Never the history rows: block data are taken from scratch, so the save is independent of `kofs`.
- `ks.load(slabs)`: for slot u, chunk block `i` of `ltab[u]` is written to slab history row blocks `[nrb*(1+i), nrb*(2+i))` (= the `kofs` of a chunk starting at `i*512`; history entry j at slot Cc+j). Stops at the first SKIP and at the slab capacity. One launch per owner (4), each one core per user. Run BEFORE the first replay of the hit (eager, or in the trace at its beginning), the scratch rows and the rest of the history are untouched.
- Low-level: `key_save(slab, sid, store, users, nrb)`, `key_load(store, ltab, slab, users, nrb)`.
- No aliasing: the store is a separate tensor; save only reads the slab; load only writes history rows of users listed in `ltab` (verified by the device test).

## 3. Halo store (`HaloStore`)

- Constructor `HaloStore(md, U, halo_a=None, halo_b=None)`. Part A = layers 0..20 (`PART_A`), part B = layers 21..39 (`PART_B`). `store[part]` = per device `[K_part * NL_part, 1, 128, 512]` bf16 TILE zeros; layer index j (position in the part) of slot k = tensor rows `[(k*NL + j)*128 ...)` (64 tiles = 128 KiB). Slot = 2.625 MiB/chip (A) or 2.375 MiB/chip (B). Defaults 128 A + 48 B slots = 450 MiB/chip. `halo_b=0` allocates no B store.
- The halos are the persistent per-layer tensors of `PrefillAttention` (`pa.halo`, `[U,1,128,512]` bf16 TILE, allocated in `alloc_dyn`, REPLICATED mapper, updated in place by the trace). The model passes them as `halos = {layer: tensor}` (or a 40-list): `hs.save(part, halos)` / `hs.restore(part, halos)`.
- `save`: store slot `save_tab[part][u]`, layer j <- `halo_j[u]` (all layers of the part, one launch of 8 cores per part). Call at the END of a chunk trace (after the last layer's in-place halo update), so the saved halo is the state AFTER the chunk = the boundary `s_end`. For part B the host sets SKIP for chunk ends outside the bounded-replay zone (halo of layers >= 21 not updated there).
- `restore`: `halo_j[u]` <- store slot `restore_tab[part][u]`, IN PLACE (the buffer address of every halo stays the same: asserted by the device test). Eager, before the hit's first replay, device local; or at the start of the trace. A restore of part A only leaves the layers 21..39 halos untouched (valid use: `s <= p0` rule; the owner of the block index decides).
- Low-level: `halo_copy(tab, store, halos, users, 'save'|'restore')`.
- A user's halo is only touched when its table entry is not SKIP: other slots of the replay keep their state (per-(row, slot) masks).

## 4. Allocation timing, trace and thread constraints

- ALL tensors (stores, tables) must be created at model build before the first trace capture/replay and AFTER the existing persistent tensors of the single-trace prefill (`alloc_dyn` of every layer, `SingleCtx.ensure`): they are allocated from the same DRAM allocator, so they will shift later allocations; keep them in one place (`DSV41Model` build, right after `setup_dyn`), under `memitem(md, "apc_paged_store")`. They are never reallocated or `deallocate`d, and written only in place (kernels write in place, tables via `h2d`).
- The kernels are `ttnn.generic_op` programs without outputs; buffer addresses travel as runtime args, the program hash depends only on shapes. First call compiles (JIT): warm every program (save and restore of both parts, key save/load for the 2 shapes) once, eagerly, with all-SKIP tables BEFORE capture (`KeyStore.warm(slabs)`, `HaloStore.warm(halos)`).
- Main thread only for launches (`generic_op`); table refreshes with `h2d` may run in the prefetch thread under `h2d.recording()` (the recorded copies are replayed by the main thread before the replay, as for `kofs`).
- No synchronize inside a capture. Device ops are in order on CQ 0: an eager restore enqueued before the first replay needs no extra sync; `BlockIndex.finish_chunk` may be called as soon as the replay is enqueued.
- No CCL, no fabric; SPMD: the same program on all 32 devices with per-device data.

## 5. What the model/other agents need to wire (M1b, not done by this branch)

1. Build: `KeyStore(md, Up, {layer: ratio for owners 2, 8, 14, 20})` and `HaloStore(md, Up)` once; `zero all tables = SKIP`.
2. Per replay before enqueue: `set_save(sid)` with `cp.key_slot` / `cp.halo_a` / `cp.halo_b` of the chunk of each (row, slot) (`-1` for slots not saving); the trace ends with `ks.save(owner slabs)`, `hs.save("A", halos)`, `hs.save("B", halos)`.
3. Hit: `set_load(plan.key_slots)`, `set_restore("A", plan.halo_a)`, `set_restore("B", plan.halo_b)`; enqueue `ks.load`, `hs.restore("A")`, `hs.restore("B")` before the user's first replay; then `idx.restored(req)`.
4. The user's page table already starts with the shared pages (pool is not touched by this module).

## Changelog
- 2026-10-31 02:40 UTC: first version (code written, CPU tests green, device tests pending a free host).
