<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Frame-table patch for `sm_reader.cpp`, `sm_append.cpp`, `scan_reader.cpp` (for their owner to apply)

State: 2026-11-20 00:02 UTC, against the files as they are at HEAD 1a0b0e323a5. Nothing in `fused/sparse_mla/` or
`fused/indexer_scan/` was edited. The replacements below are kept as data in `fused/kv_store/patch.py` (each old text
must occur exactly once in the owner's current file; `tests/test_patch.py::test_patch_applies` fails when a file
moved away from them), and the diffs in section 2 are generated from it.

## 0. What changes

Today the three kernels derive the slot from the batch row (`u * SLOT_STRIDE`, `slots[user]`) and address one
contiguous slot per (user, layer). With the patch each of them asks ONE function where a stripe row is:

    #include "kv_frames.hpp"                                    // fused/kv_store/kernels
    kv_frames::Loc at = kv_frames::locate(table, slot_bytes, row, ROW_STRIDE);   // -> at.bank, at.offset
    get_noc_addr_from_bank_id<true>(at.bank, buffer_address + at.offset)

- `table`: the frame bases of the conversation the decode row serves: `TABLE_FRAMES` = 128 uint32, entry k = byte
  offset inside a DRAM bank of the layer-0 shard of the frame that holds stripe rows [256 k, 256 k + 256);
  `UNMAPPED` (0xFFFFFFFF) where the conversation has no frame. Written by the host from
  `KvStore.table(geometry)` (`fused/kv_store/manager.py`). One table per decode row serves every layer, bank, chip.
- `slot_bytes` = `layer * geometry.slot_bytes(1)`: the layer's (rank's) slot step inside a bank.
- `ROW_STRIDE` = 704 for a packed latent row; 0 gives the 32-row index-key block that holds the row.

The same patched kernel serves both layouts, the geometry is two numbers on the host (`frames.Geometry`):

| storage | tensor per chip | `slot_bytes(1)` | table entry of frame f |
|---|---|---|---|
| today's slot-major cache, capacity C frames (step 1) | `[users * layers, 1, C * 256, w]` | `C * shard_bytes` | identity: `(user * layers * C + k) * shard_bytes` (`Geometry.contiguous(kind, C).identity_table(user)`) |
| frame-major pool (KV_STORE.md 4) | `[frames * layers, 1, 256, w]` | `shard_bytes` | `f * layers * shard_bytes` (`Geometry.pool(kind).frame_base(f)`) |

`shard_bytes` = 22,528 (latent) / 4,352 (index keys). So the patch can go in BEFORE the storage is re-allocated:
with identity tables the patched kernels read and write the very bytes they do today (section 4).

What stays as it is: the selection ids (stripe rows of the conversation, `row(p) = (p // 4096) * 128 + p % 128`),
the append's chip / row arithmetic, the scan's visible-row arithmetic, every circular buffer the kernels push or
pop, every wait. The patch changes addresses only; the scan reader additionally cuts its packets at frame
boundaries (2.3).

## 1. The table as an L1 tensor, and how it is passed

`fused/kv_store/device.py` `FrameTables(device, cores, rows, tables=1)`: an L1 tensor, HEIGHT_SHARDED over `cores`,
`[cores * rows, 128]` uint32 ROW_MAJOR; every core holds the same `rows` x 128 words (512 B per decode row: 4 KiB
for 8 rows, 8 KiB for 16). A kernel reaches it through a circular buffer laid over the tensor
(`FrameTables.cb(index)` = `ttnn.cb_descriptor_from_sharded_tensor`); `get_read_ptr(cb)` is row 0. Nothing is
pushed to or popped from that buffer (no producer, no consumer: the one-producer rule does not come into it).

- Passed as: one named compile-time argument (the buffer id) plus the tensor in the op's io list. No runtime
  argument, no DRAM read at launch, nothing per pass. A layer program that keeps an L1 arena can equally give the
  kernel the table's L1 address and cast it; the function takes any pointer to uint32.
- Updated by the host with `FrameTables.write(store.table(geometry))` when `KvStore.table_version` changed (a
  request starts or ends on a row, or a bound conversation gains a frame; not every pass). Measured on one chip:
  0.22 ms for the 16 row-reader cores, 0.32 ms for 64 scan cores (log `mk10_patch_board_sm`); for 32 chips that
  is about 7 to 10 ms per change if the chips are written one after the other (estimate, not measured).
- Under a trace: the tensor's address is fixed, the host rewrites its content between replays like the step and
  selection tensors.
- KV_STORE.md 4 proposed a DRAM tensor read at every launch instead; with the L1 tensor there is no read at all.
  If the host write of 32 chips x 80 cores turns out too slow for the engine, a small program of ours can copy a
  DRAM binding tensor into the L1 tensors (not built).

## 2. The kernels: exact changes

Line counts (removed, added): `sm_reader.cpp` 11 / 11, `sm_append.cpp` 15 / 16, `scan_reader.cpp` 6 / 22.
All three add `fused/kv_store/kernels` to `compiler_include_paths`.

### `fused/sparse_mla/kernels/sm_reader.cpp`

```diff
--- a/sparse_mla/kernels/sm_reader.cpp
+++ b/sparse_mla/kernels/sm_reader.cpp
@@ -22,2 +22,3 @@
 #include "sm_common.hpp"
+#include "kv_frames.hpp"

@@ -38,6 +39,5 @@
     constexpr uint32_t SCALE_BLOCKS = get_named_compile_time_arg_val("scale_blocks");
-    constexpr uint32_t ROWS_PER_SLOT = get_named_compile_time_arg_val("rows_per_slot");
-    constexpr uint32_t SLOT_STRIDE = get_named_compile_time_arg_val("slot_stride");
-    constexpr uint32_t BANKS = get_named_compile_time_arg_val("banks");
-    constexpr uint32_t SHARD_ROWS = get_named_compile_time_arg_val("shard_rows");
+    constexpr uint32_t cb_frames = get_named_compile_time_arg_val("cb_frames");    // the frame tables (L1 tensor)
+    constexpr uint32_t SLOT_BYTES = get_named_compile_time_arg_val("slot_bytes");  // bank bytes per layer step
+    static_assert(get_named_compile_time_arg_val("banks") == 8 && get_named_compile_time_arg_val("shard_rows") == 32);
     static_assert(USERS <= sm::MAX_USERS && CHUNK % sm::TILE_ROWS == 0 && STRIDE % 64 == 0);
@@ -61,2 +61,4 @@
     const uint32_t layer = get_common_arg_val<uint32_t>(2);
+    volatile tt_l1_ptr uint32_t* frames = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_frames));
+    const uint32_t layer_bytes = layer * SLOT_BYTES;

@@ -105,3 +107,3 @@

-        const uint32_t page_base = (u * SLOT_STRIDE + layer) * ROWS_PER_SLOT;
+        volatile tt_l1_ptr uint32_t* table = frames + u * kv_frames::TABLE_FRAMES;
         for (uint32_t i = 0; i < own; ++i) {
@@ -114,6 +116,4 @@
             for (uint32_t r = 0; r < rows; ++r) {
-                const uint32_t page = page_base + idx[first + r];
-                const uint64_t src = get_noc_addr_from_bank_id<true>(
-                    sm::kv_row_bank<BANKS, SHARD_ROWS>(page),
-                    kv_addr + sm::kv_row_offset<BANKS, SHARD_ROWS>(page, STRIDE));
+                const kv_frames::Loc at = kv_frames::locate(table, layer_bytes, idx[first + r], STRIDE);
+                const uint64_t src = get_noc_addr_from_bank_id<true>(at.bank, kv_addr + at.offset);
                 noc_async_read_one_packet(src, dst + r * STRIDE, STRIDE);
```

### `fused/sparse_mla/kernels/sm_append.cpp`

```diff
--- a/sparse_mla/kernels/sm_append.cpp
+++ b/sparse_mla/kernels/sm_append.cpp
@@ -18,2 +18,3 @@
 #include "sm_common.hpp"
+#include "kv_frames.hpp"

@@ -25,6 +26,5 @@
     constexpr uint32_t STRIDE = get_named_compile_time_arg_val("row_stride");
-    constexpr uint32_t ROWS_PER_SLOT = get_named_compile_time_arg_val("rows_per_slot");
-    constexpr uint32_t SLOT_STRIDE = get_named_compile_time_arg_val("slot_stride");
-    constexpr uint32_t BANKS = get_named_compile_time_arg_val("banks");
-    constexpr uint32_t SHARD_ROWS = get_named_compile_time_arg_val("shard_rows");
+    constexpr uint32_t cb_frames = get_named_compile_time_arg_val("cb_frames");    // the frame tables (L1 tensor)
+    constexpr uint32_t SLOT_BYTES = get_named_compile_time_arg_val("slot_bytes");  // bank bytes per layer step
+    static_assert(get_named_compile_time_arg_val("banks") == 8 && get_named_compile_time_arg_val("shard_rows") == 32);
     constexpr uint32_t BLOCK = get_named_compile_time_arg_val("stripe_block");
@@ -41,2 +41,4 @@
     const uint32_t layer = get_common_arg_val<uint32_t>(3);
+    volatile tt_l1_ptr uint32_t* frames = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_frames));
+    const uint32_t layer_bytes = layer * SLOT_BYTES;

@@ -52,3 +54,3 @@
     const uint32_t scratch = get_write_ptr(cb_row);
-    uint32_t page_of[sm::MAX_USERS];
+    kv_frames::Loc at_of[sm::MAX_USERS];
     uint32_t mine = 0;
@@ -57,7 +59,8 @@
         const uint32_t row = (position / (BLOCK * CHIPS)) * BLOCK + position % BLOCK;
-        if ((position / BLOCK) % CHIPS != chip || row >= ROWS_PER_SLOT) {
-            continue;
+        volatile tt_l1_ptr uint32_t* table = frames + u * kv_frames::TABLE_FRAMES;
+        if ((position / BLOCK) % CHIPS != chip || !kv_frames::mapped(table, row)) {
+            continue;  // another chip's position, or beyond the conversation's frames
         }
         noc_async_read(rows.get_noc_addr(u), scratch + mine * STRIDE, ROW_BYTES);
-        page_of[mine++] = (u * SLOT_STRIDE + layer) * ROWS_PER_SLOT + row;
+        at_of[mine++] = kv_frames::locate(table, layer_bytes, row, STRIDE);
     }
@@ -65,5 +68,3 @@
     for (uint32_t i = 0; i < mine; ++i) {
-        const uint64_t dst = get_noc_addr_from_bank_id<true>(
-            sm::kv_row_bank<BANKS, SHARD_ROWS>(page_of[i]),
-            kv_addr + sm::kv_row_offset<BANKS, SHARD_ROWS>(page_of[i], STRIDE));
+        const uint64_t dst = get_noc_addr_from_bank_id<true>(at_of[i].bank, kv_addr + at_of[i].offset);
         noc_async_write(scratch + i * STRIDE, dst, ROW_BYTES);
```

### `fused/indexer_scan/kernels/scan_reader.cpp`

```diff
--- a/indexer_scan/kernels/scan_reader.cpp
+++ b/indexer_scan/kernels/scan_reader.cpp
@@ -20,2 +20,3 @@
 #include "scan_common.hpp"
+#include "kv_frames.hpp"

@@ -40,2 +41,4 @@
 constexpr uint32_t k_capacity_blocks = get_compile_time_arg_val(9);  // key blocks cb_k holds
+constexpr uint32_t cb_frames = 7;  // the frame tables of fused/kv_store (an L1 tensor); free in scan_common.hpp
+static_assert(stream_depth > 0, "the paged key stream is the pipelined one");
 // Measurement only: push the key pages without reading them from DRAM.
@@ -60,3 +63,3 @@
     const uint32_t bank = get_arg_val<uint32_t>(1);
-    const uint32_t slot_offset = get_arg_val<uint32_t>(2);  // bytes from the buffer address to this slot, in a bank
+    const uint32_t slot_offset = get_arg_val<uint32_t>(2);  // this layer's slot step in a bank (Geometry.slot_bytes(rank))
     const uint32_t chip = get_arg_val<uint32_t>(3);         // this chip's index in the block-cyclic layout
@@ -67,2 +70,5 @@
     const uint32_t stream_vc = get_arg_val<uint32_t>(8);  // response virtual channel of the pipelined key stream
+    // frame bases of this core's user: block j of this bank's stream is the bank's one shard of frame j
+    volatile tt_l1_ptr uint32_t* table =
+        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_frames)) + user * kv_frames::TABLE_FRAMES;

@@ -120,3 +126,13 @@
         }
-        const uint32_t blocks = min_u32(blocks_per_read, padded_blocks - issued);
+        // a packet holds frames that lie back to back in this bank, and does not run over the end of cb_k
+        uint32_t blocks = min_u32(blocks_per_read, padded_blocks - issued);
+        blocks = min_u32(blocks, (k_end - k_write) / key_block_bytes);
+        if (issued < num_blocks) {
+            uint32_t run = 1;
+            while (run < blocks && issued + run < num_blocks &&
+                   table[issued + run] == table[issued] + run * key_block_bytes) {
+                ++run;
+            }
+            blocks = run;
+        }
         if (!cb_pages_reservable_at_back(cb_k, (issued - landed + blocks) * key_block_tiles)) {
@@ -136,3 +152,3 @@
             noc_async_read_one_packet_with_state_with_trid(
-                static_cast<uint32_t>(bank_base), issued * key_block_bytes, k_write, slot + 1);
+                static_cast<uint32_t>(bank_base), table[issued], k_write, slot + 1);
         }
```

Notes on the three.

2.1 `sm_reader.cpp`. `ROWS_PER_SLOT` and `SLOT_STRIDE` are gone from the kernel; `banks` and `shard_rows` are only
asserted (8 and 32: `kv_frames` is written for them). `idx[first + r]` is the selection's stripe row as before.
The reader does not check `UNMAPPED`: the selection only names rows below the context, and the manager maps the
frames of the context before the row is bound (`KvStore.mapped_tokens(row)`).

2.2 `sm_append.cpp`. "Beyond the cache" (`row >= ROWS_PER_SLOT`) becomes "beyond the conversation's frames"
(`!kv_frames::mapped(table, row)`), so an append that would need a new frame is skipped until the host has
reserved it: the engine must call `KvStore.reserve(key, tokens)` for the positions a pass may write (or reserve
prompt + max output at `begin`, as the simulator does) and rewrite the table. The append core needs the table too
(one more `FrameTables` core, or the readers' table tensor extended over it).

2.3 `scan_reader.cpp`. Runtime argument 2 keeps its place and its use (`bank_base = bank address + key_addr +
slot_offset`) but is now the layer's slot step, `geometry.slot_bytes(rank)`, with the user's part in the table.
Block j of a (user, bank) stream is the bank's ONE shard of frame j, at `table[j]`, so the packet read takes
`table[issued]` instead of `issued * key_block_bytes`. A packet may only span frames that lie back to back in the
bank: `issue_packet` cuts it at the first frame that does not (and at the end of `cb_k`, which fixed-size packets
never crossed but packets of mixed size can). With identity tables on today's cache every packet is 3 blocks as
today; in a frame-major pool consecutive frames are 21 slots apart, so every packet is 1 block. The blocks pushed
to `cb_k` are `padded_blocks` in every case (`patch.scan_packets` is the host mirror the test checks before the
board run: the compute kernel is never left waiting). The non-pipelined path (`stream_depth == 0`) is not paged:
a static_assert says so. The buffer id 7 is free in `scan_common.hpp`; the constant can move there.

## 3. The Python side

`fused/sparse_mla/op.py` (as in the working tree at 00:00 UTC; names, not line numbers, since the file is being
changed):
- `R_ROWS .. R_KOUT = range(9)`: add `R_FRAMES = 9` (free on the reader cores).
- `SparseMla.__init__`: take the geometry (`frames.Geometry`) and a `FrameTables` over the 16 reader-column cores
  (`self.readers`) with `rows = cfg.users`; one more over the append core.
- `_reader_named()`: replace `("rows_per_slot", cfg.rows_per_slot)` and `("slot_stride", cfg.num_layers)` by
  `("cb_frames", R_FRAMES)` and `("slot_bytes", geometry.slot_bytes(1))` (the other kernels that receive this list
  do not read the two removed names: `sm_convert.cpp`, `sm_dump.cpp`, `sm_attn*.cpp`, `sm_mcast.cpp` were grepped).
- `_reader_cbs()`: append `tables.cb(R_FRAMES)`.
- `_reader_kernels()` / `read_only_program()`: `compiler_include_paths=[.../fused/kv_store/kernels]` on the
  `sm_reader.cpp` descriptor; the table tensor joins every `ttnn.generic_op([...])` io list that runs the reader.
- `check_kv()`: `tuple(kv.shape)[1:] == (1, cfg.rows_per_slot, cfg.row_bytes)` becomes
  `kv_store.device.check_layout(kv, geometry, n)` (rows per slot a multiple of 256, shard bytes 22,528).
- `append_program()`: the same two named arguments (`("cb_frames", 2)`), `append_tables.cb(2)` in `cbs`, the
  include path, the tensor in `append()`'s io list.
- `fused/sparse_mla/layout.py`: `SparseMlaConfig.rows_per_slot` keeps its meaning for the selection (the bound of
  a stripe row: 32,768 for 1M); `kv_row_location` stays for today's layout, `kv_store.frames.locate` is its paged
  form (tests that count banks and pages).

`fused/indexer_scan/indexer_scan.py`:
- `Scan.__init__(..., slots, ...)`: `slots[u]` (today `user * 21 + rank`) becomes the layer's `rank` plus the
  geometry and a `FrameTables` over `self.cores` (64 scan cores, `rows = cfg.users`; a core reads its own user's
  row, `table = base + user * 128`). The shape assert of line 303 becomes `check_layout`.
- `append_to()` line 458: `slot_offset = self.slots[user] * (cfg.bank_blocks * KEY_BLOCK_BYTES)` becomes
  `slot_offset = self.geometry.slot_bytes(self.rank)` (the same for every user).
- `append_to()`: `cbs.append(self.frames.cb(7))`; the reader's `KernelDescriptor` gets the kv_store include path;
  `_io()` returns the table tensor too.
- `ScanConfig.stripe_rows` keeps its meaning (the bound of visible rows and of the score buffers).
  `blocks_per_read` may stay 3: the reader cuts packets itself.

Total on the Python side: about 15 changed lines in `op.py`, about 10 in `indexer_scan.py` (estimate; I did not
write them into the owners' files). `tests/test_patch.py` shows every one of these steps done from outside (the
`swap()` helper and the two `FrameTables`).

## 4. What was verified, and how

`tests/test_patch.py`: patched COPIES (`kv_store/patched/`, written by `patch.patched_sources()`, not committed)
are swapped into the program descriptors the owners' own Python builds; everything else is the owners' code.
One chip, logs `mk10_patch_mock` (compiled), `mk10_patch_board_sm`, `mk10_patch_board_scan`.

| kernel | case | result |
|---|---|---|
| `sm_reader` | `read_rows` (reader + converter + dump) of 2,067 selected rows, counts 1 / 37 / 512 / 1024 / 0 / 64 / 129 / 300, today's storage (8 users x 2 layers x 1,024 rows), identity tables | output tensor bit-equal to the unpatched reader's; traced 47.3 us against 45.7 us |
| `sm_reader` | the same rows in a frame-major pool, the 32 frames scattered (user 0: 26, 17, 11, 10) | bit-equal again; 47.2 us |
| `sm_append` | twin storages, unpatched and patched append of 8 rows (5 on this chip, 2 on others, 1 beyond the frames) | storages byte-equal, 5 rows changed |
| `scan_reader` | tree scan, 8 users x 1,024 stripe rows, ragged contexts; identity tables, and a pool with scattered frames | 4,796 valid fused words equal to the unpatched scan; 30.8 us unpatched, 30.7 us identity, 32.7 us pool |
| `scan_reader` | 8 users x 8,192 stripe rows, the AgentX contexts capped at 262,144 | 15,744 words equal; 42.3 us unpatched, 42.1 us identity, 41.6 us pool (packets of 1 block, 4 in flight), 41.4 us (8 in flight) |

Not verified: the patch inside `attend_program` and the layer program (only `read_rows`, the append and the scan
on their own), 16 decode rows, more than one chip, and the Python edits of section 3 as edits.

## 5. What it costs

- Per row access (measured, `tests/test_frames_device.py::test_cost_per_row_access`, log `mk10_frames_board`):
  22.9 ns per address through the table against 21.3 ns through `sm::kv_row_bank / kv_row_offset` on one
  data-movement core, loop and checksum included: +1.6 ns per row (one L1 load in place of two multiplies).
  For a pass that reads 64 rows per user on a chip that is 0.8 us of processor time spread over the readers of 8
  users; `read_rows` of 2,067 rows measured +1.6 us on the traced program (47.3 against 45.7 us), which also
  holds whatever the extra buffer costs at launch.
- Append: one lookup per row written.
- Scan: one L1 load per packet in place of a multiply. Packets of 1 block (frame-major pool) measured no slower
  than packets of 3 at the AgentX-like shape (41.6 against 42.3 us), +1.9 us at 1,024 rows; KV_STORE.md 4
  expected a cost here and suggested 24,576-token frames to avoid it: not needed.
- L1: 512 B per decode row on every core that holds the table (4 KiB at 8 rows, 8 KiB at 16).
- Launch: nothing read, no runtime argument added; one more buffer configuration on the cores.
- Per additional decode row: 512 B of L1 per table core and 128 words of host write per binding change; nothing in
  the kernels' time. The table holds 128 frames per row (1,048,576 tokens); `TABLE_FRAMES` is one constant.

## 6. Order of application (suggestion)

1. Apply section 2 and 3 with `Geometry.contiguous(kind, capacity_frames)` and identity tables on today's
   storage: every existing golden must hold bit for bit (it did for the three programs above).
2. Allocate the storage frame-major (`KvStore.pool_shape(kind)`: the prefill stack's allocator with
   `seq_len = 8192, num_users = frames, bank_aligned = True`) and switch the geometry to `Geometry.pool(kind)`;
   tables from `KvStore.table(...)`.
3. Bind rows through `KvStore.bind / unbind`; rewrite the tables when `table_version` changes.
