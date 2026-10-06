# TensorAccessor walker on the Quasar address generator: how it works, what it costs, and the redesign

Status:

- Sections 1–5 describe the original walker (records, LRU, parking), as measured. It has been replaced.
- Section 6 is the redesign. Step 1 (6.7) is implemented in `tensor_accessor_addrgen.h` on `abhullar/addr-gen-tests`:
  compile-checked for interleaved and sharded kernels, not yet run on the emulator. Step 2 (push) is not started.
- Section 7 lists what still has to be measured or decided.
- Cycle numbers are measured on emu-quasar-2x3 with `rdcycle` (`AddrgenRawPerf` and `TensorAccessorAddrgenPerf` in
  `tests/tt_metal/tt_metal/data_movement/quasar_examples/quasar_addrgen/test_tensor_accessor_addrgen.cpp`).

Related: `addrgen_compiler_interface.md` (what the compiler could take over), `overlay/addrgen_api.hpp` and
`overlay/addrgen_state.hpp` (the address-generator primitives), `address_generators.md` (hardware spec excerpt).

## 1. Purpose

The NoC API asks one question per transfer: **what is the NoC address of page `p` of tensor `T` (+ `offset` bytes)?**
The software answer is `T.get_noc_addr(p, offset)`: bank = `p mod banks`, offset in bank = `(p / banks) * page_size`,
and so on.

The walker answers the same question with the hardware address generator. The generator cannot answer "address of
page `p`" directly; it can only produce **the next address in a pattern it was programmed with**. The walker's job is
to bridge the two models:

| What the caller provides | What the hardware provides |
|---|---|
| Random access: any `p`, on any call, for any tensor | A sequence: the next address in a programmed pattern |

Everything in today's walker exists to make a sequence generator look like a random-access function.

## 2. Hardware it uses

- **2 address generators** per DM core, each with a **source side** and a **destination side**: 4 independent sequence
  producers. Today's code calls them **slots**: slots 0 and 1 (source sides) serve reads, slots 2 and 3 (destination
  sides) serve writes.
- Each side has its own bank loop (base, size, skip, order), inner loop and outer loop. The two sides of one generator
  share only the bank shift (`MISC.bank_offset`). DRAM and L1 windows both use shift 26 on the current maps, so a DRAM
  walk and an L1 walk can run on the two sides of one generator.
- **Programming a side:** about 10 register writes. One programming covers a "run" of pages.
- **Pop(n):** returns the current address and advances by `n`. About 6 cycles.
- **Push:** writes the current address into the generator's command buffer (`SRC_ADDR` for the source side,
  `DEST_ADDR` for the destination side) and advances. It does not issue; `issue_cmdbuf` follows. Generator N always
  pushes into command buffer N (the push opcode only names the command buffer, `cmdbuf * 32 + 28`).
- **Save/restore** of a side's position: 3 register reads to save (today each is followed by a fence, see 7.1), about
  10 writes to restore.

Today the address is popped back to the RISC-V and handed to the ordinary NoC V3 calls (`noc_async_read` →
`ncrisc_noc_fast_read`). Nothing writes command-buffer registers.

## 3. State (original walker)

All of it lives in thread-local memory (budget: 400 B, used: ~380 B).

- **`walkers[4]`** (80 B each): one record per tensor walk being tracked. Records are not slots: a record can be
  resident (in a slot) or parked (saved, not in a slot).
  - **Identity:**
    - `key`: the binding id, or the accessor object's address when the accessor was built without a Metal 2.0 binding
      token (legacy runtime-address accessors have no binding id).
    - `dir`: read or write. One tensor read and written in the same kernel is two walks, because reads and writes use
      different sides.
    - `kind`: page ids, shard bases (ShardView) or pages within a shard (`shard_pages()`).
    - `bank_base`, `page_size`: guard against a different accessor reusing a dead one's address as its key.
  - **Position:** `next` is the page id the hardware produces on the next pop; `run_end` is the first page id the
    current programming does not cover.
  - **Stride learning:** `last`, `last_delta`, `has_prev`. The walker watches the gap between consecutive page ids;
    once the same gap repeats (pages 0, 2, 4, …), each pop advances by that gap instead of 1.
  - **Address glue:** `hi_bits`, the ATT window bits ORed onto every pop.
  - **Slot bookkeeping:** `slot1` (which slot holds it, or none), `parked1` (which entry of `parked_pos` holds its saved
    position, or none), `last_use` (LRU stamp, below).
  - **The programming itself,** kept so a parked walk can be restored: strides, ends, bank fields.
- **`slot_walk1[4]`:** for each of the 4 slots, which record occupies it (record index + 1; 0 = empty).
- **`parked_pos[2]`:** the saved hardware position (`BANK_CURRENT`, `INNER_ADDRESS`, `OUTER_ADDRESS`) of up to 2 walks
  that lost their slot but may be resumed.
- **`walker_clock`:** a counter incremented on every transfer and stamped into the record used (`last_use`). When a
  slot is needed, the record with the oldest stamp (least recently used, "LRU") is evicted.

## 4. Per-transfer algorithm (original walker)

`tensor_accessor::transfer_noc_addr(T, p, offset)` (`api/tensor/transfer_noc_addr.h`) →
`tt_addrgen::try_transfer_noc_addr`:

```
1. FIND THE WALK for (T, direction, kind)                                            acquire_walker()
   fast: check the 2 slots of this direction; match key, kind, bank_base, page_size      ~28 cycles
   slow: search all 4 records (both directions, resident or parked)                 acquire_walker_slow()
         - found but parked -> pick a victim slot by LRU, save the victim's position,
                               restore this walk's registers                            +~230 cycles
         - not found        -> claim a free record (or evict the LRU one), take a slot
2. IS THE HARDWARE AT p?
   if new walk, or p < next (behind), or p >= run_end (past the programming):
         SEEK (below): compute p's address in software, program the side so its
               next output is p, write ~10 registers                                    hundreds of cycles
3. STAMP the LRU clock: w.last_use = ++walker_clock                                     (2 + 3: ~30 cycles)
4. POP p                                                                               pop_index()
   fast: p == next and the learned stride repeats -> pop(stride), next = p + stride      ~30 cycles
   slow: p is ahead, or no stride learned yet:
         if p > next, skip ahead in hardware: pop and discard (p - next) addresses
         recompute the gap from the previous call; if it repeated, adopt it as the stride
         update last, last_delta, has_prev, next
   either way the pop goes through a 4-way switch on the run-time slot number           pop_walker() / with_slot()
5. return pop | hi_bits, + offset
```

Why a walk switch costs ~230 cycles: parking itself is free. Switching costs the save of the outgoing walk (3 register
reads, each followed by a fence that waits for all outstanding memory operations) plus the restore of the incoming one
(~10 register writes including a read-modify-write of `MISC`).

**Seek recipes.** A seek programs a side so that its next output is page `p`'s address, and records how many
following pages that programming covers (`run_end`). How depends on where the pages live:

- **Interleaved** (`seek_interleaved`): page `p` is in bank `p % N` at offset `(p / N) * page_size`. That is the
  generator's bank-innermost loop: N banks, inner stride one page, starting at bank `p % N`, offset
  `(p / N) * page_size`. Every later page comes out in order, so one programming covers the rest of the tensor.
- **Sharded, contiguous run** (`seek_single_bank` + `contiguous_run`): pages are consecutive in memory only within a
  shard row. Program one bank, starting at `p`'s address, stride one page, valid to the end of that run. The next run
  needs a new seek.
- **Sharded, cross-bank** (`plan_cross_bank`): when the shards along a row sit in consecutive banks at the same local
  offset, one programming covers a whole band. The inner loop walks one shard's segment, the bank loop moves to the
  next shard's bank, the outer loop moves to the next row.
- **`shard_pages()`** (`WalkKind::ShardPages`): a shard's pages are contiguous in its bank, so one single-bank walk.
- **ShardView** (`WalkKind::ShardBases`): shard bases rotate across banks, so a bank-innermost walk with a stride of
  one shard.

## 5. What it cost per sequential transfer (original walker)

| Step | Cycles | Why |
|---|---|---|
| 1 find | ~28 | Chained thread-local loads: slot byte → record → 4 field compares |
| 2 + 3 seek check + LRU stamp | ~30 | Loads of `next` and `run_end`, compares, a clock load/increment/store |
| 4 pop | ~30 | Loads, compares and stores of the stride state, the slot switch, the pop itself (6) |
| 5 OR + offset | ~2 | |
| **Total** | **~100** | Software: 29 (L1) / ~80 (DRAM). Raw pop: 6 |

The same loop with the walk's state in local variables (registers) costs **17 cycles**. The instruction counts are
small; the cost is dependent loads and stores to the thread-local records on an in-order core.

Other measured costs (cycles per transfer):

| Case | Hardware path | Software |
|---|---|---|
| 3 tensors round-robin (a restore on every transfer) | ~386 | ~86 |
| 5 tensors round-robin (more walks than records: a seek on every transfer) | ~580–680 | ~78 |
| Raw pop + issue read, no walker | 16 | 42 (L1) / ~100 (DRAM) |
| Push + issue read, no walker | 10 | same |

The overhead has one root cause: every call arrives as just `(T, p)`, so each call rediscovers from thread-local
memory which walk this is, where the hardware is and what stride the caller uses. The redesign keeps that state where
the compiler can hold it in registers, and stops using the hardware where it cannot help.

## 6. Redesign

### 6.1 Ground rules

1. **Metal 2.0 only.** Every accessor on this path is built from a binding token, so its binding id is a compile-time
   constant in its type (`DSpec::binding_id`). Accessors without one use software.
2. **Push into the command buffer** instead of popping (step 2). The address goes from the generator straight into the
   command buffer; the RISC-V writes only the local address and length, then issues.
3. **Hardware for streams, software for the rest.** The hardware serves a tensor while it is walked in order or at a
   steady stride. A request that breaks the stream uses software for that transfer, and the hardware is re-taken as
   soon as access continues at the stride again (6.4).
4. **No user-visible change.** Kernels keep `noc.async_read(ta, dfb, size, {.page_id = p})`. The state lives with the
   hardware side (6.3), not in the kernel or the accessor: copies of an accessor (an iterator holds one) can't disagree
   about where the hardware is.

### 6.2 Hardware resources

- **Pairing is fixed; the role is not.** Generator N pushes only into command buffer N. But whether a command buffer
  performs a read or a write is just its `MISC` register (plus the request/response VCs). The NoC V3 write path already
  rewrites `MISC` on every call; the read path relies on command buffer 1 being left in read mode since boot.
- So **either generator can serve a read or a write**, as long as its command buffer is set to that direction before
  the push + issue:
  - read: push the generator's **source** side into `SRC_ADDR` (remote), write `DEST_ADDR` (local);
  - write: push the **destination** side into `DEST_ADDR` (remote), write `SRC_ADDR` (local).
- That gives up to **4 push-capable walks**: at most one read walk and one write walk per generator (its two sides),
  with the two walks on one generator sharing the bank shift.
- **Bank shift per generator.** The bit where the bank number goes (`MISC.bank_offset`) is one field per generator,
  shared by its two sides, and maps can give DRAM and worker windows different shifts (grendel_qsr1: DRAM 33, worker
  24; quasar_aether_2x3: 26 for both). A walk's shift is its memory's window, a compile-time property of the accessor
  (`walk_shift`), so a walk only takes a side whose sibling (the other side of the same generator: 0↔3, 1↔2) is free
  or uses the same shift (`shift_fits`) -- when claiming, reloading, or choosing a side to spill. If neither side of
  its direction fits, the request uses software. (The original walker instead `static_assert`ed equal shifts, so no
  kernel built against grendel_qsr1.)
- Cost of using a command buffer for the other direction: rewriting `MISC` and the VCs, and restoring read mode on
  command buffer 1 afterwards (the software read path assumes it). To measure: 7.2.
- Side order (`tensor_accessor_addrgen.h`): side 0 = generator 1 source (reads), side 1 = generator 0 source (reads),
  side 2 = generator 0 destination (writes), side 3 = generator 1 destination (writes). A direction's first walk gets
  side 0 or 2, the ones that push into the NoC API's existing read and write command buffers without switching `MISC`.
  Under pop (step 1) all four sides are used; under push, sides 1 and 3 need the switch.

### 6.3 State

One `SideState` per hardware side, thread-local at a fixed address, plus one parked walk and one byte per direction:
4 × 64 B + 88 B + 4 B ≈ 348 B (the original walker used ~380 B). The hit path's fields are independent loads at constant
offsets; nothing is reached through another load.

| Field | Meaning |
|---|---|
| `owner` | Walk key on this side: binding id + walk kind, a compile-time constant of the accessor's type; 0 = free |
| `next` | Index the hardware produces on the next pop |
| `stride` | Indices the hardware advances per pop (1 in order, T for a T-thread reader) |
| `run_end` | First index the current programming does not cover (sharded runs) |
| `last` | Index of the previous request |
| `last_gap` | Gap between the last two forward requests; a gap that repeats becomes the stride |
| `miss_gap` | Gap of the last request served in software; a repeat re-takes the hardware |
| `shard`, `last_addr`, `has_base` | `shard_pages()` and ShardView bookkeeping |
| programming (strides, ends, bank fields) | What a reload writes back, together with the saved position |

| Per thread | Meaning |
|---|---|
| `parked[1]` | A spilled walk: its `SideState` and its saved position (`BANK_CURRENT`, `INNER_ADDRESS`, `OUTER_ADDRESS`) |
| `last_side[2]` | Per direction, the side used last. With two sides per direction the other one is the least recently used, so this byte replaces the LRU clock. |

What went away: `walkers[4]`, the identity compares (key/kind/bank_base/page_size), `slot_walk1`, the per-transfer LRU
clock, the run-time 4-way slot switch (each side is a template parameter), and `hi_bits`: the ATT window bits are
the outer loop's start (`outer_start = window.compare`, measured to give pops equal to `get_noc_addr`), so every pop is
the complete NoC address. The outer loop's end sentinel is 2^62 (`kOuterEndSentinel`), above every map's compare bits
(grendel_qsr1 has a window at bit 48), with a `static_assert` over the active map.

`dir` is not stored: it is the NoC call's direction, known at compile time, and it selects the two sides to look at.

### 6.4 Per-transfer algorithm

```
request index i of walk K (binding id + kind, compile-time) in direction Dir:
  find:  side A (first of Dir) owner == K -> serve on A; else side B owner == K -> serve on B   (hit: 1-2 compares)
         else take a side: a free side of Dir, else the least recently used one (the side not used last)
           - the side's walk is spilled: position read back (3 fenced reads), parked with its programming
           - K parked -> reload: programming + position written back, then serve on that side
           - K new    -> claim: seek to i, stride 1
         TT_TA_ADDRGEN_NO_SPILL: no free side -> software (first use is sticky)
  serve on side S:
    i == next, i < run_end                        -> pop(stride)                                 hit
    next < i < run_end, i - next <= 64            -> skip in hardware: pop(i - next), then pop(step)
                                                    step = the request gap if it repeated, else 1
    i == next, i >= run_end                       -> re-seek (the next run of a sharded stream)
    otherwise (behind, or a large jump):
      gap = i - last (0 if behind)
      gap == stride or gap == miss_gap            -> re-seek at i with stride gap (the stream continues)
      the walk was streaming                      -> re-seek at i, keeping the stride (next block / column / pass)
      else                                        -> software for this request; miss_gap = gap; last = i
    "streaming": the walk's last request was served by its current programming (a hit or a hardware skip)
```

- The hit path is two independent loads and compares, the pop, and two stores. Under push it becomes the command-buffer
  writes, push and issue: about 10–17 cycles measured as "push + issue" and "register state" in section 5.
- A request that breaks the stream costs what software costs, plus a compare. It never re-seeks blindly, so random
  access does not pay hundreds of cycles per transfer, and a stream that resumes is picked up on its next request.
- **Blocked access** (matmul runs of R pages, then a jump of W): a jump of up to 64 pages is skipped in hardware; the
  step stays 1 because the gap didn't repeat, so the run continues on the hit path.
- **Multi-threaded readers and writers** (the DFB tests: thread t of T handles pages t, t+T, t+2T, …): each thread has
  its own generators, so each walk is that thread's own stream. The second request skips ahead, the third finds the gap
  repeated and makes T the stride, and from then on each request is a hit. Under push the advance is
  `PUSH_*_POP_X(cmdbuf, T - 1)`, because a push's skip count is in addition to its own advance (7.4).
  - `TensorAccessor::strided_pages()` / `strided_shard_pages()` already know T (`get_num_threads()`, the iterator's
    `stride_` in `pages_address_iterator.h`), so a later step can pass the stride directly instead of detecting it.
- **`shard_pages()`:** a new shard starts a new run (re-seek); padding pages the iterator skips are skipped in hardware.
- **ShardView:** consecutive transfers into the same shard reuse the base already popped for it.

### 6.5 Spill and reload

Implemented at run time (the default), so the path the compiler would take over is concrete: what is saved, what is
written back, and what it costs.

- **Spill** (`take_side`, `save_side`): read the side's position back (3 register reads, each fenced, 7.1) and park it
  with the walk's `SideState`, which already holds the walk's programming. Spill plus reload measured ~230 cycles with
  the original walker; the split between the two isn't measured.
- **Reload** (`restore_side`): write the programming and the position back (~10 register writes, including a
  read-modify-write of `MISC`), and continue the walk exactly where it stopped: no software seek.
- **Victim:** the least recently used side of the direction (the one not used last). The pool holds one parked walk;
  a walk spilled while it's full, or whose programming doesn't fit the narrowed fields, is forgotten and re-seeks when
  it comes back.
- **What LRU can't know:** whether the evicted tensor comes back. Round-robin over more tensors than sides reloads on
  every transfer (~386 cycles measured with the original walker, against ~86 in software), which is why the decision
  belongs to the compiler: it sees the loop, can keep the right tensors on the sides, place spill and reload at region
  boundaries the way it places register spills (`addrgen_compiler_interface.md`, Asks 1–2), or send the extras to
  software.
- `TT_TA_ADDRGEN_NO_SPILL` selects first use is sticky instead: no spills, the extra walks use software. Both are
  tested (`TensorAccessorAddrgenContention` Spill / NoSpill rows).

### 6.6 Compiler's part

- Choose which tensors get the sides in a loop (instead of first-use sticky), and send the rest to software. A loop that
  round-robins more streams than sides is visible to the compiler, which can pick software for the extras.
- Place spill/reload at region boundaries.
- Pass the stride where it is a compile-time expression, instead of detecting it.

### 6.7 Implementation status

**Step 1 (done, not yet run):** the policy and state above, still popping the address back to the RISC-V and issuing
through the NoC V3 calls.

- `tensor_accessor_addrgen.h`: `SideState`, `parked`, `last_side`; `walk` / `serve` / `take_side` / `reseek`;
  `save_side` / `restore_side` for spill and reload; the seek recipes now return a
  `Seek` (programming + run end) instead of writing a record (`plan_interleaved`, `plan_sharded`, `plan_cross_bank`,
  `plan_single_bank`, `plan_shard_bases`).
- `transfer_noc_addr.h`: `TransferStats` adds `fallbacks` / `write_fallbacks` (requests the policy sent to software),
  counted apart from `sw_ineligible` (banks the recipe can't walk); `restores` counts reloads. The test kernels report
  them as words 10 and 11.
- Tests (`test_tensor_accessor_addrgen.cpp`): single-tensor kernels expect no fallbacks except `Strided` (≤ 2: the jump
  back from the even pages to page 1); contention Spill rows expect each walk seeked once and a reload on every transfer
  after the first three, NoSpill rows the third tensor in software; the mixed suite expects the reads to share two
  source sides by reloading; the raw breakdown lost its sections for the old walker's internals.
- Compile check (offline, real JIT commands, `-Werror`): reader and writer kernels interleaved; sharded in page-id,
  ShardView and strided modes (runtime rank) and `shard_pages()` (static rank); the mixed 3-read/1-write kernel, with
  and without `TT_TA_ADDRGEN_NO_SPILL`.

**Step 1 results (2026-10-05, emu-quasar-2x3):** the addrgen suite passes (446 tests; the rest skip on a 2x3
emulator). Cycles per transfer, address only unless noted; "before" is the original walker with its fast paths:

| Case | Before | Step 1 | Software |
|---|---|---|---|
| Raw kernel, 1 sequential tensor, L1 | ~100–130 | 42 | 44 |
| Raw kernel, 1 sequential tensor, DRAM | ~100–130 | 43 | 117 |
| Benchmark, 1 tensor sequential, L1 | 96 | 66 | 29 |
| Benchmark, 1 tensor sequential, DRAM | 94 | 70 | 80 |
| Benchmark, 1 tensor sequential, DRAM, read + barrier per page | 249 | 227 | 250 |
| Benchmark, 2 tensors round-robin, DRAM | 91 | 69 | 85 |
| Random, L1 | 158 | 104 | 31 |
| Blocked (matmul-like), DRAM | 100 | 128 | 79 |
| 3 tensors round-robin (spill/reload every transfer), DRAM | 388 | 775 | 86 |

(The benchmark kernel costs more per transfer than the raw kernel because its loop dispatches through a lambda over 5
tensors.) Two regressions, fixed in step 1b:

- Spill/reload copied whole walk states three times per reload (evicted walk to the stack, parked walk to the side,
  evicted walk to the pool). Now the side's walk and the parked one swap in place (`swap_walks`), and a claim parks the
  side's walk with one copy.
- Every backward jump of a regular pattern (next block, next column, next pass) cost a software request plus a re-seek.
  Now a walk that was streaming (its last request was served by the current programming) re-seeks right away; only a
  walk whose previous request also missed uses software.

**Step 1b results (2026-10-06, emu-quasar-2x3):** with both fixes, plus learning a stride only from consecutive
requests (a skip doesn't count as streaming, and a hit's gap is its stride, so the hit path stores nothing extra).
Cycles per transfer, 2 KB pages; "orig" is the original walker:

| Case | Address: orig → now (SW) | Read + barrier per page: orig → now (SW) | Batched: now (SW) |
|---|---|---|---|
| Raw kernel, 1 sequential tensor, L1 / DRAM | ~100–130 → 43 / 44 (44 / 119) | | |
| 1 tensor, DRAM | 94 → 80 (81) | 249 → 229 (250) | 70 (90) |
| 2 tensors, DRAM | 91 → 71 (86) | 281 → 239 (262) | 82 (101) |
| Blocked, DRAM | 100 → 91 (79) | 259 → 242 (250) | 85 (91) |
| Transpose-like, DRAM | 108 → 110 (79) | 274 → 258 (251) | 100 (91) |
| 1 tensor, L1 | 96 → 70 (29) | 216 → 192 (164) | 68 (37) |
| Random, DRAM | 149 → 161 (76) | 309 → 315 (255) | 157 (92) |
| 3 tensors, reload every transfer, DRAM | 388 → 640 (86) | 619 → 849 (258) | 694 (106) |

- **DRAM streams win end to end** (1–2 tensors, blocked): 3–9% with a barrier per page, 7–22% batched.
- **L1, 1 tensor loses** because software is unusually cheap on this emulator: 2 L1 banks, a power of two known at
  compile time, so the address math is a shift, a mask and a table load (~29 cycles). The hardware path's cost is the
  same on L1 and DRAM (~43 cycles in the raw kernel), so it wins where software costs more than that. A device with
  many L1 banks, not a power of two, needs a division or multiply-and-shift in software; not measurable on 2x3 (the 9x4
  emulator build could).
- **Transpose-like loses** because the test tensor's columns are only 8 pages: every jump back to the next column is a
  re-seek (48 for 384 transfers), and its few hundred cycles are spread over 8 transfers. A column-major walk is a
  two-level loop the generator supports directly (inner: down a column, wrapping after Ht rows; outer: one page per
  column), so one programming could cover the whole tensor; stride detection can't see the second level, the compiler
  can (Ask 3).
- **Random** stays on software (376 of 384 requests) but pays ~80 cycles for checking the walk first; worth trimming.
- **Where a reload's ~400 cycles go** (instrumented): save the side's position ~75 (3 fenced reads), swap the walk states
  ~85–95, write the programming and position back ~125–155, serve ~115–140 (a hit is ~20: the first pop after
  reprogramming waits ~100 cycles for the generator). About 300 of the 400 is the hardware's switching cost, which
  compiler-placed spills pay too: a side switched between tensors every transfer costs more than software, so
  round-robin over more tensors than sides should go to software (`TT_TA_ADDRGEN_NO_SPILL` behaviour) unless the
  compiler can keep the switches rare.

**Step 1c: hit-path bookkeeping (2026-10-06, emu-quasar-2x3; suite passes).** What the hit path did that it didn't
need to, and what replaced it:

1. Owner and next index are adjacent, so one 64-bit load gets both; likewise stride and run end (`load_pair`).
2. `last` is not stored on a hit: while the walk is streaming the previous request is `next - stride`.
3. `streaming` and `last_side` are written only when they change, and `last_side` not at all without spills.
4. Stats bookkeeping (`PopInfo`) compiles to nothing outside `TT_TA_ADDRGEN_STATS` / `TT_TA_ADDRGEN_TRACE` builds.
5. Every hit condition is marked likely, so the hit is one straight line with no taken branches.
6. The seek planner's inputs (accessor pointer, shard id, NoC id) go through the hit path as plain arguments to a
   stateless planner type, instead of a lambda object the compiler built on the stack for every transfer.

Cycles per transfer, address only unless noted (2 KB pages; "1b" is the previous table):

| Case | 1b | 1–4 | 1–6 | Software |
|---|---|---|---|---|
| Raw kernel, 1 sequential tensor, L1 / DRAM | 43 / 44 | 45 / 47 | 36 / 40 | 44 / 117 |
| 1 tensor, DRAM | 80 | 46 | 40 | 81 |
| 1 tensor, DRAM, batched reads | 70 | 40 | 38 | 90 |
| 2 tensors, DRAM | 71 | 47 | 45 | 86 |
| Blocked, DRAM | 91 | 71 | 63 | 79 |
| Transpose-like, DRAM | 110 | 85 | 78 | 79 |
| 1 tensor, L1 | 70 | 66 | 60 | 29 |
| Random, DRAM | 161 | 166 | 149 | 76 |

(Steps 1–4 cut most of the benchmark kernel's cost; the raw kernel's walker barely moved until 5–6, whose stack stores
and taken branches it shared.) Reload costs are unchanged: round-robin over more tensors than sides stays far behind
software.

**Step 2: push (2026-10-06, emu-quasar-2x3; suite passes).** On a hit on side 0 (reads) or side 2 (writes), the walker pushes the
address into the command buffer instead of popping it: `push_src_pop_x(ADDRGEN_1, stride - 1)` writes read command
buffer 1's `SRC_ADDR`, `push_dest_pop_x(ADDRGEN_0, stride - 1)` write command buffer 0's `DEST_ADDR`. The issue stays in
the NoC V3 calls:

- `ncrisc_noc_fast_read` / `ncrisc_noc_fast_write` take a compile-time `src_in_cmd_buf` / `dest_in_cmd_buf`: skip that
  one register write, do everything else (VCs, local address, length, transaction id, issue, counters).
- `Noc::async_read` / `async_write` and the DFB implicit-sync overloads ask tensor endpoints for
  `src_addr_or_cmd_buf` / `dst_addr_or_cmd_buf` (`noc_traits.h`). The result is either an address (miss, seek, skip,
  side 1/3, non-zero offset: today's path) or `kAddrInCmdBuf`, and then they issue through the flagged V3 call.
- Endpoints: TensorAccessor, PageView, `pages()` and `shard_pages()` pages. ShardView (base plus offset) and the
  type-erased wrapper never push. Stateful APIs (`set_*_state`) never ask for a pushed address.
- Off under the address trace and watcher NoC sanitizing, which need the address in software, and with
  `TT_TA_ADDRGEN_NO_PUSH`.
- Only hits push; the slow path returns the address as before. Measured bound (raw kernel): pop + issue 16.8 cycles
  vs push + issue 9.7 for 64 B pages, the same (~31) for 2 KB pages, where the issue dominates.
- The `*_or_cmd_buf` addresses go through `Noc::get_src_ptr_or_cmd_buf` / `get_dst_ptr_or_cmd_buf`, which emit the
  op-to-op R/W notes like `get_src_ptr` / `get_dst_ptr` (bypassing them dropped the notes from the ELF).
- Tests: the stats report gains `pushes` (word 12). Kernels with one tensor per direction must have
  `pushes + seeks + skips == hw` (every in-order hit pushes); ShardView and wrapper kernels must push nothing; kernels
  with more tensors per direction pop on their second side and on reloads.

Results (cycles per transfer, 1 tensor, DRAM; before → with push):

| Case | 2 KB pages | 64 B pages |
|---|---|---|
| Read + barrier per page | 199 → 196 | 147 → 144 |
| Reads, one barrier | 38.3 → 38.0 | 36.9 → 34.9 |

1–3 cycles per transfer: the walker's bookkeeping (~36–40 cycles per hit) is what remains, not the pop. 254 of 256
reads pushed (the other 2 are the walk's seeks). The L1 rows of this run also moved (1 tensor, address only: 61 → 35)
in a section that never pushes, so that change is code layout in the benchmark kernel, not push; L1 benchmark numbers
move with unrelated code changes and should be read with that in mind.

Sides 1 and 3 keep popping; measure 7.2 before using them for push.

**Per-transfer breakdown with push (2026-10-06, `AddrgenRawPerf`, emu-quasar-2x3).** One sequential interleaved
tensor read into L1, one barrier at the end; cycles per transfer, DRAM 64 B pages (other sizes and L1 agree within a
few cycles):

| Section | Cycles | |
|---|---|---|
| push alone (count-less, or `push_src_pop_x` with a register count) | 0 | the push instruction is free |
| push + DEST_ADDR + LEN + issue | 10 | |
| push + NoC V3 issue (`ncrisc_noc_fast_read<src_in_cmd_buf>`) | 12 | the floor: V3's per-transfer rewrites cost ~2 |
| walker with push, nothing issued | 24 | the walker's bookkeeping |
| walker with pop, nothing issued | 38 | pop also waits for the address to come back |
| walker + push + V3 issue | 40 | floor + ~28 of walker |
| walker + pop + `noc_async_read` (before push) | 41 | in an issue loop, push saves ~0.5 |
| `Noc::async_read(tensor, scratchpad, …)` | 44 | the Noc / traits layer adds ~4 |
| software `get_noc_addr` + `noc_async_read` | 94 (L1: 48) | |

- About 28 of the 44 cycles are the walker's run-time bookkeeping: owner and next-index checks, the stride load, the
  `streaming` / `last_side` flags, the pushed-marker compare. All of it is what static side assignment and walks
  programmed outside the loop remove (`addrgen_compiler_interface.md`, Asks 2–3), so it is left to the compiler rather
  than hand-tuned; the target is the ~12-cycle floor plus the Noc layer.
- Keeping the command buffer's VCs and length across transfers would save ~2 cycles; not worth a context layer for
  that alone.
- The hit path is only fast inlined. With four call sites GCC outlined the push variant of `transfer_noc_addr` and each
  transfer paid ~12 more cycles for the call; the benchmark's walker sections are `flatten` so they measure the inlined
  path. Inlining (or `always_inline`) is also the compiler's part.

## 7. To measure or decide

1. **Fence placement on save.** Today each of the 3 position reads is followed by a fence, because back-to-back
   `rd_reg` hung emu-quasar-2x3. Tried on 2026-10-05 (emu-quasar-2x3):
   - 3 reads back to back, one fence at the end: the 3-tensor contention suite passed (thousands of saves, where the
     save is out of line with other code between the reads), but the first `AddrgenLoopProbe` spill case hung.
   - One fence first, then 3 reads back to back: hung on the first spill case.
   - Each read's result consumed by a register move before the next read, no fences: not measured (the emulator
     did not come up for that run; the hung sessions were still holding it).

   So back-to-back reads are a real hazard, not an emulator quirk of one kernel; keep a fence per read until the HW
   team says what the read needs to wait for. Open question for them: does `rd_reg` need the previous `rd_reg`'s
   response, or all memory operations, to complete?
2. **Cost of switching a command buffer's direction** (`MISC` + VCs), to decide whether to use all 4 sides (6.2).
3. **Where the window bits go under push:** the command buffer's `SRC_BASE`/`DEST_BASE` (the hardware's base, needs
   confirming that plain transfers on that buffer ignore it) or the outer-loop start (measured to work: pops equal
   `get_noc_addr`).
4. **Push skip semantics:** `PUSH_SRC_POP_X(cb, n)` advances `n + 1`, while `pop_x(n)` advances `n` (measured). The
   count-less push builtin advances by 1.
5. **Stateful NoC APIs** (`set_*_state` / `*_with_state`, used by the prefetcher pipe and remote circular buffer) rely
   on command-buffer registers persisting between calls; a push between them would overwrite `DEST_ADDR`. Rule for
   now: no push while a stateful sequence is open on that command buffer.
