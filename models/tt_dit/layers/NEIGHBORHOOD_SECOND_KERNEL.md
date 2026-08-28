# Handoff: second kernel for neighborhood SDPA (interior / edge split)

**Branch:** `wip/na-second-kernel` (off `na-integration`)
**Status:** the split **launches**. Skip of the other brick set **does not fire**. Split is currently *slower* than one mixed kernel. Do not ship this for speed until skip works.
**Hardware:** Blackhole, measured on one device / 1×1 mesh in the component test. Decode numbers are 4×8 DiffVAE stage 5.

This note is the full context for why the second kernel exists, what we measured, how the split is wired, what we already tried, and how to pick it up. Broader NA history (relative mask table, volatile word loops, chunking) lives in [`NEIGHBORHOOD_STRIDE1_FINDINGS.md`](NEIGHBORHOOD_STRIDE1_FINDINGS.md). Terminology is in [`NEIGHBORHOOD_ATTENTION.md`](NEIGHBORHOOD_ATTENTION.md).

---

## 1. What you are being asked to finish

Make stride-1 neighborhood SDPA **skip** the bricks it does not own, so the interior program only tight-gathers interiors and the edge program only classifies edges.

**Success looks like:**

| Signal | Today (skip dead) | Target (skip live) |
|---|---|---|
| `test_neighborhood_sdpa_components.py` window11 | **~530 ms** | **~210–250 ms** |
| Decode `neighborhood-sdpa` n=2 (`DIFFVAE_BLOCK_PROF=1`) | **~737 ms** | **~400–500 ms** (~200–250 each) |
| ns/slot on window11 | ~84 (both programs walk all) | interior ~20 on 73% + edge ~64 on 27% |

If window11 is still ~530 ms, skip did not fire. If it is ~416 ms, you fell back to one mixed kernel. If it is ~130 ms, you are timing interior-only (`path_mode=1`) and edges are wrong. If it is ~52 ms, both programs handshake-only (`if (true)` skip).

Correctness: `test_neighborhood_sdpa.py` must still pass. Split writes the **same** output tensor twice (interior then edge); if the writer skip is also dead, the edge launch **overwrites interiors** with classify garbage.

---

## 2. Why neighborhood SDPA is slow (not softmax)

Stage-5 exact NA (stride `(1,1,1)`, window `11×11×11`) attends each query to a box of keys. After the relative mask table landed, mask *content* is cheap DMA. The remaining wall is the **reader walking every gather slot for every query brick**.

On the real first-band shard that is:

- volume `(84, 272, 480)` sites, W-sharded 8 ways
- owned `(84, 272, 60)`, resident `(84, 272, 72)` (halo)
- brick `(2, 8, 2)` sites = 32 sites = one tile row
- **42840 query bricks**, gather box **147 bricks** at window 11
- host stamp: **31104 interior / 11736 edge** (~73% / ~27%)

So the reader does `42840 × 147` slot iterations per call, unless it skips.

TRISC (QK / softmax / PV) is **~20 ms**. You cannot see that until you skip the slot loop (`probe=7`, `skip_slots`). Drain/qk probes still wait on the reader filling K/V/mask CBs, so they cannot isolate compute.

Decode target that started this work: two stage-5 attention calls at **~200–250 ms each** (~400 ms total). We were seeing **~737 ms** (`n=2`).

---

## 3. Measurements that proved it

All component-test numbers are **host wall-clock**, 3 timed iterations after warmup, one Blackhole device. Overlapping stages do **not** sum. Re-run:

```bash
# from tt-metal repo root; kernel-only edits JIT, host C++ needs -b
./run_na.sh -f models/tt_dit/tests/unit/test_neighborhood_sdpa_components.py -t 240 -s 90
```

Leave `tail -f generated/na_run.log` in another pane. `run_na.sh` kills stale `/dev/tenstorrent/0` holders and truncates that log.

### 3.1 Ablations at window 11 (the bound is the 147-loop)

| probe | Python `probe=` | what the reader does | ms | ns/slot |
|---|---|---|---|---|
| full (`path_mode=0` split, skip dead) | 0 | walk every slot | **528–532** | 84 |
| skip_kv | 1 | no K/V DRAM, still the 147-loop | 26 | 4 |
| skip_slots | 7 | no 147-loop, still reserve/push K/V/mask | **26** | 4 |
| skip_slots_drain | 8 | skip_slots + compute drain | 6–9 | 1 |

How to read it:

```
walk      = window11 - skip_slots        ≈ 505 ms   ← the bound
compute   = skip_slots - skip_slots_drain ≈ 20 ms   ← QK/softmax/PV
handshake = skip_slots_drain              ≈ 6–9 ms
K/V DRAM  = window11 - skip_kv           ≈ 505 ms   (same as walk: DRAM is inside the loop)
```

`skip_kv ≈ skip_slots` means K/V noc is not the extra cost once you still walk slots (classify / tile_offset / mask index). The 147-loop **is** the wall.

Probes 3–6 (drain / qk / softmax / pv) **deadlock** if you add them without keeping the CB handshake. Do not add probes 5/6.

### 3.2 Window sweep (query bricks fixed, gather shrinks)

Same 42840 query bricks, windows 3 / 7 / 11 (gather 27 / 75 / 147). Skip not firing:

| label | gather | ms | us/qbrick | ns/slot |
|---|---|---|---|---|
| window3 | 27 | ~88 | 2.06 | 76 |
| window7 | 75 | ~260 | 6.06 | 81 |
| window11 | 147 | ~532 | 12.4 | 84 |

Slope **~3.7 ms per gather brick**. ns/slot **holds**. Time is walking slots, not Q, origin setup, or dispatch.

### 3.3 One binary vs two: I-cache mix

~73% of bricks are unclamped interiors: DMA the relative-table K/V along a regular raster, no `fill_mask_tile`. ~27% are volume/shard edges: classify, clamp, generate or pick a non-canonical mask.

| binary | loop in the ELF | walks | ms | ns/slot |
|---|---|---|---|---|
| mixed (old default) | tight gather **and** classify | every brick | **~416** | **64** |
| interior-only (`path_mode=1`) | tight gather only | every brick (edges get the interior path — wrong) | **~130** | **20** |
| edge-only (`path_mode=2`) | classify only | every brick | **~400** | 64 |

Interior is **3× faster per slot** when classify is not in the I-cache working set. Sharing a body with `fill_mask_tile` is what made the 147-walk 64 ns/slot (skip_slots is 26 ms for the same handshake + compute).

JIT compile time is a cheap fingerprint of which loop is in the binary: interior reader **~520–530 ms**, edge reader **~700–710 ms**. If both compile in ~520 ms, both built the tight loop.

### 3.4 Split only wins if skip fires

Paper:

```
0.73 × 130 ms   +   0.27 × 400 ms   +   handshake on the rest
≈ 95 + 108 + overhead
≈ 210–250 ms
```

That is the ~200–250 ms per call decode wanted.

**Without skip**, `path_mode=0` launches interior **then** edge, and **each** walks every brick:

```
interior tight-all (~130) + edge classify-all (~400) = ~530 ms
```

Worse than the mixed kernel (~416). **Do not land the split for perf until skip works.**

### 3.5 Skip path is live; the condition is not

| skip condition | window11 | meaning |
|---|---|---|
| none (shipped split) | **528–532 ms** | both walk all |
| `if (true)` in **both** programs | **52 ms** | 2 × skip_slots; handshake only |
| `if (true)` in **interior only** | **459 ms** | interior handshake + edge classifies all |
| `if (gather_time != 0xDEADBEEF)` interior (diagnostic, reverted) | should match 459 if L1 origin[0] is readable | not left in the tree |

52 ms proves `handshake_skip_work_item` + `continue` is compiled and taken. 459 ms proves `if constexpr (compile_interior)` **does** peel the interior program (tight gather DCE'd; interior JIT dropped ~526 → ~429 ms).

The **predicate** (`edge_token == 0xFFFFFFFF`, bit 31 of gather_width, etc.) never matches at runtime, even though:

- Host stamp is correct: `origin stamp: chunks=42840 bit31=31104 col6=31104 col7_edge=11736`
- Device round-trip via `ttnn.to_torch` is identical (`origin page=64`, full row is one Blackhole DRAM page)
- Interior ELF **does** contain `lw` of column 7 and `beq` vs `-1`, or `bltz` on gather_width, depending on the attempt

So: DRAM has the flag. The skip **branch** exists. The value the kernel compares is not the stamped token (wrong pointer, DCE of the compare, or signed/`bltz` on a word that always looks negative).

---

## 4. How the split is supposed to work

### 4.1 Launch

`neighborhood_sdpa_device_operation.cpp`: when `path_mode == 0` (Python default) **and** stride is `(1,1,1)` **and** an interior/relative mask is present **and** not per-brick-mask **and** not `DIFFVAE_NA_TABLE_ALWAYS` **and** `probe == 0`:

```text
launch path_mode=1 (interior) → output tensor
launch path_mode=2 (edge)     → same output tensor
```

Probes disable split so skip_slots/skip_kv time one unsplit reader.

### 4.2 Two TUs, one `.cpp`

Factory picks source by `path_mode`:

| path_mode | reader | writer |
|---|---|---|
| 0 (unsplit / probes) | `neighborhood_reader.cpp` | `neighborhood_writer.cpp` |
| 1 interior | `neighborhood_reader_interior.cpp` | `neighborhood_writer_interior.cpp` |
| 2 edge | `neighborhood_reader_edge.cpp` | `neighborhood_writer_edge.cpp` |

Wrappers are:

```cpp
#define NA_HAS_PATH_SKIP
#include "neighborhood_reader.cpp"
```

`NA_HAS_PATH_SKIP` compiles the skip block. Unsplit probes do not skip bricks.

`compile_interior` selects which **gather loop** is in the ELF:

- `#if NA_PATH_KIND == 1/2` if the factory `-D` actually reaches the preprocessor (it is hashed; whether it is a real `-D` has been flaky — see §6)
- else `interior_table_supported && path_mode != 2`

`path_mode != 2` **does** peel: interior compile ~526 ms vs edge ~704 ms. `if constexpr (path_mode == 1)` was DCE'd (0/1-ish). Prefer compares against **2** (`!= 2`, `== 2`, `skip_kv == 2`).

Interior loop: raster K/V DMA, relative table already in `cb_mask`.
Edge loop: `gather_edge_flash_chunk` / classify / `fill_mask_tile`.

They **must not** both exist in one binary (64 ns/slot).

### 4.3 Host stamp (origin table)

`neighborhood_plan.cpp` sets `use_interior_table_by_chunk[i] = 1` iff the chunk's gather is the canonical relative-table mapping **and** the query brick is unclamped.

`neighborhood_sdpa_nanobind.cpp` writes each 16-word row (`GATHER_ORIGIN_ROW_BYTES = 64`):

| col | name | interior | edge |
|---|---|---|---|
| 0–1 | gather t/h | origin | origin |
| 2 | gather_width | `width \| (1<<31)` | `width` |
| 3–5 | signed shard origin | | |
| 6 | `use_interior_table` | **1** | **0** |
| 7 | `skip_edge_token` | **0** | **`0xFFFFFFFF`** |
| 8–15 | pad | 0 | 0 |

Column 6 is 0/1 — **do not compare it** (DCE). Column 7 is 0 vs all-ones because 0/1 never matched. Bit 31 of col 2 is the same flag riding a word the reader already loads for addressing.

Python already prints the stamp and a device round-trip in `_upload()`.

### 4.4 Skip (reader)

After origin NOC read + `async_read_barrier`, **before** the 147-loop:

- Interior: if `skip_edge_token == 0xFFFFFFFF` → `handshake_skip_work_item` + `continue`
- Edge: if `skip_edge_token != 0xFFFFFFFF` → same

Handshake still `reserve`/`push` K, V, mask for every KV chunk so compute does not deadlock (`CWFW` on CB front). Then `cb_gather_origin.push_back/pop_front`.

Skip **must** happen after Q is reserved/filled for this work item; the current code `cb_query.push_back` then handshake then `continue`.

### 4.5 Skip (writer)

Writer runs in parallel with the reader. It has its own origin CB (`cb_writer_origin`) and must **not DRAM-write** bricks the other launch owns, or the edge pass overwrites interiors.

Same polarity as the reader. `write_this_chunk = false` then skip the `noc.async_write`. Assigning a 0/1 bool has been DCE-suspicious; prefer `if (token == 0xFFFFFFFF) continue` in the write loop.

### 4.6 Compute

Unchanged flash (`neighborhood_sdpa.cpp`). Probe 8 maps to compute drain. Probes 1/2/7 keep shipped compute.

---

## 5. File map

| File | Role |
|---|---|
| `ttnn/.../neighborhood_sdpa_device_operation.cpp` | Auto-split launch (mode 1 then 2) |
| `ttnn/.../neighborhood_sdpa_program_factory.cpp` | Wrapper sources, `-DNA_PATH_KIND`, `skip_if_bit` 2/3, pack skip into `tile_bytes` high half |
| `ttnn/.../neighborhood_sdpa_nanobind.cpp` | Stamp origin table; Python `path_mode` |
| `ttnn/.../neighborhood_plan.cpp` | `use_interior_table_by_chunk` |
| `ttnn/.../kernels/neighborhood_kernel_args.hpp` | Origin columns, `path_mode` / `skip_unowned` / `skip_if_bit` |
| `ttnn/.../dataflow/neighborhood_reader.cpp` | Shared reader; skip + both gather loops behind `if constexpr` |
| `ttnn/.../dataflow/neighborhood_reader_{interior,edge}.cpp` | Thin `#include` + `NA_HAS_PATH_SKIP` |
| `ttnn/.../dataflow/neighborhood_writer.cpp` | Writer skip |
| `ttnn/.../dataflow/neighborhood_writer_{interior,edge}.cpp` | Thin includes |
| `models/tt_dit/tests/unit/test_neighborhood_sdpa_components.py` | Timing wall; **wrong output** except as a perf probe |
| `models/tt_dit/tests/unit/test_neighborhood_sdpa.py` | Correctness |
| `run_na.sh` | Single-log loop, `-b` rebuild, stall watchdog, board reset on hang |
| `na_stall_watchdog.py` | Kills a silent hang after `-s` seconds |

JIT cache: `~/.cache/tt-metal-cache/`. Kernel **hash does not include included `.cpp` body** — hash is compile args + defines + entry path. Changing `neighborhood_reader.cpp` still **rebuilds** via `.dephash` (you will see `compiled ... in 520 ms`). Wrapper `#define NA_SKIP_REV` does **not** change the hash. ELF path example:

`~/.cache/tt-metal-cache/<id>/kernels/neighborhood_reader_interior/<hash>/ncrisc/ncrisc.elf`

Disassemble with:

```bash
tt-metal/runtime/sfpi/compiler/bin/riscv-tt-elf-objdump -d ncrisc.elf
```

`kernel_main` is inlined into `_start`. Search for `handshake_skip_work_item`.

Host C++ changes (factory, nanobind, kernel_args, plan, device op): `./run_na.sh -b ...`. Kernel `.cpp` under `device/kernels/`: JIT, no `-b`.

---

## 6. This RISC toolchain's DCE traps

Discovered the hard way. Treat as constraints, not style.

| Form | Result |
|---|---|
| `if (x)` / `if (!x)` / `if (x == 0)` / `if (x == 1)` | **DCE'd**; skip never taken |
| bool-returning helpers (`na_path_skips_chunk`, `na_should_skip`, `na_skip_kind`) | Always false |
| `if constexpr (path_mode == 1)` | DCE'd |
| `#if NA_PATH_KIND == 1` around skip | Behaved as 0 (`if (false)`); defines hashed but skip still dead |
| Wrapper `#define NA_SKIP_IF` before include | Not enough; included reader still walked all |
| Extra runtime arg for skip polarity | Dropped (8-arg pattern is 5 buffers + 3 uint32s) |
| `if (skip_kv == 2)` / `if (skip_kv == 0)` | **Works** (probes actually skip the slot loop) |
| `path_mode != 2` for `compile_interior` | **Works** (compile times differ) |
| `if (true) { handshake; continue; }` | **Works** (52 ms / 459 ms) |
| `if (token == 0xFFFFFFFF)` in the ELF | Branch is **emitted**; still never taken at runtime on the value loaded |

So: skip **control flow** can be live; **0/1 data** and **bool helpers** are poison; compare-to-2/3 survives for compile args (`skip_kv`). A compare-to-`0xFFFFFFFF` **is in the binary** and still does not take — that is a **load / value** bug, not “the if was deleted.”

`packed_width < 0x80000000u` compiled as **signed `bltz`**. Tensix L1 addresses live in `0xFFBxxxxx` (always negative as i32). If you accidentally `bltz` an **address** instead of the loaded word, you always take “process.” If the **data** word always has bit 31 set, interior never skips.

---

## 7. What we already tried (all left skip dead unless noted)

1. One mixed kernel — **~416 ms**. Baseline to beat.
2. Split without skip — **~530 ms**. Slower. Current tree.
3. `na_path_skips_chunk` / `na_should_skip` / `na_skip_kind` — DCE to false.
4. `(2 + (gather_width>>31)) == skip_if_bit` with skip_if 2/3 compile-arg — no match.
5. Factory `-DNA_PATH_KIND` / `-DNA_SKIP_IF` + `./run_na.sh -b` — hashes changed, skip still dead.
6. Extra runtime arg for skip_if — dropped.
7. Pack `skip_if << 16` into `tile_bytes` (`tile_and_skip`) — still ~531 ms (was still a bit-31 skip_token).
8. `#if NA_PATH_KIND == 1/2` around `if (!bit31)` / `if (bit31)` — ~530 ms.
9. `skip_edge_token` 0 vs `0xFFFFFFFF`, late load from `origin_row[7]` after query push — ELF had `lw off 28; li -1; beq` **once**; still ~531 ms. Suspected **a1 reused** (not the origin dest).
10. Early load of gather_width bit 31 immediately after barrier — compiled as `bltz`; still ~531 ms.
11. `CoreLocalMem<volatile uint32_t>(origin_write_pointer)[7]` immediately after barrier — still ~531 ms.
12. `if (true)` skip — **52 ms / 459 ms**. Path live.

Leftover experimental junk still in tree (safe to delete once skip works): `na_path_skips_chunk`, `na_skip_kind`, `na_should_skip`, `NA_SKIP_IF` macros, packing skip_if into `tile_bytes`, `NA_SKIP_REV` in wrappers, origin round-trip prints in the component test.

---

## 8. Disassembly notes (so you do not repeat the “is it even compiled?” loop)

Interior reader ELF (`neighborhood_reader_interior/<hash>/ncrisc/ncrisc.elf`):

- `handshake_skip_work_item` is a real function; **one** `jal` site from `_start`.
- After origin noc (`li a4, 64` size, `fence`): a load from `8(a*)` or `28(a*)` then `beq` vs `-1` or `bltz`.
- If you `if (true)` skip interiors, that `jal` is unconditional and interior JIT **drops ~100 ms** (tight gather DCE).

Edge reader: skip `jal` exists; polarity was `bltz` → handshake when bit 31 set (skip interiors).

If both programs **process** every brick with those polarities, they cannot be looking at the **same** word (interior would need bit 31 always set, edge always clear). That is the smoking gun for “wrong pointer / different value,” not “branch deleted.”

---

## 9. How to run

### Timing (the loop you want)

```bash
cd tt-metal
# Terminal A
tail -f generated/na_run.log

# Terminal B — kernel-only
./run_na.sh -f models/tt_dit/tests/unit/test_neighborhood_sdpa_components.py -t 240 -s 90

# after factory / nanobind / kernel_args / device op
./run_na.sh -b -f models/tt_dit/tests/unit/test_neighborhood_sdpa_components.py -t 240 -s 90
```

Watch for:

```
origin stamp: chunks=42840 bit31=31104 col6=31104 col7_edge=11736
origin page=64
window11: ??? ms
```

`compiled neighborhood_reader_interior/... in 526 ms` vs `..._edge/... in 706 ms` means the two loops peeled.

### Correctness

```bash
./run_na.sh -b -f models/tt_dit/tests/unit/test_neighborhood_sdpa.py -t 180 -s 60
```

### Decode (after skip works)

```bash
DIFFVAE_BLOCK_PROF=1 DIFFVAE_STAGE5_BACKEND=bricked_sp_w_sharded DIFFVAE_S5_GNA_STRIDE=1,1,1 \
  bash models/tt_dit/experimental/scripts/run_ltx25_diffvae.sh --timeout=0
```

Look at `neighborhood-sdpa` n=2. Want ~400–500 ms total, not ~737.

A hang holds the chip. `run_na.sh` resets the board on timeout/stall (`tt-smi -r all`). Watcher (`-w`): `CWFW` = waiting on CB front (classic skip/handshake mismatch).

### Interior-only timing (wrong edges, useful)

Python `path_mode=1` on the component test (you may need to thread it through `_time_probe`; default is 0). Expect ~130 ms window11 if the tight loop is the only walk.

---

## 10. Suggested next experiments (in order)

1. **`if (true)` interior-only** — confirm you still get ~459 ms on this machine. If not, you are not running the split wrappers.
2. **Skip on a compile-time constant 2/3 only** — interior binary `if (2 == 2) skip all`. Proves `if (x == 2)` in the skip site (you already know `skip_kv == 2` works in the KV loop).
3. **Force the origin dest into a stack slot** before the noc command, reload that address after `fence`, `lw` column 7. Do not reuse `a1`/`a2` from the query path. Compare to `0xFFFFFFFF` **or** to `3` if you restamp col 7 as 2 vs 3 (needs `-b`).
4. **Stamp col 7 as 2 (interior) / 3 (edge)** and `if (token == 3)` — same shape as `skip_kv == 2`. Avoids signed `bltz` and 0/1.
5. **Dump one L1 row** (watcher / `DPRINT`) after the origin read: words 0, 2, 6, 7. If they are not the host row, TensorAccessor `page_id=chunk_index` or CB write ptr is wrong. Page size **is** 64 on BH (`buffer_aligned_page_size()` printed in the test).
6. **Do not** add more bool helpers or `if (interior)` flags.

Until skip fires, the faster shippable kernel is still the **single mixed** reader (~416 ms), not this split.

---

## 11. Decode / product context (why anyone cares)

LTX-2.5 DiffVAE stage 5 is ~80% of decode. Exact NA (stride 1) is the quality path. The relative table already cut mask generation from ~34 s decode to the current regime; FINDINGS §§1–8. The leftover per **op** is this 147-slot walk × I-cache mix.

Reference fused-sdpa at stride 1 was ~673 ms/block with ~21 keys/query (bigger query groups). We process 32 queries per gather and 147 key bricks. Per-key we are not worse; we visit more keys **and** walk them at 64 ns/slot instead of 20.

Per-brick sub-box (only ~54 keys of the 147-box actually in-window) is the **next** architectural cut after skip works. It is not this task.

---

## 12. Contacts / branch

Work was on `wip/na-second-kernel`. Do not merge for perf while window11 is ~530 ms. Correctness may still pass if the edge launch overwrites interiors with something that happens to look close on small tests — check PCC on a volume that actually has interiors (the unit tests used to hide that; FINDINGS §1).
