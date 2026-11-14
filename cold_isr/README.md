# Cold DFB ISR latency: why our branch is slower than main

Issue #55788. Investigation run on 2026-10-15/16 on the Quasar 2x3_DISPATCH-userbits emulator, with waveforms.

## Question

The latency we track is from the interrupt to the data being ready: RoCC interrupt rise (T1) to the tile counter update reaching Neo0.
For the first (cold) DFB ISR of the base case it is:

| Build | Cold ISR (cycles) |
|---|---|
| origin/main (3be67656b28) | ~360-380 |
| our branch | ~490 |

Is the difference caused by code or by data layout?

**Answer: it is caused by memory and cache state, not by instructions.** Three causes were found:

1. **Vector-line fetch.** All 8 DM harts write the shared interrupt table at boot. The line ends up dirty in whichever core's D$ wrote it last, so the ISR's first I$ fetch must wait for L2 to pull it back from that D$ (31 cycles instead of 13). In main, DM0 happens to evict the line to L2 about 6k cycles before the ISR. **Fixed and verified in an experiment (VB / VF1).**
2. **Stack alignment.** Our changes shrank `_start1`'s stack frame from 256 to 112 bytes. As a result, the ISR's 96-byte register save spans 3 D$ lines instead of 2. With otherwise identical code this costs about 54 cycles on the cold ISR and about 12 on every warm ISR.
3. **Handler-line fetch.** The I$ line holding the handler entry (0xEE00) is sometimes a slow refill (51-75 cycles) and sometimes fast (13). This changes between builds that differ only in an unrelated immediate, so it is not a fixed property of the code address. **Not yet explained; this is the next open item.**

## Cache facts used

These come from `risc_common.h`. They were confirmed against the observed refill latencies.

- L1 D$: 4 KB, 2-way, 64 B lines, private, write-back. Set = `(addr >> 6) & 31`.
- L1 I$: 4 KB, 2-way, private.
- L2: 128 KB, 4-way, 64 B lines, shared by all 8 DM cores, coherent with each L1 D$. Set = `(addr >> 6) & 511`.
- Observed refill latency, measured from request accepted to the first data beat:

  | Refill | Clean line in L2 | Otherwise |
  |---|---|---|
  | I$ | ~13 | 31-75 |
  | D$ | ~18 | 54-72 |

## Method

- Every arm is one waveform run of `dfb_init_timing_bench --case base`. Waveforms are exported with the README's `isr_tc.rc`, and latency is computed with `quasar-isr-latency/isr_tc_latency.py` using that ELF's handler PCs.
- Captures are deterministic: rerunning the same binary gives the same numbers. Any change to code or layout can flip cache states, though. So each arm that changes behaviour has a **null control with identical code and layout**:
  - an opaque `li` flag turns the change off at runtime; or
  - for the stack arms, only an immediate differs.
- Cache traffic: `tools/probe_cache.rc` re-exports, from the same `waves.ztdb`, DM0's I$ and D$ TileLink A/C/D channels, which Simzilla rebuilds from the saved waveform. Note that `tile_prci_domain_N` names for the other cores did not resolve. The rc adds these signals:
  - `.../tile_prci_domain/element_reset_domain_rockettile/frontend/icache/auto_master_out_{a_valid,a_ready,a_bits_address,d_valid}`
  - `.../dcache/auto_out_{a_valid,a_ready,a_bits_address,a_bits_opcode,c_valid,c_bits_address,d_valid}`
  - `.../dcache/io_cpu_s2_nack`
- Analysis tools:

  | Tool | Purpose |
  |---|---|
  | `tools/cache_tl.py` | Timeline around the ISR: retires, refills, writebacks |
  | `tools/refills.py` | Per-refill latency table |
  | `tools/l2set_scan.py` | Every DM0 transaction to one L2 set over the whole capture |
  | `tools/isr_entry_trace.py`, `tools/isr_retire_trace.py` | Per-instruction retire timing |

## Results

Times are cycles. The cold ISR breaks down as:

- **entry**: T1 to the first handler instruction;
- **code**: handler start to the counter write;
- a fixed 8 cycles for the counter update to reach Neo0.

Warm is the second ISR's code time.

### Baselines and placement sweep

| Arm | Build | Handler | Cold | Entry | Code | Warm code | zrun |
|---|---|---|---|---|---|---|---|
| W0 | ours (production layout) | 0xee28 | 493 | 129 | 356 | 78 | 176 |
| B0 | base 3a44129a501, no uncommitted changes | 0xee28 | 421 | 87 | 326 | 64 | 190 |
| M16 | origin/main + pad | 0xee38 | 375 | 35 | 332 | 70 | 187 |
| M32 | origin/main + pad | 0xee48 | 397 | 105 | 284 | 66 | 188 |
| M48 | origin/main + pad | 0xee58 | 359 | 65 | 286 | 72 | 189 |
| W16 / W32 / W36 / W40 / W44 / W48 / W52 / W56 / W192 | ours + pad in `.text.unlikely` | shifted | 528 / 423 / 453 / 439 / 447 / 415 / 415 / 473 / 523 | | | | 177-186 |

- **Base vs ours.** B0 and W0 have the handler at the same address, and their handler instructions are identical. Only the data addresses and struct field offsets differ. So of ours' +112 over main, about +40 is upstream (base vs main) and +72 comes from our changes.
- **Placement.** Moving the handler does not close the gap. Ours stays at 415-528 across 10 placements, and main stays at 359-397.

### Breakdown of entry, W0 vs B0 vs main

- **Poll loop.** Identical in both builds, with a 40-cycle period. Its phase relative to T1 does not predict anything.
- **T1 → vector.** 51-55 in ours and base; 33 in every main capture. The difference is the vector-line I$ refill: 31 vs 13.
  - E2D shows this is not the in-flight poll load. With no load in flight at T1, T1 → vector was still 52 (`retire 0xed34` at +52).
- **Vector → handler.** Three groups:

  | Vector → handler | Arms |
  |---|---|
  | 2 (I$ hit) | M16 |
  | 32 (fast refill) | B0, W16/32/48/52, M48 |
  | 72-80 (slow refill) | W0/36/40/44/56/192, M32 |

  In W0 the handler-line request is accepted immediately but its data takes 59 cycles. In B0, with the same address, it takes 13.

### Breakdown of handler code, W0 vs B0

Retire traces are identical after the register save. All of the +30 is in the 11 `sd` stores:

- B0 stalls twice (2 stack lines). W0 stalls three times (3 stack lines), and one of those misses first writes back a dirty line.
- The cause is `_start1`, which has main inlined:

  | Build | `_start1` frame | Interrupted sp mod 64 | ISR save area | Lines |
  |---|---|---|---|---|
  | base | 256 B | 0 | [sp-88, sp-8) | 2 |
  | ours | 112 B | 16 | same span | 3 |

### Experiments

| Arm | What it changes (vs its control) | Cold | Entry | Code | Warm code | zrun | Result |
|---|---|---|---|---|---|---|---|
| E1N | pad function in the handler's I$ line, not called | 433 | 83 | 342 | 72 | 191 | control |
| E1C | same, called just before polling (handler line already in I$) | 385 | 53 | 324 | 78 | 193 | entry −30: the handler-line refill is real |
| E2D | 64-iteration spin between polls (no load in flight at T1) | 504 | 142 | 354 | 76 | 194 | T1 → vector unchanged: not the poll load |
| S64 | `addi sp,-64` around the poll (3 stack lines, control) | 483 | 83 | 392 | 72 | 196 | control |
| S16 | `addi sp,-16` around the poll (2 stack lines) | 477 | 131 | 338 | 60 | 195 | **code −54, warm −12**. The handler line happened to be slow here (57 vs 13), which hides the gain in the total. |
| L2W0 | warm-up code present, flag off (control) | 467 | 141 | 318 | 78 | 198 | control |
| L2W1 | before polling, load and `cflush.d.l1` the vector, handler and stack lines (leaves them clean in L2) | **381** | 77 | 296 | 88 | 197 | equals main. Refills: vector 13, handler 25, stack 32/13/54. |
| VF0 | vector flush present, flag off (control) | 467 | 131 | 328 | 56 | 201 | vector refill 31 |
| VF1 | before polling, DM0 loads the vector line and `cflush.d.l1` it | 429 | 109 | 312 | 61 | 200 | **vector refill 13; entry −22** |
| VB | only hart 0 writes the interrupt table, then `cflush.d.l1` at boot; all harts still set mtvec | 497 | 109 | 380 | 82 | 199 | **vector refill 13.** The line was flushed at boot (~75k cycles earlier) and was still in L2. This arm also has 3 stack lines and a slow handler line. |

## Finding 1: the vector line (resolved)

- `device_setup()` runs on every DM hart, and so does `setup_isr_csrs()` (`dm.cc`).
- All harts share one firmware image (`dm0.elf`) and one interrupt table at `MEM_INTERRUPT_TABLE_BASE` (0xED00). So all 8 harts store the same words into the same line at boot. Each store takes the line into that core's D$, and the last writer keeps it dirty.

What happened to line 0xED00 in each build (`tools/l2set_scan.py` on DM0's traffic):

| Build | DM0's history of line 0xED00 | At the ISR's I$ fetch |
|---|---|---|
| main (M48) | DM0 writes it last and keeps it dirty. At −5,945 its descriptor-copy loop reads config line 0x38500, which is in the same D$ set (20), and this evicts the vector line to L2. | Clean in L2: **13** |
| base (B0) | DM0 writes it last and never evicts it. | L2 probes DM0's own D$ (visible writeback at +23): **31** |
| ours (W0) | DM0's copy is probed away at −74,681 and −74,563 while DM0 is only computing (no DM0 access to that set), so another core requested it. | Dirty in another core's D$ (inferred; other cores are not visible): **31** |

Main's fast vector fetch therefore depends on two lucky events: DM0 wins the boot race, and an unrelated config read evicts the line shortly before the ISR.

**Fix (VB, recommended):** only hart 0 writes the interrupt table and then writes the line back to L2; every hart still sets mtvec. This costs nothing per launch, and it also removes 8 cores storing to one line at boot.

- Caveat: until hart 0 has written the table, an exception on another hart would vector through stale entries. The entries are identical once written, because all harts run the same image.

**Alternative (VF1):** a per-launch load plus `cflush.d.l1` of the vector line before arming.

## Finding 2: stack alignment (cause confirmed, fix not chosen)

The ISR saves 11 registers (88 bytes) in a 96-byte frame. Whether that spans 2 or 3 D$ lines depends on the interrupted sp mod 64, which is set by `_start1`'s frame size and changes whenever `_start1` changes.

Fix options:

- Align the ISR frame to 64 bytes inside the handler. The handler is a compiler-generated `interrupt` function, so this needs either a naked or asm prologue or a check that the compiler supports it.
- Keep sp at a fixed alignment while DM0 waits, for example by adjusting it in `wait_subordinates` as S16 does.

## Finding 3: handler line 0xEE00 (open)

The handler-entry I$ refill is 13 in some arms and 51-75 in others, sometimes at the same address with identical code (S16 57 vs S64 13). Line 0xEE00 also contains `_start`, which all harts execute at boot.

The working hypothesis mirrors Finding 1: the line's coherence or L2 state after boot depends on what the other harts did. L2W1 brought it down to 25 but not to 13. Next steps:

- run `l2set_scan.py` on 0xee00 across arms;
- find what puts the line in a slow state.

## Other corrections made along the way

- **Wrong:** "the trap waits behind the in-flight uncached poll load". E2D disproves it.
- **Wrong:** "other cores' traffic evicted the vector line from L2". The line was never cleanly in L2 in ours or base.
- Placement sensitivity is real but does not explain the gap. The samples are discrete, deterministic states, not noise.

## State after the investigation

- The tree's `dm.cc` is the default; arm files live only in `cold_isr/arms/`.
- `ENABLE_WAVEFORM=0` is set in `emu-quasar-2x3_DISPATCH-userbits/run.sh`.
- `wt/main` is at origin/main 3be67656b28 and clean, but its build is still from the base checkout, so rebuild it before use.
- zrun directories 176-201 are under `.../emu_out/2x3_DISPATCH/test_umd_remote_2x3_DISPATCH_0/`. The `probe_cache.rc` exports are each run's newest `export.N`. zrun.1, 209 and 210 were not touched.

## Files

| Path | Contents |
|---|---|
| `arms/` | DM0 firmware sources for each arm, as full copies of `dm.cc`. Base and main arms are for `wt/main` checked out at the respective commit. |
| `tools/` | Capture scripts (`wave_arms*.sh`, `emu_lib.sh`), the offline compile check (`ofl_arm.sh`), the export signal list and the analysis scripts |
| `data/` | Status logs with every arm's result line, cache timelines (`ctl_*`), L2-set scans (`l2s_*`) and retire and entry traces |
