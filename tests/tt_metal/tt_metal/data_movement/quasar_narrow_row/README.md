# Quasar narrow-row pack-untilize via iDMA (test id 918)

End-to-end narrow-row pack-untilize on Quasar: **stock hardware pack-untilize, then an iDMA
pass to squeeze out the padding.** Every shape is verified datum for datum against a golden,
so this is a correctness test that also reports cycles.

**Result: 2.5-2.9x faster than the current workaround**, at every one of 23 row widths from
2 B to 512 B per row, all verified. For the 32x252 reference workload, 786 -> 313 cycles.

## The problem

`_llk_pack_untilize_` cannot produce an untilized row narrower than a whole number of tiles on
Quasar. The packer's untilize output stride is face-granular, and the compute API exposes it
only in whole tiles — `api/compute/pack_untilize.h` static_asserts `narrow_row == false`,
`row_num_datums == TILE_C_DIM` and `dense == false` for `ARCH_QUASAR`. Wormhole and Blackhole
had a 16-byte output stride register; Quasar does not.

So untilizing a matrix whose width is not a multiple of 32 leaves junk at the end of every
row. The current workaround makes the *consumer* deal with it: the NOC issues a separate read
per row so each row's valid prefix is taken and the junk skipped. That replaces one read with
32, and the per-transaction overhead dominates.

## What this test does

```
                 ct_dim tiles, tilized                 (DRAM -> src_dfb, reader)
                            |
   stage 1   stock pack_untilize_block<ct_dim>         (TRISC, zone PACK_UNTILIZE)
                            v
          pad_dfb: 32 rows x ct_dim*32 datums          <- tile-aligned, trailing junk
             [ keep .......... | junk ]
             [ keep .......... | junk ]      32 rows
                            |
   stage 2   iDMA gather, one transaction per row       (DM core, zone COMPACT)
                            v
          out_l1:  32 rows x matrix_w datums           <- dense, no junk
             [ keep .......... ]
             [ keep .......... ]
```

with

```
matrix_w = (ct_dim - 1) * 32 + last_tile_w
```

i.e. full tiles `0 .. ct_dim-2` at 32 datums each, plus a last tile cut to `last_tile_w`. This
is exactly the layout the RV_PACR per-face-row path produces; the host golden is the same one.

## Why this instead of RV_PACR

RV_PACR drives the packer by hand, one 16-datum DEST face-row per op — 64 ops per tile, with
the face de-interleave and the row stride recomputed in software per op. It works, and it is
the only way to get a narrow row out of the packer itself, but tt-llk measures it at
1439–2418 cyc/tile against 77.4 for the hardware path.

Here the hardware does the untilize at full speed and iDMA only moves 32 rows afterwards.
Three consequences beyond the cycle count, all of them from iDMA addressing L1 by the **byte**
where RV_PACR's output address is **16-byte granular**:

| | RV_PACR | this path |
|---|---|---|
| face de-interleave | software `remap()` per op | done by the HW untilize |
| minimum narrow width | 8 datums (16-bit fmt), 16 (8-bit fmt) | none — see `SubFaceWidths` |
| width not a multiple of 8 | writes a full 16-datum face-row that spills into the next row, fixed by a second pass | nothing spills; one pass |
| ops issued | 64 per tile | 32 per 32-row block, whatever the width |

The cost is a second L1 buffer: the compaction is not done in place. Overlapping source and
destination in one DMA would need the engine to respect read-before-write across rows, which
is not guaranteed. In the real use case the dense buffer is what the consumer reads anyway.

## The engines

Selected per run by `engine_mode`. The two that ship produce byte-identical output and are
verified at every shape, which is what makes the timing comparison fair.

| mode | engine | |
|---|---|---|
| 1 | **iDMA per-row** | 32 transactions, addresses from the address generator, one drain at the end, fanned out over all 8 VCs. **The shipping engine and the default** — correct in every run at every width and channel count. |
| 2 | NOC per-row | 32 stateful NOC reads from this core's own L1. **The current workaround, and the bar to beat.** Stateful means only the addresses change per call, so it is the cheapest NOC read available — not a straw man. |
| 0 | iDMA scatter-list | ONE transaction; the hardware walks a 32-entry address list. 3% faster and **wrong** — see the bug below. Reachable only from `ScatterListMapping`; no correctness test selects it. |

The addressing follows `quasar_examples/quasar_idma/kernels/idma_1d_strided_example.cpp`
exactly: set the address generator's base and inner loop once, then `push_both_addrgen_0()` +
`issue_cmdbuf_0()` per row. All four addrgen examples do one push and one issue per address —
the 2D, face and interleaved loops change *which* address is yielded, not how many issues it
takes — so the address generator cannot amortise the per-issue cost, and ~8.7 cyc/row is the
floor. What this kernel adds over the examples is the iDMA channel autoincrement
(`req_vc_inc_en`), which neither iDMA example uses and which is worth 3.6x at 512 B/row.

### The scatter-list bug (why mode 0 is not shipped)

Entry `r` is the **offset** of padded row `r`, and the kernel passes the `pad` DFB's L1 base as
`SCATTER_BASE_ADDR`, so the host can build the list without knowing where the DFB landed.
Entries are 8 bytes and the engine takes the low 32 bits.

`ScatterListMapping` inverts the output back to the source position of every datum — the
stimulus makes each datum name its own place — and the answer at 8 channels is:

```
source index sequence: 0 x16, 16, 17, 18, ... 31        unwritten = 0
```

Every destination slot is written exactly once, so **nothing is overwritten and the
destination auto-increment is correct**. What fails is `SCATTER_INDEX`: it does not advance
for the first 16 entries, so the first 16 destinations all receive entry 0's offset, then the
index jumps to 16 and tracks correctly. 16 entries x 8 B = 128 B, one list fetch block.

Within that stuck prefix the source address **alternates** between src row 0 col 0 and col 8
— offset +0 and +16 B. So the list *index* is stuck at 0 while the address the engine forms
still moves, between two values 16 B apart. That rules out the simplest reading ("it reuses
entry 0 until the fetch lands") and is the level of detail the overlay owners will want.

- One channel is correct; eight is always wrong — consistent with the engine consuming entries
  before the first block lands.
- **Compaction is irrelevant**: `last_tile_w = 32` gives `out_row_bytes == pad_row_bytes` and
  fails identically. This is a plain scatter-list bug, not a narrow-row one.
- 4 rows correct but 2 rows wrong, which does not fit a clean block-latency story, so treat
  the 128 B reading as strong but not proven.

One more property worth knowing if anyone revives this: **every run must use a distinct list
address.** The engine fetches a list once per address; later writes to the same address do not
take effect, so a second run at the same address silently replays the first one's list. The
test uses a process-global slot counter so two tests cannot collide even if the allocator
hands them the same buffer base.

### `MAX_BYTES_IN_PACKET`

`CMDBUF_RESET` zeroes it to its rdl default of 0, which means *never split*. Any kernel that
resets the command buffer and forgets to reprogram it gets no packet split at all, and since
fan-out round-robins **packets**, every channel knob then goes silently inert — the tell is all
channel counts landing within a couple of percent of each other. This kernel reprograms it
explicitly after every reset.

## Running

Health check first — one tile, half width:

```bash
TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/unit_tests_data_movement \
  --gtest_filter="*QuasarNarrowRowUntilize.SingleTileHalfWidth*"
```

If that fails, run the plain iDMA functional examples (`*QuasarIdmaOps*`) to tell a broken
kernel apart from a broken emulator.

Everything, with timing. `TT_METAL_DEVICE_PROFILER=1` is required — without it the zone and
stamp macros compile to no-ops and no CSV is produced:

```bash
TT_METAL_SLOW_DISPATCH_MODE=1 \
TT_METAL_DEVICE_PROFILER=1 \
TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=1333 \
./build/test/tt_metal/unit_tests_data_movement \
  --gtest_filter="*QuasarNarrowRowUntilize*"
```

Then report:

```bash
python3 tests/tt_metal/tt_metal/data_movement/quasar_narrow_row/report_narrow_row_from_csv.py
```

Requires `ARCH_NAME=quasar` and `TT_METAL_SIMULATOR` pointing at an `emu-quasar-*` build (a
path, not `1`). The 1x3 emu has no fast-dispatch cores, hence slow dispatch.

## The test bodies

| test | what it covers |
|---|---|
| `SingleTileHalfWidth` | one tile, `last_tile_w = 16`. Start here. |
| `WidthSweep` | `last_tile_w` 8 / 16 / 24 / 32 — the widths RV_PACR supports, so the two paths compare on equal ground. 32 is the degenerate whole-tile case (`matrix_w == pad_w`, the gather becomes a straight copy), a good null check on the address math. |
| `TileRowSweep` | `ct_dim` 1 / 2 / 4 / 8 x `last_tile_w` 8 / 16 / 32. The shape a real matrix has, and where rows grow long enough for the payload to start dominating the per-entry cost. `ct_dim` 8 is the half-sync 16-bit DEST limit for `pack_untilize`. |
| `SubFaceWidths` | `last_tile_w` 1 / 2 / 3 / 4 / 12 / 20 — below the RV_PACR floor, and the odd ones make `out_row_bytes` odd, probing byte- rather than word-granular placement. |
| `EngineComparison` | all three engines over three shapes chosen around the measured knee — 32, 224 and 504 B/row — plus the two iDMA engines at 8 channels where bytes dominate. |
| `ChannelSweep` | whether sub-splitting a row into packets helps fan-out or just costs issue, below the knee and well past it. |
| `WorkaroundSweep` | **the adoption decision.** The full grid: `ct_dim` 1/2/4/8 x `last_tile_w` 8/16/24/32, each run as the workaround, as iDMA at 1 channel, and as iDMA at 8 channels. 48 runs. |
| `NarrowExtremes` | the high-waste end — `last_tile_w` 1/2/4/8, where up to 97% of the padded row is junk and reading it whole is not an option. Workaround vs iDMA, 16 runs. |
| `ScatterListMapping` | diagnostic, asserts nothing. Inverts the output back to the source position of every datum and root-causes the scatter-list bug above. Leads with the per-row engine as a control. |

## Correctness and timing are joined, not assumed

The profiler CSV records cycles and knows nothing about whether a run was right, so a report
built from it alone will present the cycle count of a wrong computation as a result. That is
not hypothetical: every scatter-list 8-channel run measured so far has been both fast and
wrong. Each run therefore appends its verdict to
`generated/profiler/.logs/narrow_row_results.csv`, in the same order as the profiler log, and
the report joins the two — flagging `!! WRONG OUTPUT` and excluding those rows from the
verdict table. If the verdict file is missing, the report says correctness is unknown rather
than implying everything passed.

The join is **by index**, so every run that emits a profiler zone must also write a verdict
row, including the diagnostics. `ScatterListMapping` therefore records its runs as
`VERDICT_DIAGNOSTIC`, and the report drops them. That is not tidiness: the mapping probes run
the *shipping* engine at a shape the sweeps also cover, with `compact_iterations = 1`, so
leaving them in both poisons that shape's average and marks the group "correctness unknown",
which hands the verdict to whatever arm is left. It reported "workaround wins" at 504 B once
for exactly that reason. The report also refuses to join at all if the two files disagree on
length, rather than producing a plausible wrong answer from a stale pairing.

## Measured results (emu-quasar-1x3)

`WorkaroundSweep` + `NarrowExtremes`, 23 row widths, 32-row block, `Float16_b`, single DM
core. Every run verified datum for datum. Two of the three lines are flat, and that is the
whole result.

**The workaround is payload-blind.** 781.6-790.2 cycles across a **256x span in payload** —
24.5 cyc/row at every width. Entirely issue-bound; the bytes are free. Its cost is ~785 cycles
for a 32-row block no matter how wide the matrix is, so iDMA's advantage cannot grow with
width — it narrows slightly, 2.88x at 16 B/row to 2.51x at 504 B.

**iDMA at 8 channels is also flat.** 273.6-313.4 cycles over the same range, 8.6-9.8 cyc/row.
At 512 B/row the data time spread over 8 VCs is only 4 cyc/row, well under the ~9-cycle issue
floor, so this should stay flat to roughly 1250 B/row.

| B/row | junk | NOC | iDMA 1 ch | iDMA 8 ch | speed-up |
|---|---|---|---|---|---|
| 2 | 96.9% | 786.2 | 294.6 | — | 2.67x |
| 16 | 75.0% | 786.8 | 273.8 | 273.6 | 2.88x |
| 64 | 0.0% | 785.1 | 279.8 | 279.8 | 2.81x |
| 128 | 0.0% | 781.6 | 345.9 | 278.0 | 2.81x |
| 256 | 0.0% | 781.9 | 601.2 | 294.4 | 2.66x |
| **504** | **1.6%** | **786.2** | **1120.9** | **313.4** | **2.51x** |
| 512 | 0.0% | 785.9 | 1117.4 | 313.1 | 2.51x |

**Always fan out.** One channel is flat only to a ~100 B knee, then grows at **15.9 B/cyc —
one iDMA VC's 16 B/cyc** — and crosses the workaround at **~347 B/row**. On a 504 B row a
one-channel implementation runs at 0.70x: slower than the code it replaces. Eight channels is
never more than 0.4% behind one channel anywhere in the grid, so there is no width threshold
to tune.

**Byte-granular placement is free.** 66 and 70 B rows (odd datum widths, destinations at
2 mod 4) cost 8.70 and 8.55 cyc/row against 8.56 and 8.52 for the adjacent even widths — 1-2%,
and not even monotonic.

### Two things these numbers do not say

**Stage 1 is a broken measurement.** It reports a flat ~233 cycles at ct_dim 1, 4 *and* 8 —
29.1 cyc/tile at ct_dim 8, below tt-llk's steady-state 77.4, which is impossible as a rate.
`pack_untilize_block` returns once the MOP is issued, so the zone closes before the pack
drains and the drain is absorbed by the following `push_back`. Both paths share stage 1, so
the comparison above is unaffected, but **do not quote the end-to-end `cyc/tile` column or its
"vs RV_PACR" multipliers.** Using tt-llk's 77.4 cyc/tile instead, the *whole* narrow-row
untilize improves ~1.5x at eight tiles wide and ~2.5x at one — iDMA helps most where
narrow-row matters most, because the pack is cheap there.

**The consumer-side saving is not counted.** Every number above is the producer-side
compaction only. The workaround makes the consumer issue 32 reads; this path leaves it one.

### Check `junk%` before adopting this at all

At 32x252 only 1.6% of the padded row is waste. If the consumer's row stride is negotiable it
can read the padded buffer whole and discard 256 bytes, which beats both paths and costs
nothing to build. Compaction is only unavoidable where the junk share is large — 75-97% at one
tile wide, which is also where iDMA's margin is best.

## Reading the numbers

`report_narrow_row_from_csv.py` walks the CSV in execution order rather than grouping by
`run host ID`, which slow dispatch never increments. It prints stage 1 and stage 2 separately
and then pairs them by index for an end-to-end figure, comparing against the tt-llk RV_PACR
reference (1439.5 / 1470.4 / 2418.0 / 2029.6 cyc/tile for width 8 / 16 / 24 / 32) and against
77.4 cyc/tile for normal untilize.

Two caveats the script also prints:

- **Stage 1 is measured single-shot** (`COMPUTE_LOOP_FACTOR = 1`), so it carries pipeline fill
  and reads high against a steady-state 77.4. That is not a tuning choice: the Quasar
  tile-counter model consumes a tile per unpack, so re-packing resident tiles without
  re-streaming them gives valid timing but undefined data from the second iteration on, and
  these runs check their data. Stage 2 repeats freely — an L1→L1 gather is idempotent.
- The stage pairing assumes each program run emits exactly one zone of each kind. The script
  refuses to print totals if the two counts disagree rather than reporting numbers it cannot
  trust.

## Files

| file | |
|---|---|
| `test_pack_untilize_narrow_row.cpp` | host: shapes, stimulus, golden, program spec |
| `kernels/narrow_row_untilize_compute.cpp` | stage 1, TRISC |
| `kernels/narrow_row_compact_dm.cpp` | stage 2, DM core, all three engines |
| `report_narrow_row_from_csv.py` | CSV reconstruction and tables |

The reader is the shipping
`tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/dram/direct_reader_unary_2_0.cpp`.
