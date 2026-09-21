# Quasar narrow-row pack-untilize via iDMA (test id 918)

End-to-end narrow-row pack-untilize on Quasar: **stock hardware pack-untilize, then one iDMA
transaction to squeeze out the padding.** Every shape and every engine is verified datum for
datum against a golden, so this is a correctness test that also reports cycles.

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
   stage 2   ONE iDMA transaction                      (DM core, zone COMPACT)
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
| ops issued | 64 per tile | 1 per tile-row (scatter-list) |

The cost is a second L1 buffer: the compaction is not done in place. Overlapping source and
destination in one DMA would need the engine to respect read-before-write across rows, which
is not guaranteed. In the real use case the dense buffer is what the consumer reads anyway.

## The three engines

All three produce byte-identical output and all three are verified, which is what makes the
timing comparison fair. Selected per run by `engine_mode`:

| mode | engine | what it costs |
|---|---|---|
| 0 | **iDMA scatter-list** | ONE transaction; the hardware walks a 32-entry address list in L1 and the destination auto-increments. One RISC issue for the whole matrix. **This is the proposal.** |
| 1 | iDMA per-row | 32 transactions, addresses from the address generator, one drain at the end. Separates "iDMA is fast" from "one issue is fast". |
| 2 | NOC per-row | 32 stateful NOC reads from this core's own L1. **The current workaround, and the bar to beat.** Stateful means only the addresses change per call, so it is the cheapest NOC read available — not a straw man. |

### The scatter list

Entry `r` is the **offset** of padded row `r`, and the kernel passes the `pad` DFB's L1 base as
`SCATTER_BASE_ADDR`, so the host can build the list without knowing where the DFB landed.
Entries are 8 bytes and the engine takes the low 32 bits. That encoding is not documented
anywhere in this tree; it was resolved empirically (a reversed-permutation list, which can only
verify if the hardware really reads the entries written).

Two configuration facts that are easy to get wrong:

- `apply_scatter_to_dest = false` + `dest_addr_inc_en = true` is what makes it a *compacting*
  gather: the list drives the strided source side, the destination walks contiguously.
- **Every run must use a distinct list address.** The engine fetches a list once per address;
  later writes to the same address do not take effect, so a second run at the same address
  silently replays the first one's list. The test uses a process-global slot counter, not a
  per-test one, so two tests cannot collide even if the allocator hands them the same base.

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
| `ScatterListDrainProbe` | diagnostic, not perf: scatter-list with `num_iterations = 1` so there is no re-arm inside the zone. Confirms or refutes the drain race. |

## Correctness and timing are joined, not assumed

The profiler CSV records cycles and knows nothing about whether a run was right, so a report
built from it alone will present the cycle count of a wrong computation as a result. That is
not hypothetical: every scatter-list 8-channel run measured so far has been both fast and
wrong. Each run therefore appends its verdict to
`generated/profiler/.logs/narrow_row_results.csv`, in the same order as the profiler log, and
the report joins the two — flagging `!! WRONG OUTPUT` and excluding those rows from the
verdict table. If the verdict file is missing, the report says correctness is unknown rather
than implying everything passed.

## Measured so far (emu-quasar-1x3)

First run, 2026-09-21, `SingleTileHalfWidth` + `WidthSweep` + `SubFaceWidths`, scatter-list on
one channel:

- **The compaction is descriptor-bound and flat at ~8.3 cyc/row up to ~80 B/row**, then costs
  roughly one cycle per 16-20 B — i.e. one iDMA VC becomes the limit. Measured cyc/row: 8.38
  at 16 B, 8.33 at 64 B, 8.52 at 72 B, 9.34 at 88 B, 10.20 at 104 B.
- That reproduces the independently measured scatter-list cost model **78 cyc/transaction +
  5.8 cyc/entry** almost exactly: it predicts `78 + 32*5.8 = 263.6` and the flat region
  measured 263.5-269 on a completely different harness.
- **Byte-granular placement is free.** 66 and 70 B rows (odd datum widths, destinations at
  2 mod 4) cost 8.70 and 8.55 cyc/row against 8.56 and 8.52 for the adjacent even widths —
  1-2%, and not even monotonic.
- **Stage 1 single-shot is pipeline fill, not pack throughput.** ct_dim 1 cost 225-237 cycles
  and ct_dim 2 cost 213 — two tiles took *less* wall time than one, so the marginal per-tile
  cost is buried under a ~215-cycle fixed latency. Do not read the stage-1 `cyc/tile` column
  as a pack rate, and do not read a falling end-to-end `cyc/tile` at higher `ct_dim` as
  efficiency; it is that constant divided by more tiles.
- End to end against RV_PACR at `ct_dim` 1: 2.9-4.8x faster, and that *understates* it,
  because stage 1 is carrying the fixed cost above while the RV_PACR reference is
  steady-state.

Consequence for fan-out, and why `ChannelSweep` changed: both iDMA engines already emit one
packet per row, so 32 rows always give the round-robin more packets than it has channels.
Sub-splitting on top of that (the original policy, which would have turned 504 B rows into
256 packets of 64 B) multiplies the packet count while the bytes stay the same. The default
`max_packet_bytes = out_row_bytes` is the right one even when fanning out.

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
