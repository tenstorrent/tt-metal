# Quasar narrow-row pack-untilize via iDMA (test id 918)

End-to-end narrow-row pack-untilize on Quasar: **stock hardware pack-untilize, then an iDMA
gather to squeeze out the padding.** Every shape is verified datum for datum against a golden.

**The gather runs at about 2.8× the NOC-read-per-row workaround at 70 B/row and 2.5× at
504 B.** A development sweep over 23 row widths first measured 2.5–2.9×, but its NOC baseline
was on the *any-len* read path — the default `max_page_size` is `NOC_MAX_BURST_SIZE + 1`, so
every call also wrote the length register and computed a packet count for the barrier. Rows
here are at most 512 B against a 65536 B burst limit, so the one-packet path was always
available, and the kernel now selects it explicitly. (Neither path chunks in software: the
overlay packetizes via `MAX_BYTES_IN_PACKET`, so the whole difference is one register write
plus a shift-and-add.)

Re-measuring on the corrected baseline moves the result by under 1%. Per-block cost
difference between the two engines on the same shape:

| B/row | any-len baseline | one-packet baseline |
|---|---|---|
| 64 / 70 | 505.9 | 508 |
| 504 | 474.1 | 470 |

**How solid these are.** The two ratios are reconstructed as `iDMA + delta`, not read off an
isolated gather zone — the shipped test carries no profiler instrumentation, so the measured
zone is the whole stage-2 kernel including its wait on the pack. They are also single samples:
at 32 B/row one channel came out 210 cyc *faster* than eight, which cannot be real, so treat
±200 cyc as the noise floor and disregard that row's delta. The two rows above are the
trustworthy ones, and they agree with the sweep to within 1%.

The shape of the result is firmer than the ratio, because it is structural: both engines are
issue-bound at these sizes, the workaround's cost is independent of payload, and a single iDMA
channel becomes data-bound at one VC's 16 B/cycle and loses to the workaround on long rows.

## The problem

`_llk_pack_untilize_` cannot produce an untilized row narrower than a whole number of tiles on
Quasar. The packer's untilize output stride is face-granular, and the compute API exposes it
only in whole tiles — `api/compute/pack_untilize.h` static_asserts `narrow_row == false`,
`row_num_datums == TILE_C_DIM` and `dense == false` for `ARCH_QUASAR`. Wormhole and Blackhole
had a 16-byte output stride register; Quasar does not.

So untilizing a matrix whose width is not a multiple of 32 leaves junk at the end of every
row. The current workaround makes the *consumer* deal with it: the NOC issues a separate read
per row so each row's valid prefix is taken and the junk skipped — 32 reads where one would do.

## What this test does

```
                 ct_dim tiles, tilized                 (DRAM -> src_dfb, reader)
                            |
   stage 1   stock pack_untilize_block<ct_dim>         (TRISC)
                            v
          pad_dfb: 32 rows x ct_dim*32 datums          <- tile-aligned, trailing junk
             [ keep .......... | junk ]
             [ keep .......... | junk ]      32 rows
                            |
   stage 2   iDMA gather, one transaction per row      (DM core)
                            v
          out_l1:  32 rows x matrix_w datums           <- dense, no junk
             [ keep .......... ]
             [ keep .......... ]
```

with `matrix_w = (ct_dim - 1) * 32 + last_tile_w` — full tiles at 32 datums each, plus a last
tile cut to `last_tile_w`. This is the layout the RV_PACR per-face-row path produces; the host
golden is the same one.

## Why this instead of RV_PACR

RV_PACR drives the packer by hand, one 16-datum DEST face-row per op — 64 ops per tile, with
the face de-interleave and the row stride recomputed in software per op. It works, and it is
the only way to get a narrow row out of the packer itself, but tt-llk measures it at
1439–2418 cyc/tile against 77.4 for the hardware path.

Here the hardware does the untilize at full speed and iDMA only moves 32 rows afterwards.
Three consequences beyond the cycle count, all from iDMA addressing L1 by the **byte** where
RV_PACR's output address is **16-byte granular**:

| | RV_PACR | this path |
|---|---|---|
| face de-interleave | software `remap()` per op | done by the HW untilize |
| minimum narrow width | 8 datums (16-bit fmt), 16 (8-bit fmt) | none — see `SubFaceWidths` |
| width not a multiple of 8 | writes a full 16-datum face-row that spills into the next row, fixed by a second pass | nothing spills; one pass |
| ops issued | 64 per tile | 32 per 32-row block, whatever the width |

The cost is a second L1 buffer: the compaction is not done in place. In the real use case the
dense buffer is what the consumer reads anyway.

## The two engines

Selected per run by `engine_mode`. Both produce byte-identical output and `EngineParity`
verifies both, which is what makes the performance comparison like-for-like.

| mode | engine | |
|---|---|---|
| 0 | **iDMA gather** | one transaction per row, addresses from the address generator, fanned out over all 8 VCs, one drain at the end. **The proposal.** |
| 1 | NOC per-row | one stateful NOC read per row from this core's own L1. **The current workaround, and the bar to beat.** Stateful means only the addresses change per call, so it is the cheapest NOC read available — not a straw man. |

The two wire values are an `EngineMode` enum in `kernels/narrow_row_engine_mode.hpp`, included by both the host test and the kernel. It is a host/device contract carried in one uint32 runtime arg, so two independent lists of constants would agree until someone added a third engine — and then compile cleanly on both sides while taking the wrong branch on device.

The addressing follows `quasar_examples/quasar_idma/kernels/idma_1d_strided_example.cpp`
exactly: set the address generator's base and inner loop once, then `push_both_addrgen_0()` +
`issue_cmdbuf_0()` per row. All four addrgen examples issue once per address — the 2D, face and
interleaved loops change *which* address is yielded, not how many issues it takes — so the
address generator cannot amortise per-issue cost, and ~8.7 cyc/row is the floor. What this
kernel adds over the examples is the iDMA channel autoincrement (`req_vc_inc_en`), which
neither iDMA example uses and which is worth 3.6× at 512 B/row.

### Two things that are easy to get wrong

- **`MAX_BYTES_IN_PACKET`.** `CMDBUF_RESET` zeroes it to its rdl default of 0, which means
  *never split*. Fan-out round-robins **packets**, so a kernel that resets the command buffer
  and forgets to reprogram this gets no split at all and every channel knob goes silently
  inert — the tell is all channel counts landing within a couple of percent of each other.
- **`get_read_ptr()` is a byte address on DM but 16-byte units on TRISC** (`cb_addr_shift`),
  and on Quasar DM it carries `MEM_L1_UNCACHED_BASE`. The NOC API strips that alias itself;
  the overlay cmdbuf API does not. Hence `l1_phys()` in the kernel.

## Development measurements (emu-quasar-1x3) — any-len NOC baseline, see above

A 23-width sweep was run during development; it is not part of the shipped test, which keeps
38 runs of correctness coverage. Two of three curves are flat, and that is the whole result:

**The workaround is payload-blind.** 781.6–790.2 cycles for a 32-row block across a **256×
span in payload** — 24.5 cyc/row at every width. Entirely issue-bound; the bytes are free. So
its cost is ~785 cycles no matter how wide the matrix is, and iDMA's advantage cannot grow
with width — it narrows slightly, 2.87× at 16 B/row to 2.51× at 504 B.

**iDMA at 8 channels is also flat.** 273.6–313.4 cycles over the same range, 8.6–9.8 cyc/row.
At 512 B/row the data time spread over 8 VCs is only 4 cyc/row, well under the ~9-cycle issue
floor, so it should stay flat to roughly 1250 B/row.

| B/row | junk | NOC | iDMA 1 ch | iDMA 8 ch | speed-up |
|---|---|---|---|---|---|
| 2 | 96.9% | 786.2 | 294.5 | — | 2.67× |
| 16 | 75.0% | 785.7 | 273.6 | 273.8 | 2.87× |
| 64 | 0.0% | 785.4 | 279.9 | 279.5 | 2.81× |
| 128 | 0.0% | 783.8 | 345.9 | 278.0 | 2.82× |
| 256 | 0.0% | 783.6 | 601.2 | 295.2 | 2.65× |
| **504** | **1.6%** | **787.5** | **1120.9** | **313.4** | **2.51×** |
| 512 | 0.0% | 787.0 | 1117.4 | 313.2 | 2.51× |

**Always fan out.** One channel is flat only to a ~100 B knee, then grows at **15.9 B/cyc —
one iDMA VC's 16 B/cyc** — and crosses the workaround at **~347 B/row**. On a 504 B row a
one-channel implementation runs at 0.70×: slower than the code it replaces. Eight channels is
never more than 0.4% behind one channel anywhere in the grid, so there is no threshold to tune.
`EngineParity` still covers one channel, because it must be *correct* even though it must not
be shipped.

**Byte-granular placement is free.** 66 and 70 B rows (odd datum widths, destinations at
2 mod 4) cost 8.70 and 8.55 cyc/row against 8.56 and 8.52 for the adjacent even widths.

### What these numbers do not say

- **The consumer-side saving is not counted.** All of it is producer-side compaction. The
  workaround makes the consumer issue 32 reads; this path leaves it one.
- **The NOC baseline is a local L1 loopback**, not a remote consumer across the NoC. It
  measured perfectly payload-blind, so the cost is all per-issue and a remote pipelined read
  should pay the same — but that is not measured.
- **Whole-operation, not step.** Using tt-llk's steady-state 77.4 cyc/tile for the pack, the
  *whole* narrow-row untilize improves ~1.5× at eight tiles wide and ~2.5× at one — iDMA helps
  most where narrow-row matters most, because the pack is cheap there.
- **Emulator, not silicon.** The ratio should hold, since both engines are issue-bound and
  that is structural, but absolute counts should be re-taken before they enter a model.

### Check `junk%` before adopting this at all

At 32×252 only 1.6% of the padded row is waste. If the consumer's row stride is negotiable it
can read the padded buffer whole and discard 256 bytes, which beats both paths and costs
nothing to build. Compaction is only unavoidable where the junk share is large — 75–97% at one
tile wide, which is also where iDMA's margin is best.

## A scatter-list variant was tried and dropped

An alternative gather issues **one** transaction and lets the hardware walk a 32-entry address
list in L1. It is ~3% faster and does **not** reliably produce correct output, so it is not in
this test. At 8 channels the source index sequence comes out as

```
0 x16, 16, 17, 18, ... 31        with unwritten = 0
```

Every destination slot is written exactly once — nothing is overwritten, and the destination
auto-increment is correct. What fails is `SCATTER_INDEX`: it does not advance for the first 16
entries (16 × 8 B = 128 B, one list fetch block), so those destinations all receive entry 0's
offset, then the index jumps to 16 and tracks correctly. One channel is correct; the failure is
independent of the transfer geometry (`last_tile_w = 32`, with no compaction at all, fails
identically). Worth filing against the overlay separately.

The instrumented version that produced this — a mapping dump that inverts the output back to
the source position of every datum — is in git history on this branch.

## Running

```bash
export ARCH_NAME=quasar
export TT_METAL_SIMULATOR=<path to an emu-quasar-* build>

TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/unit_tests_data_movement \
  --gtest_filter="*QuasarNarrowRowUntilize*"
```

Run the **whole** `*Quasar*` suite before trusting a green result. This kernel shares
`addrgen_0` on core {0,0} with the addrgen and im2col tests, which program outer-loop,
face-size and banking registers it does not; a reset that inherited their state would pass
under the filter above and fail only in the full suite. (`HostHugepagePcieLoopback`, from
main, hangs on this emulator — exclude it with `:-*HostHugepagePcieLoopback*`.)

38 runs. The 1x3 emu has no fast-dispatch cores, hence slow dispatch.

| test | |
|---|---|
| `WidthAndTileRowSweep` | `ct_dim` 1/2/4/8 × `last_tile_w` 8/16/24/32 — the layout contract |
| `SubFaceWidths` | `last_tile_w` 1/2/3/4/12/20 at `ct_dim` 1 **and** 2 — below the RV_PACR floor. `ct_dim` 1 is the case that tests the claim: it produces 2 B to 40 B rows, where `ct_dim` 2's leading full tile would keep every row at 66 B or wider. The odd widths make `matrix_w` odd, so every other destination row starts at a 2 mod 4 byte offset |
| `EngineParity` | iDMA at all-channels and 1 channel, and the NOC workaround, at 32 / 70 / 504 B rows, plus one over-range channel request that exercises the kernel's clamp. 70 B (`ct_dim` 2, `last_tile_w` 3) puts every odd destination row at a 2 mod 4 offset, so the NOC arm cross-checks the byte-granular placement instead of only covering 8-byte multiples |
