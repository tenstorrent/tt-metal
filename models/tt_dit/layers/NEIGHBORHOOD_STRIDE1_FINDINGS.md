# Stride-1 neighborhood attention: what the chunk lever can and cannot buy

Measured 2026-08-25 on 4x8 Blackhole, LTX-2.5 DiffVAE decode, 145 frames at 1920x1088,
ring/2-link, TP4, deterministic stages W-sharded, `bricked_sp_w_sharded` stage 5.
Every number below is `test_decode_wsp_timing` with `DIFFVAE_BLOCK_PROF=1`.

## 1. The uploaded interior-mask table is WRONG at stride 1

Against `neighborhood_reference.neighborhood_attention_3d`, volume (12,16,16), window 11,
stride 1, chunk (1,1,1) -- a volume large enough to have genuine INTERIOR bricks:

| mask path | PCC vs torch |
|---|---|
| uploaded host table (`_build_regime_masks`) | **0.914** -- fails the op's own 0.99 bar |
| generated on device (`fill_mask_tile`) | 0.996 -- bf16 noise |

It agrees only when the volume is small enough that every brick is boundary-clamped and the
interior regime never fires, which is why the unit tests (volumes <= (8,32,32), window 3) never
caught it. The GNA path is unaffected: there `stride % brick == 0`, the window origin snaps to a
brick boundary and every chunk shares one window -- the regime the table was built for.

**Generation is free at chunk (1,1,1):** 33981 ms generated vs 34121 ms uploaded. So the fix costs
nothing -- generate at stride 1 rather than upload.

## 2. Per-brick masks are correct, and they do not pay

`DIFFVAE_NA_CHUNK_BRICKS` + `DIFFVAE_NA_UNSAFE_CHUNK` decouple the chunk from the stride;
per-brick masks (`reader_arg::per_brick_mask`, `compute_arg::mask_subblock_stride`) make the
result independent of the chunk, verified against the torch oracle:

    chunk (1,1,1)  PCC 0.99639672
    chunk (2,2,2)  PCC 0.99639672
    chunk (2,1,1)  PCC 0.99639672

Identical to eight decimals -- chunk is a pure scheduling knob, as intended. But:

| stride-1 config | mask tiles/query | K/V tiles/query | sum | decode |
|---|---|---|---|---|
| chunk (1,1,1) | 175 | 175 | 350 | 34.0 s |
| chunk (2,2,2), broadcast mask (WRONG output) | 36 | 36 | 72 | 11.3 s |
| chunk (2,2,2), per-brick mask (correct) | 288 | 36 | 324 | 36.4 s |

Cost tracks `(mask tiles + K/V tiles)` per query brick at roughly 12 ms/tile/block. A bigger chunk
divides the K/V gather by the bricks sharing it, but a CORRECT mask multiplies by them: every
brick needs its own tile for every slot of the shared box. The two cancel. The 11.3 s was only
ever reachable with the wrong mask.

## 3. What would actually win: per-brick SUB-BOX

A query brick's window is ~54 bricks of the 288-brick shared gather. The other 234 slots are
outside its window entirely -- and today we generate a -inf mask tile for each AND matmul against
it. Iterating the flash loop over each brick's own sub-box instead:

    54 mask + 36 K/V = 90 tiles per query   against 350 at chunk (1,1,1)

~3.9x, i.e. plausibly ~8.7 s at exact stride-1 quality -- under the artifacty GNA path's 9.2 s.
This cuts generation and matmul by the same factor, which is why it beats tuning the chunk: it
attacks the term that scales with the shared box rather than the one that scales with the gather.

The obstacle is that one `matmul_blocks` call currently spans all query bricks x all slots, with
the online-softmax state carried across that whole block; per-brick sub-boxes means restructuring
that loop, and the flash bookkeeping is the delicate part.

## Knobs added for this investigation

    DIFFVAE_NA_CHUNK_BRICKS=t,h,w   force the chunk, in BRICKS (brick is (2,4,4) sites)
    DIFFVAE_NA_UNSAFE_CHUNK=1       lift the plan's chunk==stride check
    DIFFVAE_NA_PER_BRICK_MASK=0|1   force the mask mode; defaults on when chunk != stride
    DIFFVAE_NA_HALO_TOPOLOGY=ring   retest the halo on ring once neighbor_pad is fixed
    DIFFVAE_NA_HALO_LINKS=n         halo link count only (dead end -- not the ring hang)
    DIFFVAE_NA_HALO_PERSISTENT=0    halo without the persistent buffer (dead end)

`DIFFVAE_NA_UNSAFE_CHUNK` without `PER_BRICK_MASK` is a PERF PROBE ONLY: it reproduces the
broadcast-mask bug deliberately, and its frames are not shippable.

## 4. Measured against the reference at the SAME stride (added after the sections above)

`DIFFVAE_GNA=0` puts the reference executor at stride 1 too, which is the only honest comparison
for exact NA. Its block picker chooses `_blk=(6,8,4)` (t=78) and `(7,8,4)` (t=77) -- 192 and 224
queries per gather, box-minimising rather than volume-maximising.

| stride-1, per block | reference | ours, chunk (1,1,1) |
|---|---|---|
| K/V staging | 218.4 ms (kv-allgather 93.5 + kv-wrow 124.9) | 203.8 ms (halo+brick-permute) |
| the kernel | **672.7 ms** (fused-sdpa) | **3319.6 ms** (neighborhood-sdpa) |
| attention total | 1237.3 ms | 3766.9 ms |
| keys/query | **21** | **175** |
| decode TOTAL | **13.8 s** | 34.0 s |

Staging is a wash. The kernel is 4.9x slower while processing 8.3x more keys per query -- so per
key this op is already the more efficient of the two. The gap is entirely that we group 32 queries
per gather where they group 192.

Two things the reference does that this op does not, both in
`kernels/windowed_loop_geometry.hpp` and `kernels/dataflow/windowed_mask_gen.hpp`:

  * **K-range narrowing.** `windowed_k_chunk_range()` returns the contiguous [k_lo, k_hi) that the
    Q chunk's windows actually touch; everything outside is never read, never matmul'd, never
    masked. We visit every gather slot for every query brick.
  * **Mask generation on the WRITER core**, in parallel with the reader streaming K/V and the
    compute doing math. We generate in the reader, serially, on the critical path.

### What each fix is worth (measured or derived from the table above)

  bigger chunk (4,2,2)   175 -> 22.5 keys/query, matching their 21. THE dominant term.
                         Already implemented and PCC-verified; needs the mask fixes to pay.
  copy, don't generate   at stride 1 the interior mask is translation invariant:
                         mask(brick b, slot s) depends only on (s - b). ~441 distinct tiles for
                         the whole plan against the ~25M tiles/block we currently generate.
                         A shifted L1 copy is ~10x cheaper per tile than fill_mask_tile's
                         per-element window arithmetic (~16 cycles/element measured).
  narrowing              360 -> 175 mask tiles per brick at chunk (4,2,2); ~1.65x, NOT the 5.3x
                         first estimated (a BRICK's window union is 7x5x5=175 bricks, not the
                         6x3x3=54 of a single QUERY).
  mask gen on writer     hides what remains behind compute. Alone it only gets 34 s -> ~24 s,
                         because the writer then becomes the critical path.

Rough combination of the first three: compute ~230 ms/block + mask ~218 ms/block ~= 450 ms/block,
under the reference's 672.7.

## 5. Chunking cannot help this op -- the mask is the whole bill

Five CORRECT-output configs at stride 1, 145 frames (per-brick masks throughout):

| chunk | mask tiles/brick | fetch sites/query | decode |
|---|---|---|---|
| (1,1,1) | 175 | 175 | 34.0 s |
| (2,1,1) | 200 | 100 | 34.9 s |
| (2,2,2) | 288 | 36 | 36.4 s |
| (8,1,1) | 350 | 43.8 | 36.4 s |
| (2,2,2) broadcast -- WRONG output | **36** | 36 | **11.3 s** |

Time tracks mask tiles per brick and NOTHING else. Fetch swings 5x (175 -> 36) with no effect,
so the gather was never the bottleneck and amortising it across a bigger chunk buys nothing: a
correct mask costs one tile per (query brick, gather slot), and a bigger chunk raises that count.
The only fast configuration is the one that reuses a single wrong mask.

Two candidate explanations for the per-tile cost were tested and BOTH are dead:

  * content computation -- uploaded copy vs device generation at chunk (1,1,1): 34121 vs 33981 ms.
    Copying is not cheaper, so the window arithmetic inside fill_mask_tile is not the cost.
  * CB handshake granularity -- chunk (2,1,1) keeps cb_mask at 16 tiles (against 64 at (2,2,2))
    and still lands at 34.9 s. Coarser batching is not the cost either.

What remains untested is what the reference actually does: generate the mask on the WRITER core
(`dataflow/windowed_mask_gen.hpp` is included by writer_interleaved.cpp), overlapped with the
reader streaming K/V and the compute doing math -- hiding the cost rather than reducing it. This
op generates in the reader, serially, on the critical path.

### Where that leaves the two executors

  exact NA      reference 13.8 s | this op 34.0 s. Narrowing (350 -> 175 tiles/brick at an
                T-elongated chunk) projects ~20 s -- worth doing, not enough to win. Beating 13.8 s
                needs writer-side mask generation on top.
  GNA           this op 9.2 s at 9.00 keys/query | reference 8.8 s at 15.2, on a stride never more
                aggressive than the reference's own on any axis or band. This is where the bricked
                layout pays, and it is shippable today.

### Why T-elongated chunks, if narrowing is built

Gather slots linearise T-major (`slot / (G_h*G_w)` gives t). A chunk elongated ONLY in T keeps the
gather's h/w extent fixed -- (10,5,5), (14,5,5), (22,5,5) for chunks (4,1,1), (8,1,1), (16,1,1) --
so each brick's window sub-box spans full h/w and a contiguous t range, i.e. a CONTIGUOUS slot
range, exactly the shape `windowed_k_chunk_range()` returns in the reference. Elongating in h or w
grows the gather to (.,6,6) and the sub-boxes become strided, which is what makes narrowing
invasive there.

## 6. CORRECTION to section 5: the cost is mask CONTENT, not mask writes

Section 5 concluded that time tracks mask tiles written. That was wrong, and the experiment that
settles it is `DIFFVAE_NA_MASK_MEMSET_ONLY=1` -- same tile count, same CBs, same matmul, only
`classify_brick` + `fill_mask_tile` replaced by a constant fill:

| chunk (2,2,2), stride 1, per-brick | decode |
|---|---|
| real masks | 36.4 s |
| identical tiles, constant memset | **10.1 s** |

**26 of the 36 seconds is mask content computation -- 72% of the decode.** Writing the tiles is
nearly free. Two earlier conclusions fall:

  * "copy is not cheaper than generate" (34121 vs 33981 at chunk (1,1,1)) could never have shown
    a difference: the mask is only 175 of 875 tiles moved there, so even a large saving is buried.
  * "chunking cannot help" is false. With content removed, chunk (2,2,2) runs at 10.1 s -- BELOW
    the reference's 13.8 s -- at 36 fetch sites/query.

### The fix: precompute the mask, do not generate it per tile

At stride 1 the interior mask is TRANSLATION INVARIANT: mask(query brick b, key brick s) depends
only on (s - b), because shifting the query brick by one brick shifts its window by exactly one
brick and the gather slots ARE bricks. So the whole plan needs ~441 distinct tiles (the relative
offset range), not the ~25M tiles/block currently generated. Generate once, then every (brick,
slot) is an L1 copy -- which the memset run shows is nearly free.

Projected: chunk (2,2,2) + precomputed table ~= 10-12 s against the reference's 13.8 s, at
identical exact-NA quality. Narrowing becomes secondary -- it removes tiles that would be cheap
copies anyway.

The correctness oracle already exists: per-brick masks give identical PCC (0.99639672) across
chunks (1,1,1), (2,1,1) and (2,2,2), so a copy-based implementation can be checked against it.

## 7. Root cause: per-word volatile loops, and a regime table that is a GNA construct

The mask cost is not the window arithmetic. Every mask tile -- copied, generated or memset -- is
filled by a 512-iteration volatile word loop on the RISC, and THAT is the cost:

| fill | work per tile | 145f decode |
|---|---|---|
| memset | 512 volatile stores | **11.0 s** |
| copy from the L1 resident set | 512 loads + 512 stores | 34.1 s |
| generate (fill_mask_tile) | 1024 elements + stores | 34.0 s |

Copy and generate are indistinguishable because both are dominated by the loop, which is why the
copy-vs-generate comparison in section 6 could not resolve anything.

The reader's own comment names the trap: staging a regime's tiles in L1 (`cb_resident_mask`) to
avoid re-reading DRAM forces the per-slot fill to be a word loop, because *NOC refuses a local
source and destination*. The optimisation defeats itself.

**Landed:** the per-slot fill now `async_read`s straight from the DRAM mask tensor into cb_mask --
DMA that overlaps with compute, the same path that already fetches four K/V tiles per slot,
covered by the existing async_read_barrier. Safe: GNA 9326 ms against 9313 ms baseline, no
regression, no quality change (GNA touches only 54 mask tiles per chunk, so mask was never its
bottleneck).

**But it is dormant at stride 1**, which measured 34119 ms against 34121: `chunk_regime()` returns
NO_REGIME there, so the code never reaches the uploaded path and generates every tile. The regime
scheme requires every site in the group to share one window classification -- natural under GNA,
where the group IS one window, and impossible at stride 1 where every site has its own origin.
That is also why the uploaded table is numerically WRONG at stride 1 (PCC 0.914): one shared
pattern per regime cannot describe bricks whose windows sit at different origins.

### What is actually needed at stride 1

A RELATIVE-OFFSET table, not a regime table: mask(query brick b, key brick s) depends only on
(s - b), so ~175 distinct tiles at chunk (1,1,1) (441 at chunk (2,2,2)) describe the whole plan
against the ~25M generated per block. Build host-side, upload to DRAM, index by (s - b) in the
reader, and deliver it through the NOC read that is now in place.

Ceiling, from the memset floors: ~11-13 s at exact NA against the reference's 13.8 s -- with no
chunking, no narrowing and no matmul_blocks changes. Chunk (1,1,1) floor 11.0 s, chunk (2,2,2)
floor 10.1 s, so chunking buys almost nothing once the mask is cheap.

## 8. Relative table: LANDED, correct, and 2x faster -- with two things left

Implemented: `_build_relative_masks` (host, 175 tiles keyed on key_brick - query_brick),
`relative_mask` compile arg, and a reader path that `async_read`s the tile straight from DRAM into
cb_mask. No L1 staging, so the word loop is gone. Both the per-brick and the normal (chunk 1)
paths go through `relative_table_index()`.

**Correctness fixed.** Uploaded mask against the torch reference at stride 1, volume (12,16,16):

    old regime table   PCC 0.914   (wrong -- keyed on the absolute slot, and the gather origin's
                                    brick phase is NOT constant: 75 distinct values at 1080p)
    relative table     PCC 0.99639672  -- identical to device generation, i.e. exact

It also works where no mask could be uploaded before: the regime sets are enumerated against one
shard origin, so the W-SHARDED path always generated. The relative table depends on neither the
gather origin nor the shard origin, so one upload serves every shard.

**Speed, 145 frames, exact NA:**

| | decode | note |
|---|---|---|
| before | 34.1 s | generated every tile |
| relative table, gate as written | 32.3 s | gate admits only ~10% of bricks |
| relative table, gate bypassed | **17.5 s** | every brick served (edges wrong -- diagnostic) |
| memset floor | 11.0 s | |
| reference at stride 1 | 13.8 s | |

Fitting `f * 17.5 + (1 - f) * 34.1` puts the gated run at f ~= 0.10.

### The two remaining items

1. `brick_window_is_unclamped()` rejects ~90% of bricks and should reject ~10%. Suspect the
   W-sharded halo: local w = 0 maps to global -8 through the negative shard_origin, so
   `first - half < 0` fires across the halo. Worth 32.3 -> ~19 s with correct output.
2. NOC reads are not free: 17.5 s against the 11.0 s memset floor, ~6.5 s of DMA cost with every
   tile coming from the table. Closing the gap to the reference needs this too.

`DIFFVAE_NA_TABLE_ALWAYS=1` bypasses the gate (edge bricks get the interior pattern -- wrong
frames, right timing). `DIFFVAE_NA_MASK_MEMSET_ONLY=1` gives the content-free floor.

### Gate: still the blocker, and re-deriving it did not help

Rewriting `brick_window_is_unclamped` to ask `window_origin_on_axis` directly -- origin ==
(site - window/2) at both ends of the brick -- left the decode at 32285 ms, unchanged. So the
predicate is still rejecting ~90% of bricks and the reason is NOT the hand-derived bounds.

Next suspect, untested: `extents.shard_origin`. It is read per chunk out of the gather-origin
table and reinterpreted through static_cast<int32_t>; if the T/H components are not what this
predicate assumes (0 for the W-sharded case), `first` lands out of range and every brick reads
as clamped. Instrumenting one chunk's shard_origin would settle it in a single run.

The prize is fixed and measured: `DIFFVAE_NA_TABLE_ALWAYS=1` (all bricks served) runs 17.5 s
against 32.3 s gated, so the gate alone is worth ~15 s.

## 9. After the relative table: the bound is the gather-slot walk, so two kernels

The relative table and the host-stamped interior gate landed. Decode `neighborhood-sdpa` is still
~737 ms for n=2 stage-5 calls (~370 ms each). The target for those two was ~400 ms **total**
(~200–250 ms each). Softmax is not that gap.

Measured on one W-shard of the stage-5 first band, host wall-clock,
`test_neighborhood_sdpa_components.py` (volume `(84,272,480)`, owned `(84,272,60)`, brick
`(2,8,2)`, window 11 → gather **147** bricks, **42840** query bricks). Origin stamp:
**31104 interior / 11736 edge** (~73% interior).

### Compute is ~20 ms. The reader walks 147 slots.

| probe | what it skips | ms | ns/slot |
|---|---|---|---|
| window11 (full) | nothing | **528–532** | 84 |
| skip_kv | K/V DRAM, still the 147-loop | 26 | 4 |
| skip_slots | the 147-loop, still CB handshake | 26 | 4 |
| skip_slots_drain | handshake only (compute drained) | 6–9 | 1 |

`window11 - skip_slots` ≈ 505 ms is the slot walk. `skip_slots - skip_slots_drain` ≈ 20 ms is
QK/softmax/PV. Drain/qk probes could not isolate this because they still waited on the reader
filling those CBs.

Windows 3 / 7 / 11 (query bricks fixed, gather 27 → 75 → 147) hold **76–84 ns/slot**. Slope is
~3.7 ms per gather brick. Time is walking slots, not dispatch or Q.

### Classify and the tight gather must not share a binary

~73% of bricks are unclamped interiors: one DMA gather along the relative table, no
`fill_mask_tile`. The other 27% are edges and need classify. Putting both loops in **one** ELF
mixes them in the I-cache:

| binary | what it walks | ms | ns/slot |
|---|---|---|---|
| mixed (classify + tight gather) | every brick | **~416** | 64 |
| interior-only (`path_mode=1`, tight gather) | every brick, interior loop | **~130** | **20** |
| edge-only (classify) | every brick, classify loop | **~400** | 64 |

Interior is 3× faster per slot when classify is not in the working set. That is the whole reason
for a second kernel: peel the two loops into two programs, skip-slots handshake the bricks the
other program owns, write the same output.

On paper, skip firing:

    0.73 × 130 ms  +  0.27 × 400 ms  +  handshake on the rest  ≈  **210–250 ms**

That is the ~200–250 ms per call the decode wanted.

### Split without skip is slower than the mixed kernel

`path_mode=0` already launches interior then edge. Each program still walks **every** brick
unless the skip condition matches. Then you pay both full walks:

    interior tight-all (~130) + edge classify-all (~400)  =  **~530 ms**

Worse than one mixed kernel (~416). The skip **path** is live — forcing it drops the wall:

| skip condition | window11 |
|---|---|
| none (current split) | **528–532 ms** |
| `if (true)` in **both** programs | **52 ms** (handshake only, both launches) |
| `if (true)` in **interior only** | **459 ms** (interior handshake + edge classifies all) |

52 ms ≈ 2 × skip_slots. 459 ms is edge-only. So handshake/continue is compiled and taken; the
**predicate** never matches the host-stamped interior bit / `skip_edge_token` (0 vs `0xFFFFFFFF`
in origin-table column 7). Round-trip of that table from DRAM is correct (31104 / 11736); the
kernel load at skip time is not seeing it. 0/1 compares and bool-returning helpers are DCE'd on
this RISC toolchain — that is why the token is `0xFFFFFFFF` and skip polarity is `if (x == 2)`
shaped like `skip_kv`, not `if (interior)`.

Until skip fires, keep the mixed kernel for speed, or land the predicate. The second kernel is
the I-cache split, not extra math.
