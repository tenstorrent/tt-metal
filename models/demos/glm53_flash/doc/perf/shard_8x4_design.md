# Sequence-sharded DSA pool on the 8x4 Galaxy mesh (TP32): design

Decision 2026-11-06 (user): resident KV on all 32 chips, port the 2x4 shard "the full thing". Today
`kernels/mla/dsa_shard.hpp` static-asserts `LINE in {1, 4}` (the TP2 pair / the TP8 2x4 mesh) and the exchange is
2 rounds of all-gather + 2 rounds of reduce-scatter with a combine after each, the merge over NCHIPS = 2 LINE lists.

## 1. Geometry (shard_plan.Geometry(8, 4))

| quantity | 2x4 (TP8) | 8x4 (TP32) |
|---|---|---|
| rank r | 4 row + col | 4 row + col (row-major, rows 0..7) |
| pair axis (axis 0) length P | 2 (the column pair) | 8 (the column) |
| line axis (axis 1) length L | 4 (the row) | 4 (the row) |
| NCHIPS = P L | 8 | 32 |
| heads per chip (hpc) | 8 = [8 r, 8 r + 8) | 2 = [2 r, 2 r + 2) |
| head tile g (32 heads) | mesh row g = ranks [4 g, 4 g + 4) | mesh rows 4 g .. 4 g + 3 = ranks [16 g, 16 g + 16) |
| tile(r) | pair_pos(r) | pair_pos(r) // (P / 2) |
| PIECE_ROWS (a chip's q / o rows in a tile) | 8 | 2 |
| pieces per tile | 4 (= L) | 16 (= NCHIPS / 2) |
| pass 0 / pass 1 | the other / own tile, 32 heads each | unchanged (the attention compute shape is the same) |
| AG1 | axis 0 pair: 2 blocks | axis 0 column: 8 blocks (block ag_blk(r) = line_pos P + pair_pos) |
| AG2 | axis 1 row multicast of the 2-block bundle -> 8 blocks | row multicast of the 8-block bundle -> 32 blocks |
| q assembly (ix_dm1) | 2 tiles x 4 pieces of 8 rows | 2 tiles x 16 pieces of 2 rows |
| RS1 | axis 0 pair exchange of the whole pass-0 partial (36 fp32 pages), combine(own pass 1, landed) | axis 0 column REDUCE-SCATTER: destination row i' gets the 8-head slice [8 i', 8 i' + 8) of the 64-head partial (rows 8 (i' % 4).. of the pass whose tile is i' // 4); the landing chip combines its own slice + 7 landed |
| RS2 | axis 1 row scatter of 8-row slices (4 destinations), combine own + 3 | axis 1 row scatter of 2-row sub-slices of the column-combined 8-head slice (4 destinations: owner (i, j) gets heads [8 i + 2 j, +2)), combine own + 3, finalise o = O / l |
| merge helpers | NCHIPS = 8 chunk cores | 32 chunk cores (cl >= 32 per lane: lanes <= 2 at 64 chunk cores -> t <= 16 on the shard) |

So on 8x4 the TP8 "RS2" machinery (per-destination scatter of 8-row slices, N-way combine in column order) becomes
RS1 along the column with 8 destinations, and a new 2-row slice format serves RS2 along the row. RS1's "pair peer"
(`peer = 1 - pair_pos`, patched per chip) disappears: every column chip is a destination.

## 2. Formats

- AG block (`AG8` today: 10 pages = 4 q pages of 8 rows x 16 tiles + 5 candidate pages + 1 pad): the 2-row q piece
  needs 16 tiles x 2 rows x 32 el x 2 B = 2 KB = 1 page -> `AG32`: 1 q page + 5 candidate pages (6 pages; keep pairs:
  6). y_ag = 32 blocks x t x 6 pages = 384 KB per token per chip.
- q piece runs: rows 2 j, 2 j + 1 of a 32x32 tile = in faces (2 j / 16) 2 + {0, 1}: run A = face (2j/16)*2 at
  ((2 j) % 16) 16 elements, 32 elements; run B = the next face. `slice_off(j, tb, rows=2)` generalises the 8-row form
  (run A at (j rows / 16) (tb / 2) + ((j rows) % 16) (tb / 64), tb / 32 bytes; run B = + tb / 4).
- RS1 slice (8 rows of each of the DT = 16 O tiles (fp32, otb = 4096) + ST = 2 stats tiles): the TP8 RS2 slice format
  (rs2_layout: 16 x 1 KB + 2 x 1 KB = 18 KB -> 9 pages, RS2_PPC = 10). x_rs1 = 8 blocks x t x 10 pages.
- RS2 slice (2 rows): 16 x 256 B + 2 x 256 B = 4.5 KB -> 3 pages (RS2B_PPC = 4, pairs). x_rs2 = 4 blocks x t x 4 pages.
- The combine tile on the landing side keeps the full 32-row tiles with junk rows (as today): the own slice is written
  into rows [8 i, 8 i + 8) (RS1) / [8 i + 2 j, +2) (RS2) and the landed slices overwrite the same rows; combine over
  the whole tile is harmless (junk rows are never read: the finalise takes rows [8 i + 2 j, +2)).

## 3. Streams (tp_dsa_shard._new_streams) and fabric

- ag1: FabricStreamAG axis 0, n = 8, local copy, y = 32 blocks; ag2: bundle = the column's 8 blocks (chunks = 8 x
  ag_chunks) along axis 1. ag_wait unchanged (producers per token). Packet headers: FS_NHDR x 2 directions = 8 <= 12.
- rs1: FabricStreamScatter axis 0 (8 positions, 7 destinations per chip, up / down hops <= 7), x = x_rs1 (8 blocks),
  y = y_rs1 (8 blocks: block = the source's position); rs2: FabricStreamScatter axis 1 (4 positions) with the 2-row
  format. Both gated by the AR lanes' link 1 as today; the link audit's (link, direction) keys are unchanged.
- The 1D fabric lines along the 8-axis (axis 0) exist on the 8x4 (GLM53_FABRIC=1d; fabric_ar's lanes already use
  "4 connections on the 8-axis").

## 4. Kernels (the 2x4-specific code)

| file | 2x4-specific | 8x4 change |
|---|---|---|
| dsa_shard.hpp | LINE, NCHIPS = 2 LINE, PIECE_ROWS = 32 / LINE, pair_pos / line_pos / ag_blk, AG8 pages, RS2 10 pages, qasm | MLA_DSA_SHARD_PAIR (P) + LINE (L): NCHIPS = P L, hpc = 64 / NCHIPS, PIECE_ROWS = 32 / (NCHIPS / 2), tile(r), RS1_ROWS = 64 / P, RS2_ROWS = RS1_ROWS / L, AG32 pages, rs1 / rs2 page maps, slice_off(j, tb, rows) |
| lat_dm1.cpp | writes the chip's 8-row q piece into x_ag (agx_) | 2-row piece (1 page) |
| ix_dm1.cpp | assembles 2 tiles from 4 pieces (slice_off, qasm) | 16 pieces of 2 rows |
| ix_dm0.cpp | the merge over NCHIPS = 8 helper lists | NCHIPS = 32 (helpers = chunk cores 0..31; cl >= 32) |
| att_dsa_shard_dm1.cpp | RS1 = the pair message (36 pages) + combine(own, landed); RS2 = 8-row slices to L peers + combine own + L - 1 | RS1 = 8-row slices to P - 1 column peers (the pass / rows selected per destination) + combine own + P - 1; RS2 = 2-row slices to L - 1 row peers + combine own + L - 1 |
| att_c.cpp | combine (LINE uses: the round-2 count) | counts P / L |
| att_dsa_shard_dm0.cpp | pass tiles from q_asm (unchanged) | - |

## 5. Phases and checks

1. Host planner: `Geometry(8, 4)` (pair_axis 0 of 8, tile(r), tile_ranks, slices), `streams` (rs1 as a scatter),
   `block` / `holders` / `header_budget`, `emulate_exchange` with the column reduce-scatter + the row scatter, checked
   against the direct 32-way combine; `helper_merge` over 32 lists vs `serial_merge`. New host-only test
   (torch only, no ttnn): tests/test_shard_plan_8x4.py. No device.
2. Kernels per section 4 behind `MLA_DSA_SHARD_PAIR` (default 2 = today's builds unchanged).
3. `tp_dsa_shard.py` streams / audits for P = 8; `mk_mla.py` shard_lanes_check (helpers >= NCHIPS).
4. Device unit on the 8x4 (the shard job on a short context, numerics vs the unsharded TP32 ids 785 1196 702 ...),
   then the e2e rows: 32 K x 32, 128 K x 8 / 16 / 32, 512 K x 4-32, 1 M x 4-16.

## 6. Device results (2026-11-06, chain_mk126 on l2-34; 45 layers, tile cache v3, TP prefill)

| Context | Users | Decode | TTFT | Notes |
|---|---|---|---|---|
| 4 K | 4 identical | 26.8 ms = 37.4 tok/s/u | 8.6 s | ids 785 1196 702 3267 ... (the 2x4 sharded ids); unsharded TP32 18.8 ms / 4.2 s |
| 32 K | 4 identical | 27.2 ms = 36.8 tok/s/u | 35.4 s | ids = the 2x4 sharded 32 K row exactly; unsharded TP32 20.5 ms / 35.1 s |
| 128 K | 8 distinct | 43.3 ms = 23.1 tok/s/u | 290 s | 8 x 128 K resident (the unsharded pool holds 4) |
| 512 K | 4 distinct | 28.9 ms = 34.6 tok/s/u | 626 s | 6 GB trace region; the unsharded pool OOMs at 512 K x 1 |
| 1 M | 1 | 19.3 ms = 51.7 tok/s/u | 345 s | 8 GB trace region, 1,402 / 1,649 MiB DRAM free per bank after compile |
| 1 M | 4 distinct | 28.2 ms = 35.4 tok/s/u | 1,379 s | 4 x 1 M resident, 1,534 / 1,516 MiB DRAM per bank after gc |
| 32 K | 16 distinct | 68.3 ms = 14.6 tok/s/u | 143 s | with the shared m (section 7); 2 token lanes at t = 16 (32 helpers per lane): the decode perf item |
| 128 K | 16 distinct | 71.3 ms = 14.0 tok/s/u | 583 s | with the shared m; 16 x 128 K resident (the unsharded pool holds 4) |

Prefill runs 3.0-3.7 K tok/s flat from 32 K to 1 M (the TP prefill path, unchanged by the shard). The exchange costs
+7-8 ms per decode step at 4 K / 32 K against the unsharded TP32 (the optimisation item), and the per-user decode
at 128 K x 8 (43.3 ms) is the first measurement of the sharded split-K at 8 token lanes on 32 helpers.

## 7. The L1 floor (B = 16): closed

`program.cpp validate_circular_buffer_region` ("Statically allocated circular buffers in program N clash with L1
buffers ... 1253248 ... 1253888") came from the MoE decode program of `tp_dsa.TPMoE` -- its static CBs are the same
961,920 B at every row count -- against a floor that every sparse_dsa layer build dragged ~4 KB lower (each layer's
MLA / MoE call allocated and freed its own intermediates; the small ones fragmented the freed holes, the large ones
landed below). The fix is in the allocation lifecycle, not the CBs:

- `GLM53_MOE_SHARE_M` (default on, 0e928122d73): one shared-expert intermediate `m` per stack (10 x 2 KB per bank);
- `GLM53_DSA_POOL` (default on, faa901ac4c2 .. bbe4485aeea): the sparse_dsa MLA / MoE programs build under an
  `AllocPool` of their own (`stack_model._dsa_pools` / `_dsa_layer`; s1, the MLA -> MoE hand-off, from its child pool;
  the alias mode mirrored from the step pool) -- one L1 set for every layer, nothing allocated or freed per layer or
  capture; the floor sits flat at 1293248 (B = 4) / ~1261000 (B = 16) across all 11 layers.
  Persisting the hand-offs in a dict (tried first) made the MoE build see the MLA set and clashed; sharing the KDA
  stacks' pool changed its prefix-alias bases (the v1 two-pool B = 16 row drifted at id 7) -- both superseded.

Validated at HEAD: 32 K x 4 ids identical (27.1 ms), 32 K x 16 (68.7 ms); the longer rows of section 6 were measured
before the pool landed and are re-run at HEAD for the record. Section 6's 4 K row agrees with the 2x4 shard on the
first 9 ids (the contract rows are the 32 K ones, identical 24/24).
