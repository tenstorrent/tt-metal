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

Re-validated on 2026-11-09 at fd64cbc8e88 .. 2482b4cf898 (the DSA `AllocPool` default; the TTFT column above is the batch's prefill back to back, as the e2e measures it): 32 K x 4 27.3 ms = 36.7 tok/s/u (TTFT 35.4 s, ids 24/24 = the table), 128 K x 16 distinct 72.0 ms = 13.9 (TTFT 583 s with the 262144 window ladder through the cache length; a 12 tok/s floor, no id gate at 16 distinct users), 1 M x 1 19.0 ms = 52.7 (TTFT 345 s, ids 24/24). 128 K x 8, 512 K x 4 and 1 M x 4 stand on the 2026-11-06 .. 08 measurements above (their HEAD re-runs: 50tok_plan.md). A 32 K x 4 row at the 1 M cache length (the served 1 M profile's: its 1 M ladder) gives 27.5 ms = 36.4, TTFT 38.2 s, ids 785 1196 702 3267 291 264 1293 57681 ... -- the reference for the served 32 K texts.

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

Validated at the pool-v3 commit bbe4485aeea (2026-11-09 01:41 / 01:57): 32 K x 4 ids identical (27.1 ms), 32 K x 16
(68.7 ms); the later re-validation of the rows is in section 6. Section 6's 4 K row agrees with the 2x4 shard on the
first 9 ids (the contract rows are the 32 K ones, identical 24/24).

## 8. Token lanes past 2: KPH chips per merge helper (2026-11-10, MLA_REV 64)

Every token lane held NCHIPS = 32 merge helpers among its chunk cores, so 64 chunk cores gave 2 lanes and t = 16 ran
8 tokens per lane in series (~4.4 ms per extra token per lane per step: the per-token select / merge handshakes and
root combines are latency, not bandwidth). `GLM53_DSA_SHARD_KPH` (galaxy default `auto`): helper h holds chips
h KPH .. h KPH + KPH - 1's candidate lists (2 staging pages, one histogram over the union per round, per-list tie
counts), so a lane needs 32 / KPH helpers -- `auto` = KPH 1 / 1 / 2 / 4 at 1 / 2 / 4 / 8 lanes; T_g and the per-chip
quotas equal KPH 1's. Lane-local slots TL = 4 when a lane holds <= 4 tokens keep the kept blocks / mailbox within the
2-lane bytes (one rise: 1 M at t 4, 32.3 -> 34.2 KB per owner). 16 lanes need 16 free lane cores (14 today).

| e2e (45 layers, greedy) | 2 lanes | KPH auto | ids |
|---|---|---|---|
| 4 K x 4 identical | 26.8 ms = 37.4 tok/s/u | 22.9 ms = 43.7 | identical |
| 4 K x 8 distinct | 41.2 ms = 24.3 | 29.2 ms = 34.3 | user 0 identical |
| 4 K x 16 distinct | 67.4 ms = 14.8 | 42.6 ms = 23.5 | user 0 equal through id 21, coherent |
| 32 K x 4 identical | 27.3 ms = 36.7 | 22.8 ms = 43.8 | identical 24 / 24 |

B 1 is unchanged (one lane: KPH 1; 4 K 18.9 ms = 52.8). At 4 K x 16 the unsharded TP32 pool runs 32.1 ms, so the shard
now adds 10.5 ms (35.3 with 2 lanes).

## 9. The merge window at B = 16 is the all-gather's bytes: the key-packed AG (2026-11-10, MLA_REV 68)

Eager probe, chip 0, B 16 (8 lanes x 2 tokens, KPH 4), pos 4095, both rev-65 RS knobs on (test_mk_backend_probe,
layer 3; marks are the max over the chip's cores of a kind unless noted):

| mark | site | us |
|---|---|---|
| owner_dm1 0 | att_dsa_shard_dm1.cpp phase start (after mhc_pre) | 59.9 |
| ix_dm0 1 / 2 | ix_dm0.cpp: the I1 hand-off / the 9 index tiles in | 84.1 / 85.6 |
| ix_dm0 3 | ix_dm0.cpp: the lane's 2 tokens selected (8 rounds + tie each), record counts in x_ag | 210.4 (216.7 max) |
| owner_dm1 1 | att_dsa_shard_dm1.cpp: the last chunk core's records in x_ag = the AG1 lane's last producer | 217.4 |
| ix_dm0 4, ix_dm1 3 | SEM_AGC: AG2's waiter, all 32 blocks of y_ag landed | 752.6 |
| ix_dm1 4 | the q tiles of the lane's 2 tokens assembled (behind the merge) | 819 - 824 |
| ix_dm0 5 | both tokens' T_g + per-core quotas broadcast | 871.3 |
| owner_dm1 2 | CB_ASEL written (every lane token) | 872.0 |

The 217 -> 872 us window is the all-gather (535 us, 82 %) then the global merge (119 us: 2 tokens x 12 sequential
steps -- start, 8 radix rounds, tie, quotas, the rank helper's per-core quotas -- ~5 us each); the local select (9
report / broadcast handshakes per token) runs before it. The AG is bytes on link 0: both rounds are line multicasts,
so the column's last hop carries 7 blocks (AG1) and the row's last hop 3 bundles of 8 (AG2) = 31 blocks of t x 6 pages
= 5.8 MB at t = 16 (~11 GB/s), and 5 / 6 of a block is 16-B (key, core) records whose core id only the record's own
chip reads.

`GLM53_DSA_SHARD_AGK=1` (opt-in; dsa_shard.hpp AGK): x_ag keeps the AG32 records (no producer change); the AG1 lane's
NCRISC packs each staged chunk in place (fs_read.cpp FS_KEYPACK: page 1 = the n = min(header, 512) keys in record
order, zeros after) and ships 2 pages per token; AG2 multicasts the packed bundles (y_ag = 32 blocks x t x 2 pages).
The count is not shipped: chip q's list of a sparse token holds exactly min(512, n_closed_local(p, q)) records. The
merge helpers read one key page per list; the helper of chip rank takes its per-core tie order from its own x_ag. T_g,
the per-chip / per-core quotas and the tie order are rev 65's bitwise (tt/shard_plan.py emulate_agk_launch,
tests/test_shard_plan_8x4.py). Staging CBs: AG1 4 x 6 pages (unchanged), AG2 48 -> 16 KB; no new semaphores.
Expected: the AG at a third of its bytes, ~535 -> ~180-200 us per DSA layer at B 16 (~-3.7 ms per 45-layer step at
4 K x 16), ~-170 us at B 8 and ~-85 us at B 4 (the bytes scale with t; the per-hop drains / hand-offs do not), plus
one key page per list instead of up to 5 record pages in the merge at >= 64 K. The next cut is the local select +
merge handshakes (~245 us at B 16: one report / broadcast per round carrying the lane's tokens).

Device (chain_mk162, e7f7dc8443f): probe B 16 pos 4095 4022.2 -> 3867.6 us (-155 us per DSA layer); the AG window
548 -> 431 us (-21 % for a third of the bytes). Both formats run t + 8 t = 144 staging batches and 144 chunk-end
signals per source per launch at t 16, so the AG is per-chunk latency bound (~2.7 us a chunk), not link bound.
e2e 4 K x 16: 41.0 ms (KPH 42.6); user 0's 128-token text is byte-identical to the 2-lane rev-63 run (67.4 ms, KPH 1)
and differs from the two rev-64 KPH runs, which had already left the 2-lane text. AGK and rev 64 take the same T_g,
quotas and tie order on the host (shard_plan.helper_merge / agk_merge, a page-level port of stream_lists fuzzed over
list lengths 0 .. 512); the knob-off kernels of rev 65 - 68 equal rev 64's (att_c TRISC1: two spill slots swapped).
The text alone does not pin a bitwise difference (user 0 has a handful of attractors across configs: the B 4 fold-
combine run lands on the same text); the deciding runs are a second AGK row, an AGK-off row of the same tree and a
T_g / quota dump per lane token (hist tensor words 8 + j) with AGK on vs off on one eager step.

`GLM53_DSA_SHARD_AGB=1` (on AGK): y_ag [32 x 32 rows, 64 t] -- one 32-row chunk per block holding its t tokens' packed
pages, the same page ids as AGK (no consumer change): AG1 signals once per block (fs_send.cpp FS_KP_XPER), AG2 ships
the block chunks in 8-page batches (3 slots: rev 65's 48 KB). Per launch at t 16: 16 + 32 batches (was 144) and 1 + 8
signals per source (was 16 + 128). Expected: the AG at ~48 / 144 of its per-chunk cost, ~431 -> ~190 us per DSA layer
at B 16 (~-2.6 ms per 45-layer step), ~-110 us at B 8, ~-55 us at B 4.

## 10. The shard tail knobs and the 8x4 defaults (2026-11-10, MLA_REV 68)

Each knob is its own kernel define / stream argument, so any subset composes: `GLM53_DSA_SHARD_COMB` the roots'
combine (att_c / att_dsa_shard_dm1), `GLM53_DSA_SHARD_RS1_EARLY` the roots' RS1 signals + the RS1 lane's gate
(fs_read.cpp FS_WAIT2), `GLM53_DSA_SHARD_AGK` the AG1 lane's pack (fs_read / fs_send FS_KEYPACK) + the y_ag readers
(merge helpers, ix_dm1). AGK and RS1_EARLY both edit fs_read.cpp but different streams: FS_KEYPACK builds the AG1 lane
(plain FS_WAIT gate), FS_WAIT2 the RS1 scatter lane; one build never takes both (FS_KEYPACK `#error`s under
FS_SCATTER). With every knob off the shard kernels are instruction-identical to rev 64 (472b7d63e85; att_c TRISC1 up to
the compiler's two stack-slot variants). `tt/galaxy_defaults.py` turns on AGK, AGB and RS1_EARLY (2026-11-11).

| knob | probe: B 16, pos 4095, layers 0-3 (one DSA layer) | e2e (45 layers, greedy) | ids | 8x4 default |
|---|---|---|---|---|
| none (rev 64, KPH auto) | 4021.6 us | 4 K x 16 42.6 ms, 32 K x 4 22.8 ms | the reference | |
| AGK | 3867.6 us (AG window 548 -> 431 us) | 4 K x 16 41.0 ms, 32 K x 4 22.9 ms | 32 K x 4 identical | on |
| RS1_EARLY | 3925.0 us (rev 65 build) | 4 K x 16 41.6 ms (rev 67); on 5bf638d7f74 + timing fix: 36.8 -> 35.5 ms | bit-identical at B 4 and, since the scratch fix, at B 16 (3 runs) | on |
| COMB | 3906.8 us (rev 65 build) | 4 K x 16 41.2 ms | moves at a near-tie (strict gate fails) | off |
| COMB + RS1_EARLY | 3807.9 us | 4 K x 16 40.0 ms, 32 K x 4 22.1 ms | as COMB | off |
| AGB (on AGK) | - | 4 K x 16 36.8 -> 35.7 ms; with RS1_EARLY 34.3 ms | bit-identical at B 16 (3 runs); 32 K x 4 identical with RS1_EARLY | on |

The rows ran on the single-knob trees (wt_mrg e7f7dc8443f, wt_rsc ab4bd43c343 / 4d61833e021, wt_rs1 4e59a2c95c0),
whose knob-off kernels equal this tree's. RS1_EARLY only moves send timing (the two-field gates are sound for every
lane skew: shard_plan.rs1_early_gates); rev 67 alone at 4 K x 16 distinct had left the KPH run's user 0 at decode id 9,
which was the routed-expert scratch race (section 12): on the fixed stack RS1_EARLY and AGB give the knob-off digest in
every one of 3 in-process runs at 4 K x 16.

## 11. Batch scaling at 1 M context per user (2026-11-11, #64931 86b8fac2888 defaults)

| users at 1 M each | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| decode step | 18.9 ms | 19.9 | 24.3 | 30.6 | 44.5 |
| tok/s per user | 52.8 | 50.2 | 41.1 | 32.7 | 22.5 |
| TTFT (all users back to back) | 345 s | 690 s | 1378 s | 2779 s | 5520 s |

Decode is timed inside the e2e's one generate_batch call (`Glm53Generator.last_batch_timing`, 86b8fac2888). The
2026-11-10 table (19.1 / 20.8 / 25.4 / 36.1 / 46.2 ms) subtracted a separately timed prefill, so the difference of two
1 M prefills landed in every step: two such runs at 1 M x 8 read 36.1 and 26.8 ms around the one-call 30.6 ms. Of the
1 M x 16 step, ~7.7 ms is context (4 K x 16: 36.8 ms).

16 x 1 M is resident on one 8x4. The MoE x-row coalesce (`GLM53_MKX_XROWS_COAL`, default) adds -2.8 ms at 4 K x 16
(40.8 -> 38.0 ms) and -0.8 ms at 32 K x 4 (ids identical).

## 12. The routed experts' scratch between stacked layers (2026-11-11, #64931 5bf638d7f74)

At 12 or more users the outputs changed from run to run, in multi-layer sparse_kda stack programs only (B 12: rows 10-11,
B 14: rows 8-13, B 16: every row; one layer per program was clean). Cause: `xa.cpp` of layer l reads the list page's
weight rows in place (and, with `MKX_RW_XA`, scatters into `xw.cpp`'s CB_RAW scratch) after `MKX_SEM_LIST`, but `xc.cpp`
pops the list page as soon as it has read n. On the 8x4 the third reader of a bank sits out every dense phase, so its
`xw.cpp` for layer l + 1 reserves that page and zero-fills the weight rows while `xa.cpp` of layer l is still reading them
(the window is long enough only at t >= 12).

Fix: `xa.cpp` clears `MKX_SEM_LIST` after its last access to either page and `xw.cpp` waits for that 0 before its first
write (`GLM53_MKX_RAW_SYNC`, default on; no L1, no new semaphore; `=0` restores the previous binaries). The routing-weight
scatter moves back to `xw.cpp` (`GLM53_MKX_RW_XA=0`, default). `tests/test_mkx_xrows_coal_host.py` holds the byte map and
an exhaustive interleaving model of one reader over 1-3 stacked layers (the race is reachable without the hand-back with
`MKX_RW_XA` on and off; with it, none and no deadlock).

| Check (8x4, 45 layers unless noted) | Before | After |
|---|---|---|
| Pinned-input replays, B 14 / B 16 (3-6 sparse layers), 21 each | several digests | 1 digest |
| 4 K x 16 e2e, 3 runs | outputs differ, 37.5-38.0 ms | identical, 36.8 ms (27.1 tok/s per user) |
| 32 K x 16, 16 identical prompts | outputs differ | one output, 35.5 ms (28.1 tok/s per user) |
| 32 K x 4 | ids d032bb00 | ids d032bb00, 22.3 ms |
| Served 128 K profile, 16 identical prompts | several texts | one text, median ITL 37.6 ms |
