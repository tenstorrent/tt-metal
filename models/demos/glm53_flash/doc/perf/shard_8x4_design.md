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

Each knob is its own kernel define / stream argument, so any subset composes: `GLM53_DSA_SHARD_COMB` the roots' combine
(att_c / att_dsa_shard_dm1), `GLM53_DSA_SHARD_RS1_EARLY` the roots' RS1 signals + the RS1 lane's gate (fs_read.cpp
FS_WAIT2), `GLM53_DSA_SHARD_AGK` the AG1 lane's pack (fs_read / fs_send FS_KEYPACK) + the y_ag readers (merge helpers,
ix_dm1), `GLM53_DSA_SHARD_RSB` the two scatters' chunk geometry (stream arguments only, below), `GLM53_DSA_SHARD_RSNP`
(on RSB) the scatter lanes' staged pages (fs_read / fs_send FS_SUBPC / FS_SUBD, below). AGK and RS1_EARLY both
edit fs_read.cpp but different streams: FS_KEYPACK builds the AG1 lane (plain FS_WAIT gate), FS_WAIT2 the RS1 scatter
lane; one build never takes both (FS_KEYPACK `#error`s under FS_SCATTER; FS_SUBD `#error`s without FS_SCATTER or with
FS_KEYPACK, and composes with FS_WAIT2). With every knob off the shard kernels are
instruction-identical to rev 64 (472b7d63e85; att_c TRISC1 up to the compiler's two stack-slot variants).
`tt/galaxy_defaults.py` turns on AGK, AGB and RS1_EARLY (2026-11-11), then RSB, R8 and RSNP (an explicit RSB=0 drops
the RSNP default, as AGK=0 drops AGB's).

| knob | probe: B 16, pos 4095, layers 0-3 (one DSA layer) | e2e (45 layers, greedy) | ids | 8x4 default |
|---|---|---|---|---|
| none (rev 64, KPH auto) | 4021.6 us | 4 K x 16 42.6 ms, 32 K x 4 22.8 ms | the reference | |
| AGK | 3867.6 us (AG window 548 -> 431 us) | 4 K x 16 41.0 ms, 32 K x 4 22.9 ms | 32 K x 4 identical | on |
| RS1_EARLY | 3925.0 us (rev 65 build) | 4 K x 16 41.6 ms (rev 67); on 5bf638d7f74 + timing fix: 36.8 -> 35.5 ms | bit-identical at B 4 and, since the scratch fix, at B 16 (3 runs) | on |
| COMB | 3906.8 us (rev 65 build) | 4 K x 16 41.2 ms | moves at a near-tie (strict gate fails) | off |
| COMB + RS1_EARLY | 3807.9 us | 4 K x 16 40.0 ms, 32 K x 4 22.1 ms | as COMB | off |
| AGB (on AGK) | - | 4 K x 16 36.8 -> 35.7 ms; with RS1_EARLY 34.3 ms | bit-identical at B 16 (3 runs); 32 K x 4 identical with RS1_EARLY | on |
| RSB | 2977.0 -> 2925.4 us (layers 0-3 with UG16; RS1 wait 325 -> 287, RS2 58 -> 45) | 4 K x 16 33.3 -> 32.7 ms | bit-identical at B 16 (3 runs); 32 K x 4 identical | on |
| R8 (8-bit radix digits) | 2925.5 -> 2899.6 us (B 16 pos 4095; merge 121 -> 89.5 us) | 4 K x 16 32.7 (both); 1 M x 16 40.5 -> 38.2 ms | bit-identical at B 16 (3 runs); 32 K x 4 and 1 M x 16 user 0 identical | on |
| RSNP (on RSB) | 2925.7 -> 2876.2 us (B 16, layers 0-3 traced; RS1 wait 287.5 -> 246.9, RS2 landed 45.2 -> 35.2; root-local marks flat) | 4 K x 16 32.7 -> 32.2 ms | bit-identical at B 16 (3 runs, digest e2f3c86090b7); 32 K x 4 ids and text identical (21.2 ms) | on |

The rows ran on the single-knob trees (wt_mrg e7f7dc8443f, wt_rsc ab4bd43c343 / 4d61833e021, wt_rs1 4e59a2c95c0),
whose knob-off kernels equal this tree's. RS1_EARLY only moves send timing (the two-field gates are sound for every
lane skew: shard_plan.rs1_early_gates); rev 67 alone at 4 K x 16 distinct had left the KPH run's user 0 at decode id 9,
which was the routed-expert scratch race (section 12): on the fixed stack RS1_EARLY and AGB give the knob-off digest in
every one of 3 in-process runs at 4 K x 16.

### Prefill: the fused mHC mix (2026-11-12, default on the 8x4)

`GLM53_PF_MHC_POST=fused` runs each sublayer's mHC pre- and post-mix as one program on the flat residual
(`prefill_tp._layer_fused_mhc`, `_mhc_mix`) instead of the composite op chain. Both are exact fp32 (3e-8 rel rms vs
float64, doc/mhc/README.md); only the summation order differs, so the greedy ids move at a near-tie.

| row | composite (ops) | fused | ids |
|---|---|---|---|
| 1 M x 1 TTFT / decode | 344.7 s / 52.8 tok/s | 332.1 s / 53.6 tok/s | first 24 identical (694d88ef), text splits at ~35 tokens |
| 32 K x 4 TTFT / decode | 35.7 s / 47.5 tok/s | 34.1 s / 47.5 tok/s | 785 ... 1293, then 1467 vs 57681 (the new CI expectation) |
| ISL 128 x 16 distinct, TTFT | 7.56 s | 7.01 s | user 0 identical (9c01868e) |
| ISL 128 x 16 identical prompts | agree | agree (6.98 s) | - |

The packed prefill (`GLM53_PF_PACKED`) needs it but stays opt-in: a packed segment's rows land on other chips than the
per-user path's, whose reduce-scatter sums in another order, so identical prompts in different pack slots move at a
near-tie (16 identical prompts, 2048:8: slot 6 differs in both replays).

### Prefill: the routed-MoE kernel options (2026-11-12, default on the 8x4)

`tt/galaxy_defaults.py` turns on the seven TP-prefill MoE options of the pf_g2 group at full precision:
`GLM53_PF_MOE_KB=808`, `XGATHER`, `V2`, `RM_SKIP`, `PADMASK`, `XFFN_VALID`, `XFFN_SNAKE` (prefill_tp.py). LoFi
(`GLM53_PF_MOE_FID=LoFi`) stays off: alone it bought 0.8 % at 32 K x 4.

| geometry | TTFT before | TTFT after | ids |
|---|---|---|---|
| 1 M x 1 | 332.1 s | 281.4 s (-15.3 %) | identical (128 / 128) |
| 32 K x 4 | 34.1 s | 27.4 s (-19.6 %) | near-tie after 7 tokens (the ops-post sequence of before the fused post) |
| ISL 128 x 16 distinct (128 K geometry) | 7.01 s | 5.06 s (-27.8 %) | user 0 near-tie at token 6 ("the text" -> "a text") |

Decode is unchanged (18.6 / 21.0 / 31.8 ms per step). The tier-1 32 K x 4 gate moves to the new first ten ids.

### RSB: the reduce-scatters in block chunks (2026-11-11, default on the 8x4)

Today both scatters signal per token. RS1 (FabricStreamScatter axis 0, one lane) ships per destination t chunks of
10 pages (one 8-row slice, 9 pages + a pad page) as 10-page batches of 5 two-page packets; the chunk's last packet is a
fused write + atomic inc with flush of the receiver's ARR counter (fs_send.cpp `col + 2 == PPC`), so the receiving
router drains its NoC writes before every inc (fabric_edm_packet_transmission.hpp, CHUNK_ENCODING_SEMINC_FLUSH). That
is 7 t signals per source per launch (112 at t 16), and RS2 (axis 1, 4-page chunks of one 2-row slice) adds 3 t (48).
Nothing pipelines on them: only the receiver's waiter (the lane BRISC) reads ARR, it waits for all t chunks of every
source, then raises the roots' SEM_RSC once (FS_CONS); the roots wait SEM_RSC >= 1 / >= 2.

`GLM53_DSA_SHARD_RSB=1` (8x4 only, independent of the other knobs): x_rs1 / y_rs1 / x_rs2b / y_rs2b as [n x 32, W t]
(shard_plan.rs_stream_args): one 32-row chunk per destination block holding its t slices at the knob-off page ids,
so the roots' writes and landing reads do not change, and the chunk-end packet goes out once per destination. No
kernel change: the scatter's x and y chunks coincide, so fs_send's `col + 2 == PPC` with PPC = 10 t / 4 t is the
block's last pair (AGB needed FS_KP_XPER only because AGK's x chunk is not its y chunk). The waiters wait for 1 per
source. Staging CBs keep their bytes (RS1 4 x 10 pages = 80 KB, RS2 2 x 8 pages = 32 KB), no new CBs, no new
semaphores. Host: shard_plan.fs_scatter_sends / rs_stream_counts / audit_rs_blocks, tests/test_shard_plan_8x4.py.
Offline compile of fs_send / fs_read (t 4 / 8 / 16, RS1_EARLY on / off, probe on / off, AGK + AGB): knob off equals
3b0ee5b0342's objdump, knob on compiles.

| per source per launch | t 4 | t 8 | t 16 |
|---|---|---|---|
| RS1 chunk-end signals | 28 -> 7 | 56 -> 7 | 112 -> 7 |
| RS2 chunk-end signals | 12 -> 3 | 24 -> 3 | 48 -> 3 |
| RS2 staging batches | 12 -> 6 | 24 -> 12 | 48 -> 24 |
| RS1 batches / packets (RS1 + RS2) | 28 / 164 | 56 / 328 | 112 / 656 (unchanged) |

Expected: AGB removed 135 signals + 96 batches per source at t 16 for ~100 us per DSA layer (36.8 -> 35.7 ms over
11 layers), at most ~0.7 us per signal. At that rate RSB is worth up to ~100 us per DSA layer at B 16 (~1.1 ms per
4 K x 16 step), ~50 us at B 8, ~20 us at B 4: probe owner_dm1 marks 5 -> 6 (pass 1 done -> RS1 landed: 324 us at B 16)
and 7 -> 8 (row scatter shipped -> RS2 landed); 6 -> 7 and 8 -> 9 are root-local and must not move. The bound: RS1's
cross-half traffic is 16 t x 20 KB per direction on the column's middle link (5.1 MB at t 16; 324 us is ~16 GB/s),
so if RS1 is link-bound its share shrinks to the signals' router stalls only -- the rs1 lane's opened -> sent span
(probe row 238) tells which. RS2 (512 KB on the row's middle link at t 16) is not byte-bound.

### R8: 8-bit radix digits in the local select and the merge (2026-11-11, MLA_REV 70, default on the 8x4)

Today (dsa_common.hpp 4-bit digits) a sparse token's local select is 8 radix rounds + the tie round on the lane's
coordinator (ix_dm0), one report / broadcast handshake each: every chunk core histograms its compacted keys into its
16-word slot of the coordinator's histogram tensor and raises H_SEM, the coordinator sums the C slots (sum_hist), picks
the digit, writes (prefix, k_rem) into every owner's mailbox and raises SEL_SEM, the cores compact. The global merge
runs the same rounds on the NHELP helpers: 12 sequential steps per token (start, 8 rounds, tie, quotas, the rank
helper's per-core quotas), ~5 us each at B 16 (section 9's 119-us merge = 2 tokens x 12). fp32 keys hardly filter in the
first 4-bit rounds (sign + 3 exponent bits, then 4 more), so at long context the cores also scan nearly all their keys
about three times.

`GLM53_DSA_SHARD_R8=1` (kernel define MLA_DSA_R8; 8x4 only, composes with every other knob): ROUNDS = 32 / 8 = 4. Round r
is the 4-bit rounds 2 r and 2 r + 1 composed with the same integers, in one handshake. Each core counts its matching
keys in 256 uint16 fine bins (the 8-bit digit) in its OWN mailbox past the messages (dsa_shard.hpp `r8_hist`,
`R8_FINE_OFF` = MBOX_SHARD_BYTES) and reports the 16 coarse sums (the digit's high nibble) in the unchanged 64-B slot.
The coordinator sums the coarse reports and picks the coarse digit exactly as round 2 r does, reads the chosen bucket's
16 fine counts from every core (one 32-B NoC read each, one barrier: ix_dm0 `r8_fine`) into the cores' slots, sums them
and picks the fine digit as round 2 r + 1 does (k_rem; in the local select also each core's keys above both digits).
The broadcast carries the 8-bit prefix; the tie round, the (quota, offset) pairs, the per-chip / per-core quotas and the
tie order are unchanged. Per token: local select 9 -> 5 handshakes (+ 4 pulls), merge 12 -> 8 steps (+ 4 pulls), and
the cores' near-full key scans ~3 -> ~2 (round 0 already covers 7 exponent bits).

Why the pull: a plain 256-bin report would put C x 1 KB of slots on each lane's index core (64 KB at t 1) and a 256-word
sum per core per round on the coordinator (~1.3 us per core at rev 52's ~5 ns per word: ~80 us a round at C 64); the
pull keeps the slots and the sums at today's 16 words per core per level. L1: none -- the histogram tensor and CB_AKEY
are unchanged, the mailbox drops 4 select + 4 merge message slots per lane token (128 B x TL) and gains the 512-B fine
area: 4800 B at TL 4 (unchanged), 9600 -> 9088 B at TL 8. No new semaphores or CBs. The uint16 counts need <= 65535 keys
per core (static_assert on MLA_DSA_KEYS; a merge helper holds <= KPH x 512).

Host: tt/shard_plan.py `local_select` / `helper_merge` / `agk_merge` with digit_bits=8 run the kernels' two levels step by
step (`r8_hist` with the packed coarse sum, `radix_round`); tests/test_shard_plan_8x4.py checks R8 == 4-bit bitwise (T /
T_g, k_rem, the cores' counts above, (quota, offset) pairs, records, per-chip / per-core quotas, done; each level's sums
= the 4-bit rounds' sums) == the sorted top-k, over list lengths 0 .. 512, ties, duplicates on the digit edges, keys
sharing 24 bits, 513 candidates, KPH 1 / 2 / 4, 1-8 lanes and the AGK launch. Offline compile (`_r8_kc/cmp_r8.sh`): knob
off, ix_dm0 / att_dsa_shard_dm1 / ix_dm1 at t 1 / 4 / 8 / 16 (+ probe, TL 8, rev 64, dense window + COMB, 2x4) equal
0687018fd13's objdump; knob on compiles (text: ix_dm0 +0.7-1.2 KB, att_dsa_shard_dm1 +0.1-0.4 KB), ix_dm1 unchanged.

Expected at B 16, 4 K: 2 tokens per lane x (4 + 4) handshakes x ~5 us minus 16 pulls at ~1 us: ~-65 us per DSA layer
(~-0.7 ms per 45-layer step); about half at B 4 (1 token per lane); at 1 M also one of the ~3 near-full key scans of each
select and merge pass. Probe (test_mk_backend_probe, MLA_PROBE=1): owner_dm1_max 0 -> 1 (the local select) and 1 -> 2
(AG + merge), ix_dm0 4 -> 5 (the merge alone: ~121 us at B 16 today) must drop; the AG window and everything after mark 2
must not move.

### RSNP: the reduce-scatters without pad pages (2026-11-11, MLA_REV 71, default on the 8x4)

Why the pad page exists: fs_send.cpp ships PAGE PAIRS -- one fabric scatter packet = 2 x 2 KiB to two NoC addresses --
and a pair never straddled a batch or a chunk (`static_assert(PPC % NB == 0 && NB % 2 == 0)`), so the per-token
slice chunk was rounded up to an even page count: the RS1 slice (16 O tiles + 2 stats tiles, 8 rows of fp32 each) is
18 KB = 9 pages -> RS2_PPC = 10, the RS2 slice 4.5 KB = 3 pages -> RS2B_PPC = 4. Nothing else needs it: the roots write
bytes [0, DT ob + ST pb) of a slice and every landing read (land_rows, the fold pulls) reads the same range, so the
pad page is never written (x_rs1 / x_rs2b are ttnn.zeros) and never read; it is no alignment or consumer requirement
beyond the 10 / 4-page stride. So 1 / 10 of RS1's and 1 / 4 of RS2's bytes and packets are pad.

`GLM53_DSA_SHARD_RSNP=1` (8x4 only, needs RSB: on per-token chunks an odd page count keeps its pad): fabric_stream.py
`sub=(stride, data)` -> FS_SUBPC / FS_SUBD on both lane kernels. fs_read stages only the slices' data pages in order
(x page (base + chunk) PPC + sub_col(s)); fs_send pairs them as staged -- slice k's 9th page with slice k + 1's 1st --
and writes each page of the pair to its own RSB y page (the scatter write's two addresses are independent; consecutive
pages already land in different DRAM banks); the chunk-end signal stays on the block's last pair (`col + 2 == DPC`). Odd
t keeps the last slice's first pad page in the last pair (as RSB). Unchanged: x / y shapes and page ids, the roots, the
signals and waiter counts, every runtime arg; only NB and the two defines differ. Batches: RS1 12 pages in 3 slots
(72 KB, within RSB's 80 KB; AG2's 3-slot ring), RS2 RSB's 8 in 2 slots. Host: shard_plan.shard_rsnp / rs_slice_bytes /
fs_sub_col / fs_scatter_packets / rsnp_header_check, rs_stream_args(rsnp=), rs_stream_counts(rsnp=) (bytes, middle-link
bytes); tests/test_shard_plan_8x4.py (test_rev65_knobs, test_block_chunks_keep_pages, test_block_audits,
test_shard_knobs_compose, test_galaxy_defaults_shard_tail: 314 pass on RSB, 318 with R8's).

| per source per launch, RSB -> RSNP | t 4 | t 8 | t 16 |
|---|---|---|---|
| RS1 / RS2 pages per destination | 40 -> 36 / 16 -> 12 | 80 -> 72 / 32 -> 24 | 160 -> 144 / 64 -> 48 |
| RS1 packets / batches | 140 / 28 -> 126 / 21 | 280 / 56 -> 252 / 42 | 560 / 112 -> 504 / 84 |
| RS2 packets / batches | 24 / 6 -> 18 / 6 | 48 / 12 -> 36 / 9 | 96 / 24 -> 72 / 18 |
| column / row middle link, per direction | 1.25 -> 1.125 MiB / 128 -> 96 KiB | 2.5 -> 2.25 MiB / 256 -> 192 KiB | 5.0 -> 4.5 MiB / 512 -> 384 KiB |
| chunk-end signals (RS1 + RS2) | 7 + 3 | 7 + 3 | 7 + 3 |

Offline (no device): fs_send / fs_read compiled as fabric_stream.py builds them (_rsnp_kc/cmp_rsnp.sh): knob off
(RSB and pre-RSB args, RS1_EARLY and probe on / off, AGK + AGB) objdump-identical to 0687018fd13 at t 4 / 8 / 16; RSNP
compiles at t 1 / 2 / 3 / 4 / 8 / 16 with RS1_EARLY and probe on / off; FS_SUBD without FS_SCATTER and a batch not
dividing DPC fail to compile. On R8 (3be1572d06a, MLA_REV 70 -> 71; _rsnp2_kc/cmp_rsnp2.sh): fs_send / fs_read knob
off objdump-identical to 3be1572d06a and RSNP compiling at t 4 / 8 / 16 (RS1_EARLY, probe on / off); ix_dm0 /
att_dsa_shard_dm1 / ix_dm1 (R8 on / off, RS1_EARLY on / off, probe) objdump-identical with -DMLA_REV=71 for 70 -- the
define's value is the only difference (no kernel reads it; MLA_SRC 0xc7a14a50 unchanged: RSNP edits no hashed
directory). FabricStreamScatter.parts() under a stub ttnn for all 32 chips, t 1 .. 32, RS1_EARLY on /
off (_rsnp_kc/rsnp_args.py) with a line-by-line port of the two kernels' loops: RSB's runtime args, RSB's pages minus
the pads in RSB's order, one signal per destination; the landed y bit-identical to RSB's and the bytes the landing roots
read identical even with garbage in x's pad pages. Expected: if RS1 stays bytes-bound at RSB's rate, its wait (probe
owner_dm1 marks 5 -> 6) drops 287 -> ~258 us at B 16; RS2 (7 -> 8) 45 -> at most ~34 us (not fully bytes-bound): up to
~40 us per DSA layer (~0.4 ms per 4 K x 16 step), proportionally less at B 4 / 8; marks 6 -> 7 and 8 -> 9 are root-local
and must not move. The device run: ids / top-4 values bit-identical to the RSB default (4 K x 16 distinct, 3 runs;
32 K x 4), then the probe.

Device (wt_rsnp 3c8a0a3bf92 = these kernels / streams on 0687018fd13): 4 K x 16 distinct, 3 in-process runs, every run
the RSB default's digest (e2f3c86090b7), 32.7 -> 32.2 ms; probe B 16 layers 0-3 (traced) 2925.7 -> 2876.2 us: RS1 wait
287.5 -> 246.9 us (-40.6 us, more than the bytes alone: 10 % fewer packets and 25 % fewer RS1 batches too), RS2 landed
45.2 -> 35.2 us, the root-local marks flat. `tt/galaxy_defaults.py` turns it on.

### LOCAL_ALL: no radix rounds under 512 local pools (2026-11-12, MLA_REV 72, opt-in)

Traced-replay marks of every chip (test_mk_backend_probe MK_BACKEND_SHARD_PROBE_TRACED, B 1, POS 4095, layer 3; the
eager marks carried ~650 us of host per-device dispatch skew, the traced chips start within 6.6 us): index projections +
local select ~246 us (I1 ~72, the select ~174), AG 34, merge 47, attention 305 per DSA layer. At POS 4095 a chip holds
32 closed pools, yet the select runs R8's 4 rounds + the tie round: 5 report / broadcast handshakes of the lane's 64
owners with the index core, which only re-derive "every pool is kept". `GLM53_DSA_SHARD_LOCAL_ALL=1` (dsa_shard.hpp
LOCAL_ALL, the kernels' MLA_DSA_SHARD_LOCAL_ALL): a sparse token whose chip has `n_closed_local <= 512` skips those
MSGS messages on the coordinator (ix_dm0.cpp: header count = n_closed_local) and on every owner (att_dsa_shard_dm1.cpp
phase A, every group: all valid keys kept in local order, the group-0 records at `local_all_offset` = the valid pools of
the cores before). The SEL_SEM counts stay equal on both sides (the merge reuses the counter). Exact: the records,
offsets, header and the merge inputs equal the rounds' (shard_plan.local_select_all vs local_select in 4-bit and R8
digits, random keys and heavy ties, 0 .. 600 pools on 64 / 16 / 8 lane cores; a 32-chip merge at p = 66000 where chips
0..3 hold 532-544 pools and take the rounds: test_local_all_matches_the_rounds). Every chip takes it below 64 K tokens
(p = 65535: 512 on every chip). Expected: ~-140 us per DSA layer at B 1 (~-1.5 ms per 45-layer step at <= 64 K). Above
512 local pools the rounds stay; `GLM53_DSA_SHARD_ROUND_MARKS=1` (probe builds, R8 only) marks the coordinator's
rounds (decode_probe "rounds": per round the owners' share -- round 0 includes the I4 scoring -- and the coordinator's
sums + digits + broadcast) for the next cut.

### SEL2: cheaper local-select rounds past 512 local pools (2026-11-13, MLA_REV 73, opt-in)

Past ~64 K tokens of context (a chip over 512 closed local pools) LOCAL_ALL no longer applies and the slowest chip's
rounds gate the all-gather (POS 66000: chips 0-3 keep the rounds, the others' AG1 waits ~130 us for them). The rev-72
round marks put a coordinator round at ~0.4 us per participating chunk core (27.0 us at B 1 with one lane of 64 cores,
4.3 us at B 16 with 8 per lane): per core 16 coarse words summed, `above[c]` updated by stack read-modify-writes over
the bins above the coarse digit, a 32-B fine pull, 16 uint16 fine loads, `above[c]` again over the fine bins, one
message write and one semaphore increment. `GLM53_DSA_SHARD_SEL2=1` (dsa_shard.hpp SEL2, MLA_DSA_SHARD_SEL2; needs R8)
keeps every integer and drops most of that work:

- no `above[c]` on the coordinator: a core's keys above the final T are exactly the keys its compactions drop upward
  (a key > T leaves at the first digit where it differs from T, with the larger digit), so each core counts them while
  it compacts and reports (eq, gt) in the tie round; the record offsets are `off += gt + quota` in core order;
- packed sums: the coarse report is 8 uint32 words of 2 uint16 counts (`r8_hist_packed`), like the pulled fine slice;
  the coordinator adds 8 words per core in registers and splits the halves (`sel2_sum`): exact while a count's lane
  total stays <= 65535 (n_closed_local <= 65535, contexts below 8 M tokens);
- the (quota, offset) message: owner k gets the 16-B chunk with its own pair (`send_own`) instead of the C x 8-B table.

Host proof: `shard_plan.local_select_sel2` against `local_select` (R8), `test_sel2_matches_the_rounds` (lanes 1 / 4 / 8,
0 .. 8192 keys per lane, random / heavy ties / all equal / fp32-clustered): prefixes, k_rem, both levels' sums, (quota,
offset), header count, kept pairs and records bitwise; the upward drops equal the rounds' above counts;
`test_sel2_packed_sums_and_quota_chunks` (the 65535 bound and the carry past it). Expected (per-core cost ~0.4 ->
~0.15-0.2 us): a B 1 round 27 -> ~11-14 us, ~55-65 us per DSA layer at 1 M x 1 (~0.6-0.7 ms per 45-layer step), small
at B 16 (8 cores per lane). Validation: `_logs/mesh/chain_dsa_2.sh`.

## 11. Batch scaling at 1 M context per user (2026-11-11, #64931 685aaccbc5e defaults)

| users at 1 M each | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| decode step | 18.9 ms | 19.9 | 23.6 | 29.4 | 42.1 |
| tok/s per user | 52.8 | 50.2 | 42.4 | 34.1 | 23.8 |
| TTFT (all users back to back) | 345 s | 690 s | 1378 s | 2779 s | 5518 s |

1 and 2 users ran on 86b8fac2888 (before RS1_EARLY + AGB, whose savings scale with t); 4 / 8 / 16 users there were 24.3 /
30.6 / 44.5 ms, with user 0's ids and text identical to these rows.

Decode is timed inside the e2e's one generate_batch call (`Glm53Generator.last_batch_timing`, 86b8fac2888). The
2026-11-10 table (19.1 / 20.8 / 25.4 / 36.1 / 46.2 ms) subtracted a separately timed prefill, so the difference of two
1 M prefills landed in every step: two such runs at 1 M x 8 read 36.1 and 26.8 ms around the one-call 30.6 ms. Of the
1 M x 16 step, ~7.8 ms is context (4 K x 16: 34.3 ms).

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

Device (chain_mk192, 232d27083c7): the probe's RS1 wait drops 38 us of 325 (the signals were ~12 % of it; at t 16 RS1 moves
5.1 MB per direction over the column's middle link, ~16 GB/s: bytes-bound), RS2 13 us of 58; -52 us per DSA layer, -0.6 ms
per 4 K x 16 step. The next RS cut is bytes (the pad pages: 10 % of RS1, 25 % of RS2).
