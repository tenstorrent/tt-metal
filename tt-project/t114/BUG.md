# conv3d (halo mode): blocking sweep hung a BH galaxy 2x4 submesh; direct-reader fallback drops the halo

Draft for an upstream tt-metal issue. Not filed yet.

## Summary

Two problems, found while sweeping conv3d blockings for the LTX-2.5 VAE decoder:

1. **Hang.** A halo-mode `ttnn.experimental.conv3d` blocking sweep stopped producing output and froze an
   active-eth core. The device needed a broker reset (`glx_reset` job 486, fabric check 488).
   The hanging blocking is one of ten (list below). It is not yet isolated.
2. **Silent wrong output (found while triaging).** When the L1 prefetch shard does not fit,
   `conv3d_program_factory.cpp` falls back to the direct reader with only a `log_debug`. The direct
   reader has no halo branch: it clamps or zero-pads at the shard edge and never reads `halo_buffer`.
   In halo mode the output is wrong at every chip seam, and nothing errors. `validate()` already rejects
   halo mode for the other direct-reader cases (no spatial reuse, dilation). It misses this one.

## Setup

- Machine: Blackhole galaxy g14blx03. The mesh opens as (4,8), then `create_submesh(MeshShape(2,4))`.
  FABRIC_1D, trace region on.
- Build: tt-metal at 83c11ee2b34 (C++), test tree at 7e25dc0dbad (Python only differs).
- Test: `models/tt_dit/tests/models/ltx/bruteforce_conv3d_sweep_ltx.py::test_bruteforce_sweep_ltx25_544p_145f_halo -k exact_s2_res`
  (`SWEEP_MAX_COMBOS=300`, `near_table=True`, `max_t_block=8`, `hw_product=(16,32,64)`).
- Conv: C_in = C_out = 512, kernel (3,3,3), stride 1, bf16, bias, fp32 dest acc, HiFi2 (harness default).
  Halo mode: unpadded input (1, 73, 34, 30, 512) per device, padding (1,1,1), `halo_buffer` from
  `neighbor_pad_halo_only`, `logical_h_mask = logical_w_mask = 0`. Output per device (73, 34, 30).
  Grid from `compute_with_storage_grid_size()`.
- Blocking = (C_in_block, C_out_block, T_out_block, H_out_block, W_out_block).

## Symptoms (broker job 484, 2026-10-03)

- 08:33:28 job starts. 08:33:49 first combo, the table blocking (64,256,1,8,4), runs at 13597 us.
- 08:36:18 last line: `[140/300] (64,128,6,8,2) 18506us`.
- The harness prints only every 10th ok combo or a new best. So the hang is in combos 141 to 150.
- No output for 300 s. The broker reaped the job as hung at 08:41:28 (exit 130).
- Post-job gate: host PCI OK, ARC heartbeat OK on all 32 chips,
  **eth-heartbeat FROZEN** (an active-eth core's heartbeat stopped advancing).
- Incident: `/var/lib/tt-device-broker/health/incidents/20261003T084256Z_unhealthy_484` on g14blx03.
- Recovery: `glx_reset` job 486, fabric check job 488 (all 128 links healthy). Device back at 08:44 UTC.

## Candidate blockings (job 484 combos 141-150, in launch order)

| # | blocking | L1 prefetch shard | notes |
|---|----------|-------------------|-------|
| 141 | (64,64,3,8,8) | fits | |
| 142 | (64,64,3,16,4) | fits | |
| 143 | (64,128,6,8,8) | **no** (shard 102400 B > 91136 B left) | direct reader, halo dropped |
| 144 | (64,128,6,16,4) | **no** (shard 110592 B > 91136 B left) | direct reader, halo dropped |
| 145 | (64,32,3,4,4) | fits | |
| 146 | (64,32,3,8,2) | fits | |
| 147 | (64,128,5,4,4) | fits | |
| 148 | (64,128,5,8,2) | fits | |
| 149 | (64,128,7,4,4) | fits | |
| 150 | (64,128,7,8,2) | fits | |

All ten use C_in_block 64, so there are 8 C_in blocks reduced over semaphores (fp32 partials),
with the same parallel split as combos 137 to 140, which passed (c_in 8 x c_out 4 or 8 x t 2 to 4,
weight chain sharing).

**143 and 144 are the prime suspects.** They are the first fallbacks in the sweep with T_out_block > 1,
and they have the largest output block (M_t x N_t = 12 x 4 tiles). Other CBs fill about 93% of
the usable L1 (1165312 of 1256448 B). Earlier fallbacks 7 and 8 (T_out_block 1) did not hang.
This is a code-reading result only. No single combo has been run on its own yet.

## Fix proposed (branch ttp/t115-conv3d-hang-triage, 5bce3778127, not compiled)

- `conv3d_program_factory.cpp`: `TT_FATAL(!halo_mode, ...)` on the direct-reader fallback.
- Sweep harness: `prefetch_shard_fits()` copies the factory budget exactly. Halo sweeps drop blockings
  that do not fit before launch.
- Five LTX table blockings would now trip the guard and need new blockings first. They are the 1024-ch s0
  convs on 4x8 (keys `(4,8,1024,1024,(3,3,3),22,{5,10},{4,8})` and `(4,8,128,1024,(3,3,3),{21,22},5,4)`)
  and `(2,4,128,128,(3,3,3),147,136,120)`. If those convs run in halo mode today, their output is
  wrong at the chip seams.

## Repro and bisect

`tmp/blx03/t115/driver115.sh` runs one broker job per blocking: 141, 142, 145-150, with 143/144
left out on purpose. Each job opens the full mesh and then `create_submesh(2,4)`.
`SWEEP_ONLY_BLOCKINGS` makes the sweep time just that blocking, and the harness prints the blocking
before each launch. A drop reruns the combo (skip after 2 drops in a row); a hung combo is recorded and
the bisect goes on unless the broker stays unhealthy for 30 min after it.
`DRY_RUN=1 bash tmp/blx03/t115/driver115.sh` prints the plan without touching a device.

## Bisect status (#122, 2026-10-06)

- Build: blx03 ~/fasth3/t48 @9f2b28b766 (C++ = 64571a953b2, TT_FATAL halo guard compiled). Python overlay
  ttp/t114 @bc134f7c656 staged at g14blx03:/var/tmp/fasth3/t115/src.
- Driver launched 02:55 UTC (pid 79490), waiting for #119's driver and for broker health. At launch the
  broker held the device: 8/32 chips off the bus after a failed glx_reset (02:44 UTC), not our job.
- Per-combo outcomes: g14blx03:/var/tmp/fasth3/t115/outcomes.txt (filled in here when the run ends).
- 143 is not run on purpose (task update: 143/144 excluded); a clean TT_FATAL check for it is a followup.

## Bisect result (#122, blx03, 2026-10-06, t48 @64571a953b2 + guard 5bce3778127)
Each combo ran as its own broker job: full mesh opened, then create_submesh(2,4). Combos 143/144 were excluded.

| combo (Cin,Cout,T,H,W) | job | outcome |
|---|---|---|
| 141 (64,64,3,8,8)  | 269 | PASS |
| 142 (64,64,3,16,4) | 270 | PASS |
| (64,32,3,4,4)      | 271 | PASS |
| (64,32,3,8,2)      | 272 | PASS |
| (64,128,5,4,4)     | 273 | HANG: no output for 300 s, reaped; post-job eth heartbeat frozen (incident 20261006T040240Z_unhealthy_273) |
| (64,128,5,8,2)     | 276 | HANG, same signature (20261006T041344Z_unhealthy_276) |
| (64,128,7,4,4)     | 280 | HANG, same signature (20261006T042443Z_unhealthy_280) |
| (64,128,7,8,2)     | 285 | HANG, same signature (20261006T043535Z_unhealthy_285) |

Between hangs the broker ran its own galaxy recovery (job 275 and similar), and the device passed its health gate before the next combo started. Every combo with Cout=128 still hung on that freshly recovered device, so the hang follows the config and is not left over from a wedged mesh. The guard did not fire: these blockings pass prefetch_shard_fits() but still hang. So the job-484 hang is NOT limited to 143/144. It covers Cin_block=64 with Cout_block=128 (T=5 and T=7 tested). Cout 32 and 64 are fine.
Combo 143 alone was not run: four reproducible hangs in a row already answer the question, and 143 is expected to hit the TT_FATAL guard before any device dispatch.

## Root cause (#149, code reading only, 2026-10-06)

**The vol2col_rm CB write pointer runs past the CB end.** This depends on the block's patch count, not on Cout_block.
Cout=128 and T in {5,7} are fully confounded in the bisect above. Job 484 combo 140 (64,128,6,8,2) ran with
Cout=128 and passed.

- `conv3d_program_factory.cpp:173-175` (before the guard) sizes the vol2col_rm CB:
  - `min(n, 32)` pages when n = T_out_block*H_out_block*W_out_block is a multiple of 32;
  - otherwise `min(n, 64)` pages.
- Per block, the reader's ChunkWriter (`kernels/reader_vol2col.cpp:142-208`, used at :1088 and :1178) pushes
  32-page chunks plus an `n % 32` tail. Compute tilizes and pops the same chunks (`kernels/compute.cpp:422-445`).
- When n > 64 and n % 32 != 0, each block ends at an offset of `n % 32` into the CB, so the next block starts
  there. Its 32-page chunks then cross the 64-page CB end. Example: n=80. Block 0 pushes 0-32, 32-64 (wraps),
  then 0-16. Block 1 pushes 16-48, then 48-80, which is 16 pages past the end.
- `cb_push_back` (`tt_metal/hw/inc/api/dataflow/dataflow_api.h:216-217`) wraps only when
  `fifo_wr_ptr == fifo_limit`. At 80 > 64 it never wraps again, so the reader keeps writing upward through L1:
  vol2col_tiled, weights, interm, the prefetch shard, then past the end of L1. That corrupts the CBs and sends
  NoC writes to invalid addresses, and the core hangs.
- Compute's `llk_pop_tiles` (`llk_io_unpack.h:62`) wraps on `>=`, so the two sides also stop agreeing on where
  data is.
- Each core runs hundreds of blocks, so the overrun starts in block 1.

`tt-project/t149/sim_vol2col_cb.py` replays the pointer arithmetic and matches all 9 data points:

| (T,H,W) | patches | sim | device |
|---|---|---|---|
| (5,4,4), (5,8,2) | 80 | overrun in block 1 | HANG (273, 276) |
| (7,4,4), (7,8,2) | 112 | overrun in block 1 | HANG (280, 285) |
| (3,8,8), (3,16,4) | 192 (aligned) | ok | PASS (269, 270) |
| (3,4,4), (3,8,2) | 48 (<= 64, CB = 48 pages) | ok | PASS (271, 272) |
| (6,8,2) | 96 (aligned) | ok | PASS (job 484 combo 140, Cout=128) |

I rebuilt job 484's combo order on CPU: `build_all_blockings(512,512,(3,3,3),36,32,75)` sorted near
(64,256,1,8,4), 731 combos, positions 141-150 matching this file. No hazardous combo comes before 141, so no
earlier pass contradicts this cause. The same sizing line is on upstream main, so the bug exists upstream too.

Alternatives, ranked (all weaker):
1. L1 overflow from CB sizing. This would raise a host allocation error, not hang.
2. A weight-chain or reduction semaphore deadlock tied to Cout=128. Ruled out because combo 140 has the same
   parallel split (c_in 8, c_out 4, t 4, Chain weight share) and passed.
3. Matmul subblock or dst limits at N_t=4. Combo 140 has the same N_t and passed.
4. The frozen eth heartbeat. This is likely a side effect of the wedged NoC or the teardown, not a separate
   cause; this part is only weakly explained.

## Guard (#149)

- Factory: a TT_FATAL right before the vol2col_rm sizing rejects `n > 64 && n % 32 != 0` in every mode, not only
  halo. It names the blocking. It is not compiled or device-tested yet (this task did no device work).
  - It also rejects a hazardous blocking whose core would run only one block, which cannot overrun. That is
    deliberate: the count of blocks per core depends on the shape and the grid, so this is the narrow and
    simple rule.
- Sweep harness: `vol2col_chunks_fit(t,h,w)` in `bruteforce_conv3d_sweep.py` drops these combos from every
  sweep.
- CPU tests in `test_conv3d_sweep_halo_cpu.py`:
  - the predicate rejects the 4 hanging combos and accepts the 4 passing ones plus (64,128,6,8,2);
  - no `_BLOCKINGS`, `_DEFAULT_BLOCKINGS` or `_FP32_BLOCKINGS` entry is hazardous, even after the T-relaxed
    path clamps T_out_block down.
- A proper fix (not done): size vol2col_rm at `n` pages when n is unaligned, so each block ends exactly at the
  limit, or pad the tail push and pop to 32 pages. Then the guard can go.

## Audit (#149)

`tt-project/t149/audit_blockings.py` checks every table blocking. It also checks the smaller T_out_block values
the T-relaxed path can clamp to.
- 0 hits in `_BLOCKINGS`: 219 entries, 136 of them for mesh 4x8, which is what LTX_CONV3D_BLOCKING_MESH=4,8
  selects.
- 0 hits in `_DEFAULT_BLOCKINGS` (42), in `_FP32_BLOCKINGS` with the H3 audio entries (198), and in
  `_H3_ENCODER_BLOCKINGS` (9).
- The unaligned table blockings are all at most 56 patches.
- No model code builds a Conv3dConfig outside these tables.

Production decode cannot hit this hang. It came only from sweep combos, so no table blocking needs replacing.

## Device check of the guard (#151, blx03 job 294, 2026-10-06 04:54 UTC)

- Build: blx03 ~/fasth3/t48 = c4409b1fa2 + the C++ guard from aa4ade43a28, committed there as bf7db12a14 and
  rebuilt incrementally (guard string confirmed in `_ttnncpp.so`). Python: the t115 overlay bc134f7c656, which has
  no Python-side `vol2col_chunks_fit()` filter, so the C++ guard is what rejects the blockings.
- One broker job: full mesh opened, then `create_submesh(2,4)`. exact_s2_res conv3d (C_in = C_out = 512, k=3,
  T=75, H=36, W=32, halo mode). `SWEEP_ONLY_BLOCKINGS="64,128,6,8,2;64,128,5,4,4;64,32,5,4,4"`, passing one first.
  Scripts: `tmp/blx03/t151/` (build151.sh, driver151.sh, run151.sh).

| blocking | patches | expected | outcome |
|---|---|---|---|
| (64,128,6,8,2) | 96, aligned | runs | PASS, 16409 us/op traced (table (64,256,1,8,4) = 12717 us) |
| (64,128,5,4,4) | 80, unaligned | TT_FATAL, no hang | TT_FATAL at `conv3d_program_factory.cpp:189` in `create_descriptor`, before dispatch. No hang (it hung in job 273 without the guard) |
| (64,32,5,4,4) | 80, unaligned | rejected | TT_FATAL, same message |

- The process kept going after both TT_FATALs and exited 0 (1 ok, 2 failed). Job ran 16 s.
- Post-job gate: host-pci OK 32/32, ARC heartbeat advancing on all 32 chips, no reset needed. No drop, no tray-2
  (chip 12) event during the job.
- Result JSON: `tt-project/t149/job294_exact_s2_res_512x512.json`; log g14blx03:/var/tmp/fasth3/t151/run151_job294.log.

The guard works on device: it turns the hang into a host-side error, and an aligned 96-patch blocking with
Cout_block=128 still runs.
