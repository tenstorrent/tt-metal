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
before each launch. The driver stops at the first drop or hung job.
`DRY_RUN=1 bash tmp/blx03/t115/driver115.sh` prints the plan without touching a device.
