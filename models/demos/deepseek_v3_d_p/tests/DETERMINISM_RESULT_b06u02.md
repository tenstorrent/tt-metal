# b06u02 — non-determinism confirmed (Blackhole Galaxy, 2026-07-29)

**Logical Tensix core (6,8) — physical (7,10) per the mesh translation — on physical device 14
(PCI `0000:47:00.0`, mesh shard 21) has an intermittently wrong matmul FPU path. DRAM and SFPU
on the same die are bit-exact. Not a tt-metal bug — but not proven to be a bad die either: this
box runs fw `19.8.1.0` / KMD `2.8.0`, and both known-passing boxes run the newer pair. Firmware
owns the operating point, so it can be the trigger and the weakest core the place it shows. See
"Caveat and open items".**

Host `bh-glx-b06u02`, HEAD `da6f15e849a`, driver `tenstorrent 2.8.0`, fw bundle `19.8.1.0`,
32 Blackhole boards.
Second independent confirmation run: **6 failed, 21 passed, 14:12.** Log:
`/data/nmilicevic/b06u02_det_confirm.log`.

## Result per test

| test | b06u02 | b07u08 | evidence on b06u02 |
|---|---|---|---|
| `ccl_determinism` ×12 | pass | pass | bit-exact, `ndiff=0/183500800` |
| `ccl_chain_determinism` ×6 | pass | pass | bit-exact past the depth-2 semaphore pools |
| `local_op[readback]` | pass | pass | DRAM path clean, 8 iters × 32 chips |
| `local_op[eltwise]` | pass | pass | SFPU/pack path clean |
| `local_op[seq3200-matmul1]` | **FAIL** | pass | `run_to_run={21: 1415} chip_vs_chip0={21: 1530}` |
| `local_op[seq3200-matmul2]` | **FAIL** | pass | `run_to_run={21: 306621} chip_vs_chip0={21: 246411}` |
| `local_op[seq640-matmul1]` | **FAIL** | — | `run_to_run={21: 146} chip_vs_chip0={21: 137}` |
| `local_op[seq640-matmul2]` | **FAIL** | — | `run_to_run={21: 22888} chip_vs_chip0={21: 19699}` |
| `local_compute[seq3200]` | **FAIL** | pass | `run_to_run={21: 418500} chip_vs_chip0={21: 442497}` (1.8%) |
| `core_locality[base]` | **FAIL** | pass | rows 80-89, cols 72-83, blk (M 8/10, N 6/12) |
| `core_locality[halfN]` | **FAIL** | pass | rows 80-89, cols 36-41, blk (M 8/10, N 6/12) |
| `core_locality[halfM]` | **FAIL** | pass | rows 40-44, cols 72-83, blk (M 8/10, N 6/12) |
| `matmul_core_sweep` | **FAIL** | — | `{(6, 8): {21: 44}}` — 1 of 120 cores, 85 s |
| `report_device_mapping` | pass | pass | row 5 → dev [10, **14**, 22, 18], identical on both boxes |

Every failure is a matmul test, and every one names shard 21 alone. `local_compute` uses **no
collective at all** — identical inputs replicated to all 32 chips — so the fabric is not
involved in the failing path.

## A core, not an address

The output-tile → core mapping scales with tile counts; an address range does not. Rescaling
the matmul moves the rectangle and leaves the block index fixed:

| matmul | output tiles | row tiles hit | col tiles hit | block index |
|---|---|---|---|---|
| 3200 × 4608 | 100 × 144 | 80-89 (10 tall) | 72-83 (12 wide) | **M 8 of 10, N 6 of 12** |
| 3200 × 2304 | 100 × 72 | 80-89 (10 tall) | 36-41 (6 wide) | **M 8 of 10, N 6 of 12** |
| 1600 × 4608 | 50 × 144 | 40-44 (5 tall) | 72-83 (12 wide) | **M 8 of 10, N 6 of 12** |
| 640 × 4608 | 20 × 144 | 16-17 (2 tall) | 72-83 (12 wide) | **M 8 of 10, N 6 of 12** |

An address-bound fault (bad DRAM page, bad L1 range) would have stayed at rows 80-89 /
cols 72-83 while the block index drifted.

## Chunked prefill hits the same core, and hides it better

The 640-row shape is the per-chip sequence of a 5120-token chunk (ISL 5120 / SP 8), so it is the
size chunked prefill actually runs, not a synthetic control. It still fails, and still on core
(6,8): `per_core_M` drops to 2 at Mt 20, so row tiles 16-17 give M block `16 // 2` = 8 and col
tiles 72-83 give N block `72 // 12` = 6 — the same block on the same grid, on all 8 iterations.

What changes is the size of the signal. 5x less work per core gives ~17x fewer differing
elements (about 21 per iteration against about 350 at seq 3200) and `maxabs` 1.5e-3 against
8e-3. A PCC gate is further from noticing this at 5120 than at 25600, so a chunked run that
passes accuracy CI is not evidence the core is fine — only that the fault got smaller.

## The core, named

`test_matmul_core_sweep` removes the block-index-to-coordinate inference. It confines a matmul
to a chosen window of cores with `allowed_worker_cores`, hands each core the same 10x12 output
tiles over the full K that the failing matmul does, and walks all 120 cores of the 12x10 grid.
Exactly one core fails, and it is the one the block index predicted:

| window | cores under test | verdict |
|---|---|---|
| 1x1 at (6,8) | (6,8) alone | **FAIL**, chip 21, 2 elements |
| 2x2 at (5,8) | (5,8) (6,8) (5,9) (6,9) | **FAIL**, chip 21, 16 elements, all inside (6,8) |
| 2x2 at (6,8) | (6,8) (7,8) (6,9) (7,9) | **FAIL**, chip 21, 26 elements, all inside (6,8) |
| the other 116 | — | bit-exact, 10 iterations, 32 chips |

The 2x2 windows are the load-bearing evidence: four cores run identical work inside one window
and only (6,8) diverges, so the result does not rest on the window origin resolving the way the
factory documents. Two independent methods — block-index arithmetic over a 120-core matmul, and
a confined matmul that names its cores — agree on the same core of the same chip.

The grid confirms the arithmetic: 12x10, so Mt 100 / per_core_M 10 = 10 M-blocks = grid y and
Nt 144 / per_core_N 12 = 12 N-blocks = grid x. With `transpose_mcast=False` block (M 8, N 6)
is logical (x=6, y=8).

Physical (7,10) is the *mesh* translation. Harvesting is per chip (`ENABLED_TENSIX_COL` dev 14 =
`0x3ffd`), so confirm that translation against device 14 before filing a harvest request.

A 1x1 window cannot sit on the last grid row or column: the 2D factory reads its neighbours at
`start_core_x + 1` / `start_core_y + 1` unconditionally while building the mcast ranges
(`matmul_multicore_reuse_mcast_2d_program_factory.cpp:1173,1176`), so an origin there asks for a
core that does not exist and the op throws `No core coordinate found at location`. Those 21
cores are covered by the 2x2 pass instead.

## Magnitudes

`maxabs` 1.8e-3 to 8.3e-3 on 0.02-scale activations — low-order accumulation bits, not
garbage. Only 0.1-0.3% of the elements *inside* that core's own block differ on any run
(e.g. 207 of 122880 in the base shape). **The diff counts do not reproduce between runs; the
chip, the rectangle and the block index reproduce every time.** Compare the two runs:
`local_compute` gave `{21: 385005}` on 2026-07-28 and `{21: 418500}` today.

## Reproduce (~3 min of device work, no weights, no model, no trace)

```
cd /data/nmilicevic/tt-metal && source python_env/bin/activate && export TT_METAL_HOME=/data/nmilicevic/tt-metal PYTHONPATH=/data/nmilicevic/tt-metal && mpirun --bind-to none --pernode --tag-output python3 -u -m pytest models/demos/deepseek_v3_d_p/tests/test_det_ccl_micro.py -p no:randomly -s -q -k "local_op or matmul_core or device_mapping"
```

Full suite: drop the `-k`. The 18 CCL tests are the slow half and pass on the bad box too.

## fw 19.12 / KMD 2.10 on a fresh box — clean (2026-07-30)

Host `bh-glx-110-c10u08`, fw `19.12.0.0`, KMD `2.10.0` — the newer pair. Same repo, same test
file (identical to the pushed branch tip). Matmul-compute subset, **8 passed in 5:37**:

```
all 120 cores bit-exact across 10 iterations and all 32 chips   # test_matmul_core_sweep
local matmul1 seq=3200 / matmul2 seq=3200 / matmul1 seq=640 / matmul2 seq=640  -> bit-exact
matmul 3200x4608 / 3200x2304 / 1600x4608 (core_locality)                       -> bit-exact
```

The sweep names no core. Core (6,8) is bit-exact here, and so is the seq640 chunked shape —
the smallest signal, the one a PCC gate would miss first. Log: `/data/nmilicevic/c10u08_matmul_fw1912.log`.

The firmware correlation now has two boxes on each side: **fail on 19.8.1 / KMD 2.8** (b06u02
and the other reported box), **pass on 19.12 / KMD 2.10** (b07u08, c10u08). It is still a
correlation across *different silicon* — a fresh healthy box passes whether the fix is the
firmware or the silicon lottery, so this does not yet establish cause. The one experiment that
separates them is unchanged: bring b06u02 itself to the newer pair and rerun.

## Caveat and open items

- **Firmware/KMD is the leading candidate for the *cause*, and it is uncontrolled.** b06u02 is
  on fw `19.8.1.0` / KMD `2.8.0`, the passing boxes b07u08 and c10u08 on `19.12.0` / `2.10.0`,
  and the two reported-failing boxes are both on the older pair. Firmware owns AICLK, VDD, DVFS and harvesting, so a core
  that is marginal at 19.8.1's operating point and fine at 19.12.0's produces exactly this
  signature. These tests establish *where* the fault manifests, not why. Three constraints:
  a uniform firmware cannot select 1 chip of 32 by itself (all 32 report `fw_bundle 19.8.1.0`,
  `asic_fmax 1350`, `vdd 0.70-0.90`, `THERM_TRIP_COUNT 0x0`, `GDDR_UNCORR_ERRS 0x0`), so the
  mechanism has to be firmware interacting with something per-chip; the KMD code itself is
  unlikely, since readback and eltwise are bit-exact on the same die so host DMA/MMIO is clean;
  and **harvesting** is the one firmware-only mechanism that would fully exonerate the silicon
  — `ENABLED_TENSIX_COL` is per-chip (dev 14 = `0x3ffd`, 13 of 14 columns enabled), so a
  different harvest under 19.12.0 could simply stop mapping the marginal core.
  Close it in this order: (1) flash 19.12.0 + KMD 2.10 here and rerun the 8-test subset — one
  variable, same silicon (`fw_pack-19.8.1.fwbundle` is kept for rollback); (2) run the subset
  on b07u02 and record *which* shard fails; (3) diff `tt-smi -s --snapshot_no_tty` against a
  passing box on `ENABLED_TENSIX_COL`, `asic_fmax`, and the VDD limits.
- Confirm the logical → physical translation of core (6,8) against device 14 itself rather than
  the mesh handle, since harvesting is per chip and a harvest request needs the die's own
  coordinate. The Python `MeshDevice` binding exposes `get_device_id` but no per-device handle,
  so this needs the C++ API or `tt-triage`.
- **A wedged board is a no-result, not a failure.** If every test errors in ~20 s with
  `MMIO per-op timeout: 4B load took N us (budget=2 ms)` at `distributed.py:671`, the box failed
  at `open_mesh_device` before any kernel ran. `tt-smi -glx_reset_auto` clears it — do not use
  `tt-smi -r` on these Galaxy boxes. `tt-smi -ls` enumerating 32 chips is not proof that ttnn
  can open the mesh.

Bisection method and the decision ladder: `DETERMINISM_BISECTION.md`.
Reference-box detail: `DETERMINISM_RESULT_b07u08.md`.
