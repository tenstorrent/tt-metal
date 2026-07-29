# b06u02 — non-determinism confirmed (Blackhole Galaxy, 2026-07-29)

**One Tensix core on physical device 14 (PCI `0000:47:00.0`, mesh shard 21) has an
intermittently wrong matmul FPU path. DRAM and SFPU on the same die are bit-exact. Not a
tt-metal bug — but not proven to be a bad die either: this box runs fw `19.8.1.0` / KMD
`2.8.0`, and both known-passing boxes run the newer pair. Firmware owns the operating point,
so it can be the trigger and the weakest core the place it shows. See "Caveat and open
items".**

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
| `local_op[matmul1]` | **FAIL** | pass | `run_to_run={21: 1415} chip_vs_chip0={21: 1530}` |
| `local_op[matmul2]` | **FAIL** | pass | `run_to_run={21: 306621} chip_vs_chip0={21: 246411}` |
| `local_compute[seq3200]` | **FAIL** | pass | `run_to_run={21: 418500} chip_vs_chip0={21: 442497}` (1.8%) |
| `core_locality[base]` | **FAIL** | pass | rows 80-89, cols 72-83, blk (M 8/10, N 6/12) |
| `core_locality[halfN]` | **FAIL** | pass | rows 80-89, cols 36-41, blk (M 8/10, N 6/12) |
| `core_locality[halfM]` | **FAIL** | pass | rows 40-44, cols 72-83, blk (M 8/10, N 6/12) |
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

An address-bound fault (bad DRAM page, bad L1 range) would have stayed at rows 80-89 /
cols 72-83 while the block index drifted. Which logical core x/y block (8,6) resolves to
depends on `transpose_mcast` in the chosen program config.

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

## Caveat and open items

- **Firmware/KMD is the leading candidate for the *cause*, and it is uncontrolled.** b06u02 is
  on fw `19.8.1.0` / KMD `2.8.0`, b07u08 on `19.12.0` / `2.10.0`, and the two reported-failing
  boxes are both on the older pair. Firmware owns AICLK, VDD, DVFS and harvesting, so a core
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
- Resolve block (8,6) to a logical core x/y via `transpose_mcast` so the core can be named in a
  harvest request.
- **A wedged board is a no-result, not a failure.** If every test errors in ~20 s with
  `MMIO per-op timeout: 4B load took N us (budget=2 ms)` at `distributed.py:671`, the box failed
  at `open_mesh_device` before any kernel ran. `tt-smi -glx_reset_auto` clears it — do not use
  `tt-smi -r` on these Galaxy boxes. `tt-smi -ls` enumerating 32 chips is not proof that ttnn
  can open the mesh.

Bisection method and the decision ladder: `DETERMINISM_BISECTION.md`.
Reference-box detail: `DETERMINISM_RESULT_b07u08.md`.
