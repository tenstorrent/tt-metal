# b06u02 — non-determinism confirmed (Blackhole Galaxy, 2026-07-29)

**Logical Tensix core (6,8) — physical (7,10) per the mesh translation — on mesh device 14
(on b06u02 PCI `0000:47:00.0`, mesh shard 21, row 5 col 1) has an intermittently wrong matmul
FPU path. DRAM and SFPU on the same die are bit-exact. Not a tt-metal bug. Firmware does NOT
fix it: a second box on fw `19.12.0` / KMD `2.10.0` (`c10u08`) reproduced the identical fault
at the same mesh position on rerun (2026-07-30, below). The fault is intermittent — c10u08
passed the whole matmul subset once and failed it 15 min later, so a single clean run is a
false negative, not a cleared box. See "fw 19.12 / KMD 2.10 does NOT fix it" and "Caveat and
open items".**

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

## fw 19.12 / KMD 2.10 does NOT fix it — intermittent, same mesh position (2026-07-30)

Host `bh-glx-110-c10u08`, fw `19.12.0.0`, KMD `2.10.0` — the newer pair. Same repo, same test
file (identical to the pushed branch tip). The matmul subset was run **twice, ~15 min apart**:

| run | verdict | core sweep |
|---|---|---|
| first | **8 passed** in 5:37 | all 120 cores bit-exact |
| rerun | **8 failed** in 4:17 | `matmul is core-dependent: {(6, 8): {21: 43}}` |

The rerun signature is identical to b06u02 — logical core (6,8), shard 21, every shape incl.
the seq640 chunk:

```
local matmul1 seq3200 {21: 2683}  matmul2 seq3200 {21: 294390}
local matmul1 seq640  {21: 159}   matmul2 seq640  {21: 26982}
core_locality base/halfN/halfM: all shard 21
```

Two things follow, and they overturn the earlier "clean box" reading:

- **fw 19.12 / KMD 2.10 does not immunize.** The same box reproduced the fault on the newer
  firmware. So the earlier fw correlation (fail on 19.8.1, pass on 19.12) was under-sampled:
  b07u08's 27/27 and c10u08's first 8/8 were each *one* run, and this fault is intermittent
  enough to pass a whole run. A single clean pass does not clear a box — a box is only cleared
  by many runs. Any "pass" in this investigation that rests on one run is now suspect.
- **Same mesh position on two independent boxes.** The mesh mapping is byte-identical across
  b06u02, b07u08 and c10u08 (row 5 = shard `[20,21,22,23]` → device `[10,14,22,18]`), so shard
  21 = mesh device 14 = row 5 col 1 on all three. The fault lands at that exact position on two
  different physical boxes. That is hard to explain as a random bad-core lottery; it points at
  something systematic about that topological slot (row 5 col 1) or the harvest pattern that
  maps a marginal core there — and it needs the *per-die physical* coordinate on each box to
  confirm whether it is the same physical slot or the same logical position landing on
  different dies.

The model-level CI case fails too. The original Blaze "Transformer Determinism" test —
`test_ds_prefill_transformer` with `with_determinism`, 5 layers, isl 25600, random weights, the
exact command this investigation started from — was re-run on c10u08 and **failed** (2 variants,
`mesh-8x4` and `fabric2d-mesh-8x4`; `-k mesh-8x4` selects both by substring). Per-layer
determinism PCC drops with depth as the marginal per-core perturbation accumulates:

```
mesh-8x4:          layer_0 0.999970 -> layer_4 0.997489  norm 0.997486  logits 0.998720
fabric2d-mesh-8x4: layer_0 0.999956 -> layer_4 0.997377  norm 0.997374  logits 0.995651
```

So both the ~3-min matmul probe and the full 24-min model test reproduce on fw 19.12 — the CI
failure this whole investigation started from is present on the newer firmware, not just the old.

Logs: `/data/nmilicevic/c10u08_matmul_fw1912.log` (matmul rerun, failing),
`/data/nmilicevic/c10u08_ci_transformer_det_fw1912.log` (CI model test, failing),
`/data/nmilicevic/c10u08_device_mapping.log` (mapping).

## Caveat and open items

- **Firmware/KMD is no longer the leading candidate — the c10u08 rerun demoted it.** The
  earlier ordering (fail on 19.8.1, pass on 19.12) rested on single runs; a fw-19.12 box has now
  reproduced the fault, so 19.12 does not immunize. What survives is a *positional, intermittent*
  signal: mesh device 14 / row 5 col 1, on two independent boxes and both fw pairs, present on
  some runs and not others. Firmware may still modulate how often it trips (operating point,
  DVFS), but it is not the on/off switch. The old telemetry constraints still hold and still
  matter: a uniform firmware cannot select 1 chip of 32 by itself (on b06u02 all 32 report
  identical `asic_fmax 1350`, `vdd 0.70-0.90`, `THERM_TRIP_COUNT 0x0`, `GDDR_UNCORR_ERRS 0x0`),
  and readback/eltwise are bit-exact on the same die so host DMA/MMIO is clean — the mechanism is
  something per-position interacting with the matmul FPU path, not the KMD.
  Reordered next steps: (1) **characterize the hit-rate** — run the sweep-only test N times on
  c10u08 and on b06u02 and report *fails/N*, so "pass" stops meaning "one lucky run"; (2) get the
  **per-die physical** coordinate of core (6,8) on each box (C++ API or `tt-triage`, since the
  Python `MeshDevice` binding has no per-device handle) and check whether row 5 col 1 is the same
  physical slot fleet-wide or the same logical position landing on different dies; (3) diff
  `tt-smi -s --snapshot_no_tty` at that slot against a slot that never fails, on
  `ENABLED_TENSIX_COL` (dev 14 = `0x3ffd`), `asic_fmax`, and VDD limits. Flashing b06u02 to
  19.12 is no longer decisive — 19.12 already fails on c10u08.
- **A wedged board is a no-result, not a failure.** If every test errors in ~20 s with
  `MMIO per-op timeout: 4B load took N us (budget=2 ms)` at `distributed.py:671`, the box failed
  at `open_mesh_device` before any kernel ran. `tt-smi -glx_reset_auto` clears it — do not use
  `tt-smi -r` on these Galaxy boxes. `tt-smi -ls` enumerating 32 chips is not proof that ttnn
  can open the mesh.

Bisection method and the decision ladder: `DETERMINISM_BISECTION.md`.
Reference-box detail: `DETERMINISM_RESULT_b07u08.md`.
