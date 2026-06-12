# Handoff: minimal_matmul N-slice × K-par auto-heuristic (run + re-fit on Blackhole)

This branch (`cglagovich/minimal-matmul-mcast-prefetch`) adds **split-K (K-parallelism)** to
`ttnn.experimental.minimal_matmul` and an **auto-slicer that jointly picks N-slicing (S) and K-par (Pk)**.
It was developed and tuned on **Wormhole 8×8**. This doc is everything a fresh instance needs to (a) run
the correctness tests, (b) run the N/K-par perf sweep, and (c) **re-fit the heuristic for Blackhole**.

## 0. What's on the branch
- **Split-K plan B** (fused on-device column reduction → single `[M,N]`, no host sum). Commit history:
  `…plan A2`, `…plan B`, `…auto-slicer: jointly pick N-slice (S) and K-par (Pk)`.
- **Auto heuristic** in `ttnn/.../minimal_matmul/device/minimal_matmul_program_factory.cpp`: when neither
  `TT_MM_NUM_SLICES` nor `TT_MM_K_SLICES` is pinned, the factory chooses `(S, Pk)` and engages fused
  K-par. **Currently gated OFF on non-power-of-2 grids (so it is a NO-OP on BH grid.y=10 until re-fit).**
- Tools (this dir): `nkpar_parse.py`, `nkpar_backtest.py`; test harness
  `tests/ttnn/nightly/unit_tests/operations/experimental/test_zz_{ksplit,fluxsweep}.py`.
- WH results for reference: `minimal_matmul_nslice_kpar_flux.md` (28 big shapes) and `…_compose.md` (LTX);
  design/resume doc: `minimal_matmul_ksplit_plan.md` (repo root).

## 1. Environment + build
```bash
source /home/cglagovich/bh_env.sh && source python_env/bin/activate   # Blackhole (ARCH_NAME=blackhole)
git submodule update --init --recursive    # if you switched commits
bash build_metal.sh                        # CPP changed -> rebuild
# reset a hung device: tt-smi -r
```

## 2. Correctness (run first, ~1 min)
```bash
# Fused split-K (plan B) -> [M,N], PCC vs torch. NOTE: M=1-tile shapes need Pk == grid.y so
# rows_per_group==1 (else M pads and the no-M-padding TT_FATAL fires). On BH grid.y=10, Pk must divide
# grid.y AND be a power of 2 -> Pk in {1,2}; for M=1 shapes use Pk=2 with M sized to >=... or pick a
# shape with M_tiles divisible by rows_per_group. Start with a multi-M-tile shape:
FL_M=320 FL_K=6144 FL_N=512 TT_MM_K_SLICES=2 TT_MM_NUM_SLICES=1 \
  pytest tests/ttnn/nightly/unit_tests/operations/experimental/test_zz_ksplit.py::test_ksplit_fused -s
# A2 host-summed variant (output [Pk*M,N], host reshapes+sums): ::test_ksplit
```
Always eyeball the printed `pcc` (expect ≥0.999). A dirty device → `pcc=nan` on the next op; `tt-smi -r`.

## 3. The auto heuristic — how it decides (the thing to re-fit)
```
G = grid.y;  small = min(Mt,Nt);  out = Mt*Nt;  cores = grid.x*grid.y
if small > 2*G:        (S,Pk) = (1,1)                       # not delivery-bound (today's engage gate)
D  = Kt * cores / out                                       # "K-dominance"
Pk = 8 if D>=280 else 4 if D>=40 else 2 if D>=20 else 1     # << these constants are WH-fit
if Nt >= 256: Pk = 1                                        # wide-N: in1 DRAM-bound, K-par regresses
clamp Pk so grid.y%Pk==0, Kt%Pk==0, Kt/Pk>=8
S  = grid.y / Pk        # rows_per_group=1 (no M-padding); spends the whole row budget S*Pk=grid.y
```
Knobs (no rebuild): `TT_MM_KPAR_D8/D4/D2/NWIDE/MINKB`. Disable entirely: `TT_MM_NO_AUTO_KPAR=1`.
Force a specific split (bypasses the heuristic): `TT_MM_NUM_SLICES=S TT_MM_K_SLICES=Pk TT_MM_K_FUSED=1`.
Arch-specific constants live in the `KParParams kp{...}` struct in the factory.

## 4. Run the N/K-par sweep (fast: one device session, ~20-40 min for all sliced shapes)
The harness opens the device ONCE, loops every (shape × combo), clears the program cache per combo so the
factory re-reads the `TT_MM_*` env, and flushes the device profiler per combo (the on-device buffer caps
at ~1000 ops). **Set the grid** via `FC_GRIDX/FC_GRIDY` (BH = 11×10):
```bash
T=tests/ttnn/nightly/unit_tests/operations/experimental/test_zz_fluxsweep.py::test_flux_sweep
FC_GRIDX=11 FC_GRIDY=10 FC_REPS=20 \
FC_SHAPES=tools/mm_sweep/shapes_big.txt FC_MANIFEST=/tmp/flux_manifest.json \
  python -m tracy -p -r -o /tmp/flux_runs -m "pytest -q $T"
```
- Combos are generated per shape to FILL the row budget (`combos_for` is grid-generic; on BH GY=10 it
  yields only `auto` + `(S=5,Pk=2)` since Pk must be a power of 2 dividing 10).
- Output: raw device data lands in `/tmp/flux_runs/.logs/cpp_device_perf_report.csv` (24 rows/combo =
  1 pcc + 3 warmup + 20 reps). The tracy `-r` post-process may warn/abort — **ignore it, the raw CSV is
  what we parse.**

Parse → readable report + best combo per shape:
```bash
python tools/mm_sweep/nkpar_parse.py /tmp/flux_runs /tmp/flux_manifest.json /tmp/bh_nkpar.md
```

## 5. Re-fit the heuristic for Blackhole (the actual task)
1. **Run the sweep** (§4) to get the measured `(S,Pk)` matrix for BH.
2. **Grid-search the thresholds** against it (hard no-regression constraint, maximize geomean):
   ```bash
   python tools/mm_sweep/nkpar_backtest.py /tmp/flux_runs /tmp/flux_manifest.json 10 11   # grid_y grid_x
   ```
   It prints BEST PARAMS `(d8,d4,d2,nwide,min_kt)`, geomean vs oracle, regressions, per-shape regret.
   (On BH the only K-par combo is Pk=2, so the search mostly learns *which* shapes to engage on, not how
   much Pk — the D-thresholds collapse toward an on/off boundary.)
3. **Write the constants into the factory**: in `minimal_matmul_program_factory.cpp`, set
   `KParParams kp{...}` per arch (branch on `device->arch()` at the marked spot) using the searched values.
4. **Lift the safety gate**: the heuristic currently does `if (num_slices > 1 && grid_y_pow2)`. For BH,
   `S = grid.y/Pk = 10/2 = 5` (a non-power-of-2 slice count). Before removing `grid_y_pow2`, CONFIRM the
   partition is correct at `num_slices=5` — run the fused correctness test (§2) at `TT_MM_NUM_SLICES=5
   TT_MM_K_SLICES=2 TT_MM_K_FUSED=1` and check PCC. If good, replace the gate with the validity it really
   needs (e.g. `grid.y % (S*Pk) == 0` already enforced by the downstream TT_FATAL).
5. **Verify end-to-end**: `test_flux_autoverify` times old (`TT_MM_NO_AUTO_KPAR=1`) vs new (auto) per
   shape — confirm geomean > 1 and **min ≥ ~0.99 (no regressions)** before declaring done:
   ```bash
   T=...::test_flux_autoverify
   FC_GRIDX=11 FC_GRIDY=10 python -m tracy -p -r -o /tmp/flux_av -m "pytest -q $T"
   # then pair old/new from /tmp/flux_av/.logs/cpp_device_perf_report.csv (24 rows/combo, 2 combos/shape)
   ```

## 6. Blackhole gotchas (learned the hard way)
- **grid.y = 10 is not a power of 2.** Only `Pk ∈ {1,2}` (must be pow2 AND divide 10); `S = grid.y/Pk`
  gives `S=5`. The WH budget logic assumed pow2 grids — hence the safety gate in §5.4. The pre-existing
  auto-N-slicer also has pow2 assumptions; if skinny shapes crash a sliced run, that's the N-slicer, not
  K-par (see `minimal-matmul-blackhole-study` notes).
- **WH levers don't transfer.** Block sizes, prefetch gates, and these K-par constants are all grid-fit.
- **`shapes_big.txt`** is the shape list (M K N per line). `auto_S>1` shapes are the sweep's subset.
- **Don't trust an empty PCC grep** = pass; the auto path can silently corrupt if a block isn't a
  subblock multiple. Always read the printed pcc.
- Reset on any hang/crash before the next measurement: `tt-smi -r` (or `tt-smi -glx_reset` on wh-glx).
