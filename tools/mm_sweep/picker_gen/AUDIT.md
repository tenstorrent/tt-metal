# Regime-A picker + planner audit

Source of truth (read 2026-07-22, branch `cglagovich/regime-a-ltxflux-opt`):
- Picker: `ttnn/cpp/.../regime_a_matmul/device/regime_a_matmul_config.cpp` (`auto_select_config`)
- Planner / feasibility: `ttnn/cpp/.../regime_a_matmul/device/regime_a_matmul_plan.hpp` (`build_plan`)

The offline Python mirror `regime_a_model.py` reproduces both **exactly**; this file documents the
behaviour it mirrors so the sweep + heuristic can be reasoned about independently.

## 1. Config vector

`RegimeAMatmulConfig = (Pk, Ns, Sm, kb, nsb)` = `(k_slices, n_slices, m_slices, k_block_tiles,
n_subblock_tiles)`. All tile/slice counts. Fixed platform constants (plan.hpp):

| const | value | meaning |
|---|---|---|
| `kNumBanks` (G) | 8 | in1 DRAM width-shard fixed to 8 banks |
| `kTileBytesBf16` | 2048 | bf16 tile |
| `kTileBytesFp32` | 4096 | fp32 tile |
| `kL1BudgetBytes` | 1440·1024 | usable L1/core (BH) |
| `kMinCores` | 16 | picker core-count window low |
| `kMaxCores` | 104 | picker core-count window high |

Grid on BH p150b = 11×10 = 110 cells (no holes in v1) → `available = 110`.

## 2. Planner geometry (`build_plan`)

Normalises `Pk,Ns,Sm,kb = max(1,·)`; `nsb=0 ⇒ N_own`. Then:

- `N_band = ceil(Nt/8)` (per-bank physical width; also in1 shard cols).
- `K_slice_capacity = rup(ceil(Kt/Pk), kb·8)` — K tiles/slice, rounded so `K_num_blocks_eff = cap/kb`
  is a multiple of 8.
- `W = K_num_blocks_eff / 8` — ring shard width (in0 ring granularity).
- `M_block_capacity = ceil(Mt/Sm)`.
- `N_own = ceil(N_band/Ns)`; `N_sub = nsb (or N_own)`; `N_bpc = ceil(N_own/N_sub)`;
  `N_slice_capacity = N_bpc·N_sub`.
- `preaders = Pk·Ns·Sm`; `num_cores = 8·preaders`; `mfac = Ns·Sm` (reduction stride).
- `waste_k = Pk·K_slice_capacity/Kt − 1`; `waste_n = 8·Ns·N_slice_capacity/Nt − 1`.

**CB sizing / L1** (bytes):
```
cb0 = M_block_capacity · K_slice_capacity        (bf16, in0 k-slice resident)
cb1 = 4 · kb · N_sub                              (bf16, in1 depth 4)
cb2 = 2 · M_block_capacity · N_sub                (bf16, out depth 2)
cb3 = M_block_capacity · N_sub                    (fp32 accumulator)
cb7 = 2 · M_block_capacity · N_sub  if Pk>1 else 0 (bf16 reduce running sum)
l1  = (cb0+cb1+cb2+cb7)·2048 + cb3·4096
```

## 3. Feasibility (the pre-launch reject set)

`build_plan` returns an error (→ config rejected before device launch) when ANY of:
1. `Mt|Kt|Nt == 0`.
2. `Sm > Mt`.
3. `Pk > Kt` (empty k-slice).
4. `opt0/opt1` not size 8 (device gives 8 bank-adjacent cores → always OK on BH).
5. **`nt_width_shard_feasible(Nt)` false**: `7·ceil(Nt/8) >= Nt`. Function of Nt only.
   Small Nt (e.g. Nt=1..7 within a band) can't shard 8 banks without an empty bank.
6. `nsb > N_own`.
7. `K_num_blocks_eff % 8 != 0` (internal; can't trigger given the rup).
8. `num_cores > available` (110): `8·Pk·Ns·Sm > 110` ⇒ `Pk·Ns·Sm ≤ 13`.
9. **`l1_bytes > kL1BudgetBytes`** — the dominant deep-K / wide-N limiter (cb0 ∝ M_block·K_slice_cap).
10. collision-free placement failure (grid too full) — practically subsumed by (8) on 11×10.
11. empty ownership (`valid_k|valid_m|valid_n == 0`) — reduce Pk/Ns/Sm.

The **picker's** `pick_plan` mirrors a SUPERSET of these with two EXTRA pruning gates it applies before
the L1 check (so the picker only ever *considers* a stricter feasible set than the planner accepts):
- `cores` in `[kMinCores=16, kMaxCores=104]` ⇒ `2 ≤ Pk·Ns·Sm ≤ 13`.
- `waste_k > 0.20` reject; `waste_n > 0.20` reject.

The sweep must enumerate the **planner** feasible set (device-launchable), NOT the picker's stricter
set — the directive explicitly forbids silently pruning candidates with the picker.

## 4. Production picker (`auto_select_config`)

Two-tier, keyed on tile dims `(Mt,Kt,Nt)`:

### 4a. Lookup table (`kTable`) — 44 entries
The 20 FLUX/LTX production shapes (100% oracle) + 23 measured M-scaling campaign winners (+3..+32% vs
fallback, gated zero-regression) + a few extras. If `(Mt,Kt,Nt)` hits, that config is returned verbatim.
This is the "hardcoded shape table" the directive wants to **replace with formulas**.

### 4b. Cost-model fallback (miss) — two steps
Constants: `Csat=24, Acap=6, Kbcap=2, Kk=0.5, Aa=2.0, Ovl=1.0, Start=0.0, Wst=0.5` (grid-searched,
geomean 96.8% on the 3262-config oracle); `Rk=0.8, MSplitMargin=0.03, NbandMax=2`.

**Candidate ranges:** `Pk∈1..12, Ns∈1..6, kb∈{1,2,4,8}, nsb∈1..N_own`.

**Step 1 — Sm=1 ANCHOR:** enumerate all feasible (via `pick_plan`) Sm=1 candidates, pick min
`pick_cost`:
```
readT   = Kt·Nt / min(cores, 24)
comp_pc = M_block · N_own · K_slice_cap
area    = min(M_block·nsb, 6);  kbe = min(kb,2)
compT   = comp_pc / ( (kbe/(kbe+0.5)) · (area/(area+2.0)) )
ovlT    = 1.0 · comp_pc / N_bpc
cost    = (max(readT,compT) + ovlT) · (1 + 0.5·(waste_k+waste_n))
```
`readT` = critical-bank DRAM read time (cores capped at 24 = the ~measured read-saturation core count);
`compT` = compute time with diminishing returns from deeper kb and larger M·nsb tiles; `ovlT` =
un-overlapped compute tail (÷ N_bpc pipelining); waste multiplier penalises schedule zero-fill.

**Step 2 — NARROW-N M-split hysteresis:** only if `N_band ≤ 2` AND `Mt ≥ 2`, enumerate feasible Sm∈2..Mt
candidates, min `pick_cost_v3 = pick_cost + Rk·(Pk−1)·M_block·N_own` (adds a split-K reduction penalty).
Adopt the best Sm>1 **only if** its v3 cost `< anchor_v3_cost·(1−0.03)`. Else return the anchor.
By construction: wide-N shapes never get Sm>1 from the fallback, and the Sm=1 ranking is byte-identical
to the pre-v3 deployed model → zero regression.

## 5. What the directive asks us to change

Replace 4a (the 44-entry lookup) with **formulas + a few justified boundary conditions** that generalise
to arbitrary Mt=1..8 shapes, keeping the hierarchical structure (choose (Pk,Ns,Sm) first from
parallelism / critical-bank DRAM work / in0-ring cost / M-split forwarding / split-K reduction /
utilisation; then (kb,nsb) from block count / W / transaction granularity / compute shape / L1). The
current fallback (4b) is already a formula but is known to mis-pick on M-scaling / deep-K shapes (the
reason 23 table entries exist). The offline heuristic in `regime_a_model.py` is where the replacement is
developed and measured; **no C++ change in this phase.**

## 6. Known mis-pick signatures (from prior campaign memory + table entries)
- Fallback prefers N-split (Ns>1) where **M-split (Sm>1)** wins on narrow-N low-AI Mt≥4
  (reduction/forwarding dominate). Table entries `{8,·,·}` M-split winners encode this.
- Fallback under-provisions **kb/nsb** on deep-K / wide shapes (in1-read-bound): needs `kb≥2, nsb≥3`
  floor (`128x15360x1536` etc.).
- Efficiency falls monotonically with Mt (median %512 ≈ 91/89/81/59 for Mt=1/2/4/8) — the picker must be
  most careful for large Mt.
