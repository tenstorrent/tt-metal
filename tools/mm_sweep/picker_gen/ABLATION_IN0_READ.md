# in0-read ablation (baseline / skip-redundant / skip-all)

Test-only compile-gated diagnostic (`diag_in0_read_mask`, TT_REGIME_A_DIAG_IN0 env; reflection program-cache-hashed; public API + mask-0 binaries unchanged). Skipped reads preserve CB reserve/push/pop, in0 ring forwarding, semaphores, compute, reduction, output writes exactly (no zero-fill, no removed downstream) — outputs for masks 1/2 are intentionally invalid; PCC asserted only for mask 0.

One persistent device session per relaunch; 2 warmup + 16 timed resident-input iters/mode; 3 relaunches/shape with mode block order reversed on alternate relaunches; kernel wall + per-RISC spans from the device profiler (run-host-id demux). Tuples `(Ns,Pk,Sm,kb,nsb)`. Commit `ce79cca7f79` + this diagnostic; BH p150b, 1.35 GHz, fw 19.5.0, KMD 2.4.1; peak DRAM ref 512 GB/s.

> **Skip deltas are UPPER BOUNDS** on any cross-Ns in0-sharing win: they remove the DRAM reads entirely and **exclude the NoC-copy cost** that real sharing (one Ns group reads, then distributes to the other rings) would add.

## Summary

| shape | cfg | baseline us | skip-redundant us (Δ, %) | skip-all us (Δ, %) | per-RISC B/N/T (base→skipAll) | theo redundant / all-in0 DRAM us | exposed frac R / A | exposure R / A | baseline PCC (replay) |
|---|---|---|---|---|---|---|---|---|---|
| 256x2048x2048 | (2, 2, 3, 4, 4) | 36.74 | 36.31 (+0.42, +1.2%) | 33.34 (+3.39, +9.2%) | 36.55/35.71/35.36 → 32.91/32.89/32.02 | 2.0 / 4.1 | 0.208 / 0.829 | partially-exposed / partially-exposed | 1.00000 (1.00000) |
| 256x2048x6144 | (3, 2, 2, 2, 4) | 84.78 | 79.09 (+5.68, +6.7%) | 77.27 (+7.51, +8.9%) | 83.48/84.51/84.02 → 77.17/76.37/76.05 | 4.1 / 6.1 | 1.387 / 1.222 | fully-exposed / fully-exposed | 1.00002 (1.00002) |
| 512x6144x2304 | (2, 6, 1, 2, 1) | 170.76 | 164.67 (+6.09, +3.6%) | 145.69 (+25.07, +14.7%) | 170.79/165.93/169.88 → 146.29/141.49/145.44 | 12.3 / 24.6 | 0.495 / 1.02 | partially-exposed / fully-exposed | 1.00002 (1.00002) |
| 512x6144x4608 | (2, 6, 1, 4, 1) | 224.47 | 230.59 (-6.12, -2.7%) | 200.00 (+24.46, +10.9%) | 224.2/220.61/223.76 → 199.98/192.11/199.18 | 12.3 / 24.6 | -0.498 / 0.995 | hidden / fully-exposed | 1.00009 (1.00009) |

## Raw spread

| shape | mode | median us | IQR us | spread% | n |
|---|---|---|---|---|---|
| 256x2048x2048 | baseline | 36.74 | 0.70 | 6.9 | 48 |
| 256x2048x2048 | skip_redundant | 36.31 | 0.68 | 7.2 | 48 |
| 256x2048x2048 | skip_all | 33.34 | 1.42 | 16.2 | 48 |
| 256x2048x6144 | baseline | 84.78 | 1.84 | 6.2 | 48 |
| 256x2048x6144 | skip_redundant | 79.09 | 2.62 | 10.0 | 48 |
| 256x2048x6144 | skip_all | 77.27 | 2.19 | 8.2 | 48 |
| 512x6144x2304 | baseline | 170.76 | 2.02 | 4.3 | 48 |
| 512x6144x2304 | skip_redundant | 164.67 | 4.49 | 8.3 | 48 |
| 512x6144x2304 | skip_all | 145.69 | 3.07 | 5.3 | 48 |
| 512x6144x4608 | baseline | 224.47 | 2.02 | 3.1 | 48 |
| 512x6144x4608 | skip_redundant | 230.59 | 4.13 | 7.1 | 48 |
| 512x6144x4608 | skip_all | 200.00 | 1.59 | 3.3 | 48 |

## Interpretation

- **exposed fraction** = measured skip delta ÷ theoretical DRAM time of the removed reads (all-in0 for skip-all, redundant duplicates for skip-redundant). ~0 ⇒ the reads are hidden behind compute/in1/ring (removing them frees no wall time); ~1 ⇒ fully exposed on the critical path.
- **256x2048x2048**: skip-redundant +1.2% (exposed 0.208 ⇒ partially-exposed); skip-all +9.2% (exposed 0.829 ⇒ partially-exposed). The cross-Ns dedup opportunity here removes 2.0 us of redundant DRAM traffic; realizable upside is **at most** the skip-redundant delta (+0.42 us) minus NoC-copy cost.
- **256x2048x6144**: skip-redundant +6.7% (exposed 1.387 ⇒ fully-exposed); skip-all +8.9% (exposed 1.222 ⇒ fully-exposed). The cross-Ns dedup opportunity here removes 4.1 us of redundant DRAM traffic; realizable upside is **at most** the skip-redundant delta (+5.68 us) minus NoC-copy cost.
- **512x6144x2304**: skip-redundant +3.6% (exposed 0.495 ⇒ partially-exposed); skip-all +14.7% (exposed 1.02 ⇒ fully-exposed). The cross-Ns dedup opportunity here removes 12.3 us of redundant DRAM traffic; realizable upside is **at most** the skip-redundant delta (+6.09 us) minus NoC-copy cost.
- **512x6144x4608**: skip-redundant -2.7% (exposed -0.498 ⇒ hidden); skip-all +10.9% (exposed 0.995 ⇒ fully-exposed). The cross-Ns dedup opportunity here removes 12.3 us of redundant DRAM traffic; realizable upside is **at most** the skip-redundant delta (-6.12 us) minus NoC-copy cost.
