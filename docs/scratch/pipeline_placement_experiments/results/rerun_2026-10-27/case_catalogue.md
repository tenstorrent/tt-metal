## Appendix: every instance, its definition, verdict and fastest engine

Results directory `results/rerun_2026-10-27`. Times in seconds; `cap` = wall-clock cap, `lim` = the DFS engine's 10M call limit (undecided), `no?` = a SAT "no" under the default conflict budget (not used as a proof).

### Single-mesh carve: the original 17 cases (1x1 stages allowed)

| Case | Grid | Stages | Area / slices | Verdict | Decided by | py dfs | py dfs2 | py pos | fp sat | fp dfs |
|---|---|---|---|---|---|---|---|---|---|---|
| 8x8: 1x1, 14x 2x2, 1x1 | 8x8 | 14x 2x2, 2x 1x1 | 58/64 | feasible | fp dfs 0.001 s | yes 0.002 | yes 0.001 | yes 0.01 | yes 0.015 | yes 0.001 |
| 8x8: 1x1, 15x 2x2, 3x 1x1 (fills) | 8x8 | 15x 2x2, 4x 1x1 | 64/64 | feasible | py dfs 0.171 s | yes 0.171 | yes 0.247 | yes 0.362 | yes 0.222 | yes 0.592 |
| 8x8 ring of 64 x 1x1 (fills) | 8x8 | 64x 1x1 | 64/64 | feasible | py dfs 0.003 s | yes 0.003 | yes 0.003 | yes 0.014 | yes 0.015 | yes 0.005 |
| 8x16: 1x1, 30x 2x2, 1x1 | 8x16 | 30x 2x2, 2x 1x1 | 122/128 | feasible | fp dfs 0.004 s | yes 0.015 | yes 0.004 | yes 3.98 | yes 0.024 | yes 0.004 |
| 8x16: 1x1, 2x4 x14, 2x2 x3, 1x1 (fills) | 8x16 | 14x 2x4, 3x 2x2, 3x 1x1 | 127/128 | feasible | fp sat 2.82 s | yes 9.91 | yes 37.8 | yes 4.53 | yes 2.82 | cap |
| 8x16 ring of 128 x 1x1 (fills) | 8x16 | 128x 1x1 | 128/128 | feasible | py dfs 0.01 s | yes 0.01 | yes 0.013 | yes 0.059 | yes 0.069 | yes 0.066 |
| 16x16: 1x1, 62x 2x2, 1x1 | 16x16 | 62x 2x2, 2x 1x1 | 250/256 | feasible | fp sat 0.132 s | cap | cap | yes 0.485 | yes 0.132 | cap |
| random 8x8: 30 stages | 8x8 | 7x 2x2, 8x 1x2, 5x 2x1, 10x 1x1 | 64/64 | feasible | py dfs 35.7 s | yes 35.7 | yes 36.2 | yes 59.3 | yes 120 | cap |
| random 8x8: 29 stages | 8x8 | 9x 2x2, 6x 1x2, 2x 2x1, 12x 1x1 | 64/64 | feasible | py dfs2 0.02 s | yes 3.24 | yes 0.02 | yes 15.1 | yes 63.1 | cap |
| random 8x8: 31 stages | 8x8 | 8x 2x2, 4x 1x2, 4x 2x1, 15x 1x1 | 63/64 | feasible | py dfs2 0.129 s | yes 1.81 | yes 0.129 | yes 3.13 | yes 1.15 | cap |
| random 8x8: 34 stages | 8x8 | 7x 2x2, 6x 1x2, 3x 2x1, 18x 1x1 | 64/64 | feasible | py dfs2 0.159 s | yes 43.7 | yes 0.159 | yes 49.2 | yes 1.41 | yes 21.1 |
| random 8x8: 37 stages | 8x8 | 5x 2x2, 6x 1x2, 6x 2x1, 20x 1x1 | 64/64 | feasible | py dfs 0.067 s | yes 0.067 | yes 1.38 | yes 0.523 | yes 0.209 | cap |
| random 8x16 s1.0: 37 stages | 8x16 | 5x 2x4, 14x 2x2, 4x 1x2, 3x 2x1, 11x 1x1 | 121/128 | feasible | fp sat 5.64 s | yes 41.1 | yes 30 | yes 13.1 | yes 5.64 | yes 25 |
| random 8x16 s1.1: 32 stages | 8x16 | 8x 2x4, 10x 2x2, 5x 1x2, 4x 2x1, 5x 1x1 | 127/128 | open | every engine capped | cap | cap | cap | cap | cap |
| random 8x16 s1.2: 32 stages | 8x16 | 6x 2x4, 13x 2x2, 4x 1x2, 5x 2x1, 4x 1x1 | 122/128 | feasible | py dfs 0.079 s | yes 0.079 | yes 11.2 | yes 13.6 | yes 0.359 | cap |
| 7x7: 1x1, 10x 2x2 (only 9 fit) | 7x7 | 10x 2x2, 1x1 | 41/49 | infeasible | py dfs2 proves it in 0.001 s | no 2.21 | no 0.001 | no 1.91 | no? 2.12 | no 0.658 |
| 5x5 ring of 25 x 1x1 (odd: none) | 5x5 | 25x 1x1 | 25/25 | infeasible | py pos proves it in 0.008 s | no 0.621 | no 0.251 | no 0.008 | no? 0.003 | no 4.41 |

### Single-mesh carve: heterogeneous random sweep (1x1 to 2x4)

| Case | Grid | Stages | Area / slices | Verdict | Decided by | py dfs | py dfs2 | py pos | fp sat | fp dfs |
|---|---|---|---|---|---|---|---|---|---|---|
| sweep 4x8 fill60 s10: 8 stages 21/32 | 4x8 | 4x 2x2, 1x2, 3x 1x1 | 21/32 | feasible | fp dfs 0 s | yes 0.002 | yes 0.001 | yes 0.011 | yes 0.001 | yes 0 |
| sweep 4x8 fill60 s11: 6 stages 21/32 | 4x8 | 5x 2x2, 1x1 | 21/32 | feasible | fp dfs 0 s | yes 0.001 | yes 0 | yes 0.004 | yes 0.001 | yes 0 |
| sweep 4x8 fill80 s10: 10 stages 29/32 | 4x8 | 6x 2x2, 1x2, 3x 1x1 | 29/32 | feasible | py dfs2 0.001 s | yes 0.003 | yes 0.001 | yes 0.022 | yes 0.015 | yes 0.003 |
| sweep 4x8 fill80 s11: 8 stages 27/32 | 4x8 | 6x 2x2, 1x2, 1x1 | 27/32 | feasible | fp dfs 0 s | yes 0.001 | yes 0.001 | yes 0.007 | yes 0.001 | yes 0 |
| sweep 4x8 fill95 s10: 11 stages 31/32 | 4x8 | 6x 2x2, 1x2, 2x1, 3x 1x1 | 31/32 | feasible | py dfs2 0.002 s | yes 0.003 | yes 0.002 | yes 0.025 | yes 0.011 | yes 0.024 |
| sweep 4x8 fill95 s11: 9 stages 29/32 | 4x8 | 6x 2x2, 2x 1x2, 1x1 | 29/32 | feasible | fp dfs 0 s | yes 0.001 | yes 0.001 | yes 0.008 | yes 0.009 | yes 0 |
| sweep 8x4 fill60 s10: 8 stages 21/32 | 8x4 | 4x 2x2, 1x2, 3x 1x1 | 21/32 | feasible | fp dfs 0 s | yes 0.002 | yes 0.001 | yes 0.006 | yes 0.001 | yes 0 |
| sweep 8x4 fill60 s11: 6 stages 21/32 | 8x4 | 5x 2x2, 1x1 | 21/32 | feasible | fp dfs 0 s | yes 0.001 | yes 0 | yes 0.003 | yes 0.001 | yes 0 |
| sweep 8x4 fill80 s10: 10 stages 29/32 | 8x4 | 6x 2x2, 1x2, 3x 1x1 | 29/32 | feasible | fp dfs 0 s | yes 0.003 | yes 0.001 | yes 0.023 | yes 0.008 | yes 0 |
| sweep 8x4 fill80 s11: 8 stages 27/32 | 8x4 | 6x 2x2, 1x2, 1x1 | 27/32 | feasible | fp dfs 0 s | yes 0.001 | yes 0.001 | yes 0.011 | yes 0.001 | yes 0 |
| sweep 8x4 fill95 s10: 11 stages 31/32 | 8x4 | 6x 2x2, 1x2, 2x1, 3x 1x1 | 31/32 | infeasible | py dfs2 proves it in 0.018 s | no 0.045 | no 0.018 | no 0.045 | no 0.031 | no 0.041 |
| sweep 8x4 fill95 s11: 9 stages 29/32 | 8x4 | 6x 2x2, 2x 1x2, 1x1 | 29/32 | feasible | fp dfs 0 s | yes 0.001 | yes 0.001 | yes 0.018 | yes 0.001 | yes 0 |
| sweep 2x16 fill60 s10: 8 stages 21/32 | 2x16 | 4x 2x2, 1x2, 3x 1x1 | 21/32 | infeasible | py dfs2 proves it in 0.001 s | no 0.002 | no 0.001 | no 0.002 | no 0.009 | no 0.003 |
| sweep 2x16 fill60 s11: 6 stages 21/32 | 2x16 | 5x 2x2, 1x1 | 21/32 | infeasible | fp dfs proves it in 0 s | no 0.001 | no 0.001 | no 0.001 | no 0.005 | no 0 |
| sweep 2x16 fill80 s10: 10 stages 29/32 | 2x16 | 6x 2x2, 1x2, 3x 1x1 | 29/32 | infeasible | py dfs2 proves it in 0.001 s | no 0.002 | no 0.001 | no 0.003 | no 0.009 | no 0.002 |
| sweep 2x16 fill80 s11: 8 stages 27/32 | 2x16 | 6x 2x2, 1x2, 1x1 | 27/32 | infeasible | fp dfs proves it in 0.001 s | no 0.002 | no 0.001 | no 0.002 | no 0.007 | no 0.001 |
| sweep 2x16 fill95 s10: 11 stages 31/32 | 2x16 | 6x 2x2, 1x2, 2x1, 3x 1x1 | 31/32 | infeasible | py dfs2 proves it in 0.001 s | no 0.002 | no 0.001 | no 0.003 | no 0.008 | no 0.003 |
| sweep 2x16 fill95 s11: 9 stages 29/32 | 2x16 | 6x 2x2, 2x 1x2, 1x1 | 29/32 | infeasible | fp dfs proves it in 0.001 s | no 0.002 | no 0.001 | no 0.003 | no 0.007 | no 0.001 |
| sweep 8x8 fill60 s10: 12 stages 39/64 | 8x8 | 2x4, 6x 2x2, 1x2, 2x1, 3x 1x1 | 39/64 | feasible | py dfs2 0.001 s | yes 3.54 | yes 0.001 | yes 0.038 | yes 0.016 | yes 0.002 |
| sweep 8x8 fill60 s11: 12 stages 45/64 | 8x8 | 2x4, 8x 2x2, 2x 1x2, 1x1 | 45/64 | feasible | fp dfs 0.001 s | yes 1.97 | yes 0.001 | yes 0.026 | yes 0.005 | yes 0.001 |
| sweep 8x8 fill80 s10: 17 stages 52/64 | 8x8 | 2x4, 8x 2x2, 2x 1x2, 2x 2x1, 4x 1x1 | 52/64 | feasible | fp dfs 0.002 s | yes 0.005 | yes 0.002 | yes 0.006 | yes 0.007 | yes 0.002 |
| sweep 8x8 fill80 s11: 15 stages 52/64 | 8x8 | 2x4, 9x 2x2, 3x 1x2, 2x 1x1 | 52/64 | feasible | py dfs2 0.001 s | yes 0.008 | yes 0.001 | yes 0.012 | yes 0.021 | yes 0.002 |
| sweep 8x8 fill95 s10: 20 stages 63/64 | 8x8 | 2x 2x4, 8x 2x2, 3x 1x2, 2x 2x1, 5x 1x1 | 63/64 | infeasible | py dfs2 proves it in 18.8 s | cap | no 18.8 | no 37.7 | no 37.9 | cap |
| sweep 8x8 fill95 s11: 19 stages 61/64 | 8x8 | 2x4, 10x 2x2, 4x 1x2, 2x1, 3x 1x1 | 61/64 | feasible | fp dfs 0.003 s | yes 0.278 | yes 1.06 | yes 0.054 | yes 0.332 | yes 0.003 |
| sweep 8x16 fill60 s10: 26 stages 78/128 | 8x16 | 2x 2x4, 10x 2x2, 4x 1x2, 4x 2x1, 6x 1x1 | 78/128 | feasible | py dfs2 0.004 s | yes 1.48 | yes 0.004 | yes 0.059 | yes 0.024 | yes 0.008 |
| sweep 8x16 fill60 s11: 22 stages 81/128 | 8x16 | 3x 2x4, 11x 2x2, 4x 1x2, 2x1, 3x 1x1 | 81/128 | feasible | py dfs2 0.004 s | cap | yes 0.004 | yes 0.043 | yes 0.068 | yes 0.007 |
| sweep 8x16 fill80 s10: 33 stages 104/128 | 8x16 | 3x 2x4, 13x 2x2, 5x 1x2, 6x 2x1, 6x 1x1 | 104/128 | feasible | py dfs2 0.005 s | cap | yes 0.005 | yes 0.161 | yes 0.981 | yes 0.011 |
| sweep 8x16 fill80 s11: 28 stages 110/128 | 8x16 | 5x 2x4, 14x 2x2, 4x 1x2, 2x1, 4x 1x1 | 110/128 | feasible | py dfs2 0.004 s | cap | yes 0.004 | yes 0.041 | yes 0.087 | yes 0.009 |
| sweep 8x16 fill95 s10: 36 stages 122/128 | 8x16 | 5x 2x4, 13x 2x2, 5x 1x2, 7x 2x1, 6x 1x1 | 122/128 | feasible | py dfs2 0.006 s | cap | yes 0.006 | yes 59.9 | yes 13.9 | cap |
| sweep 8x16 fill95 s11: 30 stages 122/128 | 8x16 | 6x 2x4, 15x 2x2, 4x 1x2, 2x1, 4x 1x1 | 122/128 | feasible | py dfs2 0.005 s | cap | yes 0.005 | yes 14.5 | yes 9.06 | yes 0.011 |

### Single-mesh carve: no stage smaller than 1x2

| Case | Grid | Stages | Area / slices | Verdict | Decided by | py dfs | py dfs2 | py pos | fp sat | fp dfs |
|---|---|---|---|---|---|---|---|---|---|---|
| no-1x1 4x8 fill60 s20: 3 stages 20/32 | 4x8 | 2x 2x4, 2x2 | 20/32 | feasible | fp dfs 0 s | yes 0 | yes 0 | yes 0.001 | yes 0.01 | yes 0 |
| no-1x1 4x8 fill60 s21: 5 stages 22/32 | 4x8 | 2x4, 3x 2x2, 2x1 | 22/32 | feasible | fp dfs 0 s | yes 0.001 | yes 0 | yes 0.002 | yes 0.001 | yes 0 |
| no-1x1 4x8 fill80 s20: 6 stages 28/32 | 4x8 | 2x 2x4, 2x 2x2, 1x2, 2x1 | 28/32 | feasible | fp dfs 0 s | yes 0.001 | yes 0 | yes 0.003 | yes 0.009 | yes 0 |
| no-1x1 4x8 fill80 s21: 6 stages 26/32 | 4x8 | 2x4, 4x 2x2, 2x1 | 26/32 | feasible | fp dfs 0 s | yes 0.001 | yes 0 | yes 0.001 | yes 0.001 | yes 0 |
| no-1x1 4x8 fill95 s20: 7 stages 30/32 | 4x8 | 2x 2x4, 2x 2x2, 1x2, 2x 2x1 | 30/32 | infeasible | fp dfs proves it in 0.002 s | no 0.003 | no 0.002 | no 0.009 | no 0.019 | no 0.002 |
| no-1x1 4x8 fill95 s21: 7 stages 30/32 | 4x8 | 2x4, 5x 2x2, 2x1 | 30/32 | feasible | fp dfs 0 s | yes 0.001 | yes 0 | yes 0.008 | yes 0.001 | yes 0 |
| no-1x1 8x4 fill60 s20: 5 stages 24/32 | 8x4 | 2x 2x4, 2x2, 2x 2x1 | 24/32 | infeasible | fp dfs proves it in 0 s | no 0.003 | no 0.001 | no 0.002 | no 0.001 | no 0 |
| no-1x1 8x4 fill60 s21: 4 stages 20/32 | 8x4 | 2x 2x4, 1x2, 2x1 | 20/32 | infeasible | fp dfs proves it in 0 s | no 0.003 | no 0.001 | no 0.001 | no 0.001 | no 0 |
| no-1x1 8x4 fill80 s20: 6 stages 28/32 | 8x4 | 2x 2x4, 2x 2x2, 2x 2x1 | 28/32 | infeasible | fp dfs proves it in 0 s | no 0.002 | no 0.001 | no 0.002 | no 0.006 | no 0 |
| no-1x1 8x4 fill80 s21: 5 stages 28/32 | 8x4 | 3x 2x4, 1x2, 2x1 | 28/32 | infeasible | fp dfs proves it in 0 s | no 0.003 | no 0.001 | no 0.001 | no 0.001 | no 0 |
| no-1x1 8x4 fill95 s20: 7 stages 32/32 | 8x4 | 2x 2x4, 3x 2x2, 2x 2x1 | 32/32 | feasible | fp dfs 0 s | yes 0.001 | yes 0 | yes 0.001 | yes 0.001 | yes 0 |
| no-1x1 8x4 fill95 s21: 7 stages 32/32 | 8x4 | 3x 2x4, 3x 1x2, 2x1 | 32/32 | infeasible | fp dfs proves it in 0 s | no 0.003 | no 0.003 | no 0.001 | no 0.006 | no 0 |
| no-1x1 2x16 fill60 s20: 5 stages 24/32 | 2x16 | 2x 2x4, 2x2, 1x2, 2x1 | 24/32 | infeasible | fp dfs proves it in 0 s | no 0.002 | no 0.001 | no 0.001 | no 0.001 | no 0 |
| no-1x1 2x16 fill60 s21: 5 stages 22/32 | 2x16 | 2x 2x4, 3x 2x1 | 22/32 | infeasible | fp dfs proves it in 0 s | no 0.001 | no 0 | no 0.001 | no 0.001 | no 0 |
| no-1x1 2x16 fill80 s20: 6 stages 28/32 | 2x16 | 2x 2x4, 2x 2x2, 1x2, 2x1 | 28/32 | infeasible | fp dfs proves it in 0 s | no 0.001 | no 0.001 | no 0.002 | no 0.006 | no 0 |
| no-1x1 2x16 fill80 s21: 7 stages 26/32 | 2x16 | 2x 2x4, 1x2, 4x 2x1 | 26/32 | infeasible | fp dfs proves it in 0 s | no 0.001 | no 0 | no 0.002 | no 0.001 | no 0 |
| no-1x1 2x16 fill95 s20: 7 stages 30/32 | 2x16 | 2x 2x4, 2x 2x2, 1x2, 2x 2x1 | 30/32 | infeasible | fp dfs proves it in 0 s | no 0.001 | no 0.001 | no 0.002 | no 0.001 | no 0 |
| no-1x1 2x16 fill95 s21: 9 stages 32/32 | 2x16 | 2x 2x4, 2x2, 1x2, 5x 2x1 | 32/32 | infeasible | py dfs2 proves it in 0 s | no 0.001 | no 0 | no 0.002 | no 0.001 | no 0.001 |
| no-1x1 8x8 fill60 s20: 11 stages 40/64 | 8x8 | 2x 2x4, 3x 2x2, 4x 1x2, 2x 2x1 | 40/64 | feasible | fp dfs 0.001 s | yes 0.003 | yes 0.001 | yes 0.006 | yes 0.004 | yes 0.001 |
| no-1x1 8x8 fill60 s21: 9 stages 40/64 | 8x8 | 3x 2x4, 2x 2x2, 1x2, 3x 2x1 | 40/64 | feasible | fp dfs 0.001 s | yes 0.002 | yes 0.001 | yes 0.009 | yes 0.004 | yes 0.001 |
| no-1x1 8x8 fill80 s20: 14 stages 54/64 | 8x8 | 3x 2x4, 4x 2x2, 4x 1x2, 3x 2x1 | 54/64 | feasible | fp dfs 0.001 s | yes 0.008 | yes 0.014 | yes 0.015 | yes 0.005 | yes 0.001 |
| no-1x1 8x8 fill80 s21: 12 stages 54/64 | 8x8 | 4x 2x4, 3x 2x2, 2x 1x2, 3x 2x1 | 54/64 | feasible | fp dfs 0.002 s | yes 0.013 | yes 0.005 | yes 0.008 | yes 0.005 | yes 0.002 |
| no-1x1 8x8 fill95 s20: 16 stages 62/64 | 8x8 | 3x 2x4, 6x 2x2, 4x 1x2, 3x 2x1 | 62/64 | feasible | fp sat 0.829 s | yes 1.47 | yes 2.12 | yes 1.18 | yes 0.829 | yes 8.33 |
| no-1x1 8x8 fill95 s21: 15 stages 62/64 | 8x8 | 4x 2x4, 4x 2x2, 4x 1x2, 3x 2x1 | 62/64 | feasible | fp dfs 0.002 s | yes 0.018 | yes 0.002 | yes 0.093 | yes 0.059 | yes 0.002 |
| no-1x1 8x16 fill60 s20: 20 stages 82/128 | 8x16 | 4x 2x4, 9x 2x2, 3x 1x2, 4x 2x1 | 82/128 | feasible | py dfs2 0.004 s | yes 0.015 | yes 0.004 | yes 0.052 | yes 0.018 | yes 0.005 |
| no-1x1 8x16 fill60 s21: 22 stages 80/128 | 8x16 | 3x 2x4, 9x 2x2, 8x 1x2, 2x 2x1 | 80/128 | feasible | py dfs2 0.004 s | yes 0.736 | yes 0.004 | yes 0.027 | yes 0.053 | yes 0.005 |
| no-1x1 8x16 fill80 s20: 24 stages 106/128 | 8x16 | 6x 2x4, 11x 2x2, 3x 1x2, 4x 2x1 | 106/128 | feasible | py dfs2 0.004 s | yes 0.026 | yes 0.004 | yes 0.217 | yes 0.146 | yes 0.006 |
| no-1x1 8x16 fill80 s21: 28 stages 104/128 | 8x16 | 4x 2x4, 12x 2x2, 9x 1x2, 3x 2x1 | 104/128 | feasible | py dfs2 0.005 s | yes 2.61 | yes 0.005 | yes 0.191 | yes 0.028 | yes 0.007 |
| no-1x1 8x16 fill95 s20: 30 stages 122/128 | 8x16 | 6x 2x4, 13x 2x2, 4x 1x2, 7x 2x1 | 122/128 | feasible | py dfs2 0.265 s | yes 5.03 | yes 0.265 | yes 3.93 | yes 2.8 | cap |
| no-1x1 8x16 fill95 s21: 32 stages 126/128 | 8x16 | 6x 2x4, 13x 2x2, 10x 1x2, 3x 2x1 | 126/128 | open | every engine capped | cap | cap | cap | cap | cap |
| no-1x1 4x8 ring of 16 x 1x2 | 4x8 | 16x 1x2 | 32/32 | feasible | fp dfs 0 s | yes 0.002 | yes 0.001 | yes 0.025 | yes 0.037 | yes 0 |
| no-1x1 4x8 ring of 16 x 2x1 | 4x8 | 16x 2x1 | 32/32 | feasible | fp dfs 0 s | yes 0.001 | yes 0.001 | yes 0.002 | yes 0.002 | yes 0 |
| no-1x1 4x8 ring: 8x 1x2, 8x 2x1 | 4x8 | 8x 1x2, 8x 2x1 | 32/32 | feasible | py dfs2 0.003 s | yes 0.014 | yes 0.003 | yes 0.006 | yes 0.011 | yes 0.004 |
| no-1x1 8x8 ring of 32 x 1x2 | 8x8 | 32x 1x2 | 64/64 | feasible | fp dfs 0.002 s | yes 6.05 | yes 0.191 | yes 1.48 | yes 0.316 | yes 0.002 |
| no-1x1 8x8 ring of 16 x 2x2 | 8x8 | 16x 2x2 | 64/64 | feasible | fp dfs 0.001 s | yes 0.008 | yes 0.002 | yes 0.162 | yes 0.125 | yes 0.001 |
| no-1x1 8x8 ring: 8x 2x2, 16x 1x2 | 8x8 | 8x 2x2, 16x 1x2 | 64/64 | feasible | fp dfs 0.001 s | yes 0.041 | yes 0.002 | yes 0.981 | yes 1.59 | yes 0.001 |
| no-1x1 8x8 ring: 4x 2x4, 8x 2x2, 8x 1x2 | 8x8 | 4x 2x4, 8x 2x2, 8x 1x2 | 80/64 | infeasible | area 80 > 64 slices | no 0.002 | no 0.001 | no 2.07 | no 1.26 | no 34.1 |
| no-1x1 8x16 ring of 64 x 1x2 | 8x16 | 64x 1x2 | 128/128 | open | every engine capped | cap | cap | cap | cap | cap |
| no-1x1 8x16 ring of 32 x 2x2 | 8x16 | 32x 2x2 | 128/128 | feasible | fp dfs 0.005 s | yes 3.94 | yes 0.006 | yes 10 | yes 9.42 | yes 0.005 |
| no-1x1 8x16 ring: 8x 2x4, 16x 2x2, 16x 1x2 | 8x16 | 8x 2x4, 16x 2x2, 16x 1x2 | 160/128 | infeasible | area 160 > 128 slices | no 0.006 | no 0.003 | cap | cap | cap |
| no-1x1 16x16 ring of 64 x 2x2 | 16x16 | 64x 2x2 | 256/256 | open | every engine capped | cap | cap | cap | cap | cap |
| no-1x1 16x16 ring of 128 x 1x2 | 16x16 | 128x 1x2 | 256/256 | open | every engine capped | cap | cap | cap | cap | cap |
| no-1x1 5x8 ring of 20 x 1x2 (odd rows) | 5x8 | 20x 1x2 | 40/40 | feasible | fp dfs 0 s | yes 0.013 | yes 0.003 | yes 0.03 | yes 0.354 | yes 0 |
| no-1x1 6x6 ring: 9x 2x2 (area 36, 3x3 of 2x2: odd cycle) | 6x6 | 9x 2x2 | 36/36 | infeasible | py dfs2 proves it in 0.011 s | no 0.024 | no 0.011 | no 0.145 | no 0.11 | no 0.014 |
| no-1x1 7x8: 14x 2x2 (only 12 fit) | 7x8 | 14x 2x2 | 56/56 | infeasible | py dfs2 proves it in 0.001 s | no 1.19 | no 0.001 | no 16.3 | no 9.45 | no 2.52 |

### Multi-mesh rings and chains of one-slice stages: the original 18 (`multi2`), 60 s cap

| Case | Cables | Colour test | Verdict | Decided by | resolver | topo auto | topo sat | topo dfs | fp auto | fp sat | fp dfs |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 32-rank meshes ring, 1 cable/pair, 32/32 per mesh, seed 0 | m0r0-m1r2 m1r1-m0r3 | INFEASIBLE (colour proof) | infeasible | colour test (same-colour cable slices on a tight mesh) | cap | cap | cap | no 37.7 | cap | cap | cap |
| 2 32-rank meshes ring, 1 cable/pair, 32/32 per mesh, seed 1 | m0r0-m1r2 m1r0-m0r2 | feasible (colour) | feasible | topo sat 0.066 s; colour test agrees | yes 0.271 | yes 0.067 | yes 0.066 | lim | yes 0.846 | yes 0.856 | cap |
| 2 32-rank meshes ring, 4 cable/pair, 32/32 per mesh, seed 0 | 8 cables, 4 per pair | open (no single-segment choice) | feasible | topo auto 0.013 s | yes 0.038 | yes 0.013 | yes 0.013 | yes 0.013 | yes 0.014 | yes 0.014 | yes 0.087 |
| 2 32-rank meshes ring, 4 cable/pair, 32/32 per mesh, seed 1 | 8 cables, 4 per pair | feasible (colour) | feasible | resolver 0.005 s; colour test agrees | yes 0.005 | yes 0.013 | yes 0.014 | yes 0.074 | yes 0.015 | yes 0.015 | yes 0.441 |
| 4 32-rank meshes ring, 1 cable/pair, 32/32 per mesh, seed 0 | m0r3-m1r3 m1r1-m2r1 m2r2-m3r2 m3r2-m0r3 | INFEASIBLE (colour proof) | infeasible | colour test (same-colour cable slices on a tight mesh) | cap | cap | cap | cap | cap | cap | cap |
| 4 32-rank meshes ring, 1 cable/pair, 32/32 per mesh, seed 1 | m0r3-m1r3 m1r0-m2r3 m2r3-m3r3 m3r1-m0r0 | INFEASIBLE (colour proof) | infeasible | colour test (same-colour cable slices on a tight mesh) | cap | cap | cap | cap | cap | cap | cap |
| 4 32-rank meshes ring, 4 cable/pair, 32/32 per mesh, seed 0 | 16 cables, 4 per pair | feasible (colour) | feasible | topo auto 0.059 s; colour test agrees | yes 25 | yes 0.059 | yes 0.059 | cap | yes 0.079 | yes 0.072 | cap |
| 4 32-rank meshes ring, 4 cable/pair, 32/32 per mesh, seed 1 | 16 cables, 4 per pair | feasible (colour) | feasible | topo sat 0.389 s; colour test agrees | cap | yes 0.413 | yes 0.389 | cap | cap | yes 66 | cap |
| 8 32-rank meshes ring, 1 cable/pair, 32/32 per mesh, seed 0 | 8 cables, 1 per pair | INFEASIBLE (colour proof) | infeasible | colour test (same-colour cable slices on a tight mesh) | cap | cap | cap | cap | cap | cap | cap |
| 8 32-rank meshes ring, 1 cable/pair, 32/32 per mesh, seed 1 | 8 cables, 1 per pair | INFEASIBLE (colour proof) | infeasible | colour test (same-colour cable slices on a tight mesh) | cap | cap | cap | cap | cap | cap | cap |
| 8 32-rank meshes ring, 4 cable/pair, 32/32 per mesh, seed 0 | 32 cables, 4 per pair | feasible (colour) | open | every engine capped | cap | cap | cap | cap | cap | cap | cap |
| 8 32-rank meshes ring, 4 cable/pair, 32/32 per mesh, seed 1 | 32 cables, 4 per pair | feasible (colour) | feasible | topo sat 34.3 s; colour test agrees | cap | yes 34.7 | yes 34.3 | cap | cap | yes 322 | cap |
| 2 32-rank meshes line, 1 cable/pair, 32/32 per mesh, seed 0 | m0r0-m1r2 | feasible (colour) | feasible | resolver 0.005 s; colour test agrees | yes 0.005 | yes 0.119 | yes 0.118 | lim | yes 4.08 | yes 4.11 | cap |
| 2 32-rank meshes line, 4 cable/pair, 32/32 per mesh, seed 0 | m0r2-m1r0 m0r0-m1r2 m0r1-m1r1 m0r3-m1r3 | feasible (colour) | feasible | resolver 0.005 s; colour test agrees | yes 0.005 | yes 0.013 | yes 0.014 | yes 0.262 | yes 0.014 | yes 0.014 | yes 1.8 |
| 4 32-rank meshes line, 1 cable/pair, 32/32 per mesh, seed 0 | m0r3-m1r3 m1r1-m2r1 m2r2-m3r2 | INFEASIBLE (colour proof) | infeasible | colour test (same-colour cable slices on a tight mesh) | cap | cap | cap | cap | cap | cap | cap |
| 4 32-rank meshes line, 4 cable/pair, 32/32 per mesh, seed 0 | 12 cables, 4 per pair | feasible (colour) | feasible | topo auto 0.328 s; colour test agrees | yes 21.9 | yes 0.328 | yes 0.337 | lim | cap | yes 170 | cap |
| 2 32-rank meshes ring, 2 cable/pair, 24/32 per mesh, seed 0 | m0r2-m1r0 m0r3-m1r3 m1r0-m0r3 m1r1-m0r2 | open (not tight) | feasible | topo dfs 0.001 s | yes 0.003 | yes 0.009 | yes 0.009 | yes 0.001 | yes 0.011 | yes 0.011 | yes 0.003 |
| 4 32-rank meshes ring, 2 cable/pair, 24/32 per mesh, seed 0 | 8 cables, 2 per pair | open (not tight) | feasible | topo sat 0.043 s | yes 30.2 | yes 0.044 | yes 0.043 | cap | yes 0.05 | yes 0.051 | cap |

### Multi-mesh rings and chains of one-slice stages: the wider sweep (`multi3`), 30 s cap

| Case | Cables | Colour test | Verdict | Decided by | resolver | topo auto | topo sat | topo dfs | fp auto | fp sat | fp dfs |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 32-rank meshes ring, 1 cable/pair, 32/32 per mesh, seed 10 | m0r1-m1r0 m1r0-m0r3 | feasible (colour) | feasible | topo sat 0.057 s; colour test agrees | yes 0.069 | yes 0.058 | yes 0.057 | cap | yes 0.458 | yes 0.462 | cap |
| 2 32-rank meshes ring, 1 cable/pair, 32/32 per mesh, seed 11 | m0r2-m1r0 m1r3-m0r1 | INFEASIBLE (colour proof) | infeasible | colour test (same-colour cable slices on a tight mesh) | cap | cap | cap | cap | cap | cap | cap |
| 2 32-rank meshes ring, 1 cable/pair, 32/32 per mesh, seed 12 | m0r2-m1r0 m1r2-m0r0 | feasible (colour) | feasible | resolver 0.009 s; colour test agrees | yes 0.009 | yes 0.049 | yes 0.049 | cap | yes 0.745 | yes 0.742 | cap |
| 2 32-rank meshes ring, 2 cable/pair, 32/32 per mesh, seed 10 | m0r3-m1r3 m0r1-m1r0 m1r3-m0r3 m1r2-m0r1 | feasible (colour) | feasible | topo auto 0.049 s; colour test agrees | yes 16 | yes 0.049 | yes 0.049 | cap | yes 2.09 | yes 2.1 | cap |
| 2 32-rank meshes ring, 2 cable/pair, 32/32 per mesh, seed 11 | m0r0-m1r1 m0r2-m1r0 m1r1-m0r3 m1r0-m0r0 | feasible (colour) | feasible | resolver 0.005 s; colour test agrees | yes 0.005 | yes 0.013 | yes 0.012 | cap | yes 0.015 | yes 0.015 | cap |
| 2 32-rank meshes ring, 2 cable/pair, 32/32 per mesh, seed 12 | m0r0-m1r3 m0r1-m1r2 m1r1-m0r1 m1r0-m0r0 | open (no single-segment choice) | feasible | topo dfs 0.002 s | yes 0.006 | yes 0.012 | yes 0.015 | yes 0.002 | yes 0.014 | yes 0.014 | yes 0.016 |
| 2 32-rank meshes ring, 4 cable/pair, 32/32 per mesh, seed 10 | 8 cables, 4 per pair | feasible (colour) | feasible | topo auto 0.015 s; colour test agrees | cap | yes 0.015 | yes 0.015 | yes 0.074 | yes 0.377 | yes 0.38 | yes 0.442 |
| 2 32-rank meshes ring, 4 cable/pair, 32/32 per mesh, seed 11 | 8 cables, 4 per pair | feasible (colour) | feasible | resolver 0.005 s; colour test agrees | yes 0.005 | yes 0.013 | yes 0.012 | yes 0.076 | yes 0.015 | yes 0.015 | yes 0.429 |
| 2 32-rank meshes ring, 4 cable/pair, 32/32 per mesh, seed 12 | 8 cables, 4 per pair | feasible (colour) | feasible | topo auto 0.069 s; colour test agrees | cap | yes 0.069 | yes 0.069 | cap | yes 2.06 | yes 2.1 | cap |
| 4 32-rank meshes ring, 1 cable/pair, 32/32 per mesh, seed 10 | m0r1-m1r1 m1r0-m2r0 m2r2-m3r2 m3r3-m0r3 | INFEASIBLE (colour proof) | infeasible | colour test (same-colour cable slices on a tight mesh) | cap | cap | cap | cap | cap | cap | cap |
| 4 32-rank meshes ring, 1 cable/pair, 32/32 per mesh, seed 11 | m0r1-m1r1 m1r3-m2r3 m2r3-m3r2 m3r2-m0r0 | INFEASIBLE (colour proof) | infeasible | colour test (same-colour cable slices on a tight mesh) | cap | cap | cap | cap | cap | cap | cap |
| 4 32-rank meshes ring, 1 cable/pair, 32/32 per mesh, seed 12 | m0r1-m1r1 m1r2-m2r1 m2r0-m3r3 m3r2-m0r1 | INFEASIBLE (colour proof) | infeasible | colour test (same-colour cable slices on a tight mesh) | cap | cap | cap | cap | cap | cap | cap |
| 4 32-rank meshes ring, 2 cable/pair, 32/32 per mesh, seed 10 | 8 cables, 2 per pair | feasible (colour) | feasible | resolver 0.032 s; colour test agrees | yes 0.032 | yes 0.059 | yes 0.059 | cap | cap | cap | cap |
| 4 32-rank meshes ring, 2 cable/pair, 32/32 per mesh, seed 11 | 8 cables, 2 per pair | open (no single-segment choice) | open | every engine capped | cap | cap | cap | cap | cap | cap | cap |
| 4 32-rank meshes ring, 2 cable/pair, 32/32 per mesh, seed 12 | 8 cables, 2 per pair | feasible (colour) | open | every engine capped | cap | cap | cap | cap | cap | cap | cap |
| 4 32-rank meshes ring, 4 cable/pair, 32/32 per mesh, seed 10 | 16 cables, 4 per pair | feasible (colour) | feasible | topo sat 2.72 s; colour test agrees | cap | yes 2.73 | yes 2.72 | cap | yes 12.1 | yes 12.1 | cap |
| 4 32-rank meshes ring, 4 cable/pair, 32/32 per mesh, seed 11 | 16 cables, 4 per pair | feasible (colour) | feasible | resolver 0.031 s; colour test agrees | yes 0.031 | yes 0.06 | yes 0.062 | cap | yes 12.9 | yes 12.7 | cap |
| 4 32-rank meshes ring, 4 cable/pair, 32/32 per mesh, seed 12 | 16 cables, 4 per pair | feasible (colour) | feasible | topo auto 0.059 s; colour test agrees | yes 24.5 | yes 0.059 | yes 0.059 | cap | yes 0.072 | yes 0.07 | cap |
| 8 32-rank meshes ring, 1 cable/pair, 32/32 per mesh, seed 10 | 8 cables, 1 per pair | INFEASIBLE (colour proof) | infeasible | colour test (same-colour cable slices on a tight mesh) | cap | cap | cap | cap | cap | cap | cap |
| 8 32-rank meshes ring, 1 cable/pair, 32/32 per mesh, seed 11 | 8 cables, 1 per pair | INFEASIBLE (colour proof) | infeasible | colour test (same-colour cable slices on a tight mesh) | cap | cap | cap | cap | cap | cap | cap |
| 8 32-rank meshes ring, 1 cable/pair, 32/32 per mesh, seed 12 | 8 cables, 1 per pair | INFEASIBLE (colour proof) | infeasible | colour test (same-colour cable slices on a tight mesh) | cap | cap | cap | cap | cap | cap | cap |
| 8 32-rank meshes ring, 2 cable/pair, 32/32 per mesh, seed 10 | 16 cables, 2 per pair | open (no single-segment choice) | open | every engine capped | cap | cap | cap | cap | cap | cap | cap |
| 8 32-rank meshes ring, 2 cable/pair, 32/32 per mesh, seed 11 | 16 cables, 2 per pair | feasible (colour) | open | every engine capped | cap | cap | cap | cap | cap | cap | cap |
| 8 32-rank meshes ring, 2 cable/pair, 32/32 per mesh, seed 12 | 16 cables, 2 per pair | feasible (colour) | open | every engine capped | cap | cap | cap | cap | cap | cap | cap |
| 8 32-rank meshes ring, 4 cable/pair, 32/32 per mesh, seed 10 | 32 cables, 4 per pair | feasible (colour) | feasible | topo auto 2.18 s; colour test agrees | cap | yes 2.18 | yes 2.23 | cap | cap | cap | cap |
| 8 32-rank meshes ring, 4 cable/pair, 32/32 per mesh, seed 11 | 32 cables, 4 per pair | feasible (colour) | open | every engine capped | cap | cap | cap | cap | cap | cap | cap |
| 8 32-rank meshes ring, 4 cable/pair, 32/32 per mesh, seed 12 | 32 cables, 4 per pair | feasible (colour) | feasible | topo sat 14.2 s; colour test agrees | cap | yes 14.3 | yes 14.2 | cap | cap | cap | cap |
| 2 32-rank meshes line, 1 cable/pair, 32/32 per mesh, seed 10 | m0r1-m1r0 | feasible (colour) | feasible | resolver 0.829 s; colour test agrees | yes 0.829 | yes 2.81 | yes 2.8 | cap | cap | cap | cap |
| 2 32-rank meshes line, 1 cable/pair, 32/32 per mesh, seed 11 | m0r2-m1r0 | feasible (colour) | feasible | resolver 0.007 s; colour test agrees | yes 0.007 | yes 0.156 | yes 0.153 | cap | yes 8.5 | yes 8.46 | cap |
| 2 32-rank meshes line, 2 cable/pair, 32/32 per mesh, seed 10 | m0r3-m1r3 m0r1-m1r0 | feasible (colour) | open | every engine capped | cap | cap | cap | cap | cap | cap | cap |
| 2 32-rank meshes line, 2 cable/pair, 32/32 per mesh, seed 11 | m0r0-m1r1 m0r2-m1r0 | feasible (colour) | feasible | resolver 0.006 s; colour test agrees | yes 0.006 | yes 0.047 | yes 0.047 | cap | yes 4.02 | yes 4.02 | cap |
| 2 32-rank meshes line, 4 cable/pair, 32/32 per mesh, seed 10 | m0r0-m1r0 m0r2-m1r3 m0r1-m1r1 m0r3-m1r2 | feasible (colour) | feasible | resolver 0.006 s; colour test agrees | yes 0.006 | yes 0.066 | yes 0.066 | yes 0.006 | yes 1.48 | yes 1.47 | yes 0.031 |
| 2 32-rank meshes line, 4 cable/pair, 32/32 per mesh, seed 11 | m0r0-m1r3 m0r2-m1r0 m0r3-m1r2 m0r1-m1r1 | feasible (colour) | feasible | topo dfs 0.007 s; colour test agrees | yes 5.9 | yes 0.049 | yes 0.048 | yes 0.007 | yes 0.51 | yes 0.514 | yes 0.031 |
| 4 32-rank meshes line, 1 cable/pair, 32/32 per mesh, seed 10 | m0r1-m1r1 m1r0-m2r0 m2r2-m3r2 | INFEASIBLE (colour proof) | infeasible | colour test (same-colour cable slices on a tight mesh) | cap | cap | cap | cap | cap | cap | cap |
| 4 32-rank meshes line, 1 cable/pair, 32/32 per mesh, seed 11 | m0r1-m1r1 m1r3-m2r3 m2r3-m3r2 | feasible (colour) | feasible | resolver 1.05 s; colour test agrees | yes 1.05 | cap | cap | cap | cap | cap | cap |
| 4 32-rank meshes line, 2 cable/pair, 32/32 per mesh, seed 10 | 6 cables, 2 per pair | feasible (colour) | feasible | resolver 0.077 s; colour test agrees | yes 0.077 | yes 2.15 | yes 2.16 | cap | cap | cap | cap |
| 4 32-rank meshes line, 2 cable/pair, 32/32 per mesh, seed 11 | 6 cables, 2 per pair | feasible (colour) | open | every engine capped | cap | cap | cap | cap | cap | cap | cap |
| 4 32-rank meshes line, 4 cable/pair, 32/32 per mesh, seed 10 | 12 cables, 4 per pair | feasible (colour) | feasible | resolver 0.032 s; colour test agrees | yes 0.032 | yes 0.057 | yes 0.056 | cap | yes 0.07 | yes 0.067 | cap |
| 4 32-rank meshes line, 4 cable/pair, 32/32 per mesh, seed 11 | 12 cables, 4 per pair | feasible (colour) | feasible | topo sat 1.02 s; colour test agrees | cap | yes 1.02 | yes 1.02 | cap | cap | cap | cap |
| 8 32-rank meshes line, 1 cable/pair, 32/32 per mesh, seed 10 | 7 cables, 1 per pair | INFEASIBLE (colour proof) | infeasible | colour test (same-colour cable slices on a tight mesh) | cap | cap | cap | cap | cap | cap | cap |
| 8 32-rank meshes line, 1 cable/pair, 32/32 per mesh, seed 11 | 7 cables, 1 per pair | INFEASIBLE (colour proof) | infeasible | colour test (same-colour cable slices on a tight mesh) | cap | cap | cap | cap | cap | cap | cap |
| 8 32-rank meshes line, 2 cable/pair, 32/32 per mesh, seed 10 | 14 cables, 2 per pair | feasible (colour) | open | every engine capped | cap | cap | cap | cap | cap | cap | cap |
| 8 32-rank meshes line, 2 cable/pair, 32/32 per mesh, seed 11 | 14 cables, 2 per pair | feasible (colour) | open | every engine capped | cap | cap | cap | cap | cap | cap | cap |
| 8 32-rank meshes line, 4 cable/pair, 32/32 per mesh, seed 10 | 28 cables, 4 per pair | feasible (colour) | feasible | topo auto 1.76 s; colour test agrees | cap | yes 1.76 | yes 1.77 | cap | cap | cap | cap |
| 8 32-rank meshes line, 4 cable/pair, 32/32 per mesh, seed 11 | 28 cables, 4 per pair | feasible (colour) | open | every engine capped | cap | cap | cap | cap | cap | cap | cap |
| 2 32-rank meshes ring, 1 cable/pair, 24/32 per mesh, seed 10 | m0r1-m1r0 m1r0-m0r3 | open (not tight) | feasible | topo auto 0.01 s | yes 0.31 | yes 0.01 | yes 0.011 | yes 1.95 | yes 0.011 | yes 0.011 | yes 11.9 |
| 2 32-rank meshes ring, 1 cable/pair, 28/32 per mesh, seed 10 | m0r1-m1r0 m1r0-m0r3 | open (not tight) | feasible | fp auto 0.251 s | yes 0.316 | yes 0.312 | yes 0.314 | yes 3.89 | yes 0.251 | yes 0.258 | yes 25.7 |
| 2 32-rank meshes ring, 2 cable/pair, 24/32 per mesh, seed 10 | m0r3-m1r3 m0r1-m1r0 m1r3-m0r3 m1r2-m0r1 | open (not tight) | feasible | topo auto 0.01 s | cap | yes 0.01 | yes 0.01 | lim | yes 0.012 | yes 0.011 | cap |
| 2 32-rank meshes ring, 2 cable/pair, 28/32 per mesh, seed 10 | m0r3-m1r3 m0r1-m1r0 m1r3-m0r3 m1r2-m0r1 | open (not tight) | feasible | topo auto 0.011 s | cap | yes 0.011 | yes 0.011 | cap | yes 0.013 | yes 0.013 | cap |
| 4 32-rank meshes ring, 1 cable/pair, 24/32 per mesh, seed 10 | m0r1-m1r1 m1r0-m2r0 m2r2-m3r2 m3r3-m0r3 | open (not tight) | feasible | resolver 0.394 s | yes 0.394 | yes 2.88 | yes 2.88 | cap | yes 1.44 | yes 1.44 | cap |
| 4 32-rank meshes ring, 1 cable/pair, 28/32 per mesh, seed 10 | m0r1-m1r1 m1r0-m2r0 m2r2-m3r2 m3r3-m0r3 | open (not tight) | feasible | resolver 0.415 s | yes 0.415 | yes 25.2 | yes 25.3 | cap | yes 10.2 | yes 9.92 | cap |
| 4 32-rank meshes ring, 2 cable/pair, 24/32 per mesh, seed 10 | 8 cables, 2 per pair | open (not tight) | feasible | resolver 0.022 s | yes 0.022 | yes 0.043 | yes 0.044 | cap | yes 0.05 | yes 0.051 | cap |
| 4 32-rank meshes ring, 2 cable/pair, 28/32 per mesh, seed 10 | 8 cables, 2 per pair | open (not tight) | feasible | resolver 0.022 s | yes 0.022 | yes 0.408 | yes 0.413 | cap | yes 0.506 | yes 0.495 | cap |
| 8 32-rank meshes ring, 1 cable/pair, 24/32 per mesh, seed 10 | 8 cables, 1 per pair | open (not tight) | feasible | fp sat 18.1 s | cap | cap | cap | cap | yes 18.1 | yes 18.1 | cap |
| 8 32-rank meshes ring, 1 cable/pair, 28/32 per mesh, seed 10 | 8 cables, 1 per pair | open (not tight) | open | every engine capped | cap | cap | cap | cap | cap | cap | cap |
| 8 32-rank meshes ring, 2 cable/pair, 24/32 per mesh, seed 10 | 16 cables, 2 per pair | open (not tight) | open | every engine capped | cap | cap | cap | cap | cap | cap | cap |
| 8 32-rank meshes ring, 2 cable/pair, 28/32 per mesh, seed 10 | 16 cables, 2 per pair | open (not tight) | open | every engine capped | cap | cap | cap | cap | cap | cap | cap |

### Multi-mesh heterogeneous rings (`hetero`, footprint only, 60 s cap)

| Case | Area / slices | Positions | Verdict | Decided by | fp sat | fp dfs |
|---|---|---|---|---|---|---|
| hetero 2 meshes 4x8, 2 cables/pair, 16 stages 56/64, seed 0 | 56/64 | 240 | feasible | sat 0.026 s | yes 0.026 | yes 0.048 |
| hetero 2 meshes 4x8, 2 cables/pair, 20 stages 56/64, seed 1 | 56/64 | 240 | feasible | sat 0.007 s | yes 0.007 | yes 0.021 |
| hetero 2 meshes 4x8, 2 cables/pair, 23 stages 58/64, seed 2 | 58/64 | 210 | feasible | sat 0.249 s | yes 0.249 | yes 1.56 |
| hetero 2 meshes 4x8, 4 cables/pair, 12 stages 58/64, seed 0 | 58/64 | 176 | feasible | sat 0.021 s | yes 0.021 | yes 0.042 |
| hetero 2 meshes 4x8, 4 cables/pair, 19 stages 57/64, seed 1 | 57/64 | 240 | feasible | sat 0.007 s | yes 0.007 | yes 3.34 |
| hetero 2 meshes 4x8, 4 cables/pair, 13 stages 58/64, seed 2 | 58/64 | 240 | feasible | sat 0.004 s | yes 0.004 | yes 0.016 |
| hetero 4 meshes 4x8, 2 cables/pair, 34 stages 116/128, seed 0 | 116/128 | 480 | feasible | sat 0.84 s | yes 0.84 | cap |
| hetero 4 meshes 4x8, 2 cables/pair, 34 stages 116/128, seed 1 | 116/128 | 480 | feasible | sat 1.35 s | yes 1.35 | cap |
| hetero 4 meshes 4x8, 2 cables/pair, 34 stages 116/128, seed 2 | 116/128 | 480 | feasible | sat 2.23 s | yes 2.23 | cap |
| hetero 4 meshes 4x8, 4 cables/pair, 33 stages 111/128, seed 0 | 111/128 | 480 | feasible | sat 0.457 s | yes 0.457 | cap |
| hetero 4 meshes 4x8, 4 cables/pair, 28 stages 113/128, seed 1 | 113/128 | 480 | feasible | sat 0.439 s | yes 0.439 | cap |
| hetero 4 meshes 4x8, 4 cables/pair, 29 stages 111/128, seed 2 | 111/128 | 480 | feasible | sat 0.125 s | yes 0.125 | yes 10.6 |

### Multi-mesh rings with no stage smaller than 1x2 (`multi4`, footprint only, 60 s cap)

| Case | Area / slices | Positions | Verdict | Decided by | fp sat | fp dfs |
|---|---|---|---|---|---|---|
| no-1x1 domino 2 meshes 4x8, 1 cables/pair, 32 stages 64/64, seed 0 | 64/64 | 56 | open | every engine capped | cap | cap |
| no-1x1 domino 2 meshes 4x8, 1 cables/pair, 32 stages 64/64, seed 1 | 64/64 | 56 | open | every engine capped | cap | cap |
| no-1x1 domino 2 meshes 4x8, 2 cables/pair, 32 stages 64/64, seed 0 | 64/64 | 56 | feasible | dfs 0.012 s | yes 19.8 | yes 0.012 |
| no-1x1 domino 2 meshes 4x8, 2 cables/pair, 32 stages 64/64, seed 1 | 64/64 | 56 | feasible | sat 35.5 s | yes 35.5 | cap |
| no-1x1 domino 2 meshes 4x8, 4 cables/pair, 32 stages 64/64, seed 0 | 64/64 | 56 | feasible | dfs 0.011 s | yes 1.24 | yes 0.011 |
| no-1x1 domino 2 meshes 4x8, 4 cables/pair, 32 stages 64/64, seed 1 | 64/64 | 56 | feasible | dfs 0.55 s | yes 1.98 | yes 0.55 |
| no-1x1 domino 4 meshes 4x8, 1 cables/pair, 64 stages 128/128, seed 0 | 128/128 | 112 | open | every engine capped | cap | cap |
| no-1x1 domino 4 meshes 4x8, 1 cables/pair, 64 stages 128/128, seed 1 | 128/128 | 112 | open | every engine capped | cap | cap |
| no-1x1 domino 4 meshes 4x8, 2 cables/pair, 64 stages 128/128, seed 0 | 128/128 | 112 | open | every engine capped | cap | cap |
| no-1x1 domino 4 meshes 4x8, 2 cables/pair, 64 stages 128/128, seed 1 | 128/128 | 112 | open | every engine capped | cap | cap |
| no-1x1 domino 4 meshes 4x8, 4 cables/pair, 64 stages 128/128, seed 0 | 128/128 | 112 | open | every engine capped | cap | cap |
| no-1x1 domino 4 meshes 4x8, 4 cables/pair, 64 stages 128/128, seed 1 | 128/128 | 112 | open | every engine capped | cap | cap |
| no-1x1 domino 8 meshes 4x8, 4 cables/pair, 128 stages 256/256, seed 0 | 256/256 | 224 | open | every engine capped | cap | cap |
| no-1x1 domino 8 meshes 4x8, 4 cables/pair, 128 stages 256/256, seed 1 | 256/256 | 224 | open | every engine capped | cap | cap |
| no-1x1 mixed 2 meshes 4x8, 1 cables/pair, 15 stages 60/64, seed 0 | 60/64 | 176 | feasible | dfs 0.004 s | yes 0.031 | yes 0.004 |
| no-1x1 mixed 2 meshes 4x8, 1 cables/pair, 16 stages 64/64, seed 1 | 64/64 | 176 | infeasible | sat proves it in 0.813 s | no 0.813 | no 1.92 |
| no-1x1 mixed 2 meshes 4x8, 2 cables/pair, 17 stages 58/64, seed 0 | 58/64 | 120 | feasible | sat 0.022 s | yes 0.022 | yes 1.11 |
| no-1x1 mixed 2 meshes 4x8, 2 cables/pair, 17 stages 58/64, seed 1 | 58/64 | 176 | feasible | sat 0.051 s | yes 0.051 | yes 0.089 |
| no-1x1 mixed 2 meshes 4x8, 4 cables/pair, 13 stages 58/64, seed 0 | 58/64 | 176 | feasible | sat 0.004 s | yes 0.004 | yes 0.229 |
| no-1x1 mixed 2 meshes 4x8, 4 cables/pair, 19 stages 60/64, seed 1 | 60/64 | 176 | feasible | sat 0.035 s | yes 0.035 | yes 2.51 |
| no-1x1 mixed 4 meshes 4x8, 1 cables/pair, 30 stages 118/128, seed 0 | 118/128 | 352 | feasible | sat 4.93 s | yes 4.93 | cap |
| no-1x1 mixed 4 meshes 4x8, 1 cables/pair, 26 stages 118/128, seed 1 | 118/128 | 352 | feasible | sat 0.347 s | yes 0.347 | cap |
| no-1x1 mixed 4 meshes 4x8, 2 cables/pair, 30 stages 118/128, seed 0 | 118/128 | 352 | feasible | sat 1.91 s | yes 1.91 | cap |
| no-1x1 mixed 4 meshes 4x8, 2 cables/pair, 28 stages 116/128, seed 1 | 116/128 | 352 | feasible | sat 0.785 s | yes 0.785 | cap |
| no-1x1 mixed 4 meshes 4x8, 4 cables/pair, 30 stages 122/128, seed 0 | 122/128 | 352 | feasible | sat 1.27 s | yes 1.27 | cap |
| no-1x1 mixed 4 meshes 4x8, 4 cables/pair, 30 stages 116/128, seed 1 | 116/128 | 352 | feasible | sat 0.556 s | yes 0.556 | cap |
| no-1x1 mixed 8 meshes 4x8, 4 cables/pair, 62 stages 232/256, seed 0 | 232/256 | 704 | feasible | sat 24.6 s | yes 24.6 | cap |
| no-1x1 mixed 8 meshes 4x8, 4 cables/pair, 56 stages 232/256, seed 1 | 232/256 | 704 | feasible | sat 35.7 s | yes 35.7 | cap |

### Single-slice rings on one grid and across two meshes (`all` / `rings`, 60 s cap): every engine applies

| Case | Verdict | Decided by | resolver | topo auto | topo sat | topo dfs | fp auto |
|---|---|---|---|---|---|---|---|
| 4x8 ring of 32 x 1x1 | feasible | topo dfs 0 s | yes 0.001 | yes 0.003 | yes 0.003 | yes 0 | yes 0.007 |
| 8x8 ring of 64 x 1x1 | feasible | topo dfs 0.001 s | yes 0.004 | yes 0.013 | yes 0.012 | yes 0.001 | yes 0.014 |
| 8x16 ring of 128 x 1x1 | feasible | topo dfs 0.007 s | yes 0.032 | yes 0.061 | yes 0.06 | yes 0.007 | yes 0.069 |
| 16x16 ring of 256 x 1x1 | feasible | topo dfs 0.049 s | yes 0.244 | yes 0.33 | yes 0.331 | yes 0.049 | yes 0.378 |
| 5x5 ring of 25 x 1x1 (odd: none) | infeasible | topo sat proves it in 0.002 s | no 2.71 | no 0.004 | no 0.002 | no 1.53 | no 0.004 |
| 7x7 ring of 49 x 1x1 (odd: none) | infeasible | topo sat proves it in 0.011 s | cap | no 0.012 | no 0.011 | no 26.9 | no 0.013 |
| 8x8 ring of 60 x 1x1 (4 spare) | feasible | topo dfs 0.001 s | yes 0.006 | yes 0.013 | yes 0.012 | yes 0.001 | yes 0.014 |
| two 4x4 meshes, cables (0,0),(0,1) | feasible | topo dfs 0.001 s | yes 0.004 | yes 0.003 | yes 0.003 | yes 0.001 | yes 0.004 |
| two 4x4 meshes, cables same colour | infeasible | topo auto proves it in 1.75 s | no 3.51 | no 1.75 | no 1.77 | no 4.48 | no 4.8 |
| two 8x8 meshes, cables (0,0),(0,1) | feasible | topo auto 0.06 s | cap | yes 0.06 | yes 0.06 | cap | yes 0.067 |
| two 8x8 meshes, cables same colour | infeasible | colour: same-colour cable slices (or an odd grid), no Hamiltonian cycle | cap | cap | cap | cap | cap |
| two 4x8 meshes, cables same colour | infeasible | topo dfs proves it in 38.6 s | cap | cap | cap | no 38.6 | cap |

### Pre-carved slots, 2x2 and 2x4 (`slots`, 60 s cap)

| Case | Verdict | Decided by | resolver | topo auto | topo sat | topo dfs |
|---|---|---|---|---|---|---|
| 8x8 mesh tiled 2x2: ring of 16 x 2x2 | feasible | resolver 0 s | yes 0 | - | yes 0.001 | yes 0 |
| 8x8 mesh tiled 2x2: ring of 15 x 2x2 (1 spare slot) | infeasible | topo sat proves it in 0.001 s | no 0.052 | - | no 0.001 | no 0.011 |
| 8x8 mesh tiled 2x4: ring of 8 x 2x4 | feasible | resolver 0 s | yes 0 | - | yes 0 | yes 0 |
| 16x16 mesh tiled 2x2: ring of 64 x 2x2 | feasible | topo dfs 0.002 s | yes 0.004 | - | yes 0.013 | yes 0.002 |
| 4x8 mixed tiling: ring 2x4,2x2,2x2,2x2,2x2,2x4 (feasible) | feasible | resolver 0 s | yes 0 | - | yes 0 | yes 0 |
| 4x8 mixed tiling: ring 2x4,2x2,2x2,2x4,2x2,2x2 (none) | infeasible | resolver proves it in 0 s | no 0 | - | no 0 | no 0 |
| 2 meshes of 4x8 tiled 2x2, 4 cables/pair: ring of 16 x 2x2 | feasible | resolver 0 s | yes 0 | - | yes 0.001 | yes 0 |
| 4 meshes of 4x8 tiled 2x2, 4 cables/pair: ring of 32 x 2x2 | feasible | topo dfs 0 s | yes 0.001 | - | yes 0.003 | yes 0 |
| 2 meshes of 4x8 mixed tiling, 4 cables/pair: ring of 12 heterogeneous | feasible | resolver 0 s | yes 0 | - | yes 0 | yes 0 |
| 4 meshes of 4x8 tiled 2x2, 4 cables/pair: chain of 32 x 2x2 | feasible | topo dfs 0 s | yes 0.001 | - | yes 0.003 | yes 0 |

### Pre-carved slots, 1x2 and 2x1 dominoes (`slots2`, 60 s cap)

| Case | Verdict | Decided by | resolver | topo auto | topo sat | topo dfs |
|---|---|---|---|---|---|---|
| no-1x1 4x8 tiled 1x2: ring of 16 x 1x2 | feasible | resolver 0 s | yes 0 | - | yes 0.001 | yes 0 |
| no-1x1 4x8 tiled 2x1: ring of 16 x 2x1 | feasible | resolver 0 s | yes 0 | - | yes 0.001 | yes 0 |
| no-1x1 8x8 tiled 1x2: ring of 32 x 1x2 | feasible | topo dfs 0 s | yes 0.001 | - | yes 0.003 | yes 0 |
| no-1x1 8x8 tiled 1x2: ring of 31 x 1x2 (1 spare slot) | infeasible | topo sat proves it in 0.004 s | no 57.7 | - | no 0.004 | no 26.3 |
| no-1x1 8x8 half 2x2 / half 1x2: ring of 8 x 2x2 then 16 x 1x2 | feasible | topo dfs 0 s | yes 0.002 | - | yes 0.001 | yes 0 |
| no-1x1 8x8 half 2x2 / half 1x2: ring alternating 2x2,1x2 (none: 2x2 slots touch only 2 of 16 dominoes each) | infeasible | resolver proves it in 0 s | no 0 | - | no 0 | no 0 |
| no-1x1 16x16 tiled 1x2: ring of 128 x 1x2 | feasible | topo dfs 0.014 s | yes 0.034 | - | yes 0.06 | yes 0.014 |
| no-1x1 2 meshes of 4x8 tiled 1x2, 1 cables/pair: ring of 32 x 1x2 | feasible | resolver 0.005 s | yes 0.005 | - | yes 0.013 | yes 0.005 |
| no-1x1 2 meshes of 4x8 tiled 1x2, 4 cables/pair: ring of 32 x 1x2 | feasible | topo dfs 0 s | yes 0.001 | - | yes 0.003 | yes 0 |
| no-1x1 4 meshes of 4x8 tiled 1x2, 1 cables/pair: ring of 64 x 1x2 | feasible | topo sat 0.044 s | yes 1.7 | - | yes 0.044 | cap |
| no-1x1 4 meshes of 4x8 tiled 1x2, 4 cables/pair: ring of 64 x 1x2 | feasible | resolver 0.005 s | yes 0.005 | - | yes 0.013 | yes 0.037 |
| no-1x1 4 meshes of 4x8 tiled 2x1, 4 cables/pair: chain of 64 x 2x1 | feasible | topo dfs 0.002 s | yes 0.004 | - | yes 0.013 | yes 0.002 |
| no-1x1 2 meshes of 4x8 half 2x2 / half 1x2, 4 cables/pair: ring of 2 x (4 x 2x2, 8 x 1x2) | feasible | topo dfs 0 s | yes 0.001 | - | yes 0.001 | yes 0 |
