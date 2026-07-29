# Host-minimization solve — complete experiment log

Full record of every experiment and result. Companion docs:
`HOST_MINIMIZATION_INVESTIGATION.md` (goals + directions), `warm_solve_experiment_results.md` (tables),
`72stage_intermesh_solve_profile.txt` (profile), `WARM_SOLVE_EXPERIMENT_PLAN.md`, `WARM_SOLVE_SPEEDUP_BRAINSTORM.md`,
`SAME_RANK_HOST_SOLVE_SIMPLIFICATION_PLAN.md`.

Setup: direct 36/20/16-rank `generate_rank_bindings` on the SC36/SC20/SC16 mocks + the pipeline-sweep MGDs.
Modes via `TT_TOPO_SAT_MIN_MODE` (0 baseline, 1 skipdescent, 2 greedy, 3 hardcap-only, 4 atmostk);
`TT_TOPO_SAT_NO_MINHOST=1` disables minimization (plain embedding). Metrics from `[intermesh-solve]` +
`[topo-sat-profile]` (build `-DTT_METAL_ENABLE_LOGGING=ON`, run `TT_LOGGER_LEVEL=debug`).

## Experiment 1 — 72-stage intermesh profile (where time goes)
Full-fill 4x4 72-stage, SC36. Inter-mesh solve = 291.7 ms total: encode 23.3 ms, `minimize.warm_solve`
258.4 ms (88%). Encoding is cheap; the SOLVE dominates. (Detail: 72stage_intermesh_solve_profile.txt.)

## Experiment 2 — 5-mode comparison, 150s cap, single solution (150 runs)
Modes x {SC36,SC20,SC16} x {2x4,4x4} x fitting stages. Reached-k_min / found>=1 / timeout:
baseline 21/22/8 ; skipdescent 22/23/7 ; hardcap-only 22/23/7 ; atmostk 19/20/10 ; greedy 2/25/5.
Full per-run detail below.

| mode | mock | shape | stage | k_min | result | hosts(occ) | greedy_ok | intermesh_ms | wall_ms |
|---|---|---|---|---|---|---|---|---|---|
| baseline | SC16 | 2x4 | 16 | 4 | OK | 4 | NA | 3.0 | 6455 |
| baseline | SC16 | 2x4 | 32 | 8 | ok(non-min) | 9 | NA | 743.8 | 1934 |
| baseline | SC16 | 2x4 | 64 | 16 | OK | 16 | NA | 59.3 | 6060 |
| baseline | SC16 | 4x4 | 8 | ? | TIMEOUT | ? | NA | 0.1 | 151019 |
| baseline | SC16 | 4x4 | 16 | ? | TIMEOUT | ? | NA | 0.0 | 151018 |
| baseline | SC16 | 4x4 | 32 | ? | TIMEOUT | ? | NA | 0.2 | 151017 |
| baseline | SC20 | 2x4 | 16 | 4 | OK | 4 | NA | 4.7 | 12286 |
| baseline | SC20 | 2x4 | 32 | 8 | OK | 8 | NA | 1195.8 | 6262 |
| baseline | SC20 | 2x4 | 64 | 16 | OK | 16 | NA | 9986.3 | 17689 |
| baseline | SC20 | 2x4 | 80 | 20 | OK | 20 | NA | 19322.9 | 27635 |
| baseline | SC20 | 4x4 | 8 | 8 | OK | 8 | NA | 1.8 | 5939 |
| baseline | SC20 | 4x4 | 16 | 16 | OK | 16 | NA | 2.7 | 5928 |
| baseline | SC20 | 4x4 | 32 | 32 | OK | 32 | NA | 4.3 | 1336 |
| baseline | SC20 | 4x4 | 40 | 40 | OK | 40 | NA | 53.8 | 1510 |
| baseline | SC36 | 2x4 | 16 | 4 | OK | 4 | NA | 24.4 | 13052 |
| baseline | SC36 | 2x4 | 32 | 8 | OK | 8 | NA | 3229.7 | 12924 |
| baseline | SC36 | 2x4 | 64 | 16 | TIMEOUT | ? | NA | ? | 152030 |
| baseline | SC36 | 2x4 | 80 | 20 | OK | 20 | NA | 111302.9 | 114616 |
| baseline | SC36 | 2x4 | 96 | 24 | TIMEOUT | ? | NA | ? | 152026 |
| baseline | SC36 | 2x4 | 112 | 28 | TIMEOUT | ? | NA | ? | 152027 |
| baseline | SC36 | 2x4 | 128 | 32 | TIMEOUT | ? | NA | ? | 152028 |
| baseline | SC36 | 2x4 | 144 | 36 | TIMEOUT | ? | NA | ? | 152020 |
| baseline | SC36 | 4x4 | 8 | 8 | OK | 8 | NA | 4.3 | 6327 |
| baseline | SC36 | 4x4 | 16 | 16 | OK | 16 | NA | 6.1 | 6316 |
| baseline | SC36 | 4x4 | 32 | 32 | OK | 32 | NA | 9.8 | 6898 |
| baseline | SC36 | 4x4 | 40 | 40 | OK | 40 | NA | 12.3 | 11549 |
| baseline | SC36 | 4x4 | 48 | 48 | OK | 48 | NA | 16.0 | 7710 |
| baseline | SC36 | 4x4 | 56 | 56 | OK | 56 | NA | 29.2 | 7647 |
| baseline | SC36 | 4x4 | 64 | 64 | OK | 64 | NA | 19.8 | 6052 |
| baseline | SC36 | 4x4 | 72 | 72 | OK | 72 | NA | 190.2 | 7824 |
| skipdescent | SC16 | 2x4 | 16 | 4 | OK | 4 | NA | 3.0 | 11156 |
| skipdescent | SC16 | 2x4 | 32 | 8 | ok(non-min) | 9 | NA | 336.5 | 6239 |
| skipdescent | SC16 | 2x4 | 64 | 16 | OK | 16 | NA | 58.6 | 6374 |
| skipdescent | SC16 | 4x4 | 8 | ? | TIMEOUT | ? | NA | 0.1 | 151017 |
| skipdescent | SC16 | 4x4 | 16 | ? | TIMEOUT | ? | NA | 0.1 | 151023 |
| skipdescent | SC16 | 4x4 | 32 | ? | TIMEOUT | ? | NA | 0.2 | 151018 |
| skipdescent | SC20 | 2x4 | 16 | 4 | OK | 4 | NA | 4.4 | 6196 |
| skipdescent | SC20 | 2x4 | 32 | 8 | OK | 8 | NA | 34.7 | 7050 |
| skipdescent | SC20 | 2x4 | 64 | 16 | OK | 16 | NA | 2281.1 | 11648 |
| skipdescent | SC20 | 2x4 | 80 | 20 | OK | 20 | NA | 19401.5 | 27862 |
| skipdescent | SC20 | 4x4 | 8 | 8 | OK | 8 | NA | 1.8 | 6238 |
| skipdescent | SC20 | 4x4 | 16 | 16 | OK | 16 | NA | 2.7 | 6165 |
| skipdescent | SC20 | 4x4 | 32 | 32 | OK | 32 | NA | 4.9 | 6759 |
| skipdescent | SC20 | 4x4 | 40 | 40 | OK | 40 | NA | 55.3 | 11869 |
| skipdescent | SC36 | 2x4 | 16 | 4 | OK | 4 | NA | 17.2 | 12115 |
| skipdescent | SC36 | 2x4 | 32 | 8 | OK | 8 | NA | 479.7 | 6049 |
| skipdescent | SC36 | 2x4 | 64 | 16 | OK | 16 | NA | 18544.2 | 26287 |
| skipdescent | SC36 | 2x4 | 80 | 20 | OK | 20 | NA | 25736.4 | 40351 |
| skipdescent | SC36 | 2x4 | 96 | 24 | TIMEOUT | ? | NA | ? | 152026 |
| skipdescent | SC36 | 2x4 | 112 | 28 | TIMEOUT | ? | NA | ? | 152026 |
| skipdescent | SC36 | 2x4 | 128 | 32 | TIMEOUT | ? | NA | ? | 152027 |
| skipdescent | SC36 | 2x4 | 144 | 36 | TIMEOUT | ? | NA | ? | 152029 |
| skipdescent | SC36 | 4x4 | 8 | 8 | OK | 8 | NA | 4.5 | 6289 |
| skipdescent | SC36 | 4x4 | 16 | 16 | OK | 16 | NA | 6.1 | 6326 |
| skipdescent | SC36 | 4x4 | 32 | 32 | OK | 32 | NA | 10.0 | 1797 |
| skipdescent | SC36 | 4x4 | 40 | 40 | OK | 40 | NA | 11.4 | 6451 |
| skipdescent | SC36 | 4x4 | 48 | 48 | OK | 48 | NA | 13.5 | 7488 |
| skipdescent | SC36 | 4x4 | 56 | 56 | OK | 56 | NA | 15.0 | 13250 |
| skipdescent | SC36 | 4x4 | 64 | 64 | OK | 64 | NA | 17.1 | 7531 |
| skipdescent | SC36 | 4x4 | 72 | 72 | OK | 72 | NA | 185.6 | 13299 |
| greedy | SC16 | 2x4 | 16 | 4 | OK | 4 | true | 2.9 | 11504 |
| greedy | SC16 | 2x4 | 32 | 8 | ok(non-min) | 0 | false | 39.8 | 6729 |
| greedy | SC16 | 2x4 | 64 | 16 | OK | 16 | true | 10.7 | 6035 |
| greedy | SC16 | 4x4 | 8 | ? | TIMEOUT | ? | NA | 0.0 | 151023 |
| greedy | SC16 | 4x4 | 16 | ? | TIMEOUT | ? | NA | 0.1 | 151031 |
| greedy | SC16 | 4x4 | 32 | ? | TIMEOUT | ? | NA | 0.2 | 151049 |
| greedy | SC20 | 2x4 | 16 | 4 | ok(non-min) | 0 | false | 7.5 | 7366 |
| greedy | SC20 | 2x4 | 32 | 8 | ok(non-min) | 0 | false | 15.5 | 7214 |
| greedy | SC20 | 2x4 | 64 | 16 | ok(non-min) | 0 | false | 1073.5 | 6182 |
| greedy | SC20 | 2x4 | 80 | 20 | ok(non-min) | 0 | false | 9717.0 | 18220 |
| greedy | SC20 | 4x4 | 8 | 8 | ok(non-min) | 0 | false | 2.1 | 6253 |
| greedy | SC20 | 4x4 | 16 | 16 | ok(non-min) | 0 | false | 3.9 | 1216 |
| greedy | SC20 | 4x4 | 32 | 32 | ok(non-min) | 0 | false | 6.3 | 6063 |
| greedy | SC20 | 4x4 | 40 | 40 | ok(non-min) | 0 | false | 474.8 | 7311 |
| greedy | SC36 | 2x4 | 16 | 4 | ok(non-min) | 0 | false | 17.8 | 6265 |
| greedy | SC36 | 2x4 | 32 | 8 | ok(non-min) | 0 | false | 60.6 | 6380 |
| greedy | SC36 | 2x4 | 64 | 16 | ok(non-min) | 0 | false | 2239.4 | 6132 |
| greedy | SC36 | 2x4 | 80 | 20 | ok(non-min) | 0 | false | 2014.8 | 11093 |
| greedy | SC36 | 2x4 | 96 | 24 | ok(non-min) | 0 | false | 1868.1 | 16567 |
| greedy | SC36 | 2x4 | 112 | 28 | ok(non-min) | 0 | false | 1446.8 | 11000 |
| greedy | SC36 | 2x4 | 128 | 32 | TIMEOUT | 0 | false | ? | 151029 |
| greedy | SC36 | 2x4 | 144 | 36 | TIMEOUT | 0 | false | ? | 152028 |
| greedy | SC36 | 4x4 | 8 | 8 | ok(non-min) | 0 | false | 4.1 | 12000 |
| greedy | SC36 | 4x4 | 16 | 16 | ok(non-min) | 0 | false | 12.5 | 6431 |
| greedy | SC36 | 4x4 | 32 | 32 | ok(non-min) | 0 | false | 13.3 | 7089 |
| greedy | SC36 | 4x4 | 40 | 40 | ok(non-min) | 0 | false | 970.8 | 6257 |
| greedy | SC36 | 4x4 | 48 | 48 | ok(non-min) | 0 | false | 1348.5 | 14350 |
| greedy | SC36 | 4x4 | 56 | 56 | ok(non-min) | 0 | false | 670.4 | 6063 |
| greedy | SC36 | 4x4 | 64 | 64 | ok(non-min) | 0 | false | 530.3 | 7181 |
| greedy | SC36 | 4x4 | 72 | 72 | ok(non-min) | 0 | false | 625.6 | 12118 |
| hardcap-only | SC16 | 2x4 | 16 | 4 | OK | 4 | NA | 3.1 | 6319 |
| hardcap-only | SC16 | 2x4 | 32 | 8 | ok(non-min) | 0 | NA | 183.4 | 7219 |
| hardcap-only | SC16 | 2x4 | 64 | 16 | OK | 16 | NA | 58.6 | 1551 |
| hardcap-only | SC16 | 4x4 | 8 | ? | TIMEOUT | ? | NA | 0.0 | 151041 |
| hardcap-only | SC16 | 4x4 | 16 | ? | TIMEOUT | ? | NA | 0.1 | 151052 |
| hardcap-only | SC16 | 4x4 | 32 | ? | TIMEOUT | ? | NA | 0.2 | 151037 |
| hardcap-only | SC20 | 2x4 | 16 | 4 | OK | 4 | NA | 4.2 | 6296 |
| hardcap-only | SC20 | 2x4 | 32 | 8 | OK | 8 | NA | 99.0 | 5919 |
| hardcap-only | SC20 | 2x4 | 64 | 16 | OK | 16 | NA | 10793.7 | 18917 |
| hardcap-only | SC20 | 2x4 | 80 | 20 | OK | 20 | NA | 19065.7 | 33203 |
| hardcap-only | SC20 | 4x4 | 8 | 8 | OK | 8 | NA | 1.9 | 6159 |
| hardcap-only | SC20 | 4x4 | 16 | 16 | OK | 16 | NA | 2.9 | 1247 |
| hardcap-only | SC20 | 4x4 | 32 | 32 | OK | 32 | NA | 4.4 | 6162 |
| hardcap-only | SC20 | 4x4 | 40 | 40 | OK | 40 | NA | 53.8 | 7161 |
| hardcap-only | SC36 | 2x4 | 16 | 4 | OK | 4 | NA | 43.3 | 11951 |
| hardcap-only | SC36 | 2x4 | 32 | 8 | OK | 8 | NA | 552.2 | 7815 |
| hardcap-only | SC36 | 2x4 | 64 | 16 | OK | 16 | NA | 11315.9 | 19215 |
| hardcap-only | SC36 | 2x4 | 80 | 20 | TIMEOUT | ? | NA | ? | 152029 |
| hardcap-only | SC36 | 2x4 | 96 | 24 | OK | 24 | NA | 36786.0 | 40772 |
| hardcap-only | SC36 | 2x4 | 112 | 28 | TIMEOUT | ? | NA | ? | 152027 |
| hardcap-only | SC36 | 2x4 | 128 | 32 | TIMEOUT | ? | NA | ? | 152029 |
| hardcap-only | SC36 | 2x4 | 144 | 36 | TIMEOUT | ? | NA | ? | 152029 |
| hardcap-only | SC36 | 4x4 | 8 | 8 | OK | 8 | NA | 5.3 | 12074 |
| hardcap-only | SC36 | 4x4 | 16 | 16 | OK | 16 | NA | 5.9 | 6354 |
| hardcap-only | SC36 | 4x4 | 32 | 32 | OK | 32 | NA | 9.5 | 7042 |
| hardcap-only | SC36 | 4x4 | 40 | 40 | OK | 40 | NA | 12.2 | 7178 |
| hardcap-only | SC36 | 4x4 | 48 | 48 | OK | 48 | NA | 12.8 | 6501 |
| hardcap-only | SC36 | 4x4 | 56 | 56 | OK | 56 | NA | 19.4 | 7613 |
| hardcap-only | SC36 | 4x4 | 64 | 64 | OK | 64 | NA | 21.3 | 2459 |
| hardcap-only | SC36 | 4x4 | 72 | 72 | OK | 72 | NA | 187.9 | 6057 |
| atmostk | SC16 | 2x4 | 16 | 4 | OK | 4 | NA | 3.2 | 6072 |
| atmostk | SC16 | 2x4 | 32 | 8 | ok(non-min) | 0 | NA | 597.3 | 1772 |
| atmostk | SC16 | 2x4 | 64 | 16 | OK | 16 | NA | 58.8 | 1560 |
| atmostk | SC16 | 4x4 | 8 | ? | TIMEOUT | ? | NA | 0.1 | 151022 |
| atmostk | SC16 | 4x4 | 16 | ? | TIMEOUT | ? | NA | 0.1 | 151025 |
| atmostk | SC16 | 4x4 | 32 | ? | TIMEOUT | ? | NA | 0.2 | 151019 |
| atmostk | SC20 | 2x4 | 16 | 4 | OK | 4 | NA | 4.4 | 12469 |
| atmostk | SC20 | 2x4 | 32 | 8 | OK | 8 | NA | 244.1 | 6276 |
| atmostk | SC20 | 2x4 | 64 | 16 | TIMEOUT | ? | NA | ? | 151025 |
| atmostk | SC20 | 2x4 | 80 | 20 | OK | 20 | NA | 19369.6 | 28146 |
| atmostk | SC20 | 4x4 | 8 | 8 | OK | 8 | NA | 1.9 | 6215 |
| atmostk | SC20 | 4x4 | 16 | 16 | OK | 16 | NA | 2.7 | 6181 |
| atmostk | SC20 | 4x4 | 32 | 32 | OK | 32 | NA | 4.6 | 6827 |
| atmostk | SC20 | 4x4 | 40 | 40 | OK | 40 | NA | 53.3 | 1513 |
| atmostk | SC36 | 2x4 | 16 | 4 | OK | 4 | NA | 43.4 | 7356 |
| atmostk | SC36 | 2x4 | 32 | 8 | OK | 8 | NA | 3002.6 | 11919 |
| atmostk | SC36 | 2x4 | 64 | 16 | TIMEOUT | ? | NA | ? | 152047 |
| atmostk | SC36 | 2x4 | 80 | 20 | TIMEOUT | ? | NA | ? | 152040 |
| atmostk | SC36 | 2x4 | 96 | 24 | TIMEOUT | ? | NA | ? | 152026 |
| atmostk | SC36 | 2x4 | 112 | 28 | TIMEOUT | ? | NA | ? | 152028 |
| atmostk | SC36 | 2x4 | 128 | 32 | TIMEOUT | ? | NA | ? | 152049 |
| atmostk | SC36 | 2x4 | 144 | 36 | TIMEOUT | ? | NA | ? | 152022 |
| atmostk | SC36 | 4x4 | 8 | 8 | OK | 8 | NA | 4.4 | 6177 |
| atmostk | SC36 | 4x4 | 16 | 16 | OK | 16 | NA | 6.0 | 6378 |
| atmostk | SC36 | 4x4 | 32 | 32 | OK | 32 | NA | 9.4 | 7015 |
| atmostk | SC36 | 4x4 | 40 | 40 | OK | 40 | NA | 12.2 | 7596 |
| atmostk | SC36 | 4x4 | 48 | 48 | OK | 48 | NA | 12.9 | 7091 |
| atmostk | SC36 | 4x4 | 56 | 56 | OK | 56 | NA | 29.1 | 9186 |
| atmostk | SC36 | 4x4 | 64 | 64 | OK | 64 | NA | 23.4 | 7530 |
| atmostk | SC36 | 4x4 | 72 | 72 | OK | 72 | NA | 226.9 | 8267 |

(total 150 runs)

## Experiment 3 — NO minimization baseline (plain embedding), 2x4, SC36
Isolates base-embedding difficulty from minimization. All 2x4 sizes are solvable given time:

| stage | k_min | hosts used (non-min) | solve time |
|---|---|---|---|
| 16 | 4 | 5 | 8.7 ms |
| 32 | 8 | 12 | 15.6 ms |
| 64 | 16 | 23 | 32.2 ms |
| 80 | 20 | 29 | 42.6 ms |
| 96 | 24 | 32 | 90.8 ms |
| 112 | 28 | 36 | 63.2 ms |
| 128 | 32 | 36 | 146 s (300s cap) |
| 144 | 36 | 36 | ~910 s / ~15 min (1800s cap) |
=> Embedding is ALWAYS possible; cost only spikes at near-/full-fill (128, 144). 150s "NO SOLUTION" earlier
   was purely a timeout.

## Experiment 4 — hard-cap-only, LONGER timeout (30 min), on the 150s-timeout SC36 cases
Tests whether the MINIMIZATION solves with more time. (Live — appended as each finishes.)

Progress log (hardcap_long/results.txt):
```
[17:13:17] waiting for no-minhost run to finish before starting (avoid contention)...
[17:29:47] starting hard-cap-only long runs
[17:29:47] START 2x4-80 (hardcap-only, kmin=20, timeout 1800s)
[17:36:29] 2x4-80 : SOLVED  wall=401s  kmin=20  | hardcap_only target<=20 : 385003.6 ms (status=10, occupied=20, ok=true | [intermesh-solve] attempt 1 : 385056.8 ms (success=true
[17:36:29] START 2x4-112 (hardcap-only, kmin=28, timeout 1800s)
```
