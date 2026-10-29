# Prefill towers, stage 4 (E10.2): work log

Run folder `/local/ttuser/gtobar/tt-ops/runs/embeddinggemma-2/e10/02-attention/` (below: `$U`): `scripts/`,
`logs/`, `run/` (bench records `final.jsonl`, telemetry CSVs), `search/` (one JSON line per in-model
candidate), `tracy/{control,after}/`; HF fp32 references shared with E10.1 (`e10/refs/`). Same host and card
as E10.1 (P150a, 150 W TDP, firmware 19.8.0.0), grid 12 x 10 unless a line says 13 x 10. Precision file
unchanged.

Every device command ran through `$U/scripts/run.sh` (card-free check, env from `e1/env.sh`, `HF_HUB_OFFLINE=1`,
pinned snapshot, `TT_METAL_OPERATION_TIMEOUT_SECONDS` 60 / 120, `timeout -s KILL`, `__start` with commit and
dirty files, `__end wall_s`, `__tt_exit`), one at a time, in tmux sessions
`tt-bringup_embeddinggemma2_e1002_<id>`. The card was never held by another process (`$U/logs/card_wait.log`
absent). No hang, no reset.

## Commits

| commit | content |
|---|---|
| `c57d7ecab2` | E10.1 review minors (README M1 sustained clock effect, M3 bit-identity scope), doc only |
| `b7b727c1b7` | vision SDPA table (12 x 10, q256 k512), audio rel-shift as transpose / one-hot matmul (12 x 10, S 256 / 768), identity tests |
| `fb6cfaefd5` | vision 12 x 10 entry q256 k256 (q256 k512 failed the 12 x 10 full-model gate on an L1 clash) |
| (this docs commit) | `doc/prefill_towers/02-attention/` |

## Runs (in order)

| log | what | result |
|---|---|---|
| `sweep_vision_vs1` | vision SDPA chunk sizes, 12 x 10 grid, 16 candidates (code at `c57d7ecab2` + search hooks, uncommitted) | q256 k512 -3.03 %, max abs 0; 5 L1 errors (exact text in `search_tables.md`); drift 0.003 ms |
| `sweep_vision_vs2` | sub-grids, `max_cores_per_head_batch` 4 / 8 / 32, windowed (no dense mask) path | all sub-grids and windowed slower (+16 to +72 %); core limit no effect; drift 0.002 ms |
| `sweep_vision_vs3` | around q256 k512 (q 320, k 320, core limit, sub-grids, windowed q256 k512) | q256 k512 -2.98 % again; nothing better; drift 0.008 ms |
| `relshift_micro` | standalone: gather vs transpose / one-hot matmul / transpose (bf16 and bfp8 P, HiFi4 / LoFi, transpose vs permute), traced 50 calls | HiFi4 bit-exact vs torch.gather; 50.5 -> 18.1 us (256), 100.9 -> 21.9 us (768); LoFi max abs 0.125 |
| `sweep_audio_ar1` | in-model gather vs matmul path (code with the tables) | gather +2.65 % / +8.55 % vs matmul; max abs 0; drift 0.001 ms |
| `identity_test`, `identity_planted` | host-only identity test; planted 13 x 10 entries | 14 passed; with planted entries the 13 x 10 checks fail (2 failed) |
| `final_{v12,a12}_{new,ctl}` | 12 x 10 A/B, `EMBEDDING_GEMMA2_BENCH_ALL=1`, after then control | table in README |
| `final_{v13,a13}_{new,ctl}` | 13 x 10 A/B, burst + check | identical within 0.002 ms, same check digits |
| `sweep_audio_ar1_g13` | 13 x 10 gather vs matmul path (search only) | matmul +1.63 % (256), -8.32 % (768), max abs 0; not adopted |
| `track_v12`, `track_a12` | `TT_METAL_TRACE_ALLOC_TRACKING=1` replay + check, contract shapes | exit 0, checks pass |
| `tracy_{control,after}_*` | Tracy, one warmed eager forward per contract shape | `attention_ops.md` |
| `gates_{12x10,13x10}_*` | `scripts/gates.sh` on `b7b727c1b7` | all pass except 12 x 10 full: 3 failed (vision SDPA L1 clash, mixed image + audio items); 13 x 10 all pass |
| `gates_12x10_full_rerun` | the same full-model gate again | same 3 failures: reproducible |
| `fit12_{256,512 256,256 128,512 128,256}` | `scripts/test_full_sdpa_fit.py`: test_full_model.py with the 2560 entry forced, 12 x 10 | q256 k512 6 failed; the other three 30 passed. (A first round, `fit_*` / `fitall_*`, forgot `EMBEDDING_GEMMA2_GRID=12x10`, ran on 13 x 10 and was discarded; its logs were deleted.) |
| `final2_v12_{new,ctl}`, `final2_v13_{new,ctl}`, `track2_v12`, `tracy_after2_vision_b1_s2560` | closure 2 on `fb6cfaefd5` (`scripts/closure2.sh`) | README table; tracking passes; SDPA 297.9 us per layer |
| `gates_{12x10,13x10}_r2_*` | gates on `fb6cfaefd5` | all pass at both grids |

## Findings

- Vision SDPA at q 128 ran 240 units in two full waves; q 256 halves the waves and the K / V re-reads (each unit
  streams the whole K / V of its head) for the same per-core work. With k 512 the output is unchanged (max abs 0,
  -3.0 %), but its circular buffers do not fit next to the L1 buffers the full model holds after earlier items
  (12 x 10 full-model gate); k 256 fits and gives -1.7 % burst / -2.6 % sustained. Larger q chunks (320, 512) or
  k chunks (640+) do not fit even in the tower; the exact errors are in `search_tables.md`.
- Lesson: a tower bench alone does not prove L1 fit for the tower inside the full model; the full-model gate is
  the L1 check for a CB-heavier config.
- The `max_cores_per_head_batch` limit (default 16) never binds here: 12 heads on 120 cores is 10 cores per head.
- The windowed SDPA path computes the same output as the dense bfp4 mask at batch 1 (max abs 0) but is 24 %
  slower; the dense mask stays.
- The audio rel-shift gather cost ~49 / 99 us per layer although it moves 3 / 9 tiles per core; the stock
  transpose / matmul / transpose replacement costs 10.7 / 19.5 us. It is exact only at HiFi4 (and only for a bf16
  X); the code restricts it to bf16 X and uses the attention's HiFi4 compute config.
- At 13 x 10 the same rewrite helps 1 x 768 but hurts 1 x 256 (+1.6 %); the 13 x 10 path is unchanged by
  contract, recorded as a P150 candidate.
- Process note: four anchor replacements in `tt/audio/attention.py` (import, constructor argument, attribute,
  `from_state_dict` return) were applied with a Python string replace instead of the Edit tool, against the
  unit's editing rule; the diff is the same as the Edit calls would give (`git show b7b727c1b7`). Generated
  tables (`search_tables.md`, `attention_ops.md`, `perf_summary.json`) were copied from `$U`.

## Self-check (prefill-stage-review, owner mode, not independent)

- Headline and control from the same harness, session, commit and regime; sustained reported with telemetry.
- Kept changes justified by full-model wall time (burst and sustained), Tracy only for attribution.
- Every contract shape re-measured on the final code; none regressed; 13 x 10 unchanged (bench + host identity
  test). The vision change is not bit-identical (max abs 0.667); its HF gates went up (PCC 0.99787, cos 0.99978).
- Configs gated on grid and shape, batch 1; the 13 x 10 path has no entry; planted-entry negative controls.
- L1 rejections carry the exact error text. Mask paths: masked vs windowed equal (max abs 0).
- Trace allocation tracking on the replay of every contract shape.
- Not done: score-path sharding for audio was not swept (the block matmuls read L1-interleaved operands);
  stage 5 item.
