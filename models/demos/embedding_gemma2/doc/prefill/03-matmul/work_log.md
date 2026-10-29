# Prefill stage 3 (matmul, E9.3): work log

Run folder: `/local/ttuser/gtobar/tt-ops/runs/embeddinggemma-2/e9/03-matmul/` (below: `$U`): `scripts/`,
`logs/`, `search/` (one JSON line per in-model candidate), `tracy/{before,after}/`, `run_ab/` (harness
records), `dump/` (gate replay outputs), `analysis/`. Host sjc-snva-tp100, card 0 (P150a, 150 W default
TDP), grid 12 x 10 (`EMBEDDING_GEMMA2_GRID=12x10`; the device is opened on it and checked) unless a line
says 13 x 10. Precision `sdpa_nofp32acc` (unchanged). Every device command ran through `$U/scripts/run.sh`
(card-free check polling every 60 s, env dump, `timeout -s KILL`, `TT_METAL_OPERATION_TIMEOUT_SECONDS=60`,
120 for Tracy and sweeps, `__start` with commit and dirty files, `__end wall_s`, `__tt_exit`), one at a time;
long chains ran in tmux sessions `tt-bringup_embeddinggemma2_e903_<id>`. The card was never held by another
process (`$U/logs/card_wait.log` absent). Environment as in `../01-baseline/work_log.md`.

Scope change (person, relayed 2026-10-29 ~08:40): batch 1 only. Carry-over items b and c and the b8 x 512 /
b32 x 128 re-measurements were dropped; the closure gate suite is unchanged (153 tests, s32768, demo at both
grids). The tuned table therefore applies at batch 1 only (`lookup_spec(..., batch)`); batched calls keep
the stage-2 rule on every grid.

## Commits

| commit | content |
|---|---|
| `daa4784ae9` | commit 1 (carry-over): `02-topology` after profiles (`profiles_after/`), README op counts from profiles, work_log self-check and the batch-1 drop line |
| `50ad98eee4` | `tt/linear.py` (`MMSpec`, `TUNED_CONFIGS[(12, 10)]`, `MM_OVERRIDES`, `EG2_MM_TUNED`, 1D / chunked-2D / minimal-config paths, `role=`), role names at the call sites (`tt/attention.py`, `tt/mlp.py`, `tt/per_layer_input.py`, `tt/text_model.py`) |
| (this docs commit) | `doc/prefill/03-matmul/` |

## Method

- Search tool `$U/scripts/mm_sweep.py` (+ `cands.py` candidate sets): one device session per bucket group,
  the real model (`TextEncoder.from_pretrained`, session `max_seq_len` = max(bucket, 8192) as the harness),
  the harness input (3/4-full real input padded into the bucket). Per candidate: overrides put into
  `tt.linear.MM_OVERRIDES` (only the named (role, bucket)), eager forward (compile; an exception is recorded
  verbatim and the candidate skipped), trace capture, 3 warm replays, burst = 5 windows of n back-to-back
  replays + one sync after 2 s idle (n = 200 ms / forward, 3 ... 20), last hidden of the replay compared with
  the control's, trace released. A control (no override) runs first and last in every session; deltas are
  vs their mean, and their difference is the drift. The harness burst (3 s idle) and this burst agree:
  control 25.086 ms (sweep) vs 25.103 ms (stage-2 `sweep_new`) at 2048.
- Rounds: gelu placement at >= 8192 (`gelu_long`), N = 2048 roles at 2048 / 4096 (`n2048`), N = 512 roles
  (`narrow`), 128-1024 (`small`), in0_block_w 16 everywhere it applies (`n2048_r2`), smaller minimal blocks
  (`narrow_r2`), combinations at >= 8192 (`long_combo`); then `final`: every kept spec alone and all together
  per bucket, with `EG2_MM_TUNED=0` so the control is the stage-2 config.
- Keep rule: a spec is kept for a bucket when it lowers the full-forward burst median alone by more than the
  session drift, in its search round and again in `final`. Specs whose gain was inside the drift at 32768 /
  65536 (down, ple_layer) were removed before the closure runs.
- Host-only config proof `$U/scripts/config_dump.py`: records the calls `linear` / `gated_product` issue
  per role, shape and grid (ttnn ops replaced by recorders) for the current tree and for `a12d330bb3`.

## Commands (chronological)

| time | label (`$U/logs/<label>.log`) | command | result |
|---|---|---|---|
| 07:52-07:54 | `tracy_before_b1_s{128,2048,8192,65536}` | `$U/scripts/tracy.sh before 128 2048 8192 65536` (`perf.py::test_tracy_forward`, HEAD `a12d330bb3`, clean) | 4 passed; stage-2 after / stage-3 before profiles |
| 07:57 | `sweep_smoke_2048` | `sweep.sh smoke 2048` (tool check) | control 25.086 ms, gelu in the multiply -0.6 % |
| 07:59-08:04 | `sweep_gelu_long_{8192,16384}` | `sweep.sh gelu_long <B>` | search tables (gate, ple_gate) |
| 08:04-08:12 | `sweep_n2048_2048-4096` | `sweep.sh n2048 2048,4096` | in0_block_w 16 wins on up; L1 output probe |
| 08:12-08:26 | `sweep_narrow_2048-4096-8192` | `sweep.sh narrow 2048,4096,8192` | minimal (4,8,4), 2D for o_sliding / ple_proj at < 8192 |
| 08:26 | `c1b_*`, 08:29 `c1c_*` | batch > 1 carry-over checks (dropped afterwards by the scope change; informational) | passed / bitwise |
| 08:26-08:29 | `tracy_before_b1_s{256,512,1024,4096,16384,32768}` | `EG2_MM_TUNED=0 tracy.sh before ...` (table empty, same calls as `a12d330bb3`) | 6 passed |
| 08:29-08:44 | `sweep_n2048_r2_1024-2048-4096`, `sweep_narrow_r2_*` | round 2 | combos -1.2 ... -2.7 % |
| 08:44-09:08 | `sweep_small_128-256-512-1024` | `sweep.sh small 128,256,512,1024` | 1D wins at 128 only |
| 09:08-09:30 | `sweep_long_combo_{8192,16384}`, `sweep_gelu_long_{32768,65536}` | round 2 at >= 8192 | gate + ple_gate gelu in the multiply -4.8 / -4.5 % |
| 09:28-09:35 | `sweep_n2048_r2_128-256-512`, `sweep_narrow_1024` | `EG2_MM_TUNED=0` round 3 | 256 gains; 512 none |
| 09:35-09:55 | `sweep_final_*` | `$U/scripts/chain_final.sh` (`EG2_MM_TUNED=0`, set `final`) | every kept spec alone and all together, `final_sweep.md` |
| 09:55-09:58 | `soak_1x2048_12x10`, `soak_1x4096_12x10` | `$U/scripts/chain3.sh` (closure chain, HEAD `50ad98eee4`; its 2048 / 4096 entries equal the final ones): `02-topology/scripts/soak.py B 2000 100` | 2000 replays each, 0 mismatches |
| 09:57 | (commit) | removed down / ple_layer at 32768 / 65536 and gate at 65536 from the table (final-sweep gains inside the drift); amended to `5207e7e7b5` before the next device run | |
| 09:57 | `config_dump.log` | `config_dump.py --out $U/analysis/config_dump.json` at `5207e7e7b5` (CPU) | 13 x 10: 0 of 168 (role, shape) calls differ from `a12d330bb3`; 12 x 10: 39 differ (the 39 entries); `--tuned-off`: 0 differ at both grids |
| 09:58-10:02 | `gate_track` | `TT_METAL_TRACE_ALLOC_TRACKING=1 EMBEDDING_GEMMA2_PREFILL_SHAPES=1 EMBEDDING_GEMMA2_PREFILL_HF_CHECK=1 pytest prefill.py` | 20 passed; 10 / 10 gates |
| 10:02-10:23 | `sweep_new` | `EMBEDDING_GEMMA2_PREFILL_ALL=1 EMBEDDING_GEMMA2_PREFILL_DUMP=$U/dump/sweep_new pytest prefill.py` | 40 passed in 1258 s |
| 10:23-10:44 | `sweep_ctl` | same with `EG2_MM_TUNED=0` | 40 passed in 1280 s |
| 10:44-10:47 | `tracy_after_b1_s{128,2048,8192,65536,4096,16384}` | `$U/scripts/tracy.sh after ...` | 6 passed; `profiles_after/`, `roles_after.md` (`op_tables_after_e93.log`: CSV sha256) |
| 10:47-10:58 | `gates_e93_{12x10,13x10}_{text,s32768,demo}` | `$U/scripts/gates.sh <tag>` (same test list as stage 2) | 153 passed (1 deselected) / 1 / 2 at both grids |

The first `chain3.sh` attempt at the two gate lines failed to start (`gates.sh` not executable:
"Permission denied", nothing ran); the gates ran afterwards from tmux `..._e903_gates`.

Analysis (CPU): `roles.py <bucket>:<csv> ...` (`roles_before.md`, `roles_after.md`), `search_by_role.py`
(`search_tables.md`), `search_table.py search/final_*.jsonl --md final_sweep.md`, `config_table.py`
(`config_table.md` rows), `build_summary.py $U perf_summary.json 5207e7e7b5`.

## Decisions and findings

- 110 cores at 12 x 10: `mcast2d_config` takes per_core_N = ceil(64 / 12) = 6 for N = 2048, so 11 of 12
  columns hold output blocks. Every grid that uses more cores leaves the busiest core with at least 42
  tiles at 2048 tokens; all measured alternatives were slower (README). Not a config bug; kept the grid and
  raised in0_block_w to 16.
- gelu: the stage-2 rule fused gelu_tanh into `minimal_matmul` from 8192 tokens (CB budget rejects the 2D
  config at per_core_M >= 26). `minimal_matmul` + fused gelu costs about 2x the plain call (416 vs 188 us at
  8192); moving the gelu into the multiply's pre-activation is the largest single gain of the stage
  (-3.5 % at 8192). The multiply output, last hidden PCC vs HF and the gate values change in the 4th-5th
  digit (gate table in README); no dtype or fidelity changed.
- L1 output for up / gate at 2048 / 4096 (-3 to -5.6 %): recorded only. It changes activation placement,
  which stage 5 owns, and would need the K/V-norm soak protocol.
- Batch 1 only (scope change): `lookup_spec(..., batch)` returns no spec for batched calls, so b8 x 512,
  b32 x 128 and the batched PCC tests run the stage-2 configs (config_dump: every batched shape identical
  to `a12d330bb3` at 12 x 10 too).
- Per-bucket session key as the harness: buckets <= 8192 share one session (`max_seq_len` 8192); 16384,
  32768, 65536 one each.

## Anomaly ledger

| observation | evidence | control | resolution |
|---|---|---|---|
| Burst drift at 65536 is 0.2-0.7 % between the two controls of a session (0.01-0.03 % up to 16384) | `final_65536.jsonl` ctl_start 1565.405 / ctl_end 1576.497 ms; `gelu_long_65536` 1569.3 / 1572.3 | same code, same session | controlled: a 65536 spec is kept only when its gain exceeds the drift; gate, down, ple_layer at 65536 and down, ple_layer at 32768 were dropped for that reason (README) |
| gate at 65536: the profile predicts about 32 ms less device time (2691 -> about 1361 us per call x 24, before the multiply's added SFPU work) but no gate candidate moved the 65536 burst beyond the drift | `gelu_long_65536`: -0.17 % (gelu in the multiply) ... -0.60 % (2D chunked), drift 0.19 %; `final_65536`: -0.67 %, drift 0.71 % | same session controls | not kept; not attributed (no profile of the candidate at 65536, clock power-limited during the burst windows). Open item |
| Last hidden PCC vs the control 0.996-0.998 for gelu-placement and family changes (not bitwise) | `search/*.jsonl` `pcc_vs_ctl` | HF gate on the final table (README) | expected: a different accumulation / rounding order through 24 layers; the contract gate vs HF fp32 passes at every bucket with the kept table (`gate_track`, `sweep_new`) |
| Sustained AICLK at >= 8192 is 12-25 MHz lower after than before | `perf_summary.json` telemetry: 1318 / 1237 / 1156 / 1100 vs 1331 / 1256 / 1181 / 1112 MHz | same session, back to back | recorded as in stage 2: less light work between the heavy ops at the same power limit **[I]**; sustained still -1.9 to -4.8 % |
| 65536 after profile: kernel sum barely changes although ple_gate is 1.77 ms faster per layer | `roles_after.md` vs `roles_before.md` (other roles +2-3 % per call) | the stage-2 profile at the same bucket | **fixed (E9.4)**: the captures convert cycles at 1331.0 vs 1293.0 MHz; in cycles unchanged ops move -1.9 / +1.6 %, the kernel sum drops 55.2 Mcyc (-41.5 ms at 1331 MHz) = the ple_gate change (README "Open", `e9/04-attention/logs/c1_profile_clock_attr_65536.log`) |

## E9.4 commit 1 (stage-3 review remediation)

| time | run | command | result |
|---|---|---|---|
| 12:15-12:19 | `c1_ctl` | `e9/04-attention/scripts/c1_ab.sh ctl` (prefill.py, buckets 128,256,1024,4096,8192, HF_CHECK + SUSTAINED + TELEMETRY, 12 x 10, code `05a8aa15d7`) | 15 passed |
| 12:20-12:24 | `c1_new` | same, `tt/linear.py` without gate@256, ple_gate@128, qkv_sliding@1024, ple_layer@8192, ple_gate@4096 | 15 passed; `logs/c1_ab_table.md` (README "E9.4 amendments"): +0.05 ... +0.27 %, 256 PCC 0.99336 -> 0.99432 |
| - | (host) | `profile_clock_attr.py tracy/{before,after}/b1_s65536/ops_perf_results.csv` | 65536 anomaly attributed to the capture clock (above) |
| - | (host) | `pytest tests/perf/test_tuned_config_identity.py` with a planted `TUNED_CONFIGS[(13, 10)]` entry, then reverted | 1 failed (`down@b1_s4096` named), then 6 passed (`logs/c1_identity_{planted,clean}.log`) |

## Self-check against prefill-stage-review (not independent)

| check | status |
|---|---|
| before / after in the stage-1 regime, burst and sustained, same session, same commit, all 10 buckets | pass (`sweep_new` / `sweep_ctl`, `5207e7e7b5`) |
| every kept config improves full-forward burst wall time alone | pass (`final_sweep.md`; also in its search round); entries inside the drift removed |
| all kept configs confirmed together in sustained | pass (`sweep_new` vs `sweep_ctl`) |
| no shape regresses beyond noise | pass (512 +0.00 % with identical calls; all others faster) |
| configs gated on geometry; other geometry unchanged | pass (`TUNED_CONFIGS[(12, 10)]` only; `config_dump.py`: 13 x 10 identical; 13 x 10 gates pass) |
| precision held | pass (no dtype / fidelity / fp32-acc change; gelu placement only) |
| PCC gates 10 / 10, thresholds and tests unchanged | pass |
| trace allocation tracking on replay | pass (`gate_track`) |
| search tables with failed candidates and exact errors | pass (`search_tables.md`, 9 illegal) |
| every material role at its floor or with a blocker | pass with blockers (README "Floors and blockers") |
| K/V norm / L1 hazard soak | pass (2048 / 4096, 0 mismatches) |
| gaps | 65536 gate undecided (drift); after profiles at 6 of 10 buckets; independent review pending |
