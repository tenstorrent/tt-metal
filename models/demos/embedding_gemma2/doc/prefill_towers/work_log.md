# Prefill towers (E10.1): work log

Run folder `/local/ttuser/gtobar/tt-ops/runs/embeddinggemma-2/e10/` (below: `$U`): `scripts/`, `logs/`,
`run/` (bench records: `baseline.jsonl`, `final.jsonl`, `final2.jsonl`, telemetry CSVs), `search/` (one
JSON line per in-model candidate), `tracy/{before,after,after2}/`, `refs/` (cached HF fp32 tower outputs),
`analysis/`. Host sjc-snva-tp100, card 0 (P150a, 150 W TDP, firmware 19.8.0.0, PCIe x16), grid 12 x 10
(`EMBEDDING_GEMMA2_GRID`, default of the tower benches; the device reports (12, 10) and the bench checks
it) unless a line says 13 x 10. Precision file unchanged (`doc/datatype_sweep/selected_precision_config.json`).

Every device command ran through `$U/scripts/run.sh` (card-free check polling every 60 s, env from
`e1/env.sh` with `HF_HUB_OFFLINE=1` and the pinned snapshot, `TT_METAL_OPERATION_TIMEOUT_SECONDS=60`
(120 for sweeps and Tracy), `timeout -s KILL`, `__start` with commit and dirty files, `__end wall_s`,
`__tt_exit`), one at a time; chains ran in tmux sessions `tt-bringup_embeddinggemma2_e10_<id>`. The card
was never held by another process (`$U/logs/card_wait.log` absent).

## Commits

| commit | content |
|---|---|
| `36736a6ca8` | harness: `bench_vision.py` / `bench_audio.py` on `open_eg2_device` (12 x 10 default), lean default + `EMBEDDING_GEMMA2_BENCH_{CONTRACT,SUSTAINED,TELEMETRY,CHECK,ALL}`; `test_tracy_audio_forward[b1_s256]` |
| `36bed7c5b2` | topology rewrites (every grid) and the 12 x 10 tuned tables (vision, audio) |
| `a1501dbbb0` | vision patch projection output back in DRAM (see Findings) |
| (this docs commit) | `doc/prefill_towers/` |

## Harness options (bench_vision.py / bench_audio.py)

| option (env) | adds | measured cost (warm caches) |
|---|---|---|
| (none) | headline shape (vision 1 x 2560, audio 1 x 768), burst | 25 s vision, ~35 s audio |
| `EMBEDDING_GEMMA2_BENCH_CONTRACT=1` | every batch-1 contract shape (audio 1 x 256 + 1 x 768) | +~15 s per shape |
| `EMBEDDING_GEMMA2_BENCH_SUSTAINED=1` | >= 30 s and >= 300 replays per shape, ~500 ms windows, median of the last third | +~33 s per shape |
| `EMBEDDING_GEMMA2_BENCH_TELEMETRY=1` | sysfs AICLK / power / current / vcore / temperature every 100 ms during sustained | +0 |
| `EMBEDDING_GEMMA2_BENCH_CHECK=1` | soft-token PCC / embedding cosine of the traced replay vs HF fp32 (cached in `EMBEDDING_GEMMA2_BENCH_REFS`) | +2-4 s per shape the first time, then +0 |
| `EMBEDDING_GEMMA2_BENCH_ALL=1` | all of the above | 55 s vision, 105 s audio (`final_*_new` wall) |

## Commands (chronological; wall clock of the host)

| time | label (`$U/logs/<label>.log`) | command | result |
|---|---|---|---|
| 11:05 | `smoke_vision` | `EMBEDDING_GEMMA2_BENCH_CHECK=1 bench_vision.py --json run/smoke.jsonl` (relative path) | burst 16.272 ms, check pass; exit 1 at the JSON write (relative path under `run.sh`'s cwd); absolute paths afterwards |
| 11:05 | `smoke_audio` | `..._CHECK=1 ..._CONTRACT=1 bench_audio.py --json $U/run/smoke.jsonl` | 7.585 / 12.192 ms, check pass |
| 11:07-11:11 | `base_v12`, `base_a12`, `base_v13`, `base_a13` | `$U/scripts/baseline.sh` (HEAD `36736a6ca8`, clean): `EMBEDDING_GEMMA2_BENCH_ALL=1` per tower at 12 x 10, then 13 x 10 (vision, audio `--cases 1x768`) | `run/baseline.jsonl`; README "Baseline" |
| 11:11-11:12 | `tracy_before_audio_b1_s{256,768}` | `$U/scripts/tracy.sh before audio_b1_s256 audio_b1_s768` (`python -m tracy -r -p -v -m pytest test_audio_perf.py::test_tracy_audio_forward[...]`) | CSV sha256 in `$U/analysis/csv_sha256.txt`; audio files at `36736a6ca8` (the tree had uncommitted vision edits only) |
| 11:11 | `tracy_before_vision_b1_s2560` (first) | same for vision | failed: `NameError: vision_spec` (the pooler edit was half applied in the working tree); rerun at 11:51 with the control switches (log overwritten) |
| 11:12-11:28 | `sweep_vision_v{1,2,3,4}` | `$U/scripts/sweep.sh vision 1x2560 <set>` | `search/vision_v*.jsonl`, `search_tables.md` |
| 11:15-11:42 | `sweep_audio_a{0..5}` | `sweep.sh audio <shapes> <set>` | `search/audio_a*.jsonl` |
| 11:43 | (CPU) `config_dump.log` | `config_dump.py --out $U/analysis/config_dump.json` and `--tuned-off` at `36bed7c5b2` | 12x10: 6 of 89 differ; 13x10: 0; tuned off: 0 / 0 |
| 11:43-11:50 | `final_{v12,a12}_{new,ctl}`, `final_{v13,a13}_{new,ctl}` | `$U/scripts/closure.sh` (HEAD `36bed7c5b2`): after then control (`EG2_*` off), `_BENCH_ALL=1` at 12 x 10, `_BENCH_CHECK=1` at 13 x 10 | `run/final.jsonl`; audio rows of the README table. Vision at 13 x 10 not bit-identical: Findings |
| 11:50 | `track_v12`, `track_a12` | `TT_METAL_TRACE_ALLOC_TRACKING=1 EMBEDDING_GEMMA2_BENCH_CHECK=1 bench_* (contract)` | exit 0, checks pass |
| 11:51-11:52 | `tracy_before_vision_b1_s2560`, `tracy_after_{vision_b1_s2560,audio_b1_s256,audio_b1_s768}` | `env <control switches> tracy.sh before vision_b1_s2560`; `tracy.sh after ...` | profiles; the vision after profile showed the patch projection on 30 cores |
| 11:52 | `gates_12x10_vision` (first) | killed (tmux session and its pytest, own processes only) when the patch-embedder fix was made, so no gate ran on mixed code | - |
| 11:53-11:58 | `sweep_vision_v3off_g12`, `sweep_vision_v3off_g13`, `final2_v{12,13}_{new,ctl}`, `track2_v12`, `tracy_after2_vision_b1_s2560` | `$U/scripts/closure2.sh` (HEAD `a1501dbbb0`) | vision rows of the README table, bit-identity of t2 / t3 at both grids |
| 11:58-12:12 | `gates_{12x10,13x10}_{vision,audio,full,demo,text}` | same chain: `$U/scripts/gates.sh <grid>`: `pytest tests/pcc/vision`, `tests/pcc/audio`, `test_full_model.py test_full_model_trace.py`, `demo/demo.py`, `test_text_model.py::test_text_model[b1_s512]` | 91 / 117 / 41 / 2 / 1 passed at both grids (`$U/logs/gates_summary.log`) |

Analysis (CPU): `op_table.py <csv> <tower> <bucket>` (`prefill_op_table` with the tower signposts; 120
cores from the CSV), `roles.py` (`roles_before.md`, `roles_after.md`), `layer_ops.py` (`layer_ops.md`),
`search_tables.py` (`search_tables.md`), `build_summary.py` (`perf_summary.json`, README table).

## Method notes

- Search tool `$U/scripts/sweep.py` + `cands.py`: one device session, the real tower
  (`VisionEncoder` / `AudioEncoder.from_pretrained`), the bench inputs. Per candidate: in-process overrides
  (`VISION_MM_OVERRIDES`, `VISION_L1_WIDE_OVERRIDES`, `AUDIO_MM_OVERRIDES`, `REL_PM_OVERRIDES`, module
  switches), `capture_all` (eager compile pass, capture), 3 warm replays, 5 windows of 10 (vision) / 20
  (audio) replays after 2 s idle, replay output vs the control's and vs the cached HF fp32 reference,
  `release_traces`. A failing candidate records the exception and the session continues. Control first
  and last; drift 0.000-0.007 ms in every set.
- Keep rule: full-forward burst median below the control mean by more than the session drift, alone and
  in the final same-session A/B; accuracy gate on the replay vs HF fp32 (soft-token PCC and embedding
  cosine >= 0.99), and the tower / full-model / demo tests at both grids.
- Topology rewrites must be bit-identical (max abs 0 vs the switch off); matmul config changes may change
  the accumulation order and are judged on the HF gate.

## Decisions and findings

| observation | evidence | resolution |
|---|---|---|
| Vision is 6.3 % faster on 12 x 10 than on 13 x 10 (burst and sustained) | `base_v12` / `base_v13` | SDPA work units divide 120 cores evenly **[I]**; stage 4 owns the SDPA table |
| The patch-embedder L1 rewrite (first form) changed the patch projection's default program config: 30 cores, 97.7 vs 64.2 us at 12 x 10 (Tracy after), and the 13 x 10 output (soft-token PCC 0.99658 -> 0.99730, min token cosine 0.708 -> 0.924) | `tracy/after/vision_b1_s2560`, `final_v13_{new,ctl}` | projection output kept in DRAM (`a1501dbbb0`); the 13 x 10 output is then identical to the control (`final2_v13_*`, `v3off_g13` max abs 0) and 12 x 10 gains a further 0.03 ms (15.161 -> 15.133) |
| `h` in L1 alone gives -0.2 %; with the 2D down config -4.7 %; the 2D down with `h` in DRAM was 2x slower in E5b | `vision_v1` | kept together as one tuned entry |
| down in0_block_w 12 and 16 tie (15.259 / 15.257 ms, drift 0.007) | `vision_v4` | 12 kept: smaller circular buffers (L1 margin) and the higher HF PCC (0.99778 vs 0.99760); 24 fails with a CB / L1-buffer clash (exact error in `search_tables.md`) |
| Transposed 2D and 1D configs lose for every audio role (up to +17 % all together) | `audio_a1`, `audio_a2` | the E5c 2D rule stays for those roles |
| 768 up in0_block_w 4: -0.02 % with down bw16, inside 2x the drift | `audio_a4` | not kept |
| Vision sustained baseline differs between sessions (17.513 vs 17.807 / 17.720 ms; AICLK 1250 vs 1225 / 1231 MHz) | `base_v12`, `final_v12_ctl`, `final2_v12_ctl` | thermal history at the power limit; every before / after comparison is same session |
| The gelu(gate) * up multiply does not speed up with `h` in L1 (147 -> 144 us per layer) | Tracy before / after | NoC-bound on interleaved L1 inputs: stage 5 (sharded MLP chain) |

## Self-check against prefill-stage-review (not independent)

| check | status |
|---|---|
| headline from the same harness / regime / power limit / commit as its before | yes: same-session control on the final commit, both regimes |
| sustained measured for every kept change | yes (all kept changes together, telemetry) |
| kept changes justified by full-forward wall time | yes (in-model burst sweeps + final A/B) |
| every contract shape re-measured, none regressed | yes (3 / 3 at 12 x 10; 13 x 10 also faster) |
| config changes geometry-gated, other geometry untouched | yes (`config_dump`: 13 x 10 0 of 89) |
| PCC gates unchanged | yes (no threshold, test or reference edited) |
| trace changes: allocation tracking on replay | yes (`track2_v12`, `track_a12`) |
| rejections with exact errors | yes (`search_tables.md`) |
| independent review | pending (orchestrator) |
