# HAL LLK generation experiment: results package

Results of the HAL generation **run-8 series (8a–8e)**, plus the earlier runs **7a–7e** as an attachment. In each run a Claude Code agent writes the tt-llk LLK functions used by `test_eltwise_binary` (ELWADD, Blackhole p150b) on top of HAL, knowing HAL only from its documentation. This folder is committed on the branch `ctr-sbratic/hal-llk-generation-results` of tenstorrent/tt-metal (draft PR, not for merge). The branch is based on the HAL commit the runs used, so the PR diff contains only this folder.

## Start here
| File | What it is |
|---|---|
| `run_one.sh` | **One command that runs one run end to end** (agent + harness checks, postcheck, cycles, static counts). See "Reproduce one run" below. |
| `SUMMARY_RUN8.md` | **Key numbers of the run-8 series on one page:** tests passed, code size, TTI, TT, cycles, per run and as the median. |
| `slide/slide_draft.md` | One-slide draft for the run-8 series: experiment + results (code size, TTI, TT, speed). |
| `proposals/regression_rule_draft.md` | Draft of the prompt rule "if the code is worse than the original, try another way with HAL, then report". |
| `proposals/set_b_design_draft.md` | Draft design of set (b): spec + API descriptions, kernel deleted, HAL and RAW in parallel. |

`SUMMARY_RUN8.md` and the slide are regenerated with `python3 scripts/make_summary_run8.py`.

**Attachment: all runs, incl. 7a–7e.** `RESULTS.md` has the full tables for every run (asserts off and on, per-kernel cycles, agent cost), including runs 7a–7e. Those runs did the same task with an earlier prompt that has no section on the instruction cost of per-tile code. Regenerate it with `python3 scripts/make_results.py`.

## Reproduce one run (`run_one.sh`)
`run_one.sh` runs one run with the run-8 configuration end to end: the harness (agent + harness checks), the postcheck, the cycle measurement and the static counts with LLK asserts on and off. It needs a machine or container that sees exactly one Blackhole board, a venv with the tt-llk test requirements (pytest, tt-smi, pandas) and a logged-in Claude Code.

```bash
git clone --branch ctr-sbratic/hal-llk-generation-results https://github.com/tenstorrent/tt-metal.git /proj_sw/user_dev/$USER/hal-repro/tt-metal
cd /proj_sw/user_dev/$USER/hal-repro/tt-metal/hal_llk_generation_results
CHECK_ONLY=1 VENV=<venv> CLAUDE_DIR=<folder with claude> ELFRAY=<elfray binary> ./run_one.sh repro1   # checks only
VENV=<venv> CLAUDE_DIR=<folder with claude> ELFRAY=<elfray binary> ./run_one.sh repro1              # starts in the background
tail -f /proj_sw/user_dev/$USER/hal-repro/run_one/repro1/progress.txt
```

- **Checks before anything starts:** exactly one board visible (the harness resets the board with `tt-smi -r`), no other pytest / agent running, Claude Code logged in (version 2.1.287 as in the runs, otherwise a warning), the harness files equal to `scripts/harness/MD5SUMS.txt`, the three commits present in the clone (the script fetches the two branches named under "Configuration" if not), and this folder not readable by the agent: the harness denies the agent the repo, the results and the other folders next to the runs, so the clone must be under `/proj_sw/user_dev/$USER` or under `WORK`. A tt-metal checkout in `$HOME` gives a warning, because the agent is not denied `$HOME`.
- **Output:** harness run folder `<WORK>/hal-runs/<run id>/` with `logs/SUMMARY.txt`; postcheck in `<run folder>/postcheck_<name>/`; cycles in `<WORK>/perf/<name>/perf_summary.json`; static counts in `<WORK>/static_counts/<name>/`. Compare them with `SUMMARY_RUN8.md`.
- **Settings** are listed at the top of the script: `WORK` (default `/proj_sw/user_dev/$USER/hal-repro`), `REPO`, `PROMPT_SET=7` for the prompt of runs 7a–7e, `STEPS`, `RUN_DIR` (later steps on an existing run), `DRY_RUN=1` (no agent). `ELFRAY` is optional; elfray is not part of this repo, and without it the TT/TTI counts are skipped.
- **Duration:** the agent has the limits of the runs (4 h, 500 turns, $60); in runs 8a–8e it worked 84–223 min and cost $22.80–48.44 (`RESULTS.md` section 5). Harness setup and checks add about 30 min, the later steps about 40 min.
- **Same as in runs 8a–8e:** the harness command and parameters (`scripts/launch/driver_8x.sh` step 1) and the postcheck (`postcheck_7x.sh`). **Different:** cycles and static counts are measured in a hard-link copy of the run's own workspace; for the runs, a copy of run 8a's workspace with the run's generated files was used (`setup_ws.sh`, `build_counts.sh`), with the same base commit and test files. The dynamic analysis is not part of the script.
- **Tested** on bh-30 on 2026-11-23, without a new agent run. (1) Harness with `DRY_RUN=1`: setup and original test, 1660 passed as frozen; the prompt the agent would get is identical to runs 8a–8e (md5 3dbea124). (2) The later steps on a copy of run 8a (`RUN_DIR=`) reproduced run 8a's numbers: postcheck 4 of 4 with 1660 passed and the same TT/TTI counts; cycles 32.80 vs 41.98 per tile, ratio 1.324 (1.226–1.449); the same static counts with asserts on and off.

## Layout
```
runs/run<id>/
  harness_logs/     harness outputs: SUMMARY.txt, run_meta.txt (all parameters), prompt_used.md,
                    metrics (v2 rendering), codesize, codestats, elfray, pytest logs of the
                    baseline / HAL / dirty-state checks, audit files, the harness log
  postcheck/        independent check: reset->HAL, no reset->original, no reset->HAL, reset->HAL,
                    plus static TT/TTI counts of the harness build (asserts on)
  static_counts/    compile-only rebuild with LLK asserts on (control, must equal postcheck) and
                    off; SOURCE.txt = md5 checks that the rebuilt code is the run's code
  perf/             perf_data of 3 HAL perf runs + 3 original perf runs on the same board
  generated_code/   the agent's hal_llk/ headers and the HAL test files
  metrics_v2.json   process metrics recomputed with metrics.py v2
run_one.sh                   one command for one run (see "Reproduce one run")
perf/perf_summary_hal.json   cycle summary used by SUMMARY_RUN8.md and RESULTS.md (HAL runs only)
scripts/            harness/ (English versions of the harness files, VERSIONS.md, MD5SUMS.txt),
                    postcheck/, static_counts/, perf/, launch/, make_summary_run8.py, make_results.py
```

**Language.** Everything in this folder is English.
- Runs 8b–8e used the English harness. Their harness files are copied as they are. `harness_logs/translation_check.txt`, when present, is the check that no Serbian words are in them.
- The harness of runs 7a–8a wrote its outputs in Serbian (team convention at the time). Their text files in `runs/run*/harness_logs/` are English copies, made with a phrase-table translator. `harness_logs/translation_check.txt` records, for every file, that its numbers are identical to the Serbian original and that no Serbian words are left. The translator itself is not included because it contains the Serbian phrases; the Serbian originals stay in the run folders on bh-30.
- The harness's v1 `metrics.json` / `metrics.md` (7a–7e) are replaced by v2 (`metrics_v2.json`; `harness_logs/metrics.md` is its rendering).
- The harness `codesize_*.json` is left out (for 7a–8a one of its values was Serbian). `static_counts/assert_on/codesize.json` is the same measurement of the same build.
- `scripts/harness/` holds the English versions of the harness scripts; `VERSIONS.md` lists them next to the versions the runs used.

## Configuration
From `runs/run*/harness_logs/run_meta.txt`. `make_summary_run8.py` and `RESULTS.md` check that these are identical in all runs.
- tt-metal commits: BASE `c0e38f1ff70b7ee70c526fb44cbbf16736a341ed` (frozen test set: `test_eltwise_binary -k add` = 1660 passed / 20 skipped), HAL `b9e522d39ae534f0281499dae11b425574d89d0d`, HAL docs `d8c6fa7c22fb89e109178778a537e5860fae0ac6`.
- Doc pages marked as not valid for the agent: address_counters, replay, gpr_ops, atomic.
- Model `claude-opus-5-5`, Claude Code 2.1.287, limits 4 h / 500 turns / $60.
- One Blackhole p150b board per run on bh-30, SFPI 7.84.0. 8c, 8d and 8e ran at the same time on three different boards.
- Prompt: 8a–8e `prompt-run6b-cfgcost.md`, which adds a section on the instruction cost of per-tile code; 7a–7e `prompt-run6b.md`.

**Where the commits are.** BASE and HAL: branch `ctr-sbratic/hal-c0e38f1f-movd2a` (the HAL commit; its parent is BASE). It is the base branch of this PR. HAL docs: branch `ctr-sbratic/hal-docs-pr59009`.

## How each number was produced
The scripts are in `scripts/`. Paths are bh-30 paths (`$B` = the user folder on bh-30). For a new run, `run_one.sh` does steps 1–4 with the same parameters.

1. **Generation run (harness),** inside the board container. Run-8 series: `scripts/launch/driver_8x.sh <name>` step 1 (8c–8e were started by `launch_8x.sh` from `orchestrate_8cde.sh`; 8b was started by hand with the same harness command; 8a by `driver_8a.sh`):
   ```bash
   cd $B/harness/hal-harness-run5rules
   MODE=B TESTS="test_eltwise_binary:add" HAL_FILTER=0 PROMPT_FILE=$PWD/prompt-run6b-cfgcost.md \
   HAL_COMMIT=b9e522d39ae534f0281499dae11b425574d89d0d DOCS_COMMIT=d8c6fa7c22fb89e109178778a537e5860fae0ac6 \
   DOCS_INVALID="address_counters_hal_interface.html replay_hal_interface.html gpr_ops_hal_interface.html atomic_hal_interface.html" \
   MODEL=claude-opus-5-5 TIMEOUT=4h MAX_TURNS=500 MAX_BUDGET=60 ELFRAY=$B/tools/elfray \
   REPO=$B/repos/tt-metal RUNS_ROOT=$B/results/hal-runs RESULTS_ROOT=$B/results/hal-results ./run_hal_gen.sh
   ```
   The harness runs the original test (must give 1660 passed), starts the agent, then runs the HAL test after `tt-smi -r`, the dirty-state check, code size and metrics, and writes `SUMMARY.txt`. Runs 7a–7e: same command with `PROMPT_FILE=prompt-run6b.md` (`scripts/launch/launch_7a.sh`, `driver_7bcde.sh`).
2. **Postcheck:** `scripts/postcheck/postcheck_7x.sh <run dir> <name>` (driver_8x step 2).
3. **Cycles:** `scripts/perf/setup_ws.sh <name> <run dir>` builds a workspace with the HAL perf kernel; `driver_perf.sh` alternates original and HAL perf runs on the run's board after `tt-smi -r`, with `TT_LLK_DISABLE_ASSERTS=1` (driver_8x step 3). `perf_summary_8x.py` (8b–8e) and `retro_summary.py` (7a–8a) make `perf/perf_summary_hal.json` with the same computation.
4. **Static counts, asserts on/off:** `scripts/static_counts/build_counts.sh <name> <run dir>` for one run; `driver_asserts_off_8x.sh` (8b–8e) and `driver_asserts_off.sh` (7a–8a) for several. Each runs two `--compile-producer` builds of the original and the HAL test (`-k add`), without and with `TT_LLK_DISABLE_ASSERTS=1`, and then `tt_tti_counts.py`. `compare.py` checks that the asserts-on build equals the postcheck.
5. **SUMMARY_RUN8.md, slide, RESULTS.md:** `python3 scripts/make_summary_run8.py` and `python3 scripts/make_results.py`. They read only the files in this folder.

## Known open points
- **Run 8b is much slower than the other four.** HAL ÷ original cycles per tile: 8b 3.49×, 8a/8c/8d/8e 1.32–1.45×. In 8b most of the difference is in the pack thread (pack measured alone: 4.37×, the others 1.46–1.50×); unpack and math are like the other runs. The measurement is stable (largest spread over the 3 repeats 0.14%), and the original gives the same 32.80 cycles per tile on board 2 as on the other boards. 8b's postcheck (2026-11-23) passed 4 of 4, and its static-count control matched. 8b's agent also took the longest (223 min, $48.44; `RESULTS.md` section 5).
- **Run 8b, report and dynamic analysis.** 8b's English harness report (`report_en/perf_cycles.*`, copied as `perf/compare_8b.*` for the other runs) is made after its dynamic analysis, which was still running when this package was assembled.
- **Run 7b, first full pass.** `metrics_v2.json` gives iteration 23 at 52.5 min. An earlier manual check of the 7b pytest commands (2026-11-06) put the first full `-k add` pass at iteration 24, 62.4 min. These have not been reconciled.
- **Asserts.** Earlier run reports compared the original WITH LLK asserts against generated code. All numbers in `SUMMARY_RUN8.md` and the slide are asserts off on both sides (`RESULTS.md` section 2); section 3 has the asserts-on numbers.

## Not included on purpose
- Agent transcripts and the agent's own final reports. They are in the local run folders and can be added if wanted.
- The dynamic (executed-instruction) analysis, which is kept separately.
- RAW-arm runs. They were deleted on 2026-11-21 because they were generated with the wrong protocol; RAW will be redone as set (b).
- Compile logs, ELF files and harness archives.

## Meeting action items: status
| Item | Status |
|---|---|
| Draft PR on GitHub with all results | this branch (draft PR); the three commits are on the branches named under "Configuration" |
| Structure all results; every number regenerable or committed | this package (`SUMMARY_RUN8.md`, `RESULTS.md` + the two generators in `scripts/`) |
| One command for all tests / pre-computed results | `run_one.sh <name>` runs one run end to end; `python3 scripts/make_summary_run8.py` / `make_results.py` regenerate the results from the committed data |
| One slide (experiments + code size, TT, TTI) | `slide/slide_draft.md`, run-8 series only (the PR link still has to be filled in) |
| Numbers for other teams ("how simple a HAL call is") | not done: the metric still has to be agreed |
| Prompt rule for regressions | `proposals/regression_rule_draft.md` (draft, not in any prompt yet) |
| Two test sets: (a) conversion to HAL, (b) spec-only HAL + RAW in parallel | (b): `proposals/set_b_design_draft.md`; (a): not started |
