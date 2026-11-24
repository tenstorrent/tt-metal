# hal-llk-gen reference

The short overview is in [README.md](README.md). This file has the details.

## Stages

| Stage | What happens |
|---|---|
| `harness` | Builds a workspace from the profile's commits and runs the original test; the passed count must equal `EXPECT_PASSED`. Starts the agent (`claude -p`) with the profile's prompt and limits. Afterwards it runs the HAL test after `tt-smi -r`, then the original test right after it without a reset (dirty state). It checks for raw instructions, forbidden-path access, MCP calls and HAL probing, and writes code statistics, code size and process metrics. |
| `postcheck` | The 4-step state sequence on the run's workspace: `tt-smi -r` then HAL test; no reset, original test; no reset, HAL test; `tt-smi -r`, HAL test. With elfray, also TTI/TT counts of the harness build. |
| `perf` | The LLK perf test (`PERF_TEST`, `-k PERF_K`) for the original and the generated LLKs. The two alternate `PERF_REPS` times, each after `tt-smi -r`, with LLK asserts off. |
| `static` | Compile-only builds of the original and HAL tests, once with LLK asserts on and once off. Gives code size per build and, with elfray, TTI and TT counts. |
| `dyn` | Every variant runs with one TRISC single-stepped through the tt-exalens GDB server, which counts the executed Tensix instructions. The output goes to a separate folder (see below). |

Default stages: `harness postcheck perf static`. The later stages run only if the agent ended normally
(exit 0). `continue NAME --stages "..."` runs stages on an existing run later, for example after a perf
failure.

## Commands

| Command | Purpose |
|---|---|
| `check` | Checks the machine, tools, commits and agent isolation. Starts nothing, but creates the default venv when it is missing. |
| `dry-run [NAME]` | Harness setup and original test only, no agent. |
| `run NAME [NAME ...]` | Full runs, one after another, on this board. `--repeat N` turns one NAME into `NAME-1 ... NAME-N`. |
| `continue NAME --stages "S ..."` | More stages on an existing run (not `harness`). |
| `trivial [NAME]` | Test validity check, no agent. With empty LLK function bodies, every selected variant must fail. |
| `follow NAME` | Attaches to a running batch: shows progress until it ends, then the summaries. |
| `status [NAME]` | Lists all runs with their state and result, or shows one run's progress. |
| `stop NAME [--now]` | Starts no further stage. `--now` also terminates the running stage, including the agent. |
| `summary NAME` | Prints a run's summary and rewrites `summary.{txt,md,json}`. |
| `compare NAME ... \| --all` | Prints a table over runs with medians, the original LLK and the reference series. |
| `profiles` | Lists the profiles. |
| `selftest` | Offline tests of this tool: no card, no agent, no network. |

Options: `--profile P`, `--stages "S ..."`, `--repeat N`, `--detach`.

Run `follow`, `status` and `stop --now` on the machine or in the container that started the run. Elsewhere the
process id means nothing, so `status` shows `running? (other host)` until the run's last line is written,
and `stop --now` refuses (a plain `stop` still works through the shared STOP file).

## Profiles

A profile (`profiles/<name>.env`) pins everything that defines the experiment: commits, prompt (with its md5),
test selection, expected passed count, model, Claude Code version and agent limits.

| Profile | Tests (BASE_COMMIT) | HAL (HAL_COMMIT) | HAL docs (DOCS_COMMIT) | Prompt |
|---|---|---|---|---|
| `run8` (default) | `c0e38f1ff70b`, 1660 passed | `b9e522d39ae5` | `d8c6fa7c22fb`, 4 pages marked not valid | `run8.md` (md5 `3dbea124`) |
| `run7` | same | same | same | `run7.md` (md5 `411819d2`) |
| `hal-test-gen` | same | `485e3823fc18` (branch `sbratic/HAL-test-gen`) | same as run8 | `run8.md` |

The prompts are byte-identical to the prompts of runs 7a-7e (`run7.md`) and 8a-8e (`run8.md`). `run8.md`
adds a section on the instruction cost of per-tile code. If a commit is missing in the checkout, `check`
fetches the branches in the profile's `FETCH_REFS` from `FETCH_REMOTE` (default `origin`).

Any profile value can be overridden from the environment for one invocation, for example
`HAL_COMMIT=<sha> ./hal-llk-gen run 9c --profile run8`. The value is recorded in the run's `run.env`.

## Settings

Settings are read in this order: the profile, then `local.env` next to the script (optional, not tracked
by git; template `local.env.example`, other path with `HAL_LLK_GEN_ENV`), then the environment, then the
command-line options. `check` prints where the settings and the venv came from.

| Variable | Meaning | Default |
|---|---|---|
| `WORK` | Output root. Must not be inside the tt-metal checkout. | `/proj_sw/user_dev/$USER/hal-llk-gen-work`, or `$HOME/hal-llk-gen-work` |
| `VENV` | Python venv with pytest; pandas for `perf`, tt-exalens for `dyn` | The active venv. Otherwise `$WORK/venv`, created once by `check`/`run` from `tt_metal/tt-llk/tests/requirements.txt` (log `$WORK/venv-setup.log`) |
| `CLAUDE_DIR` | Folder with the `claude` binary, put first in `PATH` | `claude` from `PATH` or `~/.local/bin` |
| `ELFRAY` | elfray binary for TTI/TT counts | `elfray` from `PATH`; if none, code size only |
| `REPO` | tt-metal checkout that holds the commits | the checkout this tool is in |
| `PERF_REPS` | Repetitions per side in `perf` | 3 |
| `DYN_KERNELS` | Kernels for `dyn` | `original hal` |
| `DYN_REFERENCE` | Reference stepping of the original kernel for `dyn` | - |
| `EXTRA_DENY` | More paths the agent must not read | - |
| `SIBLING_ROOTS` | Folders whose other subfolders the agent must not read | `/proj_sw/user_dev/$USER /localdev/$USER` |
| `STRICT_VERSIONS` | 1 = stop if the Claude Code version differs from the profile | 0 (warning) |
| `KEEP_WS_COPY` | 1 = keep the hard-link workspace copy used by perf/static/dyn | 0 |

## Requirements

- One Blackhole board visible to the machine or container. The harness and the checks run `tt-smi -r`,
  which resets every board the machine sees, and nothing else may use the board meanwhile.
- `tt-smi` in `PATH` (system-wide in IRD containers) and a Python venv with `pytest`. The harness installs
  the test requirements of the frozen test set into the venv (`pip install -r tests/requirements.txt`). The
  `perf` stage also needs `pandas`, and `dyn` needs `tt-exalens`. Both are in the tt-llk test requirements.
- Claude Code, logged in (`claude auth login`). The profiles were validated with version 2.1.287. Auto-update
  is disabled during runs.
- The tt-metal checkout must contain the profile's commits (`check` fetches them if `FETCH_REFS` holds them).
- Optional: elfray for the TTI/TT counts.
- About 2 GB free in `WORK` per run (the workspace is about 1.2 GB), plus about the size of the venv once.

## Output

```
$WORK/
  runs/<name>/              one folder per run
    run.env                 resolved configuration (profile, overrides, tool commit and checksum)
    progress.log            one line per event (what the terminal shows)
    stages.tsv              stage, status, exit code, start, end
    harness.log             full harness output; <stage>.log for the other stages
    harness_run -> ../../hal-runs/<run id>   (path also in run_dir.txt)
    postcheck/              checks.txt + the pytest logs of the 4 steps, tt_tti_counts.{md,json}
    perf/                   perf_summary.{md,json}, orig_r<i>/ and hal_r<i>/ (pytest output, perf CSV)
    static/                 assert_on/ and assert_off/: codesize.{md,json}, tt_tti_counts.{md,json}, compile log
    dyn_status.txt          dynamic stage: completion and error counts only
    summary.txt / .md / .json
  hal-runs/<run id>/        harness run: ws/ (workspace, generated code in ws/tt_metal/tt-llk/tests/hal_llk/)
                            and logs/ (SUMMARY.txt, run_meta.txt, metrics.{md,json}, codesize, codestats, checks,
                            transcript.jsonl, final_report.md)
  hal-results/<run id>.tar.gz   archive of logs/ and the generated code
  batches/<time>_<name>/    progress.log, console.log and compare.{txt,md} of one invocation
  dynamic-analysis/<name>/  dynamic counts (only with the dyn stage)
  venv/                     default venv (only when no venv is set or active)
```

## The summary

The summary is built only from files the stages write. It never reads the agent's transcript content or
its final report. The summary of run `run8-test1` (profile run8, made with this tool):

```
==================== hal-llk-gen run 'run8-test1' ====================
Result:      PASS   (7 of 7 test checks ok)
Profile:     run8   base c0e38f1f  HAL b9e522d3  docs d8c6fa7c  prompt run8.md (3dbea124)
Agent:       claude-opus-5-5, Claude Code 2.1.287, tests: test_eltwise_binary:add
             exit 0, 94.1 min, $25.72, 247 turns, 58 pytest commands
             first full pass of the test selection: iteration 36 of 40 (73.0 min)
             pytest commands by outcome: denied 5, collect 1, compile_only 10, compile_error 3, hang 7, pass 20, fail 12
Test checks (pytest result lines):
  [ok  ] original (baseline, after tt-smi -r)   test_eltwise_binary      1,660 passed, 0 failed, 20 skipped
  [ok  ] HAL after tt-smi -r                    test_eltwise_binary      1,660 passed, 0 failed, 20 skipped
  [ok  ] original right after HAL, no reset     test_eltwise_binary      1,660 passed, 0 failed, 20 skipped
  [ok  ] postcheck 1: tt-smi -r, HAL            test_eltwise_binary      1,660 passed, 0 failed, 20 skipped
  [ok  ] postcheck 2: no reset, original        test_eltwise_binary      1,660 passed, 0 failed, 20 skipped
  [ok  ] postcheck 3: no reset, HAL             test_eltwise_binary      1,660 passed, 0 failed, 20 skipped
  [ok  ] postcheck 4: tt-smi -r, HAL            test_eltwise_binary      1,660 passed, 0 failed, 20 skipped
Code:        1,193 LOC, 10 hex literals, 0 raw instruction lines, 0 NOT_IN_HAL marks, 217 hal:: uses
Isolation:   1 forbidden-path tool calls, 0 HAL probing traces, 0 MCP calls, 28 denied tool calls
Static, LLK asserts off, average per build over 10 builds, unpack+math+pack [tt_tti_counts]:
                                  original   generated    change      run 8 median
  .text bytes                      5,800.8     4,032.8    -30.5%  4,070.8 (-29.8%)
  TTI                                 78.0       137.0    +75.6%    153.6 (+96.9%)
  TT (all paths)                      41.0        54.6    +33.2%     58.1 (+41.7%)
  (TT: 6.0 stores per build not resolved statically, see static/assert_off/tt_tti_counts.md)
Static, LLK asserts on (framework default), ...
Cycles:      L1->L1 per tile, asserts off, 3 reps: original 32.80, generated 42.78, ratio 1.348 (1.268-1.478), faster in 0 of 16 variants
             one thread alone (ratio): unpack 1.00, math 0.99, pack 1.50; repetition spread max 0.33%
             run 8 series median ratio: 1.418
Stages:      harness ok, postcheck ok, perf ok, static ok
```

- **Result** is `PASS` when every test check passed. A check passes when pytest exited 0, reported 0 failed
  and 0 errors, and the passed count equals `EXPECT_PASSED`. With the 4 postcheck steps there are 7 checks
  for one test. `PASS (harness checks only)` means postcheck did not run. `FAIL` means at least one check
  failed. `AGENT DID NOT FINISH` means the agent's exit code was not 0 (124 = timeout). `INCOMPLETE` means
  the harness did not finish. `DRY RUN` means no agent was started.
- **First full pass** is the agent's first pytest command that ran the whole selection and passed it
  (`harness/metrics.py`).
- **Code** gives counts in the generated code: lines of code, hex literals, lines with raw `TTI_`/`TT_OP_`
  instructions, `NOT_IN_HAL` marks (code the agent wrote outside HAL, with a reason) and `hal::` uses.
- **Isolation** counts four things:
  - the agent's tool calls whose text names a forbidden path, denied or not
    (`logs/forbidden_access.txt` lists them);
  - traces of probing the HAL implementation through the compiler;
  - MCP calls;
  - denied tool calls.
- **Static** values are averages per build over the builds of the test selection, summed over the unpack,
  math and pack kernels.
  - `.text` is the size of the `.text` section.
  - TTI is the number of Tensix instructions embedded in the code.
  - TT is the number of Tensix instructions built at run time and written to the instruction buffer. It
    comes from a static analysis over all paths. Stores the analysis cannot resolve are reported, not
    dropped.

  These are counts in the compiled code, not executed instructions. The `run 8 median` column comes from
  `reference/run_series.json`, which holds the static counts and cycle ratios of runs 7a-7e and 8a-8e.
- **Cycles** compares the perf test's L1-to-L1 cycles per tile, generated against original. The value per
  variant is the median over the repetitions. The ratio shown is the median over variants, so a ratio
  above 1 means slower. The perf test does not check results.

`compare` prints one row per run (result, agent minutes, cost, first full pass, `.text`, TTI, TT, cycle ratio)
plus the median, the original LLK and the reference series median.

## Agent isolation

The agent must know HAL only from its documentation and must not see the original LLKs. The harness enforces
this with Claude Code permission rules in `<run>/ws/hal-settings.json`. The base rules are in
`harness/settings/hal-settings-B.json`. For every run the harness adds:

- read-deny for the HAL source and the original LLKs in the workspace (base rules);
- read-deny for the tt-metal checkout (`REPO`), every other harness run, `hal-results`, `$WORK/runs`,
  `$WORK/batches`, `$WORK/dynamic-analysis`, this tool and `EXTRA_DENY`;
- read-deny for every other folder in `SIBLING_ROOTS`, except the folder that holds the venv and the path to
  the run itself. Each path is written as given, as its real path and under its symlink names;
- deny rules (base rules) for:
  - `git`, search, hex-dump and decompression tools, Python and the compiler binaries in `Bash`;
  - editing the tests and the docs;
  - `WebFetch` and `WebSearch`.

  claude.ai connectors are disabled.

`check` refuses a `WORK` inside the checkout or inside this tool. It also warns about tt-metal checkouts in
`$HOME`, because the rules do not deny `$HOME` (only `~/.claude`). After a run, `logs/forbidden_access.txt`,
`logs/probe_check.txt` and `logs/mcp_calls.txt` show any attempts. They are summarized under Isolation.

Several runs on several boards (one container per board) can share one `WORK`. Start them at least 20 s
apart: a run denies only the run folders that already exist when it starts. Runs started with one
`run A B C` command run one after another, so they cannot see each other while working.

## Dynamic analysis output

The `dyn` stage writes the counts, pytest logs and per-variant status records to `$WORK/dynamic-analysis/<name>/`.
The run folder gets only `dyn_status.txt`: completion, error counts, wall time and pytest result lines.
This keeps the counts in one place that can be withheld from people or agents who must not see them, and it
is denied to every later agent run.

## Tests

`./hal-llk-gen selftest` (or `make selftest`) does the following:

- runs `bash -n` on every script and parses every Python file;
- checks the profiles and the pinned prompt checksums;
- runs the unit tests in `tests/`, which cover the summary on synthetic runs, the perf ratios on synthetic
  CSV files and the agent metrics on a synthetic transcript.

It needs no card, agent or network.

## Files

| Path | Content |
|---|---|
| `hal-llk-gen`, `Makefile` | Command-line tool and make wrapper |
| `local.env.example` | Template for the optional machine settings |
| `profiles/` | Experiment configurations |
| `harness/run_hal_gen.sh` | The harness: workspace, spec, permissions, baseline, agent, independent checks |
| `harness/make_spec.py` | Signature-only spec of `llk_lib/` and `common/inc/` (bodies and raw encodings removed); with `--trivial`, empty bodies |
| `harness/metrics.py` | Process metrics from the agent's stream-json transcript (cost, tokens, iterations, first full pass) |
| `harness/codesize.py`, `codestats.py`, `elfray_tt_tti.py`, `probe_check.py` | Code size, code statistics, elfray TT/TTI estimate of the harness build, HAL probing traces |
| `harness/prompts/`, `harness/settings/` | Agent prompts and base permission rules |
| `stages/` | postcheck, perf, static and dyn stages (each usable on its own, see the header of each script) |
| `python/run_summary.py` | Run summary and comparison table |
| `python/perf_summary.py`, `perf_compare.py` | Cycle ratios from the perf CSV files |
| `python/tt_tti_counts.py` | `.text`, TTI and TT counts per build (TT by dataflow analysis over all paths) |
| `python/dyn/` | Dynamic analysis pytest plugin and its number-free status summary |
| `reference/run_series.json` | Static counts and cycle ratios of runs 7a-7e and 8a-8e |
| `tests/` | Offline tests |

## Known limits

- The prompts, `TESTS` and the perf stage are written for `test_eltwise_binary -k add`. The harness supports
  other tests and several tests per run, but a new test needs its own prompt and perf test settings.
- The perf test checks no results. Only 4 of its 16 Elwadd variants use formats that the functional test checks.
- TT is a static estimate (see `tt_tti_counts.md`, column `unresolved`).
- The `hal-test-gen` profile gives the agent the HAL docs of run 8, because that branch has no `docs/hal/`.
