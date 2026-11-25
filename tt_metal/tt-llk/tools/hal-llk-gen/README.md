# hal-llk-gen: HAL LLK generation experiments

One run of `hal-llk-gen` is one experiment: a Claude Code agent writes the LLK functions used by an LLK test
from scratch on top of HAL, knowing HAL only from its documentation. The tool then checks the generated code
on the board, measures its cycles, counts its static code and prints a summary. All profiles use
`test_eltwise_binary -k add` (ELWADD) on Blackhole. Details: [REFERENCE.md](REFERENCE.md).

## What a run does

| Stage | What it does | Board | Time |
|---|---|---|---|
| `harness` | Builds a workspace from the profile's commits and runs the original test (must give the expected passed count). Starts the agent with the profile's prompt and limits. Then runs the HAL test after `tt-smi -r` and the original test right after it, and checks the generated code for raw instructions, forbidden-path access and HAL probing. | yes | agent 1.5-4 h, plus up to 30 min |
| `postcheck` | Runs the HAL and original tests again in 4 board states (with and without `tt-smi -r`). | yes | 15 min |
| `perf` | Runs the LLK perf test for the original and the generated LLKs, alternating, with asserts off. Gives cycles per tile and the generated/original ratio. | yes | 10 min |
| `static` | Compile-only builds of both versions, asserts on and off. Gives `.text` size and, with elfray, TTI/TT instruction counts. | no | 5 min |
| `dyn` (optional, off by default) | Single-steps the kernels through the tt-exalens GDB server and counts executed Tensix instructions. | yes | about 1 h per kernel and TRISC |

A test check passes when pytest exits 0 with 0 failed and the passed count equals the profile's
`EXPECT_PASSED` (1660 for the run8 profile). The stages after `harness` run only if the agent ended normally.

## How to run

You need a machine or container that sees exactly one Blackhole board, and Claude Code logged in
(`claude auth login`). No configuration file is needed.

```bash
cd tt_metal/tt-llk/tools/hal-llk-gen
./hal-llk-gen check      # checks only; the first time it also creates the venv $WORK/venv (a few minutes)
./hal-llk-gen run 9a     # one full run with the default profile (run8); summary at the end
```

| Command | Purpose |
|---|---|
| `check` | Checks the board, tools, commits and agent isolation. Starts nothing. |
| `dry-run [NAME]` | Setup and original test only, no agent (about 20 min). |
| `run NAME ... [--repeat N]` | Full runs, one after another. `--repeat 3` turns `9` into `9-1 9-2 9-3`. |
| `run NAME --profile P --stages "..."` | Another profile (`./hal-llk-gen profiles`) or other stages, e.g. add `dyn`. |
| `continue NAME --stages "perf static"` | More stages on an existing run. |
| `follow NAME` / `status [NAME]` / `stop NAME` | Follow, list or stop runs (run these where the run was started). |
| `summary NAME` / `compare NAME ... \| --all` | Print a run's summary, or a table over runs. |
| `selftest` | Offline tests of the tool (no board, no agent). |

A run works in the background. The terminal shows its progress and then the summary. Ctrl-C only
detaches, and `follow NAME` attaches again. Every command also works through make, for example
`make check`, `make run NAME=9a` or `make run NAME=9 REPEAT=3`.

## What it outputs and where

The terminal shows one line per event, then the run summary:

```
2026-11-23T19:28:40Z  run8-test1 postcheck done: postcheck: 4 of 4 pytest runs exit 0
2026-11-23T19:39:53Z  run8-test1 perf      done: L1->L1 per tile: original 32.80, generated 42.78, ratio 1.348 (1.268-1.478), faster in 0 of 16 variants
2026-11-23T19:43:34Z  run8-test1 run       end: PASS   (7 of 7 test checks ok)

==================== hal-llk-gen run 'run8-test1' ====================
Result:      PASS   (7 of 7 test checks ok)
Profile:     run8   base c0e38f1f  HAL b9e522d3  docs d8c6fa7c  prompt run8.md (3dbea124)
Agent:       claude-opus-5-5, Claude Code 2.1.287, tests: test_eltwise_binary:add
             exit 0, 94.1 min, $25.72, 247 turns, 58 pytest commands
...          test checks, code counts, isolation, static counts, cycles (see REFERENCE.md)
```

The summary has these parts:
- the result;
- the agent's time, cost and turns;
- each test check;
- code counts and isolation counts;
- `.text`/TTI/TT against the original LLKs and the run 8 median;
- the cycle ratio.

Everything goes to `WORK` (default `/proj_sw/user_dev/$USER/hal-llk-gen-work`, else `~/hal-llk-gen-work`):

```
$WORK/runs/<name>/        the run: summary.txt/.md/.json, progress.log, run.env (exact configuration),
                          harness.log and the postcheck/, perf/, static/ folders
$WORK/hal-runs/<run id>/  the agent's workspace (generated code in ws/tt_metal/tt-llk/tests/hal_llk/)
                          and logs/ (harness summary, metrics, code size, transcript)
$WORK/hal-results/        a .tar.gz archive per run (logs and generated code)
$WORK/batches/            the log of each `run` invocation, and a comparison when it had several runs
$WORK/dynamic-analysis/   dyn stage output only
$WORK/venv/               the default venv
```

## Settings (optional)

Defaults work without any file. To change one for a machine, copy `local.env.example` to `local.env` (not
tracked by git), or set the variable for one command, e.g. `WORK=/path ./hal-llk-gen run 9a`.

| Variable | Default |
|---|---|
| `WORK` | `/proj_sw/user_dev/$USER/hal-llk-gen-work`, else `~/hal-llk-gen-work` |
| `VENV` | the active venv, else `$WORK/venv` (made from `tt_metal/tt-llk/tests/requirements.txt`) |
| `CLAUDE_DIR` | `claude` from `PATH` or `~/.local/bin` (the profiles were made with 2.1.287; another version gives a warning) |
| `ELFRAY` | `elfray` from `PATH`; without it, no TTI/TT counts |

The experiment itself (commits, prompt, model, limits) is pinned in `profiles/<name>.env`.
[REFERENCE.md](REFERENCE.md) lists all settings and describes the profiles, the summary fields, agent
isolation, the files of this tool and its known limits.
