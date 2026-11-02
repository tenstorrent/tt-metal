# llk-tools MCP server

Gives AI assistants (Claude Code, Cursor, any MCP client) LLK tools that run on the
machine with the card. Use it to check a change for perf regressions locally, before
you push. Do not start the CI perf gate for this: it uses CI runner time and reports
later.

| tool | what it does |
|---|---|
| `list_perf_tests` | the perf test modules that `perf_compare` can measure on an arch |
| `perf_compare` | one perf test, two commits, on this card: verdict, report, log tail. Wraps `.claude/scripts/perf_compare_commits.sh` |
| `perf_placement_check` | `perf_compare`, then the same at 4 more code positions, as stage 2 of the gate does: each regression is `regression` or `placement` |

## Local first

1. Commit the change. Only committed code is measured.
2. `perf_compare` with `dry_run` to see the refs and what is cached.
3. `perf_compare` for each perf test that the change can touch.
4. If a point regresses, `perf_placement_check` on that test, or repeat with more
   `iterations`.
5. Add the `llk-perf-gate` PR label only when no card is attached, or as the final
   confirmation. The merge queue runs the perf gate on every LLK change anyway.

The same flow without MCP:

```bash
S=tt_metal/tt-llk/.claude/scripts/perf_compare_commits.sh
$S blackhole perf_math_matmul --dry-run
$S blackhole perf_math_matmul                    # HEAD vs the branch point
$S blackhole perf_math_matmul --code-shift 24    # one stage-2 position (needs #62981)
```

## What it compares

The defaults are the merge gate's (`.github/workflows/merge-gate.yaml`):

| | blackhole | wormhole |
|---|---|---|
| run types | `L1_TO_L1` | `L1_TO_L1` |
| threshold | 2 % | 13 %; `perf_matmul` and `perf_math_matmul` 25 % |
| min cycles | 30 | 30 |
| asserts | off (`TT_LLK_DISABLE_ASSERTS=1`) | off |

A point regresses when it is more than the threshold slower AND more than 30 cycles
slower, median against median over the iterations (default 3 per side). The compare
module is `perf/regression_compare.py`, the gates' module. The baseline is a commit
measured on this card, not the warehouse baseline that the gates use. Thus a local
verdict and a gate verdict can differ when the warehouse baseline is old.

`run_types="all"` compares every run type. `threshold`, `module_threshold` and
`min_cycles` replace the defaults.

## Wormhole noise

Wormhole results on main move from run to run with test order and worker
assignment. The fixes (#60892, #58068, #60894) are not merged. Before you call a
Wormhole point a regression:

- repeat with more `iterations` (cached runs are reused, so only the new runs cost);
- for one fixed order, set `CONSUMER_JOBS=1` (part of the cache key, so both sides
  are measured again);
- use `perf_placement_check`. A move at some code positions only is placement, not
  a regression.

A move that stays at every position and in every repeat is a regression.

## Placement check

`perf_placement_check` measures both commits with the TRISC code moved by 0, 4, 24,
44 and 60 bytes (`LLK_CODE_SHIFT`). The outcomes are those of stage 2 of the gate
(#62981, #63686): `regression` (slower at all 5 positions), `placement` (slower at
some positions only) and `not re-measured` (a position failed). Both commits must
have the `LLK_CODE_SHIFT` knob. Until #62981 is merged, the tool says that the knob
is missing and measures nothing.

## Set up

1. Install the SDK in the LLK test environment: `tests/.venv/bin/pip install mcp`
   (1.x or 2.x). The server uses `tests/.venv/bin/python` when it exists, and
   `python3` if not. The sweeps run with the same interpreter first on `PATH`.
2. Start the assistant from `tt_metal/tt-llk`. `.mcp.json` (Claude Code) and
   `.cursor/mcp.json` (Cursor) already list the server as `llk-tools`; approve it
   when asked.

A full comparison takes from minutes to about an hour. If your client stops waiting
first, raise its tool timeout (Claude Code: `MCP_TOOL_TIMEOUT`, in ms), or ask for a
`dry_run` first and run fewer `iterations`. When the client cancels a call, the server
stops the sweep. Runs are cached per commit, so a repeated or interrupted comparison
only measures what is missing. A sweep holds the card lock that `run_test.sh` uses
(`/tmp/tt-llk-test-<arch>.lock`), so two sweeps do not run on one card at the same time.

Ask in plain words, for example "check my branch for perf regressions in
perf_math_matmul on blackhole".
