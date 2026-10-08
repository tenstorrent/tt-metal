# LLK SFPU report

Before/after perf and accuracy of the SFPU ops a change touches, on silicon, as one
PR comment. Built for reviewing SFPU PRs -- most of them bounty PRs from forks --
where the question is always the same: *how much faster or slower is it, on which
architecture, and what did it do to accuracy and the edge cases?*

## On a PR

Comment on the PR (write access required, same as `/test`):

```
/llk-sfpu-test                      # auto: ops whose machine code changed, Wormhole + Blackhole
/llk-sfpu-test tanh exp             # exactly these ops
/llk-sfpu-test --arch bh            # one architecture
/llk-sfpu-test --base main          # current main vs main + this PR's kernel diff
```

Three workflows take it from there:

| workflow | does |
|---|---|
| `llk-sfpu-test.md` (gh-aw) | reads the request, the PR and its bounty issue; dispatches `llk-sfpu-report` on main; posts a "running" note |
| `llk-sfpu-report.yaml` | plans (pins the head SHA, finds the merge-base), measures one leg per arch, renders `report.md`; never comments |
| `llk-sfpu-summary.md` (gh-aw) | on completion, posts the report with a short AI summary on top, hiding older reports |

`llk-sfpu-report` can also be dispatched by hand from the Actions tab.

## End-to-end flow

```
reviewer: "/llk-sfpu-test tanh --arch bh"
   │
   ▼
llk-sfpu-test.md (gh-aw, issue_comment, write access only)            ubuntu runners
   1. 👀 on the comment
   2. pre-agent: PR metadata, changed files, diff, the bounty issue(s) it closes,
      the valid MathOperation names; PR number + head SHA written as facts outside
      the agent sandbox
   3. agent (claude-sonnet-5): turns the request into inputs -- ops (empty = auto),
      arch, base -- and extracts the bounty issue's targets and the PR's claims
   4. post-step: rewrites the dispatch from the facts (PR number, head SHA, ref main),
      validates ops/arch/base, fails closed
   5. safe outputs: dispatch llk-sfpu-report on main; one "running" comment
   │
   ▼
llk-sfpu-report.yaml (workflow_dispatch on main)
   plan              ubuntu-slim   pins the head SHA; merge-base from the compare API;
                                   notes if the PR moved since; picks the SKUs
   build-images      LLK CI image
   load-test-matrix  tests/pipeline_reorg/llk_sfpu_report_tests.yaml (one leg per arch)
   measure (per arch) N150 / P150b, LLK container, read-only token, no secrets
      ci_run.sh: checkout main, fetch the PR head + merge-base by SHA, then cli.py run:
        overlay   base and head trees (device C++ only from the PR)
        detect    compile every unary / typecast / binary perf variant on both sides,
                  compare math .text -> the ops whose code changed (max 8, own kernel first)
        perf      compile once per side and schedule, 3 interleaved device runs per side,
                  L1_TO_L1 + MATH_ISOLATE, SFPLOADMACRO on/off; gate comparer
        accuracy  test_sfpu_report_accuracy.py on both sides; host-side ULP / exact
                  comparison, special-input diff
        -> summary-<arch>.json, report-<arch>.md
   collect           ubuntu-slim   renders one report.md for all archs; meta.json
                                   (PR, SHA, run, agent context); artifact llk-sfpu-report
   │  (workflow_run: completed)
   ▼
llk-sfpu-summary.md (gh-aw)
   1. pre-agent: downloads the artifact; the PR number comes from meta.json
   2. agent: 3-6 sentences -- perf and accuracy verdicts, against the bounty targets
   3. post-step: the comment is the rendered report, with the agent's text placed in
      a labelled box at the marker; the agent never carries the tables
   4. safe outputs: one comment on the PR; hides older reports and the "running" note
```

What runs where, and with what: only the `measure` legs touch hardware; they run
main's host code and compile the PR's device C++, with `contents: read`, no secrets
and no persisted credentials. The two agents see PR text only as data and can only
emit a dispatch (the first) or a comment (the second), both rewritten by
deterministic post-steps.

## Locally

On a machine with a Wormhole or Blackhole card, from the tt-llk test venv
(`tests/setup_external_testing_env.sh`):

```bash
cd tt_metal/tt-llk
git fetch origin refs/pull/54080/head:refs/remotes/pr/54080
python3 sfpu_report/cli.py --arch wormhole --head pr/54080 run            # auto-detect
python3 sfpu_report/cli.py --arch wormhole --head pr/54080 run --ops Tanh # given ops
python3 sfpu_report/cli.py --arch wormhole --head pr/54080 detect         # which ops changed
python3 sfpu_report/report.py /tmp/llk-sfpu-report/summary-*.json         # render
```

`--base <ref>` sets the baseline (default: the merge-base with `origin/main`), `--work`
the scratch directory (default `/tmp/llk-sfpu-report`), `--formats Float16_b,Float32`
narrows perf and accuracy to those input formats, and `--check` makes the run a
pass/fail test: it exits 1 when the report lists a regression. Every PR comment ends
with these commands, pinned to the SHAs it measured, so anyone can reproduce a row.

## Reading the report

The top of the comment lists every regression (⚠️), one line each; the tables below
show only rows that changed, regressions first, and fold everything else into
`<details>`. A regression is:

- **perf**: slower beyond the LLK perf gate's thresholds (BH 2%, WH 8%, and more than
  30 cycles per loop), in `MATH_ISOLATE` or `L1_TO_L1`;
- **accuracy**: a higher max ULP (any rise below 16 steps, or more than 1%), or at
  least 0.1% of the lanes net worse, or new non-finite results; for comparisons and
  integer ops, more wrong lanes;
- **edge case**: a special input whose result changes kind (finite ↔ NaN/inf, sign,
  zero), or a NaN input that stops returning NaN.

## How it works

**Two sides, one harness.** `overlay.py` builds two sparse worktrees. By default
(`--mode merge-base`) they are the merge-base and the PR head; with `--mode rebase`
(`--base main` on a PR) they are the tool revision and the tool revision plus the PR's
device-code diff. Either way, only device C++ comes from the PR: the pytest harness,
its plugins and this tool always run from the tool checkout, with `LLK_HOME` pointing
the harness at a side's tree. A PR can change what runs on the Tensix, never what runs
on the runner. If the merge-base is too old for the current harness, the run falls
back to rebase mode and the report says so.

**Which ops.** `detect.py` compiles every variant of `perf_eltwise_unary_sfpu.py`,
`perf_eltwise_unary_typecast.py` and `perf_eltwise_binary_sfpu.py` on both sides (no
device needed) and compares the
`.text` of each variant's math ELF, matched across sides by its generated `build.h`.
An op is measured when its code changed, which catches ops that only include a
changed header. Both sides compile through the same symlinked path, because profiler
zone ids hash `__FILE__`.

**Perf.** `perf.py` compiles each side once, then alternates base/head device runs
(3 each by default), speed of light, `L1_TO_L1` and `MATH_ISOLATE`, with and without
SFPLOADMACRO. The raw CSVs go through the LLK perf gate's own comparer
(`tt-llk/perf/regression_compare.py`): median vs median, a regression is slower than
the threshold (BH 2%, WH 8%) and by more than 30 cycles per loop.

Binary rows are per operand tile, the unit the binary perf tests have used since
#57137: their `tile_cnt` counts both input tiles, so a binary row is about half a
result tile. The binary float family also sweeps the Dest broadcast; those variants
get their own rows, labelled like `SfpuElwadd (bcast Row)`.

**Accuracy.** `tests/python_tests/test_sfpu_report_accuracy.py` runs each op over every
finite input of bf16 and fp16 (fp32: every 65,536th value) and over a tile of special
values (NaN, ±inf, ±0, subnormals, the format's extremes), and saves the raw results.
`accuracy.py` compares the two sides lane by lane with the nightly ULP sweep's helpers
(`helpers/ulp.py`, `helpers/ulp_sweep.py`): max/mean ULP, lanes that got worse or
better, non-finite disagreements, and edge-case classes. No budget is involved: both
sides run on the same device against the same golden.

## Scope

| family | detected from | perf | accuracy |
|---|---|---|---|
| elementwise unary SFPU | `perf_eltwise_unary_sfpu.py` | ✅ | every finite bf16/fp16 input, fp32 strided, special values |
| typecast | `perf_eltwise_unary_typecast.py` | ✅ | – |
| elementwise binary SFPU (v1.1) | `perf_eltwise_binary_sfpu.py` | ✅ | random pairs from the functional test's domain, special-value cross product; comparisons and integer ops compared exactly |

Binary ops use the functional driver `sfpu_binary()` of `test_eltwise_binary_sfpu.py`, so
operand layout, domains and golden are the functional test's. Ops the harness cannot
feed in full say so in the report (logsigmoid is measured on x in [-8, 3.9]: its x > 4
branch reads a device-computed exp(-x) the harness cannot supply). Not covered: `Mask`,
`AddTopRow`, `CopyDest` (not elementwise functions of their inputs).

Later: ternary and structural SFPU kernels (reduce, topk, cumsum, welford), plots,
ttnn-level timing, Quasar (needs the emulator). A changed kernel the report cannot
attribute to a covered op is listed under the report's notes.

## Developing without a device

`run --simulator` runs detection and accuracy on ttsim (`TT_METAL_SIMULATOR`, with
`TT_METAL_DISABLE_SFPLOADMACRO=1`); perf is skipped, since ttsim cycles are not
silicon's. ttsim does not model every format the report measures (Int32 and fp32
binary SFPU inputs abort it), so a simulator run is a check of the tool, not of a PR.
The hardware-free tests are `tests/python_tests/test_sfpu_report_hw_free.py`.

## Testing on GitHub

1. **The measuring side runs before merge through a test branch.**
   `workflow_dispatch` only works for workflows on the default branch, so
   `nstamatovic/llk-sfpu-test-ci` carries one extra commit that runs the report
   through `llk-bit-exact.yaml`:
   `gh workflow run llk-bit-exact.yaml --ref nstamatovic/llk-sfpu-test-ci -f pr_number=<N>`.
   The report is in the run's Summary tab and in the `llk-sfpu-report` artifact.
   That commit never goes into the real PR.
2. **The two gh-aw workflows only run from main** (`issue_comment` and `workflow_run`
   use the default branch's file). Test them after merge on a real PR.
3. **Repo plumbing:** `owner_id` in `tests/pipeline_reorg/llk_sfpu_report_tests.yaml`
   (a Slack ID), the `llk.on_demand` budget in `.github/time_budget.yaml`, lock files
   compiled with gh-aw v0.89.21 (`gh aw compile`), a version main already uses.
4. **Sign-offs:** infra for the on-demand budget and runner use; security for
   running fork device code from a comment.
