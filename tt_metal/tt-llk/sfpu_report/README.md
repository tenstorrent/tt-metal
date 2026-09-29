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
the scratch directory (default `/tmp/llk-sfpu-report`).

## How it works

**Two sides, one harness.** `overlay.py` builds two sparse worktrees. By default
(`--mode merge-base`) they are the merge-base and the PR head; with `--mode rebase`
(`--base main` on a PR) they are the tool revision and the tool revision plus the PR's
device-code diff. Either way, only device C++ comes from the PR: the pytest harness,
its plugins and this tool always run from the tool checkout, with `LLK_HOME` pointing
the harness at a side's tree. A PR can change what runs on the Tensix, never what runs
on the runner. If the merge-base is too old for the current harness, the run falls
back to rebase mode and the report says so.

**Which ops.** `detect.py` compiles every variant of `perf_eltwise_unary_sfpu.py` and
`perf_eltwise_unary_typecast.py` on both sides (no device needed) and compares the
`.text` of each variant's math ELF, matched across sides by its generated `build.h`.
An op is measured when its code changed, which catches ops that only include a
changed header. Both sides compile through the same symlinked path, because profiler
zone ids hash `__FILE__`.

**Perf.** `perf.py` compiles each side once, then alternates base/head device runs
(3 each by default), speed of light, `L1_TO_L1` and `MATH_ISOLATE`, with and without
SFPLOADMACRO. The raw CSVs go through the LLK perf gate's own comparer
(`tt-llk/perf/regression_compare.py`): median vs median, a regression is slower than
the threshold (BH 2%, WH 8%) and by more than 30 cycles per loop.

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
