# Proposed changes to `pytest.ini`, `run_safe_pytest.sh` and `eval_test_runner.sh`

**Date:** 2026-09-08 · **pytest:** 9.0.3 (current release) · **Host:** Blackhole p150b, 32 cores

Every number below is **measured wall time, unprofiled**. Every table is tagged with the
experiment it came from — **(A)**, **(B)** or **(C)** — defined in the next section. Read that
section first; the three experiments measure different things at different scales, and mixing
their numbers is the easiest way to misread this document.

---

## 0. What was actually run

### Experiment A — real hardware, full golden suite, **collect-only**

**What is under test:** **production `ttnn.rms_norm`** — not a generated or agent-authored op.
The eval adapter `eval/adapters/ttnn_rms_norm.py` forwards the golden-test API straight to
`ttnn.rms_norm(input, epsilon=…, weight=gamma, memory_config=…, compute_kernel_config=…)`.

**tt-metal build:** worktree `wt-llk-refrun`, branch `mstaletovic/llk-refrun` @ `6e82b245136`,
based on `llk_helper_library` (a tt-metal development branch, **not** `main`), Release + Tracy.
The `rms_norm` implementation itself is stock production TTNN
(`ttnn/cpp/ttnn/operations/normalization/rmsnorm/` → `ttnn::prim::layer_norm`).

**Suite:** `eval/golden_tests/rms_norm` — the full cartesian, **40,828 collected cases**.
Of those, **~33,900 are skipped at runtime** as structurally `INVALID`, so **6,928 test bodies
actually execute**.

**Mode — this is the important caveat.** The run is a `up_front_collect` warm pass with
compilation disabled:

```bash
EVAL_TARGET_MODULE=eval.adapters.ttnn_rms_norm \
UP_FRONT_COLLECT=1 UP_FRONT_REAL_ALLOC=1 UP_FRONT_COLLECT_NO_COMPILE=1 \
PYTHONPATH="$TT_METAL:$EVAL_REPO" \
scripts/run_safe_pytest.sh --no-precompile --run-all \
    "$EVAL_REPO/eval/golden_tests/rms_norm" \
    -p tests.plugins.up_front_collect [-o addopts=...]
```

That means, explicitly:

- **No kernel compilation.** `UP_FRONT_COLLECT_NO_COMPILE=1` skips the session-end parallel JIT.
- **No device execution.** The bodies run under `NO_DISPATCH` graph capture; nothing is enqueued.
- **No PCC / no numerical checking.** The plugin no-ops `check_output` / `comp_pcc`; outputs would
  be garbage anyway.
- **What *is* real:** pytest's full collection and per-test protocol over 40,828 items, all
  fixtures, all host-side Python in the golden helpers, and real host-side **program
  construction** in ttnn/tt_metal for each op (3,649 programs built, 3,554 distinct).

So Experiment A measures the **warm pass**, and nothing else. It is the right target for
proposal #3, and it is where the `-rA` behaviour was first isolated, but it is **not** a graded
run.

### Experiment B — real hardware, graded run, 4,000-case slice

A genuine graded run — real device execution, real PCC checks, real inline kernel compilation —
over the **first 4,000 cases** of the same suite (365 failed / 360 passed / 3,275 skipped).
Used only to check whether reporting flags matter at small item counts. Cold-cache run: 391.3 s.
Warm-cache runs: 27.94 s (quiet) vs 28.72 s (repo default addopts).

### Experiment C — device-free synthetic pytest suite

No ttnn, no device, no tt-metal at all: a 20-line parametrized `test_synth.py` with eval-shaped
node ids and a configurable skip / fail / pass mix (source in §6). This is where every
*scaling* claim comes from, because it lets item count, pass count and fail count be varied
independently in seconds instead of minutes.

**All 20,000-item and 40,000-item numbers in this document are Experiment C.** They are *not*
rms_norm runs.

### Experiment D — real hardware, 1,200-case graded slice, `--tb` A/B

The same golden suite, first 1,200 cases, run as a **genuine graded run** (real execution, real
PCC, real compilation) twice — once with `--tb=short`, once with `--tb=line` — keeping both
JUnit XMLs. 101 failed / 123 passed / 976 skipped in each arm. Used to measure what `--tb`
actually changes in the stored data, rather than inferring it. Feeds §2's risk section and §7.

### What was NOT measured — state this when the PRs are raised

**No full 40,828-case graded run was measured with the proposed flags on hardware.** The graded-run
claims (#2) rest on Experiment C plus the code path being identical. A single before/after graded
run would close that gap and is the obvious thing to ask for in review.

---

## Summary

| # | Change | Magnitude | Evidence | Risk |
|---|---|---|---|---|
| 1 | `pytest.ini`: `-rA` → `-ra` | warm pass 166 s → 58.9 s | **A** + **C** | Low |
| 2 | eval graded run: `--tb=short` → `--tb=line` | 48.3 s → 19.6 s at 20 k with #1 | **C** + **D** | Low |
| 3 | Warm passes: neutralize inherited addopts | 166 s → 54.6 s | **A** | Very low |
| 4 | `run_safe_pytest.sh`: stop clobbering `PYTHONPATH` | silent total loss of precompile | observed | Very low |
| 5 | Make eval independent of the host `pytest.ini` | robustness | reasoning | Very low |
| 6 | Report the quadratic upstream | fixes the class | **C** | n/a |
| 7 | Strip the C++ backtrace from `failure_message` | 2,000 → 235 chars stored; usable dashboard previews | **D** | Low |

---

## Root cause (shared by #1, #2, #3)

`_pytest/terminal.py`:

```python
def _get_teardown_reports(self, nodeid):          # :1141
    reports = self.getreports("")                 # builds a NEW list of EVERY report
    return [r for r in reports                    # then scans that list again
            if r.when == "teardown" and r.nodeid == nodeid]

def getreports(self, name):                       # :1063
    return [x for x in self.stats.get(name, ()) if not hasattr(x, "_pdbshown")]
```

`stats[""]` holds every passed setup and teardown — ~2 entries per test. To answer *"did this
test's teardown print anything?"* pytest allocates a fresh ~2N-element list (one `hasattr` per
element), then walks it again — ~3N operations **to find at most one report**, uncached. A dict
lookup written as two linear scans.

Called once per report from two places:

- `summary_passes_combined` (`:1139`) — per **passed** report. Needs `P` in reportchars, i.e. `-rA`.
- `summary_failures_combined` (`:1199`) — per **failed** report. Gated only by `--tb`; **no `-r`
  requirement at all**.

Cost is `O((passed + failed) × total_reports)`.

**It is the lookup, not the printing** — Experiment C, 20,000 items, 1,660 failed / 1,660 passed,
**totals**:

| variant | with `-rA` | with no `-r` |
|---|---|---|
| untouched | 49.8 s | 36.2 s |
| `_get_teardown_reports` stubbed to `[]`, **all printing kept** | **22.5 s** | **22.3 s** |
| all writing stubbed out, **all scanning kept** | 50.5 s | 39.8 s |

Removing every byte of output changes nothing. Removing the scan removes the cost.

Only `--tb=line` (different branch) and `--tb=no` (function skipped) avoid the failure-side call.
`short` and `long` both pay in full.

---

## 1. `pytest.ini` — `-rA` → `-ra`

### Issue
`addopts` applies `-rA` ("short summary for **all** outcomes, including passed") repo-wide. `-rA`
is the only reportchars setting that enables `summary_passes`, hence the passed-side quadratic.

### How it manifests
Time disappears *after* the last test, invisible to any in-session timer: the pytest process was
**59 s old at `pytest_sessionfinish`** while pytest reported 166 s — ~6,900 log lines were written
afterwards. It reads as "pytest is slow", not as a flag.

**Experiment A** — full rms_norm warm pass, 40,828 collected / 6,928 bodies, collect-only.
Figures are pytest's own reported duration; each row is one run, `-o addopts=` set explicitly:

| `addopts` | pytest duration |
|---|---|
| `--import-mode=importlib` (baseline) | 54.6 s |
| `--import-mode=importlib --junitxml=…` | 57.3 s |
| `--import-mode=importlib -rA` | **163.1 s** |
| repo default (`-vvs -rA --durations=25 --junitxml=…`) | 166 s |

So `-rA` accounts for ~108 s of the ~111 s the repo default adds.

The four rows above are directly comparable: same script, same suite, same machine, only
`-o addopts=` differs, and none of them loads the profiling plugin. The repo-default
configuration was run five times over the course of this work and landed at 165.7, 166.7, 168.0,
172.8 and 173.7 s — **166 s in the tables is a representative run, not a best case**.

### Fix
```diff
-addopts = --import-mode=importlib -vvs -rA --durations=25 --junitxml=generated/test_reports/most_recent_tests.xml
+addopts = --import-mode=importlib -vvs -ra --durations=25 --junitxml=generated/test_reports/most_recent_tests.xml
```
`-ra` is "all outcomes **except** passed" — the form pytest's own docs recommend. Failures,
errors, skips, xfails and xpasses all stay. Only per-test `PASSED` lines go.

### Risk: **Low**
- Nothing in this repo parses the `-r` summary: checked `scripts/` and `.github/workflows/` for
  anything grepping/`awk`-ing/regexing `PASSED`/`FAILED`; the only hits are unrelated literals.
- eval's emergency fallback (`eval_test_runner.sh:622`) greps `' PASSED$'`, which is the **`-vvs`
  progress** format (`nodeid PASSED`), **not** the `-rA` summary format (`PASSED nodeid`). `-vvs`
  is untouched.
- Residual risk: an external consumer outside this repo reading those summary lines. Not visible
  from here.

### Magnitude: **Large, and grows as the op improves**
**Experiment C** — suite held at 40,828 items, varying how many pass. `-rA` **cost** (delta vs
the same run without `-rA`); linear at ~0.028 s per passing test:

| passing tests | 6,777 | 12,248 | 18,373 | 22,986 | 40,828 |
|---|---|---|---|---|---|
| cost of `-rA` | 196.7 s | 552.8 s * | 519.7 s | 637.2 s | **1,094.8 s** |

\* breaks monotonicity — machine contention, not signal.

The rms_norm reference run passes only 6,765 today because production TTNN refuses ~half the
cells. **As those refusals are fixed this cost grows toward the 18-minute figure.** Our
skip-heavy suite is the *mild* case: Experiment C at 20,000 items, `-rA` cost is **27.0 s** with
our 83% skip rate and **133.5 s** if every test passes.

---

## 2. eval graded run — `--tb=short` → `--tb=line`

### Issue
The graded run **already passes `--tb=short`** (`eval_test_runner.sh:574` sim branch, `:592`
hardware branch). `short` takes the `else` branch of `summary_failures_combined` and pays the
failure-side quadratic in full. **This is not gated by `-r`, so #1 does not touch it.**

### How it manifests
Same signature as #1 — time after the last test. It scales with the **failure** count, so it is
worst exactly when a run is going badly.

**Experiment C** — 20,000 items, graded-run mix (83.4% skipped, 1,660 failed, 1,660 passed).
**Totals**, one flag varied at a time:

| addopts | total |
|---|---|
| *(no `-r` at all)* | 37.1 s |
| `-ra` | 39.6 s |
| `-rA` | 52.9 s |
| `--tb=line` | **19.6 s** |
| `--tb=no` | **19.5 s** |

The identical run with **zero failures** is 18.4 s. So 18.7 s of what *bare* `pytest` spends here
is the failure path — paid with no reporting flags set at all.

> This is the row that is easy to misread against §1. The **52.9 s** here is a *total* for a
> 20,000-case **synthetic** run; the **163.1 s** in §1 is a *total* for a 40,828-case **real**
> collect pass. Different experiments, different scales.

### Fix
```diff
     pytest "${TEST_DIR}" \
         "${PYTEST_FORWARD_ARGS[@]}" \
         --junitxml="${JUNIT_XML}" \
         -p eval.hang_plugin -p eval.metrics_plugin -p eval.axes_plugin \
-        --tb=short \
+        --tb=line \
         -q \
```
Both occurrences: `eval_test_runner.sh:574` and `:592`.

### Risk: **Low** — measured on real failures, not reasoned about

`--tb` is baked into `longrepr` at report creation (`nodes.py:534`,
`_repr_failure_py(excinfo, style=tbstyle)`), and the JUnit writer emits `str(report.longrepr)`
(`junitxml.py:207`), so in principle trimming `--tb` shrinks what is stored.

**Experiment D** (added for this question): the same real 1,200-case graded slice of
`eval/golden_tests/rms_norm` run twice — `--tb=short` (today) and `--tb=line` — keeping both
JUnit XMLs, then parsed through eval's own `classify_failures.parse_junit_xml`.
101 failures in each arm, all real `TT_FATAL`s from production `ttnn.rms_norm`:

| | `--tb=short` (today) | `--tb=line` |
|---|---|---|
| XML `message` attribute, median | 12,118 chars | **12,118 chars — byte-identical, 101/101** |
| XML body (traceback), median | 13,086 chars | 12,522 chars (−4%) |
| stored `failure_message` (capped at 2,000) | 2,000 | 2,000 — **101/101 truncated in both arms** |
| `failure_category` | — | **identical, 101/101** |
| dashboard preview (first 120 chars) | — | **identical, 101/101** |

**Nothing that reaches the database or the dashboard changes.** The reason is §7 below: a real
`TT_FATAL` message carries a ~12 KB C++ backtrace, so the exception message alone blows past the
2,000-char storage cap long before the Python traceback is reached. The Python frames that
`--tb=line` removes were already being truncated away.

My earlier rating of this as Low–Medium was based on a *synthetic* failure with a short exception
message, where the Python traceback was a meaningful fraction of the text. On real failures it is
noise at the 4% level.

**Residual risk:** the per-frame lines in `pytest_stdout.log` (`helpers.py:133: in run_rms_norm`)
do disappear for anyone debugging by reading that log rather than re-running the case. Given the
12 KB backtrace sitting next to them, this is a small loss.

### Magnitude: **Medium–Large, worst when runs are failing**
**Experiment C** — 20,000 items, graded mix, `--junitxml` on, realistic flag combinations:

| addopts | total |
|---|---|
| `-vvs -rA --durations=25` (today) | 48.3 s |
| `-vvs -ra --durations=25` (#1 only) | 37.5 s |
| `-vvs -ra --durations=25 --tb=line` (#1 + #2) | **19.6 s** |

At the real 40,828-item scale both terms are several times larger — **extrapolated, not measured**
(see §0).

---

## 3. Warm passes — neutralize the inherited addopts

### Issue
Both precompile warm passes inherit the repo-wide addopts, so each produces ~40,000 verbose
progress lines, a full `-r` summary, a durations table and a 40,000-entry JUnit XML — into a log
that is grepped for exactly one line (`UP_FRONT_COLLECT_RESULT:`). The warm pass also has no
`--junitxml` of its own, so it writes the shared
`generated/test_reports/most_recent_tests.xml` for no reason.

### How it manifests
The warm pass takes ~3× longer than the work it does. It runs before the graded run with output
redirected, so nobody looks.

### Fix
`eval_test_runner.sh:430`:
```diff
-        pytest "${TEST_DIR}" "${PYTEST_FORWARD_ARGS[@]}" -p tests.plugins.up_front_collect > "$clog" 2>&1
+        pytest "${TEST_DIR}" "${PYTEST_FORWARD_ARGS[@]}" -p tests.plugins.up_front_collect \
+            -o addopts=--import-mode=importlib -q > "$clog" 2>&1
```
`run_safe_pytest.sh:369`: identical addition.

### Risk: **Very low**
The only consumer of the log is the `UP_FRONT_COLLECT_RESULT:` grep, printed by the plugin itself
and independent of pytest verbosity. `--import-mode=importlib` is retained because dropping it
changes import semantics. Worst case is a less readable log when debugging a warm pass.

### Magnitude: **Large, on every eval run**
**Experiment A**, full rms_norm warm pass: **166 s → 54.6 s**. (54.6 s includes the
`torch.manual_seed` stub already applied to `up_front_collect.py`; without it, 58.9 s.)

---

## 4. `run_safe_pytest.sh:368` clobbers `PYTHONPATH`

### Issue
```bash
LOGURU_LEVEL=ERROR PYTHONPATH="$PRECOMPILE_PLUGIN_DIR" \
```
`PRECOMPILE_PLUGIN_DIR` is `$REPO_DIR`. The assignment **replaces** the inherited `PYTHONPATH`
instead of prepending, for the warm-pass subprocess.

### How it manifests
Any suite needing paths beyond the repo root — the eval golden tests need the eval repo on the
path — fails to import during the warm pass. It does not fail loudly; the script logs

> `PRECOMPILE ✗ warmup FAILED (pytest exit N) … -> warmed NOTHING; running COLD.`

and the graded run compiles inline and serially. A performance cliff behind a plausible-looking
log line. Observed directly while setting up Experiment A.

### Fix
```diff
-        LOGURU_LEVEL=ERROR PYTHONPATH="$PRECOMPILE_PLUGIN_DIR" \
+        LOGURU_LEVEL=ERROR PYTHONPATH="$PRECOMPILE_PLUGIN_DIR${PYTHONPATH:+:$PYTHONPATH}" \
```
Repo root stays first, so `tests.plugins.up_front_collect` still resolves locally — the reason the
override exists.

### Risk: **Very low**
Strictly widens the search path with the original entry still first.

### Magnitude: **Silent total loss of precompile whenever it triggers**
Not a few percent — the entire parallel warm pass is skipped.

---

## 5. Make eval independent of the host repo's `pytest.ini`

### Issue
Whether the eval suites inherit `-rA` depends on checkout layout: pytest walks up from the test
path for its ini, so nested inside a tt-metal clone it finds tt-metal's `pytest.ini`, standalone
it finds none. #1 only helps the nested case, and lands in a different repository from the code it
protects.

### Fix
Have `eval_test_runner.sh` state its reporting policy explicitly on the graded run — e.g.
`-o addopts=--import-mode=importlib` plus the flags it already passes, adding `-ra` if a summary
is wanted.

### Risk: **Very low**
Makes explicit what is implicit. Requires re-passing anything load-bearing from tt-metal's
addopts: `--durations=25` is not, `--junitxml` is already overridden, and **`-vvs` is**
load-bearing (the `' PASSED$'` fallback at `:622`).

### Magnitude: **Robustness, not speed**

---

## 6. Report the quadratic upstream

### Fix (upstream)
Build a `nodeid → teardown report` index once. Note the tempting one-word alternative — moving
`_handle_teardown_sections(rep.nodeid)` inside the `if rep.sections:` guard above it — is
**wrong**: `rep.sections` is the *call* report's output while teardown output lives on the
teardown report, so it would silently drop teardown output for tests that printed nothing during
the call.

### Reproduction — Experiment C, no ttnn, no device
```python
# test_synth.py
import os, pytest
N = int(os.environ.get("SYNTH_N", "40000"))
_IDS = [f"1x1x32x{64+i%97}-dtype=FLOAT32-gamma_layout=ROW_MAJOR-layout=TILE-"
        f"memory_layout=BLOCK_SHARDED-rank=4-idx={i}" for i in range(N)]

@pytest.mark.parametrize("case", range(N), ids=_IDS)
def test_op(case):
    if (case % 1000) / 1000.0 < 0.833:
        pytest.skip("matches INVALID entry")
    assert True
```
```bash
SYNTH_N=40000 pytest test_synth.py -o addopts=    -q   #  36 s
SYNTH_N=40000 pytest test_synth.py -o addopts=-rA -q   # 236 s
SYNTH_N=40000 pytest test_synth.py -o addopts=-ra -q   #  36 s
```

Scaling with suite size at a fixed 16.6% pass rate — `-rA` **cost** (delta):

| items | 2,500 | 5,000 | 10,000 | 20,000 | 40,000 |
|---|---|---|---|---|---|
| cost | 0.36 s | 1.40 s | 5.21 s | 27.24 s | 200.15 s |

10.2× the items, 135× the cost. The baseline without `-rA` is clean linear
(3.1 → 5.2 → 9.4 → 18.3 → 36.0 s).

Most persuasive framing upstream is the ordinary one: **20,000 passing tests, `-rA`, 133 s of
pure list scanning.**

---

## 7. Strip the C++ backtrace out of `failure_message`

Not a performance item. Found while checking whether §2 would change the dashboard — it does not,
but this does.

### Issue
`classify_failures.parse_junit_xml` stores `failure_message = f"{message}\n{traceback}"[:2000]`
(`classify_failures.py:312, 322`). A production `TT_FATAL` exception message embeds a full C++
backtrace — dozens of `--- /path/to/_ttnncpp.so(+0x…) [0x…]` lines — so the 2,000-char budget is
spent on symbol dumps instead of on why the test failed.

**Experiment D**, 101 real `TT_FATAL` failures:

| | median |
|---|---|
| `message` attribute, full | **12,118 chars** |
| same, truncated at `\nbacktrace:` | **258 chars** |
| messages containing a C++ backtrace | 101 / 101 |

**The backtrace is 97.9% of the message.**

### How it manifests
**In the database:** of run 1006's 3,705 failed/error rows, **3,409 (92%) are truncated at the
2,000-char cap** — `min=604, p50=2000, p90=2000, p99=2000, max=2000`. Everything past the cap is
discarded, and what was kept is mostly hex addresses.

**In the dashboard:** the tests table renders `failure_message[:120]` in the cell
(`dashboard.py:5529`), with the full text behind a click (`data-full` / `toggleMsg`). Today those
120 characters look like this:

```
RuntimeError: TT_FATAL @ /localdev/mstaletovic/metal_metal/wt-llk-refrun/ttnn/cpp/ttnn/operations/normalization/layernor
```

The preview is exhausted by an absolute build path and cut off mid-word, before reaching the
assertion. Two different assertions in the *same* source file would be indistinguishable in the
table.

### Fix
Trim the backtrace (and optionally relativize the build path) before storing, in
`classify_failures.py` where `full_text` is built:

```python
_BACKTRACE = "\nbacktrace:"

def _trim(text: str) -> str:
    # The C++ backtrace is ~98% of a TT_FATAL message and is identical for every
    # instance of a given assertion; it survives in the raw pytest log.
    head, sep, _ = text.partition(_BACKTRACE)
    return head if sep else text
```

Same failure, stored and previewed after trimming:

```
RuntimeError: TT_FATAL @ ttnn/cpp/ttnn/operations/normalization/layernorm/device/layernorm_device_operation.cpp:167: a.m
```

**Stored length: 2,000 → 235 chars**, no truncation, and the `info:` explanation now survives
into the DB instead of being cut off.

### Risk: **Low**
- Classification is unaffected: every pattern in `classify_failures.PATTERNS` matches exception
  text (`Out of Memory`, `CompilationError`, `TypeError: …`, `NotImplementedError`,
  `severity=bug`), all of which appear *before* the `backtrace:` marker. Worth a re-run of the
  classifier over an existing XML to confirm category-for-category equality before landing.
- Messages with no `backtrace:` marker (e.g. `CheckOutputError`, which is already 604–642 chars
  and never truncated) pass through untouched.
- The backtrace remains in `pytest_stdout.log`, so nothing is permanently lost.
- Applies to **new** runs only; existing rows keep their truncated text.

### Magnitude
- **92% of failure rows currently truncated → ~0%.**
- ~88% less text stored per failing row (2,000 → ~235 chars); ~7.4 MB → ~0.9 MB per full run.
- The dashboard preview starts showing the assertion instead of a path prefix — the actual
  day-to-day benefit.

---

## Sequencing

- **tt-metal PR:** #1 (`pytest.ini`) + #4 (`PYTHONPATH`) + #3's `run_safe_pytest.sh` half.
- **tt_ops_code_gen PR:** #2 + #3's `eval_test_runner.sh` half + #5.
- **Upstream pytest:** #6.

#1, #3, #4 and #7 are safe to land as-is. #2 is now also low-risk on the evidence of Experiment
D, but a before/after full graded run would still replace the extrapolation in §0.

#7 is independent of the performance work and is arguably the most user-visible improvement
here.
