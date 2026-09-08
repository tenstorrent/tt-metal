# Profiling the up-front collect pass

**Date:** 2026-09-08 · **Workload:** production `ttnn.rms_norm` against the full
`eval/golden_tests/rms_norm` suite — 40,828 cases, 6,928 real bodies (the rest are `INVALID`
cells), Blackhole p150b.
**Precondition:** run with the Metal 2.0 defer-compile fix in place
(see `UP_FRONT_COLLECT_EAGER_COMPILE.md`), so the collect pass no longer JIT-compiles inline.
All timings are **collect-only** (`UP_FRONT_COLLECT_NO_COMPILE=1`) — no kernel compilation.

---

## Headline

| configuration | wall |
|---|---|
| as invoked today | **166 s** |
| `-rA` → `-ra` | **58.9 s** |
| + stub `torch.manual_seed` during collect | **54.6 s** |

**3× on the collect pass. The dominant term is a quadratic loop in pytest's `-rA` handling —
one character of one flag.**

For scale: the same pass took **6,924 s** before the defer-compile fix. That fix removed the
inline compilation; this writeup is about what was left underneath it.

## The main finding: a quadratic loop in pytest's `-rA` handling

`pytest.ini:4` applies repo-wide:

```ini
addopts = --import-mode=importlib -vvs -rA --durations=25 --junitxml=generated/test_reports/most_recent_tests.xml
```

Isolating each flag on the full collect pass:

| addopts | wall | delta |
|---|---|---|
| `--import-mode=importlib` (baseline) | 54.6 s | — |
| `+ --junitxml=…` | 57.3 s | +2.7 s |
| `+ -rA` | **163.1 s** | **+108.5 s** |
| full default | 166 s | +111 s |

`-rA` alone is 108 s of the 111 s. Everything else together is ~3 s.

### Root cause — `_pytest/terminal.py`, and it is a real pytest bug

`-r` selects which outcome categories get a "short test summary" line. The **`A`** in `-rA`
means *all categories including passed*, which enables `summary_passes`:

```python
def summary_passes_combined(self, which_reports, sep_title, needed_opt):   # terminal.py:1124
    if self.config.option.tbstyle != "no":
        if self.hasopt(needed_opt):                       # "P" — only with -rA
            for rep in self.getreports(which_reports):    # every PASSED report
                ...
                self._handle_teardown_sections(rep.nodeid)

def _get_teardown_reports(self, nodeid):                  # terminal.py:1140
    reports = self.getreports("")                         # ← full scan of ALL reports
    return [r for r in reports if r.when == "teardown" and r.nodeid == nodeid]   # ← again

def getreports(self, name):                               # terminal.py:1063
    return [x for x in self.stats.get(name, ()) if not hasattr(x, "_pdbshown")]
```

For **every passed report**, pytest performs **two full linear scans of every report in the
session** to find that test's teardown report — a lookup that should be a dict keyed by nodeid.
The cost is `O(passed × total_reports)`.

Profiling a 20,000-case synthetic run confirms it exactly: `terminal.py:1143` and
`terminal.py:1063` are the top two entries by self time (23.3 s and 14.1 s of 88 s), driven by
**134 million `hasattr` calls**, and `getreports` is called 3,340 times — precisely the number of
passed tests.

This is not our suite being unusual. It is a quadratic loop that only becomes visible once a
single pytest session has tens of thousands of tests, which is exactly what an eval golden
cartesian is.

### Device-free reproduction

A plain parametrized suite with eval-shaped nodeids, ~83% skipped — no ttnn, no device:

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
SYNTH_N=40000 pytest test_synth.py -o addopts=      -q   # 36 s
SYNTH_N=40000 pytest test_synth.py -o addopts=-rA   -q   # 236 s
SYNTH_N=40000 pytest test_synth.py -o addopts=-ra   -q   # 36 s
```


| items | no `-r` | `-rA` | cost of `-rA` |
|---|---|---|---|
| 2,500 | 3.07 s | 3.42 s | 0.36 s |
| 5,000 | 5.21 s | 6.61 s | 1.40 s |
| 10,000 | 9.43 s | 14.64 s | 5.21 s |
| 20,000 | 18.31 s | 45.55 s | 27.24 s |
| 40,000 | 35.96 s | **236.11 s** | **200.15 s** |

The baseline is clean linear (3.1 → 5.2 → 9.4 → 18.3 → 36.0). The `-rA` delta quadruples per
doubling up to 20 k, then worsens (7.3× on the last doubling, as GC pressure compounds).

### Fix — `-rA` → `-ra`

Three mitigations, measured on the 40,000-case synthetic:

| addopts | wall | keeps failure/skip summary? |
|---|---|---|
| *(none)* | 35.9 s | no summary at all |
| `-rA` | 234.4 s | yes |
| **`-ra`** | **36.2 s** | **yes — all categories except passed** |
| `-rA --tb=no` | 36.4 s | yes (`summary_passes_combined` early-returns) |
| `-rfEsxX` | 36.6 s | yes (explicit categories) |

**`-ra` costs nothing and loses nothing anyone wants.** A per-test `PASSED` line for 6,800 passing
tests is not information; the failures and skips are, and `-ra` keeps them. Changing `pytest.ini`
from `-rA` to `-ra` fixes this everywhere at once, for every suite in the repo.

If the warm pass is to be trimmed separately, `eval_test_runner.sh:430` can also pass
`-o addopts=--import-mode=importlib -q` — it discards this output entirely (`$clog` is only
grepped for the `UP_FRONT_COLLECT_RESULT:` line).

Worth reporting upstream: the fix in pytest is to index teardown reports by nodeid once instead
of rescanning per passed report.

### The same loop is on the failure path — and it is NOT gated by `-r` at all

`summary_failures_combined` (`terminal.py:1199`) ends its per-report loop with the identical
call:

```python
for rep in reports:
    msg = self._getfailureheadline(rep)
    self.write_sep("_", msg, red=True, bold=True)
    self._outrep_summary(rep)
    self._handle_teardown_sections(rep.nodeid)      # ← same full-scan-per-report
```

and `summary_failures()` invokes it with **no `needed_opt`**, so this runs whenever there are
failures and `--tb != "no"`. **Removing `-r` entirely does not help.** The cost is
`O(failed × total_reports)` on top of `O(passed × total_reports)`.

Measured, 20,000 items with the graded-run mix (83.4% skipped, 1,660 failed, 1,660 passed):

| addopts | wall |
|---|---|
| *(nothing — no `-r` at all)* | 37.1 s |
| `-ra` | 39.6 s |
| `-rA` | 52.9 s |
| `--tb=line` | **19.6 s** |
| `--tb=no` | **19.5 s** |

The same run with zero failures costs 18.4 s. So of the 37.1 s that bare `pytest` spends,
**18.7 s is the failure path** — paid with no reporting flags set whatsoever.

Only `line` and `no` escape: the `style == "line"` branch skips `_handle_teardown_sections`,
and `style == "no"` skips the whole function. `--tb=short` and `--tb=long` both take the
`else` branch and pay in full.

### `--tb=line` is safe for eval — the JUnit XML keeps what the classifier needs

`--tb` feeds `_repr_failure_py(excinfo, style=tbstyle)` (`nodes.py:534`), and the JUnit writer
emits `str(report.longrepr)` (`junitxml.py:207`), so trimming `--tb` does shrink the XML. It
does **not** cost anything that is actually parsed:

| `--tb=` | `message` attribute | body | contains `TT_FATAL` |
|---|---|---|---|
| default | `RuntimeError: TT_FATAL @ …:214: gamma.layout() …` | 1170 chars | yes |
| `short` | *identical* | 324 chars | yes |
| `line` | *identical* | 224 chars | yes |
| `no` | *identical* | 224 chars | yes |

The `message` attribute is **byte-identical across every style** — it comes from
`longrepr.reprcrash.message`, the exception line, not the traceback. And every pattern in
`classify_failures.PATTERNS` matches exception text (`Out of Memory`, `CompilationError`,
`TypeError: .*missing \d+ required`, `NotImplementedError`, …), never a source frame. Under
`--tb=line` the body still carries `E   RuntimeError: TT_FATAL @ …`, which is what
`full_text = f"{message}\n{traceback}"` is classified on.

What is lost is the source-context traceback in the terminal log — for a run where the same
refusal repeats thousands of times, and where reproducing one case is a single pytest command.

### Recommended: `-ra --tb=line`

Measured on 20,000 items with the graded-run failure mix, `--junitxml` on:

| addopts | wall |
|---|---|
| `-vvs -rA --durations=25` (today) | 48.3 s |
| `-vvs -ra --durations=25` | 37.5 s |
| **`-vvs -ra --durations=25 --tb=line`** | **19.6 s** |

`-ra` alone recovers only the passed half. `--tb=line` recovers the failure half. Together they
take a 40,828-case graded run from paying both quadratics to paying neither, while keeping one
summary line per failure and per skip group.

`-vvs` must stay: the no-XML fallback at `eval_test_runner.sh:622` greps `' PASSED$'`, which is
the verbose progress format.

### Does the graded run need any of this? No.

The graded run has the same 40,828 items and pays **both** quadratics — the passed one via
`-rA`, the failure one unconditionally. Checking what actually consumes pytest's output there:

- **`--junitxml` — required.** `classify_failures.parse_junit_xml` reads the XML; that is the
  results path.
- **`-vvs` — required.** The emergency fallback at `eval_test_runner.sh:622` greps
  `' PASSED$'` / `' FAILED$'` when no XML was produced. That is the **verbose progress** format
  (`nodeid PASSED`), not the `-rA` summary format (`PASSED nodeid`).
- **`-rA` — consumed by nothing.** No parser reads it, and a 40,828-line summary is not
  human-readable anyway.
- **the FAILURES traceback section — consumed by nothing.** The classifier reads the XML, whose
  `message` attribute is unaffected by `--tb`.

## Where the remaining 54.6 s goes

Wall-clock phase timers, no cProfile:

| | |
|---|---|
| call phase (6,928 bodies) | 17.4 s — 2.5 ms/body |
| setup | 6.3 s |
| teardown | 1.6 s |
| interpreter + torch/ttnn import | 3.5 s |
| pytest collection (building 40,828 items) | 0.2 s |
| remainder — pytest per-item protocol | ~25 s |

The ~25 s remainder is pytest's own machinery across all 40,828 items: report objects, hook
dispatch, fixture resolution. **34,063 of those items are `INVALID` cells skipped at *runtime***
— generated as items, then reported as skips. Deselecting them at collection time would remove
most of that, but it is a change to `test_golden.py`'s parametrization, not to the plugin.

Inside the call phase, the actual `ttnn` op call — `ttnn/ttnn/decorators.py:709`, whose profiler
"self time" *is* the C++ program build behind the nanobind boundary — runs ~0.95 ms/op. That is
the irreducible payload: building a program is what the collect pass exists to do.

## The plugin's protections: audited, and they work

`tests/plugins/up_front_collect.py` claims to sidestep host-side work under `NO_DISPATCH`.
Measured on the full run:

- **shape-only `from_torch`** — 12,526 shallow allocations, **4** fallbacks. Working as designed.
- **dedup** — 3,649 ops stashed → 3,554 unique. Checked properly rather than assumed: a probe
  collecting 5 identical ops yields `unique=1` with the device program cache both **enabled and
  disabled**, so the `hash == 0` synthetic-key fallback in `ProgramCollector::collect` is not
  silently firing and destroying dedup.
- **verifier no-ops** (`check_output`, `comp_pcc`) — no measurable readback/metrics cost remains.

### The one gap: `torch.manual_seed`

The plugin stubs `torch.randn`/`rand`, but not `torch.manual_seed`, which the golden harness
calls once per case. `torch.manual_seed` also seeds the **cuda, mtia and xpu** backends, and each
of those `_lazy_call` records a full Python stack via `traceback.format_stack()` for an
initialization that never happens on a box without those devices — three stack captures per call.

Fixed by seeding only the CPU generator (identical host RNG behavior):

```python
def _fast_manual_seed(seed):
    return torch.random.default_generator.manual_seed(int(seed))
```

**Measured: 58.9 s → 54.6 s (7%).** This is the only code change made in this investigation.

## Not defects, but worth knowing

**47% of bodies stash nothing.** 3,254 of 6,928 bodies end in a `TT_FATAL` from
`layernorm_device_operation` validation — production TTNN refusing that cell. Verified with
`UP_FRONT_LOG_SWALLOWED=1`: every swallowed body reports `stashed=0`. This is *correct* for a
characterization run (those cells fail in the graded run too, so there is nothing to warm), but
it means roughly half the collect pass is setup + body + refusal + swallow for zero programs.
Inherent to measuring an op against a suite it rejects half of; not a plugin bug.

**`_release_device_tensors_in_traceback`** (`eval/golden_tests/conftest.py:139`) fires on all
~34,000 skipped cells, because a skip carries an `excinfo` too. It then walks every frame calling
`isinstance(..., ttnn.Tensor)` — 6.5 M calls. Guarding it:

```python
if call.excinfo is not None and not call.excinfo.errisinstance(Skipped):
```

measured 173.6 s → 164.0 s **in the verbose configuration**. Not re-measured after the `-rA` fix,
so treat that as an upper bound. Reverted here to keep the submodule pin clean.

**`_drives_capture`** calls `inspect.getsource` once per body (431 k `tokenize` calls) for a
handful of distinct test functions. Memoizing on `item.function` is free. Small.

## Method notes

- **cProfile lies about magnitude here.** It put `torch.manual_seed` at 8.2 s; the real,
  unprofiled saving was 4.3 s. It inflates Python-frame-heavy paths (traceback machinery
  especially) by 2–3×. Every number in this document is unprofiled wall time; cProfile was used
  only to *rank* candidates and to find callers (`pstats.print_callers`), never to size them.
- **The 166 s was not visible from inside pytest.** The process was only **59 s old** at
  `pytest_sessionfinish` while pytest reported 166 s — ~6,900 log lines were written *after* the
  session ended. Phase timers registered inside the session will never see this cost; only
  process-level wall time does.

## Reproducing

```bash
export PYTHONPATH="$TT_METAL:$EVAL_REPO"
export EVAL_TARGET_MODULE=eval.adapters.ttnn_rms_norm
export UP_FRONT_COLLECT=1 UP_FRONT_REAL_ALLOC=1 UP_FRONT_COLLECT_NO_COMPILE=1

# slow (inherits pytest.ini addopts)
pytest $EVAL_REPO/eval/golden_tests/rms_norm -p tests.plugins.up_front_collect

# fast
pytest $EVAL_REPO/eval/golden_tests/rms_norm -p tests.plugins.up_front_collect \
    -o addopts=--import-mode=importlib -q
```

## Summary of recommendations

| # | change | measured | where |
|---|---|---|---|
| 1 | `-rA` → `-ra` | removes the passed-side quadratic | `pytest.ini:4` |
| 2 | `--tb=line` for eval runs | removes the failure-side quadratic | `eval_test_runner.sh` |
| 1+2 | together, graded-run mix @ 20 k | **48.3 s → 19.6 s** | |
| 3 | stub `torch.manual_seed` during collect | **58.9 s → 54.6 s** | `up_front_collect.py` (done) |
| 4 | drop reporting entirely for the warm pass | folded into #1/#2 | `eval_test_runner.sh:430` |
| 5 | skip the traceback walk for skipped cells | ≤9.6 s (verbose config) | eval `golden_tests/conftest.py:139` |
| 6 | memoize `_drives_capture` | not measured | `up_front_collect.py` |
| 7 | deselect `INVALID` cells at collection | up to ~25 s | `test_golden.py` parametrization |
| 8 | report both quadratics upstream | — | pytest `terminal.py:1140` |

### Scaling reference (measured, device-free synthetic)

Cost of `-rA` at a fixed 40,828-item suite, varying how many pass — linear, ~0.028 s per passed
test:

| passed | 6,777 | 12,248 | 18,373 | 22,986 | 40,828 |
|---|---|---|---|---|---|
| cost | 196.7 s | 552.8 s* | 519.7 s | 637.2 s | **1,094.8 s** |

\* outlier, breaks monotonicity — machine contention, not signal.

An all-passing suite is the worst case, and a skip-heavy one like ours is the mild case: at
20,000 items, ours costs 27.0 s and an all-passing suite costs 133.5 s. Our golden structure
mitigates this bug; it does not cause it.
