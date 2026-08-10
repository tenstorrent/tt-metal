---
name: llk-wave-debug
description: >-
  Inspect VCS/FSDB waveforms to localize a Quasar LLK runtime hang, timeout, or
  mismatch when log/source debugging cannot pin the boundary. Use only after
  the simulator reaches device-ready; not for compile, pre-device-ready
  simulator, or confirmed environment failures. Quasar only.
user_invocable: true
---

# /llk-wave-debug — Deterministic LLK waveform diagnosis

Use the checked-in Python interface for waveform discovery and interpretation.
Do not guess RTL signal paths in a prompt or manually eyeball a waveform before
running the deterministic diagnosis.

## Usage

```text
/llk-wave-debug /proj_sw/user_dev/me/run/failure.fsdb failure=hang
/llk-wave-debug failure=mismatch          (reads $LLK_DEBUG_FSDB, else asks)
```

## Scope

**Quasar only.** The private tool ships signal profiles for `quasar` and
nothing else; `--arch blackhole` or `--arch wormhole` fails with
`unsupported architecture ... currently available: quasar`. For a WH/BH runtime
failure use `/debug-kernel` instead.

## Inputs

- Obtain the failing `.fsdb` path from the user or `LLK_DEBUG_FSDB`. There is no
  automatic discovery — do not go hunting through test logs for one, because a
  `run_test.sh` log will never name an FSDB (see below).
- Classify the observed failure as `hang`, `timeout`, `mismatch`, or `unknown`.
- Identify the **core under test**. Diagnosis defaults to scope
  `gen_y[1].gen_x[0]`; a failure on any other core needs an explicit `--scope`
  or every signal silently fails to resolve.

**This skill diagnoses an FSDB that already exists; it does not create one.**
Waves are a launch-time decision, so there is no way to recover them for a run
that has already finished. In particular the `emu-*` simulators that
`run_test.sh` drives for Quasar produce no FSDB at all — a waveform comes from a
VCS RTL run. If no FSDB exists, say so and continue with log/source debugging
rather than re-running anything. The tool's `capture` subcommand can launch a
wave-enabled run, but only do that when the user explicitly asks for one.

## Security boundary

- Keep FSDBs, catalogues, traces, resolved RTL paths, and evidence bundles on
  Weka. Never add them to a tt-metal commit or PR.
- Keep architecture profiles, signal-resolution rules, and detectors in the
  private `llk_code_gen` repository. Never copy them into this skill — that is
  why this file names no signal, profile, or detector.
- Report logical signal names and detector classifications by default. Include
  raw RTL hierarchy only when the user explicitly needs it in a private
  debugging context.

## Workflow

### 1. Check tool availability

From the `tt-llk` root, run:

```bash
python .claude/scripts/llk_wave_debug.py --version
```

The launcher resolves `/proj_sw/user_dev/llk_code_gen` by default. Use
`LLK_CODEGEN_PRIVATE_ROOT` only when the private checkout is mounted elsewhere;
every path in this skill is then relative to that root instead.

If the launcher or private checkout cannot be reached, record the exact error
and continue ordinary log/source diagnosis. Waveform tooling is supplementary —
it must never block or change the outcome of the work that invoked it.

`--version` proves only that the private entry point loads. Use
`python .claude/scripts/llk_wave_debug.py --help` for the current subcommand list
rather than assuming the set documented here is complete.

### 2. Run the diagnosis

Write output to a private Weka directory:

```bash
python .claude/scripts/llk_wave_debug.py diagnose \
  --failure-kind "$FAILURE_KIND" \
  --output-dir "/proj_sw/user_dev/llk_wave_debug_runs/${CASE_NAME}-$(date -u +%Y%m%dT%H%M%SZ)" \
  "$FSDB"
```

Pass `--scope` when the failing core is not the default. `diagnose` writes
`evidence.json` (for you) and `evidence.md` (for a human) into `--output-dir`,
alongside the trace, summary, catalogue, and resolution records, and echoes the
evidence to stdout.

Expect this to be slow on a large FSDB — cataloguing millions of signals over
SSH takes minutes. Raise `--backend-timeout` well above its 120 s default and
run it in the background rather than letting a foreground timeout kill it
mid-catalogue.

#### Remote FSDBs

The tool reads the FSDB over SSH when `LLK_DEBUG_HOST` or the existing
`SSH_MACHINE_NAME` is set, and locally otherwise. Do not pass a host argument
when the environment already resolves the backend. Use `--insecure-host-key`
only for an intentionally ephemeral internal host.

A remote FSDB is not visible on the local filesystem, so do not "verify" the
path with `ls` before diagnosing — a remote-mode path that fails a local stat
is expected, not an error. When a host is set, the tool also needs the Aether
workspace: `LLK_DEBUG_REMOTE_CWD` or `AETHER_WORKSPACE`, unless it can infer
one from the FSDB path.

### 3. Interpret evidence

`evidence.json` carries a `status` that is only ever one of two values:

- `status: findings` — one or more registered deterministic detectors matched.
  Report the highest-severity causal finding first.
- `status: inconclusive` — no registered detector matched. This is **not**
  evidence that the kernel is correct.

There is no third value. If the command exited non-zero there is no
`evidence.json` at all — read stderr, match it against **Common failures**
below, and continue log/source debugging. Never report a waveform conclusion
from a run that did not write evidence.

Before you report `inconclusive`, check that the quiescence detectors could
physically fire. Compare `summary.quiescent_threshold_fs` against
`summary.end_time_fs`:

```bash
python3 -c "import json,sys; s=json.load(open(sys.argv[1]))['summary']; \
print('capture', s['end_time_fs'], 'fs | threshold', s['quiescent_threshold_fs'], \
'fs | usable:', s['quiescent_threshold_fs'] < s['end_time_fs'])" evidence.json
```

`--quiescent` defaults to `1us`, and a targeted capture is often shorter than
that — a 988 ns trace can never contain a 1 µs idle gap, so every quiescence
detector is silently disabled and `inconclusive` means nothing. Re-run with
`--quiescent` set well below the capture length (a few percent of it) before
drawing any conclusion. Also compare `last_activity_fs` against
`end_time_fs`: a trace that goes idle in its first few nanoseconds and stays
idle is a capture-window problem, not a kernel finding.

Cross-check each finding against the failed kernel, test log, and capture time
range. Treat generic PC no-progress or terminal quiescence as an effect unless
the evidence establishes that it is the earliest causal boundary. Include the
`tool_source` revision and dirty/clean state in the report.

### 4. Handle an inconclusive result

Read the private command reference before reaching for a subcommand:

```bash
cat "${LLK_CODEGEN_PRIVATE_ROOT:-/proj_sw/user_dev/llk_code_gen}/tools/llk_wave_debug/codegen/llk_wave_debug/README.md"
```

It documents every subcommand, the available signal profiles, and the
recommended progression for an unfamiliar failure:

```text
inspect → signals → query/summarize → diagnose
```

Start with `inspect` — an FSDB truncated by a simulator crash or a size cap
looks exactly like a quiescent design to the detectors, and only its metadata
and time range distinguish the two. Then use `signals`, `query`, `summarize`,
`compare` (passing vs failing run), `utilization` (is the FPU doing anything at
all), or `detect` (re-run detectors on an existing `trace.json`) to answer one
specific unresolved question — not to browse.

Prefer logical profiles and names over hard-coded hierarchy. When a new
invariant is proven, add its detector and synthetic unit trace to the private
repository; do not extend this skill with signal paths.

## Common failures

| Symptom | Cause | Fix |
|---------|-------|-----|
| `LLK waveform debugger is not available` (exit 2) | Private checkout absent | Set `LLK_CODEGEN_PRIVATE_ROOT`, or continue without waves |
| `failed to run fsdbdebug: ... No such file or directory` | Verdi/FSDB toolchain not on `PATH` in local mode | Diagnose from the sim host via `LLK_DEBUG_HOST`/`SSH_MACHINE_NAME`, which sources the toolchain there |
| `no signals selected or resolved` | Wrong `--scope` for the failing core | Pass the core's actual scope; confirm with `signals` |
| `unsupported architecture ...` | Non-Quasar arch requested | Out of scope — use `/debug-kernel` |
| `selection resolved N signals, exceeding --limit` | Too broad a `--match`/profile set | Narrow the selection or raise `--limit` |
| `--remote-cwd is required when the SSH setup script is relative` | Host set, workspace not inferable | Set `LLK_DEBUG_REMOTE_CWD` or `AETHER_WORKSPACE` |
| `SSH FSDB command failed` | Host unreachable or toolchain missing there | Verify the host; check `command.stderr.log` |
| Command killed with no output on a large FSDB | `--backend-timeout` default is 120 s; cataloguing millions of vars over SSH exceeds it | Raise `--backend-timeout` (600–1200 s for a ~100 MB FSDB) and run it in the background |
| `inconclusive` with a mostly-idle trace | `--quiescent` (default `1us`) ≥ capture length, so quiescence detectors cannot fire | Re-run with `--quiescent` below the capture length — see step 3 |
| `ssh ... failed with exit code 10: *WARN* This file(...) does not exist.` | Path wrong, or right but on a different host than the backend is reading | Check the path on the backend host, not locally |

## Report

Return a concise waveform section containing:

- status and failure kind;
- primary deterministic finding, or why the result is inconclusive;
- the relevant event ordering and timestamps;
- evidence path and private tool revision;
- the next source-level check or explicit limitation.

Do not claim a root cause that is absent from `evidence.json`.
