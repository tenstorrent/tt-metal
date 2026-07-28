---
name: llk-wave-debug
description: Diagnose LLK runtime hangs, timeouts, data mismatches, and no-progress failures from VCS FSDB waveforms using deterministic private Weka tooling. Use when an LLK test produces an .fsdb path, the user asks to inspect waves or VCS, or normal log/source debugging cannot localize a Quasar runtime failure. Do not use for compile errors or pre-ready environment failures.
---

# /llk-wave-debug — Deterministic LLK waveform diagnosis

Use the checked-in Python interface for waveform discovery and interpretation.
Do not guess RTL signal paths in a prompt or manually eyeball a waveform before
running the deterministic diagnosis.

## Usage

```text
/llk-wave-debug /path/to/failure.fsdb failure=hang
/llk-wave-debug diagnose waves from the latest Quasar CodeGen failure
```

## Inputs

- Obtain the failing `.fsdb` path from the user, `LLK_DEBUG_FSDB`, or the
  applicable tester `run.log`.
- Classify the observed failure as `hang`, `timeout`, `mismatch`, or `unknown`.
- For a CodeGen run, obtain its `LOG_DIR`, cycle, and simulator-attempt number.

Do not generate a new waveform unless the user explicitly asks for another VCS
run. Do not invoke this skill for compilation failures, simulator failures
before the device-ready marker, or confirmed environment failures.

## Security boundary

- Keep FSDBs, catalogues, traces, resolved RTL paths, and evidence bundles on
  Weka. Never add them to a tt-metal commit or PR.
- Keep architecture profiles, signal-resolution rules, and detectors in the
  private `llk_code_gen` repository. Never copy them into this skill.
- Report logical signal names and detector classifications by default. Include
  raw RTL hierarchy only when the user explicitly needs it in a private
  debugging context.

## Workflow

### 1. Check tool availability

From the `tt-llk` root, run:

```bash
python codegen/scripts/llk_debug.py --version
```

The launcher resolves `/proj_sw/user_dev/llk_code_gen` by default. Use
`LLK_CODEGEN_PRIVATE_ROOT` only when the private checkout is mounted elsewhere.
If the launcher or private checkout is unavailable, record the exact error and
continue ordinary log/source diagnosis. Waveform tooling must not block a
CodeGen run.

### 2. Choose the execution path

For an automated CodeGen tester run, use the fail-open bridge:

```bash
python codegen/scripts/optional_wave_debug.py \
  --log-dir "$LOG_DIR" \
  --cycle "$CYCLE" \
  --attempt "$ATTEMPT" \
  --failure-kind "$FAILURE_KIND" \
  --fsdb "$FSDB"
```

Omit `--fsdb` when `LLK_DEBUG_FSDB` is set or the existing
`test_logs_cycleN/run.log` names the waveform. The bridge appends to the
existing `agent_tester_cycleN.md` and writes private output under
`test_logs_cycleN/wave_debug_attemptN/`. It deliberately returns zero for
missing tooling, missing waveforms, backend errors, timeouts, and malformed
evidence. Read the appended status; do not infer success from the exit code.

For an interactive diagnosis outside a CodeGen run, write output to a private
Weka directory:

```bash
python codegen/scripts/llk_debug.py diagnose \
  --failure-kind "$FAILURE_KIND" \
  --output-dir "/proj_sw/user_dev/llk_wave_debug_runs/${CASE_NAME}-$(date -u +%Y%m%dT%H%M%SZ)" \
  "$FSDB"
```

The tool uses `LLK_DEBUG_HOST` or the existing `SSH_MACHINE_NAME` when the FSDB
backend is remote. Do not require a host argument when the current environment
already resolves the backend. Use `--insecure-host-key` only for an intentionally
ephemeral internal host.

### 3. Interpret evidence

Read `evidence.json` before proposing a cause:

- `status: findings` means one or more registered deterministic detectors
  matched. Report the highest-severity causal finding first.
- `status: inconclusive` means no registered detector matched. It is not
  evidence that the kernel is correct.
- `unavailable` or `failed` means waveform diagnosis contributed no evidence.
  Continue normal debugging without changing the tester outcome.

Cross-check each finding against the failed kernel, tester log, and capture time
range. Treat generic PC no-progress or terminal quiescence as an effect unless
the evidence establishes that it is the earliest causal boundary. Include the
`tool_source` revision and dirty/clean state in the report.

### 4. Handle an inconclusive result

Use `signals`, `query`, `summarize`, and `compare` only to answer a specific
unresolved question. Read the private command reference first:

```text
/proj_sw/user_dev/llk_code_gen/tools/llk_wave_debug/codegen/llk_debug/README.md
```

Prefer logical profiles and names over hard-coded hierarchy. When a new
invariant is proven, add its detector and synthetic unit trace to the private
repository; do not extend this skill with signal paths.

## Report

Return a concise waveform section containing:

- status and failure kind;
- primary deterministic finding, or why the result is inconclusive;
- the relevant event ordering and timestamps;
- evidence path and private tool revision;
- the next source-level check or explicit limitation.

Do not claim a root cause that is absent from `evidence.json`.
