# Agent Analysis Instructions for tt-triage CI Output

## 1. Role and Context

You are a triage analysis agent. You will receive a batch of tt-triage output files extracted from CI hang job logs. Each file represents **one triage invocation** (one "run") on one or more devices after a test hang was detected.

Your job:
1. **Classify every triage script's result** as PASS, EXPECTED, or UNEXPECTED
2. **Identify known error patterns** from the catalog below
3. **Flag any new/unknown errors** not in the catalog
4. **Return structured JSON** per file

## 2. File Structure

Each file contains sections delimited by `{script_name}.py:` headers on their own line. Below each header is the script's output: either `pass`/`fail` status, Rich-formatted tables (box-drawing characters), or error/warning messages.

**Scripts appear in this execution order** (not all may be present):

| # | Script | Type | Purpose |
|---|--------|------|---------|
| 1 | `dump_configuration.py` | Data dump | Runtime config table (always first) |
| 2 | `check_arc.py` | Checker | ARC heartbeat/uptime/postcode |
| 3 | `check_cb_inactive.py` | Checker | Circular buffer inactivity |
| 4 | `check_eth_status.py` | Checker | Ethernet link status |
| 5 | `check_noc_locations.py` | Checker | NOC endpoint reachability |
| 6 | `device_info.py` | Data dump | Hardware info table |
| 7 | `device_telemetry.py` | Data dump | Live telemetry table |
| 8 | `dump_running_operations.py` | Data dump | Operation queue |
| 9 | `check_binary_integrity.py` | Checker | Firmware/kernel binary verification |
| 10 | `check_core_magic.py` | Checker | Core firmware type verification |
| 11 | `check_noc_status.py` | Checker | NOC transaction counter mismatch |
| 12 | `dump_aggregated_callstacks.py` | Data dump | Callstacks grouped by operation |
| 13 | `dump_callstacks.py` | Data dump | Per-core callstacks |
| 14 | `dump_fast_dispatch.py` | Data dump | Dispatcher state |
| 15 | `dump_lightweight_asserts.py` | Data dump | Assert messages from firmware |
| 16 | `dump_watcher_ringbuffer.py` | Data dump | Watcher ring buffer contents |
| 17 | `firmware_versions.py` | Data dump | Firmware version info |
| 18 | `system_info.py` | Data dump | Host system info |
| 19 | `check_broken_components.py` | Checker | Broken device/core summary (runs last) |
| 20 | `dump_risc_debug_signals.py` | Data dump | Debug bus signals (conditional) |

**How to parse**: Search for lines matching `^{name}.py:$` (script name followed by colon at start of line). The section content is everything from that line until the next script header or end of file.

## 3. Classification

There are two levels of classification: **triage-level** (did triage complete?) and **script-level** (what happened to each script?).

### 3.0 Triage-Level Outcome

Classify the overall triage execution as one of:

| Outcome | Meaning | How to detect |
|---------|---------|---------------|
| **COMPLETED** | Triage ran all scripts to completion | All expected scripts have a section in the output (some may be UNEXPECTED — that's OK, triage still completed) |
| **FAILED_TO_START** | Triage crashed during init — zero scripts ran | File is `[NO TRIAGE SECTION FOUND]` or contains only a traceback with no script headers |
| **ABORTED** | Triage stopped mid-execution | Scripts in the middle of the execution order are missing while earlier scripts ran. Caused by CI timeout, host death, or unhandled exception that killed the process. Record which script it stopped at. |

**Important**: A script hitting a traceback (e.g., Errno 24) does NOT abort triage. Triage wraps each script in try/except and continues. ABORTED is only when triage **actually stops running** — the process is killed or exits.

### 3.1 Script-Level Classification

For EACH script section found in a file, classify the result as:

| Classification | Meaning | When to use |
|---|---|---|
| **PASS** | Script ran successfully, no issues found | The word `pass` appears after the header, OR a table is present with no error/warning lines |
| **EXPECTED** | Script found a legitimate diagnostic issue | The script detected something it was *designed* to detect: a real hardware/firmware problem that the hanging test caused |
| **UNEXPECTED** | Script hit any problem outside its design | Resource exhaustion, missing files, tracebacks, unsafe memory access, cascade damage from earlier scripts. Whether it produced partial output or a full traceback — any problem that isn't a legitimate diagnostic finding is UNEXPECTED. |

Scripts that are not present in the file are not classified — they are tracked at the triage level (ABORTED or conditionally disabled).

### Key principle
A script that finds a real problem (NOC mismatch, binary corruption, ARC failure) is doing its job correctly. That's EXPECTED, not a failure. UNEXPECTED covers everything else that went wrong — the drill-down report breaks down the specific error types behind each UNEXPECTED.

### 3.2 Detecting triage abort (truncation)

After parsing, check whether triage was ABORTED:

If scripts in the middle of the execution order (Section 2) are missing while earlier scripts ran, triage was killed mid-execution. Set the triage-level outcome to **ABORTED** and record which script was the last to run.

**Normal ABSENT scripts** (not an abort):
- `dump_risc_debug_signals.py` absent when no broken cores were detected → conditionally disabled
- `check_broken_components.py` may not appear if triage config disabled it

**ABORTED indicators**:
- Core scripts like `check_noc_status.py` or `dump_callstacks.py` missing while `check_arc.py` ran → triage was killed
- CI timeout message (`##[error]The action...has timed out`) in the file → E14/E24

**Tracebacks within a script section**:
- Classify the script as **UNEXPECTED** (not a separate status). Triage continues after a traceback — it does NOT abort. The traceback is just another type of unexpected outcome for that script.

## 4. Per-Script Classification Rules

### Checker Scripts

**`check_arc.py`**
| PASS | EXPECTED | UNEXPECTED |
|------|----------|------------|
| Table with normal heartbeat (9-11 hb/s), healthy uptime | Heartbeat stopped/abnormal, postcode error, clock anomaly | `unsafe access at address 0x880030060` (can't read ARC on remote WH device), any Traceback |

**`check_cb_inactive.py`**
| PASS | EXPECTED | UNEXPECTED |
|------|----------|------------|
| `pass` — circular buffers inactive | CB activity detected (potential NOC hang indicator) | `Skipping: 'tensix'` (no cores found), any Traceback |

**`check_eth_status.py`**
| PASS | EXPECTED | UNEXPECTED |
|------|----------|------------|
| `pass` — ethernet links healthy | Port down, no heartbeat, high retrain count | Any Traceback |

**`check_noc_locations.py`**
| PASS | EXPECTED | UNEXPECTED |
|------|----------|------------|
| `pass` — NOC locations valid | Wrong NOC location detected | Any Traceback |

**`check_binary_integrity.py`**
| PASS | EXPECTED | UNEXPECTED |
|------|----------|------------|
| `pass` — on-device binaries match host ELFs | `Data mismatch in section .text` (binary corruption on device) | `ELF file ... does not exist` (missing build artifact), `Skipping: {risc} is not halted` (core couldn't be inspected), `Skipping: 'tensix'`, any Traceback |

**`check_core_magic.py`**
| PASS | EXPECTED | UNEXPECTED |
|------|----------|------------|
| `pass` — core magic numbers valid | `core_magic_number mismatch` (wrong firmware type loaded) | Any Traceback |

**`check_noc_status.py`**
| PASS | EXPECTED | UNEXPECTED |
|------|----------|------------|
| `pass` — NOC transaction counters consistent | `Mismatched state:` followed by NOC counter data (NOC is hung — the most common diagnostic finding) | `Cannot find global variable noc_mode in ELF DWARF` (DWARF info gap), `Skipping: {risc} is not halted`, `Skipping: 'tensix'`/`'active_eth'`/`'idle_eth'`, `unsafe access at address`, any Traceback |

**`check_broken_components.py`**
| PASS | EXPECTED | UNEXPECTED |
|------|----------|------------|
| No broken components listed | Lists broken cores/devices — either broken by the hang OR broken by triage (`Was halted by triage but is no longer halted`) | Traceback or unhandled exception |

> **Note on "halted by triage but no longer halted" (E04)**
>
> This message means triage halted a core for inspection and it did not come back up afterwards. This is a **known limitation of the halt/resume hardware mechanism** — triage cannot reliably re-start every core it halts, and there is currently no known software fix. The `check_broken_components` script is *designed* to detect and report this condition, so from every perspective it should be treated as an **EXPECTED** outcome for the script: the script is working correctly and is telling us about a real hardware limitation.
>
> E04 is still recorded in `known_patterns_found` so we can track its rate over time, but its category is **`environment`**, not `triage_bug`. Do not include it in "Actionable Triage Bugs" recommendations — there is nothing to fix in triage.

### Data Dump Scripts

For data dump scripts, there is generally no "EXPECTED failure" — they either produce their table (PASS) or hit an error (UNEXPECTED). The exception is `dump_running_operations.py`: when the op table surfaces a known hang signature (see E25 below), classify as EXPECTED — same convention as checker scripts that detect recognized diagnostic patterns.

**`dump_configuration.py`**
| PASS | UNEXPECTED |
|------|------------|
| Configuration table present | Traceback, missing inspector data |

**`device_info.py`**
| PASS | UNEXPECTED |
|------|------------|
| Device info table present | `unsafe access at address 0x880030060` in Postcode column, Traceback |

**`device_telemetry.py`**
| PASS | UNEXPECTED |
|------|------------|
| Telemetry table present | Traceback |

**`dump_running_operations.py`**
| PASS | EXPECTED | UNEXPECTED |
|------|----------|------------|
| Operations table present with Op Name and Params populated for current op (Previous Op Name may be N/A — that's fine) | (a) Current Op Name is `N/A` **AND fast dispatch is disabled** — op tracking not supported under slow dispatch yet. (b) `Prev Op Name` contains `Matmul` (case-insensitive) — **E25 likely di/dt**, a known hang signature surfaced by the table. | Current Op Name is `N/A` or current Params is `N/A` **while fast dispatch is enabled** (metadata resolution failed), any Traceback |

> **How to determine fast-dispatch mode**: check the `dump_configuration.py` section for the rtOption row `fast_dispatch`. Fast dispatch is enabled when `fast_dispatch │ true` appears; it is disabled when `fast_dispatch │ false` appears (or when the `TT_METAL_SLOW_DISPATCH_MODE` env var is set to a truthy value). If fast dispatch is disabled, an `N/A` current op is expected — we don't currently resolve op metadata under slow dispatch.

> **E25 — MatMul preceded hang (likely di/dt)**: when any row's `Prev Op Name` column contains `Matmul` (case-insensitive), classify the script as **EXPECTED** and emit an `E25` entry in `known_patterns_found`. Rationale: a large MatMul ends → current drops sharply → next op hits a destabilized power rail and hangs. We don't have a software fix for this, but triage's op table surfaces the signature cleanly, so we treat it like other known-and-recognized diagnostic findings (E04, E11, E16).
>
> When emitting E25, include a `details` field that records:
> 1. The `TT_MM_THROTTLE_PERF` env var from `dump_configuration.py` (value or `unset` if the row is absent).
> 2. The `Op Name` of the **current** (hung) op for context (not the MatMul).
>
> Example: `{"pattern_id": "E25", "count": 1, "script": "dump_running_operations.py", "details": "throttle=unset, current_op=AllGatherAsyncDeviceOperation"}`. When `TT_MM_THROTTLE_PERF` is **set** but the hang still happened, that's a stronger signal — firmware-level mitigation didn't prevent the hang, so the throttle level may be insufficient.
>
> If both (a) fast-dispatch N/A and (b) E25 apply to the same file, prefer E25 as the EXPECTED reason — it's more specific. Always emit the E25 pattern entry either way.

**`dump_aggregated_callstacks.py`**
| PASS | UNEXPECTED |
|------|------------|
| Callstack table present | `[Errno 24] Too many open files`, `fabric_erisc_router ... does not exist`, `Core is in reset`, `PC was not in range of any provided ELF files` on **non-erisc** cores (tensix brisc/trisc/ncrisc), any Traceback |

Note: `PC was not in range of any provided ELF files` on **erisc** cores is **informational** (PASS) — erisc can context-switch to base firmware, putting the PC outside user ELF range. On tensix/worker cores this should not happen and indicates a tooling or ELF resolution issue.

Note: `Kernel Name: N/A` with `Op Id: 0` is **informational** when the core is idle or in firmware. It is UNEXPECTED if the core is clearly running a user kernel (check if the callstack shows a user function, not just `main()`).

**`dump_callstacks.py`**
| PASS | UNEXPECTED |
|------|------------|
| Callstack table present | `[Errno 24] Too many open files`, `fabric_erisc_router ... does not exist`, `Failed to halt {risc} core`, `Core is in reset`, `PC was not in range of any provided ELF files` on **non-erisc** cores, any Traceback |

Note: `PC was not in range of any provided ELF files` on **erisc** cores is informational (PASS). On tensix/worker cores it is UNEXPECTED.

**`dump_fast_dispatch.py`**
| PASS | UNEXPECTED |
|------|------------|
| Dispatch state table present | `Failed to halt {risc} core`, `Failed to read symbol`, any Traceback |

**`dump_lightweight_asserts.py`**
| PASS | UNEXPECTED |
|------|------------|
| `pass` or assert data present | `[Errno 24]`, `ELF file ... does not exist`, any Traceback |

**`dump_watcher_ringbuffer.py`**
| PASS | UNEXPECTED |
|------|------------|
| Table or `pass` | Any Traceback |

**`firmware_versions.py`**
| PASS | UNEXPECTED |
|------|------------|
| Version table present | Any Traceback |

**`system_info.py`**
| PASS | UNEXPECTED |
|------|------------|
| System info table present | `[Errno 24] Too many open files: '/etc/os-release'` (FD exhaustion cascade), any Traceback |

**`dump_risc_debug_signals.py`**
| PASS | UNEXPECTED |
|------|------------|
| Data written or `pass` | Any Traceback |

## 4B. Triage Init Failures (No Script Output)

Some files will NOT start with `dump_configuration.py:`. Instead they contain a traceback or error from triage's initialization phase — before any scripts could run. This happens when:

- **exalens fails to connect** to the device (UMD driver crash, device inaccessible)
- **Inspector RPC is unavailable** (no serialized data, log directory missing)
- **FW init failure** on retry (device reset failed after previous hang)
- **Python import error** (missing ttexalens package, version mismatch)
- **ARC core failure** during exalens startup

### How to detect

If a file does NOT contain any `{script_name}.py:` header lines (no `dump_configuration.py:`, no `check_arc.py:`, etc.), it is an init failure. Also check for files containing only `[NO TRIAGE SECTION FOUND]`.

### How to classify

- Set ALL scripts to `"ABSENT"` in the output
- Add a top-level field `"init_failure": true` to the JSON output
- Add a field `"init_failure_reason"` with the error text (first 500 chars)
- Match against init-specific patterns E19-E22 below

### Common init failure messages

| Pattern | Example text |
|---------|-------------|
| exalens/UMD crash | `RuntimeError`, `Segmentation fault`, `core dumped` before any script output |
| Inspector RPC unavailable | `There is no Inspector RPC data, cannot continue` |
| FW init failure on retry | `Device N init: failed to initialize FW! Try resetting the board` |
| Module not found | `Module 'tt-exalens' not found`, `ModuleNotFoundError` |
| Device broken by previous triage | `Triage broke device with:` |

## 5. Known Error Pattern Catalog

When you encounter an error, check against these 24 known patterns. Record the pattern ID, count, and affected script.

| ID | Name | Regex | Category | Affected Script(s) | is_triage_bug |
|---|---|---|---|---|---|
| E01 | FD Exhaustion (Errno 24) | `\[Errno 24\] Too many open files` | triage_bug | dump_callstacks, dump_aggregated_callstacks, system_info | true |
| E02 | FD Exhaustion Crashes system_info | `Too many open files.*os-release` | triage_bug | system_info.py | true |
| E03 | Missing Fabric ERISC Router ELF | `fabric_erisc_router.*does not exist` | triage_bug | dump_callstacks, dump_aggregated_callstacks, dump_lightweight_asserts | true |
| E04 | Cores Broken During Triage (known HW halt/resume limitation) | `Was halted by triage but is no longer halted` | environment | check_broken_components.py | false |
| E05 | Unsafe ARC Memory Access | `unsafe access at address 0x880030060` | environment | check_arc.py, device_info.py | false |
| E06 | Missing noc_mode DWARF Variable | `Cannot find global variable noc_mode in ELF DWARF` | environment | check_noc_status.py | false |
| E07 | ttexalens SyntaxWarning | `SyntaxWarning: invalid escape sequence` | triage_bug | ttexalens (import) | true |
| E08 | Core Not Halted (skip) | `Skipping: (brisc\|erisc\d?) is not halted` | environment | check_noc_status.py, check_binary_integrity.py | false |
| E09 | Failed to Halt Core | `Failed to halt (brisc\|erisc) core at` | environment | check_noc_status.py, dump_fast_dispatch.py, dump_callstacks.py | false |
| E10 | No Cores Available | `Skipping: '(tensix\|active_eth\|idle_eth)'` | environment | multiple | false |
| E11 | Binary Integrity Mismatch | `Data mismatch in section \.text` | diagnostic | check_binary_integrity.py | false |
| E12 | PC Not in ELF Range | `PC was not in range of any provided ELF files` | informational | dump_callstacks.py | false |
| E13 | Core Is In Reset | `Core is in reset` | environment | dump_callstacks.py | false |
| E14 | Test Action Timeout | `##\[error\]The action.*has timed out after \d+ minutes` | environment | CI infrastructure (not triage) | false |
| E15 | No Triage Section | `^\[NO TRIAGE SECTION FOUND\]$` | environment | N/A | false |
| E16 | NOC Transaction Mismatch | `Mismatched state:.*NOC\d` | diagnostic | check_noc_status.py | false |
| E17 | Unknown Motherboard Warning | `Unknown motherboard` | environment | Metal runtime (not triage) | false |
| E18 | N/A Kernel Name in Callstacks | (Kernel Name column shows N/A with Op Id 0) | informational | dump_aggregated_callstacks.py | false |
| E19 | Triage Init: Inspector RPC Unavailable | `There is no Inspector RPC data, cannot continue\|Log directory.*does not exist` | init_failure | triage init (pre-script) | true |
| E20 | Triage Init: FW Init Failure | `failed to initialize FW\|Device.*init.*failed` | init_failure | triage init (pre-script) | false |
| E21 | Triage Init: Module Not Found | `Module.*not found\|ModuleNotFoundError\|tt-exalens not found` | init_failure | triage init (pre-script) | true |
| E22 | Triage Init: Device Broken by Previous Triage | `Triage broke device with:` | init_failure | triage init (pre-script) | true |
| E23 | Triage Mid-Script Crash | `Traceback \(most recent call last\)` in last script section with subsequent scripts ABSENT | triage_bug | last script before crash | true |
| E24 | Triage Output Truncated | File ends abruptly mid-script (incomplete table, no status line, no subsequent scripts) | environment | last script in file | false |
| E25 | Likely di/dt — MatMul preceded hang | `Prev Op Name` column in `dump_running_operations.py` contains `Matmul` (case-insensitive) | diagnostic | dump_running_operations.py | false |

### Category definitions
- **init_failure**: Triage crashed during initialization before producing any script output. The most severe failure — zero diagnostic data is produced.
- **triage_bug**: A bug or limitation in the triage tool itself. Actionable — should be fixed.
- **environment**: A device/hardware/config limitation. Not a triage bug, but may degrade diagnostic quality.
- **diagnostic**: A legitimate finding — the script correctly detected a real issue. This is triage working as designed.
- **informational**: Expected edge case, not an error. Noted for completeness.

## 6. New Error Detection

After checking all 24 known patterns (E01-E24), scan for ANY of these indicators that do NOT match a known pattern:

1. Lines containing `Traceback` followed by exception text
2. Lines containing `Error:` or `Exception:` (outside of table data)
3. Script status lines showing `fail` where the failure text doesn't match any known pattern
4. Lines containing `Warning:` that are not E07 (SyntaxWarning)
5. Lines containing `Skipping:` that are not E08, E10
6. Lines containing `does not exist` that are not E03
7. Lines containing `unsafe access` that are not E05
8. Any `OSError`, `RuntimeError`, `ValueError` that are not E01/E02

For each new error found, record:
- The script it appeared in
- The full error text (first 500 characters)
- A suggested human-readable pattern name
- A suggested regex that would match this pattern

## 7. Per-Test Correlation

Each file has metadata including `test_function`. After analyzing all files in your batch:
- Group your findings by `test_function`
- Note which tests consistently cause specific UNEXPECTED results
- Note which tests consistently trigger specific known patterns

## 8. Output Schema

Return a JSON array. One entry per file:

```json
[
  {
    "file_key": "70203412821_run1",
    "job_id": "70203412821",
    "run_number": 1,
    "total_runs_in_job": 21,
    "test_function": "test_demo_text",
    "host_name": "tt-metal-ci-vm-bh-gh-08",
    "arch": "blackhole",
    "init_failure": false,
    "init_failure_reason": null,
    "scripts": {
      "dump_configuration.py": {
        "status": "PASS",
        "details": null
      },
      "check_arc.py": {
        "status": "PASS",
        "details": null
      },
      "check_noc_status.py": {
        "status": "EXPECTED",
        "details": "5 mismatched state findings on erisc1 NOC0"
      },
      "dump_callstacks.py": {
        "status": "UNEXPECTED",
        "details": "Errno 24 — 47 cores failed to dump callstacks"
      },
      "system_info.py": {
        "status": "UNEXPECTED",
        "details": "Errno 24 — OSError /etc/os-release"
      }
    },
    "known_patterns_found": [
      {"pattern_id": "E01", "count": 47, "script": "dump_callstacks.py"},
      {"pattern_id": "E02", "count": 1, "script": "system_info.py"},
      {"pattern_id": "E16", "count": 5, "script": "check_noc_status.py"}
    ],
    "new_errors": [
      {
        "script": "some_script.py",
        "error_text": "Never-before-seen error message...",
        "suggested_name": "Descriptive Pattern Name",
        "suggested_regex": "regex_here"
      }
    ]
  }
]
```

### Field requirements
- `scripts`: Include ALL 20 scripts from Section 2. For each:
  - `status`: one of `"PASS"`, `"EXPECTED"`, `"UNEXPECTED"`
  - `details`: null for PASS, description for others
  - `absent_reason`: (only for ABSENT) — one of: `"not triggered"` (normal, e.g. dump_risc_debug_signals when no broken cores), `"triage crashed during {script}"`, `"output truncated"`, `null` (unknown)
- `init_failure`: true if no script sections found at all (see Section 4B)
- `known_patterns_found`: Only include patterns actually detected (count > 0).
- `new_errors`: Only include genuinely new errors not covered by E01-E24.
- `arch`: Detect from the first 10KB of the file — look for "blackhole" or "wormhole" (case-insensitive).

## 9. Analysis Approach

For each file in your batch:

1. **Grep first**: Search the file for key patterns: `fail`, `error`, `warning`, `skip`, `broken`, `traceback`, `exception`, `does not exist`, `timeout`, `Errno`, `Skipping:`, `Unsafe`, `Mismatched state`
2. **Read the first 50 lines** to identify which scripts ran and the architecture
3. **Read sections around each match** to get context
4. **Classify each script** using the rules in Section 4
5. **Match known patterns** using the catalog in Section 5
6. **Flag new errors** using the rules in Section 6
7. **Build the JSON output**

Keep your response under 4000 words. Focus on structured JSON output, not narrative.
