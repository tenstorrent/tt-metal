# Handoff: reconcile triage consumers after `--json-path` → `--llm-output` switch

**Author:** rebase agent (mstaletovic/agent_eval onto llk_helper_library)
**Date:** 2026-06-02
**Branch:** `mstaletovic/agent_eval` (rebased onto `origin/llk_helper_library`)
**Owner to pick this up:** any agent comfortable in the `tt_ops_code_gen` eval system + tt-metal scripts/docs.

---

## 1. Background — what changed and the hard constraint

During the rebase of `mstaletovic/agent_eval` onto `llk_helper_library`, the personal-branch
JSON-triage feature was **dropped** so that `tools/triage/triage.py` stays **byte-identical to
`llk_helper_library`/main**. Upstream `triage.py` does **not** support `--output-format` /
`--json-path`; it exposes a different machine-readable mode instead:

| Old (dropped) personal-branch interface | Upstream interface (what we now have) |
|---|---|
| `tt-triage --output-format=json` | `tt-triage --llm-output` |
| `tt-triage --json-path=<f>` (writes JSON) | `tt-triage --llm-output-path=<f>` (writes report) |
| output: JSON `{"scripts": [...]}` | output: **CSV-formatted tables** (`CsvSerializer`) |

The two device test wrappers were updated to the upstream interface (commit
`d25b9b1a0b7 scripts: use tt-triage --llm-output for hang triage instead of --json-path`):

- `scripts/run_safe_pytest.sh`
- `scripts/tt-probe.sh`

They now run, on a dispatch-timeout hang:

```sh
python3 tools/tt-triage.py --disable-progress --skip-version-check \
    --llm-output --llm-output-path=${TRIAGE_REPORT} > ${TRIAGE_LOG} 2>&1
```

where `TRIAGE_REPORT="generated/tt-triage/triage.txt"` (was `generated/tt-triage/triage.json`).
The stdout marker also changed: `SAFE_PYTEST: JSON triage: <path>` → `SAFE_PYTEST: triage report: <path>`
(and the `TT_PROBE:` equivalent).

### ⛔ Hard constraint for this task
**Do NOT re-introduce JSON output into `tools/triage/triage.py`.** It must remain identical to
`llk_helper_library`. All fixes below adapt *consumers* to the upstream `--llm-output` /
`triage.txt` (CSV) world. If a consumer genuinely needs structured data, parse the CSV report or
the Rich/CSV table text — do not patch the tool.

### What the report now looks like
`--llm-output` uses `CsvSerializer` → **CSV-formatted tables** of the same per-core / per-RISC-V
triage data (kernel name, go message, waypoint, PC, callstack, etc.). The *field names and values*
your grep targets key on (`cb_wait_front`, `cb_reserve_back`, `noc_async_*_barrier`, `ASSERT`,
`Kernel Callstack`, `Go Message`, `DONE`/`GO`) **still appear** in the CSV report — only the
*container format* changed (CSV rows instead of a JSON object). So grep-based triage guidance
carries over; only `json.load()`-style structural parsing must change.

---

## 2. Action items

### A. FUNCTIONAL BREAK — fix first (silent hang-detection failure)

**`tt_metal/third_party/tt_ops_code_gen/scripts/hooks/kw-test-fail.sh`** (Claude Code hook in the
eval submodule).

- Line ~35: `TRIAGE_JSON="${REPO_ROOT}/generated/tt-triage/triage.json"`
- Line ~41: "Only treat the triage file as a hang signal if it was written recently …" (recency check on that path)
- Line ~53: injects `additionalContext: "HANG DETECTED…"` to the agent when the file is present+recent.

**Problem:** the wrappers now write `generated/tt-triage/triage.txt`, so this hook's
`triage.json` check **never matches** → hangs are no longer surfaced to eval agents. Silent.

**Fix:** point the hook at the new report path (`generated/tt-triage/triage.txt`), rename the var
(`TRIAGE_JSON` → `TRIAGE_REPORT`), keep the recency check. The "read the triage callstacks" guidance
in the injected context is still valid (the CSV report contains callstacks). Verify the hook still
fires (see §3).

> Note: also confirm whether this hook is referenced from a `settings.json` `hooks` block in the
> submodule's `.claude/`; if the hook is wired by filename only, the path fix is sufficient.

### B. DOCS — stale, must update (describe `--llm-output` / `triage.txt`, drop JSON)

The hang-triage section documents the old JSON flow. Update to the upstream flow. The
**`scripts/run_safe_pytest.sh` § "Hang triage"** content lives in **two byte-identical copies**
(`diff -q` is empty) — keep them in sync:

1. **`tt_metal/third_party/tt_ops_code_gen/CLAUDE.md`** — *canonical* (eval system maintains it).
2. **`.claude/CLAUDE.md`** (repo root) — an identical synced copy. Whatever propagates the
   submodule's CLAUDE.md to the repo root must be re-run, or edit both identically.

   Specific stale lines in each (line numbers ~ as of this writing):
   - `…and JSON triage on hang…` → "…and an llm-friendly triage report on hang…"
   - "writes a machine-readable **JSON** report to: `generated/tt-triage/triage.json`" →
     `generated/tt-triage/triage.txt`
   - "The last line … prints `SAFE_PYTEST: JSON triage: <path>`" → `SAFE_PYTEST: triage report: <path>`
   - "**Reading the triage JSON**: Load with `json.load()`. The top-level structure is
     `{"scripts": [...]}` …" → replace with: the report is a CSV-formatted text file; open it / grep
     it directly. **Keep the grep-targets list** (`cb_wait_front`, `cb_reserve_back`,
     `noc_async_*_barrier`, `ASSERT`, `Kernel Callstack`, `Go Message`, `Kernel Name`) — they still apply.
   - Standalone-triage examples:
     `python3 tools/tt-triage.py --output-format=json` → `python3 tools/tt-triage.py --llm-output`
     `python3 tools/tt-triage.py --json-path=out.json` → `python3 tools/tt-triage.py --llm-output --llm-output-path=out.txt`

3. **`ttnn/cpp/ttnn/kernel_lib/agents/llk_helpers_hq.md`** (~line 171): "…runs `tt-triage` and emits
   the **JSON triage report**." → "…emits the triage report." (drop "JSON").

4. **`tt_metal/third_party/tt_ops_code_gen/eval/sim_mode_design.md`** (~line 91): table cell
   "dispatch timeout fired, `tt-triage` **JSON** produced" → "…triage report produced." (minor; design doc.)

### C. CONSISTENCY — optional, decide explicitly

**`tt_metal/third_party/tt_ops_code_gen/eval/eval_test_runner.sh`** (the eval system's *own* device
runner, parallel to `run_safe_pytest.sh`).

- Line ~115 runs `python3 ${TRIAGE_SCRIPT} --disable-progress > ${TRIAGE_LOG} 2>&1` — i.e. **default
  Rich-table output**. It **never used `--json-path`, so it is NOT broken** and still works against
  upstream `triage.py`.
- **Decision needed:** for consistency with the wrappers, you may switch it to `--llm-output` so its
  triage dump is the same machine-readable CSV. **If you do**, then:
  - **`tt_metal/third_party/tt_ops_code_gen/scripts/summarize-triage.py`** — its docstring says it
    "Parses the **Rich table** output from tt-triage". If `eval_test_runner.sh` (or anything that
    feeds `summarize-triage.py`) switches to `--llm-output` (CSV), this parser must be updated to
    parse the CSV format instead of Rich tables, or it will silently produce empty/garbled summaries.
  - If you leave `eval_test_runner.sh` on default Rich output, `summarize-triage.py` stays correct —
    but then triage formatting is inconsistent between the wrappers (CSV) and the eval runner (Rich).
  - Recommendation: pick one format end-to-end. Given the wrappers are now CSV, prefer switching the
    runner + `summarize-triage.py` to CSV — but confirm with the repo owner before touching
    `summarize-triage.py` parsing.

---

## 3. How to verify

There is a purpose-built hang op on this branch:
`tests/ttnn/unit_tests/operations/intentional_hang/` (writer kernel hangs deliberately).

```sh
# from the repo root, with the python_env active
scripts/run_safe_pytest.sh --dev tests/ttnn/unit_tests/operations/intentional_hang/test_intentional_hang.py
```

Expected after the fixes:
1. The run detects a dispatch-timeout hang and prints `SAFE_PYTEST: triage report: …/triage.txt`.
2. `generated/tt-triage/triage.txt` exists and contains the CSV triage tables (grep for
   `cb_wait_front` / `Kernel Callstack`).
3. **kw-test-fail.sh fires** — the eval agent receives the "HANG DETECTED" `additionalContext`
   (this is what regresses if §2.A isn't fixed). Confirm by running under the eval pipeline (or
   invoke the hook manually with a fresh `triage.txt` present).

Sanity grep that nothing still expects the old artifact:
```sh
git grep -nE 'triage\.json|--json-path|--output-format=?json|JSON triage' \
    -- . ':(exclude)HANDOFF_triage_llm_output.md'
git -C tt_metal/third_party/tt_ops_code_gen grep -nE 'triage\.json|--json-path|--output-format' -- .
```
Both should return only intentional/unrelated hits after the work is done.

---

## 4. File checklist

- [ ] `tt_metal/third_party/tt_ops_code_gen/scripts/hooks/kw-test-fail.sh` — **functional fix** (json → txt)
- [ ] `tt_metal/third_party/tt_ops_code_gen/CLAUDE.md` — hang-triage section (canonical)
- [ ] `.claude/CLAUDE.md` — same edit (synced copy) / re-run the sync
- [ ] `ttnn/cpp/ttnn/kernel_lib/agents/llk_helpers_hq.md` — drop "JSON" (~L171)
- [ ] `tt_metal/third_party/tt_ops_code_gen/eval/sim_mode_design.md` — drop "JSON" (~L91, minor)
- [ ] *(decision)* `tt_metal/third_party/tt_ops_code_gen/eval/eval_test_runner.sh` + `scripts/summarize-triage.py` — format consistency
- [ ] Verify with `intentional_hang` op (§3)

## 5. Confirmed NON-issues (don't waste time here)
- `tt_ops_code_gen/eval/pipeline.py` `--output-format json` (~L121/L181) → that's the **`claude` CLI**, not tt-triage. Not a triage consumer.
- The many `json.loads(...)` in `eval/` (ingest.py, db.py, score.py, verify_supported.py, etc.) parse
  `test_results.json` / `test_axes.json` / score files — **unrelated** to triage.
- `eval/hang_plugin.py` + `eval/score.py` detect hangs from **pytest stdout patterns**, not from the
  triage artifact — unaffected.
- The marker rename (`JSON triage:` → `triage report:`) has no programmatic consumer found.
