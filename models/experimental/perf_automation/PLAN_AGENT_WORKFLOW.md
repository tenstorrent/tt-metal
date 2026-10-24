# PLAN — Agentic Performance-Optimization Workflow

> **Purpose.** Automate transformer-model performance optimization on Tenstorrent hardware
> (Wormhole / Blackhole, TT-NN / Metal) as an **evaluator–optimizer loop**: a lead agent
> proposes optimization levers from a tagged playbook; a deterministic profiler + PCC gate
> scores them; the loop keeps wins, reverts losses, and repeats until a target metric is met
> or a budget/iteration cap is hit.
>
> **Audience.** This document is the build plan for a **coding agent** working with the
> **Claude Agent SDK**. The agent builds the system **test-first (TDD red/green)**: for every
> task, write the failing test first (🔴 RED), implement the minimum to pass (🟢 GREEN), then
> refactor. Do not write implementation before its test exists.
>
> **Canonical copy:** this file on `cust-models-02` — edit HERE. Any copy elsewhere is a
> stale snapshot.
>
> **Status legend.** ✅ known / specified · 🟡 partially known · 🔴 **TBD** — leave a stub and a
> `# TBD(<id>)` marker; the human will fill these (see §11 TBD Register).

---

## STATUS & CHECKLIST — updated 2026-06-11

> **Active branch `gtobarTT/perf_automation_loop`** carries the Agent Loop build below.
> The loop now runs ROUTE → SELECT → **PLAN** → APPLY → VERIFY → GATE_PCC → REMEASURE → DECIDE
> end-to-end; suite 170 passed. See "Loop branch build log" for what's real vs injected.

### ✅ Done (all verified on cust-models-02, real hardware; suite: 99 passed)

- **M0 scaffold** — `.env.agent` sole-source loader (§3.1, fail-fast, never logged),
model roles (`get_model`), SDK env wiring incl. `ANTHROPIC_SMALL_FAST_MODEL`
- **M1 infra** — `atomic_write`, `Checkpoint` (WAL intent/done), `Ledger` (append-only,
idempotent, truncated-final-line tolerant), `Run`/`Manifest` (write-once, second-granular
ids), `check_exit` (metric-direction-aware: min/max)
- **M2 routing** — `build_index`/`route` (closed-vocabulary validation: unknown dim or
value raises), `read_section`, `cache_playbook` (content-hash), coverage lint over the
parser's emittable classes; 52 GUIDELINES sections route-tagged
- **M3 stage-1 tools** — `OP_CLASS_MAP`, `environment_check`, 3-stage `tracy_tool`
(RUN → real tt-perf-report REFINE → TAG+BUCKET, **µs units verified against raw ns**),
`read_model_files` (gatherer-only, evidence notes, fatal/warning flags)
- **Integration / Before Loop** — `before_loop` driver, 6 banner stages + `events.jsonl`,
real boundaries: `tt-smi -s` probe, SDK discovery sub-agent, **lead review gate**
(gather/approve split), preflight collect (hard-fail on 0 selected), real tracy stage-1
(no-pipes streaming log, process-group kill, `-o` directed output + regex cross-check +
watermark fallback)
- **UX decisions (user-set)** — folder+metric CLI (no freeform prompt); `--input 128` /
`--input 128x128` matcher (no match = HARD STOP; ambiguous = HARD STOP; `-k` expert-only);
`--devices single|all|ids` (sets `**TT_VISIBLE_DEVICES`** — the UMD var; `TT_METAL_*`
alone does NOT gate fabric topology); default case = FIRST collected, loudly logged;
default metric = `**device_ms**` (profiled kernel time; wall kept as reference)
- **Real baseline** — bge_m3 S128 single-chip: device 12.1 ms, 7 tagged buckets
(run `2026-06-10T16-31-51`; matmuls HiFi2 → fidelity walk available, reductions grid=tiny)

### ⬜ Remaining (build order)

- **M4 engine** — ✅ DONE. `engine.py` (dumb dispatcher, `stop_after`/`--until`, cycle+unknown
guards), `states.py` (names + `TRANSITIONS` + repair budgets), `loop_context.py` (the `ctx`
seam: state/manifest/ledger + `record_agent_call` + `current_profile`/`model_root`), `loop.py`
entry, `looplog.py` (one terse line per stage). Resume-from-midstate covered.
- **M5 agent edge** — ✅ DONE (real handlers, injectable agent/hardware leaves):
  - **ROUTE** real — top bucket → `route()` → writes `route_brief_<N>.json` (candidates + playbook text
    + **op-class-filtered model map**, the ast skeleton from `model_map.py`). Routes on the
    **current** profile, not the frozen baseline.
  - **SELECT** real — lead picks one untried lever from the brief; enum-constrained; fallback to
    `untried[0]`.
  - **PLAN** real (NEW stage, §8.4.5) — lead reads brief+map, reads the target file (Read/Grep),
    emits a localized edit spec `{file, location, change}`; **NOOP** skips already-applied levers.
  - **APPLY** real — records clean SHA, editor applies the **spec**; git-diff ground truth
    (scoped to the model dir) so editor mis-reports/errors never crash or misattribute.
  - **VERIFY** real — **syntax-only** (`ast.parse`); the standalone import check was dropped
    (false-positive on package imports — GATE_PCC catches runtime errors).
  - **REPAIR_CODE / REPAIR_PCC** — ⬜ still MOCK (registry uses `mocks.*`); real self-heal
    (re-edit with the error, ≤5/≤2) is the next build.
  - editor on a dedicated **`edit` model role** (inherits the SUB/haiku tier from `.env`; it transcribes the lead's localized PLAN spec, so it needs no lead-class reasoning);
    **prompt logging** persists every agentic call's prompt+response to `runs/<id>/prompts/`.
- **M6 gates** — partial: **GATE_PCC** real (single-stage e2e verdict ok/pcc_low/crash; `parse_pcc`
verified live = 0.9999), **REMEASURE** real (median device_ms + run spread + iter profile;
tracy injectable), **DECIDE** real (keep/discard; noise floor still the deferred placeholder).
⬜ remaining: COMMIT/REVERT real git (**path-scoped**), and the REPAIR handlers above.
- **M7 memory** — ledger `hypothesis` rows, dashboard, resume brief, lever-eval over the
ledger (per-tag win rate: did `lever -> Δdevice_ms` match the hypothesis?)
- **M8 go-real residue** — fps / tok_s metric sources; multi-chip (`--devices all`) once
the box fabric issue is fixed
- **Open TBDs** — regime-source, count-thresh, noise-N/floor, pcc-thresholds,
git-branch-policy, brief-K, model-lead default
- **Box issue (not ours)** — with all 8 chips visible, fabric auto-discovery degrades to a
2x1 mesh and chip 0 leaves the control plane (regressed 2026-06-10 ~14:30–16:00; survives
`tt-smi -r`). Single-chip unaffected via `TT_VISIBLE_DEVICES`.

### 🧭 Onboarding (read this if you're joining)

1. **Canonical copies live HERE** (`cust-models-02:/localdev/gtobar/tt-metal/models/experimental/
  perf_automation/`). Local clones elsewhere are stale snapshots. Branch:`  gtobarTT/perf_automation`; commit style` [perf_automation] ...`.
2. **Read `progress.txt` end to end first** — it is the append-only coordination ledger between
  the human, the lead agent, and building agents. NEVER edit past entries (standing rule);
   corrections get a new dated entry. Reviews and directives land there.
3. **Setup**: activate your own tt-metal python environment (build/env per the tt-metal
  docs; the harness assumes `import ttnn` works and `tt-smi`/`tt-perf-report` are on PATH).
   The ONLY project-specific environment is `.env.agent` (§3.1) — required keys
   `LITELLM_BASE_URL`, `LITELLM_API_KEY`; NEVER commit it (no .gitignore here by choice).
4. **Verify**: `python -m pytest -q` → 99 passed. Then a mock pipeline run:
  `python -m agent.before_loop agent --mock-env --mock-model-files --mock-tracy`.
5. **Real run**: `python -m agent.before_loop <model_root> --input 128 [--target N]`.
  Artifacts land in `runs/<id>/` (manifest = audit record, state = checkpoint,
   events.jsonl = stage log, profiles/ = CSVs + tracy log).
6. **Design rules you must not break** (§3, §4): deterministic core / agentic edge (no LLM
  inside tools); sub-agents GATHER, the lead APPROVES; code validates form, agents validate
   meaning; closed tag vocabulary (§4.1, extend the registry before inventing values);
   no `priority` metadata — candidate ordering is agent judgment; pytest machine parsing
   always uses `-o addopts=`.

---

## 0. How to use this plan (for the building agent)

1. Work milestone by milestone (§6 build order). Do not skip ahead — later milestones assume
  earlier ones are green.
2. **TDD discipline, every task:**
  - 🔴 **RED** — write the named test asserting the behavior. Run it. Confirm it *fails for the
   right reason* (not an import error).
  - 🟢 **GREEN** — write the minimum implementation to make that test (and all prior tests) pass.
  - ♻️ **REFACTOR** — clean up with the tests still green.
3. **Deterministic core, agentic edge.** Tools are deterministic Python — *no LLM inside a tool*.
  The agent only makes decisions (which lever, when to stop reasoning). Sub-agents exist only to
   buy an **isolated context window** for a focused job (edit a file, judge a diff).
4. **Mock before real.** Every hardware-touching tool (Tracy, PCC, the model build) gets a
  **mock** implementation first so the whole loop is testable without a device. Swapping mock →
   real is the final milestone and changes no control logic.
5. When you hit a 🔴 TBD, implement against the **documented interface** with a mock/stub, add a
  `# TBD(<id>)` comment, and keep going. Never block on a TBD.
6. Never print secrets. Credentials come **only** from the local `.env.agent` file (§3.1);
  never hardcode, never log, never fall back to the shell environment.

---

## 1. Architecture overview

Two stages. **Stage 1 (Before Loop)** runs once to prime context. **Stage 2 (Agent Loop)** is the
evaluator–optimizer loop, driven by an explicit **state machine** with a durable checkpoint so it
can resume after a break or crash.

```
╔══════════════════ STAGE 1 · BEFORE LOOP (run once) ═══════════════════╗
   environment_check ─▶ cache_playbook ─▶ read_model_files ─▶ baseline_profile
   (HW facts)          (playbook index)   (file map, PCC      (Tracy median-of-3
                                            paths)              → baseline_ms)
   ── writes ──▶ runs/<id>/manifest.json (+ .cache/playbook_index.json)
╚════════════════════════════════════════════════════════════════════════╝
                                   │
╔══════════════════ STAGE 2 · AGENT LOOP (state machine) ═══════════════╗
                                   ▼
   ROUTE ─▶ SELECT ─▶ PLAN ─▶ APPLY ─▶ VERIFY ─▶ GATE_PCC ─▶ REMEASURE ─▶ DECIDE ─┐
   (route   (agent    (one      (e2e PCC     (Tracy       (keep/  │
    code)    picks)    edit)     vs thr)      median-3)    discard/│
     ▲                                                             │
     │                                              ┌──────────────┤
     │                              COMMIT ◀─ keep ─┤              │
     │                              REVERT ◀─ discard ─────────────┘
     │                                 │
     └──────── CHECK_EXIT ◀── LOG ◀─────┘
                   │
                   ▼
        DONE  /  STOPPED  /  FAILED        (terminal states)
╚════════════════════════════════════════════════════════════════════════╝
```

**Inner repair loop (§8.5.1–8.5.2):** between APPLY and REMEASURE the flow can cycle back to
REPAIR. `VERIFY` (parse+import) and `GATE_PCC` (e2e PCC) each return a verdict; the Engine routes
a **parse/import/run crash → REPAIR (≤5 `code_fix_attempts`)** and a **PCC-below-threshold →
REPAIR (≤2 `pcc_fix_attempts`)**, re-VERIFYing after every repair. Attempts exhausted → revert +
mark the lever `tried` (reason `edit_failed` / `pcc_failed`) + LOG, then back to SELECT.

**State semantics**


| State                | Side effect?          | Idempotency key / resume rule                                                   |
| -------------------- | --------------------- | ------------------------------------------------------------------------------- |
| `PRECHECK`           | writes manifest       | skip if `manifest.json` present                                                 |
| `PROFILE_BASELINE`   | read-only             | re-run; record `baseline_ms`                                                    |
| `ROUTE`              | none (pure fn)        | re-run, identical output                                                        |
| `SELECT` (agent)     | none                  | record `current_lever`; resume → APPLY (don't re-ask)                           |
| `APPLY` (one edit)   | mutates tree          | record `git_sha_clean` first; resume = `git reset --hard <clean>` then re-apply |
| `VERIFY`             | read-only             | re-run on the on-disk edit (`ast.parse` + import); returns `ok`/`parse`/`import` |
| `REPAIR` (agent)     | mutates tree          | edits on current tree from captured error; resume = re-VERIFY; counters in state |
| `GATE_PCC`           | read-only (device)    | single-stage e2e PCC vs threshold; returns `ok`/`crash`/`pcc_low`               |
| `REMEASURE`          | read-only (expensive) | re-run all N; no partial checkpoint                                             |
| `DECIDE`             | none (pure fn)        | re-run                                                                          |
| `COMMIT` / `REVERT`  | git                   | guard: no-op if HEAD already at target SHA                                      |
| `LOG`                | append ledger         | `experiment_id` key — skip if row already present                               |
| `CHECK_EXIT`         | none                  | reads checkpoint counters → continue / DONE / STOPPED                           |


**Terminal states:** `DONE` (target met) · `STOPPED` (budget / max-iter / no-untried-levers floor)
· `FAILED` (clean state unrecoverable — even baseline won't build/profile).

---

## 2. File & state layout

One self-contained directory per run. The **checkpoint is the single entry point**; everything
else is referenced by relative path. Four lifecycles, never merged:

```
.cache/
  playbook_index.json        # DERIVED CACHE — content-hash of playbook/, shared across runs

runs/
  2026-06-09T14-22/          # one run = one place to look / archive / delete
    manifest.json            # IMMUTABLE: env (card/grid/bw) + model file map + config
                             #            (target_ms, max_iter, budget_usd, baseline cmd)
    state.json               # MUTABLE checkpoint (atomic write, WAL) — the one live file
    ledger.jsonl             # APPEND-ONLY: one row per experiment; carries `hypothesis`/`note`
    events.jsonl             # APPEND-ONLY: stage spans {ts, stage, status, detail, iteration}
    agent_calls.jsonl        # APPEND-ONLY: one row per query() {ts, iteration, stage, role,
                             #   model, tokens_in/out, cost_usd, latency_s, prompt_sha,
                             #   response} — the input CHECK_EXIT's budget gate is missing
    dashboard.html           # rendered from ledger.jsonl
    profiles/
      baseline_profile.json  # stage-1 baseline buckets — the FIXED reference
      iter_03_profile.json   # re-bucketed profile after iter 3's committed edit
                             #   -> ROUTE reads the latest of these (current_profile)
      run0_raw.csv           # raw Tracy + tt-perf-report CSVs (evidence)
  latest -> 2026-06-09T14-22 # symlink: resume = read runs/latest/state.json
```

**Write patterns (enforced by tests):**

- `state.json` → **atomic**: write `*.tmp`, `fsync`, `os.replace()`. WAL ordering: checkpoint
*intent* → do side effect → checkpoint *done*.
- `ledger.jsonl` → **append one line**; never rewrite. A crash truncates at most the last line.
- `manifest.json` → **write once** at PRECHECK; read-only thereafter.
- `playbook_index.json` → **hash-keyed**; rebuilt automatically when `playbook/` changes; never
hand-managed; not part of crash recovery.

**state.json schema (control only — narrative lives in the ledger):**

```json
{
  "run_id": "2026-06-09T14-22",
  "state": "GATE_PCC",
  "iteration": 4,
  "metric": {
    "name": "device_ms",
    "unit": "ms",
    "direction": "min",
    "baseline": 20.0,
    "current": 14.2,
    "target": 12.0
  },
  "max_iter": 25,
  "budget_usd": 5.0,
  "cost_usd": 0.83,
  "git_sha_clean": "abc123",
  "current_bucket": "ff1-matmul",
  "current_profile": "profiles/iter_03_profile.json",
  "candidates": ["mlp-fidelity-walk", "mlp-subblock"],
  "tried": ["mlp-subblock"],
  "current_lever": "mlp-fidelity-walk",
  "crash_retries": 0,
  "code_fix_attempts": 0,
  "pcc_fix_attempts": 0,
  "last_error": null
}
```

**ledger.jsonl row schema (facts + interpretation):**

```json
{"experiment_id": "2026-06-09T14-22#4", "iter": 4, "lever": "mlp-fidelity-walk",
 "bucket": "ff1-matmul", "before_ms": 18.0, "after_ms": 14.0,
 "pcc_single": 0.999, "pcc_full": 0.998, "status": "keep", "git_sha": "def456",
 "hypothesis": "Compute side mostly harvested; remaining time is the QKV reshard. Next: attack data-movement, not matmul."}
```

- `status` ∈ `{keep, discard, crash, baseline}`.
- `hypothesis` is a **forward-looking snapshot**; "current hypothesis" = the last non-null
`hypothesis` in the file (no separate field in state.json). Write only when the thinking shifts.

---

## 3. Claude Agent SDK building blocks

Use `claude-agent-sdk` (already pinned in the POC `.venv`). Key pieces and where each is used:


| SDK piece                                                                          | Use it for                                                                                                                                      |
| ---------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| `@tool(name, desc, schema)`                                                        | every deterministic tool (route, read_section, tracy, pcc, git, log)                                                                            |
| `create_sdk_mcp_server(name, ver, [tools])`                                        | bundle tools into an in-process MCP server (surface as `mcp__<server>__<tool>`)                                                                 |
| `ClaudeAgentOptions(...)`                                                          | per-agent config: `model`, `system_prompt`, `mcp_servers`, `allowed_tools`, `permission_mode`, `setting_sources`, `max_turns`, `max_budget_usd` |
| `query(prompt, options)`                                                           | one bounded agent call (the SELECT step, and each sub-agent job)                                                                                |
| `ClaudeSDKClient(...)`, `resume=` / `continue_conversation=`                       | same-session continuation only — **not** used for multi-day warm start (see §9)                                                                 |
| Sub-agent = a separate `query()` with a **narrow toolset + focused system prompt** | context isolation for `edit_file` and `pcc_judge`                                                                                               |
| `max_turns` (inner loop) vs Python `for` (outer loop)                              | `max_turns` bounds model turns per iteration; the run cap is your own loop counter                                                              |
| `max_budget_usd`                                                                   | hard per-call cost cap; degrade gracefully on hit                                                                                               |


### 3.1 Credentials — `.env.agent` is the ONLY credential source

The workflow talks to models through the LiteLLM proxy. Credentials come from a local
`**.env.agent`** file at the `perf_automation/` root — **and from nowhere else**: not ambient
shell env, not hardcoded values, not CI secrets injected some other way. One file, one source
of truth, auditable.

```bash
# .env.agent  (gitignored — NEVER commit)
LITELLM_BASE_URL=https://<your-litellm-proxy>
LITELLM_API_KEY=sk-...
# optional overrides:
# AGENT_MODEL_LEAD=...        # lead agent (default TBD(model-lead))
# AGENT_MODEL_SUB=anthropic/claude-sonnet-4-6
```

Rules (enforced by code + tests, not convention):

1. **Load only from `.env.agent`** at startup; map to `ANTHROPIC_BASE_URL` /
  `ANTHROPIC_AUTH_TOKEN` / `ANTHROPIC_API_KEY` for the SDK process (the POC wiring).
2. **Fail fast with a clear prompt when missing/incomplete.** If the file doesn't exist or
  either required key is empty, do not fall back to the shell environment — exit before any
   state is created with an actionable message:
   `Missing .env.agent — create perf_automation/.env.agent with LITELLM_BASE_URL=... and LITELLM_API_KEY=... then re-run.`
   This check runs in `PRECHECK`, before anything else.
3. **The key never leaves the process env**: never logged, never printed, never written into
  `manifest.json` / `state.json` / `ledger.jsonl` / dashboards / error messages.
4. `.env.agent` is in `.gitignore` from M0 (this lives inside the tt-metal repo — a committed
  key is an incident).

TDD:

- 🔴 `test_env_agent_is_sole_source`: with `LITELLM_API_KEY` set in the shell env but no
`.env.agent`, startup still fails with the prompt message (no silent fallback).
- 🔴 `test_env_agent_missing_or_incomplete_prompts`: absent file / missing key / empty value →
exit with the actionable message, and no `runs/<id>/` directory is created.
- 🔴 `test_env_agent_loads_and_maps`: valid file → SDK env vars populated correctly.
- 🔴 `test_secret_never_in_artifacts`: run one mock iteration, then grep every produced
artifact (state, ledger, manifest, dashboard, logs) for the key value → zero hits.

Models: `anthropic/claude-sonnet-4-6` for sub-agents; lead agent model 🟡 TBD (likely
Opus 4.8 — `TBD(model-lead)`); both overridable via `.env.agent`.

**Division of labor (do not violate):** the **harness** does deterministic routing
(`route()`), counter bookkeeping, git, and the state machine. The **agent** only: (a) picks a
lever from a closed candidate list in SELECT, (b) drives focused sub-agent edits. The agent
**never invents the routing key** and **never decides PCC pass/fail** (that's a numeric gate).

---

## 4. Standardized tag vocabulary — THE REGISTRY ✅

Routing = string-equality matching between a **bucket's tags** (computed automatically by
`tracy_tool`, §7.4) and the tags each playbook section **declares** in its `<!-- route -->`
block. Tags are **hardware-relative and model-agnostic** — derived from TT-NN kernel names and
hardware peaks (the `tt-perf-report` model), never from model structure. No "FF1", no "QKV" in
tags: that drill-down happens *inside* a section via shapes (GUIDELINES `09 §5`).

Ground truth: the real CSV schema is **resolved** — `ops_perf_results_*.csv`, 105 columns
(sample at project root, `ops_perf_results_2026_06_01_12_44_08.csv`). Key columns: `OP CODE`,
`DEVICE KERNEL DURATION [ns]`, `MATH FIDELITY`, `CORE COUNT`, `OP TO OP LATENCY [ns]`,
`ATTRIBUTES` (full ComputeKernelConfig!), `INPUT/OUTPUT_*_MEMORY/DATATYPE/Y/X`, `PM `* columns.

### 4.1 The eight dimensions


| tag        | values                                                                                             | derived from                                                         | notes                                                                                                                       |
| ---------- | -------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| `op_class` | `matmul, attention, reduction, eltwise, datamove, embedding, conv_pool, ccl, host_fallback, other` | substring map on `OP CODE` (§4.2)                                    | classes = **lever-equivalence classes**: members share the same playbook levers                                             |
| `bound`    | `dram, flop, both, slow, host`                                                                     | tt-perf-report `Bound` column (≥65% vs fidelity-adjusted arch peaks) | the physics; FLOP% only meaningful for matmul/conv — eltwise/datamove tag `slow` (correct: they route on `rank`/`dispatch`) |
| `rank`     | `time, count`                                                                                      | duration-sum ranking vs call-count ranking (`09 §4`)                 | time → tune the op; count → remove/fuse the op                                                                              |
| `fidelity` | `lofi, hifi2, hifi3, hifi4, na`                                                                    | `MATH FIDELITY` column                                               | routing-relevant: encodes remaining fidelity-walk headroom (a LoFi op at high FLOP% is at its true ceiling)                 |
| `grid`     | `full, partial, tiny`                                                                              | `CORE COUNT` vs available worker cores                               | `tiny` < 10 cores (tt-perf-report red flag)                                                                                 |
| `dispatch` | `ok, gappy`                                                                                        | median `OP TO OP LATENCY` > 6.5 µs                                   | medians only — never sum o2o (captures include inter-iteration pauses)                                                      |
| `memory`   | `dram_interleaved, l1_interleaved, sharded`                                                        | `INPUT/OUTPUT_0_MEMORY`                                              | routes L1/sharding sections                                                                                                 |
| `regime`   | `prefill, decode, na`                                                                              | M dim of input0 (decode when M ≤ 32 🟡 `TBD(regime-source)`)         | generative LLMs only; `na` elsewhere                                                                                        |


### 4.2 `op_class` substring map (data, not code)

`OP CODE` is TT-NN's **closed kernel vocabulary** — models can't invent op codes, only compose
them. First match wins; no match → `other` + coverage-lint warning (maintenance = add one line):

```python
OP_CLASS_MAP = [
    (("Matmul", "Linear"),                                          "matmul"),
    (("SDPA", "ScaledDotProduct", "FlashDecode", "PagedAttention",
      "NlpCreateHeads", "NlpConcatHeads", "RotaryEmbedding"),       "attention"),
    (("LayerNorm", "RMSNorm", "GroupNorm", "Softmax", "Reduce",
      "ArgMax", "TopK", "Moreh"),                                   "reduction"),
    (("BinaryNg", "Binary", "Unary", "Eltwise", "Where"),           "eltwise"),
    (("Reshape", "Tilize", "Untilize", "Typecast", "Transpose",
      "Permute", "Concat", "Slice", "Pad", "Copy", "Move",
      "InterleavedToSharded", "ShardedToInterleaved", "Reshard"),   "datamove"),
    (("Embedding",),                                                "embedding"),
    (("Conv", "Halo", "Pool", "GridSample", "Upsample"),            "conv_pool"),
    (("AllGather", "ReduceScatter", "AllReduce", "LineAllGather"),  "ccl"),
]
```

🔴 `TBD(genericop)` — what `GenericOpDeviceOperation` is in our models (96 calls / 8.8 ms in the
sample CSV); classify `other` until answered.

### 4.3 Thresholds (defaults adopted from tt-perf-report — do not hand-roll)

- `bound`: DRAM% / FLOP% ≥ **65%**, computed by tt-perf-report against arch peaks
(WH 288 GB/s, BH 512 GB/s; per-fidelity TFLOPs tables). We inherit, never reimplement.
- `dispatch` gappy: median op-to-op gap > **6.5 µs**.
- `grid`: `tiny` < **10** cores; `full` = available worker core count.
- `rank=count`: high call count + tiny µs/call — exact cut 🟡 `TBD(count-thresh)`.

### 4.4 Two consumers, two artifacts

`tracy_tool` output feeds two different consumers — keep them separate:

- `**tags`** → consumed by `route()` (deterministic index search). The agent never sees the index.
- `**stack_report` + `lever_state**` → consumed by the **agent** at SELECT. `lever_state` is
parsed from the `ATTRIBUTES` column (`math_fidelity`, `fp32_dest_acc_en`, `packer_l1_acc`,
`math_approx_mode`) — the levers **already pulled**. Soft context only: it informs the agent's
pick (skip exhausted levers); it does not route. Only universal tags route; the one hard
exhaustion filter is the `fidelity` tag itself (e.g. a fidelity-walk section declares
`fidelity: hifi4,hifi2` and is auto-excluded for LoFi buckets).

### 4.5 Contributor guide — tagging playbook sections (for anyone adding files)

Every *lever* section in `GUIDELINES/` gets a stable anchor + route block:

```markdown
## 6. Walk fidelity for MLP matmuls {#mlp-fidelity-walk}
<!-- route
op_class: matmul
rank: time
bound: flop,both
fidelity: hifi4,hifi3,hifi2
lever_type: walk
-->
```

Rules:

1. **Only declare dimensions the section genuinely keys on** — omitted dim = `*` (matches all).
  Over-constraining hides the section; under-constraining floods candidates.
2. Values are comma-OR lists from §4.1's closed vocabulary. Never invent new values — extend
  §4.1 first if the vocabulary is missing something.
3. **No priority field.** Candidates are returned in document order; the *agent* decides which
  to read/try first using the stack report + `lever_state` — ordering is a judgment call, not
   metadata.
4. `lever_type` ∈ `single-shot | walk | search` — drives the APPLY branch (§8.4).
5. **Process docs don't get route blocks** (`00, 07, 09, 10, AGENT_INDEX` are methodology — they
  are loaded by role, not routed by bottleneck). Only lever sections in `01–06, 08` are routed.
6. After editing, run the **coverage lint**: every tag-tuple the parser can emit must match ≥1
  section; every section must be reachable by ≥1 emittable tuple. Gaps are playbook TODOs.

---

## 5. Shared infrastructure — TDD tasks

> Build these first (Milestone M1); the whole system stands on them.

### I-1 · `atomic_write(path, data)`

- 🔴 `test_atomic_write_all_or_nothing`: monkeypatch `os.replace` to raise after the tmp file is
written; assert the target file is unchanged and **no partial/tmp file** is left at the target.
- 🟢 write `<path>.tmp`, `fsync`, `os.replace(tmp, path)`; clean tmp on failure.

### I-2 · Checkpoint with WAL semantics

- 🔴 `test_checkpoint_roundtrip`: save then load state.json → equal dict.
- 🔴 `test_resume_returns_recorded_state`: write `state="APPLY"`, load → dispatcher returns the
`APPLY` handler, not `START`.
- 🔴 `test_crash_between_intent_and_done_reverts`: write intent (`current_lever`, `git_sha_clean`)
with no done-marker → resume detects "in-flight APPLY" and resets to `git_sha_clean`.
- 🟢 `Checkpoint` class: `save(state_dict)` (uses `atomic_write`), `load()`, `mark_intent()`,
`mark_done()`, `is_in_flight()`.

### I-3 · Append-only ledger

- 🔴 `test_ledger_append_only`: append two rows → file has exactly 2 lines, first unchanged.
- 🔴 `test_ledger_idempotent_by_experiment_id`: appending a row whose `experiment_id` already
exists is a no-op.
- 🔴 `test_current_hypothesis_is_last_nonnull`: rows with `hypothesis` = [A, null, B] → current = B.
- 🟢 `Ledger.append(row)`, `Ledger.rows()`, `Ledger.current_hypothesis()`.

### I-4 · Run directory + manifest

- 🔴 `test_new_run_creates_dirs_and_latest_symlink`.
- 🔴 `test_manifest_write_once`: second write raises / is rejected.
- 🟢 `Run.create(config)`, `Run.open(run_id)`, `Run.latest()`, `Manifest.write(...)`.

### I-5 · Counters & exit policy

The goal metric is **named and directional** — not always latency. Supported now:
`device_ms` (DEFAULT, direction `min` — profiled device-kernel time, the optimization
target), `wall_ms` (direction `min`, harness clock incl. compile — reference only),
`fps` (direction `max`), `throughput_tok_s` (direction `max`);
`tok_s_per_user` later. "Target met" / "improved" are judged per `metric.direction`:
`min` → `current <= target`; `max` → `current >= target`.

- 🔴 `test_check_exit_target_met_min_metric` (wall_ms 14 ≤ 12? no → continue; 11.9 → `DONE`).
- 🔴 `test_check_exit_target_met_max_metric` (fps: current 6.5 ≥ target 6.45 → `DONE`).
- 🔴 `test_check_exit_budget_exceeded` / `test_check_exit_max_iter` → `STOPPED`.
- 🔴 `test_check_exit_no_untried_levers` → `STOPPED` (floor).
- 🔴 `test_check_exit_otherwise_continue`.
- 🟢 `check_exit(state) -> Literal["continue","DONE","STOPPED"]`. Counters live in `state.json`,
incremented at `LOG`, read here (survive resume).

---

## 6. Build order (milestones)


| M      | Theme                     | Deliverable                                                                                                             | Depends on |
| ------ | ------------------------- | ----------------------------------------------------------------------------------------------------------------------- | ---------- |
| **M0** | Scaffold                  | `.venv`, SDK install, `.env.agent` loader + fail-fast validation (§3.1), `.gitignore`, `pytest` green on a smoke test   | —          |
| **M1** | Infra (§5)                | atomic write, checkpoint+WAL, ledger, run dir, exit policy                                                              | M0         |
| **M2** | Routing (§7.2)            | `router.py` formalized (`build_index`, `route`, `read_section`) + `cache_playbook` + coverage lint                      | M1         |
| **M3** | Stage 1 (§7)              | `environment_check`, `read_model_files` sub-agent, **mock** `tracy_tool` baseline                                       | M2         |
| **M4** | State machine (§8)        | engine over states with a **mock evaluator** (port `loop.py`), per-stage resume test, `agent_calls.jsonl` + cost wiring | M1–M3      |
| **M5** | Agent edge (§8.3, §8.5, §8.5.1–2) | `SELECT`, `edit_file` sub-agent, `VERIFY`, `REPAIR` self-heal loop                                             | M4         |
| **M6** | Gates & decide (§8.6–8.9) | single-stage e2e PCC verdict, median-3 remeasure, keep/discard DECIDE, commit/revert                                    | M5         |
| **M7** | Memory (§9)               | ledger `hypothesis`, dashboard render, **resume brief** (warm start), lever-eval over ledger                            | M6         |
| **M8** | Go real                   | swap mocks → real Tracy + PCC + model build (TBD-heavy)                                                                 | M7         |


Each milestone ends green (all tests pass) before the next begins.

---

## 7. STAGE 1 — Before Loop

### 7.1 `environment_check` (tool)

- **Purpose:** capture HW facts so levers use the real device grid.
- **In:** none. **Out:** `{card, arch, grid_x, grid_y, dram_bw, ...}` → into `manifest.json`.
- 🔴 `test_environment_check_parses_mock` (fixture stands in for `tt-smi`/script output).
- 🟢 run script 🔴 `TBD(env-script)` (exact command + output format), parse to dict.
- ♻️ on real HW, replace the fixture with the live command; schema unchanged.

### 7.2 `cache_playbook` (tool) + router core

- **Purpose:** build `.cache/playbook_index.json` from `playbook/*.md` (`{#id}` anchors +
`<!-- route -->` blocks). Already prototyped in `poc/router.py` — formalize and test.
- **Out:** `index = [{id, title, file, lever_type, op_class, bound, rank, fidelity, grid, dispatch, memory, regime}, ...]` (dims per §4.1; omitted dim = `*`; no priority —
candidates return in document order, the agent orders them).
- 🔴 `test_build_index_harvests_route_blocks` (use the real `GUIDELINES/*.md`).
- 🔴 `test_index_cache_invalidates_on_content_change` (hash changes → rebuild).
- 🔴 `test_route_matches_by_tag_equality`: `(op_class=matmul, rank=time, bound=flop)` → the
matmul lever sections, in document order.
- 🔴 `test_route_fidelity_exhaustion`: a `fidelity: hifi4,hifi3,hifi2` section is **excluded**
for a `fidelity=lofi` bucket (the walk is exhausted).
- 🔴 `test_route_wildcard`: `*` / omitted dim on either side matches.
- 🔴 `test_read_section_by_anchor`: returns text from `{#id}` to next `##` .
- 🔴 `test_coverage_lint_flags_uncovered_key`.
- 🟢 `build_index()`, `route(...)`, `read_section(id)`, `coverage_lint(index, possible_keys)`.

### 7.3 `read_model_files` (sub-agent)

- **Purpose:** discover the model's files and the PCC entry points (end_to_end, attention, etc.).
- **Out:** file map → `manifest.json` (`{pcc: {end_to_end: path, attention: path}, model_files: [...]}`).
- 🔴 `test_read_model_files_returns_pathmap` against a mock model tree fixture.
- 🟢 sub-agent `query()` with read-only tools + narrow prompt; validate the returned JSON shape.
- 🔴 `TBD(model-tree)` — real model repo layout & which files define PCC entry points.

### 7.4 `tracy_tool` — profile pipeline (tool, **mock first**)

- **Purpose:** one tool call, three deterministic stages — no LLM inside:
  ```
  1. RUN      tracy -m pytest <end_to_end> ...      → ops_perf_results_*.csv  (raw, 105 cols)
  2. REFINE   tt-perf-report raw.csv --csv report.csv \
              --start-signpost <S> --end-signpost <E> --no-advice
              → per-op: device time, op-to-op gap, cores, DRAM %, FLOP %, Bound, fidelity
  3. TAG+BUCKET  group by OP_CLASS_MAP → buckets; normalize tags (§4.1);
                 join raw-CSV cols report.csv lacks (ATTRIBUTES → lever_state, MEMORY)
  ```
- **In:** `{pcc_path, batch_size, seq_len, runs, id_range?}`. **Out:**
`{stack_report, buckets:[{id, device_ms, pct, count, tags{...§4.1}, lever_state{...}}], wall_ms, artifacts:{raw_csv, report_csv}}`. Both CSVs land in `runs/<id>/profiles/`.
- **Why tt-perf-report as a dependency (not reimplemented):** inherits the arch-peak tables and
Bound physics (§4.3); `--start/--end-signpost` excludes inter-iteration pauses (the sample CSV
has 11.8 s of op-to-op vs 120 ms kernel — sums of o2o are poison); `--id-range` gives
single-layer isolation for the PCC gates (`09 §2`) with zero extra code.
- 🔴 `test_tracy_parse_real_schema`: parse a fixture cut from the **real** 105-col CSV
(`ops_perf_results_2026_06_01_12_44_08.csv`) → expected buckets/tags.
- 🔴 `test_opclass_map_covers_sample`: every OP CODE in the sample maps to a class ≠ crash;
unknown codes → `other` + lint warning.
- 🔴 `test_attributes_join`: `lever_state` correctly joined from raw CSV when report.csv lacks it.
- 🔴 `test_tracy_median_of_n`: given N wall times, returns the **median** (noise floor §8.7).
- 🔴 `test_o2o_uses_median_never_sum`.
- 🟢 mock stage 1 returns the fixture CSV; stages 2–3 run for real (tt-perf-report is pure
CSV→CSV, no hardware needed). Only stage 1 is swapped in M8.
- 🔴 `TBD(end2end-cmd)` (exact pytest target + args), 🔴 `TBD(signposts)` (signpost names the
test emits — required for `--start/--end-signpost`).

---

## 8. STAGE 2 — Agent Loop (state machine)

### 8.1 Engine

- **In:** `runs/latest/state.json` (`state` field) + `handlers` map. **Out:** same file advanced one transition (atomic save after each); loop until a terminal state.
- 🔴 `test_engine_runs_states_in_order` (mock handlers record their order).
- 🔴 `test_engine_resumes_from_checkpoint_state` (set `state=GATE_PCC` → starts there).
- 🔴 `test_engine_stops_on_terminal` (DONE/STOPPED/FAILED end the loop).
- 🟢 `Engine` with `handlers: dict[State, callable]`, checkpoint save after every transition,
dispatch on `state.json`. **BUILT (2026-06-11)** as a walking skeleton: `agent/engine.py`
(dumb dispatcher, cycle/unknown-state guards), `agent/states.py` (names + `TRANSITIONS` +
repair budgets), `agent/loop_context.py` (the `ctx` seam), `agent/handlers/` (one file per
stage; `route.py` + `log_exit.py` REAL, `mocks.py` for the rest), `agent/loop.py` (entry from
`runs/latest`). `tests/test_engine.py` walks ROUTE→…→DONE with mocks (8 tests). Members swap a
mock for a real module in `agent/handlers/__init__.py`, one line at a time, and re-run.

### 8.2 `ROUTE`

- **In:** `state.json` (`metric`,`tried`) + the **current** profile (`ctx.current_profile()` = `profiles/iter_<N>_profile.json` of the last committed model, or `baseline_profile.json` on iteration 0) + `.cache/playbook_index.json`. **Out:** `state.json` gains `current_bucket` and `candidates` (section ids from `route()`).
- 🔴 `test_route_state_picks_top_bucket_then_routes`: from the CURRENT profile's buckets (not the frozen baseline — the bottleneck moves as changes land), select the top
bucket (🔴 `TBD(bucket-select-policy)` — top-by-ms? agent-assisted?), call `route()`, write
`current_bucket` + `candidates` to state.
- 🟢 deterministic; agent not involved.

### 8.3 `SELECT` (agent — the only reasoning step)

- **In:** `state.json` (`candidates`,`tried`) + the playbook section bodies. **Out:** `state.json.selected_lever` (one id ∈ candidates, ∉ tried) + one `agent_calls.jsonl` row. *(the only loop stage that needs an API key)*
- 🔴 `test_select_commits_untried_candidate`: agent given `candidates` + `tried` → commits an
untried id; harness validates against the closed list; fallback to `untried[0]` on
limit/invalid pick.
- 🔴 `test_select_records_lever_for_resume` (resume skips re-asking).
- 🟢 **BUILT** (`select_agent.py`, `handlers/select.py`): the LEAD model reads ROUTE's brief
(no tools — the brief already carries the candidate text + model map), commits one id from the
closed `candidates` list (validated ∉ tried), graceful `except` → `untried[0]`. → PLAN.

### 8.4 ~~lever-type branch~~ — REMOVED (decision 2026-06-11)

No branching on lever type, no `sweep_tool`. **Every lever is exactly one edit** applied by
APPLY (§8.5). A multi-step idea (e.g. a fidelity walk HiFi2→3→4) is simply three normal loop
iterations, each picking the next untried step. `lever_type` stays in the route metadata as an
informational tag only — nothing branches on it; `TBD(sweep-space)` is dropped.

### 8.4.5 `PLAN` (agent — localize the edit)  ✅ BUILT (loop branch)

- **In:** `state.json.selected_lever` + the lever's playbook section + the op-class-filtered
**model map** (`model_map.py` ast skeleton: classes/fns/`ttnn.*` ops with line+scope+var). **Out:**
`state.json.edit_spec` = `{file, location, change}`. → APPLY (or → REVERT if NOOP).
- The LEAD reads the map and the target file (Read/Grep, **no Edit**) and emits a *localized*
instruction so the editor's job is mechanical (fixes the 24-turn "editor wanders the whole model"
failure). If the lever is already applied, `change="NOOP: ..."` → discard `already_applied`
cheaply (no edit/gate/measure) — the redundant-lever guard.
- 🟢 `plan_agent.py` (`build_plan_prompt`/`_validate_spec`/`make_plan_runner`, lead model),
`handlers/plan.py`; planner injectable (`ctx.deps["plan_runner"]`); planning failure → APPLY
(improvise). Rationale + references (aider repo-map, just-in-time retrieval): see §8.12.

### 8.5 `APPLY` (sub-agent `edit_file`)

- **In:** `state.json.selected_lever` + `manifest.json` (`pathmap.model_files`) + a clean git tree. **Out:** `state.json.git_sha_clean` recorded, then the target file(s) edited in place — exactly one edit per lever.
- 🔴 `test_apply_records_clean_sha_before_edit`.
- 🔴 `test_apply_resume_resets_then_reapplies` (idempotent from clean SHA).
- 🟢 **BUILT** (`edit_agent.py`, `handlers/apply.py`): record `git_sha_clean`, then the
`edit` sub-agent applies the **PLAN spec** (`build_spec_prompt`) — or improvises from the lever
if no spec. Ground truth = `git diff <clean>` **scoped to the model dir** (the editor's
self-report is advisory; mis-reports/errors never crash or misattribute). No real edit landed
→ REPAIR_CODE. → VERIFY.

### 8.5.1 `VERIFY` (deterministic — no device, no agent)  ✅ BUILT (loop branch)

- **In:** the edited files (resolved from `state.last_edit.files`). **Out:** verdict `ok` |
`parse_error` (+ error text). **SYNTAX-ONLY** (`ast.parse`): the standalone import check was
**dropped** — model files use absolute package imports (`from models.common…`), so loading one
out of its package context gives a FALSE `import_error` on even a perfect edit. Real
import/runtime errors are caught downstream by GATE_PCC (which runs the test in the correct
environment → `crash` verdict → REPAIR_CODE).
- 🟢 BUILT (`handlers/verify.py`): `ast.parse` each edited file → ok / parse_error;
parse_error → REPAIR_CODE (or REVERT once code-fix budget spent).
- 🔴 (superseded) `test_verify_catches_syntax_error` (`ast.parse` on each edited file).
- 🔴 `test_verify_catches_import_error` (import in a subprocess so a crash can't kill the loop).
- 🟢 `ast.parse` every edited file, then import the module(s) in a child process; capture stderr.
The cheapest rung of the ladder — most agent typos die here with zero device time.

### 8.5.2 `REPAIR` (agent — the self-heal loop)

- **In:** the failing verdict (error text) + `selected_lever` intent + `manifest.model_files` + the relevant counter. **Out:** a fresh edit on the tree, the matching counter incremented, then re-enter `VERIFY`; one `agent_calls.jsonl` row per attempt.
- **Two modes, one edit sub-agent, different prompt:**
  - **code repair** (`parse_error`/`import_error`/`GATE_PCC=crash`): *"your edit failed with `<error>`; fix the bug, KEEP the optimization intent, do not delete it."* Budget `code_fix_attempts ≤ 5`.
  - **pcc repair** (`GATE_PCC=pcc_low`): *"your edit applied but PCC=`<v>` < `<thr>`; re-apply more conservatively (dtype / memory-config), preserve correctness."* Budget `pcc_fix_attempts ≤ 2`.
- 🔴 `test_repair_code_retries_to_5_then_abandons` → revert, mark `tried` reason `edit_failed`, LOG, → SELECT.
- 🔴 `test_repair_pcc_retries_to_2_then_discards` → revert, mark `tried` reason `pcc_failed`, LOG, → SELECT.
- 🔴 `test_repair_increments_correct_counter`; `test_repair_resume_reads_counter_from_state`.
- 🔴 `test_repair_records_agent_call` (telemetry row, role `repair_code`/`repair_pcc`).
- 🔴 `test_repair_noop_edit_is_abandoned` (if the "fix" reverts to the clean tree → not a real repair → abandon; the lazy-fix guard).
- 🟢 same narrow-tool edit sub-agent as APPLY with an error-augmented prompt; counters
checkpointed (reset on the next SELECT); `budget_usd` hard-checked before each call.

### 8.6 `GATE_PCC` (single-stage e2e, deterministic)

- **In:** the edited tree + `manifest.json` (`pathmap.pcc.end_to_end.{path,threshold}` — threshold already extracted at discovery). **Out:** verdict `ok` (pcc ≥ threshold) | `crash` (exception, no number → code-repair) | `pcc_low` (number < threshold → pcc-repair) + the measured value.
- 🔴 `test_gate_pass_when_pcc_ge_threshold`.
- 🔴 `test_gate_pcc_low_routes_to_pcc_repair`; `test_gate_crash_routes_to_code_repair`.
- 🔴 `test_gate_threshold_from_manifest`.
- 🟢 run the **end-to-end** PCC test, parse the measured PCC from output, compare to the manifest
threshold. **Single-stage for now** (user decision 2026-06-11 — no single-layer→full ladder).
🔴 `TBD(pcc-parse)` — how the test surfaces the number (structured print vs regex on the assert).
Reduction-op special-casing (GUIDELINES `07`) deferred.

### 8.7 `REMEASURE` (noise floor)

- **In:** the edited tree, N (`TBD(noise-N)`), `tracy_tool`. **Out:** `after` metric value = **median** of N runs (Δ vs `before` discarded as noise if under `TBD(noise-floor)`), AND the full re-bucketed `profiles/iter_<N>_profile.json` (tracy_tool returns buckets for free) — recorded in `last_decision.profile`, promoted to `current_profile` by COMMIT on a keep.
- 🔴 `test_remeasure_median_of_3`; `test_remeasure_rejects_below_noise_floor` (Δ under the floor →
treated as noise, not a win).
- 🟢 median-of-N (N 🔴 `TBD(noise-N)`, GUIDELINES suggests 3) with absolute floor 🔴
`TBD(noise-floor-us)` (≈50/200 µs). A tracy crash *here* is post-PCC (the edit ran correctly) →
infra flakiness, not an edit bug: retry the measurement once, else discard reason `measure_failed`
(this is the only crash path that does NOT go to REPAIR).

### 8.8 `DECIDE` (keep / discard — crashes handled upstream by REPAIR)

- **In:** pure tuple `(before, after)` — by the time DECIDE runs, PCC already passed (§8.6) and the measurement succeeded (§8.7). **Out:** `keep` | `discard`. No file I/O, no deps — easiest stage to test standalone.
- 🔴 `test_decide_keep_when_improved` — "improved" per `metric.direction` (min: lower wins; max: higher wins); noise floor applies in metric units.
- 🔴 `test_decide_discard_when_no_improvement` (Δ under floor or wrong direction → discard, reason `no_gain`).
- 🟢 pure function of (before, after, direction, floor). Edit/run crashes never reach here — they
are absorbed by REPAIR (§8.5.2); an unmeasurable lever is discarded by REMEASURE (§8.7).

### 8.9 `COMMIT` / `REVERT` (git, idempotent)

- **In:** the decision + `state.json.git_sha_clean`. **Out:** keep → commit, new `git_sha_clean` at HEAD; discard/crash → `git reset --hard <clean>`. Idempotent (safe to re-run on resume).
- 🔴 `test_commit_noop_if_head_already_there`; `test_revert_resets_to_clean_sha`.
- 🟢 `commit(msg)` updates `git_sha_clean` AND promotes `last_decision.profile` to `state.current_profile` (so the next ROUTE routes on the new bottleneck); `revert()` = `git reset --hard <clean>` and leaves `current_profile` unchanged. Branch naming
🔴 `TBD(git-branch-policy)`.

### 8.10 `LOG` → `CHECK_EXIT`

- **In:** `state.json` (`iteration`,`metric`,`cost_usd`,`max_iter`,`budget_usd`) + the decision result (incl. optional `hypothesis`). **Out:** one `ledger.jsonl` row appended, `dashboard.html` re-rendered, counters incremented, exit verdict `continue|DONE|STOPPED`.
- 🔴 `test_log_appends_row_with_hypothesis`; `test_log_idempotent`.
- 🔴 `test_check_exit_routes_correctly` (reuses §5 I-5).
- 🟢 append ledger row (incl. `hypothesis` when shifted), render `dashboard.html`, increment
counters, then `check_exit`.

---

### 8.11 Work split for parallel development + standalone fixtures

The loop splits into two halves along a single interface — **`state.json` + the run directory**.
Each member works against a checked-in fixture, so neither blocks on the other's stages or on an
API key. Drop these under `tests/fixtures/loop/`.

| Member | Stages | Needs API key? | Starts from fixture |
|---|---|---|---|
| **Member 1 — decide & act** | Engine (§8.1), ROUTE (§8.2), SELECT (§8.3), APPLY (§8.5), **VERIFY (§8.5.1)**, **REPAIR (§8.5.2)** | SELECT + REPAIR (mock both otherwise) | `after_before_loop/`, `repair_inputs/` |
| **Member 2 — evaluate & record** | GATE_PCC (§8.6, single-stage → verdict), REMEASURE (§8.7), DECIDE (§8.8), COMMIT/REVERT (§8.9), LOG→CHECK_EXIT (§8.10), resume brief (§9) | **none** | `before_evaluate/` |

Interface contract: Member 1 produces a `state.json` with `selected_lever` set and an edited
tree (+ `git_sha_clean`); Member 2 consumes exactly that and never needs to run SELECT/APPLY.
Member 1 mocks SELECT's single output line (`selected_lever`) when no key is present.

**The inner repair loop (§8.5.1–8.5.2) spans both members — the seam is the verdict, not code.**
`VERIFY` (Member 1) and `GATE_PCC` (Member 2) each return a plain dict the Engine routes on:
```json
{"status": "ok" | "parse_error" | "import_error" | "crash" | "pcc_low",
 "pcc": 0.74, "error": "<captured text>"}
```
The Engine's transition table (shared, owned by whoever builds §8.1) maps
`parse_error|import_error|crash → REPAIR(code, ≤5)` and `pcc_low → REPAIR(pcc, ≤2)`. So Member 2
writes a verdict *producer* and never imports REPAIR; Member 1 writes REPAIR against a **verdict
fixture** and never runs the real PCC test. Neither blocks the other.

#### Fixture A — `after_before_loop/` (Member 1 entry: what ROUTE reads)

No API key needed to develop or test ROUTE — it is pure function of these two files.

`state.json`:
```json
{
  "run_id": "FIXTURE", "state": "BEFORE_LOOP_DONE", "iteration": 0,
  "metric": {"name": "device_ms", "unit": "ms", "direction": "min",
             "baseline": 12.091, "current": 12.091, "target": 11.0},
  "max_iter": 25, "budget_usd": 5.0, "cost_usd": 0.0,
  "tokens_in": 0, "tokens_out": 0,
  "git_sha_clean": null, "candidates": [], "tried": [], "crash_retries": 0,
  "last_error": null
}
```
`profiles/baseline_profile.json` (trimmed to what ROUTE uses — the real S128 baseline):
```json
{
  "device_ms": 12.091, "wall_ms": 13291,
  "buckets": [
    {"id": "matmul",    "device_ms": 6.741, "pct": 55.7, "count": 96,
     "tags": {"op_class": "matmul", "fidelity": "hifi2", "rank": "time", "bound": "flop"}},
    {"id": "reduction", "device_ms": 2.052, "pct": 16.9, "count": 50,
     "tags": {"op_class": "reduction", "grid": "tiny", "rank": "count"}},
    {"id": "attention", "device_ms": 1.634, "pct": 13.5, "count": 48,
     "tags": {"op_class": "attention", "fidelity": "hifi2"}},
    {"id": "eltwise",   "device_ms": 1.024, "pct": 8.5,  "count": 78,
     "tags": {"op_class": "eltwise", "fidelity": "hifi4", "rank": "count"}},
    {"id": "datamove",  "device_ms": 0.492, "pct": 4.1,  "count": 32,
     "tags": {"op_class": "datamove"}}
  ]
}
```
**Expected ROUTE output** (written back into `state.json`): with a top-by-ms policy,
`current_bucket = "matmul"`, `candidates = ["mlp-fidelity-walk", "subblock-unlock",
"fuse-activation-matmul", ...]` (whatever `route()` returns for those tags). SELECT then picks
one; if no API key, mock `selected_lever = "mlp-fidelity-walk"` and hand off to Member 2.

#### Fixture B — `before_evaluate/` (Member 2 entry: no upstream, no key)

Member 2 never runs SELECT/APPLY. This fixture is "an edit has already been applied; now
evaluate and record it." The three evaluate stages are testable from plain values:

- **DECIDE** is a pure function — call it directly (PCC already passed, measurement already done):
  `decide(before=12.091, after=11.41, direction="min", floor_ms=0.05) -> "keep"`
  `decide(before=12.091, after=12.06, ...) -> "discard"` (Δ under noise floor → reason `no_gain`)
- **REMEASURE** median: `remeasure([12.10, 11.41, 11.55]) -> 11.55` (median, never mean/sum).
- **GATE_PCC** returns a verdict, not a bool:
  `pcc_gate(measured=0.997, threshold=0.99) -> {"status": "ok", "pcc": 0.997}`
  `pcc_gate(measured=0.74,  threshold=0.99) -> {"status": "pcc_low", "pcc": 0.74}`
  exception during the run → `{"status": "crash", "error": "<traceback>"}`.

`state.json` (mid-loop, what LOG→CHECK_EXIT reads):
```json
{
  "run_id": "FIXTURE", "state": "DECIDE", "iteration": 3,
  "metric": {"name": "device_ms", "unit": "ms", "direction": "min",
             "baseline": 12.091, "current": 11.83, "target": 11.0},
  "max_iter": 25, "budget_usd": 5.0, "cost_usd": 0.42,
  "tokens_in": 41000, "tokens_out": 3100,
  "git_sha_clean": "abc1234", "candidates": ["mlp-fidelity-walk", "subblock-unlock"],
  "tried": ["fuse-activation-matmul"], "crash_retries": 0, "last_error": null
}
```
`decision` (the result LOG consumes, produced by DECIDE+REMEASURE+GATE_PCC):
```json
{
  "result": "keep", "lever": "mlp-fidelity-walk",
  "before": 11.83, "after": 11.41, "pcc": 0.997,
  "hypothesis": "matmul bucket was fidelity-bound at HiFi2; one step to HiFi3 bought 0.42 ms with PCC intact"
}
```
**Expected LOG→CHECK_EXIT output:**
- `ledger.jsonl` += one row: `{iteration:3, lever:"mlp-fidelity-walk", before:11.83,
  after:11.41, delta:-0.42, pcc:0.997, result:"keep", hypothesis:"..."}`
- `state.json`: `iteration -> 4`, `metric.current -> 11.41`, `tried += "mlp-fidelity-walk"`.
- exit verdict: `continue` (11.41 > target 11.0, iteration 4 < 25, cost < budget). Change
  `after` to `10.9` → verdict `DONE`; set `iteration` near `max_iter` → `STOPPED`.

#### Fixture C — `repair_inputs/` (Member 1 entry: REPAIR without GATE_PCC or a key)

REPAIR consumes a verdict + the lever intent; develop it against these without running PCC. Mock
the edit sub-agent's reply (a patched file) when no key is present — the loop control (counters,
abandon thresholds) is what's under test, not the model.

`verdict_code.json` (drives the ≤5 code-repair path):
```json
{"status": "import_error", "file": "mlp.py",
 "error": "ImportError: cannot import name 'dram_matmul_config' from 'ttnn'",
 "selected_lever": "mlp-fidelity-walk"}
```
Expected: REPAIR edits, `code_fix_attempts` 0→1, re-enter VERIFY. Force 5 failures →
ABANDON: revert to `git_sha_clean`, `tried += "mlp-fidelity-walk"` (reason `edit_failed`), LOG.

`verdict_pcc.json` (drives the ≤2 pcc-repair path):
```json
{"status": "pcc_low", "pcc": 0.74, "threshold": 0.99,
 "selected_lever": "mlp-fidelity-walk"}
```
Expected: `pcc_fix_attempts` 0→1, re-edit conservatively, re-VERIFY→re-GATE. Force 2 failures →
DISCARD (reason `pcc_failed`), LOG, → SELECT. A "fix" that reverts to the clean tree → treated as
no-op → abandon immediately (lazy-fix guard, `test_repair_noop_edit_is_abandoned`).

These fixtures + the pure-function calls let both members reach green tests with **zero hardware
and zero API spend**; the real wiring is swapped in only at integration.

---

### 8.12 Loop branch build log — decisions, bugs-found-on-hardware, references

**New files (`gtobarTT/perf_automation_loop`):** `engine.py`, `states.py`, `loop_context.py`,
`loop.py`, `looplog.py`, `gitio.py`, `model_map.py`, `plan_agent.py`, `select_agent.py`,
`edit_agent.py`, and `handlers/{route,select,plan,apply,verify,gate_pcc,remeasure,decide,log_exit}.py`
(+ `mocks.py`). Demos: `demo_walk.py`, `demo_route.py`, `demo_model_map.py`,
`experiments/walk_to_decide.py`. Suite 170 passed.

**Architecture decisions made while building:**
- **Editor model** = dedicated `edit` role, inherits SUB (haiku) from `.env`; applies the lead's localized PLAN spec verbatim (the reasoning/localization lives in PLAN, run on the lead).
  Editing a model to apply a lever is design+coding, not transcription — haiku produced edits
  that didn't import. (`sub`/haiku stays for discovery only.)
- **PLAN stage** (§8.4.5) + **model map** (§3 of the localization design): the lead emits a
  localized `{file, location, change}` spec so the editor's job is mechanical. Bounds editor
  context: the map is a budgeted, op-class-filtered ast skeleton (signatures + op lines, never
  bodies) + just-in-time `read_region`. Makes a cheaper editor viable later.
- **NOOP guard:** PLAN returns `NOOP` when a lever is already applied (e.g. BGE-M3 already fuses
  QKV) → discard `already_applied` with no edit/gate/measure spent.
- **VERIFY = syntax-only** (§8.5.1). **Prompt logging:** every agentic call's full prompt+response
  persists as JSON to `runs/<id>/prompts/<NNN>_<stage>.json`; `agent_calls.jsonl` carries
  `agent_call_id`, `prompt_file`, `prompt_sha`, and `response_sha`.

**Bugs found by running on real hardware (and fixed):**
- VERIFY standalone-import check → false `import_error` on package-import files → dropped (syntax-only).
- APPLY git-diff ground truth was repo-wide → attributed unrelated uncommitted repo edits to the
  model edit → **scoped to the model dir**.
- Editor hit the 24-turn cap wandering the whole model with only a lever name → the PLAN spec fixes it.
- APPLY now never crashes the loop on an editor error (git-diff truth → REPAIR_CODE/REVERT).
- `GATE_PCC.parse_pcc` verified live on the real e2e test (read 0.9999 correctly).

**Still MOCK (next builds):** `REPAIR_CODE` / `REPAIR_PCC` (real self-heal: re-edit with the
error, ≤5/≤2) and `COMMIT` / `REVERT` (real path-scoped git). DECIDE noise floor stays the
deferred placeholder (reads `last_decision.spread` when wired).

**New state.json fields:** `current_profile`, `current_bucket`, `candidates`, `tried`,
`selected_lever`, `select_reasoning`, `edit_spec`, `last_edit`, `last_verdict`, `last_decision`,
`code_fix_attempts`, `pcc_fix_attempts`. **New run-dir artifacts:** `route_brief_<N>.json`,
`profiles/iter_<N>_profile.json`, `prompts/*.json`.

**References (localization design):** aider repo-map (tree-sitter symbol skeleton + PageRank +
token budget) https://aider.chat/docs/repomap.html ; 2026 just-in-time-retrieval-beats-stuffing +
graph-nav-beats-retrieval-for-architecture-heavy-code https://sourcegraph.com/blog/agentic-coding .

---

## 9. Memory & warm start (resume after days away)

**Principle:** persist **conclusions, not the transcript.** Do not rely on SDK session resume for
multi-day gaps (cache TTL, transcript cruft). Reconstruct understanding from durable artifacts.

- **Where I am** → `state.json` (checkpoint).
- **What happened** → `ledger.jsonl` rows (facts).
- **What it means / next** → the last non-null `hypothesis` in the ledger.

### 9.1 `update_hypothesis` (agent self-memory)

- 🔴 `test_agent_can_write_hypothesis`: at iteration end the agent may call `update_hypothesis(text)`
→ appended to the current ledger row.
- 🟢 a tool that sets the `hypothesis` on the row being written at `LOG`.

### 9.2 `compose_resume_brief()`

- 🔴 `test_resume_brief_contains_state_tail_and_hypothesis`: brief = current numbers (from
checkpoint) + last K ledger notes + current hypothesis; **does not** replay the transcript.
- 🟢 deterministic string builder; injected as the **first user message** on resume so the agent
is warm in one turn.
- 🔴 `TBD(brief-K)` — how many ledger rows to include.

---

## 10. Cross-cutting controls

- **Cost & iterations** — tracked in `state.json`, surfaced in `dashboard.html`; enforced in
`CHECK_EXIT` and per-call `max_budget_usd`.
- **Secrets** — credentials read from `.env.agent` only (§3.1); fail fast with a user prompt if
absent; tests assert no shell-env fallback and that the key never appears in any artifact.
- **Determinism boundary** — a test asserts tools contain no model calls
(`test_tools_have_no_llm`), keeping the core deterministic.

### 10.1 Observability (added 2026-06-10)

Industry model: **traces** (parent-child spans) + **metrics** (cost/latency) + **evaluation
over traces** (score decisions, feed back). We keep it plain JSONL beside the run — no
platforms; OTel GenAI export is an optional later add-on, never a dependency.


| Layer            | Artifact                                                                                                                                       | Status                           |
| ---------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------- |
| workflow trace   | `events.jsonl` — add `iteration` so loop stages nest under iterations                                                                          | partial (Before Loop only)       |
| agent telemetry  | `agent_calls.jsonl` — one row PER `query()`: role (discovery/lead/select/edit), model, tokens in/out, cost, latency, prompt_sha, full response | M4 — closes the dead budget gate |
| decision audit   | sub-agent raw JSON + lead reasoning persisted verbatim (manifest / agent_calls)                                                                | partial (verdict only)           |
| run metrics      | `state.cost_usd` accumulated from agent_calls                                                                                                  | M4                               |
| eval over traces | per-tag lever win rate from ledger (`hypothesis` vs measured Δ) — the playbook improves itself                                                 | M7                               |


Rules: telemetry is append-only and never read by stage logic (except cost into CHECK_EXIT);
responses logged whole (cheap, invaluable on replay); the LiteLLM key never appears.

---

## 11. TBD register (for the human to fill)

**Decided constants (not TBD):** repair budgets `code_fix_attempts ≤ 5` (parse/import/run crash),
`pcc_fix_attempts ≤ 2` (PCC below threshold); GATE_PCC is single-stage e2e using the
manifest-extracted threshold (no single-layer→full ladder). — user, 2026-06-11.


| id                                    | What's needed                                                                                                                                                                                                                                                                                                                  | Blocks |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------ |
| ~~`TBD(tracy-csv-schema)`~~           | ✅ RESOLVED — 105-col `ops_perf_results_*.csv`; real sample at project root                                                                                                                                                                                                                                                     | —      |
| ~~`TBD(opcode-map)`~~                 | ✅ RESOLVED — `OP_CLASS_MAP` §4.2 (closed TT-NN kernel vocabulary)                                                                                                                                                                                                                                                              | —      |
| ~~`TBD(kn-split)`~~                   | ✅ RESOLVED differently — (K,N) split is in-section drill-down (`09 §5`), **not** a routing tag. Sample: (1024→4096)=FF1, (4096→1024)=FF2, (1024→3072)=fused QKV, (1024→1024)=O-proj                                                                                                                                            | —      |
| ~~`TBD(util-thresh)`~~                | ✅ RESOLVED — tt-perf-report `Bound` at ≥65% vs fidelity-adjusted peaks (§4.3)                                                                                                                                                                                                                                                  | —      |
| `TBD(genericop)`                      | 🟡 likely RESOLVED — `GENERAL/03 §5c`: BGE-M3's custom `generic_op` head-split/concat-heads kernels (2/layer × 48 = 96 calls matches the sample). Map to `attention` for these models; confirm. Generic_op is user-defined, so op code alone can't classify — long-term, disambiguate via `COMPUTE KERNEL SOURCE`/`ATTRIBUTES` | §4.2   |
| `TBD(signposts)`                      | Signpost names the end_to_end test emits (for `--start/--end-signpost`)                                                                                                                                                                                                                                                        | §7.4   |
| `TBD(end2end-cmd)`                    | 🟡 pipeline resolved (§7.4); still need exact pytest target + args                                                                                                                                                                                                                                                             | §7.4   |
| `TBD(count-thresh)`                   | count vs µs/call threshold for `rank=count`                                                                                                                                                                                                                                                                                    | §4.3   |
| `TBD(regime-source)`                  | 🟡 proposed: decode when M ≤ 32 — confirm                                                                                                                                                                                                                                                                                      | §4.1   |
| `TBD(model-lead)`                     | lead-agent model id (Opus 4.8?)                                                                                                                                                                                                                                                                                                | §3     |
| `TBD(env-script)`                     | environment_check command + output format                                                                                                                                                                                                                                                                                      | §7.1   |
| `TBD(model-tree)`                     | real model repo layout; which files define PCC entry points                                                                                                                                                                                                                                                                    | §7.3   |
| `TBD(bucket-select-policy)`           | how to pick which bucket to attack each iteration                                                                                                                                                                                                                                                                              | §8.2   |
| `TBD(pcc-parse)`                      | how the e2e test surfaces its PCC number (structured print vs regex on the assert) — GATE_PCC needs to read it                                                                                                                                                                                                                  | §8.6   |
| `TBD(noise-N)`, `TBD(noise-floor-us)` | median sample count + absolute noise floor                                                                                                                                                                                                                                                                                     | §8.7   |
| `TBD(git-branch-policy)`              | branch/commit naming for experiments                                                                                                                                                                                                                                                                                           | §8.9   |
| `TBD(brief-K)`                        | # of ledger rows in the resume brief                                                                                                                                                                                                                                                                                           | §9.2   |


---

## 12. Definition of done

- All milestone tests green with **mocks** (M0–M7) — the full loop runs end-to-end, resumes after
a simulated mid-stage crash, auto-reverts a crashing lever (retry-once), stops on target/budget/
floor, and produces a warm resume brief — **without hardware**.
- M8 swaps mocks for real Tracy/PCC/build with **zero change to control logic**; only the TBDs are
filled in.
- Every 🔴 TBD is either resolved or still carries a `# TBD(<id>)` marker traceable to §11.
