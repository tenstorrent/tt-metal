# End-to-end bring-up flow (Items 1–8)

This document explains how the 8 modules added in the 2026-06-02 audit
fit together. Each is independently testable; the orchestrator (Item 8)
chains them in the order the design specified.

## TL;DR

```
Convergence (per-component PCC all pass)
    │
    ▼
┌──────────────────────────────────────────┐
│ STEP 1 — Template reuse (Item 6 / 7)     │
│ find_template_for_model(model_type)      │
│   promoted? → STATUS_TEMPLATE_REUSED ────┐
│   else → fall through                    │
└──────────────────────────────────────────┘
    │                                      │
    ▼                                      │
┌──────────────────────────────────────────┐
│ STEP 2 — LLM verify (Item 2)             │
│ run_llm_verify_pass(...)                 │
│   PASS → STATUS_VERIFY_PASSED ───────────┤
│   FAIL → fall through with diagnostic    │
└──────────────────────────────────────────┘
    │                                      │
    ▼                                      │
┌──────────────────────────────────────────┐
│ STEP 3 — LLM synthesis (Item 3)          │
│ run_e2e_synthesis_loop(...)              │
│   converged → register (6) + promote (7) │
│             STATUS_SYNTHESIS_CONVERGED ──┤
│   failed → STATUS_SYNTHESIS_FAILED       │
│                                          │
│   inside the loop, late_discoveries:     │
│     parse TODO[late-graduate] markers ──►│
│     classify (Item 5)                    │
│     Case A → inline op (LLM handles)     │
│     Case B → CPU glue                    │
│     Case C → late-graduate (Item 4)      │
└──────────────────────────────────────────┘
```

## Item-by-item reference

| Item | Module | Public surface |
|---|---|---|
| 1 | `agentic/probe.py` | `compare_hf_tt_probes` — pure comparator; the runner / log / persist helpers live in `cli.py` |
| 2 | `_cli_helpers/llm_verify.py` | `run_llm_verify_pass` + `build_verify_prompt` + `parse_verify_verdict` |
| 3 | `_cli_helpers/e2e_synthesizer.py` | `run_e2e_synthesis_loop` + `build_synthesis_prompt` + `extract_late_discovery_markers` |
| 4 | `_cli_helpers/late_graduate.py` | `run_late_graduate` + `LateGraduateComponentSpec` |
| 5 | `_cli_helpers/late_discovery_classifier.py` | `run_classify_pass` + `heuristic_classify` + `MissingPieceClassification` |
| 6 | `_cli_helpers/family_template_registry.py` | `register_template` + `find_template_for_model` + `demote_template` |
| 7 | `_cli_helpers/template_promotion.py` | `mark_promoted` + `auto_promote_after_register` + `is_template_promoted` |
| 8 | `_cli_helpers/e2e_orchestrator.py` | `run_e2e_bringup` — chains Items 1–7 |

## Activation (production use)

The orchestrator is opt-in until exercised on real bring-ups. To enable:

```bash
export TT_HW_PLANNER_USE_E2E_ORCHESTRATOR=1
python -m scripts.tt_hw_planner auto-up <model_id> --box QB2
```

Without the env var, the legacy Path A flow runs unchanged. Template
registration (Item 6 / 7) still fires on Path A success so the registry
populates regardless of the orchestrator flag.

## CLI management commands

```bash
# List registered chained templates
python -m scripts.tt_hw_planner template-list

# Include demoted entries
python -m scripts.tt_hw_planner template-list --all

# Force-promote a template (skip multi-model threshold)
python -m scripts.tt_hw_planner template-promote <family_key>

# Demote a regressed template (forces re-synthesis next bring-up)
python -m scripts.tt_hw_planner template-demote <family_key> --reason "HF v5 regression"
```

## Status labels

| Status | Meaning | Action |
|---|---|---|
| `TEMPLATE_REUSED` | promoted family template used as-is | run demo, done |
| `VERIFY_PASSED` | LLM static review accepted existing chain | run demo, done |
| `SYNTHESIS_CONVERGED` | LLM rewrote chain, end-to-end PCC ≥ 0.99 | register template; promote if threshold met |
| `SYNTHESIS_FAILED` | synthesis exhausted iter budget | surface diagnostic; do NOT promote |
| `ERROR` | orchestrator crashed | UNVERIFIED outcome; fall through to legacy flow |

## Failure semantics (everything is best-effort)

- Every step has injectable seams (LLM, pytest, HF). Tests use stubs;
  real callers use defaults.
- Every step's exception is caught and degrades to `None` / `FAIL` so
  the caller's escalation path is never blocked.
- Synthesis failure never promotes a template (no false sibling
  confirmations).
- Demoted templates are skipped by default — operator promotes via
  CLI when ready.

## Known limitations

- `_make_default_pytest_runner` injects `HF_MODEL`/`PLANNER_TARGET_HF_MODEL`
  into the subprocess env, but doesn't reuse `_run_focused_pytest`'s
  kill-stale + device-reset logic. Sufficient for the synthesis loop's
  single-test target; revisit if synthesis grows to multi-test runs.
- `resolve_hf_forward_source` (Item 2) uses `inspect.getsource` on the
  HF class; may fail on `trust_remote_code=True` models with unusual
  inheritance. Caller treats `None` return as "skip verify, fall through
  to synthesis".
- The orchestrator path runs in addition to the legacy strict gate, not
  in place of it. Both fire; their outcomes compose. Flip the default
  once exercised on real bring-ups.
- Late-graduate Case C execution (`run_late_graduate`) requires a
  `component_iterate` callable from the cli layer; the synthesizer
  accumulates Case-C discoveries in `result.late_discoveries` for the
  orchestrator (or its caller) to execute.
