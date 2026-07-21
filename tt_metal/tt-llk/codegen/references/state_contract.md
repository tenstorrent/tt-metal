# Quasar CodeGen — `state.py` Dependency Graph

`boot` = `--worktree-dir` file · `run` = `--log-dir "$LOG_DIR"` file.
`★` = agent must write those keys back into `run`.
`SKIP_TESTER` (boolean, from router; default `false`): when true the writer validates the kernel against its existing tests in a 5-run fix loop and writes the `★` tester counts itself; the orchestrator skips the tester and refiner and routes the writer's PASSED/FAILED straight to optimizer / failed.
`HIDE_EXISTING_KERNEL` (boolean, from router; default `false`): when true, `execute_step_hide_existing_kernel` (Step 2b) git-removes and commits the target op's existing files on the worktree branch before the analyzer runs, so the pipeline regenerates blind. `execute_step_setup_run` mirrors it from `boot` into `run` exactly like `SKIP_TESTER`.

## Orchestrator ⇄ agents

Left box into an agent = what it consumes · right box out = what it produces.

```mermaid
flowchart LR
  ROUTER["router (begin_setup → run.json step=setup, pre-worktree)"] --> RB["KERNEL_NAME, TARGET_ARCH, SFPI_MODE, SKIP_TESTER, HIDE_EXISTING_KERNEL,<br/>WORKTREE_BRANCH, LOG_DIR_BASE,<br/>LOG_DIR, RUN_ID, START_TIME (from begin_setup)"] --> ORCH(["orchestrator"])

  ORCH --> aI["kernel identity"] --> ANA["analyzer"] --> aO["analysis.md,<br/>error line"] --> ORCH
  ORCH --> wI["analysis, CYCLE,<br/>GENERATED_KERNEL, SKIP_TESTER"] --> WR["writer"] --> wO["PASSED / FAILED, error line, compile count<br/>(SKIP_TESTER: also TESTS_TOTAL, TESTS_PASSED,<br/>TESTER_COMPILE_COUNT, PHASE_DEBUGS)"] --> ORCH
  ORCH --> tI["compiled kernel,<br/>CYCLE, LOG_DIR"] --> TST["tester ★"] --> tO["PASS / STUCK / ENV_ERROR,<br/>TESTS_TOTAL, TESTS_PASSED, TESTS_GENERATED,<br/>TESTER_COMPILE_COUNT, PHASE_DEBUGS,<br/>FORMATS_TESTED_JSON, FORMATS_EXCLUDED_JSON"] --> ORCH
  ORCH --> rI["PREV_RESULT,<br/>failure summary, CYCLE"] --> RE["refiner"] --> rO["REFINED / ESCALATE,<br/>reason"] --> ORCH
  ORCH --> oI["passing kernel,<br/>SFPI_MODE"] --> OP["optimizer ★"] --> oO["OPTIMIZED,<br/>OPTIMIZATION_TYPE"] --> ORCH
  ORCH --> fI["kernel"] --> FM["format"] --> ORCH

  ORCH --> FIN["finalize → run.json + runs.jsonl"]

  classDef inb fill:#eef,stroke:#88a;
  classDef outb fill:#efe,stroke:#8a8;
  classDef star fill:#fde,stroke:#c39,stroke-width:2px;
  class RB,aI,wI,tI,rI,oI,fI inb;
  class aO,wO,tO,rO,oO outb;
  class TST,OP star;
```

## Orchestrator internal state (run file, in order)

```mermaid
flowchart TB
  Z["begin_setup (pre-worktree, router)<br/>RUN_ID, LOG_DIR, START_TIME, KERNEL_NAME, TARGET_ARCH<br/>→ run.json status=running, step=setup"]
  A["setup_run<br/>reuses RUN_ID/LOG_DIR/START_TIME from begin_setup;<br/>WORKTREE_DIR, GIT_COMMIT, CODEGEN_VERSION, PROMPT, BATCH_ID, MODEL, RUN_TYPE"]
  B["set_kernel_identity<br/>KERNEL_TYPE, REF_ARCH, KERNEL_PATH, GENERATED_KERNEL"]
  C["write_initial_run_json<br/>advance setup → analyzer + patch kernel identity;<br/>seeds SESSION_ID, PROJECT_CWD, CYCLE, MAX_CYCLES, REFINEMENT_COUNT,<br/>COMPILATION_ATTEMPTS, DEBUG_CYCLES, PHASES_TOTAL, PHASES_COMPLETED,<br/>LINES_GENERATED, TOKENS_JSON, OBSTACLE, +★ defaults"]
  D["writer/tester/refiner loop<br/>PHASE_COMPILES, PHASE_COMPILE_ERRORS_JSON, PHASE_TEST_DETAILS,<br/>PREV_RESULT, STATUS, FINAL_RESULT, error/diagnosis/reason lines"]
  E["finalize → run.json + runs.jsonl"]
  Z --> A --> B --> C --> D --> E
```
