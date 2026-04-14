# LLK CodeGen

AI-powered kernel generation for Tenstorrent LLK (Low-Level Kernel) libraries. Ports kernels from a reference architecture (Blackhole) to a target architecture (Quasar) using a multi-agent pipeline orchestrated by Claude Code.

## Quick Start

```bash
cd codegen
claude
> Generate gelu for Quasar
```

## Setup

### 1. Test Environment (required)
```bash
cd ../tests
./setup_testing_env.sh
```

### 2. Atlassian Access (optional)
On first use, Claude will prompt you to authenticate with Atlassian.
The MCP server is pre-configured in `.mcp.json`.

## Usage

| Command | Description |
|---------|-------------|
| `Generate sigmoid for Quasar` | Generate SFPU kernel |
| `Generate reduce for Quasar` | Generate math kernel |
| `Generate pack_untilize for Quasar` | Generate pack kernel |

### Batch Generation

```bash
./scripts/batch_generate.sh --wave 1              # all wave 1 sequentially
./scripts/batch_generate.sh --wave 1 --parallel    # all wave 1 in parallel
./scripts/batch_generate.sh --kernel abs           # single kernel
```

---

## Architecture

### Pipeline Overview

```
                          "Generate fill for Quasar"
                                    |
                                    v
                        ┌───────────────────────┐
                        │     ORCHESTRATOR       │
                        │   (codegen/CLAUDE.md)  │
                        │                        │
                        │  Spawns agents in      │
                        │  sequence, tracks       │
                        │  metrics, manages       │
                        │  phase loop             │
                        └───────────┬─────────────┘
                                    |
           ┌────────────────────────┼──────────────────────────┐
           |                        |                          |
           v                        v                          v
    ┌─────────────┐        ┌──────────────┐          ┌──────────────────┐
    │  Step 0     │        │  Step 1      │          │  Step 2          │
    │  SETUP      │        │  RESEARCH    │          │  ANALYZE         │
    │             │───────>│              │─────────>│                  │
    │  Create     │        │  Arch Lookup │          │  Analyzer        │
    │  LOG_DIR,   │        │  Agent       │          │  Agent           │
    │  metrics    │        │              │          │                  │
    └─────────────┘        └──────────────┘          └────────┬─────────┘
                                                              |
                                                              v
                                              ┌───────────────────────────┐
                                              │       PHASE LOOP          │
                                              │  (one per sub-kernel)     │
                                              │                           │
                                              │  For each phase 1..N:     │
                                              └─────────────┬─────────────┘
                                                            |
                     ┌──────────────────────────────────────┼───────────┐
                     |                                      |           |
                     v                                      v           v
              ┌─────────────┐                      ┌──────────────┐
              │  Step 3a    │                      │  Step 3b     │
              │  PLAN       │─────────────────────>│  WRITE       │
              │             │                      │              │
              │  Planner    │                      │  Writer      │
              │  Agent      │                      │  Agent       │
              └─────────────┘                      └──────┬───────┘
                                                          |
                                                   compile check
                                                          |
                                            ┌─────────────┼─────────────┐
                                            |  PASS       |             |  FAIL
                                            v             |             v
                                     ┌─────────────┐     |     ┌──────────────┐
                                     │  Step 3d    │     |     │  Step 3c     │
                                     │  TEST WRITE │     |     │  DEBUG       │
                                     │  (if needed)│     |     │              │
                                     │  Test Writer│     |     │  Debugger    │
                                     │  Agent      │     |     │  Agent       │
                                     └──────┬──────┘     |     └──────┬───────┘
                                            |            |            |
                                            v            |      fix + recompile
                                     ┌─────────────┐    |            |
                                     │  Step 3e    │<───┘────────────┘
                                     │  TEST       │
                                     │             │
                                     │  Phase      │
                                     │  Tester     │
                                     └──────┬──────┘
                                            |
                                     ┌──────┼──────┐
                                     |  PASS       |  FAIL
                                     v             v
                              next phase      debug + retest
                              or done         (max 2 cycles)
                                     |
                                     v
                        ┌───────────────────────┐
                        │  Step 4               │
                        │  FINAL REGRESSION     │
                        │                       │
                        │  Regression Tester    │
                        │  (full test suite)    │
                        └───────────┬───────────┘
                                    |
                                    v
                        ┌───────────────────────┐
                        │  Step 5               │
                        │  OPTIMIZE             │
                        │  (SFPU only, optional)│
                        │                       │
                        │  Optimizer Agent       │
                        │  (replay buffers)     │
                        └───────────┬───────────┘
                                    |
                                    v
                        ┌───────────────────────┐
                        │  Step 9b              │
                        │  FORMAT               │
                        │                       │
                        │  pre-commit hooks     │
                        └───────────┬───────────┘
                                    |
                                    v
                        ┌───────────────────────┐
                        │  Step 10              │
                        │  REPORT               │
                        │                       │
                        │  Write metrics to     │
                        │  runs.jsonl + LOG_DIR │
                        └───────────────────────┘
```

### Agent Inventory

| Agent | Playbook | Purpose | Inputs | Outputs |
|-------|----------|---------|--------|---------|
| **Arch Lookup** | `llk-arch-lookup.md` | Fetch architecture docs from Confluence + DeepWiki | Kernel name, type | `artifacts/{op}_arch_research.md` |
| **Analyzer** | `llk-analyzer.md` | Analyze reference implementation, identify sub-kernel phases | Reference kernel, arch research | `artifacts/{op}_analysis.md` |
| **Planner** | `llk-planner.md` | Design implementation spec with instruction mappings | Analysis, arch research | `artifacts/{op}_phase{N}_spec.md` |
| **Writer** | `llk-kernel-writer.md` | Generate kernel code from spec, run compile check | Phase spec | `tt_llk_{arch}/.../ckernel_sfpu_{op}.h` |
| **Debugger** | `llk-debugger.md` | Fix compilation or runtime errors (max 5 attempts) | Kernel + error output | Fixed kernel file |
| **Test Writer** | `llk-test-writer.md` | Create C++ + Python test (or extend existing multi-op test) | Kernel file | Test files in `tests/` |
| **Phase Tester** | `llk-phase-tester.md` | Run per-phase simulator tests | Compiled kernel | Test pass/fail |
| **Regression Tester** | `llk-regression-tester.md` | Run full test suite after all phases complete | Complete kernel | Final pass/fail |
| **Optimizer** | `llk-optimizer.md` | Add replay buffers for SFPU kernels | Tested kernel | Optimized kernel |
| **Prettifier** | `llk-prettifier.md` | Refactor for readability (currently disabled) | Working kernel | Cleaned kernel |

### Data Flow Between Agents

```
                    Confluence / DeepWiki
                            |
                            v
                    ┌──────────────┐
                    │  Arch Lookup │──> artifacts/{op}_arch_research.md ──┐
                    └──────────────┘                                      |
                                                                         |
                    Reference kernel                                     |
                    (tt_llk_blackhole/)                                   |
                            |                                            |
                            v                                            v
                    ┌──────────────┐                              ┌─────────────┐
                    │   Analyzer   │──> artifacts/{op}_analysis.md│             │
                    └──────────────┘              |               │             │
                                                  v               │             │
                                           ┌─────────────┐       │   Shared    │
                                           │   Planner   │<──────│   Context   │
                                           └──────┬──────┘       │             │
                                                  |               │             │
                                    artifacts/{op}_phase{N}_spec.md             │
                                                  |               │             │
                                                  v               │             │
                                           ┌─────────────┐       │             │
                                           │   Writer    │<──────│             │
                                           └──────┬──────┘       └─────────────┘
                                                  |
                                    tt_llk_quasar/.../ckernel_sfpu_{op}.h
                                                  |
                              ┌───────────────────┼───────────────────┐
                              v                   v                   v
                       ┌─────────────┐     ┌─────────────┐     ┌───────────┐
                       │  Debugger   │     │ Test Writer │     │  Phase    │
                       │  (if error) │     │ (if needed) │     │  Tester   │
                       └─────────────┘     └─────────────┘     └───────────┘
```

### Artifact Files

Each run produces these artifacts in `codegen/artifacts/`:

| Artifact | Producer | Consumer | Content |
|----------|----------|----------|---------|
| `{op}_arch_research.md` | Arch Lookup | Analyzer, Planner, Debugger | ISA details, register layouts, instruction constraints |
| `{op}_analysis.md` | Analyzer | Planner, Test Writer | Reference breakdown, sub-kernel phases, format support |
| `{op}_phase{N}_spec.md` | Planner | Writer | Instruction mappings, function signatures, pseudocode |
| `{op}_report.md` | Orchestrator | User | Final summary with metrics |

### Run Logging

Each run creates a self-contained directory at `/proj_sw/user_dev/llk_code_gen/quasar/{RUN_ID}/`:

```
2026-04-01_fill_quasar_81473bae/
├── run.json                        # Full metrics (tokens, timing, phases, failures)
├── cli_output.json                 # Raw Claude CLI JSON output
├── instructions/                   # Snapshot of agent playbooks used
│   ├── llk-analyzer.md
│   ├── llk-planner.md
│   └── ...
├── agent_analyzer.md               # Agent reasoning logs
├── agent_planner.md
├── agent_writer.md
├── agent_test_writer.md
├── agent_phase_tester.md
├── fill_arch_research.md           # Artifact copies
├── fill_analysis.md
├── fill_phase1_spec.md
├── fill_phase2_spec.md
├── fill_phase3_spec.md
├── fill_report.md
├── ckernel_sfpu_fill.h             # Generated kernel
├── ref_ckernel_sfpu_fill.h         # Reference kernel (for comparison)
├── sfpu_fill_quasar_test.cpp       # Generated test source
├── test_sfpu_fill_quasar.py        # Generated test script
└── emu_*.log                       # Simulator logs
```

All runs are also indexed in `runs.jsonl` (one JSON line per run) for dashboarding.

### Key Design Principles

1. **Discovery over hardcoding** -- Agents fetch architecture details from Confluence/DeepWiki at runtime rather than relying on embedded knowledge
2. **Incremental phases** -- Write one sub-kernel, compile, test, repeat. Never write the whole file at once.
3. **Instruction encoding drives API** -- `TTI_` macros require compile-time constants; function parameter types must preserve constexpr-ness
4. **Target-first design** -- Use reference for semantics only; derive implementation patterns from existing target code
5. **Prefer extending existing tests** -- Add to multi-op tests (e.g., `test_sfpu_nonlinear_quasar.py`) rather than creating new test files when compatible
