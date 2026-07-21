# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Git Policy: Read-Only

Read-only git commands are allowed (`git rev-parse`, `git log`, `git status`, `git diff`, `git show`). **NEVER push, commit, checkout, restore, reset, or otherwise modify** the repo via git.

## LLK CodeGen System

To generate kernels for a target architecture, start Claude from the `codegen/` folder:
```bash
cd codegen
claude
> Generate gelu for Quasar
```

The codegen system discovers architectural patterns dynamically from Confluence, DeepWiki, assembly.yaml, and existing implementations. Each target architecture has its own orchestrator and agents under `codegen/agents/{arch}/`. See [codegen/CLAUDE.md](codegen/CLAUDE.md) for the routing table.

### Editing CodeGen Workflows & Agent Playbooks

**IMPORTANT: following these rules is MANDATORY.** When changing CodeGen workflow, orchestrator, or agent playbook files:

- Replace the whole sentence with a correct one — do not patch around the existing wording.
- Write every instruction as a command to follow; do not explain why it changed.
- Keep it terse — do not overexplain.

For the Quasar workflow specifically (`codegen/agents/quasar/*.md` and its step scripts):

- Derive inputs from what the orchestrator passes and what `state.py` provides; do not hand-construct paths or duplicate state the store already holds.
- Scope architecture- or kernel-type-specific rules to the type they apply to (e.g. "For SFPU kernels, …").

## Repository Structure

```
tt-llk/
├── tt_llk_quasar/           # Quasar LLK implementations
├── tt_llk_blackhole/        # Blackhole LLK (reference)
├── tt_llk_wormhole_b0/      # Wormhole LLK (reference)
├── codegen/                 # AI code generation system
│   ├── CLAUDE.md            # Pointer to orchestrator
│   ├── agents/{arch}/       # Per-arch orchestrator + agent playbooks
│   ├── references/          # Knowledge base
│   ├── artifacts/           # Generated artifacts
│   └── scripts/             # Tools
└── tests/                   # Test infrastructure
```

## Commands

### Environment Setup

Check for `tests/.venv` to decide which setup applies:

- **`tests/.venv` exists** — external setup was used previously (`source ./setup_external_testing_env.sh` created the venv and installed `requirements.txt`). Activate it before running tests: `source tests/.venv/bin/activate`.
- **No `tests/.venv`** — you are inside the **tt-metal Docker image** (the default for work; Python test deps like `ml_dtypes`, `torch`, `tt-exalens` are already installed system-wide). Just run the arch-specific setup with `CHIP_ARCH`:

  ```bash
  cd tests
  CHIP_ARCH=quasar ./setup_testing_env.sh   # fetches SFPI for the target arch; no venv needed
  ```

See `tests/README.md` for details.

### MCP Servers
Pre-configured in `.mcp.json`. Atlassian requires authentication on first use.
- **Atlassian** — Primary source for architecture knowledge. See `codegen/skills/llk-arch-lookup/SKILL.md` for full page index. Key pages:
  - Page ID `48300268` — uarch tree root (80+ sub-pages with detailed microarchitecture)
  - Page ID `1256423592` — Quasar/Trinity SFPU Micro-Architecture Spec (primary SFPU reference)
  - Page ID `1170505767` — Tensix SFPU Instruction Set Architecture (per-instruction details)
  - Page ID `1613201604` — Tensix ISA (164 child pages, one per instruction)
  - Page ID `84508873` — Tensix NEO High Level Specification (general architecture overview)
  - Page ID `1612808713` — REPLAY instruction (replay buffer for ITERATIONS loops)
- **DeepWiki** — Query `tenstorrent/tt-isa-documentation` for reference architecture ISA docs

### Compilation Check
```bash
cd codegen
source ../tests/.venv/bin/activate
CHIP_ARCH=quasar python scripts/compiler.py ../tests/sources/quasar/sfpu_{op}_quasar_test.cpp \
    -t "MATH_OP(mathop=MathOperation.{Op})" \
    -t "APPROX_MODE()" \
    -r "TILE_COUNT(1)" \
    -r "NUM_FACES()" \
    -v
```
Template (`-t`) and runtime (`-r`) params use classes from `tests/python_tests/helpers/test_variant_parameters.py`.

### Run Tests

Tests run as a **two-step compile-then-run flow**:

1. **Compile step** — `--compile-producer -n 15` (no `--run-simulator`). Builds ELFs for all selected variants in parallel via pytest-xdist; never touches the simulator.
2. **Simulator step** — `--run-simulator --compile-consumer` (no `-n 15`; xdist is not supported under the simulator). Consumes the pre-built ELFs and executes them.

```bash
source tests/.venv/bin/activate
cd tests/python_tests/quasar

# Step A: parallel compile
CHIP_ARCH=quasar pytest -x --compile-producer -n 15 test_{op}_quasar.py

# Step B: run on simulator
TT_UMD_SIMULATOR_PATH=/proj_sw/user_dev/$USER/tt-umd-simulators/build/emu-quasar-1x3 CHIP_ARCH=quasar pytest -x --run-simulator --compile-consumer --port=5556 test_{op}_quasar.py
```

When running inside codegen agents, never call pytest or flock directly — use `.claude/scripts/run_test.sh` (`count` / `compile` / `simulate` / `run`), which serialises every invocation on one global lock (`/tmp/tt-llk-test.lock`), fetches SFPI when missing, sets `TT_UMD_SIMULATOR_PATH`, reaps stale emulator jobs, watches the run and kills a hang gracefully (exit 5). It is a single blocking call — invoke it like pytest and wait for the verdict; there is no resume loop and no timeout to tune (the only wait is on the lock). See `.claude/skills/run-test/SKILL.md`.

## Key Files

| File | Purpose |
|------|---------|
| `tt_llk_quasar/instructions/assembly.yaml` | Quasar ISA (277KB, use grep) |
| `tt_llk_blackhole/common/inc/sfpu/*.h` | Reference SFPU implementations |
| `tt_llk_quasar/common/inc/sfpu/*.h` | Quasar SFPU implementations |
