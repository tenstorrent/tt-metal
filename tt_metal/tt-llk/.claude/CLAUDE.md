# TT-LLK — AI Assistant Instructions

## Repository

Header-only library for testing Tensix Kernels across three architectures: Wormhole B0, Blackhole, and Quasar.

### Tensix

Specialized processing unit for AI workloads containing five 32-bit RISC-V cores (B, T0, T1, T2, NC) and a 3-way threaded coprocessor:
- **T0 (Unpack)**: Data movement from L1 to register files with format conversion
- **T1 (Math)**: Matrix multiplication, element-wise operations (FPU), and 32-lane SIMD (SFPU/Vector Unit)
- **T2 (Pack)**: Data movement from register files to L1 with format conversion

Execution model: RISC-V cores push instructions to corresponding coprocessor threads. Each thread has an independent frontend pipeline; the shared backend executes across threads with re-ordering.

### Key Directories

| Path | Contents |
|------|----------|
| `tt_llk_wormhole_b0/` | Wormhole B0 — llk_lib, instructions, common headers |
| `tt_llk_blackhole/` | Blackhole — same structure as WH |
| `tt_llk_quasar/` | Quasar — different file naming (semantic, not letter-based) |
| `tests/python_tests/` | Python test suite |
| `tests/sources/` | C++ test source files |
| `tests/hw_specific/` | Architecture-specific test files |
| `docs/` | Documentation |

### Architecture File Naming

- **WH/BH** (letter-based): `llk_unpack_A.h`, `llk_unpack_AB.h`, `llk_math_eltwise_binary_sfpu.h`
- **Quasar** (semantic): `llk_unpack_unary_operand.h`, `llk_unpack_binary_operands.h`, `llk_math_eltwise_binary_broadcast.h`

## Skills & Agents

**MANDATORY**: Before dispatching agents or taking action for the triggers below, you MUST first invoke the corresponding skill (or `Read` its `SKILL.md`) and follow the instructions inside it. Never manually dispatch sage agents without first loading the relevant skill — the skill defines the correct orchestration pattern, prompt structure, and quality checklist.

**Note:** Skills live in `.claude/skills/<name>/SKILL.md` relative to this file. Claude Code auto-discovers them when launched from inside `tt-llk/` (nested-subdirectory discovery). If the `Skill` tool doesn't list them, fall back to `Read(".claude/skills/<name>/SKILL.md")`.

| Trigger | Skill | What it does |
|---------|-------|--------------|
| Architecture, instruction, or LLK questions | `arch-lookup` (`.claude/skills/arch-lookup/SKILL.md`) | Orchestrates sage agents in parallel, aggregates results |
| Running tests | `run-test` (`.claude/skills/run-test/SKILL.md`) | Dispatches llk-test-runner with correct scenario flags |
| Debugging kernel errors | `debug-kernel` (`.claude/skills/debug-kernel/SKILL.md`) | Dispatches llk-debugger with inferred arch/kernel type |
| Porting kernels between architectures | `port-kernel` (`.claude/skills/port-kernel/SKILL.md`) | Launches source + target sages, reads test harness |

Trigger examples for `arch-lookup`:
- "How does SFPMAD work?", "What is BroadcastType?", "How does unpack handle Float16?"
- "How do T0/T1/T2 synchronize?", "What does [instruction] do?"
- "Explain [LLK function] parameters", "What are the differences between WH and QSR?"
- Any question about Tensix hardware, ISA instructions, or LLK implementation details

## Coding Style

- **`const` placement**: write `const <type>` (e.g. `const uint32_t x`), never `<type> const` (e.g. `uint32_t const x`). Semantics are identical, but the codebase uses `const <type>` everywhere — match it. Applies to all type qualifiers in the same position (e.g. `volatile`, `constexpr` modifiers on declarations).

### Documentation

We use lightweight Doxygen docstrings — high-signal, low-noise (`@brief`, `@param`, `@tparam`, `@ref`, `@pre`, `@post`), with `@pre`/`@post`/`@ref` encoding the init/execute/uninit contract and pairing the per-thread (T0/T1/T2) halves of an op. Avoid bloat tags (`@details`, `@author`, `@date`, `@version`, `@todo`, `@remark`, and `@return` on void functions). Applies to LLK lib (`_llk_*`), LLK API (`llk_*`), and the Compute API (`tt_metal/hw/inc/api/compute/`) — the Compute API keeps its published prose+table format (it feeds the public Sphinx docs), not `@param` tags.

**When writing or updating docstrings**, follow `.claude/references/doxygen-style.md`.

## Test Infrastructure

### Two-Phase Test Flow
1. **Compile-producer**: `pytest --compile-producer -n N -x ./test_name.py` — compiles all variants in parallel
2. **Compile-consumer**: `pytest --compile-consumer -x ./test_name.py` — runs compiled variants on hardware

### Key Concepts
- `CHIP_ARCH` env var selects the target architecture (`blackhole`, `wormhole`, `quasar`)
- Tests run from the `tests/` directory
- Test isolation: tests can affect each other via HW state leaking between kernel reconfigurations (reconfig escapes)
- Key files: `tests/python_tests/conftest.py`, `tests/python_tests/helpers/test_config.py`

### Device Reset
When tests fail with `TENSIX TIMED OUT`, the device may be hung. Reset with:
```bash
tt-smi -r  # reset device
```
Only reset for runtime failures (TIMEOUT, runtime ASSERTION). Never reset for compile errors or reconfig escapes — resetting masks reconfig bugs.

## Metal Integration

LLK is consumed by tt-metal through a 4-layer stack. **When you change an LLK function signature, add a new op, or modify unpack/pack behavior**, you must also update the corresponding metal files.

**Mandatory:** Before completing any LLK change, read `.claude/references/metal-integration.md` and follow the propagation checklist.

Key layers above tt-llk:
1. **CKernels LLK API** (`tt_metal/hw/ckernels/{arch}/metal/llk_api/`) — almost always needs updating
2. **Compute API** (`tt_metal/hw/inc/api/compute/`) — update if the public interface changes
3. **TTNN bypass files** — some TTNN operations directly include LLK headers, bypassing the API

## MCP Integration

This repo uses three MCP servers: DeepWiki, Atlassian (Confluence), and Glean.
- **DeepWiki** — ISA documentation for WH/BH (NOT available for Quasar)
- **Atlassian** — Confluence hardware documentation
- **Glean** — Internal documentation search
