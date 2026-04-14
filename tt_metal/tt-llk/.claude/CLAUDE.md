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

<context-specifics>
    <rule description="Use for architecture, instruction, or LLK implementation questions. Orchestrates sage agents.">
        @arch-lookup
        <trigger-examples>
            - "How does SFPMAD work?"
            - "What is BroadcastType?"
            - "How does unpack handle Float16?"
            - "How do T0/T1/T2 synchronize?"
            - "What is the L1 size/latency?"
            - "What does [instruction] do?"
            - "Explain [LLK function] parameters"
        </trigger-examples>
    </rule>
    <rule description="Use when running tests. Delegate to llk-test-runner.">
        @run-test
    </rule>
    <rule description="Use when debugging kernel compilation or runtime errors.">
        @debug-kernel
    </rule>
    <rule description="Use when porting a kernel between architectures.">
        @port-kernel
    </rule>
</context-specifics>

## Test Infrastructure

### Two-Phase Test Flow
1. **Compile-producer**: `pytest --compile-producer -n N -x ./test_name.py` — compiles all variants in parallel
2. **Compile-consumer**: `pytest --compile-consumer -x ./test_name.py` — runs compiled variants on hardware

### Key Concepts
- `CHIP_ARCH` env var selects the target architecture (`blackhole`, `wormhole`, `quasar`)
- Tests run from the `tests/` directory
- Logs: `/tmp/llk_test/compile.log` (compilation), `/tmp/llk_test/run.log` (execution)
- Test isolation: tests can affect each other via HW state leaking between kernel reconfigurations (reconfig escapes)
- Key files: `tests/python_tests/conftest.py`, `tests/python_tests/helpers/test_config.py`

### Device Reset
When tests fail with `TENSIX TIMED OUT`, the device may be hung. Reset with:
```bash
tt-smi -ls          # find PCI device ID
tt-smi -r <PCI_ID>  # reset device
```
Only reset for runtime failures (TIMEOUT, runtime ASSERTION). Never reset for compile errors or reconfig escapes — resetting masks reconfig bugs.

## MCP Integration

This repo uses three MCP servers: DeepWiki, Atlassian (Confluence), and Glean.
- **DeepWiki** — ISA documentation for WH/BH (NOT available for Quasar)
- **Atlassian** — Confluence hardware documentation
- **Glean** — Internal documentation search
