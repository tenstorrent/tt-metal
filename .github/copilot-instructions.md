# TT-Metal / TT-Metalium — Copilot PR Review Instructions

> Path-specific review criteria live in `.github/instructions/`.
> This file covers cross-cutting concerns that apply to every PR.

## Codebase Snapshot

| Path | What lives there |
|------|-----------------|
| `tt_metal/` | Core runtime, device APIs, dispatch, allocators, firmware |
| `tt_metal/hw/` | Firmware, SOC descriptors, hardware includes |
| `tt_metal/hw/ckernels/` | Compute kernels (math, unpack, pack) |
| `tt_metal/tt-llk/` | Low-level kernel library (SFPU ops, per-architecture) |
| `ttnn/` | High-level op layer, Python/C++ integration (nanobind) |
| `tt-train/` | Training library built on ttnn |
| `models/` | Model implementations and demos |
| `tools/` | Profiler, debugger, scaleout tooling |
| `.github/` | CI/CD workflows and infra |

## Review Language

Respond in **English**. Be terse. Use code blocks for every actionable diff.

## Review Priorities

### 🔴 CRITICAL (Block merge)
- **Correctness**: logic errors, data corruption risks, race conditions
- **ABI Breakage**: struct layout change, symbol deleted from a public header in `tt_metal/api/` or `ttnn/api/ttnn/`
- **Kernel Safety**: missing bounds check on L1 tile addressing; broken synchronization barrier order
- **Security**: hardcoded credentials, secrets, or tokens anywhere in the diff

### 🟡 IMPORTANT (Requires discussion)
- **Missing test coverage** for new public API or changed behavior
- **New dependency** added without infra team awareness
- **API contract change** without versioning or deprecation path

### 🟢 SUGGESTION (Non-blocking)
- **Naming and readability**: names that don't match surrounding conventions
- **Simplification**: complex logic that could be expressed more clearly

## Code Quality Principles

### Names must reflect actual behavior
A function named `write_to_all_chips()` that only writes to one chip is misleading and reviewable as a defect. Names are documentation — if the implementation scope narrows or widens, the name must track it. Flag any mismatch between what a symbol promises and what it delivers.

### Flag duplication — suggest commonization
When the same logic appears in more than one place, it will inevitably drift. Flag duplicated code blocks and suggest extracting a shared helper. Constants that appear in both a header and a builder file should live in one canonical location.

### Magic numbers require a derivation
Bare numeric literals in code are invisible assumptions. Every hardcoded offset, size, or threshold should either be derived from a named constant or accompanied by a comment explaining where the value comes from and under what conditions it might change.

### Complex conditions belong in named variables
When an `if` condition involves multiple conjuncts or non-obvious logic, hoist it into a descriptively named `bool`. The variable name serves as the comment the reader would otherwise have to reconstruct mentally.

```cpp
// Difficult to parse at review time
if (conn_type == FabricConnectionType::Transient && channel_idx == 0 && !is_mux_target) { ... }

// Clear intent
const bool is_transient_direct_to_router = conn_type == FabricConnectionType::Transient
                                        && channel_idx == 0
                                        && !is_mux_target;
if (is_transient_direct_to_router) { ... }
```

## Comment Format

Use this format for every finding:

````
**[🔴/🟡/🟢] Category: Short title**

What the issue is and where (file:line).

**Why it matters:** one sentence on impact.

**Suggested fix:**
```cpp
// minimal diff
```
````

## Testing Expectations

- New public API → unit test in the nearest `tests/` target
- Changed behavior in a hot path → micro-benchmark or reference to existing perf harness
- Bug fix → regression test that would have caught the original bug

## Security & Reliability

- **No secrets**: flag any hardcoded token, key, password, or internal IP with a port
- **Kernel/firmware**: bounds checks on all NOC address calculations; mailbox and semaphore release ordering must be preserved exactly

## Docs-Only PRs

Title must be prefixed with `[skip ci]` to avoid running unnecessary CI checks.
