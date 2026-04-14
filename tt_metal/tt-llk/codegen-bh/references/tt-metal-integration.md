# tt-metal Integration Reference

tt-metal is the primary consumer of our LLK kernels. Understanding how tt-metal wraps and calls our LLK functions is critical for ensuring API compatibility.

---

## Relationship

```
User kernel code
    ↓ calls
Compute API  (tt_metal/hw/inc/api/compute)
    ↓ calls
LLK API      (tt_metal/hw/ckernels/blackhole/metal/llk_api)
    ↓ calls
LLK kernels  (tt-llk: tt_llk_blackhole/)
```

Our LLK kernels are the lowest layer. The LLK API in tt-metal wraps our `_llk_*` internal functions with public-facing signatures. The compute API is the user-facing layer above that.

---

## Key Paths in tt-metal (Blackhole)

| Path | What It Contains |
|------|-----------------|
| `tt_metal/hw/ckernels/blackhole/metal/llk_api/` | BH LLK API wrappers — these call our `_llk_*` functions directly. **Primary source for expected function signatures.** |
| `tt_metal/hw/inc/api/compute/` | Compute API — higher-level functions that users call in their kernels. Shows how LLK API is invoked with specific template params and defaults. |

---

## How to Query tt-metal

Use the deepwiki MCP tool to ask questions about tt-metal:

```
mcp__deepwiki__ask_question
  repo: "tenstorrent/tt-metal"
  question: "How does the Blackhole LLK API call _llk_math_{op}_ ? What template parameters and arguments does it pass?"
```

```
mcp__deepwiki__ask_question
  repo: "tenstorrent/tt-metal"
  question: "What does the compute API for {op} look like in Blackhole? What parameters does it expose?"
```

---

## What to Look For

### In LLK API wrappers (`llk_api/`)
- **Function signatures**: What arguments and template params does tt-metal pass to our `_llk_*` functions?
- **Default values**: What defaults does tt-metal assume for optional params?
- **Template instantiations**: What concrete template param combinations does tt-metal use?
- **Init/uninit pairing**: Does tt-metal always call uninit after init? What args does it pass?

### In Compute API (`compute/`)
- **User-facing interface**: What does the end user actually call? This shows the "intent" behind the operation.
- **Parameter mapping**: How do user-facing params map to LLK API params?
- **Calling conventions**: Order of init → configure → execute → uninit calls.

---

## When to Use This

| Agent | When to Query tt-metal |
|-------|----------------------|
| **Analyzer** | During Step 1.5 — check tt-metal's LLK API wrappers alongside the test harness to understand the full customer contract |
| **Planner** | When designing function signatures — verify planned API matches what tt-metal expects to call |
| **Kernel Writer** | Before writing code — final verification that signatures match tt-metal's wrappers (third source alongside test harness + parent file) |
| **Debugger** | When runtime failures suggest API contract mismatch — check if tt-metal passes different args than what the kernel expects |
