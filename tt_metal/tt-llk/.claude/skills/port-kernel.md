---
name: port-kernel
description: Get structured porting guidance when moving a kernel between architectures. Launches sages for source and target, reads test harness.
user_invocable: true
---

# /port-kernel — Porting Guidance

## Usage

```
/port-kernel reduce --from wormhole --to quasar
/port-kernel pack_untilize --from blackhole --to quasar
/port-kernel eltwise_binary --from wormhole --to blackhole
```

## What to Do

1. Parse: kernel name, source architecture, target architecture
2. Launch two tasks IN PARALLEL:

**Task A — Source sage**: Launch the sage for the source architecture
```
Agent tool:
  subagent_type: "sage-{source_arch}"
  description: "Source analysis: {kernel}"
  prompt: |
    Find the implementation of {kernel} on {source_arch}.
    Report: algorithm, function signatures, template parameters,
    key constructs (MOP, replay buffer, address modifiers),
    data format handling, dependencies.
```

**Task B — Target sage**: Launch the sage for the target architecture
```
Agent tool:
  subagent_type: "sage-{target_arch}"
  description: "Target conventions: {kernel}"
  prompt: |
    Find the conventions for {kernel_type} kernels on {target_arch}.
    Search for: closest existing kernel of the same type,
    file naming pattern, instruction conventions, MOP patterns,
    config write patterns, any existing partial implementation of {kernel}.
```

3. While sages run, read the **test harness** for the target architecture:
   - Search `tests/sources/` for matching test files
   - Extract `#ifdef ARCH_{TARGET}` sections for expected function signatures
   - Note template parameters and argument types

4. Aggregate into structured porting guidance:

```markdown
## Porting: {kernel} ({source} → {target})

### Algorithm Summary
[From source sage — what the kernel does]

### Source Implementation
[Key functions, patterns, data format handling]

### Target Conventions
[From target sage — how similar kernels are structured on target]

### API Contract (from test harness)
[Expected function signatures, template params from test files]

### Key Differences
[What changes between source and target — naming, instructions, patterns]

### Recommended Template
[Which existing target kernel to use as a starting point]

### Porting Checklist
- [ ] Function signatures match test harness expectations
- [ ] Uses target arch instruction conventions
- [ ] Init/uninit symmetry maintained
- [ ] All data format paths handled
```

5. Reference `.claude/references/porting-guide.md` for detailed methodology
