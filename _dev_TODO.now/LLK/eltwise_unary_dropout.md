# LLK Pattern Usage: Eltwise Unary Dropout

This document describes how the **eltwise_unary** pattern is applied for the dropout operation.

> **Note**: Primitive definitions (requires/produces states, signatures) are in
> `LLK/primitives_catalog.md`. This file documents pattern-specific usage
> and validation contracts.

---

## Pattern Overview

```yaml
pattern_info:
  pattern: eltwise_unary
  operation: dropout
  compute_primitive: dropout_tile
  category: sfpu_compute

  # Reference to primitive catalog
  primitives_used:
    - { name: dropout_tile, defined_in: "LLK/primitives_catalog.md#dropout_tile" }
    - { name: dropout_kernel_init, defined_in: "LLK/primitives_catalog.md#dropout_kernel_init" }
    - { name: copy_tile, defined_in: "LLK/primitives_catalog.md#copy_tile", derived: true }
    - { name: pack_tile, defined_in: "LLK/primitives_catalog.md#pack_tile", derived: true }
```

---

## Dropout-Specific Details

### Parameter Conversion

Dropout requires converting float parameters to integer representations for the SFPU.

```yaml
param_conversion:
  - name: probability_int
    from: { name: prob, type: float, range: "[0, 1)" }
    to: { type: uint32_t }
    expr: "(uint32_t)(prob * INT_MAX)"
    passed_as: compile_arg[2]

  - name: scale_bits
    from: { name: scale, type: float, range: "> 0" }
    to: { type: uint32_t }
    expr: "std::bit_cast<uint32_t>(scale)"
    passed_as: compile_arg[3]

  - name: seed
    from: { name: seed, type: uint32_t }
    to: { type: uint32_t }
    passed_as: runtime_arg[0]
    note: "Runtime arg because seed may change between invocations"
```

### Dropout-Specific Initialization

```yaml
dropout_init:
  # Additional init required beyond pattern's base_init
  additional_init: dropout_kernel_init

  # Why: dropout_tile requires DROPOUT_RNG_INITIALIZED state
  # Which is produced by: dropout_kernel_init
  # Which requires: SFPU_INITIALIZED (provided by pattern's init_sfpu)

  derived_init_sequence:
    - { fn: init_sfpu, args: [cb_in, cb_out] }        # From pattern base
    - { fn: dropout_kernel_init, args: [seed] }       # Dropout-specific
```

---

## AST Validation Contract

What the validator expects to find in the C++ compute kernel.

```yaml
ast_contract:
  kernel_path: "device/kernels/compute/dropout_kernel.cpp"

  # Expected init block
  init_functions:
    required:
      - init_sfpu
      - dropout_kernel_init
    order: strict  # init_sfpu must come before dropout_kernel_init

  # Expected loop structure
  loop_structure:
    type: nested
    outer:
      iterator_pattern: "block_*"
      limit_from: compile_arg[0]
      name: per_core_block_cnt
    inner:
      iterator_pattern: "tile_*"
      limit_from: compile_arg[1]
      name: per_core_block_dim

  # Expected sequence in inner loop
  inner_loop_sequence:
    - tile_regs_acquire
    - cb_wait_front
    - copy_tile
    - dropout_tile      # The compute primitive
    - tile_regs_commit
    - tile_regs_wait
    - pack_tile
    - cb_pop_front
    - tile_regs_release

  # Variable bindings for validation
  variable_bindings:
    compile_arg[0]: per_core_block_cnt
    compile_arg[1]: per_core_block_dim
    compile_arg[2]: int_probability
    compile_arg[3]: int_scale_factor
    runtime_arg[0]: seed

  # CB usage patterns
  cb_usage:
    c_0:  # Input CB
      operations: [cb_wait_front, copy_tile, cb_pop_front]
      role: consumer
      init_usage: [init_sfpu]
    c_2:  # Output CB
      operations: [cb_reserve_back, pack_tile, cb_push_back]
      role: producer
      init_usage: [init_sfpu]
```

---

## Execution Pattern (Derived)

This section shows the complete derived execution pattern.
Validators use this to check C++ implementation correctness.

```yaml
execution_pattern:
  structure: nested

  outer_loop:
    name: block_loop
    iterator: block_index
    limit_var: per_core_block_cnt
    limit_source: compile_arg[0]
    before_inner:
      - { op: cb_reserve_back, args: [cb_out, per_core_block_dim] }
    after_inner:
      - { op: cb_push_back, args: [cb_out, per_core_block_dim] }

  inner_loop:
    name: tile_loop
    iterator: tile_index
    limit_var: per_core_block_dim
    limit_source: compile_arg[1]
    per_tile:
      # DST lifecycle with state annotations
      - { op: tile_regs_acquire, state_change: "DST: RELEASED → ACQUIRED" }
      - { op: cb_wait_front, args: [cb_in, 1], state_change: "CB_in: → HAS_DATA" }
      - { op: copy_tile, args: [cb_in, 0, 0], state_change: "DST: ACQUIRED → HAS_DATA" }
      - { op: dropout_tile, args: [0, prob_int, scale_bits], state_change: "DST: HAS_DATA → MODIFIED" }
      - { op: tile_regs_commit, state_change: "DST: MODIFIED → COMMITTED" }
      - { op: tile_regs_wait, state_change: "DST: COMMITTED → WAITED" }
      - { op: pack_tile, args: [0, cb_out], state_change: "CB_out: RESERVED → WRITTEN" }
      - { op: cb_pop_front, args: [cb_in, 1], state_change: "CB_in: HAS_DATA → FREED" }
      - { op: tile_regs_release, state_change: "DST: WAITED → RELEASED" }
```

---

## State Transitions

Dropout-specific view of state transitions during execution.

```yaml
state_flow:
  # DST register states during one tile iteration
  dst_lifecycle:
    - { state: RELEASED, after: "previous tile_regs_release (or initial)" }
    - { state: ACQUIRED, after: tile_regs_acquire }
    - { state: HAS_DATA, after: copy_tile }
    - { state: MODIFIED, after: dropout_tile }
    - { state: COMMITTED, after: tile_regs_commit }
    - { state: WAITED, after: tile_regs_wait }
    - { state: RELEASED, after: tile_regs_release }

  # RNG state (persistent across tiles)
  rng_lifecycle:
    - { state: UNINITIALIZED, at: "kernel start" }
    - { state: INITIALIZED, after: dropout_kernel_init }
    note: "RNG state advances with each dropout_tile call"
```

---

## Invariants

Operation-specific invariants that must hold.

```yaml
invariants:
  - id: V1
    rule: "RNG state advances deterministically per tile"
    implication: "Same seed + same tile order = same masks"

  - id: V2
    rule: "Same seed + same input = same output"
    tested_by: "determinism test with fixed seed"

  - id: V3
    rule: "Each element mask is independent"
    implication: "Bernoulli(p) for each element, not correlated"

  - id: V4
    rule: "dropout_tile modifies DST in-place"
    implication: "No additional DST register needed"

  - id: V5
    rule: "Scale applied after masking"
    formula: "output = input * mask * scale"
```

---

## Preconditions

Conditions that must hold for correct execution.

```yaml
preconditions:
  # DST register constraint
  - id: L1
    entity: idst
    attr: value
    rel: "<="
    value: 7
    reason: "DST has 8 registers (0-7)"

  # Init requirements
  - id: L2
    entity: init_sfpu
    attr: called
    rel: "=="
    value: true
    reason: "Required by dropout_kernel_init"

  - id: L3
    entity: dropout_kernel_init
    attr: called
    rel: "=="
    value: true
    reason: "Required by dropout_tile"

  # Parameter constraints
  - id: L4
    entity: probability_int
    attr: derived_from
    rel: "=="
    value: "prob * INT_MAX"

  - id: L5
    entity: scale_bits
    attr: derived_from
    rel: "=="
    value: "bit_cast(scale)"
```
