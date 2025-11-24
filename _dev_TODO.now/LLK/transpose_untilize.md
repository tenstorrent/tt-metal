# LLK Pattern Usage: Transpose + Untilize

This document describes how the **transpose_untilize** pattern is applied for operations
that perform in-tile transpose followed by untilization to row-major format.

> **Note**: Primitive definitions (requires/produces states, signatures) are in
> `LLK/primitives_catalog.md`. This file documents pattern-specific usage
> and validation contracts.

---

## Pattern Overview

```yaml
pattern_info:
  pattern: transpose_untilize
  operations: [convert_to_chw]
  compute_primitives:
    - transpose_wh_tile
    - pack_untilize_dest

  # Reference to primitive catalog
  primitives_used:
    - { name: transpose_wh_tile, defined_in: "LLK/primitives_catalog.md#transpose_wh_tile" }
    - { name: pack_untilize_dest, defined_in: "LLK/primitives_catalog.md#pack_untilize_dest" }

  # Pattern characteristics
  characteristics:
    - requires_cleanup: true    # pack_untilize_uninit
    - input_format: TILE
    - output_format: ROW_MAJOR

  # Optimization support
  optimization_support:
    - name: dst_batching
      max_batch_size: 8         # DST capacity
      enabled_when: "zero_copy CB pattern"
```

---

## Base Implementation (Canonical)

The simplest correct implementation - processes one tile at a time.
This serves as the **reference for testing** and validation.

```yaml
base_implementation:
  description: "Simple tile-by-tile processing, always correct"

  loop:
    type: simple
    iterator: tile_idx
    limit: total_tiles
    body:
      - cb_wait_front          # Wait for 1 input tile
      - tile_regs_acquire      # DST: RELEASED → ACQUIRED
      - transpose_wh_tile      # Transpose 1 tile to DST
      - tile_regs_commit       # DST: ACQUIRED → COMMITTED
      - cb_pop_front           # Release input CB slot
      - cb_reserve_back        # Reserve output CB space
      - tile_regs_wait         # DST: COMMITTED → WAITED
      - pack_untilize_dest     # Pack 1 tile from DST to CB
      - tile_regs_release      # DST: WAITED → RELEASED
      - cb_push_back           # Signal output ready

  # Equivalent C++ (pseudo-code)
  pseudo_code: |
    for (uint32_t tile = 0; tile < total_tiles; tile++) {
        cb_wait_front(cb_in, 1);
        tile_regs_acquire();
        transpose_wh_tile(cb_in, 0, 0);
        tile_regs_commit();
        cb_pop_front(cb_in, 1);
        cb_reserve_back(cb_out, 1);
        tile_regs_wait();
        pack_untilize_dest<1>(cb_out, 1);
        tile_regs_release();
        cb_push_back(cb_out, 1);
    }
```

---

## Optimization: DST Batching

Performance optimization that processes multiple tiles per DST cycle.

```yaml
optimization:
  id: OPT1
  name: dst_batching

  # Why this optimization works
  rationale:
    - "Zero-copy CB: data already in L1 SRAM (sharded), no fetch overhead"
    - "Reader kernel just signals data availability, no actual data movement"
    - "DST register file can hold 8 tiles simultaneously"
    - "pack_untilize_dest<1>(cb, N) can unpack N tiles in one call"
    - "Amortizes tile_regs_acquire/release cost across batch"

  # When to apply
  preconditions:
    - { entity: input_cb.pattern, rel: "==", value: zero_copy }
    - { entity: memory_layout, rel: "in", value: [HEIGHT_SHARDED, WIDTH_SHARDED] }

  parameters:
    BATCH_SIZE: 8
    reason: "DST capacity limit"

  # Transformed loop structure
  optimized_structure:
    type: "batch loop + leftover loop"

    loops:
      - name: batch_loop
        count: "total_tiles / BATCH_SIZE"
        calls: "transpose<BATCH_SIZE>()"

      - name: leftover_loop
        count: "total_tiles % BATCH_SIZE"
        calls: "transpose<1>()"

    helper_template:
      name: transpose
      template_param: N
      note: "FORCE_INLINE for inlining at call site"

  # Validation
  validation:
    - "Output identical to base implementation"
    - "Handles leftover tiles correctly"
    - "Performance: ~8x fewer tile_regs cycles for large tile counts"
```

### Initialization Sequence

```yaml
init_sequence:
  # Complete derived sequence for this pattern
  required: true
  order: strict

  calls:
    - fn: compute_kernel_hw_startup
      args: [cb_in, cb_out]
      note: "General compute kernel initialization"

    - fn: pack_untilize_init
      args: [cb_in, cb_out]
      produces: PACK_UNTILIZE_INITIALIZED
      note: "Setup for untilization"

    - fn: transpose_wh_init
      args: [cb_in, cb_out]
      produces: TRANSPOSE_INITIALIZED
      note: "Setup for transpose operation"

    - fn: pack_untilize_dest_init<1>
      args: [cb_out]
      requires: PACK_UNTILIZE_INITIALIZED
      produces: PACK_UNTILIZE_DEST_INITIALIZED
      note: "Template param 1 for single-tile granularity"

cleanup:
  required: true
  calls:
    - fn: pack_untilize_uninit
      args: [cb_out]
      note: "Required cleanup after untilization"
```

---

## AST Validation Contract

What the validator expects to find in the C++ compute kernel.

```yaml
ast_contract:
  kernel_path: "device/kernels/convert_to_chw.cpp"

  # Expected init block
  init_functions:
    required:
      - compute_kernel_hw_startup
      - pack_untilize_init
      - transpose_wh_init
      - pack_untilize_dest_init
    order: flexible  # Order can vary as long as dependencies satisfied

  # Expected cleanup
  cleanup_functions:
    required:
      - pack_untilize_uninit
    position: after_all_loops

  # Expected loop structure
  loop_structure:
    type: sequential
    loops:
      - name: batch_loop
        iterator_pattern: "i"
        limit_expr: "num_batches"
        body: helper_call  # Calls transpose<8>
      - name: leftover_loop
        iterator_pattern: "idx"
        limit_expr: "leftover"
        body: helper_call  # Calls transpose<1>

  # Helper template validation
  helper_template:
    name: transpose
    template_param: N
    expected_sequence:
      - cb_wait_front
      - tile_regs_acquire
      - transpose_wh_tile  # Called N times
      - tile_regs_commit
      - cb_pop_front
      - cb_reserve_back
      - tile_regs_wait
      - pack_untilize_dest
      - tile_regs_release
      - cb_push_back

  # Variable bindings
  variable_bindings:
    compile_arg[0]: cb_in
    compile_arg[1]: cb_transpose_in
    runtime_arg[0]: total_tiles

  # CB usage patterns
  cb_usage:
    c_0:  # Input CB (zero-copy from sharded input)
      operations: [cb_wait_front, transpose_wh_tile, cb_pop_front]
      role: consumer
      allocation: globally_allocated
    c_2:  # Intermediate CB (for untilized output)
      operations: [cb_reserve_back, pack_untilize_dest, cb_push_back]
      role: producer
      allocation: standard
```

---

## Execution Pattern (Derived)

Complete derived execution pattern for validation.

```yaml
execution_pattern:
  structure: sequential

  constants:
    BATCH_SIZE: 8

  # Phase 1: Process full batches
  batch_loop:
    name: batch_loop
    iterator: i
    limit_expr: "num_batches = total_tiles / BATCH_SIZE"
    per_iteration:
      calls: "transpose<BATCH_SIZE>()"
      expands_to:
        - { op: cb_wait_front, args: [cb_in, 8], state_change: "CB_in: → HAS_DATA" }
        - { op: tile_regs_acquire, state_change: "DST: RELEASED → ACQUIRED" }
        - { op: transpose_wh_tile, args: [cb_in, 0, 0], repeat: 8, state_change: "DST: ACQUIRED → WRITTEN" }
        - { op: tile_regs_commit, state_change: "DST: WRITTEN → COMMITTED" }
        - { op: cb_pop_front, args: [cb_in, 8], state_change: "CB_in: HAS_DATA → FREED" }
        - { op: cb_reserve_back, args: [cb_out, 8], state_change: "CB_out: → RESERVED" }
        - { op: tile_regs_wait, state_change: "DST: COMMITTED → WAITED" }
        - { op: pack_untilize_dest<1>, args: [cb_out, 8], state_change: "CB_out: RESERVED → WRITTEN" }
        - { op: tile_regs_release, state_change: "DST: WAITED → RELEASED" }
        - { op: cb_push_back, args: [cb_out, 8], state_change: "CB_out: WRITTEN → PUSHED" }

  # Phase 2: Process leftover tiles
  leftover_loop:
    name: leftover_loop
    iterator: idx
    limit_expr: "leftover = total_tiles % BATCH_SIZE"
    per_iteration:
      calls: "transpose<1>()"
      expands_to:
        - { op: cb_wait_front, args: [cb_in, 1] }
        - { op: tile_regs_acquire }
        - { op: transpose_wh_tile, args: [cb_in, 0, 0] }
        - { op: tile_regs_commit }
        - { op: cb_pop_front, args: [cb_in, 1] }
        - { op: cb_reserve_back, args: [cb_out, 1] }
        - { op: tile_regs_wait }
        - { op: pack_untilize_dest<1>, args: [cb_out, 1] }
        - { op: tile_regs_release }
        - { op: cb_push_back, args: [cb_out, 1] }
```

---

## State Transitions

Pattern-specific view of state transitions during batch execution.

```yaml
state_flow:
  # DST register states during one batch
  dst_lifecycle:
    - { state: RELEASED, at: "batch start" }
    - { state: ACQUIRED, after: tile_regs_acquire }
    - { state: WRITTEN, after: "all transpose_wh_tile calls" }
    - { state: COMMITTED, after: tile_regs_commit }
    - { state: WAITED, after: tile_regs_wait }
    - { state: RELEASED, after: tile_regs_release }

  # CB states
  cb_input_lifecycle:
    - { state: HAS_DATA, after: cb_wait_front }
    - { state: FREED, after: cb_pop_front }

  cb_output_lifecycle:
    - { state: RESERVED, after: cb_reserve_back }
    - { state: WRITTEN, after: pack_untilize_dest }
    - { state: PUSHED, after: cb_push_back }

  key_observation: |
    Note the unusual order: cb_pop_front comes BEFORE cb_reserve_back.
    This is because transpose_wh_tile reads directly from CB (not via copy_tile),
    so input can be released before output is reserved.
```

---

## Invariants

Pattern-specific invariants that must hold.

```yaml
invariants:
  - id: V1
    rule: "All tiles in batch processed atomically"
    implication: "DST holds all batch tiles between acquire and release"

  - id: V2
    rule: "DST exclusive to one batch at a time"
    implication: "No concurrent batch processing"

  - id: V3
    rule: "acquire → commit → wait → release per batch"
    implication: "Complete DST lifecycle every batch"

  - id: V4
    rule: "cb_push count == cb_pop count over lifetime"
    implication: "CB synchronization balanced"

  - id: V5
    rule: "Cleanup called after all processing"
    implication: "pack_untilize_uninit runs after both loops"
```

---

## Preconditions

Conditions that must hold for correct execution.

```yaml
preconditions:
  # Batching constraint
  - id: L1
    entity: batch_size
    attr: value
    rel: "<="
    value: 8
    reason: "DST capacity limit"

  # Format constraints
  - id: L2
    entity: cb_in
    attr: format
    rel: "=="
    value: TILE
    reason: "Input must be tiled for transpose"

  - id: L3
    entity: cb_out
    attr: format
    rel: "=="
    value: ROW_MAJOR
    reason: "Output is untilized to row-major"

  # Init requirement
  - id: L4
    entity: init_sequence
    attr: called
    rel: "=="
    value: true
    reason: "All init functions must be called before loops"
```
