# Common Optimization Patterns

Reusable optimization patterns that transform base algorithms for better performance.

**Usage**: When generating optimized kernel code, apply these patterns to the base algorithm.
Each pattern defines structural transformations and required compile-time/runtime args.

---

## Pattern: Block-Level CB Reservation (OPT_BLOCK_CB)

Amortizes CB synchronization overhead by reserving/pushing at block level.

```yaml
optimization:
  id: OPT_BLOCK_CB
  name: "Block-Level CB Reservation"
  applicable_to: [eltwise_unary, eltwise_binary]

  description: |
    Instead of reserve/push per tile, do it per block.
    Reduces CB synchronization calls by factor of block_size.

  preconditions:
    - "Operation can process tiles independently"
    - "Output CB has capacity >= block_size tiles"

  required_compile_args:
    - name: per_core_block_cnt
      type: uint32_t
      source: "num_tiles_per_core / block_size"
      desc: "Number of blocks to process"

    - name: per_core_block_dim
      type: uint32_t
      source: "block_size"
      desc: "Tiles per block"

  transformation:
    from: |
      // Base: per-tile reserve/push
      for (uint32_t tile = 0; tile < total_tiles; tile++) {
          cb_reserve_back(cb_out, 1);
          process_tile();
          cb_push_back(cb_out, 1);
      }

    to: |
      // Optimized: per-block reserve/push
      for (uint32_t block = 0; block < per_core_block_cnt; block++) {
          cb_reserve_back(cb_out, per_core_block_dim);
          for (uint32_t tile = 0; tile < per_core_block_dim; tile++) {
              process_tile();
          }
          cb_push_back(cb_out, per_core_block_dim);
      }

  validation:
    - "per_core_block_cnt * per_core_block_dim == total_tiles"
    - "Output identical to base"
```

---

## Pattern: DST Batching (OPT_DST_BATCH)

Process multiple tiles per DST acquire/release cycle.

```yaml
optimization:
  id: OPT_DST_BATCH
  name: "DST Batching"
  applicable_to: [transpose_untilize, reduce]

  description: |
    Fill multiple DST slots before commit/release cycle.
    Amortizes tile_regs_* overhead across batch.

  preconditions:
    - "CB is zero-copy (sharded) or pre-filled"
    - "Operation reads directly from CB to DST"

  parameters:
    - name: BATCH_SIZE
      type: "compile-time constant"
      value: 8
      constraint: "<= DST capacity (8 for 16-bit, 4 for 32-bit)"
      reason: "DST can hold 8 tiles max in half-sync mode"

  required_runtime_args:
    - name: total_tiles
      type: uint32_t
      desc: "Total tiles to process"

  derived_values:
    - name: num_batches
      expr: "total_tiles / BATCH_SIZE"
    - name: leftover
      expr: "total_tiles % BATCH_SIZE"

  transformation:
    from: |
      // Base: one tile at a time
      for (uint32_t tile = 0; tile < total_tiles; tile++) {
          tile_regs_acquire();
          process_one_tile(tile);
          tile_regs_commit();
          tile_regs_wait();
          pack_output();
          tile_regs_release();
      }

    to: |
      constexpr int BATCH_SIZE = 8;
      uint32_t num_batches = total_tiles / BATCH_SIZE;
      uint32_t leftover = total_tiles % BATCH_SIZE;

      // Full batches
      for (uint32_t b = 0; b < num_batches; b++) {
          tile_regs_acquire();
          for (uint32_t i = 0; i < BATCH_SIZE; i++) {
              process_to_dst_slot(i);
          }
          tile_regs_commit();
          tile_regs_wait();
          pack_all_slots(BATCH_SIZE);
          tile_regs_release();
      }

      // Leftover tiles
      for (uint32_t i = 0; i < leftover; i++) {
          tile_regs_acquire();
          process_to_dst_slot(0);
          tile_regs_commit();
          tile_regs_wait();
          pack_slot(0);
          tile_regs_release();
      }

  helper_function: |
    template <int N>
    FORCE_INLINE void process_batch(uint32_t cb_in, uint32_t cb_out) {
        cb_wait_front(cb_in, N);
        tile_regs_acquire();
        for (uint32_t i = 0; i < N; i++) {
            // Operation-specific: transpose, reduce, etc.
            operation_tile(cb_in, i, i);  // CB[i] → DST[i]
        }
        tile_regs_commit();
        cb_pop_front(cb_in, N);
        cb_reserve_back(cb_out, N);
        tile_regs_wait();
        pack_output<N>(cb_out);
        tile_regs_release();
        cb_push_back(cb_out, N);
    }

  validation:
    - "num_batches * BATCH_SIZE + leftover == total_tiles"
    - "Output identical to base"
```

---

## Pattern: Loop Unrolling for SFPU (OPT_SFPU_UNROLL)

Unroll inner loop for better SFPU pipeline utilization.

```yaml
optimization:
  id: OPT_SFPU_UNROLL
  name: "SFPU Loop Unrolling"
  applicable_to: [eltwise_unary]

  description: |
    Unroll the inner tile processing loop to keep SFPU pipeline fed.
    Typically unroll by 4-8 iterations.

  preconditions:
    - "Compute primitive is SFPU-based"
    - "block_size >= unroll_factor"

  parameters:
    - name: UNROLL_FACTOR
      type: "compile-time constant"
      typical_values: [4, 8]

  transformation:
    from: |
      for (uint32_t tile = 0; tile < block_size; tile++) {
          tile_regs_acquire();
          cb_wait_front(cb_in, 1);
          copy_tile(cb_in, 0, 0);
          sfpu_op(0);
          tile_regs_commit();
          // ... rest ...
      }

    to: |
      // Assuming block_size % UNROLL_FACTOR == 0
      for (uint32_t tile = 0; tile < block_size; tile += UNROLL_FACTOR) {
          tile_regs_acquire();

          // Unrolled: process UNROLL_FACTOR tiles to different DST slots
          for (uint32_t i = 0; i < UNROLL_FACTOR; i++) {
              cb_wait_front(cb_in, 1);
              copy_tile(cb_in, 0, i);
              sfpu_op(i);
              cb_pop_front(cb_in, 1);
          }

          tile_regs_commit();
          tile_regs_wait();

          for (uint32_t i = 0; i < UNROLL_FACTOR; i++) {
              pack_tile(i, cb_out);
          }

          tile_regs_release();
      }
```

---

## Applying Optimizations to Base Algorithm

```yaml
workflow:
  1_base_algorithm:
    input: "Algorithm from per-OP Section 1"
    output: "Simple tile-by-tile loop"

  2_select_optimizations:
    input: "HW constraints, operation characteristics"
    output: "List of applicable optimization patterns"

  3_apply_optimizations:
    process: |
      For each selected pattern:
        - Add required compile-time args
        - Add required runtime args
        - Transform loop structure

  4_add_state_machine_ops:
    process: |
      Insert tile_regs_acquire/commit/wait/release
      Insert cb_wait_front/pop_front/reserve_back/push_back
      Based on pattern requirements

  5_generate_kernel:
    output: "Complete kernel code matching actual implementation"
```
