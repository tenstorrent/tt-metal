# Micro-Op Architecture for Fused Kernels

This document describes the architecture for composable micro-ops that can be chained together into fused operations with a single unified kernel file.

## Overview

When TT-Metal compiles a kernel, it compiles the same source file multiple times with different preprocessor defines to target different RISC cores:

| Core Type | Define | Role |
|-----------|--------|------|
| **BRISC** | `COMPILE_FOR_BRISC` | Data movement - typically reader |
| **NCRISC** | `COMPILE_FOR_NCRISC` | Data movement - typically writer |
| **TRISC0** | `COMPILE_FOR_TRISC=0`, `TRISC_UNPACK` | Unpack from CB to SRCA/SRCB |
| **TRISC1** | `COMPILE_FOR_TRISC=1`, `TRISC_MATH` | Math operations on FPU |
| **TRISC2** | `COMPILE_FOR_TRISC=2`, `TRISC_PACK` | Pack from DEST to CB |

Our micro-op architecture leverages this by:
1. Each micro-op fully encapsulates reader/writer/compute logic
2. A single `run()` method internally dispatches based on core type
3. Fused kernels just instantiate and run micro-ops in sequence
4. Runtime args are passed directly to micro-ops (no `get_arg_val` inside micro-ops)

## Directory Structure

```
models/demos/deepseek_v3_b1/
├── micro_ops/
│   ├── common/
│   │   └── micro_op_api.hpp       # Common includes, macros, type helpers
│   ├── rmsnorm/
│   │   └── rmsnorm.hpp            # RMSNorm micro-op
│   ├── matmul/
│   │   └── matmul.hpp             # Matmul micro-op
│   ├── gather/
│   │   └── gather.hpp             # Gather micro-op
│   └── mcast/
│       └── mcast.hpp              # Multicast micro-op
├── fused_ops/
│   └── rmsnorm_matmul/
│       ├── kernel.cpp             # Single unified kernel file
│       └── op.py                  # Python op wrapper
```

## Common API

### `micro_ops/common/micro_op_api.hpp`

```cpp
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <type_traits>

// ============================================================================
// Core type detection - constexpr bools for use with constexpr if
// ============================================================================
#if defined(COMPILE_FOR_BRISC)
    inline constexpr bool is_brisc = true;
    inline constexpr bool is_ncrisc = false;
    inline constexpr bool is_trisc = false;
    #include "dataflow_api.h"
#elif defined(COMPILE_FOR_NCRISC)
    inline constexpr bool is_brisc = false;
    inline constexpr bool is_ncrisc = true;
    inline constexpr bool is_trisc = false;
    #include "dataflow_api.h"
#elif defined(COMPILE_FOR_TRISC)
    inline constexpr bool is_brisc = false;
    inline constexpr bool is_ncrisc = false;
    inline constexpr bool is_trisc = true;
    #include "compute_kernel_api.h"
#endif

// ============================================================================
// Unified kernel entry point macro
// ============================================================================
#if defined(COMPILE_FOR_TRISC)
    #define KERNEL_ENTRY namespace NAMESPACE { void MAIN
    #define KERNEL_END }
#else
    #define KERNEL_ENTRY void kernel_main()
    #define KERNEL_END
#endif

// ============================================================================
// Type helper: Select type based on current core
// Usage: using RTArgs = SelectByCore<ReaderArgs, WriterArgs, ComputeArgs>;
// ============================================================================
template<typename Reader, typename Writer, typename Compute>
using SelectByCore = std::conditional_t<
    is_brisc,
    Reader,
    std::conditional_t<is_ncrisc, Writer, Compute>
>;
```

## Micro-Op Template

Each micro-op follows this pattern:

```cpp
#pragma once

#include "micro_ops/common/micro_op_api.hpp"

// Phase-specific includes (must use preprocessor for includes)
#if defined(COMPILE_FOR_BRISC)
// Reader-specific includes
#endif

#if defined(COMPILE_FOR_TRISC)
// Compute-specific includes
#endif

namespace micro_ops {

template <
    uint32_t input_cb,
    uint32_t output_cb,
    uint32_t num_tiles,
    /* other compile-time params */>
class MyOp {
public:
    // ========================================================================
    // Phase-specific RTArgs - only relevant fields exist per core
    // ========================================================================
    struct ReaderArgs {
        uint32_t some_reader_param = 0;
    };

    struct WriterArgs {
        uint32_t some_writer_param = 0;
    };

    struct ComputeArgs {
        // Empty if no compute runtime args needed
    };

    // Compile-time type selection - only one struct exists per compilation
    using RTArgs = SelectByCore<ReaderArgs, WriterArgs, ComputeArgs>;

    // Constructor receives pre-loaded args (no get_arg_val inside micro-op)
    MyOp(const RTArgs& args) : args_(args) {}

    // Single entry point - dispatches internally
    void run() {
        if constexpr (is_brisc) {
            run_reader();
        } else if constexpr (is_ncrisc) {
            run_writer();
        } else if constexpr (is_trisc) {
            run_compute();
        }
    }

private:
    RTArgs args_;

    void run_reader() {
        // Use args_.some_reader_param
    }

    void run_writer() {
        // Use args_.some_writer_param
    }

    void run_compute() {
        // Compute implementation
    }
};

}  // namespace micro_ops
```

## RMSNorm Micro-Op

### `micro_ops/rmsnorm/rmsnorm.hpp`

```cpp
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "micro_ops/common/micro_op_api.hpp"

// Phase-specific includes
#if defined(COMPILE_FOR_BRISC)
#include "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/generate_reduce_scaler.hpp"
#endif

#if defined(COMPILE_FOR_TRISC)
#define REDUCE_OP PoolType::SUM
#define REDUCE_DIM ReduceDim::REDUCE_SCALAR
#include "compute_kernel_api/reduce.h"
#include "compute_kernel_api/bcast.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"
#include "compute_kernel_api/eltwise_unary/rsqrt.h"
#endif

namespace micro_ops {

/**
 * RMSNorm micro-op
 *
 * Performs: output = (input / rms(input)) * gamma
 *
 * Template params:
 *   input_cb     - Input circular buffer (sharded)
 *   gamma_cb     - Gamma weights circular buffer (sharded)
 *   scalars_cb   - CB for epsilon and reduction scalar
 *   interm_cb    - Intermediate CB for computation
 *   output_cb    - Output circular buffer
 *   num_tiles    - Number of tiles in input
 *   fp32_acc     - Use FP32 accumulation
 *   tiny_tile    - Use 16x32 tiles instead of 32x32
 *   pop_input    - Whether to pop input CB after use
 */
template <
    uint32_t input_cb,
    uint32_t gamma_cb,
    uint32_t scalars_cb,
    uint32_t interm_cb,
    uint32_t output_cb,
    uint32_t num_tiles,
    bool fp32_acc = false,
    bool tiny_tile = false,
    bool pop_input = true>
class RMSNorm {
public:
    // ========================================================================
    // Phase-specific RTArgs - fields only exist for the core that uses them
    // ========================================================================
    struct ReaderArgs {
        uint32_t epsilon_packed = 0;
        uint32_t scalar_packed = 0;
    };

    struct WriterArgs {
        // RMSNorm as intermediate op has no writer args
    };

    struct ComputeArgs {
        // RMSNorm has no compute runtime args
    };

    // Select based on core type - zero-cost compile-time selection
    using RTArgs = SelectByCore<ReaderArgs, WriterArgs, ComputeArgs>;

    // Constructor receives pre-loaded args
    RMSNorm(const RTArgs& args) : args_(args) {}

    void run() {
        if constexpr (is_brisc) {
            run_reader();
        } else if constexpr (is_ncrisc) {
            run_writer();
        } else if constexpr (is_trisc) {
            run_compute();
        }
    }

private:
    RTArgs args_;

    // ========================================================================
    // BRISC (Reader)
    // ========================================================================
    void run_reader() {
        // Generate scalar tiles using args passed from fused kernel
        cb_reserve_back(scalars_cb, 2);
        volatile tt_l1_ptr uint16_t* epsilon_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(scalars_cb));
        epsilon_ptr[0] = args_.epsilon_packed;
        cb_push_back(scalars_cb, 1);

        generate_reduce_scaler<tiny_tile>(scalars_cb, args_.scalar_packed);
        cb_push_back(scalars_cb, 1);

        // Signal sharded inputs are ready
        cb_reserve_back(input_cb, num_tiles);
        cb_push_back(input_cb, num_tiles);
        cb_reserve_back(gamma_cb, num_tiles);
        cb_push_back(gamma_cb, num_tiles);
    }

    // ========================================================================
    // NCRISC (Writer)
    // ========================================================================
    void run_writer() {
        // RMSNorm as intermediate op: no writer action needed
        // Output goes to next op's input CB
    }

    // ========================================================================
    // TRISC (Compute)
    // ========================================================================
    void run_compute() {
        constexpr uint32_t epsilon_index = 0;
        constexpr uint32_t scalar_index = 1;

        // Init
        binary_op_init_common(input_cb, input_cb, output_cb);
        cb_wait_front(scalars_cb, 2);
        cb_wait_front(gamma_cb, num_tiles);
        rsqrt_tile_init();

        // Square the input
        {
            mul_tiles_init(input_cb, input_cb);
            cb_wait_front(input_cb, num_tiles);
            cb_reserve_back(interm_cb, num_tiles + 1);
            tile_regs_acquire();
            for (uint32_t i = 0; i < num_tiles; i++) {
                mul_tiles(input_cb, input_cb, i, i, i);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile_block(0, interm_cb, num_tiles);
            tile_regs_release();
            cb_push_back(interm_cb, num_tiles);
        }

        // Reduce sum of squares
        {
            cb_wait_front(interm_cb, num_tiles);
            reduce_init<PoolType::SUM, ReduceDim::REDUCE_SCALAR, fp32_acc>(interm_cb, scalars_cb, interm_cb);
            tile_regs_acquire();
            for (uint32_t i = 0; i < num_tiles; i++) {
                reduce_tile<PoolType::SUM, ReduceDim::REDUCE_SCALAR, fp32_acc>(
                    interm_cb, scalars_cb, i, scalar_index, 0);
            }
        }

        // Add epsilon and compute 1/RMS
        {
            binary_dest_reuse_tiles_init<ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(scalars_cb);
            binary_dest_reuse_tiles<ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
                scalars_cb, epsilon_index, 0);
            rsqrt_tile<false, true>(0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, interm_cb);
            tile_regs_release();
            reduce_uninit();
            cb_pop_front(interm_cb, num_tiles);
            cb_push_back(interm_cb, 1);
        }

        // Multiply input by 1/RMS
        {
            cb_wait_front(interm_cb, 1);
            cb_reserve_back(output_cb, num_tiles);
            mul_tiles_bcast_scalar_init_short(input_cb, interm_cb);
            tile_regs_acquire();
            for (uint32_t i = 0; i < num_tiles; i++) {
                mul_tiles_bcast_scalar(input_cb, interm_cb, i, 0, i);
            }
            if constexpr (pop_input) {
                cb_pop_front(input_cb, num_tiles);
            }
        }

        // Multiply by gamma
        {
            binary_dest_reuse_tiles_init<ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(gamma_cb);
            for (uint32_t i = 0; i < num_tiles; i++) {
                binary_dest_reuse_tiles<ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(gamma_cb, i, i);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile_block(0, output_cb, num_tiles);
            tile_regs_release();
            cb_pop_front(interm_cb, 1);
            cb_push_back(output_cb, num_tiles);
        }
    }
};

}  // namespace micro_ops
```

## Matmul Micro-Op

### `micro_ops/matmul/matmul.hpp`

```cpp
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "micro_ops/common/micro_op_api.hpp"

#if defined(COMPILE_FOR_TRISC)
#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api/tile_move_copy.h"
#endif

namespace micro_ops {

/**
 * Single-tile output Matmul micro-op
 *
 * Computes: output[1,1] = in0[1,K] @ in1[K,1]
 *
 * Template params:
 *   in0_cb       - Input 0 circular buffer (activations)
 *   in1_cb       - Input 1 circular buffer (weights, sharded)
 *   output_cb    - Output circular buffer
 *   num_tiles_k  - K dimension in tiles
 *   pop_inputs   - Whether to pop input CBs after use
 */
template <
    uint32_t in0_cb,
    uint32_t in1_cb,
    uint32_t output_cb,
    uint32_t num_tiles_k,
    bool pop_inputs = true>
class Matmul {
public:
    // ========================================================================
    // Phase-specific RTArgs
    // ========================================================================
    struct ReaderArgs {
        // No reader runtime args for this simple matmul
    };

    struct WriterArgs {
        // No writer runtime args
    };

    struct ComputeArgs {
        // No compute runtime args
    };

    using RTArgs = SelectByCore<ReaderArgs, WriterArgs, ComputeArgs>;

    Matmul(const RTArgs& = {}) {}  // Default constructible

    void run() {
        if constexpr (is_brisc) {
            run_reader();
        } else if constexpr (is_ncrisc) {
            run_writer();
        } else if constexpr (is_trisc) {
            run_compute();
        }
    }

private:
    // ========================================================================
    // BRISC (Reader)
    // ========================================================================
    void run_reader() {
        // Signal sharded weight tensor is ready
        cb_reserve_back(in1_cb, num_tiles_k);
        cb_push_back(in1_cb, num_tiles_k);

        // in0_cb comes from previous op's output - already signaled
    }

    // ========================================================================
    // NCRISC (Writer)
    // ========================================================================
    void run_writer() {
        // Wait for output (for final op in chain)
        cb_wait_front(output_cb, 1);
    }

    // ========================================================================
    // TRISC (Compute)
    // ========================================================================
    void run_compute() {
        constexpr uint32_t out_subblock_h = 1;
        constexpr uint32_t out_subblock_w = 1;
        constexpr uint32_t in0_block_w = 1;

        mm_block_init(in0_cb, in1_cb, output_cb, false, out_subblock_w, out_subblock_h, in0_block_w);

        cb_wait_front(in0_cb, num_tiles_k);
        cb_wait_front(in1_cb, num_tiles_k);
        cb_reserve_back(output_cb, 1);

        tile_regs_acquire();
        for (uint32_t k = 0; k < num_tiles_k; k++) {
            matmul_tiles(in0_cb, in1_cb, k, k, 0, false);
        }
        tile_regs_commit();

        if constexpr (pop_inputs) {
            cb_pop_front(in0_cb, num_tiles_k);
            cb_pop_front(in1_cb, num_tiles_k);
        }

        tile_regs_wait();
        pack_tile(0, output_cb);
        tile_regs_release();

        cb_push_back(output_cb, 1);
    }
};

}  // namespace micro_ops
```

## Fused Op Kernel Example

### `fused_ops/rmsnorm_matmul/kernel.cpp`

The fused kernel is clean - it loads all runtime args and passes them to micro-ops:

```cpp
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// RMSNorm -> Matmul fused operation
// Single kernel file, compiles correctly for all RISC cores

#include "micro_ops/common/micro_op_api.hpp"
#include "micro_ops/rmsnorm/rmsnorm.hpp"
#include "micro_ops/matmul/matmul.hpp"

// ============================================================================
// Compile-time args (defined at file scope, like constants)
// ============================================================================
constexpr uint32_t input_cb         = get_compile_time_arg_val(0);
constexpr uint32_t gamma_cb         = get_compile_time_arg_val(1);
constexpr uint32_t scalars_cb       = get_compile_time_arg_val(2);
constexpr uint32_t interm_cb        = get_compile_time_arg_val(3);
constexpr uint32_t rmsnorm_out_cb   = get_compile_time_arg_val(4);  // RMSNorm output = Matmul in0
constexpr uint32_t matmul_weight_cb = get_compile_time_arg_val(5);
constexpr uint32_t output_cb        = get_compile_time_arg_val(6);
constexpr uint32_t num_tiles        = get_compile_time_arg_val(7);
constexpr uint32_t num_k_tiles      = get_compile_time_arg_val(8);
constexpr bool fp32_acc             = get_compile_time_arg_val(9);
constexpr bool tiny_tile            = get_compile_time_arg_val(10);

// ============================================================================
// Micro-op type aliases (for cleaner code below)
// ============================================================================
using RMSNormOp = micro_ops::RMSNorm<
    input_cb, gamma_cb, scalars_cb, interm_cb, rmsnorm_out_cb,
    num_tiles, fp32_acc, tiny_tile>;

using MatmulOp = micro_ops::Matmul<
    rmsnorm_out_cb, matmul_weight_cb, output_cb, num_k_tiles>;

// ============================================================================
// Runtime args - fused kernel owns all arg loading
// Aggregates RTArgs from all micro-ops, structured like compile-time args
// ============================================================================
struct RTArgs {
    RMSNormOp::RTArgs rmsnorm;
    MatmulOp::RTArgs matmul;

    // Load all runtime args - phase detection hidden here
    static RTArgs load() {
        RTArgs rt;

        // Reader runtime args (BRISC only)
        if constexpr (is_brisc) {
            uint32_t idx = 0;
            rt.rmsnorm.epsilon_packed = get_arg_val<uint32_t>(idx++);
            rt.rmsnorm.scalar_packed = get_arg_val<uint32_t>(idx++);
            // matmul has no reader args
        }

        // Writer runtime args (NCRISC only)
        if constexpr (is_ncrisc) {
            uint32_t idx = 0;
            // rmsnorm has no writer args
            // matmul has no writer args
        }

        // Compute runtime args (TRISC only)
        if constexpr (is_trisc) {
            uint32_t idx = 0;
            // rmsnorm has no compute args
            // matmul has no compute args
        }

        return rt;
    }
};

// ============================================================================
// Kernel entry - clean and simple
// ============================================================================
KERNEL_ENTRY {
    // Load runtime args (single line, mirrors compile-time args above)
    auto rt = RTArgs::load();

    // Create micro-ops with their args
    RMSNormOp rmsnorm(rt.rmsnorm);
    MatmulOp matmul(rt.matmul);

    // Execute pipeline
    rmsnorm.run();
    matmul.run();

} KERNEL_END
```

## How `SelectByCore` Works

The `SelectByCore` helper uses `std::conditional_t` to select types at compile time:

```cpp
template<typename Reader, typename Writer, typename Compute>
using SelectByCore = std::conditional_t<
    is_brisc,
    Reader,
    std::conditional_t<is_ncrisc, Writer, Compute>
>;
```

**Evaluation flow:**
1. If `is_brisc == true` → Result is `Reader`
2. Else if `is_ncrisc == true` → Result is `Writer`
3. Else → Result is `Compute`

**What happens at compile time:**

| Core Being Compiled | `is_brisc` | `is_ncrisc` | `is_trisc` | `RTArgs` becomes |
|---------------------|------------|-------------|------------|------------------|
| BRISC | `true` | `false` | `false` | `ReaderArgs` |
| NCRISC | `false` | `true` | `false` | `WriterArgs` |
| TRISC | `false` | `false` | `true` | `ComputeArgs` |

**Zero runtime cost** - this is purely compile-time type selection.

## Key Design Principles

### 1. Micro-Ops Are Pure
Micro-ops receive data, they don't load it. No `get_arg_val` inside micro-ops - all arg loading happens in the fused kernel's `RTArgs::load()`.

### 2. Phase-Specific RTArgs
Each micro-op defines `ReaderArgs`, `WriterArgs`, `ComputeArgs` structs. Only the relevant struct exists for each compilation via `SelectByCore`.

### 3. Parallel Structure
Runtime args mirror compile-time args in organization:
- **Compile-time**: `constexpr` values at file scope
- **Runtime**: `RTArgs` struct with `load()` method

### 4. Single `run()` Method
Each micro-op has one `run()` method that internally dispatches. The fused kernel just calls `run()` in sequence.

### 5. CB Chaining
Output CB of one op becomes input CB of the next:
```
input_cb → [RMSNorm] → rmsnorm_out_cb → [Matmul] → output_cb
```

### 6. Fused Kernel Owns Arg Layout
The fused kernel's `RTArgs::load()` defines the complete arg layout. Micro-ops don't know their arg indices.

## Python Op Wrapper

The Python side creates kernel descriptors using the same unified kernel file:

```python
# All three kernels use the SAME source file
kernel_source = "models/demos/deepseek_v3_b1/fused_ops/rmsnorm_matmul/kernel.cpp"

# Runtime args per kernel type
reader_rt_args = [epsilon_packed, scalar_packed]  # Matches RTArgs::load() for BRISC
writer_rt_args = []                                # Matches RTArgs::load() for NCRISC
compute_rt_args = []                               # Matches RTArgs::load() for TRISC

# Reader kernel
reader_kernel = ttnn.KernelDescriptor(
    kernel_source=kernel_source,
    source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
    core_ranges=core_grid,
    compile_time_args=compile_time_args,
    runtime_args=[[reader_rt_args]],
    config=ttnn.ReaderConfigDescriptor(),
)

# Writer kernel
writer_kernel = ttnn.KernelDescriptor(
    kernel_source=kernel_source,
    source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
    core_ranges=core_grid,
    compile_time_args=compile_time_args,
    runtime_args=[[writer_rt_args]],
    config=ttnn.WriterConfigDescriptor(),
)

# Compute kernel
compute_kernel = ttnn.KernelDescriptor(
    kernel_source=kernel_source,
    source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
    core_ranges=core_grid,
    compile_time_args=compile_time_args,
    runtime_args=[[compute_rt_args]],
    config=ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.LoFi,
        fp32_dest_acc_en=fp32_acc,
    ),
)

program_descriptor = ttnn.ProgramDescriptor(
    kernels=[reader_kernel, writer_kernel, compute_kernel],
    cbs=[...],
)
```

The kernel config type (`ReaderConfigDescriptor`, `WriterConfigDescriptor`, `ComputeConfigDescriptor`) tells the compiler which `COMPILE_FOR_*` defines to set.

## Benefits

1. **Single kernel file** per fused op - easier to maintain
2. **Reusable micro-ops** - compose new fused ops by combining existing micro-ops
3. **Pure micro-ops** - no I/O inside micro-ops, just receive data and compute
4. **Type-safe RTArgs** - each phase only has its relevant fields
5. **Clean separation** - phase logic hidden inside micro-ops
6. **Type-safe CB chaining** - template params ensure CBs are connected correctly
7. **Zero runtime overhead** - all dispatch is compile-time via `constexpr if` and `std::conditional_t`
