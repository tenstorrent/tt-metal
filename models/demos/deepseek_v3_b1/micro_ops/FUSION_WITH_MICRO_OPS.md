# Micro-Op Architecture for Fused Kernels

This document describes the architecture for composable micro-ops that can be chained together into fused operations with a single unified kernel file.

## Overview

When TT-Metal compiles a kernel, it compiles the same source file multiple times with different preprocessor defines to target different RISC cores:

| Core Type | Define | Role |
|-----------|--------|------|
| **BRISC** | `COMPILE_FOR_BRISC` | Data movement - typically writer |
| **NCRISC** | `COMPILE_FOR_NCRISC` | Data movement - typically reader |
| **TRISC0** | `COMPILE_FOR_TRISC=0`, `TRISC_UNPACK` | Unpack from CB to SRCA/SRCB |
| **TRISC1** | `COMPILE_FOR_TRISC=1`, `TRISC_MATH` | Math operations on FPU |
| **TRISC2** | `COMPILE_FOR_TRISC=2`, `TRISC_PACK` | Pack from DEST to CB |

Our micro-op architecture leverages this by:
1. Each micro-op is a **struct containing nested CTArgs types and an Op class**
2. CTArgs are RISC-specific compile-time argument structs (different layout per RISC)
3. The Op class uses preprocessor-based dispatch (`#if defined(COMPILE_FOR_*)`)
4. Fused kernels instantiate `OpName::Op<CTArgs>` and invoke via `operator()`

## Directory Structure

```
models/demos/deepseek_v3_b1/micro_ops/
├── unified_kernels/
│   ├── kernel_op_api.hpp      # Common includes, macros, type helpers
│   ├── matmul.hpp             # Matmul micro-op (struct with nested types)
│   └── matmul_kernel.cpp      # Unified kernel using Matmul op
```

## Core API

### `unified_kernels/kernel_op_api.hpp`

```cpp
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

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
#define KERNEL_ENTRY      \
    namespace NAMESPACE { \
    void MAIN
#define KERNEL_END }
#else
#define KERNEL_ENTRY void kernel_main()
#define KERNEL_END
#endif

// ============================================================================
// Type helper: Select type based on current RISC core
// Usage: using RTArgs = SelectByRISCV<ReaderArgs, WriterArgs, ComputeArgs>;
// Note: ReaderConfigDescriptor -> NCRISC, WriterConfigDescriptor -> BRISC
// ============================================================================
template <typename Reader, typename Writer, typename Compute>
using SelectByRISCV = std::conditional_t<is_ncrisc, Reader, std::conditional_t<is_brisc, Writer, Compute>>;
```

## Micro-Op Design Pattern

Each micro-op follows a **nested struct pattern** to keep the namespace clean:

```cpp
namespace deepseek_b1_ops {

struct OpName {
    // ========================================================================
    // RISC-specific CTArgs - each RISC has different compile-time args layout
    // ========================================================================

    template <uint32_t Param1, uint32_t Param2, ...>
    struct ReaderCTArgs {
        static constexpr uint32_t param1 = Param1;
        static constexpr uint32_t param2 = Param2;
        // ... reader-specific compile-time constants
    };

    template <uint32_t Param1, ...>
    struct WriterCTArgs {
        static constexpr uint32_t param1 = Param1;
        // ... writer-specific compile-time constants
    };

    template <uint32_t Param1, uint32_t Param2, bool Flag, ...>
    struct ComputeCTArgs {
        static constexpr uint32_t param1 = Param1;
        static constexpr uint32_t param2 = Param2;
        static constexpr bool flag = Flag;
        // ... compute-specific compile-time constants
    };

    // ========================================================================
    // Op - the actual operation, templated on CTArgs
    // ========================================================================
    template <typename CTArgs>
    class Op {
    public:
        // Optional: Phase-specific runtime args
        struct ReaderArgs { /* runtime args for reader */ };
        struct WriterArgs { /* runtime args for writer */ };
        struct ComputeArgs { /* runtime args for compute */ };

        using RTArgs = SelectByRISCV<ReaderArgs, WriterArgs, ComputeArgs>;

        // Entry point - callable like a function
        void operator()(const RTArgs& = {}) { impl(); }

    private:
        void impl() {
#if defined(COMPILE_FOR_NCRISC)
            // Reader logic using CTArgs::param1, CTArgs::param2, etc.
#endif

#if defined(COMPILE_FOR_BRISC)
            // Writer logic using CTArgs::param1, etc.
#endif

#if defined(COMPILE_FOR_TRISC)
            // Compute logic using CTArgs::param1, CTArgs::param2, CTArgs::flag, etc.
#endif
        }
    };
};

}  // namespace deepseek_b1_ops
```

### Key Design Decisions

1. **Nested struct pattern**: `Matmul::ReaderCTArgs`, `Matmul::WriterCTArgs`, `Matmul::ComputeCTArgs` keeps the namespace clean. Only one `Matmul` symbol is exposed instead of 4 separate symbols.

2. **RISC-specific CTArgs**: Each RISC can have a completely different set of compile-time args. This avoids passing unused parameters and enables optimal code generation.

3. **Preprocessor dispatch in Op**: The `impl()` method uses `#if defined(COMPILE_FOR_*)` to include only the relevant code path for each RISC.

4. **Callable via `operator()`**: The Op can be invoked like a function: `matmul()` instead of `matmul.run()`.

## Matmul Micro-Op

### `unified_kernels/matmul.hpp`

```cpp
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "kernel_op_api.hpp"

#if defined(COMPILE_FOR_BRISC)
#include "dataflow_api.h"
#elif defined(COMPILE_FOR_NCRISC)
#include "dataflow_api.h"
#elif defined(COMPILE_FOR_TRISC)
#include "compute_kernel_api.h"
#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api/tile_move_copy.h"
#endif

namespace deepseek_b1_ops {

// ============================================================================
// Single-tile output Matmul micro-op
//
// Computes: output[1,1] = in0[1,K] @ in1[K,1]
// ============================================================================
struct Matmul {
    // ========================================================================
    // Compile-time args structs - different layout per RISC
    // ========================================================================

    // Reader CTArgs: [in0_cb, in1_cb, num_tiles_k]
    template <uint32_t In0CB, uint32_t In1CB, uint32_t NumTilesK, bool PopInputs = true>
    struct ReaderCTArgs {
        static constexpr uint32_t in0_cb = In0CB;
        static constexpr uint32_t in1_cb = In1CB;
        static constexpr uint32_t num_tiles_k = NumTilesK;
        static constexpr bool pop_inputs = PopInputs;
    };

    // Writer CTArgs: [out_cb]
    template <uint32_t OutCB>
    struct WriterCTArgs {
        static constexpr uint32_t out_cb = OutCB;
    };

    // Compute CTArgs: [in0_cb, in1_cb, out_cb, interm_cb, num_tiles_k, fp32_acc]
    template <
        uint32_t In0CB,
        uint32_t In1CB,
        uint32_t OutCB,
        uint32_t IntermCB,
        uint32_t NumTilesK,
        bool FP32Acc,
        bool PopInputs = true>
    struct ComputeCTArgs {
        static constexpr uint32_t in0_cb = In0CB;
        static constexpr uint32_t in1_cb = In1CB;
        static constexpr uint32_t out_cb = OutCB;
        static constexpr uint32_t interm_cb = IntermCB;
        static constexpr uint32_t num_tiles_k = NumTilesK;
        static constexpr bool fp32_acc = FP32Acc;
        static constexpr bool pop_inputs = PopInputs;
    };

    // ========================================================================
    // Op - the actual operation, templated on CTArgs
    // ========================================================================
    template <typename CTArgs>
    class Op {
    public:
        // Phase-specific RTArgs (none for this simple matmul)
        struct ReaderArgs {};
        struct WriterArgs {};
        struct ComputeArgs {};

        using RTArgs = SelectByRISCV<ReaderArgs, WriterArgs, ComputeArgs>;

        void operator()(const RTArgs& = {}) { impl(); }

    private:
        void impl() {
#if defined(COMPILE_FOR_NCRISC)
            // ================================================================
            // NCRISC (Reader) - ReaderConfigDescriptor compiles as NCRISC
            // ================================================================
            // Both in0 and in1 are backed by sharded tensors - just signal they're ready
            cb_reserve_back(CTArgs::in0_cb, CTArgs::num_tiles_k);
            cb_push_back(CTArgs::in0_cb, CTArgs::num_tiles_k);

            cb_reserve_back(CTArgs::in1_cb, CTArgs::num_tiles_k);
            cb_push_back(CTArgs::in1_cb, CTArgs::num_tiles_k);
#endif

#if defined(COMPILE_FOR_BRISC)
            // ================================================================
            // BRISC (Writer) - WriterConfigDescriptor compiles as BRISC
            // ================================================================
            // Wait for output tile to be ready
            // Note: out_cb is backed by sharded tensor, data written directly to L1
            cb_wait_front(CTArgs::out_cb, 1);
#endif

#if defined(COMPILE_FOR_TRISC)
            // ================================================================
            // TRISC (Compute)
            // ================================================================
            constexpr uint32_t out_subblock_h = 1;
            constexpr uint32_t out_subblock_w = 1;
            constexpr uint32_t in0_block_w = 1;  // Process one K tile at a time

            // Initialize matmul
            mm_block_init(
                CTArgs::in0_cb, CTArgs::in1_cb, CTArgs::out_cb, false, out_subblock_w, out_subblock_h, in0_block_w);

            // Wait for all input tiles (both from sharded tensors in L1)
            cb_wait_front(CTArgs::in0_cb, CTArgs::num_tiles_k);
            cb_wait_front(CTArgs::in1_cb, CTArgs::num_tiles_k);

            // Reserve output
            cb_reserve_back(CTArgs::out_cb, 1);

            // Accumulate across K dimension
            tile_regs_acquire();

            for (uint32_t k = 0; k < CTArgs::num_tiles_k; k++) {
                matmul_tiles(CTArgs::in0_cb, CTArgs::in1_cb, k, k, 0, false);
            }

            tile_regs_commit();

            // Pop inputs
            if constexpr (CTArgs::pop_inputs) {
                cb_pop_front(CTArgs::in0_cb, CTArgs::num_tiles_k);
                cb_pop_front(CTArgs::in1_cb, CTArgs::num_tiles_k);
            }

            // Pack output
            tile_regs_wait();
            pack_tile(0, CTArgs::out_cb);
            tile_regs_release();

            cb_push_back(CTArgs::out_cb, 1);
#endif
        }
    };  // class Op

};  // struct Matmul

}  // namespace deepseek_b1_ops
```

## Unified Kernel

### `unified_kernels/matmul_kernel.cpp`

The kernel file:
1. Includes the op header
2. Creates a type alias for the op struct
3. Defines RISC-specific CTArgs using `#if defined(COMPILE_FOR_*)`
4. Instantiates and invokes the Op

```cpp
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Matmul unified kernel
// Single kernel file, compiles correctly for all RISC cores
// Each RISC has its own CTArgs struct with different compile-time arg layout

#include "kernel_op_api.hpp"
#include "matmul.hpp"

KERNEL_ENTRY {
    using Matmul = deepseek_b1_ops::Matmul;

// ============================================================================
// NCRISC (Reader) - ReaderConfigDescriptor compiles as NCRISC
// Compile-time args: [in0_cb, in1_cb, num_tiles_k]
// ============================================================================
#if defined(COMPILE_FOR_NCRISC)
    using CTArgs = Matmul::ReaderCTArgs<
        get_compile_time_arg_val(0),  // in0_cb
        get_compile_time_arg_val(1),  // in1_cb
        get_compile_time_arg_val(2),  // num_tiles_k
        true                          // pop_inputs
        >;
#endif  // COMPILE_FOR_NCRISC

// ============================================================================
// BRISC (Writer) - WriterConfigDescriptor compiles as BRISC
// Compile-time args: [out_cb]
// ============================================================================
#if defined(COMPILE_FOR_BRISC)
    using CTArgs = Matmul::WriterCTArgs<get_compile_time_arg_val(0)  // out_cb
                                        >;
#endif  // COMPILE_FOR_BRISC

// ============================================================================
// TRISC (Compute) - ComputeConfigDescriptor compiles as TRISC
// Compile-time args: [in0_cb, in1_cb, out_cb, interm_cb, num_tiles_k, fp32_acc]
// ============================================================================
#if defined(COMPILE_FOR_TRISC)
    using CTArgs = Matmul::ComputeCTArgs<
        get_compile_time_arg_val(0),  // in0_cb
        get_compile_time_arg_val(1),  // in1_cb
        get_compile_time_arg_val(2),  // out_cb
        get_compile_time_arg_val(3),  // interm_cb
        get_compile_time_arg_val(4),  // num_tiles_k
        get_compile_time_arg_val(5),  // fp32_acc
        true                          // pop_inputs
        >;
#endif  // COMPILE_FOR_TRISC

    Matmul::Op<CTArgs> matmul;
    matmul();
}
KERNEL_END
```

## RISC-Specific Compile-Time Args

A key feature of this design is that **each RISC can have a completely different compile-time args layout**:

| RISC | CTArgs Type | Compile-Time Args |
|------|-------------|-------------------|
| NCRISC (Reader) | `Matmul::ReaderCTArgs` | `[in0_cb, in1_cb, num_tiles_k]` |
| BRISC (Writer) | `Matmul::WriterCTArgs` | `[out_cb]` |
| TRISC (Compute) | `Matmul::ComputeCTArgs` | `[in0_cb, in1_cb, out_cb, interm_cb, num_tiles_k, fp32_acc]` |

This is possible because:
1. Each RISC kernel is a separate compilation with its own `KernelDescriptor`
2. Each descriptor can pass different `compile_time_args` arrays
3. The kernel uses `#if defined(COMPILE_FOR_*)` to select the appropriate CTArgs type

## Python Op Wrapper

```python
# All three kernels use the SAME source file
kernel_source = "models/demos/deepseek_v3_b1/micro_ops/unified_kernels/matmul_kernel.cpp"

# ============================================================================
# Compile-time args: RISC-SPECIFIC (different layout per RISC)
# ============================================================================

# Reader: [in0_cb, in1_cb, num_tiles_k]
reader_ct_args = [in0_cb, in1_cb, num_tiles_k]

# Writer: [out_cb]
writer_ct_args = [out_cb]

# Compute: [in0_cb, in1_cb, out_cb, interm_cb, num_tiles_k, fp32_acc]
compute_ct_args = [in0_cb, in1_cb, out_cb, interm_cb, num_tiles_k, fp32_acc]

# ============================================================================
# Kernel descriptors
# ============================================================================

# Reader kernel (NCRISC)
reader_kernel = ttnn.KernelDescriptor(
    kernel_source=kernel_source,
    source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
    core_ranges=core_grid,
    compile_time_args=reader_ct_args,
    runtime_args=[[]],
    config=ttnn.ReaderConfigDescriptor(),
)

# Writer kernel (BRISC)
writer_kernel = ttnn.KernelDescriptor(
    kernel_source=kernel_source,
    source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
    core_ranges=core_grid,
    compile_time_args=writer_ct_args,
    runtime_args=[[]],
    config=ttnn.WriterConfigDescriptor(),
)

# Compute kernel (TRISC)
compute_kernel = ttnn.KernelDescriptor(
    kernel_source=kernel_source,
    source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
    core_ranges=core_grid,
    compile_time_args=compute_ct_args,
    runtime_args=[[]],
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

## Fused Op Example

For a fused RMSNorm → Matmul op, the pattern extends naturally:

```cpp
KERNEL_ENTRY {
    using RMSNorm = deepseek_b1_ops::RMSNorm;
    using Matmul = deepseek_b1_ops::Matmul;

#if defined(COMPILE_FOR_NCRISC)
    using RMSNormCTArgs = RMSNorm::ReaderCTArgs<...>;
    using MatmulCTArgs = Matmul::ReaderCTArgs<...>;
#endif

#if defined(COMPILE_FOR_BRISC)
    using RMSNormCTArgs = RMSNorm::WriterCTArgs<...>;
    using MatmulCTArgs = Matmul::WriterCTArgs<...>;
#endif

#if defined(COMPILE_FOR_TRISC)
    using RMSNormCTArgs = RMSNorm::ComputeCTArgs<...>;
    using MatmulCTArgs = Matmul::ComputeCTArgs<...>;
#endif

    RMSNorm::Op<RMSNormCTArgs> rmsnorm;
    Matmul::Op<MatmulCTArgs> matmul;

    rmsnorm();
    matmul();
}
KERNEL_END
```

## Benefits

1. **Clean namespace**: Each op is a single struct (`Matmul`, `RMSNorm`) instead of multiple symbols (`MatmulReaderCTArgs`, `MatmulWriterCTArgs`, etc.)

2. **RISC-specific optimization**: Each RISC gets exactly the compile-time args it needs, no wasted template parameters

3. **Single kernel file**: One `.cpp` file compiles correctly for all RISC cores

4. **Composable**: Ops can be chained together in fused kernels

5. **Type-safe**: CTArgs are template parameters, so mismatches are caught at compile time

6. **Zero runtime overhead**: All dispatch is compile-time via `#if defined()` and `constexpr`

7. **Familiar syntax**: `Matmul::Op<CTArgs> matmul; matmul();` reads naturally
