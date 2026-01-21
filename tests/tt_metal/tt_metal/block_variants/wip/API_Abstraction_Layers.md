# API Abstraction Layers

This document explains the different abstraction layers in Tenstorrent's software stack, from the lowest-level hardware interfaces to high-level compute kernels.

## Overview

The Tenstorrent software stack consists of multiple abstraction layers that progressively hide hardware complexity while providing increasingly developer-friendly interfaces. Each layer builds upon the previous one to create a comprehensive programming model.

```
┌─────────────────────────────────────┐
│           Compute Kernels           │  ← Highest Level
├─────────────────────────────────────┤
│            Compute API              │  ← Multi-threaded abstraction
├─────────────────────────────────────┤
│           Low Level API             │  ← Single-threaded with tt-metal concepts
├─────────────────────────────────────┤
│       LLKs (Low Level Kernels)      │  ← Lowest Level
└─────────────────────────────────────┘
```

## 1. LLKs (Low Level Kernels)

**Purpose**: Hardware-level interface for standalone software development

### Key Characteristics:

- **Standalone Development**: Allow for completely independent software development without dependencies on higher-level frameworks
- **Hardware-Direct**: Direct interface to Tensix hardware capabilities
- **Framework Agnostic**: Don't use tt-metal concepts like circular buffers
- **Minimal Abstraction**: Closest possible interface to the actual hardware

### Repository:
- [tt-llk GitHub Repository](https://github.com/tenstorrent/tt-llk)

### Use Cases:
- Hardware validation and testing
- Performance benchmarking at the lowest level
- Custom hardware-specific optimizations
- Bring-up and debugging of new hardware features

---

## 2. Low Level API

**Purpose**: Single-threaded functions that expose the underlying programming model

### Key Characteristics:

- **tt-metal Integration**: Uses tt-metal concepts such as circular buffers
- **Single-threaded**: Functions operate on individual threads
- **Programming Model Exposure**: Explicitly exposes the fact that TRISCs (Tensix RISC cores) are specialized for different Tensix Execution Units (EXUs)
- **Thread Specialization**: Developers must understand which TRISC drives which Tensix EXU

### Programming Model Details:

The Low Level API requires developers to understand the underlying hardware architecture:

- **UNPACK Thread**: Handles data unpacking operations
- **MATH Thread**: Performs mathematical computations
- **PACK Thread**: Handles data packing operations

Each thread corresponds to specific Tensix execution units and must be programmed explicitly.

### Example Usage Pattern:
```cpp
// Developer must explicitly handle each thread
UNPACK((llk_unpack_A_init(...)));
MATH((llk_math_eltwise_unary_datacopy_init(...)));
PACK((llk_pack_init(...)));
```

---

## 3. Compute API

**Purpose**: Multi-threaded functions that hide the programming model complexity

### Key Characteristics:

- **Multi-threaded Abstraction**: Functions appear single-threaded to the developer but internally coordinate multiple threads
- **Programming Model Hiding**: Developers don't need to know which TRISC drives which Tensix EXU
- **Simplified Development**: Greatly simplifies kernel development by abstracting away hardware threading details
- **High-level Operations**: Provides operation-centric rather than thread-centric interfaces

### Benefits:

1. **Ease of Use**: Developers can focus on algorithmic logic rather than hardware threading
2. **Reduced Complexity**: No need to coordinate between different specialized threads
3. **Better Maintainability**: Higher-level abstractions are easier to maintain and debug
4. **Faster Development**: Accelerated kernel development cycle

### Example Usage Pattern:
```cpp
// Single function call handles all thread coordination internally
copy_tile(input_cb, tile_index, dest_tile_index);
```

---

## 4. Compute Kernels

**Purpose**: Complete kernel implementations using the Compute API

### Key Characteristics:

- **Application-Ready**: Full kernel implementations for common operations
- **Compute API Built**: Leverage the simplified Compute API for implementation
- **Operation-Focused**: Organized around mathematical and data processing operations
- **Production-Ready**: Optimized and tested for real-world workloads

### Examples:
- Matrix multiplication kernels
- Elementwise operation kernels
- Reduction operation kernels
- Data movement and transformation kernels

---

## Abstraction Layer Benefits

### For LLKs:
- **Maximum Control**: Complete hardware access and control
- **Minimal Overhead**: Direct hardware interface with minimal abstraction penalty
- **Hardware Validation**: Ideal for testing and validating hardware functionality

### For Low Level API:
- **tt-metal Integration**: Seamless integration with tt-metal infrastructure
- **Explicit Control**: Fine-grained control over threading and execution
- **Performance Optimization**: Ability to optimize at the thread level

### For Compute API:
- **Developer Productivity**: Significantly faster kernel development
- **Reduced Errors**: Less chance for threading-related bugs
- **Maintainability**: Easier to understand and maintain code

### For Compute Kernels:
- **Ready-to-Use**: Immediate availability of common operations
- **Optimized**: Pre-optimized implementations for best performance
- **Tested**: Thoroughly validated implementations

---

## Choosing the Right Abstraction Level

### Use LLKs when:
- Developing standalone applications
- Performing hardware validation
- Requiring maximum hardware control
- Working without tt-metal framework

### Use Low Level API when:
- Need fine-grained thread control
- Implementing custom optimizations
- Requiring explicit programming model access
- Building new Compute API functions

### Use Compute API when:
- Developing application kernels
- Want simplified programming model
- Need faster development cycles
- Focusing on algorithmic implementation

### Use Compute Kernels when:
- Need standard operations
- Want production-ready implementations
- Require optimized performance
- Building applications quickly

---

## Migration Path

The abstraction layers are designed to allow gradual migration and mixed usage:

1. **Bottom-Up Development**: Start with LLKs for hardware validation, move up to higher levels for application development
2. **Top-Down Optimization**: Begin with Compute Kernels, drop to lower levels for specific optimizations
3. **Mixed Approach**: Use different abstraction levels for different parts of the same application based on requirements

This layered approach provides flexibility while maintaining performance and enabling rapid development across different use cases and expertise levels.
