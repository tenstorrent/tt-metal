# TT-Metal JIT Build System

This directory contains the Just-In-Time (JIT) compilation infrastructure for Metal kernels and firmware.

## Overview

Metal uses JIT compilation to build RISC-V kernels and firmware at runtime using the SFPI (Tensix Floating Point Instruction) compiler (`riscv32-tt-elf-g++`). The JIT build system:

- Compiles data movement (DM) kernels for BRISC, NCRISC, and ERISC processors
- Compiles compute kernels for TRISC processors (TRISC0, TRISC1, TRISC2)
- Links kernels with firmware and linker scripts
- Manages build artifacts in a cache directory
- Tracks source dependencies for incremental builds

## Key Components

- **`build.hpp/cpp`**: Core JIT build logic
  - `JitBuildEnv`: Build environment (compiler paths, flags, defines)
  - `JitBuildState`: State for a specific kernel/firmware build
  - Compilation, linking, and dependency tracking

- **`build_env_manager.hpp/cpp`**: Singleton managing build environments per device
- **`depend.hpp/cpp`**: Dependency parsing and hash-based change detection
- **`jit_build_utils.hpp/cpp`**: Utilities (FNV1a hash, command execution)
- **`kernel_args.hpp/cpp`**: Kernel argument handling

## CCache Support for Kernel Compilation

### Overview

Metal's JIT kernel compilation supports **ccache** to accelerate repeated builds. When enabled via `TT_METAL_CCACHE_KERNEL_SUPPORT=1`, ccache wraps the SFPI compiler invocations for both compilation and linking stages.

**Important**: This applies to **all kernel types**, including:
- Data movement kernels (BRISC, NCRISC, ERISC)
- Compute kernels (TRISC0, TRISC1, TRISC2) compiled with the SFPI GCC compiler

### Enabling CCache

Set the environment variable before running your application:

```bash
export TT_METAL_CCACHE_KERNEL_SUPPORT=1
```

There are no additional flags or configuration needed. The single `TT_METAL_CCACHE_KERNEL_SUPPORT` variable enables ccache for all kernel types.

### Redis Remote Storage Backend

CCache can be configured to use Redis as a remote storage backend, enabling cache sharing across multiple machines (e.g., in CI/CD environments).

**Configuration**: Use standard ccache environment variables:

```bash
export TT_METAL_CCACHE_KERNEL_SUPPORT=1
export CCACHE_REMOTE_STORAGE="redis://user:password@host:port"
export CCACHE_REMOTE_ONLY=true           # Optional: use only remote cache
export CCACHE_TEMPDIR=/tmp/ccache        # Required for remote-only mode
export CCACHE_DIR=~/.ccache              # Local cache directory (if not remote-only)
```

**Automatic Configuration**:
- `CCACHE_BASEDIR` is automatically set to the Metal root directory if not already set
- This ensures cache keys are stable across different checkout directories
- Cache keys include: git hash, architecture, compiler flags, defines, source files, firmware, linker scripts

### Cache Key Stability

The JIT build system ensures cache keys are deterministic and include all inputs:

1. **Git commit hash**: Included in output path
2. **Build configuration**: Architecture (Wormhole, Blackhole), processor type, processor ID
3. **Compiler flags**: All `-D` defines, `-I` includes, optimization levels
4. **Source dependencies**: Source files, headers (via gcc `-MMD`), firmware ELF, linker scripts
5. **Compiler version**: Implicitly tracked by ccache

Changing any of these invalidates the cache entry.

### Logging and Debugging

Enable detailed logging to observe ccache behavior:

```bash
export TT_METAL_LOGGER_LEVEL=Debug
export TT_METAL_LOGGER_TYPES=BuildKernels
export TT_METAL_LOG_KERNELS_COMPILE_COMMANDS=1
```

Logs will show:
- `CCACHE: Kernel compilation will use ccache`
- `CCACHE: Remote storage configured` (if using Redis)
- `CCACHE: Set CCACHE_BASEDIR=<path>`
- `CCACHE_DEBUG: Full compile command: ccache <compiler> <args>`

For ccache-specific debugging:

```bash
export CCACHE_DEBUG=1
export CCACHE_LOGFILE=/tmp/ccache.log
```

### Performance Considerations

**Cache Hits**: When ccache finds a matching entry:
- Compilation is skipped entirely (object files retrieved from cache)
- Linking still occurs (to produce final ELF)
- Build time reduced by ~80-90% for cache hits

**Cache Misses**: First time compiling a kernel or after source changes:
- Slight overhead from ccache hash computation (~1-2%)
- Object files are stored in cache for future use

**Remote Storage**: Redis backend adds network latency:
- Upload time after first compilation
- Download time on cache hit (typically faster than compilation)
- Shared cache benefits outweigh latency in CI environments

### Testing

Run the kernel ccache tests to verify functionality:

```bash
# Enable ccache for tests
export TT_METAL_CCACHE_KERNEL_SUPPORT=1

# Run ccache-specific tests (includes SFPI compute kernels)
pytest tests/tt_metal/tt_metal/api/test_kernel_compile_cache.cpp
```

Tests verify:
- Cache hits on repeated compilation of identical kernels
- Cache invalidation when source changes
- Different processors produce different cache entries
- SFPI compute kernels are properly cached

## Build Artifacts

Default cache location: `~/.cache/tt-metal-cache/<git-hash>/<build-key>/`

Structure:
```
~/.cache/tt-metal-cache/
└── <git-hash>/
    └── <build-key>/
        ├── firmware/
        │   ├── brisc/
        │   ├── ncrisc/
        │   ├── trisc0/
        │   ├── trisc1/
        │   └── trisc2/
        └── kernels/
            └── <kernel-name>/
                ├── brisc/
                ├── ncrisc/
                ├── trisc0/
                ├── trisc1/
                └── trisc2/
```

Each directory contains:
- `.o` files (object files)
- `.elf` files (executables)
- `.d` files (Makefile dependencies)
- `.dephash` files (dependency hashes for incremental builds)
- `.SUCCESS` marker (indicates successful build)

## Environment Variables

### Build Control
- `TT_METAL_HOME`: Metal installation root
- `TT_METAL_CACHE_DIR`: Override default cache location

### CCache
- `TT_METAL_CCACHE_KERNEL_SUPPORT=1`: Enable ccache for kernel compilation
- `CCACHE_BASEDIR`: Base directory for path normalization (auto-set to Metal root)
- `CCACHE_DIR`: Local cache directory (default: `~/.ccache`)
- `CCACHE_REMOTE_STORAGE`: Remote storage URL (e.g., Redis)
- `CCACHE_REMOTE_ONLY`: Use only remote cache, no local cache
- `CCACHE_TEMPDIR`: Temp directory for remote-only mode

### Logging
- `TT_METAL_LOGGER_LEVEL`: Logging level (Debug, Info, Warning, Error)
- `TT_METAL_LOGGER_TYPES`: Logger types (BuildKernels)
- `TT_METAL_LOG_KERNELS_COMPILE_COMMANDS=1`: Log full compile/link commands

### Debugging
- `TT_METAL_FORCE_JIT_COMPILE=1`: Force recompilation (ignore cache)
- `CCACHE_DEBUG=1`: Enable ccache debug output
- `CCACHE_LOGFILE`: Path to ccache debug log

## Implementation Notes

### SFPI Compiler

Metal uses the SFPI RISC-V GCC compiler toolchain:
- Path: `runtime/sfpi/compiler/bin/riscv32-tt-elf-g++` (or `/opt/tenstorrent/sfpi`)
- Target: RISC-V 32-bit with Tensix extensions
- Flags: `-O3 -std=c++17 -flto=auto -ffast-math -fno-exceptions`
- Architecture-specific: `-mcpu=tt-wh-tensix` or `-mcpu=tt-bh-tensix`

### Linker Scripts

Each processor type has a dedicated linker script:
- `runtime/hw/toolchain/<arch>/kernel_brisc.ld`
- `runtime/hw/toolchain/<arch>/kernel_ncrisc.ld`
- `runtime/hw/toolchain/<arch>/kernel_trisc0.ld`
- `runtime/hw/toolchain/<arch>/kernel_trisc1.ld`
- `runtime/hw/toolchain/<arch>/kernel_trisc2.ld`

Kernels link against weakened firmware symbols via `--just-symbols`.

### Dependency Tracking

The build system uses GCC's `-MMD` flag to generate Makefile-style dependency files (`.d`). These track source file includes. The `depend.cpp` module:
1. Parses `.d` files to extract dependencies
2. Computes FNV1a hashes of all dependency contents
3. Stores hashes in `.dephash` files
4. Checks hashes on subsequent builds to determine if recompilation is needed

This is orthogonal to ccache: dependency tracking handles local incremental builds, while ccache provides distributed/persistent caching.

## See Also

- [Code Indexing for Kernels](../../tech_reports/code-indexing/kernel-code-indexing.md)
- [Metalium Programming Guide](../../METALIUM_GUIDE.md)
- [SFPI Repository](https://github.com/tenstorrent/sfpi)
