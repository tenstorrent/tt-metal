# Tenstorrent Hardware Architecture Overview

Quick reference for understanding Tensix architecture and common execution patterns.

---

## Memory Hierarchy

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              HOST (CPU)                                     │
│                                  │                                          │
│                                  ▼                                          │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                         DRAM (Off-chip)                               │  │
│  │  • Large capacity (~GB)                                               │  │
│  │  • Slower access                                                      │  │
│  │  • INTERLEAVED layout: data striped across DRAM banks                │  │
│  │  • Contains: Input tensors, Output tensors, Weight tensors            │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                  │                                          │
│                        NOC (Network on Chip)                                │
│                                  │                                          │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                         Tensix Cores Grid                             │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │  │
│  │  │  Core(0,0)  │  │  Core(0,1)  │  │  Core(0,2)  │  │    ...      │   │  │
│  │  │    ┌────┐   │  │    ┌────┐   │  │    ┌────┐   │  │             │   │  │
│  │  │    │ L1 │   │  │    │ L1 │   │  │    │ L1 │   │  │             │   │  │
│  │  │    │SRAM│   │  │    │SRAM│   │  │    │SRAM│   │  │             │   │  │
│  │  │    └────┘   │  │    └────┘   │  │    └────┘   │  │             │   │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘   │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key Terminology

```yaml
terminology:
  L1:
    aliases: [SRAM, "Local Memory"]
    location: "On-chip, per Tensix core"
    capacity: "~1-2 MB per core"
    access: "Fast, local to core"
    usage: "Circular Buffers (CBs), intermediate data"

  DRAM:
    aliases: ["Global Memory", "Off-chip Memory"]
    location: "Off-chip"
    capacity: "~GB total"
    access: "Slower, requires NOC transfer"
    usage: "Input/output tensors, weights"

  INTERLEAVED:
    meaning: "Data layout where tensor pages are striped across DRAM banks"
    implication: "Data resides in DRAM, requires Reader/Writer kernels"
    pattern: "Reader-Compute-Writer"

  HEIGHT_SHARDED:
    meaning: "Each core's L1 holds a horizontal slice of the tensor"
    implication: "Data pre-loaded in L1, zero-copy input"
    pattern: "Signal-Compute-Output"

  WIDTH_SHARDED:
    meaning: "Each core's L1 holds a vertical slice of the tensor"
    implication: "Data pre-loaded in L1, zero-copy input"

  CB:
    full_name: "Circular Buffer"
    location: "Allocated in L1 (SRAM)"
    purpose: "Synchronization between Reader-Compute-Writer kernels"
    indices: "0-31 (c_0, c_1, ... c_31)"
```

---

## Tensix Core Architecture

Each Tensix core contains multiple RISC-V processors running in parallel:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           TENSIX CORE                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐             │
│  │   BRISC         │  │   NCRISC        │  │   TRISC (x3)    │             │
│  │   (Reader)      │  │   (Writer)      │  │   (Compute)     │             │
│  │                 │  │                 │  │                 │             │
│  │  • DRAM → L1    │  │  • L1 → DRAM    │  │  • Unpack       │             │
│  │  • NOC reads    │  │  • NOC writes   │  │  • Math/SFPU    │             │
│  │  • CB push      │  │  • CB pop       │  │  • Pack         │             │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘             │
│           │                    │                    │                       │
│           └───────────────────────────────────────────────────┐             │
│                               │                               │             │
│  ┌────────────────────────────▼───────────────────────────────▼──────────┐  │
│  │                        L1 SRAM (~1-2 MB)                              │  │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐    │  │
│  │  │  CB[0]   │ │  CB[1]   │ │  CB[2]   │ │  CB[3]   │ │   ...    │    │  │
│  │  │  Input   │ │Input(bin)│ │  Output  │ │Intermed. │ │          │    │  │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘    │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                    Compute Engine (TRISC)                             │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │  │
│  │  │  DST Registers (8 tiles × 32×32 elements)                       │  │  │
│  │  │  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐│  │  │
│  │  │  │DST0 │ │DST1 │ │DST2 │ │DST3 │ │DST4 │ │DST5 │ │DST6 │ │DST7 ││  │  │
│  │  │  └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘│  │  │
│  │  └─────────────────────────────────────────────────────────────────┘  │  │
│  │  ┌─────────────────────┐  ┌─────────────────────┐                     │  │
│  │  │  FPU (Matrix Ops)   │  │  SFPU (Elem-wise)   │                     │  │
│  │  └─────────────────────┘  └─────────────────────┘                     │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Reader-Compute-Writer Pattern

**When**: `memory_layout == INTERLEAVED` (data in DRAM)

This is the fundamental execution pattern for operations on DRAM-resident data.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    READER-COMPUTE-WRITER PATTERN                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  DRAM                    L1 (SRAM)                    DRAM                  │
│  ┌──────┐               ┌──────────┐               ┌──────┐                │
│  │Input │               │ CB[in]   │               │Output│                │
│  │Tensor│               │          │               │Tensor│                │
│  └──┬───┘               └────┬─────┘               └──▲───┘                │
│     │                        │                        │                     │
│     │    ┌───────────────────┼───────────────────┐    │                     │
│     │    │                   │                   │    │                     │
│     ▼    ▼                   ▼                   ▼    │                     │
│  ┌──────────┐         ┌──────────────┐         ┌──────────┐                │
│  │  READER  │────────▶│   COMPUTE    │────────▶│  WRITER  │                │
│  │  Kernel  │         │    Kernel    │         │  Kernel  │                │
│  │  (BRISC) │         │   (TRISC)    │         │ (NCRISC) │                │
│  └──────────┘         └──────────────┘         └──────────┘                │
│                                                                             │
│  Phase 1:              Phase 2:                 Phase 3:                    │
│  DRAM → CB[in]         CB[in] → DST → CB[out]   CB[out] → DRAM             │
│                                                                             │
│  noc_async_read_tile   copy_tile                noc_async_write_tile       │
│  cb_push_back          compute_op               cb_wait_front              │
│                        pack_tile                                            │
│                        cb_push_back                                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Kernel Roles

```yaml
kernels:
  reader:
    processor: BRISC
    role: "Fetch data from DRAM to L1"
    data_flow: "DRAM → CB[input]"
    key_ops:
      - noc_async_read_tile   # DRAM → L1
      - noc_async_read_barrier # Wait for transfer
      - cb_push_back          # Signal data ready

  compute:
    processor: "TRISC (3 sub-processors: Unpack, Math, Pack)"
    role: "Transform data in L1"
    data_flow: "CB[in] → DST → CB[out]"
    key_ops:
      - cb_wait_front   # Wait for input
      - copy_tile       # CB → DST
      - <compute_op>    # SFPU/FPU operation
      - pack_tile       # DST → CB
      - cb_pop_front    # Release input
      - cb_push_back    # Signal output ready

  writer:
    processor: NCRISC
    role: "Store data from L1 to DRAM"
    data_flow: "CB[output] → DRAM"
    key_ops:
      - cb_wait_front          # Wait for data
      - noc_async_write_tile   # L1 → DRAM
      - noc_async_write_barrier # Wait for transfer
      - cb_pop_front           # Release CB space
```

### CB Synchronization

```
Reader                    Compute                   Writer
   │                         │                         │
   │  cb_push_back(in,1)     │                         │
   │ ──────────────────────▶ │                         │
   │                         │  cb_wait_front(in,1)    │
   │                         │  (blocks until data)    │
   │                         │                         │
   │                         │  ... compute ...        │
   │                         │                         │
   │                         │  cb_push_back(out,1)    │
   │                         │ ──────────────────────▶ │
   │                         │                         │  cb_wait_front(out,1)
   │                         │  cb_pop_front(in,1)     │
   │                         │ ◀────────────────────── │
   │  (space freed in CB)    │                         │  ... write ...
   │ ◀────────────────────── │                         │
   │                         │                         │  cb_pop_front(out,1)
```

---

## Zero-Copy Pattern (Sharded)

**When**: `memory_layout == HEIGHT_SHARDED | WIDTH_SHARDED | BLOCK_SHARDED`

Data is pre-loaded in L1 before kernel execution. No DRAM transfers needed.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        ZERO-COPY PATTERN (Sharded)                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  L1 already contains data (globally allocated CB points to shard)          │
│                                                                             │
│  ┌──────────┐               ┌──────────────┐               ┌──────────┐    │
│  │ CB[in]   │ ────────────▶ │   COMPUTE    │ ────────────▶ │ CB[out]  │    │
│  │ (shard)  │               │    Kernel    │               │ (shard)  │    │
│  │          │               │              │               │          │    │
│  │ No read! │               │  cb_wait_front(in)          │ No write!│    │
│  │ Signal   │               │  compute...                 │ Data     │    │
│  │ only     │               │  cb_push_back(out)          │ stays    │    │
│  └──────────┘               └──────────────┘               └──────────┘    │
│                                                                             │
│  Reader kernel:            Compute kernel:           Writer kernel:         │
│  cb_push_back(in, n)       (same as interleaved)     (may be no-op or      │
│  (signal data ready,       cb_pop_front/push_back    just remaps address)  │
│   no actual read)                                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

CB allocation for sharded:
  - set_globally_allocated_address(cb, shard_address)
  - CB directly points to tensor shard in L1
  - No data copy, just synchronization signals
```

---

## Program Loading

```yaml
program_loading:
  steps:
    1. Host creates Program:
       - "Program contains kernel definitions and CB configurations"

    2. Host specifies kernels:
       - "Reader kernel (BRISC)"
       - "Compute kernel (TRISC)"
       - "Writer kernel (NCRISC)"

    3. Host specifies core range:
       - "Which cores execute this program"
       - "Work distribution: tiles split across cores"

    4. Host enqueues program:
       - "Kernels compiled and loaded to cores"
       - "Runtime args set per-core"

    5. Execution:
       - "All 3 kernels run in parallel on each core"
       - "Synchronized via CB push/pop"

  code_pattern: |
    auto program = CreateProgram();

    // Create CBs (in L1/SRAM)
    CircularBuffer cb_in = CreateCircularBuffer(program, c_0, ...);
    CircularBuffer cb_out = CreateCircularBuffer(program, c_2, ...);

    // Create kernels
    auto reader = CreateKernel(program, "reader.cpp", core_range, ReaderConfig);
    auto compute = CreateKernel(program, "compute.cpp", core_range, ComputeConfig);
    auto writer = CreateKernel(program, "writer.cpp", core_range, WriterConfig);

    // Set runtime args per core
    for (auto core : cores) {
        SetRuntimeArgs(program, reader, core, {src_addr, num_tiles, start_id});
        SetRuntimeArgs(program, compute, core, {seed});
        SetRuntimeArgs(program, writer, core, {dst_addr, num_tiles, start_id});
    }

    // Execute
    EnqueueProgram(command_queue, program);
```

---

## Tile Format

```yaml
tile:
  dimensions:
    height: 32
    width: 32
    total_elements: 1024

  memory_layout: |
    Tile is stored as 4 faces (16x16 each):
    ┌─────────┬─────────┐
    │ Face 0  │ Face 1  │
    │ (16x16) │ (16x16) │
    ├─────────┼─────────┤
    │ Face 2  │ Face 3  │
    │ (16x16) │ (16x16) │
    └─────────┴─────────┘

  data_formats:
    - Float16
    - BFloat16
    - Float32
    - Int32
    - etc.

  size_calculation: |
    tile_size = 32 * 32 * sizeof(element_type)
    # For Float16: 32 * 32 * 2 = 2048 bytes
    # For Float32: 32 * 32 * 4 = 4096 bytes
```

---

## Pattern Selection Rules

```yaml
pattern_selection:
  - conditions:
      - "memory_layout == INTERLEAVED"
      - "input/output in DRAM"
    pattern: "Reader-Compute-Writer"
    reader: "noc_async_read_tile loop"
    writer: "noc_async_write_tile loop"

  - conditions:
      - "memory_layout == HEIGHT_SHARDED"
      - "input in L1 shard"
    pattern: "Zero-Copy + Compute + Writer"
    reader: "cb_push_back only (signal)"
    writer: "depends on output layout"

  - conditions:
      - "input_layout == TILE"
      - "output_layout == ROW_MAJOR"
    pattern: "Includes untilize step"
    compute: "Uses pack_untilize instead of pack_tile"

  - conditions:
      - "input_layout == ROW_MAJOR"
      - "output_layout == TILE"
    pattern: "Includes tilize step"
    compute: "Uses tilize instead of copy_tile"
```

---

## Quick Reference: Where Things Live

| Item | Location | Notes |
|------|----------|-------|
| Input Tensor (INTERLEAVED) | DRAM | Must be read by Reader kernel |
| Input Tensor (SHARDED) | L1 (SRAM) | Pre-placed, zero-copy |
| Circular Buffers | L1 (SRAM) | Sync between kernels |
| DST Registers | Compute Engine | 8 tile slots for compute |
| Output Tensor (INTERLEAVED) | DRAM | Must be written by Writer kernel |
| Output Tensor (SHARDED) | L1 (SRAM) | Already in place after compute |
| Kernel Code | L1 (SRAM) | Loaded at program start |
| Compile-time Args | Kernel binary | Baked in at compile |
| Runtime Args | L1 (SRAM) | Set before execution |

---

## Hardware Constants

```yaml
tile:
  TILE_HEIGHT: 32
  TILE_WIDTH: 32
  TILE_HW: 1024  # 32 * 32

memory_layouts:
  - INTERLEAVED       # Pages striped across DRAM banks
  - HEIGHT_SHARDED    # Rows partitioned across L1
  - WIDTH_SHARDED     # Columns partitioned across L1
  - BLOCK_SHARDED     # 2D blocks across L1
```

---

## Work Distribution

```yaml
work_distribution:
  function: "tt::tt_metal::split_work_to_cores"
  purpose: "Distribute tiles evenly across available cores"

  input:
    - grid_size       # Available compute grid
    - total_tiles     # Work to distribute

  output:
    - num_cores       # Cores utilized
    - all_cores       # Set of all cores
    - core_group_1    # Cores with more tiles (if uneven)
    - core_group_2    # Cores with fewer tiles (may be empty)
    - tiles_g1        # Tiles per core in group 1
    - tiles_g2        # Tiles per core in group 2

  when_uneven: |
    If total_tiles % num_cores != 0:
      core_group_1 gets ceil(tiles/cores) tiles each
      core_group_2 gets floor(tiles/cores) tiles each
    Creates two separate compute kernels with different compile args

  sharded_case:
    rule: "Each core processes its local shard"
    no_work_split: true
```

---

## Pattern Implications

```yaml
pattern_implications:
  sharded_input:
    condition: "input.is_sharded() == true"
    implies:
      - "Reader uses signal-only pattern (cb_push_back only)"
      - "CB uses set_globally_allocated_address()"
      - "No DRAM read needed"

  sharded_output:
    condition: "output.is_sharded() == true"
    implies:
      - "Writer may skip DRAM write"
      - "CB uses set_globally_allocated_address()"
      - "Data stays in L1 after compute"

  untilize:
    condition: "input.layout == TILE && output.layout == ROW_MAJOR"
    implies:
      - "Compute uses pack_untilize_dest"
      - "Requires pack_untilize_init + pack_untilize_uninit"
```

---

## Circular Buffer Configuration

```yaml
cb_configuration:
  standard_fields:
    - name: string              # e.g., "cb_in", "cb_out"
    - index: CBIndex            # c_0, c_1, c_2, ...
    - size: string              # Total size expression
    - page_size: string         # Single page size
    - format: DataFormat        # Float16, BFloat16, etc.

  zero_copy_fields:
    - zero_copy: bool           # Default: false
    - buffer: string            # Required if zero_copy: true

  zero_copy_pattern: |
    // Setup for sharded tensors
    auto config = CircularBufferConfig(size, {{index, format}})
        .set_page_size(index, page_size)
        .set_globally_allocated_address(*buffer);  // Zero-copy to shard
```
