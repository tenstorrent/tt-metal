# TTNN Operation Table Templates

Standard table formats for analyzer and planner outputs.

## Tensor Format Table

| Property | Input Tensor | Output Tensor |
|----------|--------------|---------------|
| **Logical shape** | [N, H, W, C] | [N, H, W, C] |
| **Dimension convention** | NHWC | NHWC |
| **Tensor layout** | TILE_LAYOUT | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED | INTERLEAVED |
| **Buffer type** | DRAM | DRAM |
| **Data type** | BFLOAT16 | BFLOAT16 |

## Circular Buffer Table

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_in0 | cb_input | Input tiles | 2 tiles | 1 tile | Double | Reader | Compute | Block |

**Purpose values**: Input staging, Output staging, Scratchpad, Accumulator, Intermediate
**Buffering values**: Single (capacity=block), Double (capacity=2x block), Multi
**Lifetime values**: Block (per iteration), Row (across iterations), Program (entire kernel)

## Core Distribution Table

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 1D / 2D |
| **Grid dimensions** | rows x cols |
| **Total cores** | N |
| **Work per core** | M tiles |
| **Load balancing** | equal / round-robin / weighted |

## Compile-Time Arguments Table

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | num_tiles_c | uint32_t | Tiles along C dimension |

## Runtime Arguments Table

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Source buffer address |
| 1 | num_tiles | uint32_t | Tiles to process |

## Kernel Specification Table

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader | RISCV_0 | NOC0 | DRAM | CB_in | Read tiles |
| compute | RISCV_2 | N/A | CB_in | CB_out | FPU ops |
| writer | RISCV_1 | NOC1 | CB_out | DRAM | Write tiles |

## Data Flow Table

| Stage | Kernel | Reads From | Writes To | CB Operations |
|-------|--------|------------|-----------|---------------|
| 1 | Reader | DRAM | CB_in | reserve_back, push_back |
| 2 | Compute | CB_in | CB_out | wait_front, pop_front, push_back |
| 3 | Writer | CB_out | DRAM | wait_front, pop_front |

## Input/Output Requirements Table

| Property | Requirement | Error Message Hint |
|----------|-------------|-------------------|
| Rank | Must be 4D | "must be 4D" |
| Layout | TILE_LAYOUT | "must be in TILE layout" |
| Dtype | bfloat16 or float32 | "unsupported dtype" |
| Device | Must be on device | "must be on device" |

## Work Unit Table

| Attribute | Value |
|-----------|-------|
| **Granularity** | tile / block / row |
| **Unit size** | N tiles |
| **Total units** | formula or value |
| **Loop structure** | description |
