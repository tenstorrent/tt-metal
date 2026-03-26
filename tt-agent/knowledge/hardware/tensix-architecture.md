# Tensix Architecture

Stable silicon facts. Update only when hardware changes.

## Processing Elements

Each Tensix core contains five 32-bit in-order RISC-V CPUs:

| CPU | Role | Coprocessor thread |
|-----|------|--------------------|
| DM0 (B) | Reader: moves data from DRAM/L1 into CBs | — |
| DM1 (NC) | Writer: moves data from CBs to DRAM/L1 | — |
| T0 (Unpack) | Unpacks tiles from L1 into register file | T0 |
| T1 (Math) | Executes FPU/SFPU operations | T1 |
| T2 (Pack) | Packs results from register file to L1 | T2 |

## Compute Units

**FPU (Matrix Unit):** Matrix multiplication and element-wise operations on tiles.
One operation per cycle: 8×16 × 16×16 = 8×16 output tiles.
Math fidelity controls precision/throughput trade-off (LoFi, HiFi2, HiFi4).

**SFPU (Vector Unit):** 32-lane SIMD on 32-bit float/int values. Used for
transcendental functions, activations, and operations not expressible as matmul.
Slower than FPU for bulk operations.

## Memory

**L1 SRAM:** 1.5 MB per Tensix core. Holds circular buffers, sharded tensor tiles,
and kernel stack. Statically allocated before kernel launch — no dynamic allocation.

**DRAM:** Off-chip. Accessed via NOC. High latency, high bandwidth.
Tensors too large for L1 live in DRAM and are streamed through L1 via circular buffers.

## NOC (Network on Chip)

Two NOCs run in opposite directions forming a 2D torus across the chip:
- NOC0: conventionally used for reads (DM0 pulls from DRAM/remote L1)
- NOC1: conventionally used for writes (DM1 pushes to DRAM/remote L1)

Operations: `noc_async_read`, `noc_async_write`, `noc_async_read_barrier`,
`noc_async_write_barrier`. Transfers are asynchronous — always issue a barrier
before reading data that was written via NOC.

## Execution Model

SPMD: all cores run the same kernel binary with different runtime args (typically
their own coordinates). The host distributes work by assigning different tensor
offsets to each core via runtime args.

## Hardware Targets

| Target | Notes |
|--------|-------|
| Wormhole B0 | Current production hardware |
| Blackhole | Next-gen, higher core count |
| Quasar | Research/future |
