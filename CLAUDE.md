# tt-metal

tt-metal is Tenstorrent's low-level programming framework for Tensix-based AI accelerators (Wormhole B0, Blackhole, Quasar). It exposes bare-metal control over a mesh of RISC-V cores, each with matrix (FPU) and vector (SFPU) compute units, coordinated through explicit data movement over two NOC networks. The stack spans from hand-written kernels up through the `ttnn` Python operator library used by model authors.

See `METALIUM_GUIDE.md` for the full architecture reference.
See `tt-agent/skills/` for agentic workflow instructions (writing ops, debugging, etc.).

## Abstraction Levels

**Kernel** (C++ on RISC-V) -- Each Tensix core runs up to 5 Baby RISC-V CPUs: two data-movement processors (DM0/reader, DM1/writer) and three compute processors (Unpack, Math, Pack). Kernels are bare-metal C++ compiled per-processor. Coordination between reader/compute/writer happens via circular buffers in L1 SRAM. Execution model is SPMD: same kernel binary, different core coordinates.

**Operator** (host C++) -- Host-side C++ that creates `Program` objects, configures circular buffers, sets runtime args, and dispatches to device. This is where grid shapes, memory layouts, sharding strategies, and kernel binaries are assembled. Canonical example: `ttnn/cpp/ttnn/operations/eltwise/binary/device/`.

**Model** (Python using ttnn) -- Python code composing `ttnn` operators into full models. Handles weight loading, KV-cache management, attention masks, and multi-device distribution. See `models/tt_transformers/tt/` for production LLM implementations.

## Hardware Invariants

- **Tile granularity**: Native compute unit is 32x32 elements. All CBs and compute operate on tiles.
- **L1 SRAM**: 1.5 MB per Tensix core. Holds circular buffers, intermediate data, and sharded tensors.
- **No dynamic allocation in kernels**: All memory (CBs, semaphores, buffers) is statically configured from the host before launch.
- **Circular buffers**: Producer/consumer coordination primitive. Reader pushes tiles into a CB; compute consumes and pushes to another CB; writer drains. `cb_reserve_back` / `cb_push_back` and `cb_wait_front` / `cb_pop_front` are the core API.
- **Explicit NOC data movement**: Two NOCs in opposite directions form a 2D torus. Kernels explicitly issue `noc_async_read` / `noc_async_write` + barrier. No implicit caching or coherence.
- **Data formats**: BFLOAT16, BFLOAT8_B, BFLOAT4_B, FLOAT32, UINT16, UINT32. Accumulation typically in FLOAT32 (DST register).

## Quality Bar

PCC (Pearson Correlation Coefficient) > 0.999 against PyTorch reference for all ops and models. Test with `ttnn.pearson_correlation_coefficient` or equivalent.

## Key Repo Directories

```
Kernels & examples:
  tt_metal/programming_examples/          # Bare-metal kernel examples

Kernel APIs:
  tt_metal/hw/inc/api/dataflow/dataflow_api.h   # Reader/writer NOC + CB API
  tt_metal/hw/inc/api/compute/                   # Compute kernel API (tiles, math)

Circular buffer config:
  tt_metal/api/tt-metalium/circular_buffer_config.hpp

Operator implementations:
  ttnn/cpp/ttnn/operations/                      # All ttnn ops
  ttnn/cpp/ttnn/operations/eltwise/binary/device/ # Canonical op example

Model implementations:
  models/tt_transformers/tt/                     # Production LLM layers
  models/common/                                 # Shared model utilities

Model demos & tests:
  models/demos/                                  # End-to-end model demos
  models/tt_transformers/tests/                  # Model-level tests

Unit tests:
  tests/ttnn/unit_tests/                         # ttnn op unit tests

Architecture docs:
  tech_reports/                                  # Deep-dive technical reports

Low-level kernels (LLK):
  tt_metal/third_party/tt_llk/                   # Tenstorrent low-level kernels
```
